"""Session storage: JSONL, manifest, summary, recovery, and path safety.

Every test writes into a pytest ``tmp_path``.  Nothing touches the real
``artifacts/`` directory.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from engagevr.protocol.envelope import (
    REPLAY_LABEL,
    SYNTHETIC_LABEL,
    MessageProvenance,
    ReplayMetadata,
)
from engagevr.protocol.messages import (
    DisconnectReason,
    MessageSource,
    MessageType,
)
from engagevr.protocol.version import PROTOCOL_VERSION
from engagevr.schemas.session import DataSource
from engagevr.storage import (
    RECORDING_DISCLAIMER,
    DropRecord,
    IngestionMetadata,
    InvalidSessionIdError,
    JsonlFormatError,
    JsonlWriter,
    SessionStore,
    SessionStoreError,
    read_jsonl,
    read_jsonl_tolerant,
    session_directory,
    validate_session_id,
    write_json_atomic,
)
from engagevr.storage.session_store import (
    EVENTS_FILENAME,
    MANIFEST_FILENAME,
    SUMMARY_FILENAME,
)
from engagevr.synchronization.ordering import OrderingAnomaly
from tests.unit.test_protocol import make_envelope

BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def ingestion(index: int = 0, **overrides: object) -> IngestionMetadata:
    fields: dict[str, object] = {
        "arrival_index": index,
        "server_received_at_utc": BASE,
        "server_monotonic_seconds": float(index),
        "transport": "in_process",
        "client_id": "client-1",
        "client_role": "simulator",
    }
    fields.update(overrides)
    return IngestionMetadata.model_validate(fields)


# --- session id safety -----------------------------------------------------


class TestSessionIdValidation:
    @pytest.mark.parametrize(
        "session_id",
        [
            "demo-session",
            "a",
            "S1",
            "sim-0123abcd",
            "run.2026-01-01",
            "A_b".replace("_", "-"),
        ],
    )
    def test_safe_identifiers_are_accepted(self, session_id: str) -> None:
        assert validate_session_id(session_id) == session_id

    @pytest.mark.parametrize(
        "session_id",
        [
            "",
            ".",
            "..",
            "../escape",
            "a/b",
            "a\\b",
            "/absolute",
            ".hidden",
            "with space",
            "with\x00null",
            "sess:1",
            "sess;1",
            "~root",
            "a" * 129,
        ],
    )
    def test_unsafe_identifiers_are_rejected(self, session_id: str) -> None:
        with pytest.raises(InvalidSessionIdError):
            validate_session_id(session_id)

    def test_path_traversal_cannot_escape_the_root(self, tmp_path: Path) -> None:
        for attempt in ("../../etc", "..", "a/../../b"):
            with pytest.raises(InvalidSessionIdError):
                session_directory(tmp_path, attempt)

    def test_resolved_directory_is_inside_the_root(self, tmp_path: Path) -> None:
        resolved = session_directory(tmp_path, "demo")
        assert resolved.parent == tmp_path.resolve()


# --- JSONL -----------------------------------------------------------------


class TestJsonl:
    def test_append_only_writer_adds_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        with JsonlWriter(path) as writer:
            writer.write({"a": 1})
            writer.write({"a": 2})
        with JsonlWriter(path) as writer:
            writer.write({"a": 3})

        lines = list(read_jsonl(path))
        assert [line.record["a"] for line in lines] == [1, 2, 3]
        assert [line.line_number for line in lines] == [1, 2, 3]

    def test_flush_every_is_honoured(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        writer = JsonlWriter(path, flush_every=3)
        writer.write({"a": 1})
        writer.write({"a": 2})
        assert path.read_text() == "", "not yet flushed"
        writer.write({"a": 3})
        assert path.read_text() != "", "flushed at the third record"
        writer.close()

    def test_flush_every_zero_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            JsonlWriter(tmp_path / "x.jsonl", flush_every=0)

    def test_malformed_line_reports_its_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        path.write_text('{"a":1}\n{"a":2}\nnot json at all\n')
        with pytest.raises(JsonlFormatError) as info:
            list(read_jsonl(path))
        assert info.value.line_number == 3
        assert "not json at all" in info.value.excerpt
        assert "events.jsonl:3" in str(info.value)

    def test_non_object_line_is_rejected_with_its_number(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        path.write_text('{"a":1}\n[1,2,3]\n')
        with pytest.raises(JsonlFormatError) as info:
            list(read_jsonl(path))
        assert info.value.line_number == 2
        assert "expected a JSON object" in info.value.detail

    def test_tolerant_reader_collects_bad_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        path.write_text('{"a":1}\ntorn{\n{"a":3}\n')
        good, bad = read_jsonl_tolerant(path)
        assert [line.record["a"] for line in good] == [1, 3]
        assert bad == [2]

    def test_atomic_write_leaves_no_temporary_file(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        write_json_atomic(path, {"ok": True})
        assert json.loads(path.read_text()) == {"ok": True}
        assert list(tmp_path.glob(".*tmp")) == []

    def test_atomic_write_replaces_the_previous_document(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        write_json_atomic(path, {"version": 1})
        write_json_atomic(path, {"version": 2})
        assert json.loads(path.read_text()) == {"version": 2}


# --- recorder and store ----------------------------------------------------


class TestSessionRecorder:
    def test_manifest_is_written_on_open(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo", configuration={"seed": 42})
        recorder.close()

        manifest = store.read_manifest("demo")
        assert manifest.session_id == "demo"
        assert manifest.protocol_version == PROTOCOL_VERSION
        assert manifest.configuration == {"seed": 42}
        assert manifest.disclaimer == RECORDING_DISCLAIMER
        assert (tmp_path / "demo" / MANIFEST_FILENAME).is_file()

    def test_events_are_appended_in_arrival_order(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        # Deliberately out of sequence order: arrival order must win.
        for index, sequence in enumerate([0, 5, 2]):
            recorder.append(make_envelope(sequence_number=sequence), ingestion(index))
        recorder.close()

        stored = list(store.iter_messages("demo"))
        assert [m.envelope.sequence_number for m in stored] == [0, 5, 2]
        assert [m.ingestion.arrival_index for m in stored] == [0, 1, 2]

    def test_summary_is_atomic_and_counts_everything(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        recorder.append(
            make_envelope(message_type=MessageType.SESSION_START), ingestion(0)
        )
        recorder.append(make_envelope(sequence_number=1), ingestion(1))
        recorder.append(
            make_envelope(message_type=MessageType.SESSION_END, sequence_number=2),
            ingestion(2),
        )
        summary = recorder.close(disconnect_reason=DisconnectReason.ORDERLY)

        assert summary.event_count == 3
        assert summary.message_type_counts["task_event"] == 1
        assert summary.source_counts["python_simulator"] == 3
        assert summary.completed is True
        assert summary.disconnect_reason is DisconnectReason.ORDERLY
        assert summary.synthetic_message_count == 3
        assert summary.replay_message_count == 0
        assert summary.first_received_at_utc == BASE
        assert (tmp_path / "demo" / SUMMARY_FILENAME).is_file()
        assert store.read_summary("demo") == summary

    def test_incomplete_session_is_not_marked_completed(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        recorder.append(make_envelope(), ingestion(0))
        summary = recorder.close()
        assert summary.completed is False

    def test_anomalies_are_counted(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        recorder.append(
            make_envelope(),
            ingestion(0, anomalies=[OrderingAnomaly.SEQUENCE_REVERSAL]),
        )
        summary = recorder.close()
        assert summary.anomaly_counts["sequence_reversal"] == 1

    def test_replay_messages_are_counted_separately(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        envelope = make_envelope().with_replay_metadata(
            ReplayMetadata(
                source_session_id="a",
                replay_session_id="demo",
                replay_index=0,
                replay_speed=0.0,
                replayed_at_utc=BASE,
            )
        )
        recorder.append(envelope, ingestion(0, transport="replay"))
        summary = recorder.close()
        assert summary.replay_message_count == 1
        assert summary.synthetic_message_count == 1

    def test_drops_are_recorded_and_counted(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        recorder.record_drop(
            DropRecord(
                message_id="m1",
                message_type=MessageType.TELEMETRY,
                source=MessageSource.PYTHON_SIMULATOR,
                sequence_number=7,
                dropped_at_utc=BASE,
                queue="storage",
                reason="storage queue was full",
            )
        )
        summary = recorder.close()

        assert summary.dropped_message_count == 1
        assert summary.dropped_message_types["telemetry"] == 1
        dropped = (tmp_path / "demo" / "dropped.jsonl").read_text()
        assert "storage queue was full" in dropped

    def test_append_after_close_is_refused(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        recorder.close()
        with pytest.raises(SessionStoreError, match="closed"):
            recorder.append(make_envelope(), ingestion(0))

    def test_context_manager_records_an_internal_failure(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        with pytest.raises(RuntimeError):
            with store.open_recorder("demo") as recorder:
                recorder.append(make_envelope(), ingestion(0))
                raise RuntimeError("boom")
        summary = store.read_summary("demo")
        assert summary is not None
        assert summary.disconnect_reason is DisconnectReason.INTERNAL_FAILURE


class TestSessionStore:
    def test_listing_skips_non_sessions(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        store.open_recorder("alpha").close()
        store.open_recorder("beta").close()
        (tmp_path / "not-a-session").mkdir()
        (tmp_path / "loose-file.txt").write_text("x")

        assert store.list_sessions() == ["alpha", "beta"]

    def test_listing_an_absent_root_is_empty(self, tmp_path: Path) -> None:
        assert SessionStore(tmp_path / "nope").list_sessions() == []

    def test_exists_rejects_unsafe_ids_without_raising(self, tmp_path: Path) -> None:
        assert SessionStore(tmp_path).exists("../etc") is False

    def test_reading_a_missing_session_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SessionStoreError, match="no manifest"):
            SessionStore(tmp_path).read_manifest("ghost")

    def test_summary_is_none_when_the_session_never_closed(
        self, tmp_path: Path
    ) -> None:
        store = SessionStore(tmp_path)
        store.open_recorder("demo")  # deliberately not closed
        assert store.read_summary("demo") is None

    def test_corrupted_stored_message_reports_its_line(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        recorder.append(make_envelope(), ingestion(0))
        recorder.close()

        path = store.events_path("demo")
        path.write_text(
            path.read_text()
            + json.dumps({"envelope": {"protocol_version": "9.9"}, "ingestion": {}})
            + "\n"
        )
        with pytest.raises(SessionStoreError) as info:
            list(store.iter_messages("demo"))
        assert ":2:" in str(info.value)

    def test_record_missing_envelope_is_reported(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        recorder.close()
        store.events_path("demo").write_text('{"not_an_envelope": true}\n')
        with pytest.raises(SessionStoreError, match="'envelope' and 'ingestion'"):
            list(store.iter_messages("demo"))


class TestRecovery:
    def test_interrupted_session_is_recovered(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        for n in range(3):
            recorder.append(make_envelope(sequence_number=n), ingestion(n))
        # Simulate a crash: no close(), so no summary.json.
        assert store.read_summary("demo") is None

        recovered = store.recover("demo")
        assert recovered.recovered is True
        assert recovered.completed is False
        assert recovered.event_count == 3
        assert recovered.malformed_line_numbers == []

    def test_torn_final_line_is_reported_not_fatal(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        recorder.append(make_envelope(sequence_number=0), ingestion(0))
        recorder.append(make_envelope(sequence_number=1), ingestion(1))
        path = store.events_path("demo")
        with path.open("a") as handle:
            handle.write('{"envelope": {"protocol_ver')  # torn write

        recovered = store.recover("demo")
        assert recovered.event_count == 2, "the intact lines are still readable"
        assert recovered.malformed_line_numbers == [3]

    def test_recovery_does_not_modify_the_recording(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        recorder.append(make_envelope(), ingestion(0))
        before = store.events_path("demo").read_bytes()

        store.recover("demo")
        assert store.events_path("demo").read_bytes() == before

    def test_recovery_sees_a_recorded_session_end(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        recorder.append(
            make_envelope(message_type=MessageType.SESSION_END), ingestion(0)
        )
        recovered = store.recover("demo")
        assert recovered.completed is True

    def test_recovering_a_missing_stream_raises(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        with pytest.raises(SessionStoreError, match="no event stream"):
            store.recover("ghost")


class TestPrivacyInvariants:
    """What a recording must never be able to contain."""

    FORBIDDEN = (
        "frame",
        "image",
        "pixel",
        "landmark",
        "mediapipe",
        "engagement",
        "cognitive_load",
        "confidence",
        "prediction",
        "bpm",
        "heart_rate",
        "email",
        "password",
        "secret",
        "token",
    )

    #: Prose fields that *describe* what is excluded, and therefore
    #: legitimately name the excluded things. Scanning them would make
    #: the disclaimer trip its own test.
    PROSE_KEYS = frozenset({"disclaimer", "note", "reason", "delay_unavailable_reason"})

    @classmethod
    def _scan(cls, value: object, path: str = "") -> list[str]:
        """Return every forbidden token found outside a prose field."""
        found: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = str(key).lower()
                for token in cls.FORBIDDEN:
                    if token in lowered:
                        found.append(f"{path}.{key} (key)")
                if key in cls.PROSE_KEYS:
                    continue
                found.extend(cls._scan(item, f"{path}.{key}"))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                found.extend(cls._scan(item, f"{path}[{index}]"))
        elif isinstance(value, str):
            lowered = value.lower()
            for token in cls.FORBIDDEN:
                if token in lowered:
                    found.append(f"{path} (value)")
        return found

    def test_a_recorded_session_contains_no_forbidden_field(
        self, tmp_path: Path
    ) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        for n in range(5):
            recorder.append(make_envelope(sequence_number=n), ingestion(n))
        recorder.close()

        for filename in (MANIFEST_FILENAME, SUMMARY_FILENAME):
            document = json.loads((tmp_path / "demo" / filename).read_text())
            assert self._scan(document, filename) == []

        for line in (tmp_path / "demo" / EVENTS_FILENAME).read_text().splitlines():
            assert self._scan(json.loads(line), EVENTS_FILENAME) == []

    def test_the_envelope_cannot_carry_binary_frame_data(self) -> None:
        from pydantic import ValidationError

        from engagevr.protocol.messages import TelemetryPayload

        with pytest.raises(ValidationError):
            TelemetryPayload(
                component="capture",
                metrics={"frame": object()},  # type: ignore[dict-item]
            )

    def test_synthetic_label_survives_a_storage_round_trip(
        self, tmp_path: Path
    ) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        recorder.append(make_envelope(), ingestion(0))
        recorder.close()

        stored = next(iter(store.iter_messages("demo")))
        assert stored.envelope.provenance.data_source is DataSource.SYNTHETIC
        assert stored.envelope.provenance.synthetic_label == SYNTHETIC_LABEL

    def test_replay_label_survives_a_storage_round_trip(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        recorder.append(
            make_envelope().with_replay_metadata(
                ReplayMetadata(
                    source_session_id="a",
                    replay_session_id="demo",
                    replay_index=0,
                    replay_speed=0.0,
                    replayed_at_utc=BASE,
                )
            ),
            ingestion(0),
        )
        recorder.close()

        stored = next(iter(store.iter_messages("demo")))
        assert stored.envelope.replay is not None
        assert stored.envelope.replay.replay_label == REPLAY_LABEL
        assert stored.envelope.provenance.synthetic_label == SYNTHETIC_LABEL

    def test_live_provenance_carries_no_synthetic_label(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("demo")
        envelope = make_envelope().model_copy(
            update={
                "provenance": MessageProvenance(
                    data_source=DataSource.LIVE, synthetic_label=None, producer="p"
                )
            }
        )
        recorder.append(envelope, ingestion(0))
        summary = recorder.close()
        assert summary.synthetic_message_count == 0
