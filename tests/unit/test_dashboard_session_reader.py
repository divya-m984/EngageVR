"""The tail-safe session reader and the session catalogue.

The properties that matter here are the ones a live view depends on: a
recording being written to must not look corrupt, a genuinely bad line
must not vanish, and reading must never change what is on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engagevr.dashboard.session_catalogue import (
    UNAVAILABLE_STATEMENTS,
    build_session_catalogue,
    read_session,
)
from engagevr.dashboard.session_reader import (
    PARTIAL_TRAILING_NOTE,
    SessionReadError,
    decode_records,
    file_checksums,
    read_json_document,
    read_stream,
    readable_session_id,
    sequence_observations,
)
from engagevr.schemas.dashboard_session import (
    DashboardSessionMode,
    DashboardSessionStatus,
    SessionRecordProblem,
)
from tests.unit import session_fixtures as sfx

REPLAY = DashboardSessionMode.REPLAY
LIVE = DashboardSessionMode.LIVE


@pytest.fixture
def session_root(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


class TestAbsentAndEmptyRoots:
    def test_an_absent_root_is_a_state_not_an_exception(self, tmp_path: Path) -> None:
        catalogue = build_session_catalogue(tmp_path / "nowhere", mode=REPLAY)
        assert catalogue.root_exists is False
        assert catalogue.is_empty
        assert catalogue.warnings
        assert "does not exist" in catalogue.warnings[0].message

    def test_an_empty_root_lists_no_session(self, session_root: Path) -> None:
        session_root.mkdir(parents=True)
        catalogue = build_session_catalogue(session_root, mode=REPLAY)
        assert catalogue.root_exists is True
        assert catalogue.is_empty

    def test_a_file_in_the_root_is_not_a_session(self, session_root: Path) -> None:
        session_root.mkdir(parents=True)
        (session_root / "notes.txt").write_text("synthetic", encoding="utf-8")
        catalogue = build_session_catalogue(session_root, mode=REPLAY)
        assert catalogue.is_empty

    def test_an_invalid_directory_name_is_reported_not_listed(
        self, session_root: Path
    ) -> None:
        sfx.write_session(session_root, "synthetic-ok")
        bad = session_root / ".hidden"
        bad.mkdir()
        (bad / "manifest.json").write_text("{}", encoding="utf-8")
        catalogue = build_session_catalogue(session_root, mode=REPLAY)
        assert catalogue.session_ids() == ("synthetic-ok",)
        assert any(
            "not a valid session identifier" in w.message for w in catalogue.warnings
        )


class TestSessionDiscovery:
    def test_a_completed_session_is_discovered(self, session_root: Path) -> None:
        sfx.write_session(session_root, "synthetic-completed")
        catalogue = build_session_catalogue(session_root, mode=REPLAY)
        assert catalogue.session_ids() == ("synthetic-completed",)
        summary = catalogue.sessions[0]
        assert summary.provenance.status is DashboardSessionStatus.COMPLETED
        assert summary.session_end_recorded is True

    def test_an_active_session_is_discovered_and_is_not_failed(
        self, session_root: Path
    ) -> None:
        sfx.write_session(
            session_root, "synthetic-active", with_summary=False, completed=False
        )
        catalogue = build_session_catalogue(session_root, mode=LIVE)
        summary = catalogue.sessions[0]
        assert summary.provenance.status is DashboardSessionStatus.ACTIVE_OR_INCOMPLETE
        assert summary.provenance.status is not DashboardSessionStatus.FAILED

    def test_an_interrupted_session_is_visible_and_inspectable(
        self, session_root: Path
    ) -> None:
        sfx.write_session(
            session_root,
            "synthetic-interrupted",
            completed=False,
            disconnect_reason="client_disconnect",
        )
        read = read_session(session_root / "synthetic-interrupted", mode=REPLAY)
        assert read.summary.provenance.status is DashboardSessionStatus.INTERRUPTED
        assert read.summary.decoded_record_count > 0

    def test_a_recorded_failure_is_reported_as_failed(self, session_root: Path) -> None:
        sfx.write_session(
            session_root,
            "synthetic-failed",
            completed=False,
            disconnect_reason="internal_failure",
        )
        read = read_session(session_root / "synthetic-failed", mode=REPLAY)
        assert read.summary.provenance.status is DashboardSessionStatus.FAILED
        assert read.summary.provenance.status_reason

    def test_a_missing_manifest_makes_the_session_unreadable(
        self, session_root: Path
    ) -> None:
        sfx.write_session(session_root, "synthetic-no-manifest", with_manifest=False)
        read = read_session(session_root / "synthetic-no-manifest", mode=REPLAY)
        assert read.summary.provenance.status is DashboardSessionStatus.UNREADABLE
        assert "manifest.json" in (read.summary.provenance.status_reason or "")

    def test_an_absent_event_stream_is_its_own_status(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-no-stream")
        (directory / "events.jsonl").unlink()
        read = read_session(directory, mode=REPLAY)
        assert read.summary.provenance.status is (
            DashboardSessionStatus.STREAM_UNAVAILABLE
        )

    def test_a_removed_directory_is_reported_not_crashed(
        self, session_root: Path
    ) -> None:
        read = read_session(session_root / "synthetic-gone", mode=REPLAY)
        assert read.summary.provenance.status is DashboardSessionStatus.UNREADABLE
        assert read.summary.complete_record_count == 0


class TestTailSafety:
    def test_a_partial_trailing_line_does_not_crash_or_count(
        self, session_root: Path
    ) -> None:
        sfx.write_session(
            session_root,
            "synthetic-partial",
            with_summary=False,
            completed=False,
            partial_trailing='{"envelope": {"message_ty',
        )
        read = read_session(session_root / "synthetic-partial", mode=LIVE)
        assert read.summary.partial_trailing_line is True
        assert read.summary.malformed_record_count == 0
        assert read.summary.decoded_record_count == read.summary.complete_record_count

    def test_a_partial_trailing_line_is_explained_not_called_corrupt(
        self, session_root: Path
    ) -> None:
        sfx.write_session(session_root, "synthetic-partial", partial_trailing='{"half')
        read = read_session(session_root / "synthetic-partial", mode=LIVE)
        note = read.summary.partial_trailing_note or ""
        assert note == PARTIAL_TRAILING_NOTE
        assert "not treated as corruption" in note
        assert "normal appearance" in note

    def test_a_final_line_without_a_newline_is_treated_as_partial(
        self, session_root: Path
    ) -> None:
        sfx.write_session(
            session_root, "synthetic-unterminated", trailing_newline=False
        )
        read = read_session(session_root / "synthetic-unterminated", mode=LIVE)
        assert read.summary.partial_trailing_line is True

    def test_a_torn_multibyte_tail_does_not_raise(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-torn")
        path = directory / "events.jsonl"
        path.write_bytes(path.read_bytes() + b'{"envelope": "\xe2\x82')
        read = read_session(directory, mode=LIVE)
        assert read.summary.partial_trailing_line is True
        assert read.summary.malformed_record_count == 0

    def test_a_malformed_interior_line_stays_visible(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-malformed")
        sfx.corrupt_line(directory, 3)
        read = read_session(directory, mode=REPLAY)
        assert read.summary.malformed_record_count == 1
        assert read.summary.malformed_line_numbers == (3,)
        bad = [record for record in read.records if not record.decoded]
        assert bad[0].problem is SessionRecordProblem.MALFORMED_JSON
        assert bad[0].problem_detail

    def test_a_line_that_is_not_a_stored_record_is_reported(
        self, session_root: Path
    ) -> None:
        directory = sfx.write_session(session_root, "synthetic-structure")
        sfx.corrupt_line(directory, 2, json.dumps({"something": "else"}))
        read = read_session(directory, mode=REPLAY)
        bad = [record for record in read.records if not record.decoded]
        assert bad[0].problem is SessionRecordProblem.INVALID_STRUCTURE

    def test_a_protocol_invalid_message_is_reported(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-protocol")
        broken = json.dumps(
            {
                "envelope": sfx.envelope(
                    session_id="synthetic-protocol",
                    message_type="session_start",
                    sequence_number=99,
                    payload={"participant_id": ""},
                ),
                "ingestion": sfx.ingestion(arrival_index=99),
            }
        )
        sfx.corrupt_line(directory, 2, broken)
        read = read_session(directory, mode=REPLAY)
        bad = [record for record in read.records if not record.decoded]
        assert bad[0].problem is SessionRecordProblem.PROTOCOL_INVALID

    def test_invalid_ingestion_metadata_is_reported(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-ingestion")
        line = json.dumps(
            {
                "envelope": sfx.envelope(
                    session_id="synthetic-ingestion",
                    message_type="heartbeat",
                    sequence_number=5,
                    payload={
                        "heartbeat_id": "synthetic-heartbeat-0001",
                        "client_monotonic_seconds": 1.0,
                    },
                ),
                "ingestion": {"arrival_index": -1},
            }
        )
        sfx.corrupt_line(directory, 2, line)
        read = read_session(directory, mode=REPLAY)
        bad = [record for record in read.records if not record.decoded]
        assert bad[0].problem is SessionRecordProblem.INGESTION_INVALID

    def test_a_blank_line_is_skipped_not_counted(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-blank")
        path = directory / "events.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        read = read_session(directory, mode=REPLAY)
        assert read.summary.malformed_record_count == 0


class TestIncrementalReading:
    def test_a_new_appended_record_appears_on_the_next_read(
        self, session_root: Path
    ) -> None:
        directory = sfx.write_session(
            session_root, "synthetic-live", with_summary=False, completed=False
        )
        first = read_session(directory, mode=LIVE)
        before = first.summary.complete_record_count

        sfx.append_record(
            directory,
            sfx.stored(
                sfx.envelope(
                    session_id="synthetic-live",
                    message_type="heartbeat",
                    sequence_number=90,
                    payload={
                        "heartbeat_id": "synthetic-heartbeat-0002",
                        "client_monotonic_seconds": 2.0,
                    },
                    offset=30,
                ),
                sfx.ingestion(arrival_index=90, offset=30),
            ),
        )
        second = read_session(directory, mode=LIVE)
        assert second.summary.complete_record_count == before + 1
        assert second.records[-1].message_type == "heartbeat"

    def test_reading_from_a_cursor_parses_only_what_is_new(
        self, session_root: Path
    ) -> None:
        directory = sfx.write_session(session_root, "synthetic-cursor")
        full = read_session(directory, mode=LIVE)
        total = full.summary.complete_record_count

        tail = read_session(directory, mode=LIVE, start_line=total - 2)
        assert tail.summary.parsed_record_count == 2
        assert tail.summary.complete_record_count == total
        assert tail.summary.parse_start_line == total - 2
        assert tail.summary.fully_parsed is False

    def test_a_partial_line_appended_between_reads_is_transient(
        self, session_root: Path
    ) -> None:
        directory = sfx.write_session(
            session_root, "synthetic-transient", with_summary=False, completed=False
        )
        line = sfx.stored(
            sfx.envelope(
                session_id="synthetic-transient",
                message_type="heartbeat",
                sequence_number=91,
                payload={
                    "heartbeat_id": "synthetic-heartbeat-0003",
                    "client_monotonic_seconds": 3.0,
                },
            ),
            sfx.ingestion(arrival_index=91),
        )
        sfx.append_record(directory, line[:40], newline=False)
        mid = read_session(directory, mode=LIVE)
        assert mid.summary.partial_trailing_line is True
        assert mid.summary.malformed_record_count == 0

        path = directory / "events.jsonl"
        text = path.read_text(encoding="utf-8")
        path.write_text(text[: -len(line[:40])] + line + "\n", encoding="utf-8")
        done = read_session(directory, mode=LIVE)
        assert done.summary.partial_trailing_line is False
        assert done.summary.malformed_record_count == 0
        assert done.records[-1].message_type == "heartbeat"

    def test_a_negative_cursor_is_refused(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-cursor")
        with pytest.raises(SessionReadError, match="negative"):
            read_stream(directory / "events.jsonl", start_line=-1)


class TestOrderingIsPreserved:
    def test_records_keep_recorded_arrival_order(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-order")
        read = read_session(directory, mode=REPLAY)
        lines = [record.line_number for record in read.records]
        assert lines == sorted(lines)
        arrivals = [record.arrival_index for record in read.records]
        assert arrivals == sorted(a for a in arrivals if a is not None)

    def test_a_reversal_is_shown_not_re_sorted(self, session_root: Path) -> None:
        session_id = "synthetic-reversal"
        lines = sfx.default_lines(session_id)
        lines.append(
            sfx.stored(
                sfx.envelope(
                    session_id=session_id,
                    message_type="heartbeat",
                    sequence_number=1,
                    payload={
                        "heartbeat_id": "synthetic-heartbeat-0004",
                        "client_monotonic_seconds": 4.0,
                    },
                    offset=40,
                ),
                sfx.ingestion(arrival_index=40, offset=40),
            )
        )
        directory = sfx.write_session(session_root, session_id, lines=lines)
        read = read_session(directory, mode=REPLAY)
        kinds = {o.kind for o in read.summary.sequence_observations}
        assert "sequence_reversal" in kinds
        assert read.records[-1].message_type == "heartbeat"

    def test_a_duplicate_sequence_number_is_shown(self, session_root: Path) -> None:
        session_id = "synthetic-duplicate"
        lines = sfx.default_lines(session_id)
        lines.append(lines[-1])
        directory = sfx.write_session(session_root, session_id, lines=lines)
        read = read_session(directory, mode=REPLAY)
        kinds = {o.kind for o in read.summary.sequence_observations}
        assert "duplicate_sequence_number" in kinds
        assert read.summary.decoded_record_count == len(lines)

    def test_a_gap_is_reported_without_inventing_a_message(
        self, session_root: Path
    ) -> None:
        session_id = "synthetic-gap"
        lines = sfx.default_lines(session_id)
        lines.append(
            sfx.stored(
                sfx.envelope(
                    session_id=session_id,
                    message_type="heartbeat",
                    sequence_number=50,
                    payload={
                        "heartbeat_id": "synthetic-heartbeat-0005",
                        "client_monotonic_seconds": 5.0,
                    },
                    offset=50,
                ),
                sfx.ingestion(arrival_index=50, offset=50),
            )
        )
        before = len(lines)
        directory = sfx.write_session(session_root, session_id, lines=lines)
        read = read_session(directory, mode=REPLAY)
        gaps = [
            o
            for o in read.summary.sequence_observations
            if o.kind == "missing_sequence_range"
        ]
        assert gaps and "no message has been invented" in gaps[0].detail
        assert read.summary.decoded_record_count == before

    def test_recorded_anomalies_are_surfaced_verbatim(self, session_root: Path) -> None:
        session_id = "synthetic-anomaly"
        lines = sfx.default_lines(session_id)
        lines.append(
            sfx.stored(
                sfx.envelope(
                    session_id=session_id,
                    message_type="heartbeat",
                    sequence_number=60,
                    payload={
                        "heartbeat_id": "synthetic-heartbeat-0006",
                        "client_monotonic_seconds": 6.0,
                    },
                    offset=60,
                ),
                sfx.ingestion(
                    arrival_index=60,
                    offset=60,
                    anomalies=("excessive_transport_delay",),
                    detail="synthetic fixture anomaly",
                ),
            )
        )
        directory = sfx.write_session(session_root, session_id, lines=lines)
        read = read_session(directory, mode=REPLAY)
        assert ("excessive_transport_delay", 1) in read.summary.recorded_anomaly_counts

    def test_sequence_observations_are_empty_for_a_clean_recording(
        self, session_root: Path
    ) -> None:
        directory = sfx.write_session(session_root, "synthetic-clean")
        read = read_session(directory, mode=REPLAY)
        assert sequence_observations(read.records) == ()


class TestProvenance:
    def test_a_synthetic_recording_is_labelled_synthetic(
        self, session_root: Path
    ) -> None:
        directory = sfx.write_session(session_root, "synthetic-labelled")
        read = read_session(directory, mode=REPLAY)
        provenance = read.summary.provenance
        assert provenance.is_synthetic is True
        assert provenance.data_sources == ("synthetic",)
        assert provenance.synthetic_message_count == read.summary.decoded_record_count

    def test_a_live_data_source_is_not_scientifically_eligible(
        self, session_root: Path
    ) -> None:
        directory = sfx.write_session(
            session_root, "synthetic-live-source", synthetic=False
        )
        read = read_session(directory, mode=LIVE)
        provenance = read.summary.provenance
        assert provenance.data_sources == ("live",)
        assert provenance.is_synthetic is False
        assert provenance.scientific_evaluation_eligible is False
        assert "not automatically eligible" in provenance.eligibility_reason

    def test_a_replayed_recording_keeps_both_labels(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-replayed", replayed=True)
        read = read_session(directory, mode=REPLAY)
        provenance = read.summary.provenance
        assert provenance.replayed_message_count == read.summary.decoded_record_count
        assert provenance.is_synthetic is True
        assert all(record.is_replayed for record in read.records if record.decoded)

    def test_an_unread_session_does_not_claim_to_be_non_synthetic(
        self, session_root: Path
    ) -> None:
        directory = sfx.write_session(session_root, "synthetic-empty", lines=[])
        read = read_session(directory, mode=LIVE)
        provenance = read.summary.provenance
        assert provenance.provenance_established is False
        assert provenance.data_sources == ()

    def test_the_mode_travels_with_the_provenance(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-mode")
        assert read_session(directory, mode=LIVE).summary.provenance.mode is LIVE
        assert read_session(directory, mode=REPLAY).summary.provenance.mode is REPLAY


class TestUnavailableQuantities:
    def test_estimates_are_stated_unavailable_not_omitted(
        self, session_root: Path
    ) -> None:
        directory = sfx.write_session(session_root, "synthetic-unavailable")
        read = read_session(directory, mode=REPLAY)
        joined = " ".join(read.summary.unavailable_statements)
        for expected in (
            "Engagement estimate: Unavailable",
            "Cognitive-load estimate: Unavailable",
            "Signal-quality report: Unavailable",
            "Abstention decision: Unavailable",
        ):
            assert expected in joined

    def test_the_statements_are_the_module_constants(self) -> None:
        assert len(UNAVAILABLE_STATEMENTS) == 5


class TestReadingChangesNothing:
    def test_no_source_file_is_modified_by_a_read(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-untouched")
        before = sfx.directory_digests(directory)
        read_session(directory, mode=REPLAY)
        read_session(directory, mode=LIVE)
        build_session_catalogue(session_root, mode=REPLAY)
        assert sfx.directory_digests(directory) == before

    def test_no_file_is_created_by_a_read(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-nocreate")
        before = sorted(p.name for p in directory.iterdir())
        read_session(directory, mode=REPLAY)
        assert sorted(p.name for p in directory.iterdir()) == before

    def test_two_reads_of_one_recording_agree_exactly(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-stable")
        first = read_session(directory, mode=REPLAY)
        second = read_session(directory, mode=REPLAY)
        assert first.summary == second.summary
        assert first.records == second.records


class TestChecksumsAndHelpers:
    def test_checksums_cover_every_present_file(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-digest")
        names = {name for name, _ in file_checksums(directory)}
        assert names == {
            "manifest.json",
            "events.jsonl",
            "summary.json",
            "dropped.jsonl",
        }

    def test_checksums_skip_an_absent_file(self, session_root: Path) -> None:
        directory = sfx.write_session(
            session_root, "synthetic-nosummary", with_summary=False, completed=False
        )
        names = {name for name, _ in file_checksums(directory)}
        assert "summary.json" not in names

    def test_readable_session_id_uses_the_stores_allowlist(self) -> None:
        assert readable_session_id("synthetic-session-01") is True
        assert readable_session_id("../escape") is False
        assert readable_session_id(".hidden") is False
        assert readable_session_id("") is False

    def test_read_json_document_reports_the_path_and_the_reason(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "manifest.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SessionReadError, match="not valid JSON"):
            read_json_document(path)

    def test_read_json_document_refuses_a_non_object(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(SessionReadError, match="not a JSON object"):
            read_json_document(path)

    def test_decoding_no_lines_produces_no_records(self) -> None:
        assert decode_records([]) == ()

    def test_an_absent_stream_is_reported_with_a_reason(self, tmp_path: Path) -> None:
        snapshot = read_stream(tmp_path / "events.jsonl")
        assert snapshot.stream_present is False
        assert snapshot.unavailable_reason
        assert "not present" in snapshot.unavailable_reason

    def test_an_empty_stream_is_present_with_no_records(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        path.write_text("", encoding="utf-8")
        snapshot = read_stream(path)
        assert snapshot.stream_present is True
        assert snapshot.complete_line_count == 0
        assert snapshot.partial_trailing_line is False


class TestCatalogueLimits:
    def test_a_record_limit_is_stated_not_silent(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-limited")
        read = read_session(directory, mode=REPLAY, max_records=2)
        assert read.summary.parsed_record_count == 2
        assert read.summary.unparsed_record_count > 0
        assert any("were not read" in w.message for w in read.summary.warnings)

    def test_the_full_count_survives_a_limit(self, session_root: Path) -> None:
        directory = sfx.write_session(session_root, "synthetic-limited")
        full = read_session(directory, mode=REPLAY)
        limited = read_session(directory, mode=REPLAY, max_records=1)
        assert limited.summary.complete_record_count == (
            full.summary.complete_record_count
        )

    def test_the_catalogue_is_sorted_by_name(self, session_root: Path) -> None:
        for name in ("synthetic-c", "synthetic-a", "synthetic-b"):
            sfx.write_session(session_root, name)
        catalogue = build_session_catalogue(session_root, mode=REPLAY)
        assert catalogue.session_ids() == (
            "synthetic-a",
            "synthetic-b",
            "synthetic-c",
        )

    def test_find_returns_the_named_session(self, session_root: Path) -> None:
        sfx.write_session(session_root, "synthetic-findable")
        catalogue = build_session_catalogue(session_root, mode=REPLAY)
        assert catalogue.find("synthetic-findable") is not None
        assert catalogue.find("synthetic-absent") is None
