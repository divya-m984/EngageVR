"""Protocol envelope, payload, versioning, and validation tests.

None of these tests requires a webcam, a model asset, a display server,
network access, a running server, Unity, or any participant data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from engagevr.protocol import (
    ACCEPTED_MAJOR_VERSIONS,
    CRITICAL_MESSAGE_TYPES,
    PAYLOAD_MODELS,
    PROTOCOL_VERSION,
    REPLAY_LABEL,
    SYNTHETIC_LABEL,
    AcknowledgementPayload,
    AdaptationAcknowledgementPayload,
    AdaptationCommandName,
    AdaptationCommandPayload,
    ClientHelloPayload,
    ClientRole,
    HeartbeatAcknowledgementPayload,
    MessageEnvelope,
    MessageProvenance,
    MessageSource,
    MessageType,
    ProtocolErrorCode,
    ProtocolValidationError,
    ProtocolVersionError,
    ReplayMetadata,
    ServerHelloPayload,
    SessionStartPayload,
    TaskEventPayload,
    build_envelope,
    decode_envelope,
    decode_json,
    decode_message,
    is_critical,
    is_supported_version,
    parse_protocol_version,
    render_protocol_json_schema,
    require_supported_version,
)
from engagevr.protocol.json_schema import (
    SCHEMA_RELATIVE_PATH,
    build_protocol_json_schema,
)
from engagevr.schemas.events import EventType, ResponseOutcome, TaskEventDetail
from engagevr.schemas.session import DataSource

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPOSITORY_ROOT / "protocol" / "fixtures"


def default_payload(message_type: MessageType) -> object:
    """A minimal valid payload for each message type.

    Used so a test envelope of any type passes full payload validation,
    not just envelope validation.
    """
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    match message_type:
        case MessageType.CLIENT_HELLO:
            return ClientHelloPayload(
                role=ClientRole.SIMULATOR,
                client_name="test",
                client_version="1",
                protocol_version=PROTOCOL_VERSION,
            )
        case MessageType.SERVER_HELLO:
            return ServerHelloPayload(
                accepted=True,
                server_name="s",
                server_version="1",
                protocol_version=PROTOCOL_VERSION,
                session_id="test-session",
                assigned_client_id="c",
                server_time_utc=now,
                heartbeat_interval_seconds=10.0,
                connection_timeout_seconds=30.0,
                maximum_message_bytes=1024,
            )
        case MessageType.SESSION_START:
            return SessionStartPayload(
                participant_id="p",
                task_id="t",
                started_at_utc=now,
                blocks=1,
                trials_per_block=1,
                difficulty_level=1,
            )
        case MessageType.SESSION_END:
            from engagevr.protocol.messages import SessionEndPayload

            return SessionEndPayload(
                ended_at_utc=now, completed=True, reason="task_completed"
            )
        case MessageType.TASK_EVENT:
            return TaskEventPayload(
                event=TaskEventDetail(event_type=EventType.TRIAL_STARTED, trial_id=0)
            )
        case MessageType.TASK_STATE:
            from engagevr.protocol.messages import TaskState, TaskStatePayload

            return TaskStatePayload(state=TaskState.RUNNING)
        case MessageType.TELEMETRY:
            from engagevr.protocol.messages import TelemetryPayload

            return TelemetryPayload(component="test", metrics={"n": 1})
        case MessageType.ADAPTATION_COMMAND:
            return AdaptationCommandPayload(
                command_id="cmd-1",
                command=AdaptationCommandName.PAUSE_TASK,
                reason="test",
                issued_at_utc=now,
            )
        case MessageType.ADAPTATION_ACKNOWLEDGEMENT:
            return AdaptationAcknowledgementPayload(
                command_id="cmd-1", accepted=True, applied_at_utc=now
            )
        case MessageType.HEARTBEAT:
            from engagevr.protocol.messages import HeartbeatPayload

            return HeartbeatPayload(heartbeat_id="hb-1", client_monotonic_seconds=1.0)
        case MessageType.HEARTBEAT_ACKNOWLEDGEMENT:
            return HeartbeatAcknowledgementPayload(
                heartbeat_id="hb-1",
                client_monotonic_seconds=1.0,
                server_received_at_utc=now,
                server_sent_at_utc=now,
            )
        case MessageType.REPLAY_CONTROL:
            from engagevr.protocol.messages import ReplayAction, ReplayControlPayload

            return ReplayControlPayload(
                action=ReplayAction.START, source_session_id="src"
            )
        case MessageType.ACKNOWLEDGEMENT:
            return AcknowledgementPayload(
                acknowledged_message_id="m",
                acknowledged_message_type=MessageType.TASK_EVENT,
                acknowledged_sequence_number=0,
                server_received_at_utc=now,
                stored=True,
            )
        case MessageType.PROTOCOL_ERROR:
            from engagevr.protocol.messages import ProtocolErrorPayload

            return ProtocolErrorPayload(
                error_code=ProtocolErrorCode.INTERNAL_ERROR, detail="test"
            )
    raise AssertionError(f"no default payload for {message_type!r}")


def make_envelope(
    *,
    message_type: MessageType = MessageType.TASK_EVENT,
    session_id: str = "test-session",
    source: MessageSource = MessageSource.PYTHON_SIMULATOR,
    sequence_number: int = 0,
    payload: object | None = None,
) -> MessageEnvelope:
    """A valid envelope with sensible defaults, for reuse across tests."""
    body = payload if payload is not None else default_payload(message_type)
    return build_envelope(
        message_type=message_type,
        session_id=session_id,
        source=source,
        sequence_number=sequence_number,
        payload=body,  # type: ignore[arg-type]
        sent_at_utc=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        sent_at_monotonic_seconds=100.0,
    )


# --- versioning ------------------------------------------------------------


class TestProtocolVersion:
    def test_current_version_is_supported(self) -> None:
        assert is_supported_version(PROTOCOL_VERSION)
        assert parse_protocol_version(PROTOCOL_VERSION)[0] in ACCEPTED_MAJOR_VERSIONS

    def test_newer_minor_version_is_accepted(self) -> None:
        assert is_supported_version("1.99")

    def test_different_major_version_is_rejected(self) -> None:
        assert not is_supported_version("2.0")
        with pytest.raises(ProtocolVersionError, match="unsupported protocol major"):
            require_supported_version("2.0")

    @pytest.mark.parametrize("value", ["", "1", "1.", ".1", "one.two", "1.2.3", "v1.0"])
    def test_malformed_versions_are_rejected(self, value: str) -> None:
        assert not is_supported_version(value)
        with pytest.raises(ProtocolVersionError):
            parse_protocol_version(value)


# --- envelope --------------------------------------------------------------


class TestEnvelope:
    def test_valid_envelope_round_trips(self) -> None:
        envelope = make_envelope()
        raw = json.dumps(envelope.to_json_dict())
        decoded = decode_message(raw)

        assert decoded.envelope.message_id == envelope.message_id
        assert decoded.envelope.sequence_number == 0
        assert decoded.message_type is MessageType.TASK_EVENT
        assert isinstance(decoded.payload, TaskEventPayload)

    def test_negative_sequence_number_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MessageEnvelope(
                message_type=MessageType.HEARTBEAT,
                session_id="s",
                source=MessageSource.BACKEND,
                sequence_number=-1,
                sent_at_utc=datetime.now(UTC),
                sent_at_monotonic_seconds=0.0,
            )

    def test_naive_timestamp_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            MessageEnvelope(
                message_type=MessageType.HEARTBEAT,
                session_id="s",
                source=MessageSource.BACKEND,
                sequence_number=0,
                sent_at_utc=datetime(2026, 1, 1, 12, 0, 0),
                sent_at_monotonic_seconds=0.0,
            )

    def test_non_utc_offset_is_accepted_and_preserved(self) -> None:
        """A tz-aware non-UTC timestamp is unambiguous, so it is kept."""
        offset = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone_plus_two())
        envelope = MessageEnvelope(
            message_type=MessageType.HEARTBEAT,
            session_id="s",
            source=MessageSource.BACKEND,
            sequence_number=0,
            sent_at_utc=offset,
            sent_at_monotonic_seconds=0.0,
        )
        assert envelope.sent_at_utc.utcoffset() == timedelta(hours=2)

    def test_unknown_envelope_field_is_rejected(self) -> None:
        raw = make_envelope().to_json_dict()
        raw["surprise"] = 1
        with pytest.raises(ProtocolValidationError) as info:
            decode_envelope(raw)
        assert info.value.error_code is ProtocolErrorCode.INVALID_ENVELOPE

    def test_monotonic_timestamp_is_preserved_verbatim(self) -> None:
        envelope = make_envelope()
        decoded = decode_message(json.dumps(envelope.to_json_dict()))
        assert decoded.envelope.sent_at_monotonic_seconds == 100.0


class TestProvenance:
    def test_synthetic_requires_the_label(self) -> None:
        with pytest.raises(ValidationError, match="SYNTHETIC"):
            MessageProvenance(data_source=DataSource.SYNTHETIC, synthetic_label=None)

    def test_live_must_not_carry_the_label(self) -> None:
        with pytest.raises(ValidationError, match="must be null"):
            MessageProvenance(
                data_source=DataSource.LIVE, synthetic_label=SYNTHETIC_LABEL
            )

    def test_default_provenance_is_labelled_synthetic(self) -> None:
        provenance = MessageProvenance()
        assert provenance.data_source is DataSource.SYNTHETIC
        assert provenance.synthetic_label == SYNTHETIC_LABEL


class TestReplayMetadata:
    def test_adding_replay_metadata_does_not_mutate_the_original(self) -> None:
        original = make_envelope()
        metadata = ReplayMetadata(
            source_session_id="a",
            replay_session_id="b",
            replay_index=0,
            replay_speed=0.0,
            replayed_at_utc=datetime.now(UTC),
        )
        replayed = original.with_replay_metadata(metadata)

        assert original.replay is None
        assert replayed.replay is not None
        assert replayed.replay.replay_label == REPLAY_LABEL
        assert replayed.source is original.source
        assert replayed.sequence_number == original.sequence_number
        assert replayed.sent_at_utc == original.sent_at_utc
        assert replayed.message_id == original.message_id

    def test_replay_label_cannot_be_changed(self) -> None:
        with pytest.raises(ValidationError, match="replay_label"):
            ReplayMetadata(
                replay_label="LIVE",
                source_session_id="a",
                replay_session_id="b",
                replay_index=0,
                replay_speed=0.0,
                replayed_at_utc=datetime.now(UTC),
            )


# --- validation pipeline ---------------------------------------------------


class TestValidation:
    def test_unsupported_version_is_reported_as_such(self) -> None:
        raw = make_envelope().to_json_dict()
        raw["protocol_version"] = "2.0"
        with pytest.raises(ProtocolValidationError) as info:
            decode_envelope(raw)
        assert info.value.error_code is ProtocolErrorCode.UNSUPPORTED_PROTOCOL_VERSION
        assert info.value.fatal is True

    def test_unknown_message_type_lists_the_known_ones(self) -> None:
        raw = make_envelope().to_json_dict()
        raw["message_type"] = "telepathy"
        with pytest.raises(ProtocolValidationError) as info:
            decode_envelope(raw)
        assert info.value.error_code is ProtocolErrorCode.UNKNOWN_MESSAGE_TYPE
        assert "task_event" in info.value.detail

    def test_invalid_payload_is_distinguished_from_invalid_envelope(self) -> None:
        raw = make_envelope().to_json_dict()
        raw["payload"] = {
            "event": {"event_type": "response_registered", "reaction_time_ms": -1.0}
        }
        with pytest.raises(ProtocolValidationError) as info:
            decode_message(json.dumps(raw))
        assert info.value.error_code is ProtocolErrorCode.INVALID_PAYLOAD

    def test_invalid_json_is_reported(self) -> None:
        with pytest.raises(ProtocolValidationError) as info:
            decode_json("{not json")
        assert info.value.error_code is ProtocolErrorCode.INVALID_JSON

    def test_non_object_top_level_is_rejected(self) -> None:
        with pytest.raises(ProtocolValidationError) as info:
            decode_json("[1, 2, 3]")
        assert info.value.error_code is ProtocolErrorCode.INVALID_ENVELOPE

    def test_oversized_message_is_rejected_before_parsing(self) -> None:
        with pytest.raises(ProtocolValidationError) as info:
            decode_json("{}" + " " * 2048, maximum_message_bytes=64)
        assert info.value.error_code is ProtocolErrorCode.MESSAGE_TOO_LARGE
        assert info.value.fatal is True

    def test_rejection_reason_is_preserved_verbatim(self) -> None:
        raw = make_envelope().to_json_dict()
        raw["message_type"] = "a_very_specific_wrong_name"
        with pytest.raises(ProtocolValidationError) as info:
            decode_envelope(raw)
        assert "a_very_specific_wrong_name" in info.value.detail
        assert info.value.message_type == "a_very_specific_wrong_name"

    def test_every_message_type_has_a_payload_model(self) -> None:
        assert set(PAYLOAD_MODELS) == set(MessageType)


class TestCriticality:
    def test_declared_critical_types(self) -> None:
        for message_type in CRITICAL_MESSAGE_TYPES:
            envelope = make_envelope(message_type=message_type)
            assert is_critical(envelope) is True

    def test_task_completed_is_critical(self) -> None:
        payload = TaskEventPayload(
            event=TaskEventDetail(event_type=EventType.TASK_COMPLETED)
        )
        envelope = make_envelope(message_type=MessageType.TASK_EVENT, payload=payload)
        assert is_critical(envelope, payload) is True
        # Also detectable from the raw envelope alone.
        assert is_critical(envelope) is True

    def test_ordinary_task_event_is_not_critical(self) -> None:
        envelope = make_envelope()
        assert is_critical(envelope) is False

    def test_heartbeat_is_not_critical(self) -> None:
        assert is_critical(make_envelope(message_type=MessageType.HEARTBEAT)) is False


# --- payload rules ---------------------------------------------------------


class TestPayloadRules:
    def test_set_difficulty_requires_an_integer(self) -> None:
        with pytest.raises(ValidationError, match="integer"):
            AdaptationCommandPayload(
                command_id="c",
                command=AdaptationCommandName.SET_DIFFICULTY,
                value="three",
                reason="r",
                issued_at_utc=datetime.now(UTC),
            )

    def test_pause_task_takes_no_value(self) -> None:
        with pytest.raises(ValidationError, match="takes no value"):
            AdaptationCommandPayload(
                command_id="c",
                command=AdaptationCommandName.PAUSE_TASK,
                value=1,
                reason="r",
                issued_at_utc=datetime.now(UTC),
            )

    def test_expiry_must_follow_issue(self) -> None:
        now = datetime.now(UTC)
        with pytest.raises(ValidationError, match="after issued_at_utc"):
            AdaptationCommandPayload(
                command_id="c",
                command=AdaptationCommandName.PAUSE_TASK,
                reason="r",
                issued_at_utc=now,
                expires_at_utc=now - timedelta(seconds=1),
            )

    def test_command_cannot_target_an_observer(self) -> None:
        with pytest.raises(ValidationError, match="task client role"):
            AdaptationCommandPayload(
                command_id="c",
                command=AdaptationCommandName.PAUSE_TASK,
                reason="r",
                issued_at_utc=datetime.now(UTC),
                target_role=ClientRole.OBSERVER,
            )

    def test_rejected_acknowledgement_needs_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="rejection_reason"):
            AdaptationAcknowledgementPayload(command_id="c", accepted=False)

    def test_accepted_acknowledgement_must_not_carry_a_rejection(self) -> None:
        with pytest.raises(ValidationError, match="must not carry"):
            AdaptationAcknowledgementPayload(
                command_id="c", accepted=True, rejection_reason="why"
            )

    def test_acknowledgement_cannot_be_stored_and_dropped(self) -> None:
        with pytest.raises(ValidationError, match="both stored and dropped"):
            AcknowledgementPayload(
                acknowledged_message_id="m",
                acknowledged_message_type=MessageType.TASK_EVENT,
                acknowledged_sequence_number=0,
                server_received_at_utc=datetime.now(UTC),
                stored=True,
                dropped=True,
            )

    def test_rejected_server_hello_needs_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="rejection_reason"):
            ServerHelloPayload(
                accepted=False,
                server_name="s",
                server_version="1",
                protocol_version=PROTOCOL_VERSION,
                session_id="s",
                assigned_client_id="c",
                server_time_utc=datetime.now(UTC),
                heartbeat_interval_seconds=10.0,
                connection_timeout_seconds=30.0,
                maximum_message_bytes=1024,
            )

    def test_heartbeat_acknowledgement_echoes_client_monotonic(self) -> None:
        reply = HeartbeatAcknowledgementPayload(
            heartbeat_id="h",
            client_monotonic_seconds=42.5,
            server_received_at_utc=datetime.now(UTC),
            server_sent_at_utc=datetime.now(UTC),
        )
        assert reply.client_monotonic_seconds == 42.5

    def test_payloads_are_closed_to_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ClientHelloPayload(
                role=ClientRole.SIMULATOR,
                client_name="c",
                client_version="1",
                protocol_version=PROTOCOL_VERSION,
                engagement_estimate=0.8,  # type: ignore[call-arg]
            )

    def test_session_start_participant_is_pseudonymous_only(self) -> None:
        payload = SessionStartPayload(
            participant_id="p001",
            task_id="t",
            started_at_utc=datetime.now(UTC),
            blocks=1,
            trials_per_block=1,
            difficulty_level=1,
        )
        # There is no field on this model that can hold a name, email, or
        # other real-world identifier.
        assert payload.participant_id == "p001"
        assert not {"name", "email", "full_name", "date_of_birth"} & set(
            SessionStartPayload.model_fields
        )


# --- JSON Schema and shared fixtures ---------------------------------------


class TestJsonSchema:
    def test_schema_is_generated_with_a_branch_per_message_type(self) -> None:
        document = build_protocol_json_schema()
        assert document["x-protocol-version"] == PROTOCOL_VERSION
        assert len(document["oneOf"]) == len(MessageType)
        titles = {branch["title"] for branch in document["oneOf"]}
        assert titles == {t.value for t in MessageType}

    def test_schema_generation_is_deterministic(self) -> None:
        assert render_protocol_json_schema() == render_protocol_json_schema()

    def test_checked_in_schema_matches_the_models(self) -> None:
        path = REPOSITORY_ROOT / SCHEMA_RELATIVE_PATH
        assert path.is_file(), f"{path} is missing"
        assert path.read_text(encoding="utf-8") == render_protocol_json_schema(), (
            "the checked-in protocol schema is out of date; regenerate it with "
            "'uv run python scripts/generate_protocol_artifacts.py'"
        )


class TestSharedFixtures:
    """The fixtures the Unity C# tests also parse."""

    @staticmethod
    def index() -> dict[str, object]:
        return json.loads((FIXTURE_ROOT / "index.json").read_text(encoding="utf-8"))

    def test_index_exists_and_names_the_protocol_version(self) -> None:
        assert self.index()["protocol_version"] == PROTOCOL_VERSION

    def test_every_message_type_has_a_valid_fixture(self) -> None:
        covered = set()
        for name in self.index()["valid"]:  # type: ignore[union-attr]
            raw = json.loads((FIXTURE_ROOT / "valid" / name).read_text())
            covered.add(raw["message_type"])
        assert covered == {t.value for t in MessageType}

    def test_every_valid_fixture_decodes(self) -> None:
        for name in self.index()["valid"]:  # type: ignore[union-attr]
            path = FIXTURE_ROOT / "valid" / name
            decoded = decode_message(path.read_bytes())
            assert decoded.envelope.protocol_version == PROTOCOL_VERSION

    def test_every_invalid_fixture_is_rejected_with_its_declared_code(self) -> None:
        for case in self.index()["invalid"]:  # type: ignore[union-attr]
            path = FIXTURE_ROOT / "invalid" / case["file"]
            with pytest.raises(ProtocolValidationError) as info:
                decode_message(path.read_bytes())
            assert info.value.error_code.value == case["expected_error_code"], (
                f"{case['file']} produced {info.value.error_code.value}"
            )

    def test_fixtures_round_trip_byte_stably(self) -> None:
        """A decoded fixture re-serializes to the same JSON object."""
        for name in self.index()["valid"]:  # type: ignore[union-attr]
            path = FIXTURE_ROOT / "valid" / name
            original = json.loads(path.read_text())
            decoded = decode_message(path.read_bytes())
            assert decoded.envelope.to_json_dict() == original, name

    def test_fixtures_are_unity_compatible_json(self) -> None:
        """No construct Unity's C# reader cannot represent.

        Specifically: no NaN, no Infinity, no integer outside the double
        exact range, and no non-string object key.
        """
        for name in self.index()["valid"]:  # type: ignore[union-attr]
            text = (FIXTURE_ROOT / "valid" / name).read_text()
            assert "NaN" not in text, name
            assert "Infinity" not in text, name
            # json.loads with a strict parser rejects NaN/Infinity too.
            json.loads(text, parse_constant=_reject_constant)

    def test_replayed_synthetic_fixture_carries_both_labels(self) -> None:
        decoded = decode_message(
            (FIXTURE_ROOT / "valid" / "replayed-synthetic-task-event.json").read_bytes()
        )
        assert decoded.envelope.provenance.synthetic_label == SYNTHETIC_LABEL
        assert decoded.envelope.replay is not None
        assert decoded.envelope.replay.replay_label == REPLAY_LABEL

    def test_no_fixture_contains_engagement_or_frame_data(self) -> None:
        forbidden = (
            "engagement",
            "cognitive_load",
            "attention",
            "fatigue",
            "frame_data",
            "image",
            "landmark",
            "pixels",
        )
        for name in self.index()["valid"]:  # type: ignore[union-attr]
            text = (FIXTURE_ROOT / "valid" / name).read_text().lower()
            for token in forbidden:
                assert token not in text, f"{name} contains {token!r}"


def _reject_constant(value: str) -> object:
    raise AssertionError(f"JSON constant {value!r} is not representable in Unity")


def timezone_plus_two() -> object:
    from datetime import timezone

    return timezone(timedelta(hours=2))


# --- task event schema -----------------------------------------------------


class TestTaskEventDetail:
    def test_reaction_time_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            TaskEventDetail(
                event_type=EventType.RESPONSE_REGISTERED, reaction_time_ms=-0.001
            )

    def test_zero_reaction_time_is_permitted(self) -> None:
        detail = TaskEventDetail(
            event_type=EventType.RESPONSE_REGISTERED, reaction_time_ms=0.0
        )
        assert detail.reaction_time_ms == 0.0

    def test_missing_response_stays_none(self) -> None:
        detail = TaskEventDetail(event_type=EventType.TRIAL_COMPLETED)
        assert detail.observed_response is None
        assert detail.reaction_time_ms is None
        assert detail.response_correct is None

    def test_timeout_is_distinct_from_incorrect(self) -> None:
        timeout = TaskEventDetail(
            event_type=EventType.RESPONSE_TIMEOUT,
            response_outcome=ResponseOutcome.TIMEOUT,
        )
        incorrect = TaskEventDetail(
            event_type=EventType.RESPONSE_REGISTERED,
            response_outcome=ResponseOutcome.INCORRECT,
            response_correct=False,
            observed_response="k",
            reaction_time_ms=300.0,
        )
        assert timeout.response_outcome is not incorrect.response_outcome
        assert timeout.response_correct is None
        assert incorrect.response_correct is False

    def test_timeout_may_not_carry_a_reaction_time(self) -> None:
        with pytest.raises(ValidationError, match="reaction_time_ms"):
            TaskEventDetail(
                event_type=EventType.RESPONSE_TIMEOUT, reaction_time_ms=100.0
            )

    def test_timeout_may_not_carry_an_observed_response(self) -> None:
        with pytest.raises(ValidationError, match="observed_response"):
            TaskEventDetail(
                event_type=EventType.RESPONSE_TIMEOUT, observed_response="j"
            )

    def test_timeout_has_no_correctness(self) -> None:
        with pytest.raises(ValidationError, match="no correctness"):
            TaskEventDetail(
                event_type=EventType.TRIAL_COMPLETED,
                response_outcome=ResponseOutcome.TIMEOUT,
                response_correct=False,
            )

    def test_non_task_event_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not a task event"):
            TaskEventDetail(event_type=EventType.SESSION_STARTED)

    def test_all_thirteen_vocabulary_members_are_accepted(self) -> None:
        from engagevr.schemas.events import TASK_EVENT_TYPES

        assert len(TASK_EVENT_TYPES) == 13
        for event_type in TASK_EVENT_TYPES:
            TaskEventDetail(event_type=event_type)

    def test_identifiers_survive_for_later_windowing(self) -> None:
        detail = TaskEventDetail(
            event_type=EventType.RESPONSE_REGISTERED,
            task_id="t",
            block_id=1,
            trial_id=2,
            stimulus_id="s",
            task_elapsed_ms=100.0,
            trial_elapsed_ms=50.0,
        )
        # Every field needed to join this event onto a synchronized
        # feature window is present on the event itself.
        assert (detail.task_id, detail.block_id, detail.trial_id) == ("t", 1, 2)
        assert detail.task_elapsed_ms == 100.0

    def test_legacy_task_event_still_works_and_can_carry_a_detail(self) -> None:
        from engagevr.schemas.events import TaskEvent

        detail = TaskEventDetail(
            event_type=EventType.RESPONSE_REGISTERED,
            trial_id=3,
            difficulty_level=2,
            response_correct=True,
            reaction_time_ms=250.0,
        )
        event = TaskEvent.from_detail(
            session_id="s", monotonic_timestamp=1.0, detail=detail
        )
        assert event.trial_index == 3
        assert event.correct is True
        assert event.reaction_time_ms == 250.0
        assert event.detail is detail
