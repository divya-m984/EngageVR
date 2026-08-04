#!/usr/bin/env python
"""Regenerate the checked-in protocol JSON Schema and contract fixtures.

The fixtures under ``protocol/fixtures/`` are the **shared contract**
between the Python implementation and the Unity C# client: both parse
the same files in their own test suites, so a field rename on one side
fails the other side's tests.

Run::

    uv run python scripts/generate_protocol_artifacts.py

A test asserts that the checked-in files match what this script would
produce, so the artefacts cannot silently drift from the models.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engagevr.protocol.envelope import (
    SYNTHETIC_LABEL,
    MessageProvenance,
    ReplayMetadata,
    build_envelope,
)
from engagevr.protocol.json_schema import (
    SCHEMA_RELATIVE_PATH,
    render_protocol_json_schema,
)
from engagevr.protocol.messages import (
    AcknowledgementPayload,
    AdaptationAcknowledgementPayload,
    AdaptationCommandName,
    AdaptationCommandPayload,
    ClientHelloPayload,
    ClientRole,
    HeartbeatAcknowledgementPayload,
    HeartbeatPayload,
    MessageSource,
    MessageType,
    ProtocolErrorCode,
    ProtocolErrorPayload,
    ReplayAction,
    ReplayControlPayload,
    ServerHelloPayload,
    SessionEndPayload,
    SessionStartPayload,
    TaskEventPayload,
    TaskState,
    TaskStatePayload,
    TelemetryPayload,
)
from engagevr.protocol.version import PROTOCOL_VERSION
from engagevr.schemas.events import EventType, ResponseOutcome, TaskEventDetail
from engagevr.schemas.session import DataSource

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "protocol" / "fixtures"
VALID_ROOT = FIXTURE_ROOT / "valid"
INVALID_ROOT = FIXTURE_ROOT / "invalid"
INDEX_PATH = FIXTURE_ROOT / "index.json"

#: Fixtures are fully deterministic: fixed ids, fixed timestamps, fixed
#: monotonic readings. A regenerated fixture set is byte-identical, so a
#: diff means the protocol changed, not that the clock moved.
FIXED_UTC = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
FIXED_MONOTONIC = 1000.0
SESSION_ID = "fixture-session"

SYNTHETIC_PROVENANCE = MessageProvenance(
    data_source=DataSource.SYNTHETIC,
    synthetic_label=SYNTHETIC_LABEL,
    producer="engagevr.protocol.fixtures",
)
BACKEND_PROVENANCE = MessageProvenance(
    data_source=DataSource.LIVE,
    synthetic_label=None,
    producer="engagevr.protocol.fixtures",
)


def _envelope(
    name: str,
    message_type: MessageType,
    payload: Any,
    *,
    source: MessageSource,
    sequence_number: int,
    provenance: MessageProvenance,
    correlation_id: str | None = None,
    replay: ReplayMetadata | None = None,
) -> tuple[str, dict[str, Any]]:
    envelope = build_envelope(
        message_type=message_type,
        session_id=SESSION_ID,
        source=source,
        sequence_number=sequence_number,
        payload=payload,
        provenance=provenance,
        correlation_id=correlation_id,
        sent_at_utc=FIXED_UTC,
        sent_at_monotonic_seconds=FIXED_MONOTONIC + sequence_number,
        message_id=f"fixture-{name}",
    )
    if replay is not None:
        envelope = envelope.with_replay_metadata(replay)
    return name, envelope.to_json_dict()


def build_valid_fixtures() -> list[tuple[str, dict[str, Any]]]:
    """One representative valid message per message type, plus variants."""
    fixtures: list[tuple[str, dict[str, Any]]] = []

    fixtures.append(
        _envelope(
            "client-hello",
            MessageType.CLIENT_HELLO,
            ClientHelloPayload(
                role=ClientRole.UNITY,
                client_name="engagevr-unity-client",
                client_version="0.1.0",
                protocol_version=PROTOCOL_VERSION,
                capabilities=["task_event", "adaptation_acknowledgement"],
            ),
            source=MessageSource.UNITY_CLIENT,
            sequence_number=0,
            provenance=SYNTHETIC_PROVENANCE,
        )
    )
    fixtures.append(
        _envelope(
            "server-hello",
            MessageType.SERVER_HELLO,
            ServerHelloPayload(
                accepted=True,
                server_name="engagevr-backend",
                server_version="0.1.0",
                protocol_version=PROTOCOL_VERSION,
                session_id=SESSION_ID,
                assigned_client_id="fixture-client",
                server_time_utc=FIXED_UTC,
                heartbeat_interval_seconds=10.0,
                connection_timeout_seconds=30.0,
                maximum_message_bytes=262144,
            ),
            source=MessageSource.BACKEND,
            sequence_number=0,
            provenance=BACKEND_PROVENANCE,
            correlation_id="fixture-client-hello",
        )
    )
    fixtures.append(
        _envelope(
            "session-start",
            MessageType.SESSION_START,
            SessionStartPayload(
                participant_id="synthetic_participant",
                task_id="reaction_task_v1",
                started_at_utc=FIXED_UTC,
                blocks=2,
                trials_per_block=10,
                difficulty_level=1,
                configuration={
                    "seed": 42,
                    "stimulus_duration_ms": 800.0,
                    "synthetic_label": SYNTHETIC_LABEL,
                },
            ),
            source=MessageSource.UNITY_CLIENT,
            sequence_number=1,
            provenance=SYNTHETIC_PROVENANCE,
        )
    )
    fixtures.append(
        _envelope(
            "task-event-stimulus-presented",
            MessageType.TASK_EVENT,
            TaskEventPayload(
                event=TaskEventDetail(
                    event_type=EventType.STIMULUS_PRESENTED,
                    task_id="reaction_task_v1",
                    block_id=0,
                    trial_id=3,
                    stimulus_id="square-b0t3",
                    stimulus_category="square",
                    expected_response="j",
                    difficulty_level=1,
                    task_elapsed_ms=4200.0,
                    trial_elapsed_ms=500.0,
                )
            ),
            source=MessageSource.UNITY_CLIENT,
            sequence_number=2,
            provenance=SYNTHETIC_PROVENANCE,
        )
    )
    fixtures.append(
        _envelope(
            "task-event-response-registered",
            MessageType.TASK_EVENT,
            TaskEventPayload(
                event=TaskEventDetail(
                    event_type=EventType.RESPONSE_REGISTERED,
                    task_id="reaction_task_v1",
                    block_id=0,
                    trial_id=3,
                    stimulus_id="square-b0t3",
                    stimulus_category="square",
                    expected_response="j",
                    observed_response="j",
                    response_correct=True,
                    response_outcome=ResponseOutcome.CORRECT,
                    reaction_time_ms=412.5,
                    difficulty_level=1,
                    task_elapsed_ms=4612.5,
                    trial_elapsed_ms=912.5,
                )
            ),
            source=MessageSource.UNITY_CLIENT,
            sequence_number=3,
            provenance=SYNTHETIC_PROVENANCE,
        )
    )
    fixtures.append(
        _envelope(
            "task-event-response-timeout",
            MessageType.TASK_EVENT,
            TaskEventPayload(
                event=TaskEventDetail(
                    event_type=EventType.RESPONSE_TIMEOUT,
                    task_id="reaction_task_v1",
                    block_id=0,
                    trial_id=4,
                    stimulus_id="circle-b0t4",
                    stimulus_category="circle",
                    expected_response="k",
                    response_outcome=ResponseOutcome.TIMEOUT,
                    difficulty_level=1,
                    task_elapsed_ms=6112.5,
                    trial_elapsed_ms=1500.0,
                )
            ),
            source=MessageSource.UNITY_CLIENT,
            sequence_number=4,
            provenance=SYNTHETIC_PROVENANCE,
        )
    )
    fixtures.append(
        _envelope(
            "task-state",
            MessageType.TASK_STATE,
            TaskStatePayload(
                state=TaskState.RUNNING,
                task_id="reaction_task_v1",
                block_id=0,
                trial_id=4,
                difficulty_level=1,
                stimulus_interval_ms=500.0,
            ),
            source=MessageSource.UNITY_CLIENT,
            sequence_number=5,
            provenance=SYNTHETIC_PROVENANCE,
        )
    )
    fixtures.append(
        _envelope(
            "telemetry",
            MessageType.TELEMETRY,
            TelemetryPayload(
                component="unity_client",
                metrics={"frames_per_second": 59.8, "dropped_frames": 0},
            ),
            source=MessageSource.UNITY_CLIENT,
            sequence_number=6,
            provenance=SYNTHETIC_PROVENANCE,
        )
    )
    fixtures.append(
        _envelope(
            "adaptation-command-set-difficulty",
            MessageType.ADAPTATION_COMMAND,
            AdaptationCommandPayload(
                command_id="cmd-0001",
                command=AdaptationCommandName.SET_DIFFICULTY,
                value=3,
                reason="manual operator request during a transport test",
                issued_at_utc=FIXED_UTC,
                target_role=ClientRole.UNITY,
                is_manual=True,
            ),
            source=MessageSource.BACKEND,
            sequence_number=1,
            provenance=BACKEND_PROVENANCE,
        )
    )
    fixtures.append(
        _envelope(
            "adaptation-command-pause",
            MessageType.ADAPTATION_COMMAND,
            AdaptationCommandPayload(
                command_id="cmd-0002",
                command=AdaptationCommandName.PAUSE_TASK,
                reason="manual operator request",
                issued_at_utc=FIXED_UTC,
                target_role=ClientRole.UNITY,
            ),
            source=MessageSource.BACKEND,
            sequence_number=2,
            provenance=BACKEND_PROVENANCE,
        )
    )
    fixtures.append(
        _envelope(
            "adaptation-acknowledgement-accepted",
            MessageType.ADAPTATION_ACKNOWLEDGEMENT,
            AdaptationAcknowledgementPayload(
                command_id="cmd-0001",
                accepted=True,
                applied_at_utc=FIXED_UTC,
            ),
            source=MessageSource.UNITY_CLIENT,
            sequence_number=7,
            provenance=SYNTHETIC_PROVENANCE,
            correlation_id="fixture-adaptation-command-set-difficulty",
        )
    )
    fixtures.append(
        _envelope(
            "adaptation-acknowledgement-rejected",
            MessageType.ADAPTATION_ACKNOWLEDGEMENT,
            AdaptationAcknowledgementPayload(
                command_id="cmd-0002",
                accepted=False,
                rejection_reason=(
                    "pause_task requires state 'running'; the task is 'idle'"
                ),
            ),
            source=MessageSource.UNITY_CLIENT,
            sequence_number=8,
            provenance=SYNTHETIC_PROVENANCE,
            correlation_id="fixture-adaptation-command-pause",
        )
    )
    fixtures.append(
        _envelope(
            "heartbeat",
            MessageType.HEARTBEAT,
            HeartbeatPayload(heartbeat_id="hb-0001", client_monotonic_seconds=1234.5),
            source=MessageSource.UNITY_CLIENT,
            sequence_number=9,
            provenance=SYNTHETIC_PROVENANCE,
        )
    )
    fixtures.append(
        _envelope(
            "heartbeat-acknowledgement",
            MessageType.HEARTBEAT_ACKNOWLEDGEMENT,
            HeartbeatAcknowledgementPayload(
                heartbeat_id="hb-0001",
                client_monotonic_seconds=1234.5,
                server_received_at_utc=FIXED_UTC,
                server_sent_at_utc=FIXED_UTC,
            ),
            source=MessageSource.BACKEND,
            sequence_number=3,
            provenance=BACKEND_PROVENANCE,
            correlation_id="fixture-heartbeat",
        )
    )
    fixtures.append(
        _envelope(
            "replay-control",
            MessageType.REPLAY_CONTROL,
            ReplayControlPayload(
                action=ReplayAction.START,
                source_session_id="recorded-session",
                speed=5.0,
            ),
            source=MessageSource.REPLAY,
            sequence_number=0,
            provenance=BACKEND_PROVENANCE,
        )
    )
    fixtures.append(
        _envelope(
            "acknowledgement",
            MessageType.ACKNOWLEDGEMENT,
            AcknowledgementPayload(
                acknowledged_message_id="fixture-task-event-stimulus-presented",
                acknowledged_message_type=MessageType.TASK_EVENT,
                acknowledged_sequence_number=2,
                server_received_at_utc=FIXED_UTC,
                stored=True,
            ),
            source=MessageSource.BACKEND,
            sequence_number=4,
            provenance=BACKEND_PROVENANCE,
            correlation_id="fixture-task-event-stimulus-presented",
        )
    )
    fixtures.append(
        _envelope(
            "protocol-error",
            MessageType.PROTOCOL_ERROR,
            ProtocolErrorPayload(
                error_code=ProtocolErrorCode.UNKNOWN_MESSAGE_TYPE,
                detail="unknown message_type 'not_a_real_type'",
                offending_message_id="fixture-bad",
                offending_message_type="not_a_real_type",
                offending_sequence_number=99,
                fatal=False,
            ),
            source=MessageSource.BACKEND,
            sequence_number=5,
            provenance=BACKEND_PROVENANCE,
        )
    )
    fixtures.append(
        _envelope(
            "session-end",
            MessageType.SESSION_END,
            SessionEndPayload(
                ended_at_utc=FIXED_UTC,
                completed=True,
                reason="task_completed",
            ),
            source=MessageSource.UNITY_CLIENT,
            sequence_number=10,
            provenance=SYNTHETIC_PROVENANCE,
        )
    )
    # A replayed synthetic message: SYNTHETIC in provenance AND REPLAY in
    # the replay block. This fixture exists so that both labels surviving
    # a round trip is a contract, not an implementation detail.
    fixtures.append(
        _envelope(
            "replayed-synthetic-task-event",
            MessageType.TASK_EVENT,
            TaskEventPayload(
                event=TaskEventDetail(
                    event_type=EventType.TRIAL_COMPLETED,
                    task_id="reaction_task_v1",
                    block_id=0,
                    trial_id=3,
                    difficulty_level=1,
                )
            ),
            source=MessageSource.PYTHON_SIMULATOR,
            sequence_number=11,
            provenance=SYNTHETIC_PROVENANCE,
            replay=ReplayMetadata(
                source_session_id="recorded-session",
                replay_session_id="replay-of-recorded-session",
                replay_index=42,
                replay_speed=5.0,
                replayed_at_utc=FIXED_UTC,
                original_arrival_index=42,
            ),
        )
    )
    return fixtures


def build_invalid_fixtures() -> list[tuple[str, str, dict[str, Any]]]:
    """Malformed messages, each paired with the error code it must produce."""
    base = dict(build_valid_fixtures()[3][1])  # a valid task_event

    def variant(**changes: Any) -> dict[str, Any]:
        copy = json.loads(json.dumps(base))
        for key, value in changes.items():
            if value is _DELETE:
                copy.pop(key, None)
            else:
                copy[key] = value
        return copy

    cases: list[tuple[str, str, dict[str, Any]]] = [
        (
            "unsupported-major-version",
            "unsupported_protocol_version",
            variant(protocol_version="2.0"),
        ),
        (
            "malformed-version",
            "unsupported_protocol_version",
            variant(protocol_version="one-point-oh"),
        ),
        (
            "unknown-message-type",
            "unknown_message_type",
            variant(message_type="telepathy"),
        ),
        (
            "missing-message-type",
            "invalid_envelope",
            variant(message_type=_DELETE),
        ),
        (
            "negative-sequence-number",
            "invalid_envelope",
            variant(sequence_number=-1),
        ),
        (
            "naive-timestamp",
            "invalid_envelope",
            variant(sent_at_utc="2026-01-01T12:00:00"),
        ),
        (
            "unknown-envelope-field",
            "invalid_envelope",
            variant(unexpected_field="rejected because the envelope is closed"),
        ),
        (
            "negative-reaction-time",
            "invalid_payload",
            variant(
                payload={
                    "event": {
                        "event_type": "response_registered",
                        "reaction_time_ms": -5.0,
                    }
                }
            ),
        ),
        (
            "timeout-with-reaction-time",
            "invalid_payload",
            variant(
                payload={
                    "event": {
                        "event_type": "response_timeout",
                        "reaction_time_ms": 300.0,
                    }
                }
            ),
        ),
        (
            "non-task-event-in-task-event",
            "invalid_payload",
            variant(payload={"event": {"event_type": "session_started"}}),
        ),
        (
            "unknown-payload-field",
            "invalid_payload",
            variant(
                payload={
                    "event": {"event_type": "trial_started"},
                    "extra": "payloads are closed too",
                }
            ),
        ),
        (
            "synthetic-without-label",
            "invalid_envelope",
            variant(
                provenance={
                    "data_source": "synthetic",
                    "synthetic_label": None,
                    "producer": "test",
                }
            ),
        ),
    ]
    return cases


class _Delete:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<delete>"


_DELETE = _Delete()


def _dump(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Write the schema and every fixture; return the index document."""
    (root / SCHEMA_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / SCHEMA_RELATIVE_PATH).write_text(
        render_protocol_json_schema(), encoding="utf-8"
    )

    valid = build_valid_fixtures()
    invalid = build_invalid_fixtures()

    for name, document in valid:
        _dump(root / "protocol" / "fixtures" / "valid" / f"{name}.json", document)
    for name, _code, document in invalid:
        _dump(root / "protocol" / "fixtures" / "invalid" / f"{name}.json", document)

    index = {
        "protocol_version": PROTOCOL_VERSION,
        "schema": str(SCHEMA_RELATIVE_PATH).replace("\\", "/"),
        "note": (
            "These fixtures are the shared contract between the Python "
            "implementation and the Unity C# client. Both test suites parse "
            "these exact files. Regenerate with "
            "'uv run python scripts/generate_protocol_artifacts.py'."
        ),
        "valid": [f"{name}.json" for name, _ in valid],
        "invalid": [
            {"file": f"{name}.json", "expected_error_code": code}
            for name, code, _ in invalid
        ],
    }
    _dump(root / "protocol" / "fixtures" / "index.json", index)
    return index


def main() -> int:
    index = generate()
    print(f"protocol version:   {index['protocol_version']}")
    print(f"schema:             {index['schema']}")
    print(f"valid fixtures:     {len(index['valid'])}")
    print(f"invalid fixtures:   {len(index['invalid'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
