"""Minimal session recordings for the live, replay, and report tests.

Written line by line rather than through
:class:`~engagevr.storage.session_store.SessionRecorder`, for two
reasons.  A test must be able to produce states a well-behaved recorder
never produces — a torn final line, a malformed interior line, a
duplicated sequence number — and the dashboard tests must not depend on
the writer they are asserting the dashboard never uses.

All fixture data is obviously synthetic.  Participant identifiers are of
the form ``synthetic-participant-0001``; there is no name, no email
address, no image, and nothing that could be mistaken for a real person.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engagevr.protocol.envelope import REPLAY_LABEL, SYNTHETIC_LABEL
from engagevr.protocol.version import PROTOCOL_VERSION
from engagevr.storage.manifest import RECORDING_DISCLAIMER, SESSION_FORMAT_VERSION

#: Pseudonymous, obviously software-generated participant labels.
PARTICIPANTS: tuple[str, ...] = (
    "synthetic-participant-0001",
    "synthetic-participant-0002",
)

#: A fixed base instant, so a recording is byte-identical run to run.
BASE_SECOND = 0


def manifest(session_id: str) -> dict[str, Any]:
    """A ``manifest.json`` carrying the fields the dashboard reads."""
    return {
        "session_format_version": SESSION_FORMAT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "session_id": session_id,
        "created_at_utc": "2026-01-01T00:00:00Z",
        "engagevr_version": "0.1.0",
        "configuration": {"task": {"blocks": 1, "trials_per_block": 2}},
        "disclaimer": RECORDING_DISCLAIMER,
    }


def summary(
    session_id: str,
    *,
    event_count: int,
    completed: bool = True,
    disconnect_reason: str | None = "orderly",
    recovered: bool = False,
    dropped: int = 0,
) -> dict[str, Any]:
    """A ``summary.json`` for a finished or interrupted recording."""
    return {
        "session_format_version": SESSION_FORMAT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "session_id": session_id,
        "event_count": event_count,
        "message_type_counts": {},
        "source_counts": {},
        "anomaly_counts": {},
        "dropped_message_count": dropped,
        "dropped_message_types": {},
        "completed": completed,
        "disconnect_reason": disconnect_reason,
        "recovered": recovered,
        "malformed_line_numbers": [],
        "synthetic_message_count": event_count,
        "replay_message_count": 0,
        "disclaimer": RECORDING_DISCLAIMER,
    }


def timestamp(offset: int) -> str:
    """A deterministic UTC timestamp, offset seconds from the base."""
    second = BASE_SECOND + offset
    minute, remainder = divmod(second, 60)
    return f"2026-01-01T00:{minute:02d}:{remainder:02d}Z"


def envelope(
    *,
    session_id: str,
    message_type: str,
    sequence_number: int,
    payload: dict[str, Any],
    source: str = "python_simulator",
    offset: int = 0,
    synthetic: bool = True,
    replayed: bool = False,
    data_source: str | None = None,
) -> dict[str, Any]:
    """One protocol envelope, as a JSON-compatible dict.

    ``data_source`` names one of the project's own
    :class:`~engagevr.schemas.session.DataSource` members and overrides
    the ``synthetic`` shorthand.  It exists so a test can build the
    ``public_dataset`` provenance no recording on this machine currently
    carries, without inventing a provenance string the protocol would
    reject.
    """
    recorded_source = data_source or ("synthetic" if synthetic else "live")
    is_synthetic = recorded_source == "synthetic"
    provenance = {
        "data_source": recorded_source,
        "synthetic_label": SYNTHETIC_LABEL if is_synthetic else None,
        "producer": "tests.unit.session_fixtures",
    }
    replay = (
        {
            "replay_label": REPLAY_LABEL,
            "source_session_id": f"{session_id}-origin",
            "replay_session_id": session_id,
            "replay_index": sequence_number,
            "replay_speed": 0.0,
            "replayed_at_utc": timestamp(offset),
            "original_arrival_index": sequence_number,
        }
        if replayed
        else None
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "message_id": f"{message_type}-{sequence_number:04d}",
        "message_type": message_type,
        "session_id": session_id,
        "source": source,
        "sequence_number": sequence_number,
        "sent_at_utc": timestamp(offset),
        "sent_at_monotonic_seconds": float(offset),
        "payload": payload,
        "correlation_id": None,
        "provenance": provenance,
        "replay": replay,
    }


def ingestion(
    *,
    arrival_index: int,
    offset: int = 0,
    anomalies: tuple[str, ...] = (),
    detail: str | None = None,
) -> dict[str, Any]:
    """Ingestion metadata for one stored record."""
    return {
        "arrival_index": arrival_index,
        "server_received_at_utc": timestamp(offset),
        "server_monotonic_seconds": float(offset),
        "transport": "in_process",
        "client_id": "synthetic-client-0001",
        "client_role": "simulator",
        "anomalies": list(anomalies),
        "anomaly_detail": detail,
        "expected_sequence_number": None,
        "apparent_transport_delay_seconds": None,
        "delay_unavailable_reason": None,
    }


def stored(envelope_body: dict[str, Any], ingestion_body: dict[str, Any]) -> str:
    """One line of ``events.jsonl``."""
    return json.dumps(
        {"envelope": envelope_body, "ingestion": ingestion_body},
        separators=(",", ":"),
    )


def default_lines(
    session_id: str,
    *,
    synthetic: bool = True,
    replayed: bool = False,
    with_session_end: bool = True,
    with_adaptation: bool = False,
    data_source: str | None = None,
) -> list[str]:
    """A small, well-formed recording: hello, start, two trials, end."""
    lines: list[str] = []
    index = 0

    def add(message_type: str, payload: dict[str, Any]) -> None:
        nonlocal index
        lines.append(
            stored(
                envelope(
                    session_id=session_id,
                    message_type=message_type,
                    sequence_number=index,
                    payload=payload,
                    offset=index,
                    synthetic=synthetic,
                    replayed=replayed,
                    data_source=data_source,
                ),
                ingestion(arrival_index=index, offset=index),
            )
        )
        index += 1

    add(
        "client_hello",
        {
            "role": "simulator",
            "client_name": "engagevr-test-fixture",
            "client_version": "0.1.0",
            "protocol_version": PROTOCOL_VERSION,
            "capabilities": [],
        },
    )
    add(
        "session_start",
        {
            "participant_id": PARTICIPANTS[0],
            "task_id": "synthetic-reaction-task",
            "started_at_utc": timestamp(1),
            "blocks": 1,
            "trials_per_block": 2,
            "difficulty_level": 2,
            "configuration": {},
        },
    )
    add(
        "task_state",
        {
            "state": "running",
            "task_id": "synthetic-reaction-task",
            "block_id": 0,
            "trial_id": 0,
            "difficulty_level": 2,
            "stimulus_interval_ms": 800.0,
        },
    )
    for trial in range(2):
        add(
            "task_event",
            {
                "event": {
                    "event_type": "trial_completed",
                    "task_id": "synthetic-reaction-task",
                    "block_id": 0,
                    "trial_id": trial,
                    "response_outcome": "correct",
                    "reaction_time_ms": 400.0 + trial,
                }
            },
        )
    if with_adaptation:
        add(
            "adaptation_command",
            {
                "command_id": "synthetic-command-0001",
                "command": "set_difficulty",
                "value": 3,
                "reason": "synthetic fixture command; issued by a test, not a policy",
                "issued_at_utc": timestamp(index),
                "expires_at_utc": None,
                "target_role": "simulator",
                "target_client_id": None,
                "is_manual": True,
            },
        )
        add(
            "adaptation_acknowledgement",
            {
                "command_id": "synthetic-command-0001",
                "accepted": True,
                "applied_at_utc": timestamp(index),
                "rejection_reason": None,
                "duplicate": False,
            },
        )
    if with_session_end:
        add(
            "session_end",
            {
                "ended_at_utc": timestamp(index),
                "completed": True,
                "reason": "synthetic fixture session reached its planned end",
            },
        )
    return lines


def write_session(
    root: Path,
    session_id: str,
    *,
    lines: list[str] | None = None,
    with_summary: bool = True,
    completed: bool = True,
    disconnect_reason: str | None = "orderly",
    with_manifest: bool = True,
    partial_trailing: str | None = None,
    trailing_newline: bool = True,
    synthetic: bool = True,
    replayed: bool = False,
    with_adaptation: bool = False,
    data_source: str | None = None,
) -> Path:
    """Write one recording and return its directory.

    ``partial_trailing`` appends text with no terminating newline, which
    is what a file being written to looks like from outside.
    """
    directory = root / session_id
    directory.mkdir(parents=True, exist_ok=True)
    if with_manifest:
        (directory / "manifest.json").write_text(
            json.dumps(manifest(session_id), indent=2) + "\n", encoding="utf-8"
        )
    body = (
        lines
        if lines is not None
        else default_lines(
            session_id,
            synthetic=synthetic,
            replayed=replayed,
            with_session_end=completed,
            with_adaptation=with_adaptation,
            data_source=data_source,
        )
    )
    text = "".join(f"{line}\n" for line in body)
    if not trailing_newline and text.endswith("\n"):
        text = text[:-1]
    if partial_trailing is not None:
        text += partial_trailing
    (directory / "events.jsonl").write_text(text, encoding="utf-8")
    (directory / "dropped.jsonl").write_text("", encoding="utf-8")
    if with_summary:
        (directory / "summary.json").write_text(
            json.dumps(
                summary(
                    session_id,
                    event_count=len(body),
                    completed=completed,
                    disconnect_reason=disconnect_reason,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return directory


def append_record(directory: Path, line: str, *, newline: bool = True) -> None:
    """Append one line to a recording, as a recorder would.

    Uses a plain rewrite rather than the project's writer: these tests
    simulate a producer, and the dashboard under test must never contain
    such a call itself.
    """
    path = directory / "events.jsonl"
    existing = path.read_text(encoding="utf-8")
    path.write_text(existing + line + ("\n" if newline else ""), encoding="utf-8")


def corrupt_line(directory: Path, line_number: int, text: str = "{not json") -> None:
    """Replace one complete line with something that will not parse."""
    path = directory / "events.jsonl"
    lines = path.read_text(encoding="utf-8").split("\n")
    lines[line_number - 1] = text
    path.write_text("\n".join(lines), encoding="utf-8")


def file_digest(path: Path) -> str:
    """SHA-256 of a file, for before/after comparisons in tests."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_digests(directory: Path) -> dict[str, str]:
    """Digest of every file in a session directory."""
    return {
        path.name: file_digest(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


__all__ = [
    "PARTICIPANTS",
    "append_record",
    "corrupt_line",
    "default_lines",
    "directory_digests",
    "envelope",
    "file_digest",
    "ingestion",
    "manifest",
    "stored",
    "summary",
    "timestamp",
    "write_session",
]
