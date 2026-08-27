"""Tail-safe, read-only reading of a Milestone 4 session recording.

Why this is not :class:`~engagevr.storage.session_store.SessionStore`
--------------------------------------------------------------------
``SessionStore`` has exactly the right layout knowledge and exactly the
wrong failure behaviour for this job.  ``iter_messages`` raises on the
first bad line, which would blank a live view because a writer was
mid-flush; ``recover`` is tolerant but rebuilds a whole summary and is
not incremental.  Neither distinguishes *a torn final line while the
recorder is running* from *a malformed line in the middle of a finished
recording*, and on a live page those two need opposite reactions.

So this module reuses the store's **layout constants and its identifier
validation** — the parts that are pure — and does its own parsing.  It
imports no writer, constructs no recorder, and opens no file for
anything but reading.

What "tail-safe" means here
---------------------------
The recording is read as a whole-file snapshot on each pass.  A line is
*complete* only when a newline terminated it.  Anything after the last
newline is a **partial trailing line**: the normal appearance of a file
that is being appended to right now.  It is reported as a transient
state and is never parsed, never counted, and never called corruption.

A complete line that will not decode is a different matter.  It stays
visible, with its 1-based line number and the reason, because a bad line
that silently vanished would be a gap the reader never learned about.

Ordering
--------
Records are presented in the order they appear in the file, which is
recorded arrival order.  They are never re-sorted by sequence number.
Sequence irregularities are *observed and reported* from the numbers
that were recorded; nothing is renumbered, reordered, or filled in.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engagevr.protocol.messages import MessageType
from engagevr.protocol.validation import ProtocolValidationError, decode_stored_message
from engagevr.schemas.dashboard_session import (
    DashboardSessionRecord,
    SessionRecordProblem,
    SessionSequenceObservation,
)
from engagevr.storage.session_store import (
    DROPS_FILENAME,
    EVENTS_FILENAME,
    MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    InvalidSessionIdError,
    validate_session_id,
)

#: Files this module will read, and nothing else.
SESSION_FILENAMES: tuple[str, ...] = (
    MANIFEST_FILENAME,
    EVENTS_FILENAME,
    SUMMARY_FILENAME,
    DROPS_FILENAME,
)

#: How much of an offending line is echoed back to the reader.
EXCERPT_LIMIT = 160

#: Stated whenever a trailing line has no terminating newline.
PARTIAL_TRAILING_NOTE = (
    "The final line of this recording has no terminating newline, so it is "
    "not yet a complete record. While a recorder is running this is the "
    "normal appearance of a file being appended to, and it is expected to "
    "resolve on the next read. It is not counted, not parsed, and not "
    "treated as corruption."
)

#: Payload fields worth showing per message type, in display order.
PAYLOAD_DISPLAY_FIELDS: dict[str, tuple[str, ...]] = {
    MessageType.CLIENT_HELLO.value: ("role", "client_name", "client_version"),
    MessageType.SERVER_HELLO.value: ("accepted", "session_id", "rejection_reason"),
    MessageType.SESSION_START.value: (
        "participant_id",
        "task_id",
        "blocks",
        "trials_per_block",
        "difficulty_level",
    ),
    MessageType.SESSION_END.value: ("completed", "reason"),
    MessageType.TASK_STATE.value: (
        "state",
        "task_id",
        "block_id",
        "trial_id",
        "difficulty_level",
        "stimulus_interval_ms",
    ),
    MessageType.TELEMETRY.value: ("component",),
    MessageType.ADAPTATION_COMMAND.value: (
        "command_id",
        "command",
        "value",
        "reason",
        "target_role",
        "is_manual",
    ),
    MessageType.ADAPTATION_ACKNOWLEDGEMENT.value: (
        "command_id",
        "accepted",
        "applied_at_utc",
        "rejection_reason",
        "duplicate",
    ),
    MessageType.PROTOCOL_ERROR.value: ("error_code", "detail", "fatal"),
    MessageType.ACKNOWLEDGEMENT.value: (
        "acknowledged_message_type",
        "stored",
        "dropped",
    ),
    MessageType.REPLAY_CONTROL.value: ("action", "source_session_id", "speed"),
}

#: Fields lifted out of a ``task_event`` payload's nested event body.
TASK_EVENT_FIELDS: tuple[str, ...] = (
    "event_type",
    "task_id",
    "block_id",
    "trial_id",
    "response_outcome",
    "reaction_time_ms",
    "difficulty_level",
)


class SessionReadError(ValueError):
    """A session directory or file could not be read."""


@dataclass(frozen=True, slots=True)
class RawLine:
    """One complete line of a recording, parsed or not."""

    line_number: int
    record: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True, slots=True)
class StreamSnapshot:
    """What one read pass saw in ``events.jsonl``.

    ``complete_line_count`` counts every newline-terminated, non-blank
    line in the file, whether or not this pass parsed it.  A live view
    uses it as the cursor for the next pass.
    """

    lines: tuple[RawLine, ...]
    complete_line_count: int
    partial_trailing_line: bool
    partial_trailing_excerpt: str | None
    stream_present: bool
    unavailable_reason: str | None = None


def session_paths(directory: Path) -> tuple[Path, ...]:
    """Every file of a session directory this module may read."""
    return tuple(directory / name for name in SESSION_FILENAMES)


def readable_session_id(name: str) -> bool:
    """Whether a directory name is a valid session identifier.

    Reuses the store's own allowlist rather than restating it, so the
    dashboard cannot accept an identifier the store would reject.
    """
    try:
        validate_session_id(name)
    except InvalidSessionIdError:
        return False
    return True


def read_json_document(path: Path) -> dict[str, Any]:
    """Read one JSON document of a session directory.

    Raises
    ------
    SessionReadError
        With the path and the parser's own message, because "the session
        is corrupt" is not something a reader can act on.
    """
    if not path.is_file():
        raise SessionReadError(f"{path.name} is not present in {path.parent}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SessionReadError(f"{path} could not be read: {exc}") from exc
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise SessionReadError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SessionReadError(
            f"{path} holds a {type(parsed).__name__}, not a JSON object"
        )
    return parsed


def read_stream(events_path: Path, *, start_line: int = 0) -> StreamSnapshot:
    """Snapshot ``events.jsonl``, parsing from ``start_line`` onward.

    ``start_line`` is a count of already-consumed complete lines, which
    is what makes a live refresh incremental: a page that has presented
    1 000 records asks for line 1 000 and parses only what arrived since.
    Lines before it are still counted, so the totals stay right.

    The whole file is snapshotted in one read.  A writer appending
    between two reads therefore cannot produce a half-line in the middle
    of this pass's view; the only incomplete line is the last one, and it
    is reported separately.
    """
    if start_line < 0:
        raise SessionReadError(f"start_line must not be negative, got {start_line}")
    if not events_path.is_file():
        return StreamSnapshot(
            lines=(),
            complete_line_count=0,
            partial_trailing_line=False,
            partial_trailing_excerpt=None,
            stream_present=False,
            unavailable_reason=(
                f"{events_path.name} is not present, so this session has no "
                "readable event stream. Nothing has been substituted for it."
            ),
        )
    try:
        raw = events_path.read_bytes()
    except OSError as exc:
        return StreamSnapshot(
            lines=(),
            complete_line_count=0,
            partial_trailing_line=False,
            partial_trailing_excerpt=None,
            stream_present=False,
            unavailable_reason=f"{events_path} could not be read: {exc}",
        )

    # A torn multi-byte character at the tail must not raise. Replacing
    # it keeps the interior lines readable and makes the damaged line
    # fail its own JSON parse, which is exactly where it should surface.
    text = raw.decode("utf-8", errors="replace")
    if not text:
        return StreamSnapshot(
            lines=(),
            complete_line_count=0,
            partial_trailing_line=False,
            partial_trailing_excerpt=None,
            stream_present=True,
        )

    chunks = text.split("\n")
    trailing = chunks.pop()
    partial = bool(trailing.strip())

    lines: list[RawLine] = []
    complete = 0
    for index, chunk in enumerate(chunks, start=1):
        stripped = chunk.strip()
        if not stripped:
            continue
        complete += 1
        if complete <= start_line:
            continue
        lines.append(_parse_line(index, stripped))

    return StreamSnapshot(
        lines=tuple(lines),
        complete_line_count=complete,
        partial_trailing_line=partial,
        partial_trailing_excerpt=trailing[:EXCERPT_LIMIT] if partial else None,
        stream_present=True,
    )


def _parse_line(line_number: int, text: str) -> RawLine:
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        return RawLine(
            line_number=line_number,
            record=None,
            error=f"invalid JSON ({exc}): {text[:EXCERPT_LIMIT]!r}",
        )
    if not isinstance(parsed, dict):
        return RawLine(
            line_number=line_number,
            record=None,
            error=(
                f"expected a JSON object, got {type(parsed).__name__}: "
                f"{text[:EXCERPT_LIMIT]!r}"
            ),
        )
    return RawLine(line_number=line_number, record=parsed, error=None)


def decode_records(lines: Iterable[RawLine]) -> tuple[DashboardSessionRecord, ...]:
    """Turn snapshot lines into presentable records.

    A line that will not decode becomes a record carrying its problem
    rather than disappearing.  The protocol validator is the same one the
    session store uses, so a record this dashboard shows as sound is
    sound by the project's own definition, not by a looser local one.
    """
    return tuple(_decode_line(line) for line in lines)


def _decode_line(line: RawLine) -> DashboardSessionRecord:
    if line.record is None:
        return DashboardSessionRecord(
            line_number=line.line_number,
            problem=SessionRecordProblem.MALFORMED_JSON,
            problem_detail=line.error or "the line could not be parsed",
        )
    envelope_raw = line.record.get("envelope")
    ingestion_raw = line.record.get("ingestion")
    if not isinstance(envelope_raw, dict) or not isinstance(ingestion_raw, dict):
        return DashboardSessionRecord(
            line_number=line.line_number,
            problem=SessionRecordProblem.INVALID_STRUCTURE,
            problem_detail=(
                "a stored record must carry object 'envelope' and 'ingestion' "
                "fields; this line carries "
                f"{sorted(line.record)!r}"
            ),
        )
    try:
        envelope = decode_stored_message(envelope_raw).envelope
    except ProtocolValidationError as exc:
        return DashboardSessionRecord(
            line_number=line.line_number,
            problem=SessionRecordProblem.PROTOCOL_INVALID,
            problem_detail=f"the message is not valid under the protocol: {exc.detail}",
        )
    except ValueError as exc:
        return DashboardSessionRecord(
            line_number=line.line_number,
            problem=SessionRecordProblem.PROTOCOL_INVALID,
            problem_detail=f"the message could not be validated: {exc}",
        )

    ingestion = _ingestion_fields(ingestion_raw)
    if isinstance(ingestion, str):
        return DashboardSessionRecord(
            line_number=line.line_number,
            problem=SessionRecordProblem.INGESTION_INVALID,
            problem_detail=ingestion,
        )

    replay = envelope.replay
    return DashboardSessionRecord(
        line_number=line.line_number,
        arrival_index=ingestion["arrival_index"],
        sequence_number=envelope.sequence_number,
        message_type=envelope.message_type.value,
        source=envelope.source.value,
        message_id=envelope.message_id,
        correlation_id=envelope.correlation_id,
        sent_at_utc=envelope.sent_at_utc.isoformat(),
        server_received_at_utc=ingestion["server_received_at_utc"],
        transport=ingestion["transport"],
        data_source=envelope.provenance.data_source.value,
        synthetic_label=envelope.provenance.synthetic_label,
        producer=envelope.provenance.producer or None,
        replay_label=replay.replay_label if replay is not None else None,
        replay_source_session_id=(
            replay.source_session_id if replay is not None else None
        ),
        recorded_anomalies=ingestion["anomalies"],
        anomaly_detail=ingestion["anomaly_detail"],
        expected_sequence_number=ingestion["expected_sequence_number"],
        payload_summary=payload_summary(envelope.message_type.value, envelope.payload),
    )


def _ingestion_fields(raw: dict[str, Any]) -> dict[str, Any] | str:
    """Pull the ingestion fields a view shows, or state what is wrong.

    Read field by field rather than through the writer's model, so that
    a recording written by a newer version with an extra field is still
    presentable rather than rejected by ``extra="forbid"``.
    """
    arrival = raw.get("arrival_index")
    if not isinstance(arrival, int) or isinstance(arrival, bool) or arrival < 0:
        return (
            f"ingestion.arrival_index must be a non-negative integer, got {arrival!r}"
        )
    received = raw.get("server_received_at_utc")
    if received is not None and not isinstance(received, str):
        return f"ingestion.server_received_at_utc must be a string, got {received!r}"
    transport = raw.get("transport")
    if transport is not None and not isinstance(transport, str):
        return f"ingestion.transport must be a string, got {transport!r}"
    anomalies_raw = raw.get("anomalies", [])
    if not isinstance(anomalies_raw, list):
        return f"ingestion.anomalies must be a list, got {anomalies_raw!r}"
    anomalies = tuple(str(entry) for entry in anomalies_raw)
    detail = raw.get("anomaly_detail")
    expected = raw.get("expected_sequence_number")
    if expected is not None and (
        not isinstance(expected, int) or isinstance(expected, bool) or expected < 0
    ):
        return (
            "ingestion.expected_sequence_number must be a non-negative "
            f"integer or null, got {expected!r}"
        )
    return {
        "arrival_index": arrival,
        "server_received_at_utc": received,
        "transport": transport,
        "anomalies": anomalies,
        "anomaly_detail": str(detail) if detail is not None else None,
        "expected_sequence_number": expected,
    }


def payload_summary(
    message_type: str, payload: dict[str, Any]
) -> tuple[tuple[str, str], ...]:
    """Flat display fields for one payload.

    Only named fields are lifted out.  A payload is task and transport
    telemetry, but listing it wholesale would put whatever a future
    message type carries onto a page nobody reviewed.
    """
    rows: list[tuple[str, str]] = []
    if message_type == MessageType.TASK_EVENT.value:
        event = payload.get("event")
        if isinstance(event, dict):
            for name in TASK_EVENT_FIELDS:
                if name in event and event[name] is not None:
                    rows.append((name, str(event[name])))
        return tuple(rows)
    for name in PAYLOAD_DISPLAY_FIELDS.get(message_type, ()):
        if name in payload and payload[name] is not None:
            rows.append((name, str(payload[name])))
    return tuple(rows)


def sequence_observations(
    records: Sequence[DashboardSessionRecord],
) -> tuple[SessionSequenceObservation, ...]:
    """Sequence irregularities visible in the recorded numbers.

    Derived from ``sequence_number`` values that were recorded, per
    source, in recorded arrival order.  This reports; it does not repair.
    No message is invented for a gap, nothing is reordered, and a
    reversal that actually happened stays a reversal.

    The receiver's own :class:`~engagevr.synchronization.ordering.\
OrderingAnomaly` records are shown separately and take precedence: they
    are what the receiver saw at the time.
    """
    seen: dict[str, set[int]] = {}
    highest: dict[str, int] = {}
    observations: list[SessionSequenceObservation] = []
    for record in records:
        source = record.source
        number = record.sequence_number
        if source is None or number is None:
            continue
        previous = seen.setdefault(source, set())
        if number in previous:
            observations.append(
                SessionSequenceObservation(
                    source=source,
                    line_number=record.line_number,
                    kind="duplicate_sequence_number",
                    detail=(
                        f"sequence number {number} appears more than once for "
                        f"source {source!r}. The recording shows it twice; "
                        "neither occurrence has been removed."
                    ),
                )
            )
        previous.add(number)
        top = highest.get(source)
        if top is None:
            highest[source] = number
            continue
        if number < top:
            observations.append(
                SessionSequenceObservation(
                    source=source,
                    line_number=record.line_number,
                    kind="sequence_reversal",
                    detail=(
                        f"sequence number {number} arrived after {top} for "
                        f"source {source!r}. Arrival order is preserved as "
                        "recorded; the records have not been re-sorted."
                    ),
                )
            )
        elif number > top + 1:
            observations.append(
                SessionSequenceObservation(
                    source=source,
                    line_number=record.line_number,
                    kind="missing_sequence_range",
                    detail=(
                        f"sequence numbers {top + 1}-{number - 1} were never "
                        f"recorded for source {source!r}. They may have been "
                        "dropped under backpressure or never sent; no message "
                        "has been invented for them."
                    ),
                )
            )
        highest[source] = max(top, number)
    return tuple(observations)


def file_checksums(directory: Path) -> tuple[tuple[str, str], ...]:
    """SHA-256 of every session file present, for audit.

    Reading a file to hash it is a read.  The digests let a reader
    confirm afterwards that inspecting a recording did not change it.
    """
    digests: list[tuple[str, str]] = []
    for path in session_paths(directory):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
        except OSError:  # pragma: no cover - vanished mid-scan
            continue
        digests.append((path.name, digest.hexdigest()))
    return tuple(digests)


__all__ = [
    "EXCERPT_LIMIT",
    "PARTIAL_TRAILING_NOTE",
    "PAYLOAD_DISPLAY_FIELDS",
    "SESSION_FILENAMES",
    "TASK_EVENT_FIELDS",
    "RawLine",
    "SessionReadError",
    "StreamSnapshot",
    "decode_records",
    "file_checksums",
    "payload_summary",
    "read_json_document",
    "read_stream",
    "readable_session_id",
    "sequence_observations",
    "session_paths",
]
