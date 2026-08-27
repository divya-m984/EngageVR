"""Discovery and summarisation of recorded sessions.

Kept deliberately separate from
:mod:`engagevr.dashboard.catalogue`.  That module catalogues Milestone
5-8 **experiment runs**: directories carrying a metrics document, a split
manifest, a dataset fingerprint, and an eligibility declaration.  A
Milestone 4 **session recording** has none of those and answers a
different question.  Merging the two catalogues would let a task
recording appear in a run selector, which is the point at which a
transport log starts reading like an experiment result.

Status, and why ``failed`` is rare here
---------------------------------------
A session with no ``summary.json`` is reported as *active or incomplete*.
From outside the process the two are indistinguishable — neither has a
summary — and both are legitimate.  ``failed`` is used only when the
recording's own summary states a failure through its disconnect reason.
An interrupted session stays fully inspectable: nothing here requires a
summary before a recording may be read.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engagevr.dashboard.session_reader import (
    PARTIAL_TRAILING_NOTE,
    SessionReadError,
    StreamSnapshot,
    decode_records,
    read_json_document,
    read_stream,
    readable_session_id,
    sequence_observations,
)
from engagevr.protocol.messages import MessageType
from engagevr.schemas.dashboard import DashboardWarning, DashboardWarningLevel
from engagevr.schemas.dashboard_session import (
    SESSION_CONTENT_NOTE,
    DashboardSessionCatalogue,
    DashboardSessionMode,
    DashboardSessionProvenance,
    DashboardSessionRecord,
    DashboardSessionStatus,
    DashboardSessionSummary,
    SessionAdaptationCounts,
)
from engagevr.storage.session_store import (
    EVENTS_FILENAME,
    MANIFEST_FILENAME,
    SUMMARY_FILENAME,
)

#: Records one catalogue pass decodes per session before it stops and
#: says so. A catalogue listing must stay cheap; the session pages read
#: with a higher limit once a reader has chosen one.
CATALOGUE_RECORD_LIMIT = 500

#: Disconnect reasons that mean the recording itself reports a failure.
FAILURE_DISCONNECT_REASONS: frozenset[str] = frozenset(
    {"invalid_protocol", "internal_failure"}
)

#: Stated for every quantity a session recording structurally cannot hold.
UNAVAILABLE_STATEMENTS: tuple[str, ...] = (
    "Engagement estimate: Unavailable. A session recording cannot carry "
    "one; the protocol payloads have no field for it.",
    "Cognitive-load estimate: Unavailable, for the same reason.",
    "Signal-quality report: Unavailable. Quality is measured by the "
    "capture pipeline and stored separately from the task protocol.",
    "Abstention decision: Unavailable. Selective prediction is a "
    "Milestone 7 artifact and is not part of a session recording.",
    "Model confidence and prediction interval: Unavailable. No estimator "
    "output is recorded here, and none has been produced to fill the gap.",
)


@dataclass(frozen=True, slots=True)
class SessionRead:
    """One read pass over one recording.

    Holds the parsed records alongside the summary so a page renders both
    from a single pass rather than opening the file twice and risking two
    different views of a file that is being appended to.
    """

    directory: Path
    summary: DashboardSessionSummary
    records: tuple[DashboardSessionRecord, ...]
    snapshot: StreamSnapshot
    manifest: dict[str, Any] | None = None
    stored_summary: dict[str, Any] | None = None


def read_session(
    directory: Path,
    *,
    mode: DashboardSessionMode,
    start_line: int = 0,
    max_records: int | None = None,
) -> SessionRead:
    """Read one session directory, tolerating every stage failing.

    Never raises for a recording's own state.  A missing manifest, an
    unparseable summary, a malformed line, and an absent event stream all
    become stated facts on the returned summary, because a dashboard that
    crashed on one bad recording would hide every good one beside it.
    """
    warnings: list[DashboardWarning] = []
    session_id = directory.name

    manifest, manifest_error = _read_optional(directory / MANIFEST_FILENAME)
    if manifest_error:
        warnings.append(
            DashboardWarning(
                level=DashboardWarningLevel.ERROR,
                message=manifest_error,
                subject=session_id,
            )
        )
    stored_summary, summary_error = _read_optional(directory / SUMMARY_FILENAME)
    if summary_error:
        warnings.append(
            DashboardWarning(
                level=DashboardWarningLevel.ERROR,
                message=summary_error,
                subject=session_id,
            )
        )

    snapshot = read_stream(directory / EVENTS_FILENAME, start_line=start_line)
    if snapshot.unavailable_reason:
        warnings.append(
            DashboardWarning(
                level=DashboardWarningLevel.WARNING,
                message=snapshot.unavailable_reason,
                subject=session_id,
            )
        )

    lines = snapshot.lines
    if max_records is not None and len(lines) > max_records:
        warnings.append(
            DashboardWarning(
                level=DashboardWarningLevel.INFORMATION,
                message=(
                    f"{len(lines) - max_records} further record(s) exist in "
                    "this recording and were not read in this pass. The "
                    "recording on disk is complete; only this view is "
                    "limited."
                ),
                subject=session_id,
            )
        )
        lines = lines[:max_records]

    recorded_id = _text(manifest, "session_id")
    if recorded_id is not None and recorded_id != session_id:
        warnings.append(
            DashboardWarning(
                level=DashboardWarningLevel.WARNING,
                message=(
                    f"This directory is named {session_id!r} but the recording "
                    f"inside declares session_id={recorded_id!r}. The recorded "
                    "identifier is the provenance; the directory name is only "
                    "where the bytes happen to sit. Both are shown, and "
                    "neither has been rewritten to match the other."
                ),
                subject=session_id,
            )
        )

    records = decode_records(lines)
    status, status_reason = _status(
        manifest=manifest,
        manifest_error=manifest_error,
        stored_summary=stored_summary,
        snapshot=snapshot,
    )
    provenance = _provenance(
        session_id=session_id,
        directory=directory,
        mode=mode,
        manifest=manifest,
        records=records,
        status=status,
        status_reason=status_reason,
        warnings=tuple(warnings),
    )
    summary = _summary(
        provenance=provenance,
        snapshot=snapshot,
        start_line=start_line,
        records=records,
        stored_summary=stored_summary,
        warnings=tuple(warnings),
    )
    return SessionRead(
        directory=directory,
        summary=summary,
        records=records,
        snapshot=snapshot,
        manifest=manifest,
        stored_summary=stored_summary,
    )


def build_session_catalogue(
    session_root: Path,
    *,
    mode: DashboardSessionMode = DashboardSessionMode.REPLAY,
    record_limit: int = CATALOGUE_RECORD_LIMIT,
) -> DashboardSessionCatalogue:
    """List every readable recording under ``session_root``.

    A root that does not exist is a state to display, not an exception to
    raise: a fresh clone has recorded no session and the dashboard must
    still start.
    """
    root = Path(session_root)
    warnings: list[DashboardWarning] = []
    if not root.is_dir():
        return DashboardSessionCatalogue(
            session_root=str(root),
            root_exists=False,
            warnings=(
                DashboardWarning(
                    level=DashboardWarningLevel.INFORMATION,
                    message=(
                        f"The session root {root} does not exist. No session "
                        "has been recorded on this machine yet; that is not "
                        "an error."
                    ),
                    subject=str(root),
                ),
            ),
        )

    sessions: list[DashboardSessionSummary] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        if not readable_session_id(entry.name):
            warnings.append(
                DashboardWarning(
                    level=DashboardWarningLevel.INFORMATION,
                    message=(
                        f"{entry.name!r} is not a valid session identifier, so "
                        "it is not listed as a session."
                    ),
                    subject=entry.name,
                )
            )
            continue
        if (
            not (entry / MANIFEST_FILENAME).is_file()
            and not (entry / EVENTS_FILENAME).is_file()
        ):
            continue
        sessions.append(
            read_session(entry, mode=mode, max_records=record_limit).summary
        )

    return DashboardSessionCatalogue(
        session_root=str(root),
        root_exists=True,
        sessions=tuple(sessions),
        warnings=tuple(warnings),
    )


def _read_optional(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        return read_json_document(path), None
    except SessionReadError as exc:
        return None, str(exc)


def _status(
    *,
    manifest: dict[str, Any] | None,
    manifest_error: str | None,
    stored_summary: dict[str, Any] | None,
    snapshot: StreamSnapshot,
) -> tuple[DashboardSessionStatus, str | None]:
    if manifest_error is not None:
        return DashboardSessionStatus.UNREADABLE, manifest_error
    if manifest is None:
        return (
            DashboardSessionStatus.UNREADABLE,
            f"{MANIFEST_FILENAME} is not present, so this directory records no "
            "session identity of its own.",
        )
    if not snapshot.stream_present:
        return (
            DashboardSessionStatus.STREAM_UNAVAILABLE,
            snapshot.unavailable_reason
            or f"{EVENTS_FILENAME} is not present in this session directory.",
        )
    if stored_summary is None:
        return DashboardSessionStatus.ACTIVE_OR_INCOMPLETE, None
    reason = stored_summary.get("disconnect_reason")
    if isinstance(reason, str) and reason in FAILURE_DISCONNECT_REASONS:
        return (
            DashboardSessionStatus.FAILED,
            f"This session's summary records disconnect_reason={reason!r}.",
        )
    if stored_summary.get("completed") is True:
        return DashboardSessionStatus.COMPLETED, None
    return DashboardSessionStatus.INTERRUPTED, None


def _provenance(
    *,
    session_id: str,
    directory: Path,
    mode: DashboardSessionMode,
    manifest: dict[str, Any] | None,
    records: tuple[DashboardSessionRecord, ...],
    status: DashboardSessionStatus,
    status_reason: str | None,
    warnings: tuple[DashboardWarning, ...],
) -> DashboardSessionProvenance:
    sources = sorted(
        {record.data_source for record in records if record.data_source is not None}
    )
    synthetic = sum(1 for record in records if record.is_synthetic)
    decoded = sum(1 for record in records if record.decoded)
    replayed = sum(1 for record in records if record.is_replayed)
    return DashboardSessionProvenance(
        session_id=_text(manifest, "session_id") or session_id,
        directory_name=session_id,
        session_directory=str(directory),
        mode=mode,
        session_format_version=_text(manifest, "session_format_version"),
        protocol_version=_text(manifest, "protocol_version"),
        engagevr_version=_text(manifest, "engagevr_version"),
        created_at_utc=_text(manifest, "created_at_utc"),
        data_sources=tuple(sources),
        synthetic_message_count=synthetic,
        non_synthetic_message_count=decoded - synthetic,
        replayed_message_count=replayed,
        status=status,
        status_reason=status_reason,
        warnings=warnings,
    )


def _summary(
    *,
    provenance: DashboardSessionProvenance,
    snapshot: StreamSnapshot,
    start_line: int,
    records: tuple[DashboardSessionRecord, ...],
    stored_summary: dict[str, Any] | None,
    warnings: tuple[DashboardWarning, ...],
) -> DashboardSessionSummary:
    decoded = [record for record in records if record.decoded]
    malformed = [record for record in records if not record.decoded]

    types = Counter(
        record.message_type for record in decoded if record.message_type is not None
    )
    sources = Counter(record.source for record in decoded if record.source is not None)
    anomalies: Counter[str] = Counter()
    for record in decoded:
        anomalies.update(record.recorded_anomalies)

    sent = [r.sent_at_utc for r in decoded if r.sent_at_utc]
    received = [r.server_received_at_utc for r in decoded if r.server_received_at_utc]

    return DashboardSessionSummary(
        provenance=provenance,
        complete_record_count=snapshot.complete_line_count,
        parse_start_line=start_line,
        parsed_record_count=len(records),
        decoded_record_count=len(decoded),
        malformed_record_count=len(malformed),
        malformed_line_numbers=tuple(record.line_number for record in malformed),
        partial_trailing_line=snapshot.partial_trailing_line,
        partial_trailing_note=(
            PARTIAL_TRAILING_NOTE if snapshot.partial_trailing_line else None
        ),
        message_type_counts=_ordered(types),
        source_counts=_ordered(sources),
        recorded_anomaly_counts=_ordered(anomalies),
        sequence_observations=sequence_observations(decoded),
        dropped_message_count=_integer(stored_summary, "dropped_message_count"),
        first_sent_at_utc=sent[0] if sent else None,
        last_sent_at_utc=sent[-1] if sent else None,
        first_received_at_utc=received[0] if received else None,
        last_received_at_utc=received[-1] if received else None,
        session_end_recorded=any(
            record.message_type == MessageType.SESSION_END.value for record in decoded
        ),
        disconnect_reason=_text(stored_summary, "disconnect_reason"),
        summary_recovered=_boolean(stored_summary, "recovered"),
        task_state=_latest_field(decoded, MessageType.TASK_STATE.value, "state"),
        current_difficulty_level=_latest_difficulty(decoded),
        adaptation=_adaptation(decoded),
        unavailable_statements=(*UNAVAILABLE_STATEMENTS, SESSION_CONTENT_NOTE),
        warnings=warnings,
    )


def _adaptation(records: list[DashboardSessionRecord]) -> SessionAdaptationCounts:
    commands = 0
    acknowledgements = 0
    accepted = 0
    rejected = 0
    applied = 0
    for record in records:
        if record.message_type == MessageType.ADAPTATION_COMMAND.value:
            commands += 1
        elif record.message_type == MessageType.ADAPTATION_ACKNOWLEDGEMENT.value:
            acknowledgements += 1
            fields = dict(record.payload_summary)
            if fields.get("accepted") == "True":
                accepted += 1
            elif fields.get("accepted") == "False":
                rejected += 1
            if fields.get("applied_at_utc"):
                applied += 1
    # A recording can hold an acknowledgement whose command was dropped
    # under backpressure or never stored. Reporting more acknowledgements
    # than commands would be refused by the model, so the counts are
    # clamped and the discrepancy is visible as an anomaly instead.
    acknowledgements = min(acknowledgements, commands)
    accepted = min(accepted, acknowledgements)
    rejected = min(rejected, acknowledgements - accepted)
    applied = min(applied, accepted)
    return SessionAdaptationCounts(
        commands_recorded=commands,
        acknowledgements_recorded=acknowledgements,
        accepted_recorded=accepted,
        rejected_recorded=rejected,
        applied_timestamp_recorded=applied,
    )


def _latest_field(
    records: list[DashboardSessionRecord], message_type: str, field: str
) -> str | None:
    for record in reversed(records):
        if record.message_type != message_type:
            continue
        value = dict(record.payload_summary).get(field)
        if value is not None:
            return value
    return None


def _latest_difficulty(records: list[DashboardSessionRecord]) -> int | None:
    for record in reversed(records):
        if record.message_type not in (
            MessageType.TASK_STATE.value,
            MessageType.SESSION_START.value,
        ):
            continue
        value = dict(record.payload_summary).get("difficulty_level")
        if value is None:
            continue
        try:
            level = int(value)
        except ValueError:
            return None
        return level if level >= 0 else None
    return None


def _ordered(counts: Counter[str]) -> tuple[tuple[str, int], ...]:
    """Counts sorted by name, so two scans of one file agree exactly."""
    return tuple(sorted(counts.items()))


def _text(document: dict[str, Any] | None, key: str) -> str | None:
    if document is None:
        return None
    value = document.get(key)
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _integer(document: dict[str, Any] | None, key: str) -> int | None:
    if document is None:
        return None
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _boolean(document: dict[str, Any] | None, key: str) -> bool | None:
    if document is None:
        return None
    value = document.get(key)
    return value if isinstance(value, bool) else None


__all__ = [
    "CATALOGUE_RECORD_LIMIT",
    "FAILURE_DISCONNECT_REASONS",
    "UNAVAILABLE_STATEMENTS",
    "SessionRead",
    "build_session_catalogue",
    "read_session",
]
