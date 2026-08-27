"""View models for the live-observation and replay modes.

Framework-free, like every other ``views_*`` module: it turns a
:class:`~engagevr.dashboard.session_catalogue.SessionRead` into labelled
tables, and the Streamlit layer above it does nothing but render them.

Two rules from the artifact views carry over unchanged and are the
reason this module exists rather than the pages formatting things
themselves.

*A missing value is never a zero.*  Everything numeric goes through
:mod:`engagevr.dashboard.formatting`, so a field the recording never
carried reads *Unavailable*.

*A label is not decoration.*  A record that was synthetic when it was
written says so in its own row, a record that was already a replay says
that too, and neither is inferable from the mode heading — a live
observation of a recording full of synthetic messages is still
synthetic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from engagevr.dashboard import formatting as fmt
from engagevr.dashboard import presentation
from engagevr.dashboard.session_catalogue import SessionRead
from engagevr.schemas.dashboard import (
    DashboardError,
    LabelledTable,
    MetricDisplayValue,
    MetricKind,
)
from engagevr.schemas.dashboard_session import (
    DashboardReplayState,
    DashboardSessionCatalogue,
    DashboardSessionMode,
    DashboardSessionRecord,
    DashboardSessionStatus,
    DashboardSessionSummary,
)

#: Wording of each session status, so the state is never colour-only.
SESSION_STATUS_TEXT: dict[DashboardSessionStatus, str] = {
    DashboardSessionStatus.COMPLETED: (
        "Completed — a session_end was recorded and the summary was written"
    ),
    DashboardSessionStatus.INTERRUPTED: (
        "Interrupted — the summary records that this session did not reach "
        "its planned end. It remains fully inspectable."
    ),
    DashboardSessionStatus.ACTIVE_OR_INCOMPLETE: (
        "Active or incomplete — no summary.json exists yet. This session may "
        "still be running, or it may have stopped. Both are legitimate and "
        "neither is a failure."
    ),
    DashboardSessionStatus.FAILED: (
        "FAILED — this session's own summary records a failure"
    ),
    DashboardSessionStatus.UNREADABLE: (
        "UNREADABLE — the session manifest is absent or could not be parsed"
    ),
    DashboardSessionStatus.STREAM_UNAVAILABLE: (
        "STREAM UNAVAILABLE — no event stream is present in this directory"
    ),
}

#: Mode wording, stated on every page so the evidence source is explicit.
MODE_TEXT: dict[DashboardSessionMode, str] = {
    DashboardSessionMode.ARTIFACT: (
        "Evidence source: persisted experiment artifacts written by the "
        "Milestone 5-8 runners."
    ),
    DashboardSessionMode.LIVE: (
        "Evidence source: a session recording on this machine, re-read as it "
        "is appended to. Every record shown was persisted by the recorder "
        "before this view saw it."
    ),
    DashboardSessionMode.REPLAY: (
        "Evidence source: a session recording on this machine, already "
        "complete or already interrupted. This is a presentation of those "
        "records, not a re-emission of them."
    ),
}

#: Headline wording of each mode, shown as ``Mode: <headline>``.
#:
#: Separate from :data:`MODE_TEXT` because a heading and an explanation
#: are read at different moments.  ``LIVE OBSERVATION`` in particular is
#: never shortened to ``LIVE``: the word that must survive skimming is
#: *observation*, since this view observes a recording and runs no
#: inference.
MODE_HEADLINE: dict[DashboardSessionMode, str] = {
    DashboardSessionMode.ARTIFACT: "EXPERIMENT ARTIFACTS",
    DashboardSessionMode.LIVE: "LIVE OBSERVATION",
    DashboardSessionMode.REPLAY: "SESSION REPLAY",
}

#: Floor on the automatic live-refresh interval, in seconds.
#:
#: Conservative on purpose.  A research tool re-reading a file several
#: times a second spends its budget on filesystem traffic to produce an
#: animation, and a faster cadence buys nothing: the recorder appends
#: task and transport telemetry, not video frames.
MINIMUM_LIVE_REFRESH_SECONDS = 2.0

#: Columns of the record table, in display order.
RECORD_COLUMNS: tuple[str, ...] = (
    "line",
    "arrival index",
    "sequence",
    "source",
    "message type",
    "sent (sender clock)",
    "received (receiver clock)",
    "transport",
    "data source",
    "synthetic",
    "replayed",
    "recorded anomalies",
    "decode problem",
)

#: Said next to any prediction-shaped field on a session page.
NO_ESTIMATOR_NOTE = (
    "Nothing on this page is a model output. No estimator was loaded, no "
    "inference was run, and no engagement or cognitive-load value exists in "
    "a session recording to display."
)

#: Said wherever a task event is shown.
TASK_EVENT_NOTE = (
    "A task event is a software telemetry record of what the task program "
    "observed. Accuracy, reaction time, and timeout counts are not "
    "engagement, attention, cognitive-load, or fatigue measurements."
)


def mode_statement(mode: DashboardSessionMode) -> str:
    """The evidence-source sentence for one mode."""
    return MODE_TEXT[mode]


def mode_headline(mode: DashboardSessionMode) -> str:
    """The ``Mode: ...`` headline for one mode."""
    return MODE_HEADLINE[mode]


def live_refresh_interval(seconds: float) -> float:
    """Validate a configured live-refresh interval, in seconds.

    Only the live mode may refresh on a timer, and only at a cadence a
    reader could have asked for deliberately.  Every rejected value fails
    loudly rather than being clamped: silently substituting a default for
    a misconfigured interval would leave a page refreshing at a rate
    nobody chose, which is precisely the sort of unattributed behaviour a
    read-only observability tool must not have.

    :raises DashboardError: if the value is not a finite number greater
        than zero, or is below :data:`MINIMUM_LIVE_REFRESH_SECONDS`.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, int | float):
        raise DashboardError(
            "dashboard.live_refresh_seconds must be a number of seconds, got "
            f"{seconds!r}"
        )
    value = float(seconds)
    if not math.isfinite(value):
        raise DashboardError(
            "dashboard.live_refresh_seconds must be finite; a non-finite "
            f"interval could not schedule a refresh, got {seconds!r}"
        )
    if value <= 0.0:
        raise DashboardError(
            "dashboard.live_refresh_seconds must be greater than zero; an "
            "interval of zero or less would re-read the recording without "
            f"pause, got {value!r}"
        )
    if value < MINIMUM_LIVE_REFRESH_SECONDS:
        raise DashboardError(
            "dashboard.live_refresh_seconds must be at least "
            f"{MINIMUM_LIVE_REFRESH_SECONDS:g}s. A faster cadence spends a "
            "research tool's budget on filesystem traffic and buys no "
            f"information, got {value!r}"
        )
    return value


def refresh_statement(seconds: float) -> str:
    """What the automatic live refresh does, in one sentence."""
    return (
        f"Automatic refresh: every {seconds:g} seconds. This view re-reads the "
        "recording on disk with the read-only session reader and redraws what "
        "it finds. It runs no model, opens no camera, sends nothing, and "
        "writes nothing. Real-time observation is not real-time inference."
    )


#: Said when the configured interval was refused, so the reason a live
#: page is *not* refreshing on its own is never left to be inferred.
MANUAL_REFRESH_ONLY_NOTE = (
    "Automatic refresh is switched off because the configured interval was "
    "refused. This page still re-reads the recording whenever you press "
    "Read new records, and nothing else about it has changed."
)


def status_statement(summary: DashboardSessionSummary) -> str:
    """The status sentence for one recording, with its stated reason."""
    text = SESSION_STATUS_TEXT[summary.provenance.status]
    reason = summary.provenance.status_reason
    return f"{text} ({reason})" if reason else text


def catalogue_table(
    catalogue: DashboardSessionCatalogue, *, max_rows: int
) -> LabelledTable:
    """Every discovered recording, with its status and composition."""
    rows = []
    for session in catalogue.sessions:
        provenance = session.provenance
        rows.append(
            (
                provenance.directory_name,
                provenance.session_id,
                provenance.status.value,
                fmt.text(session.complete_record_count),
                fmt.text(session.malformed_record_count),
                fmt.text(provenance.synthetic_message_count),
                fmt.text(provenance.replayed_message_count),
                fmt.text(", ".join(provenance.data_sources) or None),
                data_source_labels(provenance.data_sources),
                fmt.text(provenance.scientific_evaluation_eligible),
                fmt.text(session.partial_trailing_line),
            )
        )
    return fmt.build_table(
        title="Recorded sessions",
        columns=(
            "directory",
            "recorded session id",
            "status",
            "complete records",
            "malformed records",
            "synthetic records",
            "already-replayed records",
            "recorded data sources",
            "data source labels",
            "scientifically eligible",
            "partial trailing line",
        ),
        rows=rows,
        source_artifact=catalogue.session_root,
        max_rows=max_rows,
        caption=(
            "A recorded session is not an experiment run and is listed "
            "separately from one. The directory is where the bytes sit; the "
            "recorded session id is the provenance, and the two are shown "
            "apart rather than reconciled. Scientific eligibility is false "
            "for every session because the recording format declares none."
        ),
    )


def data_source_labels(values: Sequence[str]) -> str:
    """Every recorded data source, each with its display label."""
    if not values:
        return fmt.text(None)
    return "; ".join(presentation.data_source_label(value) for value in values)


def data_source_table(summary: DashboardSessionSummary) -> LabelledTable:
    """Each recorded data source, its label, and what it does not establish.

    A recorded ``data_source`` is terse — ``public_dataset``, ``live`` —
    and both of those are read by some readers as a promise about
    validity.  The label is spelled out and the disclaimer travels in the
    same row, so neither can be seen without the other.
    """
    sources = summary.provenance.data_sources
    rows = [
        (
            value,
            presentation.data_source_label(value),
            "no",
            presentation.data_source_statement(value),
        )
        for value in sources
    ]
    return fmt.build_table(
        title="Recorded data sources",
        columns=(
            "recorded value",
            "label",
            "scientifically eligible",
            "what it establishes",
        ),
        rows=rows,
        source_artifact="events.jsonl",
        caption=(
            "Read from the provenance block of the records this pass "
            "decoded. A data source says where the bytes came from. Neither "
            "'public_dataset' nor 'live' makes anything here scientifically "
            "eligible, and no dashboard control can change either field."
        ),
    )


#: Shown in place of the data-source table when nothing recorded one.
NO_DATA_SOURCE_NOTE = (
    "No record read in this pass carried a data source, so the provenance of "
    "this recording is not established. That is not the same as saying the "
    "records are not synthetic."
)


def session_metrics(summary: DashboardSessionSummary) -> tuple[MetricDisplayValue, ...]:
    """The counts a session page shows as cards."""
    return (
        fmt.count(
            "Complete records",
            summary.complete_record_count,
            source_artifact="events.jsonl",
        ),
        fmt.count(
            "Decoded this pass",
            summary.decoded_record_count,
            source_artifact="events.jsonl",
        ),
        fmt.count(
            "Malformed records",
            summary.malformed_record_count,
            source_artifact="events.jsonl",
        ),
        fmt.count(
            "Dropped under backpressure",
            summary.dropped_message_count,
            source_artifact="summary.json",
            unavailable_reason=(
                "this session has written no summary yet, so the recorder's "
                "drop count is not available"
            ),
        ),
    )


def provenance_table(summary: DashboardSessionSummary) -> LabelledTable:
    """Identity and provenance of one recording, as recorded."""
    provenance = summary.provenance
    rows = [
        ("recorded session id", fmt.text(provenance.session_id)),
        ("directory name", fmt.text(provenance.directory_name)),
        ("session directory", fmt.text(provenance.session_directory)),
        ("mode", provenance.mode.value),
        ("session format version", fmt.text(provenance.session_format_version)),
        ("protocol version", fmt.text(provenance.protocol_version)),
        ("engagevr version", fmt.text(provenance.engagevr_version)),
        ("created (recorder clock)", fmt.text(provenance.created_at_utc)),
        (
            "recorded data sources",
            fmt.text(", ".join(provenance.data_sources) or None),
        ),
        ("data source labels", data_source_labels(provenance.data_sources)),
        ("synthetic records", fmt.text(provenance.synthetic_message_count)),
        ("non-synthetic records", fmt.text(provenance.non_synthetic_message_count)),
        ("already-replayed records", fmt.text(provenance.replayed_message_count)),
        (
            "scientifically eligible",
            fmt.text(provenance.scientific_evaluation_eligible),
        ),
        ("eligibility reason", provenance.eligibility_reason),
        ("status", status_statement(summary)),
        ("session_end recorded", fmt.text(summary.session_end_recorded)),
        ("disconnect reason", fmt.text(summary.disconnect_reason)),
        ("summary rebuilt from a partial stream", fmt.text(summary.summary_recovered)),
    ]
    return fmt.build_table(
        title="Session provenance",
        columns=("field", "value"),
        rows=rows,
        source_artifact="manifest.json, summary.json, events.jsonl",
        caption=(
            "Every value is read from the recording. None is inferred from "
            "the directory name, and filesystem modification time takes no "
            "part in identity. Where the recorded id and the directory name "
            "disagree, both are shown and neither is rewritten."
        ),
    )


def record_table(
    records: Sequence[DashboardSessionRecord],
    *,
    title: str,
    max_rows: int,
    source_artifact: str = "events.jsonl",
) -> LabelledTable:
    """Records in recorded arrival order, never re-sorted."""
    rows = [
        (
            str(record.line_number),
            fmt.text(record.arrival_index),
            fmt.text(record.sequence_number),
            fmt.text(record.source),
            fmt.text(record.message_type),
            fmt.text(record.sent_at_utc),
            fmt.text(record.server_received_at_utc),
            fmt.text(record.transport),
            fmt.text(record.data_source),
            fmt.text(record.synthetic_label),
            fmt.text(record.replay_label),
            fmt.text(", ".join(record.recorded_anomalies) or None),
            fmt.text(record.problem_detail),
        )
        for record in records
    ]
    return fmt.build_table(
        title=title,
        columns=RECORD_COLUMNS,
        rows=rows,
        source_artifact=source_artifact,
        max_rows=max_rows,
        caption=(
            "Recorded arrival order, exactly as the line order in the file. "
            "Records are never re-sorted by sequence number, and a record "
            "that would not decode is listed here rather than dropped."
        ),
    )


def record_detail_table(record: DashboardSessionRecord) -> LabelledTable:
    """One record's own fields, including its payload display fields."""
    rows: list[tuple[str, str]] = [
        ("line", str(record.line_number)),
        ("arrival index", fmt.text(record.arrival_index)),
        ("sequence number", fmt.text(record.sequence_number)),
        ("message type", fmt.text(record.message_type)),
        ("source", fmt.text(record.source)),
        ("message id", fmt.text(record.message_id)),
        ("correlation id", fmt.text(record.correlation_id)),
        ("sent (sender clock)", fmt.text(record.sent_at_utc)),
        ("received (receiver clock)", fmt.text(record.server_received_at_utc)),
        ("transport", fmt.text(record.transport)),
        ("data source", fmt.text(record.data_source)),
        ("synthetic label", fmt.text(record.synthetic_label)),
        ("producer", fmt.text(record.producer)),
        ("replay label", fmt.text(record.replay_label)),
        ("replayed from session", fmt.text(record.replay_source_session_id)),
        (
            "recorded anomalies",
            fmt.text(", ".join(record.recorded_anomalies) or None),
        ),
        ("anomaly detail", fmt.text(record.anomaly_detail)),
        ("expected sequence number", fmt.text(record.expected_sequence_number)),
        ("decode problem", fmt.text(record.problem.value if record.problem else None)),
        ("decode problem detail", fmt.text(record.problem_detail)),
    ]
    rows.extend((f"payload.{name}", value) for name, value in record.payload_summary)
    return fmt.build_table(
        title=f"Record at line {record.line_number}",
        columns=("field", "value"),
        rows=rows,
        source_artifact="events.jsonl",
        caption=(f"Payload fields are task and transport telemetry. {TASK_EVENT_NOTE}"),
    )


def message_type_table(summary: DashboardSessionSummary) -> LabelledTable:
    """Message types decoded in this pass."""
    return fmt.build_table(
        title="Message types decoded",
        columns=("message type", "records"),
        rows=[(name, str(value)) for name, value in summary.message_type_counts],
        source_artifact="events.jsonl",
        caption="Counted from the records this pass decoded.",
    )


def anomaly_table(summary: DashboardSessionSummary) -> LabelledTable:
    """Anomalies the receiver recorded, and irregularities visible now.

    The two are listed side by side and labelled apart.  A recorded
    anomaly is what the receiver observed at ingestion; a derived
    observation is what the stored sequence numbers show when read back.
    Neither is repaired.
    """
    rows: list[tuple[str, str, str, str]] = [
        (
            "recorded by the receiver",
            name,
            "",
            f"{count} record(s) carry this anomaly.",
        )
        for name, count in summary.recorded_anomaly_counts
    ]
    rows.extend(
        (
            "derived from recorded sequence numbers",
            observation.kind,
            str(observation.line_number),
            observation.detail,
        )
        for observation in summary.sequence_observations
    )
    return fmt.build_table(
        title="Ordering anomalies",
        columns=("origin", "kind", "line", "detail"),
        rows=rows,
        source_artifact="events.jsonl",
        caption=(
            "Anomalies are classified, never corrected. No message has been "
            "reordered, renumbered, deduplicated, or invented for a gap."
        ),
    )


def unavailable_table(summary: DashboardSessionSummary) -> LabelledTable:
    """What a session recording cannot carry, stated rather than omitted."""
    return fmt.build_table(
        title="Not present in a session recording",
        columns=("quantity", "state"),
        rows=[
            (statement.split(":", 1)[0], statement.split(":", 1)[1].strip())
            if ":" in statement
            else (statement, "Unavailable")
            for statement in summary.unavailable_statements
        ],
        source_artifact="engagevr.dashboard.session_catalogue",
        caption=NO_ESTIMATOR_NOTE,
    )


def adaptation_table(summary: DashboardSessionSummary) -> LabelledTable:
    """Adaptation messages present in the recording.

    Transported commands and their replies.  A session recording holds no
    policy proposal and no built-but-unsent command, so those are absent
    rather than inferred, and how often a command was sent is not how
    well anything worked.
    """
    counts = summary.adaptation
    rows = [
        ("adaptation commands recorded", str(counts.commands_recorded)),
        ("acknowledgements recorded", str(counts.acknowledgements_recorded)),
        ("acknowledgements accepting", str(counts.accepted_recorded)),
        ("acknowledgements rejecting", str(counts.rejected_recorded)),
        (
            "acknowledgements carrying an applied timestamp",
            str(counts.applied_timestamp_recorded),
        ),
        ("policy proposals", "Unavailable — not recorded in this format"),
        ("commands built but not sent", "Unavailable — not recorded in this format"),
    ]
    return fmt.build_table(
        title="Adaptation messages in this recording",
        columns=("lifecycle state", "count"),
        rows=rows,
        source_artifact="events.jsonl",
        caption=(
            f"{counts.note} How often a command was transported is not a "
            "measure of adaptation effectiveness, and no such measure exists "
            "anywhere in this repository."
        ),
    )


def task_state_metrics(
    summary: DashboardSessionSummary,
) -> tuple[MetricDisplayValue, ...]:
    """Recorded task state and difficulty, as last seen in the stream."""
    return (
        fmt.count(
            "Difficulty level last recorded",
            summary.current_difficulty_level,
            source_artifact="events.jsonl",
            unavailable_reason=(
                "no task_state or session_start record in this pass carried a "
                "difficulty level"
            ),
        ),
        fmt.metric(
            "Records read in this pass",
            summary.parsed_record_count,
            kind=MetricKind.COUNT,
            source_artifact="events.jsonl",
        ),
    )


def timing_table(summary: DashboardSessionSummary) -> LabelledTable:
    """First and last timestamps, kept on the clocks that produced them."""
    return fmt.build_table(
        title="Recorded timestamps",
        columns=("field", "value"),
        rows=[
            ("first sent (sender clock)", fmt.text(summary.first_sent_at_utc)),
            ("last sent (sender clock)", fmt.text(summary.last_sent_at_utc)),
            (
                "first received (receiver clock)",
                fmt.text(summary.first_received_at_utc),
            ),
            ("last received (receiver clock)", fmt.text(summary.last_received_at_utc)),
        ],
        source_artifact="events.jsonl",
        caption=(
            "Sender and receiver timestamps come from different clocks and "
            "are never subtracted from one another here."
        ),
    )


def replay_state_for(read: SessionRead, *, position: int = 0) -> DashboardReplayState:
    """A cursor sized to what this read actually parsed."""
    total = len(read.records)
    if total == 0:
        return DashboardReplayState(total=0, position=0)
    return DashboardReplayState(total=total, position=max(0, min(position, total - 1)))


def live_appended_records(
    read: SessionRead,
) -> tuple[DashboardSessionRecord, ...]:
    """Records this pass added, which for an incremental read is all of them.

    A live view starts its next pass where this one stopped, so what a
    pass returns *is* what is new.  Stated as its own function so a page
    cannot accidentally re-render the whole file and call it an update.
    """
    return read.records


__all__ = [
    "MANUAL_REFRESH_ONLY_NOTE",
    "MINIMUM_LIVE_REFRESH_SECONDS",
    "MODE_HEADLINE",
    "MODE_TEXT",
    "NO_DATA_SOURCE_NOTE",
    "NO_ESTIMATOR_NOTE",
    "RECORD_COLUMNS",
    "SESSION_STATUS_TEXT",
    "TASK_EVENT_NOTE",
    "adaptation_table",
    "anomaly_table",
    "catalogue_table",
    "data_source_labels",
    "data_source_table",
    "live_appended_records",
    "live_refresh_interval",
    "message_type_table",
    "mode_headline",
    "mode_statement",
    "provenance_table",
    "record_detail_table",
    "record_table",
    "refresh_statement",
    "replay_state_for",
    "session_metrics",
    "status_statement",
    "task_state_metrics",
    "timing_table",
    "unavailable_table",
]
