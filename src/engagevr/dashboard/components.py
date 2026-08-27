"""Reusable Streamlit components.

This is the first module in the dashboard that imports Streamlit.
Everything beneath it — the catalogue, the loaders, the view builders,
the formatting — is framework-free, which is why the unit tests never
need a server.

:func:`provenance_banner` is the component every result-bearing page must
call.  It renders the software-self-check banner when the run is
synthetic, states the scientific-eligibility flag in words, and prints
the run's identity.  It is a function rather than a convention so that a
test can check each page calls it, and so that no page can render a
synthetic result under a heading that says nothing about where the
numbers came from.

Colour is never the only carrier of meaning here.  Every status has text
beside it, every class label is written out, and no green tick ever
stands for *psychologically good*.
"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from engagevr.dashboard import formatting as fmt
from engagevr.dashboard import presentation, views_session
from engagevr.dashboard.views_session import (
    MODE_HEADLINE,
    MODE_TEXT,
    NO_ESTIMATOR_NOTE,
    SESSION_STATUS_TEXT,
)
from engagevr.schemas.dashboard import (
    DASHBOARD_DISCLAIMER,
    PROVENANCE_PROPAGATION_NOTE,
    SYNTHETIC_BANNER,
    ArtifactIntegrityStatus,
    ConfusionMatrixView,
    DashboardProvenance,
    DashboardRunStatus,
    DashboardWarning,
    DashboardWarningLevel,
    LabelledChart,
    LabelledTable,
    MetricDisplayValue,
)
from engagevr.schemas.dashboard_session import (
    SESSION_CONTENT_NOTE,
    DashboardSessionSummary,
)

#: Wording of each integrity status, so the state is never colour-only.
INTEGRITY_TEXT: dict[ArtifactIntegrityStatus, str] = {
    ArtifactIntegrityStatus.NOT_CHECKED: "Checksums not verified",
    ArtifactIntegrityStatus.VALID: "Checksums match the recorded digests",
    ArtifactIntegrityStatus.MISMATCHED: (
        "CHECKSUM MISMATCH — the bytes on disk differ from what this run recorded"
    ),
    ArtifactIntegrityStatus.CHECKSUM_FILE_UNAVAILABLE: (
        "No checksums.json — integrity could not be checked"
    ),
    ArtifactIntegrityStatus.REFERENCED_FILE_MISSING: (
        "A file named in checksums.json is missing"
    ),
    ArtifactIntegrityStatus.CHECKSUM_FILE_CORRUPT: (
        "checksums.json could not be parsed"
    ),
}

#: Wording of each run status.
STATUS_TEXT: dict[DashboardRunStatus, str] = {
    DashboardRunStatus.COMPLETED: "Completed",
    DashboardRunStatus.FAILED: "FAILED — this run recorded a failure",
    DashboardRunStatus.INCOMPLETE: (
        "INCOMPLETE — a required artifact is missing; this is not a successful run"
    ),
    DashboardRunStatus.CORRUPT: "CORRUPT — an artifact could not be read",
    DashboardRunStatus.UNSUPPORTED: (
        "UNSUPPORTED — this artifact declares a version this dashboard cannot interpret"
    ),
    DashboardRunStatus.UNKNOWN: "UNKNOWN — nothing recognisable was found",
}


def provenance_banner(provenance: DashboardProvenance) -> None:
    """Render the mandatory provenance block at the top of a page.

    Not an expander, not a footer, not a tooltip.  A reader who does not
    scroll must still see whether these numbers are a software
    self-check.
    """
    if provenance.requires_synthetic_banner:
        st.error(f"**{SYNTHETIC_BANNER}**", icon="⚠️")
    st.warning(DASHBOARD_DISCLAIMER)

    eligibility = (
        "scientific_evaluation_eligible = **false**. Nothing on this page "
        "is scientific evidence."
        if not provenance.scientific_evaluation_eligible
        else "scientific_evaluation_eligible = **true** as recorded by this artifact."
    )
    st.markdown(eligibility)

    left, middle, right = st.columns(3)
    with left:
        st.markdown("**Run**")
        st.text(provenance.run_id)
        st.caption(f"directory: {provenance.run_directory}")
        st.caption(f"family: {provenance.family.value}")
    with middle:
        st.markdown("**Data**")
        st.text(f"source: {presentation.data_source_label(provenance.data_source)}")
        st.text(f"synthetic: {fmt.text(provenance.is_synthetic)}")
        st.text(f"evaluation mode: {fmt.text(provenance.evaluation_mode)}")
    with right:
        st.markdown("**Target**")
        st.text(f"target: {fmt.text(provenance.target_name)}")
        st.text(f"task type: {fmt.text(provenance.task_type)}")
        st.text(f"model source: {fmt.text(provenance.model_source)}")

    st.markdown(presentation.data_source_statement(provenance.data_source))
    st.text(f"status: {STATUS_TEXT[provenance.status]}")
    st.text(f"integrity: {INTEGRITY_TEXT[provenance.integrity]}")
    if provenance.dataset_fingerprint:
        st.caption(f"dataset fingerprint: {provenance.dataset_fingerprint}")
    if provenance.split_manifest_fingerprint:
        st.caption(f"split fingerprint: {provenance.split_manifest_fingerprint}")
    if provenance.finished_at_utc:
        st.caption(
            f"finished (display metadata only, not provenance): "
            f"{provenance.finished_at_utc}"
        )
    if provenance.failure_reason:
        st.error(f"Failure reason: {provenance.failure_reason}")
    render_warnings(provenance.warnings)
    st.caption(PROVENANCE_PROPAGATION_NOTE)
    st.divider()


def session_provenance_banner(summary: DashboardSessionSummary) -> None:
    """The mandatory provenance block for a live or replay view.

    The session equivalent of :func:`provenance_banner`, and separate
    from it because the two provenance contracts differ: an experiment
    run declares its scientific eligibility, a session recording has no
    field in which to declare one.  Sharing a banner would mean sharing a
    claim.

    A live display is the most persuasive thing this repository can put
    on a screen, so the scope statement is rendered before anything else
    and never inside an expander.
    """
    provenance = summary.provenance
    if provenance.is_synthetic:
        st.error(f"**{SYNTHETIC_BANNER}**", icon="⚠️")
    st.warning(DASHBOARD_DISCLAIMER)
    st.markdown(
        f"**Mode: {MODE_HEADLINE[provenance.mode]}** — {MODE_TEXT[provenance.mode]}"
    )
    st.markdown(
        "scientific_evaluation_eligible = **false**. " + provenance.eligibility_reason
    )
    for statement in presentation.data_source_statements(provenance.data_sources):
        st.markdown(statement)
    if not provenance.provenance_established:
        st.info(
            "No record has been read from this session yet, so its data "
            "source is not established. That is not the same as saying the "
            "records are not synthetic.",
        )

    left, middle, right = st.columns(3)
    with left:
        st.markdown("**Session**")
        st.text(provenance.session_id)
        st.caption(f"directory: {provenance.session_directory}")
        st.caption(f"format: {fmt.text(provenance.session_format_version)}")
    with middle:
        st.markdown("**Records**")
        st.text(
            f"data sources: {views_session.data_source_labels(provenance.data_sources)}"
        )
        st.text(f"synthetic: {provenance.synthetic_message_count}")
        st.text(f"already replayed: {provenance.replayed_message_count}")
    with right:
        st.markdown("**State**")
        st.text(f"status: {provenance.status.value}")
        st.text(f"protocol: {fmt.text(provenance.protocol_version)}")
        st.text(f"complete records: {summary.complete_record_count}")

    st.text(SESSION_STATUS_TEXT[provenance.status])
    if provenance.status_reason:
        st.caption(provenance.status_reason)
    if summary.partial_trailing_line and summary.partial_trailing_note:
        st.info(summary.partial_trailing_note)
    if summary.malformed_record_count:
        st.error(
            f"{summary.malformed_record_count} complete record(s) could not be "
            "decoded and are listed with their line numbers rather than "
            "hidden: lines "
            + ", ".join(str(n) for n in summary.malformed_line_numbers)
            + ".",
            icon="🛑",
        )
    render_warnings(summary.warnings)
    st.caption(SESSION_CONTENT_NOTE)
    st.caption(NO_ESTIMATOR_NOTE)
    st.caption(PROVENANCE_PROPAGATION_NOTE)
    st.divider()


def render_warnings(warnings: Sequence[DashboardWarning]) -> None:
    """Render warnings at their declared level, with text not just colour."""
    for entry in warnings:
        subject = f"[{entry.subject}] " if entry.subject else ""
        message = f"{subject}{entry.message}"
        if entry.level is DashboardWarningLevel.ERROR:
            st.error(f"**Error.** {message}", icon="🛑")
        elif entry.level is DashboardWarningLevel.WARNING:
            st.warning(f"**Warning.** {message}", icon="⚠️")
        else:
            st.info(f"**Note.** {message}")


def render_table(
    table: LabelledTable | None, *, empty_reason: str | None = None
) -> None:
    """Render a labelled table, or state why there is none."""
    if table is None:
        st.info(
            empty_reason
            or "This view is unavailable for the selected run. No value has "
            "been substituted for it.",
        )
        return
    st.markdown(f"**{table.title}**")
    if table.rows:
        st.dataframe(
            [dict(zip(table.columns, row, strict=True)) for row in table.rows],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "This table has no rows. That is what the artifact recorded; it "
            "is not a zero.",
        )
    note = fmt.truncation_note(table)
    if note:
        st.warning(note, icon="⚠️")
    if table.caption:
        st.caption(table.caption)
    if table.source_artifact:
        st.caption(f"Source: {table.source_artifact}")


def render_chart(chart: LabelledChart | None) -> None:
    """Render a labelled chart, or state exactly why it cannot be drawn."""
    if chart is None:
        st.info(
            "This chart is unavailable for the selected run.",
        )
        return
    st.markdown(f"**{chart.title}**")
    if chart.subtitle:
        st.caption(chart.subtitle)
    if not chart.available:
        st.info(
            f"Not drawn: {chart.unavailable_reason}",
        )
        _chart_footer(chart)
        return

    rows: list[dict[str, float | None]] = []
    x_values = sorted({x for series in chart.series for x in series.x_values})
    for x in x_values:
        row: dict[str, float | None] = {chart.x_axis_label: x}
        for series in chart.series:
            lookup = dict(zip(series.x_values, series.y_values, strict=True))
            row[series.name] = lookup.get(x)
        rows.append(row)
    st.line_chart(
        rows,
        x=chart.x_axis_label,
        y=[series.name for series in chart.series],
        x_label=chart.x_axis_label,
        y_label=chart.y_axis_label,
    )
    with st.expander("Table of the plotted values"):
        st.dataframe(rows, width="stretch", hide_index=True)
    _chart_footer(chart)


def render_scatter(chart: LabelledChart | None) -> None:
    """Render a scatter chart, or state why it cannot be drawn."""
    if chart is None:
        st.info("This chart is unavailable for the selected run.")
        return
    st.markdown(f"**{chart.title}**")
    if chart.subtitle:
        st.caption(chart.subtitle)
    if not chart.available:
        st.info(f"Not drawn: {chart.unavailable_reason}")
        _chart_footer(chart)
        return
    rows = [
        {
            chart.x_axis_label: x,
            chart.y_axis_label: y,
            "series": series.name,
        }
        for series in chart.series
        for x, y in zip(series.x_values, series.y_values, strict=True)
        if y is not None
    ]
    st.scatter_chart(
        rows,
        x=chart.x_axis_label,
        y=chart.y_axis_label,
        color="series",
        x_label=chart.x_axis_label,
        y_label=chart.y_axis_label,
    )
    _chart_footer(chart)


def render_bar_chart(chart: LabelledChart | None) -> None:
    """Render a histogram or bar chart, or state why it cannot be drawn."""
    if chart is None:
        st.info("This chart is unavailable for the selected run.")
        return
    st.markdown(f"**{chart.title}**")
    if chart.subtitle:
        st.caption(chart.subtitle)
    if not chart.available:
        st.info(f"Not drawn: {chart.unavailable_reason}")
        _chart_footer(chart)
        return
    rows = [
        {chart.x_axis_label: x, series.name: y}
        for series in chart.series
        for x, y in zip(series.x_values, series.y_values, strict=True)
    ]
    st.bar_chart(
        rows,
        x=chart.x_axis_label,
        y=[series.name for series in chart.series],
        x_label=chart.x_axis_label,
        y_label=chart.y_axis_label,
    )
    _chart_footer(chart)


def _chart_footer(chart: LabelledChart) -> None:
    if chart.x_axis_note:
        st.caption(chart.x_axis_note)
    if chart.source_artifact:
        st.caption(f"Source: {chart.source_artifact}")


def render_metrics(metrics: Sequence[MetricDisplayValue], *, columns: int = 4) -> None:
    """Render compact metric cards. An absent value shows *Unavailable*."""
    if not metrics:
        return
    slots = st.columns(min(columns, len(metrics)))
    for index, entry in enumerate(metrics):
        with slots[index % len(slots)]:
            st.metric(entry.name, fmt.format_value(entry))
            if not entry.available and entry.unavailable_reason:
                st.caption(entry.unavailable_reason)
            elif entry.source_artifact:
                st.caption(f"Source: {entry.source_artifact}")


def render_confusion_matrix(matrix: ConfusionMatrixView) -> None:
    """Render a confusion matrix as a labelled table.

    A table rather than a heat map: colour intensity is not a reliable
    carrier of a count, and the row axis wording is the point of this
    view.
    """
    st.markdown(
        f"**Confusion matrix** — rows: {matrix.row_axis_label}; "
        f"columns: {matrix.column_axis_label}"
    )
    rows = []
    for label, counts in zip(matrix.labels, matrix.counts, strict=True):
        row: dict[str, str] = {matrix.row_axis_label: label}
        for column, value in zip(matrix.labels, counts, strict=True):
            row[f"predicted {column}"] = str(value)
        row["row total"] = str(sum(counts))
        rows.append(row)
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        f"Total windows: {matrix.total}. Row totals are the recorded support "
        "for each label."
    )
    if matrix.source_artifact:
        st.caption(f"Source: {matrix.source_artifact}")


def unavailable(reason: str) -> None:
    """State why a whole section cannot be rendered."""
    st.info(reason)


def section(title: str, note: str | None = None) -> None:
    """A section heading with an optional standing note."""
    st.subheader(title)
    if note:
        st.caption(note)


__all__ = [
    "INTEGRITY_TEXT",
    "STATUS_TEXT",
    "provenance_banner",
    "render_bar_chart",
    "render_chart",
    "render_confusion_matrix",
    "render_metrics",
    "render_scatter",
    "render_table",
    "render_warnings",
    "section",
    "session_provenance_banner",
    "unavailable",
]
