"""The live-observation and replay pages.

Two pages, one context type, and one hard rule enforced in code rather
than in prose: a page renders only its own mode.
:func:`live_session_page` refuses a replay context and
:func:`replay_page` refuses a live one, so neither can be reached under
the other's heading by a routing mistake.  A live view is the most
persuasive thing this dashboard can show, and "it said LIVE at the top"
is not a guarantee worth relying on.

The record a page shows is a record the recorder had already persisted
before that page existed.

Refresh policy
--------------
The **live** page — and only the live page — refreshes on its own, using
``st.fragment(run_every=...)`` at the interval configured in
``dashboard.live_refresh_seconds`` (DEC-094, revised).  Each firing
re-reads the recording with the read-only session reader and redraws
what it finds; nothing is cached across firings, so an appended record
appears on the next one.  The manual **Read new records** button remains
for a reader who wants the next pass now.

The interval is validated by
:func:`engagevr.dashboard.views_session.live_refresh_interval` before it
reaches Streamlit, and a refused value switches the timer off with the
reason stated rather than being clamped to a cadence nobody chose.

Replay does not auto-advance and the artifact observatory does not poll.
Replay's cursor moves only when one of its own controls is used, which is
why the fragment is constructed inside :func:`live_session_page` rather
than around anything shared.

*Real-time observation is not real-time inference.*  What refreshes is a
view of records another subsystem already wrote.  Neither page runs a
model, opens a camera, starts a server, opens a socket, re-emits a
message, or writes to the recording it is reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from engagevr.dashboard import components as ui
from engagevr.dashboard import views_session as views
from engagevr.dashboard.session_catalogue import SessionRead, read_session
from engagevr.dashboard.session_report import (
    SessionReportError,
    build_report,
    report_to_json,
    report_to_markdown,
)
from engagevr.schemas.dashboard import DashboardError
from engagevr.schemas.dashboard_session import (
    LIVE_OBSERVATION_NOTE,
    REPLAY_PRESENTATION_NOTE,
    DashboardReplayState,
    DashboardSessionCatalogue,
    DashboardSessionMode,
    DashboardSessionStatus,
    DashboardSessionSummary,
)

#: Records the live page shows in its tail table.
LIVE_TAIL_RECORDS = 50

#: Session-state keys, namespaced so nothing else collides with them.
LIVE_COUNT_KEY = "engagevr_live_previous_record_count"
REPLAY_POSITION_KEY = "engagevr_replay_position"
REPLAY_SESSION_KEY = "engagevr_replay_session_id"
REPLAY_SLIDER_KEY = "engagevr_replay_slider"


@dataclass(frozen=True)
class SessionPageContext:
    """Everything a session page needs.

    ``mode`` travels as a value rather than being implied by which
    function was called, so a page can check it and a report can record
    which mode produced it.
    """

    mode: DashboardSessionMode
    session_root: Path
    catalogue: DashboardSessionCatalogue
    #: Directory of the selected recording, not its recorded session id.
    #: The two can differ, and only the directory addresses a file.
    session_directory_name: str | None
    max_table_rows: int
    live_refresh_seconds: float
    enable_session_report_export: bool = True


def _require_mode(
    context: SessionPageContext, expected: DashboardSessionMode, label: str
) -> bool:
    """Refuse to render a page under the wrong evidence mode."""
    if context.mode is expected:
        return True
    st.error(
        f"This is the {label} view, but the selected evidence source is "
        f"{context.mode.value!r}. The two are different kinds of evidence and "
        "are never rendered under each other's heading. Choose "
        f"{expected.value!r} in the sidebar.",
        icon="🛑",
    )
    return False


def _require_session(context: SessionPageContext) -> SessionRead | None:
    """Read the selected recording, or state why nothing is shown."""
    catalogue = context.catalogue
    if not catalogue.root_exists:
        ui.render_warnings(catalogue.warnings)
        ui.unavailable(
            f"The session root {catalogue.session_root} does not exist. No "
            "session has been recorded on this machine yet; the dashboard "
            "runs, there is simply nothing to observe."
        )
        return None
    if catalogue.is_empty:
        ui.render_warnings(catalogue.warnings)
        ui.unavailable(
            f"No recorded session was found under {catalogue.session_root}. "
            "One can be produced with a documented CLI command such as "
            "`uv run python -m engagevr task-sim`."
        )
        return None
    if context.session_directory_name is None:
        ui.unavailable("No session is selected. Choose one in the sidebar.")
        return None

    selected = catalogue.find(context.session_directory_name)
    if selected is None:
        ui.unavailable(
            f"No recording named {context.session_directory_name!r} is listed "
            "under this session root. It may have been removed while this "
            "page was open."
        )
        return None

    # Addressed by directory, never by the recorded session id: a
    # recording copied for comparison keeps its recorded id under a new
    # folder name, and resolving the id would then read the wrong file or
    # none at all.
    directory = Path(selected.provenance.session_directory)
    try:
        return read_session(
            directory, mode=context.mode, max_records=context.max_table_rows
        )
    except OSError as exc:
        st.error(
            f"The session directory {directory} could not be read: {exc}. It "
            "may have been removed while this page was open. Nothing has been "
            "substituted for it.",
            icon="🛑",
        )
        return None


def _render_common(read: SessionRead, context: SessionPageContext) -> None:
    """The blocks both session pages share, in the same order."""
    summary = read.summary
    ui.session_provenance_banner(summary)
    st.caption(views.mode_statement(context.mode))

    ui.render_metrics(list(views.session_metrics(summary)))
    ui.render_metrics(list(views.task_state_metrics(summary)))
    if summary.unparsed_record_count:
        st.info(
            f"{summary.unparsed_record_count} further complete record(s) exist "
            "in this recording and were not read into this view because of "
            "the configured display limit. The recording on disk is complete.",
        )

    ui.section("Session provenance")
    ui.render_table(views.provenance_table(summary))

    ui.section(
        "Recorded data sources",
        "Where the bytes came from, in words. Not a statement about validity.",
    )
    ui.render_table(
        views.data_source_table(summary),
        empty_reason=views.NO_DATA_SOURCE_NOTE,
    )

    ui.section("Recorded timestamps")
    ui.render_table(views.timing_table(summary))

    ui.section("Composition")
    ui.render_table(views.message_type_table(summary))

    ui.section(
        "Ordering anomalies",
        "Recorded and derived irregularities, neither of them repaired.",
    )
    ui.render_table(views.anomaly_table(summary))

    ui.section(
        "Adaptation messages",
        "Transport lifecycle only. How often a command was sent is not how "
        "well anything worked.",
    )
    ui.render_table(views.adaptation_table(summary))

    ui.section(
        "Measurement quality and estimates",
        views.NO_ESTIMATOR_NOTE,
    )
    ui.render_table(views.unavailable_table(summary))
    st.caption(
        "Measurement quality is not engagement, is not cognitive load, and is "
        "not model confidence. None of the three is recorded here."
    )


def _render_export(read: SessionRead, context: SessionPageContext) -> None:
    """The session-report export, offered from both session modes."""
    st.divider()
    ui.section(
        "Session report",
        "A deterministic presentation artifact built from the records above. "
        "It is not an experiment result and creates no new evidence.",
    )
    if not context.enable_session_report_export:
        ui.unavailable(
            "Session-report export is switched off in the dashboard configuration."
        )
        return
    try:
        report = build_report(read, mode=context.mode)
    except SessionReportError as exc:
        ui.unavailable(f"No report was built: {exc}")
        return

    st.caption(
        f"Report fingerprint: `{report.report_fingerprint}` (schema "
        f"{report.report_schema_version}). The fingerprint covers the report's "
        "content. Export time takes no part in it, so the same recording "
        "reported twice gives the same fingerprint."
    )
    if report.synthetic_banner:
        st.error(f"**{report.synthetic_banner}**", icon="⚠️")
    st.caption(
        "The exported report permanently carries is_synthetic, "
        "scientific_evaluation_eligible = false, the standing disclaimer, and "
        "the software-self-check banner where it applies. There is no export "
        "path that removes them."
    )

    json_text = report_to_json(report)
    markdown_text = report_to_markdown(report)
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download session report (JSON)",
            data=json_text,
            file_name=f"engagevr-session-report-{report.session_id}.json",
            mime="application/json",
        )
    with right:
        st.download_button(
            "Download session report (Markdown)",
            data=markdown_text,
            file_name=f"engagevr-session-report-{report.session_id}.md",
            mime="text/markdown",
        )
    st.caption(
        "Downloading sends a copy to the browser. The recording on disk is "
        "not touched, and the report lists a SHA-256 of every source file so "
        "that can be verified afterwards."
    )
    with st.expander("Preview the report"):
        st.markdown(markdown_text)


# --- Live observation ----------------------------------------------------


def live_session_page(context: SessionPageContext) -> None:
    """Read-only observation of a session recording as it is written.

    The heading, the scope warning, and the refresh statement are drawn
    once, outside the auto-refreshing fragment: a statement about what
    this view is may not be something a timer can remove.  Everything
    that depends on the recording is drawn inside the fragment, so each
    firing re-reads the file rather than redrawing a cached read.
    """
    st.title("Live session observation")
    st.warning(LIVE_OBSERVATION_NOTE, icon="⚠️")
    if not _require_mode(context, DashboardSessionMode.LIVE, "live observation"):
        return
    st.markdown(f"**Mode: {views.mode_headline(DashboardSessionMode.LIVE)}**")

    interval = _live_refresh_interval(context)
    if interval is None:
        st.caption(views.MANUAL_REFRESH_ONLY_NOTE)
        _render_live_body(context)
        return

    st.caption(views.refresh_statement(interval))

    @st.fragment(run_every=interval)
    def _auto_refreshing_live_view() -> None:
        _render_live_body(context)

    _auto_refreshing_live_view()


def _live_refresh_interval(context: SessionPageContext) -> float | None:
    """The validated automatic-refresh interval, or ``None`` if refused.

    A refused interval is reported with its reason and switches the timer
    off.  It is deliberately not clamped to the minimum: a page
    refreshing at a cadence nobody configured is exactly the sort of
    unattributed behaviour this dashboard must not have.
    """
    try:
        return views.live_refresh_interval(context.live_refresh_seconds)
    except DashboardError as exc:
        st.error(
            f"The configured automatic-refresh interval was refused: {exc}. "
            "No timer has been started. This page still re-reads the "
            "recording when you press Read new records, and nothing else "
            "about it has changed.",
            icon="🛑",
        )
        return None


def _render_live_body(context: SessionPageContext) -> None:
    """Everything a live pass re-reads and redraws.

    Called once per refresh, and it starts by reading the recording
    again.  Nothing between here and the file is cached, which is the
    whole point: a cached read would render a live view that cannot show
    an appended record.
    """
    read = _require_session(context)
    if read is None:
        return
    summary = read.summary
    _render_live_controls(summary, context)
    _render_common(read, context)

    st.divider()
    ui.section(
        "Most recent records",
        "Recorded arrival order. Nothing here is a model output.",
    )
    tail = read.records[-LIVE_TAIL_RECORDS:]
    ui.render_table(
        views.record_table(
            tail,
            title=f"Last {len(tail)} record(s)",
            max_rows=context.max_table_rows,
        )
    )
    st.caption(views.TASK_EVENT_NOTE)
    _render_export(read, context)


def _render_live_controls(
    summary: DashboardSessionSummary, context: SessionPageContext
) -> None:
    """The refresh control and what changed since the previous read."""
    del context  # the cadence is stated by the page, not by this block
    previous = st.session_state.get(LIVE_COUNT_KEY)
    current = summary.complete_record_count
    st.button("Read new records")
    st.caption(
        "Automatic refresh re-reads this recording on its own; this button "
        "asks for the next pass now. Every pass reads the file on disk. "
        "Nothing is interpolated, extrapolated, or fabricated between two "
        "passes, and no value on this page is produced by a model."
    )
    if isinstance(previous, int):
        appended = current - previous
        if appended > 0:
            st.success(
                f"{appended} new complete record(s) since the previous read.",
                icon="✅",
            )
        elif appended == 0:
            st.info("No new complete record since the previous read.")
        else:
            st.error(
                f"This recording now holds {current} complete record(s), fewer "
                f"than the {previous} seen previously. A session recording is "
                "append-only, so the file has been replaced or truncated "
                "outside this dashboard.",
                icon="🛑",
            )
    st.session_state[LIVE_COUNT_KEY] = current

    status = summary.provenance.status
    if status is DashboardSessionStatus.COMPLETED:
        st.info(
            "This session is completed: a session_end was recorded and the "
            "summary was written. No further record will appear, so this is "
            "an observation of a finished recording.",
        )
    elif status is DashboardSessionStatus.ACTIVE_OR_INCOMPLETE:
        st.info(
            "This session has written no summary yet. It may still be "
            "running, or it may have stopped. Both are legitimate states and "
            "neither is a failure.",
        )


# --- Replay --------------------------------------------------------------


def replay_page(context: SessionPageContext) -> None:
    """Deterministic read-only navigation through a recorded session."""
    st.title("Session replay")
    st.warning(REPLAY_PRESENTATION_NOTE, icon="⚠️")
    if not _require_mode(context, DashboardSessionMode.REPLAY, "replay"):
        return
    st.markdown(f"**Mode: {views.mode_headline(DashboardSessionMode.REPLAY)}**")
    st.caption(
        "Replay does not advance on its own. The cursor moves only when one "
        "of the controls below is used, and the live mode's automatic "
        "refresh does not reach this page."
    )

    read = _require_session(context)
    if read is None:
        return
    _render_common(read, context)

    st.divider()
    ui.section(
        "Replay position",
        "Navigation through records already on disk. Nothing is re-emitted, "
        "re-simulated, re-inferred, or repaired.",
    )
    state = _replay_state(read, context)
    if state.is_empty:
        ui.unavailable(
            "This recording holds no complete record yet, so there is nothing "
            "to step through."
        )
        _render_export(read, context)
        return

    st.markdown(f"**Record {state.human_position}**")
    st.progress((state.position + 1) / state.total)
    record = read.records[state.position]
    ui.render_table(views.record_detail_table(record))
    if record.is_replayed:
        st.info(
            "This record already carried replay metadata when it was written: "
            f"it was originally recorded under session "
            f"{record.replay_source_session_id!r} and re-emitted into this "
            "one. Presenting it here does not make it a new observation.",
        )
    if record.is_synthetic:
        st.caption(
            "This record was permanently labelled SYNTHETIC when it was "
            "written. Replaying it does not make it participant data."
        )
    if not record.decoded:
        st.error(
            f"This record could not be decoded: {record.problem_detail}",
            icon="🛑",
        )

    st.divider()
    ui.section("Full recorded sequence")
    ui.render_table(
        views.record_table(
            read.records,
            title="Every record read, in recorded arrival order",
            max_rows=context.max_table_rows,
        )
    )
    st.caption(views.TASK_EVENT_NOTE)
    _render_export(read, context)


def _replay_state(
    read: SessionRead, context: SessionPageContext
) -> DashboardReplayState:
    """The replay cursor, kept in session state and clamped to the recording.

    The slider is the single owner of the position, and the buttons move
    it by writing its key before it is created.  Having the buttons keep
    a *separate* position would mean the slider silently overrode every
    button press on the following rerun, which is precisely the sort of
    quiet disagreement between two controls that makes a navigation view
    untrustworthy.

    Every move goes through
    :class:`~engagevr.schemas.dashboard_session.DashboardReplayState`, so
    the clamping at both ends is the model's, not this function's.
    """
    del context  # the cursor depends on the recording, not the settings
    key = read.summary.provenance.directory_name
    total = len(read.records)
    if st.session_state.get(REPLAY_SESSION_KEY) != key:
        st.session_state[REPLAY_SESSION_KEY] = key
        st.session_state[REPLAY_SLIDER_KEY] = 1
    if total == 0:
        st.session_state[REPLAY_POSITION_KEY] = 0
        return DashboardReplayState(total=0, position=0)

    stored = st.session_state.get(REPLAY_SLIDER_KEY, 1)
    current = stored if isinstance(stored, int) else 1
    state = views.replay_state_for(read, position=current - 1)
    st.session_state[REPLAY_SLIDER_KEY] = state.position + 1

    first, back, forward, last = st.columns(4)
    for column, label, action, disabled in (
        (first, "Jump to beginning", "first", state.at_first),
        (back, "Step backward", "backward", state.at_first),
        (forward, "Step forward", "forward", state.at_last),
        (last, "Jump to end", "last", state.at_last),
    ):
        with column:
            st.button(
                label,
                disabled=disabled,
                on_click=_move_cursor,
                args=(action, state.total),
            )

    if state.total > 1:
        st.slider(
            "Position in the recorded sequence",
            min_value=1,
            max_value=state.total,
            key=REPLAY_SLIDER_KEY,
        )

    st.session_state[REPLAY_POSITION_KEY] = state.position
    return state


def _move_cursor(action: str, total: int) -> None:
    """Move the cursor before the page reruns.

    A callback rather than a return value from
    :func:`streamlit.button`, so the position is already updated when the
    script re-runs.  Reading the click after the controls were drawn
    would leave every button's enabled state one interaction behind what
    the page displays — a small lag, but one that shows "step forward"
    as available on the last record.
    """
    stored = st.session_state.get(REPLAY_SLIDER_KEY, 1)
    current = stored if isinstance(stored, int) else 1
    position = max(0, min(current - 1, max(total - 1, 0)))
    state = DashboardReplayState(total=total, position=position)
    moved = {
        "first": state.first,
        "last": state.last,
        "forward": state.step_forward,
        "backward": state.step_backward,
    }[action]()
    st.session_state[REPLAY_SLIDER_KEY] = moved.position + 1


#: The pages of the two session modes, keyed by mode.
SESSION_PAGES: dict[DashboardSessionMode, str] = {
    DashboardSessionMode.LIVE: "live_session_page",
    DashboardSessionMode.REPLAY: "replay_page",
}


__all__ = [
    "LIVE_COUNT_KEY",
    "LIVE_TAIL_RECORDS",
    "REPLAY_POSITION_KEY",
    "REPLAY_SESSION_KEY",
    "REPLAY_SLIDER_KEY",
    "SESSION_PAGES",
    "SessionPageContext",
    "live_session_page",
    "replay_page",
]
