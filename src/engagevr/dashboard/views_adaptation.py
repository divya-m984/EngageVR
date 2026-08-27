"""Milestone 8 adaptation-controller views.

This page reports **controller behaviour**.  It has no effectiveness
card, no benefit metric, and no field in which one could be written,
because Milestone 8 produced no evidence that any adaptation helped
anyone.  How often a controller acts is not how well it works.

Three separations are load-bearing.

**Proposal, command, dispatch, acknowledgement.**  These are four
distinct events and they are counted in four distinct fields of
:class:`~engagevr.schemas.dashboard.AdaptationLifecycleCounts`, which
refuses an ordering that could not have happened.  For the current
Milestone 8 runs the honest reading is: proposals above zero, commands
built above zero, dispatched zero, acknowledged zero.  Nothing reached
Unity, and the page must not imply that it did.

**Hold is a decision.**  Most windows hold, and each hold records which
guard produced it.  A hold is not a failure of the controller; it is the
controller working.

**Static is a condition.**  A run in static experiment mode holds on
every window by design.  It is a legitimate experimental control, not a
malfunction and not a statement about anybody's engagement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from engagevr.dashboard import formatting as fmt
from engagevr.dashboard.loaders import (
    DEFAULT_MAX_TABLE_ROWS,
    ArtifactReadError,
    filter_rows,
    read_parquet,
    try_document,
    unreadable_artifact_warning,
)
from engagevr.dashboard.presentation import (
    ADAPTATION_NOTE,
    CONTROLLER_COMPARISON_NOTE,
    STATIC_MODE_NOTE,
)
from engagevr.schemas.dashboard import (
    AdaptationDashboardData,
    AdaptationLifecycleCounts,
    ChartSeries,
    DashboardRunSummary,
    DashboardWarning,
    DashboardWarningLevel,
    LabelledChart,
    MetricKind,
)

#: Columns of ``adaptation_trace.parquet`` the difficulty plot needs.
TRACE_COLUMNS: tuple[str, ...] = (
    "session_id",
    "window_order",
    "current_difficulty",
    "proposed_difficulty",
    "decision_kind",
    "resolved_direction",
    "policy_reasons",
    "command_built",
    "lifecycle_status",
    "cooldown_after",
    "persistence_after",
)

#: Lifecycle statuses that imply the command left this process.
_DISPATCHED_ONWARDS: frozenset[str] = frozenset(
    {"dispatched", "acknowledged", "applied", "rejected"}
)

#: Lifecycle statuses reachable only from a real Milestone 4 reply.
#: ``rejected`` counts as an acknowledgement: the environment answered.
_ACKNOWLEDGED_ONWARDS: frozenset[str] = frozenset(
    {"acknowledged", "applied", "rejected"}
)

#: Metrics reported about the controller, with the display kind of each.
#: None of them is effectiveness. Window counts are counts and render as
#: whole numbers; only the adaptation fraction is a real-valued ratio,
#: and calling a count "2.0000" would suggest a measurement precision the
#: quantity does not have.
CONTROLLER_METRICS: tuple[tuple[str, str, MetricKind], ...] = (
    ("evaluated_windows", "Evaluated windows", MetricKind.COUNT),
    ("gate_eligible_windows", "Milestone 7 gate eligible", MetricKind.COUNT),
    ("gate_blocked_windows", "Milestone 7 gate blocked", MetricKind.COUNT),
    ("hold_decisions", "HOLD decisions", MetricKind.COUNT),
    ("adaptation_proposals", "Adaptation proposals", MetricKind.COUNT),
    ("increases", "Proposed increases", MetricKind.COUNT),
    ("decreases", "Proposed decreases", MetricKind.COUNT),
    ("direction_reversals", "Direction reversals", MetricKind.COUNT),
    (
        "minimum_proposal_spacing_windows",
        "Minimum proposal spacing (windows)",
        MetricKind.COUNT,
    ),
    (
        "longest_same_direction_streak",
        "Longest same-direction streak (windows)",
        MetricKind.COUNT,
    ),
    ("blocked_oscillation_attempts", "Blocked oscillation attempts", MetricKind.COUNT),
    (
        "eligible_window_adaptation_fraction",
        "Proposals per eligible window (fraction)",
        MetricKind.REAL,
    ),
)


def load_adaptation(
    run: DashboardRunSummary, *, max_rows: int = DEFAULT_MAX_TABLE_ROWS
) -> AdaptationDashboardData:
    """Build the controller view for one Milestone 8 run."""
    provenance = run.provenance
    document, error = try_document(run, "adaptation_summary.json")
    if document is None:
        return AdaptationDashboardData(
            provenance=provenance,
            warnings=(
                unreadable_artifact_warning(run, "adaptation_summary.json", str(error)),
            ),
            unavailable_reason=f"adaptation_summary.json is unavailable: {error}",
        )

    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        return AdaptationDashboardData(
            provenance=provenance,
            unavailable_reason=(
                "adaptation_summary.json records no controller metrics, so "
                "there is nothing to display for this run."
            ),
        )

    configuration = document.get("configuration")
    configuration = configuration if isinstance(configuration, dict) else {}
    experiment_mode = _optional_str(configuration.get("experiment_mode"))
    enabled = configuration.get("enabled")

    warnings: list[DashboardWarning] = []
    if experiment_mode == "static":
        warnings.append(
            DashboardWarning(
                level=DashboardWarningLevel.INFORMATION,
                message=STATIC_MODE_NOTE,
                subject=run.directory_name,
            )
        )

    lifecycle, trace_warning, trace = _lifecycle(run, metrics)
    if trace_warning is not None:
        warnings.append(trace_warning)

    return AdaptationDashboardData(
        provenance=provenance,
        experiment_mode=experiment_mode,
        policy_mode=_optional_str(configuration.get("mode")),
        adaptation_enabled=bool(enabled) if isinstance(enabled, bool) else None,
        configuration_fingerprint=_optional_str(
            document.get("configuration_fingerprint")
        ),
        evaluated_windows=_optional_count(metrics.get("evaluated_windows")),
        gate_eligible_windows=_optional_count(metrics.get("gate_eligible_windows")),
        gate_blocked_windows=_optional_count(metrics.get("gate_blocked_windows")),
        hold_decisions=_optional_count(metrics.get("hold_decisions")),
        increases=_optional_count(metrics.get("increases")),
        decreases=_optional_count(metrics.get("decreases")),
        lifecycle=lifecycle,
        hold_reason_table=_hold_reason_table(metrics),
        guard_table=_guard_table(configuration),
        spacing_table=_spacing_table(metrics),
        scenario_table=_scenario_table(run, max_rows=max_rows),
        session_table=_session_table(metrics, max_rows=max_rows),
        difficulty_trace=_difficulty_chart(run, trace, provenance.is_synthetic),
        lifecycle_table=_lifecycle_table(lifecycle),
        action_frequency_comparison_table=_comparison_table(document),
        session_ids=tuple(
            str(session)
            for session in document.get("session_ids") or ()
            if session is not None
        ),
        warnings=tuple(warnings),
    )


def _lifecycle(
    run: DashboardRunSummary, metrics: Mapping[str, Any]
) -> tuple[
    AdaptationLifecycleCounts | None,
    DashboardWarning | None,
    dict[str, list[Any]] | None,
]:
    """Count the four lifecycle events, keeping them separate.

    The proposal count comes from the summary; the command, dispatch,
    and acknowledgement counts come from the trace, whose
    ``lifecycle_status`` column is the only record of how far each
    proposal actually got.
    """
    proposals = _optional_count(metrics.get("adaptation_proposals"))
    if proposals is None:
        return None, None, None
    try:
        trace = read_parquet(run, "adaptation_trace.parquet", TRACE_COLUMNS)
    except ArtifactReadError as exc:
        return (
            None,
            unreadable_artifact_warning(run, "adaptation_trace.parquet", str(exc)),
            None,
        )
    # A HOLD records no lifecycle status at all, so a null here means the
    # window produced no proposal rather than a proposal that stalled.
    statuses = [str(value) for value in trace["lifecycle_status"] if value is not None]
    built = sum(1 for value in trace["command_built"] if value is True)
    dispatched = sum(1 for value in statuses if value in _DISPATCHED_ONWARDS)
    acknowledged = sum(1 for value in statuses if value in _ACKNOWLEDGED_ONWARDS)
    applied = sum(1 for value in statuses if value == "applied")
    return (
        AdaptationLifecycleCounts(
            proposals=proposals,
            commands_built=built,
            commands_dispatched=dispatched,
            acknowledgements_recorded=acknowledged,
            applied_confirmed=applied,
        ),
        None,
        trace,
    )


def _lifecycle_table(lifecycle: AdaptationLifecycleCounts | None) -> Any:
    if lifecycle is None:
        return None
    return fmt.build_table(
        title="Adaptation lifecycle",
        columns=("lifecycle stage", "count", "what this stage means"),
        rows=[
            (
                "Proposal",
                str(lifecycle.proposals),
                "The policy decided a change would be appropriate under its "
                "rules. Nothing has been produced for transmission.",
            ),
            (
                "Command built",
                str(lifecycle.commands_built),
                "A set_difficulty payload was constructed. It exists in "
                "memory; it has not been sent anywhere.",
            ),
            (
                "Dispatched",
                str(lifecycle.commands_dispatched),
                "The command was transmitted to a running environment. "
                "Milestone 8 sends nothing, so this is zero.",
            ),
            (
                "Acknowledged",
                str(lifecycle.acknowledgements_recorded),
                "The environment confirmed receipt. This requires a real "
                "reply and cannot be reached without a dispatch.",
            ),
            (
                "Applied",
                str(lifecycle.applied_confirmed),
                "The environment confirmed the change took effect. No "
                "adaptation in this repository has reached this state.",
            ),
        ],
        source_artifact="adaptation_trace.parquet",
        caption=(
            "These five counts are never added together and never collapsed "
            "into one 'adaptations' number. A proposal is not a dispatched "
            "adaptation and a dispatched adaptation is not an applied one."
        ),
    )


def _hold_reason_table(metrics: Mapping[str, Any]) -> Any:
    counts = metrics.get("hold_reason_counts")
    if not isinstance(counts, dict) or not counts:
        return None
    clean = {str(k): int(v) for k, v in counts.items() if isinstance(v, int)}
    if not clean:
        return None
    return fmt.counts_table(
        title="Why the controller held",
        counts=clean,
        key_column="hold reason",
        source_artifact="adaptation_summary.json",
        caption=(
            "Every evaluated window produced exactly one decision. A HOLD is "
            "a first-class decision with a recorded reason, not a failure to "
            "decide. " + ADAPTATION_NOTE
        ),
    )


def _guard_table(configuration: Mapping[str, Any]) -> Any:
    difficulty = configuration.get("difficulty")
    difficulty = difficulty if isinstance(difficulty, dict) else {}
    if not configuration:
        return None
    return fmt.build_table(
        title="Configured guards",
        columns=("guard", "value", "what it does"),
        rows=[
            (
                "Minimum persistence (dwell)",
                fmt.text(configuration.get("minimum_persistence_windows")),
                "Consecutive windows that must agree on a direction before "
                "any proposal. A hold resets this count; it does not decay.",
            ),
            (
                "Cooldown",
                fmt.text(configuration.get("cooldown_windows")),
                "Windows that must pass after a proposal before another may "
                "be made. Counted in windows, and it ticks on blocked "
                "windows too.",
            ),
            (
                "Session adaptation budget",
                fmt.text(configuration.get("max_adaptations_per_session")),
                "The most proposals one session may produce, after which "
                "every window holds.",
            ),
            (
                "Difficulty bounds",
                f"{fmt.text(difficulty.get('minimum'))} to "
                f"{fmt.text(difficulty.get('maximum'))}",
                "The policy holds at a bound rather than proposing a level outside it.",
            ),
            (
                "Step size",
                fmt.text(difficulty.get("step")),
                "One fixed step per proposal. The step is never scaled by confidence.",
            ),
            (
                "Conflict resolution",
                fmt.text(configuration.get("conflict_resolution")),
                "What happens when the two targets suggest opposite "
                "directions. There is no prefer-increase option.",
            ),
            (
                "Regression mapping enabled",
                fmt.text(configuration.get("regression_mapping_enabled")),
                "Whether continuous targets are mapped to ordinal bands. No "
                "band boundary has been measured, so this is off.",
            ),
        ],
        source_artifact="adaptation_summary.json",
        caption=(
            "Every value here is an engineering default chosen to be "
            "conservative. None was derived from evidence, and none is "
            "psychologically validated, pedagogically optimal, therapeutic, "
            "or known to be safe."
        ),
    )


def _spacing_table(metrics: Mapping[str, Any]) -> Any:
    rows = []
    for key, label, kind in CONTROLLER_METRICS:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            rows.append((label, "Unavailable"))
            continue
        rows.append((label, fmt.format_value(fmt.metric(label, value, kind=kind))))
    return fmt.build_table(
        title="Controller behaviour",
        columns=("diagnostic", "value"),
        rows=rows,
        source_artifact="adaptation_summary.json",
        caption=ADAPTATION_NOTE,
    )


def _session_table(metrics: Mapping[str, Any], *, max_rows: int) -> Any:
    proposals = metrics.get("proposals_by_session")
    finals = metrics.get("final_difficulty_by_session")
    if not isinstance(proposals, dict) or not proposals:
        return None
    finals = finals if isinstance(finals, dict) else {}
    rows = [
        (
            str(session),
            str(count),
            fmt.text(finals.get(session)),
        )
        for session, count in sorted(proposals.items())
    ]
    return fmt.build_table(
        title="Per-session controller behaviour",
        columns=("session (synthetic scenario)", "proposals", "final difficulty"),
        rows=rows,
        source_artifact="adaptation_summary.json",
        max_rows=max_rows,
        caption=(
            "Sessions here are hand-written controller scenarios, not "
            "recordings of anybody. Proposal counts describe the software."
        ),
    )


def _scenario_table(run: DashboardRunSummary, *, max_rows: int) -> Any:
    document, _error = try_document(run, "scenarios.json")
    if document is None:
        return None
    scenarios = document.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        return None
    rows = []
    for entry in scenarios:
        if not isinstance(entry, dict):
            continue
        rows.append(
            (
                fmt.text(entry.get("name")),
                fmt.text(entry.get("session_id")),
                fmt.text(entry.get("window_count")),
                fmt.text(entry.get("description")),
                fmt.text(entry.get("expectation")),
            )
        )
    return fmt.build_table(
        title="Controller scenarios",
        columns=("scenario", "session", "windows", "description", "expectation"),
        rows=rows,
        source_artifact="scenarios.json",
        max_rows=max_rows,
        caption=str(
            document.get("disclaimer")
            or "Deterministic controller tests. Each window state was chosen "
            "to make one branch of the policy run."
        ),
    )


def _difficulty_chart(
    run: DashboardRunSummary,
    trace: Mapping[str, Sequence[Any]] | None,
    is_synthetic: bool,
) -> LabelledChart | None:
    """Difficulty against window order, one series per session.

    The plotted values are the ``current_difficulty`` the run recorded at
    each window.  Nothing is recomputed: the policy is not re-run to
    produce this picture.
    """
    title = "Difficulty state by window"
    subtitle = (
        "Synthetic controller scenario — software diagnostic only"
        if is_synthetic
        else "Recorded controller trace"
    )
    if trace is None:
        return LabelledChart(
            title=title,
            subtitle=subtitle,
            x_axis_label="window order within the session",
            y_axis_label="task difficulty level (recorded)",
            series=(),
            source_artifact="adaptation_trace.parquet",
            unavailable_reason=(
                "adaptation_trace.parquet is unavailable for this run, so the "
                "difficulty trace cannot be drawn."
            ),
        )
    sessions = sorted({str(s) for s in trace["session_id"] if s is not None})
    series: list[ChartSeries] = []
    for session in sessions:
        rows = filter_rows(trace, "session_id", session)
        pairs = [
            (float(order), float(level))
            for order, level in zip(
                rows["window_order"], rows["current_difficulty"], strict=True
            )
            if isinstance(order, int | float)
            and isinstance(level, int | float)
            and not isinstance(order, bool)
            and not isinstance(level, bool)
        ]
        pairs.sort()
        if pairs:
            series.append(
                ChartSeries(
                    name=session,
                    x_values=tuple(x for x, _ in pairs),
                    y_values=tuple(y for _, y in pairs),
                )
            )
    return LabelledChart(
        title=title,
        subtitle=subtitle,
        x_axis_label="window order within the session",
        y_axis_label="task difficulty level (recorded)",
        series=tuple(series),
        x_axis_note=(
            "Difficulty as the run recorded it at each window. A flat line "
            "means the controller held, which is the ordinary outcome. This "
            "is not a participant's response to anything."
        ),
        source_artifact="adaptation_trace.parquet",
        unavailable_reason=(
            None if series else "no window recorded a difficulty level"
        ),
    )


def _comparison_table(document: Mapping[str, Any]) -> Any:
    comparison = document.get("naive_comparison")
    if not isinstance(comparison, dict):
        return None
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        return None
    keys = (
        ("adaptation_proposals", "Proposals", MetricKind.COUNT),
        ("hold_decisions", "HOLD decisions", MetricKind.COUNT),
        ("direction_reversals", "Direction reversals", MetricKind.COUNT),
        (
            "minimum_proposal_spacing_windows",
            "Minimum proposal spacing (windows)",
            MetricKind.COUNT,
        ),
        (
            "eligible_window_adaptation_fraction",
            "Proposals per eligible window (fraction)",
            MetricKind.REAL,
        ),
    )
    rows = []
    for key, label, kind in keys:
        rows.append(
            (
                label,
                fmt.format_value(fmt.metric(label, metrics.get(key), kind=kind)),
                fmt.format_value(fmt.metric(label, comparison.get(key), kind=kind)),
            )
        )
    return fmt.build_table(
        title="Software-controller action-frequency comparison",
        columns=("diagnostic", "conservative policy", "guard-free controller"),
        rows=rows,
        source_artifact="adaptation_summary.json",
        caption=CONTROLLER_COMPARISON_NOTE,
    )


def hold_reason_detail(
    run: DashboardRunSummary, session_id: str, *, max_rows: int = DEFAULT_MAX_TABLE_ROWS
) -> Any:
    """Per-window decisions of one session, read from the trace."""
    try:
        trace = read_parquet(run, "adaptation_trace.parquet", TRACE_COLUMNS)
    except ArtifactReadError:
        return None
    rows_data = filter_rows(trace, "session_id", session_id)
    rows = []
    for index in range(len(rows_data["window_order"])):
        reasons = rows_data["policy_reasons"][index]
        rows.append(
            (
                fmt.text(rows_data["window_order"][index]),
                fmt.text(rows_data["decision_kind"][index]),
                fmt.text(rows_data["resolved_direction"][index]),
                fmt.text(
                    ", ".join(str(r) for r in reasons)
                    if isinstance(reasons, list)
                    else reasons
                ),
                fmt.text(rows_data["current_difficulty"][index]),
                fmt.text(rows_data["proposed_difficulty"][index]),
                fmt.text(rows_data["persistence_after"][index]),
                fmt.text(rows_data["cooldown_after"][index]),
                fmt.text(rows_data["command_built"][index]),
                fmt.text(rows_data["lifecycle_status"][index]),
            )
        )
    if not rows:
        return None
    return fmt.build_table(
        title=f"Per-window decisions: {session_id}",
        columns=(
            "window",
            "decision",
            "direction",
            "reasons",
            "difficulty before",
            "difficulty proposed",
            "persistence after",
            "cooldown after",
            "command built",
            "lifecycle status",
        ),
        rows=rows,
        source_artifact="adaptation_trace.parquet",
        max_rows=max_rows,
        caption=(
            "One row per evaluated window, exactly as the run recorded it. "
            "No decision is recomputed here."
        ),
    )


def _optional_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


__all__ = [
    "CONTROLLER_METRICS",
    "TRACE_COLUMNS",
    "hold_reason_detail",
    "load_adaptation",
]
