"""Milestone 7 uncertainty and selective-prediction views.

This is the page with the strictest vocabulary in the dashboard, and the
strictness is structural: :class:`~engagevr.schemas.dashboard.UncertaintyDashboardData`
refuses to hold a calibrated-confidence field on a regression run and
refuses to hold an interval field on a classification run.  A page cannot
show the wrong control because the view model will not carry it.

The two coverage axes move in opposite directions and never share a grid.

* Classification sweeps ``confidence_threshold``.  Raising it is
  *stricter*, so coverage may only fall.
* Regression sweeps ``maximum_interval_width``.  Raising it is *more
  permissive*, so coverage may only rise.

Both axes are named on the chart with their units.  Neither is relabelled
"uncertainty threshold", and ``1 - interval_width`` is never computed:
an interval width is in the target's own units, is not confined to
``[0, 1]``, and is not convertible into a confidence score.

Finally, the three selective outcomes are reconciled rather than
normalised.  ``accepted + abstained + unavailable`` must equal the
evaluated window count.  When it does not, the page shows an artifact
validation error, because a mismatch means the artifact is wrong and
smoothing it over would hide that.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from engagevr.dashboard import formatting as fmt
from engagevr.dashboard.aggregation import curve_series, histogram_series, is_monotonic
from engagevr.dashboard.loaders import (
    DEFAULT_MAX_TABLE_ROWS,
    ArtifactReadError,
    numeric_list,
    read_parquet,
    try_document,
    unreadable_artifact_warning,
)
from engagevr.dashboard.presentation import SELECTIVE_PREDICTION_NOTE, term_caption
from engagevr.schemas.dashboard import (
    DashboardError,
    DashboardRunSummary,
    DashboardWarning,
    DashboardWarningLevel,
    LabelledChart,
    MetricKind,
    SelectiveAccounting,
    UncertaintyDashboardData,
)

#: Axis name recorded by a classification coverage curve.
CONFIDENCE_AXIS = "confidence_threshold"
#: Axis name recorded by a regression coverage curve.
INTERVAL_WIDTH_AXIS = "maximum_interval_width"

#: Units of each axis, restated on the chart so neither is mistaken.
AXIS_UNITS: dict[str, str] = {
    CONFIDENCE_AXIS: (
        "probability in [0, 1]. The axis value is compared against a "
        "calibrated class probability. Raising it is STRICTER, so coverage "
        "may only fall."
    ),
    INTERVAL_WIDTH_AXIS: (
        "the regression target's own units. The axis value is compared "
        "against a prediction-interval width. It is NOT a probability, is "
        "not confined to [0, 1], and is not convertible into a confidence "
        "score. Raising it is MORE PERMISSIVE, so coverage may only rise."
    ),
}

#: The monotonicity contract of each axis.
AXIS_RULE: dict[str, str] = {
    CONFIDENCE_AXIS: "coverage is non-increasing as the threshold rises",
    INTERVAL_WIDTH_AXIS: "coverage is non-decreasing as the width limit rises",
}


def load_uncertainty(
    run: DashboardRunSummary, *, max_rows: int = DEFAULT_MAX_TABLE_ROWS
) -> UncertaintyDashboardData:
    """Build the uncertainty view for one Milestone 7 run."""
    provenance = run.provenance
    document, error = try_document(run, "uncertainty.json")
    if document is None:
        return UncertaintyDashboardData(
            provenance=provenance,
            task_type=provenance.task_type or "unknown",
            warnings=(
                unreadable_artifact_warning(run, "uncertainty.json", str(error)),
            ),
            unavailable_reason=f"uncertainty.json is unavailable: {error}",
        )

    task_type = str(document.get("task_type", provenance.task_type or "unknown"))
    warnings: list[DashboardWarning] = []
    accounting, accounting_warning = _accounting(document, run)
    if accounting_warning is not None:
        warnings.append(accounting_warning)

    reason_table = _reason_table(document)
    threshold_table, threshold_warning = _threshold_table(run, max_rows=max_rows)
    if threshold_warning is not None:
        warnings.append(threshold_warning)

    curve, curve_axis, curve_warning = _coverage_curve(run, task_type)
    if curve_warning is not None:
        warnings.append(curve_warning)
    risk_curve = _risk_coverage_curve(run, task_type)

    if task_type == "classification":
        return _classification_view(
            run,
            document,
            accounting=accounting,
            reason_table=reason_table,
            threshold_table=threshold_table,
            curve=curve,
            curve_axis=curve_axis,
            risk_curve=risk_curve,
            warnings=warnings,
            max_rows=max_rows,
        )
    if task_type == "regression":
        return _regression_view(
            run,
            document,
            accounting=accounting,
            reason_table=reason_table,
            threshold_table=threshold_table,
            curve=curve,
            curve_axis=curve_axis,
            risk_curve=risk_curve,
            warnings=warnings,
            max_rows=max_rows,
        )
    return UncertaintyDashboardData(
        provenance=provenance,
        task_type=task_type,
        accounting=accounting,
        abstention_reason_table=reason_table,
        threshold_table=threshold_table,
        warnings=tuple(warnings),
        unavailable_reason=(
            f"task type {task_type!r} is neither classification nor "
            "regression, so this dashboard cannot choose a selective view "
            "for it. The axis is not guessed."
        ),
    )


def _classification_view(
    run: DashboardRunSummary,
    document: Mapping[str, Any],
    *,
    accounting: SelectiveAccounting | None,
    reason_table: Any,
    threshold_table: Any,
    curve: LabelledChart | None,
    curve_axis: str | None,
    risk_curve: LabelledChart | None,
    warnings: list[DashboardWarning],
    max_rows: int,
) -> UncertaintyDashboardData:
    """Build the classification half. No interval field is ever set."""
    columns, column_warning = _selective_columns(run)
    if column_warning is not None:
        warnings.append(column_warning)

    confidence_chart = _histogram_chart(
        columns.get("confidence_score"),
        title="Calibrated classification confidence",
        x_label="calibrated confidence (probability in [0, 1])",
        note=term_caption("calibrated_confidence"),
        lower=0.0,
        upper=1.0,
    )
    entropy_chart = _histogram_chart(
        columns.get("entropy"),
        title="Predictive entropy",
        x_label="entropy (nats)",
        note=term_caption("predictive_entropy"),
        lower=0.0,
    )
    margin_chart = _histogram_chart(
        columns.get("margin"),
        title="Probability margin (top class minus second)",
        x_label="margin (probability difference in [0, 1])",
        note=term_caption("probability_margin"),
        lower=0.0,
        upper=1.0,
    )

    if curve_axis is not None and curve_axis != CONFIDENCE_AXIS:
        warnings.append(
            DashboardWarning(
                level=DashboardWarningLevel.ERROR,
                message=(
                    f"this classification run recorded a coverage curve swept "
                    f"over {curve_axis!r}, but a classification target is "
                    f"selective on {CONFIDENCE_AXIS!r}. The curve is not shown "
                    "under the wrong axis name."
                ),
                subject=run.directory_name,
            )
        )
        curve = None

    return UncertaintyDashboardData(
        provenance=run.provenance,
        task_type="classification",
        accounting=accounting,
        abstention_reason_table=reason_table,
        threshold_table=threshold_table,
        probability_calibration_status=_calibration_status(document),
        calibrated_confidence_histogram=confidence_chart,
        predictive_entropy_histogram=entropy_chart,
        probability_margin_histogram=margin_chart,
        confidence_coverage_curve=curve,
        risk_coverage_curve=risk_curve,
        coverage_axis=CONFIDENCE_AXIS,
        coverage_axis_units=AXIS_UNITS[CONFIDENCE_AXIS],
        coverage_monotonicity_rule=AXIS_RULE[CONFIDENCE_AXIS],
        warnings=tuple(warnings),
    )


def _regression_view(
    run: DashboardRunSummary,
    document: Mapping[str, Any],
    *,
    accounting: SelectiveAccounting | None,
    reason_table: Any,
    threshold_table: Any,
    curve: LabelledChart | None,
    curve_axis: str | None,
    risk_curve: LabelledChart | None,
    warnings: list[DashboardWarning],
    max_rows: int,
) -> UncertaintyDashboardData:
    """Build the regression half. No confidence field is ever set."""
    columns, column_warning = _selective_columns(run)
    if column_warning is not None:
        warnings.append(column_warning)

    width_chart = _histogram_chart(
        columns.get("interval_width"),
        title="Regression prediction-interval width",
        x_label="interval width (regression target units)",
        note=term_caption("interval_width"),
        lower=0.0,
    )
    interval_table = _interval_table(columns, max_rows=max_rows)

    if curve_axis is not None and curve_axis != INTERVAL_WIDTH_AXIS:
        warnings.append(
            DashboardWarning(
                level=DashboardWarningLevel.ERROR,
                message=(
                    f"this regression run recorded a coverage curve swept over "
                    f"{curve_axis!r}, but a regression target is selective on "
                    f"{INTERVAL_WIDTH_AXIS!r}. A regression target has no class "
                    "probability to threshold, so the curve is not shown."
                ),
                subject=run.directory_name,
            )
        )
        curve = None

    configuration = document.get("configuration")
    maximum_width = None
    if isinstance(configuration, dict):
        maximum_width = configuration.get("maximum_interval_width")

    return UncertaintyDashboardData(
        provenance=run.provenance,
        task_type="regression",
        accounting=accounting,
        abstention_reason_table=reason_table,
        threshold_table=threshold_table,
        interval_width_histogram=width_chart,
        width_coverage_curve=curve,
        risk_coverage_curve=risk_curve,
        interval_table=interval_table,
        empirical_interval_coverage=fmt.metric(
            "Empirical interval coverage",
            _empirical_interval_coverage(document),
            kind=MetricKind.PROBABILITY,
            source_artifact="uncertainty.json",
            unavailable_reason=(
                "no fold recorded an empirical interval coverage; the "
                "conformal calibration may have been unavailable"
            ),
        ),
        configured_maximum_interval_width=fmt.metric(
            "Configured maximum interval width",
            maximum_width,
            kind=MetricKind.INTERVAL_WIDTH,
            units="target units",
            source_artifact="uncertainty.json",
            unavailable_reason=(
                "no maximum interval width was configured, so no window was "
                "abstained on width grounds"
            ),
        ),
        coverage_axis=INTERVAL_WIDTH_AXIS,
        coverage_axis_units=AXIS_UNITS[INTERVAL_WIDTH_AXIS],
        coverage_monotonicity_rule=AXIS_RULE[INTERVAL_WIDTH_AXIS],
        warnings=tuple(warnings),
    )


def _accounting(
    document: Mapping[str, Any], run: DashboardRunSummary
) -> tuple[SelectiveAccounting | None, DashboardWarning | None]:
    """Accepted, abstained, unavailable, reconciled against the total."""
    total = document.get("total_window_count")
    accepted = document.get("accepted_count")
    abstained = document.get("abstained_count")
    unavailable = document.get("unavailable_count")
    values = (total, accepted, abstained, unavailable)
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return None, DashboardWarning(
            level=DashboardWarningLevel.ERROR,
            message=(
                "this run does not record all four of total, accepted, "
                "abstained, and unavailable window counts, so selective "
                "accounting cannot be shown. No count has been inferred."
            ),
            subject=run.directory_name,
        )
    assert isinstance(total, int)
    assert isinstance(accepted, int)
    assert isinstance(abstained, int)
    assert isinstance(unavailable, int)
    if accepted + abstained + unavailable == total:
        return (
            SelectiveAccounting(
                evaluated_window_count=total,
                accepted_count=accepted,
                abstained_count=abstained,
                unavailable_count=unavailable,
            ),
            None,
        )
    message = (
        f"ARTIFACT VALIDATION ERROR: accepted ({accepted}) + abstained "
        f"({abstained}) + unavailable ({unavailable}) = "
        f"{accepted + abstained + unavailable}, but this run recorded {total} "
        "evaluated windows. The three outcomes must partition the evaluated "
        "windows exactly. The mismatch is shown rather than normalised away."
    )
    return (
        SelectiveAccounting(
            evaluated_window_count=total,
            accepted_count=accepted,
            abstained_count=abstained,
            unavailable_count=unavailable,
            reconciles=False,
            reconciliation_error=message,
        ),
        DashboardWarning(
            level=DashboardWarningLevel.ERROR,
            message=message,
            subject=run.directory_name,
        ),
    )


def _reason_table(document: Mapping[str, Any]) -> Any:
    counts = document.get("abstention_reason_counts")
    if not isinstance(counts, dict) or not counts:
        return None
    clean = {str(k): int(v) for k, v in counts.items() if isinstance(v, int)}
    if not clean:
        return None
    return fmt.counts_table(
        title="Abstention reasons",
        counts=clean,
        key_column="reason",
        source_artifact="uncertainty.json",
        caption=(
            "Each reason is a distinct decision. An abstention is not an "
            "error and is never counted as one. " + SELECTIVE_PREDICTION_NOTE
        ),
    )


def _threshold_table(
    run: DashboardRunSummary, *, max_rows: int
) -> tuple[Any, DashboardWarning | None]:
    document, error = try_document(run, "thresholds.json")
    if document is None:
        return None, unreadable_artifact_warning(run, "thresholds.json", str(error))
    rows = [
        (
            "Population confidence threshold",
            fmt.text(document.get("population_confidence_threshold")),
        ),
        ("Provenance", fmt.text(document.get("population_threshold_provenance"))),
        ("Acceptance rule", fmt.text(document.get("acceptance_rule"))),
        ("Threshold estimation enabled", fmt.text(document.get("estimation_enabled"))),
        (
            "Personalized thresholds enabled",
            fmt.text(document.get("personalized_thresholds_enabled")),
        ),
        ("Personalized rule", fmt.text(document.get("personalized_threshold_rule"))),
    ]
    return (
        fmt.build_table(
            title="Applied thresholds",
            columns=("field", "value"),
            rows=rows,
            source_artifact="thresholds.json",
            max_rows=max_rows,
            caption=(
                "A threshold in this repository is an engineering default or "
                "is estimated from inner calibration groups only. None is read "
                "off an outer-test coverage curve."
            ),
        ),
        None,
    )


def _coverage_curve(
    run: DashboardRunSummary, task_type: str
) -> tuple[LabelledChart | None, str | None, DashboardWarning | None]:
    """The recorded coverage curve, drawn under its own axis name."""
    document, error = try_document(run, "coverage_curve.json")
    if document is None:
        return (
            None,
            None,
            unreadable_artifact_warning(run, "coverage_curve.json", str(error)),
        )
    curve = document.get("curve")
    if not isinstance(curve, dict):
        return None, None, None

    axis = curve.get("axis")
    if not isinstance(axis, str):
        return (
            None,
            None,
            DashboardWarning(
                level=DashboardWarningLevel.ERROR,
                message=(
                    "this run's coverage_curve.json records no 'axis' field. "
                    "It was written by a version of the pipeline that shared "
                    "one grid between the two coverage axes. The curve is not "
                    "displayed, because which axis it was swept over cannot "
                    "be established and must not be guessed."
                ),
                subject=run.directory_name,
            ),
        )

    reason = curve.get("points_unavailable_reason")
    points = curve.get("points")
    if not isinstance(points, list) or not points:
        return (
            LabelledChart(
                title=_curve_title(axis),
                x_axis_label=_axis_label(axis),
                y_axis_label="selective coverage (accepted / evaluated)",
                series=(),
                x_axis_note=AXIS_UNITS.get(axis),
                source_artifact="coverage_curve.json",
                unavailable_reason=str(
                    reason
                    or "this run recorded no coverage-curve points. The curve "
                    "is unavailable; it has not been fabricated."
                ),
            ),
            axis,
            None,
        )

    flattened = []
    for point in points:
        if not isinstance(point, dict):
            continue
        coverage_point = point.get("coverage_point")
        coverage = (
            coverage_point.get("coverage") if isinstance(coverage_point, dict) else None
        )
        flattened.append({"threshold": point.get("threshold"), "coverage": coverage})
    series = curve_series(
        "selective coverage", flattened, x_key="threshold", y_key="coverage"
    )
    recorded_monotonic = curve.get("coverage_is_monotonic")
    observed = (
        is_monotonic(series.y_values, non_increasing=axis == CONFIDENCE_AXIS)
        if series is not None
        else None
    )
    note = AXIS_UNITS.get(axis, "")
    if observed is not None:
        note = (
            f"{note} {AXIS_RULE.get(axis, '')} "
            f"Recorded as monotonic: {recorded_monotonic}; observed in the "
            f"plotted points: {observed}."
        )
    return (
        LabelledChart(
            title=_curve_title(axis),
            subtitle=f"Run {run.provenance.run_id}",
            x_axis_label=_axis_label(axis),
            y_axis_label="selective coverage (accepted / evaluated)",
            series=(series,) if series is not None else (),
            x_axis_note=note.strip(),
            source_artifact="coverage_curve.json",
            unavailable_reason=(
                None if series is not None else "no coverage value was recorded"
            ),
        ),
        axis,
        None,
    )


def _risk_coverage_curve(
    run: DashboardRunSummary, task_type: str
) -> LabelledChart | None:
    document, _error = try_document(run, "coverage_curve.json")
    if document is None:
        return None
    curve = document.get("curve")
    if not isinstance(curve, dict):
        return None
    points = curve.get("risk_coverage")
    if not isinstance(points, list) or not points:
        return None
    series = curve_series(
        "empirical risk", points, x_key="coverage", y_key="empirical_risk"
    )
    if series is None:
        reasons = {
            str(p.get("unavailable_reason"))
            for p in points
            if isinstance(p, dict) and p.get("unavailable_reason")
        }
        return LabelledChart(
            title="Empirical risk against coverage",
            x_axis_label="selective coverage (accepted / evaluated)",
            y_axis_label="empirical risk over accepted windows",
            series=(),
            source_artifact="coverage_curve.json",
            unavailable_reason=(
                "; ".join(sorted(reasons))
                if reasons
                else "no empirical risk was recorded at any coverage point"
            ),
        )
    return LabelledChart(
        title="Empirical risk against coverage",
        subtitle=f"Run {run.provenance.run_id}",
        x_axis_label="selective coverage (accepted / evaluated)",
        y_axis_label="empirical risk over accepted windows",
        series=(series,),
        x_axis_note=(
            "Empirical risk is one minus accepted-set accuracy, over accepted "
            "windows only. It is an empirical software quantity: not a bound, "
            "not a guarantee, and not a statement about any person."
        ),
        source_artifact="coverage_curve.json",
    )


#: Quantities read from ``selective_predictions.parquet``: the record of
#: what the selective layer decided, one row per evaluated window.
_SELECTIVE_COLUMNS: tuple[str, ...] = (
    "confidence_score",
    "interval_width",
    "interval_lower_bound",
    "interval_upper_bound",
)

#: Quantities read from ``predictions.parquet``: the *unselected* record
#: of what the model said before any threshold was applied. Entropy and
#: margin live there, and are read from there rather than being
#: recomputed from a probability vector.
_PREDICTION_COLUMNS: tuple[str, ...] = (
    "entropy",
    "normalized_entropy",
    "margin",
)


def _selective_columns(
    run: DashboardRunSummary,
) -> tuple[dict[str, list[float | None]], DashboardWarning | None]:
    """Numeric columns for the distribution charts, from two artifacts.

    Only the columns a chart displays are read.  A column an artifact
    does not carry is simply absent from the result, and the chart that
    wanted it renders as unavailable with that stated rather than being
    reconstructed from a different column.
    """
    columns: dict[str, list[float | None]] = {}
    warning: DashboardWarning | None = None
    try:
        selective = read_parquet(
            run,
            "selective_predictions.parquet",
            ("window_id",),
            optional_columns=_SELECTIVE_COLUMNS,
        )
    except ArtifactReadError as exc:
        warning = unreadable_artifact_warning(
            run, "selective_predictions.parquet", str(exc)
        )
    else:
        columns.update(
            {
                name: numeric_list(values)
                for name, values in selective.items()
                if name in _SELECTIVE_COLUMNS
            }
        )
    try:
        predictions = read_parquet(
            run,
            "predictions.parquet",
            ("window_id",),
            optional_columns=_PREDICTION_COLUMNS,
        )
    except ArtifactReadError:
        # The unselected table is optional for this page: without it the
        # entropy and margin charts say so and nothing else changes.
        return columns, warning
    columns.update(
        {
            name: numeric_list(values)
            for name, values in predictions.items()
            if name in _PREDICTION_COLUMNS
        }
    )
    return columns, warning


def _histogram_chart(
    values: Sequence[float | None] | None,
    *,
    title: str,
    x_label: str,
    note: str,
    lower: float | None = None,
    upper: float | None = None,
) -> LabelledChart:
    """A distribution chart, or an explicit unavailable one."""
    if values is None:
        return LabelledChart(
            title=title,
            x_axis_label=x_label,
            y_axis_label="number of windows",
            series=(),
            x_axis_note=note,
            source_artifact="selective_predictions.parquet",
            unavailable_reason=(
                "this run's selective_predictions.parquet carries no column "
                "for this quantity, so it is unavailable. It has not been "
                "derived from another column."
            ),
        )
    finite = [value for value in values if value is not None]
    series = histogram_series("windows", finite, lower=lower, upper=upper)
    return LabelledChart(
        title=title,
        x_axis_label=x_label,
        y_axis_label="number of windows",
        series=(series,) if series is not None else (),
        x_axis_note=note,
        source_artifact="selective_predictions.parquet",
        unavailable_reason=(
            None
            if series is not None
            else "every row recorded this quantity as unavailable"
        ),
    )


def _interval_table(
    columns: Mapping[str, Sequence[float | None]], *, max_rows: int
) -> Any:
    widths = columns.get("interval_width")
    if widths is None:
        return None
    present = [value for value in widths if value is not None]
    if not present:
        return None
    from engagevr.dashboard.aggregation import mean, median

    return fmt.build_table(
        title="Prediction-interval summary",
        columns=("statistic", "value (regression target units)"),
        rows=[
            (
                "Windows with a recorded interval",
                fmt.format_value(
                    fmt.count("windows", len(present), source_artifact="parquet")
                ),
            ),
            (
                "Windows with no recorded interval",
                fmt.format_value(
                    fmt.count(
                        "windows", len(widths) - len(present), source_artifact="parquet"
                    )
                ),
            ),
            (
                "Mean interval width",
                fmt.format_value(
                    fmt.metric(
                        "mean width",
                        mean(present),
                        kind=MetricKind.INTERVAL_WIDTH,
                        units="target units",
                    )
                ),
            ),
            (
                "Median interval width",
                fmt.format_value(
                    fmt.metric(
                        "median width",
                        median(present),
                        kind=MetricKind.INTERVAL_WIDTH,
                        units="target units",
                    )
                ),
            ),
            (
                "Narrowest recorded interval",
                fmt.format_value(
                    fmt.metric(
                        "min width",
                        min(present),
                        kind=MetricKind.INTERVAL_WIDTH,
                        units="target units",
                    )
                ),
            ),
            (
                "Widest recorded interval",
                fmt.format_value(
                    fmt.metric(
                        "max width",
                        max(present),
                        kind=MetricKind.INTERVAL_WIDTH,
                        units="target units",
                    )
                ),
            ),
        ],
        source_artifact="selective_predictions.parquet",
        max_rows=max_rows,
        caption=term_caption("interval_width"),
    )


def _calibration_status(document: Mapping[str, Any]) -> str | None:
    folds = document.get("folds")
    if not isinstance(folds, list):
        return None
    statuses = {
        str(fold.get("probability_calibration_status"))
        for fold in folds
        if isinstance(fold, dict) and fold.get("probability_calibration_status")
    }
    if not statuses:
        return None
    return ", ".join(sorted(statuses))


def _empirical_interval_coverage(document: Mapping[str, Any]) -> float | None:
    """Window-weighted mean of the per-fold recorded interval coverage."""
    folds = document.get("folds")
    if not isinstance(folds, list):
        return None
    weighted = 0.0
    total = 0
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        applied = fold.get("applied_selective_metrics")
        if not isinstance(applied, dict):
            continue
        coverage = applied.get("empirical_interval_coverage")
        point = applied.get("coverage_point")
        accepted = point.get("accepted_count") if isinstance(point, dict) else None
        if (
            isinstance(coverage, int | float)
            and not isinstance(coverage, bool)
            and isinstance(accepted, int)
            and not isinstance(accepted, bool)
            and accepted > 0
        ):
            weighted += float(coverage) * accepted
            total += accepted
    if total == 0:
        return None
    return weighted / total


def _curve_title(axis: str) -> str:
    if axis == CONFIDENCE_AXIS:
        return "Selective coverage against confidence threshold"
    if axis == INTERVAL_WIDTH_AXIS:
        return "Selective coverage against maximum interval width"
    raise DashboardError(f"unknown coverage axis {axis!r}")


def _axis_label(axis: str) -> str:
    if axis == CONFIDENCE_AXIS:
        return "confidence threshold (calibrated probability)"
    if axis == INTERVAL_WIDTH_AXIS:
        return "maximum interval width (regression target units)"
    raise DashboardError(f"unknown coverage axis {axis!r}")


__all__ = [
    "AXIS_RULE",
    "AXIS_UNITS",
    "CONFIDENCE_AXIS",
    "INTERVAL_WIDTH_AXIS",
    "load_uncertainty",
]
