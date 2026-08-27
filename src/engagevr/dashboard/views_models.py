"""Classification and regression result views over ``metrics.json``.

These views serve every family that writes a ``MetricsDocument``:
baseline, fusion, personalization, and uncertainty runs all record their
per-model results in the same shape, so one reader covers all four.

Three rules are enforced by construction rather than by convention.

**No champion.**  Nothing here computes or displays a best model.  Rows
may be ordered by a metric, and when they are, the caption says
*descriptive sorting only*.  The current pipeline records no
scientifically justified champion status, so the dashboard has no field
in which to render one.

**No ground truth.**  A confusion matrix from a synthetic run is labelled
*observed synthetic label* on the row axis.  This repository has no
participant ground truth, and the axis caption is exactly where that
would silently be claimed.

**No fabricated numbers.**  A fold whose metric was not computable is
``None`` in the artifact and *Unavailable* on the page.  It never becomes
zero, and a model whose folds all failed shows a column of *Unavailable*
rather than a column that reads as the worst possible score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from engagevr.dashboard import formatting as fmt
from engagevr.dashboard.aggregation import histogram_series, paired_series, residuals
from engagevr.dashboard.loaders import (
    DEFAULT_MAX_TABLE_ROWS,
    ArtifactReadError,
    filter_rows,
    numeric_list,
    read_parquet,
    string_list,
    try_document,
    unreadable_artifact_warning,
)
from engagevr.dashboard.presentation import DESCRIPTIVE_SORTING_NOTE
from engagevr.schemas.dashboard import (
    ChartSeries,
    ClassificationDashboardData,
    ConfusionMatrixView,
    DashboardRunSummary,
    DashboardWarning,
    DashboardWarningLevel,
    LabelledChart,
    RegressionDashboardData,
)

#: Aggregate metric names a classification page shows, in display order.
CLASSIFICATION_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)

#: Aggregate metric names a regression page shows, in display order.
REGRESSION_METRICS: tuple[str, ...] = (
    "mean_absolute_error",
    "root_mean_squared_error",
    "median_absolute_error",
    "r_squared",
)

#: Calibration metric names, which are reported separately from accuracy.
CALIBRATION_METRICS: tuple[str, ...] = (
    "brier_score",
    "log_loss",
    "expected_calibration_error",
)

#: Row-axis wording when the labels came from a synthetic generator.
SYNTHETIC_ROW_AXIS = "observed synthetic label"
#: Row-axis wording when the labels have real recorded provenance.
RECORDED_ROW_AXIS = "observed recorded label"
#: Column axis, which is the same either way.
PREDICTION_AXIS = "predicted class"


def row_axis_label(is_synthetic: bool) -> str:
    """Confusion-matrix row-axis wording for this run's provenance."""
    return SYNTHETIC_ROW_AXIS if is_synthetic else RECORDED_ROW_AXIS


def _aggregate_map(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entries = result.get("aggregate")
    if not isinstance(entries, list):
        return {}
    return {
        str(entry.get("name")): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("name") is not None
    }


def _aggregate_cell(entry: Mapping[str, Any] | None, name: str) -> str:
    """Render one aggregate as ``mean +/- sd`` or ``Unavailable``."""
    if entry is None:
        return fmt.format_value(
            fmt.metric(name, None, unavailable_reason="not recorded")
        )
    mean = fmt.metric(name, entry.get("mean"))
    if not mean.available:
        return fmt.format_value(mean)
    deviation = fmt.metric(f"{name} sd", entry.get("standard_deviation"))
    if deviation.available:
        return f"{fmt.format_value(mean)} ± {fmt.format_value(deviation)}"
    return fmt.format_value(mean)


def _results(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = document.get("results")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def aggregate_table(
    document: Mapping[str, Any],
    metric_names: Sequence[str],
    *,
    source_artifact: str,
    sort_by: str | None = None,
) -> Any:
    """One row per model, one column per aggregate metric.

    ``sort_by`` orders rows by a metric descending.  That is a display
    convenience and the caption says so; it does not select, recommend,
    or promote anything.
    """
    results = _results(document)
    if not results:
        return None
    rows: list[tuple[str, ...]] = []
    order: list[tuple[float, str]] = []
    for index, result in enumerate(results):
        aggregates = _aggregate_map(result)
        name = str(result.get("model_name", f"model-{index}"))
        cells = [name, str(result.get("model_kind", ""))]
        cells.extend(
            _aggregate_cell(aggregates.get(metric), metric) for metric in metric_names
        )
        failed = result.get("failed_folds")
        cells.append(str(len(failed)) if isinstance(failed, dict) else "0")
        rows.append(tuple(cells))
        if sort_by is not None:
            entry = aggregates.get(sort_by)
            value = entry.get("mean") if isinstance(entry, dict) else None
            order.append(
                (
                    float(value)
                    if isinstance(value, int | float) and not isinstance(value, bool)
                    else float("-inf"),
                    name,
                )
            )
    caption = (
        "Unweighted mean over the folds in which the metric was defined, with "
        "the standard deviation across those folds."
    )
    if sort_by is not None:
        indices = sorted(range(len(rows)), key=lambda i: (-order[i][0], order[i][1]))
        rows = [rows[i] for i in indices]
        caption = f"Sorted by {sort_by}. {DESCRIPTIVE_SORTING_NOTE} {caption}"
    return fmt.build_table(
        title="Aggregate results by model",
        columns=("model", "kind", *metric_names, "failed folds"),
        rows=rows,
        source_artifact=source_artifact,
        caption=caption,
    )


def fold_table(
    document: Mapping[str, Any],
    metric_names: Sequence[str],
    *,
    task_type: str,
    source_artifact: str,
    max_rows: int,
) -> Any:
    """One row per model and fold."""
    key = (
        "fold_classification_metrics"
        if task_type == "classification"
        else "fold_regression_metrics"
    )
    rows: list[tuple[str, ...]] = []
    for result in _results(document):
        folds = result.get(key)
        if not isinstance(folds, list):
            continue
        name = str(result.get("model_name", ""))
        for index, fold in enumerate(folds):
            if not isinstance(fold, dict):
                continue
            cells = [
                name,
                str(index),
                fmt.text(fold.get("sample_count")),
                fmt.text(fold.get("independent_group_count")),
            ]
            cells.extend(
                fmt.format_value(fmt.metric(metric, fold.get(metric)))
                for metric in metric_names
            )
            rows.append(tuple(cells))
    if not rows:
        return None
    return fmt.build_table(
        title="Fold-level results",
        columns=(
            "model",
            "fold",
            "windows",
            "independent groups",
            *metric_names,
        ),
        rows=rows,
        source_artifact=source_artifact,
        max_rows=max_rows,
        caption=(
            "Every fold as the run recorded it. A metric whose prerequisites "
            "were unmet is Unavailable, never zero."
        ),
    )


def _per_class_table(
    document: Mapping[str, Any], *, source_artifact: str, max_rows: int
) -> Any:
    rows: list[tuple[str, ...]] = []
    for result in _results(document):
        folds = result.get("fold_classification_metrics")
        if not isinstance(folds, list):
            continue
        name = str(result.get("model_name", ""))
        for fold_index, fold in enumerate(folds):
            if not isinstance(fold, dict):
                continue
            per_class = fold.get("per_class")
            if not isinstance(per_class, list):
                continue
            for entry in per_class:
                if not isinstance(entry, dict):
                    continue
                rows.append(
                    (
                        name,
                        str(fold_index),
                        fmt.text(entry.get("label")),
                        fmt.text(entry.get("support")),
                        fmt.format_value(
                            fmt.metric("precision", entry.get("precision"))
                        ),
                        fmt.format_value(fmt.metric("recall", entry.get("recall"))),
                        fmt.format_value(fmt.metric("f1", entry.get("f1"))),
                    )
                )
    if not rows:
        return None
    return fmt.build_table(
        title="Per-class results",
        columns=(
            "model",
            "fold",
            "class",
            "support",
            "precision",
            "recall",
            "F1",
        ),
        rows=rows,
        source_artifact=source_artifact,
        max_rows=max_rows,
        caption="Support is the number of evaluated windows carrying that class.",
    )


def confusion_matrices(
    document: Mapping[str, Any], *, is_synthetic: bool, source_artifact: str
) -> tuple[ConfusionMatrixView, ...]:
    """Aggregate confusion matrices, one per model, from stored counts."""
    views: list[ConfusionMatrixView] = []
    for result in _results(document):
        matrix = result.get("aggregate_confusion_matrix")
        if not isinstance(matrix, dict):
            continue
        labels = matrix.get("labels")
        counts = matrix.get("counts")
        if not isinstance(labels, list) or not isinstance(counts, list):
            continue
        try:
            views.append(
                ConfusionMatrixView(
                    labels=tuple(str(label) for label in labels),
                    counts=tuple(tuple(int(cell) for cell in row) for row in counts),
                    row_axis_label=row_axis_label(is_synthetic),
                    column_axis_label=PREDICTION_AXIS,
                    source_artifact=(
                        f"{source_artifact} :: {result.get('model_name')}"
                    ),
                )
            )
        except (TypeError, ValueError):
            continue
    return tuple(views)


def _calibration_table(
    document: Mapping[str, Any], *, source_artifact: str, max_rows: int
) -> Any:
    rows: list[tuple[str, ...]] = []
    for result in _results(document):
        aggregates = _aggregate_map(result)
        name = str(result.get("model_name", ""))
        labels = sorted(
            {
                key.split(".", 1)[0]
                for key in aggregates
                if "." in key and key.split(".", 1)[1] in CALIBRATION_METRICS
            }
        )
        for label in labels:
            rows.append(
                (
                    name,
                    label,
                    *(
                        _aggregate_cell(aggregates.get(f"{label}.{metric}"), metric)
                        for metric in CALIBRATION_METRICS
                    ),
                )
            )
    if not rows:
        return None
    return fmt.build_table(
        title="Probability calibration",
        columns=("model", "probability set", *CALIBRATION_METRICS),
        rows=rows,
        source_artifact=source_artifact,
        max_rows=max_rows,
        caption=(
            "A calibrated probability is not certainty and is not signal "
            "quality. It states how often outcomes of this kind occurred at "
            "this predicted probability in the evaluated folds."
        ),
    )


def reliability_chart(
    run: DashboardRunSummary, document: Mapping[str, Any]
) -> LabelledChart | None:
    """Reliability diagram built from the bins the run already recorded.

    Descriptive only.  The bins, their edges, their mean confidence, and
    their empirical accuracy were all computed and stored by the run.
    Nothing here fits or refits a calibration model.
    """
    series: list[ChartSeries] = []
    for result in _results(document):
        folds = result.get("fold_classification_metrics")
        if not isinstance(folds, list):
            continue
        name = str(result.get("model_name", ""))
        confidences: list[float] = []
        accuracies: list[float | None] = []
        merged: dict[float, tuple[float, float, int]] = {}
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            for calibration in fold.get("calibration") or ():
                if not isinstance(calibration, dict):
                    continue
                if calibration.get("label") == "uncalibrated":
                    continue
                for entry in calibration.get("bins") or ():
                    if not isinstance(entry, dict):
                        continue
                    count = entry.get("count")
                    confidence = entry.get("mean_confidence")
                    accuracy = entry.get("empirical_accuracy")
                    if not isinstance(count, int) or count <= 0:
                        continue
                    if not isinstance(confidence, int | float):
                        continue
                    if not isinstance(accuracy, int | float):
                        continue
                    lower = entry.get("lower_edge")
                    if not isinstance(lower, int | float):
                        continue
                    key = round(float(lower), 6)
                    prior = merged.get(key, (0.0, 0.0, 0))
                    merged[key] = (
                        prior[0] + float(confidence) * count,
                        prior[1] + float(accuracy) * count,
                        prior[2] + count,
                    )
        for _edge, (confidence_sum, accuracy_sum, total) in sorted(merged.items()):
            confidences.append(confidence_sum / total)
            accuracies.append(accuracy_sum / total)
        if confidences:
            series.append(
                ChartSeries(
                    name=name,
                    x_values=tuple(confidences),
                    y_values=tuple(accuracies),
                )
            )
    if not series:
        return LabelledChart(
            title="Reliability diagram",
            x_axis_label="mean calibrated confidence in bin",
            y_axis_label="empirical accuracy in bin",
            series=(),
            source_artifact="metrics.json",
            unavailable_reason=(
                "this run recorded no populated calibration bins, so a "
                "reliability diagram cannot be drawn from stored values. It "
                "has not been reconstructed by recomputing calibration."
            ),
        )
    return LabelledChart(
        title="Reliability diagram",
        subtitle=f"Run {run.provenance.run_id}",
        x_axis_label="mean calibrated confidence in bin",
        y_axis_label="empirical accuracy in bin",
        series=tuple(series),
        x_axis_note=(
            "Equal-width bins of the maximum calibrated class probability, "
            "pooled across folds by window count. Descriptive only: no "
            "calibration model is fitted or refitted here."
        ),
        source_artifact="metrics.json",
    )


def load_classification(
    run: DashboardRunSummary,
    *,
    document_name: str = "metrics.json",
    max_rows: int = DEFAULT_MAX_TABLE_ROWS,
    sort_by: str | None = None,
) -> ClassificationDashboardData:
    """Build the classification view for one run."""
    provenance = run.provenance
    document, error = try_document(run, document_name)
    if document is None:
        return ClassificationDashboardData(
            provenance=provenance,
            warnings=(unreadable_artifact_warning(run, document_name, str(error)),),
            unavailable_reason=f"{document_name} is unavailable: {error}",
        )
    if document.get("task_type") != "classification":
        return ClassificationDashboardData(
            provenance=provenance,
            unavailable_reason=(
                f"this run's target is a {document.get('task_type')} target, "
                "so it has no classes, no class probabilities, and no "
                "confusion matrix."
            ),
        )

    labels = _class_labels(document)
    warnings: list[DashboardWarning] = []
    if not labels:
        warnings.append(
            DashboardWarning(
                level=DashboardWarningLevel.WARNING,
                message=(
                    "no class labels were recorded, so class-indexed views "
                    "are unavailable for this run"
                ),
                subject=run.directory_name,
            )
        )

    return ClassificationDashboardData(
        provenance=provenance,
        class_labels=labels,
        model_names=tuple(
            str(result.get("model_name", "")) for result in _results(document)
        ),
        aggregate_table=aggregate_table(
            document,
            CLASSIFICATION_METRICS,
            source_artifact=document_name,
            sort_by=sort_by,
        ),
        fold_table=fold_table(
            document,
            CLASSIFICATION_METRICS,
            task_type="classification",
            source_artifact=document_name,
            max_rows=max_rows,
        ),
        per_class_table=_per_class_table(
            document, source_artifact=document_name, max_rows=max_rows
        ),
        confusion_matrices=confusion_matrices(
            document,
            is_synthetic=provenance.is_synthetic,
            source_artifact=document_name,
        ),
        calibration_table=_calibration_table(
            document, source_artifact=document_name, max_rows=max_rows
        ),
        reliability_chart=reliability_chart(run, document),
        calibration_method=_calibration_method(run),
        warnings=tuple(warnings),
    )


def load_regression(
    run: DashboardRunSummary,
    *,
    document_name: str = "metrics.json",
    max_rows: int = DEFAULT_MAX_TABLE_ROWS,
    sort_by: str | None = None,
) -> RegressionDashboardData:
    """Build the regression view for one run."""
    provenance = run.provenance
    document, error = try_document(run, document_name)
    if document is None:
        return RegressionDashboardData(
            provenance=provenance,
            warnings=(unreadable_artifact_warning(run, document_name, str(error)),),
            unavailable_reason=f"{document_name} is unavailable: {error}",
        )
    if document.get("task_type") != "regression":
        return RegressionDashboardData(
            provenance=provenance,
            unavailable_reason=(
                f"this run's target is a {document.get('task_type')} target, "
                "so it has no continuous predictions to plot against observed "
                "values."
            ),
        )

    observed_label = "synthetic target" if provenance.is_synthetic else "observed value"
    scatter, residual_hist, residual_scatter, warnings = _regression_charts(
        run, observed_label
    )
    return RegressionDashboardData(
        provenance=provenance,
        model_names=tuple(
            str(result.get("model_name", "")) for result in _results(document)
        ),
        aggregate_table=aggregate_table(
            document,
            REGRESSION_METRICS,
            source_artifact=document_name,
            sort_by=sort_by,
        ),
        fold_table=fold_table(
            document,
            REGRESSION_METRICS,
            task_type="regression",
            source_artifact=document_name,
            max_rows=max_rows,
        ),
        observed_versus_predicted=scatter,
        residual_histogram=residual_hist,
        residual_versus_predicted=residual_scatter,
        observed_axis_label=observed_label,
        warnings=warnings,
    )


def _regression_charts(
    run: DashboardRunSummary, observed_label: str
) -> tuple[
    LabelledChart | None,
    LabelledChart | None,
    LabelledChart | None,
    tuple[DashboardWarning, ...],
]:
    """Scatter and residual charts from stored predictions only."""
    try:
        data = read_parquet(
            run,
            "predictions.parquet",
            ("model_name", "true_value", "predicted_value"),
        )
    except ArtifactReadError as exc:
        reason = (
            f"predictions.parquet is unavailable for this run, so observed-"
            f"versus-predicted and residual views cannot be drawn: {exc}"
        )
        empty = LabelledChart(
            title="Observed versus predicted",
            x_axis_label=f"{observed_label} (recorded)",
            y_axis_label="predicted value (recorded)",
            series=(),
            unavailable_reason=reason,
            source_artifact="predictions.parquet",
        )
        return (
            empty,
            empty.model_copy(update={"title": "Residual distribution"}),
            empty.model_copy(update={"title": "Residual versus predicted"}),
            (unreadable_artifact_warning(run, "predictions.parquet", str(exc)),),
        )

    models = sorted({str(name) for name in data["model_name"] if name is not None})
    scatter_series: list[ChartSeries] = []
    residual_scatter_series: list[ChartSeries] = []
    residual_values: list[float] = []
    dropped_total = 0
    for model in models:
        rows = filter_rows(data, "model_name", model)
        observed = numeric_list(rows["true_value"])
        predicted = numeric_list(rows["predicted_value"])
        series, dropped = paired_series(model, observed, predicted)
        dropped_total += dropped
        if series is not None:
            scatter_series.append(series)
        predicted_kept, residual_kept, _ = residuals(observed, predicted)
        if predicted_kept:
            residual_scatter_series.append(
                ChartSeries(
                    name=model,
                    x_values=predicted_kept,
                    y_values=tuple(residual_kept),
                )
            )
            residual_values.extend(residual_kept)

    dropped_note = (
        None
        if not dropped_total
        else (
            f"{dropped_total} row(s) had no recorded observed or predicted "
            "value and are excluded from these charts rather than plotted as "
            "zero."
        )
    )
    scatter = LabelledChart(
        title="Observed versus predicted",
        subtitle=f"Run {run.provenance.run_id}",
        x_axis_label=f"{observed_label} (recorded)",
        y_axis_label="predicted value (recorded)",
        series=tuple(scatter_series),
        x_axis_note=dropped_note,
        source_artifact="predictions.parquet",
        unavailable_reason=(
            None if scatter_series else "no paired observed and predicted values"
        ),
    )
    histogram = histogram_series("residual", residual_values, bins=30)
    residual_hist = LabelledChart(
        title="Residual distribution",
        subtitle=f"residual = {observed_label} - predicted",
        x_axis_label="residual (target units)",
        y_axis_label="number of windows",
        series=(histogram,) if histogram is not None else (),
        source_artifact="predictions.parquet",
        unavailable_reason=(
            None if histogram is not None else "no residual could be computed"
        ),
    )
    residual_scatter = LabelledChart(
        title="Residual versus predicted",
        x_axis_label="predicted value (recorded)",
        y_axis_label="residual (target units)",
        series=tuple(residual_scatter_series),
        x_axis_note=(
            "Residuals are differences between two values this run already "
            "recorded. No model is re-run and no statistical test is implied."
        ),
        source_artifact="predictions.parquet",
        unavailable_reason=(
            None if residual_scatter_series else "no residual could be computed"
        ),
    )
    return scatter, residual_hist, residual_scatter, ()


def _class_labels(document: Mapping[str, Any]) -> tuple[str, ...]:
    for result in _results(document):
        matrix = result.get("aggregate_confusion_matrix")
        if isinstance(matrix, dict) and isinstance(matrix.get("labels"), list):
            return tuple(str(label) for label in matrix["labels"])
        folds = result.get("fold_classification_metrics")
        if isinstance(folds, list):
            for fold in folds:
                support = fold.get("class_support") if isinstance(fold, dict) else None
                if isinstance(support, dict) and support:
                    return tuple(str(key) for key in support)
    return ()


def _calibration_method(run: DashboardRunSummary) -> str | None:
    document, _error = try_document(run, "manifest.json")
    if document is None:
        return None
    method = document.get("calibration_method")
    return None if method is None else str(method)


def predicted_class_distribution(
    run: DashboardRunSummary, model_name: str
) -> tuple[dict[str, int], str | None]:
    """Counts of each predicted class for one model, from stored rows."""
    try:
        data = read_parquet(
            run, "predictions.parquet", ("model_name", "predicted_value")
        )
    except ArtifactReadError as exc:
        return {}, str(exc)
    rows = filter_rows(data, "model_name", model_name)
    counts: dict[str, int] = {}
    for value in string_list(rows["predicted_value"]):
        key = "unavailable" if value is None else value
        counts[key] = counts.get(key, 0) + 1
    return counts, None


__all__ = [
    "CALIBRATION_METRICS",
    "CLASSIFICATION_METRICS",
    "PREDICTION_AXIS",
    "RECORDED_ROW_AXIS",
    "REGRESSION_METRICS",
    "SYNTHETIC_ROW_AXIS",
    "aggregate_table",
    "confusion_matrices",
    "fold_table",
    "load_classification",
    "load_regression",
    "predicted_class_distribution",
    "reliability_chart",
    "row_axis_label",
]
