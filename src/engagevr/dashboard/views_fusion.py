"""Milestone 6 fusion and personalization views.

Both pages exist mainly to keep two pairs of concepts apart.

**Expert disagreement is not calibrated uncertainty.**  How much the
per-modality experts differ is a property of an ensemble.  Whether a
probability means what it says is a property of a calibration.  They are
displayed in separate tables under separate names, and neither table's
heading uses the other's word.

**A fusion support weight is not a probability of correctness.**  A
quality-aware weight says how much a modality contributed to a fused
prediction, given how well that modality could be measured.  It is not a
confidence, and calling it one would turn a measurement statement into a
model-performance statement.

The personalization page has a third rule.  Population and personalized
results are reported side by side over identical evaluation windows, and
the difference between them is labelled a *difference*.  On synthetic
data the personalized result is frequently worse, and that is displayed
exactly as recorded: hiding a negative difference would be the same act
as claiming a positive one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from engagevr.dashboard import formatting as fmt
from engagevr.dashboard.loaders import (
    DEFAULT_MAX_TABLE_ROWS,
    try_document,
    unreadable_artifact_warning,
)
from engagevr.dashboard.presentation import (
    DESCRIPTIVE_SORTING_NOTE,
    PERSONALIZATION_NOTE,
    term_caption,
)
from engagevr.schemas.dashboard import (
    ChartSeries,
    DashboardRunSummary,
    DashboardWarning,
    DashboardWarningLevel,
    FusionDashboardData,
    LabelledChart,
    PersonalizationDashboardData,
)

#: Prefix of every fusion diagnostic recorded in ``fusion_metrics.json``.
FUSION_PREFIX = "fusion."
#: Prefix of the disagreement diagnostics inside that namespace.
DISAGREEMENT_PREFIX = "fusion.disagreement."
#: Prefix of the per-modality support weights.
WEIGHT_PREFIX = "fusion.mean_normalized_weight."
#: Prefix of the per-modality missing rates.
MISSING_PREFIX = "fusion.missing_modality_rate."


def load_fusion(
    run: DashboardRunSummary, *, max_rows: int = DEFAULT_MAX_TABLE_ROWS
) -> FusionDashboardData:
    """Build the fusion view for one Milestone 6 fusion run."""
    provenance = run.provenance
    document, error = try_document(run, "fusion_metrics.json")
    if document is None:
        return FusionDashboardData(
            provenance=provenance,
            warnings=(
                unreadable_artifact_warning(run, "fusion_metrics.json", str(error)),
            ),
            unavailable_reason=f"fusion_metrics.json is unavailable: {error}",
        )

    strategies = document.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        return FusionDashboardData(
            provenance=provenance,
            unavailable_reason=(
                "fusion_metrics.json records no fusion strategy, so there is "
                "nothing to display for this run."
            ),
        )

    modalities = tuple(
        str(m) for m in document.get("modalities") or () if m is not None
    )
    task_type = str(document.get("task_type", ""))
    metric_names = (
        ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        if task_type == "classification"
        else ("mean_absolute_error", "root_mean_squared_error", "r_squared")
    )

    warnings: list[DashboardWarning] = []
    strategy_rows = []
    weight_rows = []
    disagreement_rows = []
    availability_rows = []
    for entry in strategies:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("strategy", ""))
        aggregates = _named(entry.get("aggregate"))
        fusion_aggregates = _named(entry.get("fusion_aggregate"))
        strategy_rows.append(
            (
                name,
                fmt.text(entry.get("expert_model_name")),
                fmt.text(entry.get("calibrated_experts")),
                fmt.text(entry.get("valid_fold_count")),
                fmt.text(entry.get("total_fold_count")),
                *(_mean_cell(aggregates.get(metric)) for metric in metric_names),
                _mean_cell(fusion_aggregates.get("fusion.coverage")),
            )
        )
        for modality in modalities:
            weight = fusion_aggregates.get(f"{WEIGHT_PREFIX}{modality}")
            missing = fusion_aggregates.get(f"{MISSING_PREFIX}{modality}")
            if weight is not None or missing is not None:
                weight_rows.append((name, modality, _mean_cell(weight)))
                availability_rows.append((name, modality, _mean_cell(missing)))
        for key, value in sorted(fusion_aggregates.items()):
            if key.startswith(DISAGREEMENT_PREFIX):
                disagreement_rows.append(
                    (name, key.removeprefix(DISAGREEMENT_PREFIX), _mean_cell(value))
                )

    strategy_table = fmt.build_table(
        title="Fusion strategies",
        columns=(
            "strategy",
            "expert model",
            "calibrated experts",
            "valid folds",
            "total folds",
            *metric_names,
            "fusion coverage",
        ),
        rows=strategy_rows,
        source_artifact="fusion_metrics.json",
        caption=(
            "Strategies were compared on identical grouped outer folds. "
            + DESCRIPTIVE_SORTING_NOTE
        ),
    )

    weight_table = (
        fmt.build_table(
            title="Fusion support weights by modality",
            columns=("strategy", "modality", "mean normalized weight"),
            rows=weight_rows,
            source_artifact="fusion_metrics.json",
            max_rows=max_rows,
            caption=term_caption("fusion_support_weight"),
        )
        if weight_rows
        else None
    )
    disagreement_table = (
        fmt.build_table(
            title="Expert disagreement diagnostics",
            columns=("strategy", "diagnostic", "mean over folds"),
            rows=disagreement_rows,
            source_artifact="fusion_metrics.json",
            max_rows=max_rows,
            caption=term_caption("ensemble_disagreement"),
        )
        if disagreement_rows
        else None
    )
    availability_table = (
        fmt.build_table(
            title="Missing-modality rate",
            columns=("strategy", "modality", "mean missing rate"),
            rows=availability_rows,
            source_artifact="fusion_metrics.json",
            max_rows=max_rows,
            caption=(
                "The fraction of evaluated windows in which this modality "
                "contributed nothing. Modality availability is a measurement "
                "property. The absence of a modality is not the absence of "
                "engagement or cognitive load."
            ),
        )
        if availability_rows
        else None
    )

    expert_table, expert_warning = _expert_table(run, max_rows=max_rows)
    if expert_warning is not None:
        warnings.append(expert_warning)
    robustness_table, robustness_chart, robustness_warning = _robustness(
        run, metric_names[0], max_rows=max_rows
    )
    if robustness_warning is not None:
        warnings.append(robustness_warning)

    return FusionDashboardData(
        provenance=provenance,
        strategies=tuple(
            str(entry.get("strategy", ""))
            for entry in strategies
            if isinstance(entry, dict)
        ),
        modalities=modalities,
        strategy_table=strategy_table,
        expert_table=expert_table,
        fusion_support_weight_table=weight_table,
        expert_disagreement_table=disagreement_table,
        modality_availability_table=availability_table,
        robustness_table=robustness_table,
        robustness_chart=robustness_chart,
        warnings=tuple(warnings),
    )


def _expert_table(
    run: DashboardRunSummary, *, max_rows: int
) -> tuple[Any, DashboardWarning | None]:
    document, error = try_document(run, "experts.json")
    if document is None:
        return None, unreadable_artifact_warning(run, "experts.json", str(error))
    experts = document.get("experts")
    if not isinstance(experts, list) or not experts:
        return None, None
    rows = []
    for entry in experts:
        if not isinstance(entry, dict):
            continue
        rows.append(
            (
                fmt.text(entry.get("modality")),
                fmt.text(entry.get("fold_index")),
                fmt.text(entry.get("model_name")),
                fmt.text(entry.get("trained")),
                fmt.text(entry.get("calibrated")),
                fmt.text(entry.get("calibration_method")),
                fmt.text(entry.get("fit_row_count")),
                fmt.text(entry.get("fit_group_count")),
                fmt.text(entry.get("unavailable_reason")),
            )
        )
    return (
        fmt.build_table(
            title="Modality experts",
            columns=(
                "modality",
                "fold",
                "estimator",
                "trained",
                "calibrated",
                "calibration method",
                "fit windows",
                "fit groups",
                "unavailable reason",
            ),
            rows=rows,
            source_artifact="experts.json",
            max_rows=max_rows,
            caption=(
                "One estimator per modality. An expert that could not be "
                "trained records why, and contributes nothing rather than "
                "contributing a default."
            ),
        ),
        None,
    )


def _robustness(
    run: DashboardRunSummary, headline_metric: str, *, max_rows: int
) -> tuple[Any, LabelledChart | None, DashboardWarning | None]:
    document, error = try_document(run, "robustness.json")
    if document is None:
        return (
            None,
            None,
            unreadable_artifact_warning(run, "robustness.json", str(error)),
        )
    results = document.get("results")
    if not isinstance(results, list) or not results:
        return None, None, None

    rows = []
    by_strategy: dict[str, list[tuple[str, float | None]]] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        aggregates = _named(entry.get("aggregate"))
        scenario = str(entry.get("scenario_name", ""))
        strategy = str(entry.get("strategy", ""))
        rows.append(
            (
                scenario,
                strategy,
                _joined(entry.get("present_modalities")),
                _joined(entry.get("absent_modalities")),
                fmt.text(entry.get("evaluated")),
                fmt.text(entry.get("evaluated_window_count")),
                fmt.text(entry.get("fused_window_count")),
                fmt.text(entry.get("unavailable_fusion_count")),
                _mean_cell(aggregates.get("fusion.coverage")),
                _mean_cell(aggregates.get(headline_metric)),
                fmt.text(entry.get("unavailable_reason")),
            )
        )
        value = aggregates.get(headline_metric)
        mean = value.get("mean") if isinstance(value, dict) else None
        by_strategy.setdefault(strategy, []).append(
            (
                scenario,
                float(mean)
                if isinstance(mean, int | float) and not isinstance(mean, bool)
                else None,
            )
        )

    table = fmt.build_table(
        title="Missing-modality robustness scenarios",
        columns=(
            "scenario",
            "strategy",
            "present modalities",
            "absent modalities",
            "evaluated",
            "evaluated windows",
            "fused windows",
            "unavailable fusions",
            "fusion coverage",
            headline_metric,
            "unavailable reason",
        ),
        rows=rows,
        source_artifact="robustness.json",
        max_rows=max_rows,
        caption=(
            "Each scenario withholds modalities from the same recorded rows. "
            "Missing-modality behaviour is a property of this software under "
            "an artificial mask, not of any person or any real recording "
            "condition."
        ),
    )

    scenario_names = sorted(
        {scenario for values in by_strategy.values() for scenario, _ in values}
    )
    index_of = {name: float(i) for i, name in enumerate(scenario_names)}
    series = []
    for strategy, values in sorted(by_strategy.items()):
        ordered = sorted(values, key=lambda pair: index_of[pair[0]])
        series.append(
            ChartSeries(
                name=strategy,
                x_values=tuple(index_of[scenario] for scenario, _ in ordered),
                y_values=tuple(value for _, value in ordered),
            )
        )
    chart = LabelledChart(
        title=f"{headline_metric} by robustness scenario",
        subtitle="Software self-check under an artificial modality mask",
        x_axis_label="scenario index (see the table for names)",
        y_axis_label=headline_metric,
        series=tuple(s for s in series if any(v is not None for v in s.y_values)),
        x_axis_note=(
            "Scenario order matches the table above. A gap means the metric "
            "was not computable for that scenario, not that it was zero."
        ),
        source_artifact="robustness.json",
        unavailable_reason=(
            None
            if any(any(v is not None for v in s.y_values) for s in series)
            else f"{headline_metric} was not recorded for any scenario"
        ),
    )
    return table, chart, None


def load_personalization(
    run: DashboardRunSummary, *, max_rows: int = DEFAULT_MAX_TABLE_ROWS
) -> PersonalizationDashboardData:
    """Build the personalization view for one Milestone 6 run."""
    provenance = run.provenance
    document, error = try_document(run, "personalization.json")
    if document is None:
        return PersonalizationDashboardData(
            provenance=provenance,
            warnings=(
                unreadable_artifact_warning(run, "personalization.json", str(error)),
            ),
            unavailable_reason=f"personalization.json is unavailable: {error}",
        )

    population = _named(document.get("population_aggregate"))
    personalized = _named(document.get("personalized_aggregate"))
    metric_names = sorted(set(population) | set(personalized))

    paired_rows = []
    delta_rows = []
    for name in metric_names:
        left = population.get(name)
        right = personalized.get(name)
        paired_rows.append((name, _mean_cell(left), _mean_cell(right)))
        delta_rows.append((name, _delta_cell(left, right)))

    paired_table = (
        fmt.build_table(
            title="Population and personalized results",
            columns=("metric", "population", "personalized"),
            rows=paired_rows,
            source_artifact="personalization.json",
            caption=(
                "Both columns are computed over identical evaluation windows. "
                + PERSONALIZATION_NOTE
            ),
        )
        if paired_rows
        else None
    )
    delta_label = (
        "Δ metric on synthetic software-check data"
        if provenance.is_synthetic
        else "Δ metric (personalized - population)"
    )
    delta_table = (
        fmt.build_table(
            title=delta_label,
            columns=("metric", "personalized - population"),
            rows=delta_rows,
            source_artifact="personalization.json",
            caption=(
                "This column is a difference, not a benefit and not an "
                "improvement. A negative value is displayed exactly as "
                "recorded. " + PERSONALIZATION_NOTE
            ),
        )
        if delta_rows
        else None
    )

    coverage_table = fmt.build_table(
        title="Personalization coverage",
        columns=("field", "value"),
        rows=[
            (
                "Personalized subjects",
                fmt.text(document.get("personalized_subject_count")),
            ),
            ("Cold-start subjects", fmt.text(document.get("cold_start_subject_count"))),
            (
                "Subjects with no personalization available",
                fmt.text(document.get("unavailable_personalization_count")),
            ),
            (
                "Personalization coverage",
                fmt.format_value(
                    fmt.metric(
                        "coverage",
                        document.get("personalization_coverage"),
                        source_artifact="personalization.json",
                    )
                ),
            ),
            (
                "Calibration windows used",
                fmt.text(document.get("total_calibration_window_count")),
            ),
            (
                "Evaluation windows",
                fmt.text(document.get("total_evaluation_window_count")),
            ),
            (
                "Windows excluded for calibration/evaluation overlap",
                fmt.text(document.get("total_excluded_overlap_window_count")),
            ),
        ],
        source_artifact="personalization.json",
        caption=(
            "A cold-start subject fell back to the population model because "
            "too few of its own calibration windows were available. Falling "
            "back is the designed behaviour, not a failure."
        ),
    )

    fold_table, subject_table, warnings = _personalization_folds(
        document, max_rows=max_rows, run=run
    )

    evaluation_windows = document.get("total_evaluation_window_count")
    window_count = (
        int(evaluation_windows)
        if isinstance(evaluation_windows, int) and evaluation_windows >= 0
        else None
    )
    return PersonalizationDashboardData(
        provenance=provenance,
        paired_metric_table=paired_table,
        metric_delta_table=delta_table,
        coverage_table=coverage_table,
        fold_table=fold_table,
        subject_diagnostic_table=subject_table,
        population_evaluation_window_count=window_count,
        personalized_evaluation_window_count=window_count,
        calibration_window_count=_optional_count(
            document.get("total_calibration_window_count")
        ),
        cold_start_subject_count=_optional_count(
            document.get("cold_start_subject_count")
        ),
        personalized_subject_count=_optional_count(
            document.get("personalized_subject_count")
        ),
        warnings=warnings,
    )


def _personalization_folds(
    document: Mapping[str, Any], *, max_rows: int, run: DashboardRunSummary
) -> tuple[Any, Any, tuple[DashboardWarning, ...]]:
    folds = document.get("folds")
    if not isinstance(folds, list) or not folds:
        return None, None, ()
    warnings: list[DashboardWarning] = []
    fold_rows = []
    subject_rows = []
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        fold_rows.append(
            (
                fmt.text(fold.get("fold_index")),
                fmt.text(fold.get("evaluated")),
                fmt.text(fold.get("population_training_subject_count")),
                fmt.text(fold.get("evaluated_subject_count")),
                fmt.text(fold.get("personalized_subject_count")),
                fmt.text(fold.get("cold_start_subject_count")),
                fmt.text(fold.get("calibration_window_count")),
                fmt.text(fold.get("evaluation_window_count")),
                fmt.text(fold.get("excluded_overlap_window_count")),
                fmt.text(fold.get("unavailable_reason")),
            )
        )
        for correction in fold.get("corrections") or ():
            if not isinstance(correction, dict):
                continue
            subject_rows.append(
                (
                    fmt.text(correction.get("subject_id")),
                    fmt.text(fold.get("fold_index")),
                    fmt.text(correction.get("applied")),
                    fmt.text(correction.get("cold_start")),
                    fmt.text(correction.get("calibration_sample_count")),
                    fmt.text(correction.get("unavailable_reason")),
                )
            )
    fold_table = fmt.build_table(
        title="Per-fold personalization",
        columns=(
            "fold",
            "evaluated",
            "population training subjects",
            "evaluated subjects",
            "personalized subjects",
            "cold-start subjects",
            "calibration windows",
            "evaluation windows",
            "excluded overlap windows",
            "unavailable reason",
        ),
        rows=fold_rows,
        source_artifact="personalization.json",
        max_rows=max_rows,
    )
    subject_table = (
        fmt.build_table(
            title="Subject-wise software evaluation",
            columns=(
                "subject (pseudonymous)",
                "fold",
                "personalization applied",
                "cold start",
                "calibration windows",
                "unavailable reason",
            ),
            rows=subject_rows,
            source_artifact="personalization.json",
            max_rows=max_rows,
            caption=(
                "A group-level software diagnostic. Subject identifiers are "
                "pseudonymous labels; this table does not rank subjects and "
                "carries no judgement about any of them."
            ),
        )
        if subject_rows
        else None
    )
    counts = {
        fold.get("evaluation_window_count")
        for fold in folds
        if isinstance(fold, dict) and fold.get("evaluated")
    }
    if len(counts) > 1 and None not in counts:
        warnings.append(
            DashboardWarning(
                level=DashboardWarningLevel.INFORMATION,
                message=(
                    "evaluation-window counts differ between folds, which is "
                    "expected under grouped cross-validation. The population "
                    "and personalized results within each fold still cover "
                    "identical windows."
                ),
                subject=run.directory_name,
            )
        )
    return fold_table, subject_table, tuple(warnings)


def _named(entries: object) -> dict[str, Mapping[str, Any]]:
    if not isinstance(entries, list):
        return {}
    return {
        str(entry.get("name")): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("name") is not None
    }


def _mean_cell(entry: Mapping[str, Any] | None) -> str:
    if entry is None:
        return fmt.format_value(
            fmt.metric("value", None, unavailable_reason="not recorded")
        )
    metric = fmt.metric("value", entry.get("mean"))
    if not metric.available:
        reason = entry.get("unavailable_reason")
        return fmt.format_value(metric) if not reason else "Unavailable"
    deviation = fmt.metric("sd", entry.get("standard_deviation"))
    if deviation.available:
        return f"{fmt.format_value(metric)} ± {fmt.format_value(deviation)}"
    return fmt.format_value(metric)


def _delta_cell(
    population: Mapping[str, Any] | None, personalized: Mapping[str, Any] | None
) -> str:
    left = population.get("mean") if isinstance(population, dict) else None
    right = personalized.get("mean") if isinstance(personalized, dict) else None
    if not _is_number(left) or not _is_number(right):
        return "Unavailable"
    difference = float(right) - float(left)  # type: ignore[arg-type]
    rendered = fmt.format_value(fmt.metric("delta", difference))
    return f"+{rendered}" if difference > 0 else rendered


def _joined(values: object) -> str:
    """Comma-joined modality names, or ``Unavailable`` when there are none."""
    if not isinstance(values, list) or not values:
        return fmt.text(None)
    return fmt.text(", ".join(str(value) for value in values))


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _optional_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def modality_names(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Modality names recorded by a fusion document."""
    names: Sequence[Any] = document.get("modalities") or ()
    return tuple(str(name) for name in names)


__all__ = [
    "DISAGREEMENT_PREFIX",
    "FUSION_PREFIX",
    "MISSING_PREFIX",
    "WEIGHT_PREFIX",
    "load_fusion",
    "load_personalization",
    "modality_names",
]
