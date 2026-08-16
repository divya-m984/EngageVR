"""Fusion-specific diagnostics: coverage, contribution, and disagreement.

These are diagnostics *about the fusion*, kept separate from the
classification and regression metrics of :mod:`engagevr.training.metrics`,
which are reused unchanged.  A fusion result that scored well while fusing
only a third of its windows is a different result from one that fused all
of them, and coverage is what makes the difference visible.

Undefined stays undefined.  A diagnostic whose prerequisites are unmet is
``None`` with a stated reason; zero is never substituted, because zero is a
legitimate value and a reader could not tell the two apart.

Expert disagreement here is an **ensemble-disagreement diagnostic**, not an
uncertainty estimate and not a trigger for abstention.  Milestone 7 owns
uncertainty-aware inference.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from engagevr.schemas.experiments import AggregateMetric
from engagevr.schemas.fusion import (
    ExpertDisagreementSummary,
    FusionDiagnostics,
    FusionModality,
    FusionPrediction,
)
from engagevr.schemas.targets import TaskType
from engagevr.training.fusion import (
    distinct_predicted_classes,
    mean_pairwise_probability_distance,
    prediction_spread,
    probability_entropy,
)

#: Fewest available experts before a window can exhibit disagreement.
MINIMUM_DISAGREEMENT_EXPERTS = 2


def disagreement_summary(
    predictions: Sequence[FusionPrediction],
    task_type: TaskType,
) -> ExpertDisagreementSummary:
    """Summarise how far the modality experts differed from one another.

    Only windows with at least two available experts contribute: a single
    expert cannot disagree with anything.  Windows with fewer are counted
    in ``insufficient_expert_window_count`` and contribute to no summary.
    """
    evaluated: list[FusionPrediction] = []
    insufficient = 0
    for prediction in predictions:
        available = [p for p in prediction.modality_predictions if p.available]
        if len(available) >= MINIMUM_DISAGREEMENT_EXPERTS:
            evaluated.append(prediction)
        else:
            insufficient += 1

    if not evaluated:
        return ExpertDisagreementSummary(
            task_type=task_type,
            evaluated_window_count=0,
            insufficient_expert_window_count=insufficient,
            unavailable_reason=(
                "no evaluated window had two or more available modality "
                "experts, so no disagreement diagnostic is defined"
            ),
        )

    if task_type is TaskType.CLASSIFICATION:
        distinct: list[float] = []
        unanimous: list[float] = []
        distances: list[float] = []
        entropies: list[float] = []
        for prediction in evaluated:
            available = [p for p in prediction.modality_predictions if p.available]
            labels = [
                p.predicted_class for p in available if p.predicted_class is not None
            ]
            count = distinct_predicted_classes(labels)
            distinct.append(float(count))
            unanimous.append(1.0 if count == 1 else 0.0)
            vectors = [p.probabilities for p in available if p.probabilities]
            distance = mean_pairwise_probability_distance(vectors)
            if distance is not None:
                distances.append(distance)
            if prediction.fused and prediction.probabilities:
                entropies.append(probability_entropy(prediction.probabilities))
        unanimous_fraction = float(np.mean(unanimous))
        return ExpertDisagreementSummary(
            task_type=task_type,
            evaluated_window_count=len(evaluated),
            insufficient_expert_window_count=insufficient,
            mean_distinct_predicted_classes=float(np.mean(distinct)),
            unanimous_fraction=unanimous_fraction,
            disagreement_fraction=1.0 - unanimous_fraction,
            mean_pairwise_probability_distance=(
                float(np.mean(distances)) if distances else None
            ),
            mean_fused_probability_entropy=(
                float(np.mean(entropies)) if entropies else None
            ),
        )

    deviations: list[float] = []
    ranges: list[float] = []
    for prediction in evaluated:
        values = [
            p.predicted_value
            for p in prediction.modality_predictions
            if p.available and p.predicted_value is not None
        ]
        deviation, spread = prediction_spread(values)
        if deviation is not None and spread is not None:
            deviations.append(deviation)
            ranges.append(spread)
    return ExpertDisagreementSummary(
        task_type=task_type,
        evaluated_window_count=len(evaluated),
        insufficient_expert_window_count=insufficient,
        mean_prediction_standard_deviation=(
            float(np.mean(deviations)) if deviations else None
        ),
        mean_prediction_range=float(np.mean(ranges)) if ranges else None,
    )


def fusion_diagnostics(
    predictions: Sequence[FusionPrediction],
    modalities: Sequence[FusionModality],
    task_type: TaskType,
) -> FusionDiagnostics:
    """Coverage, contribution, weight, and disagreement diagnostics."""
    sample_count = len(predictions)
    fused = [p for p in predictions if p.fused]
    fused_count = len(fused)

    # Early and stacked fusion carry no per-window weights: early fusion
    # fits one estimator over concatenated features, and a stacker's
    # coefficients are a property of the meta-model rather than of a
    # window. For those, contribution is read from availability and the
    # mean weight stays unavailable rather than being invented.
    weighted = any(prediction.fusion_weights for prediction in fused)

    missing_rate: dict[str, float] = {}
    contribution_counts: dict[str, int] = {}
    mean_weight: dict[str, float | None] = {}
    for modality in modalities:
        name = modality.value
        if sample_count:
            absent = sum(
                1 for p in predictions if modality not in set(p.available_modalities)
            )
            missing_rate[name] = float(absent) / float(sample_count)
        else:
            missing_rate[name] = 0.0
        if weighted:
            weights = [
                next(
                    (
                        w.normalized_weight
                        for w in prediction.fusion_weights
                        if w.modality is modality
                    ),
                    0.0,
                )
                for prediction in fused
            ]
            contribution_counts[name] = sum(1 for w in weights if w > 0.0)
            mean_weight[name] = float(np.mean(weights)) if weights else None
        else:
            contribution_counts[name] = sum(
                1 for p in fused if modality in set(p.available_modalities)
            )
            mean_weight[name] = None

    expert_counts = [float(len(p.available_modalities)) for p in predictions]

    return FusionDiagnostics(
        sample_count=sample_count,
        fused_count=fused_count,
        unavailable_fusion_count=sample_count - fused_count,
        coverage=(float(fused_count) / float(sample_count) if sample_count else None),
        mean_available_expert_count=(
            float(np.mean(expert_counts)) if expert_counts else None
        ),
        missing_modality_rate=missing_rate,
        modality_contribution_counts=contribution_counts,
        mean_normalized_weight=mean_weight,
        disagreement=disagreement_summary(predictions, task_type),
    )


def pool_diagnostics(
    entries: Sequence[FusionDiagnostics],
    modalities: Sequence[FusionModality],
    task_type: TaskType,
) -> FusionDiagnostics | None:
    """Combine per-fold diagnostics into one pooled record.

    Counts are summed and rates are weighted by the number of windows each
    fold contributed, so the pooled record describes the whole evaluated
    set.  This is deliberately different from the fold-level *aggregates*,
    which weight folds equally: a pooled count answers "how many windows",
    an equal-weight aggregate answers "how did a typical fold behave", and
    conflating the two hides a large fold.
    """
    usable = [entry for entry in entries if entry is not None]
    if not usable:
        return None
    sample_count = sum(entry.sample_count for entry in usable)
    fused_count = sum(entry.fused_count for entry in usable)

    def _weighted(
        values: Sequence[float | None], weights: Sequence[int]
    ) -> float | None:
        pairs = [
            (float(v), float(w))
            for v, w in zip(values, weights, strict=True)
            if v is not None and w > 0
        ]
        total = sum(w for _v, w in pairs)
        if total <= 0.0:
            return None
        return float(sum(v * w for v, w in pairs) / total)

    sample_weights = [entry.sample_count for entry in usable]
    fused_weights = [entry.fused_count for entry in usable]

    missing_rate: dict[str, float] = {}
    contribution_counts: dict[str, int] = {}
    mean_weight: dict[str, float | None] = {}
    for modality in modalities:
        name = modality.value
        pooled_missing = _weighted(
            [entry.missing_modality_rate.get(name) for entry in usable], sample_weights
        )
        missing_rate[name] = 0.0 if pooled_missing is None else pooled_missing
        contribution_counts[name] = sum(
            entry.modality_contribution_counts.get(name, 0) for entry in usable
        )
        mean_weight[name] = _weighted(
            [entry.mean_normalized_weight.get(name) for entry in usable], fused_weights
        )

    disagreements = [
        entry.disagreement for entry in usable if entry.disagreement is not None
    ]
    pooled_disagreement: ExpertDisagreementSummary | None = None
    if disagreements:
        weights = [d.evaluated_window_count for d in disagreements]
        pooled_disagreement = ExpertDisagreementSummary(
            task_type=task_type,
            evaluated_window_count=sum(weights),
            insufficient_expert_window_count=sum(
                d.insufficient_expert_window_count for d in disagreements
            ),
            mean_distinct_predicted_classes=_weighted(
                [d.mean_distinct_predicted_classes for d in disagreements], weights
            ),
            unanimous_fraction=_weighted(
                [d.unanimous_fraction for d in disagreements], weights
            ),
            disagreement_fraction=_weighted(
                [d.disagreement_fraction for d in disagreements], weights
            ),
            mean_pairwise_probability_distance=_weighted(
                [d.mean_pairwise_probability_distance for d in disagreements], weights
            ),
            mean_fused_probability_entropy=_weighted(
                [d.mean_fused_probability_entropy for d in disagreements], weights
            ),
            mean_prediction_standard_deviation=_weighted(
                [d.mean_prediction_standard_deviation for d in disagreements], weights
            ),
            mean_prediction_range=_weighted(
                [d.mean_prediction_range for d in disagreements], weights
            ),
            unavailable_reason=(
                None
                if sum(weights)
                else (
                    "no evaluated window had two or more available modality "
                    "experts, so no disagreement diagnostic is defined"
                )
            ),
        )

    return FusionDiagnostics(
        sample_count=sample_count,
        fused_count=fused_count,
        unavailable_fusion_count=sample_count - fused_count,
        coverage=(float(fused_count) / float(sample_count) if sample_count else None),
        mean_available_expert_count=_weighted(
            [entry.mean_available_expert_count for entry in usable], sample_weights
        ),
        missing_modality_rate=missing_rate,
        modality_contribution_counts=contribution_counts,
        mean_normalized_weight=mean_weight,
        disagreement=pooled_disagreement,
    )


#: Scalar diagnostic fields aggregated across folds.
FUSION_AGGREGATE_FIELDS: tuple[str, ...] = (
    "coverage",
    "mean_available_expert_count",
)

#: Disagreement fields aggregated across folds, per task type.
CLASSIFICATION_DISAGREEMENT_FIELDS: tuple[str, ...] = (
    "mean_distinct_predicted_classes",
    "unanimous_fraction",
    "disagreement_fraction",
    "mean_pairwise_probability_distance",
    "mean_fused_probability_entropy",
)
REGRESSION_DISAGREEMENT_FIELDS: tuple[str, ...] = (
    "mean_prediction_standard_deviation",
    "mean_prediction_range",
)


def aggregate_fusion_diagnostics(
    fold_diagnostics: Sequence[FusionDiagnostics | None],
    modalities: Sequence[FusionModality],
    task_type: TaskType,
    *,
    total_fold_count: int,
) -> tuple[AggregateMetric, ...]:
    """Aggregate fold diagnostics with the unweighted mean over valid folds.

    Folds are weighted equally, matching
    :func:`engagevr.training.metrics.aggregate_fold_metrics`: a size-weighted
    mean would let one large participant dominate, which is the opposite of
    what grouped cross-validation is for.
    """
    aggregates: list[AggregateMetric] = []

    for field in FUSION_AGGREGATE_FIELDS:
        values = [
            None if entry is None else getattr(entry, field, None)
            for entry in fold_diagnostics
        ]
        aggregates.append(_aggregate(f"fusion.{field}", values, total_fold_count))

    for modality in modalities:
        name = modality.value
        weight_values = [
            None if entry is None else entry.mean_normalized_weight.get(name)
            for entry in fold_diagnostics
        ]
        aggregates.append(
            _aggregate(
                f"fusion.mean_normalized_weight.{name}",
                weight_values,
                total_fold_count,
            )
        )
        missing_values = [
            None if entry is None else entry.missing_modality_rate.get(name)
            for entry in fold_diagnostics
        ]
        aggregates.append(
            _aggregate(
                f"fusion.missing_modality_rate.{name}",
                missing_values,
                total_fold_count,
            )
        )

    fields = (
        CLASSIFICATION_DISAGREEMENT_FIELDS
        if task_type is TaskType.CLASSIFICATION
        else REGRESSION_DISAGREEMENT_FIELDS
    )
    for field in fields:
        values = [
            None
            if entry is None or entry.disagreement is None
            else getattr(entry.disagreement, field, None)
            for entry in fold_diagnostics
        ]
        aggregates.append(
            _aggregate(f"fusion.disagreement.{field}", values, total_fold_count)
        )
    return tuple(aggregates)


def _aggregate(
    name: str, values: Sequence[float | None], total_fold_count: int
) -> AggregateMetric:
    defined = [float(v) for v in values if v is not None]
    return AggregateMetric(
        name=name,
        mean=float(np.mean(defined)) if defined else None,
        standard_deviation=float(np.std(defined, ddof=0)) if defined else None,
        valid_fold_count=len(defined),
        total_fold_count=total_fold_count,
        fold_values=tuple(float(v) if v is not None else None for v in values),
        unavailable_reason=(
            None
            if defined
            else f"{name} was undefined in every evaluated fold, so no aggregate exists"
        ),
    )


__all__ = [
    "CLASSIFICATION_DISAGREEMENT_FIELDS",
    "FUSION_AGGREGATE_FIELDS",
    "MINIMUM_DISAGREEMENT_EXPERTS",
    "REGRESSION_DISAGREEMENT_FIELDS",
    "aggregate_fusion_diagnostics",
    "disagreement_summary",
    "fusion_diagnostics",
    "pool_diagnostics",
]
