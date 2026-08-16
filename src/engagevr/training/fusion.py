"""Multimodal-fusion algebra: modality groups, weights, and combination.

This module holds the deterministic, dependency-free core of Milestone 6:
which columns belong to which modality, how weights are derived, how
available modality outputs are combined, and how their disagreement is
described.  Everything here is pure: no estimator is fitted, no fold is
touched, and no artifact is written.

Fusion is not the Milestone 5 ablation
--------------------------------------
``all_available`` in :mod:`engagevr.training.ablation` means "no feature
group was removed".  It fits one model on whatever columns survived and
records nothing about modality structure.  Fusion, here, tracks modality
membership, modality availability, modality quality, missing-modality
patterns, the fusion configuration, and fusion-specific robustness
diagnostics, and it offers architectures — early feature fusion, late
decision-level fusion, quality-aware late fusion, stacked fusion — that the
ablation set does not contain.  Renaming an ablation would not produce any
of that.

Missing is never zero
---------------------
A modality that produced no prediction is represented through
**availability**.  It never contributes a zero-valued measurement, a zero
probability vector, or a uniform probability vector standing in for a real
one.  Its effective weight is zero by construction, the remaining weights
are renormalised over the experts that did contribute, and the exclusion is
recorded with a reason.

Nothing here is a neural architecture.  Concatenating feature groups is
concatenation; it is not attention.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from engagevr.schemas.features import (
    AVAILABILITY_PREFIX,
    FEATURE_PREFIX,
    FeatureCatalog,
    FeatureModality,
    modality_available_column,
    modality_quality_column,
)
from engagevr.schemas.fusion import (
    PROBABILITY_SUM_TOLERANCE,
    FusionModality,
    FusionStrategy,
    MissingQualityPolicy,
    ModalityPrediction,
    ModalityWeight,
    QualitySource,
    QualityWeightingConfiguration,
)

#: Catalogue modality behind each fusion modality.  ``FeatureModality.QUALITY``
#: appears in no entry: capture-quality diagnostics are support signals.
FEATURE_MODALITY_OF: dict[FusionModality, FeatureModality] = {
    FusionModality.BEHAVIOURAL: FeatureModality.BEHAVIOURAL,
    FusionModality.HEAD_POSE: FeatureModality.HEAD_POSE,
    FusionModality.RPPG: FeatureModality.RPPG,
    FusionModality.TASK: FeatureModality.TASK,
}

#: Catalogue modalities that are support/context signals rather than
#: measurement modalities, and may therefore never carry a fusion expert.
SUPPORT_MODALITIES: frozenset[FeatureModality] = frozenset({FeatureModality.QUALITY})

#: Human-readable description of each implemented strategy.
STRATEGY_DESCRIPTIONS: dict[FusionStrategy, str] = {
    FusionStrategy.EARLY: (
        "Early feature fusion: the permitted features of every selected "
        "modality are concatenated into one predictor matrix, with modality "
        "availability and per-feature missingness carried as separate "
        "columns, and one estimator is fitted on the result inside each "
        "training fold."
    ),
    FusionStrategy.UNIFORM_LATE: (
        "Uniform late fusion: one estimator per modality; the available "
        "experts' outputs are combined with equal weights, renormalised "
        "over the experts that actually produced a prediction."
    ),
    FusionStrategy.QUALITY_LATE: (
        "Quality-aware late fusion: one estimator per modality; the "
        "available experts' outputs are combined with weights derived from "
        "modality availability and recorded modality signal quality, using "
        "the documented equation, then renormalised over the contributors."
    ),
    FusionStrategy.VALIDATION_WEIGHTED_LATE: (
        "Validation-weighted late fusion: per-modality weights are estimated "
        "from inner validation groups drawn only from the outer training "
        "portion, never from the outer test fold, then applied to the "
        "available experts and renormalised."
    ),
    FusionStrategy.STACKED_LATE: (
        "Stacked fusion: an interpretable meta-estimator is fitted on "
        "out-of-fold modality-expert predictions generated inside the outer "
        "training groups, then applied to expert predictions for the "
        "untouched outer test groups."
    ),
}


class FusionError(ValueError):
    """A fusion operation cannot be performed as requested."""


class FusionUnavailableError(FusionError):
    """Fusion produced no result for a window, and said why."""


def parse_modality(name: str) -> FusionModality:
    """Parse a modality name, refusing support signals with an explanation.

    Raises
    ------
    FusionError
        If ``name`` is not a fusion modality.
    """
    try:
        return FusionModality(name)
    except ValueError as exc:
        valid = ", ".join(m.value for m in FusionModality)
        if name == FeatureModality.QUALITY.value:
            raise FusionError(
                "'quality' is not a fusion modality. Capture-quality "
                "diagnostics, availability flags, and missingness indicators "
                "are support/context signals: they describe a measurement, "
                "they are not one. They may inform explicitly named "
                "quality-aware weighting, but they must never become another "
                f"measurement modality. Valid modalities: {valid}."
            ) from exc
        raise FusionError(
            f"unknown fusion modality {name!r}; valid modalities: {valid}"
        ) from exc


def parse_strategy(name: str) -> FusionStrategy:
    """Parse a strategy name, accepting hyphens as well as underscores.

    Raises
    ------
    FusionError
        If ``name`` is not an implemented strategy.
    """
    normalised = name.strip().lower().replace("-", "_")
    aliases = {"validation_late": FusionStrategy.VALIDATION_WEIGHTED_LATE.value}
    normalised = aliases.get(normalised, normalised)
    try:
        return FusionStrategy(normalised)
    except ValueError as exc:
        valid = ", ".join(s.value.replace("_", "-") for s in FusionStrategy)
        raise FusionError(
            f"unknown fusion strategy {name!r}; implemented strategies: "
            f"{valid}. No deep or neural fusion is implemented."
        ) from exc


def _catalog_feature_of(column: str) -> str | None:
    """Catalogue feature a predictor column belongs to, or ``None``."""
    if column.startswith(FEATURE_PREFIX):
        return column.removeprefix(FEATURE_PREFIX)
    if column.startswith(AVAILABILITY_PREFIX):
        return column.removeprefix(AVAILABILITY_PREFIX)
    return None


def modality_expert_columns(
    modality: FusionModality,
    predictor_columns: Sequence[str],
    catalog: FeatureCatalog,
    *,
    include_modality_quality: bool = False,
) -> tuple[str, ...]:
    """Columns one modality expert may see, in catalogue order.

    An expert sees the measured features of its own modality, their
    per-feature availability flags, and its own modality-availability flag.
    It sees its modality-quality column only when explicitly configured:
    quality's declared role in this milestone is fusion weighting, and
    letting it into every expert silently would blur that boundary.
    """
    wanted = FEATURE_MODALITY_OF[modality]
    selected: list[str] = []
    for column in predictor_columns:
        feature = _catalog_feature_of(column)
        if feature is None:
            continue
        try:
            entry = catalog.get(feature)
        except KeyError:  # pragma: no cover - refused earlier by assert_no_leakage
            continue
        if entry.modality is wanted:
            selected.append(column)
    availability = modality_available_column(modality.value)
    if availability in set(predictor_columns):
        selected.append(availability)
    if include_modality_quality:
        quality = modality_quality_column(modality.value)
        if quality in set(predictor_columns):
            selected.append(quality)
    return tuple(selected)


def early_fusion_columns(
    modalities: Sequence[FusionModality],
    predictor_columns: Sequence[str],
    catalog: FeatureCatalog,
    *,
    include_modality_quality: bool = False,
) -> tuple[str, ...]:
    """The early-fusion predictor matrix columns, in catalogue order.

    Catalogue order is preserved rather than modality order, so a change to
    modality ordering in a configuration does not silently change the column
    order of the matrix.

    Raises
    ------
    FusionError
        If fewer than two modalities are requested, or if a requested
        modality contributes no column.
    """
    if len(set(modalities)) < 2:
        raise FusionError(
            "early feature fusion requires at least two modality groups; with "
            "one group there is nothing to fuse"
        )
    per_modality = {
        modality: modality_expert_columns(
            modality,
            predictor_columns,
            catalog,
            include_modality_quality=include_modality_quality,
        )
        for modality in modalities
    }
    empty = sorted(m.value for m, columns in per_modality.items() if not columns)
    if empty:
        raise FusionError(
            f"modality group(s) {empty} contribute no permitted predictor to "
            "this dataset, so early fusion over the requested groups is not "
            "available. The absence is reported rather than silently reduced "
            "to whatever happened to survive."
        )
    keep = {column for columns in per_modality.values() for column in columns}
    return tuple(column for column in predictor_columns if column in keep)


def missing_modality_pattern(
    availability: Mapping[FusionModality, bool],
    modalities: Sequence[FusionModality],
) -> str:
    """A stable textual key naming which modalities were absent."""
    absent = [m.value for m in modalities if not availability.get(m, False)]
    return "none" if not absent else "+".join(sorted(absent))


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _WeightDraft:
    """One modality's weight before normalisation across contributors."""

    modality: FusionModality
    base: float
    availability: float
    quality_used: float | None
    quality_source: QualitySource
    normalized_quality: float | None
    raw: float
    exclusion: str | None


def resolve_base_weights(
    modalities: Sequence[FusionModality],
    configured: Mapping[str, float] | None = None,
) -> dict[FusionModality, float]:
    """Base weight per modality; deterministic 1.0 unless configured.

    An empty configuration produces equal weights, which is the control.
    No optimised weight set is shipped as a default.

    Raises
    ------
    FusionError
        On a non-finite or non-positive configured weight.
    """
    resolved: dict[FusionModality, float] = {}
    for modality in modalities:
        weight = 1.0
        if configured:
            weight = float(configured.get(modality.value, 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise FusionError(
                f"base weight for modality {modality.value!r} must be finite "
                f"and positive; got {weight!r}"
            )
        resolved[modality] = weight
    return resolved


def build_fusion_weights(
    *,
    modalities: Sequence[FusionModality],
    predictions: Mapping[FusionModality, ModalityPrediction],
    base_weights: Mapping[FusionModality, float],
    quality: Mapping[FusionModality, float | None] | None = None,
    quality_config: QualityWeightingConfiguration | None = None,
) -> tuple[ModalityWeight, ...]:
    """Compute per-modality weights for one window.

    The documented equation is::

        raw_effective_weight_m = base_weight_m * availability_m * normalized_quality_m
        normalized_weight_m    = raw_effective_weight_m / sum over contributors

    ``availability_m`` is 1 when modality ``m`` produced a prediction for
    this window and 0 otherwise, so an unavailable modality has zero
    effective weight by construction and is never substituted with a
    placeholder value.

    When ``quality_config`` is ``None`` (or disabled) the quality factor is
    fixed at 1.0 and every weight records ``quality_source=not_used``: this
    is the deterministic equal-weight control.

    Raises
    ------
    FusionError
        On a negative or non-finite base weight or quality value.
    """
    use_quality = quality_config is not None and quality_config.enabled
    drafts: list[_WeightDraft] = []

    for modality in modalities:
        prediction = predictions.get(modality)
        available = prediction is not None and prediction.available
        base = float(base_weights.get(modality, 1.0))
        if not math.isfinite(base) or base <= 0.0:
            raise FusionError(
                f"base weight for modality {modality.value!r} must be finite "
                f"and positive; got {base!r}"
            )

        quality_value: float | None = None
        source = QualitySource.NOT_USED
        normalized_quality: float | None = None
        exclusion: str | None = None

        if not available:
            exclusion = (
                prediction.unavailable_reason
                if prediction is not None and prediction.unavailable_reason
                else (
                    f"modality {modality.value!r} produced no prediction for "
                    "this window"
                )
            )
        elif use_quality:
            assert quality_config is not None  # narrowed by use_quality
            recorded = None if quality is None else quality.get(modality)
            if recorded is None:
                if (
                    quality_config.missing_quality_policy
                    is MissingQualityPolicy.EXCLUDE
                ):
                    source = QualitySource.UNAVAILABLE
                    exclusion = (
                        f"modality {modality.value!r} recorded no signal "
                        "quality and the missing-quality policy is 'exclude'; "
                        "missing quality is never treated as perfect quality"
                    )
                else:
                    quality_value = float(quality_config.missing_quality_fallback)
                    source = QualitySource.DOCUMENTED_FALLBACK
            else:
                quality_value = float(recorded)
                source = QualitySource.MEASURED
                if not math.isfinite(quality_value):
                    raise FusionError(
                        f"modality {modality.value!r} reported a non-finite "
                        "signal quality; quality must be a finite value in "
                        "[0, 1]"
                    )
                if not 0.0 <= quality_value <= 1.0:
                    raise FusionError(
                        f"modality {modality.value!r} reported signal quality "
                        f"{quality_value!r}, which is outside the valid range "
                        "[0, 1]"
                    )
            if exclusion is None and quality_value is not None:
                normalized_quality = quality_value
                if quality_value < quality_config.minimum_quality:
                    exclusion = (
                        f"modality {modality.value!r} signal quality "
                        f"{quality_value:.4f} is below the configured minimum "
                        f"{quality_config.minimum_quality:.4f}; the "
                        "measurement is unusable, which is a statement about "
                        "the signal and not about the person"
                    )

        availability = 1.0 if available and exclusion is None else 0.0
        factor = 1.0 if not use_quality else (normalized_quality or 0.0)
        raw = base * availability * factor
        if exclusion is None and available and use_quality:
            assert quality_config is not None  # narrowed by use_quality
            if raw < quality_config.minimum_effective_weight:
                exclusion = (
                    f"modality {modality.value!r} effective weight "
                    f"{raw:.3e} is below the minimum effective weight "
                    f"{quality_config.minimum_effective_weight:.3e}"
                )
        if exclusion is not None:
            raw = 0.0
            availability = 0.0

        drafts.append(
            _WeightDraft(
                modality=modality,
                base=base,
                availability=availability,
                quality_used=quality_value,
                quality_source=source,
                normalized_quality=normalized_quality,
                raw=raw,
                exclusion=exclusion,
            )
        )

    total = sum(draft.raw for draft in drafts)
    weights: list[ModalityWeight] = []
    for draft in drafts:
        contributed = draft.exclusion is None and draft.raw > 0.0 and total > 0.0
        exclusion = draft.exclusion
        if not contributed and exclusion is None:
            exclusion = (
                "every candidate weight was zero, so no modality could "
                "contribute to this window"
            )
        weights.append(
            ModalityWeight(
                modality=draft.modality,
                base_weight=draft.base,
                availability=draft.availability,
                quality_used=draft.quality_used,
                quality_source=draft.quality_source,
                normalized_quality=draft.normalized_quality,
                raw_effective_weight=draft.raw,
                normalized_weight=(draft.raw / total) if contributed else 0.0,
                contributed=contributed,
                exclusion_reason=None if contributed else exclusion,
            )
        )
    return tuple(weights)


def contributing_modalities(
    weights: Sequence[ModalityWeight],
) -> tuple[FusionModality, ...]:
    """Modalities whose normalised weight is non-zero, in supplied order."""
    return tuple(weight.modality for weight in weights if weight.contributed)


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------


def align_probability_vector(
    probabilities: Sequence[float],
    source_vocabulary: Sequence[str],
    target_vocabulary: Sequence[str],
) -> tuple[float, ...]:
    """Reorder a probability vector onto ``target_vocabulary``.

    Classes the source never saw receive zero and the vector is
    renormalised, so the result is always a distribution over the target
    vocabulary.  Combining vectors whose columns mean different things is
    the classic silent fusion bug; this is the guard against it.

    Raises
    ------
    FusionError
        If the vector does not match its own vocabulary, carries a
        non-finite or negative value, or has no mass on the target
        vocabulary.
    """
    if len(probabilities) != len(source_vocabulary):
        raise FusionError(
            f"{len(probabilities)} probabilities were supplied for "
            f"{len(source_vocabulary)} source classes"
        )
    for value in probabilities:
        if not math.isfinite(value):
            raise FusionError(f"probability {value!r} is not finite")
        if value < 0.0:
            raise FusionError(f"probability {value!r} is negative")
    position = {label: index for index, label in enumerate(target_vocabulary)}
    aligned = [0.0] * len(target_vocabulary)
    for value, label in zip(probabilities, source_vocabulary, strict=True):
        index = position.get(label)
        if index is not None:
            aligned[index] += float(value)
    total = sum(aligned)
    if total <= 0.0:
        raise FusionError(
            "no probability mass survives alignment onto the target class "
            f"vocabulary {list(target_vocabulary)}; the vocabularies do not "
            "overlap and the vectors must not be combined"
        )
    return tuple(value / total for value in aligned)


def fuse_class_probabilities(
    contributions: Sequence[tuple[float, Sequence[float]]],
    vocabulary: Sequence[str],
) -> tuple[float, ...]:
    """Weighted average of aligned probability vectors.

    ``contributions`` pairs a non-negative weight with a probability vector
    already aligned to ``vocabulary``.  The result is renormalised so it
    sums to exactly one within floating-point tolerance.

    Raises
    ------
    FusionError
        If there is no contribution, a weight is negative or non-finite, a
        probability is negative or non-finite, a vector has the wrong
        length, or the total mass is zero.
    """
    if not contributions:
        raise FusionError(
            "probability fusion requires at least one available expert; there "
            "is nothing to combine"
        )
    size = len(vocabulary)
    combined = [0.0] * size
    weight_total = 0.0
    for weight, vector in contributions:
        if not math.isfinite(weight):
            raise FusionError(f"fusion weight {weight!r} is not finite")
        if weight < 0.0:
            raise FusionError(
                f"fusion weight {weight!r} is negative; a negative weight "
                "would subtract an expert's evidence rather than down-weight it"
            )
        if len(vector) != size:
            raise FusionError(
                f"a contributed probability vector has {len(vector)} entries "
                f"but {size} classes are declared"
            )
        weight_total += weight
        for index, value in enumerate(vector):
            if not math.isfinite(value):
                raise FusionError(f"probability {value!r} is not finite")
            if value < 0.0:
                raise FusionError(f"probability {value!r} is negative")
            combined[index] += weight * float(value)
    if weight_total <= 0.0:
        raise FusionError(
            "every contributed fusion weight was zero, so no fused probability exists"
        )
    total = sum(combined)
    if total <= 0.0:
        raise FusionError(
            "the weighted probability vectors carry no mass; no fused "
            "distribution exists"
        )
    fused = [value / total for value in combined]
    drift = abs(sum(fused) - 1.0)
    if drift > PROBABILITY_SUM_TOLERANCE:  # pragma: no cover - numeric guard
        raise FusionError(
            f"fused probabilities sum to {sum(fused)!r} after renormalisation"
        )
    return tuple(fused)


def fuse_regression_predictions(
    contributions: Sequence[tuple[float, float]],
) -> float:
    """Weighted average of available numeric modality estimates.

    An unavailable expert is simply absent from ``contributions``; it is
    never replaced with zero, and the weights are normalised over the
    experts that are present.

    Raises
    ------
    FusionError
        If there is no contribution, a weight or value is negative or
        non-finite, or the weights sum to zero.
    """
    if not contributions:
        raise FusionError(
            "regression fusion requires at least one available expert; there "
            "is nothing to combine, and an absent estimate is never replaced "
            "with zero"
        )
    total = 0.0
    accumulated = 0.0
    for weight, value in contributions:
        if not math.isfinite(weight):
            raise FusionError(f"fusion weight {weight!r} is not finite")
        if weight < 0.0:
            raise FusionError(f"fusion weight {weight!r} is negative")
        if not math.isfinite(value):
            raise FusionError(
                f"expert prediction {value!r} is not finite; an expert that "
                "cannot produce a finite value must report unavailable"
            )
        total += weight
        accumulated += weight * float(value)
    if total <= 0.0:
        raise FusionError(
            "every contributed fusion weight was zero, so no fused prediction exists"
        )
    fused = accumulated / total
    if not math.isfinite(fused):  # pragma: no cover - numeric guard
        raise FusionError("the fused regression prediction is not finite")
    return float(fused)


# ---------------------------------------------------------------------------
# Disagreement diagnostics
# ---------------------------------------------------------------------------


def distinct_predicted_classes(labels: Sequence[str]) -> int:
    """Number of different labels among the available experts."""
    return len(set(labels))


def mean_pairwise_probability_distance(
    vectors: Sequence[Sequence[float]],
) -> float | None:
    """Mean Euclidean distance between every pair of probability vectors.

    ``None`` when fewer than two vectors are supplied: a single expert
    cannot disagree with anything.
    """
    if len(vectors) < 2:
        return None
    distances: list[float] = []
    for first in range(len(vectors)):
        for second in range(first + 1, len(vectors)):
            a = vectors[first]
            b = vectors[second]
            if len(a) != len(b):
                raise FusionError(
                    "probability vectors of different lengths cannot be "
                    "compared; align them to one class vocabulary first"
                )
            distances.append(
                math.sqrt(
                    sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True))
                )
            )
    return float(sum(distances) / len(distances))


def probability_entropy(vector: Sequence[float]) -> float:
    """Shannon entropy of a probability vector, in nats."""
    total = 0.0
    for value in vector:
        probability = float(value)
        if probability > 0.0:
            total -= probability * math.log(probability)
    return float(total)


def prediction_spread(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Population standard deviation and max-min range of expert predictions.

    ``(None, None)`` when fewer than two predictions are supplied.
    """
    if len(values) < 2:
        return None, None
    numbers = [float(v) for v in values]
    mean = sum(numbers) / len(numbers)
    variance = sum((v - mean) ** 2 for v in numbers) / len(numbers)
    return float(math.sqrt(variance)), float(max(numbers) - min(numbers))


__all__ = [
    "FEATURE_MODALITY_OF",
    "STRATEGY_DESCRIPTIONS",
    "SUPPORT_MODALITIES",
    "FusionError",
    "FusionUnavailableError",
    "align_probability_vector",
    "build_fusion_weights",
    "contributing_modalities",
    "distinct_predicted_classes",
    "early_fusion_columns",
    "fuse_class_probabilities",
    "fuse_regression_predictions",
    "mean_pairwise_probability_distance",
    "missing_modality_pattern",
    "modality_expert_columns",
    "parse_modality",
    "parse_strategy",
    "prediction_spread",
    "probability_entropy",
    "resolve_base_weights",
]
