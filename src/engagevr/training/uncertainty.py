"""Confidence, conformal intervals, thresholds, and selective metrics.

Every function here is **pure**: it takes numbers and returns numbers or
typed records.  Nothing fits an estimator, nothing reads a file, and
nothing consults a wall clock, so each rule below can be tested against
its stated equation directly rather than through a training run.

Documented equations
--------------------
- **Predictive entropy** — ``H(p) = -sum_c p_c * log(p_c)`` in **nats**
  (natural logarithm), with the convention ``0 * log(0) = 0``.  A
  normalised variant ``H(p) / log(K)`` over ``K >= 2`` classes is returned
  alongside and lies in ``[0, 1]``.
- **Top-two margin** — ``margin = p_(1) - p_(2)``, the largest minus the
  second-largest predicted class probability.  A ranking diagnostic.
- **Classification confidence** — ``max_c p_calibrated(c | x)``, together
  with the class that attained it.  The word *confidence* is used only
  when the probabilities satisfied the calibration contract; otherwise the
  identical number is named ``selection_score``.
- **Selective acceptance** — ``accept if score >= tau``.  The boundary is
  **inclusive**.
- **Split conformal residual quantile** — for residuals
  ``r_i = |y_i - yhat_i|`` and miscoverage ``alpha``,
  ``k = ceil((n + 1) * (1 - alpha))`` and ``q`` is the ``k``-th smallest
  residual (1-indexed).  When ``k > n`` the rule cannot be met and the
  interval is unavailable.
- **Coverage** — ``accepted / total``, where ``total`` counts every
  evaluated window including those with no prediction, so
  ``accepted + abstained + unavailable = total`` reconciles exactly.
- **Empirical risk** — ``1 - accepted_accuracy``, over accepted windows.

What this module refuses to do
------------------------------
It will not compute a confidence score from an uncalibrated vector and
call it calibrated, will not turn an interval width into a probability,
will not multiply signal quality into a model probability, and will not
select a threshold from outer-test outcomes.  Each of those is a distinct
refusal with its own message.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from engagevr.schemas.experiments import (
    CalibrationMetrics,
    ClassificationMetrics,
    RegressionMetrics,
)
from engagevr.schemas.targets import TaskType
from engagevr.schemas.uncertainty import (
    CLASSIFICATION_ACCEPTANCE_RULE,
    COVERAGE_AXIS_DIRECTION,
    COVERAGE_AXIS_FOR_TASK,
    REGRESSION_ACCEPTANCE_RULE,
    AbstentionReason,
    CoverageAxis,
    CoveragePoint,
    EstimatedThresholdRecord,
    EvidenceGateConfiguration,
    MonotonicDirection,
    PersonalThresholdRecord,
    ProbabilityCalibrationStatus,
    RiskCoveragePoint,
    SelectiveMetrics,
    SelectivePredictionConfiguration,
    ThresholdObjective,
    ThresholdSource,
    UncertaintyMethod,
)
from engagevr.training.metrics import classification_metrics, regression_metrics

#: Tolerance for "this probability vector sums to one".
PROBABILITY_SUM_TOLERANCE = 1e-9

#: numpy quantile method used for the personal threshold.  ``"lower"``
#: returns an *observed* confidence value rather than interpolating a value
#: no window produced, which keeps the threshold explainable and exactly
#: reproducible across platforms.  Typed as a ``Literal`` so a typo cannot
#: silently change the quantile convention the artifacts claim.
PERSONAL_QUANTILE_METHOD: Literal["lower"] = "lower"

#: Anything that can stand in for a numeric vector here: a plain sequence
#: from a schema record, or the array a previous step already built.
Vector = Sequence[float] | np.ndarray


class UncertaintyError(ValueError):
    """An uncertainty computation cannot be performed as requested."""


# ---------------------------------------------------------------------------
# Classification confidence and diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfidenceComponents:
    """The four separately-stored classification diagnostics for one window.

    They are returned as four fields rather than one number on purpose:
    collapsing them would destroy exactly the distinctions Milestone 7
    exists to preserve.
    """

    predicted_class: str
    maximum_probability: float
    maximum_probability_class: str
    entropy: float
    normalized_entropy: float | None
    margin: float


def assert_probability_vector(
    probabilities: Vector, vocabulary: Sequence[str], *, context: str
) -> np.ndarray:
    """Validate and return a probability vector as a float array.

    Raises
    ------
    UncertaintyError
        If the vector is the wrong length, non-finite, negative, or does
        not sum to one.  A vector failing any of these is not a
        distribution, and a maximum taken from it would not be a
        probability.
    """
    values = np.asarray(list(probabilities), dtype=float)
    if values.size != len(vocabulary):
        raise UncertaintyError(
            f"{context}: {values.size} probabilities for {len(vocabulary)} class(es)"
        )
    if values.size == 0:
        raise UncertaintyError(
            f"{context}: an empty probability vector is not a distribution"
        )
    if not np.isfinite(values).all():
        raise UncertaintyError(f"{context}: the probability vector is not finite")
    if (values < 0.0).any():
        raise UncertaintyError(
            f"{context}: the probability vector has a negative entry"
        )
    total = float(math.fsum(float(v) for v in values))
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise UncertaintyError(
            f"{context}: probabilities sum to {total!r}, not 1.0. A vector that "
            "is not a distribution is refused rather than renormalised here: "
            "renormalisation belongs where the vector is produced, so that the "
            "record shows what the model actually emitted."
        )
    return values


def predictive_entropy(probabilities: Vector) -> float:
    """``H(p) = -sum_c p_c * log(p_c)`` in nats, with ``0 * log(0) = 0``.

    Entropy is a *diagnostic*.  It is not signal quality, and it is not
    automatically calibrated uncertainty: an entropy computed from a
    miscalibrated vector describes that miscalibrated vector.
    """
    values = np.asarray(list(probabilities), dtype=float)
    positive = values[values > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def normalized_entropy(probabilities: Vector) -> float | None:
    """``H(p) / log(K)`` over ``K`` classes, or ``None`` when ``K < 2``.

    With one class the maximum entropy is zero and the ratio is undefined;
    it is returned as ``None`` rather than as zero, which would read as a
    perfectly certain prediction.
    """
    count = len(list(probabilities))
    if count < 2:
        return None
    return float(predictive_entropy(probabilities) / math.log(count))


def prediction_margin(probabilities: Vector) -> float:
    """``p_(1) - p_(2)``: the top-two probability margin.

    A ranking diagnostic, not calibrated confidence.  With a single class
    the margin is the whole mass, which is why a single-class vocabulary is
    refused wherever a margin is required.
    """
    values = np.asarray(list(probabilities), dtype=float)
    if values.size == 0:
        raise UncertaintyError("a margin is undefined for an empty probability vector")
    if values.size == 1:
        raise UncertaintyError(
            "a top-two margin is undefined for a single-class vocabulary: there "
            "is no second class to subtract"
        )
    ordered = np.sort(values)[::-1]
    return float(ordered[0] - ordered[1])


def confidence_components(
    probabilities: Vector, vocabulary: Sequence[str], *, context: str
) -> ConfidenceComponents:
    """Predicted class, maximum probability, entropy, and margin, separately.

    The predicted class is the ``argmax``.  Ties are broken by the first
    position in the declared class vocabulary, which is a fixed ordering,
    so two runs of one configuration agree.
    """
    values = assert_probability_vector(probabilities, vocabulary, context=context)
    position = int(np.argmax(values))
    label = str(vocabulary[position])
    return ConfidenceComponents(
        predicted_class=label,
        maximum_probability=float(values[position]),
        maximum_probability_class=label,
        entropy=predictive_entropy(values),
        normalized_entropy=normalized_entropy(values),
        margin=prediction_margin(values),
    )


def confidence_method(
    status: ProbabilityCalibrationStatus,
) -> UncertaintyMethod:
    """The method name a probability vector of this status may claim.

    Raises
    ------
    UncertaintyError
        If no probability vector exists at all, in which case neither a
        confidence score nor a selection score is defined.
    """
    if status is ProbabilityCalibrationStatus.CALIBRATED:
        return UncertaintyMethod.MAX_CALIBRATED_PROBABILITY
    if status is ProbabilityCalibrationStatus.UNCALIBRATED:
        return UncertaintyMethod.MAX_UNCALIBRATED_PROBABILITY
    raise UncertaintyError(
        "no probability vector is available, so neither a calibrated "
        "confidence score nor an uncalibrated selection score is defined. "
        "The window abstains with reason "
        f"{AbstentionReason.MODEL_PREDICTION_UNAVAILABLE.value!r}."
    )


# ---------------------------------------------------------------------------
# Selective acceptance
# ---------------------------------------------------------------------------


def accepts_at_threshold(score: float, threshold: float) -> bool:
    """``score >= threshold``: the inclusive-boundary acceptance rule.

    A score exactly equal to the threshold is **accepted**.  The convention
    is stated here, recorded on every decision, and pinned by a test,
    because an off-by-one-epsilon disagreement between the rule and the
    curve would silently shift every reported coverage.
    """
    if not math.isfinite(score):
        raise UncertaintyError(
            f"a non-finite selection score ({score!r}) cannot be compared with a "
            "threshold; the window has no usable prediction"
        )
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise UncertaintyError(
            f"threshold {threshold!r} is not a finite value in [0, 1]; a "
            "threshold is compared against a probability"
        )
    return score >= threshold


def accepts_interval_width(width: float | None, maximum: float | None) -> bool:
    """``width <= maximum``: the inclusive-boundary width rule.

    A missing interval is **never** treated as width zero, which would read
    as a perfectly certain prediction and would be accepted by every
    threshold.  It returns ``False`` and the caller records
    ``prediction_interval_unavailable``.
    """
    if width is None:
        return False
    if not math.isfinite(width) or width < 0.0:
        raise UncertaintyError(
            f"interval width {width!r} is not a finite non-negative number"
        )
    if maximum is None:
        return True
    # Zero is permitted *here* so a width sweep may include its strictest
    # endpoint, where only an exactly-zero-width interval would be accepted.
    # A zero configured as the run's operating policy is refused by
    # ``SelectivePredictionConfiguration`` instead, because a policy that
    # abstains on every window is a configuration mistake rather than a
    # curve endpoint.
    if not math.isfinite(maximum) or maximum < 0.0:
        raise UncertaintyError(
            f"maximum_interval_width {maximum!r} is not a finite non-negative number"
        )
    return width <= maximum


# ---------------------------------------------------------------------------
# Evidence gate — kept separate from model confidence
# ---------------------------------------------------------------------------


def evaluate_evidence_gate(
    *,
    configuration: EvidenceGateConfiguration,
    prediction_available: bool,
    available_modalities: Sequence[str],
    modality_quality: Mapping[str, float | None],
    probability_calibrated: bool | None,
) -> tuple[bool, tuple[AbstentionReason, ...]]:
    """Whether there was enough usable *measurement* to act on this window.

    This answers a different question from the confidence threshold, and
    the two are never combined arithmetically.  Signal quality is not
    multiplied into a model probability: no probabilistic model in this
    repository justifies treating a camera diagnostic as a likelihood term,
    so the two gate independently and each failure keeps its own reason
    code.

    Returns ``(passed, reasons)`` with reasons in canonical order.
    """
    reasons: set[AbstentionReason] = set()

    if configuration.require_prediction_available and not prediction_available:
        reasons.add(AbstentionReason.MODEL_PREDICTION_UNAVAILABLE)

    if configuration.enabled:
        present = {str(name) for name in available_modalities}
        if len(present) < configuration.minimum_available_modalities:
            reasons.add(AbstentionReason.INSUFFICIENT_MEASUREMENT_EVIDENCE)
        for modality in configuration.required_modalities:
            if modality.value not in present:
                reasons.add(AbstentionReason.REQUIRED_MODALITY_UNAVAILABLE)
                break

        gate = configuration.minimum_signal_quality
        if gate is not None:
            for name in sorted(present):
                value = modality_quality.get(name)
                if value is None:
                    # Absence of a recorded quality is not a low quality. It
                    # fails only when the configuration explicitly says an
                    # unmeasured quality is unacceptable evidence.
                    if configuration.treat_missing_quality_as_failure:
                        reasons.add(AbstentionReason.SIGNAL_QUALITY_BELOW_GATE)
                        break
                    continue
                if not math.isfinite(value) or value < gate:
                    reasons.add(AbstentionReason.SIGNAL_QUALITY_BELOW_GATE)
                    break

        if (
            configuration.require_probability_calibration_for_classification_confidence
            and probability_calibrated is False
        ):
            reasons.add(AbstentionReason.PROBABILITY_CALIBRATION_UNAVAILABLE)

    ordered = tuple(r for r in AbstentionReason if r in reasons)
    return (not ordered), ordered


# ---------------------------------------------------------------------------
# Split conformal prediction intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConformalFit:
    """The residual quantile of one fold, and how it was obtained."""

    available: bool
    quantile: float | None
    order_statistic: int | None
    sample_count: int
    alpha: float
    unavailable_reason: str | None = None


def absolute_residuals(
    y_true: Sequence[float], y_predicted: Sequence[float]
) -> np.ndarray:
    """``r_i = |y_i - yhat_i|`` over the calibration rows.

    Raises
    ------
    UncertaintyError
        If any truth or prediction is non-finite.  A residual computed from
        a non-finite prediction is not a residual, and silently dropping it
        would shrink the quantile's sample count without saying so.
    """
    truth = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_predicted), dtype=float)
    if truth.size != predicted.size:
        raise UncertaintyError(
            f"{truth.size} calibration label(s) against {predicted.size} "
            "prediction(s); residuals are undefined"
        )
    if truth.size and not np.isfinite(truth).all():
        raise UncertaintyError("a calibration label is not finite")
    if predicted.size and not np.isfinite(predicted).all():
        raise UncertaintyError(
            "a calibration prediction is not finite; a model that cannot "
            "produce a finite prediction must fail rather than emit one"
        )
    return np.asarray(np.abs(truth - predicted), dtype=float)


def conformal_order_statistic(sample_count: int, alpha: float) -> int:
    """``k = ceil((n + 1) * (1 - alpha))``, the 1-indexed order statistic."""
    if not 0.0 < alpha < 1.0:
        raise UncertaintyError(
            f"alpha must lie strictly between 0 and 1; got {alpha!r}. alpha is a "
            "nominal miscoverage rate, so alpha=0 would demand an infinite "
            "interval and alpha=1 an empty one."
        )
    if sample_count < 0:
        raise UncertaintyError("sample_count cannot be negative")
    return math.ceil((sample_count + 1) * (1.0 - alpha))


def minimum_conformal_samples(alpha: float) -> int:
    """Fewest calibration residuals for which ``k <= n`` can hold.

    ``k = ceil((n + 1)(1 - alpha)) <= n`` first holds at
    ``n = ceil(1 / alpha) - 1``.  Below it the finite-sample rule selects an
    order statistic that does not exist, which is the honest statement that
    the calibration set is too small — not a licence to widen the interval
    to infinity.
    """
    if not 0.0 < alpha < 1.0:
        raise UncertaintyError(
            f"alpha must lie strictly between 0 and 1; got {alpha!r}"
        )
    return math.ceil(1.0 / alpha) - 1


def fit_conformal_quantile(residuals: Vector, *, alpha: float) -> ConformalFit:
    """The split-conformal residual quantile, or a stated refusal.

    The exact convention is
    ``k = ceil((n + 1) * (1 - alpha))``; ``q`` is the ``k``-th smallest
    residual, 1-indexed.  When ``k > n`` no such order statistic exists and
    the fit is unavailable with a reason.

    Marginal coverage of at least ``1 - alpha`` holds when the calibration
    and test points are **exchangeable**.  Under grouped cross-validation
    the calibration and test rows come from *different people*, so
    exchangeability is an assumption about between-subject variation that
    this repository has never tested.  See
    ``docs/UNCERTAINTY_AND_ABSTENTION.md``.
    """
    values = np.asarray(list(residuals), dtype=float)
    n = int(values.size)
    if n and not np.isfinite(values).all():
        raise UncertaintyError("a calibration residual is not finite")
    if n and (values < 0.0).any():
        raise UncertaintyError("an absolute residual cannot be negative")

    k = conformal_order_statistic(n, alpha)
    if n == 0:
        return ConformalFit(
            available=False,
            quantile=None,
            order_statistic=None,
            sample_count=0,
            alpha=alpha,
            unavailable_reason=(
                "no calibration residual is available, so no conformal quantile "
                "exists. The interval is reported as unavailable rather than "
                "assumed to be zero-width."
            ),
        )
    if k > n:
        return ConformalFit(
            available=False,
            quantile=None,
            order_statistic=None,
            sample_count=n,
            alpha=alpha,
            unavailable_reason=(
                f"the finite-sample rule needs the {k}-th smallest of {n} "
                f"calibration residual(s) at alpha={alpha}, which does not "
                f"exist. At least {minimum_conformal_samples(alpha)} residual(s) "
                "are required. The interval is unavailable; it is never widened "
                "to infinity and never fabricated."
            ),
        )
    ordered = np.sort(values)
    return ConformalFit(
        available=True,
        quantile=float(ordered[k - 1]),
        order_statistic=k,
        sample_count=n,
        alpha=alpha,
    )


def conformal_interval(
    predicted_value: float, quantile: float
) -> tuple[float, float, float]:
    """``[yhat - q, yhat + q]`` and its width, all finite."""
    if not math.isfinite(predicted_value):
        raise UncertaintyError(
            "a non-finite point prediction has no interval; a model that cannot "
            "produce a finite prediction must fail rather than emit one"
        )
    if not math.isfinite(quantile) or quantile < 0.0:
        raise UncertaintyError(
            f"the conformal quantile {quantile!r} is not a finite non-negative number"
        )
    lower = predicted_value - quantile
    upper = predicted_value + quantile
    return float(lower), float(upper), float(upper - lower)


def project_interval_to_range(
    lower: float, upper: float, *, minimum: float, maximum: float
) -> tuple[float, float]:
    """Clip bounds to a target's declared range as a PRESENTATION step.

    The raw bounds remain the interval of record, and empirical interval
    coverage is always computed on them.  Clipping narrows an interval
    without any statistical justification for doing so, so a clipped
    interval must never be scored as though it were the conformal one.
    """
    if minimum >= maximum:
        raise UncertaintyError(
            f"the target range [{minimum!r}, {maximum!r}] is empty or inverted"
        )
    return float(max(lower, minimum)), float(min(upper, maximum))


def interval_contains(lower: float, upper: float, value: float) -> bool:
    """Whether ``value`` lies within ``[lower, upper]``, bounds inclusive."""
    return bool(lower <= value <= upper)


# ---------------------------------------------------------------------------
# Threshold estimation from permitted groups only
# ---------------------------------------------------------------------------


def select_population_threshold(
    *,
    scores: Sequence[float],
    correct: Sequence[bool],
    group_ids: Sequence[str],
    grid: Sequence[float],
    objective: ThresholdObjective,
    target: float,
    minimum_samples: int,
    minimum_groups: int,
    fold_index: int,
    calibration_group_ids: Sequence[str],
    outer_test_group_ids: Sequence[str],
) -> EstimatedThresholdRecord:
    """Choose a threshold from calibration rows only, or refuse.

    ``scores`` and ``correct`` must come from the fold's **calibration
    groups**, which are disjoint from both the rows that fitted the model
    and the outer-test rows.  No outer-test label participates, and the
    record says so in a field a reader can check.

    Tie-breaking: among grid points meeting the objective the **smallest**
    is chosen, which maximises coverage among admissible thresholds.  The
    grid is walked in ascending order, so the choice is deterministic.

    An objective that no grid point meets yields ``available=false`` with a
    reason.  No threshold is invented to satisfy an unreachable target, and
    no threshold is chosen because it makes a result look better.
    """
    overlap = set(calibration_group_ids) & set(outer_test_group_ids)
    if overlap:
        raise UncertaintyError(
            f"fold {fold_index}: threshold selection was handed group(s) "
            f"{sorted(overlap)} that are also outer-test groups. Selecting a "
            "threshold on the data it is reported against tunes the policy to "
            "its own evaluation."
        )

    score_values = np.asarray(list(scores), dtype=float)
    correct_values = np.asarray(list(correct), dtype=bool)
    if score_values.size != correct_values.size:
        raise UncertaintyError(
            f"fold {fold_index}: {score_values.size} score(s) against "
            f"{correct_values.size} outcome(s)"
        )
    if score_values.size and not np.isfinite(score_values).all():
        raise UncertaintyError(f"fold {fold_index}: a selection score is not finite")

    total = int(score_values.size)
    group_count = len(set(group_ids))
    base = {
        "fold_index": fold_index,
        "objective": objective,
        "objective_target": target,
        "search_grid": tuple(float(t) for t in grid),
        "calibration_group_ids": tuple(sorted(set(calibration_group_ids))),
        "calibration_sample_count": total,
        "calibration_group_count": group_count,
        "outer_test_group_ids": tuple(sorted(set(outer_test_group_ids))),
    }

    if total < minimum_samples:
        return EstimatedThresholdRecord(
            available=False,
            unavailable_reason=(
                f"threshold selection requires at least {minimum_samples} "
                f"calibration row(s); {total} are available. The configured "
                "population threshold is applied instead."
            ),
            **base,
        )
    if group_count < minimum_groups:
        return EstimatedThresholdRecord(
            available=False,
            unavailable_reason=(
                f"threshold selection requires at least {minimum_groups} "
                f"independent calibration group(s); {group_count} are available. "
                "A threshold chosen inside one person describes that person."
            ),
            **base,
        )

    best: tuple[float, float, float] | None = None
    for threshold in grid:
        accepted = score_values >= float(threshold)
        accepted_count = int(accepted.sum())
        coverage = accepted_count / total
        if accepted_count == 0:
            continue
        accuracy = float(correct_values[accepted].mean())
        if objective is ThresholdObjective.TARGET_ACCEPTED_ACCURACY:
            achieved, meets = accuracy, accuracy >= target
        elif objective is ThresholdObjective.TARGET_EMPIRICAL_RISK:
            risk = 1.0 - accuracy
            achieved, meets = risk, risk <= target
        else:
            achieved, meets = coverage, coverage >= target
        if not meets:
            continue
        if objective is ThresholdObjective.TARGET_COVERAGE:
            # Coverage falls as the threshold rises, so the LARGEST
            # admissible threshold is the selective one; ascending order
            # means the last match wins.
            best = (float(threshold), achieved, coverage)
        elif best is None:
            best = (float(threshold), achieved, coverage)

    if best is None:
        return EstimatedThresholdRecord(
            available=False,
            unavailable_reason=(
                f"no threshold on the configured grid reaches the objective "
                f"{objective.value!r} at target {target!r} on this fold's "
                "calibration groups. The configured population threshold is "
                "applied instead; a threshold is never invented to satisfy an "
                "unreachable target."
            ),
            **base,
        )

    threshold, achieved, coverage = best
    return EstimatedThresholdRecord(
        available=True,
        selected_threshold=threshold,
        achieved_value=achieved,
        achieved_coverage=coverage,
        **base,
    )


def personal_confidence_threshold(
    *,
    subject_id: str,
    fold_index: int,
    calibration_scores: Sequence[float],
    calibration_window_ids: Sequence[str],
    evaluation_window_ids: Sequence[str],
    population_threshold: float,
    configuration: SelectivePredictionConfiguration,
    calibration_start_utc: str | None = None,
    calibration_end_utc: str | None = None,
    evaluation_start_utc: str | None = None,
    temporal_order_verified: bool = False,
    unavailable_reason: str | None = None,
) -> PersonalThresholdRecord:
    """One subject's threshold from their EARLIER windows, or a fallback.

    The rule is deliberately small::

        tau_raw = quantile(calibration confidence, 1 - target_coverage)
        lambda  = n / (n + kappa)
        tau_s   = (1 - lambda) * tau_population + lambda * tau_raw

    and it consumes **no labels at all** — only the confidence scores the
    population model assigned to the subject's own calibration windows.  An
    evaluation label therefore cannot reach it by any path, which is a
    stronger statement than "we were careful not to pass one".

    The failure it addresses is real: a subject the model is uniformly less
    confident about would otherwise be abstained on entirely by a
    population threshold, which is a measurement artefact presented as a
    property of that person.  Shrinkage toward the population value keeps a
    thin calibration set from moving the threshold far.

    Below the configured minimum evidence — or when the caller supplies
    ``unavailable_reason`` because the subject had no valid temporal split —
    the population threshold is applied and the reason is recorded.
    """
    windows = tuple(str(w) for w in calibration_window_ids)
    evaluation = tuple(str(w) for w in evaluation_window_ids)
    scores = np.asarray(list(calibration_scores), dtype=float)
    n = int(scores.size)
    shared = {
        "subject_id": subject_id,
        "fold_index": fold_index,
        "population_threshold": population_threshold,
        "calibration_window_ids": windows,
        "evaluation_window_ids": evaluation,
        "calibration_sample_count": n,
        "calibration_start_utc": calibration_start_utc,
        "calibration_end_utc": calibration_end_utc,
        "evaluation_start_utc": evaluation_start_utc,
        "temporal_order_verified": temporal_order_verified,
        "target_coverage": configuration.personal_target_coverage,
    }

    def fallback(reason: str) -> PersonalThresholdRecord:
        return PersonalThresholdRecord(
            personalized_threshold=None,
            applied_threshold=population_threshold,
            threshold_source=(
                ThresholdSource.POPULATION_FALLBACK
                if configuration.personalized_thresholds_enabled
                else ThresholdSource.CONFIGURED_POPULATION
            ),
            personalization_applied=False,
            fallback_to_population=True,
            fallback_reason=reason,
            **shared,
        )

    if not configuration.personalized_thresholds_enabled:
        return fallback(
            "personalized confidence thresholds are disabled in this run; the "
            "population threshold applies to every subject"
        )
    if unavailable_reason:
        return fallback(unavailable_reason)
    if not temporal_order_verified:
        return fallback(
            f"subject {subject_id!r} has no verified calibration-before-"
            "evaluation ordering, so no window can be shown to precede the "
            "windows the threshold would be applied to"
        )
    minimum = configuration.minimum_personal_calibration_windows
    if n < minimum:
        return fallback(
            f"subject {subject_id!r} has {n} calibration window(s), fewer than "
            f"the {minimum} required for a personal threshold. The population "
            "threshold applies; no per-subject rule is fitted from evidence "
            "too thin to support one."
        )
    if not np.isfinite(scores).all():
        return fallback(
            f"subject {subject_id!r} has a non-finite calibration confidence "
            "score, so no quantile of their confidence distribution exists"
        )
    if not windows:
        return fallback(
            f"subject {subject_id!r} has calibration scores but no recorded "
            "window ids, so the threshold's provenance could not be audited"
        )

    quantile_position = 1.0 - configuration.personal_target_coverage
    raw = float(np.quantile(scores, quantile_position, method=PERSONAL_QUANTILE_METHOD))
    kappa = configuration.personal_shrinkage_constant
    shrinkage = n / (n + kappa)
    personalized = (1.0 - shrinkage) * population_threshold + shrinkage * raw
    personalized = float(min(1.0, max(0.0, personalized)))

    return PersonalThresholdRecord(
        personalized_threshold=personalized,
        applied_threshold=personalized,
        threshold_source=ThresholdSource.PERSONALIZED,
        personalization_applied=True,
        fallback_to_population=False,
        shrinkage=float(shrinkage),
        raw_personal_quantile=float(min(1.0, max(0.0, raw))),
        **shared,
    )


# ---------------------------------------------------------------------------
# Selective metrics and coverage curves
# ---------------------------------------------------------------------------


def coverage_point(
    *,
    threshold: float | None,
    accepted_count: int,
    abstained_count: int,
    unavailable_count: int,
    axis: CoverageAxis = CoverageAxis.CONFIDENCE_THRESHOLD,
    threshold_unavailable_reason: str | None = None,
) -> CoveragePoint:
    """Counts and rates at one axis value, with an explicit denominator.

    The denominator is **every evaluated window**, including those for
    which no prediction existed.  Excluding them would let a run raise its
    reported coverage by producing fewer predictions.

    ``axis`` records what the value is a threshold *on*, because the two
    axes move in opposite directions and an unlabelled number could be read
    either way.
    """
    total = accepted_count + abstained_count + unavailable_count
    return CoveragePoint(
        axis=axis,
        threshold=None if threshold is None else float(threshold),
        threshold_unavailable_reason=threshold_unavailable_reason,
        total_window_count=total,
        accepted_count=accepted_count,
        abstained_count=abstained_count,
        unavailable_count=unavailable_count,
        coverage=(accepted_count / total) if total else 0.0,
        abstention_rate=(abstained_count / total) if total else 0.0,
    )


def selective_classification_metrics(
    *,
    threshold: float,
    y_true: Sequence[str],
    y_predicted: Sequence[str],
    probabilities: Sequence[Sequence[float]] | None,
    labels: Sequence[str],
    group_ids: Sequence[str],
    accepted: Sequence[bool],
    unavailable: Sequence[bool],
    calibration_bins: int = 10,
) -> SelectiveMetrics:
    """Classification performance over accepted rows, beside its coverage.

    Abstained rows are **not** scored.  They are not counted as errors,
    they do not become a class, and they do not enter any confusion matrix.
    Their number is reported so an accepted-set score cannot be read as a
    whole-set score.
    """
    from engagevr.training.metrics import calibration_metrics

    accepted_mask = np.asarray(list(accepted), dtype=bool)
    unavailable_mask = np.asarray(list(unavailable), dtype=bool)
    if (accepted_mask & unavailable_mask).any():
        raise UncertaintyError(
            "a window cannot be both accepted and have no available prediction"
        )
    total = int(accepted_mask.size)
    accepted_count = int(accepted_mask.sum())
    unavailable_count = int(unavailable_mask.sum())
    abstained_count = total - accepted_count - unavailable_count

    point = coverage_point(
        threshold=threshold,
        accepted_count=accepted_count,
        abstained_count=abstained_count,
        unavailable_count=unavailable_count,
        axis=CoverageAxis.CONFIDENCE_THRESHOLD,
    )
    unavailable_metrics: dict[str, str] = {}

    if accepted_count == 0:
        unavailable_metrics["accepted_metrics"] = (
            f"no window was accepted at threshold {threshold!r}, so every "
            "accepted-set metric is undefined. It is reported as unavailable "
            "rather than as zero, which is a legitimate score."
        )
        return SelectiveMetrics(
            axis=CoverageAxis.CONFIDENCE_THRESHOLD,
            threshold=float(threshold),
            coverage_point=point,
            unavailable_metrics=unavailable_metrics,
        )

    rows = np.flatnonzero(accepted_mask)
    truth = [str(y_true[int(i)]) for i in rows]
    predicted = [str(y_predicted[int(i)]) for i in rows]
    groups = [str(group_ids[int(i)]) for i in rows]

    calibration: list[CalibrationMetrics] = []
    if probabilities is not None:
        matrix = np.asarray([list(probabilities[int(i)]) for i in rows], dtype=float)
        calibration.append(
            calibration_metrics(
                label="accepted",
                probabilities=matrix,
                y_true=truth,
                labels=list(labels),
                bin_count=calibration_bins,
            )
        )

    metrics = classification_metrics(
        y_true=truth,
        y_predicted=predicted,
        labels=list(labels),
        group_ids=groups,
        calibration=tuple(calibration),
    )
    risk = None if metrics.accuracy is None else float(1.0 - metrics.accuracy)
    if risk is None:
        unavailable_metrics["empirical_risk"] = (
            "empirical risk is 1 - accepted accuracy, and accuracy is undefined"
        )
    return SelectiveMetrics(
        axis=CoverageAxis.CONFIDENCE_THRESHOLD,
        threshold=float(threshold),
        coverage_point=point,
        accepted_classification=metrics,
        empirical_risk=risk,
        accepted_class_support=dict(metrics.class_support),
        unavailable_metrics=unavailable_metrics,
    )


def selective_regression_metrics(
    *,
    maximum_interval_width: float | None,
    y_true: Sequence[float],
    y_predicted: Sequence[float],
    group_ids: Sequence[str],
    accepted: Sequence[bool],
    unavailable: Sequence[bool],
    interval_lower: Sequence[float | None],
    interval_upper: Sequence[float | None],
    interval_width: Sequence[float | None],
    no_maximum_reason: str | None = None,
) -> SelectiveMetrics:
    """Regression performance over accepted rows, beside its coverage.

    The axis value here is a **maximum interval width in the target's own
    units**, never a confidence score.  ``None`` means no width policy was
    applied at this point, and it is recorded as an absent threshold with a
    stated reason rather than as ``0.0``, which would mean "abstain unless
    the interval has zero width".

    Empirical interval coverage is computed on the **raw** conformal
    bounds, and only over accepted rows that actually carry an interval. A
    row with no interval contributes to neither the coverage numerator nor
    its denominator, because a missing interval is not a failed one.
    """
    if maximum_interval_width is None and not no_maximum_reason:
        raise UncertaintyError(
            "a regression selective point with no maximum_interval_width must "
            "state why it has none; an absent width policy is not a maximum "
            "width of zero"
        )
    if maximum_interval_width is not None and (
        not math.isfinite(maximum_interval_width) or maximum_interval_width < 0.0
    ):
        raise UncertaintyError(
            f"maximum_interval_width {maximum_interval_width!r} is not a finite "
            "non-negative width"
        )
    threshold = maximum_interval_width
    accepted_mask = np.asarray(list(accepted), dtype=bool)
    unavailable_mask = np.asarray(list(unavailable), dtype=bool)
    if (accepted_mask & unavailable_mask).any():
        raise UncertaintyError(
            "a window cannot be both accepted and have no available prediction"
        )
    total = int(accepted_mask.size)
    accepted_count = int(accepted_mask.sum())
    unavailable_count = int(unavailable_mask.sum())
    abstained_count = total - accepted_count - unavailable_count

    point = coverage_point(
        threshold=threshold,
        accepted_count=accepted_count,
        abstained_count=abstained_count,
        unavailable_count=unavailable_count,
        axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
        threshold_unavailable_reason=(
            None if threshold is not None else no_maximum_reason
        ),
    )
    unavailable_metrics: dict[str, str] = {}

    if accepted_count == 0:
        unavailable_metrics["accepted_metrics"] = (
            f"no window was accepted at maximum interval width {threshold!r}, "
            "so every accepted-set metric is undefined. It is reported as "
            "unavailable rather than as zero."
        )
        return SelectiveMetrics(
            axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
            threshold=threshold,
            coverage_point=point,
            unavailable_metrics=unavailable_metrics,
        )

    rows = [int(i) for i in np.flatnonzero(accepted_mask)]
    truth = [float(y_true[i]) for i in rows]
    predicted = [float(y_predicted[i]) for i in rows]
    groups = [str(group_ids[i]) for i in rows]
    metrics = regression_metrics(y_true=truth, y_predicted=predicted, group_ids=groups)

    widths: list[float] = []
    covered: list[bool] = []
    for i in rows:
        width = interval_width[i]
        if width is not None:
            widths.append(float(width))
        low, high = interval_lower[i], interval_upper[i]
        if low is not None and high is not None:
            covered.append(interval_contains(float(low), float(high), float(y_true[i])))
    empirical = float(np.mean(covered)) if covered else None
    if empirical is None:
        unavailable_metrics["empirical_interval_coverage"] = (
            "no accepted window carries an interval, so empirical interval "
            "coverage is undefined"
        )
    return SelectiveMetrics(
        axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
        threshold=threshold,
        coverage_point=point,
        accepted_regression=metrics,
        empirical_interval_coverage=empirical,
        mean_interval_width=(float(np.mean(widths)) if widths else None),
        median_interval_width=(float(np.median(widths)) if widths else None),
        unavailable_metrics=unavailable_metrics,
    )


def risk_coverage_points(
    points: Sequence[SelectiveMetrics],
) -> tuple[RiskCoveragePoint, ...]:
    """Extract the risk-coverage representation from selective metrics."""
    records: list[RiskCoveragePoint] = []
    for point in points:
        reason: str | None = None
        if point.empirical_risk is None:
            reason = (
                point.unavailable_metrics.get("accepted_metrics")
                or point.unavailable_metrics.get("empirical_risk")
                or (
                    "empirical risk is 1 - accepted accuracy, which is undefined "
                    "at this threshold"
                )
            )
        records.append(
            RiskCoveragePoint(
                axis=point.axis,
                threshold=point.threshold,
                coverage=point.coverage_point.coverage,
                empirical_risk=point.empirical_risk,
                accepted_count=point.coverage_point.accepted_count,
                unavailable_reason=reason,
            )
        )
    return tuple(records)


def area_under_risk_coverage(
    points: Sequence[RiskCoveragePoint],
) -> tuple[float | None, str | None]:
    """Trapezoidal mean risk against coverage, or a stated refusal.

    Points are sorted by **ascending coverage**, the trapezoidal rule is
    applied, and the area is divided by the covered coverage span so the
    result reads as a mean risk rather than an unnormalised area whose
    magnitude depends on how much of the coverage axis the grid happened to
    span.

    This is descriptive only.  A lower value computed on synthetic data
    establishes nothing about safety, superiority, or usefulness.
    """
    usable = [p for p in points if p.empirical_risk is not None]
    if len(usable) < 2:
        return None, (
            f"the area under the risk-coverage curve needs at least two points "
            f"with a defined risk; {len(usable)} are available"
        )
    ordered = sorted(usable, key=lambda p: (p.coverage, p.threshold or 0.0))
    coverage = np.asarray([p.coverage for p in ordered], dtype=float)
    risk = np.asarray([float(p.empirical_risk or 0.0) for p in ordered], dtype=float)
    span = float(coverage[-1] - coverage[0])
    if span <= 0.0:
        return None, (
            "every point with a defined risk sits at the same coverage, so the "
            "curve has zero span and no area under it"
        )
    area = float(np.trapezoid(risk, coverage))
    return float(area / span), None


def coverage_is_monotonic(
    points: Sequence[CoveragePoint],
    *,
    direction: MonotonicDirection,
) -> bool:
    """Whether coverage moves only in ``direction`` as the axis value rises.

    The required direction depends on what the axis *is*, and the two axes
    disagree:

    * ``confidence_threshold`` — non-increasing.  Raising an inclusive
      confidence threshold can only remove windows from the accepted set.
    * ``maximum_interval_width`` — non-decreasing.  Raising the widest
      acceptable interval can only admit windows, because the rule is
      ``accept if interval_width <= W_max`` and a wide interval is the
      uncertain case.

    A ``False`` here means the curve was not built from one shared set of
    predictions, or was swept in the wrong direction.  Either is a bug
    rather than a finding.

    Raises
    ------
    UncertaintyError
        If any point has no axis value, or the points mix axes; neither can
        be ordered into a curve.
    """
    if not points:
        return True
    axes = {point.axis for point in points}
    if len(axes) > 1:
        raise UncertaintyError(
            "coverage monotonicity is undefined over a mixture of axes "
            f"({sorted(a.value for a in axes)}): the two move in opposite "
            "directions"
        )
    axis = next(iter(axes))
    expected = COVERAGE_AXIS_DIRECTION[axis]
    if direction is not expected:
        raise UncertaintyError(
            f"the {axis.value!r} axis is {expected.value}, so it cannot be "
            f"checked for being {direction.value}"
        )
    values = [point.threshold for point in points]
    if any(value is None for value in values):
        raise UncertaintyError(
            "a coverage curve point has no axis value, so the points cannot be "
            "ordered; an operating point is not a curve"
        )
    ordered = sorted(points, key=lambda p: p.threshold or 0.0)
    if direction is MonotonicDirection.NON_INCREASING:
        return all(
            ordered[index].coverage >= ordered[index + 1].coverage - 1e-12
            for index in range(len(ordered) - 1)
        )
    return all(
        ordered[index].coverage <= ordered[index + 1].coverage + 1e-12
        for index in range(len(ordered) - 1)
    )


def coverage_axis_for(task_type: TaskType) -> CoverageAxis:
    """The one coverage axis a task type is selective on."""
    return COVERAGE_AXIS_FOR_TASK[task_type]


def expected_monotonic_direction(axis: CoverageAxis) -> MonotonicDirection:
    """The direction coverage must move in as ``axis`` increases."""
    return COVERAGE_AXIS_DIRECTION[axis]


def reason_counts(
    reasons: Sequence[Sequence[AbstentionReason]],
) -> dict[str, int]:
    """Count each abstention reason, in canonical order."""
    counts: dict[str, int] = {}
    for reason in AbstentionReason:
        total = sum(1 for group in reasons if reason in set(group))
        if total:
            counts[reason.value] = total
    return counts


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------


def build_uncertainty_run_id(
    *,
    target_name: str,
    task_type: str,
    evaluation_mode: str,
    dataset_fingerprint: str,
    split_manifest_fingerprint: str,
    random_seed: int,
    configuration: SelectivePredictionConfiguration,
    calibration_method: str,
    engagevr_version: str,
) -> str:
    """Deterministic identifier over everything that defines the run.

    No wall clock, no output directory, and no random component
    participates, so re-running one configuration reproduces the identifier
    rather than accumulating near-duplicate directories.
    """
    gate = configuration.evidence_gate
    payload = {
        "target_name": target_name,
        "task_type": task_type,
        "evaluation_mode": evaluation_mode,
        "dataset_fingerprint": dataset_fingerprint,
        "split_manifest_fingerprint": split_manifest_fingerprint,
        "random_seed": random_seed,
        "prediction_source": configuration.prediction_source.value,
        "modalities": sorted(m.value for m in configuration.modalities),
        "model_classification": configuration.model_classification,
        "model_regression": configuration.model_regression,
        "probability_calibration_method": calibration_method,
        "confidence_source": configuration.confidence_source.value,
        "confidence_definition": configuration.confidence_equation,
        "population_confidence_threshold": (
            configuration.population_confidence_threshold
        ),
        "threshold_grid": list(configuration.threshold_grid),
        "estimate_population_threshold": configuration.estimate_population_threshold,
        "threshold_objective": configuration.threshold_objective.value,
        "threshold_objective_target": configuration.threshold_objective_target,
        "minimum_threshold_selection_samples": (
            configuration.minimum_threshold_selection_samples
        ),
        "minimum_threshold_selection_groups": (
            configuration.minimum_threshold_selection_groups
        ),
        "personalized_thresholds_enabled": (
            configuration.personalized_thresholds_enabled
        ),
        "personal_calibration_windows": configuration.personal_calibration_windows,
        "minimum_personal_calibration_windows": (
            configuration.minimum_personal_calibration_windows
        ),
        "minimum_evaluation_windows": configuration.minimum_evaluation_windows,
        "personal_target_coverage": configuration.personal_target_coverage,
        "personal_shrinkage_constant": configuration.personal_shrinkage_constant,
        "personal_quantile_method": PERSONAL_QUANTILE_METHOD,
        "interval_method": configuration.interval_method.value,
        "alpha": configuration.alpha,
        "maximum_interval_width": configuration.maximum_interval_width,
        "interval_width_grid": (
            None
            if configuration.interval_width_grid is None
            else list(configuration.interval_width_grid)
        ),
        "clip_interval_to_target_range": configuration.clip_interval_to_target_range,
        "evidence_gate": {
            "enabled": gate.enabled,
            "require_prediction_available": gate.require_prediction_available,
            "require_probability_calibration": (
                gate.require_probability_calibration_for_classification_confidence
            ),
            "minimum_available_modalities": gate.minimum_available_modalities,
            "required_modalities": sorted(m.value for m in gate.required_modalities),
            "minimum_signal_quality": gate.minimum_signal_quality,
            "treat_missing_quality_as_failure": (gate.treat_missing_quality_as_failure),
        },
        "adaptation_gate_enabled": configuration.adaptation_gate_enabled,
        "engagevr_version": engagevr_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    short_mode = "sci" if evaluation_mode == "scientific" else "selfcheck"
    return f"{target_name}-uncertainty-{short_mode}-{digest}"


def acceptance_rule_for(task_type: TaskType) -> str:
    """The recorded acceptance rule text for a task type."""
    if task_type is TaskType.CLASSIFICATION:
        return CLASSIFICATION_ACCEPTANCE_RULE
    return REGRESSION_ACCEPTANCE_RULE


__all__ = [
    "PERSONAL_QUANTILE_METHOD",
    "ClassificationMetrics",
    "ConfidenceComponents",
    "ConformalFit",
    "RegressionMetrics",
    "UncertaintyError",
    "absolute_residuals",
    "acceptance_rule_for",
    "accepts_at_threshold",
    "accepts_interval_width",
    "area_under_risk_coverage",
    "assert_probability_vector",
    "build_uncertainty_run_id",
    "confidence_components",
    "confidence_method",
    "conformal_interval",
    "conformal_order_statistic",
    "coverage_axis_for",
    "coverage_is_monotonic",
    "coverage_point",
    "evaluate_evidence_gate",
    "expected_monotonic_direction",
    "fit_conformal_quantile",
    "interval_contains",
    "minimum_conformal_samples",
    "normalized_entropy",
    "personal_confidence_threshold",
    "prediction_margin",
    "predictive_entropy",
    "project_interval_to_range",
    "reason_counts",
    "risk_coverage_points",
    "select_population_threshold",
    "selective_classification_metrics",
    "selective_regression_metrics",
]
