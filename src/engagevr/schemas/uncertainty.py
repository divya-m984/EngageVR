"""Uncertainty, selective-prediction, and abstention schemas (Milestone 7).

These models are the persisted form of an uncertainty-aware run.  They are
written as JSON beside the run's artifacts so a selective result can be
inspected, and its provenance judged, **without loading any model file**.

What this module encodes structurally rather than by convention
---------------------------------------------------------------
1. *Five distinct things are never collapsed into one number.*  Signal
   quality, predicted probability, probability calibration status, model
   confidence, and ensemble disagreement each occupy their own field.
   There is no field named merely ``uncertainty`` anywhere in this module,
   because such a field would make the five indistinguishable to a reader.
2. *A maximum probability is only called confidence when it was
   calibrated.*  :class:`ClassificationConfidence` refuses to populate
   ``confidence_score`` unless ``probability_calibration_status`` is
   :attr:`ProbabilityCalibrationStatus.CALIBRATED`; an uncalibrated vector
   yields ``selection_score`` instead, under an explicitly named
   uncalibrated policy.
3. *Abstention is a decision, not a failure and not a missing value.*
   :class:`AbstentionReason` keeps the model's own refusal
   (``below_confidence_threshold``, ``interval_too_wide``) distinct from
   missing evidence (``required_modality_unavailable``), from unusable
   evidence (``signal_quality_below_gate``), and from an absent prediction
   (``model_prediction_unavailable``).
4. *An abstained window keeps its original prediction.*  Every decision
   record carries the prediction it declined to act on, so abstention is an
   additional layer and never a destructive mutation.  Nothing becomes a
   class it was not, and nothing becomes zero.
5. *A regression prediction has an interval, never a probability.*
   :class:`RegressionPredictionInterval` carries bounds, a width, a named
   method, its calibration provenance, and the nominal miscoverage it was
   built for.  There is no confidence field on it.
6. *Synthetic can never be scientifically eligible*, and a self-check
   document must carry the self-check banner — the same validator contract
   as :mod:`engagevr.schemas.experiments`, :mod:`engagevr.schemas.fusion`,
   and :mod:`engagevr.schemas.personalization`.

The adaptation *gate* modelled here answers only "may an already-chosen
action be acted upon?".  It never chooses an action, never ranks actions,
and never learns.  Adaptation policy is Milestone 8.
"""

from __future__ import annotations

import enum
import math
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from engagevr.schemas.experiments import (
    SOFTWARE_SELF_CHECK_BANNER,
    ClassificationMetrics,
    EvaluationMode,
    RegressionMetrics,
)
from engagevr.schemas.fusion import PROBABILITY_SUM_TOLERANCE, FusionModality
from engagevr.schemas.targets import TaskType

#: Note attached to every uncertainty document.
UNCERTAINTY_NOTE = (
    "Confidence, signal quality, probability calibration, and ensemble "
    "disagreement are FOUR DIFFERENT THINGS and are stored in four "
    "different fields. A confidence score describes the model's predicted "
    "probability; signal quality describes the measurement; calibration "
    "status describes whether the probability was fitted against observed "
    "outcomes; disagreement describes how far modality experts differed. "
    "None of them is safety, none is validated, and none is evidence about "
    "any person."
)

#: Note attached to every selective-prediction document.
SELECTIVE_PREDICTION_NOTE = (
    "Coverage and accepted-set performance are reported TOGETHER and never "
    "separately. An accepted-set score computed over a subset of windows is "
    "not comparable to a score over all of them unless the reader can see "
    "what fraction was accepted. A coverage curve computed on SYNTHETIC "
    "data describes the generator this repository wrote; it is not evidence "
    "of real-world calibration, reliability, safety, or usefulness."
)

#: Note distinguishing an abstention from every other kind of no-answer.
ABSTENTION_MEANING_NOTE = (
    "An ABSTENTION is a deliberate decision not to emit an actionable "
    "estimate because a declared selective-prediction rule was not "
    "satisfied. It is NOT a modality being unavailable, NOT a feature being "
    "missing, NOT a model being unavailable, NOT calibration being "
    "unavailable, and NOT a runtime error. Each of those has its own reason "
    "code, and an abstained window keeps the prediction it declined to act "
    "on rather than losing it."
)

#: Note attached to every quality field carried beside a confidence record.
QUALITY_IS_NOT_CONFIDENCE_NOTE = (
    "Signal quality describes the MEASUREMENT, never the person. Low signal "
    "quality is never low engagement and never low cognitive load. Quality "
    "is not multiplied into model confidence: the two gate an actionable "
    "estimate independently and carry separate reason codes."
)

#: Note attached to every disagreement reference carried beside a record.
DISAGREEMENT_IS_NOT_UNCERTAINTY_NOTE = (
    "Ensemble disagreement is the Milestone 6 ENSEMBLE-DISAGREEMENT "
    "DIAGNOSTIC carried here unchanged as supporting evidence. It is not a "
    "calibrated uncertainty estimate, it is not model confidence, it is not "
    "signal quality, and it does not by itself trigger abstention."
)

#: The exact predictive-entropy definition, recorded on every record.
ENTROPY_EQUATION = (
    "H(p) = -sum_c p_c * log(p_c), natural logarithm (base e), with the "
    "convention 0 * log(0) = 0. Reported in nats. A normalised variant "
    "H(p) / log(K) over K classes is reported separately as "
    "'normalized_entropy' and lies in [0, 1]."
)

#: The exact top-two margin definition, recorded on every record.
MARGIN_EQUATION = (
    "margin = p_(1) - p_(2), the largest predicted class probability minus "
    "the second largest. It is a RANKING DIAGNOSTIC, not calibrated "
    "confidence, unless a calibration contract has been separately "
    "satisfied and stated."
)

#: The exact classification confidence definition.
CONFIDENCE_EQUATION = (
    "confidence = max_c p_calibrated(c | x), the maximum CALIBRATED class "
    "probability, together with the class that attained it. It is not "
    "epistemic uncertainty, not certainty, not psychological confidence, "
    "and not safety."
)

#: The exact selective-classification acceptance rule.
CLASSIFICATION_ACCEPTANCE_RULE = (
    "accept if score >= tau, abstain if score < tau. The boundary is "
    "INCLUSIVE: a score exactly equal to tau is accepted."
)

#: The exact split-conformal interval construction.
CONFORMAL_EQUATION = (
    "Split conformal absolute-residual interval. Calibration residuals "
    "r_i = |y_i - yhat_i| are computed on calibration rows disjoint from "
    "both the rows that fitted the model and the outer-test rows. For "
    "nominal miscoverage alpha and n calibration residuals, "
    "k = ceil((n + 1) * (1 - alpha)) and q = the k-th smallest residual "
    "(1-indexed order statistic). The interval is "
    "[yhat(x) - q, yhat(x) + q]. When k > n the finite-sample rule cannot "
    "be met and the interval is UNAVAILABLE; it is never widened to "
    "infinity and never fabricated."
)

#: The exact selective-regression acceptance rule.
REGRESSION_ACCEPTANCE_RULE = (
    "accept if interval_width <= maximum_interval_width, abstain otherwise. "
    "The boundary is INCLUSIVE: a width exactly equal to the maximum is "
    "accepted. A missing interval is never treated as width zero."
)

#: The exact coverage definition used by every selective metric.
COVERAGE_EQUATION = (
    "coverage = accepted_count / total_window_count, where "
    "total_window_count is every evaluated window in scope, including those "
    "for which no model prediction was available. "
    "accepted + abstained + unavailable = total, so the three reconcile "
    "exactly. abstention_rate = abstained_count / total_window_count."
)

#: The exact empirical-risk definition used by the risk-coverage curve.
RISK_EQUATION = (
    "empirical_risk = 1 - accepted_accuracy, the misclassification rate "
    "over ACCEPTED windows only. It is an empirical software quantity under "
    "this definition; it is not a bound, not a guarantee, and not a "
    "statement about any person."
)

#: The exact area-under-risk-coverage integration rule.
AURC_EQUATION = (
    "AURC = trapezoidal integral of empirical_risk against coverage, over "
    "points sorted by ascending coverage, divided by the covered coverage "
    "span so the result is a mean risk rather than an unnormalised area. "
    "At least two points with a defined risk and distinct coverage are "
    "required; otherwise it is unavailable. It is a DESCRIPTIVE software "
    "metric: a lower value on synthetic data establishes nothing about "
    "safety, superiority, or usefulness."
)


class UncertaintyMethod(enum.StrEnum):
    """How an uncertainty representation was produced.

    Each member names a *construction*, not a claim.  ``COLD_START`` is
    absent: there is no cold start here, because a window with no usable
    evidence produces an explicit abstention with a reason rather than a
    degraded estimate.
    """

    #: Maximum calibrated class probability. Classification only.
    MAX_CALIBRATED_PROBABILITY = "max_calibrated_probability"
    #: Maximum uncalibrated class probability, explicitly named as such.
    MAX_UNCALIBRATED_PROBABILITY = "max_uncalibrated_probability"
    #: Split conformal absolute-residual interval. Regression only.
    SPLIT_CONFORMAL_ABSOLUTE_RESIDUAL = "split_conformal_absolute_residual"


#: Classification confidence sources.
CLASSIFICATION_METHODS: frozenset[UncertaintyMethod] = frozenset(
    {
        UncertaintyMethod.MAX_CALIBRATED_PROBABILITY,
        UncertaintyMethod.MAX_UNCALIBRATED_PROBABILITY,
    }
)

#: Regression interval sources.
REGRESSION_METHODS: frozenset[UncertaintyMethod] = frozenset(
    {UncertaintyMethod.SPLIT_CONFORMAL_ABSOLUTE_RESIDUAL}
)


class ProbabilityCalibrationStatus(enum.StrEnum):
    """Whether a probability vector satisfied the calibration contract.

    The contract is the Milestone 5 one, unchanged: a calibrator was fitted
    on calibration groups drawn from the training groups and disjoint from
    both the fit groups and the outer-test groups, and it produced an
    estimator.  Anything else is ``UNCALIBRATED`` or ``UNAVAILABLE``, and
    neither may be described as calibrated confidence.
    """

    #: A calibrator was fitted on disjoint groups and produced this vector.
    CALIBRATED = "calibrated"
    #: A probability vector exists but no calibrator was fitted for it.
    UNCALIBRATED = "uncalibrated"
    #: No probability vector exists at all.
    UNAVAILABLE = "unavailable"


class PredictionSource(enum.StrEnum):
    """Which model produced the prediction an uncertainty record describes.

    Only sources whose probabilities pass through the Milestone 5
    calibration step are offered.  A *late*-fusion fused probability vector
    is deliberately absent: Milestone 6 calibrates each expert and the
    early-fusion estimator but never the fused vector itself, so calling a
    fused maximum "calibrated confidence" would be false.  See
    ``docs/UNCERTAINTY_AND_ABSTENTION.md``.
    """

    #: A single Milestone 5 baseline estimator over all permitted features.
    BASELINE_MODEL = "baseline_model"
    #: The Milestone 6 early-fusion estimator over the configured modalities.
    EARLY_FUSION = "early_fusion"


class ThresholdSource(enum.StrEnum):
    """Where an applied threshold came from.

    Provenance is required on every decision: a threshold read from
    configuration and a threshold estimated from data are different kinds
    of object, and a reader must be able to tell which was applied.
    """

    #: The configured population threshold, an engineering default.
    CONFIGURED_POPULATION = "configured_population"
    #: Estimated from inner calibration groups against a stated objective.
    ESTIMATED_POPULATION = "estimated_population"
    #: Derived from one subject's earlier calibration windows.
    PERSONALIZED = "personalized"
    #: Personalization was requested but fell back to a population value.
    POPULATION_FALLBACK = "population_fallback"


class ThresholdObjective(enum.StrEnum):
    """What an estimated population threshold was chosen to achieve."""

    #: Smallest threshold whose accepted accuracy reaches the target.
    TARGET_ACCEPTED_ACCURACY = "target_accepted_accuracy"
    #: Smallest threshold whose empirical risk falls to the target.
    TARGET_EMPIRICAL_RISK = "target_empirical_risk"
    #: Largest threshold whose coverage still reaches the target.
    TARGET_COVERAGE = "target_coverage"


class CoverageAxis(enum.StrEnum):
    """What the x-axis of a coverage curve actually measures.

    The two axes are **not** interchangeable, and neither is convertible
    into the other.  A confidence threshold is compared against a
    probability and rises toward stricter; an interval-width maximum is
    compared against a width in the target's own units and rises toward
    *less* strict.  Recording which axis a curve was swept over is what
    keeps a reader from reading one direction as the other.
    """

    #: ``accept if score >= tau``. Classification only. Units: probability.
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    #: ``accept if interval_width <= W_max``. Regression only. Units: target.
    MAXIMUM_INTERVAL_WIDTH = "maximum_interval_width"


class MonotonicDirection(enum.StrEnum):
    """The direction coverage must move in as an axis value increases."""

    #: Coverage may only fall: raising the value is stricter.
    NON_INCREASING = "non_increasing"
    #: Coverage may only rise: raising the value is more permissive.
    NON_DECREASING = "non_decreasing"


#: Units of each coverage axis, recorded on every curve.
COVERAGE_AXIS_UNITS: dict[CoverageAxis, str] = {
    CoverageAxis.CONFIDENCE_THRESHOLD: (
        "probability in [0, 1]: the axis value is compared against a "
        "predicted class probability"
    ),
    CoverageAxis.MAXIMUM_INTERVAL_WIDTH: (
        "the regression target's own units: the axis value is compared "
        "against a prediction-interval width. It is NOT a probability, is "
        "not confined to [0, 1], and is not convertible into a confidence "
        "score"
    ),
}

#: The direction coverage must move in on each axis, and why.
COVERAGE_AXIS_DIRECTION: dict[CoverageAxis, MonotonicDirection] = {
    CoverageAxis.CONFIDENCE_THRESHOLD: MonotonicDirection.NON_INCREASING,
    CoverageAxis.MAXIMUM_INTERVAL_WIDTH: MonotonicDirection.NON_DECREASING,
}

#: Which axis belongs to which task type. There is no third option.
COVERAGE_AXIS_FOR_TASK: dict[TaskType, CoverageAxis] = {
    TaskType.CLASSIFICATION: CoverageAxis.CONFIDENCE_THRESHOLD,
    TaskType.REGRESSION: CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
}

#: The monotonicity contract of each axis, recorded on every curve.
COVERAGE_MONOTONICITY_RULE: dict[CoverageAxis, str] = {
    CoverageAxis.CONFIDENCE_THRESHOLD: (
        "coverage[i + 1] <= coverage[i] as tau increases. Raising an "
        "inclusive confidence threshold can only remove windows from the "
        "accepted set."
    ),
    CoverageAxis.MAXIMUM_INTERVAL_WIDTH: (
        "coverage[i + 1] >= coverage[i] as W_max increases. Raising the "
        "widest acceptable interval can only admit windows to the accepted "
        "set. This is the OPPOSITE direction from the classification axis, "
        "because a wide interval is the uncertain case."
    ),
}


class AbstentionReason(enum.StrEnum):
    """Why an actionable estimate was withheld.

    These are deliberately **not** interchangeable.  A window blocked
    because the camera signal was poor is a different event from a window
    blocked because the model was unsure, and conflating them would let a
    measurement problem be read as a statement about a person.

    Declaration order is the canonical reporting order used by
    :meth:`AbstentionDecision.primary_reason`: evidence problems are
    reported before model-confidence problems, because an estimate built on
    absent evidence should not be discussed in terms of its confidence.
    """

    #: No model prediction exists for this window at all.
    MODEL_PREDICTION_UNAVAILABLE = "model_prediction_unavailable"
    #: Fewer measurement modalities contributed than the gate requires.
    INSUFFICIENT_MEASUREMENT_EVIDENCE = "insufficient_measurement_evidence"
    #: A modality the gate requires produced nothing for this window.
    REQUIRED_MODALITY_UNAVAILABLE = "required_modality_unavailable"
    #: Recorded signal quality fell below the configured gate.
    SIGNAL_QUALITY_BELOW_GATE = "signal_quality_below_gate"
    #: No calibrated probability exists and the policy requires one.
    PROBABILITY_CALIBRATION_UNAVAILABLE = "probability_calibration_unavailable"
    #: No prediction interval could be constructed for this window.
    PREDICTION_INTERVAL_UNAVAILABLE = "prediction_interval_unavailable"
    #: The confidence score fell below the applied threshold.
    BELOW_CONFIDENCE_THRESHOLD = "below_confidence_threshold"
    #: The prediction interval was wider than the configured maximum.
    INTERVAL_TOO_WIDE = "interval_too_wide"


#: Reasons describing the EVIDENCE rather than the model's confidence.
EVIDENCE_REASONS: tuple[AbstentionReason, ...] = (
    AbstentionReason.INSUFFICIENT_MEASUREMENT_EVIDENCE,
    AbstentionReason.REQUIRED_MODALITY_UNAVAILABLE,
    AbstentionReason.SIGNAL_QUALITY_BELOW_GATE,
)

#: Reasons describing the MODEL rather than the measurement.
MODEL_REASONS: tuple[AbstentionReason, ...] = (
    AbstentionReason.MODEL_PREDICTION_UNAVAILABLE,
    AbstentionReason.PROBABILITY_CALIBRATION_UNAVAILABLE,
    AbstentionReason.PREDICTION_INTERVAL_UNAVAILABLE,
    AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,
    AbstentionReason.INTERVAL_TOO_WIDE,
)

#: Canonical reporting order for reason codes.
REASON_ORDER: tuple[AbstentionReason, ...] = tuple(AbstentionReason)


class AdaptationGateDecision(enum.StrEnum):
    """Whether an already-chosen adaptation action may be acted upon."""

    ELIGIBLE = "eligible"
    BLOCKED = "blocked"


def _assert_distribution(
    values: tuple[float, ...],
    vocabulary: tuple[str, ...],
    *,
    context: str,
) -> None:
    """Assert a probability vector is finite, non-negative, and sums to one."""
    if len(values) != len(vocabulary):
        raise ValueError(
            f"{context}: {len(values)} probabilities for {len(vocabulary)} class(es)"
        )
    if not values:
        raise ValueError(
            f"{context}: an empty probability vector is not a distribution"
        )
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{context}: probability {value!r} is not finite")
        if value < 0.0:
            raise ValueError(f"{context}: probability {value!r} is negative")
    total = math.fsum(values)
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"{context}: probabilities sum to {total!r}, not 1.0")


def _check_self_check_document(document: Any) -> None:
    """Enforce the shared self-check banner and eligibility contract."""
    if not document.disclaimers:
        raise ValueError("an uncertainty document must carry at least one disclaimer")
    if document.evaluation_mode is EvaluationMode.SOFTWARE_SELF_CHECK:
        if document.scientific_evaluation_eligible:
            raise ValueError(
                "a software self-check can never be scientifically eligible"
            )
        if not any(SOFTWARE_SELF_CHECK_BANNER in d for d in document.disclaimers):
            raise ValueError(
                "a software self-check document must carry the banner "
                f"{SOFTWARE_SELF_CHECK_BANNER!r}"
            )


class SignalQualitySummary(BaseModel):
    """Per-window measurement quality, carried BESIDE a confidence record.

    This is the measurement's own diagnostic.  It is never combined
    arithmetically with a model probability, and a low value here is a
    statement about the camera or the task signal, never about a person.
    """

    model_config = {"extra": "forbid", "frozen": True}

    available_modalities: tuple[FusionModality, ...] = ()
    unavailable_modalities: tuple[FusionModality, ...] = ()
    modality_quality: dict[str, float | None] = Field(
        default_factory=dict,
        description=(
            "Per-modality recorded quality in [0, 1]; None where the dataset "
            "recorded none. A missing quality is never read as zero quality."
        ),
    )
    minimum_recorded_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    note: str = QUALITY_IS_NOT_CONFIDENCE_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        overlap = set(self.available_modalities) & set(self.unavailable_modalities)
        if overlap:
            raise ValueError(
                "modalities are both available and unavailable: "
                f"{sorted(m.value for m in overlap)}"
            )
        for name, value in self.modality_quality.items():
            if value is None:
                continue
            if not math.isfinite(value):
                raise ValueError(f"quality for {name!r} is not finite")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"quality for {name!r} is {value!r}, outside [0, 1]")
        return self


class EnsembleDisagreementReference(BaseModel):
    """A Milestone 6 disagreement value carried beside a Milestone 7 record.

    It keeps its Milestone 6 name and its Milestone 6 meaning.  Nothing in
    this module renames it uncertainty, and nothing converts it into a
    confidence score or an interval.
    """

    model_config = {"extra": "forbid", "frozen": True}

    ensemble_disagreement: float | None = Field(
        default=None,
        description=(
            "Mean pairwise distance between modality experts' outputs on this "
            "window, as defined in Milestone 6. None when it was not computed."
        ),
    )
    expert_count: int = Field(default=0, ge=0)
    source_strategy: str | None = None
    note: str = DISAGREEMENT_IS_NOT_UNCERTAINTY_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.ensemble_disagreement is not None and not math.isfinite(
            self.ensemble_disagreement
        ):
            raise ValueError("ensemble_disagreement is not finite")
        return self


class EvidenceGateConfiguration(BaseModel):
    """Configuration of the evidence-availability gate.

    The gate is deliberately *separate* from model confidence.  It asks
    whether there was enough usable measurement to act on at all; the
    confidence threshold asks whether the model was sure enough.  Neither
    is multiplied into the other, and each has its own reason code.
    """

    model_config = {"extra": "forbid", "frozen": True}

    enabled: bool = True
    require_prediction_available: bool = True
    require_probability_calibration_for_classification_confidence: bool = True
    minimum_available_modalities: int = Field(default=1, ge=0)
    required_modalities: tuple[FusionModality, ...] = ()
    minimum_signal_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    treat_missing_quality_as_failure: bool = Field(
        default=False,
        description=(
            "When False (the default) a modality with no recorded quality does "
            "not fail the quality gate: absence of a measurement is not a low "
            "measurement. It is recorded either way."
        ),
    )
    note: str = QUALITY_IS_NOT_CONFIDENCE_NOTE


class SelectivePredictionConfiguration(BaseModel):
    """Everything that defines the selective-prediction policy of a run.

    Every threshold here is an ENGINEERING DEFAULT.  None was chosen by
    looking at a result, none is empirically optimal, and none is
    validated.
    """

    model_config = {"extra": "forbid", "frozen": True}

    prediction_source: PredictionSource = PredictionSource.BASELINE_MODEL
    modalities: tuple[FusionModality, ...] = ()
    model_classification: str = "logistic_regression"
    model_regression: str = "ridge"

    # --- classification -------------------------------------------------
    confidence_source: UncertaintyMethod = UncertaintyMethod.MAX_CALIBRATED_PROBABILITY
    population_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    #: The CLASSIFICATION confidence grid. Its values are probabilities and
    #: it is swept only over the confidence axis; it is never reused as an
    #: interval-width grid, whose values carry the target's units.
    threshold_grid: tuple[float, ...] = ()
    estimate_population_threshold: bool = False
    threshold_objective: ThresholdObjective = (
        ThresholdObjective.TARGET_ACCEPTED_ACCURACY
    )
    threshold_objective_target: float = Field(default=0.80, ge=0.0, le=1.0)
    minimum_threshold_selection_samples: int = Field(default=30, ge=1)
    minimum_threshold_selection_groups: int = Field(default=2, ge=1)

    # --- regression -----------------------------------------------------
    interval_method: UncertaintyMethod = (
        UncertaintyMethod.SPLIT_CONFORMAL_ABSOLUTE_RESIDUAL
    )
    alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    maximum_interval_width: float | None = Field(default=None, gt=0.0)
    interval_width_grid: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "The REGRESSION width sweep, in the target's own units. It is a "
            "separate surface from threshold_grid because an interval width "
            "is not a probability: a general regression target need not live "
            "in [0, 1], and a width of 2.5 is a perfectly ordinary value. "
            "None means no width sweep was configured, in which case no "
            "width coverage curve is manufactured and the run reports its "
            "operating point only."
        ),
    )
    clip_interval_to_target_range: bool = Field(
        default=False,
        description=(
            "Presentation/domain projection only. When True a second, clipped "
            "pair of bounds is recorded ALONGSIDE the raw bounds; empirical "
            "interval coverage is always computed on the raw bounds."
        ),
    )

    # --- personalization ------------------------------------------------
    personalized_thresholds_enabled: bool = False
    personal_calibration_windows: int = Field(default=5, ge=0)
    minimum_personal_calibration_windows: int = Field(default=5, ge=1)
    minimum_evaluation_windows: int = Field(default=1, ge=1)
    personal_target_coverage: float = Field(default=0.80, gt=0.0, le=1.0)
    personal_shrinkage_constant: float = Field(default=10.0, gt=0.0)
    fallback_to_population_threshold: bool = True

    # --- gates ----------------------------------------------------------
    evidence_gate: EvidenceGateConfiguration = Field(
        default_factory=EvidenceGateConfiguration
    )
    adaptation_gate_enabled: bool = True

    # --- recorded documentation ------------------------------------------
    confidence_equation: str = CONFIDENCE_EQUATION
    entropy_equation: str = ENTROPY_EQUATION
    margin_equation: str = MARGIN_EQUATION
    classification_acceptance_rule: str = CLASSIFICATION_ACCEPTANCE_RULE
    conformal_equation: str = CONFORMAL_EQUATION
    regression_acceptance_rule: str = REGRESSION_ACCEPTANCE_RULE
    coverage_equation: str = COVERAGE_EQUATION
    note: str = UNCERTAINTY_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.confidence_source not in CLASSIFICATION_METHODS:
            raise ValueError(
                f"confidence_source {self.confidence_source.value!r} is not a "
                "classification confidence method; valid: "
                f"{sorted(m.value for m in CLASSIFICATION_METHODS)}"
            )
        if self.interval_method not in REGRESSION_METHODS:
            raise ValueError(
                f"interval_method {self.interval_method.value!r} is not a "
                "regression interval method; valid: "
                f"{sorted(m.value for m in REGRESSION_METHODS)}"
            )
        if not self.threshold_grid:
            raise ValueError(
                "threshold_grid must not be empty: a coverage-versus-performance "
                "curve needs at least one threshold to evaluate"
            )
        seen: set[float] = set()
        for value in self.threshold_grid:
            if not math.isfinite(value):
                raise ValueError(f"threshold_grid entry {value!r} is not finite")
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"threshold_grid entry {value!r} is outside [0, 1]; a "
                    "threshold is compared against a probability"
                )
            if value in seen:
                raise ValueError(f"threshold_grid contains {value!r} more than once")
            seen.add(value)
        if list(self.threshold_grid) != sorted(self.threshold_grid):
            raise ValueError(
                "threshold_grid must be sorted ascending so the coverage curve "
                "is deterministic and its monotonicity is checkable"
            )
        if self.maximum_interval_width is not None and not math.isfinite(
            self.maximum_interval_width
        ):
            raise ValueError("maximum_interval_width is not finite")
        if self.interval_width_grid is not None:
            if not self.interval_width_grid:
                raise ValueError(
                    "interval_width_grid is empty. Configure at least one "
                    "width, or leave it null; an empty grid would report a "
                    "sweep that evaluated nothing."
                )
            widths: set[float] = set()
            for width in self.interval_width_grid:
                if not math.isfinite(width):
                    raise ValueError(
                        f"interval_width_grid entry {width!r} is not finite"
                    )
                if width < 0.0:
                    raise ValueError(
                        f"interval_width_grid entry {width!r} is negative; an "
                        "interval width is a non-negative distance in the "
                        "target's units"
                    )
                if width in widths:
                    raise ValueError(
                        f"interval_width_grid contains {width!r} more than once"
                    )
                widths.add(width)
            if list(self.interval_width_grid) != sorted(self.interval_width_grid):
                raise ValueError(
                    "interval_width_grid must be sorted ascending so the width "
                    "coverage curve is deterministic and its monotonicity is "
                    "checkable"
                )
        if self.personalized_thresholds_enabled:
            if self.personal_calibration_windows < (
                self.minimum_personal_calibration_windows
            ):
                raise ValueError(
                    "personal_calibration_windows "
                    f"({self.personal_calibration_windows}) is below "
                    "minimum_personal_calibration_windows "
                    f"({self.minimum_personal_calibration_windows}), so no "
                    "subject could ever reach the minimum evidence and every "
                    "one would fall back. Configure a reachable minimum "
                    "rather than a rule that cannot fire."
                )
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities contains duplicates")
        return self


class ClassificationConfidence(BaseModel):
    """One window's classification probability, confidence, and diagnostics.

    ``confidence_score`` exists only when the probability vector satisfied
    the calibration contract.  Otherwise the same number is recorded as
    ``selection_score`` under
    :attr:`UncertaintyMethod.MAX_UNCALIBRATED_PROBABILITY`, so a reader can
    never mistake an uncalibrated maximum for calibrated confidence.
    """

    model_config = {"extra": "forbid", "frozen": True}

    window_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    task_type: TaskType = TaskType.CLASSIFICATION
    fold_index: int = Field(ge=0)

    source_model: PredictionSource
    source_model_name: str = Field(min_length=1)
    source_prediction_id: str = Field(
        min_length=1,
        description=(
            "Stable identity of the prediction this record describes: "
            "'<run_id>|<fold_index>|<window_id>|<source_model>'."
        ),
    )

    class_vocabulary: tuple[str, ...]
    probabilities: tuple[float, ...]
    probability_calibration_status: ProbabilityCalibrationStatus
    probability_calibration_method: str | None = None
    probability_calibration_group_count: int = Field(default=0, ge=0)
    probability_calibration_unavailable_reason: str | None = None

    predicted_class: str = Field(min_length=1)
    maximum_probability: float = Field(ge=0.0, le=1.0)
    maximum_probability_class: str = Field(min_length=1)

    method: UncertaintyMethod
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    selection_score: float | None = Field(default=None, ge=0.0, le=1.0)

    entropy: float | None = Field(default=None, ge=0.0)
    normalized_entropy: float | None = Field(default=None, ge=0.0, le=1.0)
    margin: float | None = Field(default=None, ge=0.0, le=1.0)

    signal_quality: SignalQualitySummary | None = None
    disagreement: EnsembleDisagreementReference | None = None

    data_source: str = Field(min_length=1)
    is_synthetic: bool
    scientific_evaluation_eligible: bool

    entropy_equation: str = ENTROPY_EQUATION
    margin_equation: str = MARGIN_EQUATION
    note: str = UNCERTAINTY_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"window {self.window_id!r} (fold {self.fold_index})"
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                f"{where}: a synthetic record can never be scientifically eligible"
            )
        _assert_distribution(self.probabilities, self.class_vocabulary, context=where)
        if self.predicted_class not in self.class_vocabulary:
            raise ValueError(
                f"{where}: predicted class {self.predicted_class!r} is not in "
                f"the vocabulary {list(self.class_vocabulary)}"
            )
        if self.maximum_probability_class not in self.class_vocabulary:
            raise ValueError(
                f"{where}: maximum-probability class "
                f"{self.maximum_probability_class!r} is not in the vocabulary"
            )
        expected = max(self.probabilities)
        if abs(self.maximum_probability - expected) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError(
                f"{where}: maximum_probability is {self.maximum_probability!r} "
                f"but the largest recorded probability is {expected!r}"
            )

        calibrated = (
            self.probability_calibration_status
            is ProbabilityCalibrationStatus.CALIBRATED
        )
        if self.probability_calibration_status is (
            ProbabilityCalibrationStatus.UNAVAILABLE
        ):
            raise ValueError(
                f"{where}: a classification confidence record carries a "
                "probability vector, so its calibration status cannot be "
                "'unavailable'; use 'uncalibrated' when no calibrator was fitted"
            )
        if calibrated:
            if self.method is not UncertaintyMethod.MAX_CALIBRATED_PROBABILITY:
                raise ValueError(
                    f"{where}: a calibrated vector must record method "
                    f"{UncertaintyMethod.MAX_CALIBRATED_PROBABILITY.value!r}"
                )
            if self.confidence_score is None:
                raise ValueError(
                    f"{where}: a calibrated vector must carry a confidence_score"
                )
            if self.selection_score is not None:
                raise ValueError(
                    f"{where}: a calibrated vector records confidence_score, not "
                    "selection_score; carrying both would let a reader treat the "
                    "uncalibrated name as if it had been calibrated"
                )
            if abs(self.confidence_score - self.maximum_probability) > (
                PROBABILITY_SUM_TOLERANCE
            ):
                raise ValueError(
                    f"{where}: confidence_score {self.confidence_score!r} is not "
                    f"the maximum calibrated probability {self.maximum_probability!r}"
                )
        else:
            if self.method is not UncertaintyMethod.MAX_UNCALIBRATED_PROBABILITY:
                raise ValueError(
                    f"{where}: an uncalibrated vector must record method "
                    f"{UncertaintyMethod.MAX_UNCALIBRATED_PROBABILITY.value!r}"
                )
            if self.confidence_score is not None:
                raise ValueError(
                    f"{where}: this probability vector was not calibrated, so its "
                    "maximum may not be recorded as confidence_score. Record it "
                    "as selection_score under an explicitly uncalibrated policy."
                )
            if self.selection_score is None:
                raise ValueError(
                    f"{where}: an uncalibrated vector must carry a selection_score"
                )
            if not self.probability_calibration_unavailable_reason:
                raise ValueError(
                    f"{where}: an uncalibrated vector must state why no "
                    "calibrator was fitted"
                )

        if self.entropy is not None and not math.isfinite(self.entropy):
            raise ValueError(f"{where}: entropy is not finite")
        if self.margin is not None and not math.isfinite(self.margin):
            raise ValueError(f"{where}: margin is not finite")
        return self

    def score(self) -> float:
        """The number a threshold is compared against, calibrated or not."""
        value = (
            self.confidence_score
            if self.confidence_score is not None
            else (self.selection_score)
        )
        # Both branches of the validator guarantee exactly one is present.
        assert value is not None
        return value


class RegressionPredictionInterval(BaseModel):
    """One window's regression point prediction and its interval.

    There is no confidence field here on purpose.  A point prediction has
    no class probability, and inventing ``1 - width`` would be a
    probability-shaped number with no probabilistic meaning.
    """

    model_config = {"extra": "forbid", "frozen": True}

    window_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    task_type: TaskType = TaskType.REGRESSION
    fold_index: int = Field(ge=0)

    source_model: PredictionSource
    source_model_name: str = Field(min_length=1)
    source_prediction_id: str = Field(min_length=1)

    predicted_value: float
    interval_method: UncertaintyMethod
    calibration_succeeded: bool
    unavailable_reason: str | None = None

    lower_bound: float | None = None
    upper_bound: float | None = None
    interval_width: float | None = Field(default=None, ge=0.0)
    conformal_quantile: float | None = Field(default=None, ge=0.0)
    conformal_order_statistic: int | None = Field(default=None, ge=1)

    clipped_lower_bound: float | None = None
    clipped_upper_bound: float | None = None
    clipping_note: str | None = None

    alpha: float = Field(gt=0.0, lt=1.0)
    nominal_coverage: float = Field(ge=0.0, le=1.0)
    calibration_sample_count: int = Field(ge=0)
    calibration_group_count: int = Field(ge=0)
    calibration_group_ids: tuple[str, ...] = ()

    signal_quality: SignalQualitySummary | None = None
    disagreement: EnsembleDisagreementReference | None = None

    data_source: str = Field(min_length=1)
    is_synthetic: bool
    scientific_evaluation_eligible: bool

    equation: str = CONFORMAL_EQUATION
    note: str = UNCERTAINTY_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"window {self.window_id!r} (fold {self.fold_index})"
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                f"{where}: a synthetic record can never be scientifically eligible"
            )
        if not math.isfinite(self.predicted_value):
            raise ValueError(
                f"{where}: the regression point prediction is not finite; a "
                "model that cannot produce a finite prediction must fail rather "
                "than emit one"
            )
        if abs((self.alpha + self.nominal_coverage) - 1.0) > 1e-12:
            raise ValueError(
                f"{where}: nominal_coverage {self.nominal_coverage!r} and alpha "
                f"{self.alpha!r} must sum to 1.0"
            )

        if not self.calibration_succeeded:
            if not self.unavailable_reason:
                raise ValueError(
                    f"{where}: an unavailable interval must state why. It is "
                    "never recorded as width zero, which would read as a "
                    "perfectly certain prediction."
                )
            for name in (
                "lower_bound",
                "upper_bound",
                "interval_width",
                "conformal_quantile",
            ):
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"{where}: interval calibration failed but {name} is recorded"
                    )
            return self

        if self.lower_bound is None or self.upper_bound is None:
            raise ValueError(f"{where}: a successful interval must carry both bounds")
        if self.interval_width is None:
            raise ValueError(f"{where}: a successful interval must carry its width")
        for name in ("lower_bound", "upper_bound", "interval_width"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{where}: {name} is not finite")
        if not self.lower_bound <= self.predicted_value <= self.upper_bound:
            raise ValueError(
                f"{where}: the split-conformal interval is symmetric about the "
                f"point prediction, so {self.lower_bound!r} <= "
                f"{self.predicted_value!r} <= {self.upper_bound!r} must hold"
            )
        width = self.upper_bound - self.lower_bound
        if abs(width - self.interval_width) > 1e-9:
            raise ValueError(
                f"{where}: interval_width {self.interval_width!r} does not match "
                f"upper - lower = {width!r}"
            )
        if self.conformal_quantile is None:
            raise ValueError(
                f"{where}: a successful conformal interval must record the "
                "residual quantile it was built from"
            )
        if (self.clipped_lower_bound is None) != (self.clipped_upper_bound is None):
            raise ValueError(f"{where}: clipped bounds must be recorded as a pair")
        if self.clipped_lower_bound is not None and not self.clipping_note:
            raise ValueError(
                f"{where}: clipped bounds are a presentation projection and must "
                "say so; the raw bounds remain the interval of record"
            )
        return self


class PersonalThresholdRecord(BaseModel):
    """One held-out subject's confidence threshold and its provenance.

    The personal threshold is computed from the subject's own EARLIER
    calibration windows and **uses no labels at all** — only the confidence
    scores the population model assigned to those windows.  An evaluation
    label therefore cannot influence it by any path.
    """

    model_config = {"extra": "forbid", "frozen": True}

    subject_id: str = Field(min_length=1)
    fold_index: int = Field(ge=0)

    population_threshold: float = Field(ge=0.0, le=1.0)
    personalized_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    applied_threshold: float = Field(ge=0.0, le=1.0)
    threshold_source: ThresholdSource

    personalization_applied: bool
    fallback_to_population: bool
    fallback_reason: str | None = None

    calibration_window_ids: tuple[str, ...] = ()
    evaluation_window_ids: tuple[str, ...] = ()
    calibration_sample_count: int = Field(ge=0)
    calibration_start_utc: str | None = None
    calibration_end_utc: str | None = None
    evaluation_start_utc: str | None = None
    temporal_order_verified: bool = False

    selection_method: str = Field(
        default=(
            "tau_s = (1 - lambda) * tau_population + lambda * "
            "quantile(subject calibration confidence, 1 - target_coverage), "
            "lambda = n / (n + kappa). The quantile uses numpy's 'lower' "
            "method, so the threshold is an observed confidence value rather "
            "than an interpolated one. NO LABEL of any kind participates."
        ),
        min_length=1,
    )
    target_coverage: float | None = Field(default=None, gt=0.0, le=1.0)
    shrinkage: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_personal_quantile: float | None = Field(default=None, ge=0.0, le=1.0)
    uses_labels: bool = Field(
        default=False,
        description=(
            "Always False. Recorded explicitly so the leakage claim is "
            "auditable from the artifact alone."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"subject {self.subject_id!r} (fold {self.fold_index})"
        if self.uses_labels:
            raise ValueError(
                f"{where}: the personal threshold rule uses no labels; a record "
                "claiming otherwise describes a different mechanism"
            )
        if self.personalization_applied:
            if self.personalized_threshold is None:
                raise ValueError(
                    f"{where}: personalization was applied without a threshold"
                )
            if self.threshold_source is not ThresholdSource.PERSONALIZED:
                raise ValueError(
                    f"{where}: an applied personalization must record source "
                    f"{ThresholdSource.PERSONALIZED.value!r}"
                )
            if self.applied_threshold != self.personalized_threshold:
                raise ValueError(
                    f"{where}: applied_threshold {self.applied_threshold!r} is "
                    f"not the personalized threshold {self.personalized_threshold!r}"
                )
            if not self.calibration_window_ids:
                raise ValueError(
                    f"{where}: a personalized threshold must name the calibration "
                    "windows it was derived from"
                )
            if not self.temporal_order_verified:
                raise ValueError(
                    f"{where}: a personalized threshold requires a verified "
                    "calibration-before-evaluation ordering"
                )
        else:
            if not self.fallback_to_population:
                raise ValueError(
                    f"{where}: personalization was not applied, so the record "
                    "must mark the population fallback"
                )
            if not self.fallback_reason:
                raise ValueError(f"{where}: a fallback must state its reason")
            if self.applied_threshold != self.population_threshold:
                raise ValueError(
                    f"{where}: a fallback must apply the population threshold "
                    f"{self.population_threshold!r}, not {self.applied_threshold!r}"
                )

        overlap = set(self.calibration_window_ids) & set(self.evaluation_window_ids)
        if overlap:
            raise ValueError(
                f"{where}: window(s) {sorted(overlap)} are recorded as both "
                "calibration and evaluation windows"
            )
        return self


class EstimatedThresholdRecord(BaseModel):
    """A population threshold estimated from inner calibration groups.

    The outer-test rows of the fold play no part.  When the objective
    cannot be met on the calibration groups the record is
    ``available=false`` with a reason and the configured population
    threshold is applied instead — a threshold is never invented to satisfy
    an unreachable target.
    """

    model_config = {"extra": "forbid", "frozen": True}

    fold_index: int = Field(ge=0)
    available: bool
    unavailable_reason: str | None = None

    objective: ThresholdObjective
    objective_target: float = Field(ge=0.0, le=1.0)
    selected_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    achieved_value: float | None = None
    achieved_coverage: float | None = Field(default=None, ge=0.0, le=1.0)

    search_grid: tuple[float, ...] = ()
    tie_break_rule: str = Field(
        default=(
            "Among thresholds meeting the objective the SMALLEST is chosen, "
            "which maximises coverage among admissible thresholds. The grid is "
            "evaluated in ascending order, so the choice is deterministic."
        ),
        min_length=1,
    )

    calibration_group_ids: tuple[str, ...] = ()
    calibration_sample_count: int = Field(ge=0)
    calibration_group_count: int = Field(ge=0)
    outer_test_group_ids: tuple[str, ...] = ()
    used_outer_test_labels: bool = Field(
        default=False,
        description=(
            "Always False. Recorded explicitly so the leakage claim is "
            "auditable from the artifact alone."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"fold {self.fold_index}"
        if self.used_outer_test_labels:
            raise ValueError(
                f"{where}: threshold selection never reads outer-test labels; a "
                "record claiming otherwise describes a different mechanism"
            )
        if self.available:
            if self.selected_threshold is None:
                raise ValueError(f"{where}: an available record must carry a threshold")
            if not self.search_grid:
                raise ValueError(f"{where}: an available record must record its grid")
            if self.selected_threshold not in self.search_grid:
                raise ValueError(
                    f"{where}: the selected threshold {self.selected_threshold!r} "
                    "is not one of the searched grid points"
                )
        else:
            if not self.unavailable_reason:
                raise ValueError(f"{where}: an unavailable record must state why")
            if self.selected_threshold is not None:
                raise ValueError(
                    f"{where}: the objective was not met, so no threshold may be "
                    "recorded; the configured population threshold is applied"
                )
        overlap = set(self.calibration_group_ids) & set(self.outer_test_group_ids)
        if overlap:
            raise ValueError(
                f"{where}: threshold-selection groups overlap the outer-test "
                f"groups: {sorted(overlap)}"
            )
        return self


class AbstentionDecision(BaseModel):
    """Whether one window's estimate may be acted upon, and why.

    The original prediction stays on the record.  An abstained window is
    not a wrong prediction, not a missing prediction, and not a zero: it is
    a prediction that exists and was not acted upon.
    """

    model_config = {"extra": "forbid", "frozen": True}

    window_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    task_type: TaskType
    fold_index: int = Field(ge=0)
    source_prediction_id: str = Field(min_length=1)

    prediction_available: bool
    accepted: bool
    abstained: bool
    reasons: tuple[AbstentionReason, ...] = ()

    # --- classification --------------------------------------------------
    predicted_class: str | None = None
    class_vocabulary: tuple[str, ...] = ()
    probabilities: tuple[float, ...] = ()
    probability_calibration_status: ProbabilityCalibrationStatus | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    selection_score: float | None = Field(default=None, ge=0.0, le=1.0)
    entropy: float | None = None
    margin: float | None = None
    applied_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold_source: ThresholdSource | None = None

    # --- regression ------------------------------------------------------
    predicted_value: float | None = None
    interval_lower_bound: float | None = None
    interval_upper_bound: float | None = None
    interval_width: float | None = Field(default=None, ge=0.0)
    maximum_interval_width: float | None = Field(default=None, gt=0.0)

    # --- evidence, kept separate ----------------------------------------
    signal_quality: SignalQualitySummary | None = None
    disagreement: EnsembleDisagreementReference | None = None
    evidence_gate_passed: bool | None = None

    data_source: str = Field(min_length=1)
    is_synthetic: bool
    scientific_evaluation_eligible: bool

    acceptance_rule: str = Field(min_length=1)
    note: str = ABSTENTION_MEANING_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"window {self.window_id!r} (fold {self.fold_index})"
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                f"{where}: a synthetic record can never be scientifically eligible"
            )
        if self.accepted == self.abstained:
            raise ValueError(
                f"{where}: a decision is exactly one of accepted or abstained; "
                f"both are {self.accepted!r}"
            )
        if self.abstained and not self.reasons:
            raise ValueError(f"{where}: an abstention must state at least one reason")
        if self.accepted and self.reasons:
            raise ValueError(
                f"{where}: an accepted decision carries no abstention reason; "
                f"got {[r.value for r in self.reasons]}"
            )
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError(f"{where}: duplicate abstention reasons")
        ordered = tuple(r for r in REASON_ORDER if r in set(self.reasons))
        if self.reasons != ordered:
            raise ValueError(
                f"{where}: abstention reasons must be recorded in the canonical "
                f"order {[r.value for r in ordered]} so two runs of one "
                "configuration produce identical documents"
            )

        if not self.prediction_available:
            if not self.abstained:
                raise ValueError(
                    f"{where}: no prediction is available, so the window cannot "
                    "be accepted"
                )
            if AbstentionReason.MODEL_PREDICTION_UNAVAILABLE not in self.reasons:
                raise ValueError(
                    f"{where}: an unavailable prediction must record "
                    f"{AbstentionReason.MODEL_PREDICTION_UNAVAILABLE.value!r}"
                )
            if self.predicted_class is not None or self.predicted_value is not None:
                raise ValueError(
                    f"{where}: no prediction is available but one is recorded"
                )
            return self

        if self.task_type is TaskType.CLASSIFICATION:
            if self.predicted_class is None:
                raise ValueError(
                    f"{where}: an available classification prediction is retained "
                    "on the record even when the window is abstained"
                )
            _assert_distribution(
                self.probabilities, self.class_vocabulary, context=where
            )
            if self.predicted_class not in self.class_vocabulary:
                raise ValueError(
                    f"{where}: predicted class {self.predicted_class!r} is not in "
                    f"the vocabulary {list(self.class_vocabulary)}"
                )
            if self.predicted_value is not None:
                raise ValueError(
                    f"{where}: a classification decision must not carry a numeric "
                    "prediction"
                )
            if self.accepted and self.applied_threshold is None:
                raise ValueError(
                    f"{where}: an accepted classification decision must record the "
                    "threshold it cleared"
                )
            if self.applied_threshold is not None and self.threshold_source is None:
                raise ValueError(
                    f"{where}: a threshold is recorded without its provenance"
                )
        else:
            if self.predicted_value is None:
                raise ValueError(
                    f"{where}: an available regression prediction is retained on "
                    "the record even when the window is abstained"
                )
            if not math.isfinite(self.predicted_value):
                raise ValueError(f"{where}: the regression prediction is not finite")
            if self.predicted_class is not None or self.probabilities:
                raise ValueError(
                    f"{where}: a regression decision must not carry a class or "
                    "probabilities"
                )
            if self.interval_width is None and self.accepted:
                raise ValueError(
                    f"{where}: a regression decision cannot be accepted without an "
                    "interval; a missing interval is not width zero"
                )
            bounds = (self.interval_lower_bound, self.interval_upper_bound)
            if (bounds[0] is None) != (bounds[1] is None):
                raise ValueError(f"{where}: interval bounds must be recorded as a pair")
            if bounds[0] is not None and bounds[1] is not None:
                if not bounds[0] <= self.predicted_value <= bounds[1]:
                    raise ValueError(
                        f"{where}: the point prediction lies outside its interval"
                    )

        if self.confidence_score is not None and self.selection_score is not None:
            raise ValueError(
                f"{where}: a decision records either a calibrated confidence or "
                "an uncalibrated selection score, never both"
            )
        return self

    def primary_reason(self) -> AbstentionReason | None:
        """The first reason in canonical order, or ``None`` when accepted."""
        return self.reasons[0] if self.reasons else None


class CoveragePoint(BaseModel):
    """Accepted/abstained/unavailable counts at one axis value.

    ``axis`` says what ``threshold`` is a threshold *on*.  On the
    confidence axis it is a probability in ``[0, 1]``; on the width axis it
    is a distance in the regression target's own units and is bounded below
    by zero only.  A point that did not record its axis could be read in
    the wrong direction, which is exactly the mistake this field exists to
    prevent.
    """

    model_config = {"extra": "forbid", "frozen": True}

    axis: CoverageAxis = CoverageAxis.CONFIDENCE_THRESHOLD
    threshold: float | None = Field(default=None, ge=0.0)
    threshold_unavailable_reason: str | None = None
    total_window_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    abstained_count: int = Field(ge=0)
    unavailable_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    abstention_rate: float = Field(ge=0.0, le=1.0)
    coverage_denominator: str = "total_evaluated_windows"
    equation: str = COVERAGE_EQUATION

    @property
    def axis_units(self) -> str:
        """Units of this point's axis value."""
        return COVERAGE_AXIS_UNITS[self.axis]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.threshold is None:
            if self.axis is not CoverageAxis.MAXIMUM_INTERVAL_WIDTH:
                raise ValueError(
                    "a confidence-axis point always has a threshold: the "
                    "population confidence threshold is always applied. Only "
                    "a width-axis operating point with no configured "
                    "maximum_interval_width may omit one."
                )
            if not self.threshold_unavailable_reason:
                raise ValueError(
                    "a coverage point with no axis value must state why it "
                    "has none; an absent width threshold is not a width "
                    "threshold of zero, which would abstain on every window"
                )
        elif self.axis is CoverageAxis.CONFIDENCE_THRESHOLD and self.threshold > 1.0:
            raise ValueError(
                f"confidence threshold {self.threshold!r} is outside [0, 1]; "
                "a confidence threshold is compared against a probability. An "
                "interval width belongs on the "
                f"{CoverageAxis.MAXIMUM_INTERVAL_WIDTH.value!r} axis instead."
            )
        total = self.accepted_count + self.abstained_count + self.unavailable_count
        if total != self.total_window_count:
            raise ValueError(
                f"threshold {self.threshold!r}: accepted + abstained + "
                f"unavailable = {total}, but total_window_count is "
                f"{self.total_window_count}. The three must reconcile exactly."
            )
        if self.total_window_count == 0:
            if self.coverage != 0.0 or self.abstention_rate != 0.0:
                raise ValueError("an empty evaluation set has no coverage")
            return self
        expected = self.accepted_count / self.total_window_count
        if abs(self.coverage - expected) > 1e-12:
            raise ValueError(
                f"threshold {self.threshold!r}: coverage {self.coverage!r} is not "
                f"accepted / total = {expected!r}"
            )
        expected_abstention = self.abstained_count / self.total_window_count
        if abs(self.abstention_rate - expected_abstention) > 1e-12:
            raise ValueError(
                f"threshold {self.threshold!r}: abstention_rate "
                f"{self.abstention_rate!r} is not abstained / total = "
                f"{expected_abstention!r}"
            )
        return self


class SelectiveMetrics(BaseModel):
    """Performance over ACCEPTED windows, always beside its coverage.

    Abstained windows are not scored.  They are not counted as errors, they
    do not become a class, and they do not become zero.  Their number is
    reported so that an accepted-set score is never read as a whole-set
    score.
    """

    model_config = {"extra": "forbid"}

    axis: CoverageAxis = CoverageAxis.CONFIDENCE_THRESHOLD
    threshold: float | None = Field(default=None, ge=0.0)
    coverage_point: CoveragePoint

    accepted_classification: ClassificationMetrics | None = None
    accepted_regression: RegressionMetrics | None = None

    empirical_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    empirical_risk_equation: str = RISK_EQUATION

    empirical_interval_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_interval_width: float | None = Field(default=None, ge=0.0)
    median_interval_width: float | None = Field(default=None, ge=0.0)

    accepted_class_support: dict[str, int] = Field(default_factory=dict)
    unavailable_metrics: dict[str, str] = Field(default_factory=dict)
    note: str = SELECTIVE_PREDICTION_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.threshold != self.coverage_point.threshold:
            raise ValueError(
                f"selective metrics at threshold {self.threshold!r} carry a "
                f"coverage point for {self.coverage_point.threshold!r}"
            )
        if self.axis is not self.coverage_point.axis:
            raise ValueError(
                f"selective metrics on the {self.axis.value!r} axis carry a "
                f"coverage point on the {self.coverage_point.axis.value!r} "
                "axis. The two axes move in opposite directions, so mixing "
                "them would invert the reported curve."
            )
        if self.accepted_classification is not None:
            if (
                self.accepted_classification.sample_count
                != self.coverage_point.accepted_count
            ):
                raise ValueError(
                    "accepted classification metrics were computed over "
                    f"{self.accepted_classification.sample_count} rows but "
                    f"{self.coverage_point.accepted_count} were accepted; "
                    "accepted metrics must use exactly the accepted rows"
                )
            if self.axis is CoverageAxis.MAXIMUM_INTERVAL_WIDTH:
                raise ValueError(
                    "a width-axis point carries accepted CLASSIFICATION "
                    "metrics. An interval-width sweep evaluates a regression "
                    "target and has no class probabilities to score."
                )
        if self.accepted_regression is not None:
            if (
                self.accepted_regression.sample_count
                != self.coverage_point.accepted_count
            ):
                raise ValueError(
                    "accepted regression metrics were computed over "
                    f"{self.accepted_regression.sample_count} rows but "
                    f"{self.coverage_point.accepted_count} were accepted"
                )
            if self.axis is CoverageAxis.CONFIDENCE_THRESHOLD:
                raise ValueError(
                    "a confidence-axis point carries accepted REGRESSION "
                    "metrics. A point prediction has no class probability to "
                    "threshold; its uncertainty axis is interval width."
                )
        if self.empirical_risk is not None:
            accuracy = (
                self.accepted_classification.accuracy
                if self.accepted_classification is not None
                else None
            )
            if accuracy is None:
                raise ValueError(
                    "empirical risk is 1 - accepted accuracy and cannot be "
                    "recorded without it"
                )
            if abs(self.empirical_risk - (1.0 - accuracy)) > 1e-12:
                raise ValueError(
                    f"empirical_risk {self.empirical_risk!r} is not "
                    f"1 - accepted_accuracy = {1.0 - accuracy!r}"
                )
        return self


class RiskCoveragePoint(BaseModel):
    """One point of the risk-coverage curve."""

    model_config = {"extra": "forbid", "frozen": True}

    axis: CoverageAxis = CoverageAxis.CONFIDENCE_THRESHOLD
    threshold: float | None = Field(default=None, ge=0.0)
    coverage: float = Field(ge=0.0, le=1.0)
    empirical_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    accepted_count: int = Field(ge=0)
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.empirical_risk is None and not self.unavailable_reason:
            raise ValueError(
                f"threshold {self.threshold!r}: risk is undefined and must state "
                "why; it is never recorded as zero, which is a legitimate risk"
            )
        return self


class CoverageCurve(BaseModel):
    """The deterministic sweep of one axis, and what it may not be used for.

    Every point is computed from the **same** outer-test predictions.  No
    threshold anywhere in this repository is selected by reading this
    curve: it is a report, not a search.

    The axis is fixed by the task type and the two are validated against
    each other, so a regression curve can never be swept over a
    classification confidence score and a classification curve can never be
    swept over an interval width.  Each axis carries its own monotonicity
    direction, because raising a confidence threshold is stricter while
    raising a width maximum is more permissive.
    """

    model_config = {"extra": "forbid"}

    task_type: TaskType
    axis: CoverageAxis
    axis_values: tuple[float, ...] = ()
    points: tuple[SelectiveMetrics, ...] = ()
    risk_coverage: tuple[RiskCoveragePoint, ...] = ()
    points_unavailable_reason: str | None = None

    area_under_risk_coverage: float | None = None
    area_under_risk_coverage_unavailable_reason: str | None = None
    aurc_equation: str = AURC_EQUATION

    expected_monotonic_direction: MonotonicDirection = MonotonicDirection.NON_INCREASING
    coverage_is_monotonic: bool | None = None
    selection_note: str = (
        "No threshold in this repository is selected from this curve. The "
        "applied population threshold is a configured engineering default, "
        "or is estimated from inner calibration groups only. Reading a "
        "threshold off an outer-test curve would tune the policy on the data "
        "it is reported against."
    )
    note: str = SELECTIVE_PREDICTION_NOTE

    @property
    def axis_units(self) -> str:
        """Units of this curve's x-axis."""
        return COVERAGE_AXIS_UNITS[self.axis]

    @property
    def monotonicity_rule(self) -> str:
        """The direction contract this curve's axis must satisfy."""
        return COVERAGE_MONOTONICITY_RULE[self.axis]

    @model_validator(mode="after")
    def _check(self) -> Self:
        expected_axis = COVERAGE_AXIS_FOR_TASK[self.task_type]
        if self.axis is not expected_axis:
            raise ValueError(
                f"a {self.task_type.value} curve was swept over the "
                f"{self.axis.value!r} axis, but that task type is selective on "
                f"the {expected_axis.value!r} axis. A regression target has no "
                "class probability to threshold, so no classification "
                "confidence score can index its curve; a classification target "
                "has no prediction interval to widen."
            )
        expected_direction = COVERAGE_AXIS_DIRECTION[self.axis]
        if self.expected_monotonic_direction is not expected_direction:
            raise ValueError(
                f"the {self.axis.value!r} axis is "
                f"{expected_direction.value}, not "
                f"{self.expected_monotonic_direction.value}"
            )
        if self.area_under_risk_coverage is None and self.risk_coverage:
            if not self.area_under_risk_coverage_unavailable_reason:
                raise ValueError(
                    "the area under the risk-coverage curve is unavailable and "
                    "must state why"
                )
        if not self.points:
            if not self.points_unavailable_reason:
                raise ValueError(
                    "a curve with no points must state why it has none. An "
                    "unswept axis is reported as unavailable rather than as a "
                    "curve that happens to be empty."
                )
            if self.coverage_is_monotonic is not None:
                raise ValueError(
                    "a curve with no points has no monotonicity to report; "
                    "coverage_is_monotonic must be null rather than a "
                    "vacuously true claim"
                )
        elif self.coverage_is_monotonic is None:
            raise ValueError(
                "a curve with points must report whether coverage is monotonic "
                f"in the {self.expected_monotonic_direction.value} direction"
            )
        for point in self.points:
            if point.axis is not self.axis:
                raise ValueError(
                    f"a {self.axis.value!r} curve carries a point on the "
                    f"{point.axis.value!r} axis"
                )
        for risk_point in self.risk_coverage:
            if risk_point.axis is not self.axis:
                raise ValueError(
                    f"a {self.axis.value!r} curve carries a risk-coverage "
                    f"point on the {risk_point.axis.value!r} axis"
                )
        recorded = tuple(point.threshold for point in self.points)
        if recorded and recorded != tuple(self.axis_values):
            raise ValueError(
                f"the curve records axis values {list(recorded)} but the "
                f"configured grid is {list(self.axis_values)}; every grid value "
                "must appear exactly once, in order"
            )
        return self


class AdaptationGateRecord(BaseModel):
    """Whether an already-chosen adaptation action may be acted upon.

    This record answers one question and no other.  It does not name an
    action, does not rank actions, does not choose a difficulty, does not
    address a scene, and does not carry a reward.  Choosing what to do is
    Milestone 8.
    """

    model_config = {"extra": "forbid", "frozen": True}

    window_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    fold_index: int = Field(ge=0)
    source_prediction_id: str = Field(min_length=1)

    decision: AdaptationGateDecision
    reasons: tuple[AbstentionReason, ...] = ()

    prediction_available: bool
    prediction_abstained: bool
    evidence_gate_passed: bool
    confidence_requirement_satisfied: bool | None = None
    interval_requirement_satisfied: bool | None = None

    applied_confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    maximum_interval_width: float | None = Field(default=None, gt=0.0)

    data_source: str = Field(min_length=1)
    is_synthetic: bool
    scientific_evaluation_eligible: bool

    scope_note: str = (
        "This gate answers only whether an ALREADY-CHOSEN action may be acted "
        "upon. It does not choose an adaptation, does not rank adaptations, "
        "does not set difficulty or scene content, does not issue any "
        "transport message, does not learn, and has no cooldown, hysteresis, "
        "or reward. Adaptation policy is Milestone 8."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"window {self.window_id!r} (fold {self.fold_index})"
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                f"{where}: a synthetic record can never be scientifically eligible"
            )
        blocked = self.decision is AdaptationGateDecision.BLOCKED
        if blocked and not self.reasons:
            raise ValueError(f"{where}: a blocked gate must state at least one reason")
        if not blocked and self.reasons:
            raise ValueError(
                f"{where}: an eligible gate carries no blocking reason; got "
                f"{[r.value for r in self.reasons]}"
            )
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError(f"{where}: duplicate blocking reasons")
        ordered = tuple(r for r in REASON_ORDER if r in set(self.reasons))
        if self.reasons != ordered:
            raise ValueError(
                f"{where}: blocking reasons must be recorded in the canonical "
                f"order {[r.value for r in ordered]}"
            )
        if self.prediction_abstained and not blocked:
            raise ValueError(
                f"{where}: the prediction abstained, so no action may be acted "
                "upon; the gate cannot report eligible"
            )
        if not self.prediction_available and not blocked:
            raise ValueError(
                f"{where}: no prediction is available, so the gate cannot report "
                "eligible"
            )
        if not self.evidence_gate_passed and not blocked:
            raise ValueError(
                f"{where}: the evidence gate did not pass, so the gate cannot "
                "report eligible"
            )
        return self


class UncertaintyFoldResult(BaseModel):
    """One outer fold's uncertainty, selective, and threshold outcome."""

    model_config = {"extra": "forbid"}

    fold_index: int = Field(ge=0)
    evaluated: bool
    unavailable_reason: str | None = None

    fit_group_ids: tuple[str, ...] = ()
    probability_calibration_group_ids: tuple[str, ...] = ()
    threshold_selection_group_ids: tuple[str, ...] = ()
    conformal_calibration_group_ids: tuple[str, ...] = ()
    outer_test_group_ids: tuple[str, ...] = ()

    probability_calibration_status: ProbabilityCalibrationStatus = (
        ProbabilityCalibrationStatus.UNAVAILABLE
    )
    probability_calibration_method: str | None = None
    probability_calibration_unavailable_reason: str | None = None

    estimated_threshold: EstimatedThresholdRecord | None = None
    applied_population_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    applied_population_threshold_source: ThresholdSource | None = None
    personal_thresholds: tuple[PersonalThresholdRecord, ...] = ()

    conformal_quantile: float | None = Field(default=None, ge=0.0)
    conformal_order_statistic: int | None = Field(default=None, ge=1)
    conformal_calibration_sample_count: int = Field(default=0, ge=0)
    conformal_available: bool = False
    conformal_unavailable_reason: str | None = None

    total_window_count: int = Field(default=0, ge=0)
    accepted_count: int = Field(default=0, ge=0)
    abstained_count: int = Field(default=0, ge=0)
    unavailable_count: int = Field(default=0, ge=0)
    abstention_reason_counts: dict[str, int] = Field(default_factory=dict)

    applied_selective_metrics: SelectiveMetrics | None = None
    coverage_curve: CoverageCurve | None = None

    adaptation_gate_eligible_count: int = Field(default=0, ge=0)
    adaptation_gate_blocked_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"fold {self.fold_index}"
        if not self.evaluated and not self.unavailable_reason:
            raise ValueError(f"{where}: an unevaluated fold must state a reason")
        total = self.accepted_count + self.abstained_count + self.unavailable_count
        if total != self.total_window_count:
            raise ValueError(
                f"{where}: accepted + abstained + unavailable = {total}, but "
                f"total_window_count is {self.total_window_count}"
            )
        gated = self.adaptation_gate_eligible_count + self.adaptation_gate_blocked_count
        if gated and gated != self.total_window_count:
            raise ValueError(
                f"{where}: the adaptation gate decided {gated} window(s) but "
                f"{self.total_window_count} were evaluated"
            )
        for name, groups in (
            ("probability calibration", self.probability_calibration_group_ids),
            ("threshold selection", self.threshold_selection_group_ids),
            ("conformal calibration", self.conformal_calibration_group_ids),
            ("fit", self.fit_group_ids),
        ):
            overlap = set(groups) & set(self.outer_test_group_ids)
            if overlap:
                raise ValueError(
                    f"{where}: {name} used outer-test group(s) {sorted(overlap)}. "
                    "The outer-test fold never fits a model, a calibrator, a "
                    "conformal residual distribution, or a threshold."
                )
        overlap = set(self.fit_group_ids) & set(self.probability_calibration_group_ids)
        if overlap:
            raise ValueError(
                f"{where}: probability calibration reused fit group(s) "
                f"{sorted(overlap)}"
            )
        overlap = set(self.fit_group_ids) & set(self.conformal_calibration_group_ids)
        if overlap:
            raise ValueError(
                f"{where}: conformal calibration reused fit group(s) "
                f"{sorted(overlap)}; residuals from rows the model memorised "
                "understate the interval"
            )
        if self.conformal_available and self.conformal_quantile is None:
            raise ValueError(f"{where}: an available conformal fit must record q")
        if not self.conformal_available and self.conformal_quantile is not None:
            raise ValueError(
                f"{where}: conformal calibration was unavailable but q is recorded"
            )
        return self


class UncertaintyEvaluation(BaseModel):
    """The complete uncertainty-aware evaluation for one run."""

    model_config = {"extra": "forbid"}

    run_id: str = Field(min_length=1)
    evaluation_mode: EvaluationMode
    scientific_evaluation_eligible: bool
    target_name: str = Field(min_length=1)
    task_type: TaskType

    dataset_fingerprint: str = Field(min_length=1)
    split_manifest_fingerprint: str = Field(min_length=1)
    group_field: str = Field(min_length=1)
    group_count: int = Field(ge=0)
    fold_count: int = Field(ge=0)
    random_seed: int

    configuration: SelectivePredictionConfiguration
    predictor_columns: tuple[str, ...] = ()
    class_vocabulary: tuple[str, ...] = ()

    folds: tuple[UncertaintyFoldResult, ...] = ()

    total_window_count: int = Field(default=0, ge=0)
    accepted_count: int = Field(default=0, ge=0)
    abstained_count: int = Field(default=0, ge=0)
    unavailable_count: int = Field(default=0, ge=0)
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    abstention_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    coverage_denominator: str = "total_evaluated_windows"
    abstention_reason_counts: dict[str, int] = Field(default_factory=dict)

    personalized_subject_count: int = Field(default=0, ge=0)
    population_fallback_subject_count: int = Field(default=0, ge=0)

    adaptation_gate_eligible_count: int = Field(default=0, ge=0)
    adaptation_gate_blocked_count: int = Field(default=0, ge=0)

    disclaimers: tuple[str, ...]
    note: str = UNCERTAINTY_NOTE
    abstention_note: str = ABSTENTION_MEANING_NOTE
    selective_note: str = SELECTIVE_PREDICTION_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_self_check_document(self)
        total = self.accepted_count + self.abstained_count + self.unavailable_count
        if total != self.total_window_count:
            raise ValueError(
                f"accepted + abstained + unavailable = {total}, but "
                f"total_window_count is {self.total_window_count}"
            )
        if self.total_window_count and self.coverage is not None:
            expected = self.accepted_count / self.total_window_count
            if abs(self.coverage - expected) > 1e-12:
                raise ValueError(
                    f"coverage {self.coverage!r} is not accepted / total = {expected!r}"
                )
        return self


class UncertaintyExperimentManifest(BaseModel):
    """Resolved configuration and provenance of one uncertainty run."""

    model_config = {"extra": "forbid"}

    run_id: str = Field(min_length=1)
    evaluation_mode: EvaluationMode
    scientific_evaluation_eligible: bool
    target_name: str = Field(min_length=1)
    task_type: TaskType

    dataset_path: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=1)
    feature_catalog_version: str = Field(min_length=1)
    split_manifest_fingerprint: str = Field(min_length=1)
    split_strategy: str = Field(min_length=1)
    group_field: str = Field(min_length=1)
    group_count: int = Field(ge=0)
    fold_count: int = Field(ge=2)
    random_seed: int

    configuration: SelectivePredictionConfiguration
    probability_calibration_method: str = Field(min_length=1)
    calibration_group_fraction: float = Field(ge=0.0, lt=1.0)
    predictor_columns: tuple[str, ...] = ()

    run_id_inputs: tuple[str, ...] = (
        "dataset_fingerprint",
        "split_manifest_fingerprint",
        "target_name",
        "task_type",
        "prediction_source",
        "random_seed",
        "probability_calibration_method",
        "confidence_source",
        "confidence_definition",
        "population_confidence_threshold",
        "threshold_grid",
        "threshold_estimation",
        "personalized_threshold_method",
        "interval_method",
        "regression_alpha",
        "maximum_interval_width",
        "evidence_gate_configuration",
        "adaptation_gate_configuration",
        "modalities",
        "model_names",
        "engagevr_version",
    )
    run_id_excludes: tuple[str, ...] = (
        "started_at_utc",
        "finished_at_utc",
        "output_directory",
        "wall_clock",
    )

    disclaimers: tuple[str, ...]
    note: str = UNCERTAINTY_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_self_check_document(self)
        return self
