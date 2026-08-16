"""Multimodal-fusion schemas (Milestone 6).

These models are the persisted form of a fusion run.  They are written as
JSON beside the run's artifacts so a fused result can be inspected, and its
provenance judged, **without loading any model file**.

What this module encodes structurally rather than by convention
---------------------------------------------------------------
1. *Quality is not a modality.*  :class:`FusionModality` has exactly four
   members — behavioural, head pose, rPPG, task.  There is no ``quality``
   member, so capture-quality diagnostics, availability flags, and
   missingness indicators cannot become a fifth measurement modality by
   accident.  They are support/context signals and are named as such.
2. *A modality that produced no prediction receives no weight.*  A
   :class:`ModalityWeight` that did not contribute must carry
   ``normalized_weight == 0`` and state why it was excluded.
3. *Missing is never zero.*  An unavailable modality yields
   ``available=False`` plus a reason.  It never yields a zero-valued
   measurement, a zero probability vector, or a zero prediction.
4. *A fused probability vector is finite and sums to one*, and a fused
   regression prediction is finite.  A run that cannot satisfy that must
   report the window as unfused, not emit a placeholder.
5. *Synthetic can never be scientifically eligible*, and a self-check
   document must carry the self-check banner.  Both are validator-enforced,
   mirroring :mod:`engagevr.schemas.experiments`.

Expert disagreement recorded here is an **ensemble-disagreement
diagnostic**.  It is not a calibrated uncertainty estimate and it does not
drive abstention; formal uncertainty-aware inference and abstention are
Milestone 7.
"""

from __future__ import annotations

import enum
import math
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from engagevr.schemas.experiments import (
    SOFTWARE_SELF_CHECK_BANNER,
    AggregateMetric,
    ClassificationMetrics,
    EvaluationMode,
    RegressionMetrics,
)
from engagevr.schemas.targets import TaskType

#: Tolerance for "these probabilities sum to one".  Renormalisation in
#: IEEE-754 leaves an error of order 1e-16; anything larger than this is a
#: bug, not rounding.
PROBABILITY_SUM_TOLERANCE = 1e-9

#: Note attached to every disagreement summary.
DISAGREEMENT_NOTE = (
    "Expert disagreement is an ENSEMBLE-DISAGREEMENT DIAGNOSTIC. It "
    "describes how far the modality-specific estimators differed from one "
    "another on a window. It is not a calibrated uncertainty estimate, it "
    "is not signal quality, it is not model confidence, and it does not "
    "trigger abstention. Uncertainty-aware inference and abstention are "
    "Milestone 7."
)

#: Note attached to every quality-weighting record.
QUALITY_NOTE = (
    "Signal quality describes the MEASUREMENT, never the person. A low "
    "quality value means the camera or task signal was poor; it is never "
    "low engagement and never low cognitive load. Quality is kept in "
    "separate fields from model probability throughout."
)

#: Note attached to the fusion evaluation document.
FUSION_COMPARISON_NOTE = (
    "Fusion strategies here are compared on identical grouped outer folds "
    "for software verification only. No strategy is a champion, none is "
    "validated, and a comparison computed on SYNTHETIC data cannot select "
    "a best fusion architecture: it would describe the generator this "
    "repository wrote, not any person."
)


class FusionModality(enum.StrEnum):
    """A measurement modality that may carry a fusion expert.

    ``quality`` is deliberately **absent**.  Capture-quality diagnostics,
    per-feature availability flags, and missingness indicators are
    support/context signals: they explain a measurement, they are not one.
    They may inform explicitly named quality-aware weighting, but they must
    never silently become another measurement modality.
    """

    BEHAVIOURAL = "behavioural"
    HEAD_POSE = "head_pose"
    RPPG = "rppg"
    TASK = "task"


class FusionStrategy(enum.StrEnum):
    """Fusion architectures implemented in this milestone.

    ``early`` combines features before one estimator is fitted; the three
    ``*_late`` members fit one estimator per modality and combine their
    outputs.  No deep or neural fusion is implemented, and simple feature
    concatenation is never called attention.
    """

    EARLY = "early"
    UNIFORM_LATE = "uniform_late"
    QUALITY_LATE = "quality_late"
    VALIDATION_WEIGHTED_LATE = "validation_weighted_late"
    STACKED_LATE = "stacked_late"


#: Strategies that fit one estimator per modality and combine outputs.
LATE_FUSION_STRATEGIES: frozenset[FusionStrategy] = frozenset(
    {
        FusionStrategy.UNIFORM_LATE,
        FusionStrategy.QUALITY_LATE,
        FusionStrategy.VALIDATION_WEIGHTED_LATE,
        FusionStrategy.STACKED_LATE,
    }
)


class MissingQualityPolicy(enum.StrEnum):
    """What quality-aware fusion does when a modality reports no quality.

    ``EXCLUDE`` drops the modality from quality-aware weighting and records
    the exclusion.  ``DOCUMENTED_FALLBACK`` substitutes a single documented
    neutral value, recorded on every affected weight as
    ``quality_source=documented_fallback``.

    Neither policy treats missing quality as perfect quality, and there is
    no policy that does.  Some modalities have no quality channel at all —
    task telemetry is a software measurement with no signal-quality index —
    so "no quality recorded" is a normal condition that must be handled
    explicitly rather than assumed away.
    """

    EXCLUDE = "exclude"
    DOCUMENTED_FALLBACK = "documented_fallback"


class QualitySource(enum.StrEnum):
    """Where the quality value used in a weight came from."""

    MEASURED = "measured"
    DOCUMENTED_FALLBACK = "documented_fallback"
    NOT_USED = "not_used"
    UNAVAILABLE = "unavailable"


class QualityWeightingConfiguration(BaseModel):
    """Quality-aware weighting parameters and their documented equation."""

    model_config = {"extra": "forbid", "frozen": True}

    enabled: bool = True
    missing_quality_policy: MissingQualityPolicy = (
        MissingQualityPolicy.DOCUMENTED_FALLBACK
    )
    missing_quality_fallback: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Neutral quality substituted under the documented-fallback "
            "policy. It is the midpoint of the [0, 1] quality range, not a "
            "tuned value, and never 1.0: a missing quality reading is not "
            "evidence of a good signal."
        ),
    )
    minimum_quality: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Quality at or below which a modality is excluded from "
            "quality-aware fusion. The default of 0.0 excludes nothing: no "
            "empirically validated quality cut-off exists for these "
            "signals, so no non-zero default could be justified here."
        ),
    )
    minimum_effective_weight: float = Field(
        default=1e-9,
        ge=0.0,
        lt=1.0,
        description=(
            "Raw effective weight below which a modality is excluded. The "
            "default is a numerical guard against normalising by ~0, not a "
            "modelling threshold."
        ),
    )
    base_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-modality base weight. Empty means a deterministic equal "
            "base weight of 1.0 for every modality — the control. No "
            "optimised weight set is shipped as a default."
        ),
    )
    equation: str = (
        "raw_effective_weight_m = base_weight_m * availability_m * "
        "normalized_quality_m; normalized_weight_m = raw_effective_weight_m "
        "/ sum over contributing modalities. availability_m is 1 when "
        "modality m produced a prediction for this window and 0 otherwise, "
        "so an unavailable modality has zero effective weight by "
        "construction."
    )
    note: str = QUALITY_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        for name, weight in self.base_weights.items():
            try:
                FusionModality(name)
            except ValueError as exc:
                raise ValueError(
                    f"base_weights names {name!r}, which is not a fusion "
                    f"modality; valid: {[m.value for m in FusionModality]}. "
                    "Quality is a support signal, not a modality."
                ) from exc
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(
                    f"base weight for {name!r} must be finite and positive; "
                    f"got {weight!r}"
                )
        return self


class StackingConfiguration(BaseModel):
    """Leakage-safe stacked-fusion parameters."""

    model_config = {"extra": "forbid", "frozen": True}

    enabled: bool = False
    inner_folds: int = Field(
        default=3,
        ge=2,
        description=(
            "Grouped inner folds used to build out-of-fold expert "
            "predictions inside each outer training portion."
        ),
    )
    meta_model_classification: str = "logistic_regression"
    meta_model_regression: str = "ridge"
    design: str = (
        "The meta-model is fitted only on out-of-fold expert predictions "
        "generated inside the outer training groups. Experts are then "
        "refitted on the full outer training portion and applied to the "
        "untouched outer test groups. In-sample expert predictions never "
        "reach the meta-model, and no outer-test row influences meta "
        "fitting or calibration."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        allowed_classification = {"logistic_regression"}
        allowed_regression = {"ridge"}
        if self.meta_model_classification not in allowed_classification:
            raise ValueError(
                "stacking.meta_model_classification must be one of "
                f"{sorted(allowed_classification)}; got "
                f"{self.meta_model_classification!r}. No neural stacker is "
                "offered."
            )
        if self.meta_model_regression not in allowed_regression:
            raise ValueError(
                "stacking.meta_model_regression must be one of "
                f"{sorted(allowed_regression)}; got "
                f"{self.meta_model_regression!r}. No neural stacker is offered."
            )
        return self


class RobustnessConfiguration(BaseModel):
    """Missing-modality robustness and synthetic-dropout parameters."""

    model_config = {"extra": "forbid", "frozen": True}

    enabled: bool = True
    scenarios: tuple[str, ...] = ()
    synthetic_dropout_enabled: bool = False
    synthetic_dropout_seed: int = 42
    synthetic_dropout_probability: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.synthetic_dropout_enabled and self.synthetic_dropout_probability <= 0.0:
            raise ValueError(
                "robustness.synthetic_dropout_enabled requires a positive "
                "synthetic_dropout_probability; otherwise nothing is dropped "
                "and the setting is misleading"
            )
        return self


class FusionConfiguration(BaseModel):
    """Everything that defines the fusion behaviour of one run."""

    model_config = {"extra": "forbid", "frozen": True}

    strategies: tuple[FusionStrategy, ...]
    modalities: tuple[FusionModality, ...]
    minimum_modalities: int = Field(
        default=1,
        ge=1,
        description=(
            "Fewest modality experts that must have produced a prediction "
            "before a late-fusion result is emitted for a window."
        ),
    )
    expert_model_classification: str = "logistic_regression"
    expert_model_regression: str = "ridge"
    use_calibrated_experts: bool = True
    include_modality_quality_in_experts: bool = Field(
        default=False,
        description=(
            "Whether a modality expert may see its own modality-quality "
            "column. Off by default so quality does not silently enter "
            "every expert; quality's declared role is fusion weighting."
        ),
    )
    include_modality_quality_in_early_fusion: bool = Field(
        default=False,
        description=(
            "Whether the early-fusion matrix carries modality-quality "
            "columns. Off by default for the same reason. Per-feature "
            "availability and modality availability are always carried: "
            "they are how a missing modality is represented without "
            "fabricating a zero measurement."
        ),
    )
    quality: QualityWeightingConfiguration = Field(
        default_factory=QualityWeightingConfiguration
    )
    stacking: StackingConfiguration = Field(default_factory=StackingConfiguration)
    robustness: RobustnessConfiguration = Field(default_factory=RobustnessConfiguration)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.strategies:
            raise ValueError(
                "a fusion configuration must enable at least one strategy; "
                "an empty strategy set produces no fusion at all"
            )
        if len(set(self.strategies)) != len(self.strategies):
            duplicates = sorted(
                {s.value for s in self.strategies if list(self.strategies).count(s) > 1}
            )
            raise ValueError(f"duplicate fusion strategies: {duplicates}")
        if not self.modalities:
            raise ValueError("a fusion configuration must name at least one modality")
        if len(set(self.modalities)) != len(self.modalities):
            duplicates = sorted(
                {m.value for m in self.modalities if list(self.modalities).count(m) > 1}
            )
            raise ValueError(f"duplicate fusion modalities: {duplicates}")
        if len(self.modalities) < 2:
            raise ValueError(
                "fusion requires at least two modality groups; with one group "
                "there is nothing to fuse and the result is a unimodal model"
            )
        if self.minimum_modalities > len(self.modalities):
            raise ValueError(
                f"minimum_modalities={self.minimum_modalities} exceeds the "
                f"{len(self.modalities)} configured modalities; no window "
                "could ever satisfy it"
            )
        if self.stacking.enabled and FusionStrategy.STACKED_LATE not in self.strategies:
            raise ValueError(
                "stacking is enabled but 'stacked_late' is not among the "
                "enabled strategies; enable the strategy or disable stacking"
            )
        if FusionStrategy.STACKED_LATE in self.strategies and not self.stacking.enabled:
            raise ValueError(
                "strategy 'stacked_late' is requested but stacking.enabled is "
                "false; a stacked run must state its inner-fold design"
            )
        return self


class ModalityAvailability(BaseModel):
    """Whether one modality carried usable evidence, and how good it was."""

    model_config = {"extra": "forbid", "frozen": True}

    modality: FusionModality
    available: bool
    unavailable_reason: str | None = None
    quality: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_recorded: bool = False
    feature_count: int = Field(default=0, ge=0)
    measured_feature_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.available and not self.unavailable_reason:
            raise ValueError(
                f"modality {self.modality.value!r} is unavailable and must "
                "state a reason; silence is indistinguishable from a bug"
            )
        if self.quality is not None and not self.quality_recorded:
            raise ValueError(
                f"modality {self.modality.value!r} carries a quality value but "
                "quality_recorded is false"
            )
        return self


class ModalityPrediction(BaseModel):
    """One modality expert's output for one window.

    An expert that produced nothing carries ``available=False`` and a
    reason.  It never carries a zero prediction or a uniform probability
    vector standing in for a real one.
    """

    model_config = {"extra": "forbid", "frozen": True}

    modality: FusionModality
    available: bool
    unavailable_reason: str | None = None

    predicted_class: str | None = None
    class_vocabulary: tuple[str, ...] = ()
    probabilities: tuple[float, ...] = ()
    probabilities_are_calibrated: bool = False

    predicted_value: float | None = None
    quality: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        name = self.modality.value
        if not self.available:
            if not self.unavailable_reason:
                raise ValueError(
                    f"modality {name!r} produced no prediction and must state a reason"
                )
            if (
                self.predicted_class is not None
                or self.predicted_value is not None
                or self.probabilities
            ):
                raise ValueError(
                    f"modality {name!r} is unavailable but carries a "
                    "prediction; an unavailable modality must have no "
                    "effective prediction of any kind"
                )
            return self

        if (self.predicted_class is None) == (self.predicted_value is None):
            raise ValueError(
                f"modality {name!r} must carry exactly one of predicted_class "
                "(classification) or predicted_value (regression)"
            )
        if self.predicted_value is not None and not math.isfinite(self.predicted_value):
            raise ValueError(
                f"modality {name!r} produced a non-finite regression "
                "prediction; a model that cannot produce a finite value must "
                "report unavailable instead"
            )
        if self.predicted_class is not None:
            _assert_probability_vector(
                self.probabilities,
                self.class_vocabulary,
                context=f"modality {name!r}",
            )
            if self.predicted_class not in self.class_vocabulary:
                raise ValueError(
                    f"modality {name!r} predicted {self.predicted_class!r}, "
                    f"which is not in its class vocabulary "
                    f"{list(self.class_vocabulary)}"
                )
        elif self.probabilities:
            raise ValueError(
                f"modality {name!r} is a regression expert and must not carry "
                "class probabilities"
            )
        return self


class ModalityWeight(BaseModel):
    """The weight one modality received in one fused window.

    Raw and normalised weights are both recorded, together with the quality
    value that produced them and where that value came from, so a reader can
    reconstruct the arithmetic without re-running anything.
    """

    model_config = {"extra": "forbid", "frozen": True}

    modality: FusionModality
    base_weight: float = Field(gt=0.0)
    availability: float = Field(ge=0.0, le=1.0)
    quality_used: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_source: QualitySource = QualitySource.NOT_USED
    normalized_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_effective_weight: float = Field(ge=0.0)
    normalized_weight: float = Field(ge=0.0, le=1.0)
    contributed: bool
    exclusion_reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        name = self.modality.value
        for label, value in (
            ("base_weight", self.base_weight),
            ("raw_effective_weight", self.raw_effective_weight),
            ("normalized_weight", self.normalized_weight),
        ):
            if not math.isfinite(value):
                raise ValueError(f"modality {name!r}: {label} must be finite")
        if not self.contributed:
            if self.normalized_weight != 0.0:
                raise ValueError(
                    f"modality {name!r} did not contribute but carries "
                    f"normalized_weight={self.normalized_weight}. A modality "
                    "that produced no prediction must receive no effective "
                    "prediction weight."
                )
            if not self.exclusion_reason:
                raise ValueError(
                    f"modality {name!r} did not contribute and must state why"
                )
        elif self.availability <= 0.0:
            raise ValueError(
                f"modality {name!r} is marked as contributing but its "
                "availability is zero"
            )
        return self


class FusionPrediction(BaseModel):
    """One fused prediction for one window under one strategy.

    Everything needed to audit the fusion is on the record: which
    modalities were configured, which produced a prediction, which did not
    and why, the quality used, the weights before and after normalisation,
    and the fused output itself.
    """

    model_config = {"extra": "forbid", "frozen": True}

    window_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    task_type: TaskType
    fold_index: int = Field(ge=0)
    strategy: FusionStrategy
    scenario: str = Field(default="all_modalities", min_length=1)

    participating_modalities: tuple[FusionModality, ...]
    available_modalities: tuple[FusionModality, ...] = ()
    unavailable_modalities: tuple[FusionModality, ...] = ()
    modality_quality: dict[str, float | None] = Field(default_factory=dict)
    modality_predictions: tuple[ModalityPrediction, ...] = ()
    fusion_weights: tuple[ModalityWeight, ...] = ()

    fused: bool
    unavailable_reason: str | None = None
    predicted_class: str | None = None
    class_vocabulary: tuple[str, ...] = ()
    probabilities: tuple[float, ...] = ()
    predicted_value: float | None = None

    data_source: str = Field(min_length=1)
    is_synthetic: bool
    scientific_evaluation_eligible: bool

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"window {self.window_id!r} ({self.strategy.value})"
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                f"{where}: a synthetic prediction can never be scientifically eligible"
            )
        available = set(self.available_modalities)
        unavailable = set(self.unavailable_modalities)
        participating = set(self.participating_modalities)
        if available & unavailable:
            raise ValueError(
                f"{where}: modalities are both available and unavailable: "
                f"{sorted(m.value for m in available & unavailable)}"
            )
        stray = (available | unavailable) - participating
        if stray:
            raise ValueError(
                f"{where}: modalities outside the participating set: "
                f"{sorted(m.value for m in stray)}"
            )

        contributing = {w.modality for w in self.fusion_weights if w.contributed}
        if contributing - available:
            raise ValueError(
                f"{where}: weight given to modalities that produced no "
                f"prediction: {sorted(m.value for m in contributing - available)}"
            )

        if not self.fused:
            if not self.unavailable_reason:
                raise ValueError(f"{where}: an unfused window must state a reason")
            if (
                self.predicted_class is not None
                or self.predicted_value is not None
                or self.probabilities
            ):
                raise ValueError(
                    f"{where}: fusion was unavailable but a prediction is recorded"
                )
            return self

        if (self.predicted_class is None) == (self.predicted_value is None):
            raise ValueError(
                f"{where}: a fused window must carry exactly one of "
                "predicted_class or predicted_value"
            )
        if self.task_type is TaskType.CLASSIFICATION:
            if self.predicted_class is None:
                raise ValueError(
                    f"{where}: a classification target requires predicted_class"
                )
            _assert_probability_vector(
                self.probabilities, self.class_vocabulary, context=where
            )
            if self.predicted_class not in self.class_vocabulary:
                raise ValueError(
                    f"{where}: predicted class {self.predicted_class!r} is not "
                    f"in the vocabulary {list(self.class_vocabulary)}"
                )
        else:
            if self.predicted_value is None:
                raise ValueError(
                    f"{where}: a regression target requires predicted_value"
                )
            if not math.isfinite(self.predicted_value):
                raise ValueError(f"{where}: fused regression value is not finite")
            if self.probabilities:
                raise ValueError(
                    f"{where}: a regression fusion must not carry probabilities"
                )

        if self.fusion_weights:
            total = sum(w.normalized_weight for w in self.fusion_weights)
            if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
                raise ValueError(
                    f"{where}: normalised fusion weights sum to {total!r}, not "
                    "1.0; weights must be renormalised over the contributing "
                    "experts"
                )
        return self


class ExpertRecord(BaseModel):
    """What one modality expert was, and whether it could be trained."""

    model_config = {"extra": "forbid", "frozen": True}

    modality: FusionModality
    fold_index: int = Field(ge=0)
    model_name: str = Field(min_length=1)
    trained: bool
    unavailable_reason: str | None = None
    feature_names: tuple[str, ...] = ()
    fit_row_count: int = Field(default=0, ge=0)
    fit_group_count: int = Field(default=0, ge=0)
    calibration_row_count: int = Field(default=0, ge=0)
    calibration_group_count: int = Field(default=0, ge=0)
    calibrated: bool = False
    calibration_method: str | None = None
    calibration_unavailable_reason: str | None = None
    class_vocabulary: tuple[str, ...] = ()
    available_row_count: int = Field(default=0, ge=0)
    predicted_row_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.trained:
            if not self.unavailable_reason:
                raise ValueError(
                    f"expert {self.modality.value!r} was not trained and must "
                    "state why; refusing to train is a result, not silence"
                )
            if self.predicted_row_count:
                raise ValueError(
                    f"expert {self.modality.value!r} was not trained but "
                    "recorded predictions"
                )
        elif not self.feature_names:
            raise ValueError(
                f"expert {self.modality.value!r} was trained and must record "
                "the features it saw"
            )
        return self


class ExpertDocument(BaseModel):
    """All modality experts of one run, per fold."""

    model_config = {"extra": "forbid"}

    run_id: str = Field(min_length=1)
    evaluation_mode: EvaluationMode
    target_name: str = Field(min_length=1)
    task_type: TaskType
    experts: tuple[ExpertRecord, ...] = ()
    design: str = (
        "One estimator per modality, fitted on that modality's features "
        "only, inside the outer fold's fit groups. Calibrators, when "
        "requested, are fitted on the fold's calibration groups, which are "
        "disjoint from the fit groups and from the outer test groups."
    )
    disclaimers: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.disclaimers:
            raise ValueError("an expert document must carry at least one disclaimer")
        return self


class ExpertDisagreementSummary(BaseModel):
    """Interpretable disagreement diagnostics over the modality experts.

    This is a description of ensemble spread.  See :data:`DISAGREEMENT_NOTE`
    for what it is not.
    """

    model_config = {"extra": "forbid", "frozen": True}

    task_type: TaskType
    evaluated_window_count: int = Field(ge=0)
    insufficient_expert_window_count: int = Field(default=0, ge=0)

    mean_distinct_predicted_classes: float | None = None
    unanimous_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    disagreement_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_pairwise_probability_distance: float | None = Field(default=None, ge=0.0)
    mean_fused_probability_entropy: float | None = Field(default=None, ge=0.0)

    mean_prediction_standard_deviation: float | None = Field(default=None, ge=0.0)
    mean_prediction_range: float | None = Field(default=None, ge=0.0)

    unavailable_reason: str | None = None
    definitions: str = (
        "distinct_predicted_classes: number of different argmax labels among "
        "available experts. unanimous: every available expert predicted the "
        "same label. pairwise_probability_distance: mean Euclidean distance "
        "between every pair of available experts' probability vectors. "
        "fused_probability_entropy: Shannon entropy in nats of the fused "
        "probability vector. prediction_standard_deviation and "
        "prediction_range: population SD and max-min of the available "
        "experts' numeric predictions. Windows with fewer than two available "
        "experts are counted in insufficient_expert_window_count and "
        "contribute to no summary."
    )
    note: str = DISAGREEMENT_NOTE


class FusionDiagnostics(BaseModel):
    """Fusion-specific diagnostics for one fold or one aggregate."""

    model_config = {"extra": "forbid", "frozen": True}

    sample_count: int = Field(ge=0)
    fused_count: int = Field(ge=0)
    unavailable_fusion_count: int = Field(ge=0)
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    coverage_definition: str = (
        "coverage = fused_count / sample_count: the fraction of evaluated "
        "windows for which fusion produced a prediction at all."
    )
    mean_available_expert_count: float | None = Field(default=None, ge=0.0)
    missing_modality_rate: dict[str, float] = Field(default_factory=dict)
    modality_contribution_counts: dict[str, int] = Field(default_factory=dict)
    mean_normalized_weight: dict[str, float | None] = Field(default_factory=dict)
    weight_definition: str = (
        "mean_normalized_weight[m] is the mean, over the FUSED windows, of "
        "modality m's normalised fusion weight, counting zero for windows in "
        "which m contributed nothing. modality_contribution_counts[m] is the "
        "number of fused windows in which m carried a non-zero weight. "
        "missing_modality_rate[m] is the fraction of all evaluated windows in "
        "which m produced no prediction."
    )
    disagreement: ExpertDisagreementSummary | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.fused_count + self.unavailable_fusion_count != self.sample_count:
            raise ValueError(
                f"fused_count ({self.fused_count}) plus "
                f"unavailable_fusion_count ({self.unavailable_fusion_count}) "
                f"must equal sample_count ({self.sample_count})"
            )
        return self


class FusionFoldResult(BaseModel):
    """One strategy's result on one outer fold under one scenario."""

    model_config = {"extra": "forbid"}

    fold_index: int = Field(ge=0)
    strategy: FusionStrategy
    scenario: str = Field(default="all_modalities", min_length=1)
    evaluated: bool = True
    unavailable_reason: str | None = None
    classification_metrics: ClassificationMetrics | None = None
    regression_metrics: RegressionMetrics | None = None
    diagnostics: FusionDiagnostics | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.evaluated and not self.unavailable_reason:
            raise ValueError(
                f"fold {self.fold_index} of strategy {self.strategy.value!r} "
                "was not evaluated and must state a reason"
            )
        return self


class ValidationWeightRecord(BaseModel):
    """Late-fusion weights derived from inner validation groups.

    The groups used and the metric the weights came from are recorded, so a
    reader can confirm the weights were never derived from the outer test
    fold.
    """

    model_config = {"extra": "forbid", "frozen": True}

    fold_index: int = Field(ge=0)
    metric_name: str = Field(min_length=1)
    metric_definition: str = Field(min_length=1)
    groups_used: tuple[str, ...] = ()
    raw_scores: dict[str, float | None] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    fallback_applied: bool = False
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        for modality, weight in self.weights.items():
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"validation-derived weight for {modality!r} must be "
                    f"finite and non-negative; got {weight!r}"
                )
        if self.fallback_applied and not self.fallback_reason:
            raise ValueError(
                f"fold {self.fold_index}: a weighting fallback must state why "
                "it was applied"
            )
        return self


class StackingProvenanceRecord(BaseModel):
    """How one fold's stacked meta-model was trained."""

    model_config = {"extra": "forbid", "frozen": True}

    fold_index: int = Field(ge=0)
    available: bool
    unavailable_reason: str | None = None
    meta_model_name: str | None = None
    inner_fold_count: int = Field(default=0, ge=0)
    out_of_fold_row_count: int = Field(default=0, ge=0)
    meta_training_row_count: int = Field(default=0, ge=0)
    meta_training_group_count: int = Field(default=0, ge=0)
    outer_train_group_count: int = Field(default=0, ge=0)
    probabilities_are_calibrated: bool = False
    leakage_checks_passed: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.available and not self.unavailable_reason:
            raise ValueError(
                f"fold {self.fold_index}: an unavailable stacker must state a reason"
            )
        return self


class FusionStrategyResult(BaseModel):
    """One strategy across every outer fold, with aggregates."""

    model_config = {"extra": "forbid"}

    strategy: FusionStrategy
    description: str = Field(min_length=1)
    modalities: tuple[FusionModality, ...] = ()
    expert_model_name: str | None = None
    calibrated_experts: bool = False
    folds: tuple[FusionFoldResult, ...] = ()
    aggregate: tuple[AggregateMetric, ...] = ()
    fusion_aggregate: tuple[AggregateMetric, ...] = ()
    valid_fold_count: int = Field(default=0, ge=0)
    total_fold_count: int = Field(default=0, ge=0)
    failed_folds: dict[int, str] = Field(default_factory=dict)
    validation_weights: tuple[ValidationWeightRecord, ...] = ()
    stacking_provenance: tuple[StackingProvenanceRecord, ...] = ()
    notes: tuple[str, ...] = ()


class UnimodalControlResult(BaseModel):
    """Per-fold descriptive control: the strongest single-modality expert.

    This is **descriptive only**.  The modality reported per fold is chosen
    using that fold's own outer-test metric, which makes the value
    optimistically biased.  It exists so a reader can see roughly where the
    single modalities sat; it is never used to select a fusion strategy and
    it is never presented as a result.
    """

    model_config = {"extra": "forbid"}

    metric_name: str = Field(min_length=1)
    higher_is_better: bool
    per_fold_modality: dict[str, str] = Field(default_factory=dict)
    per_fold_value: dict[str, float | None] = Field(default_factory=dict)
    aggregate: AggregateMetric | None = None
    note: str = (
        "Descriptive control only. The reported modality is selected using "
        "the same outer fold it is scored on, so the value is optimistically "
        "biased and is not a fair comparator. It is never used to choose a "
        "fusion strategy."
    )


class MissingModalityScenario(BaseModel):
    """A deterministic evaluation scenario over modality availability."""

    model_config = {"extra": "forbid", "frozen": True}

    name: str = Field(min_length=1)
    absent_modalities: tuple[FusionModality, ...] = ()
    description: str = Field(min_length=1)

    def present(
        self, configured: tuple[FusionModality, ...]
    ) -> tuple[FusionModality, ...]:
        """Modalities this scenario leaves present, given a configuration."""
        absent = set(self.absent_modalities)
        return tuple(m for m in configured if m not in absent)


class RobustnessResult(BaseModel):
    """One strategy's behaviour under one missing-modality scenario."""

    model_config = {"extra": "forbid"}

    scenario_name: str = Field(min_length=1)
    scenario_description: str = Field(min_length=1)
    strategy: FusionStrategy
    present_modalities: tuple[FusionModality, ...] = ()
    absent_modalities: tuple[FusionModality, ...] = ()

    evaluated: bool = True
    unavailable_reason: str | None = None
    evaluated_window_count: int = Field(default=0, ge=0)
    fused_window_count: int = Field(default=0, ge=0)
    unavailable_fusion_count: int = Field(default=0, ge=0)
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    valid_fold_count: int = Field(default=0, ge=0)
    aggregate: tuple[AggregateMetric, ...] = ()
    diagnostics: FusionDiagnostics | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.evaluated and not self.unavailable_reason:
            raise ValueError(
                f"scenario {self.scenario_name!r} was not evaluated and must "
                "state a reason; an unevaluated scenario is never reported as "
                "a passing one"
            )
        if set(self.present_modalities) & set(self.absent_modalities):
            raise ValueError(
                f"scenario {self.scenario_name!r} lists modalities as both "
                "present and absent"
            )
        return self


class RobustnessDocument(BaseModel):
    """Every missing-modality scenario evaluated by one run."""

    model_config = {"extra": "forbid"}

    run_id: str = Field(min_length=1)
    evaluation_mode: EvaluationMode
    scientific_evaluation_eligible: bool
    target_name: str = Field(min_length=1)
    task_type: TaskType
    synthetic_dropout_applied: bool = False
    synthetic_dropout_seed: int | None = None
    synthetic_dropout_probability: float | None = None
    results: tuple[RobustnessResult, ...] = ()
    disclaimers: tuple[str, ...]
    note: str = (
        "Missing-modality behaviour measured here is a property of the "
        "software under deterministic scenarios. On SYNTHETIC data it is "
        "not a real-world robustness result and says nothing about how this "
        "system would behave on a person whose camera signal failed."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_self_check_document(self)
        return self


class FusionEvaluation(BaseModel):
    """Top-level fusion metrics document for one run."""

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
    modalities: tuple[FusionModality, ...] = ()
    strategies: tuple[FusionStrategyResult, ...] = ()
    unimodal_control: UnimodalControlResult | None = None
    disclaimers: tuple[str, ...]
    comparison_note: str = FUSION_COMPARISON_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_self_check_document(self)
        return self


class FusionExperimentManifest(BaseModel):
    """The fusion-specific configuration record, written as fusion_config.json.

    Run identity is deterministic: the same logical configuration on the
    same data reproduces the same ``run_id``.  No wall clock participates.
    """

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
    fold_count: int = Field(ge=0)
    random_seed: int

    fusion: FusionConfiguration
    calibration_method: str = Field(min_length=1)
    calibration_group_fraction: float = Field(ge=0.0, lt=1.0)
    early_fusion_columns: tuple[str, ...] = ()
    modality_columns: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    scenarios: tuple[MissingModalityScenario, ...] = ()

    run_id_inputs: str = (
        "dataset fingerprint, target, task type, random seed, split-manifest "
        "fingerprint, enabled strategies, modality groups, expert model "
        "types, calibration setting, quality-weighting configuration, "
        "missing-modality scenarios, and the EngageVR version. No wall clock "
        "and no random component participates."
    )
    disclaimers: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_self_check_document(self)
        return self


def _check_self_check_document(document: Any) -> None:
    """Shared self-check invariants for every fusion document."""
    if not document.disclaimers:
        raise ValueError("a fusion document must carry at least one disclaimer")
    if document.evaluation_mode is EvaluationMode.SOFTWARE_SELF_CHECK:
        if getattr(document, "scientific_evaluation_eligible", False):
            raise ValueError(
                "a software self-check can never be scientifically eligible"
            )
        if not any(SOFTWARE_SELF_CHECK_BANNER in d for d in document.disclaimers):
            raise ValueError(
                "a software self-check document must carry the banner "
                f"{SOFTWARE_SELF_CHECK_BANNER!r}"
            )


def _assert_probability_vector(
    probabilities: tuple[float, ...],
    vocabulary: tuple[str, ...],
    *,
    context: str,
) -> None:
    """Assert a probability vector is finite, non-negative, and sums to one."""
    if not probabilities:
        raise ValueError(
            f"{context}: a classification prediction requires probabilities"
        )
    if len(probabilities) != len(vocabulary):
        raise ValueError(
            f"{context}: {len(probabilities)} probabilities were supplied for "
            f"{len(vocabulary)} classes"
        )
    for value in probabilities:
        if not math.isfinite(value):
            raise ValueError(f"{context}: probability {value!r} is not finite")
        if value < 0.0:
            raise ValueError(f"{context}: probability {value!r} is negative")
    total = sum(probabilities)
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(
            f"{context}: probabilities sum to {total!r}, not 1.0; a "
            "probability vector that does not sum to one is not a "
            "distribution"
        )


__all__ = [
    "DISAGREEMENT_NOTE",
    "FUSION_COMPARISON_NOTE",
    "LATE_FUSION_STRATEGIES",
    "PROBABILITY_SUM_TOLERANCE",
    "QUALITY_NOTE",
    "ExpertDisagreementSummary",
    "ExpertDocument",
    "ExpertRecord",
    "FusionConfiguration",
    "FusionDiagnostics",
    "FusionEvaluation",
    "FusionExperimentManifest",
    "FusionFoldResult",
    "FusionModality",
    "FusionPrediction",
    "FusionStrategy",
    "FusionStrategyResult",
    "MissingModalityScenario",
    "MissingQualityPolicy",
    "ModalityAvailability",
    "ModalityPrediction",
    "ModalityWeight",
    "QualitySource",
    "QualityWeightingConfiguration",
    "RobustnessConfiguration",
    "RobustnessDocument",
    "RobustnessResult",
    "StackingConfiguration",
    "StackingProvenanceRecord",
    "UnimodalControlResult",
    "ValidationWeightRecord",
]
