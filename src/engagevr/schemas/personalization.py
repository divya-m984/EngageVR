"""Personalization schemas (Milestone 6, acceptance criterion 3).

These models are the persisted form of a personalization run.  They are
written as JSON beside the run's artifacts so a personalized result can be
inspected, and its provenance judged, **without loading any model file**.

What this module encodes structurally rather than by convention
---------------------------------------------------------------
1. *A personal baseline is estimated only from windows that precede the
   evaluation windows in time.*  :class:`PersonalCalibrationSplit` carries
   the calibration window ids, the evaluation window ids, both boundary
   timestamps, and a validator that refuses a split whose calibration
   region does not end before its evaluation region begins.
2. *Cold start is a stated outcome, never silence.*  A prediction with
   ``personalization_applied=false`` must carry a reason, and a cold-start
   prediction must reproduce the population prediction exactly.
3. *The population prediction is never overwritten.*  Every
   :class:`PersonalizedPrediction` carries both, so the population-only
   report and the personalized report describe the same windows.
4. *Missing evidence refuses rather than substitutes.*  A correction that
   could not be estimated is ``available=false`` with a reason; it never
   carries a zero bias or a zero log-odds shift standing in for a real one.
5. *Synthetic can never be scientifically eligible*, and a self-check
   document must carry the self-check banner — the same validator contract
   as :mod:`engagevr.schemas.experiments` and
   :mod:`engagevr.schemas.fusion`.

Personalized *calibration* here means adapting a population model to one
subject.  It is **not** uncertainty calibration, it is not a confidence
estimate, and nothing in this module abstains.  Confidence-based and
selective-prediction thresholds are Milestone 7.
"""

from __future__ import annotations

import enum
import math
from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from engagevr.schemas.experiments import (
    SOFTWARE_SELF_CHECK_BANNER,
    AggregateMetric,
    ClassificationMetrics,
    EvaluationMode,
    RegressionMetrics,
)
from engagevr.schemas.fusion import PROBABILITY_SUM_TOLERANCE, FusionModality
from engagevr.schemas.targets import TaskType

#: Note attached to every personalization document.
PERSONALIZATION_NOTE = (
    "Population and personalized results are reported SEPARATELY over "
    "identical evaluation windows. A difference between them computed on "
    "SYNTHETIC data is not evidence of a personalization benefit: it "
    "describes the generator this repository wrote, not any person. No "
    "personalized model here is validated, none is a champion, and none "
    "has been fitted to a real participant label."
)

#: Note distinguishing subject adaptation from uncertainty calibration.
CALIBRATION_MEANING_NOTE = (
    "'Calibration' in this document means adapting a population model to "
    "one subject from that subject's own earlier windows. It is NOT "
    "uncertainty calibration, NOT a confidence estimate, and NOT signal "
    "quality. Probability calibration of the population model is a "
    "separate, earlier step recorded in calibration.json. Confidence-based "
    "and selective-prediction thresholds are Milestone 7."
)

#: The documented z-score used for personal-baseline normalization.
PERSONAL_BASELINE_EQUATION = (
    "z_s(x) = (x - mu_s) / sigma_s, where mu_s and sigma_s are the mean and "
    "population standard deviation of feature x over subject s's permitted "
    "calibration windows only. A missing measurement stays missing. When "
    "sigma_s is at or below the zero-variance epsilon the value is centred "
    "and the scale is fixed at 1.0, recorded as "
    "scale_source='unit_scale_zero_variance'; it is never divided by ~0. "
    "When the subject has fewer than the minimum number of finite "
    "calibration values for x, the feature is passed through unchanged "
    "(mu_s=0, sigma_s=1) and recorded as not normalized, with the reason."
)

#: The documented regression correction.
REGRESSION_CORRECTION_EQUATION = (
    "b_s = mean over subject s's labelled calibration windows of "
    "(y_calibration - y_population_prediction); "
    "y_personalized = y_population_prediction + b_s. "
    "Evaluation labels take no part in b_s."
)

#: The documented classification correction.
CLASSIFICATION_CORRECTION_EQUATION = (
    "Regularised per-subject log-odds shift. With K classes, n labelled "
    "calibration windows, smoothing alpha and shrinkage constant kappa: "
    "observed_c = (count_c + alpha) / (n + alpha*K); "
    "expected_c = (sum_w p_population_c(w) + alpha) / (n + alpha*K); "
    "lambda = n / (n + kappa); "
    "delta_c = lambda * (log(observed_c) - log(expected_c)); "
    "p_personalized_c = p_population_c * exp(delta_c) / "
    "sum_k p_population_k * exp(delta_k). "
    "Both terms are smoothed identically, so delta_c is exactly zero when "
    "the subject's calibration labels match the population model's average "
    "prediction. lambda grows with calibration evidence and is zero when "
    "there is none. Evaluation labels take no part in delta_c."
)


class PersonalizationMethod(enum.StrEnum):
    """How a population model is adapted to one subject.

    No member fits a subject-specific model from scratch.  A handful of
    calibration windows cannot support one, and pretending otherwise would
    produce a per-subject estimator that describes its own calibration set.
    """

    POPULATION_ONLY = "population_only"
    PERSONAL_BASELINE = "personal_baseline"
    FEW_SHOT_CORRECTION = "few_shot_correction"
    PERSONAL_BASELINE_AND_CORRECTION = "personal_baseline_and_correction"
    COLD_START = "cold_start"


#: Methods that normalise features against a personal baseline.
BASELINE_METHODS: frozenset[PersonalizationMethod] = frozenset(
    {
        PersonalizationMethod.PERSONAL_BASELINE,
        PersonalizationMethod.PERSONAL_BASELINE_AND_CORRECTION,
    }
)

#: Methods that fit a supervised per-subject correction.
CORRECTION_METHODS: frozenset[PersonalizationMethod] = frozenset(
    {
        PersonalizationMethod.FEW_SHOT_CORRECTION,
        PersonalizationMethod.PERSONAL_BASELINE_AND_CORRECTION,
    }
)

#: Methods a configuration may request. ``COLD_START`` is an *outcome*: it
#: is what a run records when a requested personalization could not be
#: applied to a subject, so requesting it directly would be misleading.
REQUESTABLE_METHODS: frozenset[PersonalizationMethod] = frozenset(
    {
        PersonalizationMethod.POPULATION_ONLY,
        PersonalizationMethod.PERSONAL_BASELINE,
        PersonalizationMethod.FEW_SHOT_CORRECTION,
        PersonalizationMethod.PERSONAL_BASELINE_AND_CORRECTION,
    }
)


class PersonalizationConfiguration(BaseModel):
    """Everything that defines the personalization behaviour of one run."""

    model_config = {"extra": "forbid", "frozen": True}

    method: PersonalizationMethod = (
        PersonalizationMethod.PERSONAL_BASELINE_AND_CORRECTION
    )
    modalities: tuple[FusionModality, ...] = tuple(FusionModality)

    calibration_windows: int = Field(
        default=5,
        ge=0,
        description=(
            "Number of a held-out subject's earliest windows reserved for "
            "personal calibration. Zero requests cold-start mode: no "
            "calibration evidence is taken, every window is evaluated under "
            "the population model, and every subject is recorded as a cold "
            "start."
        ),
    )
    minimum_calibration_windows: int = Field(
        default=3,
        ge=1,
        description=(
            "Fewest labelled calibration windows before a supervised "
            "correction is fitted. Below it the subject falls back to the "
            "population model and is recorded as a cold start. An "
            "engineering default, not a validated threshold."
        ),
    )
    minimum_evaluation_windows: int = Field(
        default=1,
        ge=1,
        description=(
            "Fewest windows that must remain strictly after the calibration "
            "boundary before a subject can be evaluated at all."
        ),
    )
    minimum_calibration_classes: int = Field(
        default=2,
        ge=1,
        description=(
            "Fewest distinct classes the calibration labels must contain "
            "before the classification correction is fitted. With one class "
            "the shift is driven entirely by the absence of the others."
        ),
    )
    minimum_baseline_samples: int = Field(
        default=3,
        ge=1,
        description=(
            "Fewest finite calibration values a feature needs before a "
            "personal mean and scale are estimated for it."
        ),
    )
    zero_variance_epsilon: float = Field(
        default=1e-9,
        gt=0.0,
        lt=1.0,
        description=(
            "Standard deviation at or below which a feature is centred but "
            "not scaled. A numerical guard against dividing by ~0, not a "
            "modelling threshold."
        ),
    )
    classification_smoothing: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Laplace smoothing alpha applied identically to the observed "
            "class counts and to the population model's expected counts, so "
            "the shift is exactly zero when they agree."
        ),
    )
    classification_shrinkage_constant: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Shrinkage constant kappa in lambda = n / (n + kappa). Larger "
            "values trust a small calibration set less. An engineering "
            "default, not a validated value."
        ),
    )

    population_model_classification: str = "logistic_regression"
    population_model_regression: str = "ridge"
    use_calibrated_population_model: bool = True
    include_modality_quality: bool = Field(
        default=False,
        description=(
            "Whether modality-quality columns enter the population "
            "predictor matrix. Off by default: quality describes the "
            "measurement, and it is never personalised as a physiological "
            "value."
        ),
    )

    population_design: str = (
        "The population reference model is the early-fusion estimator over "
        "the configured modality groups: the permitted features of every "
        "group are concatenated into one matrix, with modality availability "
        "and per-feature missingness carried as separate columns. It is "
        "fitted only on subjects other than the held-out subject. "
        "Personalization layers on top of its output; no fusion weight is "
        "retuned, and the population prediction is retained unchanged."
    )
    baseline_equation: str = PERSONAL_BASELINE_EQUATION
    regression_correction_equation: str = REGRESSION_CORRECTION_EQUATION
    classification_correction_equation: str = CLASSIFICATION_CORRECTION_EQUATION
    note: str = CALIBRATION_MEANING_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.method not in REQUESTABLE_METHODS:
            valid = sorted(m.value for m in REQUESTABLE_METHODS)
            raise ValueError(
                f"personalization.method {self.method.value!r} cannot be "
                f"requested; valid: {valid}. 'cold_start' is an outcome a run "
                "records when personalization could not be applied to a "
                "subject, not a method to ask for. Request cold-start "
                "behaviour with calibration_windows=0."
            )
        if not self.modalities:
            raise ValueError(
                "personalization.modalities must name at least one modality group"
            )
        if len(set(self.modalities)) != len(self.modalities):
            duplicates = sorted(
                {m.value for m in self.modalities if list(self.modalities).count(m) > 1}
            )
            raise ValueError(f"duplicate personalization modalities: {duplicates}")
        if len(self.modalities) < 2:
            raise ValueError(
                "personalization layers on the fused population model, which "
                "requires at least two modality groups; with one group there "
                "is nothing to fuse"
            )
        if (
            self.calibration_windows
            and self.calibration_windows < self.minimum_calibration_windows
            and self.method in CORRECTION_METHODS
        ):
            raise ValueError(
                f"personalization.calibration_windows "
                f"({self.calibration_windows}) is below "
                f"minimum_calibration_windows "
                f"({self.minimum_calibration_windows}), so method "
                f"{self.method.value!r} could never be applied to any "
                "subject. Raise the window count or lower the minimum."
            )
        return self


class PersonalCalibrationSplit(BaseModel):
    """One held-out subject's chronological calibration/evaluation split.

    The two regions are separated by a boundary in wall-clock time, not by
    a row count: with overlapping windows a purely positional split would
    put a calibration window whose interval extends past the boundary on
    the wrong side of it.  Windows that straddle the boundary are excluded
    from both regions and listed.
    """

    model_config = {"extra": "forbid", "frozen": True}

    subject_id: str = Field(min_length=1)
    fold_index: int = Field(ge=0)
    session_ids: tuple[str, ...] = ()
    total_window_count: int = Field(ge=0)

    available: bool
    unavailable_reason: str | None = None
    cold_start: bool = False
    cold_start_reason: str | None = None

    calibration_window_ids: tuple[str, ...] = ()
    evaluation_window_ids: tuple[str, ...] = ()
    excluded_overlap_window_ids: tuple[str, ...] = ()

    calibration_start_utc: datetime | None = None
    calibration_end_utc: datetime | None = None
    evaluation_start_utc: datetime | None = None
    evaluation_end_utc: datetime | None = None

    windows_overlap: bool = False
    temporal_order_verified: bool = False
    ordering_rule: str = (
        "A subject's windows are ordered by (window_start_utc, "
        "window_end_utc, window_index, window_id). The first "
        "calibration_windows of them form the calibration region. The "
        "boundary is the latest calibration window_end_utc. A later window "
        "joins the evaluation region only if its window_start_utc is at or "
        "after that boundary; a window that straddles the boundary is "
        "excluded from both regions and listed in "
        "excluded_overlap_window_ids. Windows are never mixed at random "
        "between the two regions."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"subject {self.subject_id!r} (fold {self.fold_index})"
        if not self.available:
            if not self.unavailable_reason:
                raise ValueError(
                    f"{where}: an unavailable calibration split must state a reason"
                )
            if self.calibration_window_ids or self.evaluation_window_ids:
                raise ValueError(
                    f"{where}: an unavailable calibration split must carry no "
                    "calibration or evaluation windows"
                )
            return self

        if self.cold_start and not self.cold_start_reason:
            raise ValueError(f"{where}: a cold start must state why")
        if not self.evaluation_window_ids:
            raise ValueError(
                f"{where}: an available split must carry at least one evaluation window"
            )
        overlap = set(self.calibration_window_ids) & set(self.evaluation_window_ids)
        if overlap:
            raise ValueError(
                f"{where}: window(s) {sorted(overlap)[:5]} appear in both the "
                "calibration and the evaluation region"
            )
        stray = set(self.excluded_overlap_window_ids) & (
            set(self.calibration_window_ids) | set(self.evaluation_window_ids)
        )
        if stray:
            raise ValueError(
                f"{where}: window(s) {sorted(stray)[:5]} are recorded both as "
                "excluded and as used"
            )
        if self.calibration_window_ids:
            if self.calibration_end_utc is None or self.evaluation_start_utc is None:
                raise ValueError(
                    f"{where}: a split with calibration windows must record "
                    "both boundary timestamps so temporal ordering is provable"
                )
            if self.calibration_end_utc > self.evaluation_start_utc:
                raise ValueError(
                    f"{where}: the calibration region ends at "
                    f"{self.calibration_end_utc.isoformat()}, after the "
                    f"evaluation region begins at "
                    f"{self.evaluation_start_utc.isoformat()}. Calibration "
                    "must precede evaluation in time; otherwise the personal "
                    "baseline is estimated from the future."
                )
            if not self.temporal_order_verified:
                raise ValueError(
                    f"{where}: a split with calibration windows must record "
                    "temporal_order_verified=true"
                )
        elif not self.cold_start:
            raise ValueError(
                f"{where}: a split with no calibration window is a cold start "
                "and must be recorded as one"
            )
        return self


class PersonalBaselineStatistics(BaseModel):
    """One subject's personal baseline for one feature.

    Feature identity, unit, and modality travel with the numbers so a
    reader can see what was normalised without consulting the catalogue.
    """

    model_config = {"extra": "forbid", "frozen": True}

    subject_id: str = Field(min_length=1)
    fold_index: int = Field(ge=0)
    column: str = Field(min_length=1)
    feature_name: str = Field(min_length=1)
    modality: FusionModality
    unit: str = Field(min_length=1)

    normalized: bool
    unavailable_reason: str | None = None
    calibration_sample_count: int = Field(ge=0)
    finite_sample_count: int = Field(ge=0)
    mean: float
    scale: float = Field(gt=0.0)
    observed_standard_deviation: float | None = None
    scale_source: str = Field(min_length=1)
    source_window_ids: tuple[str, ...] = ()
    equation: str = PERSONAL_BASELINE_EQUATION

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"subject {self.subject_id!r} feature {self.column!r}"
        if not math.isfinite(self.mean) or not math.isfinite(self.scale):
            raise ValueError(f"{where}: mean and scale must both be finite")
        if not self.normalized:
            if not self.unavailable_reason:
                raise ValueError(
                    f"{where}: a feature that was not normalized must state why"
                )
            if self.mean != 0.0 or self.scale != 1.0:
                raise ValueError(
                    f"{where}: a feature that was not normalized must carry the "
                    "identity transform (mean 0.0, scale 1.0); anything else "
                    "would silently rescale it"
                )
        return self


class PopulationPrediction(BaseModel):
    """The population reference model's output for one window."""

    model_config = {"extra": "forbid", "frozen": True}

    window_id: str = Field(min_length=1)
    available: bool = True
    unavailable_reason: str | None = None
    predicted_class: str | None = None
    class_vocabulary: tuple[str, ...] = ()
    probabilities: tuple[float, ...] = ()
    probabilities_are_calibrated: bool = False
    predicted_value: float | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_prediction_payload(self, context=f"window {self.window_id!r}")
        return self


class PersonalizationCorrection(BaseModel):
    """The per-subject correction fitted from calibration windows only.

    ``calibration_targets`` records the label used for each calibration
    window by window id.  It is the audit trail that makes the leakage
    claim checkable: any evaluation window id appearing here would be a
    defect, and a test asserts none does.
    """

    model_config = {"extra": "forbid", "frozen": True}

    subject_id: str = Field(min_length=1)
    fold_index: int = Field(ge=0)
    method: PersonalizationMethod
    task_type: TaskType

    available: bool
    unavailable_reason: str | None = None
    supervised: bool = False

    calibration_sample_count: int = Field(ge=0)
    calibration_window_ids: tuple[str, ...] = ()
    calibration_targets: dict[str, str] = Field(default_factory=dict)
    calibration_class_support: dict[str, int] = Field(default_factory=dict)

    bias: float | None = None
    log_odds_shift: dict[str, float] = Field(default_factory=dict)
    shrinkage: float | None = Field(default=None, ge=0.0, le=1.0)
    smoothing: float | None = Field(default=None, gt=0.0)
    equation: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"subject {self.subject_id!r} (fold {self.fold_index})"
        if not self.available:
            if not self.unavailable_reason:
                raise ValueError(
                    f"{where}: an unavailable correction must state a reason"
                )
            if self.bias is not None or self.log_odds_shift:
                raise ValueError(
                    f"{where}: an unavailable correction must carry no "
                    "parameters; a zero bias would be indistinguishable from a "
                    "fitted one"
                )
            return self
        if not self.equation:
            raise ValueError(
                f"{where}: an applied correction must record its exact equation"
            )
        if self.task_type is TaskType.REGRESSION:
            if self.bias is None or not math.isfinite(self.bias):
                raise ValueError(
                    f"{where}: a regression correction must record a finite bias"
                )
            if self.log_odds_shift:
                raise ValueError(
                    f"{where}: a regression correction must not carry a log-odds shift"
                )
        else:
            if not self.log_odds_shift:
                raise ValueError(
                    f"{where}: a classification correction must record a "
                    "log-odds shift per class"
                )
            for label, value in self.log_odds_shift.items():
                if not math.isfinite(value):
                    raise ValueError(
                        f"{where}: log-odds shift for class {label!r} is not finite"
                    )
            if self.bias is not None:
                raise ValueError(
                    f"{where}: a classification correction must not carry a "
                    "regression bias"
                )
        if not self.calibration_window_ids:
            raise ValueError(
                f"{where}: an applied correction must record the calibration "
                "windows it was fitted on"
            )
        return self


class PersonalizedPrediction(BaseModel):
    """One evaluation window, with both the population and personalized output.

    The population prediction is preserved on every record.  A personalized
    prediction is an addition to it, never a replacement of it.
    """

    model_config = {"extra": "forbid", "frozen": True}

    window_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    task_type: TaskType
    fold_index: int = Field(ge=0)
    method: PersonalizationMethod

    population_predicted_class: str | None = None
    population_probabilities: tuple[float, ...] = ()
    population_predicted_value: float | None = None

    personalized_predicted_class: str | None = None
    personalized_probabilities: tuple[float, ...] = ()
    personalized_predicted_value: float | None = None

    class_vocabulary: tuple[str, ...] = ()
    probabilities_are_calibrated: bool = False

    personalization_applied: bool
    unavailable_reason: str | None = None
    cold_start: bool
    cold_start_reason: str | None = None
    baseline_normalized: bool = False
    supervised_correction_applied: bool = False
    normalized_feature_count: int = Field(default=0, ge=0)

    calibration_window_ids: tuple[str, ...] = ()
    calibration_sample_count: int = Field(ge=0)

    data_source: str = Field(min_length=1)
    is_synthetic: bool
    scientific_evaluation_eligible: bool

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"window {self.window_id!r} (subject {self.subject_id!r})"
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                f"{where}: a synthetic prediction can never be scientifically eligible"
            )
        if self.window_id in set(self.calibration_window_ids):
            raise ValueError(
                f"{where}: the evaluation window is also listed as one of its "
                "own calibration windows; the correction would have seen this "
                "window's label"
            )
        if self.cold_start and not self.cold_start_reason:
            raise ValueError(f"{where}: a cold start must state why")
        if not self.personalization_applied and not self.unavailable_reason:
            raise ValueError(
                f"{where}: personalization was not applied and must state why"
            )

        if self.task_type is TaskType.CLASSIFICATION:
            for label, predicted, probabilities in (
                (
                    "population",
                    self.population_predicted_class,
                    self.population_probabilities,
                ),
                (
                    "personalized",
                    self.personalized_predicted_class,
                    self.personalized_probabilities,
                ),
            ):
                if predicted is None:
                    raise ValueError(
                        f"{where}: the {label} prediction is missing a class"
                    )
                _assert_distribution(
                    probabilities,
                    self.class_vocabulary,
                    context=f"{where} {label}",
                )
                if predicted not in self.class_vocabulary:
                    raise ValueError(
                        f"{where}: the {label} class {predicted!r} is not in "
                        f"the vocabulary {list(self.class_vocabulary)}"
                    )
            if (
                self.population_predicted_value is not None
                or self.personalized_predicted_value is not None
            ):
                raise ValueError(
                    f"{where}: a classification record must not carry a "
                    "numeric prediction"
                )
        else:
            for label, value in (
                ("population", self.population_predicted_value),
                ("personalized", self.personalized_predicted_value),
            ):
                if value is None or not math.isfinite(value):
                    raise ValueError(
                        f"{where}: the {label} regression prediction must be "
                        "present and finite"
                    )
            if self.population_probabilities or self.personalized_probabilities:
                raise ValueError(
                    f"{where}: a regression record must not carry probabilities"
                )

        if not self.personalization_applied:
            if self.task_type is TaskType.CLASSIFICATION:
                same = (
                    self.personalized_predicted_class == self.population_predicted_class
                    and self.personalized_probabilities == self.population_probabilities
                )
            else:
                same = (
                    self.personalized_predicted_value == self.population_predicted_value
                )
            if not same:
                raise ValueError(
                    f"{where}: personalization was not applied, so the "
                    "personalized output must reproduce the population output "
                    "exactly; a cold start uses the population model, it does "
                    "not invent a personal one"
                )
        return self


class PersonalizationFoldResult(BaseModel):
    """One outer fold: population and personalized metrics on the same rows."""

    model_config = {"extra": "forbid"}

    fold_index: int = Field(ge=0)
    evaluated: bool = True
    unavailable_reason: str | None = None

    population_training_subject_count: int = Field(default=0, ge=0)
    held_out_subject_count: int = Field(default=0, ge=0)
    evaluated_subject_count: int = Field(default=0, ge=0)
    personalized_subject_count: int = Field(default=0, ge=0)
    cold_start_subject_count: int = Field(default=0, ge=0)
    unavailable_subject_count: int = Field(default=0, ge=0)

    calibration_window_count: int = Field(default=0, ge=0)
    evaluation_window_count: int = Field(default=0, ge=0)
    excluded_overlap_window_count: int = Field(default=0, ge=0)

    population_classification_metrics: ClassificationMetrics | None = None
    personalized_classification_metrics: ClassificationMetrics | None = None
    population_regression_metrics: RegressionMetrics | None = None
    personalized_regression_metrics: RegressionMetrics | None = None

    splits: tuple[PersonalCalibrationSplit, ...] = ()
    corrections: tuple[PersonalizationCorrection, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.evaluated and not self.unavailable_reason:
            raise ValueError(
                f"fold {self.fold_index} was not evaluated and must state a reason"
            )
        for name, population, personalized in (
            (
                "classification",
                self.population_classification_metrics,
                self.personalized_classification_metrics,
            ),
            (
                "regression",
                self.population_regression_metrics,
                self.personalized_regression_metrics,
            ),
        ):
            if population is None or personalized is None:
                continue
            if population.sample_count != personalized.sample_count:
                raise ValueError(
                    f"fold {self.fold_index}: the population {name} metrics "
                    f"cover {population.sample_count} rows and the "
                    f"personalized metrics cover {personalized.sample_count}. "
                    "Both must be computed over exactly the same evaluation "
                    "windows, otherwise the comparison is between different "
                    "data."
                )
        return self


class PersonalizationEvaluation(BaseModel):
    """Top-level personalization document for one run."""

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

    configuration: PersonalizationConfiguration
    predictor_columns: tuple[str, ...] = ()
    personalized_columns: tuple[str, ...] = ()

    folds: tuple[PersonalizationFoldResult, ...] = ()
    population_aggregate: tuple[AggregateMetric, ...] = ()
    personalized_aggregate: tuple[AggregateMetric, ...] = ()

    total_calibration_window_count: int = Field(default=0, ge=0)
    total_evaluation_window_count: int = Field(default=0, ge=0)
    total_excluded_overlap_window_count: int = Field(default=0, ge=0)
    cold_start_subject_count: int = Field(default=0, ge=0)
    personalized_subject_count: int = Field(default=0, ge=0)
    unavailable_personalization_count: int = Field(default=0, ge=0)
    personalization_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    coverage_definition: str = (
        "personalization_coverage = personalized_subject_count / "
        "(personalized_subject_count + cold_start_subject_count), over the "
        "subject-fold pairs that were evaluated at all. A subject that could "
        "not be split into a calibration and an evaluation region is counted "
        "in unavailable_personalization_count and enters neither term."
    )

    disclaimers: tuple[str, ...]
    comparison_note: str = PERSONALIZATION_NOTE
    calibration_meaning_note: str = CALIBRATION_MEANING_NOTE
    deferred_note: str = (
        "Personalized confidence thresholds, selective prediction, and "
        "abstention are NOT implemented here and are deferred to Milestone "
        "7. Nothing in this document withholds a prediction."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_self_check_document(self)
        return self


class PersonalBaselineDocument(BaseModel):
    """Every persisted personal baseline of one run.

    Only held-out subjects' baselines are persisted.  A training subject's
    baseline has no calibration/evaluation boundary to audit — every one of
    their windows is training data — so recording thousands of them would
    add bulk without adding evidence.
    """

    model_config = {"extra": "forbid"}

    run_id: str = Field(min_length=1)
    evaluation_mode: EvaluationMode
    target_name: str = Field(min_length=1)
    method: PersonalizationMethod
    statistics: tuple[PersonalBaselineStatistics, ...] = ()
    scope: str = (
        "Held-out subjects only. A training subject's baseline is estimated "
        "from all of that subject's training-fold windows, because they are "
        "training data with no evaluation region to protect; only a held-out "
        "subject has a boundary worth auditing."
    )
    equation: str = PERSONAL_BASELINE_EQUATION
    excluded_column_rule: str = (
        "Identifiers, timestamps, target columns, target provenance, "
        "availability flags, modality-availability flags, "
        "modality-quality columns, and categorical provenance fields are "
        "never personalised. Only catalogued measured features of the "
        "configured measurement modalities are. Signal quality describes "
        "the measurement and is not turned into a normalised physiological "
        "value."
    )
    disclaimers: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.disclaimers:
            raise ValueError("a baseline document must carry at least one disclaimer")
        return self


class PersonalizationExperimentManifest(BaseModel):
    """The personalization-specific configuration record.

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

    configuration: PersonalizationConfiguration
    calibration_method: str = Field(min_length=1)
    calibration_group_fraction: float = Field(ge=0.0, lt=1.0)
    predictor_columns: tuple[str, ...] = ()
    personalized_columns: tuple[str, ...] = ()

    run_id_inputs: str = (
        "dataset fingerprint, target, task type, random seed, "
        "split-manifest fingerprint, personalization method, modality "
        "groups, calibration-window count, every minimum-evidence and "
        "smoothing constant, the population model types, the probability "
        "calibration setting, and the EngageVR version. No wall clock and "
        "no random component participates."
    )
    disclaimers: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_self_check_document(self)
        return self


def _check_self_check_document(document: Any) -> None:
    """Shared self-check invariants for every personalization document."""
    if not document.disclaimers:
        raise ValueError(
            "a personalization document must carry at least one disclaimer"
        )
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


def _check_prediction_payload(prediction: Any, *, context: str) -> None:
    """Shared availability/payload invariants for a single-model prediction."""
    if not prediction.available:
        if not prediction.unavailable_reason:
            raise ValueError(
                f"{context}: an unavailable prediction must state a reason"
            )
        if (
            prediction.predicted_class is not None
            or prediction.predicted_value is not None
            or prediction.probabilities
        ):
            raise ValueError(
                f"{context}: an unavailable prediction must carry no output"
            )
        return
    if (prediction.predicted_class is None) == (prediction.predicted_value is None):
        raise ValueError(
            f"{context}: exactly one of predicted_class or predicted_value is required"
        )
    if prediction.predicted_value is not None:
        if not math.isfinite(prediction.predicted_value):
            raise ValueError(f"{context}: the regression prediction is not finite")
        if prediction.probabilities:
            raise ValueError(
                f"{context}: a regression prediction must not carry probabilities"
            )
        return
    _assert_distribution(
        prediction.probabilities, prediction.class_vocabulary, context=context
    )
    if prediction.predicted_class not in prediction.class_vocabulary:
        raise ValueError(
            f"{context}: predicted class {prediction.predicted_class!r} is not "
            f"in the vocabulary {list(prediction.class_vocabulary)}"
        )


def _assert_distribution(
    probabilities: tuple[float, ...],
    vocabulary: tuple[str, ...],
    *,
    context: str,
) -> None:
    """Assert a probability vector is finite, non-negative, and sums to one."""
    if not probabilities:
        raise ValueError(f"{context}: a class prediction requires probabilities")
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
            "probability vector that does not sum to one is not a distribution"
        )


__all__ = [
    "BASELINE_METHODS",
    "CALIBRATION_MEANING_NOTE",
    "CLASSIFICATION_CORRECTION_EQUATION",
    "CORRECTION_METHODS",
    "PERSONALIZATION_NOTE",
    "PERSONAL_BASELINE_EQUATION",
    "REGRESSION_CORRECTION_EQUATION",
    "REQUESTABLE_METHODS",
    "PersonalBaselineDocument",
    "PersonalBaselineStatistics",
    "PersonalCalibrationSplit",
    "PersonalizationConfiguration",
    "PersonalizationCorrection",
    "PersonalizationEvaluation",
    "PersonalizationExperimentManifest",
    "PersonalizationFoldResult",
    "PersonalizationMethod",
    "PersonalizedPrediction",
    "PopulationPrediction",
]
