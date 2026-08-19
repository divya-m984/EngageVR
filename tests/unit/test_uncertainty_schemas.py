"""Milestone 7 schemas: what a record structurally cannot say.

Every test here asserts a *refusal*: a shape the schema will not validate.
The point of the module is that the distinctions Milestone 7 exists to
preserve — quality versus confidence, calibrated versus uncalibrated,
abstained versus unavailable, interval versus probability — are enforced by
the persisted document rather than by convention.

No test needs a webcam, a model asset, a display server, a network, Unity,
a public dataset, or participant data.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from engagevr.schemas.experiments import (
    SOFTWARE_SELF_CHECK_BANNER,
    ClassificationMetrics,
    EvaluationMode,
)
from engagevr.schemas.fusion import FusionModality
from engagevr.schemas.targets import TaskType
from engagevr.schemas.uncertainty import (
    AbstentionDecision,
    AbstentionReason,
    AdaptationGateDecision,
    AdaptationGateRecord,
    ClassificationConfidence,
    CoverageAxis,
    CoverageCurve,
    CoveragePoint,
    EnsembleDisagreementReference,
    EstimatedThresholdRecord,
    MonotonicDirection,
    PersonalThresholdRecord,
    PredictionSource,
    ProbabilityCalibrationStatus,
    RegressionPredictionInterval,
    RiskCoveragePoint,
    SelectiveMetrics,
    SelectivePredictionConfiguration,
    SignalQualitySummary,
    ThresholdObjective,
    ThresholdSource,
    UncertaintyEvaluation,
    UncertaintyFoldResult,
    UncertaintyMethod,
)

VOCABULARY = ("low", "medium", "high")


def _confidence(**overrides: object) -> ClassificationConfidence:
    payload: dict[str, object] = {
        "window_id": "w01",
        "subject_id": "synthetic-subject-01",
        "session_id": "sess-a",
        "target_name": "engagement_class",
        "fold_index": 0,
        "source_model": PredictionSource.BASELINE_MODEL,
        "source_model_name": "logistic_regression",
        "source_prediction_id": "run|0|w01|baseline_model",
        "class_vocabulary": VOCABULARY,
        "probabilities": (0.7, 0.2, 0.1),
        "probability_calibration_status": ProbabilityCalibrationStatus.CALIBRATED,
        "predicted_class": "low",
        "maximum_probability": 0.7,
        "maximum_probability_class": "low",
        "method": UncertaintyMethod.MAX_CALIBRATED_PROBABILITY,
        "confidence_score": 0.7,
        "entropy": 0.8018,
        "normalized_entropy": 0.73,
        "margin": 0.5,
        "data_source": "synthetic",
        "is_synthetic": True,
        "scientific_evaluation_eligible": False,
    }
    payload.update(overrides)
    return ClassificationConfidence(**payload)  # type: ignore[arg-type]


def _interval(**overrides: object) -> RegressionPredictionInterval:
    payload: dict[str, object] = {
        "window_id": "w01",
        "subject_id": "synthetic-subject-01",
        "session_id": "sess-a",
        "target_name": "engagement_score",
        "fold_index": 0,
        "source_model": PredictionSource.BASELINE_MODEL,
        "source_model_name": "ridge",
        "source_prediction_id": "run|0|w01|baseline_model",
        "predicted_value": 0.5,
        "interval_method": (UncertaintyMethod.SPLIT_CONFORMAL_ABSOLUTE_RESIDUAL),
        "calibration_succeeded": True,
        "lower_bound": 0.3,
        "upper_bound": 0.7,
        "interval_width": 0.4,
        "conformal_quantile": 0.2,
        "conformal_order_statistic": 9,
        "alpha": 0.1,
        "nominal_coverage": 0.9,
        "calibration_sample_count": 10,
        "calibration_group_count": 2,
        "data_source": "synthetic",
        "is_synthetic": True,
        "scientific_evaluation_eligible": False,
    }
    payload.update(overrides)
    return RegressionPredictionInterval(**payload)  # type: ignore[arg-type]


def _decision(**overrides: object) -> AbstentionDecision:
    payload: dict[str, object] = {
        "window_id": "w01",
        "subject_id": "synthetic-subject-01",
        "session_id": "sess-a",
        "target_name": "engagement_class",
        "task_type": TaskType.CLASSIFICATION,
        "fold_index": 0,
        "source_prediction_id": "run|0|w01|baseline_model",
        "prediction_available": True,
        "accepted": True,
        "abstained": False,
        "predicted_class": "low",
        "class_vocabulary": VOCABULARY,
        "probabilities": (0.7, 0.2, 0.1),
        "probability_calibration_status": ProbabilityCalibrationStatus.CALIBRATED,
        "confidence_score": 0.7,
        "applied_threshold": 0.5,
        "threshold_source": ThresholdSource.CONFIGURED_POPULATION,
        "data_source": "synthetic",
        "is_synthetic": True,
        "scientific_evaluation_eligible": False,
        "acceptance_rule": "accept if score >= tau",
    }
    payload.update(overrides)
    return AbstentionDecision(**payload)  # type: ignore[arg-type]


class TestNoSingleUncertaintyField:
    def test_no_record_carries_a_bare_uncertainty_field(self) -> None:
        # A field named merely "uncertainty" would make quality, probability,
        # calibration status, confidence, and disagreement indistinguishable.
        for model in (
            ClassificationConfidence,
            RegressionPredictionInterval,
            AbstentionDecision,
            AdaptationGateRecord,
            SelectivePredictionConfiguration,
        ):
            assert "uncertainty" not in model.model_fields

    def test_every_record_forbids_unknown_fields(self) -> None:
        for model in (
            ClassificationConfidence,
            RegressionPredictionInterval,
            AbstentionDecision,
            AdaptationGateRecord,
            SelectivePredictionConfiguration,
            SignalQualitySummary,
            EnsembleDisagreementReference,
            PersonalThresholdRecord,
            EstimatedThresholdRecord,
            CoveragePoint,
            SelectiveMetrics,
            RiskCoveragePoint,
            UncertaintyFoldResult,
            UncertaintyEvaluation,
        ):
            assert model.model_config.get("extra") == "forbid"

    def test_quality_and_confidence_are_different_fields(self) -> None:
        record = _confidence(
            signal_quality=SignalQualitySummary(
                available_modalities=(FusionModality.RPPG,),
                modality_quality={"rppg": 0.2},
                minimum_recorded_quality=0.2,
            )
        )
        assert record.confidence_score == 0.7
        assert record.signal_quality is not None
        assert record.signal_quality.minimum_recorded_quality == 0.2

    def test_disagreement_keeps_its_milestone_6_name(self) -> None:
        reference = EnsembleDisagreementReference(ensemble_disagreement=0.4)
        assert "ensemble_disagreement" in EnsembleDisagreementReference.model_fields
        assert "uncertainty" not in EnsembleDisagreementReference.model_fields
        assert "not a calibrated uncertainty estimate" in reference.note


class TestClassificationConfidence:
    def test_a_valid_calibrated_record_validates(self) -> None:
        assert _confidence().confidence_score == pytest.approx(0.7)

    def test_an_uncalibrated_maximum_cannot_be_called_confidence(self) -> None:
        with pytest.raises(ValidationError, match="not calibrated"):
            _confidence(
                probability_calibration_status=(
                    ProbabilityCalibrationStatus.UNCALIBRATED
                ),
                method=UncertaintyMethod.MAX_UNCALIBRATED_PROBABILITY,
                confidence_score=0.7,
                probability_calibration_unavailable_reason="thin class",
            )

    def test_an_uncalibrated_record_carries_a_selection_score(self) -> None:
        record = _confidence(
            probability_calibration_status=(ProbabilityCalibrationStatus.UNCALIBRATED),
            method=UncertaintyMethod.MAX_UNCALIBRATED_PROBABILITY,
            confidence_score=None,
            selection_score=0.7,
            probability_calibration_unavailable_reason="thin class",
        )
        assert record.selection_score == pytest.approx(0.7)
        assert record.confidence_score is None
        assert record.score() == pytest.approx(0.7)

    def test_an_uncalibrated_record_must_say_why(self) -> None:
        with pytest.raises(ValidationError, match="must state why"):
            _confidence(
                probability_calibration_status=(
                    ProbabilityCalibrationStatus.UNCALIBRATED
                ),
                method=UncertaintyMethod.MAX_UNCALIBRATED_PROBABILITY,
                confidence_score=None,
                selection_score=0.7,
            )

    def test_a_calibrated_record_cannot_also_carry_a_selection_score(self) -> None:
        with pytest.raises(ValidationError, match="records confidence_score, not"):
            _confidence(selection_score=0.7)

    def test_the_method_must_match_the_calibration_status(self) -> None:
        with pytest.raises(ValidationError, match="must record method"):
            _confidence(method=UncertaintyMethod.MAX_UNCALIBRATED_PROBABILITY)

    def test_a_record_with_a_vector_cannot_claim_unavailable_calibration(self) -> None:
        with pytest.raises(ValidationError, match="cannot be 'unavailable'"):
            _confidence(
                probability_calibration_status=(
                    ProbabilityCalibrationStatus.UNAVAILABLE
                )
            )

    def test_probabilities_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match=r"not 1\.0"):
            _confidence(probabilities=(0.5, 0.2, 0.1), maximum_probability=0.5)

    def test_a_negative_probability_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="negative"):
            # Sums to one and has a legal maximum, so only the negative
            # entry can be what the validator objects to.
            _confidence(
                probabilities=(0.9, 0.2, -0.1),
                maximum_probability=0.9,
                confidence_score=0.9,
            )

    def test_a_non_finite_probability_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="not finite"):
            _confidence(probabilities=(float("nan"), 0.5, 0.5), maximum_probability=0.5)

    def test_the_maximum_must_match_the_vector(self) -> None:
        with pytest.raises(ValidationError, match="largest recorded probability"):
            _confidence(maximum_probability=0.9, confidence_score=0.9)

    def test_the_confidence_score_must_be_the_maximum(self) -> None:
        with pytest.raises(ValidationError, match="not the maximum calibrated"):
            _confidence(confidence_score=0.6)

    def test_a_predicted_class_outside_the_vocabulary_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="not in the vocabulary"):
            _confidence(predicted_class="unknown")

    def test_synthetic_can_never_be_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically eligible"):
            _confidence(scientific_evaluation_eligible=True)

    def test_the_entropy_and_margin_equations_are_recorded(self) -> None:
        record = _confidence()
        assert "log(p_c)" in record.entropy_equation
        assert "p_(1) - p_(2)" in record.margin_equation
        assert "RANKING DIAGNOSTIC" in record.margin_equation


class TestRegressionPredictionInterval:
    def test_a_valid_interval_validates(self) -> None:
        assert _interval().interval_width == pytest.approx(0.4)

    def test_there_is_no_confidence_field_on_an_interval(self) -> None:
        assert "confidence_score" not in RegressionPredictionInterval.model_fields
        assert "selection_score" not in RegressionPredictionInterval.model_fields

    def test_the_point_prediction_must_lie_inside_the_interval(self) -> None:
        with pytest.raises(ValidationError, match="must hold"):
            _interval(predicted_value=0.9)

    def test_the_width_must_match_the_bounds(self) -> None:
        with pytest.raises(ValidationError, match="does not match"):
            _interval(interval_width=0.9)

    def test_an_unavailable_interval_states_why(self) -> None:
        record = _interval(
            calibration_succeeded=False,
            lower_bound=None,
            upper_bound=None,
            interval_width=None,
            conformal_quantile=None,
            unavailable_reason="too few calibration residuals",
        )
        assert record.interval_width is None
        assert record.unavailable_reason

    def test_an_unavailable_interval_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="never recorded as width zero"):
            _interval(
                calibration_succeeded=False,
                lower_bound=None,
                upper_bound=None,
                interval_width=None,
                conformal_quantile=None,
            )

    def test_an_unavailable_interval_cannot_carry_bounds(self) -> None:
        with pytest.raises(ValidationError, match="but lower_bound is"):
            _interval(
                calibration_succeeded=False,
                unavailable_reason="too few residuals",
            )

    def test_alpha_and_nominal_coverage_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match="must sum to 1"):
            _interval(alpha=0.1, nominal_coverage=0.8)

    def test_a_non_finite_point_prediction_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="not finite"):
            _interval(predicted_value=float("nan"))

    def test_clipped_bounds_must_be_recorded_as_a_labelled_pair(self) -> None:
        with pytest.raises(ValidationError, match="recorded as a pair"):
            _interval(clipped_lower_bound=0.3)
        with pytest.raises(ValidationError, match="presentation projection"):
            _interval(clipped_lower_bound=0.3, clipped_upper_bound=0.7)

    def test_raw_bounds_survive_a_clipping_projection(self) -> None:
        record = _interval(
            lower_bound=-0.1,
            upper_bound=0.9,
            interval_width=1.0,
            predicted_value=0.4,
            conformal_quantile=0.5,
            clipped_lower_bound=0.0,
            clipped_upper_bound=0.9,
            clipping_note="PRESENTATION PROJECTION ONLY.",
        )
        assert record.lower_bound == pytest.approx(-0.1)
        assert record.clipped_lower_bound == pytest.approx(0.0)

    def test_a_successful_interval_records_its_quantile(self) -> None:
        with pytest.raises(ValidationError, match="must record the"):
            _interval(conformal_quantile=None)

    def test_synthetic_can_never_be_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically eligible"):
            _interval(scientific_evaluation_eligible=True)


class TestAbstentionDecision:
    def test_an_accepted_decision_carries_no_reason(self) -> None:
        with pytest.raises(ValidationError, match="carries no abstention reason"):
            _decision(reasons=(AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,))

    def test_an_abstention_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state at least one reason"):
            _decision(accepted=False, abstained=True)

    def test_a_decision_is_exactly_one_of_accepted_or_abstained(self) -> None:
        with pytest.raises(ValidationError, match="exactly one of"):
            _decision(accepted=True, abstained=True)

    def test_an_abstained_window_keeps_its_original_prediction(self) -> None:
        record = _decision(
            accepted=False,
            abstained=True,
            reasons=(AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,),
        )
        assert record.predicted_class == "low"
        assert record.probabilities == (0.7, 0.2, 0.1)

    def test_an_abstained_window_is_not_converted_to_a_wrong_class(self) -> None:
        record = _decision(
            accepted=False,
            abstained=True,
            reasons=(AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,),
        )
        # The class is retained, not blanked and not replaced.
        assert record.predicted_class in record.class_vocabulary

    def test_an_available_prediction_must_be_retained(self) -> None:
        with pytest.raises(ValidationError, match="retained on the record"):
            _decision(
                accepted=False,
                abstained=True,
                reasons=(AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,),
                predicted_class=None,
            )

    def test_an_unavailable_prediction_cannot_be_accepted(self) -> None:
        with pytest.raises(ValidationError, match="cannot be accepted"):
            _decision(prediction_available=False)

    def test_an_unavailable_prediction_records_its_own_reason(self) -> None:
        with pytest.raises(ValidationError, match="must record"):
            _decision(
                prediction_available=False,
                accepted=False,
                abstained=True,
                reasons=(AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,),
                predicted_class=None,
                probabilities=(),
                class_vocabulary=(),
                confidence_score=None,
            )

    def test_an_unavailable_prediction_is_distinct_from_an_abstention(self) -> None:
        record = _decision(
            prediction_available=False,
            accepted=False,
            abstained=True,
            reasons=(AbstentionReason.MODEL_PREDICTION_UNAVAILABLE,),
            predicted_class=None,
            probabilities=(),
            class_vocabulary=(),
            confidence_score=None,
            applied_threshold=None,
            threshold_source=None,
        )
        assert record.primary_reason() is (
            AbstentionReason.MODEL_PREDICTION_UNAVAILABLE
        )

    def test_reasons_must_be_recorded_in_canonical_order(self) -> None:
        with pytest.raises(ValidationError, match="canonical order"):
            _decision(
                accepted=False,
                abstained=True,
                reasons=(
                    AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,
                    AbstentionReason.SIGNAL_QUALITY_BELOW_GATE,
                ),
            )

    def test_duplicate_reasons_are_refused(self) -> None:
        with pytest.raises(ValidationError, match=r"[Dd]uplicate"):
            _decision(
                accepted=False,
                abstained=True,
                reasons=(
                    AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,
                    AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,
                ),
            )

    def test_an_accepted_decision_records_the_threshold_it_cleared(self) -> None:
        with pytest.raises(ValidationError, match="must record the"):
            _decision(applied_threshold=None)

    def test_a_threshold_without_provenance_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="without its provenance"):
            _decision(threshold_source=None)

    def test_a_regression_decision_cannot_carry_probabilities(self) -> None:
        with pytest.raises(ValidationError, match="must not carry a class"):
            _decision(
                task_type=TaskType.REGRESSION,
                predicted_value=0.5,
                interval_lower_bound=0.3,
                interval_upper_bound=0.7,
                interval_width=0.4,
                applied_threshold=None,
                threshold_source=None,
            )

    def test_a_regression_decision_cannot_be_accepted_without_an_interval(self) -> None:
        with pytest.raises(ValidationError, match="not width zero"):
            _decision(
                task_type=TaskType.REGRESSION,
                predicted_class=None,
                probabilities=(),
                class_vocabulary=(),
                confidence_score=None,
                applied_threshold=None,
                threshold_source=None,
                predicted_value=0.5,
            )

    def test_a_regression_point_must_lie_inside_its_interval(self) -> None:
        with pytest.raises(ValidationError, match="outside its interval"):
            _decision(
                task_type=TaskType.REGRESSION,
                predicted_class=None,
                probabilities=(),
                class_vocabulary=(),
                confidence_score=None,
                applied_threshold=None,
                threshold_source=None,
                predicted_value=0.9,
                interval_lower_bound=0.3,
                interval_upper_bound=0.7,
                interval_width=0.4,
            )

    def test_a_decision_cannot_carry_both_score_kinds(self) -> None:
        with pytest.raises(ValidationError, match="never both"):
            _decision(confidence_score=0.7, selection_score=0.7)


class TestThresholdRecords:
    def _personal(self, **overrides: object) -> PersonalThresholdRecord:
        payload: dict[str, object] = {
            "subject_id": "synthetic-subject-01",
            "fold_index": 0,
            "population_threshold": 0.7,
            "personalized_threshold": 0.6,
            "applied_threshold": 0.6,
            "threshold_source": ThresholdSource.PERSONALIZED,
            "personalization_applied": True,
            "fallback_to_population": False,
            "calibration_window_ids": ("w0", "w1"),
            "evaluation_window_ids": ("w2",),
            "calibration_sample_count": 2,
            "temporal_order_verified": True,
        }
        payload.update(overrides)
        return PersonalThresholdRecord(**payload)  # type: ignore[arg-type]

    def test_a_valid_personalized_record_validates(self) -> None:
        assert self._personal().applied_threshold == pytest.approx(0.6)

    def test_the_rule_may_never_claim_to_use_labels(self) -> None:
        with pytest.raises(ValidationError, match="uses no labels"):
            self._personal(uses_labels=True)

    def test_a_personalized_record_names_its_calibration_windows(self) -> None:
        with pytest.raises(ValidationError, match="must name the calibration"):
            self._personal(calibration_window_ids=())

    def test_a_personalized_record_needs_a_verified_temporal_order(self) -> None:
        with pytest.raises(ValidationError, match="verified"):
            self._personal(temporal_order_verified=False)

    def test_the_applied_threshold_must_be_the_personalized_one(self) -> None:
        with pytest.raises(ValidationError, match="not the personalized threshold"):
            self._personal(applied_threshold=0.7)

    def test_a_fallback_applies_the_population_threshold(self) -> None:
        with pytest.raises(ValidationError, match="must apply the population"):
            self._personal(
                personalization_applied=False,
                personalized_threshold=None,
                fallback_to_population=True,
                fallback_reason="too few windows",
                threshold_source=ThresholdSource.POPULATION_FALLBACK,
                applied_threshold=0.6,
            )

    def test_a_fallback_states_its_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state its reason"):
            self._personal(
                personalization_applied=False,
                personalized_threshold=None,
                fallback_to_population=True,
                threshold_source=ThresholdSource.POPULATION_FALLBACK,
                applied_threshold=0.7,
            )

    def test_a_calibration_window_cannot_also_be_an_evaluation_window(self) -> None:
        with pytest.raises(ValidationError, match="both calibration and evaluation"):
            self._personal(evaluation_window_ids=("w0",))

    def _estimated(self, **overrides: object) -> EstimatedThresholdRecord:
        payload: dict[str, object] = {
            "fold_index": 0,
            "available": True,
            "objective": ThresholdObjective.TARGET_ACCEPTED_ACCURACY,
            "objective_target": 0.8,
            "selected_threshold": 0.7,
            "search_grid": (0.5, 0.7, 0.9),
            "calibration_group_ids": ("a", "b"),
            "calibration_sample_count": 40,
            "calibration_group_count": 2,
            "outer_test_group_ids": ("c", "d"),
        }
        payload.update(overrides)
        return EstimatedThresholdRecord(**payload)  # type: ignore[arg-type]

    def test_an_estimated_threshold_may_never_claim_outer_test_labels(self) -> None:
        with pytest.raises(ValidationError, match="never reads outer-test labels"):
            self._estimated(used_outer_test_labels=True)

    def test_the_selected_threshold_must_come_from_the_searched_grid(self) -> None:
        with pytest.raises(ValidationError, match="not one of the searched"):
            self._estimated(selected_threshold=0.6)

    def test_an_unavailable_record_carries_no_threshold(self) -> None:
        with pytest.raises(ValidationError, match="no threshold may be recorded"):
            self._estimated(available=False, unavailable_reason="unreachable target")

    def test_threshold_selection_groups_cannot_overlap_the_test_groups(self) -> None:
        with pytest.raises(ValidationError, match="overlap the outer-test"):
            self._estimated(calibration_group_ids=("a", "c"))

    def test_the_tie_break_rule_is_recorded(self) -> None:
        assert "SMALLEST" in self._estimated().tie_break_rule


class TestCoverageAndSelectiveMetrics:
    def test_the_three_counts_must_reconcile(self) -> None:
        with pytest.raises(ValidationError, match="reconcile exactly"):
            CoveragePoint(
                threshold=0.5,
                total_window_count=10,
                accepted_count=3,
                abstained_count=3,
                unavailable_count=3,
                coverage=0.3,
                abstention_rate=0.3,
            )

    def test_coverage_must_equal_accepted_over_total(self) -> None:
        with pytest.raises(ValidationError, match="is not accepted / total"):
            CoveragePoint(
                threshold=0.5,
                total_window_count=10,
                accepted_count=3,
                abstained_count=5,
                unavailable_count=2,
                coverage=0.9,
                abstention_rate=0.5,
            )

    def test_accepted_metrics_must_cover_exactly_the_accepted_rows(self) -> None:
        point = CoveragePoint(
            threshold=0.5,
            total_window_count=10,
            accepted_count=3,
            abstained_count=5,
            unavailable_count=2,
            coverage=0.3,
            abstention_rate=0.5,
        )
        with pytest.raises(ValidationError, match="must use exactly the accepted"):
            SelectiveMetrics(
                threshold=0.5,
                coverage_point=point,
                accepted_classification=ClassificationMetrics(
                    sample_count=7,
                    independent_group_count=2,
                    class_support={"low": 7},
                ),
            )

    def test_the_selective_threshold_must_match_its_coverage_point(self) -> None:
        point = CoveragePoint(
            threshold=0.5,
            total_window_count=1,
            accepted_count=1,
            abstained_count=0,
            unavailable_count=0,
            coverage=1.0,
            abstention_rate=0.0,
        )
        with pytest.raises(ValidationError, match="carry a coverage point for"):
            SelectiveMetrics(threshold=0.9, coverage_point=point)

    def test_an_undefined_risk_states_why(self) -> None:
        with pytest.raises(ValidationError, match="must state why"):
            RiskCoveragePoint(threshold=0.5, coverage=0.0, accepted_count=0)

    def test_a_confidence_threshold_above_one_is_refused(self) -> None:
        with pytest.raises(ValidationError, match=r"outside \[0, 1\]"):
            CoveragePoint(
                axis=CoverageAxis.CONFIDENCE_THRESHOLD,
                threshold=2.5,
                total_window_count=1,
                accepted_count=1,
                abstained_count=0,
                unavailable_count=0,
                coverage=1.0,
                abstention_rate=0.0,
            )

    def test_a_width_above_one_is_an_ordinary_axis_value(self) -> None:
        # A regression target need not live in [0, 1], so 2.5 is a perfectly
        # ordinary interval width.
        point = CoveragePoint(
            axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
            threshold=2.5,
            total_window_count=1,
            accepted_count=1,
            abstained_count=0,
            unavailable_count=0,
            coverage=1.0,
            abstention_rate=0.0,
        )
        assert point.threshold == pytest.approx(2.5)
        assert "NOT a probability" in point.axis_units

    def test_an_absent_axis_value_must_state_why(self) -> None:
        with pytest.raises(ValidationError, match="must state why it has none"):
            CoveragePoint(
                axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
                threshold=None,
                total_window_count=1,
                accepted_count=1,
                abstained_count=0,
                unavailable_count=0,
                coverage=1.0,
                abstention_rate=0.0,
            )

    def test_a_confidence_point_always_has_a_threshold(self) -> None:
        with pytest.raises(ValidationError, match="always has a threshold"):
            CoveragePoint(
                axis=CoverageAxis.CONFIDENCE_THRESHOLD,
                threshold=None,
                threshold_unavailable_reason="none",
                total_window_count=1,
                accepted_count=1,
                abstained_count=0,
                unavailable_count=0,
                coverage=1.0,
                abstention_rate=0.0,
            )

    def test_selective_metrics_cannot_mix_axes(self) -> None:
        point = CoveragePoint(
            axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
            threshold=0.5,
            total_window_count=1,
            accepted_count=1,
            abstained_count=0,
            unavailable_count=0,
            coverage=1.0,
            abstention_rate=0.0,
        )
        with pytest.raises(ValidationError, match="opposite directions"):
            SelectiveMetrics(
                axis=CoverageAxis.CONFIDENCE_THRESHOLD,
                threshold=0.5,
                coverage_point=point,
            )


class TestCoverageCurveAxis:
    """A curve's axis is fixed by its task type and cannot be swapped."""

    def _point(self, axis: CoverageAxis, threshold: float) -> SelectiveMetrics:
        return SelectiveMetrics(
            axis=axis,
            threshold=threshold,
            coverage_point=CoveragePoint(
                axis=axis,
                threshold=threshold,
                total_window_count=2,
                accepted_count=1,
                abstained_count=1,
                unavailable_count=0,
                coverage=0.5,
                abstention_rate=0.5,
            ),
        )

    def test_a_classification_curve_uses_the_confidence_axis(self) -> None:
        curve = CoverageCurve(
            task_type=TaskType.CLASSIFICATION,
            axis=CoverageAxis.CONFIDENCE_THRESHOLD,
            axis_values=(0.5,),
            points=(self._point(CoverageAxis.CONFIDENCE_THRESHOLD, 0.5),),
            expected_monotonic_direction=MonotonicDirection.NON_INCREASING,
            coverage_is_monotonic=True,
        )
        assert curve.axis is CoverageAxis.CONFIDENCE_THRESHOLD
        assert "coverage[i + 1] <= coverage[i]" in curve.monotonicity_rule

    def test_a_regression_curve_uses_the_width_axis(self) -> None:
        curve = CoverageCurve(
            task_type=TaskType.REGRESSION,
            axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
            axis_values=(0.5,),
            points=(self._point(CoverageAxis.MAXIMUM_INTERVAL_WIDTH, 0.5),),
            expected_monotonic_direction=MonotonicDirection.NON_DECREASING,
            coverage_is_monotonic=True,
        )
        assert curve.axis is CoverageAxis.MAXIMUM_INTERVAL_WIDTH
        assert "coverage[i + 1] >= coverage[i]" in curve.monotonicity_rule

    def test_a_regression_curve_cannot_be_indexed_by_a_classification_score(
        self,
    ) -> None:
        # max_calibrated_probability, or any other confidence score, indexes
        # the confidence axis. A regression target has no class probability
        # to threshold, so its curve cannot be swept over one.
        with pytest.raises(ValidationError, match="no class probability"):
            CoverageCurve(
                task_type=TaskType.REGRESSION,
                axis=CoverageAxis.CONFIDENCE_THRESHOLD,
                axis_values=(0.5,),
                points=(self._point(CoverageAxis.CONFIDENCE_THRESHOLD, 0.5),),
                expected_monotonic_direction=MonotonicDirection.NON_INCREASING,
                coverage_is_monotonic=True,
            )

    def test_a_classification_curve_cannot_be_indexed_by_a_width(self) -> None:
        with pytest.raises(ValidationError, match="no prediction interval"):
            CoverageCurve(
                task_type=TaskType.CLASSIFICATION,
                axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
                axis_values=(0.5,),
                points=(self._point(CoverageAxis.MAXIMUM_INTERVAL_WIDTH, 0.5),),
                expected_monotonic_direction=MonotonicDirection.NON_DECREASING,
                coverage_is_monotonic=True,
            )

    def test_the_width_axis_cannot_claim_the_classification_direction(self) -> None:
        with pytest.raises(ValidationError, match="non_decreasing, not"):
            CoverageCurve(
                task_type=TaskType.REGRESSION,
                axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
                axis_values=(0.5,),
                points=(self._point(CoverageAxis.MAXIMUM_INTERVAL_WIDTH, 0.5),),
                expected_monotonic_direction=MonotonicDirection.NON_INCREASING,
                coverage_is_monotonic=True,
            )

    def test_the_confidence_axis_cannot_claim_the_width_direction(self) -> None:
        with pytest.raises(ValidationError, match="non_increasing, not"):
            CoverageCurve(
                task_type=TaskType.CLASSIFICATION,
                axis=CoverageAxis.CONFIDENCE_THRESHOLD,
                axis_values=(0.5,),
                points=(self._point(CoverageAxis.CONFIDENCE_THRESHOLD, 0.5),),
                expected_monotonic_direction=MonotonicDirection.NON_DECREASING,
                coverage_is_monotonic=True,
            )

    def test_a_curve_cannot_carry_a_point_from_the_other_axis(self) -> None:
        with pytest.raises(ValidationError, match="carries a point on the"):
            CoverageCurve(
                task_type=TaskType.CLASSIFICATION,
                axis=CoverageAxis.CONFIDENCE_THRESHOLD,
                axis_values=(0.5,),
                points=(self._point(CoverageAxis.MAXIMUM_INTERVAL_WIDTH, 0.5),),
                expected_monotonic_direction=MonotonicDirection.NON_INCREASING,
                coverage_is_monotonic=True,
            )

    def test_an_unswept_curve_must_state_why_it_has_no_points(self) -> None:
        with pytest.raises(ValidationError, match="must state why it has none"):
            CoverageCurve(
                task_type=TaskType.REGRESSION,
                axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
                expected_monotonic_direction=MonotonicDirection.NON_DECREASING,
            )

    def test_an_unswept_curve_makes_no_vacuous_monotonicity_claim(self) -> None:
        with pytest.raises(ValidationError, match="vacuously true"):
            CoverageCurve(
                task_type=TaskType.REGRESSION,
                axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
                points_unavailable_reason="no width grid was configured",
                expected_monotonic_direction=MonotonicDirection.NON_DECREASING,
                coverage_is_monotonic=True,
            )

    def test_a_swept_curve_must_report_its_monotonicity(self) -> None:
        with pytest.raises(ValidationError, match="must report whether"):
            CoverageCurve(
                task_type=TaskType.REGRESSION,
                axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
                axis_values=(0.5,),
                points=(self._point(CoverageAxis.MAXIMUM_INTERVAL_WIDTH, 0.5),),
                expected_monotonic_direction=MonotonicDirection.NON_DECREASING,
            )

    def test_the_recorded_points_must_match_the_configured_grid(self) -> None:
        with pytest.raises(ValidationError, match="must appear exactly once"):
            CoverageCurve(
                task_type=TaskType.REGRESSION,
                axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
                axis_values=(0.25, 0.5),
                points=(self._point(CoverageAxis.MAXIMUM_INTERVAL_WIDTH, 0.5),),
                expected_monotonic_direction=MonotonicDirection.NON_DECREASING,
                coverage_is_monotonic=True,
            )


class TestIntervalWidthGridConfiguration:
    def test_no_grid_is_the_default(self) -> None:
        config = SelectivePredictionConfiguration(threshold_grid=(0.0, 0.5))
        assert config.interval_width_grid is None

    def test_a_width_grid_is_not_defaulted_from_the_confidence_grid(self) -> None:
        config = SelectivePredictionConfiguration(threshold_grid=(0.0, 0.5, 1.0))
        assert config.interval_width_grid is None
        assert config.threshold_grid == (0.0, 0.5, 1.0)

    def test_widths_outside_the_unit_interval_are_accepted(self) -> None:
        config = SelectivePredictionConfiguration(
            threshold_grid=(0.0, 0.5), interval_width_grid=(0.0, 2.5, 40.0)
        )
        assert config.interval_width_grid == (0.0, 2.5, 40.0)

    def test_an_empty_width_grid_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="interval_width_grid is empty"):
            SelectivePredictionConfiguration(
                threshold_grid=(0.0, 0.5), interval_width_grid=()
            )

    def test_a_negative_width_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="is negative"):
            SelectivePredictionConfiguration(
                threshold_grid=(0.0, 0.5), interval_width_grid=(-0.1, 0.5)
            )

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_a_non_finite_width_is_refused(self, value: float) -> None:
        with pytest.raises(ValidationError, match="is not finite"):
            SelectivePredictionConfiguration(
                threshold_grid=(0.0, 0.5), interval_width_grid=(0.1, value)
            )

    def test_a_repeated_width_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="more than once"):
            SelectivePredictionConfiguration(
                threshold_grid=(0.0, 0.5), interval_width_grid=(0.5, 0.5)
            )

    def test_an_unsorted_width_grid_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="sorted ascending"):
            SelectivePredictionConfiguration(
                threshold_grid=(0.0, 0.5), interval_width_grid=(0.9, 0.1)
            )

    def test_the_width_grid_enters_the_run_identity(self) -> None:
        from engagevr.training.uncertainty import build_uncertainty_run_id

        payload: dict[str, object] = {
            "target_name": "engagement_score",
            "task_type": "regression",
            "evaluation_mode": "software_self_check",
            "dataset_fingerprint": "sha256:a",
            "split_manifest_fingerprint": "sha256:b",
            "random_seed": 42,
            "calibration_method": "sigmoid",
            "engagevr_version": "0.1.0",
        }
        without = build_uncertainty_run_id(
            configuration=SelectivePredictionConfiguration(threshold_grid=(0.0, 0.5)),
            **payload,  # type: ignore[arg-type]
        )
        with_grid = build_uncertainty_run_id(
            configuration=SelectivePredictionConfiguration(
                threshold_grid=(0.0, 0.5), interval_width_grid=(0.25, 0.5)
            ),
            **payload,  # type: ignore[arg-type]
        )
        assert without != with_grid


class TestAdaptationGateRecord:
    def _record(self, **overrides: object) -> AdaptationGateRecord:
        payload: dict[str, object] = {
            "window_id": "w01",
            "subject_id": "synthetic-subject-01",
            "session_id": "sess-a",
            "fold_index": 0,
            "source_prediction_id": "run|0|w01|baseline_model",
            "decision": AdaptationGateDecision.ELIGIBLE,
            "prediction_available": True,
            "prediction_abstained": False,
            "evidence_gate_passed": True,
            "data_source": "synthetic",
            "is_synthetic": True,
            "scientific_evaluation_eligible": False,
        }
        payload.update(overrides)
        return AdaptationGateRecord(**payload)  # type: ignore[arg-type]

    def test_the_gate_has_no_field_that_could_name_an_action(self) -> None:
        forbidden = (
            "action",
            "adaptation",
            "difficulty",
            "scene",
            "policy",
            "reward",
            "cooldown",
            "hysteresis",
        )
        for name in AdaptationGateRecord.model_fields:
            for token in forbidden:
                assert token not in name, f"{name} names {token}"

    def test_the_scope_note_denies_choosing_an_adaptation(self) -> None:
        note = self._record().scope_note
        assert "does not choose an adaptation" in note
        assert "Milestone 8" in note

    def test_an_abstained_prediction_cannot_be_eligible(self) -> None:
        with pytest.raises(ValidationError, match="cannot report eligible"):
            self._record(prediction_abstained=True)

    def test_an_unavailable_prediction_cannot_be_eligible(self) -> None:
        with pytest.raises(ValidationError, match="cannot report eligible"):
            self._record(prediction_available=False)

    def test_a_failed_evidence_gate_cannot_be_eligible(self) -> None:
        with pytest.raises(ValidationError, match="cannot report eligible"):
            self._record(evidence_gate_passed=False)

    def test_a_blocked_gate_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state at least one reason"):
            self._record(decision=AdaptationGateDecision.BLOCKED)

    def test_an_eligible_gate_carries_no_reason(self) -> None:
        with pytest.raises(ValidationError, match="carries no blocking reason"):
            self._record(reasons=(AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,))

    def test_blocking_reasons_are_recorded_in_canonical_order(self) -> None:
        with pytest.raises(ValidationError, match="canonical order"):
            self._record(
                decision=AdaptationGateDecision.BLOCKED,
                prediction_abstained=True,
                reasons=(
                    AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,
                    AbstentionReason.SIGNAL_QUALITY_BELOW_GATE,
                ),
            )


class TestConfiguration:
    def test_the_defaults_are_valid(self) -> None:
        config = SelectivePredictionConfiguration(threshold_grid=(0.0, 0.5, 1.0))
        assert config.population_confidence_threshold == pytest.approx(0.70)

    def test_an_empty_threshold_grid_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            SelectivePredictionConfiguration(threshold_grid=())

    def test_a_duplicate_grid_entry_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="more than once"):
            SelectivePredictionConfiguration(threshold_grid=(0.1, 0.1))

    def test_an_unsorted_grid_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="sorted ascending"):
            SelectivePredictionConfiguration(threshold_grid=(0.5, 0.1))

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_an_out_of_range_grid_entry_is_refused(self, value: float) -> None:
        with pytest.raises(ValidationError, match=r"outside \[0, 1\]"):
            SelectivePredictionConfiguration(threshold_grid=(0.0, value))

    def test_a_non_finite_grid_entry_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="not finite"):
            SelectivePredictionConfiguration(threshold_grid=(0.0, float("nan")))

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.5])
    def test_an_out_of_range_alpha_is_refused(self, alpha: float) -> None:
        with pytest.raises(ValidationError):
            SelectivePredictionConfiguration(threshold_grid=(0.0, 0.5), alpha=alpha)

    def test_a_regression_method_cannot_be_a_confidence_source(self) -> None:
        with pytest.raises(ValidationError, match="not a classification confidence"):
            SelectivePredictionConfiguration(
                threshold_grid=(0.0, 0.5),
                confidence_source=(UncertaintyMethod.SPLIT_CONFORMAL_ABSOLUTE_RESIDUAL),
            )

    def test_a_classification_method_cannot_be_an_interval_method(self) -> None:
        with pytest.raises(ValidationError, match="not a regression interval"):
            SelectivePredictionConfiguration(
                threshold_grid=(0.0, 0.5),
                interval_method=UncertaintyMethod.MAX_CALIBRATED_PROBABILITY,
            )

    def test_an_unreachable_personal_minimum_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="cannot fire"):
            SelectivePredictionConfiguration(
                threshold_grid=(0.0, 0.5),
                personalized_thresholds_enabled=True,
                personal_calibration_windows=2,
                minimum_personal_calibration_windows=5,
            )

    def test_duplicate_modalities_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="duplicates"):
            SelectivePredictionConfiguration(
                threshold_grid=(0.0, 0.5),
                modalities=(FusionModality.RPPG, FusionModality.RPPG),
            )

    def test_every_equation_is_recorded_on_the_configuration(self) -> None:
        config = SelectivePredictionConfiguration(threshold_grid=(0.0, 0.5))
        assert "log(p_c)" in config.entropy_equation
        assert "p_(1) - p_(2)" in config.margin_equation
        assert "ceil((n + 1)" in config.conformal_equation
        assert "INCLUSIVE" in config.classification_acceptance_rule
        assert "accepted_count / total_window_count" in config.coverage_equation


class TestSelfCheckContract:
    def _evaluation(self, **overrides: object) -> UncertaintyEvaluation:
        payload: dict[str, object] = {
            "run_id": "engagement_class-uncertainty-selfcheck-abc",
            "evaluation_mode": EvaluationMode.SOFTWARE_SELF_CHECK,
            "scientific_evaluation_eligible": False,
            "target_name": "engagement_class",
            "task_type": TaskType.CLASSIFICATION,
            "dataset_fingerprint": "abc",
            "split_manifest_fingerprint": "def",
            "group_field": "subject_id",
            "group_count": 12,
            "fold_count": 3,
            "random_seed": 42,
            "configuration": SelectivePredictionConfiguration(
                threshold_grid=(0.0, 0.5)
            ),
            "disclaimers": (SOFTWARE_SELF_CHECK_BANNER,),
        }
        payload.update(overrides)
        return UncertaintyEvaluation(**payload)  # type: ignore[arg-type]

    def test_a_self_check_can_never_be_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically eligible"):
            self._evaluation(scientific_evaluation_eligible=True)

    def test_a_self_check_must_carry_the_banner(self) -> None:
        with pytest.raises(ValidationError, match="must carry the banner"):
            self._evaluation(disclaimers=("something else",))

    def test_a_document_must_carry_a_disclaimer(self) -> None:
        with pytest.raises(ValidationError, match="at least one disclaimer"):
            self._evaluation(disclaimers=())

    def test_the_run_counts_must_reconcile(self) -> None:
        with pytest.raises(ValidationError, match="total_window_count is"):
            self._evaluation(
                total_window_count=10,
                accepted_count=3,
                abstained_count=3,
                unavailable_count=3,
            )

    def test_the_notes_keep_the_four_concepts_distinct(self) -> None:
        evaluation = self._evaluation()
        assert "FOUR DIFFERENT THINGS" in evaluation.note
        assert "NOT a modality being unavailable" in evaluation.abstention_note


class TestFoldLeakageGuards:
    def _fold(self, **overrides: object) -> UncertaintyFoldResult:
        payload: dict[str, object] = {
            "fold_index": 0,
            "evaluated": True,
            "fit_group_ids": ("a", "b"),
            "probability_calibration_group_ids": ("c",),
            "outer_test_group_ids": ("d", "e"),
        }
        payload.update(overrides)
        return UncertaintyFoldResult(**payload)  # type: ignore[arg-type]

    def test_a_valid_fold_validates(self) -> None:
        assert self._fold().evaluated

    def test_calibration_may_not_touch_an_outer_test_group(self) -> None:
        with pytest.raises(ValidationError, match="never fits a model"):
            self._fold(probability_calibration_group_ids=("c", "d"))

    def test_threshold_selection_may_not_touch_an_outer_test_group(self) -> None:
        with pytest.raises(ValidationError, match="never fits a model"):
            self._fold(threshold_selection_group_ids=("d",))

    def test_conformal_calibration_may_not_touch_an_outer_test_group(self) -> None:
        with pytest.raises(ValidationError, match="never fits a model"):
            self._fold(conformal_calibration_group_ids=("e",))

    def test_fitting_may_not_touch_an_outer_test_group(self) -> None:
        with pytest.raises(ValidationError, match="never fits a model"):
            self._fold(fit_group_ids=("a", "d"))

    def test_probability_calibration_may_not_reuse_a_fit_group(self) -> None:
        with pytest.raises(ValidationError, match="reused fit group"):
            self._fold(probability_calibration_group_ids=("a",))

    def test_conformal_calibration_may_not_reuse_a_fit_group(self) -> None:
        with pytest.raises(ValidationError, match="understate the interval"):
            self._fold(conformal_calibration_group_ids=("b",))

    def test_an_unevaluated_fold_states_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state a reason"):
            self._fold(evaluated=False)

    def test_an_available_conformal_fit_records_its_quantile(self) -> None:
        with pytest.raises(ValidationError, match="must record q"):
            self._fold(conformal_available=True)

    def test_an_unavailable_conformal_fit_records_no_quantile(self) -> None:
        with pytest.raises(ValidationError, match="but q is recorded"):
            self._fold(conformal_available=False, conformal_quantile=0.2)

    def test_the_fold_counts_must_reconcile(self) -> None:
        with pytest.raises(ValidationError, match="total_window_count is"):
            self._fold(
                total_window_count=10,
                accepted_count=1,
                abstained_count=1,
                unavailable_count=1,
            )
