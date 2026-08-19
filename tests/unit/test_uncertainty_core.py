"""Milestone 7 pure algebra: confidence, conformal, thresholds, metrics.

Every function under test here takes numbers and returns numbers, so each
documented equation is checked against the equation rather than through a
training run.

No test needs a webcam, a model asset, a display server, a network, Unity,
a public dataset, or participant data.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest

from engagevr.schemas.fusion import FusionModality
from engagevr.schemas.targets import TaskType
from engagevr.schemas.uncertainty import (
    AbstentionReason,
    CoverageAxis,
    EvidenceGateConfiguration,
    MonotonicDirection,
    ProbabilityCalibrationStatus,
    SelectivePredictionConfiguration,
    ThresholdObjective,
    ThresholdSource,
    UncertaintyMethod,
)
from engagevr.training.uncertainty import (
    PERSONAL_QUANTILE_METHOD,
    UncertaintyError,
    absolute_residuals,
    acceptance_rule_for,
    accepts_at_threshold,
    accepts_interval_width,
    area_under_risk_coverage,
    assert_probability_vector,
    build_uncertainty_run_id,
    confidence_components,
    confidence_method,
    conformal_interval,
    conformal_order_statistic,
    coverage_axis_for,
    coverage_is_monotonic,
    coverage_point,
    evaluate_evidence_gate,
    expected_monotonic_direction,
    fit_conformal_quantile,
    interval_contains,
    minimum_conformal_samples,
    normalized_entropy,
    personal_confidence_threshold,
    prediction_margin,
    predictive_entropy,
    project_interval_to_range,
    reason_counts,
    risk_coverage_points,
    select_population_threshold,
    selective_classification_metrics,
    selective_regression_metrics,
)

VOCABULARY = ("low", "medium", "high")


def _configuration(**overrides: object) -> SelectivePredictionConfiguration:
    payload: dict[str, object] = {
        "modalities": tuple(FusionModality),
        "threshold_grid": (0.0, 0.25, 0.5, 0.75, 1.0),
        "population_confidence_threshold": 0.5,
    }
    payload.update(overrides)
    return SelectivePredictionConfiguration(**payload)  # type: ignore[arg-type]


class TestTerminologyContracts:
    """The five concepts must stay structurally distinct."""

    def test_an_uncalibrated_vector_cannot_claim_the_calibrated_method(self) -> None:
        assert (
            confidence_method(ProbabilityCalibrationStatus.UNCALIBRATED)
            is UncertaintyMethod.MAX_UNCALIBRATED_PROBABILITY
        )
        assert (
            confidence_method(ProbabilityCalibrationStatus.CALIBRATED)
            is UncertaintyMethod.MAX_CALIBRATED_PROBABILITY
        )

    def test_no_probability_vector_means_no_score_of_either_kind(self) -> None:
        with pytest.raises(UncertaintyError, match="neither a calibrated"):
            confidence_method(ProbabilityCalibrationStatus.UNAVAILABLE)

    def test_signal_quality_never_enters_the_confidence_computation(self) -> None:
        # The confidence path takes a probability vector and nothing else, so
        # a quality value has no argument it could arrive through.
        parts = confidence_components((0.7, 0.2, 0.1), VOCABULARY, context="w")
        assert parts.maximum_probability == pytest.approx(0.7)

    def test_the_evidence_gate_returns_its_own_reason_codes(self) -> None:
        passed, reasons = evaluate_evidence_gate(
            configuration=EvidenceGateConfiguration(minimum_signal_quality=0.5),
            prediction_available=True,
            available_modalities=["rppg"],
            modality_quality={"rppg": 0.1},
            probability_calibrated=True,
        )
        assert not passed
        assert reasons == (AbstentionReason.SIGNAL_QUALITY_BELOW_GATE,)
        # The quality failure did NOT become a confidence failure.
        assert AbstentionReason.BELOW_CONFIDENCE_THRESHOLD not in reasons

    def test_quality_is_not_multiplied_into_the_score(self) -> None:
        # A window with terrible quality still reports the probability the
        # model emitted; the gate blocks it separately.
        parts = confidence_components((0.9, 0.05, 0.05), VOCABULARY, context="w")
        passed, _reasons = evaluate_evidence_gate(
            configuration=EvidenceGateConfiguration(minimum_signal_quality=0.9),
            prediction_available=True,
            available_modalities=["rppg"],
            modality_quality={"rppg": 0.05},
            probability_calibrated=True,
        )
        assert parts.maximum_probability == pytest.approx(0.9)
        assert not passed


class TestProbabilityVector:
    def test_a_valid_distribution_is_returned_as_floats(self) -> None:
        values = assert_probability_vector((0.2, 0.3, 0.5), VOCABULARY, context="w")
        assert values.tolist() == [0.2, 0.3, 0.5]

    def test_a_non_finite_probability_is_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="not finite"):
            assert_probability_vector((float("nan"), 0.5, 0.5), VOCABULARY, context="w")

    def test_a_negative_probability_is_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="negative entry"):
            assert_probability_vector((-0.1, 0.6, 0.5), VOCABULARY, context="w")

    def test_a_vector_that_does_not_sum_to_one_is_refused(self) -> None:
        with pytest.raises(UncertaintyError, match=r"not 1\.0"):
            assert_probability_vector((0.2, 0.2, 0.2), VOCABULARY, context="w")

    def test_a_wrong_length_vector_is_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="probabilities for"):
            assert_probability_vector((0.5, 0.5), VOCABULARY, context="w")

    def test_an_empty_vector_is_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="not a distribution"):
            assert_probability_vector((), (), context="w")


class TestEntropy:
    def test_the_formula_matches_the_documented_definition(self) -> None:
        vector = (0.5, 0.25, 0.25)
        expected = -sum(p * math.log(p) for p in vector)
        assert predictive_entropy(vector) == pytest.approx(expected)

    def test_a_certain_prediction_has_zero_entropy(self) -> None:
        assert predictive_entropy((1.0, 0.0, 0.0)) == pytest.approx(0.0)

    def test_zero_log_zero_is_treated_as_zero(self) -> None:
        # Without the convention this would be nan rather than a number.
        assert math.isfinite(predictive_entropy((1.0, 0.0)))

    def test_a_uniform_vector_reaches_the_maximum(self) -> None:
        uniform = (1 / 3, 1 / 3, 1 / 3)
        assert predictive_entropy(uniform) == pytest.approx(math.log(3))
        assert normalized_entropy(uniform) == pytest.approx(1.0)

    def test_the_normalised_variant_lies_in_the_unit_interval(self) -> None:
        for vector in ((0.9, 0.05, 0.05), (0.4, 0.4, 0.2), (1 / 3, 1 / 3, 1 / 3)):
            value = normalized_entropy(vector)
            assert value is not None
            assert 0.0 <= value <= 1.0

    def test_a_single_class_has_no_normalised_entropy(self) -> None:
        # Zero would read as a perfectly certain prediction, which is a
        # different statement from "the ratio is undefined".
        assert normalized_entropy((1.0,)) is None

    def test_entropy_is_reported_in_nats_not_bits(self) -> None:
        # log base 2 would give exactly 1.0 for a fair coin; nats give ln 2.
        assert predictive_entropy((0.5, 0.5)) == pytest.approx(math.log(2))


class TestMargin:
    def test_the_formula_matches_the_documented_definition(self) -> None:
        assert prediction_margin((0.6, 0.3, 0.1)) == pytest.approx(0.3)

    def test_the_ordering_of_the_vector_does_not_matter(self) -> None:
        assert prediction_margin((0.1, 0.6, 0.3)) == pytest.approx(0.3)

    def test_a_tie_has_zero_margin(self) -> None:
        assert prediction_margin((0.5, 0.5, 0.0)) == pytest.approx(0.0)

    def test_a_single_class_vocabulary_has_no_top_two_margin(self) -> None:
        with pytest.raises(UncertaintyError, match="no second class"):
            prediction_margin((1.0,))

    def test_an_empty_vector_has_no_margin(self) -> None:
        with pytest.raises(UncertaintyError, match="empty probability vector"):
            prediction_margin(())

    def test_the_margin_lies_in_the_unit_interval(self) -> None:
        for vector in ((1.0, 0.0, 0.0), (0.4, 0.35, 0.25), (1 / 3, 1 / 3, 1 / 3)):
            assert 0.0 <= prediction_margin(vector) <= 1.0


class TestConfidenceComponents:
    def test_the_predicted_class_is_the_argmax(self) -> None:
        parts = confidence_components((0.2, 0.5, 0.3), VOCABULARY, context="w")
        assert parts.predicted_class == "medium"
        assert parts.maximum_probability_class == "medium"

    def test_the_maximum_probability_is_exact(self) -> None:
        parts = confidence_components((0.2, 0.5, 0.3), VOCABULARY, context="w")
        assert parts.maximum_probability == pytest.approx(0.5)

    def test_a_tie_resolves_to_the_first_vocabulary_position(self) -> None:
        parts = confidence_components((0.5, 0.5, 0.0), VOCABULARY, context="w")
        assert parts.predicted_class == "low"

    def test_the_four_diagnostics_are_stored_separately(self) -> None:
        parts = confidence_components((0.7, 0.2, 0.1), VOCABULARY, context="w")
        assert parts.maximum_probability != parts.entropy
        assert parts.margin != parts.entropy
        assert parts.normalized_entropy is not None


class TestAcceptanceBoundary:
    def test_a_score_above_the_threshold_is_accepted(self) -> None:
        assert accepts_at_threshold(0.71, 0.70)

    def test_a_score_below_the_threshold_abstains(self) -> None:
        assert not accepts_at_threshold(0.69, 0.70)

    def test_the_boundary_is_inclusive(self) -> None:
        # Pinned deliberately: an off-by-one-epsilon disagreement between the
        # rule and the curve would shift every reported coverage.
        assert accepts_at_threshold(0.70, 0.70)

    def test_a_zero_threshold_accepts_everything(self) -> None:
        assert accepts_at_threshold(0.0, 0.0)

    def test_a_non_finite_score_is_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="non-finite selection score"):
            accepts_at_threshold(float("nan"), 0.5)

    @pytest.mark.parametrize("threshold", [-0.1, 1.1, float("inf"), float("nan")])
    def test_an_invalid_threshold_is_refused(self, threshold: float) -> None:
        with pytest.raises(UncertaintyError, match=r"not a finite value in \[0, 1\]"):
            accepts_at_threshold(0.5, threshold)


class TestIntervalWidthAcceptance:
    def test_a_narrow_interval_is_accepted(self) -> None:
        assert accepts_interval_width(0.2, 0.5)

    def test_a_wide_interval_abstains(self) -> None:
        assert not accepts_interval_width(0.8, 0.5)

    def test_the_width_boundary_is_inclusive(self) -> None:
        assert accepts_interval_width(0.5, 0.5)

    def test_a_missing_interval_is_never_treated_as_width_zero(self) -> None:
        # Width zero would read as a perfectly certain prediction and would
        # be accepted by every threshold.
        assert not accepts_interval_width(None, 0.5)
        assert not accepts_interval_width(None, None)

    def test_no_configured_maximum_accepts_any_finite_width(self) -> None:
        assert accepts_interval_width(99.0, None)

    def test_a_negative_width_is_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="non-negative"):
            accepts_interval_width(-1.0, 0.5)


class TestEvidenceGate:
    def test_a_satisfied_gate_passes_with_no_reasons(self) -> None:
        passed, reasons = evaluate_evidence_gate(
            configuration=EvidenceGateConfiguration(),
            prediction_available=True,
            available_modalities=["rppg", "task"],
            modality_quality={"rppg": 0.9},
            probability_calibrated=True,
        )
        assert passed and reasons == ()

    def test_too_few_modalities_blocks_independently(self) -> None:
        _passed, reasons = evaluate_evidence_gate(
            configuration=EvidenceGateConfiguration(minimum_available_modalities=2),
            prediction_available=True,
            available_modalities=["rppg"],
            modality_quality={},
            probability_calibrated=True,
        )
        assert AbstentionReason.INSUFFICIENT_MEASUREMENT_EVIDENCE in reasons

    def test_a_required_modality_blocks_independently(self) -> None:
        _passed, reasons = evaluate_evidence_gate(
            configuration=EvidenceGateConfiguration(
                required_modalities=(FusionModality.RPPG,)
            ),
            prediction_available=True,
            available_modalities=["task"],
            modality_quality={},
            probability_calibrated=True,
        )
        assert AbstentionReason.REQUIRED_MODALITY_UNAVAILABLE in reasons

    def test_missing_quality_is_not_low_quality_by_default(self) -> None:
        passed, reasons = evaluate_evidence_gate(
            configuration=EvidenceGateConfiguration(minimum_signal_quality=0.5),
            prediction_available=True,
            available_modalities=["rppg"],
            modality_quality={"rppg": None},
            probability_calibrated=True,
        )
        assert passed
        assert AbstentionReason.SIGNAL_QUALITY_BELOW_GATE not in reasons

    def test_missing_quality_can_be_configured_to_fail(self) -> None:
        _passed, reasons = evaluate_evidence_gate(
            configuration=EvidenceGateConfiguration(
                minimum_signal_quality=0.5, treat_missing_quality_as_failure=True
            ),
            prediction_available=True,
            available_modalities=["rppg"],
            modality_quality={"rppg": None},
            probability_calibrated=True,
        )
        assert AbstentionReason.SIGNAL_QUALITY_BELOW_GATE in reasons

    def test_uncalibrated_probabilities_block_when_configured(self) -> None:
        _passed, reasons = evaluate_evidence_gate(
            configuration=EvidenceGateConfiguration(),
            prediction_available=True,
            available_modalities=["rppg"],
            modality_quality={},
            probability_calibrated=False,
        )
        assert AbstentionReason.PROBABILITY_CALIBRATION_UNAVAILABLE in reasons

    def test_uncalibrated_probabilities_pass_when_not_required(self) -> None:
        passed, _reasons = evaluate_evidence_gate(
            configuration=EvidenceGateConfiguration(
                require_probability_calibration_for_classification_confidence=False
            ),
            prediction_available=True,
            available_modalities=["rppg"],
            modality_quality={},
            probability_calibrated=False,
        )
        assert passed

    def test_an_absent_prediction_blocks(self) -> None:
        _passed, reasons = evaluate_evidence_gate(
            configuration=EvidenceGateConfiguration(),
            prediction_available=False,
            available_modalities=["rppg"],
            modality_quality={},
            probability_calibrated=True,
        )
        assert AbstentionReason.MODEL_PREDICTION_UNAVAILABLE in reasons

    def test_reasons_are_returned_in_canonical_order(self) -> None:
        _passed, reasons = evaluate_evidence_gate(
            configuration=EvidenceGateConfiguration(
                minimum_available_modalities=3,
                required_modalities=(FusionModality.RPPG,),
                minimum_signal_quality=0.9,
            ),
            prediction_available=False,
            available_modalities=["task"],
            modality_quality={"task": 0.1},
            probability_calibrated=False,
        )
        assert list(reasons) == sorted(
            reasons, key=lambda r: list(AbstentionReason).index(r)
        )


class TestConformalQuantile:
    def test_the_residual_equation(self) -> None:
        residuals = absolute_residuals([1.0, 2.0, 3.0], [1.5, 1.0, 3.0])
        assert residuals.tolist() == [0.5, 1.0, 0.0]

    def test_a_non_finite_prediction_is_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="not finite"):
            absolute_residuals([1.0], [float("inf")])

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="residuals are undefined"):
            absolute_residuals([1.0, 2.0], [1.0])

    def test_the_order_statistic_convention(self) -> None:
        # k = ceil((n + 1) * (1 - alpha))
        assert conformal_order_statistic(9, 0.10) == 9
        assert conformal_order_statistic(19, 0.10) == 18
        assert conformal_order_statistic(99, 0.05) == 95

    def test_the_quantile_is_the_kth_smallest_residual(self) -> None:
        residuals = [5.0, 1.0, 3.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0]
        fit = fit_conformal_quantile(residuals, alpha=0.10)
        assert fit.available
        assert fit.order_statistic == 9
        assert fit.quantile == pytest.approx(9.0)

    def test_the_minimum_sample_rule(self) -> None:
        assert minimum_conformal_samples(0.10) == 9
        assert minimum_conformal_samples(0.05) == 19
        assert minimum_conformal_samples(0.20) == 4

    def test_too_few_residuals_make_the_interval_unavailable(self) -> None:
        fit = fit_conformal_quantile([1.0, 2.0, 3.0], alpha=0.10)
        assert not fit.available
        assert fit.quantile is None
        assert "does not exist" in (fit.unavailable_reason or "")
        # Never widened to infinity, never fabricated.
        assert "infinity" in (fit.unavailable_reason or "")

    def test_no_residuals_make_the_interval_unavailable(self) -> None:
        fit = fit_conformal_quantile([], alpha=0.10)
        assert not fit.available
        assert fit.sample_count == 0
        assert "assumed to be zero-width" in (fit.unavailable_reason or "")

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_an_out_of_range_alpha_is_refused(self, alpha: float) -> None:
        with pytest.raises(UncertaintyError, match="strictly between 0 and 1"):
            conformal_order_statistic(10, alpha)

    def test_a_negative_residual_is_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="cannot be negative"):
            fit_conformal_quantile([-1.0, 1.0], alpha=0.5)

    def test_a_non_finite_residual_is_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="not finite"):
            fit_conformal_quantile([float("nan"), 1.0], alpha=0.5)


class TestConformalInterval:
    def test_the_interval_is_symmetric_about_the_point_prediction(self) -> None:
        lower, upper, width = conformal_interval(0.4, 0.1)
        assert lower == pytest.approx(0.3)
        assert upper == pytest.approx(0.5)
        assert width == pytest.approx(0.2)

    def test_the_bounds_are_finite_and_the_width_non_negative(self) -> None:
        lower, upper, width = conformal_interval(-3.0, 2.5)
        assert math.isfinite(lower) and math.isfinite(upper)
        assert width >= 0.0

    def test_a_zero_quantile_gives_a_zero_width_interval(self) -> None:
        lower, upper, width = conformal_interval(0.5, 0.0)
        assert (lower, upper, width) == (0.5, 0.5, 0.0)

    def test_a_non_finite_point_prediction_is_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="no interval"):
            conformal_interval(float("nan"), 0.1)

    def test_a_negative_quantile_is_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="non-negative"):
            conformal_interval(0.5, -0.1)

    def test_a_semantic_range_does_not_silently_clip_the_raw_interval(self) -> None:
        lower, upper, _width = conformal_interval(0.05, 0.3)
        assert lower < 0.0  # the RAW bound escapes the target's [0, 1] range
        clipped_lower, clipped_upper = project_interval_to_range(
            lower, upper, minimum=0.0, maximum=1.0
        )
        assert clipped_lower == 0.0
        assert clipped_upper == pytest.approx(0.35)
        # The raw bounds are untouched by the projection.
        assert lower == pytest.approx(-0.25)

    def test_an_inverted_target_range_is_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="empty or inverted"):
            project_interval_to_range(0.0, 1.0, minimum=1.0, maximum=0.0)

    def test_interval_containment_is_inclusive(self) -> None:
        assert interval_contains(0.0, 1.0, 0.0)
        assert interval_contains(0.0, 1.0, 1.0)
        assert not interval_contains(0.0, 1.0, 1.0001)


class TestThresholdSelection:
    def _call(self, **overrides: object) -> object:
        payload: dict[str, object] = {
            "scores": [0.9, 0.8, 0.7, 0.6, 0.5],
            "correct": [True, True, True, False, False],
            "group_ids": ["a", "a", "b", "b", "c"],
            "grid": (0.0, 0.5, 0.6, 0.7, 0.8, 0.9),
            "objective": ThresholdObjective.TARGET_ACCEPTED_ACCURACY,
            "target": 1.0,
            "minimum_samples": 1,
            "minimum_groups": 1,
            "fold_index": 0,
            "calibration_group_ids": ["a", "b", "c"],
            "outer_test_group_ids": ["d", "e"],
        }
        payload.update(overrides)
        return select_population_threshold(**payload)  # type: ignore[arg-type]

    def test_the_smallest_admissible_threshold_is_chosen(self) -> None:
        record = self._call()
        assert record.available  # type: ignore[attr-defined]
        # 0.7 is the smallest threshold whose accepted set is all-correct.
        assert record.selected_threshold == pytest.approx(0.7)  # type: ignore[attr-defined]

    def test_the_selection_is_deterministic(self) -> None:
        first = self._call()
        second = self._call()
        assert first.selected_threshold == second.selected_threshold  # type: ignore[attr-defined]

    def test_an_impossible_objective_produces_unavailable_not_a_threshold(self) -> None:
        record = self._call(correct=[False, False, False, False, False], target=1.0)
        assert not record.available  # type: ignore[attr-defined]
        assert record.selected_threshold is None  # type: ignore[attr-defined]
        assert "never invented" in (record.unavailable_reason or "")  # type: ignore[attr-defined]

    def test_too_few_samples_produces_unavailable(self) -> None:
        record = self._call(minimum_samples=100)
        assert not record.available  # type: ignore[attr-defined]
        assert "calibration row(s)" in (record.unavailable_reason or "")  # type: ignore[attr-defined]

    def test_too_few_groups_produces_unavailable(self) -> None:
        record = self._call(minimum_groups=99)
        assert not record.available  # type: ignore[attr-defined]
        assert "independent calibration group(s)" in (  # type: ignore[attr-defined]
            record.unavailable_reason or ""
        )

    def test_selection_refuses_to_touch_an_outer_test_group(self) -> None:
        with pytest.raises(UncertaintyError, match="also outer-test groups"):
            self._call(calibration_group_ids=["a", "d"], outer_test_group_ids=["d"])

    def test_the_record_states_that_no_outer_test_label_was_read(self) -> None:
        record = self._call()
        assert record.used_outer_test_labels is False  # type: ignore[attr-defined]

    def test_the_search_grid_is_recorded(self) -> None:
        record = self._call()
        assert record.search_grid == (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)  # type: ignore[attr-defined]

    def test_the_coverage_objective_prefers_the_largest_admissible(self) -> None:
        record = self._call(objective=ThresholdObjective.TARGET_COVERAGE, target=0.6)
        assert record.available  # type: ignore[attr-defined]
        # Coverage falls as tau rises, so the strictest admissible wins.
        assert record.selected_threshold == pytest.approx(0.7)  # type: ignore[attr-defined]

    def test_the_risk_objective_is_the_complement_of_accuracy(self) -> None:
        record = self._call(
            objective=ThresholdObjective.TARGET_EMPIRICAL_RISK, target=0.0
        )
        assert record.selected_threshold == pytest.approx(0.7)  # type: ignore[attr-defined]

    def test_mismatched_score_and_outcome_lengths_are_refused(self) -> None:
        with pytest.raises(UncertaintyError, match="outcome"):
            self._call(correct=[True])


class TestPersonalThreshold:
    def _call(self, **overrides: object) -> object:
        payload: dict[str, object] = {
            "subject_id": "synthetic-subject-01",
            "fold_index": 0,
            "calibration_scores": [0.4, 0.5, 0.6, 0.7, 0.8],
            "calibration_window_ids": ("w0", "w1", "w2", "w3", "w4"),
            "evaluation_window_ids": ("w5", "w6"),
            "population_threshold": 0.7,
            "configuration": _configuration(
                personalized_thresholds_enabled=True,
                personal_calibration_windows=5,
                minimum_personal_calibration_windows=5,
                personal_target_coverage=0.8,
                personal_shrinkage_constant=10.0,
            ),
            "temporal_order_verified": True,
        }
        payload.update(overrides)
        return personal_confidence_threshold(**payload)  # type: ignore[arg-type]

    def test_the_rule_uses_no_labels_at_all(self) -> None:
        record = self._call()
        assert record.uses_labels is False  # type: ignore[attr-defined]

    def test_the_shrinkage_and_quantile_match_the_documented_rule(self) -> None:
        record = self._call()
        scores = [0.4, 0.5, 0.6, 0.7, 0.8]
        raw = float(np.quantile(scores, 0.2, method=PERSONAL_QUANTILE_METHOD))
        shrinkage = 5 / (5 + 10.0)
        expected = (1 - shrinkage) * 0.7 + shrinkage * raw
        assert record.personalized_threshold == pytest.approx(expected)  # type: ignore[attr-defined]
        assert record.shrinkage == pytest.approx(shrinkage)  # type: ignore[attr-defined]
        assert record.raw_personal_quantile == pytest.approx(raw)  # type: ignore[attr-defined]

    def test_the_quantile_returns_an_observed_value(self) -> None:
        record = self._call()
        assert record.raw_personal_quantile in {0.4, 0.5, 0.6, 0.7, 0.8}  # type: ignore[attr-defined]

    def test_insufficient_evidence_falls_back_to_the_population_threshold(self) -> None:
        record = self._call(calibration_scores=[0.5], calibration_window_ids=("w0",))
        assert not record.personalization_applied  # type: ignore[attr-defined]
        assert record.applied_threshold == pytest.approx(0.7)  # type: ignore[attr-defined]
        assert record.threshold_source is ThresholdSource.POPULATION_FALLBACK  # type: ignore[attr-defined]

    def test_a_fallback_states_its_reason(self) -> None:
        record = self._call(calibration_scores=[], calibration_window_ids=())
        assert record.fallback_reason  # type: ignore[attr-defined]
        assert "fewer than" in (record.fallback_reason or "")  # type: ignore[attr-defined]

    def test_an_unverified_temporal_order_falls_back(self) -> None:
        record = self._call(temporal_order_verified=False)
        assert not record.personalization_applied  # type: ignore[attr-defined]
        assert "precede" in (record.fallback_reason or "")  # type: ignore[attr-defined]

    def test_a_supplied_unavailable_reason_falls_back(self) -> None:
        record = self._call(unavailable_reason="no valid temporal split exists")
        assert not record.personalization_applied  # type: ignore[attr-defined]
        assert record.fallback_reason == "no valid temporal split exists"  # type: ignore[attr-defined]

    def test_disabling_personalization_applies_the_population_threshold(self) -> None:
        record = self._call(
            configuration=_configuration(personalized_thresholds_enabled=False)
        )
        assert not record.personalization_applied  # type: ignore[attr-defined]
        assert record.threshold_source is ThresholdSource.CONFIGURED_POPULATION  # type: ignore[attr-defined]

    def test_a_non_finite_calibration_score_falls_back(self) -> None:
        record = self._call(calibration_scores=[0.5, float("nan"), 0.7, 0.8, 0.9])
        assert not record.personalization_applied  # type: ignore[attr-defined]
        assert "non-finite" in (record.fallback_reason or "")  # type: ignore[attr-defined]

    def test_calibration_and_evaluation_windows_may_not_overlap(self) -> None:
        with pytest.raises(ValueError, match="both calibration and evaluation"):
            self._call(evaluation_window_ids=("w0", "w9"))

    def test_the_threshold_stays_inside_the_unit_interval(self) -> None:
        record = self._call(calibration_scores=[0.0, 0.0, 0.0, 0.0, 0.0])
        assert 0.0 <= (record.applied_threshold) <= 1.0  # type: ignore[attr-defined]


class TestCoverageAccounting:
    def test_the_coverage_equation(self) -> None:
        point = coverage_point(
            threshold=0.5, accepted_count=3, abstained_count=5, unavailable_count=2
        )
        assert point.total_window_count == 10
        assert point.coverage == pytest.approx(0.3)
        assert point.abstention_rate == pytest.approx(0.5)

    def test_the_three_counts_reconcile_with_the_total(self) -> None:
        point = coverage_point(
            threshold=0.5, accepted_count=1, abstained_count=2, unavailable_count=3
        )
        assert (
            point.accepted_count + point.abstained_count + point.unavailable_count
            == point.total_window_count
        )

    def test_the_denominator_is_recorded_explicitly(self) -> None:
        point = coverage_point(
            threshold=0.0, accepted_count=1, abstained_count=0, unavailable_count=0
        )
        assert point.coverage_denominator == "total_evaluated_windows"

    def test_an_empty_evaluation_set_has_zero_coverage(self) -> None:
        point = coverage_point(
            threshold=0.5, accepted_count=0, abstained_count=0, unavailable_count=0
        )
        assert point.coverage == 0.0

    def test_unavailable_predictions_stay_out_of_the_numerator(self) -> None:
        point = coverage_point(
            threshold=0.5, accepted_count=0, abstained_count=0, unavailable_count=4
        )
        assert point.coverage == 0.0
        assert point.abstention_rate == 0.0

    def test_reason_counts_are_recorded_in_canonical_order(self) -> None:
        counts = reason_counts(
            [
                (AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,),
                (
                    AbstentionReason.SIGNAL_QUALITY_BELOW_GATE,
                    AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,
                ),
            ]
        )
        assert list(counts) == [
            AbstentionReason.SIGNAL_QUALITY_BELOW_GATE.value,
            AbstentionReason.BELOW_CONFIDENCE_THRESHOLD.value,
        ]
        assert counts[AbstentionReason.BELOW_CONFIDENCE_THRESHOLD.value] == 2


class TestSelectiveClassificationMetrics:
    def _metrics(self, accepted: list[bool], **overrides: object) -> object:
        payload: dict[str, object] = {
            "threshold": 0.5,
            "y_true": ["low", "low", "high", "high"],
            "y_predicted": ["low", "high", "high", "low"],
            "probabilities": [
                [0.8, 0.1, 0.1],
                [0.1, 0.1, 0.8],
                [0.1, 0.1, 0.8],
                [0.8, 0.1, 0.1],
            ],
            "labels": VOCABULARY,
            "group_ids": ["a", "a", "b", "b"],
            "accepted": accepted,
            "unavailable": [False, False, False, False],
        }
        payload.update(overrides)
        return selective_classification_metrics(**payload)  # type: ignore[arg-type]

    def test_accepted_metrics_use_only_the_accepted_rows(self) -> None:
        metrics = self._metrics([True, False, True, False])
        assert metrics.accepted_classification is not None  # type: ignore[attr-defined]
        assert metrics.accepted_classification.sample_count == 2  # type: ignore[attr-defined]
        # Both accepted rows are correct, so accuracy is 1.0 — the two
        # incorrect rows were abstained, not scored as wrong.
        assert metrics.accepted_classification.accuracy == pytest.approx(1.0)  # type: ignore[attr-defined]

    def test_abstentions_are_not_counted_as_prediction_errors(self) -> None:
        all_rows = self._metrics([True, True, True, True])
        selective = self._metrics([True, False, True, False])
        assert all_rows.accepted_classification.accuracy == pytest.approx(0.5)  # type: ignore[attr-defined]
        assert selective.accepted_classification.accuracy == pytest.approx(1.0)  # type: ignore[attr-defined]

    def test_an_empty_accepted_set_returns_unavailable_not_zero(self) -> None:
        metrics = self._metrics([False, False, False, False])
        assert metrics.accepted_classification is None  # type: ignore[attr-defined]
        assert metrics.empirical_risk is None  # type: ignore[attr-defined]
        assert "rather than as zero" in metrics.unavailable_metrics["accepted_metrics"]  # type: ignore[attr-defined]

    def test_the_empirical_risk_equation(self) -> None:
        metrics = self._metrics([True, True, True, True])
        assert metrics.empirical_risk == pytest.approx(0.5)  # type: ignore[attr-defined]

    def test_unavailable_predictions_are_separate_from_abstentions(self) -> None:
        metrics = self._metrics(
            [True, False, False, False], unavailable=[False, False, True, True]
        )
        point = metrics.coverage_point  # type: ignore[attr-defined]
        assert point.accepted_count == 1
        assert point.abstained_count == 1
        assert point.unavailable_count == 2

    def test_a_window_cannot_be_both_accepted_and_unavailable(self) -> None:
        with pytest.raises(UncertaintyError, match="both accepted"):
            self._metrics(
                [True, False, False, False],
                unavailable=[True, False, False, False],
            )

    def test_class_support_among_accepted_predictions_is_recorded(self) -> None:
        metrics = self._metrics([True, False, True, False])
        assert metrics.accepted_class_support == {"low": 1, "medium": 0, "high": 1}  # type: ignore[attr-defined]

    def test_accepted_probability_calibration_is_scored_on_accepted_rows(self) -> None:
        metrics = self._metrics([True, False, True, False])
        calibration = metrics.accepted_classification.calibration  # type: ignore[attr-defined]
        assert calibration and calibration[0].sample_count == 2


class TestSelectiveRegressionMetrics:
    def _metrics(self, accepted: list[bool], **overrides: object) -> object:
        payload: dict[str, object] = {
            "maximum_interval_width": 0.5,
            "y_true": [0.0, 1.0, 0.5, 0.25],
            "y_predicted": [0.1, 0.9, 0.5, 0.75],
            "group_ids": ["a", "a", "b", "b"],
            "accepted": accepted,
            "unavailable": [False, False, False, False],
            "interval_lower": [-0.1, 0.7, 0.3, 0.55],
            "interval_upper": [0.3, 1.1, 0.7, 0.95],
            "interval_width": [0.4, 0.4, 0.4, 0.4],
        }
        payload.update(overrides)
        return selective_regression_metrics(**payload)  # type: ignore[arg-type]

    def test_accepted_metrics_use_only_the_accepted_rows(self) -> None:
        metrics = self._metrics([True, True, False, False])
        assert metrics.accepted_regression.sample_count == 2  # type: ignore[attr-defined]

    def test_empirical_interval_coverage_uses_the_raw_bounds(self) -> None:
        metrics = self._metrics([True, True, True, True])
        # Three of the four true values fall inside their interval.
        assert metrics.empirical_interval_coverage == pytest.approx(0.75)  # type: ignore[attr-defined]

    def test_the_interval_widths_are_summarised(self) -> None:
        metrics = self._metrics([True, True, True, True])
        assert metrics.mean_interval_width == pytest.approx(0.4)  # type: ignore[attr-defined]
        assert metrics.median_interval_width == pytest.approx(0.4)  # type: ignore[attr-defined]

    def test_an_empty_accepted_set_returns_unavailable_not_zero(self) -> None:
        metrics = self._metrics([False, False, False, False])
        assert metrics.accepted_regression is None  # type: ignore[attr-defined]
        assert "rather than as zero" in metrics.unavailable_metrics["accepted_metrics"]  # type: ignore[attr-defined]

    def test_a_missing_interval_never_becomes_width_zero(self) -> None:
        metrics = self._metrics(
            [True, True, True, True],
            interval_width=[0.4, None, 0.4, 0.4],
            interval_lower=[-0.1, None, 0.3, 0.55],
            interval_upper=[0.3, None, 0.7, 0.95],
        )
        # The missing row contributes to neither the width summary nor the
        # interval-coverage denominator.
        assert metrics.mean_interval_width == pytest.approx(0.4)  # type: ignore[attr-defined]
        assert metrics.empirical_interval_coverage == pytest.approx(2 / 3)  # type: ignore[attr-defined]

    def test_r_squared_stays_unavailable_when_undefined(self) -> None:
        metrics = self._metrics(
            [True, False, False, False], y_true=[0.5, 0.5, 0.5, 0.5]
        )
        assert metrics.accepted_regression.r_squared is None  # type: ignore[attr-defined]
        assert "r_squared" in metrics.accepted_regression.unavailable_metrics  # type: ignore[attr-defined]


class TestCoverageCurve:
    def _points(self, accepted_counts: list[int]) -> list[object]:
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        return [
            selective_classification_metrics(
                threshold=threshold,
                y_true=["low"] * 4,
                y_predicted=["low"] * 3 + ["high"],
                probabilities=[[1.0, 0.0, 0.0]] * 4,
                labels=VOCABULARY,
                group_ids=["a", "a", "b", "b"],
                accepted=[index < count for index in range(4)],
                unavailable=[False] * 4,
            )
            for threshold, count in zip(grid, accepted_counts, strict=True)
        ]

    def test_coverage_is_monotonic_non_increasing(self) -> None:
        points = self._points([4, 4, 3, 1, 0])
        assert coverage_is_monotonic(
            [p.coverage_point for p in points],  # type: ignore[attr-defined]
            direction=MonotonicDirection.NON_INCREASING,
        )

    def test_a_non_monotonic_curve_is_detected(self) -> None:
        points = self._points([1, 4, 3, 1, 0])
        assert not coverage_is_monotonic(
            [p.coverage_point for p in points],  # type: ignore[attr-defined]
            direction=MonotonicDirection.NON_INCREASING,
        )

    def test_the_classification_axis_cannot_be_checked_as_non_decreasing(self) -> None:
        points = self._points([4, 4, 3, 1, 0])
        with pytest.raises(UncertaintyError, match="non_increasing"):
            coverage_is_monotonic(
                [p.coverage_point for p in points],  # type: ignore[attr-defined]
                direction=MonotonicDirection.NON_DECREASING,
            )

    def test_risk_points_state_why_a_risk_is_undefined(self) -> None:
        points = self._points([4, 4, 3, 1, 0])
        risk = risk_coverage_points(points)  # type: ignore[arg-type]
        assert risk[-1].empirical_risk is None
        assert risk[-1].unavailable_reason

    def test_the_area_needs_at_least_two_defined_points(self) -> None:
        points = self._points([1, 0, 0, 0, 0])
        area, reason = area_under_risk_coverage(
            risk_coverage_points(points)  # type: ignore[arg-type]
        )
        assert area is None
        assert "at least two points" in (reason or "")

    def test_the_area_is_a_normalised_mean_risk(self) -> None:
        points = self._points([4, 4, 3, 1, 0])
        area, reason = area_under_risk_coverage(
            risk_coverage_points(points)  # type: ignore[arg-type]
        )
        assert reason is None
        assert area is not None
        assert 0.0 <= area <= 1.0

    def test_points_all_at_one_coverage_have_no_area(self) -> None:
        points = self._points([4, 4, 4, 4, 4])
        area, reason = area_under_risk_coverage(
            risk_coverage_points(points)  # type: ignore[arg-type]
        )
        assert area is None
        assert "zero span" in (reason or "")


class TestCoverageAxes:
    """The two axes are distinct, and neither is convertible into the other."""

    def test_classification_is_selective_on_the_confidence_axis(self) -> None:
        assert (
            coverage_axis_for(TaskType.CLASSIFICATION)
            is CoverageAxis.CONFIDENCE_THRESHOLD
        )

    def test_regression_is_selective_on_the_width_axis(self) -> None:
        assert (
            coverage_axis_for(TaskType.REGRESSION)
            is CoverageAxis.MAXIMUM_INTERVAL_WIDTH
        )

    def test_raising_a_confidence_threshold_is_stricter(self) -> None:
        assert (
            expected_monotonic_direction(CoverageAxis.CONFIDENCE_THRESHOLD)
            is MonotonicDirection.NON_INCREASING
        )

    def test_raising_a_width_maximum_is_more_permissive(self) -> None:
        assert (
            expected_monotonic_direction(CoverageAxis.MAXIMUM_INTERVAL_WIDTH)
            is MonotonicDirection.NON_DECREASING
        )

    def test_the_width_axis_states_it_is_not_a_probability(self) -> None:
        from engagevr.schemas.uncertainty import COVERAGE_AXIS_UNITS

        units = COVERAGE_AXIS_UNITS[CoverageAxis.MAXIMUM_INTERVAL_WIDTH]
        assert "target's own units" in units
        assert "NOT a probability" in units

    def test_mixed_axes_cannot_be_ordered_into_one_curve(self) -> None:
        confidence = coverage_point(
            threshold=0.5,
            accepted_count=1,
            abstained_count=1,
            unavailable_count=0,
            axis=CoverageAxis.CONFIDENCE_THRESHOLD,
        )
        width = coverage_point(
            threshold=0.5,
            accepted_count=1,
            abstained_count=1,
            unavailable_count=0,
            axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
        )
        with pytest.raises(UncertaintyError, match="mixture of axes"):
            coverage_is_monotonic(
                [confidence, width], direction=MonotonicDirection.NON_INCREASING
            )


class TestRegressionWidthSweep:
    """``accept if interval_width <= W_max`` over known widths.

    The four windows carry widths 0.20, 0.40, 0.60, 0.80.  Coverage must
    RISE as the maximum rises, because a larger maximum admits more
    intervals.  Nothing here normalises a width into ``[0, 1]``, and
    nothing inverts one into ``1 - width``.
    """

    WIDTHS: ClassVar[list[float]] = [0.20, 0.40, 0.60, 0.80]

    def _coverage(self, maximum: float) -> float:
        accepted = [accepts_interval_width(width, maximum) for width in self.WIDTHS]
        metrics = selective_regression_metrics(
            maximum_interval_width=maximum,
            y_true=[0.0, 1.0, 0.5, 0.25],
            y_predicted=[0.1, 0.9, 0.5, 0.75],
            group_ids=["a", "a", "b", "b"],
            accepted=accepted,
            unavailable=[False] * 4,
            interval_lower=[
                y - w / 2
                for y, w in zip([0.1, 0.9, 0.5, 0.75], self.WIDTHS, strict=True)
            ],
            interval_upper=[
                y + w / 2
                for y, w in zip([0.1, 0.9, 0.5, 0.75], self.WIDTHS, strict=True)
            ],
            interval_width=list(self.WIDTHS),
        )
        return metrics.coverage_point.coverage

    @pytest.mark.parametrize(
        ("maximum", "expected"),
        [
            (0.10, 0.00),
            (0.20, 0.25),
            (0.50, 0.50),
            (0.80, 1.00),
            (1.00, 1.00),
        ],
    )
    def test_coverage_at_each_width_threshold(
        self, maximum: float, expected: float
    ) -> None:
        assert self._coverage(maximum) == pytest.approx(expected)

    def test_coverage_is_monotonic_non_decreasing(self) -> None:
        grid = [0.10, 0.20, 0.50, 0.80, 1.00]
        points = [
            coverage_point(
                threshold=maximum,
                accepted_count=sum(1 for w in self.WIDTHS if w <= maximum),
                abstained_count=sum(1 for w in self.WIDTHS if w > maximum),
                unavailable_count=0,
                axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
            )
            for maximum in grid
        ]
        coverages = [point.coverage for point in points]
        assert coverages == sorted(coverages)
        assert coverage_is_monotonic(
            points, direction=MonotonicDirection.NON_DECREASING
        )

    def test_a_decreasing_width_curve_is_detected(self) -> None:
        points = [
            coverage_point(
                threshold=maximum,
                accepted_count=accepted,
                abstained_count=4 - accepted,
                unavailable_count=0,
                axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
            )
            for maximum, accepted in [(0.1, 4), (0.5, 2), (0.9, 0)]
        ]
        assert not coverage_is_monotonic(
            points, direction=MonotonicDirection.NON_DECREASING
        )

    def test_the_width_axis_cannot_be_checked_as_non_increasing(self) -> None:
        points = [
            coverage_point(
                threshold=0.5,
                accepted_count=2,
                abstained_count=2,
                unavailable_count=0,
                axis=CoverageAxis.MAXIMUM_INTERVAL_WIDTH,
            )
        ]
        with pytest.raises(UncertaintyError, match="non_decreasing"):
            coverage_is_monotonic(points, direction=MonotonicDirection.NON_INCREASING)

    def test_the_axis_value_is_the_width_itself_not_a_confidence(self) -> None:
        # 2.5 is an ordinary width for a target that does not live in [0, 1].
        metrics = selective_regression_metrics(
            maximum_interval_width=2.5,
            y_true=[0.0, 1.0],
            y_predicted=[0.1, 0.9],
            group_ids=["a", "b"],
            accepted=[True, True],
            unavailable=[False, False],
            interval_lower=[-1.0, 0.0],
            interval_upper=[1.5, 2.0],
            interval_width=[2.5, 2.0],
        )
        assert metrics.axis is CoverageAxis.MAXIMUM_INTERVAL_WIDTH
        assert metrics.threshold == pytest.approx(2.5)
        assert metrics.coverage_point.threshold == pytest.approx(2.5)

    def test_an_absent_maximum_is_recorded_as_absent_not_as_zero(self) -> None:
        metrics = selective_regression_metrics(
            maximum_interval_width=None,
            no_maximum_reason="no width policy was configured",
            y_true=[0.0, 1.0],
            y_predicted=[0.1, 0.9],
            group_ids=["a", "b"],
            accepted=[True, True],
            unavailable=[False, False],
            interval_lower=[-0.1, 0.7],
            interval_upper=[0.3, 1.1],
            interval_width=[0.4, 0.4],
        )
        assert metrics.threshold is None
        assert metrics.coverage_point.threshold is None
        assert metrics.coverage_point.coverage == pytest.approx(1.0)
        assert metrics.coverage_point.threshold_unavailable_reason

    def test_an_absent_maximum_must_state_why(self) -> None:
        with pytest.raises(UncertaintyError, match="not a maximum width of zero"):
            selective_regression_metrics(
                maximum_interval_width=None,
                y_true=[0.0],
                y_predicted=[0.1],
                group_ids=["a"],
                accepted=[True],
                unavailable=[False],
                interval_lower=[-0.1],
                interval_upper=[0.3],
                interval_width=[0.4],
            )

    @pytest.mark.parametrize("maximum", [-1.0, float("nan"), float("inf")])
    def test_a_non_finite_or_negative_maximum_is_refused(self, maximum: float) -> None:
        with pytest.raises(UncertaintyError, match="finite non-negative width"):
            selective_regression_metrics(
                maximum_interval_width=maximum,
                y_true=[0.0],
                y_predicted=[0.1],
                group_ids=["a"],
                accepted=[False],
                unavailable=[False],
                interval_lower=[-0.1],
                interval_upper=[0.3],
                interval_width=[0.4],
            )


class TestRunIdentity:
    def test_the_identifier_is_deterministic(self) -> None:
        payload: dict[str, object] = {
            "target_name": "engagement_class",
            "task_type": "classification",
            "evaluation_mode": "software_self_check",
            "dataset_fingerprint": "abc",
            "split_manifest_fingerprint": "def",
            "random_seed": 42,
            "configuration": _configuration(),
            "calibration_method": "sigmoid",
            "engagevr_version": "0.1.0",
        }
        first = build_uncertainty_run_id(**payload)  # type: ignore[arg-type]
        second = build_uncertainty_run_id(**payload)  # type: ignore[arg-type]
        assert first == second
        assert first.startswith("engagement_class-uncertainty-selfcheck-")

    def test_a_different_threshold_changes_the_identifier(self) -> None:
        base: dict[str, object] = {
            "target_name": "engagement_class",
            "task_type": "classification",
            "evaluation_mode": "software_self_check",
            "dataset_fingerprint": "abc",
            "split_manifest_fingerprint": "def",
            "random_seed": 42,
            "calibration_method": "sigmoid",
            "engagevr_version": "0.1.0",
        }
        first = build_uncertainty_run_id(
            configuration=_configuration(population_confidence_threshold=0.5),
            **base,  # type: ignore[arg-type]
        )
        second = build_uncertainty_run_id(
            configuration=_configuration(population_confidence_threshold=0.9),
            **base,  # type: ignore[arg-type]
        )
        assert first != second

    def test_a_different_alpha_changes_the_identifier(self) -> None:
        base: dict[str, object] = {
            "target_name": "engagement_score",
            "task_type": "regression",
            "evaluation_mode": "software_self_check",
            "dataset_fingerprint": "abc",
            "split_manifest_fingerprint": "def",
            "random_seed": 42,
            "calibration_method": "sigmoid",
            "engagevr_version": "0.1.0",
        }
        first = build_uncertainty_run_id(
            configuration=_configuration(alpha=0.1),
            **base,  # type: ignore[arg-type]
        )
        second = build_uncertainty_run_id(
            configuration=_configuration(alpha=0.2),
            **base,  # type: ignore[arg-type]
        )
        assert first != second

    def test_a_scientific_run_is_named_differently(self) -> None:
        identifier = build_uncertainty_run_id(
            target_name="engagement_class",
            task_type="classification",
            evaluation_mode="scientific",
            dataset_fingerprint="abc",
            split_manifest_fingerprint="def",
            random_seed=42,
            configuration=_configuration(),
            calibration_method="sigmoid",
            engagevr_version="0.1.0",
        )
        assert "-sci-" in identifier


class TestAcceptanceRuleText:
    def test_each_task_type_records_its_own_rule(self) -> None:
        classification = acceptance_rule_for(TaskType.CLASSIFICATION)
        regression = acceptance_rule_for(TaskType.REGRESSION)
        assert "score >= tau" in classification
        assert "interval_width <=" in regression
        assert classification != regression

    def test_both_rules_state_the_inclusive_boundary(self) -> None:
        for rule in (
            acceptance_rule_for(TaskType.CLASSIFICATION),
            acceptance_rule_for(TaskType.REGRESSION),
        ):
            assert "INCLUSIVE" in rule
