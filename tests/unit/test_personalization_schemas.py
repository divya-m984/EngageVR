"""Milestone 6 personalization schemas: what they refuse to represent.

These tests pin the invariants that are enforced structurally rather than
by convention: calibration must precede evaluation, a cold start must
reproduce the population prediction, an unavailable correction carries no
parameters, and a synthetic document can never be scientifically eligible.

No test here needs a webcam, a model asset, a display server, a network,
Unity, a public dataset, or participant data.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from engagevr.schemas.experiments import (
    SELF_CHECK_DISCLAIMER,
    EvaluationMode,
)
from engagevr.schemas.fusion import FusionModality
from engagevr.schemas.personalization import (
    REQUESTABLE_METHODS,
    PersonalBaselineDocument,
    PersonalBaselineStatistics,
    PersonalCalibrationSplit,
    PersonalizationConfiguration,
    PersonalizationCorrection,
    PersonalizationEvaluation,
    PersonalizationFoldResult,
    PersonalizationMethod,
    PersonalizedPrediction,
    PopulationPrediction,
)
from engagevr.schemas.targets import TaskType

_START = datetime(2026, 8, 16, 9, 0, 0, tzinfo=UTC)
_VOCABULARY = ("low", "medium", "high")


def _split(**overrides: object) -> PersonalCalibrationSplit:
    payload: dict[str, object] = {
        "subject_id": "s1",
        "fold_index": 0,
        "total_window_count": 6,
        "available": True,
        "calibration_window_ids": ("w00", "w01"),
        "evaluation_window_ids": ("w02", "w03"),
        "calibration_start_utc": _START,
        "calibration_end_utc": _START + timedelta(seconds=20),
        "evaluation_start_utc": _START + timedelta(seconds=20),
        "evaluation_end_utc": _START + timedelta(seconds=40),
        "temporal_order_verified": True,
    }
    payload.update(overrides)
    return PersonalCalibrationSplit(**payload)  # type: ignore[arg-type]


def _prediction(**overrides: object) -> PersonalizedPrediction:
    payload: dict[str, object] = {
        "window_id": "w02",
        "subject_id": "s1",
        "session_id": "sess-a",
        "target_name": "engagement_class",
        "task_type": TaskType.CLASSIFICATION,
        "fold_index": 0,
        "method": PersonalizationMethod.FEW_SHOT_CORRECTION,
        "population_predicted_class": "low",
        "population_probabilities": (0.6, 0.3, 0.1),
        "personalized_predicted_class": "medium",
        "personalized_probabilities": (0.2, 0.7, 0.1),
        "class_vocabulary": _VOCABULARY,
        "personalization_applied": True,
        "cold_start": False,
        "calibration_window_ids": ("w00", "w01"),
        "calibration_sample_count": 2,
        "data_source": "synthetic",
        "is_synthetic": True,
        "scientific_evaluation_eligible": False,
    }
    payload.update(overrides)
    return PersonalizedPrediction(**payload)  # type: ignore[arg-type]


class TestConfiguration:
    def test_the_defaults_are_valid(self) -> None:
        configuration = PersonalizationConfiguration()
        assert configuration.method in REQUESTABLE_METHODS
        assert configuration.modalities == tuple(FusionModality)
        assert configuration.calibration_windows == 5

    def test_cold_start_cannot_be_requested_as_a_method(self) -> None:
        with pytest.raises(ValidationError, match="cannot be requested"):
            PersonalizationConfiguration(method=PersonalizationMethod.COLD_START)

    def test_a_single_modality_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least two modality groups"):
            PersonalizationConfiguration(modalities=(FusionModality.TASK,))

    def test_duplicate_modalities_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            PersonalizationConfiguration(
                modalities=(FusionModality.TASK, FusionModality.TASK)
            )

    def test_a_window_count_below_the_minimum_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="could never be applied"):
            PersonalizationConfiguration(
                method=PersonalizationMethod.FEW_SHOT_CORRECTION,
                calibration_windows=2,
                minimum_calibration_windows=5,
            )

    def test_cold_start_mode_is_reachable_with_zero_windows(self) -> None:
        configuration = PersonalizationConfiguration(calibration_windows=0)
        assert configuration.calibration_windows == 0

    def test_every_equation_is_recorded(self) -> None:
        configuration = PersonalizationConfiguration()
        assert "z_s(x) = (x - mu_s) / sigma_s" in configuration.baseline_equation
        assert "b_s = mean" in configuration.regression_correction_equation
        assert "delta_c" in configuration.classification_correction_equation
        assert "NOT uncertainty calibration" in configuration.note


class TestCalibrationSplitSchema:
    def test_a_valid_split_is_accepted(self) -> None:
        assert _split().available

    def test_calibration_after_evaluation_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Calibration must precede"):
            _split(calibration_end_utc=_START + timedelta(seconds=60))

    def test_a_shared_window_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="both the calibration"):
            _split(evaluation_window_ids=("w01", "w03"))

    def test_a_window_recorded_as_both_excluded_and_used_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="both as excluded and as used"):
            _split(excluded_overlap_window_ids=("w02",))

    def test_calibration_windows_require_both_boundaries(self) -> None:
        with pytest.raises(ValidationError, match="both boundary timestamps"):
            _split(calibration_end_utc=None)

    def test_a_split_with_no_calibration_must_be_a_cold_start(self) -> None:
        with pytest.raises(ValidationError, match="is a cold start"):
            _split(
                calibration_window_ids=(),
                calibration_start_utc=None,
                calibration_end_utc=None,
                temporal_order_verified=False,
            )

    def test_a_cold_start_must_state_why(self) -> None:
        with pytest.raises(ValidationError, match="must state why"):
            _split(
                calibration_window_ids=(),
                calibration_start_utc=None,
                calibration_end_utc=None,
                temporal_order_verified=False,
                cold_start=True,
            )

    def test_an_unavailable_split_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state a reason"):
            _split(
                available=False,
                calibration_window_ids=(),
                evaluation_window_ids=(),
                calibration_start_utc=None,
                calibration_end_utc=None,
                temporal_order_verified=False,
            )

    def test_an_unavailable_split_carries_no_windows(self) -> None:
        with pytest.raises(ValidationError, match="must carry no calibration"):
            _split(available=False, unavailable_reason="too few windows")

    def test_the_ordering_rule_is_recorded_on_every_split(self) -> None:
        assert "never mixed at random" in _split().ordering_rule


class TestBaselineStatistics:
    def _record(self, **overrides: object) -> PersonalBaselineStatistics:
        payload: dict[str, object] = {
            "subject_id": "s1",
            "fold_index": 0,
            "column": "feat__task_correct_proportion",
            "feature_name": "task_correct_proportion",
            "modality": FusionModality.TASK,
            "unit": "proportion",
            "normalized": True,
            "calibration_sample_count": 3,
            "finite_sample_count": 3,
            "mean": 0.4,
            "scale": 0.2,
            "scale_source": "calibration_standard_deviation",
        }
        payload.update(overrides)
        return PersonalBaselineStatistics(**payload)  # type: ignore[arg-type]

    def test_a_valid_record_is_accepted(self) -> None:
        assert self._record().normalized

    def test_a_zero_scale_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._record(scale=0.0)

    def test_a_non_normalized_record_must_state_why(self) -> None:
        with pytest.raises(ValidationError, match="must state why"):
            self._record(normalized=False, mean=0.0, scale=1.0)

    def test_a_non_normalized_record_must_be_the_identity(self) -> None:
        with pytest.raises(ValidationError, match="identity transform"):
            self._record(
                normalized=False,
                unavailable_reason="too few values",
                mean=0.4,
                scale=0.2,
            )

    def test_the_equation_travels_with_the_numbers(self) -> None:
        assert "z_s(x)" in self._record().equation


class TestPopulationPrediction:
    def test_an_unavailable_prediction_carries_no_output(self) -> None:
        with pytest.raises(ValidationError, match="must carry no output"):
            PopulationPrediction(
                window_id="w0",
                available=False,
                unavailable_reason="the estimator refused",
                predicted_value=0.4,
            )

    def test_probabilities_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match=r"not 1\.0"):
            PopulationPrediction(
                window_id="w0",
                predicted_class="low",
                class_vocabulary=_VOCABULARY,
                probabilities=(0.5, 0.2, 0.2),
            )

    def test_exactly_one_output_kind_is_required(self) -> None:
        with pytest.raises(ValidationError, match="exactly one of"):
            PopulationPrediction(window_id="w0")


class TestCorrection:
    def test_an_unavailable_correction_carries_no_parameters(self) -> None:
        with pytest.raises(ValidationError, match="must carry no parameters"):
            PersonalizationCorrection(
                subject_id="s1",
                fold_index=0,
                method=PersonalizationMethod.FEW_SHOT_CORRECTION,
                task_type=TaskType.REGRESSION,
                available=False,
                unavailable_reason="too few calibration windows",
                calibration_sample_count=1,
                bias=0.0,
            )

    def test_a_regression_correction_must_record_a_finite_bias(self) -> None:
        with pytest.raises(ValidationError, match="finite bias"):
            PersonalizationCorrection(
                subject_id="s1",
                fold_index=0,
                method=PersonalizationMethod.FEW_SHOT_CORRECTION,
                task_type=TaskType.REGRESSION,
                available=True,
                calibration_sample_count=3,
                calibration_window_ids=("w0",),
                equation="documented",
            )

    def test_a_classification_correction_must_record_a_shift(self) -> None:
        with pytest.raises(ValidationError, match="log-odds shift per class"):
            PersonalizationCorrection(
                subject_id="s1",
                fold_index=0,
                method=PersonalizationMethod.FEW_SHOT_CORRECTION,
                task_type=TaskType.CLASSIFICATION,
                available=True,
                calibration_sample_count=3,
                calibration_window_ids=("w0",),
                equation="documented",
            )

    def test_an_applied_correction_must_record_its_equation(self) -> None:
        with pytest.raises(ValidationError, match="exact equation"):
            PersonalizationCorrection(
                subject_id="s1",
                fold_index=0,
                method=PersonalizationMethod.FEW_SHOT_CORRECTION,
                task_type=TaskType.REGRESSION,
                available=True,
                calibration_sample_count=3,
                calibration_window_ids=("w0",),
                bias=0.1,
            )

    def test_an_applied_correction_must_name_its_calibration_windows(self) -> None:
        with pytest.raises(ValidationError, match="calibration windows it was"):
            PersonalizationCorrection(
                subject_id="s1",
                fold_index=0,
                method=PersonalizationMethod.FEW_SHOT_CORRECTION,
                task_type=TaskType.REGRESSION,
                available=True,
                calibration_sample_count=3,
                bias=0.1,
                equation="documented",
            )


class TestPersonalizedPrediction:
    def test_a_valid_record_is_accepted(self) -> None:
        assert _prediction().personalization_applied

    def test_synthetic_can_never_be_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically"):
            _prediction(scientific_evaluation_eligible=True)

    def test_a_window_cannot_be_its_own_calibration_window(self) -> None:
        with pytest.raises(ValidationError, match="own calibration windows"):
            _prediction(calibration_window_ids=("w00", "w02"))

    def test_both_probability_vectors_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match=r"not 1\.0"):
            _prediction(personalized_probabilities=(0.2, 0.2, 0.1))

    def test_a_cold_start_must_reproduce_the_population_prediction(self) -> None:
        with pytest.raises(ValidationError, match="reproduce the population"):
            _prediction(
                personalization_applied=False,
                unavailable_reason="too few calibration windows",
                cold_start=True,
                cold_start_reason="too few calibration windows",
            )

    def test_a_cold_start_that_reproduces_the_population_is_accepted(self) -> None:
        record = _prediction(
            personalization_applied=False,
            unavailable_reason="too few calibration windows",
            cold_start=True,
            cold_start_reason="too few calibration windows",
            personalized_predicted_class="low",
            personalized_probabilities=(0.6, 0.3, 0.1),
            method=PersonalizationMethod.COLD_START,
        )
        assert record.cold_start
        assert record.personalized_predicted_class == record.population_predicted_class

    def test_an_unapplied_personalization_must_state_why(self) -> None:
        with pytest.raises(ValidationError, match="must state why"):
            _prediction(
                personalization_applied=False,
                personalized_predicted_class="low",
                personalized_probabilities=(0.6, 0.3, 0.1),
            )

    def test_a_regression_record_must_carry_two_finite_values(self) -> None:
        with pytest.raises(ValidationError, match="present and finite"):
            _prediction(
                task_type=TaskType.REGRESSION,
                population_predicted_class=None,
                personalized_predicted_class=None,
                population_probabilities=(),
                personalized_probabilities=(),
                class_vocabulary=(),
                population_predicted_value=0.4,
            )

    def test_a_regression_record_must_not_carry_probabilities(self) -> None:
        with pytest.raises(ValidationError, match="must not carry probabilities"):
            _prediction(
                task_type=TaskType.REGRESSION,
                population_predicted_class=None,
                personalized_predicted_class=None,
                class_vocabulary=(),
                population_predicted_value=0.4,
                personalized_predicted_value=0.5,
            )


class TestFoldResult:
    def test_mismatched_row_counts_are_rejected(self) -> None:
        from engagevr.schemas.experiments import RegressionMetrics

        with pytest.raises(ValidationError, match="exactly the same evaluation"):
            PersonalizationFoldResult(
                fold_index=0,
                population_regression_metrics=RegressionMetrics(
                    sample_count=10, independent_group_count=2
                ),
                personalized_regression_metrics=RegressionMetrics(
                    sample_count=8, independent_group_count=2
                ),
            )

    def test_an_unevaluated_fold_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state a reason"):
            PersonalizationFoldResult(fold_index=1, evaluated=False)


class TestDocuments:
    def _evaluation(self, **overrides: object) -> PersonalizationEvaluation:
        payload: dict[str, object] = {
            "run_id": "run-1",
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
            "configuration": PersonalizationConfiguration(),
            "disclaimers": (SELF_CHECK_DISCLAIMER,),
        }
        payload.update(overrides)
        return PersonalizationEvaluation(**payload)  # type: ignore[arg-type]

    def test_a_self_check_document_is_accepted(self) -> None:
        assert not self._evaluation().scientific_evaluation_eligible

    def test_a_self_check_can_never_be_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically"):
            self._evaluation(scientific_evaluation_eligible=True)

    def test_a_self_check_must_carry_the_banner(self) -> None:
        with pytest.raises(ValidationError, match="must carry the banner"):
            self._evaluation(disclaimers=("a plain note",))

    def test_a_document_without_a_disclaimer_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one disclaimer"):
            self._evaluation(disclaimers=())

    def test_the_deferred_note_names_milestone_seven(self) -> None:
        assert "Milestone 7" in self._evaluation().deferred_note
        assert "abstention" in self._evaluation().deferred_note

    def test_the_comparison_note_denies_a_benefit_claim(self) -> None:
        note = self._evaluation().comparison_note
        assert "not evidence of a personalization benefit" in note

    def test_the_baseline_document_records_what_is_never_personalized(self) -> None:
        document = PersonalBaselineDocument(
            run_id="run-1",
            evaluation_mode=EvaluationMode.SOFTWARE_SELF_CHECK,
            target_name="engagement_class",
            method=PersonalizationMethod.PERSONAL_BASELINE,
            disclaimers=(SELF_CHECK_DISCLAIMER,),
        )
        assert "modality-quality columns" in document.excluded_column_rule
        assert "Held-out subjects only" in document.scope
