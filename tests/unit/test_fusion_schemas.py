"""Fusion schema tests.

These assert the invariants that make a fused prediction auditable: a
probability vector is a distribution, an unavailable modality carries no
prediction and no weight, quality is not a modality, and a synthetic
document can never claim scientific eligibility.

No test here needs a webcam, a model asset, a display server, a network,
Unity, a public dataset, or participant data.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from engagevr.schemas.experiments import (
    SELF_CHECK_DISCLAIMER,
    SOFTWARE_SELF_CHECK_BANNER,
    EvaluationMode,
)
from engagevr.schemas.fusion import (
    ExpertDocument,
    ExpertRecord,
    FusionConfiguration,
    FusionDiagnostics,
    FusionEvaluation,
    FusionModality,
    FusionPrediction,
    FusionStrategy,
    MissingModalityScenario,
    MissingQualityPolicy,
    ModalityAvailability,
    ModalityPrediction,
    ModalityWeight,
    QualitySource,
    QualityWeightingConfiguration,
    RobustnessConfiguration,
    RobustnessResult,
    StackingConfiguration,
    ValidationWeightRecord,
)
from engagevr.schemas.targets import TaskType

ALL = tuple(FusionModality)
LABELS = ("low", "medium", "high")


def _weight(
    modality: FusionModality,
    *,
    normalized: float,
    contributed: bool = True,
    reason: str | None = None,
) -> ModalityWeight:
    return ModalityWeight(
        modality=modality,
        base_weight=1.0,
        availability=1.0 if contributed else 0.0,
        raw_effective_weight=normalized,
        normalized_weight=normalized,
        contributed=contributed,
        exclusion_reason=reason,
    )


def _prediction(modality: FusionModality, label: str = "low") -> ModalityPrediction:
    index = LABELS.index(label)
    probabilities = [0.1, 0.1, 0.1]
    probabilities[index] = 0.8
    return ModalityPrediction(
        modality=modality,
        available=True,
        predicted_class=label,
        class_vocabulary=LABELS,
        probabilities=tuple(probabilities),
    )


def _fused(**overrides: object) -> FusionPrediction:
    defaults: dict[str, object] = {
        "window_id": "w1",
        "subject_id": "synthetic-subject-0001",
        "session_id": "s1",
        "target_name": "engagement_class",
        "task_type": TaskType.CLASSIFICATION,
        "fold_index": 0,
        "strategy": FusionStrategy.UNIFORM_LATE,
        "participating_modalities": ALL,
        "available_modalities": (FusionModality.TASK,),
        "unavailable_modalities": tuple(m for m in ALL if m is not FusionModality.TASK),
        "modality_predictions": (_prediction(FusionModality.TASK),),
        "fusion_weights": (_weight(FusionModality.TASK, normalized=1.0),),
        "fused": True,
        "predicted_class": "low",
        "class_vocabulary": LABELS,
        "probabilities": (0.8, 0.1, 0.1),
        "data_source": "synthetic",
        "is_synthetic": True,
        "scientific_evaluation_eligible": False,
    }
    defaults.update(overrides)
    return FusionPrediction(**defaults)  # type: ignore[arg-type]


class TestFusionModality:
    def test_quality_is_not_a_modality(self) -> None:
        with pytest.raises(ValueError):
            FusionModality("quality")

    def test_exactly_four_measurement_modalities_exist(self) -> None:
        assert {m.value for m in FusionModality} == {
            "behavioural",
            "head_pose",
            "rppg",
            "task",
        }


class TestFusionConfiguration:
    def test_a_valid_configuration_is_accepted(self) -> None:
        configuration = FusionConfiguration(
            strategies=(FusionStrategy.EARLY, FusionStrategy.UNIFORM_LATE),
            modalities=ALL,
        )
        assert configuration.minimum_modalities == 1
        assert configuration.quality.enabled is True
        assert configuration.stacking.enabled is False

    def test_an_empty_strategy_set_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one strategy"):
            FusionConfiguration(strategies=(), modalities=ALL)

    def test_a_duplicate_strategy_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate fusion strategies"):
            FusionConfiguration(
                strategies=(FusionStrategy.EARLY, FusionStrategy.EARLY),
                modalities=ALL,
            )

    def test_a_duplicate_modality_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate fusion modalities"):
            FusionConfiguration(
                strategies=(FusionStrategy.EARLY,),
                modalities=(
                    FusionModality.TASK,
                    FusionModality.TASK,
                    FusionModality.RPPG,
                ),
            )

    def test_a_single_modality_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least two modality groups"):
            FusionConfiguration(
                strategies=(FusionStrategy.EARLY,),
                modalities=(FusionModality.TASK,),
            )

    def test_an_unsatisfiable_minimum_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="could ever satisfy it"):
            FusionConfiguration(
                strategies=(FusionStrategy.UNIFORM_LATE,),
                modalities=(FusionModality.TASK, FusionModality.RPPG),
                minimum_modalities=3,
            )

    def test_stacking_and_the_stacked_strategy_must_agree(self) -> None:
        with pytest.raises(ValidationError, match=r"stacking\.enabled is false"):
            FusionConfiguration(
                strategies=(FusionStrategy.STACKED_LATE,),
                modalities=ALL,
            )
        with pytest.raises(ValidationError, match="not among the enabled strategies"):
            FusionConfiguration(
                strategies=(FusionStrategy.UNIFORM_LATE,),
                modalities=ALL,
                stacking=StackingConfiguration(enabled=True),
            )

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FusionConfiguration(
                strategies=(FusionStrategy.EARLY,),
                modalities=ALL,
                attention_heads=8,  # type: ignore[call-arg]
            )


class TestQualityConfiguration:
    def test_defaults_are_the_deterministic_control(self) -> None:
        quality = QualityWeightingConfiguration()
        assert quality.base_weights == {}
        assert quality.missing_quality_fallback == 0.5
        assert (
            quality.missing_quality_policy is MissingQualityPolicy.DOCUMENTED_FALLBACK
        )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("missing_quality_fallback", 1.5),
            ("missing_quality_fallback", -0.1),
            ("minimum_quality", 1.2),
            ("minimum_effective_weight", 1.0),
        ],
    )
    def test_out_of_range_quality_values_are_rejected(
        self, field: str, value: float
    ) -> None:
        with pytest.raises(ValidationError):
            QualityWeightingConfiguration(**{field: value})

    def test_a_base_weight_for_quality_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not a fusion modality"):
            QualityWeightingConfiguration(base_weights={"quality": 1.0})

    def test_a_non_positive_base_weight_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="finite and positive"):
            QualityWeightingConfiguration(base_weights={"task": 0.0})

    def test_no_policy_treats_missing_quality_as_perfect(self) -> None:
        assert {p.value for p in MissingQualityPolicy} == {
            "exclude",
            "documented_fallback",
        }


class TestStackingAndRobustnessConfiguration:
    def test_a_neural_stacker_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="No neural stacker"):
            StackingConfiguration(meta_model_classification="mlp")

    def test_dropout_without_a_probability_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="positive"):
            RobustnessConfiguration(synthetic_dropout_enabled=True)


class TestModalityPrediction:
    def test_an_unavailable_prediction_needs_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state a reason"):
            ModalityPrediction(modality=FusionModality.RPPG, available=False)

    def test_an_unavailable_prediction_may_not_carry_a_value(self) -> None:
        with pytest.raises(ValidationError, match="no effective prediction"):
            ModalityPrediction(
                modality=FusionModality.RPPG,
                available=False,
                unavailable_reason="no evidence",
                predicted_value=0.0,
            )

    def test_probabilities_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match=r"not 1\.0"):
            ModalityPrediction(
                modality=FusionModality.TASK,
                available=True,
                predicted_class="low",
                class_vocabulary=LABELS,
                probabilities=(0.5, 0.2, 0.2),
            )

    def test_a_non_finite_probability_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not finite"):
            ModalityPrediction(
                modality=FusionModality.TASK,
                available=True,
                predicted_class="low",
                class_vocabulary=LABELS,
                probabilities=(math.nan, 0.5, 0.5),
            )

    def test_a_wrong_length_probability_vector_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="probabilities were supplied"):
            ModalityPrediction(
                modality=FusionModality.TASK,
                available=True,
                predicted_class="low",
                class_vocabulary=LABELS,
                probabilities=(0.5, 0.5),
            )

    def test_a_non_finite_regression_prediction_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-finite regression"):
            ModalityPrediction(
                modality=FusionModality.TASK,
                available=True,
                predicted_value=math.inf,
            )

    def test_a_finite_regression_prediction_is_accepted(self) -> None:
        prediction = ModalityPrediction(
            modality=FusionModality.TASK, available=True, predicted_value=0.42
        )
        assert prediction.predicted_value == pytest.approx(0.42)

    def test_quality_outside_the_unit_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModalityPrediction(
                modality=FusionModality.RPPG,
                available=True,
                predicted_value=0.1,
                quality=1.5,
            )


class TestModalityWeight:
    def test_a_non_contributing_modality_gets_no_weight(self) -> None:
        with pytest.raises(ValidationError, match="no effective prediction weight"):
            ModalityWeight(
                modality=FusionModality.RPPG,
                base_weight=1.0,
                availability=0.0,
                raw_effective_weight=0.0,
                normalized_weight=0.5,
                contributed=False,
                exclusion_reason="unavailable",
            )

    def test_an_excluded_modality_states_why(self) -> None:
        with pytest.raises(ValidationError, match="must state why"):
            ModalityWeight(
                modality=FusionModality.RPPG,
                base_weight=1.0,
                availability=0.0,
                raw_effective_weight=0.0,
                normalized_weight=0.0,
                contributed=False,
            )

    def test_quality_source_is_recorded(self) -> None:
        weight = ModalityWeight(
            modality=FusionModality.RPPG,
            base_weight=1.0,
            availability=1.0,
            quality_used=0.4,
            quality_source=QualitySource.MEASURED,
            normalized_quality=0.4,
            raw_effective_weight=0.4,
            normalized_weight=1.0,
            contributed=True,
        )
        assert weight.quality_source is QualitySource.MEASURED


class TestFusionPrediction:
    def test_a_valid_fused_prediction_round_trips(self) -> None:
        prediction = _fused()
        assert prediction.fused
        assert sum(prediction.probabilities) == pytest.approx(1.0)

    def test_an_unfused_prediction_needs_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state a reason"):
            _fused(
                fused=False,
                predicted_class=None,
                probabilities=(),
                fusion_weights=(),
            )

    def test_an_unfused_prediction_may_not_carry_a_prediction(self) -> None:
        with pytest.raises(ValidationError, match="fusion was unavailable"):
            _fused(
                fused=False,
                unavailable_reason="no expert was available",
                fusion_weights=(),
            )

    def test_probabilities_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match=r"not 1\.0"):
            _fused(probabilities=(0.5, 0.2, 0.2))

    def test_weights_must_sum_to_one_across_contributors(self) -> None:
        with pytest.raises(ValidationError, match="normalised fusion weights sum"):
            _fused(fusion_weights=(_weight(FusionModality.TASK, normalized=0.5),))

    def test_an_unavailable_modality_may_not_receive_weight(self) -> None:
        with pytest.raises(ValidationError, match="weight given to modalities"):
            _fused(
                fusion_weights=(
                    _weight(FusionModality.TASK, normalized=0.5),
                    _weight(FusionModality.RPPG, normalized=0.5),
                )
            )

    def test_available_and_unavailable_must_be_disjoint(self) -> None:
        with pytest.raises(ValidationError, match="both available and unavailable"):
            _fused(
                available_modalities=(FusionModality.TASK,),
                unavailable_modalities=(FusionModality.TASK,),
            )

    def test_synthetic_can_never_be_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically eligible"):
            _fused(scientific_evaluation_eligible=True)

    def test_a_regression_fusion_must_be_finite(self) -> None:
        with pytest.raises(ValidationError, match="not finite"):
            _fused(
                task_type=TaskType.REGRESSION,
                predicted_class=None,
                class_vocabulary=(),
                probabilities=(),
                predicted_value=math.inf,
                modality_predictions=(),
            )

    def test_a_regression_fusion_carries_no_probabilities(self) -> None:
        with pytest.raises(ValidationError, match="exactly one of"):
            _fused(
                task_type=TaskType.REGRESSION,
                predicted_value=0.5,
                modality_predictions=(),
            )

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _fused(attention_weight=0.5)


class TestSupportingDocuments:
    def test_an_unavailable_modality_availability_states_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state a reason"):
            ModalityAvailability(modality=FusionModality.RPPG, available=False)

        record = ModalityAvailability(
            modality=FusionModality.RPPG,
            available=True,
            quality=0.7,
            quality_recorded=True,
        )
        assert record.quality == pytest.approx(0.7)

    def test_an_untrained_expert_states_why(self) -> None:
        with pytest.raises(ValidationError, match="must state why"):
            ExpertRecord(
                modality=FusionModality.RPPG,
                fold_index=0,
                model_name="ridge",
                trained=False,
            )

    def test_a_trained_expert_records_its_features(self) -> None:
        with pytest.raises(ValidationError, match="must record the features"):
            ExpertRecord(
                modality=FusionModality.RPPG,
                fold_index=0,
                model_name="ridge",
                trained=True,
            )

    def test_diagnostics_counts_must_reconcile(self) -> None:
        with pytest.raises(ValidationError, match="must equal sample_count"):
            FusionDiagnostics(
                sample_count=10, fused_count=8, unavailable_fusion_count=1
            )

    def test_a_scenario_partitions_the_configured_modalities(self) -> None:
        scenario = MissingModalityScenario(
            name="missing_rppg",
            absent_modalities=(FusionModality.RPPG,),
            description="rPPG absent",
        )
        assert scenario.present(ALL) == (
            FusionModality.BEHAVIOURAL,
            FusionModality.HEAD_POSE,
            FusionModality.TASK,
        )

    def test_an_unevaluated_scenario_states_why(self) -> None:
        with pytest.raises(ValidationError, match="must state a reason"):
            RobustnessResult(
                scenario_name="only_rppg",
                scenario_description="only rPPG",
                strategy=FusionStrategy.UNIFORM_LATE,
                evaluated=False,
            )

    def test_a_negative_validation_weight_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="finite and non-negative"):
            ValidationWeightRecord(
                fold_index=0,
                metric_name="balanced_accuracy",
                metric_definition="documented",
                weights={"task": -1.0},
            )


class TestSelfCheckInvariants:
    def _evaluation(self, **overrides: object) -> FusionEvaluation:
        defaults: dict[str, object] = {
            "run_id": "engagement_class-fusion-selfcheck-000000000000",
            "evaluation_mode": EvaluationMode.SOFTWARE_SELF_CHECK,
            "scientific_evaluation_eligible": False,
            "target_name": "engagement_class",
            "task_type": TaskType.CLASSIFICATION,
            "dataset_fingerprint": "f" * 64,
            "split_manifest_fingerprint": "a" * 64,
            "group_field": "subject_id",
            "group_count": 12,
            "fold_count": 3,
            "random_seed": 42,
            "disclaimers": (SELF_CHECK_DISCLAIMER,),
        }
        defaults.update(overrides)
        return FusionEvaluation(**defaults)  # type: ignore[arg-type]

    def test_a_self_check_cannot_be_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically eligible"):
            self._evaluation(scientific_evaluation_eligible=True)

    def test_a_self_check_must_carry_the_banner(self) -> None:
        with pytest.raises(ValidationError, match="must carry the banner"):
            self._evaluation(disclaimers=("something else",))

    def test_a_document_must_carry_a_disclaimer(self) -> None:
        with pytest.raises(ValidationError, match="at least one disclaimer"):
            self._evaluation(disclaimers=())

    def test_the_banner_is_accepted(self) -> None:
        evaluation = self._evaluation()
        assert any(SOFTWARE_SELF_CHECK_BANNER in d for d in evaluation.disclaimers)

    def test_the_expert_document_requires_a_disclaimer(self) -> None:
        with pytest.raises(ValidationError, match="at least one disclaimer"):
            ExpertDocument(
                run_id="r",
                evaluation_mode=EvaluationMode.SOFTWARE_SELF_CHECK,
                target_name="engagement_class",
                task_type=TaskType.CLASSIFICATION,
                disclaimers=(),
            )
