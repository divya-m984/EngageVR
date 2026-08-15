"""Feature-group ablation tests.

An ablation is a feature-subset comparison on identical folds. These tests
assert the subsets are correct, that a missing group is reported rather
than silently reduced, and that nothing in the output claims to be a
fusion architecture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engagevr.features.catalog import FEATURE_CATALOG
from engagevr.schemas.experiments import (
    AblationDocument,
    AblationResult,
    EvaluationMode,
)
from engagevr.schemas.features import FeatureModality
from engagevr.schemas.targets import TargetName
from engagevr.training.ablation import (
    ABLATION_SPECS,
    AblationSpec,
    resolve_ablation_features,
)
from engagevr.training.calibration import CalibrationMethod
from engagevr.training.runner import RunConfiguration, run_baselines

ALL_FEATURES = frozenset(FEATURE_CATALOG.predictor_names())


class TestAblationDefinitions:
    def test_every_required_ablation_is_declared(self) -> None:
        names = {spec.name for spec in ABLATION_SPECS}
        assert names == {
            "task_only",
            "behavioural_only",
            "head_pose_only",
            "rppg_only",
            "quality_only",
            "all_available",
            "all_except_task",
            "all_except_rppg",
            "all_except_behavioural",
        }

    def test_single_modality_ablations_contain_only_that_modality(self) -> None:
        for spec in ABLATION_SPECS:
            if not spec.name.endswith("_only"):
                continue
            features, reason = resolve_ablation_features(
                spec, FEATURE_CATALOG, ALL_FEATURES
            )
            assert reason is None
            for name in features:
                assert FEATURE_CATALOG.get(name).modality in spec.modalities

    def test_all_available_covers_every_permitted_predictor(self) -> None:
        spec = next(s for s in ABLATION_SPECS if s.name == "all_available")
        features, reason = resolve_ablation_features(
            spec, FEATURE_CATALOG, ALL_FEATURES
        )
        assert reason is None
        assert set(features) == ALL_FEATURES

    def test_exclusion_ablations_remove_exactly_one_group(self) -> None:
        pairs = {
            "all_except_task": FeatureModality.TASK,
            "all_except_rppg": FeatureModality.RPPG,
            "all_except_behavioural": FeatureModality.BEHAVIOURAL,
        }
        for name, excluded in pairs.items():
            spec = next(s for s in ABLATION_SPECS if s.name == name)
            features, _reason = resolve_ablation_features(
                spec, FEATURE_CATALOG, ALL_FEATURES
            )
            modalities = {FEATURE_CATALOG.get(f).modality for f in features}
            assert excluded not in modalities
            assert len(modalities) == len(FeatureModality) - 1

    def test_features_are_returned_in_catalog_order(self) -> None:
        spec = next(s for s in ABLATION_SPECS if s.name == "all_available")
        features, _reason = resolve_ablation_features(
            spec, FEATURE_CATALOG, ALL_FEATURES
        )
        assert list(features) == list(FEATURE_CATALOG.predictor_names())

    def test_a_non_permitted_feature_never_enters_an_ablation(self) -> None:
        for spec in ABLATION_SPECS:
            features, _reason = resolve_ablation_features(
                spec, FEATURE_CATALOG, frozenset(FEATURE_CATALOG.names())
            )
            assert "rppg_method" not in features


class TestUnavailableAblations:
    def test_a_missing_feature_group_is_reported_not_silently_reduced(self) -> None:
        spec = next(s for s in ABLATION_SPECS if s.name == "rppg_only")
        without_rppg = frozenset(
            name
            for name in ALL_FEATURES
            if FEATURE_CATALOG.get(name).modality is not FeatureModality.RPPG
        )
        features, reason = resolve_ablation_features(
            spec, FEATURE_CATALOG, without_rppg
        )
        assert features == ()
        assert reason is not None
        assert "unavailable" in reason
        assert "rppg" in reason

    def test_a_partially_present_group_resolves_to_what_exists(self) -> None:
        spec = next(s for s in ABLATION_SPECS if s.name == "task_only")
        subset = frozenset({"task_correct_proportion"})
        features, reason = resolve_ablation_features(spec, FEATURE_CATALOG, subset)
        assert reason is None
        assert features == ("task_correct_proportion",)

    def test_a_modality_with_no_permitted_predictors_is_reported(self) -> None:
        spec = AblationSpec("empty", (), "no modalities")
        features, reason = resolve_ablation_features(
            spec, FEATURE_CATALOG, ALL_FEATURES
        )
        assert features == ()
        assert reason is not None

    def test_an_unavailable_result_must_state_a_reason(self) -> None:
        with pytest.raises(ValueError, match="must state"):
            AblationResult(
                ablation_name="rppg_only",
                included_modalities=("rppg",),
                feature_names=(),
                available=False,
            )


class TestSharedFolds:
    @pytest.fixture(scope="class")
    def run_result(self, m5_dataset: Path, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
        directory = tmp_path_factory.mktemp("ablation-run")
        return run_baselines(
            RunConfiguration(
                dataset_path=m5_dataset,
                target_name=TargetName.ENGAGEMENT_CLASS,
                output_directory=directory,
                n_splits=3,
                random_seed=42,
                model_names=("dummy",),
                calibration_method=CalibrationMethod.NONE,
                calibration_group_fraction=0.0,
                permutation_repeats=1,
                run_ablations=True,
            )
        )

    def test_ablations_reuse_the_run_fold_count(self, run_result) -> None:  # type: ignore[no-untyped-def]
        assert run_result.ablations is not None
        assert run_result.ablations.shared_fold_count == run_result.splits.n_splits

    def test_every_ablation_records_its_feature_names(self, run_result) -> None:  # type: ignore[no-untyped-def]
        for ablation in run_result.ablations.results:
            if ablation.available:
                assert ablation.feature_names
                assert ablation.model_name

    def test_ablations_use_one_model_so_the_subset_is_the_variable(
        self, run_result
    ) -> None:  # type: ignore[no-untyped-def]
        models = {a.model_name for a in run_result.ablations.results if a.available}
        assert len(models) == 1

    def test_repeating_the_run_reproduces_the_ablation_document(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        def once(directory: Path) -> AblationDocument:
            result = run_baselines(
                RunConfiguration(
                    dataset_path=m5_dataset,
                    target_name=TargetName.ENGAGEMENT_CLASS,
                    output_directory=directory,
                    n_splits=3,
                    random_seed=42,
                    model_names=("dummy",),
                    calibration_method=CalibrationMethod.NONE,
                    calibration_group_fraction=0.0,
                    permutation_repeats=1,
                )
            )
            assert result.ablations is not None
            return result.ablations

        first = once(tmp_path / "a")
        second = once(tmp_path / "b")
        assert first.model_dump() == second.model_dump()


class TestNoFusionClaim:
    def test_the_document_denies_being_a_fusion_architecture(self) -> None:
        document = AblationDocument(
            run_id="r",
            evaluation_mode=EvaluationMode.SOFTWARE_SELF_CHECK,
            target_name="engagement_class",
            shared_fold_count=3,
            split_strategy="stratified_group_k_fold",
            disclaimers=("SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION",),
        )
        assert "NOT a multimodal-fusion architecture" in document.note
        assert "Milestone 6" in document.note

    def test_all_available_is_described_as_a_subset_not_a_fusion(self) -> None:
        spec = next(s for s in ABLATION_SPECS if s.name == "all_available")
        assert "NOT a fusion architecture" in spec.description

    def test_the_quality_only_control_is_explained(self) -> None:
        spec = next(s for s in ABLATION_SPECS if s.name == "quality_only")
        assert "control" in spec.description
        assert "measurement conditions" in spec.description
