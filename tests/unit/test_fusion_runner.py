"""End-to-end fusion-run tests: artifacts, leakage, privacy, determinism.

These run the whole pipeline on a small deterministic SYNTHETIC dataset and
then inspect what it wrote.  Everything asserted is a property of the
software; none of it is evidence about a person.

No test here needs a webcam, a model asset, a display server, a network,
Unity, a public dataset, or participant data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engagevr.features.catalog import FEATURE_CATALOG
from engagevr.schemas.experiments import (
    SOFTWARE_SELF_CHECK_BANNER,
    EvaluationMode,
    RunStatus,
)
from engagevr.schemas.fusion import (
    FusionConfiguration,
    FusionModality,
    FusionStrategy,
    MissingQualityPolicy,
    QualityWeightingConfiguration,
    RobustnessConfiguration,
)
from engagevr.schemas.targets import TargetName, TaskType
from engagevr.training.artifacts import verify_checksums
from engagevr.training.fusion import FEATURE_MODALITY_OF
from engagevr.training.fusion_artifacts import (
    FUSION_REQUIRED_ARTIFACTS,
    build_fusion_run_id,
    split_manifest_fingerprint,
)
from engagevr.training.fusion_runner import (
    FusionConfigurationError,
    FusionRunConfiguration,
    FusionRunResult,
    run_fusion,
)
from engagevr.training.runner import (
    RunConfiguration,
    ScientificModeError,
    run_baselines,
)
from engagevr.training.splits import SplitConfigurationError

ALL = tuple(FusionModality)

#: Phrases that must never appear in a fusion artifact.
FORBIDDEN_CLAIMS = (
    "clinically validated",
    "diagnostic accuracy",
    "proven to measure",
    "production-ready",
    "champion model",
    "experimentally validated",
    "best fusion model",
    "state of the art",
)

#: Tokens that must not appear anywhere in a fusion run directory. These are
#: values and secrets, not vocabulary: the feature catalog legitimately says
#: "landmark ratio" when describing what a geometric proxy is.
FORBIDDEN_IDENTIFIERS = (
    "@example.com",
    "password",
    "api_key",
    "secret_key",
    "access_token",
    "first_name",
    "last_name",
)

#: Column-name fragments that would mean raw media reached an artifact.
FORBIDDEN_COLUMN_TOKENS = (
    "landmark",
    "raw_frame",
    "frame_bytes",
    "pixel_array",
    "thumbnail",
)


def _json_documents(directory: Path) -> dict[str, object]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    }


class TestArtifacts:
    def test_every_required_artifact_is_written(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        directory = m6_classification_run.directory
        for name in FUSION_REQUIRED_ARTIFACTS:
            assert (directory / name).exists(), name
        for name in (
            "predictions.parquet",
            "expert_predictions.parquet",
            "fusion_weights.parquet",
            "feature_importance.parquet",
            "calibration.json",
            "checksums.json",
            "manifest.json",
        ):
            assert (directory / name).exists(), name
        assert (directory / "models" / "README.txt").exists()

    def test_the_manifest_records_completion(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        manifest = m6_classification_run.manifest
        assert manifest.status is RunStatus.COMPLETED
        assert manifest.failure_reason is None
        assert manifest.evaluation_mode is EvaluationMode.SOFTWARE_SELF_CHECK
        assert manifest.scientific_evaluation_eligible is False
        assert manifest.configuration["milestone"] == 6

    def test_checksums_cover_every_artifact(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        assert verify_checksums(m6_classification_run.directory) == ()
        recorded = json.loads(
            (m6_classification_run.directory / "checksums.json").read_text()
        )
        for name in FUSION_REQUIRED_ARTIFACTS:
            assert name in recorded

    def test_no_temporary_file_survives(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        leftovers = list(m6_classification_run.directory.glob("*.tmp"))
        leftovers += list(m6_classification_run.directory.glob(".*.tmp"))
        assert leftovers == []

    def test_no_mlflow_or_dvc_artifact_appears(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        names = {p.name.lower() for p in m6_classification_run.directory.rglob("*")}
        assert not any("mlflow" in name for name in names)
        assert not any(name.endswith(".dvc") for name in names)
        assert "mlruns" not in names
        assert "dockerfile" not in names

    def test_a_failed_run_is_recorded_as_failed(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        configuration = FusionConfiguration(
            strategies=(FusionStrategy.UNIFORM_LATE,), modalities=ALL
        )
        directory = tmp_path / "failed"
        with pytest.raises(SplitConfigurationError):
            run_fusion(
                FusionRunConfiguration(
                    dataset_path=m5_dataset,
                    target_name=TargetName.ENGAGEMENT_CLASS,
                    output_directory=directory,
                    fusion=configuration,
                    n_splits=200,
                )
            )
        # A run that raises before it can build folds never reaches the
        # manifest writer, so the directory holds no manifest at all — which
        # read_manifest reports as an interrupted run, never as a success.
        assert not (directory / "manifest.json").exists()


class TestFusedPredictions:
    def test_probabilities_are_finite_and_sum_to_one(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        frame = pd.read_parquet(m6_classification_run.directory / "predictions.parquet")
        columns = [c for c in frame.columns if c.startswith("probability__")]
        fused = frame[frame.fused]
        values = fused[columns].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert np.allclose(values.sum(axis=1), 1.0)
        assert (values >= 0.0).all()

    def test_an_unfused_window_carries_no_prediction(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        frame = pd.read_parquet(m6_classification_run.directory / "predictions.parquet")
        unfused = frame[~frame.fused]
        assert not unfused.empty
        assert unfused.predicted_class.isna().all()
        assert unfused.unavailable_reason.notna().all()
        columns = [c for c in frame.columns if c.startswith("probability__")]
        assert unfused[columns].isna().all().all()

    def test_regression_predictions_are_finite(
        self, m6_regression_run: FusionRunResult
    ) -> None:
        frame = pd.read_parquet(m6_regression_run.directory / "predictions.parquet")
        fused = frame[frame.fused]
        assert not fused.empty
        assert np.isfinite(fused.predicted_value.to_numpy(dtype=float)).all()
        assert "probability__low" not in frame.columns

    def test_every_strategy_and_scenario_is_represented(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        frame = pd.read_parquet(m6_classification_run.directory / "predictions.parquet")
        assert set(frame.strategy.unique()) == {
            "early",
            "uniform_late",
            "quality_late",
        }
        assert len(frame.scenario.unique()) == 10

    def test_synthetic_rows_are_never_scientifically_eligible(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        frame = pd.read_parquet(m6_classification_run.directory / "predictions.parquet")
        assert frame.is_synthetic.all()
        assert not frame.scientific_evaluation_eligible.any()


class TestFusionWeights:
    def test_contributing_weights_sum_to_one(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        frame = pd.read_parquet(
            m6_classification_run.directory / "fusion_weights.parquet"
        )
        totals = frame.groupby(["window_id", "strategy", "scenario", "fold_index"])[
            "normalized_weight"
        ].sum()
        assert np.allclose(totals.to_numpy(dtype=float), 1.0)

    def test_a_non_contributing_modality_receives_no_weight(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        frame = pd.read_parquet(
            m6_classification_run.directory / "fusion_weights.parquet"
        )
        excluded = frame[~frame.contributed]
        assert not excluded.empty
        assert (excluded.normalized_weight == 0.0).all()
        assert (excluded.raw_effective_weight == 0.0).all()
        assert excluded.exclusion_reason.notna().all()

    def test_quality_and_uniform_weighting_differ(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        frame = pd.read_parquet(
            m6_classification_run.directory / "fusion_weights.parquet"
        )
        uniform = frame[frame.strategy == "uniform_late"]
        quality = frame[frame.strategy == "quality_late"]
        assert (uniform.quality_source == "not_used").all()
        assert (quality.quality_source == "measured").any()
        merged = uniform.merge(
            quality,
            on=["window_id", "scenario", "modality"],
            suffixes=("_uniform", "_quality"),
        )
        assert not np.allclose(
            merged.normalized_weight_uniform.to_numpy(dtype=float),
            merged.normalized_weight_quality.to_numpy(dtype=float),
        )

    def test_early_fusion_records_no_per_window_weights(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        frame = pd.read_parquet(
            m6_classification_run.directory / "fusion_weights.parquet"
        )
        assert "early" not in set(frame.strategy.unique())


class TestExpertsAndFeatureSets:
    def test_every_expert_feature_belongs_to_its_modality(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        for record in m6_classification_run.experts.experts:
            for column in record.feature_names:
                if column.startswith(("feat__", "avail__")):
                    name = column.split("__", 1)[1]
                    assert (
                        FEATURE_CATALOG.get(name).modality
                        is FEATURE_MODALITY_OF[record.modality]
                    )
                else:
                    assert column == f"modality_available__{record.modality.value}"

    def test_the_early_matrix_carries_no_target_or_identifier(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        document = json.loads(
            (m6_classification_run.directory / "fusion_config.json").read_text()
        )
        for column in document["early_fusion_columns"]:
            assert not column.startswith(("target__", "target_meta__"))
            assert column not in {
                "window_id",
                "subject_id",
                "session_id",
                "window_start_utc",
                "window_index",
            }

    def test_the_early_matrix_spans_every_configured_modality(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        document = json.loads(
            (m6_classification_run.directory / "fusion_config.json").read_text()
        )
        columns = set(document["early_fusion_columns"])
        for modality in ALL:
            assert f"modality_available__{modality.value}" in columns

    def test_modality_quality_stays_out_of_the_experts_by_default(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        for record in m6_classification_run.experts.experts:
            assert not any(
                column.startswith("modality_quality__")
                for column in record.feature_names
            )

    def test_an_untrained_expert_states_a_reason(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        for record in m6_classification_run.experts.experts:
            if not record.trained:
                assert record.unavailable_reason
                assert record.predicted_row_count == 0


class TestSplitReuseAndLeakage:
    def test_the_split_manifest_matches_a_baseline_run(
        self, m5_dataset: Path, tmp_path: Path, m6_classification_run: FusionRunResult
    ) -> None:
        baseline = run_baselines(
            RunConfiguration(
                dataset_path=m5_dataset,
                target_name=TargetName.ENGAGEMENT_CLASS,
                output_directory=tmp_path / "baseline",
                n_splits=3,
                model_names=("dummy",),
                run_ablations=False,
            )
        )
        assert baseline.splits.model_dump(mode="json") == (
            m6_classification_run.splits.model_dump(mode="json")
        )
        assert split_manifest_fingerprint(baseline.splits) == (
            split_manifest_fingerprint(m6_classification_run.splits)
        )

    def test_no_group_appears_on_both_sides_of_a_fold(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        assert m6_classification_run.splits.audit_passed
        for fold in m6_classification_run.splits.folds:
            assert not set(fold.train_groups) & set(fold.test_groups)
            assert not set(fold.calibration_groups) & set(fold.test_groups)
            assert set(fold.calibration_groups) <= set(fold.train_groups)

    def test_a_session_never_straddles_two_test_folds(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        frame = pd.read_parquet(m6_classification_run.directory / "predictions.parquet")
        per_session = frame.groupby("session_id")["fold_index"].nunique()
        assert (per_session == 1).all()

    def test_validation_weights_never_see_the_outer_test_groups(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        configuration = FusionConfiguration(
            strategies=(FusionStrategy.VALIDATION_WEIGHTED_LATE,),
            modalities=ALL,
            robustness=RobustnessConfiguration(enabled=False),
        )
        result = run_fusion(
            FusionRunConfiguration(
                dataset_path=m5_dataset,
                target_name=TargetName.ENGAGEMENT_SCORE,
                output_directory=tmp_path / "validation-weighted",
                fusion=configuration,
                n_splits=3,
            )
        )
        strategy = result.fusion_metrics.strategies[0]
        assert strategy.validation_weights
        by_fold = {f.fold_index: f for f in result.splits.folds}
        for record in strategy.validation_weights:
            fold = by_fold[record.fold_index]
            assert set(record.groups_used) <= set(fold.train_groups)
            assert not set(record.groups_used) & set(fold.test_groups)
            assert record.metric_definition
            assert all(value >= 0.0 for value in record.weights.values())


class TestDiagnosticsAndRobustness:
    def test_coverage_is_recorded_for_every_scenario(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        for entry in m6_classification_run.robustness.results:
            if not entry.evaluated:
                assert entry.unavailable_reason
                continue
            assert entry.coverage is not None
            assert 0.0 <= entry.coverage <= 1.0
            assert (
                entry.fused_window_count + entry.unavailable_fusion_count
                == entry.evaluated_window_count
            )

    def test_a_single_modality_scenario_reduces_coverage(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        by_key = {
            (r.scenario_name, r.strategy.value): r
            for r in m6_classification_run.robustness.results
        }
        reference = by_key[("all_modalities", "uniform_late")]
        only_task = by_key[("only_task", "uniform_late")]
        assert reference.coverage == pytest.approx(1.0)
        assert only_task.coverage is not None
        assert only_task.coverage < reference.coverage

    def test_missing_modality_rates_reflect_the_scenario(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        by_key = {
            (r.scenario_name, r.strategy.value): r
            for r in m6_classification_run.robustness.results
        }
        entry = by_key[("missing_rppg", "uniform_late")]
        assert entry.diagnostics is not None
        assert entry.diagnostics.missing_modality_rate["rppg"] == pytest.approx(1.0)
        assert entry.diagnostics.modality_contribution_counts["rppg"] == 0

    def test_disagreement_is_labelled_as_a_diagnostic(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        strategy = next(
            s
            for s in m6_classification_run.fusion_metrics.strategies
            if s.strategy is FusionStrategy.UNIFORM_LATE
        )
        diagnostics = strategy.folds[0].diagnostics
        assert diagnostics is not None
        disagreement = diagnostics.disagreement
        assert disagreement is not None
        assert "not a calibrated uncertainty estimate" in disagreement.note
        assert disagreement.unanimous_fraction is not None
        assert 0.0 <= disagreement.unanimous_fraction <= 1.0

    def test_a_single_expert_window_yields_no_disagreement(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        by_key = {
            (r.scenario_name, r.strategy.value): r
            for r in m6_classification_run.robustness.results
        }
        entry = by_key[("only_task", "uniform_late")]
        assert entry.diagnostics is not None
        disagreement = entry.diagnostics.disagreement
        assert disagreement is not None
        assert disagreement.evaluated_window_count == 0
        assert disagreement.insufficient_expert_window_count > 0

    def test_early_fusion_reports_no_per_modality_mean_weight(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        strategy = next(
            s
            for s in m6_classification_run.fusion_metrics.strategies
            if s.strategy is FusionStrategy.EARLY
        )
        diagnostics = strategy.folds[0].diagnostics
        assert diagnostics is not None
        assert all(v is None for v in diagnostics.mean_normalized_weight.values())


class TestSyntheticDropout:
    def test_dropout_lowers_availability_and_is_recorded(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        configuration = FusionConfiguration(
            strategies=(FusionStrategy.UNIFORM_LATE,),
            modalities=ALL,
            robustness=RobustnessConfiguration(
                enabled=False,
                synthetic_dropout_enabled=True,
                synthetic_dropout_seed=7,
                synthetic_dropout_probability=0.5,
            ),
        )
        result = run_fusion(
            FusionRunConfiguration(
                dataset_path=m5_dataset,
                target_name=TargetName.ENGAGEMENT_CLASS,
                output_directory=tmp_path / "dropout",
                fusion=configuration,
                n_splits=3,
            )
        )
        assert result.robustness.synthetic_dropout_applied
        assert result.robustness.synthetic_dropout_seed == 7
        assert result.robustness.synthetic_dropout_probability == pytest.approx(0.5)
        frame = pd.read_parquet(result.directory / "predictions.parquet")
        assert frame.available_expert_count.mean() < 3.0

    def test_scientific_mode_refuses_synthetic_dropout(
        self, m5_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The synthetic-data gate fires first on this dataset, which is
        # correct; suspending it isolates the dropout gate behind it.
        from engagevr.training import fusion_runner

        monkeypatch.setattr(
            fusion_runner, "assert_scientific_eligibility", lambda _frame: None
        )
        configuration = FusionConfiguration(
            strategies=(FusionStrategy.UNIFORM_LATE,),
            modalities=ALL,
            robustness=RobustnessConfiguration(
                enabled=False,
                synthetic_dropout_enabled=True,
                synthetic_dropout_probability=0.5,
            ),
        )
        with pytest.raises(ScientificModeError, match="fabricates an availability"):
            run_fusion(
                FusionRunConfiguration(
                    dataset_path=m5_dataset,
                    target_name=TargetName.ENGAGEMENT_CLASS,
                    output_directory=tmp_path / "scientific-dropout",
                    fusion=configuration,
                    evaluation_mode=EvaluationMode.SCIENTIFIC,
                    n_splits=3,
                )
            )

    def test_the_synthetic_data_gate_fires_first_in_scientific_mode(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        configuration = FusionConfiguration(
            strategies=(FusionStrategy.UNIFORM_LATE,),
            modalities=ALL,
            robustness=RobustnessConfiguration(
                enabled=False,
                synthetic_dropout_enabled=True,
                synthetic_dropout_probability=0.5,
            ),
        )
        with pytest.raises(ScientificModeError, match="data_source='synthetic'"):
            run_fusion(
                FusionRunConfiguration(
                    dataset_path=m5_dataset,
                    target_name=TargetName.ENGAGEMENT_CLASS,
                    output_directory=tmp_path / "scientific-dropout-order",
                    fusion=configuration,
                    evaluation_mode=EvaluationMode.SCIENTIFIC,
                    n_splits=3,
                )
            )


class TestScientificMode:
    def test_scientific_mode_refuses_a_synthetic_dataset(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        configuration = FusionConfiguration(
            strategies=(FusionStrategy.UNIFORM_LATE,), modalities=ALL
        )
        with pytest.raises(ScientificModeError, match="data_source='synthetic'"):
            run_fusion(
                FusionRunConfiguration(
                    dataset_path=m5_dataset,
                    target_name=TargetName.ENGAGEMENT_CLASS,
                    output_directory=tmp_path / "scientific",
                    fusion=configuration,
                    evaluation_mode=EvaluationMode.SCIENTIFIC,
                    n_splits=3,
                )
            )

    def test_a_missing_modality_group_is_refused(
        self, m5_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from engagevr.training import fusion_runner

        def _empty(*_args: object, **_kwargs: object) -> tuple[str, ...]:
            return ()

        monkeypatch.setattr(fusion_runner, "modality_expert_columns", _empty)
        configuration = FusionConfiguration(
            strategies=(FusionStrategy.UNIFORM_LATE,), modalities=ALL
        )
        with pytest.raises(FusionConfigurationError, match="no permitted predictor"):
            run_fusion(
                FusionRunConfiguration(
                    dataset_path=m5_dataset,
                    target_name=TargetName.ENGAGEMENT_CLASS,
                    output_directory=tmp_path / "missing-modality",
                    fusion=configuration,
                    n_splits=3,
                )
            )


class TestDeterminism:
    def test_two_identical_runs_agree(self, m5_dataset: Path, tmp_path: Path) -> None:
        configuration = FusionConfiguration(
            strategies=(FusionStrategy.UNIFORM_LATE, FusionStrategy.QUALITY_LATE),
            modalities=ALL,
            robustness=RobustnessConfiguration(enabled=False),
        )
        results = [
            run_fusion(
                FusionRunConfiguration(
                    dataset_path=m5_dataset,
                    target_name=TargetName.ENGAGEMENT_CLASS,
                    output_directory=tmp_path / name,
                    fusion=configuration,
                    n_splits=3,
                )
            )
            for name in ("repeat-a", "repeat-b")
        ]
        assert results[0].run_id == results[1].run_id
        for name in ("metrics.json", "fusion_metrics.json", "splits.json"):
            first = (results[0].directory / name).read_text()
            second = (results[1].directory / name).read_text()
            assert first == second
        checksums = [
            json.loads((result.directory / "checksums.json").read_text())
            for result in results
        ]
        for name in ("metrics.json", "fusion_metrics.json", "splits.json"):
            assert checksums[0][name] == checksums[1][name]

    def test_the_run_id_ignores_request_ordering(self) -> None:
        first = FusionConfiguration(
            strategies=(FusionStrategy.EARLY, FusionStrategy.UNIFORM_LATE),
            modalities=(FusionModality.TASK, FusionModality.RPPG),
        )
        second = FusionConfiguration(
            strategies=(FusionStrategy.UNIFORM_LATE, FusionStrategy.EARLY),
            modalities=(FusionModality.RPPG, FusionModality.TASK),
        )
        arguments = {
            "target_name": "engagement_class",
            "task_type": "classification",
            "evaluation_mode": "software_self_check",
            "dataset_fingerprint": "f" * 64,
            "split_manifest_fingerprint": "a" * 64,
            "random_seed": 42,
            "calibration_method": "sigmoid",
            "scenario_names": ["all_modalities"],
        }
        assert build_fusion_run_id(fusion=first, **arguments) == build_fusion_run_id(  # type: ignore[arg-type]
            fusion=second,
            **arguments,  # type: ignore[arg-type]
        )

    def test_the_run_id_changes_with_the_quality_policy(self) -> None:
        arguments = {
            "target_name": "engagement_class",
            "task_type": "classification",
            "evaluation_mode": "software_self_check",
            "dataset_fingerprint": "f" * 64,
            "split_manifest_fingerprint": "a" * 64,
            "random_seed": 42,
            "calibration_method": "sigmoid",
            "scenario_names": ["all_modalities"],
        }
        default = FusionConfiguration(
            strategies=(FusionStrategy.QUALITY_LATE,), modalities=ALL
        )
        strict = FusionConfiguration(
            strategies=(FusionStrategy.QUALITY_LATE,),
            modalities=ALL,
            quality=QualityWeightingConfiguration(
                missing_quality_policy=MissingQualityPolicy.EXCLUDE
            ),
        )
        assert build_fusion_run_id(fusion=default, **arguments) != build_fusion_run_id(  # type: ignore[arg-type]
            fusion=strict,
            **arguments,  # type: ignore[arg-type]
        )


class TestPrivacyAndClaims:
    def test_no_artifact_claims_scientific_validity(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        for name, document in _json_documents(m6_classification_run.directory).items():
            text = json.dumps(document).lower()
            for claim in FORBIDDEN_CLAIMS:
                assert claim not in text, f"{name} contains {claim!r}"

    def test_no_artifact_carries_a_personal_identifier(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        for name, document in _json_documents(m6_classification_run.directory).items():
            text = json.dumps(document).lower()
            for token in FORBIDDEN_IDENTIFIERS:
                assert token not in text, f"{name} contains {token!r}"

    def test_no_parquet_carries_a_raw_frame_or_landmark(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        for path in m6_classification_run.directory.glob("*.parquet"):
            frame = pd.read_parquet(path)
            for column in frame.columns:
                assert not any(
                    token in str(column).lower() for token in FORBIDDEN_COLUMN_TOKENS
                )

    def test_subjects_stay_pseudonymous(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        frame = pd.read_parquet(m6_classification_run.directory / "predictions.parquet")
        assert frame.subject_id.str.startswith("synthetic-subject-").all()

    def test_the_self_check_banner_is_persisted(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        for name in (
            "metrics.json",
            "fusion_metrics.json",
            "robustness.json",
            "experts.json",
            "fusion_config.json",
            "manifest.json",
        ):
            document = json.loads(
                (m6_classification_run.directory / name).read_text(encoding="utf-8")
            )
            assert document["evaluation_mode"] == "software_self_check"
            assert any(
                SOFTWARE_SELF_CHECK_BANNER in disclaimer
                for disclaimer in document["disclaimers"]
            ), name

    def test_the_comparison_note_denies_champion_selection(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        note = m6_classification_run.fusion_metrics.comparison_note
        assert "No strategy is a champion" in note
        assert "cannot select a best fusion architecture" in note


class TestMetricsDocument:
    def test_each_strategy_and_expert_appears(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        names = {r.model_name for r in m6_classification_run.metrics.results}
        assert {"early", "uniform_late", "quality_late"} <= names
        assert {f"unimodal_{m.value}" for m in ALL} <= names
        kinds = {r.model_kind for r in m6_classification_run.metrics.results}
        assert kinds == {"fusion", "unimodal_expert"}

    def test_the_unimodal_control_is_labelled_descriptive(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        control = m6_classification_run.fusion_metrics.unimodal_control
        assert control is not None
        assert control.metric_name == "balanced_accuracy"
        assert "optimistically biased" in control.note
        assert "never used to choose a fusion strategy" in control.note

    def test_regression_uses_error_metrics(
        self, m6_regression_run: FusionRunResult
    ) -> None:
        strategy = m6_regression_run.fusion_metrics.strategies[0]
        names = {a.name for a in strategy.aggregate}
        assert "mean_absolute_error" in names
        assert "root_mean_squared_error" in names
        assert m6_regression_run.fusion_metrics.task_type is TaskType.REGRESSION

    def test_stacking_provenance_is_recorded(
        self, m6_regression_run: FusionRunResult
    ) -> None:
        strategy = next(
            s
            for s in m6_regression_run.fusion_metrics.strategies
            if s.strategy is FusionStrategy.STACKED_LATE
        )
        assert strategy.stacking_provenance
        for record in strategy.stacking_provenance:
            if record.available:
                assert record.meta_model_name == "ridge"
                assert record.meta_training_row_count > 0
                assert len(record.leakage_checks_passed) == 3
                assert record.probabilities_are_calibrated is False

    def test_fold_results_precede_aggregates(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        for strategy in m6_classification_run.fusion_metrics.strategies:
            assert strategy.folds
            assert strategy.valid_fold_count <= strategy.total_fold_count
            for aggregate in strategy.aggregate:
                assert len(aggregate.fold_values) == strategy.valid_fold_count or (
                    aggregate.valid_fold_count <= aggregate.total_fold_count
                )


class TestCalibrationPlacement:
    def test_calibration_happens_once_before_fusion(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        document = json.loads(
            (m6_classification_run.directory / "calibration.json").read_text()
        )
        assert "PER EXPERT, BEFORE fusion" in document["placement"]
        assert "No post-fusion calibrator is fitted" in document["placement"]
        assert document["use_calibrated_experts"] is True

    def test_expert_calibration_groups_are_disjoint_from_the_test_fold(
        self, m6_classification_run: FusionRunResult
    ) -> None:
        by_fold = {f.fold_index: f for f in m6_classification_run.splits.folds}
        for record in m6_classification_run.experts.experts:
            if not record.trained:
                continue
            fold = by_fold[record.fold_index]
            assert record.calibration_group_count <= len(fold.calibration_groups)
            assert not set(fold.calibration_groups) & set(fold.test_groups)
