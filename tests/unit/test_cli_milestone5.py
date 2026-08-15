"""Milestone 5 CLI and runner tests.

No test here needs a webcam, a model asset, a display server, a network,
Unity, a public dataset, or participant data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engagevr.__main__ import main
from engagevr.features.assembly import read_dataset_metadata
from engagevr.schemas.experiments import (
    SOFTWARE_SELF_CHECK_BANNER,
    EvaluationMode,
)
from engagevr.schemas.targets import TargetName, TaskType
from engagevr.training.calibration import CalibrationMethod
from engagevr.training.preprocessing import ModellingFrame
from engagevr.training.runner import (
    RULE_FEATURE_BY_TARGET,
    RunConfiguration,
    ScientificModeError,
    assert_scientific_eligibility,
    run_baselines,
)


def small_dataset(directory: Path, *, seed: int = 42) -> Path:
    path = directory / "m5.parquet"
    code = main(
        [
            "features-demo",
            "--seed",
            str(seed),
            "--subjects",
            "9",
            "--sessions-per-subject",
            "2",
            "--windows-per-session",
            "4",
            "--output",
            str(path),
        ]
    )
    assert code == 0
    return path


class TestFeaturesDemo:
    def test_it_succeeds_and_writes_every_document(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = small_dataset(tmp_path)
        capsys.readouterr()
        assert path.exists()
        assert (tmp_path / "m5.metadata.json").exists()
        assert (tmp_path / "m5.feature_catalog.json").exists()

    def test_it_creates_missing_output_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deeper" / "m5.parquet"
        assert (
            main(
                [
                    "features-demo",
                    "--subjects",
                    "4",
                    "--windows-per-session",
                    "3",
                    "--output",
                    str(target),
                ]
            )
            == 0
        )
        assert target.exists()

    def test_it_prints_the_permanent_synthetic_disclaimer(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        small_dataset(tmp_path)
        output = capsys.readouterr().out
        assert "=== SYNTHETIC DATA ===" in output
        assert "SYNTHETIC DATA. Permanently labelled." in output
        assert "Excluded from scientific" in output
        assert "not a performance claim" in output

    def test_it_prints_the_fingerprint_and_provenance(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        small_dataset(tmp_path)
        output = capsys.readouterr().out
        assert "Dataset fingerprint:" in output
        assert "Data-source counts:     {'synthetic': 72}" in output
        assert "Scientific evaluation:  False" in output
        assert "Random seed:            42" in output

    def test_repeating_the_command_reproduces_the_fingerprint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        first = small_dataset(tmp_path / "a")
        second = small_dataset(tmp_path / "b")
        capsys.readouterr()
        assert (
            read_dataset_metadata(first).dataset_fingerprint
            == read_dataset_metadata(second).dataset_fingerprint
        )

    def test_a_different_seed_changes_the_fingerprint(self, tmp_path: Path) -> None:
        first = small_dataset(tmp_path / "a", seed=1)
        second = small_dataset(tmp_path / "b", seed=2)
        assert (
            read_dataset_metadata(first).dataset_fingerprint
            != read_dataset_metadata(second).dataset_fingerprint
        )

    def test_the_imbalanced_flag_shifts_the_class_distribution(
        self, tmp_path: Path
    ) -> None:
        balanced = read_dataset_metadata(small_dataset(tmp_path / "a"))
        code = main(
            [
                "features-demo",
                "--seed",
                "42",
                "--subjects",
                "9",
                "--sessions-per-subject",
                "2",
                "--windows-per-session",
                "4",
                "--imbalanced",
                "--output",
                str(tmp_path / "b" / "m5.parquet"),
            ]
        )
        assert code == 0
        imbalanced = read_dataset_metadata(tmp_path / "b" / "m5.parquet")
        first = next(s for s in balanced.targets if s.target_name == "engagement_class")
        second = next(
            s for s in imbalanced.targets if s.target_name == "engagement_class"
        )
        assert first.class_distribution != second.class_distribution

    def test_an_invalid_generator_configuration_returns_two(
        self, tmp_path: Path
    ) -> None:
        code = main(
            [
                "features-demo",
                "--subjects",
                "4",
                "--window-seconds",
                "5",
                "--step-seconds",
                "10",
                "--output",
                str(tmp_path / "bad.parquet"),
            ]
        )
        assert code == 2


class TestBaselineDemo:
    @pytest.fixture(scope="class")
    def dataset(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        return small_dataset(tmp_path_factory.mktemp("cli-dataset"))

    def _run(self, dataset: Path, output: Path, *extra: str) -> int:
        return main(
            [
                "baseline-demo",
                "--dataset",
                str(dataset),
                "--target",
                "engagement_class",
                "--folds",
                "3",
                "--seed",
                "42",
                "--models",
                "dummy",
                "--no-ablations",
                "--output",
                str(output),
                *extra,
            ]
        )

    def test_it_succeeds(
        self, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert self._run(dataset, tmp_path / "run") == 0
        capsys.readouterr()

    def test_it_prints_the_software_self_check_banner(
        self, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run(dataset, tmp_path / "run")
        output = capsys.readouterr().out
        assert SOFTWARE_SELF_CHECK_BANNER in output
        assert output.count(SOFTWARE_SELF_CHECK_BANNER) >= 2

    def test_it_prints_every_required_field(
        self, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run(dataset, tmp_path / "run")
        output = capsys.readouterr().out
        for label in (
            "Dataset fingerprint:",
            "Data-source counts:",
            "Target:",
            "Task type:",
            "Group field:",
            "Independent groups:",
            "Folds:",
            "Models evaluated:",
            "Metrics:",
            "Experiment directory:",
            "Scientific evaluation:  False",
            "Split audit passed:     True",
        ):
            assert label in output, label

    def test_it_prints_the_permanent_disclaimers(
        self, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run(dataset, tmp_path / "run")
        output = capsys.readouterr().out
        assert "NOT model accuracy" in output
        assert "no number printed by these commands is experimental evidence" in output
        assert "No model here is a champion and none is production-ready" in output

    def test_it_makes_no_scientific_validation_claim(
        self, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run(dataset, tmp_path / "run")
        lowered = capsys.readouterr().out.lower()
        for phrase in (
            "experimentally validated",
            "clinically validated",
            "proven",
            "measures engagement",
            "state of the art",
            "outperforms",
        ):
            assert phrase not in lowered, phrase

    def test_it_creates_the_output_directory(
        self, dataset: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "nested" / "run"
        assert self._run(dataset, output) == 0
        assert (output / "manifest.json").exists()

    def test_a_repeated_run_is_deterministic(
        self, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run(dataset, tmp_path / "a")
        self._run(dataset, tmp_path / "b")
        capsys.readouterr()

        def metrics(directory: Path) -> dict[str, object]:
            with (directory / "metrics.json").open() as handle:
                return json.load(handle)

        first = metrics(tmp_path / "a")
        second = metrics(tmp_path / "b")
        assert first == second

        def splits(directory: Path) -> dict[str, object]:
            with (directory / "splits.json").open() as handle:
                return json.load(handle)

        assert splits(tmp_path / "a") == splits(tmp_path / "b")

    @pytest.mark.parametrize(
        "target",
        [
            "engagement_class",
            "cognitive_load_class",
            "engagement_score",
            "cognitive_load_score",
        ],
    )
    def test_every_supported_target_runs(
        self,
        dataset: Path,
        tmp_path: Path,
        target: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code = main(
            [
                "baseline-demo",
                "--dataset",
                str(dataset),
                "--target",
                target,
                "--folds",
                "3",
                "--seed",
                "42",
                "--models",
                "dummy",
                "--no-ablations",
                "--output",
                str(tmp_path / target),
            ]
        )
        capsys.readouterr()
        assert code == 0
        with (tmp_path / target / "metrics.json").open() as handle:
            document = json.load(handle)
        assert document["target_name"] == target


class TestCliRefusals:
    @pytest.fixture(scope="class")
    def dataset(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        return small_dataset(tmp_path_factory.mktemp("refusal-dataset"))

    def test_a_missing_dataset_returns_two(self, tmp_path: Path) -> None:
        code = main(
            [
                "baseline-demo",
                "--dataset",
                str(tmp_path / "absent.parquet"),
                "--output",
                str(tmp_path / "run"),
            ]
        )
        assert code == 2

    def test_an_invalid_fold_count_returns_two(
        self, dataset: Path, tmp_path: Path
    ) -> None:
        code = main(
            [
                "baseline-demo",
                "--dataset",
                str(dataset),
                "--folds",
                "1",
                "--output",
                str(tmp_path / "run"),
            ]
        )
        assert code == 2

    def test_too_many_folds_for_the_group_count_returns_one(
        self, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "baseline-demo",
                "--dataset",
                str(dataset),
                "--folds",
                "50",
                "--models",
                "dummy",
                "--no-ablations",
                "--output",
                str(tmp_path / "run"),
            ]
        )
        assert code == 1
        assert "cannot support 50 folds" in capsys.readouterr().err

    def test_an_unsupported_target_is_rejected_by_the_parser(
        self, dataset: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit):
            main(
                [
                    "baseline-demo",
                    "--dataset",
                    str(dataset),
                    "--target",
                    "mood_class",
                    "--output",
                    str(tmp_path / "run"),
                ]
            )

    def test_an_unsupported_model_returns_one(
        self, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "baseline-demo",
                "--dataset",
                str(dataset),
                "--models",
                "deep_fusion_net",
                "--output",
                str(tmp_path / "run"),
            ]
        )
        assert code == 1
        assert "not implemented" in capsys.readouterr().err

    def test_scientific_mode_rejects_synthetic_data(
        self, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "baseline-train",
                "--dataset",
                str(dataset),
                "--mode",
                "scientific",
                "--models",
                "dummy",
                "--no-ablations",
                "--output",
                str(tmp_path / "run"),
            ]
        )
        assert code == 3
        error = capsys.readouterr().err
        assert "scientific evaluation refused" in error
        assert "data_source='synthetic'" in error
        assert "never evidence about a person" in error

    def test_self_check_mode_accepts_the_same_dataset(
        self, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "baseline-train",
                "--dataset",
                str(dataset),
                "--mode",
                "self-check",
                "--folds",
                "3",
                "--models",
                "dummy",
                "--no-ablations",
                "--output",
                str(tmp_path / "run"),
            ]
        )
        capsys.readouterr()
        assert code == 0


class TestScientificEligibility:
    """Unit tests for the scientific-mode gate.

    Built from in-memory frames rather than from a fabricated "live"
    dataset on disk: this project has no participant data, and writing a
    file that claims otherwise — even in a temporary directory — is not
    worth the risk of it being mistaken for one.
    """

    def _frame(
        self,
        *,
        data_sources: tuple[str, ...],
        permitted: tuple[bool, ...],
        source_types: tuple[str, ...],
    ) -> ModellingFrame:
        n = len(data_sources)
        return ModellingFrame(
            predictors=pd.DataFrame({"feat__a": np.zeros(n)}),
            target_values=np.asarray(["low"] * n, dtype=object),
            target_name=TargetName.ENGAGEMENT_CLASS,
            task_type=TaskType.CLASSIFICATION,
            subject_ids=tuple(f"s{i}" for i in range(n)),
            session_ids=tuple(f"x{i}" for i in range(n)),
            window_ids=tuple(f"w{i}" for i in range(n)),
            data_sources=data_sources,
            target_source_types=source_types,
            target_scientific_permitted=permitted,
            # The gate never reads dataset metadata; it inspects the rows.
            metadata=None,  # type: ignore[arg-type]
        )

    def test_synthetic_rows_are_refused(self) -> None:
        frame = self._frame(
            data_sources=("synthetic", "live"),
            permitted=(True, True),
            source_types=("subjective_self_report",) * 2,
        )
        with pytest.raises(ScientificModeError, match="1 of 2 rows"):
            assert_scientific_eligibility(frame)

    def test_a_prohibited_target_is_refused(self) -> None:
        frame = self._frame(
            data_sources=("live", "live"),
            permitted=(True, False),
            source_types=("subjective_self_report",) * 2,
        )
        with pytest.raises(
            ScientificModeError, match="scientific_evaluation_permitted"
        ):
            assert_scientific_eligibility(frame)

    def test_missing_target_provenance_is_refused(self) -> None:
        frame = self._frame(
            data_sources=("live", "live"),
            permitted=(True, True),
            source_types=("subjective_self_report", "unstated"),
        )
        with pytest.raises(ScientificModeError, match="target provenance is missing"):
            assert_scientific_eligibility(frame)

    def test_a_documented_non_synthetic_frame_passes_the_gate(self) -> None:
        frame = self._frame(
            data_sources=("live", "live"),
            permitted=(True, True),
            source_types=("subjective_self_report", "expert_annotation"),
        )
        assert_scientific_eligibility(frame)

    def test_the_error_names_the_permitted_source_categories(self) -> None:
        frame = self._frame(
            data_sources=("live",),
            permitted=(True,),
            source_types=("unstated",),
        )
        with pytest.raises(ScientificModeError) as excinfo:
            assert_scientific_eligibility(frame)
        message = str(excinfo.value)
        for category in (
            "subjective_self_report",
            "experiment_condition",
            "expert_annotation",
            "public_dataset_annotation",
        ):
            assert category in message


class TestRunnerBehaviour:
    @pytest.fixture(scope="class")
    def dataset(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        return small_dataset(tmp_path_factory.mktemp("runner-dataset"))

    def test_the_rule_feature_is_documented_per_target(self) -> None:
        assert set(RULE_FEATURE_BY_TARGET) == set(TargetName)
        for column in RULE_FEATURE_BY_TARGET.values():
            assert column.startswith("feat__")

    def test_every_registered_model_runs_on_a_classification_target(
        self, dataset: Path, tmp_path: Path
    ) -> None:
        result = run_baselines(
            RunConfiguration(
                dataset_path=dataset,
                target_name=TargetName.ENGAGEMENT_CLASS,
                output_directory=tmp_path / "all-models",
                n_splits=3,
                random_seed=42,
                calibration_method=CalibrationMethod.SIGMOID,
                permutation_repeats=1,
                run_ablations=False,
            )
        )
        names = {model.model_name for model in result.metrics.results}
        assert names == {
            "dummy",
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "rule_software_check",
        }

    def test_every_registered_model_runs_on_a_regression_target(
        self, dataset: Path, tmp_path: Path
    ) -> None:
        result = run_baselines(
            RunConfiguration(
                dataset_path=dataset,
                target_name=TargetName.ENGAGEMENT_SCORE,
                output_directory=tmp_path / "all-regressors",
                n_splits=3,
                random_seed=42,
                permutation_repeats=1,
                run_ablations=False,
            )
        )
        names = {model.model_name for model in result.metrics.results}
        assert names == {
            "dummy",
            "ridge",
            "random_forest",
            "hist_gradient_boosting",
            "rule_software_check",
        }
        for model in result.metrics.results:
            assert model.fold_regression_metrics
            assert not model.fold_classification_metrics

    def test_bounded_tuning_runs_without_touching_the_test_fold(
        self, dataset: Path, tmp_path: Path
    ) -> None:
        result = run_baselines(
            RunConfiguration(
                dataset_path=dataset,
                target_name=TargetName.ENGAGEMENT_CLASS,
                output_directory=tmp_path / "tuned",
                n_splits=3,
                random_seed=42,
                model_names=("logistic_regression",),
                calibration_method=CalibrationMethod.NONE,
                calibration_group_fraction=0.0,
                permutation_repeats=1,
                run_ablations=False,
                tune_hyperparameters=True,
            )
        )
        assert result.splits.audit_passed is True
        assert result.metrics.results[0].aggregate

    def test_a_self_check_run_is_never_scientifically_eligible(
        self, dataset: Path, tmp_path: Path
    ) -> None:
        result = run_baselines(
            RunConfiguration(
                dataset_path=dataset,
                target_name=TargetName.ENGAGEMENT_CLASS,
                output_directory=tmp_path / "eligibility",
                n_splits=3,
                random_seed=42,
                model_names=("dummy",),
                permutation_repeats=1,
                run_ablations=False,
            )
        )
        assert result.metrics.evaluation_mode is EvaluationMode.SOFTWARE_SELF_CHECK
        assert result.metrics.scientific_evaluation_eligible is False
        assert result.manifest.scientific_evaluation_eligible is False
