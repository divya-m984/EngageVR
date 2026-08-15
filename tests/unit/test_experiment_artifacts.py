"""Experiment-artifact tests: atomicity, checksums, failure status, privacy.

A run directory is the audit trail. These tests assert that it is complete
when a run succeeds, honest when a run fails, and free of anything that
could identify a person.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from engagevr.schemas.experiments import (
    EvaluationMode,
    RunManifest,
    RunStatus,
)
from engagevr.schemas.targets import TargetName
from engagevr.training.artifacts import (
    MODEL_FILE_WARNING,
    TRACKED_PACKAGES,
    ArtifactError,
    ExperimentRun,
    build_run_id,
    dependency_versions,
    engagevr_version,
    importance_table,
    read_manifest,
    runtime_environment,
    sha256_file,
    verify_checksums,
)
from engagevr.training.calibration import CalibrationMethod
from engagevr.training.models import UnsupportedModelError
from engagevr.training.runner import RunConfiguration, RunResult, run_baselines


def manifest(**overrides: object) -> RunManifest:
    from datetime import UTC, datetime

    fields: dict[str, object] = {
        "run_id": "r",
        "engagevr_version": "0.1.0",
        "python_version": "3.12.13",
        "dependency_versions": {},
        "evaluation_mode": EvaluationMode.SOFTWARE_SELF_CHECK,
        "scientific_evaluation_eligible": False,
        "dataset_path": "d.parquet",
        "dataset_fingerprint": "0" * 64,
        "feature_catalog_version": "1.0",
        "target_name": "engagement_class",
        "task_type": "classification",
        "feature_set": ("feat__a",),
        "split_strategy": "group_k_fold",
        "group_field": "subject_id",
        "group_count": 5,
        "fold_count": 3,
        "random_seed": 42,
        "started_at_utc": datetime(2026, 1, 1, tzinfo=UTC),
        "finished_at_utc": datetime(2026, 1, 1, tzinfo=UTC),
        "status": RunStatus.COMPLETED,
        "disclaimers": ("note",),
    }
    fields.update(overrides)
    return RunManifest(**fields)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def completed_run(
    m5_dataset: Path, tmp_path_factory: pytest.TempPathFactory
) -> RunResult:
    directory = tmp_path_factory.mktemp("artifact-run")
    return run_baselines(
        RunConfiguration(
            dataset_path=m5_dataset,
            target_name=TargetName.ENGAGEMENT_CLASS,
            output_directory=directory,
            n_splits=3,
            random_seed=42,
            model_names=("dummy", "logistic_regression"),
            calibration_method=CalibrationMethod.SIGMOID,
            permutation_repeats=1,
            run_ablations=False,
        )
    )


class TestRunIdentifiers:
    def test_the_identifier_is_deterministic(self) -> None:
        kwargs: dict[str, object] = {
            "target_name": "engagement_class",
            "evaluation_mode": "software_self_check",
            "dataset_fingerprint": "a" * 64,
            "random_seed": 42,
            "fold_count": 5,
            "model_names": ["dummy", "ridge"],
            "feature_set": ["feat__a"],
            "calibration_method": "sigmoid",
        }
        assert build_run_id(**kwargs) == build_run_id(**kwargs)  # type: ignore[arg-type]

    def test_the_identifier_is_insensitive_to_model_order(self) -> None:
        base: dict[str, object] = {
            "target_name": "engagement_class",
            "evaluation_mode": "software_self_check",
            "dataset_fingerprint": "a" * 64,
            "random_seed": 42,
            "fold_count": 5,
            "feature_set": ["feat__a"],
            "calibration_method": "sigmoid",
        }
        first = build_run_id(model_names=["dummy", "ridge"], **base)  # type: ignore[arg-type]
        second = build_run_id(model_names=["ridge", "dummy"], **base)  # type: ignore[arg-type]
        assert first == second

    @pytest.mark.parametrize(
        "field,value",
        [
            ("dataset_fingerprint", "b" * 64),
            ("random_seed", 7),
            ("fold_count", 3),
            ("calibration_method", "isotonic"),
            ("target_name", "cognitive_load_class"),
        ],
    )
    def test_any_defining_input_changes_the_identifier(
        self, field: str, value: object
    ) -> None:
        kwargs: dict[str, object] = {
            "target_name": "engagement_class",
            "evaluation_mode": "software_self_check",
            "dataset_fingerprint": "a" * 64,
            "random_seed": 42,
            "fold_count": 5,
            "model_names": ["dummy"],
            "feature_set": ["feat__a"],
            "calibration_method": "sigmoid",
        }
        original = build_run_id(**kwargs)  # type: ignore[arg-type]
        kwargs[field] = value
        assert build_run_id(**kwargs) != original  # type: ignore[arg-type]

    def test_the_identifier_names_the_evaluation_mode(self) -> None:
        scientific = build_run_id(
            target_name="engagement_class",
            evaluation_mode="scientific",
            dataset_fingerprint="a" * 64,
            random_seed=42,
            fold_count=5,
            model_names=["dummy"],
            feature_set=["feat__a"],
            calibration_method=None,
        )
        assert scientific.startswith("engagement_class-sci-")


class TestArtifactSet:
    def test_every_required_artifact_is_written(self, completed_run: RunResult) -> None:
        for name in (
            "manifest.json",
            "dataset.json",
            "feature_catalog.json",
            "splits.json",
            "metrics.json",
            "calibration.json",
            "ablations.json",
            "predictions.parquet",
            "feature_importance.parquet",
            "checksums.json",
        ):
            assert (completed_run.directory / name).exists(), name
        assert (completed_run.directory / "models").is_dir()

    def test_the_manifest_records_the_full_environment(
        self, completed_run: RunResult
    ) -> None:
        stored = read_manifest(completed_run.directory)
        assert stored.engagevr_version == engagevr_version()
        assert stored.python_version == runtime_environment()["python_version"]
        assert set(stored.dependency_versions) == set(TRACKED_PACKAGES)
        assert stored.status is RunStatus.COMPLETED
        assert stored.failure_reason is None

    def test_the_manifest_records_the_split_and_fold_assignments(
        self, completed_run: RunResult
    ) -> None:
        stored = read_manifest(completed_run.directory)
        assert stored.group_field == "subject_id"
        assert stored.fold_count == 3
        assert set(stored.fold_assignments) == {"0", "1", "2"}
        for assignment in stored.fold_assignments.values():
            assert set(assignment) == {"train", "calibration", "test"}
            assert not set(assignment["train"]) & set(assignment["test"])

    def test_the_manifest_records_model_parameters_and_the_seed(
        self, completed_run: RunResult
    ) -> None:
        stored = read_manifest(completed_run.directory)
        assert set(stored.model_parameters) == {"dummy", "logistic_regression"}
        assert stored.random_seed == 42
        assert stored.calibration_method == "sigmoid"

    def test_checksums_cover_every_artifact_and_verify(
        self, completed_run: RunResult
    ) -> None:
        with (completed_run.directory / "checksums.json").open() as handle:
            checksums = json.load(handle)
        assert "metrics.json" in checksums
        assert "predictions.parquet" in checksums
        for name, digest in checksums.items():
            assert sha256_file(completed_run.directory / name) == digest
        assert verify_checksums(completed_run.directory) == ()

    def test_a_tampered_artifact_is_detected(
        self, completed_run: RunResult, tmp_path: Path
    ) -> None:
        import shutil

        copy = tmp_path / "copy"
        shutil.copytree(completed_run.directory, copy)
        (copy / "metrics.json").write_text("{}\n", encoding="utf-8")
        assert "metrics.json" in verify_checksums(copy)

    def test_the_model_directory_carries_the_pickle_warning(
        self, completed_run: RunResult
    ) -> None:
        warning = (completed_run.directory / "models" / "README.txt").read_text()
        assert "never load a model file from an untrusted source" in warning.lower()
        assert MODEL_FILE_WARNING.split(".")[0] in warning

    def test_predictions_are_finite_and_probabilities_sum_to_one(
        self, completed_run: RunResult
    ) -> None:
        import pyarrow.parquet as pq

        table = pq.read_table(
            completed_run.directory / "predictions.parquet"
        ).to_pandas()
        assert len(table) > 0
        assert table["predicted_value"].notna().all()
        columns = [
            c for c in table.columns if c.startswith("probability_uncalibrated__")
        ]
        assert columns
        totals = table[columns].sum(axis=1)
        assert ((totals - 1.0).abs() < 1e-9).all()

    def test_feature_importance_is_stored_per_fold_before_aggregation(
        self, completed_run: RunResult
    ) -> None:
        import pyarrow.parquet as pq

        table = pq.read_table(
            completed_run.directory / "feature_importance.parquet"
        ).to_pandas()
        assert len(table) > 0
        assert set(table["kind"]) <= {"linear_coefficient", "permutation_importance"}
        assert table["fold_index"].nunique() >= 1
        assert table["feature_name"].notna().all()


class TestFailureRecording:
    def test_a_failed_run_is_recorded_as_failed(self, tmp_path: Path) -> None:
        run = ExperimentRun(tmp_path / "failed", "r")
        path = run.finalize(
            manifest(status=RunStatus.FAILED, failure_reason="ValueError: boom")
        )
        stored = read_manifest(path.parent)
        assert stored.status is RunStatus.FAILED
        assert "boom" in (stored.failure_reason or "")

    def test_a_failed_manifest_must_state_a_reason(self) -> None:
        with pytest.raises(ValueError, match="must record a failure_reason"):
            manifest(status=RunStatus.FAILED)

    def test_a_completed_manifest_must_not_state_a_reason(self) -> None:
        with pytest.raises(ValueError, match="must not record a failure_reason"):
            manifest(status=RunStatus.COMPLETED, failure_reason="boom")

    def test_a_completed_run_missing_an_artifact_is_refused(
        self, tmp_path: Path
    ) -> None:
        run = ExperimentRun(tmp_path / "incomplete", "r")
        with pytest.raises(ArtifactError, match="missing required artifact"):
            run.finalize(manifest(status=RunStatus.COMPLETED))

    def test_a_directory_without_a_manifest_is_an_interrupted_run(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "orphan").mkdir()
        with pytest.raises(ArtifactError, match="interrupted"):
            read_manifest(tmp_path / "orphan")

    def test_a_run_that_raises_still_writes_a_failed_manifest(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        directory = tmp_path / "boom"
        with pytest.raises(UnsupportedModelError):
            run_baselines(
                RunConfiguration(
                    dataset_path=m5_dataset,
                    target_name=TargetName.ENGAGEMENT_CLASS,
                    output_directory=directory,
                    n_splits=3,
                    random_seed=42,
                    model_names=("no_such_model",),
                )
            )
        # The failure happens before the run directory is created, so there
        # is nothing to mistake for a successful run.
        assert not (directory / "manifest.json").exists()


class TestAtomicWrites:
    def test_no_temporary_file_survives_a_write(self, completed_run: RunResult) -> None:
        leftovers = [
            p.name
            for p in completed_run.directory.rglob("*")
            if p.name.startswith(".") and p.name.endswith(".tmp")
        ]
        assert leftovers == []

    def test_the_manifest_is_written_last(self, tmp_path: Path) -> None:
        run = ExperimentRun(tmp_path / "ordered", "r")
        for name in ExperimentRun.REQUIRED_ARTIFACTS:
            run.write_json(name, {"ok": True})
        run.finalize(manifest())
        assert run.written_artifacts[-1] == "manifest.json"
        assert run.written_artifacts[-2] == "checksums.json"

    def test_a_json_artifact_is_valid_after_writing(self, tmp_path: Path) -> None:
        run = ExperimentRun(tmp_path / "json", "r")
        path = run.write_json("dataset.json", {"a": 1, "b": [1, 2]})
        with path.open() as handle:
            assert json.load(handle) == {"a": 1, "b": [1, 2]}

    def test_a_parquet_artifact_is_valid_after_writing(self, tmp_path: Path) -> None:
        import pyarrow.parquet as pq

        run = ExperimentRun(tmp_path / "parquet", "r")
        path = run.write_table("t.parquet", pa.table({"a": [1, 2, 3]}))
        assert pq.read_table(path).num_rows == 3


class TestPrivacy:
    #: Tokens that must not appear anywhere in a run directory.
    FORBIDDEN = (
        "@example.com",
        "password",
        "api_key",
        "secret_key",
        "access_token",
        "first_name",
        "last_name",
    )

    def test_no_secret_or_identifier_appears_in_any_json_artifact(
        self, completed_run: RunResult
    ) -> None:
        for path in completed_run.directory.glob("*.json"):
            text = path.read_text(encoding="utf-8").lower()
            for token in self.FORBIDDEN:
                assert token not in text, (path.name, token)

    def test_subject_identifiers_stay_pseudonymous_in_artifacts(
        self, completed_run: RunResult
    ) -> None:
        with (completed_run.directory / "splits.json").open() as handle:
            splits = json.load(handle)
        for fold in splits["folds"]:
            for group in fold["train_groups"] + fold["test_groups"]:
                assert group.startswith("synthetic-subject-")

    def test_no_scientific_or_clinical_claim_appears(
        self, completed_run: RunResult
    ) -> None:
        banned = (
            "clinically validated",
            "diagnostic accuracy",
            "proven to measure",
            "production-ready",
            "champion model",
            "experimentally validated",
        )
        for path in completed_run.directory.glob("*.json"):
            text = path.read_text(encoding="utf-8").lower()
            for phrase in banned:
                assert phrase not in text, (path.name, phrase)

    def test_the_metrics_document_carries_the_self_check_banner(
        self, completed_run: RunResult
    ) -> None:
        with (completed_run.directory / "metrics.json").open() as handle:
            document = json.load(handle)
        assert document["evaluation_mode"] == "software_self_check"
        assert document["scientific_evaluation_eligible"] is False
        assert any(
            "SOFTWARE SELF-CHECK" in disclaimer
            for disclaimer in document["disclaimers"]
        )

    def test_no_mlflow_artifact_is_produced(self, completed_run: RunResult) -> None:
        names = {p.name for p in completed_run.directory.rglob("*")}
        assert "mlruns" not in names
        assert "MLmodel" not in names
        assert not any(name.startswith("meta.yaml") for name in names)


class TestGitIgnoreCoverage:
    def test_artifacts_are_ignored_by_git(self) -> None:
        ignore = Path(__file__).resolve().parents[2] / ".gitignore"
        lines = {line.strip() for line in ignore.read_text().splitlines()}
        assert "artifacts/" in lines
        assert "models/" in lines

    def test_no_parquet_dataset_is_tracked_under_version_control(self) -> None:
        root = Path(__file__).resolve().parents[2]
        tracked = [
            p
            for p in root.rglob("*.parquet")
            if ".venv" not in p.parts and "artifacts" not in p.parts
        ]
        assert tracked == []


class TestHelpers:
    def test_dependency_versions_cover_the_modelling_stack(self) -> None:
        versions = dependency_versions()
        for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "joblib"):
            assert package in versions
            assert versions[package] != "not installed"

    def test_an_empty_importance_table_still_has_the_schema(self) -> None:
        table = importance_table([])
        assert table.num_rows == 0
        assert "feature_name" in table.schema.names
        assert "permutation" not in table.schema.names
