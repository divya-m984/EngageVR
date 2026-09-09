"""Local MLflow tracking.

Every test here uses a **temporary** store under ``tmp_path``, which
pytest removes on its own.  No test contacts a server, and no tracking
state survives the session.

The properties under test: a run is created with the parameters, metrics,
tags, and artifacts the source run already recorded; synthetic provenance
survives the trip intact; nothing is registered, promoted, or renamed
into an endorsement; and the source run directory is left byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engagevr.config import EngageVRConfig, load_config
from engagevr.mlops.mlflow_tracking import (
    FILE_STORE_OPT_OUT_ENV,
    LOGGED_JSON_ARTIFACTS,
    NEVER_LOGGED,
    TELEMETRY_OPT_OUT_ENV,
    TrackingError,
    assert_usable_store,
    log_run_directory,
    resolve_tracking_uri,
)
from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER
from engagevr.schemas.mlops import FORBIDDEN_STATUS_WORDS, REQUIRED_TRACKING_TAGS
from engagevr.training.artifacts import sha256_file


@pytest.fixture
def config() -> EngageVRConfig:
    return load_config()


@pytest.fixture
def store(tmp_path: Path) -> str:
    """A throwaway local file store. Removed with tmp_path."""
    directory = tmp_path / "mlflow-store"
    directory.mkdir()
    return directory.resolve().as_uri()


@pytest.fixture
def logged(m10_baseline_run: Path, config: EngageVRConfig, store: str):  # type: ignore[no-untyped-def]
    return log_run_directory(
        m10_baseline_run,
        config=config,
        tracking_uri=store,
        experiment_name="engagevr-unit-test",
    )


def directory_digest(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class TestStoreResolution:
    def test_a_relative_directory_becomes_a_local_file_uri(
        self, config: EngageVRConfig, tmp_path: Path
    ) -> None:
        uri = resolve_tracking_uri(config, base=tmp_path)
        assert uri.startswith("file://")
        assert (tmp_path / config.mlops.mlflow.tracking_uri).is_dir()

    def test_no_remote_scheme_is_configurable(self, config: EngageVRConfig) -> None:
        settings = type(config.mlops.mlflow)
        for scheme in ("http://tracker", "https://tracker", "databricks://x"):
            with pytest.raises(ValueError, match="local-first"):
                settings(tracking_uri=scheme)

    def test_a_database_backend_is_refused_with_its_reason(
        self, config: EngageVRConfig
    ) -> None:
        settings = type(config.mlops.mlflow)
        with pytest.raises(ValueError, match="pandas<3"):
            settings(tracking_uri="sqlite:///mlflow.db")

    def test_a_store_under_artifacts_is_refused_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        # MLflow's file store rejects any run path containing a component
        # named "artifacts", then reports the runs it wrote as missing.
        with pytest.raises(TrackingError, match="path-traversal"):
            assert_usable_store(tmp_path / "artifacts" / "mlflow")

    def test_the_configuration_refuses_the_same_path(
        self, config: EngageVRConfig
    ) -> None:
        settings = type(config.mlops.mlflow)
        with pytest.raises(ValueError, match="path component named 'artifacts'"):
            settings(tracking_uri="artifacts/mlflow")

    def test_tracking_is_off_by_default(self, config: EngageVRConfig) -> None:
        assert config.mlops.mlflow.enabled is False


class TestRunCreation:
    def test_a_run_is_created_in_the_named_experiment(self, logged) -> None:  # type: ignore[no-untyped-def]
        assert logged.experiment_name == "engagevr-unit-test"
        assert logged.mlflow_run_id
        assert logged.mlflow_version

    def test_the_run_is_readable_back_from_the_store(self, logged, store) -> None:  # type: ignore[no-untyped-def]
        import os

        from mlflow.tracking import MlflowClient

        os.environ[FILE_STORE_OPT_OUT_ENV] = "true"
        try:
            client = MlflowClient(tracking_uri=store)
            run = client.get_run(logged.mlflow_run_id)
        finally:
            os.environ.pop(FILE_STORE_OPT_OUT_ENV, None)
        assert run.info.status == "FINISHED"
        assert run.data.tags["engagevr.is_synthetic"] == "true"
        assert run.data.params["target"] == "engagement_class"
        assert set(logged.metrics).issubset(run.data.metrics)

    def test_the_store_is_temporary_and_local(self, logged, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        assert logged.tracking_uri.startswith("file://")
        assert str(tmp_path) in logged.tracking_uri


class TestParameters:
    def test_the_design_of_the_run_is_recorded(self, logged) -> None:  # type: ignore[no-untyped-def]
        for key in (
            "target",
            "task_type",
            "estimator",
            "feature_subset",
            "seed",
            "split_strategy",
            "fold_count",
            "calibration_method",
            "dataset_fingerprint",
            "split_fingerprint",
            "engagevr.config_fingerprint",
        ):
            assert key in logged.parameters, key

    def test_the_scientific_status_is_a_parameter_too(self, logged) -> None:  # type: ignore[no-untyped-def]
        assert logged.parameters["evaluation_mode"] == "software_self_check"
        assert logged.parameters["scientific_evaluation_eligible"] == "false"

    def test_the_fingerprints_are_not_truncated(self, logged) -> None:  # type: ignore[no-untyped-def]
        assert len(logged.parameters["dataset_fingerprint"]) == 64
        assert len(logged.parameters["split_fingerprint"]) == 64
        assert len(logged.parameters["engagevr.config_fingerprint"]) == 64


class TestMetrics:
    def test_only_already_computed_values_are_logged(
        self, logged, m10_baseline_run: Path
    ) -> None:  # type: ignore[no-untyped-def]
        document = json.loads(
            (m10_baseline_run / "metrics.json").read_text(encoding="utf-8")
        )
        recorded: dict[str, float] = {}
        for result in document["results"]:
            for aggregate in result["aggregate"]:
                if aggregate["mean"] is not None:
                    recorded[f"{result['model_name']}/{aggregate['name']}"] = float(
                        aggregate["mean"]
                    )
        assert recorded
        for key, value in recorded.items():
            assert logged.metrics[key] == pytest.approx(value)

    def test_an_unavailable_aggregate_is_skipped_with_a_reason_not_zeroed(
        self, logged, m10_baseline_run: Path
    ) -> None:  # type: ignore[no-untyped-def]
        document = json.loads(
            (m10_baseline_run / "metrics.json").read_text(encoding="utf-8")
        )
        unavailable = [
            f"{result['model_name']}/{aggregate['name']}"
            for result in document["results"]
            for aggregate in result["aggregate"]
            if aggregate["mean"] is None
        ]
        for key in unavailable:
            assert key not in logged.metrics
            assert logged.skipped_metrics[key]

    def test_no_metric_is_recomputed(self, logged) -> None:  # type: ignore[no-untyped-def]
        # Every key is namespaced by the model that produced it; nothing
        # is derived, averaged, or renamed on the way through.
        for key in logged.metrics:
            assert "/" in key

    def test_classification_and_regression_names_are_not_merged(self, logged) -> None:  # type: ignore[no-untyped-def]
        # A classification run must not carry a regression aggregate.
        assert not any("mean_absolute_error" in key for key in logged.metrics)


class TestProvenanceTags:
    def test_every_required_tag_is_present(self, logged) -> None:  # type: ignore[no-untyped-def]
        for tag in REQUIRED_TRACKING_TAGS:
            assert tag in logged.tags

    def test_the_synthetic_tags_say_synthetic_and_ineligible(self, logged) -> None:  # type: ignore[no-untyped-def]
        assert logged.tags["engagevr.data_source"] == "synthetic"
        assert logged.tags["engagevr.is_synthetic"] == "true"
        assert logged.tags["engagevr.scientific_evaluation_eligible"] == "false"
        assert logged.is_synthetic is True
        assert logged.scientific_evaluation_eligible is False

    def test_the_disclaimer_tag_carries_the_banner(self, logged) -> None:  # type: ignore[no-untyped-def]
        assert logged.tags["engagevr.disclaimer"] == SOFTWARE_SELF_CHECK_BANNER

    def test_the_limitation_tag_states_that_tracking_is_not_validation(
        self, logged
    ) -> None:  # type: ignore[no-untyped-def]
        limitation = logged.tags["engagevr.limitation"].lower()
        assert "tracking is not validation" in limitation

    def test_the_run_family_and_source_are_recorded(self, logged) -> None:  # type: ignore[no-untyped-def]
        assert logged.tags["engagevr.run_family"] == "baseline"
        assert logged.run_family == "baseline"
        assert logged.tags["engagevr.source_directory"]


class TestNoEndorsement:
    def test_nothing_is_registered(self, logged) -> None:  # type: ignore[no-untyped-def]
        assert logged.registered_model is None
        assert logged.tags["engagevr.registered_model"] == "none"

    def test_no_alias_or_stage_word_appears_anywhere(self, logged) -> None:  # type: ignore[no-untyped-def]
        haystacks = [logged.mlflow_run_name, logged.experiment_name]
        haystacks += [
            value
            for key, value in logged.tags.items()
            if not key.endswith(("disclaimer", "note", "limitation"))
        ]
        for text in haystacks:
            lowered = text.lower()
            for word in FORBIDDEN_STATUS_WORDS:
                assert word not in lowered.split(), (text, word)

    def test_the_summary_disclaims_endorsement(self, logged) -> None:  # type: ignore[no-untyped-def]
        assert logged.disclaimers
        assert any(
            "registration is not approval" in d.lower() for d in logged.disclaimers
        )


class TestArtifacts:
    def test_the_run_documents_are_logged(self, logged) -> None:  # type: ignore[no-untyped-def]
        names = {name.removeprefix("run/") for name in logged.logged_artifacts}
        for required in (
            "manifest.json",
            "metrics.json",
            "splits.json",
            "dataset.json",
        ):
            assert required in names

    def test_only_documents_on_the_allowlist_are_logged(self, logged) -> None:  # type: ignore[no-untyped-def]
        allowed = set(LOGGED_JSON_ARTIFACTS) | {"predictions.parquet"}
        for name in logged.logged_artifacts:
            assert name.removeprefix("run/") in allowed

    def test_no_model_binary_is_ever_logged(self, logged) -> None:  # type: ignore[no-untyped-def]
        for name in logged.logged_artifacts:
            assert not name.endswith((".joblib", ".pkl"))
            assert "models/" not in name

    def test_the_never_logged_list_covers_media_and_secrets(self) -> None:
        listed = " ".join(NEVER_LOGGED)
        for pattern in ("models/", "*.joblib", ".env", "secrets/", "*.mp4", "*.png"):
            assert pattern in listed


class TestImmutabilityAndIsolation:
    def test_the_source_run_is_left_byte_identical(
        self, m10_baseline_run: Path, config: EngageVRConfig, store: str
    ) -> None:
        before = directory_digest(m10_baseline_run)
        log_run_directory(m10_baseline_run, config=config, tracking_uri=store)
        assert directory_digest(m10_baseline_run) == before

    def test_no_mlflow_state_is_written_into_the_run_directory(
        self, m10_baseline_run: Path, config: EngageVRConfig, store: str
    ) -> None:
        log_run_directory(m10_baseline_run, config=config, tracking_uri=store)
        names = {p.name for p in m10_baseline_run.rglob("*")}
        assert "mlruns" not in names
        assert "MLmodel" not in names
        assert not any(name.startswith("meta.yaml") for name in names)

    def test_the_environment_is_restored_after_the_call(
        self, m10_baseline_run: Path, config: EngageVRConfig, store: str
    ) -> None:
        import os

        before = {
            name: os.environ.get(name)
            for name in (FILE_STORE_OPT_OUT_ENV, TELEMETRY_OPT_OUT_ENV)
        }
        log_run_directory(m10_baseline_run, config=config, tracking_uri=store)
        after = {
            name: os.environ.get(name)
            for name in (FILE_STORE_OPT_OUT_ENV, TELEMETRY_OPT_OUT_ENV)
        }
        assert after == before

    def test_importing_the_adapter_does_not_import_mlflow(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import engagevr.mlops.mlflow_tracking as m; "
                "print('mlflow' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"


class TestRefusals:
    def test_a_directory_with_no_conclusion_is_refused(self, tmp_path: Path) -> None:
        empty = tmp_path / "interrupted"
        empty.mkdir()
        with pytest.raises(TrackingError, match=r"no run in it reached a conclusion"):
            log_run_directory(
                empty, config=load_config(), tracking_uri=(tmp_path / "s").as_uri()
            )
