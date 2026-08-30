"""Milestone 10 schemas: strictness, provenance, and refusals.

The point of these tests is not that the models parse.  It is that they
*refuse*: a synthetic record cannot become scientifically eligible, an
endorsement word cannot enter a record, an unavailable statistic cannot
be reported as zero, a non-finite value cannot be reported as a number,
and a document from an unknown schema version is declined rather than
half-read.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from engagevr.schemas.experiments import (
    SOFTWARE_SELF_CHECK_BANNER,
    EvaluationMode,
)
from engagevr.schemas.mlops import (
    FORBIDDEN_STATUS_WORDS,
    MLOPS_DISCLAIMER,
    MLOPS_SCHEMA_VERSION,
    REQUIRED_TRACKING_TAGS,
    ConfigurationVersion,
    DeterministicArtifact,
    DriftDatasetReference,
    DriftMethod,
    DriftReport,
    DriftReportKind,
    DriftStatistic,
    DriftStatus,
    FeatureDriftResult,
    MLOpsRunSummary,
    ModelVersionManifest,
    ReproducibilityManifest,
    ReproducibilityStage,
    SmokeCheckResult,
    SmokeCheckStatus,
    SmokeReport,
    UnsupportedMLOpsSchemaError,
    assert_no_status_word,
    assert_supported_schema_version,
)

DIGEST = "a" * 64
OTHER = "b" * 64
MOMENT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def configuration_version(**overrides: object) -> ConfigurationVersion:
    payload: dict[str, object] = {
        "config_fingerprint": DIGEST,
        "engagevr_version": "0.1.0",
        "project_config_version": "0.1.0",
    }
    payload.update(overrides)
    return ConfigurationVersion.model_validate(payload)


def model_version(**overrides: object) -> ModelVersionManifest:
    payload: dict[str, object] = {
        "model_version_id": "mv-engagement_class-logistic_regression-0123456789ab",
        "target_name": "engagement_class",
        "task_type": "classification",
        "estimator_type": "linear",
        "model_name": "logistic_regression-fold0",
        "source_run_id": "engagement_class-selfcheck-0123456789ab",
        "source_run_family": "baseline",
        "source_run_directory": "artifacts/pipeline/experiments/baseline",
        "dataset_fingerprint": DIGEST,
        "split_fingerprint": OTHER,
        "feature_schema_fingerprint": "c" * 64,
        "feature_catalog_version": "1.0",
        "feature_count": 108,
        "configuration": configuration_version(),
        "serialization_library_version": "1.5.3",
        "model_artifact_path": "models/logistic_regression-fold0.joblib",
        "model_artifact_sha256": "d" * 64,
        "model_artifact_bytes": 4096,
        "engagevr_version": "0.1.0",
        "python_series": "3.12",
        "evaluation_mode": EvaluationMode.SOFTWARE_SELF_CHECK,
        "is_synthetic": True,
        "scientific_evaluation_eligible": False,
        "disclaimers": (SOFTWARE_SELF_CHECK_BANNER, MLOPS_DISCLAIMER),
    }
    payload.update(overrides)
    return ModelVersionManifest.model_validate(payload)


def tracking_tags(**overrides: str) -> dict[str, str]:
    tags = {
        "engagevr.data_source": "synthetic",
        "engagevr.is_synthetic": "true",
        "engagevr.scientific_evaluation_eligible": "false",
        "engagevr.evaluation_mode": "software_self_check",
        "engagevr.disclaimer": SOFTWARE_SELF_CHECK_BANNER,
        "engagevr.run_family": "baseline",
        "engagevr.run_id": "engagement_class-selfcheck-0123456789ab",
        "engagevr.version": "0.1.0",
    }
    tags.update(overrides)
    return tags


def tracking_summary(**overrides: object) -> MLOpsRunSummary:
    payload: dict[str, object] = {
        "tracking_uri": "file:///repo/mlruns",
        "experiment_name": "engagevr-software-self-check",
        "experiment_id": "1",
        "mlflow_run_id": "0" * 32,
        "mlflow_run_name": "baseline-engagement_class",
        "mlflow_version": "3.15.2",
        "source_run_directory": "artifacts/pipeline/experiments/baseline",
        "source_run_id": "engagement_class-selfcheck-0123456789ab",
        "run_family": "baseline",
        "tags": tracking_tags(),
        "is_synthetic": True,
        "scientific_evaluation_eligible": False,
        "created_at_utc": MOMENT,
        "disclaimers": (MLOPS_DISCLAIMER,),
    }
    payload.update(overrides)
    return MLOpsRunSummary.model_validate(payload)


def drift_report(**overrides: object) -> DriftReport:
    payload: dict[str, object] = {
        "report_kind": DriftReportKind.FEATURE_DISTRIBUTION_SHIFT,
        "reference": DriftDatasetReference(
            role="reference",
            path="artifacts/pipeline/datasets/reference.parquet",
            dataset_fingerprint=DIGEST,
            row_count=384,
            is_synthetic=True,
            scientific_evaluation_eligible=False,
        ),
        "current": DriftDatasetReference(
            role="current",
            path="artifacts/pipeline/datasets/current.parquet",
            dataset_fingerprint=OTHER,
            row_count=384,
            is_synthetic=True,
            scientific_evaluation_eligible=False,
        ),
        "minimum_samples": 30,
        "histogram_bin_count": 10,
        "report_fingerprint": DIGEST,
        "features_compared_count": 0,
        "features_exceeding_count": 0,
        "features_unavailable_count": 0,
        "is_synthetic": True,
        "scientific_evaluation_eligible": False,
        "disclaimers": (MLOPS_DISCLAIMER,),
    }
    payload.update(overrides)
    return DriftReport.model_validate(payload)


class TestSchemaVersioning:
    def test_the_current_version_is_supported(self) -> None:
        assert assert_supported_schema_version(MLOPS_SCHEMA_VERSION) == "1.0"

    def test_a_future_version_is_refused_rather_than_partially_read(self) -> None:
        with pytest.raises(UnsupportedMLOpsSchemaError, match="not supported"):
            assert_supported_schema_version("2.0")

    @pytest.mark.parametrize(
        "factory",
        [configuration_version, model_version, tracking_summary, drift_report],
    )
    def test_every_document_refuses_an_unsupported_version(
        self, factory: object
    ) -> None:
        with pytest.raises((ValidationError, UnsupportedMLOpsSchemaError)):
            factory(schema_version="99.0")  # type: ignore[operator]

    @pytest.mark.parametrize(
        "factory",
        [configuration_version, model_version, tracking_summary, drift_report],
    )
    def test_every_document_forbids_unknown_fields(self, factory: object) -> None:
        with pytest.raises(ValidationError):
            factory(nonsense_field=1)  # type: ignore[operator]


class TestStatusWords:
    @pytest.mark.parametrize("word", sorted(FORBIDDEN_STATUS_WORDS))
    def test_every_forbidden_word_is_rejected(self, word: str) -> None:
        with pytest.raises(ValueError, match="bookkeeping, not endorsement"):
            assert_no_status_word(f"the {word} model", field="test")

    def test_a_word_merely_containing_one_is_allowed(self) -> None:
        # "productionise" is a different word, and a naive substring check
        # would reject legitimate text.
        assert assert_no_status_word("preproduction-ish", field="test")

    def test_a_model_version_may_not_be_called_production(self) -> None:
        with pytest.raises(ValidationError, match="bookkeeping, not endorsement"):
            model_version(model_version_id="mv-production-engagement")

    def test_a_model_version_may_not_be_called_champion(self) -> None:
        with pytest.raises(ValidationError, match="bookkeeping, not endorsement"):
            model_version(model_name="champion")

    def test_an_experiment_may_not_be_called_production(self) -> None:
        with pytest.raises(ValidationError, match="bookkeeping, not endorsement"):
            tracking_summary(experiment_name="engagevr-production")

    def test_a_run_may_not_be_named_champion(self) -> None:
        with pytest.raises(ValidationError, match="bookkeeping, not endorsement"):
            tracking_summary(mlflow_run_name="champion-2026")

    def test_a_tag_value_may_not_carry_an_endorsement(self) -> None:
        with pytest.raises(ValidationError, match="bookkeeping, not endorsement"):
            tracking_summary(tags=tracking_tags(**{"engagevr.stage": "production"}))

    def test_a_disclaimer_tag_may_use_the_words_it_is_denying(self) -> None:
        # The disclaimer's whole job is to say "not validated, not
        # production". A blanket ban would make it unwritable.
        summary = tracking_summary(
            tags=tracking_tags(
                **{"engagevr.limitation": "not validated, not production-ready"}
            )
        )
        assert "not validated" in summary.tags["engagevr.limitation"]


class TestModelVersionManifest:
    def test_a_synthetic_version_cannot_be_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically eligible"):
            model_version(scientific_evaluation_eligible=True)

    def test_a_self_check_version_cannot_be_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically eligible"):
            model_version(is_synthetic=False, scientific_evaluation_eligible=True)

    def test_a_self_check_version_must_carry_the_banner(self) -> None:
        with pytest.raises(ValidationError, match="must carry the banner"):
            model_version(disclaimers=("something else",))

    def test_disclaimers_may_not_be_empty(self) -> None:
        with pytest.raises(ValidationError, match="at least one disclaimer"):
            model_version(disclaimers=())

    @pytest.mark.parametrize(
        "field",
        [
            "dataset_fingerprint",
            "split_fingerprint",
            "feature_schema_fingerprint",
            "model_artifact_sha256",
        ],
    )
    def test_every_digest_must_be_a_sha256(self, field: str) -> None:
        with pytest.raises(ValidationError):
            model_version(**{field: "not-a-digest"})

    def test_an_uppercase_digest_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="lowercase SHA-256"):
            model_version(model_artifact_sha256="D" * 64)

    def test_the_manifest_carries_no_wall_clock_field(self) -> None:
        fields = set(ModelVersionManifest.model_fields)
        assert not any("created_at" in name or "timestamp" in name for name in fields)

    def test_a_patch_level_python_version_is_refused(self) -> None:
        with pytest.raises(ValidationError, match=re.escape("major.minor")):
            model_version(python_series="3.12.13")

    def test_an_absolute_run_directory_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="absolute"):
            model_version(source_run_directory="/tmp/pytest-1234/run")

    def test_there_is_no_field_for_a_stage_or_alias(self) -> None:
        fields = set(ModelVersionManifest.model_fields)
        for forbidden in ("stage", "alias", "status", "promoted", "approved_by"):
            assert forbidden not in fields

    def test_a_fold_index_is_recorded_and_may_be_absent(self) -> None:
        assert model_version(fold_index=0).fold_index == 0
        assert model_version(fold_index=None).fold_index is None


class TestReproducibilityManifest:
    def _stage(self, **overrides: object) -> ReproducibilityStage:
        payload: dict[str, object] = {
            "name": "baseline",
            "kind": "experiment_run",
            "command": "uv run python -m engagevr baseline-demo",
            "logical_identity": "run_id:engagement_class-selfcheck-0123456789ab",
            "deterministic_artifacts": (
                DeterministicArtifact(
                    path="experiments/baseline/metrics.json",
                    sha256=DIGEST,
                    size_bytes=1024,
                ),
            ),
            "volatile_artifacts": {
                "experiments/baseline/manifest.json": "records started_at_utc",
            },
        }
        payload.update(overrides)
        return ReproducibilityStage.model_validate(payload)

    def _manifest(self, **overrides: object) -> ReproducibilityManifest:
        payload: dict[str, object] = {
            "engagevr_version": "0.1.0",
            "python_series": "3.12",
            "python_implementation": "CPython",
            "configuration": configuration_version(),
            "stages": (self._stage(),),
            "logical_fingerprint": DIGEST,
            "is_synthetic": True,
            "scientific_evaluation_eligible": False,
            "disclaimers": (MLOPS_DISCLAIMER,),
        }
        payload.update(overrides)
        return ReproducibilityManifest.model_validate(payload)

    def test_a_synthetic_pipeline_cannot_be_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically eligible"):
            self._manifest(scientific_evaluation_eligible=True)

    def test_stage_names_must_be_unique(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            self._manifest(stages=(self._stage(), self._stage()))

    def test_wall_clock_and_paths_are_declared_excluded_from_identity(self) -> None:
        excluded = " ".join(self._manifest().excluded_from_identity).lower()
        assert "wall-clock time" in excluded
        assert "absolute filesystem paths" in excluded
        assert "mlflow" in excluded

    def test_the_manifest_carries_no_wall_clock_field_at_all(self) -> None:
        fields = set(ReproducibilityManifest.model_fields)
        assert not any("created_at" in name or "timestamp" in name for name in fields)

    def test_a_volatile_artifact_must_say_why_and_carries_no_checksum(self) -> None:
        stage = self._stage()
        assert stage.volatile_artifacts
        for reason in stage.volatile_artifacts.values():
            assert reason
        # The model has nowhere to put a digest for a volatile file.
        assert (
            set(stage.volatile_artifacts)
            & {a.path for a in stage.deterministic_artifacts}
            == set()
        )

    def test_a_volatile_artifact_without_a_reason_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="must state why"):
            self._stage(volatile_artifacts={"experiments/baseline/manifest.json": ""})

    def test_an_absolute_artifact_path_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="absolute"):
            DeterministicArtifact(
                path="/home/someone/EngageVR/metrics.json",
                sha256=DIGEST,
                size_bytes=1,
            )

    def test_a_patch_level_python_version_is_refused(self) -> None:
        with pytest.raises(ValidationError, match=re.escape("major.minor")):
            self._manifest(python_series="3.12.13")

    def test_an_unknown_stage_kind_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="stage kind"):
            self._stage(kind="training")


class TestDriftSchemas:
    def test_there_is_no_concept_drift_report_kind(self) -> None:
        values = {kind.value for kind in DriftReportKind}
        assert "concept_drift" not in values
        assert values == {
            "feature_distribution_shift",
            "prediction_distribution_shift",
        }

    def test_a_computed_statistic_needs_a_value(self) -> None:
        with pytest.raises(ValidationError, match="carries no statistic"):
            DriftStatistic(method=DriftMethod.KOLMOGOROV_SMIRNOV)

    def test_an_unavailable_statistic_needs_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state a reason"):
            DriftStatistic(
                method=DriftMethod.KOLMOGOROV_SMIRNOV,
                status=DriftStatus.UNAVAILABLE_INSUFFICIENT_SAMPLES,
            )

    def test_an_unavailable_statistic_may_not_report_a_verdict(self) -> None:
        with pytest.raises(ValidationError, match="not a passing one"):
            DriftStatistic(
                method=DriftMethod.KOLMOGOROV_SMIRNOV,
                status=DriftStatus.UNAVAILABLE_INSUFFICIENT_SAMPLES,
                unavailable_reason="too few samples",
                exceeded=False,
            )

    def test_an_unavailable_statistic_is_not_reported_as_zero(self) -> None:
        statistic = DriftStatistic(
            method=DriftMethod.KOLMOGOROV_SMIRNOV,
            status=DriftStatus.UNAVAILABLE_ALL_VALUES_MISSING,
            unavailable_reason="every value is missing on both sides",
        )
        assert statistic.statistic is None
        assert statistic.exceeded is None

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_statistic_is_refused(self, value: float) -> None:
        with pytest.raises(ValidationError, match="not a statistic"):
            DriftStatistic(
                method=DriftMethod.KOLMOGOROV_SMIRNOV,
                statistic=value,
                threshold=0.1,
                exceeded=False,
            )

    def test_exceeded_methods_must_agree_with_the_statistics(self) -> None:
        with pytest.raises(ValidationError, match="exceeded_methods"):
            FeatureDriftResult(
                feature_name="feat__x",
                value_kind="numeric",
                reference_row_count=100,
                current_row_count=100,
                reference_present_count=100,
                current_present_count=100,
                statistics=(
                    DriftStatistic(
                        method=DriftMethod.KOLMOGOROV_SMIRNOV,
                        statistic=0.01,
                        threshold=0.1,
                        exceeded=False,
                    ),
                ),
                exceeded_methods=(DriftMethod.KOLMOGOROV_SMIRNOV,),
            )

    def test_a_synthetic_report_cannot_be_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically eligible"):
            drift_report(scientific_evaluation_eligible=True)

    def test_counts_must_agree_with_the_results(self) -> None:
        with pytest.raises(ValidationError, match="disagrees"):
            drift_report(features_compared_count=3)

    def test_the_report_carries_no_wall_clock_field(self) -> None:
        fields = set(DriftReport.model_fields)
        assert not any("created_at" in name or "timestamp" in name for name in fields)

    def test_the_terminology_note_denies_concept_drift(self) -> None:
        note = drift_report().terminology_note.lower()
        assert "not concept drift" in note
        assert "distribution shift" in note

    def test_there_is_no_overall_failure_field(self) -> None:
        fields = set(DriftReport.model_fields)
        for forbidden in ("model_failed", "alarm", "passed", "verdict", "status"):
            assert forbidden not in fields

    def test_a_dataset_side_must_be_reference_or_current(self) -> None:
        with pytest.raises(ValidationError, match="reference' or 'current"):
            DriftDatasetReference(
                role="baseline",
                path="a.parquet",
                row_count=1,
                is_synthetic=True,
                scientific_evaluation_eligible=False,
            )


class TestTrackingSummary:
    def test_every_provenance_tag_is_required(self) -> None:
        for tag in REQUIRED_TRACKING_TAGS:
            tags = tracking_tags()
            del tags[tag]
            with pytest.raises(ValidationError, match="provenance tag"):
                tracking_summary(tags=tags)

    def test_a_synthetic_run_must_be_tagged_synthetic(self) -> None:
        with pytest.raises(ValidationError, match=re.escape("engagevr.is_synthetic")):
            tracking_summary(tags=tracking_tags(**{"engagevr.is_synthetic": "false"}))

    def test_a_synthetic_run_must_be_tagged_ineligible(self) -> None:
        with pytest.raises(ValidationError, match="scientific_evaluation_eligible"):
            tracking_summary(
                tags=tracking_tags(
                    **{"engagevr.scientific_evaluation_eligible": "true"}
                )
            )

    def test_a_synthetic_run_must_carry_the_banner(self) -> None:
        with pytest.raises(ValidationError, match="must carry the banner"):
            tracking_summary(
                tags=tracking_tags(**{"engagevr.disclaimer": "looks fine"})
            )

    def test_a_synthetic_run_cannot_be_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically eligible"):
            tracking_summary(scientific_evaluation_eligible=True)

    def test_no_model_may_be_registered(self) -> None:
        with pytest.raises(ValidationError):
            tracking_summary(registered_model="engagevr-engagement")

    def test_a_skipped_metric_records_a_reason_and_not_a_zero(self) -> None:
        summary = tracking_summary(
            skipped_metrics={"logistic_regression/roc_auc": "one class in fold"}
        )
        assert "logistic_regression/roc_auc" not in summary.metrics
        assert summary.skipped_metrics["logistic_regression/roc_auc"]

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_a_non_finite_metric_is_refused(self, value: float) -> None:
        with pytest.raises(ValidationError, match="not a statistic"):
            tracking_summary(metrics={"accuracy": value})


class TestSmokeReport:
    def _report(self, **overrides: object) -> SmokeReport:
        payload: dict[str, object] = {
            "engagevr_version": "0.1.0",
            "python_version": "3.12.13",
            "checks": (
                SmokeCheckResult(
                    name="package_imports",
                    status=SmokeCheckStatus.PASSED,
                    detail="ok",
                ),
            ),
            "passed_count": 1,
            "failed_count": 0,
            "skipped_count": 0,
            "status": SmokeCheckStatus.PASSED,
            "created_at_utc": MOMENT,
            "disclaimers": (MLOPS_DISCLAIMER,),
        }
        payload.update(overrides)
        return SmokeReport.model_validate(payload)

    def test_the_banner_is_required_verbatim(self) -> None:
        assert self._report().banner == SOFTWARE_SELF_CHECK_BANNER
        with pytest.raises(ValidationError, match="must carry the banner"):
            self._report(banner="all good")

    def test_a_smoke_check_can_never_be_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically eligible"):
            self._report(scientific_evaluation_eligible=True)

    def test_counts_must_agree_with_the_checks(self) -> None:
        with pytest.raises(ValidationError, match="counts disagree"):
            self._report(passed_count=7)

    def test_a_failed_check_forces_a_failed_report(self) -> None:
        failed = SmokeCheckResult(
            name="backend",
            status=SmokeCheckStatus.FAILED,
            failure_reason="no /health route",
        )
        with pytest.raises(ValidationError, match="must have status 'failed'"):
            self._report(
                checks=(failed,),
                passed_count=0,
                failed_count=1,
                status=SmokeCheckStatus.PASSED,
            )

    def test_the_report_itself_cannot_be_skipped(self) -> None:
        with pytest.raises(ValidationError, match="the report may not"):
            self._report(
                checks=(),
                passed_count=0,
                skipped_count=0,
                status=SmokeCheckStatus.SKIPPED,
            )

    def test_a_failed_check_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state a reason"):
            SmokeCheckResult(name="x", status=SmokeCheckStatus.FAILED)

    def test_a_skipped_check_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state a reason"):
            SmokeCheckResult(name="x", status=SmokeCheckStatus.SKIPPED)

    def test_no_duration_is_recorded(self) -> None:
        # A wall-clock duration would differ between two identical runs
        # and make an otherwise deterministic report incomparable.
        fields = set(SmokeCheckResult.model_fields)
        assert not any("duration" in name or "elapsed" in name for name in fields)


class TestConfigurationVersion:
    def test_every_exclusion_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="why it was excluded"):
            configuration_version(excluded_paths=("logging.file",))

    def test_the_fingerprint_must_be_a_sha256(self) -> None:
        with pytest.raises(ValidationError):
            configuration_version(config_fingerprint="short")

    def test_the_fingerprint_inputs_are_documented(self) -> None:
        inputs = configuration_version().fingerprint_inputs.lower()
        assert "canonical json" in inputs
        assert "wall-clock" in inputs
