"""Deriving immutable, checksum-linked model versions from a run.

Three properties matter here, and each has its own group below.

**Determinism.** Re-deriving a version from the same run reproduces the
same identifier rather than minting a new one.

**Immutability of the source.** Building a version leaves the producing
run byte-identical.  That is what makes the recorded checksums mean
anything: they describe a run that was not touched in order to describe
it.

**Refusal.** A tampered artifact, an incomplete run, or a run with no
persisted estimator produces an error, not a version record.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from engagevr.config import load_config
from engagevr.mlops.model_version import (
    MANIFEST_SUFFIX,
    REFERENCED_DOCUMENTS,
    ModelVersionError,
    build_model_versions,
    read_model_version,
    summarise,
    verify_model_version,
    write_model_versions,
)
from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER, EvaluationMode
from engagevr.training.artifacts import sha256_file


def directory_digest(directory: Path) -> dict[str, str]:
    """SHA-256 of every file under ``directory``, keyed by relative path."""
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class TestBuilding:
    def test_one_version_per_persisted_estimator(self, m10_baseline_run: Path) -> None:
        versions = build_model_versions(m10_baseline_run, config=load_config())
        files = sorted((m10_baseline_run / "models").glob("*.joblib"))
        assert len(versions) == len(files)
        assert {v.model_artifact_path for v in versions} == {
            f"models/{p.name}" for p in files
        }

    def test_the_identifier_is_deterministic(self, m10_baseline_run: Path) -> None:
        config = load_config()
        first = build_model_versions(m10_baseline_run, config=config)
        second = build_model_versions(m10_baseline_run, config=config)
        assert [v.model_version_id for v in first] == [
            v.model_version_id for v in second
        ]

    def test_the_identifier_carries_no_wall_clock(self, m10_baseline_run: Path) -> None:
        versions = build_model_versions(m10_baseline_run, config=load_config())
        # Two builds a moment apart differ in created_at_utc and agree in
        # identity, which is the whole point.
        again = build_model_versions(m10_baseline_run, config=load_config())
        assert versions[0].model_version_id == again[0].model_version_id
        assert "created_at_utc" not in versions[0].model_version_inputs

    def test_the_configuration_changes_the_identifier(
        self, m10_baseline_run: Path
    ) -> None:
        config = load_config()
        changed = config.model_copy(
            update={"training": config.training.model_copy(update={"folds": 9})}
        )
        baseline = build_model_versions(m10_baseline_run, config=config)
        other = build_model_versions(m10_baseline_run, config=changed)
        assert baseline[0].model_version_id != other[0].model_version_id

    def test_every_provenance_field_is_populated(self, m10_baseline_run: Path) -> None:
        version = build_model_versions(m10_baseline_run, config=load_config())[0]
        manifest = json.loads(
            (m10_baseline_run / "manifest.json").read_text(encoding="utf-8")
        )
        assert version.source_run_id == manifest["run_id"]
        assert version.dataset_fingerprint == manifest["dataset_fingerprint"]
        assert version.target_name == manifest["target_name"]
        assert version.task_type == manifest["task_type"]
        assert version.feature_count == len(manifest["feature_set"])
        assert version.source_run_family == "baseline"
        assert len(version.split_fingerprint) == 64
        assert len(version.feature_schema_fingerprint) == 64
        assert version.configuration.config_fingerprint

    def test_the_serialization_format_and_library_are_recorded(
        self, m10_baseline_run: Path
    ) -> None:
        version = build_model_versions(m10_baseline_run, config=load_config())[0]
        assert version.serialization_format == "joblib-pickle"
        assert version.serialization_library == "joblib"
        assert version.serialization_library_version
        assert "pickle" in version.serialization_warning.lower()

    def test_the_fold_and_calibration_state_are_recorded(
        self, m10_baseline_run: Path
    ) -> None:
        versions = build_model_versions(m10_baseline_run, config=load_config())
        by_name = {v.model_name: v for v in versions}
        for name, version in by_name.items():
            assert version.fold_index == 0, name
            assert version.is_calibrated is name.endswith("-calibrated")

    def test_the_checksum_matches_the_file_on_disk(
        self, m10_baseline_run: Path
    ) -> None:
        for version in build_model_versions(m10_baseline_run, config=load_config()):
            path = m10_baseline_run / version.model_artifact_path
            assert version.model_artifact_sha256 == sha256_file(path)
            assert version.model_artifact_bytes == path.stat().st_size

    def test_referenced_documents_are_checksum_linked(
        self, m10_baseline_run: Path
    ) -> None:
        version = build_model_versions(m10_baseline_run, config=load_config())[0]
        for name in REFERENCED_DOCUMENTS:
            assert name in version.referenced_checksums
            assert version.referenced_checksums[name] == sha256_file(
                m10_baseline_run / name
            )

    def test_the_timestamped_dataset_document_is_not_referenced(
        self, m10_baseline_run: Path
    ) -> None:
        # dataset.json copies the dataset metadata verbatim, including
        # created_at_utc, so its digest changes on every execution. A
        # volatile digest inside a version record would make the record
        # volatile, and the record is a DVC-declared output.
        version = build_model_versions(m10_baseline_run, config=load_config())[0]
        assert "dataset.json" not in version.referenced_checksums
        # The dataset is still pinned, by a fingerprint that excludes the
        # wall clock by construction.
        assert len(version.dataset_fingerprint) == 64

    def test_the_record_is_byte_stable_across_two_builds(
        self, m10_baseline_run: Path
    ) -> None:
        config = load_config()
        first = build_model_versions(m10_baseline_run, config=config)[0]
        second = build_model_versions(m10_baseline_run, config=config)[0]
        assert first.model_dump(mode="json") == second.model_dump(mode="json")

    def test_a_named_subset_can_be_versioned(self, m10_baseline_run: Path) -> None:
        versions = build_model_versions(
            m10_baseline_run, config=load_config(), model_names=["dummy"]
        )
        assert versions
        assert all(v.model_name.startswith("dummy") for v in versions)

    def test_an_unknown_model_name_is_an_error(self, m10_baseline_run: Path) -> None:
        with pytest.raises(ModelVersionError, match="no persisted estimator"):
            build_model_versions(
                m10_baseline_run, config=load_config(), model_names=["nonexistent"]
            )


class TestScientificStatus:
    def test_a_synthetic_run_yields_ineligible_versions(
        self, m10_baseline_run: Path
    ) -> None:
        for version in build_model_versions(m10_baseline_run, config=load_config()):
            assert version.is_synthetic is True
            assert version.scientific_evaluation_eligible is False
            assert version.evaluation_mode is EvaluationMode.SOFTWARE_SELF_CHECK

    def test_the_banner_and_the_limitation_are_carried(
        self, m10_baseline_run: Path
    ) -> None:
        version = build_model_versions(m10_baseline_run, config=load_config())[0]
        assert any(SOFTWARE_SELF_CHECK_BANNER in d for d in version.disclaimers)
        limitation = version.limitation.lower()
        assert "not an approval" in limitation
        assert "not a release" in limitation

    def test_the_data_source_counts_are_carried(self, m10_baseline_run: Path) -> None:
        version = build_model_versions(m10_baseline_run, config=load_config())[0]
        assert set(version.data_source_counts) == {"synthetic"}


class TestImmutability:
    def test_building_a_version_does_not_touch_the_run(
        self, m10_baseline_run: Path
    ) -> None:
        before = directory_digest(m10_baseline_run)
        build_model_versions(m10_baseline_run, config=load_config())
        assert directory_digest(m10_baseline_run) == before

    def test_writing_versions_does_not_touch_the_run(
        self, m10_baseline_run: Path, tmp_path: Path
    ) -> None:
        before = directory_digest(m10_baseline_run)
        versions = build_model_versions(m10_baseline_run, config=load_config())
        write_model_versions(versions, tmp_path / "versions")
        assert directory_digest(m10_baseline_run) == before

    def test_versions_are_written_outside_the_run_directory(
        self, m10_baseline_run: Path, tmp_path: Path
    ) -> None:
        versions = build_model_versions(m10_baseline_run, config=load_config())
        written = write_model_versions(versions, tmp_path / "versions")
        for path in written:
            assert m10_baseline_run not in path.parents


class TestRoundTrip:
    def test_a_written_version_reads_back_identically(
        self, m10_baseline_run: Path, tmp_path: Path
    ) -> None:
        versions = build_model_versions(m10_baseline_run, config=load_config())
        written = write_model_versions(versions, tmp_path / "versions")
        for path, original in zip(written, versions, strict=True):
            assert path.name.endswith(MANIFEST_SUFFIX)
            assert read_model_version(path) == original

    def test_verification_passes_against_the_untouched_run(
        self, m10_baseline_run: Path
    ) -> None:
        for version in build_model_versions(m10_baseline_run, config=load_config()):
            assert verify_model_version(version, run_directory=m10_baseline_run) == ()

    def test_verification_names_a_tampered_model_file(
        self, m10_baseline_run: Path, tmp_path: Path
    ) -> None:
        import shutil

        copy = tmp_path / "run"
        shutil.copytree(m10_baseline_run, copy)
        version = build_model_versions(copy, config=load_config())[0]
        target = copy / version.model_artifact_path
        target.write_bytes(target.read_bytes() + b"tampered")
        assert verify_model_version(version, run_directory=copy) == (
            version.model_artifact_path,
        )

    def test_verification_names_a_tampered_metrics_document(
        self, m10_baseline_run: Path, tmp_path: Path
    ) -> None:
        import shutil

        copy = tmp_path / "run"
        shutil.copytree(m10_baseline_run, copy)
        version = build_model_versions(copy, config=load_config())[0]
        (copy / "metrics.json").write_text("{}", encoding="utf-8")
        assert "metrics.json" in verify_model_version(version, run_directory=copy)


class TestRefusals:
    def test_a_directory_with_no_manifest_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ModelVersionError, match=re.escape("no manifest.json")):
            build_model_versions(tmp_path, config=load_config())

    def test_a_failed_run_is_refused(
        self, m10_baseline_run: Path, tmp_path: Path
    ) -> None:
        import shutil

        copy = tmp_path / "run"
        shutil.copytree(m10_baseline_run, copy)
        manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
        manifest["status"] = "failed"
        manifest["failure_reason"] = "deliberately marked failed for this test"
        (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ModelVersionError, match="Only a completed run"):
            build_model_versions(copy, config=load_config())

    def test_a_run_with_no_estimator_is_refused(
        self, m10_baseline_run: Path, tmp_path: Path
    ) -> None:
        import shutil

        copy = tmp_path / "run"
        shutil.copytree(m10_baseline_run, copy)
        for path in (copy / "models").glob("*.joblib"):
            path.unlink()
        with pytest.raises(ModelVersionError, match=re.escape("contains no .joblib")):
            build_model_versions(copy, config=load_config())

    def test_a_model_whose_bytes_changed_is_refused_at_build_time(
        self, m10_baseline_run: Path, tmp_path: Path
    ) -> None:
        import shutil

        copy = tmp_path / "run"
        shutil.copytree(m10_baseline_run, copy)
        target = next((copy / "models").glob("*.joblib"))
        target.write_bytes(b"not the estimator that was fitted")
        with pytest.raises(ModelVersionError, match="bytes changed after the run"):
            build_model_versions(copy, config=load_config())

    def test_no_model_file_is_ever_loaded(
        self, m10_baseline_run: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A .joblib is a pickle; loading one executes code in it. Nothing
        # in this layer needs to.
        import joblib

        def explode(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("a model file was unpickled")

        monkeypatch.setattr(joblib, "load", explode)
        assert build_model_versions(m10_baseline_run, config=load_config())


class TestSummary:
    def test_the_one_line_summary_states_the_scientific_status(
        self, m10_baseline_run: Path
    ) -> None:
        version = build_model_versions(m10_baseline_run, config=load_config())[0]
        line = summarise(version)
        assert version.model_version_id in line
        assert "eligible=false" in line
