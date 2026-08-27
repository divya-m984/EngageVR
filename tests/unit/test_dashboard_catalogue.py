"""Artifact discovery, run-family detection, and integrity.

The catalogue's job is to make a bad run visible as a bad run, so most
of these tests break something on purpose and then check that the
breakage is reported rather than swallowed.
"""

from __future__ import annotations

from pathlib import Path

from engagevr.dashboard.catalogue import (
    FAMILY_SIGNATURES,
    build_catalogue,
    detect_family,
    inspect_run,
    verify_integrity,
)
from engagevr.schemas.dashboard import (
    ArtifactIntegrityStatus,
    DashboardRunFamily,
    DashboardRunStatus,
    DashboardWarningLevel,
)
from tests.unit import dashboard_fixtures as fx


class TestEmptyAndMissingRoots:
    def test_a_missing_root_is_a_state_not_an_exception(self, tmp_path: Path) -> None:
        catalogue = build_catalogue(tmp_path / "nothing-here")
        assert not catalogue.root_exists
        assert catalogue.is_empty
        assert catalogue.warnings

    def test_a_missing_root_says_what_to_do(self, tmp_path: Path) -> None:
        catalogue = build_catalogue(tmp_path / "nothing-here")
        assert "does not exist" in catalogue.warnings[0].message

    def test_an_empty_root_is_not_an_error(self, tmp_path: Path) -> None:
        catalogue = build_catalogue(tmp_path)
        assert catalogue.root_exists
        assert catalogue.is_empty
        assert "not an error" in catalogue.warnings[0].message

    def test_a_file_where_the_root_should_be_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "root"
        path.write_text("not a directory", encoding="utf-8")
        catalogue = build_catalogue(path)
        assert not catalogue.root_exists
        assert catalogue.warnings[0].level is DashboardWarningLevel.ERROR

    def test_loose_files_beside_runs_are_ignored(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path)
        (tmp_path / "notes.txt").write_text("scratch", encoding="utf-8")
        catalogue = build_catalogue(tmp_path)
        assert [run.directory_name for run in catalogue.runs] == ["fixture-baseline"]


class TestFamilyDetection:
    def test_a_baseline_run_is_found(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.family is DashboardRunFamily.BASELINE

    def test_a_fusion_run_is_found(self, tmp_path: Path) -> None:
        fx.make_fusion_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-fusion")
        assert run.provenance.family is DashboardRunFamily.FUSION

    def test_a_personalization_run_is_found(self, tmp_path: Path) -> None:
        fx.make_personalization_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-personalization")
        assert run.provenance.family is DashboardRunFamily.PERSONALIZATION

    def test_an_uncertainty_run_is_found(self, tmp_path: Path) -> None:
        fx.make_uncertainty_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-uncertainty")
        assert run.provenance.family is DashboardRunFamily.UNCERTAINTY

    def test_an_adaptation_run_is_found(self, tmp_path: Path) -> None:
        fx.make_adaptation_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-adaptation")
        assert run.provenance.family is DashboardRunFamily.ADAPTATION

    def test_a_fusion_run_is_not_read_as_a_baseline_run(self, tmp_path: Path) -> None:
        # Both write manifest.json, metrics.json, and splits.json. Only the
        # fusion documents tell them apart.
        fx.make_fusion_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-fusion")
        assert run.provenance.family is not DashboardRunFamily.BASELINE

    def test_the_directory_name_does_not_classify_a_run(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path, "m7-not-really-an-uncertainty-run")
        run = fx.summary_for(tmp_path, "m7-not-really-an-uncertainty-run")
        assert run.provenance.family is DashboardRunFamily.BASELINE

    def test_a_misleading_name_on_an_empty_directory_stays_unknown(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "m8-adaptation-something").mkdir()
        run = fx.summary_for(tmp_path, "m8-adaptation-something")
        assert run.provenance.family is DashboardRunFamily.UNKNOWN
        assert run.provenance.status is DashboardRunStatus.UNKNOWN

    def test_the_detection_note_names_the_evidence(self, tmp_path: Path) -> None:
        fx.make_adaptation_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-adaptation")
        assert run.detection_note is not None
        assert "adaptation_summary.json" in run.detection_note
        assert "directory name" in run.detection_note

    def test_detect_family_reports_unknown_for_an_empty_directory(
        self, tmp_path: Path
    ) -> None:
        family, note = detect_family(tmp_path)
        assert family is DashboardRunFamily.UNKNOWN
        assert note is not None and "guessed" in note

    def test_an_interrupted_run_keeps_its_family(self, tmp_path: Path) -> None:
        directory = fx.make_fusion_run(tmp_path)
        (directory / "fusion_metrics.json").unlink()
        run = fx.summary_for(tmp_path, "fixture-fusion")
        assert run.provenance.family is DashboardRunFamily.FUSION
        assert run.provenance.status is DashboardRunStatus.INCOMPLETE
        assert run.detection_note is not None
        assert "INCOMPLETE" in run.detection_note

    def test_every_signature_declares_its_distinguishing_artifacts(self) -> None:
        for signature in FAMILY_SIGNATURES:
            assert signature.distinguishing
            assert set(signature.distinguishing) <= set(signature.required) | set(
                signature.optional
            )


class TestOrdering:
    def test_runs_are_ordered_by_directory_name(self, tmp_path: Path) -> None:
        for name in ("zeta", "alpha", "mu"):
            fx.make_baseline_run(tmp_path, name)
        catalogue = build_catalogue(tmp_path)
        assert [run.directory_name for run in catalogue.runs] == [
            "alpha",
            "mu",
            "zeta",
        ]

    def test_two_scans_agree(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path, "one")
        fx.make_fusion_run(tmp_path, "two")
        first = build_catalogue(tmp_path)
        second = build_catalogue(tmp_path)
        assert first == second


class TestStatus:
    def test_a_completed_run_is_completed(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.status is DashboardRunStatus.COMPLETED

    def test_a_failed_run_records_its_reason(self, tmp_path: Path) -> None:
        fx.make_baseline_run(
            tmp_path, status="failed", failure_reason="the estimator did not converge"
        )
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.status is DashboardRunStatus.FAILED
        assert run.provenance.failure_reason == "the estimator did not converge"

    def test_a_run_with_no_manifest_is_incomplete(self, tmp_path: Path) -> None:
        directory = fx.make_baseline_run(tmp_path)
        (directory / "manifest.json").unlink()
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.status is DashboardRunStatus.INCOMPLETE
        assert "interrupted" in run.provenance.warnings[0].message

    def test_a_missing_required_artifact_makes_a_run_incomplete(
        self, tmp_path: Path
    ) -> None:
        directory = fx.make_baseline_run(tmp_path)
        (directory / "splits.json").unlink()
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.status is DashboardRunStatus.INCOMPLETE
        assert "splits.json" in run.missing_required_artifacts

    def test_corrupt_json_is_visible_as_corrupt(self, tmp_path: Path) -> None:
        directory = fx.make_baseline_run(tmp_path)
        fx.corrupt(directory / "manifest.json")
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.status is DashboardRunStatus.CORRUPT
        assert "not valid JSON" in run.provenance.warnings[0].message

    def test_a_corrupt_run_is_listed_rather_than_dropped(self, tmp_path: Path) -> None:
        directory = fx.make_baseline_run(tmp_path)
        fx.corrupt(directory / "manifest.json")
        fx.make_fusion_run(tmp_path)
        catalogue = build_catalogue(tmp_path)
        assert len(catalogue.runs) == 2

    def test_one_bad_run_does_not_stop_the_others(self, tmp_path: Path) -> None:
        fx.corrupt(fx.make_baseline_run(tmp_path, "broken") / "manifest.json")
        fx.make_fusion_run(tmp_path, "healthy")
        catalogue = build_catalogue(tmp_path)
        healthy = catalogue.find("healthy")
        assert healthy is not None
        assert healthy.provenance.status is DashboardRunStatus.COMPLETED

    def test_an_unsupported_catalog_version_is_refused_cleanly(
        self, tmp_path: Path
    ) -> None:
        fx.make_baseline_run(tmp_path, feature_catalog_version="99.0")
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.status is DashboardRunStatus.UNSUPPORTED
        assert "can interpret" in run.provenance.warnings[0].message

    def test_an_unsupported_run_is_not_inspectable(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path, feature_catalog_version="99.0")
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert not run.is_inspectable

    def test_an_unknown_status_word_is_flagged_not_assumed(
        self, tmp_path: Path
    ) -> None:
        directory = fx.make_baseline_run(tmp_path)
        document = fx.manifest_document(run_id="x")
        document["status"] = "probably fine"
        fx.write_json(directory / "manifest.json", document)
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.status is DashboardRunStatus.UNKNOWN


class TestIntegrity:
    def test_matching_checksums_pass(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-baseline", validate=True)
        assert run.provenance.integrity is ArtifactIntegrityStatus.VALID

    def test_a_mismatch_is_visible(self, tmp_path: Path) -> None:
        directory = fx.make_baseline_run(tmp_path)
        (directory / "metrics.json").write_text("{}", encoding="utf-8")
        run = fx.summary_for(tmp_path, "fixture-baseline", validate=True)
        assert run.provenance.integrity is ArtifactIntegrityStatus.MISMATCHED

    def test_a_mismatch_produces_an_error_level_warning(self, tmp_path: Path) -> None:
        directory = fx.make_baseline_run(tmp_path)
        (directory / "metrics.json").write_text("{}", encoding="utf-8")
        run = fx.summary_for(tmp_path, "fixture-baseline", validate=True)
        levels = {w.level for w in run.provenance.warnings}
        assert DashboardWarningLevel.ERROR in levels

    def test_a_missing_checksum_file_is_not_a_failed_check(
        self, tmp_path: Path
    ) -> None:
        directory = fx.make_baseline_run(tmp_path, with_checksums=False)
        status, offenders, message = verify_integrity(directory, validate=True)
        assert status is ArtifactIntegrityStatus.CHECKSUM_FILE_UNAVAILABLE
        assert not offenders
        assert "not the same as a failed check" in str(message)

    def test_a_referenced_file_that_vanished_is_its_own_state(
        self, tmp_path: Path
    ) -> None:
        directory = fx.make_baseline_run(tmp_path)
        (directory / "predictions.parquet").unlink()
        status, offenders, _message = verify_integrity(directory, validate=True)
        assert status is ArtifactIntegrityStatus.REFERENCED_FILE_MISSING
        assert "predictions.parquet" in offenders

    def test_a_corrupt_checksum_file_is_its_own_state(self, tmp_path: Path) -> None:
        directory = fx.make_baseline_run(tmp_path)
        fx.corrupt(directory / "checksums.json")
        status, _offenders, _message = verify_integrity(directory, validate=True)
        assert status is ArtifactIntegrityStatus.CHECKSUM_FILE_CORRUPT

    def test_verification_is_opt_in(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-baseline", validate=False)
        assert run.provenance.integrity is ArtifactIntegrityStatus.NOT_CHECKED

    def test_skipping_verification_never_reports_valid(self, tmp_path: Path) -> None:
        directory = fx.make_baseline_run(tmp_path)
        (directory / "metrics.json").write_text("{}", encoding="utf-8")
        run = fx.summary_for(tmp_path, "fixture-baseline", validate=False)
        assert run.provenance.integrity is not ArtifactIntegrityStatus.VALID

    def test_verification_does_not_modify_the_run(self, tmp_path: Path) -> None:
        directory = fx.make_baseline_run(tmp_path)
        before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        (directory / "metrics.json").write_text("{}", encoding="utf-8")
        inspect_run(directory, validate_checksums=True)
        after = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        assert set(before) == set(after)
        assert after["checksums.json"] == before["checksums.json"]


class TestArtifactAvailability:
    def test_a_required_artifact_is_marked_required(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-baseline")
        required = {a.name for a in run.artifacts if a.required}
        assert "metrics.json" in required

    def test_an_optional_artifact_absence_does_not_make_a_run_incomplete(
        self, tmp_path: Path
    ) -> None:
        fx.make_baseline_run(tmp_path, with_predictions=False)
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.status is DashboardRunStatus.COMPLETED
        absent = {a.name for a in run.artifacts if not a.present}
        assert "predictions.parquet" in absent

    def test_an_absent_artifact_states_a_reason(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path, with_predictions=False)
        run = fx.summary_for(tmp_path, "fixture-baseline")
        entry = next(a for a in run.artifacts if a.name == "predictions.parquet")
        assert entry.unavailable_reason
        assert "not present" in entry.unavailable_reason

    def test_a_present_artifact_records_its_size(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-baseline")
        entry = next(a for a in run.artifacts if a.name == "metrics.json")
        assert entry.size_bytes is not None and entry.size_bytes > 0


class TestProvenanceConflicts:
    def test_a_configuration_marker_disagreement_is_reported(
        self, tmp_path: Path
    ) -> None:
        directory = fx.make_fusion_run(tmp_path)
        document = fx.manifest_document(run_id="conflicted")
        document["configuration"] = {"milestone": 5}
        fx.write_json(directory / "manifest.json", document)
        run = fx.summary_for(tmp_path, "fixture-fusion")
        messages = " ".join(w.message for w in run.provenance.warnings)
        assert "disagree" in messages

    def test_a_self_check_run_is_never_eligible(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.is_synthetic
        assert not run.provenance.scientific_evaluation_eligible

    def test_a_contradictory_manifest_is_shown_as_corrupt(self, tmp_path: Path) -> None:
        directory = fx.make_baseline_run(tmp_path)
        document = fx.manifest_document(run_id="contradiction")
        document["scientific_evaluation_eligible"] = True
        fx.write_json(directory / "manifest.json", document)
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.status is DashboardRunStatus.CORRUPT
        assert not run.provenance.scientific_evaluation_eligible

    def test_the_data_source_comes_from_the_dataset_document(
        self, tmp_path: Path
    ) -> None:
        fx.make_baseline_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.data_source == "synthetic"

    def test_a_mixed_source_dataset_is_reported_as_mixed(self, tmp_path: Path) -> None:
        directory = fx.make_baseline_run(tmp_path)
        document = fx.dataset_document()
        document["data_source_counts"] = {"synthetic": 30, "public_dataset": 30}
        fx.write_json(directory / "dataset.json", document)
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.data_source == "mixed"

    def test_an_unreadable_dataset_document_does_not_break_the_run(
        self, tmp_path: Path
    ) -> None:
        directory = fx.make_baseline_run(tmp_path)
        fx.corrupt(directory / "dataset.json")
        run = fx.summary_for(tmp_path, "fixture-baseline")
        assert run.provenance.data_source is None
        assert run.provenance.status is DashboardRunStatus.COMPLETED


class TestCatalogueQueries:
    def test_families_are_reported_in_declaration_order(self, tmp_path: Path) -> None:
        fx.make_adaptation_run(tmp_path)
        fx.make_baseline_run(tmp_path)
        catalogue = build_catalogue(tmp_path)
        assert catalogue.families() == (
            DashboardRunFamily.BASELINE,
            DashboardRunFamily.ADAPTATION,
        )

    def test_by_family_filters(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path, "a")
        fx.make_baseline_run(tmp_path, "b")
        fx.make_fusion_run(tmp_path, "c")
        catalogue = build_catalogue(tmp_path)
        assert len(catalogue.by_family(DashboardRunFamily.BASELINE)) == 2

    def test_find_returns_none_for_an_unknown_directory(self, tmp_path: Path) -> None:
        fx.make_baseline_run(tmp_path)
        assert build_catalogue(tmp_path).find("nope") is None
