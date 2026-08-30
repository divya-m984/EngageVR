"""The integrated software self-check, run for real.

One execution of :func:`engagevr.mlops.smoke.run_system_smoke` against a
temporary directory, with every assertion drawn from that one run.  It is
the only test in the suite that exercises Milestones 4, 5, 9, and 10
together in one process.

It still needs nothing outside the repository: no webcam, no network, no
Unity, no browser, no display server, no external dataset, no participant
data, and no server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engagevr.config import load_config
from engagevr.mlops.smoke import CHECKS, run_system_smoke
from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER
from engagevr.schemas.mlops import SmokeCheckStatus, SmokeReport


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> SmokeReport:
    directory = tmp_path_factory.mktemp("system-smoke")
    return run_system_smoke(directory, config=load_config())


@pytest.fixture(scope="module")
def by_name(report: SmokeReport) -> dict[str, object]:
    return {check.name: check for check in report.checks}


class TestEveryComponentInteroperates:
    def test_the_run_passes(self, report: SmokeReport) -> None:
        failures = [
            f"{check.name}: {check.failure_reason}"
            for check in report.checks
            if check.status is SmokeCheckStatus.FAILED
        ]
        assert not failures, failures
        assert report.status is SmokeCheckStatus.PASSED

    def test_every_declared_check_ran(self, by_name: dict[str, object]) -> None:
        assert set(by_name) == {name for name, _ in CHECKS}

    @pytest.mark.parametrize(
        "name",
        [
            "package_imports",
            "configuration_loads",
            "synthetic_dataset_generated",
            "dataset_provenance_preserved",
            "baseline_pipeline_ran",
            "artifact_manifest_validated",
            "model_version_manifest_validated",
            "mlflow_tracking_local",
            "drift_diagnostic_ran",
            "dashboard_catalogue_discovered_run",
            "backend_application_created",
            "dashboard_module_imports",
            "protocol_artifacts_current",
        ],
    )
    def test_the_named_check_did_not_fail(
        self, by_name: dict[str, object], name: str
    ) -> None:
        check = by_name[name]
        assert check.status is not SmokeCheckStatus.FAILED, check.failure_reason  # type: ignore[attr-defined]

    def test_the_backend_was_constructed_without_binding_a_socket(
        self, by_name: dict[str, object]
    ) -> None:
        detail = by_name["backend_application_created"].detail  # type: ignore[attr-defined]
        assert "/health" in detail
        assert "no socket was bound" in detail

    def test_the_dashboard_was_not_started(self, by_name: dict[str, object]) -> None:
        detail = by_name["dashboard_module_imports"].detail  # type: ignore[attr-defined]
        assert "no server started" in detail
        assert "headless" in detail


class TestScientificStatus:
    def test_the_banner_is_present_verbatim(self, report: SmokeReport) -> None:
        assert report.banner == SOFTWARE_SELF_CHECK_BANNER

    def test_the_report_is_never_scientifically_eligible(
        self, report: SmokeReport
    ) -> None:
        assert report.is_synthetic is True
        assert report.scientific_evaluation_eligible is False

    def test_it_says_reproducibility_is_not_validity(self, report: SmokeReport) -> None:
        assert "Reproducibility is not validity" in report.note
        assert "Tracking is not validation" in report.note

    def test_every_provenance_bearing_check_reports_ineligibility(
        self, by_name: dict[str, object]
    ) -> None:
        for name in (
            "dataset_provenance_preserved",
            "artifact_manifest_validated",
            "mlflow_tracking_local",
            "dashboard_catalogue_discovered_run",
        ):
            check = by_name[name]
            if check.status is SmokeCheckStatus.SKIPPED:  # type: ignore[attr-defined]
                continue
            assert "eligible=false" in check.detail.replace(  # type: ignore[attr-defined]
                "scientific_evaluation_eligible=false", "eligible=false"
            ), name


class TestIsolation:
    def test_nothing_is_written_outside_the_given_directory(
        self, tmp_path: Path
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        before = {
            path.name
            for path in root.iterdir()
            if path.name not in {".git", "artifacts", "mlruns", ".dvc"}
        }
        run_system_smoke(tmp_path / "scratch", config=load_config())
        after = {
            path.name
            for path in root.iterdir()
            if path.name not in {".git", "artifacts", "mlruns", ".dvc"}
        }
        assert after == before

    def test_the_tracking_store_does_not_survive_the_check(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "scratch"
        run_system_smoke(directory, config=load_config())
        # The store was a temporary directory removed when the check
        # returned; nothing MLflow-shaped is left behind.
        names = {path.name for path in directory.rglob("*")}
        assert "mlruns" not in names
        assert "meta.yaml" not in names

    def test_a_second_run_in_a_fresh_directory_agrees(
        self, tmp_path: Path, report: SmokeReport
    ) -> None:
        second = run_system_smoke(tmp_path / "again", config=load_config())
        assert second.status is report.status
        assert {c.name for c in second.checks} == {c.name for c in report.checks}
