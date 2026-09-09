"""The Milestone 10 commands, exercised through the public CLI.

Every command must have ``--help``, an explicit output path, a stated
synthetic and scientific status, and a non-zero exit on a validation
failure.  These tests drive ``engagevr.__main__.main`` with an argv list,
which is exactly the path a user's shell takes.

Nothing here starts a server, and every store or output directory is a
``tmp_path`` pytest removes on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engagevr.__main__ import main

MLOPS_COMMANDS = (
    "mlops-demo",
    "model-manifest",
    "drift-check",
    "mlflow-log",
    "repro-manifest",
    "stage-record",
    "system-smoke",
)


class TestRegistration:
    @pytest.mark.parametrize("command", MLOPS_COMMANDS)
    def test_each_command_offers_help(
        self, command: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main([command, "--help"])
        assert exit_info.value.code == 0
        assert capsys.readouterr().out

    def test_the_version_flag_reports_the_package_version(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from engagevr import __version__

        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_the_version_has_one_source_of_truth(self) -> None:
        import tomllib

        from engagevr import __version__

        root = Path(__file__).resolve().parents[2]
        declared = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["version"]
        assert declared == __version__

    def test_the_version_was_not_bumped_to_one_point_zero_for_a_milestone(
        self,
    ) -> None:
        from engagevr import __version__

        # A milestone is not a release. Reaching Milestone 10 does not make
        # this a 1.0 research prototype.
        assert __version__ == "0.1.0"


class TestModelManifestCommand:
    def test_it_writes_one_record_per_estimator(
        self,
        m10_baseline_run: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        output = tmp_path / "versions"
        assert (
            main(
                [
                    "model-manifest",
                    "--run",
                    str(m10_baseline_run),
                    "--output",
                    str(output),
                    "--verify",
                ]
            )
            == 0
        )
        written = sorted(output.glob("*.model-version.json"))
        assert written
        printed = capsys.readouterr().out
        assert "eligible=false" in printed
        assert "champion" in printed  # only inside the refusal sentence
        assert "No version here is production" in printed

    def test_a_missing_run_directory_exits_non_zero(self, tmp_path: Path) -> None:
        assert (
            main(
                [
                    "model-manifest",
                    "--run",
                    str(tmp_path / "absent"),
                    "--output",
                    str(tmp_path / "out"),
                ]
            )
            == 2
        )

    def test_a_tampered_model_exits_non_zero(
        self, m10_baseline_run: Path, tmp_path: Path
    ) -> None:
        import shutil

        copy = tmp_path / "run"
        shutil.copytree(m10_baseline_run, copy)
        target = next((copy / "models").glob("*.joblib"))
        target.write_bytes(b"tampered")
        assert (
            main(
                [
                    "model-manifest",
                    "--run",
                    str(copy),
                    "--output",
                    str(tmp_path / "out"),
                ]
            )
            == 1
        )


class TestDriftCheckCommand:
    def test_it_writes_a_report_and_states_the_scientific_status(
        self,
        m5_dataset: Path,
        m10_shifted_dataset: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        output = tmp_path / "drift.json"
        assert (
            main(
                [
                    "drift-check",
                    "--reference",
                    str(m5_dataset),
                    "--current",
                    str(m10_shifted_dataset),
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["scientific_evaluation_eligible"] is False
        assert document["report_kind"] == "feature_distribution_shift"
        printed = capsys.readouterr().out
        assert "Scientific evaluation:  false" in printed
        assert "not concept drift" in printed.lower()

    def test_a_missing_dataset_exits_non_zero(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        assert (
            main(
                [
                    "drift-check",
                    "--reference",
                    str(m5_dataset),
                    "--current",
                    str(tmp_path / "absent.parquet"),
                    "--output",
                    str(tmp_path / "drift.json"),
                ]
            )
            == 2
        )

    def test_fail_on_shift_is_opt_in(
        self, tmp_path: Path, m5_dataset: Path, m10_shifted_dataset: Path
    ) -> None:
        # Without the flag a threshold crossing is reported, not fatal:
        # a crossing is an invitation to look, not a verdict.
        assert (
            main(
                [
                    "drift-check",
                    "--reference",
                    str(m5_dataset),
                    "--current",
                    str(m10_shifted_dataset),
                    "--output",
                    str(tmp_path / "a.json"),
                ]
            )
            == 0
        )

    def test_comparing_a_dataset_with_itself_never_fails_the_gate(
        self, tmp_path: Path, m5_dataset: Path
    ) -> None:
        assert (
            main(
                [
                    "drift-check",
                    "--reference",
                    str(m5_dataset),
                    "--current",
                    str(m5_dataset),
                    "--output",
                    str(tmp_path / "b.json"),
                    "--fail-on-shift",
                ]
            )
            == 0
        )


class TestMlflowLogCommand:
    def test_it_logs_a_run_and_writes_a_summary(
        self,
        m10_baseline_run: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        store = tmp_path / "store"
        store.mkdir()
        output = tmp_path / "summaries.json"
        assert (
            main(
                [
                    "mlflow-log",
                    "--run",
                    str(m10_baseline_run),
                    "--tracking-uri",
                    store.resolve().as_uri(),
                    "--experiment",
                    "engagevr-cli-test",
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        document = json.loads(output.read_text(encoding="utf-8"))
        assert len(document["runs"]) == 1
        run = document["runs"][0]
        assert run["is_synthetic"] is True
        assert run["scientific_evaluation_eligible"] is False
        assert run["registered_model"] is None
        printed = capsys.readouterr().out
        assert "registered model:   none" in printed
        assert "not validated, not approved" in printed

    def test_a_missing_run_directory_exits_non_zero(self, tmp_path: Path) -> None:
        assert main(["mlflow-log", "--run", str(tmp_path / "absent")]) == 2


class TestReproManifestCommand:
    """The manifest is assembled from the deterministic stage records.

    That is a deliberate behaviour change from the first Milestone 10
    implementation, which walked the filesystem and emitted a manifest
    full of "unavailable" entries for a pipeline that had never run. A
    manifest describing a pipeline nobody executed is not useful, and the
    stage records are the only place where the byte-stable-versus-
    timestamped classification has been made. So the command now refuses,
    with a message that says what to do about it.
    """

    def _run_stage(self, root: Path, stage: str) -> int:
        return main(["stage-record", "--stage", stage, "--pipeline-root", str(root)])

    def _populate(self, root: Path, source: Path) -> None:
        """Copy a completed baseline run in and record the stage."""
        import shutil

        from engagevr.mlops.pipeline import default_layout, load_parameters

        layout = default_layout(root, load_parameters().target)
        layout.experiments.mkdir(parents=True, exist_ok=True)
        if not layout.baseline_run.exists():
            shutil.copytree(source, layout.baseline_run)
        assert self._run_stage(root, "baseline") == 0

    def test_a_pipeline_that_never_ran_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            main(
                [
                    "repro-manifest",
                    "--pipeline-root",
                    str(tmp_path / "pipeline"),
                    "--output",
                    str(tmp_path / "repro.json"),
                ]
            )
            == 1
        )
        assert "Run the pipeline" in capsys.readouterr().err

    def test_a_recorded_pipeline_produces_a_manifest(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        root = tmp_path / "pipeline"
        self._populate(root, m10_baseline_run)
        output = tmp_path / "repro.json"
        assert (
            main(
                [
                    "repro-manifest",
                    "--pipeline-root",
                    str(root),
                    "--output",
                    str(output),
                    "--stages",
                    "baseline",
                ]
            )
            == 0
        )
        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["scientific_evaluation_eligible"] is False
        assert document["is_synthetic"] is True
        assert "created_at_utc" not in document
        stage = next(s for s in document["stages"] if s["name"] == "baseline")
        assert stage["deterministic_artifacts"]
        assert stage["volatile_artifacts"]

    def test_the_manifest_gets_an_execution_sidecar(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        root = tmp_path / "pipeline"
        self._populate(root, m10_baseline_run)
        output = tmp_path / "repro.json"
        assert (
            main(
                [
                    "repro-manifest",
                    "--pipeline-root",
                    str(root),
                    "--output",
                    str(output),
                    "--stages",
                    "baseline",
                ]
            )
            == 0
        )
        sidecar = output.with_name("repro.execution.json")
        assert sidecar.is_file()
        assert json.loads(sidecar.read_text(encoding="utf-8"))["created_at_utc"]

    def test_comparing_two_identical_manifests_succeeds(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        root = tmp_path / "pipeline"
        self._populate(root, m10_baseline_run)
        first = tmp_path / "first.json"
        second = tmp_path / "second.json"
        common = [
            "repro-manifest",
            "--pipeline-root",
            str(root),
            "--stages",
            "baseline",
        ]
        assert main([*common, "--output", str(first)]) == 0
        assert main([*common, "--output", str(second), "--compare", str(first)]) == 0

    def test_comparing_against_a_changed_pipeline_exits_non_zero(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        root = tmp_path / "pipeline"
        self._populate(root, m10_baseline_run)
        first = tmp_path / "first.json"
        second = tmp_path / "second.json"
        common = [
            "repro-manifest",
            "--pipeline-root",
            str(root),
            "--stages",
            "baseline",
        ]
        assert main([*common, "--output", str(first)]) == 0

        from engagevr.mlops.pipeline import default_layout, load_parameters

        layout = default_layout(root, load_parameters().target)
        (layout.baseline_run / "metrics.json").write_text("{}", encoding="utf-8")
        assert self._run_stage(root, "baseline") == 0
        assert main([*common, "--output", str(second), "--compare", str(first)]) == 1


class TestStageRecordCommand:
    def test_it_writes_a_record_and_a_sidecar(
        self, tmp_path: Path, m10_baseline_run: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import shutil

        from engagevr.mlops.pipeline import default_layout, load_parameters

        root = tmp_path / "pipeline"
        layout = default_layout(root, load_parameters().target)
        layout.experiments.mkdir(parents=True)
        shutil.copytree(m10_baseline_run, layout.baseline_run)
        assert (
            main(["stage-record", "--stage", "baseline", "--pipeline-root", str(root)])
            == 0
        )
        record = layout.stage_record("baseline")
        assert record.is_file()
        assert record.with_name("baseline.execution.json").is_file()
        printed = capsys.readouterr().out
        assert "run_id:" in printed
        assert "Scientific evaluation:  false" in printed
        assert "not checksummed" in printed

    def test_a_stage_that_never_ran_exits_non_zero(self, tmp_path: Path) -> None:
        assert (
            main(
                [
                    "stage-record",
                    "--stage",
                    "baseline",
                    "--pipeline-root",
                    str(tmp_path / "pipeline"),
                ]
            )
            == 1
        )

    def test_a_stage_with_no_record_of_its_own_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            main(
                [
                    "stage-record",
                    "--stage",
                    "drift",
                    "--pipeline-root",
                    str(tmp_path / "pipeline"),
                ]
            )
            == 2
        )
        assert "produces its own deterministic document" in capsys.readouterr().err


class TestSystemSmokeCommand:
    def test_the_json_form_carries_the_banner_and_the_status(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER

        status = main(["system-smoke", "--output", str(tmp_path / "smoke"), "--json"])
        printed = capsys.readouterr().out
        document = json.loads(printed)
        assert document["banner"] == SOFTWARE_SELF_CHECK_BANNER
        assert document["scientific_evaluation_eligible"] is False
        assert document["is_synthetic"] is True
        assert status == (0 if document["status"] == "passed" else 1)

    def test_the_report_is_written_to_the_named_directory(self, tmp_path: Path) -> None:
        directory = tmp_path / "smoke"
        main(["system-smoke", "--output", str(directory), "--json"])
        assert (directory / "smoke_report.json").is_file()

    def test_skipping_tracking_is_recorded_as_skipped_not_passed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "system-smoke",
                "--output",
                str(tmp_path / "smoke"),
                "--no-mlflow",
                "--json",
            ]
        )
        document = json.loads(capsys.readouterr().out)
        tracking = next(
            check
            for check in document["checks"]
            if check["name"] == "mlflow_tracking_local"
        )
        assert tracking["status"] == "skipped"
        assert "Skipping is not passing" in tracking["skip_reason"]
