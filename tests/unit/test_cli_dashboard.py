"""The ``dashboard``, ``dashboard-check``, and ``dashboard-sessions`` commands.

``dashboard-check`` and ``dashboard-sessions`` do everything interesting
without a server, so they are the commands these tests exercise.  For
``dashboard`` itself only the pure parts are tested — the argv it builds,
the environment it passes, the paths it resolves — because a unit test
that started a long-running web server would be testing Streamlit rather
than this repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engagevr.__main__ import main
from engagevr.dashboard.launch import (
    DEVELOPMENT_COMMAND,
    DashboardLaunchError,
    app_path,
    build_command,
    build_environment,
    describe,
)
from engagevr.schemas.dashboard import SYNTHETIC_BANNER
from tests.unit import dashboard_fixtures as fx
from tests.unit import session_fixtures as sfx


class TestRegistration:
    def test_every_command_appears_in_the_help(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        assert "dashboard" in out
        assert "dashboard-check" in out
        assert "dashboard-sessions" in out

    def test_the_sessions_help_says_it_writes_nothing(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            main(["dashboard-sessions", "--help"])
        # argparse rewraps the description, so assert on phrases that
        # survive a line break rather than on a two-word phrase.
        out = capsys.readouterr().out.lower()
        assert "writes nothing" in out
        assert "sends nothing" in out
        assert "printed, not saved" in out

    def test_the_help_says_the_dashboard_is_read_only(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            main(["dashboard", "--help"])
        # argparse rewraps the description, so assert on phrases that
        # survive a line break rather than on a hyphenated word.
        out = capsys.readouterr().out.lower()
        assert "changes nothing" in out
        assert "no dispatch" in out

    def test_the_check_help_says_it_starts_no_server(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            main(["dashboard-check", "--help"])
        assert "no server" in capsys.readouterr().out.lower()

    def test_the_existing_commands_still_work(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([]) == 1
        out = capsys.readouterr().out
        for command in ("baseline-demo", "fusion-demo", "adaptation-demo"):
            assert command in out


class TestDashboardCheck:
    def test_an_empty_root_is_reported_not_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["dashboard-check", "--artifact-root", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "Runs found:     0" in out

    def test_a_missing_root_is_reported_not_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "nothing"
        assert main(["dashboard-check", "--artifact-root", str(missing)]) == 0
        assert "Root exists:    False" in capsys.readouterr().out

    def test_every_family_is_listed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fx.make_baseline_run(tmp_path)
        fx.make_fusion_run(tmp_path)
        fx.make_personalization_run(tmp_path)
        fx.make_uncertainty_run(tmp_path)
        fx.make_adaptation_run(tmp_path)
        assert main(["dashboard-check", "--artifact-root", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        for family in (
            "baseline",
            "fusion",
            "personalization",
            "uncertainty",
            "adaptation",
        ):
            assert family in out

    def test_the_read_only_boundary_is_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["dashboard-check", "--artifact-root", str(tmp_path)])
        out = capsys.readouterr().out
        assert "READ-ONLY" in out
        assert "never opens a model file" in out

    def test_the_synthetic_banner_is_printed_for_synthetic_runs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fx.make_baseline_run(tmp_path)
        main(["dashboard-check", "--artifact-root", str(tmp_path)])
        out = capsys.readouterr().out
        assert "SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION" in out

    def test_the_eligible_count_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fx.make_baseline_run(tmp_path)
        main(["dashboard-check", "--artifact-root", str(tmp_path)])
        assert "Scientifically eligible runs:   0/1" in capsys.readouterr().out

    def test_a_corrupt_run_is_surfaced(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        directory = fx.make_baseline_run(tmp_path)
        fx.corrupt(directory / "manifest.json")
        assert main(["dashboard-check", "--artifact-root", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "corrupt" in out
        assert "Runs needing attention" in out

    def test_a_checksum_mismatch_is_surfaced(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        directory = fx.make_baseline_run(tmp_path)
        (directory / "metrics.json").write_text("{}", encoding="utf-8")
        main(["dashboard-check", "--artifact-root", str(tmp_path)])
        assert "mismatched" in capsys.readouterr().out

    def test_verification_can_be_switched_off(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fx.make_baseline_run(tmp_path)
        main(
            [
                "dashboard-check",
                "--artifact-root",
                str(tmp_path),
                "--no-validate-checksums",
            ]
        )
        assert "not_checked" in capsys.readouterr().out

    def test_the_json_form_is_machine_readable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fx.make_adaptation_run(tmp_path)
        assert (
            main(["dashboard-check", "--artifact-root", str(tmp_path), "--json"]) == 0
        )
        document = json.loads(capsys.readouterr().out)
        assert document["run_count"] == 1
        assert document["read_only"] is True
        assert document["runs"][0]["family"] == "adaptation"
        assert document["runs"][0]["is_synthetic"] is True
        assert document["runs"][0]["scientific_evaluation_eligible"] is False

    def test_the_scan_modifies_nothing(self, tmp_path: Path) -> None:
        directory = fx.make_baseline_run(tmp_path)
        before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        main(["dashboard-check", "--artifact-root", str(tmp_path)])
        after = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        assert before == after

    def test_the_default_root_comes_from_configuration(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["dashboard-check"]) == 0
        assert "artifacts/experiments" in capsys.readouterr().out


class TestLaunchCommand:
    def test_the_app_script_resolves(self) -> None:
        assert app_path().is_file()
        assert app_path().name == "app.py"

    def test_the_command_runs_the_app_script(self) -> None:
        command = build_command()
        assert "streamlit" in command
        assert "run" in command
        assert str(app_path()) in command

    def test_it_binds_to_loopback_by_default(self) -> None:
        command = build_command()
        index = command.index("--server.address")
        assert command[index + 1] == "127.0.0.1"

    def test_it_is_headless_by_default(self) -> None:
        command = build_command()
        index = command.index("--server.headless")
        assert command[index + 1] == "true"

    def test_a_browser_can_be_requested_explicitly(self) -> None:
        command = build_command(headless=False)
        index = command.index("--server.headless")
        assert command[index + 1] == "false"

    def test_usage_statistics_are_switched_off(self) -> None:
        command = build_command()
        index = command.index("--browser.gatherUsageStats")
        assert command[index + 1] == "false"

    def test_the_port_is_honoured(self) -> None:
        command = build_command(port=9123)
        assert "9123" in command

    def test_an_impossible_port_is_refused(self) -> None:
        with pytest.raises(DashboardLaunchError, match="port must be"):
            build_command(port=0)

    def test_an_empty_address_is_refused(self) -> None:
        with pytest.raises(DashboardLaunchError, match="must not be empty"):
            build_command(address="  ")

    def test_the_artifact_root_travels_in_the_environment(self) -> None:
        environment = build_environment("artifacts/experiments", base={})
        assert environment["ENGAGEVR_DASHBOARD_ARTIFACT_ROOT"] == (
            "artifacts/experiments"
        )

    def test_no_root_leaves_the_environment_alone(self) -> None:
        assert build_environment(None, base={"A": "B"}) == {"A": "B"}

    def test_the_description_names_the_development_command(self) -> None:
        text = describe(build_command(), "artifacts/experiments")
        assert DEVELOPMENT_COMMAND in text
        assert "ENGAGEVR_DASHBOARD_ARTIFACT_ROOT=artifacts/experiments" in text


class TestDashboardDryRun:
    def test_print_command_starts_no_server(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "dashboard",
                "--artifact-root",
                str(tmp_path),
                "--print-command",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "streamlit" in out
        assert str(tmp_path) in out

    def test_an_impossible_port_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "dashboard",
                "--artifact-root",
                str(tmp_path),
                "--port",
                "0",
                "--print-command",
            ]
        )
        assert code == 2
        assert "port must be" in capsys.readouterr().err

    def test_a_missing_root_warns_but_still_starts(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "dashboard",
                "--artifact-root",
                str(tmp_path / "absent"),
                "--print-command",
            ]
        )
        assert code == 0
        assert "does not exist yet" in capsys.readouterr().err


class TestDashboardSessions:
    """The server-free session path: listing and report building."""

    @pytest.fixture
    def session_root(self, tmp_path: Path) -> Path:
        root = tmp_path / "sessions"
        sfx.write_session(root, "synthetic-completed", with_adaptation=True)
        sfx.write_session(root, "synthetic-active", with_summary=False, completed=False)
        return root

    def test_listing_reports_every_session(
        self, session_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["dashboard-sessions", "--session-root", str(session_root)]) == 0
        out = capsys.readouterr().out
        assert "synthetic-completed" in out
        assert "synthetic-active" in out
        assert "Sessions found: 2" in out

    def test_listing_states_the_read_only_boundary(
        self, session_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["dashboard-sessions", "--session-root", str(session_root)])
        out = capsys.readouterr().out
        assert "READ-ONLY" in out
        assert SYNTHETIC_BANNER in out

    def test_listing_marks_every_session_ineligible(
        self, session_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["dashboard-sessions", "--session-root", str(session_root)])
        out = capsys.readouterr().out
        assert "True" not in out.split("eligible")[-1].split("SOFTWARE")[0]

    def test_an_absent_root_is_reported_not_raised(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["dashboard-sessions", "--session-root", str(tmp_path / "absent")])
        assert code == 0
        assert "Root exists:    False" in capsys.readouterr().out

    def test_a_report_is_printed_as_json(
        self, session_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "dashboard-sessions",
                "--session-root",
                str(session_root),
                "--session",
                "synthetic-completed",
            ]
        )
        assert code == 0
        document = json.loads(capsys.readouterr().out)
        assert document["session_id"] == "synthetic-completed"
        assert document["is_synthetic"] is True
        assert document["scientific_evaluation_eligible"] is False
        assert document["synthetic_banner"] == SYNTHETIC_BANNER

    def test_a_report_is_printed_as_markdown(
        self, session_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "dashboard-sessions",
                "--session-root",
                str(session_root),
                "--session",
                "synthetic-completed",
                "--format",
                "markdown",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert out.startswith("# EngageVR session report")
        assert SYNTHETIC_BANNER in out

    def test_two_reports_are_byte_identical(
        self, session_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        arguments = [
            "dashboard-sessions",
            "--session-root",
            str(session_root),
            "--session",
            "synthetic-completed",
        ]
        main(arguments)
        first = capsys.readouterr().out
        main(arguments)
        second = capsys.readouterr().out
        assert first == second

    def test_the_mode_is_recorded_in_the_report(
        self, session_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "dashboard-sessions",
                "--session-root",
                str(session_root),
                "--session",
                "synthetic-completed",
                "--mode",
                "live",
            ]
        )
        assert json.loads(capsys.readouterr().out)["source_mode"] == "live"

    def test_an_unknown_session_exits_two(
        self, session_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "dashboard-sessions",
                "--session-root",
                str(session_root),
                "--session",
                "synthetic-absent",
            ]
        )
        assert code == 2
        assert "no session directory" in capsys.readouterr().err

    def test_an_active_session_reports_without_a_summary(
        self, session_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "dashboard-sessions",
                "--session-root",
                str(session_root),
                "--session",
                "synthetic-active",
            ]
        )
        assert code == 0
        document = json.loads(capsys.readouterr().out)
        assert document["status"] == "active_or_incomplete"
        assert document["dropped_message_count"] is None

    def test_reporting_modifies_no_recording(self, session_root: Path) -> None:
        directory = session_root / "synthetic-completed"
        before = sfx.directory_digests(directory)
        main(
            [
                "dashboard-sessions",
                "--session-root",
                str(session_root),
                "--session",
                "synthetic-completed",
            ]
        )
        assert sfx.directory_digests(directory) == before

    def test_reporting_writes_no_file(self, session_root: Path) -> None:
        before = sorted(path.name for path in session_root.iterdir())
        main(["dashboard-sessions", "--session-root", str(session_root)])
        assert sorted(path.name for path in session_root.iterdir()) == before
