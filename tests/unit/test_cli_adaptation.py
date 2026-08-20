"""The ``adaptation-demo`` command.

The command must run with no server, no socket, no Unity, and no dataset,
and its output must never read as a claim that an adaptation helped
anyone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engagevr.__main__ import main
from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER


def _run(tmp_path: Path, *extra: str) -> int:
    return main(
        [
            "adaptation-demo",
            "--output",
            str(tmp_path / "run"),
            "--scenario",
            "persistent-decrease",
            *extra,
        ]
    )


class TestInvocation:
    def test_the_command_is_registered(self, tmp_path: Path) -> None:
        assert _run(tmp_path) == 0

    def test_it_writes_its_artifacts(self, tmp_path: Path) -> None:
        _run(tmp_path)
        for name in (
            "adaptation_policy_config.json",
            "adaptation_trace.parquet",
            "adaptation_summary.json",
            "scenarios.json",
            "checksums.json",
        ):
            assert (tmp_path / "run" / name).exists()

    def test_the_whole_suite_runs(self, tmp_path: Path) -> None:
        assert main(["adaptation-demo", "--output", str(tmp_path / "all")]) == 0

    def test_an_unknown_scenario_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(tmp_path, "--scenario", "no-such-thing") == 2
        assert "unknown scenario" in capsys.readouterr().err

    def test_listing_scenarios_prints_every_expectation(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["adaptation-demo", "--list-scenarios"]) == 0
        out = capsys.readouterr().out
        assert "persistent-decrease" in out
        assert "expectation:" in out


class TestExperimenterControls:
    def test_the_static_condition_proposes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(tmp_path, "--experiment-mode", "static") == 0
        out = capsys.readouterr().out
        assert "Experiment mode:               static" in out
        assert "Adaptation proposals:          0" in out

    def test_the_lock_proposes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(tmp_path, "--disable-adaptation") == 0
        out = capsys.readouterr().out
        assert "Adaptation enabled:            False" in out
        assert "Adaptation proposals:          0" in out

    def test_the_adaptive_condition_proposes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(tmp_path) == 0
        out = capsys.readouterr().out
        assert "Experiment mode:               adaptive" in out
        assert "Adaptation proposals:          1" in out


class TestDispatchIsRefused:
    def test_dispatch_is_not_the_default(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(tmp_path)
        out = capsys.readouterr().out
        assert "Commands dispatched:           0" in out
        assert "Acknowledgements recorded:     0" in out

    def test_asking_for_dispatch_produces_a_stated_refusal(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(tmp_path, "--dispatch") == 2
        err = capsys.readouterr().err
        assert "not implemented" in err
        assert "unvalidated rule in control" in err


class TestOutputLanguage:
    def test_the_self_check_banner_is_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(tmp_path)
        out = capsys.readouterr().out
        assert out.count(SOFTWARE_SELF_CHECK_BANNER) >= 2

    def test_the_demonstration_rule_is_stated(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(tmp_path)
        out = capsys.readouterr().out
        assert "ENGINEERING DEMONSTRATION RULE" in out
        assert "not psychologically validated" in out

    def test_the_milestone_boundary_is_stated(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(tmp_path)
        out = capsys.readouterr().out
        assert "blocked Milestone 7 gate always holds" in out
        assert "no override" in out

    def test_no_benefit_claim_is_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(tmp_path)
        out = capsys.readouterr().out.lower()
        for claim in (
            "improved engagement",
            "reduced cognitive load",
            "better policy",
            "optimal adaptation",
            "champion",
        ):
            assert claim not in out
        # "therapeutic" appears only inside its own denial.
        assert "not therapeutic" in out
        assert out.count("therapeutic") == 1

    def test_the_scientific_eligibility_flag_is_printed_false(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(tmp_path)
        assert "scientific_evaluation_eligible=false" in capsys.readouterr().out

    def test_the_naive_comparison_denies_the_benefit_reading(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["adaptation-demo", "--output", str(tmp_path / "all")])
        out = capsys.readouterr().out
        assert "NOT a benefit claim" in out
        assert "Neither controller has been shown to help anyone" in out

    def test_the_comparison_can_be_switched_off(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(tmp_path, "--no-naive-comparison")
        assert "NOT a benefit claim" not in capsys.readouterr().out


class TestDeterminism:
    def test_two_invocations_agree(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from engagevr.training.artifacts import sha256_file

        main(["adaptation-demo", "--output", str(tmp_path / "a")])
        main(["adaptation-demo", "--output", str(tmp_path / "b")])
        capsys.readouterr()
        assert sha256_file(tmp_path / "a" / "adaptation_trace.parquet") == sha256_file(
            tmp_path / "b" / "adaptation_trace.parquet"
        )


class TestExistingCommandsStillWork:
    def test_the_help_lists_the_new_command(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            main(["--help"])
        assert "adaptation-demo" in capsys.readouterr().out

    def test_an_unknown_command_still_prints_help(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([]) == 1
        assert "adaptation-demo" in capsys.readouterr().out
