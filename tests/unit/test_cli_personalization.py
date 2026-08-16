"""Milestone 6 CLI: ``personalization-demo`` and ``personalization-train``.

The CLI must print what was evaluated, must report the population and the
personalized result separately, must never claim personalization helped,
and must refuse a synthetic dataset in scientific mode.

No test here needs a webcam, a model asset, a display server, a network,
Unity, a public dataset, or participant data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engagevr.__main__ import main
from engagevr.config import PersonalizationConfig, load_config
from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER
from engagevr.schemas.fusion import FusionModality
from engagevr.schemas.personalization import PersonalizationMethod


def _run(
    dataset: Path,
    output: Path,
    *extra: str,
    command: str = "personalization-demo",
) -> int:
    return main(
        [
            command,
            "--dataset",
            str(dataset),
            "--folds",
            "3",
            "--seed",
            "42",
            "--calibration-windows",
            "3",
            "--minimum-calibration-windows",
            "2",
            "--output",
            str(output),
            *extra,
        ]
    )


class TestPersonalizationDemo:
    def test_classification_succeeds_and_writes_every_document(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "classification"
        code = _run(m5_dataset, output, "--target", "engagement_class")
        captured = capsys.readouterr()
        assert code == 0, captured.err
        for name in (
            "manifest.json",
            "personalization_config.json",
            "personalization.json",
            "personal_baselines.json",
            "metrics.json",
            "calibration.json",
            "splits.json",
            "predictions.parquet",
            "checksums.json",
        ):
            assert (output / name).exists(), name

    def test_regression_succeeds(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(m5_dataset, tmp_path / "regression", "--target", "engagement_score")
        captured = capsys.readouterr()
        assert code == 0, captured.err
        assert "regression" in captured.out
        assert "mean_absolute_error" in captured.out

    def test_the_output_reports_the_required_context(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(m5_dataset, tmp_path / "context")
        out = capsys.readouterr().out
        assert code == 0
        for line in (
            "Dataset fingerprint:",
            "Target:",
            "Task type:",
            "Population training subjects:",
            "Held-out subjects:",
            "Calibration windows:",
            "Evaluation windows:",
            "Personalization method:",
            "Cold-start subject-folds:",
            "Population metrics:",
            "Personalized metrics:",
            "scientific_evaluation_eligible=False",
        ):
            assert line in out, line

    def test_the_self_check_banner_and_disclaimer_are_printed(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "banner")
        out = capsys.readouterr().out
        assert SOFTWARE_SELF_CHECK_BANNER in out
        assert "=== SYNTHETIC DATA ===" in out
        assert "a medical, diagnostic, psychological, or clinical conclusion" in out

    def test_the_two_reports_are_printed_separately(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "separate")
        out = capsys.readouterr().out
        assert "Population baseline (identical evaluation windows):" in out
        assert "Personalized (identical evaluation windows):" in out
        assert out.index("Population baseline") < out.index("Personalized (")
        assert "/results/0  (model_name='population')" in out
        assert "/results/1  (model_name='personalized')" in out

    def test_no_personalization_benefit_is_claimed(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "claims")
        out = capsys.readouterr().out.lower()
        for phrase in (
            "personalization improves",
            "personalisation improves",
            "personalized model is better",
            "personalized baselines outperform",
            "best model",
            "winner",
            "champion model",
            "outperform",
        ):
            assert phrase not in out, phrase
        assert "not evidence of a personalization benefit" in out

    def test_the_deferred_milestone_seven_note_is_printed(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "deferred")
        out = capsys.readouterr().out
        assert "Milestone 7" in out
        assert "abstention" in out
        assert "NOT uncertainty calibration" in out

    def test_cold_start_mode_is_reported(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "personalization-demo",
                "--dataset",
                str(m5_dataset),
                "--folds",
                "3",
                "--calibration-windows",
                "0",
                "--method",
                "personal-baseline",
                "--output",
                str(tmp_path / "cold"),
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "Personalization coverage:      0.000" in out
        assert "Cold-start subject-folds:" in out

    def test_a_repeat_run_is_deterministic(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        outputs = [tmp_path / "repeat-a", tmp_path / "repeat-b"]
        for output in outputs:
            assert _run(m5_dataset, output) == 0
            capsys.readouterr()
        for name in ("personalization.json", "metrics.json", "splits.json"):
            assert (outputs[0] / name).read_text() == (outputs[1] / name).read_text()
        first = json.loads((outputs[0] / "manifest.json").read_text())
        second = json.loads((outputs[1] / "manifest.json").read_text())
        assert first["run_id"] == second["run_id"]


class TestArgumentHandling:
    def test_an_unknown_method_is_rejected(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit) as exit_info:
            _run(m5_dataset, tmp_path / "bad", "--method", "deep-personal-net")
        assert exit_info.value.code == 2

    def test_cold_start_is_not_offered_as_a_method(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit) as exit_info:
            _run(m5_dataset, tmp_path / "bad", "--method", "cold_start")
        assert exit_info.value.code == 2

    def test_quality_is_not_accepted_as_a_modality(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit) as exit_info:
            _run(m5_dataset, tmp_path / "bad", "--modalities", "quality", "task")
        assert exit_info.value.code == 2

    def test_a_single_modality_is_rejected(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(m5_dataset, tmp_path / "single", "--modalities", "task")
        assert code == 2
        assert "at least two" in capsys.readouterr().err

    def test_a_missing_dataset_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(tmp_path / "absent.parquet", tmp_path / "out")
        assert code == 2
        assert "does not exist" in capsys.readouterr().err

    def test_a_single_fold_is_rejected(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "personalization-demo",
                "--dataset",
                str(m5_dataset),
                "--folds",
                "1",
                "--output",
                str(tmp_path / "one-fold"),
            ]
        )
        assert code == 2
        assert "at least 2" in capsys.readouterr().err

    def test_a_negative_calibration_window_count_is_rejected(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "personalization-demo",
                "--dataset",
                str(m5_dataset),
                "--calibration-windows",
                "-1",
                "--output",
                str(tmp_path / "negative"),
            ]
        )
        assert code == 2
        assert "cannot be negative" in capsys.readouterr().err

    def test_an_impossible_minimum_is_reported(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "personalization-demo",
                "--dataset",
                str(m5_dataset),
                "--calibration-windows",
                "2",
                "--minimum-calibration-windows",
                "9",
                "--output",
                str(tmp_path / "impossible"),
            ]
        )
        assert code == 2
        assert "could never be applied" in capsys.readouterr().err

    def test_too_many_folds_is_reported_as_an_error(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "personalization-demo",
                "--dataset",
                str(m5_dataset),
                "--folds",
                "200",
                "--output",
                str(tmp_path / "many-folds"),
            ]
        )
        assert code == 1
        assert "cannot support 200 folds" in capsys.readouterr().err


class TestScientificMode:
    def test_scientific_mode_refuses_a_synthetic_dataset(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            m5_dataset,
            tmp_path / "scientific",
            "--mode",
            "scientific",
            command="personalization-train",
        )
        assert code == 3
        assert "data_source='synthetic'" in capsys.readouterr().err

    def test_train_defaults_to_the_self_check_mode(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            m5_dataset, tmp_path / "self-check", command="personalization-train"
        )
        assert code == 0
        assert SOFTWARE_SELF_CHECK_BANNER in capsys.readouterr().out


class TestConfiguration:
    def test_the_shipped_defaults_resolve(self) -> None:
        settings = load_config()
        configuration = settings.personalization.resolve()
        assert (
            configuration.method
            is PersonalizationMethod.PERSONAL_BASELINE_AND_CORRECTION
        )
        assert configuration.modalities == tuple(FusionModality)
        assert configuration.calibration_windows == 5
        assert configuration.include_modality_quality is False

    def test_cold_start_is_rejected_as_a_configured_method(self) -> None:
        with pytest.raises(ValueError, match="not a requestable method"):
            PersonalizationConfig(method="cold_start")

    def test_an_unknown_method_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a requestable method"):
            PersonalizationConfig(method="deep_personal_net")

    def test_quality_is_rejected_as_a_configured_modality(self) -> None:
        with pytest.raises(ValueError, match="not a fusion modality"):
            PersonalizationConfig(modalities=["quality", "task"])

    def test_a_duplicate_modality_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="contains duplicates"):
            PersonalizationConfig(modalities=["task", "task", "rppg"])

    def test_a_single_modality_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two groups"):
            PersonalizationConfig(modalities=["task"])

    def test_overrides_are_applied(self) -> None:
        settings = load_config()
        configuration = settings.personalization.resolve(
            method="few-shot-correction",
            calibration_windows=8,
            minimum_evaluation_windows=4,
        )
        assert configuration.method is PersonalizationMethod.FEW_SHOT_CORRECTION
        assert configuration.calibration_windows == 8
        assert configuration.minimum_evaluation_windows == 4
