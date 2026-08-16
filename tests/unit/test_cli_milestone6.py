"""Milestone 6 CLI tests: ``fusion-demo`` and ``fusion-train``.

The CLI must print what was evaluated, must never print a winner, and must
refuse a synthetic dataset in scientific mode.

No test here needs a webcam, a model asset, a display server, a network,
Unity, a public dataset, or participant data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engagevr.__main__ import main
from engagevr.config import FusionConfig, load_config
from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER
from engagevr.schemas.fusion import FusionModality, FusionStrategy


def _run(
    dataset: Path,
    output: Path,
    *extra: str,
    command: str = "fusion-demo",
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
            "--output",
            str(output),
            *extra,
        ]
    )


class TestFusionDemo:
    def test_classification_succeeds_and_writes_every_document(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "classification"
        code = _run(
            m5_dataset,
            output,
            "--target",
            "engagement_class",
            "--strategies",
            "early",
            "uniform-late",
            "quality-late",
            "--no-robustness",
        )
        captured = capsys.readouterr()
        assert code == 0, captured.err
        for name in (
            "manifest.json",
            "fusion_config.json",
            "experts.json",
            "metrics.json",
            "fusion_metrics.json",
            "robustness.json",
            "predictions.parquet",
            "expert_predictions.parquet",
            "fusion_weights.parquet",
            "checksums.json",
        ):
            assert (output / name).exists(), name

    def test_regression_succeeds(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "regression"
        code = _run(
            m5_dataset,
            output,
            "--target",
            "engagement_score",
            "--strategies",
            "uniform-late",
            "--no-robustness",
        )
        captured = capsys.readouterr()
        assert code == 0, captured.err
        assert "regression" in captured.out
        assert "mean_absolute_error" in captured.out

    def test_the_output_reports_the_required_context(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            m5_dataset,
            tmp_path / "context",
            "--strategies",
            "uniform-late",
            "--no-robustness",
        )
        out = capsys.readouterr().out
        assert code == 0
        for line in (
            "Dataset fingerprint:",
            "Split fingerprint:",
            "Target:",
            "Task type:",
            "Group field:",
            "Independent groups:",
            "Folds:",
            "Modality groups:",
            "Strategies:",
            "Expert estimator:",
            "Calibration method:",
            "Calibration placement:",
            "Quality weighting:",
            "Split audit passed:",
            "Robustness scenarios:",
            "Experiment directory:",
            "Scientific evaluation:",
        ):
            assert line in out, line

    def test_the_self_check_banner_and_disclaimer_are_printed(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(
            m5_dataset,
            tmp_path / "banner",
            "--strategies",
            "uniform-late",
            "--no-robustness",
        )
        out = capsys.readouterr().out
        assert SOFTWARE_SELF_CHECK_BANNER in out
        assert "=== SYNTHETIC DATA ===" in out
        assert "a medical, diagnostic, psychological, or clinical conclusion" in out
        assert "Milestone 7" in out

    def test_no_superiority_claim_is_printed(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(
            m5_dataset,
            tmp_path / "claims",
            "--strategies",
            "early",
            "uniform-late",
            "quality-late",
            "--no-robustness",
        )
        out = capsys.readouterr().out.lower()
        for phrase in (
            "best model",
            "best strategy",
            "the best fusion",
            "recommended model",
            "winner",
            "champion model",
            "state of the art",
            "outperform",
        ):
            assert phrase not in out, phrase
        # "best fusion architecture" appears only inside the sentence that
        # denies one can be selected from synthetic data.
        assert out.count("best fusion") == 1
        assert "cannot select a best fusion architecture" in out
        assert "no fusion strategy here is a champion" in out
        assert "none is production-ready" in out

    def test_metrics_are_printed_beside_the_synthetic_disclaimer(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(
            m5_dataset,
            tmp_path / "metrics",
            "--strategies",
            "uniform-late",
            "--no-robustness",
        )
        out = capsys.readouterr().out
        assert "balanced_accuracy" in out
        assert out.index("balanced_accuracy") < out.index(
            "must NEVER be compared with a published result"
        )

    def test_robustness_scenarios_are_reported(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            m5_dataset,
            tmp_path / "robustness",
            "--strategies",
            "uniform-late",
            "--scenarios",
            "all_modalities",
            "missing_rppg",
            "only_task",
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "missing_rppg" in out
        assert "only_task" in out
        assert "coverage=" in out

    def test_a_repeat_run_is_deterministic(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        outputs = [tmp_path / "repeat-a", tmp_path / "repeat-b"]
        for output in outputs:
            assert (
                _run(
                    m5_dataset,
                    output,
                    "--strategies",
                    "uniform-late",
                    "--no-robustness",
                )
                == 0
            )
            capsys.readouterr()
        for name in ("metrics.json", "fusion_metrics.json", "splits.json"):
            assert (outputs[0] / name).read_text() == (outputs[1] / name).read_text()
        first = json.loads((outputs[0] / "manifest.json").read_text())
        second = json.loads((outputs[1] / "manifest.json").read_text())
        assert first["run_id"] == second["run_id"]
        assert first["dataset_fingerprint"] == second["dataset_fingerprint"]


class TestArgumentHandling:
    def test_an_unknown_strategy_is_rejected(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit) as exit_info:
            _run(m5_dataset, tmp_path / "bad", "--strategies", "attention-fusion")
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
                "fusion-demo",
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

    def test_too_many_folds_is_reported_as_an_error(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "fusion-demo",
                "--dataset",
                str(m5_dataset),
                "--folds",
                "200",
                "--strategies",
                "uniform-late",
                "--no-robustness",
                "--output",
                str(tmp_path / "many-folds"),
            ]
        )
        assert code == 1
        assert "cannot support 200 folds" in capsys.readouterr().err

    def test_an_impossible_minimum_modality_count_is_rejected(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            m5_dataset,
            tmp_path / "minimum",
            "--modalities",
            "task",
            "rppg",
            "--minimum-modalities",
            "4",
        )
        assert code == 2
        assert "could ever satisfy it" in capsys.readouterr().err


class TestScientificMode:
    def test_scientific_mode_refuses_a_synthetic_dataset(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            m5_dataset,
            tmp_path / "scientific",
            "--mode",
            "scientific",
            "--strategies",
            "uniform-late",
            "--no-robustness",
            command="fusion-train",
        )
        assert code == 3
        assert "data_source='synthetic'" in capsys.readouterr().err

    def test_fusion_train_defaults_to_the_self_check_mode(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            m5_dataset,
            tmp_path / "self-check",
            "--strategies",
            "uniform-late",
            "--no-robustness",
            command="fusion-train",
        )
        assert code == 0
        assert SOFTWARE_SELF_CHECK_BANNER in capsys.readouterr().out


class TestConfiguration:
    def test_the_shipped_defaults_resolve(self) -> None:
        settings = load_config()
        configuration = settings.fusion.resolve()
        assert configuration.strategies == (
            FusionStrategy.EARLY,
            FusionStrategy.UNIFORM_LATE,
            FusionStrategy.QUALITY_LATE,
        )
        assert configuration.modalities == tuple(FusionModality)
        assert configuration.stacking.enabled is False
        assert configuration.robustness.enabled is True
        assert configuration.quality.base_weights == {}

    def test_quality_is_rejected_as_a_configured_modality(self) -> None:
        with pytest.raises(ValueError, match="not a fusion modality"):
            FusionConfig(modalities=["quality", "task"])

    def test_an_empty_strategy_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one strategy"):
            FusionConfig(enabled_strategies=[])

    def test_a_duplicate_modality_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="contains duplicates"):
            FusionConfig(modalities=["task", "task", "rppg"])

    def test_an_unknown_strategy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not an implemented strategy"):
            FusionConfig(enabled_strategies=["attention"])

    def test_an_out_of_range_quality_value_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            FusionConfig(quality={"minimum_quality": 1.5})  # type: ignore[arg-type]

    def test_requesting_the_stacked_strategy_enables_stacking(self) -> None:
        settings = load_config()
        configuration = settings.fusion.resolve(strategies=["stacked-late"])
        assert configuration.stacking.enabled is True

    def test_dropout_overrides_are_applied(self) -> None:
        settings = load_config()
        configuration = settings.fusion.resolve(
            synthetic_dropout_probability=0.25, synthetic_dropout_seed=7
        )
        assert configuration.robustness.synthetic_dropout_enabled is True
        assert configuration.robustness.synthetic_dropout_probability == 0.25
        assert configuration.robustness.synthetic_dropout_seed == 7
