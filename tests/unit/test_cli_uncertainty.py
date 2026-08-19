"""Milestone 7 CLI: ``uncertainty-demo`` and ``uncertainty-train``.

The CLI must print what was evaluated, must report coverage together with
the accepted-set result, must keep confidence separate from signal quality
and from ensemble disagreement, must never claim safety or validation, and
must refuse a synthetic dataset in scientific mode.

No test here needs a webcam, a model asset, a display server, a network,
Unity, a public dataset, or participant data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engagevr.__main__ import main
from engagevr.config import UncertaintyConfig, load_config
from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER

#: Phrases the printed output must never contain.
FORBIDDEN_OUTPUT: tuple[str, ...] = (
    "safe confidence",
    "trusted prediction",
    "optimal threshold",
    "best uncertainty method",
    "validated abstention",
    "champion",
    "clinically validated",
    "diagnostic accuracy",
    "proven to measure",
)


def _run(
    dataset: Path,
    output: Path,
    *extra: str,
    command: str = "uncertainty-demo",
    target: str = "engagement_class",
) -> int:
    return main(
        [
            command,
            "--dataset",
            str(dataset),
            "--target",
            target,
            "--folds",
            "3",
            "--seed",
            "42",
            "--output",
            str(output),
            *extra,
        ]
    )


class TestClassificationDemo:
    def test_the_demo_succeeds_on_a_synthetic_dataset(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(m5_dataset, tmp_path / "cls") == 0
        assert capsys.readouterr().out

    def test_the_self_check_banner_is_printed(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "banner")
        out = capsys.readouterr().out
        assert SOFTWARE_SELF_CHECK_BANNER in out
        assert "=== SYNTHETIC DATA ===" in out

    def test_the_required_fields_are_printed(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "fields")
        out = capsys.readouterr().out
        for label in (
            "Dataset fingerprint:",
            "Split fingerprint:",
            "Target:",
            "Task type:",
            "Source prediction model:",
            "Probability calibration:",
            "Confidence definition:",
            "Population threshold:",
            "Threshold source:",
            "Personalized thresholds:",
            "Calibration groups per fold:",
            "Outer folds:",
            "Selective coverage:",
            "Abstained:",
            "Prediction unavailable:",
            "Coverage curve:",
            "Adaptation gate:",
        ):
            assert label in out, label

    def test_the_scientific_eligibility_flag_is_printed_as_false(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "eligible")
        assert "scientific_evaluation_eligible=false" in capsys.readouterr().out

    def test_the_permanent_disclaimer_is_printed(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "disclaimer")
        out = capsys.readouterr().out
        assert "Nothing here is a medical, diagnostic, psychological, or" in out
        assert "not evidence of real-world calibration" in out

    def test_the_population_threshold_is_labelled_an_engineering_default(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "engineering")
        out = capsys.readouterr().out
        assert "ENGINEERING DEFAULT, not an optimum" in out

    def test_the_output_never_calls_disagreement_uncertainty(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "disagreement")
        out = capsys.readouterr().out.lower()
        assert "disagreement uncertainty" not in out
        assert "uncertain disagreement" not in out
        # It IS named, and named as a separate thing.
        assert "ensemble disagreement are four different things" in out

    def test_the_output_makes_no_forbidden_claim(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "claims")
        out = capsys.readouterr().out.lower()
        for claim in FORBIDDEN_OUTPUT:
            assert claim not in out, claim

    def test_the_output_denies_that_the_gate_chooses_an_adaptation(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "gate-note")
        out = capsys.readouterr().out
        assert "does not choose an adaptation" in out
        assert "Adaptation policy is Milestone 8" in out

    def test_the_artifact_locations_are_printed(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "artifacts"
        _run(m5_dataset, output)
        out = capsys.readouterr().out
        for name in (
            "metrics.json",
            "selective_metrics.json",
            "coverage_curve.json",
            "thresholds.json",
            "adaptation_gate.parquet",
            "manifest.json",
        ):
            assert name in out, name
            assert (output / name).exists(), name


class TestRegressionDemo:
    def test_the_demo_succeeds_on_a_regression_target(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        assert _run(m5_dataset, tmp_path / "reg", target="engagement_score") == 0

    def test_the_conformal_rule_and_alpha_are_printed(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "conformal", target="engagement_score")
        out = capsys.readouterr().out
        assert "Interval method:" in out
        assert "Nominal miscoverage alpha:" in out
        assert "Nominal coverage:" in out
        assert "Conformal quantile per fold:" in out
        assert "ceil((n + 1)" in out

    def test_no_confidence_definition_is_printed_for_a_point_prediction(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(m5_dataset, tmp_path / "no-confidence", target="engagement_score")
        out = capsys.readouterr().out
        assert "Confidence definition:" not in out
        assert "no meaning for a point prediction" in out

    def test_no_width_grid_reports_the_operating_point_only(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(m5_dataset, tmp_path / "no-grid", target="engagement_score") == 0
        out = capsys.readouterr().out
        assert "Interval-width grid:           not configured" in out
        assert "Coverage curve x-axis:         maximum_interval_width" in out
        assert "Coverage curve:                not evaluated" in out

    def test_a_width_grid_sweeps_the_width_axis(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            _run(
                m5_dataset,
                tmp_path / "width-grid",
                "--interval-width-grid",
                "0,0.25,0.5,0.75,1",
                target="engagement_score",
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "higher is MORE permissive" in out
        assert "Coverage curve x-axis:         maximum_interval_width" in out
        assert "Coverage non_decreasing" in out
        assert "coverage[i + 1] >= coverage[i]" in out

    def test_the_swept_regression_coverage_only_rises(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "width-rises"
        assert (
            _run(
                m5_dataset,
                output,
                "--interval-width-grid",
                "0,0.25,0.5,0.75,1",
                target="engagement_score",
            )
            == 0
        )
        document = json.loads((output / "coverage_curve.json").read_text())
        assert document["x_axis"] == "maximum_interval_width"
        coverages = [
            point["coverage_point"]["coverage"] for point in document["curve"]["points"]
        ]
        assert coverages == sorted(coverages)
        assert document["curve"]["coverage_is_monotonic"] is True

    def test_a_classification_curve_keeps_the_original_direction(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "confidence-falls"
        assert _run(m5_dataset, output, target="engagement_class") == 0
        document = json.loads((output / "coverage_curve.json").read_text())
        assert document["x_axis"] == "confidence_threshold"
        assert document["curve"]["expected_monotonic_direction"] == "non_increasing"
        coverages = [
            point["coverage_point"]["coverage"] for point in document["curve"]["points"]
        ]
        assert coverages == sorted(coverages, reverse=True)
        assert document["curve"]["coverage_is_monotonic"] is True

    def test_all_four_targets_run(self, m5_dataset: Path, tmp_path: Path) -> None:
        for target in (
            "engagement_class",
            "cognitive_load_class",
            "engagement_score",
            "cognitive_load_score",
        ):
            assert _run(m5_dataset, tmp_path / target, target=target) == 0


class TestDeterminism:
    def test_repeating_a_command_reproduces_the_documents(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        first, second = tmp_path / "rep-a", tmp_path / "rep-b"
        assert _run(m5_dataset, first) == 0
        assert _run(m5_dataset, second) == 0
        for name in ("uncertainty.json", "coverage_curve.json", "thresholds.json"):
            assert (first / name).read_text() == (second / name).read_text(), name

    def test_repeating_a_command_reproduces_the_run_identifier(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        first, second = tmp_path / "id-a", tmp_path / "id-b"
        _run(m5_dataset, first)
        _run(m5_dataset, second)
        a = json.loads((first / "manifest.json").read_text())
        b = json.loads((second / "manifest.json").read_text())
        assert a["run_id"] == b["run_id"]
        assert a["started_at_utc"] != b["started_at_utc"]


class TestScientificMode:
    def test_scientific_mode_rejects_synthetic_data(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            m5_dataset,
            tmp_path / "sci",
            "--mode",
            "scientific",
            command="uncertainty-train",
        )
        assert code == 3
        assert "scientific evaluation refused" in capsys.readouterr().err

    def test_self_check_mode_of_the_train_command_succeeds(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        assert (
            _run(
                m5_dataset,
                tmp_path / "train-self",
                "--mode",
                "self-check",
                command="uncertainty-train",
            )
            == 0
        )


class TestArgumentValidation:
    def test_a_missing_dataset_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(tmp_path / "absent.parquet", tmp_path / "out")
        assert code == 2
        assert "does not exist" in capsys.readouterr().err

    def test_a_single_fold_is_refused(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "uncertainty-demo",
                "--dataset",
                str(m5_dataset),
                "--folds",
                "1",
                "--output",
                str(tmp_path / "one"),
            ]
        )
        assert code == 2
        assert "no held-out portion" in capsys.readouterr().err

    @pytest.mark.parametrize("threshold", ["-0.5", "1.5"])
    def test_an_invalid_threshold_is_refused(
        self,
        m5_dataset: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        threshold: str,
    ) -> None:
        code = _run(m5_dataset, tmp_path / "tau", "--threshold", threshold)
        assert code == 2
        assert "must lie in [0, 1]" in capsys.readouterr().err

    @pytest.mark.parametrize("alpha", ["0", "1", "-0.2", "1.4"])
    def test_an_invalid_alpha_is_refused(
        self,
        m5_dataset: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        alpha: str,
    ) -> None:
        code = _run(
            m5_dataset, tmp_path / "alpha", "--alpha", alpha, target="engagement_score"
        )
        assert code == 2
        assert "strictly between 0 and 1" in capsys.readouterr().err

    def test_a_non_positive_maximum_width_is_refused(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            m5_dataset,
            tmp_path / "width",
            "--maximum-interval-width",
            "0",
            target="engagement_score",
        )
        assert code == 2
        assert "abstain on every window" in capsys.readouterr().err

    @pytest.mark.parametrize("grid", ["-0.5,0.5", "0.5,nan", "0.5,inf"])
    def test_an_invalid_width_grid_entry_is_refused(
        self,
        m5_dataset: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        grid: str,
    ) -> None:
        # Passed as a single ``--opt=value`` token so a leading minus reads
        # as part of the value rather than as another option.
        code = _run(
            m5_dataset,
            tmp_path / "bad-grid",
            f"--interval-width-grid={grid}",
            target="engagement_score",
        )
        assert code == 2
        assert "finite non-negative width" in capsys.readouterr().err

    def test_a_non_numeric_width_grid_entry_is_refused(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            m5_dataset,
            tmp_path / "text-grid",
            "--interval-width-grid",
            "0.5,wide",
            target="engagement_score",
        )
        assert code == 2
        assert "is not a number" in capsys.readouterr().err

    def test_a_repeated_width_grid_entry_is_refused(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            m5_dataset,
            tmp_path / "dup-grid",
            "--interval-width-grid",
            "0.5,0.5",
            target="engagement_score",
        )
        assert code == 2
        assert "repeats a width" in capsys.readouterr().err

    def test_an_empty_width_grid_is_refused(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _run(
            m5_dataset,
            tmp_path / "empty-grid",
            "--interval-width-grid",
            ",",
            target="engagement_score",
        )
        assert code == 2
        assert "is empty" in capsys.readouterr().err

    def test_an_unsupported_target_is_refused_by_argparse(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit):
            _run(m5_dataset, tmp_path / "bad-target", target="mood")

    def test_a_late_fusion_source_is_not_offered(
        self, m5_dataset: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit):
            _run(m5_dataset, tmp_path / "late", "--source", "quality_late")

    def test_personalized_thresholds_can_be_disabled(
        self, m5_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            _run(m5_dataset, tmp_path / "no-personal", "--no-personalized-thresholds")
            == 0
        )
        assert "disabled; the population threshold applies" in capsys.readouterr().out


class TestConfiguration:
    def test_the_shipped_defaults_resolve(self) -> None:
        resolved = load_config().uncertainty.resolve()
        assert resolved.population_confidence_threshold == pytest.approx(0.70)
        assert resolved.threshold_grid[0] == pytest.approx(0.0)
        assert resolved.alpha == pytest.approx(0.10)

    def test_a_late_fusion_prediction_source_is_refused_with_a_reason(self) -> None:
        with pytest.raises(ValueError, match="never the FUSED probability"):
            UncertaintyConfig(prediction_source="quality_late")

    def test_quality_is_not_a_configurable_modality(self) -> None:
        with pytest.raises(ValueError, match="not a measurement modality"):
            UncertaintyConfig(modalities=["quality", "rppg"])

    def test_duplicate_modalities_are_refused(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            UncertaintyConfig(modalities=["rppg", "rppg"])

    def test_an_empty_threshold_grid_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            UncertaintyConfig(classification={"threshold_grid": []})

    def test_a_duplicate_grid_entry_is_refused(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            UncertaintyConfig(classification={"threshold_grid": [0.5, 0.5]})

    def test_an_out_of_range_grid_entry_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
            UncertaintyConfig(classification={"threshold_grid": [0.5, 1.5]})

    def test_an_invalid_alpha_is_refused(self) -> None:
        with pytest.raises(ValueError):
            UncertaintyConfig(regression={"alpha": 0.0})

    def test_the_population_fallback_cannot_be_disabled(self) -> None:
        with pytest.raises(ValueError, match="cannot be disabled"):
            UncertaintyConfig(
                personalization={"fallback_to_population_threshold": False}
            )

    def test_an_unreachable_personal_minimum_is_refused(self) -> None:
        with pytest.raises(ValueError, match="could ever fire"):
            UncertaintyConfig(
                personalization={
                    "confidence_threshold_enabled": True,
                    "calibration_windows": 2,
                    "minimum_personal_calibration_windows": 5,
                }
            )

    def test_quality_is_not_a_required_gate_modality(self) -> None:
        with pytest.raises(ValueError, match="not a measurement modality"):
            UncertaintyConfig(evidence_gate={"required_modalities": ["quality"]})

    def test_the_grid_is_sorted_deterministically_on_resolve(self) -> None:
        resolved = UncertaintyConfig(
            classification={"threshold_grid": [0.9, 0.1, 0.5]}
        ).resolve()
        assert resolved.threshold_grid == (0.1, 0.5, 0.9)

    def test_the_shipped_defaults_configure_no_width_grid(self) -> None:
        # A width is in the target's units, so there is no scale-free default
        # sweep to ship. Null means the run reports its operating point only.
        assert load_config().uncertainty.regression.interval_width_grid is None
        assert load_config().uncertainty.resolve().interval_width_grid is None

    def test_a_width_grid_is_never_defaulted_from_the_confidence_grid(self) -> None:
        resolved = UncertaintyConfig(
            classification={"threshold_grid": [0.1, 0.5, 0.9]}
        ).resolve()
        assert resolved.threshold_grid == (0.1, 0.5, 0.9)
        assert resolved.interval_width_grid is None

    def test_a_width_grid_may_leave_the_unit_interval(self) -> None:
        resolved = UncertaintyConfig(
            regression={"interval_width_grid": [0.0, 2.5, 40.0]}
        ).resolve()
        assert resolved.interval_width_grid == (0.0, 2.5, 40.0)

    def test_a_negative_width_grid_entry_is_refused(self) -> None:
        with pytest.raises(ValueError, match="is negative"):
            UncertaintyConfig(regression={"interval_width_grid": [-0.5, 0.5]})

    def test_a_non_finite_width_grid_entry_is_refused(self) -> None:
        with pytest.raises(ValueError, match="is not finite"):
            UncertaintyConfig(regression={"interval_width_grid": [0.5, float("inf")]})

    def test_an_empty_width_grid_is_refused(self) -> None:
        with pytest.raises(ValueError, match="is empty"):
            UncertaintyConfig(regression={"interval_width_grid": []})

    def test_a_duplicate_width_grid_entry_is_refused(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            UncertaintyConfig(regression={"interval_width_grid": [0.5, 0.5]})

    def test_the_width_grid_is_sorted_deterministically_on_resolve(self) -> None:
        resolved = UncertaintyConfig(
            regression={"interval_width_grid": [0.9, 0.1, 0.5]}
        ).resolve()
        assert resolved.interval_width_grid == (0.1, 0.5, 0.9)
