"""Tests for the rppg-demo and rppg-evaluate CLI commands."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from engagevr.__main__ import main


def run_demo(tmp_path: Path, *args: str) -> tuple[int, Path]:
    out = tmp_path / "nested" / "deeper" / "rppg-demo.json"
    code = main(["rppg-demo", *args, "--output", str(out)])
    return code, out


# --- successful runs ------------------------------------------------------


def test_demo_succeeds_and_writes_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run_demo(
        tmp_path,
        "--bpm",
        "72",
        "--duration",
        "30",
        "--fps",
        "30",
        "--method",
        "pos",
        "--seed",
        "42",
    )

    assert code == 0
    assert out.exists()
    captured = capsys.readouterr().out
    assert "Requested SYNTHETIC BPM" in captured
    assert "Estimated BPM" in captured
    assert str(out) in captured


def test_demo_creates_missing_output_directories(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "c" / "demo.json"
    assert not out.parent.exists()

    code = main(["rppg-demo", "--duration", "20", "--output", str(out)])

    assert code == 0
    assert out.parent.is_dir()
    assert out.exists()


def test_demo_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    args = ["rppg-demo", "--bpm", "78", "--duration", "20", "--seed", "7"]

    assert main([*args, "--output", str(first)]) == 0
    assert main([*args, "--output", str(second)]) == 0

    a = json.loads(first.read_text())
    b = json.loads(second.read_text())
    assert a["heart_rate"] == b["heart_rate"]
    assert a["quality"]["overall_quality"] == b["quality"]["overall_quality"]


@pytest.mark.parametrize("method", ["green", "chrom", "pos"])
def test_every_method_runs(tmp_path: Path, method: str) -> None:
    out = tmp_path / f"{method}.json"

    code = main(
        ["rppg-demo", "--duration", "25", "--method", method, "--output", str(out)]
    )

    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["input_parameters"]["method"] == method
    assert payload["heart_rate"]["method"] == method


@pytest.mark.parametrize("bpm", [55.0, 72.0, 110.0])
def test_multiple_bpm_values_are_recovered(tmp_path: Path, bpm: float) -> None:
    out = tmp_path / "bpm.json"

    code = main(
        ["rppg-demo", "--bpm", str(bpm), "--duration", "30", "--output", str(out)]
    )

    assert code == 0
    payload = json.loads(out.read_text())
    check = payload["synthetic_recovery_check"]
    assert check["requested_synthetic_bpm"] == pytest.approx(bpm)
    if check["estimated_bpm"] is not None:
        # Recovery is bounded by the reported spectral resolution.
        assert check["absolute_error_bpm"] <= check["frequency_resolution_bpm"]


# --- argument validation --------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["--bpm", "0"],
        ["--bpm", "-10"],
        ["--fps", "0"],
        ["--fps", "-30"],
        ["--duration", "0"],
        ["--duration", "-5"],
        ["--dropout-rate", "1.0"],
        ["--dropout-rate", "-0.5"],
    ],
)
def test_invalid_arguments_return_non_zero(tmp_path: Path, args: list[str]) -> None:
    code = main(["rppg-demo", *args, "--output", str(tmp_path / "x.json")])

    assert code != 0


def test_unsupported_method_is_rejected(tmp_path: Path) -> None:
    """argparse choices reject an unknown method with SystemExit(2)."""
    with pytest.raises(SystemExit) as exc:
        main(["rppg-demo", "--method", "deepphys", "--output", "x.json"])

    assert exc.value.code == 2


def test_fps_too_low_for_the_band_is_rejected(tmp_path: Path) -> None:
    """A 4 Hz band edge is unusable at 5 fps (Nyquist 2.5 Hz)."""
    code = main(
        [
            "rppg-demo",
            "--fps",
            "5",
            "--duration",
            "30",
            "--output",
            str(tmp_path / "x.json"),
        ]
    )

    assert code != 0


# --- output contents ------------------------------------------------------


def test_output_carries_the_permanent_synthetic_label(tmp_path: Path) -> None:
    _, out = run_demo(tmp_path, "--duration", "20")
    payload = json.loads(out.read_text())

    assert payload["synthetic_label"] == "SYNTHETIC"
    assert payload["data_source"] == "synthetic"
    assert payload["trace_summary"]["synthetic_label"] == "SYNTHETIC"
    assert payload["session_summary"]["synthetic_label"] == "SYNTHETIC"
    assert "SYNTHETIC" in payload["_synthetic_disclaimer"]


def test_output_disclaims_scientific_validation(tmp_path: Path) -> None:
    _, out = run_demo(tmp_path, "--duration", "20")
    text = out.read_text()
    payload = json.loads(text)

    disclaimer = payload["_synthetic_disclaimer"]
    assert "NOT scientific" in disclaimer
    assert "NOT a performance claim" in disclaimer
    assert "NOT" in payload["synthetic_recovery_check"]["note"]


def test_banner_disclaims_validation_on_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_demo(tmp_path, "--duration", "20")
    captured = capsys.readouterr().out

    assert "SYNTHETIC" in captured
    assert "NOT scientific" in captured
    assert "not a medical measurement" in captured


def test_output_records_input_and_method_parameters(tmp_path: Path) -> None:
    _, out = run_demo(
        tmp_path,
        "--bpm",
        "84",
        "--duration",
        "25",
        "--fps",
        "30",
        "--seed",
        "11",
        "--method",
        "chrom",
    )
    payload = json.loads(out.read_text())

    params = payload["input_parameters"]
    assert params["requested_bpm"] == pytest.approx(84.0)
    assert params["duration_seconds"] == pytest.approx(25.0)
    assert params["fps"] == pytest.approx(30.0)
    assert params["seed"] == 11
    assert params["method"] == "chrom"
    assert payload["method_parameters"]["band_low_hz"] == pytest.approx(0.7)
    assert payload["method_parameters"]["filter_order"] == 4


def test_output_contains_quality_components(tmp_path: Path) -> None:
    _, out = run_demo(tmp_path, "--duration", "25")
    payload = json.loads(out.read_text())

    components = payload["quality"]["components"]
    assert len(components) >= 8
    for c in components:
        assert "name" in c
        assert 0.0 <= c["score"] <= 1.0
    assert payload["quality"]["aggregation"]


def test_output_values_are_finite(tmp_path: Path) -> None:
    _, out = run_demo(tmp_path, "--duration", "25")
    payload = json.loads(out.read_text())

    def check(node: object) -> None:
        if isinstance(node, dict):
            for v in node.values():
                check(v)
        elif isinstance(node, list):
            for v in node:
                check(v)
        elif isinstance(node, float):
            assert math.isfinite(node)

    check(payload)


def test_output_contains_no_raw_frames_or_identifiers(tmp_path: Path) -> None:
    """Privacy: no imagery, no pixel arrays, no personal identifiers."""
    _, out = run_demo(tmp_path, "--duration", "20")
    payload = json.loads(out.read_text())
    text = out.read_text().lower()

    # Sample-level pixel data is excluded from the persisted trace summary.
    assert "samples" not in payload["trace_summary"]
    # The waveform summary omits the full sample arrays.
    assert "values" not in payload["waveform_summary"]
    assert "timestamps" not in payload["waveform_summary"]

    for forbidden in ("participant_name", "email", "@", "base64", "jpeg", "png"):
        assert forbidden not in text


def test_rejection_reason_is_present_when_quality_is_insufficient(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "bad.json"

    code = main(
        ["rppg-demo", "--duration", "30", "--dropout-rate", "0.7", "--output", str(out)]
    )

    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["heart_rate"]["available"] is False
    assert payload["heart_rate"]["bpm"] is None
    assert payload["heart_rate"]["reason"]
    assert payload["quality"]["rejection_reasons"]
    assert payload["quality"]["acceptable"] is False

    captured = capsys.readouterr().out
    assert "unavailable" in captured
    assert "Rejection reason" in captured


def test_low_quality_is_not_reported_as_low_engagement(tmp_path: Path) -> None:
    out = tmp_path / "bad.json"
    main(
        ["rppg-demo", "--duration", "30", "--dropout-rate", "0.7", "--output", str(out)]
    )
    payload = json.loads(out.read_text())

    warnings = " ".join(payload["quality"]["warnings"])
    assert "NOT mean low engagement" in warnings
    text = out.read_text().lower()
    assert "engagement_estimate" not in text
    assert "cognitive_load_estimate" not in text


# --- rppg-evaluate --------------------------------------------------------


def test_evaluate_without_root_returns_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "rppg-evaluate",
            "--dataset",
            "ubfc-rppg",
            "--output",
            str(tmp_path / "e.json"),
        ]
    )

    assert code == 2
    assert "never downloads datasets" in capsys.readouterr().err


def test_evaluate_with_missing_root_returns_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "rppg-evaluate",
            "--dataset",
            "ubfc-rppg",
            "--root",
            str(tmp_path / "nope"),
            "--output",
            str(tmp_path / "e.json"),
        ]
    )

    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_evaluate_reports_no_metrics_without_data(
    tmp_path: Path, ubfc_fixture_root: Path
) -> None:
    out = tmp_path / "eval.json"

    code = main(
        [
            "rppg-evaluate",
            "--dataset",
            "ubfc-rppg",
            "--root",
            str(ubfc_fixture_root),
            "--output",
            str(out),
        ]
    )

    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["metrics"] is None
    assert "PENDING" in payload["_note"]
    assert payload["provenance"]["license_status"]


def test_evaluate_never_reports_synthetic_as_dataset_performance(
    tmp_path: Path, ubfc_fixture_root: Path
) -> None:
    out = tmp_path / "eval.json"
    main(["rppg-evaluate", "--root", str(ubfc_fixture_root), "--output", str(out)])
    payload = json.loads(out.read_text())

    # No error metric of any kind is emitted without a real reference.
    assert payload["metrics"] is None
    for forbidden in ("mae_bpm", "rmse_bpm", "bias_bpm", "error_std_bpm"):
        assert forbidden not in out.read_text()
    # The output must say so explicitly rather than staying silent.
    assert "No synthetic value is ever reported here" in payload["_note"]
