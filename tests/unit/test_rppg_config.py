"""Tests for rPPG configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from engagevr.config import (
    RppgConfig,
    RppgPreprocessingConfig,
    RppgQualityConfig,
    RppgRoiConfig,
    RppgSpectralConfig,
    RppgTraceConfig,
    RppgWindowConfig,
    load_config,
)
from engagevr.schemas.rppg import RoiRegion, RppgMethod

# --- defaults load from YAML ----------------------------------------------


def test_defaults_yaml_provides_an_rppg_section() -> None:
    cfg = load_config()

    assert cfg.rppg.method is RppgMethod.POS
    assert cfg.rppg.preprocessing.pulse_band_low_hz == pytest.approx(0.7)
    assert cfg.rppg.preprocessing.pulse_band_high_hz == pytest.approx(4.0)
    assert cfg.rppg.window.duration_seconds == pytest.approx(30.0)
    assert cfg.rppg.datasets.ubfc_rppg_root is None


def test_default_band_matches_the_documented_bpm_range() -> None:
    """0.7-4.0 Hz is 42-240 BPM, following de Haan & Jeanne (2013)."""
    cfg = RppgPreprocessingConfig()

    assert cfg.pulse_band_low_hz * 60.0 == pytest.approx(42.0)
    assert cfg.pulse_band_high_hz * 60.0 == pytest.approx(240.0)


def test_default_roi_regions_exclude_the_derived_combined_region() -> None:
    cfg = RppgRoiConfig()

    assert RoiRegion.COMBINED not in cfg.regions
    assert set(cfg.regions) == {
        RoiRegion.FOREHEAD,
        RoiRegion.LEFT_CHEEK,
        RoiRegion.RIGHT_CHEEK,
    }


# --- rejection of invalid values ------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_seconds": -1.0},
        {"duration_seconds": 0.0},
        {"step_seconds": -0.5},
        {"step_seconds": 0.0},
        {"min_duration_seconds": -2.0},
    ],
)
def test_negative_durations_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        RppgWindowConfig(**kwargs)


def test_step_larger_than_window_is_rejected() -> None:
    with pytest.raises(ValidationError, match="step_seconds"):
        RppgWindowConfig(duration_seconds=10.0, step_seconds=20.0)


def test_minimum_longer_than_window_is_rejected() -> None:
    with pytest.raises(ValidationError, match="min_duration_seconds"):
        RppgWindowConfig(duration_seconds=10.0, min_duration_seconds=20.0)


@pytest.mark.parametrize(
    ("low", "high"),
    [(4.0, 0.7), (2.0, 2.0), (5.0, 1.0)],
)
def test_inverted_frequency_band_is_rejected(low: float, high: float) -> None:
    with pytest.raises(ValidationError, match="pulse_band_low_hz"):
        RppgPreprocessingConfig(pulse_band_low_hz=low, pulse_band_high_hz=high)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_non_positive_frequencies_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        RppgPreprocessingConfig(pulse_band_low_hz=value)
    with pytest.raises(ValidationError):
        RppgPreprocessingConfig(pulse_band_high_hz=value)


def test_upper_frequency_at_nyquist_is_rejected() -> None:
    """At 30 fps, Nyquist is 15 Hz; the band edge must be strictly below."""
    with pytest.raises(ValidationError, match="Nyquist"):
        RppgConfig(
            trace=RppgTraceConfig(expected_fps=30.0),
            preprocessing=RppgPreprocessingConfig(
                pulse_band_low_hz=0.7, pulse_band_high_hz=15.0
            ),
        )


def test_upper_frequency_above_nyquist_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Nyquist"):
        RppgConfig(
            trace=RppgTraceConfig(expected_fps=8.0),
            preprocessing=RppgPreprocessingConfig(
                pulse_band_low_hz=0.7, pulse_band_high_hz=4.0
            ),
        )


def test_band_below_nyquist_is_accepted() -> None:
    cfg = RppgConfig(
        trace=RppgTraceConfig(expected_fps=30.0),
        preprocessing=RppgPreprocessingConfig(
            pulse_band_low_hz=0.7, pulse_band_high_hz=4.0
        ),
    )

    assert cfg.preprocessing.pulse_band_high_hz < cfg.trace.expected_fps / 2


@pytest.mark.parametrize("overlap", [-0.1, 1.0, 1.5])
def test_invalid_welch_overlap_is_rejected(overlap: float) -> None:
    with pytest.raises(ValidationError):
        RppgSpectralConfig(welch_overlap=overlap)


def test_valid_welch_overlap_is_accepted() -> None:
    assert RppgSpectralConfig(welch_overlap=0.0).welch_overlap == 0.0
    assert RppgSpectralConfig(welch_overlap=0.75).welch_overlap == 0.75


def test_welch_segment_longer_than_window_is_rejected() -> None:
    with pytest.raises(ValidationError, match="welch_segment_seconds"):
        RppgConfig(
            window=RppgWindowConfig(duration_seconds=10.0),
            spectral=RppgSpectralConfig(welch_segment_seconds=20.0),
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, 2.0])
def test_quality_threshold_outside_zero_one_is_rejected(
    threshold: float,
) -> None:
    with pytest.raises(ValidationError):
        RppgQualityConfig(threshold=threshold)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_invalid_prominence_and_ratio_bounds_are_rejected(
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        RppgSpectralConfig(min_relative_peak_prominence=value)
    with pytest.raises(ValidationError):
        RppgSpectralConfig(min_spectral_peak_ratio=value)


def test_unsupported_method_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RppgConfig(method="deepphys")  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["green", "chrom", "pos"])
def test_supported_method_names_are_accepted(method: str) -> None:
    assert RppgConfig(method=method).method.value == method  # type: ignore[arg-type]


def test_combined_region_cannot_be_configured() -> None:
    with pytest.raises(ValidationError, match="combined"):
        RppgRoiConfig(regions=[RoiRegion.COMBINED])


def test_duplicate_regions_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        RppgRoiConfig(regions=[RoiRegion.FOREHEAD, RoiRegion.FOREHEAD])


def test_empty_region_list_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RppgRoiConfig(regions=[])


def test_inverted_clipping_bounds_are_rejected() -> None:
    with pytest.raises(ValidationError, match="clipping_low"):
        RppgRoiConfig(clipping_low=200, clipping_high=100)


@pytest.mark.parametrize("value", [-1.0, 0.0])
def test_non_positive_jitter_tolerance_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        RppgTraceConfig(max_timestamp_jitter_s=value)


@pytest.mark.parametrize("value", [0.0, -30.0])
def test_non_positive_expected_fps_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        RppgTraceConfig(expected_fps=value)


@pytest.mark.parametrize("value", [0.0, 101.0, -5.0])
def test_invalid_valid_frame_percentage_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        RppgTraceConfig(min_valid_frame_pct=value)


@pytest.mark.parametrize("order", [0, -1, 20])
def test_invalid_filter_order_is_rejected(order: int) -> None:
    with pytest.raises(ValidationError):
        RppgPreprocessingConfig(filter_order=order)


@pytest.mark.parametrize("value", [0.45, 0.9, -0.1])
def test_invalid_inset_fraction_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        RppgRoiConfig(inset_fraction=value)


def test_min_valid_pixels_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RppgRoiConfig(min_valid_pixels=0)
