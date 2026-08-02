"""Tests for rPPG preprocessing: detrending, normalization, resampling, filtering."""

from __future__ import annotations

import numpy as np
import pytest

from engagevr.rppg.errors import RppgUnavailable
from engagevr.rppg.preprocessing import (
    bandpass_filter,
    bpm_to_hz,
    design_bandpass_sos,
    detrend_linear,
    hz_to_bpm,
    min_samples_for_sosfiltfilt,
    normalize_channels,
    remove_mean,
    require_finite,
    require_non_constant,
    resample_uniform,
    standardize,
    validate_timing,
)
from engagevr.schemas.rppg import UnavailableReason

FS = 30.0


def sine(freq_hz: float, seconds: float = 20.0, fs: float = FS) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return np.sin(2.0 * np.pi * freq_hz * t)


# --- validation helpers ---------------------------------------------------


def test_require_finite_accepts_finite() -> None:
    require_finite(np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_require_finite_rejects_non_finite(bad: float) -> None:
    with pytest.raises(RppgUnavailable) as exc:
        require_finite(np.array([1.0, bad, 3.0]))

    assert exc.value.reason is UnavailableReason.NON_FINITE_VALUES


def test_require_finite_rejects_empty() -> None:
    with pytest.raises(RppgUnavailable) as exc:
        require_finite(np.array([]))

    assert exc.value.reason is UnavailableReason.WINDOW_TOO_SHORT


def test_require_non_constant_rejects_flat_signal() -> None:
    with pytest.raises(RppgUnavailable) as exc:
        require_non_constant(np.full(100, 5.0))

    assert exc.value.reason is UnavailableReason.CONSTANT_SIGNAL


# --- mean removal, detrending, normalization ------------------------------


def test_remove_mean_zeroes_the_mean() -> None:
    x = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])

    out = remove_mean(x)

    assert np.allclose(out.mean(axis=0), 0.0)


def test_detrend_removes_a_linear_ramp() -> None:
    """A pure ramp must detrend to (numerically) zero."""
    ramp = np.linspace(0.0, 100.0, 300).reshape(-1, 1)

    out = detrend_linear(ramp)

    assert np.allclose(out, 0.0, atol=1e-9)


def test_detrend_preserves_oscillation_but_removes_drift() -> None:
    t = np.arange(600) / FS
    oscillation = np.sin(2.0 * np.pi * 1.2 * t)
    drifted = (oscillation + 3.0 * t).reshape(-1, 1)

    out = detrend_linear(drifted).reshape(-1)

    # The drift is gone but the oscillation amplitude survives.
    assert abs(out[-1] - out[0]) < 1.0
    assert np.std(out) == pytest.approx(np.std(oscillation), rel=0.05)


def test_detrend_rejects_single_sample() -> None:
    with pytest.raises(RppgUnavailable) as exc:
        detrend_linear(np.array([[1.0]]))

    assert exc.value.reason is UnavailableReason.WINDOW_TOO_SHORT


def test_normalize_channels_gives_unit_mean() -> None:
    x = np.array([[100.0, 200.0, 50.0], [120.0, 180.0, 60.0]])

    out = normalize_channels(x)

    assert np.allclose(out.mean(axis=0), 1.0)


def test_normalize_rejects_zero_mean_channel() -> None:
    x = np.array([[1.0, 5.0, 5.0], [-1.0, 5.0, 5.0]])

    with pytest.raises(RppgUnavailable) as exc:
        normalize_channels(x)

    assert exc.value.reason is UnavailableReason.CONSTANT_SIGNAL


def test_standardize_gives_zero_mean_unit_variance() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(50.0, 7.0, size=(500, 3))

    out = standardize(x)

    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(out.std(axis=0), 1.0)


def test_standardize_rejects_constant_channel() -> None:
    x = np.column_stack([np.arange(100.0), np.full(100, 3.0)])

    with pytest.raises(RppgUnavailable) as exc:
        standardize(x)

    assert exc.value.reason is UnavailableReason.CONSTANT_SIGNAL


# --- timing validation ----------------------------------------------------


def test_validate_timing_returns_sampling_rate() -> None:
    timestamps = np.arange(300) / FS

    fs = validate_timing(timestamps, max_jitter_s=0.02, min_duration_s=5.0)

    assert fs == pytest.approx(FS)


def test_validate_timing_rejects_duplicates() -> None:
    timestamps = np.array([0.0, 0.1, 0.1, 0.2, 0.3, 0.4])

    with pytest.raises(RppgUnavailable) as exc:
        validate_timing(timestamps, max_jitter_s=1.0, min_duration_s=0.1)

    assert exc.value.reason is UnavailableReason.DUPLICATE_TIMESTAMPS


def test_validate_timing_rejects_reversal() -> None:
    timestamps = np.array([0.0, 0.2, 0.1, 0.3, 0.4])

    with pytest.raises(RppgUnavailable) as exc:
        validate_timing(timestamps, max_jitter_s=1.0, min_duration_s=0.1)

    assert exc.value.reason is UnavailableReason.NON_MONOTONIC_TIMESTAMPS


def test_validate_timing_rejects_excessive_jitter() -> None:
    """Severely irregular sampling must not be treated as uniform."""
    rng = np.random.default_rng(3)
    timestamps = np.cumsum(rng.uniform(0.005, 0.20, size=400))

    with pytest.raises(RppgUnavailable) as exc:
        validate_timing(timestamps, max_jitter_s=0.005, min_duration_s=1.0)

    assert exc.value.reason is UnavailableReason.EXCESSIVE_TIMESTAMP_JITTER


def test_validate_timing_rejects_short_window() -> None:
    timestamps = np.arange(30) / FS

    with pytest.raises(RppgUnavailable) as exc:
        validate_timing(timestamps, max_jitter_s=0.02, min_duration_s=8.0)

    assert exc.value.reason is UnavailableReason.WINDOW_TOO_SHORT


def test_validate_timing_rejects_single_timestamp() -> None:
    with pytest.raises(RppgUnavailable) as exc:
        validate_timing(np.array([0.0]), max_jitter_s=1.0, min_duration_s=0.0)

    assert exc.value.reason is UnavailableReason.WINDOW_TOO_SHORT


# --- resampling -----------------------------------------------------------


def test_resample_onto_uniform_grid() -> None:
    timestamps = np.array([0.0, 0.03, 0.07, 0.10, 0.14, 0.17, 0.20])
    values = timestamps.reshape(-1, 1) * 2.0

    grid, out = resample_uniform(timestamps, values, 30.0)

    assert np.allclose(np.diff(grid), 1.0 / 30.0)
    # A linear input resamples to the same linear function.
    assert np.allclose(out.reshape(-1), grid * 2.0, atol=1e-9)


def test_resample_preserves_a_sinusoid() -> None:
    t = np.arange(600) / FS
    values = np.sin(2.0 * np.pi * 1.2 * t).reshape(-1, 1)

    grid, out = resample_uniform(t, values, FS)

    assert out.shape[0] == grid.size
    assert np.corrcoef(out.reshape(-1)[:590], values.reshape(-1)[:590])[0, 1] > 0.99


def test_resample_rejects_non_positive_rate() -> None:
    with pytest.raises(RppgUnavailable) as exc:
        resample_uniform(np.arange(10.0), np.arange(10.0).reshape(-1, 1), 0.0)

    assert exc.value.reason is UnavailableReason.SAMPLING_RATE_UNKNOWN


def test_resample_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="matching lengths"):
        resample_uniform(np.arange(10.0), np.arange(5.0).reshape(-1, 1), 30.0)


# --- filter design and application ----------------------------------------


def test_bandpass_uses_second_order_sections() -> None:
    sos = design_bandpass_sos(0.7, 4.0, FS, 4)

    # (n_sections, 6) is the SOS layout; direct-form would be 1-D.
    assert sos.ndim == 2
    assert sos.shape[1] == 6


def test_bandpass_passes_in_band_and_rejects_out_of_band() -> None:
    sos = design_bandpass_sos(0.7, 4.0, FS, 4)
    in_band = sine(1.2).reshape(-1, 1)
    out_of_band = sine(0.1).reshape(-1, 1)

    passed = bandpass_filter(in_band, sos)
    blocked = bandpass_filter(out_of_band, sos)

    # Ignore filter edges, which are transient by construction.
    # A unit-amplitude sine has std 1/sqrt(2); the pass band must preserve it.
    assert np.std(passed[100:-100]) == pytest.approx(
        np.std(in_band[100:-100]), rel=0.05
    )
    assert np.std(blocked[100:-100]) < 0.1


def test_bandpass_rejects_high_frequency() -> None:
    sos = design_bandpass_sos(0.7, 4.0, FS, 4)
    high = sine(10.0).reshape(-1, 1)

    blocked = bandpass_filter(high, sos)

    assert np.std(blocked[100:-100]) < 0.1


@pytest.mark.parametrize(
    ("low", "high"),
    [(4.0, 0.7), (1.0, 1.0), (-1.0, 4.0), (0.0, 4.0)],
)
def test_invalid_band_is_rejected(low: float, high: float) -> None:
    with pytest.raises(RppgUnavailable) as exc:
        design_bandpass_sos(low, high, FS, 4)

    assert exc.value.reason is UnavailableReason.INVALID_FREQUENCY_BAND


def test_upper_band_at_or_above_nyquist_is_rejected() -> None:
    """15 Hz is exactly Nyquist at 30 Hz and must be refused."""
    with pytest.raises(RppgUnavailable) as exc:
        design_bandpass_sos(0.7, 15.0, FS, 4)

    assert exc.value.reason is UnavailableReason.INVALID_FREQUENCY_BAND

    with pytest.raises(RppgUnavailable):
        design_bandpass_sos(0.7, 20.0, FS, 4)


def test_zero_sampling_rate_is_rejected() -> None:
    with pytest.raises(RppgUnavailable) as exc:
        design_bandpass_sos(0.7, 4.0, 0.0, 4)

    assert exc.value.reason is UnavailableReason.SAMPLING_RATE_UNKNOWN


def test_too_short_window_is_rejected_not_padded() -> None:
    sos = design_bandpass_sos(0.7, 4.0, FS, 4)
    required = min_samples_for_sosfiltfilt(sos)
    short = sine(1.2, seconds=(required - 5) / FS).reshape(-1, 1)

    with pytest.raises(RppgUnavailable) as exc:
        bandpass_filter(short, sos)

    assert exc.value.reason is UnavailableReason.FILTER_NOT_VIABLE


def test_minimum_length_scales_with_filter_order() -> None:
    low_order = min_samples_for_sosfiltfilt(design_bandpass_sos(0.7, 4.0, FS, 2))
    high_order = min_samples_for_sosfiltfilt(design_bandpass_sos(0.7, 4.0, FS, 8))

    assert high_order > low_order


def test_filter_rejects_non_finite_input() -> None:
    sos = design_bandpass_sos(0.7, 4.0, FS, 4)
    x = sine(1.2).reshape(-1, 1)
    x[10] = np.nan

    with pytest.raises(RppgUnavailable) as exc:
        bandpass_filter(x, sos)

    assert exc.value.reason is UnavailableReason.NON_FINITE_VALUES


def test_zero_phase_filtering_does_not_shift_the_signal() -> None:
    """sosfiltfilt must not introduce a phase lag."""
    sos = design_bandpass_sos(0.7, 4.0, FS, 4)
    x = sine(1.2, seconds=30).reshape(-1, 1)

    y = bandpass_filter(x, sos).reshape(-1)

    core = slice(150, -150)
    correlation = np.corrcoef(x.reshape(-1)[core], y[core])[0, 1]
    assert correlation > 0.99


# --- unit conversion ------------------------------------------------------


def test_hz_bpm_round_trip() -> None:
    assert hz_to_bpm(1.2) == pytest.approx(72.0)
    assert bpm_to_hz(72.0) == pytest.approx(1.2)
    assert bpm_to_hz(hz_to_bpm(1.7)) == pytest.approx(1.7)
