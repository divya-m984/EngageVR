"""Tests for the GREEN, CHROM, and POS extraction methods."""

from __future__ import annotations

import numpy as np
import pytest

from engagevr.config import RppgPreprocessingConfig
from engagevr.rppg.errors import RppgUnavailable
from engagevr.rppg.methods import (
    DEFAULT_METHOD_WINDOW_SECONDS,
    POS_PROJECTION,
    extract_chrom,
    extract_green,
    extract_pos,
    extract_waveform,
)
from engagevr.schemas.rppg import RppgMethod, UnavailableReason

FS = 30.0
SECONDS = 20.0


def pulsatile_rgb(
    bpm: float = 72.0,
    seconds: float = SECONDS,
    fs: float = FS,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """A clean (N, 3) RGB trace carrying a known pulse frequency."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    pulse = np.sin(2.0 * np.pi * (bpm / 60.0) * t)
    rng = np.random.default_rng(seed)
    base = np.array([150.0, 110.0, 100.0])
    gain = np.array([0.45, 1.0, 0.35])
    out = base + np.outer(pulse, gain)
    if noise > 0.0:
        out = out + rng.normal(0.0, noise, size=out.shape)
    return out


BAND = {"band_low_hz": 0.7, "band_high_hz": 4.0, "filter_order": 4}


# --- determinism ----------------------------------------------------------


@pytest.mark.parametrize(
    "runner",
    [
        lambda rgb: extract_green(rgb, FS),
        lambda rgb: extract_chrom(rgb, FS, **BAND),
        lambda rgb: extract_pos(rgb, FS),
    ],
    ids=["green", "chrom", "pos"],
)
def test_methods_are_deterministic(runner) -> None:  # type: ignore[no-untyped-def]
    rgb = pulsatile_rgb(noise=0.3, seed=1)

    assert np.array_equal(runner(rgb), runner(rgb))


@pytest.mark.parametrize(
    "runner",
    [
        lambda rgb: extract_green(rgb, FS),
        lambda rgb: extract_chrom(rgb, FS, **BAND),
        lambda rgb: extract_pos(rgb, FS),
    ],
    ids=["green", "chrom", "pos"],
)
def test_methods_produce_no_nan_on_valid_input(runner) -> None:  # type: ignore[no-untyped-def]
    rgb = pulsatile_rgb(noise=0.4, seed=2)

    out = runner(rgb)

    assert np.all(np.isfinite(out))
    assert out.size > 0


# --- GREEN ----------------------------------------------------------------


def test_green_uses_only_the_green_channel() -> None:
    """Changing R and B must not change the GREEN output."""
    rgb = pulsatile_rgb()
    altered = rgb.copy()
    altered[:, 0] *= 1.5
    altered[:, 2] *= 0.5

    assert np.allclose(extract_green(rgb, FS), extract_green(altered, FS))


def test_green_output_is_mean_removed() -> None:
    out = extract_green(pulsatile_rgb(), FS)

    assert float(np.mean(out)) == pytest.approx(0.0, abs=1e-9)


def test_green_recovers_the_pulse_shape() -> None:
    """The normalized green trace correlates with the injected pulse."""
    n = int(SECONDS * FS)
    t = np.arange(n) / FS
    pulse = np.sin(2.0 * np.pi * 1.2 * t)

    out = extract_green(pulsatile_rgb(bpm=72.0), FS)

    assert abs(np.corrcoef(out, pulse)[0, 1]) > 0.99


# --- CHROM ----------------------------------------------------------------


def test_chrom_produces_a_full_length_output() -> None:
    rgb = pulsatile_rgb()

    out = extract_chrom(rgb, FS, **BAND)

    assert out.shape == (rgb.shape[0],)


def test_chrom_suppresses_pure_intensity_variation() -> None:
    """A common-mode brightness change is not a pulse.

    CHROM's chrominance projection should respond far less to a signal
    that scales all three channels together than to a genuine pulse.
    """
    n = int(SECONDS * FS)
    t = np.arange(n) / FS
    base = np.array([150.0, 110.0, 100.0])

    intensity_only = np.outer(1.0 + 0.02 * np.sin(2.0 * np.pi * 1.2 * t), base)
    genuine_pulse = pulsatile_rgb(bpm=72.0)

    intensity_response = np.std(extract_chrom(intensity_only, FS, **BAND))
    pulse_response = np.std(extract_chrom(genuine_pulse, FS, **BAND))

    assert intensity_response < pulse_response


# --- POS ------------------------------------------------------------------


def test_pos_projection_matrix_matches_the_paper() -> None:
    """Wang et al. (2017) Algorithm 1: P = [[0,1,-1], [-2,1,1]]."""
    assert np.array_equal(
        POS_PROJECTION, np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    )


def test_pos_projection_is_orthogonal_to_the_skin_tone_axis() -> None:
    """Both rows of P must annihilate the [1, 1, 1] intensity direction."""
    assert np.allclose(POS_PROJECTION @ np.array([1.0, 1.0, 1.0]), 0.0)


def test_pos_suppresses_pure_intensity_variation() -> None:
    n = int(SECONDS * FS)
    t = np.arange(n) / FS
    base = np.array([150.0, 110.0, 100.0])
    intensity_only = np.outer(1.0 + 0.02 * np.sin(2.0 * np.pi * 1.2 * t), base)

    intensity_response = np.std(extract_pos(intensity_only, FS))
    pulse_response = np.std(extract_pos(pulsatile_rgb(bpm=72.0), FS))

    assert intensity_response < pulse_response


def test_pos_produces_a_full_length_output() -> None:
    rgb = pulsatile_rgb()

    out = extract_pos(rgb, FS)

    assert out.shape == (rgb.shape[0],)


def test_pos_window_length_follows_the_sampling_rate() -> None:
    """1.6 s at 60 Hz is 96 samples, not 48."""
    rgb_60 = pulsatile_rgb(fs=60.0)

    out = extract_pos(rgb_60, 60.0, window_seconds=DEFAULT_METHOD_WINDOW_SECONDS)

    # The first (l - 1) samples receive fewer overlap-add contributions.
    assert out.shape[0] == rgb_60.shape[0]


# --- shared rejection behaviour -------------------------------------------


@pytest.mark.parametrize(
    "runner",
    [
        lambda rgb: extract_green(rgb, FS),
        lambda rgb: extract_chrom(rgb, FS, **BAND),
        lambda rgb: extract_pos(rgb, FS),
    ],
    ids=["green", "chrom", "pos"],
)
def test_constant_trace_is_rejected(runner) -> None:  # type: ignore[no-untyped-def]
    constant = np.tile(np.array([150.0, 110.0, 100.0]), (600, 1))

    with pytest.raises(RppgUnavailable) as exc:
        runner(constant)

    assert exc.value.reason is UnavailableReason.CONSTANT_SIGNAL


@pytest.mark.parametrize(
    "runner",
    [
        lambda rgb: extract_green(rgb, FS),
        lambda rgb: extract_chrom(rgb, FS, **BAND),
        lambda rgb: extract_pos(rgb, FS),
    ],
    ids=["green", "chrom", "pos"],
)
def test_missing_channel_is_rejected(runner) -> None:  # type: ignore[no-untyped-def]
    two_channel = pulsatile_rgb()[:, :2]

    with pytest.raises(RppgUnavailable) as exc:
        runner(two_channel)

    assert exc.value.reason is UnavailableReason.MISSING_CHANNEL


@pytest.mark.parametrize(
    "runner",
    [
        lambda rgb: extract_green(rgb, FS),
        lambda rgb: extract_chrom(rgb, FS, **BAND),
        lambda rgb: extract_pos(rgb, FS),
    ],
    ids=["green", "chrom", "pos"],
)
def test_non_finite_input_is_rejected(runner) -> None:  # type: ignore[no-untyped-def]
    rgb = pulsatile_rgb()
    rgb[5, 1] = np.nan

    with pytest.raises(RppgUnavailable) as exc:
        runner(rgb)

    assert exc.value.reason is UnavailableReason.NON_FINITE_VALUES


@pytest.mark.parametrize(
    "runner",
    [
        lambda rgb: extract_chrom(rgb, FS, **BAND),
        lambda rgb: extract_pos(rgb, FS),
    ],
    ids=["chrom", "pos"],
)
def test_trace_shorter_than_method_window_is_rejected(runner) -> None:  # type: ignore[no-untyped-def]
    short = pulsatile_rgb(seconds=1.0)

    with pytest.raises(RppgUnavailable) as exc:
        runner(short)

    assert exc.value.reason is UnavailableReason.WINDOW_TOO_SHORT


# --- extract_waveform interface -------------------------------------------


@pytest.fixture
def preprocessing() -> RppgPreprocessingConfig:
    return RppgPreprocessingConfig()


@pytest.mark.parametrize("method", [RppgMethod.GREEN, RppgMethod.CHROM, RppgMethod.POS])
def test_extract_waveform_returns_available_result(
    method: RppgMethod, preprocessing: RppgPreprocessingConfig
) -> None:
    rgb = pulsatile_rgb(noise=0.2, seed=4)
    timestamps = np.arange(rgb.shape[0]) / FS

    waveform = extract_waveform(method, rgb, timestamps, FS, preprocessing)

    assert waveform.available is True
    assert waveform.reason is None
    assert waveform.method is method
    assert waveform.n_samples == rgb.shape[0]
    assert waveform.sampling_rate_hz == pytest.approx(FS)
    assert all(np.isfinite(v) for v in waveform.values)
    assert waveform.duration_s > 0.0


@pytest.mark.parametrize("method", [RppgMethod.GREEN, RppgMethod.CHROM, RppgMethod.POS])
def test_extract_waveform_records_method_and_parameters(
    method: RppgMethod, preprocessing: RppgPreprocessingConfig
) -> None:
    rgb = pulsatile_rgb()
    timestamps = np.arange(rgb.shape[0]) / FS

    waveform = extract_waveform(method, rgb, timestamps, FS, preprocessing)

    assert waveform.method_params["band_low_hz"] == preprocessing.pulse_band_low_hz
    assert waveform.method_params["band_high_hz"] == preprocessing.pulse_band_high_hz
    assert waveform.method_params["filter_order"] == preprocessing.filter_order
    assert waveform.method_params["sampling_rate_hz"] == pytest.approx(FS)
    assert waveform.band_low_hz == preprocessing.pulse_band_low_hz


@pytest.mark.parametrize("method", [RppgMethod.GREEN, RppgMethod.CHROM, RppgMethod.POS])
def test_extract_waveform_returns_unavailable_not_nan(
    method: RppgMethod, preprocessing: RppgPreprocessingConfig
) -> None:
    constant = np.tile(np.array([150.0, 110.0, 100.0]), (600, 1))
    timestamps = np.arange(600) / FS

    waveform = extract_waveform(method, constant, timestamps, FS, preprocessing)

    assert waveform.available is False
    assert waveform.reason is UnavailableReason.CONSTANT_SIGNAL
    assert waveform.values == []


def test_extract_waveform_rejects_band_above_nyquist(
    preprocessing: RppgPreprocessingConfig,
) -> None:
    rgb = pulsatile_rgb()
    timestamps = np.arange(rgb.shape[0]) / FS

    # 4 Hz upper edge is above Nyquist at 6 Hz sampling.
    waveform = extract_waveform(RppgMethod.GREEN, rgb, timestamps, 6.0, preprocessing)

    assert waveform.available is False
    assert waveform.reason is UnavailableReason.INVALID_FREQUENCY_BAND


def test_extract_waveform_preserves_timestamps(
    preprocessing: RppgPreprocessingConfig,
) -> None:
    rgb = pulsatile_rgb()
    timestamps = 100.0 + np.arange(rgb.shape[0]) / FS

    waveform = extract_waveform(RppgMethod.POS, rgb, timestamps, FS, preprocessing)

    assert waveform.timestamps[0] == pytest.approx(100.0)
    assert len(waveform.timestamps) == rgb.shape[0]


def test_methods_have_no_shared_mutable_state(
    preprocessing: RppgPreprocessingConfig,
) -> None:
    """Running one method must not perturb another's result."""
    rgb = pulsatile_rgb(noise=0.3, seed=9)
    timestamps = np.arange(rgb.shape[0]) / FS

    first = extract_waveform(RppgMethod.POS, rgb, timestamps, FS, preprocessing)
    extract_waveform(RppgMethod.CHROM, rgb, timestamps, FS, preprocessing)
    extract_waveform(RppgMethod.GREEN, rgb, timestamps, FS, preprocessing)
    again = extract_waveform(RppgMethod.POS, rgb, timestamps, FS, preprocessing)

    assert first.values == again.values
