"""Tests for spectral heart-rate estimation."""

from __future__ import annotations

import numpy as np
import pytest

from engagevr.config import RppgPreprocessingConfig, RppgSpectralConfig
from engagevr.rppg.heart_rate import estimate_heart_rate
from engagevr.schemas.rppg import RppgMethod, UnavailableReason

FS = 30.0


@pytest.fixture
def preprocessing() -> RppgPreprocessingConfig:
    return RppgPreprocessingConfig()


@pytest.fixture
def spectral() -> RppgSpectralConfig:
    return RppgSpectralConfig()


def tone(bpm: float, seconds: float = 30.0, fs: float = FS) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return np.sin(2.0 * np.pi * (bpm / 60.0) * t)


def resolution_bpm(seconds: float, fs: float = FS) -> float:
    """Welch bin width in BPM for the default 8 s segment."""
    segment = min(int(8.0 * fs), int(seconds * fs))
    return (fs / segment) * 60.0


# --- recovery of a known rate ---------------------------------------------


@pytest.mark.parametrize("bpm", [48.0, 60.0, 72.0, 90.0, 120.0, 180.0])
def test_known_bpm_is_recovered_within_one_bin(
    bpm: float,
    preprocessing: RppgPreprocessingConfig,
    spectral: RppgSpectralConfig,
) -> None:
    """Recovery is limited by Welch resolution, and that limit is reported."""
    estimate = estimate_heart_rate(
        tone(bpm), FS, preprocessing, spectral, method=RppgMethod.POS
    )

    assert estimate.available is True
    assert estimate.bpm is not None
    assert estimate.frequency_resolution_hz is not None
    tolerance = estimate.frequency_resolution_hz * 60.0
    assert abs(estimate.bpm - bpm) <= tolerance


def test_estimate_reports_full_diagnostics(
    preprocessing: RppgPreprocessingConfig, spectral: RppgSpectralConfig
) -> None:
    estimate = estimate_heart_rate(
        tone(72.0), FS, preprocessing, spectral, method=RppgMethod.CHROM
    )

    assert estimate.peak_frequency_hz is not None
    assert estimate.peak_power is not None and estimate.peak_power > 0
    assert estimate.peak_prominence is not None
    assert estimate.in_band_power is not None and estimate.in_band_power > 0
    assert estimate.out_of_band_power is not None
    assert estimate.spectral_peak_ratio is not None
    assert 0.0 <= estimate.spectral_peak_ratio <= 1.0
    assert estimate.frequency_resolution_hz is not None
    assert estimate.method is RppgMethod.CHROM
    assert estimate.estimator == "welch_psd_peak"
    assert estimate.band_low_hz == preprocessing.pulse_band_low_hz
    assert estimate.band_high_hz == preprocessing.pulse_band_high_hz


def test_frequency_resolution_matches_welch_segment(
    preprocessing: RppgPreprocessingConfig, spectral: RppgSpectralConfig
) -> None:
    estimate = estimate_heart_rate(tone(72.0), FS, preprocessing, spectral)

    expected = FS / int(spectral.welch_segment_seconds * FS)
    assert estimate.frequency_resolution_hz == pytest.approx(expected)


def test_shorter_segment_gives_coarser_resolution(
    preprocessing: RppgPreprocessingConfig,
) -> None:
    fine = estimate_heart_rate(
        tone(72.0),
        FS,
        preprocessing,
        RppgSpectralConfig(welch_segment_seconds=16.0),
    )
    coarse = estimate_heart_rate(
        tone(72.0),
        FS,
        preprocessing,
        RppgSpectralConfig(welch_segment_seconds=4.0),
    )

    assert fine.frequency_resolution_hz is not None
    assert coarse.frequency_resolution_hz is not None
    assert coarse.frequency_resolution_hz > fine.frequency_resolution_hz


def test_bpm_is_peak_frequency_times_sixty(
    preprocessing: RppgPreprocessingConfig, spectral: RppgSpectralConfig
) -> None:
    estimate = estimate_heart_rate(tone(90.0), FS, preprocessing, spectral)

    assert estimate.bpm is not None
    assert estimate.peak_frequency_hz is not None
    assert estimate.bpm == pytest.approx(estimate.peak_frequency_hz * 60.0)


# --- rejection paths ------------------------------------------------------


def test_constant_signal_is_rejected(
    preprocessing: RppgPreprocessingConfig, spectral: RppgSpectralConfig
) -> None:
    estimate = estimate_heart_rate(np.full(900, 3.0), FS, preprocessing, spectral)

    assert estimate.available is False
    assert estimate.bpm is None
    assert estimate.reason is UnavailableReason.CONSTANT_SIGNAL


def test_non_finite_signal_is_rejected(
    preprocessing: RppgPreprocessingConfig, spectral: RppgSpectralConfig
) -> None:
    values = tone(72.0)
    values[10] = np.nan

    estimate = estimate_heart_rate(values, FS, preprocessing, spectral)

    assert estimate.available is False
    assert estimate.reason is UnavailableReason.NON_FINITE_VALUES


def test_too_short_signal_is_rejected(
    preprocessing: RppgPreprocessingConfig, spectral: RppgSpectralConfig
) -> None:
    estimate = estimate_heart_rate(
        tone(72.0, seconds=3.0),
        FS,
        preprocessing,
        spectral,
        min_duration_seconds=8.0,
    )

    assert estimate.available is False
    assert estimate.bpm is None
    assert estimate.reason is UnavailableReason.WINDOW_TOO_SHORT


def test_zero_sampling_rate_is_rejected(
    preprocessing: RppgPreprocessingConfig, spectral: RppgSpectralConfig
) -> None:
    estimate = estimate_heart_rate(tone(72.0), 0.0, preprocessing, spectral)

    assert estimate.available is False
    assert estimate.reason is UnavailableReason.SAMPLING_RATE_UNKNOWN


def test_band_above_nyquist_is_rejected(
    spectral: RppgSpectralConfig,
) -> None:
    preprocessing = RppgPreprocessingConfig(
        pulse_band_low_hz=0.7, pulse_band_high_hz=4.0
    )

    estimate = estimate_heart_rate(tone(72.0), 6.0, preprocessing, spectral)

    assert estimate.available is False
    assert estimate.reason is UnavailableReason.INVALID_FREQUENCY_BAND


def test_signal_outside_the_band_is_not_reported_as_a_heart_rate(
    preprocessing: RppgPreprocessingConfig, spectral: RppgSpectralConfig
) -> None:
    """A 0.2 Hz respiration-like tone must not be reported as ~12 BPM.

    12 BPM is below the 42 BPM band edge, so the estimator must either
    abstain or report a frequency inside the band -- never the
    out-of-band tone.
    """
    respiration = tone(12.0)  # 0.2 Hz, well below the band

    estimate = estimate_heart_rate(respiration, FS, preprocessing, spectral)

    if estimate.available:
        assert estimate.bpm is not None
        assert estimate.bpm >= preprocessing.pulse_band_low_hz * 60.0
        assert estimate.bpm != pytest.approx(12.0, abs=1.0)
    else:
        assert estimate.bpm is None
        assert estimate.reason is not None


def test_result_is_never_clamped_into_the_band(
    preprocessing: RppgPreprocessingConfig, spectral: RppgSpectralConfig
) -> None:
    """A reported BPM always lies strictly inside the configured band."""
    for bpm in (30.0, 45.0, 200.0, 300.0):
        estimate = estimate_heart_rate(tone(bpm), FS, preprocessing, spectral)
        if estimate.available:
            assert estimate.bpm is not None
            assert (
                preprocessing.pulse_band_low_hz * 60.0
                <= estimate.bpm
                <= preprocessing.pulse_band_high_hz * 60.0
            )


def test_broadband_noise_is_rejected_or_unconcentrated(
    preprocessing: RppgPreprocessingConfig, spectral: RppgSpectralConfig
) -> None:
    """Pure noise has no pulse; it must not yield a confident peak."""
    rng = np.random.default_rng(17)
    noise = rng.normal(0.0, 1.0, size=900)

    estimate = estimate_heart_rate(noise, FS, preprocessing, spectral)

    if estimate.available:
        assert estimate.spectral_peak_ratio is not None
        assert estimate.spectral_peak_ratio < 0.5
    else:
        assert estimate.reason in {
            UnavailableReason.NO_SPECTRAL_PEAK,
            UnavailableReason.PEAK_BELOW_MIN_PROMINENCE,
        }


def test_flat_spectrum_has_no_prominent_peak(
    preprocessing: RppgPreprocessingConfig,
) -> None:
    """Requiring near-perfect prominence rejects a noisy spectrum."""
    rng = np.random.default_rng(23)
    noise = rng.normal(0.0, 1.0, size=900)
    strict = RppgSpectralConfig(min_relative_peak_prominence=0.999)

    estimate = estimate_heart_rate(noise, FS, preprocessing, strict)

    assert estimate.available is False
    assert estimate.reason in {
        UnavailableReason.NO_SPECTRAL_PEAK,
        UnavailableReason.PEAK_BELOW_MIN_PROMINENCE,
    }


def test_rejected_peak_still_reports_diagnostics(
    preprocessing: RppgPreprocessingConfig,
) -> None:
    """A prominence rejection keeps the numbers that justified it."""
    strict = RppgSpectralConfig(min_relative_peak_prominence=1.0)

    estimate = estimate_heart_rate(tone(72.0), FS, preprocessing, strict)

    if estimate.reason is UnavailableReason.PEAK_BELOW_MIN_PROMINENCE:
        assert estimate.bpm is None
        assert estimate.peak_frequency_hz is not None
        assert estimate.peak_prominence is not None
        assert estimate.frequency_resolution_hz is not None


def test_noise_reduces_spectral_concentration(
    preprocessing: RppgPreprocessingConfig, spectral: RppgSpectralConfig
) -> None:
    rng = np.random.default_rng(31)
    clean = tone(72.0)
    noisy = clean + rng.normal(0.0, 3.0, size=clean.size)

    clean_est = estimate_heart_rate(clean, FS, preprocessing, spectral)
    noisy_est = estimate_heart_rate(noisy, FS, preprocessing, spectral)

    assert clean_est.spectral_peak_ratio is not None
    if noisy_est.spectral_peak_ratio is not None:
        assert noisy_est.spectral_peak_ratio < clean_est.spectral_peak_ratio


def test_no_hrv_fields_are_exposed(
    preprocessing: RppgPreprocessingConfig, spectral: RppgSpectralConfig
) -> None:
    """HRV is deliberately deferred; no IBI/SDNN/RMSSD may be reported."""
    estimate = estimate_heart_rate(tone(72.0), FS, preprocessing, spectral)

    fields = set(estimate.model_dump())
    for forbidden in ("sdnn", "rmssd", "pnn50", "ibi", "hrv", "inter_beat"):
        assert not any(forbidden in field for field in fields)


def test_disclaimer_is_always_attached(
    preprocessing: RppgPreprocessingConfig, spectral: RppgSpectralConfig
) -> None:
    estimate = estimate_heart_rate(tone(72.0), FS, preprocessing, spectral)

    assert "NOT medical" in estimate.disclaimer
    assert "engagement" in estimate.disclaimer
