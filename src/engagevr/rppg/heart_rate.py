"""Interpretable spectral heart-rate estimation from an rPPG waveform.

Method
------
The power spectral density is estimated with Welch's method
(``scipy.signal.welch``): the window is split into overlapping segments,
each is Hann-tapered and transformed, and the periodograms are averaged.
Averaging trades frequency resolution for variance reduction, which is
the right trade for a noisy camera-derived signal where a single
periodogram is dominated by its own estimation variance.

The pulse peak is then located as the largest PSD value **within the
configured band only**.  Searching outside the band would allow
respiration (roughly 0.2-0.4 Hz), illumination flicker, or slow motion
drift to be reported as a heart rate.

The peak frequency is converted to beats per minute by ``bpm = f * 60``.

What this is not
----------------
This is a pulse-rate estimate from a camera signal.  It is not a
medical measurement, has not been validated against a reference device
in this repository, and carries no information about stress, anxiety,
engagement, fatigue, or cognitive load.

Deferred: HRV and inter-beat intervals
--------------------------------------
No HRV, IBI, SDNN, RMSSD, or pNN50 value is computed in this milestone.
Time-domain HRV requires beat-to-beat interval accuracy on the order of
milliseconds, which in turn requires validated individual peak detection
on a waveform whose morphology is trustworthy.  A spectral pulse-rate
estimate provides no such per-beat timing, and deriving intervals from
an unvalidated camera waveform would produce numbers that look precise
and mean nothing.  The minimum duration, sampling-rate, waveform-quality,
and peak-validation requirements must be established from primary
literature before any HRV feature is implemented.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.signal import peak_prominences, welch

from engagevr.config import RppgPreprocessingConfig, RppgSpectralConfig
from engagevr.rppg.errors import RppgUnavailable
from engagevr.rppg.preprocessing import hz_to_bpm, require_finite, require_non_constant
from engagevr.schemas.rppg import HeartRateEstimate, RppgMethod, UnavailableReason

_EPS = 1e-15


def _unavailable(
    reason: UnavailableReason,
    method: RppgMethod | None,
    band: tuple[float, float],
) -> HeartRateEstimate:
    return HeartRateEstimate(
        available=False,
        bpm=None,
        reason=reason,
        method=method,
        band_low_hz=band[0],
        band_high_hz=band[1],
    )


def estimate_heart_rate(
    values: npt.NDArray[np.float64],
    fs: float,
    preprocessing: RppgPreprocessingConfig,
    spectral: RppgSpectralConfig,
    *,
    method: RppgMethod | None = None,
    min_duration_seconds: float = 8.0,
) -> HeartRateEstimate:
    """Estimate pulse rate from a processed rPPG waveform.

    Returns an ``available=False`` estimate with an explicit reason for
    constant, non-finite, too-short, or spectrally-unconvincing signals.
    The result is never clamped into a plausible-looking range: if no
    acceptable in-band peak exists, the estimate is withheld.
    """
    band = (preprocessing.pulse_band_low_hz, preprocessing.pulse_band_high_hz)

    if fs <= 0.0:
        return _unavailable(UnavailableReason.SAMPLING_RATE_UNKNOWN, method, band)
    if band[0] >= band[1] or band[1] >= fs / 2.0:
        return _unavailable(UnavailableReason.INVALID_FREQUENCY_BAND, method, band)

    try:
        require_finite(values)
        require_non_constant(values)
    except RppgUnavailable as exc:
        return _unavailable(exc.reason, method, band)

    n = int(values.size)
    duration = n / fs
    if duration < min_duration_seconds:
        return _unavailable(UnavailableReason.WINDOW_TOO_SHORT, method, band)

    # Welch segmentation. The segment must fit inside the window and must
    # be long enough to resolve the low band edge.
    nperseg = round(spectral.welch_segment_seconds * fs)
    nperseg = min(nperseg, n)
    if nperseg < 2:
        return _unavailable(UnavailableReason.WINDOW_TOO_SHORT, method, band)
    noverlap = round(nperseg * spectral.welch_overlap)
    noverlap = min(noverlap, nperseg - 1)

    freqs, psd = welch(
        values,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)

    if freqs.size < 2 or not np.all(np.isfinite(psd)):
        return _unavailable(UnavailableReason.NO_SPECTRAL_PEAK, method, band)

    resolution = float(freqs[1] - freqs[0])
    if resolution <= 0.0:
        return _unavailable(UnavailableReason.NO_SPECTRAL_PEAK, method, band)

    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(in_band):
        # The band is narrower than one frequency bin at this resolution.
        return _unavailable(UnavailableReason.WINDOW_TOO_SHORT, method, band)

    band_indices = np.flatnonzero(in_band)
    peak_index = int(band_indices[int(np.argmax(psd[band_indices]))])
    peak_power = float(psd[peak_index])
    peak_freq = float(freqs[peak_index])

    if peak_power <= _EPS:
        return _unavailable(UnavailableReason.NO_SPECTRAL_PEAK, method, band)

    # A maximum that is not a local maximum is a band-edge artifact, not a
    # pulse peak. Report it explicitly rather than accepting the edge.
    if peak_index == 0 or peak_index == freqs.size - 1:
        return _unavailable(UnavailableReason.NO_SPECTRAL_PEAK, method, band)
    if not (
        psd[peak_index] > psd[peak_index - 1] and psd[peak_index] > psd[peak_index + 1]
    ):
        return _unavailable(UnavailableReason.NO_SPECTRAL_PEAK, method, band)

    prominences = peak_prominences(psd, np.array([peak_index]))[0]
    prominence = float(prominences[0])

    # Power bookkeeping over the full computed spectrum.
    total_power = float(np.trapezoid(psd, freqs))
    in_band_power = float(np.trapezoid(psd[in_band], freqs[in_band]))
    out_of_band_power = max(0.0, total_power - in_band_power)

    # Spectral concentration: how much of the in-band power sits within
    # +/- peak_bandwidth_hz of the selected peak.
    near_peak = (
        in_band
        & (freqs >= peak_freq - spectral.peak_bandwidth_hz)
        & (freqs <= peak_freq + spectral.peak_bandwidth_hz)
    )
    if np.count_nonzero(near_peak) >= 2:
        near_power = float(np.trapezoid(psd[near_peak], freqs[near_peak]))
    else:
        near_power = peak_power * resolution
    peak_ratio = (
        float(np.clip(near_power / in_band_power, 0.0, 1.0))
        if in_band_power > _EPS
        else 0.0
    )

    relative_prominence = prominence / peak_power if peak_power > _EPS else 0.0

    diagnostics = {
        "peak_frequency_hz": peak_freq,
        "peak_power": peak_power,
        "peak_prominence": prominence,
        "in_band_power": in_band_power,
        "out_of_band_power": out_of_band_power,
        "spectral_peak_ratio": peak_ratio,
        "frequency_resolution_hz": resolution,
        "band_low_hz": band[0],
        "band_high_hz": band[1],
        "method": method,
    }

    if relative_prominence < spectral.min_relative_peak_prominence:
        # Report the full diagnostics alongside the refusal, so the caller
        # can see how close the peak came to the acceptance criterion.
        return HeartRateEstimate(
            available=False,
            bpm=None,
            reason=UnavailableReason.PEAK_BELOW_MIN_PROMINENCE,
            **diagnostics,
        )

    bpm = hz_to_bpm(peak_freq)
    if not math.isfinite(bpm):  # pragma: no cover - defensive
        return _unavailable(UnavailableReason.NO_SPECTRAL_PEAK, method, band)

    return HeartRateEstimate(
        available=True,
        bpm=bpm,
        reason=None,
        **diagnostics,
    )
