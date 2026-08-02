"""Preprocessing for rPPG traces: detrending, normalization, filtering.

Every function here is independently testable and raises
:class:`~engagevr.rppg.errors.RppgUnavailable` with an explicit reason
rather than returning a degraded or NaN-filled result.

Filter design
-------------
Band-pass filters are Butterworth designs realised as **second-order
sections** (``scipy.signal.butter(..., output="sos")``) and applied with
``scipy.signal.sosfiltfilt``.  Second-order sections are used instead of
direct-form ``(b, a)`` coefficients because direct-form realisations of
even moderate-order band-pass filters are numerically ill-conditioned at
the narrow relative bandwidths involved here (a 0.7-4.0 Hz band at 30 Hz
is a small fraction of Nyquist), which manifests as a filter that is
silently unstable rather than one that fails loudly.

``sosfiltfilt`` is zero-phase: it filters forward and backward, so the
pulse waveform is not phase-shifted relative to its timestamps.  This is
valid for **offline window processing** only -- it is non-causal and
cannot be used for sample-by-sample streaming.  Because it pads the
signal at both edges, it requires the window to be longer than the
padding length; shorter windows are rejected, not zero-padded.

Pulse band
----------
The default 0.7-4.0 Hz band corresponds to 42-240 BPM, following the
40-240 BPM pulse range used by de Haan & Jeanne (2013,
DOI 10.1109/TBME.2013.2266196).  Frequencies are stored in hertz
throughout; conversion to BPM happens only at the reporting boundary.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, detrend, sosfiltfilt

from engagevr.rppg.errors import RppgUnavailable
from engagevr.schemas.rppg import UnavailableReason

#: Signals whose peak-to-peak range is below this are treated as constant.
_CONSTANT_TOLERANCE = 1e-12


def require_finite(x: npt.NDArray[np.float64]) -> None:
    """Raise unless every element is finite."""
    if x.size == 0:
        raise RppgUnavailable(UnavailableReason.WINDOW_TOO_SHORT, "empty signal")
    if not np.all(np.isfinite(x)):
        raise RppgUnavailable(
            UnavailableReason.NON_FINITE_VALUES,
            "signal contains NaN or infinity",
        )


def require_non_constant(x: npt.NDArray[np.float64]) -> None:
    """Raise when the signal carries no variation to analyse."""
    if float(np.ptp(x)) <= _CONSTANT_TOLERANCE:
        raise RppgUnavailable(
            UnavailableReason.CONSTANT_SIGNAL,
            "signal has zero dynamic range",
        )


def remove_mean(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Subtract the temporal mean along axis 0."""
    require_finite(x)
    return np.asarray(x - np.mean(x, axis=0), dtype=np.float64)


def detrend_linear(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Remove a least-squares linear trend along axis 0.

    Slow illumination drift and camera auto-exposure ramps appear as a
    near-linear baseline over a 30 s window; removing it prevents that
    energy from leaking into the low end of the pulse band.
    """
    require_finite(x)
    if x.shape[0] < 2:
        raise RppgUnavailable(
            UnavailableReason.WINDOW_TOO_SHORT,
            "linear detrending needs at least 2 samples",
        )
    return np.asarray(detrend(x, axis=0, type="linear"), dtype=np.float64)


def normalize_channels(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Divide each channel by its temporal mean.

    This is the temporal normalization step shared by CHROM and POS: it
    removes the stationary skin-colour component and expresses each
    channel as a relative modulation, which makes the projection
    coefficients independent of absolute intensity.
    """
    require_finite(x)
    means = np.mean(x, axis=0)
    if np.any(np.abs(means) < _CONSTANT_TOLERANCE):
        raise RppgUnavailable(
            UnavailableReason.CONSTANT_SIGNAL,
            "a channel has a zero temporal mean; cannot normalize",
        )
    return np.asarray(x / means, dtype=np.float64)


def standardize(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Zero-mean, unit-variance scaling along axis 0."""
    require_finite(x)
    centred = x - np.mean(x, axis=0)
    std = np.std(centred, axis=0)
    if np.any(std < _CONSTANT_TOLERANCE):
        raise RppgUnavailable(
            UnavailableReason.CONSTANT_SIGNAL,
            "a channel has zero standard deviation",
        )
    return np.asarray(centred / std, dtype=np.float64)


def validate_timing(
    timestamps: npt.NDArray[np.float64],
    *,
    max_jitter_s: float,
    min_duration_s: float,
) -> float:
    """Validate timestamp integrity and return the observed sampling rate.

    Raises rather than silently assuming uniform sampling when the
    timestamps are duplicated, reversed, or excessively jittery.
    """
    if timestamps.size < 2:
        raise RppgUnavailable(
            UnavailableReason.WINDOW_TOO_SHORT,
            "fewer than 2 timestamps",
        )
    require_finite(timestamps)

    diffs = np.diff(timestamps)
    if np.any(diffs == 0.0):
        raise RppgUnavailable(
            UnavailableReason.DUPLICATE_TIMESTAMPS,
            f"{int(np.sum(diffs == 0.0))} duplicate timestamps",
        )
    if np.any(diffs < 0.0):
        raise RppgUnavailable(
            UnavailableReason.NON_MONOTONIC_TIMESTAMPS,
            f"{int(np.sum(diffs < 0.0))} reversed timestamps",
        )

    duration = float(timestamps[-1] - timestamps[0])
    if duration < min_duration_s:
        raise RppgUnavailable(
            UnavailableReason.WINDOW_TOO_SHORT,
            f"{duration:.2f}s < required {min_duration_s:.2f}s",
        )

    jitter = float(np.std(diffs))
    if jitter > max_jitter_s:
        raise RppgUnavailable(
            UnavailableReason.EXCESSIVE_TIMESTAMP_JITTER,
            f"interval std {jitter:.4f}s > {max_jitter_s:.4f}s",
        )

    median = float(np.median(diffs))
    if median <= 0.0:
        raise RppgUnavailable(
            UnavailableReason.SAMPLING_RATE_UNKNOWN,
            "non-positive median sampling interval",
        )
    return 1.0 / median


def resample_uniform(
    timestamps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    fs: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Resample onto a uniform grid at ``fs`` Hz by linear interpolation.

    Only called after :func:`validate_timing` has confirmed the timestamps
    are strictly increasing and within the jitter tolerance.  Linear
    interpolation over sub-frame gaps is appropriate here; it is not used
    to fabricate data across long dropouts, which are rejected earlier by
    the valid-frame-percentage check.
    """
    if fs <= 0.0:
        raise RppgUnavailable(
            UnavailableReason.SAMPLING_RATE_UNKNOWN,
            "sampling rate must be positive",
        )
    require_finite(timestamps)
    require_finite(values)
    if timestamps.size != values.shape[0]:
        raise ValueError("timestamps and values must have matching lengths")

    duration = float(timestamps[-1] - timestamps[0])
    n = int(np.floor(duration * fs)) + 1
    if n < 2:
        raise RppgUnavailable(
            UnavailableReason.WINDOW_TOO_SHORT,
            "resampled grid would have fewer than 2 samples",
        )
    grid = timestamps[0] + np.arange(n, dtype=np.float64) / fs

    values_2d = values.reshape(values.shape[0], -1)
    out = np.empty((n, values_2d.shape[1]), dtype=np.float64)
    for c in range(values_2d.shape[1]):
        out[:, c] = np.interp(grid, timestamps, values_2d[:, c])
    return grid, out.reshape((n, *values.shape[1:]))


def design_bandpass_sos(
    low_hz: float,
    high_hz: float,
    fs: float,
    order: int,
) -> npt.NDArray[np.float64]:
    """Design a Butterworth band-pass filter as second-order sections.

    Raises when the band is invalid or when the upper edge reaches the
    Nyquist frequency, rather than letting ``butter`` produce a filter
    whose behaviour at the edge is undefined.
    """
    if fs <= 0.0:
        raise RppgUnavailable(
            UnavailableReason.SAMPLING_RATE_UNKNOWN,
            "sampling rate must be positive",
        )
    if not (0.0 < low_hz < high_hz):
        raise RppgUnavailable(
            UnavailableReason.INVALID_FREQUENCY_BAND,
            f"require 0 < low ({low_hz}) < high ({high_hz})",
        )
    nyquist = fs / 2.0
    if high_hz >= nyquist:
        raise RppgUnavailable(
            UnavailableReason.INVALID_FREQUENCY_BAND,
            f"high {high_hz} Hz >= Nyquist {nyquist} Hz at fs={fs} Hz",
        )
    if order < 1:
        raise RppgUnavailable(
            UnavailableReason.FILTER_NOT_VIABLE,
            "filter order must be >= 1",
        )

    sos = butter(
        order,
        [low_hz / nyquist, high_hz / nyquist],
        btype="bandpass",
        output="sos",
    )
    return np.asarray(sos, dtype=np.float64)


def min_samples_for_sosfiltfilt(sos: npt.NDArray[np.float64]) -> int:
    """Minimum window length ``sosfiltfilt`` can process without error.

    ``sosfiltfilt`` pads by ``3 * (2 * n_sections + 1) - 1`` samples by
    default and requires the input to be strictly longer than that pad.
    """
    n_sections = int(sos.shape[0])
    padlen = 3 * (2 * n_sections + 1) - 1
    return padlen + 1


def bandpass_filter(
    x: npt.NDArray[np.float64],
    sos: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Zero-phase band-pass filter for offline window processing.

    Rejects windows shorter than the filter's padding requirement rather
    than shortening the pad, which would apply a different effective
    filter to short windows than to long ones.
    """
    require_finite(x)
    required = min_samples_for_sosfiltfilt(sos)
    if x.shape[0] < required:
        raise RppgUnavailable(
            UnavailableReason.FILTER_NOT_VIABLE,
            f"{x.shape[0]} samples < {required} required for zero-phase "
            f"filtering with {sos.shape[0]} second-order sections",
        )
    out = sosfiltfilt(sos, x, axis=0)
    result = np.asarray(out, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise RppgUnavailable(
            UnavailableReason.FILTER_NOT_VIABLE,
            "filter produced non-finite output",
        )
    return result


def hz_to_bpm(hz: float) -> float:
    """Convert a frequency in hertz to beats per minute."""
    return hz * 60.0


def bpm_to_hz(bpm: float) -> float:
    """Convert beats per minute to a frequency in hertz."""
    return bpm / 60.0
