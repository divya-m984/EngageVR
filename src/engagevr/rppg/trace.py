"""Timestamped RGB trace extraction and synthetic trace generation.

An RGB trace is the sequence of spatially-averaged red, green, and blue
values taken from the skin ROI on each frame, together with the frame's
timestamps.  This is the raw input to every classical rPPG method
(Verkruysse, Svaasand & Nelson, 2008, DOI 10.1364/OE.16.021434).

Missing-data policy
-------------------
Frames without a usable ROI are recorded as samples with ``valid=False``
and no channel values.  They are **never** replaced with zero, with the
previous value, or with an interpolated value at this layer.  Downstream
resampling operates only on the valid subset, and the missing percentage
is carried forward into the quality report.

Timing integrity
----------------
Every window records its observed sampling rate, inter-sample jitter,
and any duplicate or reversed timestamps.  A trace whose timestamps are
not strictly increasing is not silently treated as uniformly sampled.
"""

from __future__ import annotations

import itertools
import math
from datetime import datetime

import numpy as np
import numpy.typing as npt

from engagevr.schemas.rppg import (
    SYNTHETIC_LABEL,
    RgbTraceSample,
    RgbTraceWindow,
    RoiObservation,
    UnavailableReason,
)
from engagevr.schemas.session import DataSource

#: Channel ranges are 8-bit; anything outside is a programming error.
_CHANNEL_MIN = 0.0
_CHANNEL_MAX = 255.0


def make_sample(
    frame_index: int,
    monotonic_timestamp: float,
    roi: RoiObservation,
    mean_rgb: tuple[float, float, float] | None,
    *,
    utc_timestamp: datetime | None = None,
) -> RgbTraceSample:
    """Build one trace sample from a ROI observation and its mean RGB.

    Non-finite or out-of-range channel values are rejected rather than
    stored, because they would silently poison every downstream filter.
    """
    if mean_rgb is None or not roi.available:
        return RgbTraceSample(
            frame_index=frame_index,
            monotonic_timestamp=monotonic_timestamp,
            utc_timestamp=utc_timestamp,
            valid=False,
            reason=roi.reason or UnavailableReason.ROI_EMPTY,
            roi_valid_pixel_count=roi.valid_pixel_count,
        )

    if not all(math.isfinite(c) for c in mean_rgb):
        return RgbTraceSample(
            frame_index=frame_index,
            monotonic_timestamp=monotonic_timestamp,
            utc_timestamp=utc_timestamp,
            valid=False,
            reason=UnavailableReason.NON_FINITE_VALUES,
            roi_valid_pixel_count=roi.valid_pixel_count,
        )

    if not all(_CHANNEL_MIN <= c <= _CHANNEL_MAX for c in mean_rgb):
        return RgbTraceSample(
            frame_index=frame_index,
            monotonic_timestamp=monotonic_timestamp,
            utc_timestamp=utc_timestamp,
            valid=False,
            reason=UnavailableReason.NON_FINITE_VALUES,
            roi_valid_pixel_count=roi.valid_pixel_count,
        )

    return RgbTraceSample(
        frame_index=frame_index,
        monotonic_timestamp=monotonic_timestamp,
        utc_timestamp=utc_timestamp,
        valid=True,
        r=mean_rgb[0],
        g=mean_rgb[1],
        b=mean_rgb[2],
        roi_valid_pixel_count=roi.valid_pixel_count,
        roi_mean_brightness=roi.mean_brightness,
    )


def check_timestamps(timestamps: list[float]) -> tuple[bool, int, int]:
    """Inspect a timestamp sequence for ordering defects.

    Returns
    -------
    (monotonic, n_duplicates, n_reversed)
        ``monotonic`` is True only when timestamps are *strictly*
        increasing, so duplicates count as a violation.
    """
    n_dup = 0
    n_rev = 0
    for prev, curr in itertools.pairwise(timestamps):
        if curr == prev:
            n_dup += 1
        elif curr < prev:
            n_rev += 1
    return (n_dup == 0 and n_rev == 0), n_dup, n_rev


def estimate_sampling_rate(timestamps: list[float]) -> float | None:
    """Estimate sampling rate (Hz) from the median inter-sample interval.

    The median is used rather than the mean so that a small number of
    dropped frames does not bias the estimate.  Returns ``None`` when
    fewer than two samples exist or the median interval is not positive.
    """
    if len(timestamps) < 2:
        return None
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
    if diffs.size == 0:
        return None
    median = float(np.median(diffs))
    if not math.isfinite(median) or median <= 0.0:
        return None
    return 1.0 / median


def timestamp_jitter(timestamps: list[float]) -> float | None:
    """Standard deviation of inter-sample intervals, in seconds."""
    if len(timestamps) < 3:
        return None
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
    value = float(np.std(diffs))
    return value if math.isfinite(value) else None


def build_window(
    samples: list[RgbTraceSample],
    *,
    session_id: str = "",
    window_index: int = 0,
    data_source: DataSource = DataSource.LIVE,
    synthetic_label: str | None = None,
    mean_capture_motion_score: float | None = None,
) -> RgbTraceWindow:
    """Assemble samples into a window with full timing diagnostics."""
    n = len(samples)
    valid = [s for s in samples if s.valid]
    n_valid = len(valid)
    n_missing = n - n_valid

    all_ts = [s.monotonic_timestamp for s in samples]
    valid_ts = [s.monotonic_timestamp for s in valid]

    monotonic, n_dup, n_rev = check_timestamps(all_ts)

    start = min(all_ts) if all_ts else None
    end = max(all_ts) if all_ts else None
    duration = (end - start) if (start is not None and end is not None) else 0.0

    return RgbTraceWindow(
        session_id=session_id,
        data_source=data_source,
        synthetic_label=synthetic_label,
        window_index=window_index,
        samples=samples,
        start_monotonic=start,
        end_monotonic=end,
        duration_s=max(0.0, duration),
        n_samples=n,
        n_valid=n_valid,
        n_missing=n_missing,
        missing_pct=(100.0 * n_missing / n) if n else 0.0,
        observed_fps=estimate_sampling_rate(valid_ts),
        timestamp_jitter_s=timestamp_jitter(valid_ts),
        timestamps_monotonic=monotonic,
        n_duplicate_timestamps=n_dup,
        n_reversed_timestamps=n_rev,
        mean_capture_motion_score=mean_capture_motion_score,
    )


def window_arrays(
    window: RgbTraceWindow,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(timestamps, rgb)`` arrays over the valid samples only.

    ``rgb`` has shape ``(N, 3)`` in R, G, B column order.
    """
    valid = [s for s in window.samples if s.valid]
    timestamps = np.asarray([s.monotonic_timestamp for s in valid], dtype=np.float64)
    rgb = np.asarray(
        [[s.r, s.g, s.b] for s in valid],
        dtype=np.float64,
    ).reshape(-1, 3)
    return timestamps, rgb


def iter_windows(
    samples: list[RgbTraceSample],
    *,
    duration_seconds: float,
    step_seconds: float,
    session_id: str = "",
    data_source: DataSource = DataSource.LIVE,
    synthetic_label: str | None = None,
) -> list[RgbTraceWindow]:
    """Split a sample sequence into overlapping fixed-duration windows.

    Windows are cut on the monotonic timestamp axis, not by sample count,
    so a dropped frame shortens the sample count rather than shifting the
    window in time.  A trailing partial window is discarded.
    """
    if duration_seconds <= 0.0 or step_seconds <= 0.0:
        raise ValueError("duration_seconds and step_seconds must be positive")
    if not samples:
        return []

    timestamps = [s.monotonic_timestamp for s in samples]
    t0 = min(timestamps)
    t_end = max(timestamps)

    windows: list[RgbTraceWindow] = []
    index = 0
    start = t0
    while start + duration_seconds <= t_end + 1e-9:
        stop = start + duration_seconds
        chunk = [s for s in samples if start <= s.monotonic_timestamp < stop]
        if chunk:
            windows.append(
                build_window(
                    chunk,
                    session_id=session_id,
                    window_index=index,
                    data_source=data_source,
                    synthetic_label=synthetic_label,
                )
            )
            index += 1
        start += step_seconds
    return windows


# --- Synthetic trace generation -------------------------------------------
#
# SYNTHETIC DATA. Generated traces are software test fixtures. They are
# not measurements, not evidence, and must never be reported as
# validation of rPPG accuracy.


#: Relative pulse amplitude per channel.
#:
#: Verkruysse et al. (2008) report that the plethysmographic modulation is
#: strongest in the green channel and weakest in blue, reflecting the
#: absorption spectrum of haemoglobin.  These ratios encode that ordering
#: for test-fixture purposes only; they are not fitted to any measurement.
_PULSE_CHANNEL_GAIN: tuple[float, float, float] = (0.45, 1.0, 0.35)

#: Baseline skin-like RGB level for generated traces (arbitrary, mid-range).
_BASELINE_RGB: tuple[float, float, float] = (150.0, 110.0, 100.0)


def generate_synthetic_rgb_trace(
    *,
    bpm: float,
    duration_seconds: float,
    fps: float,
    seed: int = 42,
    pulse_amplitude: float = 1.5,
    illumination_drift: float = 0.02,
    illumination_flicker_hz: float = 0.1,
    noise_std: float = 0.4,
    motion_artifact_rate: float = 0.02,
    motion_artifact_amplitude: float = 6.0,
    dropout_rate: float = 0.0,
    session_id: str = "",
    start_monotonic: float = 0.0,
) -> list[RgbTraceSample]:
    """Generate a deterministic SYNTHETIC RGB trace with a known pulse rate.

    The generated signal follows the additive structure of the skin
    reflection model used by Wang et al. (2017,
    DOI 10.1109/TBME.2016.2609282): a stationary skin colour, a slowly
    varying multiplicative illumination term, an additive pulsatile term
    with channel-dependent gain, and sensor noise.  Motion artifacts are
    injected as short broadband bursts.

    Parameters
    ----------
    bpm:
        Ground-truth pulse rate of the generated signal, in beats per
        minute.  Must be positive.
    duration_seconds, fps:
        Trace length and uniform sampling rate.  Both must be positive.
    seed:
        Seed for the local generator.  The same seed always yields the
        same trace; no global RNG state is touched.
    dropout_rate:
        Probability that a frame is emitted as ``valid=False`` with no
        channel values, simulating a lost ROI.

    Returns
    -------
    list[RgbTraceSample]
        SYNTHETIC samples.  Label the enclosing window with
        ``synthetic_label=SYNTHETIC_LABEL`` -- see
        :func:`build_synthetic_window`.
    """
    if bpm <= 0.0:
        raise ValueError("bpm must be positive")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")

    rng = np.random.default_rng(seed)
    n = round(duration_seconds * fps)
    if n < 1:
        raise ValueError("duration_seconds * fps must yield at least one sample")

    t = np.arange(n, dtype=np.float64) / fps
    pulse_hz = bpm / 60.0

    # Slow illumination variation: a low-frequency sinusoid plus a linear
    # drift, applied multiplicatively to the baseline skin colour.
    illumination = (
        1.0
        + illumination_drift * np.sin(2.0 * np.pi * illumination_flicker_hz * t)
        + illumination_drift * 0.5 * (t / max(t[-1], 1e-9))
    )

    # Pulsatile term with a second harmonic, as real PPG waveforms are not
    # pure sinusoids.
    pulse = np.sin(2.0 * np.pi * pulse_hz * t) + 0.25 * np.sin(
        4.0 * np.pi * pulse_hz * t
    )

    # Motion artifacts: short bursts applied equally across channels.
    motion = np.zeros(n, dtype=np.float64)
    if motion_artifact_rate > 0.0 and motion_artifact_amplitude > 0.0:
        burst_mask = rng.random(n) < motion_artifact_rate
        burst_len = max(1, round(0.2 * fps))
        for raw_idx in np.flatnonzero(burst_mask):
            idx = int(raw_idx)
            stop = min(n, idx + burst_len)
            shape = np.hanning(2 * (stop - idx) + 1)[(stop - idx) :]
            motion[idx:stop] += (
                motion_artifact_amplitude * rng.normal() * shape[: stop - idx]
            )

    samples: list[RgbTraceSample] = []
    drops = rng.random(n) < dropout_rate
    noise = rng.normal(0.0, noise_std, size=(n, 3)) if noise_std > 0 else None

    for i in range(n):
        timestamp = start_monotonic + float(t[i])
        if drops[i]:
            samples.append(
                RgbTraceSample(
                    frame_index=i,
                    monotonic_timestamp=timestamp,
                    valid=False,
                    reason=UnavailableReason.ROI_EMPTY,
                )
            )
            continue

        channels: list[float] = []
        for c in range(3):
            value = (
                _BASELINE_RGB[c] * illumination[i]
                + pulse_amplitude * _PULSE_CHANNEL_GAIN[c] * pulse[i]
                + motion[i]
            )
            if noise is not None:
                value += noise[i, c]
            channels.append(float(np.clip(value, _CHANNEL_MIN, _CHANNEL_MAX)))

        samples.append(
            RgbTraceSample(
                frame_index=i,
                monotonic_timestamp=timestamp,
                valid=True,
                r=channels[0],
                g=channels[1],
                b=channels[2],
                roi_valid_pixel_count=1000,
                roi_mean_brightness=float(np.mean(channels)),
            )
        )

    return samples


def build_synthetic_window(
    samples: list[RgbTraceSample],
    *,
    session_id: str = "",
    window_index: int = 0,
) -> RgbTraceWindow:
    """Wrap synthetic samples in a permanently SYNTHETIC-labelled window."""
    return build_window(
        samples,
        session_id=session_id,
        window_index=window_index,
        data_source=DataSource.SYNTHETIC,
        synthetic_label=SYNTHETIC_LABEL,
    )
