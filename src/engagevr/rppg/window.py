"""Per-window rPPG orchestration.

Pipeline for one window
-----------------------
1. Extract the valid RGB samples and their timestamps.
2. Validate timing integrity and derive the sampling rate.  Reject the
   window if timestamps are duplicated, reversed, or too jittery.
3. Resample onto a uniform grid (optional, and only permitted once the
   jitter check has passed).
4. Detrend and normalize the channels.
5. Run the selected rPPG method and band-pass its output.
6. Estimate the heart rate spectrally.
7. Assess signal quality.
8. **Gate**: if quality is unacceptable, replace the heart-rate estimate
   with an explicit unavailable result.

Step 8 is the point of the whole milestone: an unreliable window returns
``unavailable``, not a confident-looking number.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from engagevr.config import RppgConfig
from engagevr.rppg.errors import RppgUnavailable
from engagevr.rppg.heart_rate import estimate_heart_rate
from engagevr.rppg.methods import extract_waveform
from engagevr.rppg.preprocessing import (
    detrend_linear,
    normalize_channels,
    resample_uniform,
    validate_timing,
)
from engagevr.rppg.quality import assess_window_quality
from engagevr.rppg.trace import window_arrays
from engagevr.schemas.rppg import (
    HeartRateEstimate,
    RgbTraceWindow,
    RppgMethod,
    RppgMethodResult,
    RppgQualityComponent,
    RppgQualityReport,
    RppgWaveform,
    UnavailableReason,
)


def _empty_quality(
    window: RgbTraceWindow,
    cfg: RppgConfig,
    reason: UnavailableReason,
    detail: str,
    session_id: str,
) -> RppgQualityReport:
    """Quality report for a window rejected before any processing."""
    return RppgQualityReport(
        session_id=session_id,
        window_index=window.window_index,
        monotonic_timestamp=window.start_monotonic,
        components=[
            RppgQualityComponent(
                name="preprocessing_viability",
                score=0.0,
                value=None,
                passed=False,
                is_gate=True,
                detail=detail,
            )
        ],
        overall_quality=0.0,
        acceptable=False,
        quality_threshold=cfg.quality.threshold,
        warnings=[
            f"Window rejected before processing: {detail}",
            "Signal quality is insufficient. This means the camera signal is "
            "unreliable. It does NOT mean low engagement or high cognitive load.",
        ],
        rejection_reasons=[reason],
    )


def prepare_window(
    window: RgbTraceWindow,
    cfg: RppgConfig,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Validate timing and produce a clean, uniformly-sampled RGB matrix.

    Raises
    ------
    RppgUnavailable
        When the window cannot be preprocessed safely.
    """
    valid_pct = 100.0 * window.n_valid / window.n_samples if window.n_samples else 0.0
    if valid_pct < cfg.trace.min_valid_frame_pct:
        raise RppgUnavailable(
            UnavailableReason.TOO_FEW_VALID_FRAMES,
            f"{valid_pct:.1f}% valid frames < {cfg.trace.min_valid_frame_pct}%",
        )

    timestamps, rgb = window_arrays(window)
    fs = validate_timing(
        timestamps,
        max_jitter_s=cfg.trace.max_timestamp_jitter_s,
        min_duration_s=cfg.window.min_duration_seconds,
    )

    if cfg.preprocessing.resample_uniform:
        timestamps, rgb = resample_uniform(timestamps, rgb, fs)

    if cfg.preprocessing.detrend:
        rgb = detrend_linear(rgb)
        # Detrending removes the mean, so restore a positive offset before
        # the multiplicative normalization step, which needs a non-zero
        # temporal mean per channel.
        _, original = window_arrays(window)
        rgb = rgb + np.mean(original, axis=0)

    if cfg.preprocessing.normalize:
        rgb = normalize_channels(rgb)

    return timestamps, rgb, fs


def process_window(
    window: RgbTraceWindow,
    cfg: RppgConfig,
    *,
    method: RppgMethod | None = None,
    comparison_method: RppgMethod | None = None,
    session_id: str = "",
) -> RppgMethodResult:
    """Run the full rPPG pipeline over one window.

    Parameters
    ----------
    method:
        Overrides ``cfg.method`` for this call.
    comparison_method:
        Optional second method run purely to produce a method-agreement
        quality component.  Its own result is not returned.

    Returns
    -------
    RppgMethodResult
        Always populated.  When the window is unusable, the waveform and
        heart rate carry ``available=False`` with an explicit reason and
        the quality report carries the rejection reasons.
    """
    selected = method if method is not None else cfg.method

    try:
        timestamps, rgb, fs = prepare_window(window, cfg)
    except RppgUnavailable as exc:
        waveform = RppgWaveform(
            method=selected,
            available=False,
            reason=exc.reason,
            band_low_hz=cfg.preprocessing.pulse_band_low_hz,
            band_high_hz=cfg.preprocessing.pulse_band_high_hz,
            filter_order=cfg.preprocessing.filter_order,
        )
        quality = _empty_quality(window, cfg, exc.reason, str(exc), session_id)
        return RppgMethodResult(
            method=selected,
            method_params={"stage": "preprocessing"},
            waveform=waveform,
            quality=quality,
            heart_rate=HeartRateEstimate(
                available=False,
                bpm=None,
                reason=exc.reason,
                method=selected,
                band_low_hz=cfg.preprocessing.pulse_band_low_hz,
                band_high_hz=cfg.preprocessing.pulse_band_high_hz,
            ),
        )

    waveform = extract_waveform(selected, rgb, timestamps, fs, cfg.preprocessing)

    if waveform.available:
        heart_rate = estimate_heart_rate(
            np.asarray(waveform.values, dtype=np.float64),
            fs,
            cfg.preprocessing,
            cfg.spectral,
            method=selected,
            min_duration_seconds=cfg.window.min_duration_seconds,
        )
    else:
        heart_rate = HeartRateEstimate(
            available=False,
            bpm=None,
            reason=waveform.reason,
            method=selected,
            band_low_hz=cfg.preprocessing.pulse_band_low_hz,
            band_high_hz=cfg.preprocessing.pulse_band_high_hz,
        )

    comparison_bpm: float | None = None
    if comparison_method is not None and comparison_method != selected:
        other = extract_waveform(
            comparison_method, rgb, timestamps, fs, cfg.preprocessing
        )
        if other.available:
            other_hr = estimate_heart_rate(
                np.asarray(other.values, dtype=np.float64),
                fs,
                cfg.preprocessing,
                cfg.spectral,
                method=comparison_method,
                min_duration_seconds=cfg.window.min_duration_seconds,
            )
            comparison_bpm = other_hr.bpm

    quality = assess_window_quality(
        window,
        waveform,
        heart_rate,
        cfg,
        comparison_bpm=comparison_bpm,
        session_id=session_id,
    )

    # Quality gate: an unreliable window must not report a heart rate.
    if not quality.acceptable and heart_rate.available:
        heart_rate = heart_rate.model_copy(
            update={
                "available": False,
                "bpm": None,
                "reason": UnavailableReason.INSUFFICIENT_SIGNAL_QUALITY,
            }
        )

    return RppgMethodResult(
        method=selected,
        method_params=waveform.method_params,
        waveform=waveform,
        quality=quality,
        heart_rate=heart_rate,
    )
