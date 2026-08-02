"""Interpretable rPPG signal-quality index.

Design
------
Quality is reported as a list of named, independently-computed
components, each with its raw measured value and a 0-1 score.  The raw
value is always retained so that a reader can see *why* a component
scored what it did rather than having to trust an opaque number.

Aggregation is the **unweighted arithmetic mean of the components that
could actually be computed**.  Equal weighting is a deliberate choice:
this repository has no validated empirical basis for ranking these
components against one another, and hand-picked weights would be exactly
the kind of arbitrary unexplained constant the project rules forbid.
Components that could not be computed for a window are omitted from the
mean rather than imputed, so a missing motion score does not silently
drag the average down.

Separately from the mean, some components are **gates**.  A failed gate
forces the window to be unacceptable regardless of how well every other
component scored, because these conditions make the estimate invalid
rather than merely noisy: non-monotonic timestamps, an unusable filter,
insufficient window duration, and too few valid frames.  A gate failure
cannot be averaged away.

What quality is not
-------------------
Signal quality describes the camera measurement, not the person.  A low
score means "this signal cannot be trusted", never "this person is
disengaged", "this person is stressed", or any other statement about
their state.  rPPG quality is also kept strictly separate from model
confidence: they answer different questions and are combined, if at all,
only by a downstream policy layer that treats them as distinct inputs.
"""

from __future__ import annotations

import math

import numpy as np

from engagevr.config import RppgConfig
from engagevr.schemas.rppg import (
    HeartRateEstimate,
    RgbTraceWindow,
    RppgQualityComponent,
    RppgQualityReport,
    RppgWaveform,
    UnavailableReason,
)


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _linear_penalty(value: float, worst: float) -> float:
    """Map 0 -> 1.0 and ``worst`` (or above) -> 0.0, linearly."""
    if worst <= 0.0:
        return 0.0
    return _clamp01(1.0 - value / worst)


def assess_window_quality(
    window: RgbTraceWindow,
    waveform: RppgWaveform,
    heart_rate: HeartRateEstimate,
    cfg: RppgConfig,
    *,
    comparison_bpm: float | None = None,
    session_id: str = "",
) -> RppgQualityReport:
    """Build the quality report for one processed window.

    Parameters
    ----------
    comparison_bpm:
        BPM from a second, independently-run rPPG method.  When supplied,
        a method-agreement component is added.  Agreement between two
        methods is weak corroboration only -- both can be wrong in the
        same way, since both read the same corrupted pixels.
    """
    components: list[RppgQualityComponent] = []
    warnings: list[str] = []
    rejections: list[UnavailableReason] = []

    valid_samples = [s for s in window.samples if s.valid]

    # --- 1. ROI availability (gate) ---
    roi_availability = window.n_valid / window.n_samples if window.n_samples else 0.0
    availability_ok = 100.0 * roi_availability >= cfg.trace.min_valid_frame_pct
    components.append(
        RppgQualityComponent(
            name="roi_availability",
            score=_clamp01(roi_availability),
            value=100.0 * roi_availability,
            passed=availability_ok,
            is_gate=True,
            detail=(
                f"{window.n_valid}/{window.n_samples} frames had a usable ROI "
                f"(minimum {cfg.trace.min_valid_frame_pct}%)"
            ),
        )
    )
    if not availability_ok:
        warnings.append("Too few frames had a usable skin ROI.")
        rejections.append(UnavailableReason.TOO_FEW_VALID_FRAMES)

    # --- 2. ROI valid-pixel coverage ---
    pixel_counts = [
        s.roi_valid_pixel_count for s in valid_samples if s.roi_valid_pixel_count
    ]
    if pixel_counts:
        mean_pixels = float(np.mean(pixel_counts))
        coverage = _clamp01(mean_pixels / max(1.0, float(cfg.roi.min_valid_pixels)))
        components.append(
            RppgQualityComponent(
                name="roi_pixel_coverage",
                score=coverage,
                value=mean_pixels,
                passed=mean_pixels >= cfg.roi.min_valid_pixels,
                detail=(
                    f"mean {mean_pixels:.0f} valid ROI pixels per frame "
                    f"(minimum {cfg.roi.min_valid_pixels})"
                ),
            )
        )

    # --- 3. Missing frames ---
    components.append(
        RppgQualityComponent(
            name="missing_frames",
            score=_clamp01(1.0 - window.missing_pct / 100.0),
            value=window.missing_pct,
            passed=window.missing_pct <= (100.0 - cfg.trace.min_valid_frame_pct),
            detail=f"{window.missing_pct:.1f}% of frames were missing",
        )
    )

    # --- 4. Timestamp monotonicity (gate) ---
    monotonic_ok = window.timestamps_monotonic
    components.append(
        RppgQualityComponent(
            name="timestamp_monotonicity",
            score=1.0 if monotonic_ok else 0.0,
            value=float(window.n_duplicate_timestamps + window.n_reversed_timestamps),
            passed=monotonic_ok,
            is_gate=True,
            detail=(
                f"{window.n_duplicate_timestamps} duplicate, "
                f"{window.n_reversed_timestamps} reversed timestamps"
            ),
        )
    )
    if not monotonic_ok:
        warnings.append("Timestamps are not strictly increasing.")
        if window.n_duplicate_timestamps:
            rejections.append(UnavailableReason.DUPLICATE_TIMESTAMPS)
        if window.n_reversed_timestamps:
            rejections.append(UnavailableReason.NON_MONOTONIC_TIMESTAMPS)

    # --- 5. Sampling jitter ---
    if window.timestamp_jitter_s is not None:
        jitter_ok = window.timestamp_jitter_s <= cfg.trace.max_timestamp_jitter_s
        components.append(
            RppgQualityComponent(
                name="sampling_jitter",
                score=_linear_penalty(
                    window.timestamp_jitter_s,
                    cfg.trace.max_timestamp_jitter_s,
                ),
                value=window.timestamp_jitter_s,
                passed=jitter_ok,
                detail=(
                    f"inter-sample interval std {window.timestamp_jitter_s:.4f}s "
                    f"(tolerance {cfg.trace.max_timestamp_jitter_s:.4f}s)"
                ),
            )
        )
        if not jitter_ok:
            warnings.append("Frame timing is irregular.")
            rejections.append(UnavailableReason.EXCESSIVE_TIMESTAMP_JITTER)

    # --- 6. Illumination stability ---
    brightness = [
        s.roi_mean_brightness
        for s in valid_samples
        if s.roi_mean_brightness is not None
    ]
    if len(brightness) >= 2:
        arr = np.asarray(brightness, dtype=np.float64)
        mean_b = float(np.mean(arr))
        cv = float(np.std(arr) / mean_b) if abs(mean_b) > 1e-9 else 1.0
        illum_ok = cv <= cfg.quality.max_illumination_cv
        components.append(
            RppgQualityComponent(
                name="illumination_stability",
                score=_linear_penalty(cv, cfg.quality.max_illumination_cv),
                value=cv,
                passed=illum_ok,
                detail=(
                    f"ROI brightness coefficient of variation {cv:.4f} "
                    f"(tolerance {cfg.quality.max_illumination_cv:.4f})"
                ),
            )
        )
        if not illum_ok:
            warnings.append("ROI illumination varied substantially across the window.")

    # --- 7. Channel clipping / saturation ---
    channel_values: list[float] = []
    for s in valid_samples:
        channel_values.extend(v for v in (s.r, s.g, s.b) if v is not None)
    if channel_values:
        arr = np.asarray(channel_values, dtype=np.float64)
        clipped = float(
            np.mean(
                (arr <= float(cfg.roi.clipping_low))
                | (arr >= float(cfg.roi.clipping_high))
            )
            * 100.0
        )
        clip_ok = clipped <= cfg.quality.max_clipped_pixel_pct
        components.append(
            RppgQualityComponent(
                name="channel_clipping",
                score=_linear_penalty(clipped, cfg.quality.max_clipped_pixel_pct),
                value=clipped,
                passed=clip_ok,
                detail=(
                    f"{clipped:.1f}% of channel means at or beyond the clipping bounds"
                ),
            )
        )
        if not clip_ok:
            warnings.append("Colour channels are clipped or saturated.")

    # --- 8. Motion, inherited from the capture layer ---
    if window.mean_capture_motion_score is not None:
        motion_ok = window.mean_capture_motion_score <= cfg.quality.max_motion_score
        components.append(
            RppgQualityComponent(
                name="capture_motion",
                score=_linear_penalty(
                    window.mean_capture_motion_score,
                    cfg.quality.max_motion_score,
                ),
                value=window.mean_capture_motion_score,
                passed=motion_ok,
                detail=(
                    "mean capture motion score "
                    f"{window.mean_capture_motion_score:.1f} "
                    f"(tolerance {cfg.quality.max_motion_score:.1f})"
                ),
            )
        )
        if not motion_ok:
            warnings.append("Excessive motion during the window.")

    # --- 9. Filter viability (gate) ---
    filter_ok = waveform.available
    components.append(
        RppgQualityComponent(
            name="filter_viability",
            score=1.0 if filter_ok else 0.0,
            value=None,
            passed=filter_ok,
            is_gate=True,
            detail=(
                "waveform extracted and band-pass filtered"
                if filter_ok
                else f"waveform unavailable: {waveform.reason}"
            ),
        )
    )
    if not filter_ok and waveform.reason is not None:
        warnings.append(f"Waveform could not be produced: {waveform.reason.value}.")
        rejections.append(waveform.reason)

    # --- 10. Window duration (gate) ---
    duration_ok = window.duration_s >= cfg.window.min_duration_seconds
    components.append(
        RppgQualityComponent(
            name="window_duration",
            score=_clamp01(
                window.duration_s / max(1e-9, cfg.window.min_duration_seconds)
            ),
            value=window.duration_s,
            passed=duration_ok,
            is_gate=True,
            detail=(
                f"{window.duration_s:.1f}s window "
                f"(minimum {cfg.window.min_duration_seconds:.1f}s)"
            ),
        )
    )
    if not duration_ok:
        warnings.append("Window is shorter than the minimum analysable duration.")
        rejections.append(UnavailableReason.WINDOW_TOO_SHORT)

    # --- 11. In-band spectral concentration ---
    if heart_rate.spectral_peak_ratio is not None:
        ratio = heart_rate.spectral_peak_ratio
        ratio_ok = ratio >= cfg.spectral.min_spectral_peak_ratio
        components.append(
            RppgQualityComponent(
                name="spectral_concentration",
                score=_clamp01(ratio / max(1e-9, cfg.spectral.min_spectral_peak_ratio)),
                value=ratio,
                passed=ratio_ok,
                detail=(
                    f"{100.0 * ratio:.1f}% of in-band power near the peak "
                    f"(minimum {100.0 * cfg.spectral.min_spectral_peak_ratio:.1f}%)"
                ),
            )
        )
        if not ratio_ok:
            warnings.append("Pulse peak is not spectrally concentrated.")

    # --- 12. Peak prominence ---
    if heart_rate.peak_prominence is not None and heart_rate.peak_power:
        relative = heart_rate.peak_prominence / heart_rate.peak_power
        prom_ok = relative >= cfg.spectral.min_relative_peak_prominence
        components.append(
            RppgQualityComponent(
                name="peak_prominence",
                score=_clamp01(
                    relative / max(1e-9, cfg.spectral.min_relative_peak_prominence)
                ),
                value=relative,
                passed=prom_ok,
                detail=(
                    f"relative peak prominence {relative:.3f} "
                    f"(minimum {cfg.spectral.min_relative_peak_prominence:.3f})"
                ),
            )
        )
        if not prom_ok:
            warnings.append("Spectral peak is not prominent enough.")
            rejections.append(UnavailableReason.PEAK_BELOW_MIN_PROMINENCE)

    # --- 13. Method agreement (optional corroboration) ---
    if comparison_bpm is not None and heart_rate.bpm is not None:
        delta = abs(comparison_bpm - heart_rate.bpm)
        tolerance = cfg.quality.method_agreement_tolerance_bpm
        agree_ok = delta <= tolerance
        components.append(
            RppgQualityComponent(
                name="method_agreement",
                score=_linear_penalty(delta, tolerance),
                value=delta,
                passed=agree_ok,
                detail=(
                    f"{delta:.1f} BPM difference against a second method "
                    f"(tolerance {tolerance:.1f} BPM)"
                ),
            )
        )
        if not agree_ok:
            warnings.append("Independent rPPG methods disagree on the pulse rate.")

    # --- Aggregate ---
    scores = [c.score for c in components]
    overall = float(np.mean(scores)) if scores else 0.0
    gates_passed = all(c.passed for c in components if c.is_gate)
    acceptable = gates_passed and overall >= cfg.quality.threshold

    if not acceptable and not rejections:
        rejections.append(UnavailableReason.INSUFFICIENT_SIGNAL_QUALITY)
    if not acceptable:
        warnings.append(
            "Signal quality is insufficient. This means the camera signal is "
            "unreliable. It does NOT mean low engagement or high cognitive load."
        )

    return RppgQualityReport(
        session_id=session_id,
        window_index=window.window_index,
        monotonic_timestamp=window.start_monotonic,
        components=components,
        overall_quality=_clamp01(overall),
        acceptable=acceptable,
        quality_threshold=cfg.quality.threshold,
        warnings=warnings,
        rejection_reasons=list(dict.fromkeys(rejections)),
    )
