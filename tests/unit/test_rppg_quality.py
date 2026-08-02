"""Tests for the rPPG signal-quality index and the quality gate."""

from __future__ import annotations

import numpy as np
import pytest

from engagevr.config import RppgConfig
from engagevr.rppg.quality import assess_window_quality
from engagevr.rppg.trace import (
    build_synthetic_window,
    build_window,
    generate_synthetic_rgb_trace,
    make_sample,
)
from engagevr.rppg.window import process_window
from engagevr.schemas.rppg import (
    HeartRateEstimate,
    RgbTraceSample,
    RoiObservation,
    RoiRegion,
    RppgMethod,
    RppgWaveform,
    UnavailableReason,
)

FS = 30.0
DURATION = 30.0


@pytest.fixture
def cfg() -> RppgConfig:
    return RppgConfig()


def good_roi(pixels: int = 1000, brightness: float = 120.0) -> RoiObservation:
    return RoiObservation(
        region=RoiRegion.COMBINED,
        available=True,
        total_pixel_count=pixels,
        valid_pixel_count=pixels,
        valid_pixel_pct=100.0,
        mean_brightness=brightness,
    )


def clean_window(bpm: float = 72.0, **kwargs: float):  # type: ignore[no-untyped-def]
    samples = generate_synthetic_rgb_trace(
        bpm=bpm, duration_seconds=DURATION, fps=FS, seed=42, **kwargs
    )
    return build_synthetic_window(samples)


def component(report, name: str):  # type: ignore[no-untyped-def]
    for c in report.components:
        if c.name == name:
            return c
    return None


# --- good signal ----------------------------------------------------------


def test_clean_synthetic_signal_scores_high(cfg: RppgConfig) -> None:
    result = process_window(clean_window(), cfg, method=RppgMethod.POS)

    assert result.quality.acceptable is True
    assert result.quality.overall_quality > 0.7
    assert result.quality.rejection_reasons == []
    assert result.heart_rate.available is True


def test_report_lists_named_components_with_raw_values(
    cfg: RppgConfig,
) -> None:
    result = process_window(clean_window(), cfg, method=RppgMethod.POS)
    names = {c.name for c in result.quality.components}

    for expected in (
        "roi_availability",
        "missing_frames",
        "timestamp_monotonicity",
        "sampling_jitter",
        "illumination_stability",
        "channel_clipping",
        "filter_viability",
        "window_duration",
        "spectral_concentration",
        "peak_prominence",
    ):
        assert expected in names

    for c in result.quality.components:
        assert 0.0 <= c.score <= 1.0


def test_aggregation_is_the_documented_mean(cfg: RppgConfig) -> None:
    """overall_quality must equal the plain mean of component scores."""
    result = process_window(clean_window(), cfg, method=RppgMethod.POS)
    report = result.quality

    expected = float(np.mean([c.score for c in report.components]))
    assert report.overall_quality == pytest.approx(expected)
    assert report.aggregation == "unweighted_mean_of_available_components"


# --- degradation ----------------------------------------------------------


def test_illumination_drift_lowers_stability_score(cfg: RppgConfig) -> None:
    stable = process_window(
        clean_window(illumination_drift=0.0), cfg, method=RppgMethod.POS
    )
    drifting = process_window(
        clean_window(illumination_drift=0.5), cfg, method=RppgMethod.POS
    )

    stable_c = component(stable.quality, "illumination_stability")
    drift_c = component(drifting.quality, "illumination_stability")

    assert stable_c is not None and drift_c is not None
    assert drift_c.score < stable_c.score
    assert drift_c.value > stable_c.value


def test_motion_artifacts_lower_quality(cfg: RppgConfig) -> None:
    calm = process_window(
        clean_window(motion_artifact_rate=0.0), cfg, method=RppgMethod.POS
    )
    shaky = process_window(
        clean_window(motion_artifact_rate=0.5, motion_artifact_amplitude=40.0),
        cfg,
        method=RppgMethod.POS,
    )

    assert shaky.quality.overall_quality < calm.quality.overall_quality


def test_inherited_capture_motion_score_is_scored(cfg: RppgConfig) -> None:
    window = clean_window()
    high_motion = window.model_copy(
        update={"mean_capture_motion_score": cfg.quality.max_motion_score * 2}
    )

    report = assess_window_quality(
        high_motion,
        RppgWaveform(method=RppgMethod.POS, available=True),
        HeartRateEstimate(available=True, bpm=72.0),
        cfg,
    )

    motion = component(report, "capture_motion")
    assert motion is not None
    assert motion.passed is False
    assert motion.score == 0.0


def test_clipped_channels_lower_quality(cfg: RppgConfig) -> None:
    samples = [
        make_sample(i, i / FS, good_roi(), (255.0, 255.0, 255.0))
        for i in range(int(DURATION * FS))
    ]
    window = build_window(samples)

    report = assess_window_quality(
        window,
        RppgWaveform(method=RppgMethod.POS, available=True),
        HeartRateEstimate(available=True, bpm=72.0),
        cfg,
    )

    clipping = component(report, "channel_clipping")
    assert clipping is not None
    assert clipping.passed is False
    assert clipping.score == 0.0
    assert clipping.value == pytest.approx(100.0)


def test_missing_samples_lower_quality_and_gate_the_result(
    cfg: RppgConfig,
) -> None:
    result = process_window(clean_window(dropout_rate=0.6), cfg, method=RppgMethod.POS)

    assert result.quality.acceptable is False
    assert result.heart_rate.available is False
    assert UnavailableReason.TOO_FEW_VALID_FRAMES in result.quality.rejection_reasons


def test_excessive_jitter_is_rejected(cfg: RppgConfig) -> None:
    rng = np.random.default_rng(5)
    intervals = rng.uniform(0.005, 0.15, size=int(DURATION * FS))
    timestamps = np.cumsum(intervals)
    samples = [
        make_sample(i, float(t), good_roi(), (150.0, 110.0, 100.0))
        for i, t in enumerate(timestamps)
    ]

    result = process_window(build_window(samples), cfg, method=RppgMethod.POS)

    assert result.quality.acceptable is False
    assert result.heart_rate.available is False
    assert (
        UnavailableReason.EXCESSIVE_TIMESTAMP_JITTER in result.quality.rejection_reasons
    )


def test_non_monotonic_timestamps_fail_the_gate(cfg: RppgConfig) -> None:
    samples = [
        make_sample(i, i / FS, good_roi(), (150.0, 110.0, 100.0))
        for i in range(int(DURATION * FS))
    ]
    samples[100] = make_sample(
        100, samples[50].monotonic_timestamp, good_roi(), (150.0, 110.0, 100.0)
    )
    window = build_window(samples)

    report = assess_window_quality(
        window,
        RppgWaveform(method=RppgMethod.POS, available=True),
        HeartRateEstimate(available=True, bpm=72.0),
        cfg,
    )

    assert report.acceptable is False
    monotonic = component(report, "timestamp_monotonicity")
    assert monotonic is not None
    assert monotonic.is_gate is True
    assert monotonic.passed is False


def test_low_peak_concentration_lowers_quality(cfg: RppgConfig) -> None:
    """Broadband noise carries no concentrated pulse peak."""
    rng = np.random.default_rng(13)
    samples: list[RgbTraceSample] = []
    for i in range(int(DURATION * FS)):
        values = 120.0 + rng.normal(0.0, 4.0, size=3)
        samples.append(
            make_sample(
                i,
                i / FS,
                good_roi(brightness=float(np.mean(values))),
                (float(values[0]), float(values[1]), float(values[2])),
            )
        )

    result = process_window(build_window(samples), cfg, method=RppgMethod.POS)

    concentration = component(result.quality, "spectral_concentration")
    if concentration is not None:
        assert concentration.score < 1.0


# --- gating behaviour -----------------------------------------------------


def test_a_failed_gate_cannot_be_averaged_away(cfg: RppgConfig) -> None:
    """Otherwise-perfect components must not rescue a failed gate."""
    window = clean_window()
    broken = window.model_copy(
        update={"timestamps_monotonic": False, "n_reversed_timestamps": 1}
    )

    report = assess_window_quality(
        broken,
        RppgWaveform(method=RppgMethod.POS, available=True),
        HeartRateEstimate(available=True, bpm=72.0, spectral_peak_ratio=1.0),
        cfg,
    )

    assert report.acceptable is False
    assert report.overall_quality > cfg.quality.threshold


def test_unacceptable_quality_forces_unavailable_heart_rate() -> None:
    """The central Milestone 3 requirement."""
    cfg = RppgConfig()
    cfg.quality = cfg.quality.model_copy(update={"threshold": 0.999})

    result = process_window(clean_window(), cfg, method=RppgMethod.POS)

    assert result.quality.acceptable is False
    assert result.heart_rate.available is False
    assert result.heart_rate.bpm is None
    assert result.heart_rate.reason is UnavailableReason.INSUFFICIENT_SIGNAL_QUALITY


def test_unavailable_waveform_fails_the_filter_gate(cfg: RppgConfig) -> None:
    window = clean_window()

    report = assess_window_quality(
        window,
        RppgWaveform(
            method=RppgMethod.POS,
            available=False,
            reason=UnavailableReason.CONSTANT_SIGNAL,
        ),
        HeartRateEstimate(available=False, reason=UnavailableReason.CONSTANT_SIGNAL),
        cfg,
    )

    viability = component(report, "filter_viability")
    assert viability is not None
    assert viability.is_gate is True
    assert viability.passed is False
    assert report.acceptable is False
    assert UnavailableReason.CONSTANT_SIGNAL in report.rejection_reasons


def test_method_agreement_component_is_added_when_requested(
    cfg: RppgConfig,
) -> None:
    result = process_window(
        clean_window(),
        cfg,
        method=RppgMethod.POS,
        comparison_method=RppgMethod.CHROM,
    )

    agreement = component(result.quality, "method_agreement")
    assert agreement is not None
    assert agreement.value is not None


def test_method_disagreement_lowers_the_agreement_score(
    cfg: RppgConfig,
) -> None:
    report = assess_window_quality(
        clean_window(),
        RppgWaveform(method=RppgMethod.POS, available=True),
        HeartRateEstimate(available=True, bpm=72.0),
        cfg,
        comparison_bpm=140.0,
    )

    agreement = component(report, "method_agreement")
    assert agreement is not None
    assert agreement.passed is False
    assert agreement.score == 0.0


# --- separation from engagement -------------------------------------------


def test_quality_report_never_mentions_engagement_as_a_conclusion(
    cfg: RppgConfig,
) -> None:
    result = process_window(clean_window(dropout_rate=0.6), cfg)
    report = result.quality

    assert "NOT mean low engagement" in " ".join(report.warnings)
    assert "not engagement" in report.note.lower()

    fields = set(report.model_dump())
    for forbidden in ("engagement", "cognitive_load", "confidence", "stress"):
        assert not any(forbidden in field for field in fields)


def test_quality_is_separate_from_model_confidence(cfg: RppgConfig) -> None:
    """No confidence field may exist on an rPPG quality report."""
    result = process_window(clean_window(), cfg)

    assert "confidence" not in result.quality.model_dump()
    assert "confidence" not in result.heart_rate.model_dump()
