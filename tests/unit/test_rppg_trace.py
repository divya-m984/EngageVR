"""Tests for RGB trace extraction, timing diagnostics, and synthetic traces."""

from __future__ import annotations

import numpy as np
import pytest

from engagevr.rppg.trace import (
    build_synthetic_window,
    build_window,
    check_timestamps,
    estimate_sampling_rate,
    generate_synthetic_rgb_trace,
    iter_windows,
    make_sample,
    timestamp_jitter,
    window_arrays,
)
from engagevr.schemas.rppg import (
    SYNTHETIC_LABEL,
    RgbTraceSample,
    RoiObservation,
    RoiRegion,
    UnavailableReason,
)
from engagevr.schemas.session import DataSource


def good_roi(valid_pixels: int = 1000, brightness: float = 120.0) -> RoiObservation:
    return RoiObservation(
        region=RoiRegion.COMBINED,
        available=True,
        total_pixel_count=valid_pixels,
        valid_pixel_count=valid_pixels,
        valid_pixel_pct=100.0,
        mean_brightness=brightness,
    )


def bad_roi(reason: UnavailableReason) -> RoiObservation:
    return RoiObservation(region=RoiRegion.COMBINED, available=False, reason=reason)


# --- make_sample ----------------------------------------------------------


def test_sample_records_channel_means() -> None:
    sample = make_sample(3, 1.5, good_roi(), (150.0, 110.0, 90.0))

    assert sample.valid is True
    assert sample.r == pytest.approx(150.0)
    assert sample.g == pytest.approx(110.0)
    assert sample.b == pytest.approx(90.0)
    assert sample.frame_index == 3
    assert sample.monotonic_timestamp == pytest.approx(1.5)


def test_missing_roi_yields_invalid_sample_with_no_values() -> None:
    """A missing ROI must never be substituted with zero."""
    sample = make_sample(1, 0.5, bad_roi(UnavailableReason.ROI_TOO_SMALL), None)

    assert sample.valid is False
    assert sample.r is None
    assert sample.g is None
    assert sample.b is None
    assert sample.reason is UnavailableReason.ROI_TOO_SMALL


def test_non_finite_channel_is_rejected() -> None:
    sample = make_sample(0, 0.0, good_roi(), (float("nan"), 100.0, 100.0))

    assert sample.valid is False
    assert sample.reason is UnavailableReason.NON_FINITE_VALUES


def test_out_of_range_channel_is_rejected() -> None:
    sample = make_sample(0, 0.0, good_roi(), (300.0, 100.0, 100.0))

    assert sample.valid is False
    assert sample.reason is UnavailableReason.NON_FINITE_VALUES


# --- timing diagnostics ---------------------------------------------------


def test_strictly_increasing_timestamps_are_monotonic() -> None:
    ok, dup, rev = check_timestamps([0.0, 0.1, 0.2, 0.3])

    assert ok is True
    assert dup == 0
    assert rev == 0


def test_duplicate_timestamps_are_detected() -> None:
    ok, dup, rev = check_timestamps([0.0, 0.1, 0.1, 0.2])

    assert ok is False
    assert dup == 1
    assert rev == 0


def test_reversed_timestamps_are_detected() -> None:
    ok, dup, rev = check_timestamps([0.0, 0.2, 0.1, 0.3])

    assert ok is False
    assert dup == 0
    assert rev == 1


def test_sampling_rate_from_uniform_timestamps() -> None:
    timestamps = [i / 30.0 for i in range(90)]

    assert estimate_sampling_rate(timestamps) == pytest.approx(30.0)


def test_sampling_rate_is_robust_to_a_dropped_frame() -> None:
    """The median interval ignores one long gap."""
    timestamps = [i / 30.0 for i in range(30)]
    timestamps += [t + 0.5 for t in (1.0, 1.1, 1.2)]

    rate = estimate_sampling_rate(timestamps)

    assert rate is not None
    assert rate == pytest.approx(30.0, rel=0.05)


def test_sampling_rate_needs_two_samples() -> None:
    assert estimate_sampling_rate([1.0]) is None
    assert estimate_sampling_rate([]) is None


def test_jitter_is_zero_for_uniform_sampling() -> None:
    timestamps = [i / 30.0 for i in range(60)]

    jitter = timestamp_jitter(timestamps)

    assert jitter is not None
    assert jitter == pytest.approx(0.0, abs=1e-12)


def test_jitter_is_positive_for_irregular_sampling() -> None:
    rng = np.random.default_rng(7)
    timestamps = list(np.cumsum(rng.uniform(0.01, 0.09, size=60)))

    jitter = timestamp_jitter(timestamps)

    assert jitter is not None
    assert jitter > 0.01


# --- windows --------------------------------------------------------------


def test_window_counts_valid_and_missing_samples() -> None:
    samples = [
        make_sample(i, i / 30.0, good_roi(), (150.0, 110.0, 90.0)) for i in range(10)
    ]
    samples[3] = make_sample(3, 3 / 30.0, bad_roi(UnavailableReason.ROI_EMPTY), None)
    samples[7] = make_sample(7, 7 / 30.0, bad_roi(UnavailableReason.ROI_EMPTY), None)

    window = build_window(samples)

    assert window.n_samples == 10
    assert window.n_valid == 8
    assert window.n_missing == 2
    assert window.missing_pct == pytest.approx(20.0)


def test_window_arrays_return_only_valid_samples() -> None:
    samples = [
        make_sample(i, i / 30.0, good_roi(), (150.0, 110.0, 90.0)) for i in range(5)
    ]
    samples[2] = make_sample(2, 2 / 30.0, bad_roi(UnavailableReason.ROI_EMPTY), None)

    timestamps, rgb = window_arrays(build_window(samples))

    assert timestamps.shape == (4,)
    assert rgb.shape == (4, 3)
    assert np.all(np.isfinite(rgb))


def test_window_records_timestamp_defects() -> None:
    samples = [
        make_sample(i, t, good_roi(), (150.0, 110.0, 90.0))
        for i, t in enumerate([0.0, 0.1, 0.1, 0.05, 0.4])
    ]

    window = build_window(samples)

    assert window.timestamps_monotonic is False
    assert window.n_duplicate_timestamps == 1
    assert window.n_reversed_timestamps == 1


def test_iter_windows_splits_by_time() -> None:
    samples = [
        make_sample(i, i / 30.0, good_roi(), (150.0, 110.0, 90.0))
        for i in range(30 * 12)
    ]

    windows = iter_windows(samples, duration_seconds=5.0, step_seconds=5.0)

    assert len(windows) == 2
    assert all(w.duration_s == pytest.approx(5.0, abs=0.1) for w in windows)
    assert [w.window_index for w in windows] == [0, 1]


def test_iter_windows_overlap() -> None:
    samples = [
        make_sample(i, i / 30.0, good_roi(), (150.0, 110.0, 90.0))
        for i in range(30 * 10)
    ]

    windows = iter_windows(samples, duration_seconds=5.0, step_seconds=1.0)

    assert len(windows) > 2


def test_iter_windows_rejects_non_positive_geometry() -> None:
    with pytest.raises(ValueError, match="positive"):
        iter_windows([], duration_seconds=0.0, step_seconds=1.0)


# --- synthetic generation -------------------------------------------------


def test_synthetic_trace_is_deterministic() -> None:
    a = generate_synthetic_rgb_trace(bpm=72, duration_seconds=5, fps=30, seed=1)
    b = generate_synthetic_rgb_trace(bpm=72, duration_seconds=5, fps=30, seed=1)

    assert [s.g for s in a] == [s.g for s in b]


def test_synthetic_trace_differs_by_seed() -> None:
    a = generate_synthetic_rgb_trace(bpm=72, duration_seconds=5, fps=30, seed=1)
    b = generate_synthetic_rgb_trace(bpm=72, duration_seconds=5, fps=30, seed=2)

    assert [s.g for s in a] != [s.g for s in b]


def test_synthetic_trace_has_expected_length_and_rate() -> None:
    samples = generate_synthetic_rgb_trace(bpm=72, duration_seconds=10, fps=30, seed=3)

    assert len(samples) == 300
    rate = estimate_sampling_rate([s.monotonic_timestamp for s in samples])
    assert rate == pytest.approx(30.0)


def test_synthetic_trace_channels_are_in_range() -> None:
    samples = generate_synthetic_rgb_trace(bpm=90, duration_seconds=5, fps=30, seed=4)

    for s in samples:
        assert s.valid
        assert s.r is not None and 0.0 <= s.r <= 255.0
        assert s.g is not None and 0.0 <= s.g <= 255.0
        assert s.b is not None and 0.0 <= s.b <= 255.0


def test_synthetic_dropouts_produce_invalid_samples() -> None:
    samples = generate_synthetic_rgb_trace(
        bpm=72, duration_seconds=20, fps=30, seed=5, dropout_rate=0.3
    )

    invalid = [s for s in samples if not s.valid]
    assert invalid, "expected some dropped frames"
    assert all(s.r is None and s.g is None and s.b is None for s in invalid)


def test_synthetic_window_is_permanently_labelled() -> None:
    samples = generate_synthetic_rgb_trace(bpm=72, duration_seconds=5, fps=30, seed=6)

    window = build_synthetic_window(samples)

    assert window.synthetic_label == SYNTHETIC_LABEL
    assert window.data_source is DataSource.SYNTHETIC
    assert window.model_dump()["synthetic_label"] == "SYNTHETIC"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bpm": 0.0}, "bpm"),
        ({"bpm": -5.0}, "bpm"),
        ({"fps": 0.0}, "fps"),
        ({"duration_seconds": 0.0}, "duration_seconds"),
        ({"dropout_rate": 1.0}, "dropout_rate"),
        ({"dropout_rate": -0.1}, "dropout_rate"),
    ],
)
def test_synthetic_generator_rejects_invalid_arguments(
    kwargs: dict[str, float], match: str
) -> None:
    base: dict[str, float | int] = {
        "bpm": 72.0,
        "duration_seconds": 5.0,
        "fps": 30.0,
        "seed": 1,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=match):
        generate_synthetic_rgb_trace(**base)  # type: ignore[arg-type]


def test_pulse_is_strongest_in_green() -> None:
    """Verkruysse et al. (2008): green carries the strongest modulation."""
    samples = generate_synthetic_rgb_trace(
        bpm=72,
        duration_seconds=20,
        fps=30,
        seed=8,
        noise_std=0.0,
        illumination_drift=0.0,
        motion_artifact_rate=0.0,
    )
    r = np.std([s.r for s in samples if s.r is not None])
    g = np.std([s.g for s in samples if s.g is not None])
    b = np.std([s.b for s in samples if s.b is not None])

    assert g > r > b


def test_empty_sample_list_builds_empty_window() -> None:
    window = build_window([])

    assert window.n_samples == 0
    assert window.observed_fps is None
    assert window.duration_s == 0.0


def test_window_preserves_utc_timestamp_when_supplied() -> None:
    from datetime import UTC, datetime

    stamp = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)
    sample = make_sample(0, 0.0, good_roi(), (150.0, 110.0, 90.0), utc_timestamp=stamp)

    assert isinstance(sample, RgbTraceSample)
    assert sample.utc_timestamp == stamp
