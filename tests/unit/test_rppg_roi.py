"""Tests for face-skin ROI extraction.

No webcam, no display, no network. Frames are synthetic arrays and
landmarks are constructed directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from engagevr.config import RppgRoiConfig
from engagevr.rppg.roi import (
    extract_combined,
    extract_region,
    extract_regions,
    region_landmark_indices,
)
from engagevr.schemas.capture import NormalizedLandmark
from engagevr.schemas.rppg import RoiRegion, UnavailableReason

FRAME_H, FRAME_W = 480, 640


def make_landmarks(count: int = 478) -> list[NormalizedLandmark]:
    """A plausible face laid out across the middle of the frame.

    Landmarks are placed on a coarse grid so that every region's index
    set spans a non-degenerate box.
    """
    landmarks = []
    for i in range(count):
        landmarks.append(
            NormalizedLandmark(
                x=0.30 + 0.40 * ((i * 37) % 100) / 100.0,
                y=0.25 + 0.45 * ((i * 53) % 100) / 100.0,
                z=0.0,
            )
        )
    return landmarks


def positioned_landmarks(
    region_boxes: dict[RoiRegion, tuple[float, float, float, float]],
    count: int = 478,
) -> list[NormalizedLandmark]:
    """Place each region's landmarks at explicit normalized box corners."""
    landmarks = [NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(count)]
    for region, (x0, y0, x1, y1) in region_boxes.items():
        indices = region_landmark_indices(region)
        for n, idx in enumerate(indices):
            landmarks[idx] = NormalizedLandmark(
                x=x0 if n % 2 == 0 else x1,
                y=y0 if n < len(indices) // 2 else y1,
                z=0.0,
            )
    return landmarks


def uniform_frame(value: int = 120) -> np.ndarray:
    return np.full((FRAME_H, FRAME_W, 3), value, dtype=np.uint8)


@pytest.fixture
def cfg() -> RppgRoiConfig:
    return RppgRoiConfig()


def test_forehead_extraction_is_available(cfg: RppgRoiConfig) -> None:
    landmarks = positioned_landmarks({RoiRegion.FOREHEAD: (0.4, 0.15, 0.6, 0.30)})
    obs, pixels = extract_region(uniform_frame(120), landmarks, RoiRegion.FOREHEAD, cfg)

    assert obs.available is True
    assert obs.reason is None
    assert obs.region is RoiRegion.FOREHEAD
    assert pixels is not None
    assert obs.valid_pixel_count == pixels.shape[0]
    assert obs.valid_pixel_pct == pytest.approx(100.0)
    assert obs.mean_brightness == pytest.approx(120.0)


@pytest.mark.parametrize(
    "region",
    [RoiRegion.LEFT_CHEEK, RoiRegion.RIGHT_CHEEK],
)
def test_cheek_extraction_is_available(cfg: RppgRoiConfig, region: RoiRegion) -> None:
    landmarks = positioned_landmarks({region: (0.30, 0.45, 0.45, 0.65)})
    obs, pixels = extract_region(uniform_frame(100), landmarks, region, cfg)

    assert obs.available is True
    assert pixels is not None
    assert obs.region is region


def test_region_channel_means_are_preserved(cfg: RppgRoiConfig) -> None:
    """Distinct per-channel values survive extraction unchanged."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, :, 0] = 150
    frame[:, :, 1] = 110
    frame[:, :, 2] = 90
    landmarks = positioned_landmarks({RoiRegion.FOREHEAD: (0.4, 0.15, 0.6, 0.30)})

    _, pixels = extract_region(frame, landmarks, RoiRegion.FOREHEAD, cfg)

    assert pixels is not None
    means = pixels.mean(axis=0)
    assert means[0] == pytest.approx(150.0)
    assert means[1] == pytest.approx(110.0)
    assert means[2] == pytest.approx(90.0)


def test_box_is_clipped_to_frame_bounds(cfg: RppgRoiConfig) -> None:
    """Landmarks beyond the frame produce a box inside the frame."""
    landmarks = positioned_landmarks({RoiRegion.FOREHEAD: (-0.5, -0.5, 1.5, 1.5)})
    obs, pixels = extract_region(uniform_frame(), landmarks, RoiRegion.FOREHEAD, cfg)

    assert obs.available is True
    assert obs.x_min is not None and obs.x_min >= 0
    assert obs.y_min is not None and obs.y_min >= 0
    assert obs.x_max is not None and obs.x_max <= FRAME_W
    assert obs.y_max is not None and obs.y_max <= FRAME_H
    assert pixels is not None


def test_region_fully_outside_frame_is_rejected(cfg: RppgRoiConfig) -> None:
    landmarks = positioned_landmarks({RoiRegion.FOREHEAD: (1.2, 1.2, 1.4, 1.4)})
    obs, pixels = extract_region(uniform_frame(), landmarks, RoiRegion.FOREHEAD, cfg)

    assert obs.available is False
    assert obs.reason is UnavailableReason.ROI_OUT_OF_FRAME
    assert pixels is None


def test_missing_landmarks_return_no_face(cfg: RppgRoiConfig) -> None:
    obs, pixels = extract_region(uniform_frame(), None, RoiRegion.FOREHEAD, cfg)

    assert obs.available is False
    assert obs.reason is UnavailableReason.NO_FACE
    assert pixels is None


def test_too_few_landmarks_return_no_landmarks(cfg: RppgRoiConfig) -> None:
    obs, _ = extract_region(
        uniform_frame(), make_landmarks(10), RoiRegion.FOREHEAD, cfg
    )

    assert obs.available is False
    assert obs.reason is UnavailableReason.NO_LANDMARKS


def test_degenerate_box_is_rejected(cfg: RppgRoiConfig) -> None:
    """A zero-area region cannot yield pixels."""
    landmarks = positioned_landmarks({RoiRegion.FOREHEAD: (0.5, 0.5, 0.5, 0.5)})
    obs, pixels = extract_region(uniform_frame(), landmarks, RoiRegion.FOREHEAD, cfg)

    assert obs.available is False
    assert obs.reason is UnavailableReason.ROI_OUT_OF_FRAME
    assert pixels is None


def test_too_few_valid_pixels_is_rejected() -> None:
    """A tiny but non-empty box fails the minimum-pixel requirement."""
    cfg = RppgRoiConfig(min_valid_pixels=100_000)
    landmarks = positioned_landmarks({RoiRegion.FOREHEAD: (0.40, 0.20, 0.60, 0.30)})
    obs, pixels = extract_region(uniform_frame(), landmarks, RoiRegion.FOREHEAD, cfg)

    assert obs.available is False
    assert obs.reason is UnavailableReason.ROI_TOO_SMALL
    assert pixels is None
    assert obs.total_pixel_count > 0


def test_saturated_region_is_rejected(cfg: RppgRoiConfig) -> None:
    """Fully blown-out pixels carry no plethysmographic signal."""
    landmarks = positioned_landmarks({RoiRegion.FOREHEAD: (0.4, 0.15, 0.6, 0.30)})
    obs, pixels = extract_region(uniform_frame(255), landmarks, RoiRegion.FOREHEAD, cfg)

    assert obs.available is False
    assert obs.reason is UnavailableReason.ROI_EMPTY
    assert pixels is None
    assert obs.clipped_pixel_pct == pytest.approx(100.0)


def test_partially_clipped_region_below_threshold_is_rejected() -> None:
    """More than the tolerated fraction of clipped pixels rejects the ROI."""
    cfg = RppgRoiConfig(min_valid_pixels=1, min_valid_pixel_pct=90.0)
    frame = uniform_frame(120)
    # Saturate the top half of the frame.
    frame[: FRAME_H // 2, :, :] = 255
    landmarks = positioned_landmarks({RoiRegion.FOREHEAD: (0.3, 0.10, 0.7, 0.90)})

    obs, pixels = extract_region(frame, landmarks, RoiRegion.FOREHEAD, cfg)

    assert obs.available is False
    assert obs.reason is UnavailableReason.ROI_INSUFFICIENT_VALID_PIXELS
    assert pixels is None
    assert 0.0 < obs.valid_pixel_pct < 90.0


def test_extract_regions_covers_every_configured_region(
    cfg: RppgRoiConfig,
) -> None:
    landmarks = positioned_landmarks(
        {
            RoiRegion.FOREHEAD: (0.40, 0.15, 0.60, 0.30),
            RoiRegion.LEFT_CHEEK: (0.30, 0.45, 0.42, 0.62),
            RoiRegion.RIGHT_CHEEK: (0.58, 0.45, 0.70, 0.62),
        }
    )
    results = extract_regions(uniform_frame(), landmarks, cfg)

    assert set(results) == set(cfg.regions)
    assert all(obs.available for obs, _ in results.values())


def test_combined_pools_regions(cfg: RppgRoiConfig) -> None:
    landmarks = positioned_landmarks(
        {
            RoiRegion.FOREHEAD: (0.40, 0.15, 0.60, 0.30),
            RoiRegion.LEFT_CHEEK: (0.30, 0.45, 0.42, 0.62),
            RoiRegion.RIGHT_CHEEK: (0.58, 0.45, 0.70, 0.62),
        }
    )
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, :, 0] = 160
    frame[:, :, 1] = 120
    frame[:, :, 2] = 100

    obs, mean_rgb = extract_combined(frame, landmarks, cfg)

    assert obs.available is True
    assert obs.region is RoiRegion.COMBINED
    assert mean_rgb is not None
    assert mean_rgb[0] == pytest.approx(160.0)
    assert mean_rgb[1] == pytest.approx(120.0)
    assert mean_rgb[2] == pytest.approx(100.0)
    # Pooled counts aggregate across regions.
    assert obs.valid_pixel_count > 0


def test_combined_without_face_is_unavailable(cfg: RppgRoiConfig) -> None:
    obs, mean_rgb = extract_combined(uniform_frame(), None, cfg)

    assert obs.available is False
    assert mean_rgb is None
    assert obs.reason is UnavailableReason.NO_FACE


def test_combined_rejects_when_all_regions_saturated(cfg: RppgRoiConfig) -> None:
    landmarks = positioned_landmarks(
        {
            RoiRegion.FOREHEAD: (0.40, 0.15, 0.60, 0.30),
            RoiRegion.LEFT_CHEEK: (0.30, 0.45, 0.42, 0.62),
            RoiRegion.RIGHT_CHEEK: (0.58, 0.45, 0.70, 0.62),
        }
    )
    obs, mean_rgb = extract_combined(uniform_frame(255), landmarks, cfg)

    assert obs.available is False
    assert mean_rgb is None


def test_combined_region_argument_is_rejected(cfg: RppgRoiConfig) -> None:
    with pytest.raises(ValueError, match="extract_combined"):
        extract_region(uniform_frame(), make_landmarks(), RoiRegion.COMBINED, cfg)


def test_inset_shrinks_the_box() -> None:
    """A larger inset must produce a strictly smaller region."""
    landmarks = positioned_landmarks({RoiRegion.FOREHEAD: (0.30, 0.15, 0.70, 0.35)})
    small_inset, _ = extract_region(
        uniform_frame(),
        landmarks,
        RoiRegion.FOREHEAD,
        RppgRoiConfig(inset_fraction=0.0),
    )
    large_inset, _ = extract_region(
        uniform_frame(),
        landmarks,
        RoiRegion.FOREHEAD,
        RppgRoiConfig(inset_fraction=0.4),
    )

    assert large_inset.total_pixel_count < small_inset.total_pixel_count


def test_roi_observation_stores_no_pixels(cfg: RppgRoiConfig) -> None:
    """Privacy: the persisted schema must contain counts, never imagery."""
    landmarks = positioned_landmarks({RoiRegion.FOREHEAD: (0.4, 0.15, 0.6, 0.30)})
    obs, _ = extract_region(uniform_frame(), landmarks, RoiRegion.FOREHEAD, cfg)

    dumped = obs.model_dump()
    for value in dumped.values():
        assert not isinstance(value, (bytes, bytearray, np.ndarray))
    assert "pixels" not in dumped
    assert "image" not in dumped
