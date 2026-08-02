"""Face-skin region-of-interest extraction for rPPG.

Regions are derived from the MediaPipe Face Mesh landmark set already
produced by :mod:`engagevr.face.landmarker`.  Three skin regions are
sampled -- forehead and both cheeks -- because these are large, relatively
flat, well-perfused, and comparatively free of hair and rigid facial
motion (Verkruysse, Svaasand & Nelson, 2008, DOI 10.1364/OE.16.021434,
sample the forehead and cheek for exactly this reason).

Region definitions
------------------
Each region is defined by a set of Face Mesh landmark indices.  Their
axis-aligned bounding box is computed, then trimmed inward by
``inset_fraction`` of each half-extent.  The inset is what excludes the
hairline above the forehead box, the eyebrows and eyes below it, and the
nostrils and face outline beside the cheek boxes.  The cheek index sets
are mirror pairs in the canonical mesh (117/346, 118/347, 119/348,
100/329, 142/371, 36/266, 205/425, 50/280).

``left`` and ``right`` are named from the **image** frame, not from the
subject's anatomy.

Validity
--------
A pixel is *valid* when every channel lies strictly between the
configured clipping bounds.  Crushed-black and saturated pixels carry no
usable plethysmographic modulation, so they are excluded from the spatial
average and counted separately.

Privacy
-------
ROI pixels exist only in memory for the duration of one frame.  No ROI
image is written to disk, logged, or transmitted.  The returned
:class:`~engagevr.schemas.rppg.RoiObservation` contains counts and
coordinates only.  Nothing here infers skin tone, ethnicity, identity,
emotion, engagement, or cognitive state.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from engagevr.config import RppgRoiConfig
from engagevr.schemas.capture import NormalizedLandmark
from engagevr.schemas.rppg import RoiObservation, RoiRegion, UnavailableReason

# MediaPipe Face Mesh (468/478-point) landmark indices per region.
_REGION_LANDMARKS: dict[RoiRegion, tuple[int, ...]] = {
    # Forehead: glabella (9) up to the upper forehead (10), bounded
    # laterally by the brow-ridge corners (67, 297) and mid-brow (105, 334).
    RoiRegion.FOREHEAD: (9, 10, 67, 297, 105, 334, 151),
    # Image-left cheek (subject's right).
    RoiRegion.LEFT_CHEEK: (117, 118, 119, 100, 142, 36, 205, 50),
    # Image-right cheek (subject's left) -- mirror of the above.
    RoiRegion.RIGHT_CHEEK: (346, 347, 348, 329, 371, 266, 425, 280),
}

#: Face Mesh models below this size do not contain the indices above.
_MIN_LANDMARK_COUNT = 468


def region_landmark_indices(region: RoiRegion) -> tuple[int, ...]:
    """Return the Face Mesh landmark indices defining ``region``."""
    if region not in _REGION_LANDMARKS:
        raise ValueError(f"No landmark definition for region {region!r}")
    return _REGION_LANDMARKS[region]


def _bounding_box(
    landmarks: list[NormalizedLandmark],
    indices: tuple[int, ...],
    width: int,
    height: int,
    inset_fraction: float,
) -> tuple[int, int, int, int] | None:
    """Compute an inset, frame-clipped pixel bounding box.

    Returns ``None`` when the box degenerates or falls outside the frame.
    """
    xs: list[float] = []
    ys: list[float] = []
    for idx in indices:
        lm = landmarks[idx]
        if not (np.isfinite(lm.x) and np.isfinite(lm.y)):
            return None
        xs.append(lm.x * width)
        ys.append(lm.y * height)

    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)

    # Trim inward by a fraction of each half-extent.
    dx = (x_hi - x_lo) * inset_fraction
    dy = (y_hi - y_lo) * inset_fraction
    x_lo, x_hi = x_lo + dx, x_hi - dx
    y_lo, y_hi = y_lo + dy, y_hi - dy

    # Clip to frame bounds; x_max/y_max are exclusive.
    x_min = max(0, int(np.floor(x_lo)))
    y_min = max(0, int(np.floor(y_lo)))
    x_max = min(width, int(np.ceil(x_hi)))
    y_max = min(height, int(np.ceil(y_hi)))

    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def _unavailable(
    region: RoiRegion,
    reason: UnavailableReason,
    *,
    session_id: str,
    frame_index: int,
    monotonic_timestamp: float,
    box: tuple[int, int, int, int] | None = None,
    total_pixels: int = 0,
    valid_pixels: int = 0,
    valid_pct: float = 0.0,
    clipped_pct: float = 0.0,
) -> RoiObservation:
    return RoiObservation(
        session_id=session_id,
        frame_index=frame_index,
        monotonic_timestamp=monotonic_timestamp,
        region=region,
        available=False,
        reason=reason,
        x_min=box[0] if box else None,
        y_min=box[1] if box else None,
        x_max=box[2] if box else None,
        y_max=box[3] if box else None,
        total_pixel_count=total_pixels,
        valid_pixel_count=valid_pixels,
        valid_pixel_pct=valid_pct,
        clipped_pixel_pct=clipped_pct,
    )


def extract_region(
    rgb_frame: npt.NDArray[np.uint8],
    landmarks: list[NormalizedLandmark] | None,
    region: RoiRegion,
    cfg: RppgRoiConfig,
    *,
    session_id: str = "",
    frame_index: int = 0,
    monotonic_timestamp: float = 0.0,
) -> tuple[RoiObservation, npt.NDArray[np.float64] | None]:
    """Extract one skin region from an RGB frame.

    Parameters
    ----------
    rgb_frame:
        ``(H, W, 3)`` uint8 RGB image.  Never persisted.
    landmarks:
        Face Mesh landmarks, or ``None`` when no face was detected.
    region:
        Region to sample.  ``RoiRegion.COMBINED`` is not accepted here;
        use :func:`extract_combined`.

    Returns
    -------
    (observation, valid_pixels)
        ``valid_pixels`` is an ``(N, 3)`` float array of non-clipped RGB
        pixels, or ``None`` when the region is unavailable.  It is
        intended for immediate spatial averaging and must not be stored.
    """
    if region is RoiRegion.COMBINED:
        raise ValueError("Use extract_combined() for RoiRegion.COMBINED")

    def fail(
        reason: UnavailableReason,
        *,
        box: tuple[int, int, int, int] | None = None,
        total_pixels: int = 0,
        valid_pixels: int = 0,
        valid_pct: float = 0.0,
        clipped_pct: float = 0.0,
    ) -> tuple[RoiObservation, None]:
        return (
            _unavailable(
                region,
                reason,
                session_id=session_id,
                frame_index=frame_index,
                monotonic_timestamp=monotonic_timestamp,
                box=box,
                total_pixels=total_pixels,
                valid_pixels=valid_pixels,
                valid_pct=valid_pct,
                clipped_pct=clipped_pct,
            ),
            None,
        )

    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        return fail(UnavailableReason.ROI_EMPTY)

    if landmarks is None:
        return fail(UnavailableReason.NO_FACE)
    if len(landmarks) < _MIN_LANDMARK_COUNT:
        return fail(UnavailableReason.NO_LANDMARKS)

    height, width = rgb_frame.shape[0], rgb_frame.shape[1]
    if height <= 0 or width <= 0:
        return fail(UnavailableReason.ROI_EMPTY)

    box = _bounding_box(
        landmarks,
        region_landmark_indices(region),
        width,
        height,
        cfg.inset_fraction,
    )
    if box is None:
        return fail(UnavailableReason.ROI_OUT_OF_FRAME)

    x_min, y_min, x_max, y_max = box
    patch = rgb_frame[y_min:y_max, x_min:x_max, :].astype(np.float64)
    total = int(patch.shape[0] * patch.shape[1])
    if total == 0:
        return fail(UnavailableReason.ROI_EMPTY, box=box)

    flat = patch.reshape(-1, 3)
    keep = np.all(
        (flat > float(cfg.clipping_low)) & (flat < float(cfg.clipping_high)),
        axis=1,
    )
    valid = flat[keep]
    n_valid = int(valid.shape[0])
    valid_pct = 100.0 * n_valid / total
    clipped_pct = 100.0 - valid_pct

    def fail_measured(reason: UnavailableReason) -> tuple[RoiObservation, None]:
        return fail(
            reason,
            box=box,
            total_pixels=total,
            valid_pixels=n_valid,
            valid_pct=valid_pct,
            clipped_pct=clipped_pct,
        )

    if n_valid == 0:
        return fail_measured(UnavailableReason.ROI_EMPTY)
    if n_valid < cfg.min_valid_pixels:
        return fail_measured(UnavailableReason.ROI_TOO_SMALL)
    if valid_pct < cfg.min_valid_pixel_pct:
        return fail_measured(UnavailableReason.ROI_INSUFFICIENT_VALID_PIXELS)

    brightness = float(np.mean(valid))
    observation = RoiObservation(
        session_id=session_id,
        frame_index=frame_index,
        monotonic_timestamp=monotonic_timestamp,
        region=region,
        available=True,
        reason=None,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        total_pixel_count=total,
        valid_pixel_count=n_valid,
        valid_pixel_pct=valid_pct,
        mean_brightness=brightness,
        clipped_pixel_pct=clipped_pct,
    )
    return observation, valid


def extract_regions(
    rgb_frame: npt.NDArray[np.uint8],
    landmarks: list[NormalizedLandmark] | None,
    cfg: RppgRoiConfig,
    *,
    session_id: str = "",
    frame_index: int = 0,
    monotonic_timestamp: float = 0.0,
) -> dict[RoiRegion, tuple[RoiObservation, npt.NDArray[np.float64] | None]]:
    """Extract every configured region independently."""
    return {
        region: extract_region(
            rgb_frame,
            landmarks,
            region,
            cfg,
            session_id=session_id,
            frame_index=frame_index,
            monotonic_timestamp=monotonic_timestamp,
        )
        for region in cfg.regions
    }


def extract_combined(
    rgb_frame: npt.NDArray[np.uint8],
    landmarks: list[NormalizedLandmark] | None,
    cfg: RppgRoiConfig,
    *,
    session_id: str = "",
    frame_index: int = 0,
    monotonic_timestamp: float = 0.0,
) -> tuple[RoiObservation, tuple[float, float, float] | None]:
    """Pool all available configured regions into one RGB sample.

    Pixels from every available region are pooled before averaging, so
    each region contributes in proportion to its valid-pixel count.  This
    is the spatial-averaging step of Verkruysse et al. (2008), applied
    across regions rather than to a single patch.

    Returns
    -------
    (observation, mean_rgb)
        ``mean_rgb`` is ``None`` when no region was usable.  The returned
        observation has ``region=COMBINED`` and aggregates the per-region
        pixel counts.
    """
    per_region = extract_regions(
        rgb_frame,
        landmarks,
        cfg,
        session_id=session_id,
        frame_index=frame_index,
        monotonic_timestamp=monotonic_timestamp,
    )

    pixel_blocks: list[npt.NDArray[np.float64]] = []
    total = 0
    valid = 0
    first_reason: UnavailableReason | None = None
    for obs, pixels in per_region.values():
        total += obs.total_pixel_count
        valid += obs.valid_pixel_count
        if pixels is not None:
            pixel_blocks.append(pixels)
        elif first_reason is None:
            first_reason = obs.reason

    base = {
        "session_id": session_id,
        "frame_index": frame_index,
        "monotonic_timestamp": monotonic_timestamp,
        "region": RoiRegion.COMBINED,
        "total_pixel_count": total,
        "valid_pixel_count": valid,
        "valid_pixel_pct": (100.0 * valid / total) if total else 0.0,
        "clipped_pixel_pct": (100.0 - 100.0 * valid / total) if total else 0.0,
    }

    if not pixel_blocks:
        return (
            RoiObservation(
                available=False,
                reason=first_reason or UnavailableReason.ROI_EMPTY,
                **base,
            ),
            None,
        )

    pooled = np.concatenate(pixel_blocks, axis=0)
    means = np.mean(pooled, axis=0)
    observation = RoiObservation(
        available=True,
        reason=None,
        mean_brightness=float(np.mean(pooled)),
        **base,
    )
    return observation, (float(means[0]), float(means[1]), float(means[2]))
