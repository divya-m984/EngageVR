"""Frame-level capture quality metrics.

Quality is distinct from engagement. Poor quality must NEVER
produce a low-engagement label.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from engagevr.config import QualityConfig


def compute_brightness(gray: npt.NDArray[np.uint8]) -> float:
    """Mean pixel intensity of a grayscale frame [0-255]."""
    return float(np.mean(gray))


def compute_blur_score(gray: npt.NDArray[np.uint8]) -> float:
    """Laplacian variance as a sharpness proxy.

    Higher values indicate sharper images.
    Reference: Pertuz et al. (2013), "Analysis of focus measure operators
    for shape-from-focus", Pattern Recognition.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(laplacian))


def compute_motion_score(
    prev_gray: npt.NDArray[np.uint8] | None,
    curr_gray: npt.NDArray[np.uint8],
) -> float | None:
    """Mean absolute difference between consecutive frames [0-255].

    Returns None when no previous frame is available.
    """
    if prev_gray is None:
        return None
    if prev_gray.shape != curr_gray.shape:
        return None
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff))


def assess_brightness(brightness: float, cfg: QualityConfig) -> tuple[bool, bool]:
    """Return (underexposed, overexposed) flags."""
    return brightness < cfg.brightness_low, brightness > cfg.brightness_high


def assess_blur(blur_score: float, cfg: QualityConfig) -> bool:
    """Return True if the frame is blurry."""
    return blur_score < cfg.blur_threshold


def assess_motion(motion_score: float | None, cfg: QualityConfig) -> bool:
    """Return True if frame-to-frame motion is excessive."""
    if motion_score is None:
        return False
    return motion_score > cfg.motion_threshold


def count_missing_features(features: dict[str, object | None]) -> float:
    """Percentage of feature fields that are None [0-100]."""
    if not features:
        return 0.0
    n_missing = sum(1 for v in features.values() if v is None)
    return 100.0 * n_missing / len(features)
