"""Behavioural proxy features from facial landmarks.

All outputs are behavioural proxies. They do NOT measure engagement,
cognitive load, or emotion directly.

Eye Aspect Ratio (EAR)
----------------------
Soukupova & Cech (2016), "Real-Time Eye Blink Detection using Facial
Landmarks", 21st Computer Vision Winter Workshop.

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

MediaPipe landmark indices used (Face Mesh 478-point model):
- Right eye: 33, 160, 158, 133, 153, 144
- Left eye:  362, 385, 387, 263, 373, 380

Mouth Aspect Ratio (MAR)
------------------------
Adapted from lip landmark vertical-to-horizontal ratio.
- Upper lip: 13, Lower lip: 14
- Left corner: 61, Right corner: 291

These are geometric proxies only. Cultural and individual variation
is substantial.
"""

from __future__ import annotations

import math

from engagevr.schemas.capture import NormalizedLandmark

# MediaPipe Face Mesh landmark indices
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
_LEFT_EYE = [362, 385, 387, 263, 373, 380]

_UPPER_LIP = 13
_LOWER_LIP = 14
_LEFT_MOUTH = 61
_RIGHT_MOUTH = 291

_MIN_LANDMARK_COUNT = 468


def _dist(a: NormalizedLandmark, b: NormalizedLandmark) -> float:
    """Euclidean distance between two normalized landmarks."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def compute_ear(
    landmarks: list[NormalizedLandmark], indices: list[int]
) -> float | None:
    """Compute Eye Aspect Ratio for one eye.

    Returns None if required landmarks are unavailable.
    """
    if len(landmarks) < _MIN_LANDMARK_COUNT:
        return None
    try:
        p1 = landmarks[indices[0]]
        p2 = landmarks[indices[1]]
        p3 = landmarks[indices[2]]
        p4 = landmarks[indices[3]]
        p5 = landmarks[indices[4]]
        p6 = landmarks[indices[5]]
    except IndexError:
        return None

    horizontal = _dist(p1, p4)
    if horizontal < 1e-9:
        return None
    vertical1 = _dist(p2, p6)
    vertical2 = _dist(p3, p5)
    return (vertical1 + vertical2) / (2.0 * horizontal)


def compute_left_ear(
    landmarks: list[NormalizedLandmark],
) -> float | None:
    """Left Eye Aspect Ratio."""
    return compute_ear(landmarks, _LEFT_EYE)


def compute_right_ear(
    landmarks: list[NormalizedLandmark],
) -> float | None:
    """Right Eye Aspect Ratio."""
    return compute_ear(landmarks, _RIGHT_EYE)


def compute_mean_ear(
    landmarks: list[NormalizedLandmark],
) -> float | None:
    """Mean EAR across both eyes. None if either eye is unavailable."""
    left = compute_left_ear(landmarks)
    right = compute_right_ear(landmarks)
    if left is None or right is None:
        return None
    return (left + right) / 2.0


def detect_blink(mean_ear: float | None, threshold: float) -> bool | None:
    """Return True if mean EAR is below blink threshold.

    Returns None when EAR is unavailable (not a blink).
    """
    if mean_ear is None:
        return None
    return mean_ear < threshold


def compute_mouth_aspect_ratio(
    landmarks: list[NormalizedLandmark],
) -> float | None:
    """Mouth Aspect Ratio: vertical / horizontal lip distance.

    Returns None when landmarks are insufficient.
    """
    if len(landmarks) < _MIN_LANDMARK_COUNT:
        return None
    try:
        upper = landmarks[_UPPER_LIP]
        lower = landmarks[_LOWER_LIP]
        left = landmarks[_LEFT_MOUTH]
        right = landmarks[_RIGHT_MOUTH]
    except IndexError:
        return None

    horizontal = _dist(left, right)
    if horizontal < 1e-9:
        return None
    vertical = _dist(upper, lower)
    return vertical / horizontal


class BlinkTracker:
    """Track blink events and eye-closure duration.

    A blink is detected when mean EAR drops below the threshold for
    at least ``min_frames`` consecutive frames.
    """

    def __init__(
        self,
        *,
        ear_threshold: float = 0.21,
        min_frames: int = 3,
        fps: float = 30.0,
    ) -> None:
        self._threshold = ear_threshold
        self._min_frames = min_frames
        self._fps = fps
        self._closed_count = 0

    def update(self, mean_ear: float | None) -> tuple[bool | None, float | None]:
        """Update tracker with current EAR.

        Returns
        -------
        (blink_detected, eye_closure_duration_s)
            blink_detected: True if eyes are closed, None if EAR unavailable.
            eye_closure_duration_s: seconds eyes have been closed, or None.
        """
        if mean_ear is None:
            return None, None

        if mean_ear < self._threshold:
            self._closed_count += 1
        else:
            self._closed_count = 0

        is_closed = self._closed_count >= self._min_frames
        duration = self._closed_count / self._fps if is_closed else None
        return is_closed, duration

    def reset(self) -> None:
        self._closed_count = 0
