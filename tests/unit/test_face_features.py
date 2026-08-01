"""Tests for facial behavioural feature extraction.

Uses synthetic landmarks -- no webcam, model, or internet required.
"""

from __future__ import annotations

from engagevr.face.features import (
    BlinkTracker,
    compute_left_ear,
    compute_mean_ear,
    compute_mouth_aspect_ratio,
    compute_right_ear,
    detect_blink,
)
from engagevr.schemas.capture import NormalizedLandmark


def _make_landmarks(n: int = 478) -> list[NormalizedLandmark]:
    """Create n synthetic landmarks at distinct positions."""
    return [
        NormalizedLandmark(x=0.3 + 0.001 * i, y=0.3 + 0.001 * i, z=0.0)
        for i in range(n)
    ]


def _set_eye_landmarks(
    landmarks: list[NormalizedLandmark],
    indices: list[int],
    *,
    open_ratio: float = 0.3,
) -> None:
    """Set eye landmark positions to produce a target EAR-like ratio.

    p1 and p4 (horizontal) are spaced apart.
    p2/p6 and p3/p5 (vertical pairs) are spaced by open_ratio * horizontal.
    """
    # Horizontal: p1 at 0.3, p4 at 0.5
    cx, cy = 0.4, 0.4
    half_h = 0.1
    half_v = half_h * open_ratio

    landmarks[indices[0]] = NormalizedLandmark(x=cx - half_h, y=cy, z=0)
    landmarks[indices[3]] = NormalizedLandmark(x=cx + half_h, y=cy, z=0)
    # Vertical pairs
    landmarks[indices[1]] = NormalizedLandmark(x=cx - half_h / 2, y=cy - half_v, z=0)
    landmarks[indices[5]] = NormalizedLandmark(x=cx - half_h / 2, y=cy + half_v, z=0)
    landmarks[indices[2]] = NormalizedLandmark(x=cx + half_h / 2, y=cy - half_v, z=0)
    landmarks[indices[4]] = NormalizedLandmark(x=cx + half_h / 2, y=cy + half_v, z=0)


class TestEyeAspectRatio:
    def test_open_eyes(self):
        lm = _make_landmarks()
        _set_eye_landmarks(lm, [33, 160, 158, 133, 153, 144], open_ratio=0.3)
        _set_eye_landmarks(lm, [362, 385, 387, 263, 373, 380], open_ratio=0.3)
        ear = compute_mean_ear(lm)
        assert ear is not None
        assert ear > 0.2

    def test_closed_eyes(self):
        lm = _make_landmarks()
        _set_eye_landmarks(lm, [33, 160, 158, 133, 153, 144], open_ratio=0.05)
        _set_eye_landmarks(lm, [362, 385, 387, 263, 373, 380], open_ratio=0.05)
        ear = compute_mean_ear(lm)
        assert ear is not None
        assert ear < 0.1

    def test_insufficient_landmarks(self):
        lm = _make_landmarks(100)
        assert compute_left_ear(lm) is None
        assert compute_right_ear(lm) is None
        assert compute_mean_ear(lm) is None

    def test_left_right_independent(self):
        lm = _make_landmarks()
        _set_eye_landmarks(lm, [33, 160, 158, 133, 153, 144], open_ratio=0.3)
        _set_eye_landmarks(lm, [362, 385, 387, 263, 373, 380], open_ratio=0.1)
        left = compute_left_ear(lm)
        right = compute_right_ear(lm)
        assert left is not None
        assert right is not None
        assert left != right


class TestBlinkDetection:
    def test_blink_below_threshold(self):
        assert detect_blink(0.15, threshold=0.21) is True

    def test_no_blink_above_threshold(self):
        assert detect_blink(0.30, threshold=0.21) is False

    def test_none_ear(self):
        assert detect_blink(None, threshold=0.21) is None


class TestBlinkTracker:
    def test_no_blink_initially(self):
        tracker = BlinkTracker(ear_threshold=0.21, min_frames=3, fps=30)
        blink, duration = tracker.update(0.30)
        assert blink is False
        assert duration is None

    def test_blink_after_min_frames(self):
        tracker = BlinkTracker(ear_threshold=0.21, min_frames=3, fps=30)
        for _ in range(2):
            blink, _ = tracker.update(0.10)
            assert blink is False
        blink, duration = tracker.update(0.10)
        assert blink is True
        assert duration is not None
        assert duration > 0

    def test_blink_resets_on_open(self):
        tracker = BlinkTracker(ear_threshold=0.21, min_frames=3, fps=30)
        for _ in range(3):
            tracker.update(0.10)
        blink, _ = tracker.update(0.30)
        assert blink is False

    def test_none_ear_returns_none(self):
        tracker = BlinkTracker()
        blink, duration = tracker.update(None)
        assert blink is None
        assert duration is None

    def test_eye_closure_duration(self):
        tracker = BlinkTracker(ear_threshold=0.21, min_frames=2, fps=10)
        tracker.update(0.10)
        tracker.update(0.10)
        _, duration = tracker.update(0.10)
        assert duration is not None
        assert duration == 0.3  # 3 frames / 10 fps


class TestMouthAspectRatio:
    def test_closed_mouth(self):
        lm = _make_landmarks()
        # Upper and lower lip same y
        lm[13] = NormalizedLandmark(x=0.5, y=0.6, z=0)
        lm[14] = NormalizedLandmark(x=0.5, y=0.6, z=0)
        lm[61] = NormalizedLandmark(x=0.4, y=0.6, z=0)
        lm[291] = NormalizedLandmark(x=0.6, y=0.6, z=0)
        mar = compute_mouth_aspect_ratio(lm)
        assert mar is not None
        assert mar < 0.01

    def test_open_mouth(self):
        lm = _make_landmarks()
        lm[13] = NormalizedLandmark(x=0.5, y=0.55, z=0)
        lm[14] = NormalizedLandmark(x=0.5, y=0.7, z=0)
        lm[61] = NormalizedLandmark(x=0.4, y=0.6, z=0)
        lm[291] = NormalizedLandmark(x=0.6, y=0.6, z=0)
        mar = compute_mouth_aspect_ratio(lm)
        assert mar is not None
        assert mar > 0.3

    def test_insufficient_landmarks(self):
        lm = _make_landmarks(100)
        assert compute_mouth_aspect_ratio(lm) is None
