"""Tests for head-pose estimation and motion features."""

from __future__ import annotations

from engagevr.head_pose.estimator import estimate_head_pose
from engagevr.head_pose.features import HeadMotionTracker
from engagevr.schemas.capture import NormalizedLandmark


def _make_centered_face(
    n: int = 478,
    *,
    yaw_shift: float = 0.0,
) -> list[NormalizedLandmark]:
    """Create landmarks approximating a centered face.

    Landmark positions are arranged to match the 3D model geometry
    expected by the PnP solver. The canonical model has the nose at
    origin, chin below, eyes above, and mouth corners spread.
    """
    lm = [NormalizedLandmark(x=0.5, y=0.5, z=0) for _ in range(n)]
    # Key landmarks for PnP (arranged to match canonical 3D model)
    lm[1] = NormalizedLandmark(x=0.5 + yaw_shift, y=0.4, z=0)  # nose
    lm[152] = NormalizedLandmark(x=0.5 + yaw_shift, y=0.7, z=0)  # chin
    lm[263] = NormalizedLandmark(x=0.35, y=0.3, z=0)  # left eye
    lm[33] = NormalizedLandmark(x=0.65, y=0.3, z=0)  # right eye
    lm[61] = NormalizedLandmark(x=0.4, y=0.6, z=0)  # left mouth
    lm[291] = NormalizedLandmark(x=0.6, y=0.6, z=0)  # right mouth
    return lm


class TestHeadPoseEstimation:
    def test_returns_three_angles(self):
        lm = _make_centered_face()
        result = estimate_head_pose(lm, 640, 480)
        assert result is not None
        assert len(result) == 3

    def test_produces_finite_angles(self):
        import math

        lm = _make_centered_face()
        result = estimate_head_pose(lm, 640, 480)
        assert result is not None
        for angle in result:
            assert math.isfinite(angle)

    def test_yaw_shift(self):
        lm_center = _make_centered_face(yaw_shift=0.0)
        lm_right = _make_centered_face(yaw_shift=0.1)
        r1 = estimate_head_pose(lm_center, 640, 480)
        r2 = estimate_head_pose(lm_right, 640, 480)
        assert r1 is not None and r2 is not None
        # Shifted nose should produce different yaw
        assert r1[0] != r2[0]

    def test_insufficient_landmarks(self):
        lm = [NormalizedLandmark(x=0.5, y=0.5, z=0)] * 100
        assert estimate_head_pose(lm, 640, 480) is None


class TestHeadMotionTracker:
    def test_first_frame_returns_none(self):
        tracker = HeadMotionTracker(window_seconds=1.0)
        vel, var = tracker.update(0, 0, 0, 1.0)
        assert vel is None
        assert var is None

    def test_velocity_computed(self):
        tracker = HeadMotionTracker(window_seconds=1.0)
        tracker.update(0, 0, 0, 0.0)
        vel, _ = tracker.update(10, 0, 0, 1.0)
        assert vel is not None
        assert vel == 10.0  # 10 degrees in 1 second

    def test_variability_needs_two_readings(self):
        tracker = HeadMotionTracker(window_seconds=5.0)
        tracker.update(0, 0, 0, 0.0)
        _, var = tracker.update(10, 0, 0, 1.0)
        assert var is None  # only 1 velocity sample
        _, var = tracker.update(20, 0, 0, 2.0)
        assert var is not None

    def test_zero_dt_returns_none(self):
        tracker = HeadMotionTracker()
        tracker.update(0, 0, 0, 1.0)
        vel, _ = tracker.update(10, 0, 0, 1.0)  # same timestamp
        assert vel is None

    def test_reset(self):
        tracker = HeadMotionTracker()
        tracker.update(0, 0, 0, 0.0)
        tracker.update(10, 0, 0, 1.0)
        tracker.reset()
        vel, _ = tracker.update(0, 0, 0, 2.0)
        assert vel is None  # first after reset
