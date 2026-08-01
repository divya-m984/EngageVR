"""Tests for capture-layer schemas."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from engagevr.schemas.capture import (
    BehaviouralFeatures,
    CaptureQualityReport,
    CaptureSessionSummary,
    FrameMetadata,
    HeadPoseObservation,
    LandmarkObservation,
    NormalizedLandmark,
)


class TestFrameMetadata:
    def test_valid(self):
        m = FrameMetadata(
            session_id="s1",
            frame_index=0,
            monotonic_timestamp=1.0,
            utc_timestamp=datetime.now(UTC),
            width=640,
            height=480,
        )
        assert m.channels == 3


class TestLandmarkObservation:
    def test_no_face(self):
        obs = LandmarkObservation(
            session_id="s1",
            frame_index=0,
            monotonic_timestamp=1.0,
            face_detected=False,
        )
        assert obs.landmarks is None
        assert obs.landmark_count == 0

    def test_with_face(self):
        lm = [NormalizedLandmark(x=0.5, y=0.5, z=0.0)]
        obs = LandmarkObservation(
            session_id="s1",
            frame_index=0,
            monotonic_timestamp=1.0,
            face_detected=True,
            landmarks=lm,
            landmark_count=1,
        )
        assert obs.face_detected
        assert len(obs.landmarks) == 1


class TestBehaviouralFeatures:
    def test_no_face(self):
        bf = BehaviouralFeatures(
            session_id="s1",
            frame_index=0,
            monotonic_timestamp=1.0,
            face_present=False,
        )
        assert bf.left_ear is None
        assert bf.blink_detected is None

    def test_with_features(self):
        bf = BehaviouralFeatures(
            session_id="s1",
            frame_index=0,
            monotonic_timestamp=1.0,
            face_present=True,
            left_ear=0.3,
            right_ear=0.28,
            mean_ear=0.29,
            blink_detected=False,
            mouth_aspect_ratio=0.1,
        )
        assert bf.mean_ear == 0.29


class TestHeadPoseObservation:
    def test_unavailable(self):
        hp = HeadPoseObservation(
            session_id="s1",
            frame_index=0,
            monotonic_timestamp=1.0,
            available=False,
        )
        assert hp.yaw_deg is None

    def test_with_pose(self):
        hp = HeadPoseObservation(
            session_id="s1",
            frame_index=0,
            monotonic_timestamp=1.0,
            available=True,
            yaw_deg=5.0,
            pitch_deg=-3.0,
            roll_deg=1.0,
        )
        assert hp.yaw_deg == 5.0


class TestCaptureQualityReport:
    def test_missing_feature_pct_range(self):
        with pytest.raises(ValidationError):
            CaptureQualityReport(
                session_id="s1",
                frame_index=0,
                monotonic_timestamp=1.0,
                webcam_open=True,
                frame_read_success=True,
                missing_feature_pct=101.0,
            )

    def test_valid_report(self):
        r = CaptureQualityReport(
            session_id="s1",
            frame_index=0,
            monotonic_timestamp=1.0,
            webcam_open=True,
            frame_read_success=True,
            brightness=128.0,
            blur_score=200.0,
            face_present=True,
            landmarks_available=True,
        )
        assert r.webcam_open


class TestCaptureSessionSummary:
    def test_defaults(self):
        s = CaptureSessionSummary(
            session_id="s1",
            total_frames=100,
            dropped_frames=2,
            face_present_pct=95.0,
        )
        assert s.data_source == "live"
