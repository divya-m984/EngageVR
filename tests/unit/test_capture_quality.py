"""Tests for capture quality metrics."""

from __future__ import annotations

import numpy as np

from engagevr.capture.quality import (
    assess_blur,
    assess_brightness,
    assess_motion,
    compute_blur_score,
    compute_brightness,
    compute_motion_score,
    count_missing_features,
)
from engagevr.config import QualityConfig


def _default_quality_config() -> QualityConfig:
    return QualityConfig(
        brightness_low=40.0,
        brightness_high=220.0,
        blur_threshold=100.0,
        motion_threshold=30.0,
    )


class TestBrightness:
    def test_dark_frame(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        assert compute_brightness(gray) == 0.0

    def test_bright_frame(self):
        gray = np.full((100, 100), 255, dtype=np.uint8)
        assert compute_brightness(gray) == 255.0

    def test_mid_brightness(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        assert compute_brightness(gray) == 128.0

    def test_underexposed(self):
        cfg = _default_quality_config()
        under, over = assess_brightness(20.0, cfg)
        assert under is True
        assert over is False

    def test_overexposed(self):
        cfg = _default_quality_config()
        under, over = assess_brightness(240.0, cfg)
        assert under is False
        assert over is True

    def test_normal_exposure(self):
        cfg = _default_quality_config()
        under, over = assess_brightness(128.0, cfg)
        assert under is False
        assert over is False


class TestBlurScore:
    def test_sharp_image(self):
        # Checkerboard has high variance
        img = np.zeros((100, 100), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
        score = compute_blur_score(img)
        assert score > 100

    def test_uniform_image_is_blurry(self):
        img = np.full((100, 100), 128, dtype=np.uint8)
        score = compute_blur_score(img)
        assert score < 1.0

    def test_assess_blur(self):
        cfg = _default_quality_config()
        assert assess_blur(50.0, cfg) is True  # below threshold
        assert assess_blur(200.0, cfg) is False


class TestMotionScore:
    def test_no_previous_frame(self):
        curr = np.full((100, 100), 128, dtype=np.uint8)
        assert compute_motion_score(None, curr) is None

    def test_identical_frames(self):
        frame = np.full((100, 100), 128, dtype=np.uint8)
        score = compute_motion_score(frame, frame)
        assert score is not None
        assert score == 0.0

    def test_different_frames(self):
        prev = np.full((100, 100), 100, dtype=np.uint8)
        curr = np.full((100, 100), 200, dtype=np.uint8)
        score = compute_motion_score(prev, curr)
        assert score is not None
        assert score == 100.0

    def test_excessive_motion(self):
        cfg = _default_quality_config()
        assert assess_motion(50.0, cfg) is True
        assert assess_motion(10.0, cfg) is False
        assert assess_motion(None, cfg) is False

    def test_shape_mismatch(self):
        prev = np.zeros((100, 100), dtype=np.uint8)
        curr = np.zeros((50, 50), dtype=np.uint8)
        assert compute_motion_score(prev, curr) is None


class TestMissingFeatures:
    def test_no_missing(self):
        assert count_missing_features({"a": 1, "b": 2}) == 0.0

    def test_all_missing(self):
        assert count_missing_features({"a": None, "b": None}) == 100.0

    def test_half_missing(self):
        assert count_missing_features({"a": 1, "b": None}) == 50.0

    def test_empty_dict(self):
        assert count_missing_features({}) == 0.0
