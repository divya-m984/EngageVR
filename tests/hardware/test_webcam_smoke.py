"""Opt-in hardware smoke test for physical webcam.

Skipped unless ENGAGEVR_RUN_HARDWARE_TESTS=1 is set.
Never runs in GitHub Actions by default.
Never saves or transmits frames.

Usage::

    ENGAGEVR_RUN_HARDWARE_TESTS=1 uv run pytest -m hardware -v
"""

from __future__ import annotations

import os

import pytest

_ENABLED = os.environ.get("ENGAGEVR_RUN_HARDWARE_TESTS") == "1"
_REASON = "Set ENGAGEVR_RUN_HARDWARE_TESTS=1 to enable hardware tests"


@pytest.mark.hardware
class TestWebcamSmoke:
    @pytest.fixture(autouse=True)
    def _skip_unless_enabled(self):
        if not _ENABLED:
            pytest.skip(_REASON)

    def test_open_and_read_frames(self):
        from engagevr.capture.webcam import WebcamCapture
        from engagevr.config import load_config

        cfg = load_config()
        cam = WebcamCapture(
            camera_index=cfg.capture.camera_index,
            width=cfg.capture.width,
            height=cfg.capture.height,
            fps=cfg.capture.webcam_fps_target,
            session_id="hardware_smoke",
        )

        try:
            opened = cam.open()
            assert opened, (
                f"Cannot open camera {cfg.capture.camera_index}. Is a webcam connected?"
            )

            n_frames = 5
            timestamps: list[float] = []

            for _ in range(n_frames):
                meta, frame = cam.read_frame()
                assert meta is not None, "Failed to read frame"
                assert frame is not None
                assert frame.shape[0] > 0, "Frame height must be positive"
                assert frame.shape[1] > 0, "Frame width must be positive"
                assert frame.shape[2] == 3, "Frame must be 3-channel BGR"
                timestamps.append(meta.monotonic_timestamp)

            # Timestamps must be monotonically non-decreasing
            for i in range(1, len(timestamps)):
                assert timestamps[i] >= timestamps[i - 1], (
                    f"Timestamps not monotonic: {timestamps}"
                )

            # Verify reported dimensions
            w = cam.actual_width()
            h = cam.actual_height()
            assert w is not None and w > 0
            assert h is not None and h > 0

        finally:
            cam.release()
            assert not cam.is_opened(), "Camera not released"
