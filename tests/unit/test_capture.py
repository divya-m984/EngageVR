"""Tests for webcam capture with fake backend."""

from __future__ import annotations

import numpy as np

from engagevr.capture.webcam import FakeCaptureBackend, WebcamCapture


class TestWebcamCapture:
    def test_open_success(self):
        backend = FakeCaptureBackend()
        cam = WebcamCapture(backend=backend, session_id="s1")
        assert cam.open()
        assert cam.is_opened()
        cam.release()
        assert not cam.is_opened()

    def test_open_failure(self):
        backend = FakeCaptureBackend(fail_open=True)
        cam = WebcamCapture(backend=backend)
        assert not cam.open()
        assert not cam.is_opened()

    def test_read_frame_success(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        backend = FakeCaptureBackend(frames=[frame])
        cam = WebcamCapture(backend=backend, session_id="s1")
        cam.open()
        meta, f = cam.read_frame()
        assert meta is not None
        assert f is not None
        assert meta.width == 640
        assert meta.height == 480
        assert meta.session_id == "s1"
        assert meta.frame_index == 0
        cam.release()

    def test_read_frame_failure(self):
        backend = FakeCaptureBackend(fail_after=0)
        cam = WebcamCapture(backend=backend)
        cam.open()
        meta, frame = cam.read_frame()
        assert meta is None
        assert frame is None
        cam.release()

    def test_read_before_open(self):
        backend = FakeCaptureBackend()
        cam = WebcamCapture(backend=backend)
        meta, _frame = cam.read_frame()
        assert meta is None

    def test_frame_index_increments(self):
        backend = FakeCaptureBackend()
        cam = WebcamCapture(backend=backend, session_id="s1")
        cam.open()
        for i in range(3):
            meta, _ = cam.read_frame()
            assert meta is not None
            assert meta.frame_index == i
        assert cam.frame_index == 3
        cam.release()

    def test_timestamp_monotonicity(self):
        backend = FakeCaptureBackend()
        cam = WebcamCapture(backend=backend, session_id="s1")
        cam.open()
        timestamps = []
        for _ in range(5):
            meta, _ = cam.read_frame()
            assert meta is not None
            timestamps.append(meta.monotonic_timestamp)
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]
        cam.release()

    def test_context_manager(self):
        backend = FakeCaptureBackend()
        with WebcamCapture(backend=backend, session_id="s1") as cam:
            assert cam.is_opened()
        assert not cam.is_opened()

    def test_context_manager_failure(self):
        backend = FakeCaptureBackend(fail_open=True)
        with WebcamCapture(backend=backend) as cam:
            assert not cam.is_opened()

    def test_release_idempotent(self):
        backend = FakeCaptureBackend()
        cam = WebcamCapture(backend=backend)
        cam.open()
        cam.release()
        cam.release()  # should not raise

    def test_actual_properties(self):
        backend = FakeCaptureBackend()
        cam = WebcamCapture(backend=backend)
        cam.open()
        assert cam.actual_fps() == 30.0
        assert cam.actual_width() == 640
        assert cam.actual_height() == 480
        cam.release()

    def test_raw_video_not_stored(self):
        """Verify no raw video persistence mechanism exists by default."""
        from engagevr.config import load_config

        cfg = load_config()
        assert cfg.capture.store_raw_video is False
