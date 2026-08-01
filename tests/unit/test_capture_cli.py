"""Tests for capture CLI commands -- no webcam or model required."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from engagevr.__main__ import main


class TestCaptureCli:
    def test_capture_no_model_exits_nonzero(self, tmp_path, monkeypatch):
        """With a nonexistent model path, capture exits with model error."""
        from engagevr import config as cfg_mod

        orig = cfg_mod.load_config

        def _patched(path=None):  # type: ignore[no-untyped-def]
            c = orig(path)
            c.face.model_path = str(tmp_path / "missing.task")
            return c

        monkeypatch.setattr(cfg_mod, "load_config", _patched)
        rc = main(["capture", "--duration", "1"])
        assert rc != 0

    def test_capture_no_camera_exits_nonzero(self):
        """With model mocked but no camera, capture exits with error."""
        mock_landmarker = MagicMock()
        with patch(
            "engagevr.face.landmarker.FaceLandmarkerWrapper",
            return_value=mock_landmarker,
        ):
            rc = main(["capture", "--camera", "99", "--duration", "1"])
        assert rc != 0
        mock_landmarker.close.assert_called()
