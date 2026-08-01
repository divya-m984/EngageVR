"""Tests for configuration loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from engagevr.config import EngageVRConfig, load_config


class TestConfigDefaults:
    def test_load_defaults_yaml(self):
        cfg = load_config()
        assert cfg.project.name == "EngageVR"
        assert cfg.capture.store_raw_video is False

    def test_default_windowing(self):
        cfg = load_config()
        assert cfg.windowing.rppg_window_seconds == 30.0
        assert cfg.windowing.facial_aggregate_seconds == 5.0

    def test_default_adaptation(self):
        cfg = load_config()
        assert cfg.adaptation.cooldown_seconds == 30.0
        assert cfg.adaptation.enabled is True

    def test_signal_quality_thresholds(self):
        cfg = load_config()
        assert cfg.signal_quality.min_hrv_window_seconds == 60.0

    def test_model_confidence_thresholds(self):
        cfg = load_config()
        assert cfg.model.confidence_threshold > cfg.model.abstain_below_confidence

    def test_raw_video_disabled_by_default(self):
        cfg = load_config()
        assert cfg.capture.store_raw_video is False


class TestConfigFromYaml:
    def test_load_from_custom_path(self, tmp_path: Path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
            project:
              name: "TestProject"
            capture:
              webcam_fps_target: 15
            """)
        )
        cfg = load_config(yaml_file)
        assert cfg.project.name == "TestProject"
        assert cfg.capture.webcam_fps_target == 15
        # Unspecified sections get defaults
        assert cfg.adaptation.cooldown_seconds == 30.0

    def test_missing_file_returns_defaults(self, tmp_path: Path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.project.name == "EngageVR"

    def test_empty_file_returns_defaults(self, tmp_path: Path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        cfg = load_config(yaml_file)
        assert cfg.project.name == "EngageVR"

    def test_partial_override(self, tmp_path: Path):
        yaml_file = tmp_path / "partial.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
            adaptation:
              cooldown_seconds: 60.0
            """)
        )
        cfg = load_config(yaml_file)
        assert cfg.adaptation.cooldown_seconds == 60.0
        assert cfg.adaptation.enabled is True  # default preserved


class TestConfigValidation:
    def test_invalid_section_rejected(self):
        with pytest.raises(ValidationError):
            EngageVRConfig.model_validate(
                {"capture": {"webcam_fps_target": "not_a_number"}}
            )
