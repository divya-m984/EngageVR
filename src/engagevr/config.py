"""Configuration loading with Pydantic validation.

Loads YAML configuration from ``configs/defaults.yaml`` (or a caller-
supplied path) and validates it against typed Pydantic models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "defaults.yaml"


# --- Section models ---


class ProjectConfig(BaseModel):
    name: str = "EngageVR"
    version: str = "0.1.0"


class CaptureConfig(BaseModel):
    camera_index: int = 0
    width: int = 640
    height: int = 480
    webcam_fps_target: int = 30
    store_raw_video: bool = False
    preview: bool = False


class FaceConfig(BaseModel):
    model_path: str = "models/face_landmarker.task"
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_presence_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    blink_ear_threshold: float = Field(default=0.21, gt=0.0)
    eye_closure_min_frames: int = Field(default=3, ge=1)


class HeadPoseConfig(BaseModel):
    velocity_window_seconds: float = Field(default=1.0, gt=0.0)


class QualityConfig(BaseModel):
    brightness_low: float = Field(default=40.0, ge=0.0)
    brightness_high: float = Field(default=220.0, le=255.0)
    blur_threshold: float = Field(default=100.0, gt=0.0)
    motion_threshold: float = Field(default=30.0, gt=0.0)


class WindowingConfig(BaseModel):
    facial_aggregate_seconds: float = 5.0
    rppg_window_seconds: float = 30.0
    model_inference_seconds: float = 5.0
    baseline_calibration_seconds: float = 120.0


class SignalQualityConfig(BaseModel):
    min_face_detection_confidence: float = 0.5
    min_rppg_quality: float = 0.3
    min_hrv_window_seconds: float = 60.0


class ModelConfig(BaseModel):
    confidence_threshold: float = 0.5
    abstain_below_confidence: float = 0.3


class AdaptationConfig(BaseModel):
    enabled: bool = True
    cooldown_seconds: float = 30.0
    max_difficulty_change: int = 1
    require_min_signal_quality: float = 0.4
    require_min_confidence: float = 0.4


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    file: str | None = None


class SessionConfig(BaseModel):
    data_dir: str = "data"
    synthetic_dir: str = "data/synthetic"


# --- Root model ---


class EngageVRConfig(BaseModel):
    """Root configuration validated against ``configs/defaults.yaml``."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    face: FaceConfig = Field(default_factory=FaceConfig)
    head_pose: HeadPoseConfig = Field(default_factory=HeadPoseConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    windowing: WindowingConfig = Field(default_factory=WindowingConfig)
    signal_quality: SignalQualityConfig = Field(default_factory=SignalQualityConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    adaptation: AdaptationConfig = Field(default_factory=AdaptationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> Self:
        """Load and validate configuration from a YAML file.

        Parameters
        ----------
        path:
            Path to the YAML file.  Falls back to ``configs/defaults.yaml``
            relative to the package root.
        """
        config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
        if not config_path.exists():
            return cls()
        with open(config_path) as fh:
            raw: dict[str, object] = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)


def load_config(path: str | Path | None = None) -> EngageVRConfig:
    """Convenience wrapper for :meth:`EngageVRConfig.from_yaml`."""
    return EngageVRConfig.from_yaml(path)
