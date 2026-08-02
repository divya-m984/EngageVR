"""Configuration loading with Pydantic validation.

Loads YAML configuration from ``configs/defaults.yaml`` (or a caller-
supplied path) and validates it against typed Pydantic models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator

from engagevr.schemas.rppg import RoiRegion, RppgMethod

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


class RppgRoiConfig(BaseModel):
    """Face-skin region-of-interest sampling parameters."""

    regions: list[RoiRegion] = Field(
        default_factory=lambda: [
            RoiRegion.FOREHEAD,
            RoiRegion.LEFT_CHEEK,
            RoiRegion.RIGHT_CHEEK,
        ],
        min_length=1,
        description="Regions sampled and pooled into the combined ROI.",
    )
    inset_fraction: float = Field(
        default=0.15,
        ge=0.0,
        lt=0.45,
        description=(
            "Fraction of each ROI half-extent trimmed inward, to exclude "
            "hairline, eyebrows, eyes, nostrils, and background."
        ),
    )
    min_valid_pixels: int = Field(
        default=200,
        ge=1,
        description="Minimum non-clipped pixels for a usable region.",
    )
    min_valid_pixel_pct: float = Field(
        default=60.0,
        gt=0.0,
        le=100.0,
        description="Minimum percentage of ROI pixels that must be non-clipped.",
    )
    clipping_low: int = Field(
        default=5,
        ge=0,
        le=255,
        description="Pixel values at or below this are treated as crushed.",
    )
    clipping_high: int = Field(
        default=250,
        ge=0,
        le=255,
        description="Pixel values at or above this are treated as saturated.",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.clipping_low >= self.clipping_high:
            raise ValueError("rppg.roi.clipping_low must be < clipping_high")
        if RoiRegion.COMBINED in self.regions:
            raise ValueError(
                "rppg.roi.regions must list source regions only; "
                "'combined' is derived, not configured"
            )
        if len(set(self.regions)) != len(self.regions):
            raise ValueError("rppg.roi.regions must not contain duplicates")
        return self


class RppgTraceConfig(BaseModel):
    """RGB trace acquisition and timing-integrity parameters."""

    expected_fps: float = Field(
        default=30.0,
        gt=0.0,
        description="Nominal sampling rate; used for the Nyquist check.",
    )
    max_timestamp_jitter_s: float = Field(
        default=0.02,
        gt=0.0,
        description=(
            "Maximum tolerated standard deviation of inter-sample intervals "
            "before a window is rejected rather than resampled."
        ),
    )
    min_valid_frame_pct: float = Field(
        default=80.0,
        gt=0.0,
        le=100.0,
        description="Minimum percentage of frames with a usable ROI.",
    )


class RppgWindowConfig(BaseModel):
    """Sliding-window geometry for rPPG estimation."""

    duration_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Window length. Specification recommends 20-30 s.",
    )
    step_seconds: float = Field(
        default=1.0,
        gt=0.0,
        description="Hop between consecutive windows.",
    )
    min_duration_seconds: float = Field(
        default=8.0,
        gt=0.0,
        description=(
            "Absolute minimum analysable duration. Below this the spectral "
            "resolution is too coarse for a defensible BPM estimate."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.step_seconds > self.duration_seconds:
            raise ValueError(
                "rppg.window.step_seconds must not exceed duration_seconds"
            )
        if self.min_duration_seconds > self.duration_seconds:
            raise ValueError(
                "rppg.window.min_duration_seconds must not exceed duration_seconds"
            )
        return self


class RppgPreprocessingConfig(BaseModel):
    """Detrending, normalization, and band-pass filter design.

    Pulse-band rationale
    --------------------
    de Haan & Jeanne (2013, DOI 10.1109/TBME.2013.2266196) band-limit the
    chrominance signals to a 40-240 BPM pulse range, i.e. 0.67-4.0 Hz.
    The default band below (0.7-4.0 Hz = 42-240 BPM) follows that range.
    """

    detrend: bool = True
    normalize: bool = Field(
        default=True,
        description="Divide each channel by its temporal mean (CHROM/POS step).",
    )
    resample_uniform: bool = Field(
        default=True,
        description="Resample onto a uniform grid when jitter is within tolerance.",
    )
    filter_order: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Butterworth order, realised as second-order sections.",
    )
    pulse_band_low_hz: float = Field(default=0.7, gt=0.0)
    pulse_band_high_hz: float = Field(default=4.0, gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.pulse_band_low_hz >= self.pulse_band_high_hz:
            raise ValueError(
                "rppg.preprocessing.pulse_band_low_hz must be < pulse_band_high_hz"
            )
        return self


class RppgSpectralConfig(BaseModel):
    """Welch PSD and spectral-peak acceptance parameters."""

    welch_segment_seconds: float = Field(
        default=8.0,
        gt=0.0,
        description="Welch segment length in seconds (sets frequency resolution).",
    )
    welch_overlap: float = Field(
        default=0.5,
        ge=0.0,
        lt=1.0,
        description="Fractional overlap between Welch segments.",
    )
    peak_bandwidth_hz: float = Field(
        default=0.1,
        gt=0.0,
        description="Half-width around the peak used for spectral concentration.",
    )
    min_relative_peak_prominence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum peak prominence divided by peak power. Scale-invariant, "
            "so it does not depend on arbitrary waveform amplitude."
        ),
    )
    min_spectral_peak_ratio: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of in-band power concentrated at the peak.",
    )


class RppgQualityConfig(BaseModel):
    """Thresholds for the interpretable rPPG signal-quality index."""

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum overall quality for a heart rate to be reported.",
    )
    max_illumination_cv: float = Field(
        default=0.05,
        gt=0.0,
        description=(
            "Coefficient of variation of ROI brightness scored as fully "
            "unstable. Below this, illumination is scored as stable."
        ),
    )
    max_clipped_pixel_pct: float = Field(
        default=5.0,
        gt=0.0,
        le=100.0,
        description="Clipped-pixel percentage scored as fully saturated.",
    )
    max_motion_score: float = Field(
        default=30.0,
        gt=0.0,
        description="Capture motion score treated as fully motion-corrupted.",
    )
    method_agreement_tolerance_bpm: float = Field(
        default=10.0,
        gt=0.0,
        description="BPM disagreement between methods scored as full mismatch.",
    )


class RppgDatasetConfig(BaseModel):
    """Local roots for public datasets. Nothing is downloaded automatically."""

    ubfc_rppg_root: str | None = Field(
        default=None,
        description=(
            "Path to a locally obtained UBFC-rPPG root. Never fetched by "
            "this software; see docs/DATASETS.md."
        ),
    )


class RppgConfig(BaseModel):
    """Root rPPG configuration."""

    method: RppgMethod = RppgMethod.POS
    roi: RppgRoiConfig = Field(default_factory=RppgRoiConfig)
    trace: RppgTraceConfig = Field(default_factory=RppgTraceConfig)
    window: RppgWindowConfig = Field(default_factory=RppgWindowConfig)
    preprocessing: RppgPreprocessingConfig = Field(
        default_factory=RppgPreprocessingConfig
    )
    spectral: RppgSpectralConfig = Field(default_factory=RppgSpectralConfig)
    quality: RppgQualityConfig = Field(default_factory=RppgQualityConfig)
    datasets: RppgDatasetConfig = Field(default_factory=RppgDatasetConfig)

    @model_validator(mode="after")
    def _check(self) -> Self:
        nyquist = self.trace.expected_fps / 2.0
        if self.preprocessing.pulse_band_high_hz >= nyquist:
            raise ValueError(
                "rppg.preprocessing.pulse_band_high_hz "
                f"({self.preprocessing.pulse_band_high_hz} Hz) must be strictly "
                f"below the Nyquist frequency ({nyquist} Hz) implied by "
                f"rppg.trace.expected_fps ({self.trace.expected_fps} Hz)"
            )
        if self.spectral.welch_segment_seconds > self.window.duration_seconds:
            raise ValueError(
                "rppg.spectral.welch_segment_seconds must not exceed "
                "rppg.window.duration_seconds"
            )
        return self


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
    rppg: RppgConfig = Field(default_factory=RppgConfig)
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
