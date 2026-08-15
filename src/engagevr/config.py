"""Configuration loading with Pydantic validation.

Loads YAML configuration from ``configs/defaults.yaml`` (or a caller-
supplied path) and validates it against typed Pydantic models.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator

from engagevr.protocol.version import (
    ACCEPTED_MAJOR_VERSIONS,
    PROTOCOL_VERSION,
    ProtocolVersionError,
    parse_protocol_version,
)
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


# --- Milestone 4: real-time bridge, task simulator, storage, replay ---


class ServerConfig(BaseModel):
    """Local development server binding and connection limits.

    The server binds to loopback by default.  Binding to a
    non-loopback address requires ``allow_public_bind`` to be set
    explicitly, because this prototype has **no authentication,
    authorization, or transport encryption** and must not be reachable
    from a network.
    """

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)
    allow_public_bind: bool = Field(
        default=False,
        description="Required before binding anywhere other than loopback.",
    )
    maximum_message_bytes: int = Field(
        default=262_144,
        ge=1024,
        le=16_777_216,
        description="Largest accepted single WebSocket frame.",
    )
    heartbeat_interval_seconds: float = Field(
        default=10.0, gt=0.0, description="How often the server expects liveness."
    )
    connection_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Silence after which a connection is closed as timed out.",
    )
    acknowledgement_timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="How long a sender waits for an acknowledgement.",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.connection_timeout_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                "server.connection_timeout_seconds must exceed "
                "server.heartbeat_interval_seconds, otherwise a healthy "
                "client is timed out between heartbeats"
            )
        if not self.allow_public_bind and not _is_loopback_host(self.host):
            raise ValueError(
                f"server.host {self.host!r} is not a loopback address. This "
                "prototype has no authentication and must not be exposed. "
                "Set server.allow_public_bind: true to override deliberately."
            )
        return self


def _is_loopback_host(host: str) -> bool:
    """Whether ``host`` refers only to the local machine."""
    normalized = host.strip().strip("[]").lower()
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    return normalized.startswith("127.")


class ProtocolSettings(BaseModel):
    """Protocol version acceptance and message-integrity tolerances."""

    version: str = Field(default=PROTOCOL_VERSION)
    accepted_major_versions: list[int] = Field(
        default_factory=lambda: list(ACCEPTED_MAJOR_VERSIONS), min_length=1
    )
    maximum_clock_skew_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "A sender timestamp further in the future than this is flagged "
            "as a clock observation. It never causes a rejection."
        ),
    )
    maximum_sequence_gap: int = Field(
        default=1000,
        ge=0,
        description="Largest missing sequence range that is enumerated in full.",
    )
    maximum_transport_delay_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description=(
            "Delay beyond which an in-process message is flagged as slow. "
            "Not applied across processes, where the subtraction would "
            "measure clock offset rather than delay."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        try:
            major, _minor = parse_protocol_version(self.version)
        except ProtocolVersionError as exc:
            raise ValueError(f"protocol.version is invalid: {exc}") from exc
        if major not in self.accepted_major_versions:
            raise ValueError(
                f"protocol.version {self.version!r} has major version {major}, "
                "which is not in protocol.accepted_major_versions "
                f"{self.accepted_major_versions}"
            )
        unsupported = [
            v for v in self.accepted_major_versions if v not in ACCEPTED_MAJOR_VERSIONS
        ]
        if unsupported:
            raise ValueError(
                f"protocol.accepted_major_versions lists {unsupported}, which "
                "this build cannot parse; it implements "
                f"{list(ACCEPTED_MAJOR_VERSIONS)}"
            )
        return self


class QueueConfig(BaseModel):
    """Bounded-queue capacities for the ingestion pipeline.

    Every queue is bounded.  Unbounded queueing would let a fast
    producer consume memory without limit, which is a failure mode that
    surfaces as an out-of-memory kill rather than as a diagnosable
    backpressure event.
    """

    ingestion_capacity: int = Field(default=1024, ge=1)
    storage_capacity: int = Field(default=1024, ge=1)
    broadcast_capacity: int = Field(default=256, ge=1)
    operation_timeout_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description=(
            "How long a critical message may wait for queue space before "
            "the connection is failed. Non-critical messages are dropped "
            "immediately instead of waiting."
        ),
    )


class SessionsConfig(BaseModel):
    """Where session recordings are written and how they are flushed."""

    root_directory: str = Field(default="artifacts/sessions", min_length=1)
    flush_every_events: int = Field(
        default=1,
        ge=1,
        description="Events buffered before flushing to the OS. 1 = every event.",
    )
    atomic_summary: bool = Field(
        default=True,
        description="Write summary.json via a temporary file and atomic rename.",
    )
    recover_incomplete_sessions: bool = Field(
        default=True,
        description="Rebuild a summary from a partial event stream when reading.",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        path = PurePosixPath(self.root_directory)
        if path.is_absolute():
            raise ValueError(
                "sessions.root_directory must be a relative path inside the "
                "project; absolute roots are rejected so a configuration "
                "mistake cannot write outside the workspace"
            )
        if any(part == ".." for part in path.parts):
            raise ValueError(
                "sessions.root_directory must not contain '..' path segments"
            )
        if not self.atomic_summary:
            raise ValueError(
                "sessions.atomic_summary cannot be disabled: a non-atomic "
                "summary write can leave a half-written summary.json that "
                "reads as a valid but wrong session record"
            )
        return self


class TaskConfig(BaseModel):
    """Reaction-task geometry and synthetic-response behaviour.

    The synthetic rates below describe how the **simulator** fabricates
    responses.  They are software test parameters.  They are not
    measurements, not participant data, and not a model of human
    performance.
    """

    blocks: int = Field(default=2, ge=1)
    trials_per_block: int = Field(default=10, ge=1)
    stimulus_duration_ms: float = Field(default=800.0, gt=0.0)
    response_timeout_ms: float = Field(default=1500.0, gt=0.0)
    inter_trial_interval_ms: float = Field(default=500.0, ge=0.0)
    default_difficulty: int = Field(default=1, ge=0)
    synthetic_response_latency_ms: float = Field(
        default=450.0,
        gt=0.0,
        description="Mean fabricated reaction time. SYNTHETIC, not measured.",
    )
    synthetic_error_rate: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Fabricated incorrect-response rate."
    )
    synthetic_timeout_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Fabricated no-response rate."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.synthetic_error_rate + self.synthetic_timeout_rate > 1.0:
            raise ValueError(
                "task.synthetic_error_rate + task.synthetic_timeout_rate must "
                "not exceed 1.0; there would be no probability mass left for "
                "correct responses"
            )
        if self.response_timeout_ms < self.stimulus_duration_ms:
            raise ValueError(
                "task.response_timeout_ms must not be shorter than "
                "task.stimulus_duration_ms, otherwise a trial times out while "
                "its stimulus is still on screen"
            )
        return self


class ReplayConfig(BaseModel):
    """Replay pacing limits."""

    default_speed: float = Field(
        default=1.0,
        ge=0.0,
        description="1.0 = original timing. 0 = immediate, no sleeping.",
    )
    maximum_speed: float = Field(default=1000.0, gt=0.0)
    preserve_original_timing: bool = Field(
        default=True,
        description="Pace by recorded inter-message gaps rather than a fixed rate.",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.default_speed > self.maximum_speed:
            raise ValueError(
                "replay.default_speed must not exceed replay.maximum_speed"
            )
        return self


# --- Milestone 5: feature datasets and interpretable baselines ---


class AggregationSettings(BaseModel):
    """Minimum evidence required before a window feature is computed.

    These are software thresholds that stop a summary being computed from
    evidence too thin to support it.  They are **not** empirically
    validated cut-offs, and meeting them does not make a window
    scientifically adequate.
    """

    min_face_frames: int = Field(default=5, ge=1)
    min_face_seconds: float = Field(default=1.0, gt=0.0)
    min_pose_frames: int = Field(default=5, ge=1)
    min_quality_frames: int = Field(default=1, ge=1)
    min_resolved_trials: int = Field(default=1, ge=1)
    min_reaction_times: int = Field(default=1, ge=1)
    min_rppg_windows: int = Field(default=1, ge=1)


class FeaturesConfig(BaseModel):
    """Windowed-feature dataset geometry and storage location."""

    catalog_version: str = Field(default="1.0", min_length=1)
    window_duration_seconds: float = Field(default=10.0, gt=0.0)
    window_step_seconds: float = Field(default=10.0, gt=0.0)
    partial_window_policy: str = Field(default="drop")
    minimum_partial_duration_seconds: float = Field(default=0.0, ge=0.0)
    dataset_directory: str = Field(default="artifacts/datasets", min_length=1)
    aggregation: AggregationSettings = Field(default_factory=AggregationSettings)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.partial_window_policy not in {"drop", "keep_if_minimum"}:
            raise ValueError(
                "features.partial_window_policy must be 'drop' or "
                f"'keep_if_minimum'; got {self.partial_window_policy!r}"
            )
        if self.window_step_seconds > self.window_duration_seconds:
            raise ValueError(
                "features.window_step_seconds must not exceed "
                "window_duration_seconds, otherwise evidence between windows "
                "is silently discarded"
            )
        if self.partial_window_policy == "keep_if_minimum":
            if self.minimum_partial_duration_seconds <= 0.0:
                raise ValueError(
                    "features.partial_window_policy 'keep_if_minimum' requires "
                    "a positive minimum_partial_duration_seconds"
                )
            if self.minimum_partial_duration_seconds > self.window_duration_seconds:
                raise ValueError(
                    "features.minimum_partial_duration_seconds must not exceed "
                    "features.window_duration_seconds"
                )
        return self

    @property
    def windows_overlap(self) -> bool:
        """Whether consecutive windows share evidence."""
        return self.window_step_seconds < self.window_duration_seconds


class TrainingConfig(BaseModel):
    """Cross-validation, calibration, and experiment-tracking settings.

    Only grouped cross-validation is configurable.  There is no setting
    that enables row-level splitting: with several windows per session it
    would place the same session on both sides of a fold boundary.
    """

    experiment_directory: str = Field(default="artifacts/experiments", min_length=1)
    random_seed: int = 42
    folds: int = Field(default=5, ge=2)
    calibration_method: str = Field(default="sigmoid")
    calibration_group_fraction: float = Field(default=0.25, ge=0.0, lt=1.0)
    calibration_bins: int = Field(default=10, ge=2)
    permutation_repeats: int = Field(default=5, ge=1)
    run_ablations: bool = True
    tune_hyperparameters: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        allowed = {"none", "sigmoid", "isotonic"}
        if self.calibration_method not in allowed:
            raise ValueError(
                "training.calibration_method must be one of "
                f"{sorted(allowed)}; got {self.calibration_method!r}"
            )
        if self.calibration_method != "none" and self.calibration_group_fraction <= 0.0:
            raise ValueError(
                "training.calibration_group_fraction must be positive when a "
                "calibration method is requested; otherwise the calibrator "
                "would have to be fitted on the estimator's own training rows"
            )
        return self


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

    # Milestone 4
    server: ServerConfig = Field(default_factory=ServerConfig)
    protocol: ProtocolSettings = Field(default_factory=ProtocolSettings)
    queues: QueueConfig = Field(default_factory=QueueConfig)
    sessions: SessionsConfig = Field(default_factory=SessionsConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)

    # Milestone 5
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

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
