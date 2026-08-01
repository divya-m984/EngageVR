"""Capture-layer schemas for webcam, landmarks, features, and quality.

All behavioural outputs are observable proxies. They are NOT direct
measurements of engagement, cognitive load, or emotional state.
Poor capture quality must never be interpreted as low engagement.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class FrameMetadata(BaseModel):
    """Metadata for a single captured frame."""

    session_id: str
    frame_index: int
    monotonic_timestamp: float
    utc_timestamp: datetime
    width: int
    height: int
    channels: int = 3


class NormalizedLandmark(BaseModel):
    """A single facial landmark in normalized image coordinates."""

    x: float = Field(description="Horizontal position, 0=left, 1=right.")
    y: float = Field(description="Vertical position, 0=top, 1=bottom.")
    z: float = Field(description="Relative depth (smaller = closer).")


class LandmarkObservation(BaseModel):
    """Landmark detection result for one frame."""

    session_id: str
    frame_index: int
    monotonic_timestamp: float
    face_detected: bool
    face_confidence: float | None = None
    landmarks: list[NormalizedLandmark] | None = None
    landmark_count: int = 0


class BehaviouralFeatures(BaseModel):
    """Behavioural proxy features from facial landmarks.

    These are observable behavioural proxies, NOT direct measurements
    of engagement, cognitive load, or emotional state.
    """

    session_id: str
    frame_index: int
    monotonic_timestamp: float

    face_present: bool
    face_confidence: float | None = None

    # Eye features (dimensionless ratio)
    left_ear: float | None = Field(
        default=None, description="Left Eye Aspect Ratio (dimensionless)."
    )
    right_ear: float | None = Field(
        default=None, description="Right Eye Aspect Ratio (dimensionless)."
    )
    mean_ear: float | None = Field(
        default=None, description="Mean EAR across both eyes."
    )
    blink_detected: bool | None = None
    eye_closure_duration_s: float | None = Field(
        default=None, description="Duration of current eye closure (seconds)."
    )

    # Mouth features (dimensionless ratio)
    mouth_aspect_ratio: float | None = Field(
        default=None, description="Mouth Aspect Ratio (dimensionless)."
    )

    # Tracking stability
    face_tracking_stable: bool | None = None


class HeadPoseObservation(BaseModel):
    """Head-pose estimate from facial landmarks.

    Angles are derived from a PnP solve using a generic 3D face model.
    They are approximate and not calibrated per individual.

    Reference: OpenCV solvePnP with a canonical face model.
    """

    session_id: str
    frame_index: int
    monotonic_timestamp: float

    available: bool
    yaw_deg: float | None = Field(
        default=None, description="Yaw in degrees (+ = turning right)."
    )
    pitch_deg: float | None = Field(
        default=None, description="Pitch in degrees (+ = looking up)."
    )
    roll_deg: float | None = Field(
        default=None, description="Roll in degrees (+ = CW tilt)."
    )

    angular_velocity_deg_s: float | None = Field(
        default=None,
        description="Angular velocity magnitude (degrees/second).",
    )
    motion_variability_deg_s: float | None = Field(
        default=None,
        description=("Std dev of angular velocity over a window (degrees/second)."),
    )


class CaptureQualityReport(BaseModel):
    """Quality indicators for a captured frame.

    Quality is distinct from engagement. Poor quality must NEVER
    produce a low-engagement label.
    """

    session_id: str
    frame_index: int
    monotonic_timestamp: float

    webcam_open: bool
    frame_read_success: bool
    observed_fps: float | None = None
    resolution_width: int | None = None
    resolution_height: int | None = None

    face_present: bool = False
    face_confidence: float | None = None
    landmarks_available: bool = False

    brightness: float | None = Field(
        default=None, description="Mean pixel intensity [0-255]."
    )
    underexposed: bool = False
    overexposed: bool = False
    blur_score: float | None = Field(
        default=None,
        description="Laplacian variance (higher = sharper).",
    )
    is_blurry: bool = False
    motion_score: float | None = Field(
        default=None,
        description="Mean absolute frame diff [0-255].",
    )
    excessive_motion: bool = False

    dropped_frames: int = 0
    missing_feature_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of unavailable features [0-100].",
    )


class CaptureSessionSummary(BaseModel):
    """Summary statistics for a capture session."""

    session_id: str
    total_frames: int
    dropped_frames: int
    mean_fps: float | None = None
    face_present_pct: float
    mean_brightness: float | None = None
    mean_blur_score: float | None = None
    data_source: str = "live"
