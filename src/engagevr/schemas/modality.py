"""Modality identifiers and timestamped samples."""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class Modality(enum.StrEnum):
    """Identifiers for each signal modality."""

    FACE = "face"
    HEAD_POSE = "head_pose"
    RPPG = "rppg"
    HEART_RATE = "heart_rate"
    HRV = "hrv"
    TASK_PERFORMANCE = "task_performance"
    SUBJECTIVE = "subjective"
    EDA = "eda"
    WEARABLE_ECG = "wearable_ecg"
    WEARABLE_PPG = "wearable_ppg"
    RESPIRATION = "respiration"


class ModalitySample(BaseModel):
    """A single timestamped sample from one modality."""

    session_id: str
    modality: Modality
    monotonic_timestamp: float = Field(
        description="Session-local monotonic clock value in seconds."
    )
    raw_value: dict[str, object] | None = None
    processed_value: dict[str, object] | None = None
