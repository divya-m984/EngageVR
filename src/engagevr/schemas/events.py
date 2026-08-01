"""Common event schemas for task telemetry and system events."""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class EventType(enum.StrEnum):
    """Standard event types for task and system telemetry."""

    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    TRIAL_STARTED = "trial_started"
    TRIAL_ENDED = "trial_ended"
    STIMULUS_PRESENTED = "stimulus_presented"
    RESPONSE_RECEIVED = "response_received"
    DIFFICULTY_CHANGED = "difficulty_changed"
    BREAK_TRIGGERED = "break_triggered"
    ADAPTATION_APPLIED = "adaptation_applied"


class BaseEvent(BaseModel):
    """Timestamped event base used by all telemetry producers."""

    session_id: str
    event_type: EventType
    monotonic_timestamp: float = Field(
        description="Session-local monotonic clock value."
    )
    data: dict[str, object] = Field(default_factory=dict)
    data_source: str = "synthetic"


class TaskEvent(BaseEvent):
    """Task-performance-specific event with trial metadata."""

    trial_index: int | None = None
    difficulty_level: int | None = None
    correct: bool | None = None
    reaction_time_ms: float | None = Field(
        default=None, ge=0.0, description="Reaction time in milliseconds."
    )
