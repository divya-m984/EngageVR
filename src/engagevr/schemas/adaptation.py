"""Adaptation event and command schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AdaptationCommand(BaseModel):
    """Command sent to the task environment (Unity or simulator)."""

    command: str = Field(description="e.g. 'set_difficulty', 'trigger_break'.")
    value: int | float | str | bool
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class AdaptationEvent(BaseModel):
    """Full provenance record for one adaptation decision."""

    session_id: str
    monotonic_timestamp: float

    previous_state: dict[str, object]
    new_state: dict[str, object]

    engagement_estimate: float | None = None
    cognitive_load_estimate: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    signal_quality: float = Field(ge=0.0, le=1.0)

    triggering_evidence: str
    rule_used: str
    is_manual: bool = False
    command: AdaptationCommand
