"""Session metadata schema."""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class DataSource(enum.StrEnum):
    """Origin of data in a session."""

    SYNTHETIC = "synthetic"
    PUBLIC_DATASET = "public_dataset"
    LIVE = "live"


class ExperimentCondition(enum.StrEnum):
    """Experimental condition for the session."""

    STATIC = "static"
    ADAPTIVE = "adaptive"
    BASELINE = "baseline"


class Session(BaseModel):
    """Top-level session metadata.

    Participants are identified by pseudonymous IDs only.
    No names, emails, or biometric identifiers are stored.
    """

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    participant_id: str = Field(
        description="Pseudonymous participant identifier (no PII)."
    )
    experiment_condition: ExperimentCondition = ExperimentCondition.STATIC
    data_source: DataSource = DataSource.SYNTHETIC
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC wall-clock time at session start.",
    )
    end_time: datetime | None = None
    config_snapshot: dict[str, object] | None = Field(
        default=None,
        description="Frozen copy of the configuration used for this session.",
    )
