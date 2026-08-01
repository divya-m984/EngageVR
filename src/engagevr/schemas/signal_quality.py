"""Signal-quality schema.

Signal quality is distinct from model confidence and from engagement
level.  A poor-quality signal must never be interpreted as low
engagement.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from engagevr.schemas.modality import Modality


class ModalityQuality(BaseModel):
    """Quality assessment for a single modality within a window."""

    modality: Modality
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="0 = unusable, 1 = excellent.",
    )
    available: bool = True
    reason: str | None = Field(
        default=None,
        description="Human-readable note when quality is low or unavailable.",
    )


class SignalQualityReport(BaseModel):
    """Aggregated signal-quality report for one inference window."""

    session_id: str
    monotonic_timestamp: float
    modalities: list[ModalityQuality]
    overall_quality: float = Field(
        ge=0.0,
        le=1.0,
        description="Aggregate quality across available modalities.",
    )
