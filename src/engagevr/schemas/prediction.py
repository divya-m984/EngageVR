"""Engagement and cognitive-load prediction schema.

Predictions are model estimates, not diagnostic conclusions.
The schema supports explicit abstention when evidence is insufficient.
Confidence and signal quality are separate dimensions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from engagevr.schemas.modality import Modality


class EngagementPrediction(BaseModel):
    """Single prediction output from the ML layer.

    When ``abstain`` is True, estimate fields are None and ``reason``
    explains why the model declined to predict.
    """

    session_id: str
    monotonic_timestamp: float

    engagement_estimate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Estimated engagement level (0-1). None when abstaining.",
    )
    cognitive_load_estimate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Estimated cognitive load (0-1). None when abstaining.",
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Model confidence in this prediction. "
            "This is NOT signal quality -- see signal_quality field."
        ),
    )
    signal_quality: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Aggregate signal quality for the input window. "
            "This is NOT model confidence -- see confidence field."
        ),
    )

    available_modalities: list[Modality] = Field(default_factory=list)
    missing_modalities: list[Modality] = Field(default_factory=list)

    abstain: bool = False
    reason: str | None = Field(
        default=None,
        description="Explanation when abstaining or when confidence is low.",
    )

    data_source: str = "synthetic"
