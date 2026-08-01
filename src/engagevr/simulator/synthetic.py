"""Deterministic synthetic session generator.

Produces labelled synthetic sessions for integration testing.
ALL generated data carries ``data_source = "synthetic"`` permanently.
This data must NEVER be presented as experimental evidence.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

from engagevr.schemas.events import BaseEvent, EventType, TaskEvent
from engagevr.schemas.modality import Modality, ModalitySample
from engagevr.schemas.prediction import EngagementPrediction
from engagevr.schemas.session import DataSource, ExperimentCondition, Session
from engagevr.schemas.signal_quality import (
    ModalityQuality,
    SignalQualityReport,
)

_DATA_SOURCE = "synthetic"


class SyntheticSession:
    """Container for a complete synthetic session with all generated artefacts."""

    def __init__(
        self,
        session: Session,
        events: list[BaseEvent | TaskEvent],
        samples: list[ModalitySample],
        predictions: list[EngagementPrediction],
        quality_reports: list[SignalQualityReport],
    ) -> None:
        self.session = session
        self.events = events
        self.samples = samples
        self.predictions = predictions
        self.quality_reports = quality_reports

    def to_dict(self) -> dict[str, object]:
        """Serialize the full session to a JSON-compatible dict."""
        return {
            "session": self.session.model_dump(mode="json"),
            "events": [e.model_dump(mode="json") for e in self.events],
            "samples": [s.model_dump(mode="json") for s in self.samples],
            "predictions": [p.model_dump(mode="json") for p in self.predictions],
            "quality_reports": [
                q.model_dump(mode="json") for q in self.quality_reports
            ],
        }


def generate_synthetic_session(
    *,
    participant_id: str = "synth_participant_001",
    n_trials: int = 10,
    seed: int = 42,
    condition: ExperimentCondition = ExperimentCondition.STATIC,
) -> SyntheticSession:
    """Generate a complete synthetic session for testing.

    Parameters
    ----------
    participant_id:
        Pseudonymous identifier for the synthetic participant.
    n_trials:
        Number of task trials to generate.
    seed:
        Random seed for reproducibility.
    condition:
        Experimental condition to assign.

    Returns
    -------
    SyntheticSession
        A session with events, modality samples, predictions, and
        quality reports -- all labelled ``data_source = "synthetic"``.
    """
    rng = random.Random(seed)

    session = Session(
        participant_id=participant_id,
        experiment_condition=condition,
        data_source=DataSource.SYNTHETIC,
        start_time=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    sid = session.session_id

    events: list[BaseEvent | TaskEvent] = []
    samples: list[ModalitySample] = []
    predictions: list[EngagementPrediction] = []
    quality_reports: list[SignalQualityReport] = []

    t = 0.0

    # Session start event
    events.append(
        BaseEvent(
            session_id=sid,
            event_type=EventType.SESSION_STARTED,
            monotonic_timestamp=t,
            data_source=_DATA_SOURCE,
        )
    )

    for trial_idx in range(n_trials):
        t += rng.uniform(1.0, 3.0)
        difficulty = rng.randint(1, 5)

        # Trial start
        events.append(
            TaskEvent(
                session_id=sid,
                event_type=EventType.TRIAL_STARTED,
                monotonic_timestamp=t,
                trial_index=trial_idx,
                difficulty_level=difficulty,
                data_source=_DATA_SOURCE,
            )
        )

        # Stimulus
        t += rng.uniform(0.5, 1.0)
        events.append(
            TaskEvent(
                session_id=sid,
                event_type=EventType.STIMULUS_PRESENTED,
                monotonic_timestamp=t,
                trial_index=trial_idx,
                difficulty_level=difficulty,
                data_source=_DATA_SOURCE,
            )
        )

        # Modality samples during the trial
        face_quality = rng.uniform(0.3, 1.0)
        rppg_quality = rng.uniform(0.1, 0.9)
        overall_quality = (face_quality + rppg_quality) / 2.0

        samples.append(
            ModalitySample(
                session_id=sid,
                modality=Modality.FACE,
                monotonic_timestamp=t,
                processed_value={"blink_rate": rng.uniform(10, 25)},
            )
        )
        samples.append(
            ModalitySample(
                session_id=sid,
                modality=Modality.HEAD_POSE,
                monotonic_timestamp=t,
                processed_value={
                    "yaw": rng.uniform(-15, 15),
                    "pitch": rng.uniform(-10, 10),
                    "roll": rng.uniform(-5, 5),
                },
            )
        )
        samples.append(
            ModalitySample(
                session_id=sid,
                modality=Modality.RPPG,
                monotonic_timestamp=t,
                processed_value={"estimated_hr_bpm": rng.uniform(60, 100)},
            )
        )

        # Signal quality report
        modality_qualities = [
            ModalityQuality(
                modality=Modality.FACE,
                quality_score=round(face_quality, 3),
            ),
            ModalityQuality(
                modality=Modality.HEAD_POSE,
                quality_score=round(face_quality, 3),
            ),
            ModalityQuality(
                modality=Modality.RPPG,
                quality_score=round(rppg_quality, 3),
            ),
            ModalityQuality(
                modality=Modality.EDA,
                quality_score=0.0,
                available=False,
                reason="No EDA sensor connected.",
            ),
        ]
        quality_reports.append(
            SignalQualityReport(
                session_id=sid,
                monotonic_timestamp=t,
                modalities=modality_qualities,
                overall_quality=round(overall_quality, 3),
            )
        )

        # Response
        rt_ms = rng.uniform(200, 1500)
        correct = rng.random() > 0.3
        t += rt_ms / 1000.0
        events.append(
            TaskEvent(
                session_id=sid,
                event_type=EventType.RESPONSE_RECEIVED,
                monotonic_timestamp=t,
                trial_index=trial_idx,
                difficulty_level=difficulty,
                correct=correct,
                reaction_time_ms=round(rt_ms, 1),
                data_source=_DATA_SOURCE,
            )
        )

        # Engagement prediction
        confidence = rng.uniform(0.2, 0.95)
        should_abstain = confidence < 0.3 or overall_quality < 0.3

        predictions.append(
            EngagementPrediction(
                session_id=sid,
                monotonic_timestamp=t,
                engagement_estimate=(
                    None if should_abstain else round(rng.uniform(0.3, 0.9), 3)
                ),
                cognitive_load_estimate=(
                    None if should_abstain else round(rng.uniform(0.2, 0.8), 3)
                ),
                confidence=round(confidence, 3),
                signal_quality=round(overall_quality, 3),
                available_modalities=[
                    Modality.FACE,
                    Modality.HEAD_POSE,
                    Modality.RPPG,
                    Modality.TASK_PERFORMANCE,
                ],
                missing_modalities=[Modality.EDA, Modality.WEARABLE_ECG],
                abstain=should_abstain,
                reason=(
                    "Insufficient confidence or signal quality"
                    if should_abstain
                    else None
                ),
                data_source=_DATA_SOURCE,
            )
        )

        # Trial end
        t += rng.uniform(0.5, 1.0)
        events.append(
            TaskEvent(
                session_id=sid,
                event_type=EventType.TRIAL_ENDED,
                monotonic_timestamp=t,
                trial_index=trial_idx,
                difficulty_level=difficulty,
                data_source=_DATA_SOURCE,
            )
        )

    # Session end
    t += 1.0
    session.end_time = datetime(2026, 1, 1, 12, 30, 0, tzinfo=UTC)
    events.append(
        BaseEvent(
            session_id=sid,
            event_type=EventType.SESSION_ENDED,
            monotonic_timestamp=t,
            data_source=_DATA_SOURCE,
        )
    )

    return SyntheticSession(
        session=session,
        events=events,
        samples=samples,
        predictions=predictions,
        quality_reports=quality_reports,
    )
