"""Tests for Pydantic schemas -- validation and rejection."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from engagevr.schemas.adaptation import AdaptationCommand, AdaptationEvent
from engagevr.schemas.events import BaseEvent, EventType, TaskEvent
from engagevr.schemas.modality import Modality, ModalitySample
from engagevr.schemas.prediction import EngagementPrediction
from engagevr.schemas.session import DataSource, Session
from engagevr.schemas.signal_quality import ModalityQuality, SignalQualityReport

# --- Session ---


class TestSession:
    def test_valid_session(self):
        s = Session(participant_id="p001", data_source=DataSource.SYNTHETIC)
        assert s.participant_id == "p001"
        assert s.data_source == DataSource.SYNTHETIC
        assert s.session_id  # auto-generated

    def test_session_id_unique(self):
        s1 = Session(participant_id="p001")
        s2 = Session(participant_id="p002")
        assert s1.session_id != s2.session_id

    def test_missing_participant_id_rejected(self):
        with pytest.raises(ValidationError):
            Session.model_validate({})

    def test_invalid_data_source_rejected(self):
        with pytest.raises(ValidationError):
            Session(participant_id="p001", data_source="fabricated")  # type: ignore[arg-type]

    def test_invalid_condition_rejected(self):
        with pytest.raises(ValidationError):
            Session(
                participant_id="p001",
                experiment_condition="invalid",  # type: ignore[arg-type]
            )


# --- Modality ---


class TestModality:
    def test_all_modalities_exist(self):
        expected = {
            "face",
            "head_pose",
            "rppg",
            "heart_rate",
            "hrv",
            "task_performance",
            "subjective",
            "eda",
            "wearable_ecg",
            "wearable_ppg",
            "respiration",
        }
        assert {m.value for m in Modality} == expected

    def test_valid_sample(self):
        s = ModalitySample(
            session_id="s001",
            modality=Modality.FACE,
            monotonic_timestamp=1.5,
        )
        assert s.modality == Modality.FACE

    def test_invalid_modality_rejected(self):
        with pytest.raises(ValidationError):
            ModalitySample(
                session_id="s001",
                modality="nonexistent",  # type: ignore[arg-type]
                monotonic_timestamp=1.0,
            )


# --- Signal Quality ---


class TestSignalQuality:
    def test_valid_quality(self):
        q = ModalityQuality(modality=Modality.RPPG, quality_score=0.75)
        assert q.available is True

    def test_quality_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ModalityQuality(modality=Modality.RPPG, quality_score=1.5)

    def test_quality_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            ModalityQuality(modality=Modality.RPPG, quality_score=-0.1)

    def test_unavailable_modality(self):
        q = ModalityQuality(
            modality=Modality.EDA,
            quality_score=0.0,
            available=False,
            reason="No sensor connected.",
        )
        assert q.available is False
        assert q.reason is not None

    def test_quality_report(self):
        report = SignalQualityReport(
            session_id="s001",
            monotonic_timestamp=5.0,
            modalities=[
                ModalityQuality(modality=Modality.FACE, quality_score=0.9),
            ],
            overall_quality=0.9,
        )
        assert report.overall_quality == 0.9


# --- Prediction ---


class TestPrediction:
    def test_valid_prediction(self):
        p = EngagementPrediction(
            session_id="s001",
            monotonic_timestamp=10.0,
            engagement_estimate=0.7,
            cognitive_load_estimate=0.5,
            confidence=0.8,
            signal_quality=0.9,
            available_modalities=[Modality.FACE, Modality.RPPG],
            missing_modalities=[Modality.EDA],
            data_source="synthetic",
        )
        assert p.abstain is False
        assert p.engagement_estimate == 0.7

    def test_abstained_prediction(self):
        p = EngagementPrediction(
            session_id="s001",
            monotonic_timestamp=10.0,
            engagement_estimate=None,
            cognitive_load_estimate=None,
            confidence=0.2,
            signal_quality=0.1,
            abstain=True,
            reason="Insufficient signal quality",
            data_source="synthetic",
        )
        assert p.abstain is True
        assert p.engagement_estimate is None
        assert p.reason is not None

    def test_engagement_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            EngagementPrediction(
                session_id="s001",
                monotonic_timestamp=1.0,
                engagement_estimate=1.5,  # out of [0, 1]
                confidence=0.5,
                signal_quality=0.5,
            )

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            EngagementPrediction(
                session_id="s001",
                monotonic_timestamp=1.0,
                confidence=-0.1,
                signal_quality=0.5,
            )

    def test_signal_quality_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            EngagementPrediction(
                session_id="s001",
                monotonic_timestamp=1.0,
                confidence=0.5,
                signal_quality=2.0,
            )

    def test_data_source_defaults_to_synthetic(self):
        p = EngagementPrediction(
            session_id="s001",
            monotonic_timestamp=1.0,
            confidence=0.5,
            signal_quality=0.5,
        )
        assert p.data_source == "synthetic"


# --- Events ---


class TestEvents:
    def test_valid_base_event(self):
        e = BaseEvent(
            session_id="s001",
            event_type=EventType.SESSION_STARTED,
            monotonic_timestamp=0.0,
        )
        assert e.data_source == "synthetic"

    def test_valid_task_event(self):
        e = TaskEvent(
            session_id="s001",
            event_type=EventType.RESPONSE_RECEIVED,
            monotonic_timestamp=5.0,
            trial_index=0,
            correct=True,
            reaction_time_ms=450.0,
        )
        assert e.correct is True
        assert e.reaction_time_ms == 450.0

    def test_negative_reaction_time_rejected(self):
        with pytest.raises(ValidationError):
            TaskEvent(
                session_id="s001",
                event_type=EventType.RESPONSE_RECEIVED,
                monotonic_timestamp=5.0,
                reaction_time_ms=-10.0,
            )

    def test_invalid_event_type_rejected(self):
        with pytest.raises(ValidationError):
            BaseEvent(
                session_id="s001",
                event_type="invalid_type",  # type: ignore[arg-type]
                monotonic_timestamp=0.0,
            )

    def test_data_source_defaults_to_synthetic(self):
        e = BaseEvent(
            session_id="s001",
            event_type=EventType.SESSION_STARTED,
            monotonic_timestamp=0.0,
        )
        assert e.data_source == "synthetic"


# --- Adaptation ---


class TestAdaptation:
    def test_valid_command(self):
        cmd = AdaptationCommand(
            command="set_difficulty",
            value=3,
            reason="High engagement",
            confidence=0.8,
        )
        assert cmd.command == "set_difficulty"

    def test_command_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            AdaptationCommand(
                command="set_difficulty",
                value=1,
                reason="test",
                confidence=1.5,
            )

    def test_valid_adaptation_event(self):
        cmd = AdaptationCommand(
            command="set_difficulty",
            value=3,
            reason="High engagement and moderate load",
            confidence=0.8,
        )
        event = AdaptationEvent(
            session_id="s001",
            monotonic_timestamp=100.0,
            previous_state={"difficulty": 2},
            new_state={"difficulty": 3},
            engagement_estimate=0.8,
            cognitive_load_estimate=0.4,
            confidence=0.8,
            signal_quality=0.9,
            triggering_evidence="Sustained high engagement over 30s",
            rule_used="high_engagement_increase_difficulty",
            command=cmd,
        )
        assert event.is_manual is False
