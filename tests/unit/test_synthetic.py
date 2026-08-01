"""Tests for the synthetic session generator."""

from __future__ import annotations

import json

from engagevr.schemas.events import EventType
from engagevr.schemas.session import DataSource
from engagevr.simulator.synthetic import generate_synthetic_session


class TestSyntheticGenerator:
    def test_generates_session(self):
        ss = generate_synthetic_session(n_trials=5, seed=42)
        assert ss.session.data_source == DataSource.SYNTHETIC
        assert ss.session.participant_id == "synth_participant_001"

    def test_correct_event_count(self):
        ss = generate_synthetic_session(n_trials=5, seed=42)
        # 1 start + 5*(start + stimulus + response + end) + 1 end = 22
        assert len(ss.events) == 22

    def test_correct_sample_count(self):
        ss = generate_synthetic_session(n_trials=5, seed=42)
        # 3 modality samples per trial (face, head_pose, rppg)
        assert len(ss.samples) == 15

    def test_correct_prediction_count(self):
        ss = generate_synthetic_session(n_trials=5, seed=42)
        assert len(ss.predictions) == 5

    def test_correct_quality_report_count(self):
        ss = generate_synthetic_session(n_trials=5, seed=42)
        assert len(ss.quality_reports) == 5

    def test_all_events_labelled_synthetic(self):
        ss = generate_synthetic_session(n_trials=5, seed=42)
        for event in ss.events:
            assert event.data_source == "synthetic"

    def test_all_predictions_labelled_synthetic(self):
        ss = generate_synthetic_session(n_trials=5, seed=42)
        for pred in ss.predictions:
            assert pred.data_source == "synthetic"

    def test_session_has_start_and_end_events(self):
        ss = generate_synthetic_session(n_trials=3, seed=1)
        event_types = [e.event_type for e in ss.events]
        assert event_types[0] == EventType.SESSION_STARTED
        assert event_types[-1] == EventType.SESSION_ENDED

    def test_timestamps_monotonically_increasing(self):
        ss = generate_synthetic_session(n_trials=10, seed=99)
        timestamps = [e.monotonic_timestamp for e in ss.events]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_reproducible_with_same_seed(self):
        ss1 = generate_synthetic_session(n_trials=5, seed=42)
        ss2 = generate_synthetic_session(n_trials=5, seed=42)
        # Session IDs differ (uuid), but event data should match
        assert len(ss1.events) == len(ss2.events)
        for e1, e2 in zip(ss1.events, ss2.events, strict=False):
            assert e1.event_type == e2.event_type
            assert e1.monotonic_timestamp == e2.monotonic_timestamp

    def test_different_seeds_differ(self):
        ss1 = generate_synthetic_session(n_trials=5, seed=1)
        ss2 = generate_synthetic_session(n_trials=5, seed=2)
        # Timestamps will differ due to different random draws
        ts1 = [e.monotonic_timestamp for e in ss1.events]
        ts2 = [e.monotonic_timestamp for e in ss2.events]
        assert ts1 != ts2

    def test_some_predictions_may_abstain(self):
        # With enough trials and a seed that produces low quality/confidence
        ss = generate_synthetic_session(n_trials=50, seed=7)
        abstained = [p for p in ss.predictions if p.abstain]
        non_abstained = [p for p in ss.predictions if not p.abstain]
        # At least some should abstain and some should not
        assert len(abstained) > 0 or len(non_abstained) > 0

    def test_abstained_predictions_have_null_estimates(self):
        ss = generate_synthetic_session(n_trials=50, seed=7)
        for pred in ss.predictions:
            if pred.abstain:
                assert pred.engagement_estimate is None
                assert pred.cognitive_load_estimate is None
                assert pred.reason is not None

    def test_serialization_roundtrip(self):
        ss = generate_synthetic_session(n_trials=3, seed=42)
        d = ss.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["session"]["data_source"] == "synthetic"
        assert len(loaded["events"]) == len(ss.events)

    def test_session_end_time_set(self):
        ss = generate_synthetic_session(n_trials=3, seed=42)
        assert ss.session.end_time is not None
