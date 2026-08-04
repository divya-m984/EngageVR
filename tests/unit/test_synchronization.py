"""Clock diagnostics and sequence-ordering tests.

No real time passes in any of these tests: the clock is injected.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from engagevr.protocol.messages import MessageSource
from engagevr.synchronization import (
    ManualClock,
    OrderingAnomaly,
    SequenceTracker,
    SystemClock,
    assess_arrival,
    estimate_round_trip,
)
from tests.unit.test_protocol import make_envelope

BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


class TestManualClock:
    def test_advances_both_readings_together(self) -> None:
        clock = ManualClock(start_utc=BASE, start_monotonic=10.0)
        clock.advance(2.5)
        assert clock.monotonic() == 12.5
        assert clock.utc_now() == BASE + timedelta(seconds=2.5)

    def test_cannot_move_backwards(self) -> None:
        clock = ManualClock()
        with pytest.raises(ValueError, match="backwards"):
            clock.advance(-1.0)

    def test_system_clock_returns_aware_utc(self) -> None:
        assert SystemClock().utc_now().tzinfo is not None


class TestRoundTrip:
    def test_symmetric_case_gives_zero_offset(self) -> None:
        estimate = estimate_round_trip(
            heartbeat_id="h",
            client_sent_monotonic=0.0,
            client_received_monotonic=0.100,
            client_sent_utc=BASE,
            client_received_utc=BASE + timedelta(seconds=0.100),
            server_received_utc=BASE + timedelta(seconds=0.040),
            server_sent_utc=BASE + timedelta(seconds=0.060),
        )
        assert estimate.round_trip_seconds == pytest.approx(0.080)
        assert estimate.server_handling_seconds == pytest.approx(0.020)
        assert estimate.clock_offset_estimate_seconds == pytest.approx(0.0, abs=1e-9)
        assert estimate.offset_uncertainty_seconds == pytest.approx(0.040)

    def test_offset_is_recovered_with_its_uncertainty(self) -> None:
        offset = timedelta(seconds=3.0)
        estimate = estimate_round_trip(
            heartbeat_id="h",
            client_sent_monotonic=0.0,
            client_received_monotonic=0.100,
            client_sent_utc=BASE,
            client_received_utc=BASE + timedelta(seconds=0.100),
            server_received_utc=BASE + offset + timedelta(seconds=0.040),
            server_sent_utc=BASE + offset + timedelta(seconds=0.060),
        )
        assert estimate.clock_offset_estimate_seconds == pytest.approx(3.0)
        assert estimate.offset_uncertainty_seconds > 0.0
        assert estimate.symmetric_delay_assumed is True
        assert "ESTIMATE" in estimate.note

    def test_backwards_client_clock_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="backwards"):
            estimate_round_trip(
                heartbeat_id="h",
                client_sent_monotonic=1.0,
                client_received_monotonic=0.5,
                client_sent_utc=BASE,
                client_received_utc=BASE,
                server_received_utc=BASE,
                server_sent_utc=BASE,
            )

    def test_impossible_handling_time_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="exceeds the measured round trip"):
            estimate_round_trip(
                heartbeat_id="h",
                client_sent_monotonic=0.0,
                client_received_monotonic=0.010,
                client_sent_utc=BASE,
                client_received_utc=BASE + timedelta(seconds=0.010),
                server_received_utc=BASE,
                server_sent_utc=BASE + timedelta(seconds=5.0),
            )


class TestArrivalAssessment:
    def test_cross_process_delay_is_unavailable_with_a_reason(self) -> None:
        timing = assess_arrival(
            sent_at_utc=BASE,
            server_received_at_utc=BASE + timedelta(seconds=0.05),
            server_monotonic_seconds=1.0,
            same_process=False,
            maximum_clock_skew_seconds=5.0,
            maximum_transport_delay_seconds=2.0,
        )
        assert timing.apparent_transport_delay_seconds is None
        assert timing.delay_unavailable_reason is not None
        assert "clock offset" in timing.delay_unavailable_reason

    def test_in_process_delay_is_computed(self) -> None:
        timing = assess_arrival(
            sent_at_utc=BASE,
            server_received_at_utc=BASE + timedelta(seconds=0.05),
            server_monotonic_seconds=1.0,
            same_process=True,
            maximum_clock_skew_seconds=5.0,
            maximum_transport_delay_seconds=2.0,
        )
        assert timing.apparent_transport_delay_seconds == pytest.approx(0.05)
        assert timing.excessive_delay is False

    def test_excessive_in_process_delay_is_flagged(self) -> None:
        timing = assess_arrival(
            sent_at_utc=BASE,
            server_received_at_utc=BASE + timedelta(seconds=10.0),
            server_monotonic_seconds=1.0,
            same_process=True,
            maximum_clock_skew_seconds=5.0,
            maximum_transport_delay_seconds=2.0,
        )
        assert timing.excessive_delay is True

    def test_future_timestamp_beyond_tolerance_is_flagged(self) -> None:
        timing = assess_arrival(
            sent_at_utc=BASE + timedelta(seconds=30.0),
            server_received_at_utc=BASE,
            server_monotonic_seconds=1.0,
            same_process=False,
            maximum_clock_skew_seconds=5.0,
            maximum_transport_delay_seconds=2.0,
        )
        assert timing.future_timestamp is True
        assert timing.future_by_seconds == pytest.approx(30.0)

    def test_small_skew_is_within_tolerance(self) -> None:
        timing = assess_arrival(
            sent_at_utc=BASE + timedelta(seconds=1.0),
            server_received_at_utc=BASE,
            server_monotonic_seconds=1.0,
            same_process=False,
            maximum_clock_skew_seconds=5.0,
            maximum_transport_delay_seconds=2.0,
        )
        assert timing.future_timestamp is False
        assert timing.future_by_seconds is None


class TestSequenceTracker:
    def test_ordered_stream_has_no_anomalies(self) -> None:
        tracker = SequenceTracker()
        for n in range(5):
            observation = tracker.observe(make_envelope(sequence_number=n))
            assert observation.ok, observation.detail
        assert tracker.message_count(MessageSource.PYTHON_SIMULATOR) == 5

    def test_duplicate_message_id_is_detected(self) -> None:
        tracker = SequenceTracker()
        first = make_envelope(sequence_number=0)
        tracker.observe(first)
        repeat = first.model_copy(update={"sequence_number": 1})
        observation = tracker.observe(repeat)
        assert OrderingAnomaly.DUPLICATE_MESSAGE_ID in observation.anomalies

    def test_duplicate_sequence_number_is_detected(self) -> None:
        tracker = SequenceTracker()
        tracker.observe(make_envelope(sequence_number=0))
        observation = tracker.observe(make_envelope(sequence_number=0))
        assert OrderingAnomaly.DUPLICATE_SEQUENCE_NUMBER in observation.anomalies

    def test_sequence_reversal_is_detected(self) -> None:
        tracker = SequenceTracker()
        tracker.observe(make_envelope(sequence_number=0))
        tracker.observe(make_envelope(sequence_number=5))
        observation = tracker.observe(make_envelope(sequence_number=2))
        assert OrderingAnomaly.SEQUENCE_REVERSAL in observation.anomalies

    def test_missing_range_is_enumerated(self) -> None:
        tracker = SequenceTracker()
        tracker.observe(make_envelope(sequence_number=0))
        observation = tracker.observe(make_envelope(sequence_number=4))
        assert OrderingAnomaly.MISSING_SEQUENCE_RANGE in observation.anomalies
        assert observation.missing_sequence_numbers == (1, 2, 3)
        assert observation.expected_sequence_number == 1

    def test_excessive_gap_is_flagged_and_not_enumerated(self) -> None:
        tracker = SequenceTracker(maximum_sequence_gap=3)
        tracker.observe(make_envelope(sequence_number=0))
        observation = tracker.observe(make_envelope(sequence_number=1000))
        assert OrderingAnomaly.EXCESSIVE_SEQUENCE_GAP in observation.anomalies
        assert observation.missing_sequence_numbers == ()

    def test_sources_are_tracked_independently(self) -> None:
        tracker = SequenceTracker()
        tracker.observe(
            make_envelope(sequence_number=0, source=MessageSource.PYTHON_SIMULATOR)
        )
        tracker.observe(make_envelope(sequence_number=0, source=MessageSource.BACKEND))
        # Same sequence number, different source: not a duplicate.
        assert tracker.message_count(MessageSource.PYTHON_SIMULATOR) == 1
        assert tracker.message_count(MessageSource.BACKEND) == 1
        assert set(tracker.sources()) == {
            MessageSource.PYTHON_SIMULATOR,
            MessageSource.BACKEND,
        }

    def test_message_id_history_is_bounded(self) -> None:
        tracker = SequenceTracker(message_id_history=4)
        assert tracker.message_id_history == 4
        for n in range(10):
            tracker.observe(make_envelope(sequence_number=n))
        # Memory is bounded; the tracker still counts every message.
        assert tracker.message_count(MessageSource.PYTHON_SIMULATOR) == 10

    def test_invalid_configuration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SequenceTracker(maximum_sequence_gap=-1)
        with pytest.raises(ValueError, match="at least 1"):
            SequenceTracker(message_id_history=0)
