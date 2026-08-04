"""Replay: determinism, pacing modes, filtering, provenance, immutability."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from engagevr.config import ReplayConfig
from engagevr.protocol.envelope import REPLAY_LABEL, SYNTHETIC_LABEL, MessageProvenance
from engagevr.protocol.messages import MessageSource, MessageType
from engagevr.replay import (
    REPLAY_DISCLAIMER,
    InvalidReplaySpeedError,
    ReplayFilter,
    ReplayMode,
    ReplayPacer,
    ReplayPlayer,
    mode_for_speed,
    parse_message_type_filter,
    parse_source_filter,
    read_recorded_session,
    validate_speed,
)
from engagevr.schemas.session import DataSource
from engagevr.storage import IngestionMetadata, SessionStore
from engagevr.synchronization.clock import ManualClock
from engagevr.transport import InProcessTransport
from tests.unit.test_protocol import make_envelope

BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def recording(tmp_path: Path) -> SessionStore:
    """A five-message recording with known one-second arrival gaps."""
    store = SessionStore(tmp_path)
    recorder = store.open_recorder("recorded")
    plan = [
        (MessageType.CLIENT_HELLO, MessageSource.PYTHON_SIMULATOR),
        (MessageType.SESSION_START, MessageSource.PYTHON_SIMULATOR),
        (MessageType.TASK_EVENT, MessageSource.PYTHON_SIMULATOR),
        (MessageType.ACKNOWLEDGEMENT, MessageSource.BACKEND),
        (MessageType.SESSION_END, MessageSource.PYTHON_SIMULATOR),
    ]
    for index, (message_type, source) in enumerate(plan):
        provenance = (
            MessageProvenance(
                data_source=DataSource.LIVE, synthetic_label=None, producer="backend"
            )
            if source is MessageSource.BACKEND
            else None
        )
        envelope = make_envelope(
            message_type=message_type,
            source=source,
            sequence_number=index,
        )
        if provenance is not None:
            envelope = envelope.model_copy(update={"provenance": provenance})
        recorder.append(
            envelope,
            IngestionMetadata(
                arrival_index=index,
                server_received_at_utc=BASE + timedelta(seconds=index),
                server_monotonic_seconds=float(index),
                transport="websocket",
            ),
        )
    recorder.close()
    return store


async def no_sleep(_seconds: float) -> None:
    return None


async def play(
    store: SessionStore,
    *,
    speed: float = 0.0,
    replay_filter: ReplayFilter | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    replay_session_id: str = "replay-target",
    step_mode: bool = False,
) -> tuple[object, InProcessTransport, ReplayPlayer]:
    recording = read_recorded_session(store, "recorded")
    transport = InProcessTransport()
    player = ReplayPlayer(
        recording,
        transport=transport,
        replay_session_id=replay_session_id,
        speed=speed,
        replay_filter=replay_filter,
        clock=ManualClock(start_utc=BASE),
        sleep=sleep if sleep is not None else no_sleep,
        collect_envelopes=True,
        step_mode=step_mode,
    )
    if step_mode:
        return None, transport, player
    result = await player.run()
    return result, transport, player


# --- speed validation ------------------------------------------------------


class TestSpeedValidation:
    @pytest.mark.parametrize("speed", [0.0, 0.5, 1.0, 10.0, 1000.0])
    def test_valid_speeds_are_accepted(self, speed: float) -> None:
        assert validate_speed(speed, maximum_speed=1000.0) == speed

    def test_negative_speed_is_rejected(self) -> None:
        with pytest.raises(InvalidReplaySpeedError, match="not be negative"):
            validate_speed(-1.0, maximum_speed=1000.0)

    def test_nan_is_rejected(self) -> None:
        with pytest.raises(InvalidReplaySpeedError, match="NaN"):
            validate_speed(float("nan"), maximum_speed=1000.0)

    def test_infinity_is_rejected(self) -> None:
        with pytest.raises(InvalidReplaySpeedError, match="finite"):
            validate_speed(float("inf"), maximum_speed=1000.0)

    def test_speed_above_the_maximum_is_rejected(self) -> None:
        with pytest.raises(InvalidReplaySpeedError, match="exceeds"):
            validate_speed(5000.0, maximum_speed=1000.0)

    def test_mode_selection(self) -> None:
        assert mode_for_speed(0.0) is ReplayMode.IMMEDIATE
        assert mode_for_speed(1.0) is ReplayMode.ORIGINAL
        assert mode_for_speed(7.5) is ReplayMode.ACCELERATED

    def test_config_rejects_a_default_above_the_maximum(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="maximum_speed"):
            ReplayConfig(default_speed=10.0, maximum_speed=2.0)


class TestPacer:
    def test_immediate_never_waits(self) -> None:
        pacer = ReplayPacer(speed=0.0, sleep=no_sleep)
        assert pacer.delay_for(5.0) == 0.0

    def test_original_timing_reproduces_the_gap(self) -> None:
        pacer = ReplayPacer(speed=1.0, sleep=no_sleep)
        assert pacer.delay_for(2.5) == 2.5

    def test_accelerated_divides_the_gap(self) -> None:
        pacer = ReplayPacer(speed=5.0, sleep=no_sleep)
        assert pacer.delay_for(2.5) == 0.5

    def test_a_negative_gap_is_clamped_not_rewound(self) -> None:
        pacer = ReplayPacer(speed=1.0, sleep=no_sleep)
        assert pacer.delay_for(-3.0) == 0.0

    @pytest.mark.asyncio
    async def test_requested_delays_are_recorded(self) -> None:
        pacer = ReplayPacer(speed=2.0, sleep=no_sleep, requested_delays=[])
        await pacer.wait(4.0)
        await pacer.wait(0.0)
        assert pacer.requested_delays == [2.0, 0.0]


# --- replay behaviour ------------------------------------------------------


class TestReplayOrdering:
    @pytest.mark.asyncio
    async def test_all_messages_are_replayed_in_recorded_order(
        self, recording: SessionStore
    ) -> None:
        result, transport, _player = await play(recording)
        assert result.emitted_message_count == 5  # type: ignore[attr-defined]
        assert [e.sequence_number for e in transport.sent] == [0, 1, 2, 3, 4]
        assert [e.message_type.value for e in transport.sent] == [
            "client_hello",
            "session_start",
            "task_event",
            "acknowledgement",
            "session_end",
        ]

    @pytest.mark.asyncio
    async def test_replay_is_deterministic(self, recording: SessionStore) -> None:
        _first_result, first, _p1 = await play(recording)
        _second_result, second, _p2 = await play(recording)

        assert [e.model_dump(mode="json") for e in first.sent] == [
            e.model_dump(mode="json") for e in second.sent
        ]

    @pytest.mark.asyncio
    async def test_immediate_mode_does_not_sleep(self, recording: SessionStore) -> None:
        slept: list[float] = []

        async def record(seconds: float) -> None:
            slept.append(seconds)

        await play(recording, speed=0.0, sleep=record)
        assert slept == []

    @pytest.mark.asyncio
    async def test_original_timing_requests_the_recorded_gaps(
        self, recording: SessionStore
    ) -> None:
        slept: list[float] = []

        async def record(seconds: float) -> None:
            slept.append(seconds)

        await play(recording, speed=1.0, sleep=record)
        # First message has no preceding gap; the rest are one second apart.
        assert slept == [1.0, 1.0, 1.0, 1.0]

    @pytest.mark.asyncio
    async def test_accelerated_timing_divides_the_gaps(
        self, recording: SessionStore
    ) -> None:
        slept: list[float] = []

        async def record(seconds: float) -> None:
            slept.append(seconds)

        await play(recording, speed=4.0, sleep=record)
        assert slept == [0.25, 0.25, 0.25, 0.25]

    @pytest.mark.asyncio
    async def test_step_mode_releases_one_message_at_a_time(
        self, recording: SessionStore
    ) -> None:
        _result, transport, player = await play(recording, step_mode=True)
        task = asyncio.ensure_future(player.run())
        try:
            for expected in range(1, 6):
                player.step()
                for _ in range(50):
                    await asyncio.sleep(0)
                    if len(transport.sent) >= expected:
                        break
                assert len(transport.sent) == expected
            result = await asyncio.wait_for(task, timeout=1.0)
            assert result.emitted_message_count == 5
        finally:
            if not task.done():
                task.cancel()


class TestReplayProvenance:
    @pytest.mark.asyncio
    async def test_every_replayed_message_is_labelled_replay(
        self, recording: SessionStore
    ) -> None:
        _result, transport, _player = await play(recording)
        for index, envelope in enumerate(transport.sent):
            assert envelope.replay is not None
            assert envelope.replay.replay_label == REPLAY_LABEL
            assert envelope.replay.source_session_id == "recorded"
            assert envelope.replay.replay_session_id == "replay-target"
            assert envelope.replay.replay_index == index
            assert envelope.replay.original_arrival_index == index

    @pytest.mark.asyncio
    async def test_a_synthetic_message_stays_synthetic_and_becomes_replay(
        self, recording: SessionStore
    ) -> None:
        _result, transport, _player = await play(recording)
        synthetic = [
            e for e in transport.sent if e.source is MessageSource.PYTHON_SIMULATOR
        ]
        assert synthetic
        for envelope in synthetic:
            assert envelope.provenance.synthetic_label == SYNTHETIC_LABEL
            assert envelope.provenance.data_source is DataSource.SYNTHETIC
            assert envelope.replay is not None
            assert envelope.replay.replay_label == REPLAY_LABEL

    @pytest.mark.asyncio
    async def test_original_fields_are_not_rewritten(
        self, recording: SessionStore
    ) -> None:
        source = read_recorded_session(recording, "recorded")
        _result, transport, _player = await play(recording)

        for original, replayed in zip(source.messages, transport.sent, strict=True):
            assert replayed.message_id == original.envelope.message_id
            assert replayed.source is original.envelope.source
            assert replayed.session_id == original.envelope.session_id
            assert replayed.sequence_number == original.envelope.sequence_number
            assert replayed.sent_at_utc == original.envelope.sent_at_utc
            assert (
                replayed.sent_at_monotonic_seconds
                == original.envelope.sent_at_monotonic_seconds
            )
            assert replayed.provenance == original.envelope.provenance

    @pytest.mark.asyncio
    async def test_a_live_recorded_message_does_not_gain_a_synthetic_label(
        self, recording: SessionStore
    ) -> None:
        _result, transport, _player = await play(recording)
        backend = [e for e in transport.sent if e.source is MessageSource.BACKEND]
        assert backend
        for envelope in backend:
            assert envelope.provenance.synthetic_label is None
            assert envelope.replay is not None

    @pytest.mark.asyncio
    async def test_the_result_carries_the_replay_disclaimer(
        self, recording: SessionStore
    ) -> None:
        result, _transport, _player = await play(recording)
        assert result.replay_label == REPLAY_LABEL  # type: ignore[attr-defined]
        assert result.disclaimer == REPLAY_DISCLAIMER  # type: ignore[attr-defined]
        assert "not live" in REPLAY_DISCLAIMER
        assert "not a new session" in REPLAY_DISCLAIMER


class TestSourceImmutability:
    @pytest.mark.asyncio
    async def test_the_recording_is_byte_identical_after_replay(
        self, recording: SessionStore, tmp_path: Path
    ) -> None:
        directory = tmp_path / "recorded"
        before = {
            path.name: path.read_bytes()
            for path in sorted(directory.iterdir())
            if path.is_file()
        }

        await play(recording)
        await play(recording, speed=2.0)

        after = {
            path.name: path.read_bytes()
            for path in sorted(directory.iterdir())
            if path.is_file()
        }
        assert after == before

    @pytest.mark.asyncio
    async def test_replaying_does_not_add_files_to_the_source(
        self, recording: SessionStore, tmp_path: Path
    ) -> None:
        before = sorted(p.name for p in (tmp_path / "recorded").iterdir())
        await play(recording)
        assert sorted(p.name for p in (tmp_path / "recorded").iterdir()) == before


class TestFiltering:
    @pytest.mark.asyncio
    async def test_filtering_by_message_type(self, recording: SessionStore) -> None:
        result, transport, _player = await play(
            recording,
            replay_filter=ReplayFilter(
                message_types=frozenset({MessageType.TASK_EVENT})
            ),
        )
        assert result.emitted_message_count == 1  # type: ignore[attr-defined]
        assert result.skipped_by_filter_count == 4  # type: ignore[attr-defined]
        assert transport.sent[0].message_type is MessageType.TASK_EVENT

    @pytest.mark.asyncio
    async def test_filtering_by_source(self, recording: SessionStore) -> None:
        result, transport, _player = await play(
            recording,
            replay_filter=ReplayFilter(sources=frozenset({MessageSource.BACKEND})),
        )
        assert result.emitted_message_count == 1  # type: ignore[attr-defined]
        assert transport.sent[0].source is MessageSource.BACKEND

    @pytest.mark.asyncio
    async def test_filters_combine_conjunctively(self, recording: SessionStore) -> None:
        result, _transport, _player = await play(
            recording,
            replay_filter=ReplayFilter(
                message_types=frozenset({MessageType.TASK_EVENT}),
                sources=frozenset({MessageSource.BACKEND}),
            ),
        )
        assert result.emitted_message_count == 0  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_filtering_does_not_renumber_the_survivors(
        self, recording: SessionStore
    ) -> None:
        _result, transport, _player = await play(
            recording,
            replay_filter=ReplayFilter(
                message_types=frozenset(
                    {MessageType.SESSION_START, MessageType.SESSION_END}
                )
            ),
        )
        # The original sequence numbers are kept, so the gap the filter
        # created is visible rather than hidden.
        assert [e.sequence_number for e in transport.sent] == [1, 4]

    def test_an_empty_filter_describes_itself_as_such(self) -> None:
        assert ReplayFilter().describe() == "no filter (all messages)"
        assert ReplayFilter().is_empty is True

    def test_a_filter_describes_its_selection(self) -> None:
        description = ReplayFilter(
            message_types=frozenset({MessageType.TASK_EVENT}),
            sources=frozenset({MessageSource.BACKEND}),
        ).describe()
        assert "task_event" in description
        assert "backend" in description

    def test_unknown_filter_values_are_rejected_with_the_valid_list(self) -> None:
        with pytest.raises(ValueError, match="valid types"):
            parse_message_type_filter(["not_a_type"])
        with pytest.raises(ValueError, match="valid sources"):
            parse_source_filter(["not_a_source"])

    def test_empty_filter_arguments_produce_an_empty_set(self) -> None:
        assert parse_message_type_filter(None) == frozenset()
        assert parse_source_filter([]) == frozenset()


class TestReader:
    def test_reading_yields_the_manifest_summary_and_messages(
        self, recording: SessionStore
    ) -> None:
        session = read_recorded_session(recording, "recorded")
        assert session.session_id == "recorded"
        assert session.message_count == 5
        assert session.summary.event_count == 5
        assert session.manifest.session_id == "recorded"

    def test_gaps_are_computed_on_the_receiver_clock(
        self, recording: SessionStore
    ) -> None:
        session = read_recorded_session(recording, "recorded")
        assert session.gaps_seconds(session.messages) == (0.0, 1.0, 1.0, 1.0, 1.0)

    def test_an_interrupted_recording_is_recovered_for_replay(
        self, tmp_path: Path
    ) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("partial")
        recorder.append(
            make_envelope(),
            IngestionMetadata(
                arrival_index=0,
                server_received_at_utc=BASE,
                server_monotonic_seconds=0.0,
                transport="websocket",
            ),
        )
        # No close(): summary.json is absent.
        session = read_recorded_session(store, "partial")
        assert session.summary.recovered is True
        assert session.summary.completed is False
        assert session.message_count == 1

    def test_recovery_can_be_refused(self, tmp_path: Path) -> None:
        from engagevr.storage import SessionStoreError

        store = SessionStore(tmp_path)
        store.open_recorder("partial")
        with pytest.raises(SessionStoreError, match=r"no summary\.json"):
            read_recorded_session(store, "partial", recover_incomplete=False)

    def test_a_malformed_line_is_reported_with_its_number(
        self, recording: SessionStore, tmp_path: Path
    ) -> None:
        from engagevr.storage import JsonlFormatError

        path = tmp_path / "recorded" / "events.jsonl"
        lines = path.read_text().splitlines()
        lines.insert(2, "this is not json")
        path.write_text("\n".join(lines) + "\n")

        with pytest.raises(JsonlFormatError) as info:
            read_recorded_session(recording, "recorded")
        assert info.value.line_number == 3
