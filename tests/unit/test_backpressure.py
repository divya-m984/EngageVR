"""Bounded-queue and backpressure-policy tests.

The policy under test:

- every queue is bounded, so memory cannot grow without limit;
- a critical message waits for space and, if space never comes, fails
  the connection rather than being dropped;
- a non-critical message is dropped immediately rather than blocking;
- every drop is counted, warned about, and written to ``dropped.jsonl``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from engagevr.api.broker import BrokerSettings, QueueFullError, SessionBroker
from engagevr.protocol.messages import DisconnectReason, MessageType, TaskEventPayload
from engagevr.protocol.validation import DecodedMessage, decode_payload
from engagevr.schemas.events import EventType, TaskEventDetail
from engagevr.storage import SessionStore
from engagevr.synchronization.clock import ManualClock
from tests.unit.test_protocol import make_envelope


def decoded(
    message_type: MessageType = MessageType.TASK_EVENT,
    *,
    sequence_number: int = 0,
    payload: object | None = None,
) -> DecodedMessage:
    envelope = make_envelope(
        message_type=message_type, sequence_number=sequence_number, payload=payload
    )
    return DecodedMessage(envelope=envelope, payload=decode_payload(envelope))


class _StalledBroker(SessionBroker):
    """A broker whose storage worker never drains, so the queue fills up."""

    async def _storage_worker(self) -> None:  # type: ignore[override]
        await asyncio.Event().wait()


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path)


class TestQueueBounds:
    @pytest.mark.asyncio
    async def test_capacity_is_configurable_and_enforced(
        self, store: SessionStore
    ) -> None:
        recorder = store.open_recorder("demo")
        broker = _StalledBroker(
            session_id="demo",
            recorder=recorder,
            settings=BrokerSettings(storage_capacity=3, operation_timeout_seconds=0.05),
            clock=ManualClock(),
        )
        await broker.start()
        try:
            for n in range(3):
                result = await broker.ingest(decoded(sequence_number=n))
                assert result.stored is True
            assert broker.storage_queue_depth == 3
        finally:
            await broker.stop()

    @pytest.mark.asyncio
    async def test_a_full_queue_drops_non_critical_messages(
        self, store: SessionStore
    ) -> None:
        recorder = store.open_recorder("demo")
        broker = _StalledBroker(
            session_id="demo",
            recorder=recorder,
            settings=BrokerSettings(storage_capacity=2, operation_timeout_seconds=0.05),
            clock=ManualClock(),
        )
        await broker.start()
        try:
            await broker.ingest(decoded(sequence_number=0))
            await broker.ingest(decoded(sequence_number=1))
            overflow = await broker.ingest(decoded(sequence_number=2))

            assert overflow.stored is False
            assert overflow.dropped is True
            assert overflow.acknowledgement.stored is False
            assert overflow.acknowledgement.dropped is True
            assert broker.dropped_count == 1
        finally:
            await broker.stop()

    @pytest.mark.asyncio
    async def test_the_process_is_not_blocked_by_a_full_queue(
        self, store: SessionStore
    ) -> None:
        """Non-critical overflow returns promptly instead of waiting."""
        recorder = store.open_recorder("demo")
        broker = _StalledBroker(
            session_id="demo",
            recorder=recorder,
            settings=BrokerSettings(storage_capacity=1, operation_timeout_seconds=5.0),
            clock=ManualClock(),
        )
        await broker.start()
        try:
            await broker.ingest(decoded(sequence_number=0))
            # A five-second operation timeout must not be spent here: a
            # non-critical message is dropped immediately.
            await asyncio.wait_for(
                broker.ingest(decoded(sequence_number=1)), timeout=1.0
            )
        finally:
            await broker.stop()


class TestCriticalMessagePreservation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "message_type",
        [
            MessageType.SESSION_START,
            MessageType.SESSION_END,
            MessageType.ADAPTATION_COMMAND,
            MessageType.ADAPTATION_ACKNOWLEDGEMENT,
            MessageType.PROTOCOL_ERROR,
        ],
    )
    async def test_a_critical_message_is_never_silently_dropped(
        self, store: SessionStore, message_type: MessageType
    ) -> None:
        recorder = store.open_recorder("demo")
        broker = _StalledBroker(
            session_id="demo",
            recorder=recorder,
            settings=BrokerSettings(storage_capacity=1, operation_timeout_seconds=0.05),
            clock=ManualClock(),
        )
        await broker.start()
        try:
            await broker.ingest(decoded(sequence_number=0))
            with pytest.raises(QueueFullError) as info:
                await broker.ingest(decoded(message_type, sequence_number=1))
            assert "refusing to drop it" in str(info.value)
            assert broker.dropped_count == 0, "a critical message is never dropped"
        finally:
            await broker.stop()

    @pytest.mark.asyncio
    async def test_task_completed_is_treated_as_critical(
        self, store: SessionStore
    ) -> None:
        recorder = store.open_recorder("demo")
        broker = _StalledBroker(
            session_id="demo",
            recorder=recorder,
            settings=BrokerSettings(storage_capacity=1, operation_timeout_seconds=0.05),
            clock=ManualClock(),
        )
        await broker.start()
        try:
            await broker.ingest(decoded(sequence_number=0))
            completion = decoded(
                MessageType.TASK_EVENT,
                sequence_number=1,
                payload=TaskEventPayload(
                    event=TaskEventDetail(event_type=EventType.TASK_COMPLETED)
                ),
            )
            with pytest.raises(QueueFullError):
                await broker.ingest(completion)
            assert broker.dropped_count == 0
        finally:
            await broker.stop()

    @pytest.mark.asyncio
    async def test_an_ordinary_task_event_is_droppable(
        self, store: SessionStore
    ) -> None:
        recorder = store.open_recorder("demo")
        broker = _StalledBroker(
            session_id="demo",
            recorder=recorder,
            settings=BrokerSettings(storage_capacity=1, operation_timeout_seconds=0.05),
            clock=ManualClock(),
        )
        await broker.start()
        try:
            await broker.ingest(decoded(sequence_number=0))
            result = await broker.ingest(decoded(sequence_number=1))
            assert result.dropped is True
        finally:
            await broker.stop()


class TestDropVisibility:
    @pytest.mark.asyncio
    async def test_a_drop_is_counted_in_the_summary(self, store: SessionStore) -> None:
        recorder = store.open_recorder("demo")
        broker = _StalledBroker(
            session_id="demo",
            recorder=recorder,
            settings=BrokerSettings(storage_capacity=1, operation_timeout_seconds=0.05),
            clock=ManualClock(),
        )
        await broker.start()
        await broker.ingest(decoded(sequence_number=0))
        await broker.ingest(decoded(sequence_number=1))
        await broker.stop()

        summary = store.read_summary("demo")
        assert summary is not None
        assert summary.dropped_message_count == 1
        assert summary.dropped_message_types["task_event"] == 1

    @pytest.mark.asyncio
    async def test_a_drop_is_written_to_the_drop_log_with_a_reason(
        self, store: SessionStore, tmp_path: Path
    ) -> None:
        recorder = store.open_recorder("demo")
        broker = _StalledBroker(
            session_id="demo",
            recorder=recorder,
            settings=BrokerSettings(storage_capacity=1, operation_timeout_seconds=0.05),
            clock=ManualClock(),
        )
        await broker.start()
        await broker.ingest(decoded(sequence_number=0))
        await broker.ingest(decoded(sequence_number=1))
        await broker.stop()

        text = (tmp_path / "demo" / "dropped.jsonl").read_text()
        assert "storage queue was full" in text
        assert "task_event" in text
        assert '"queue":"storage"' in text

    @pytest.mark.asyncio
    async def test_a_drop_emits_a_warning_in_the_session_log(
        self, store: SessionStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        recorder = store.open_recorder("demo")
        broker = _StalledBroker(
            session_id="demo",
            recorder=recorder,
            settings=BrokerSettings(storage_capacity=1, operation_timeout_seconds=0.05),
            clock=ManualClock(),
        )
        await broker.start()
        with caplog.at_level(logging.WARNING, logger="engagevr.api.broker"):
            await broker.ingest(decoded(sequence_number=0))
            await broker.ingest(decoded(sequence_number=1))
        await broker.stop()

        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("dropped non-critical message" in message for message in warnings)

    @pytest.mark.asyncio
    async def test_a_full_broadcast_queue_warns_but_does_not_drop_the_event(
        self, store: SessionStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        recorder = store.open_recorder("demo")

        class _NoBroadcastDrain(SessionBroker):
            async def _broadcast_worker(self) -> None:  # type: ignore[override]
                await asyncio.Event().wait()

        broker = _NoBroadcastDrain(
            session_id="demo",
            recorder=recorder,
            settings=BrokerSettings(
                storage_capacity=64,
                broadcast_capacity=1,
                operation_timeout_seconds=0.05,
            ),
            clock=ManualClock(),
        )
        await broker.start()
        with caplog.at_level(logging.WARNING, logger="engagevr.api.broker"):
            for n in range(4):
                result = await broker.ingest(decoded(sequence_number=n))
                assert result.stored is True, "storage is unaffected by observer lag"
        await broker.stop()

        summary = store.read_summary("demo")
        assert summary is not None
        assert summary.event_count == 4
        assert summary.dropped_message_count == 0
        assert any("broadcast queue full" in r.getMessage() for r in caplog.records)


class TestQueueTimeout:
    @pytest.mark.asyncio
    async def test_the_critical_wait_is_bounded_by_the_operation_timeout(
        self, store: SessionStore
    ) -> None:
        recorder = store.open_recorder("demo")
        broker = _StalledBroker(
            session_id="demo",
            recorder=recorder,
            settings=BrokerSettings(storage_capacity=1, operation_timeout_seconds=0.1),
            clock=ManualClock(),
        )
        await broker.start()
        try:
            await broker.ingest(decoded(sequence_number=0))
            loop = asyncio.get_running_loop()
            started = loop.time()
            with pytest.raises(QueueFullError):
                await broker.ingest(decoded(MessageType.SESSION_END, sequence_number=1))
            elapsed = loop.time() - started
            assert 0.05 <= elapsed < 2.0, "the wait must be bounded, not indefinite"
        finally:
            await broker.stop()

    @pytest.mark.asyncio
    async def test_a_critical_message_succeeds_once_space_appears(
        self, store: SessionStore
    ) -> None:
        recorder = store.open_recorder("demo")
        broker = SessionBroker(
            session_id="demo",
            recorder=recorder,
            settings=BrokerSettings(storage_capacity=2, operation_timeout_seconds=1.0),
            clock=ManualClock(),
        )
        await broker.start()
        try:
            for n in range(6):
                result = await broker.ingest(
                    decoded(MessageType.SESSION_START, sequence_number=n)
                )
                assert result.stored is True
        finally:
            await broker.stop()

        summary = store.read_summary("demo")
        assert summary is not None
        assert summary.event_count == 6
        assert summary.dropped_message_count == 0


class TestBrokerLifecycle:
    @pytest.mark.asyncio
    async def test_stop_drains_writes_the_summary_and_closes_the_files(
        self, store: SessionStore
    ) -> None:
        recorder = store.open_recorder("demo")
        broker = SessionBroker(
            session_id="demo", recorder=recorder, clock=ManualClock()
        )
        await broker.start()
        for n in range(5):
            await broker.ingest(decoded(sequence_number=n))
        await broker.stop(disconnect_reason=DisconnectReason.SERVER_SHUTDOWN)

        summary = store.read_summary("demo")
        assert summary is not None
        assert summary.event_count == 5
        assert summary.disconnect_reason is DisconnectReason.SERVER_SHUTDOWN
        assert len(list(store.iter_messages("demo"))) == 5

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, store: SessionStore) -> None:
        broker = SessionBroker(
            session_id="demo",
            recorder=store.open_recorder("demo"),
            clock=ManualClock(),
        )
        await broker.start()
        await broker.stop()
        await broker.stop()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, store: SessionStore) -> None:
        broker = SessionBroker(
            session_id="demo",
            recorder=store.open_recorder("demo"),
            clock=ManualClock(),
        )
        await broker.start()
        await broker.start()
        await broker.stop()

    @pytest.mark.asyncio
    async def test_server_sequence_numbers_increase(self, store: SessionStore) -> None:
        broker = SessionBroker(
            session_id="demo",
            recorder=store.open_recorder("demo"),
            clock=ManualClock(),
        )
        assert [broker.next_server_sequence() for _ in range(3)] == [0, 1, 2]
        await broker.start()
        await broker.stop()
