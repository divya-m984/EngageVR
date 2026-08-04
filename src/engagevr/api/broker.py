"""Per-session ingestion, storage, and broadcast with bounded queues.

Pipeline
--------
::

    WebSocket handler
          |
          v
    ingestion queue  (bounded, queues.ingestion_capacity)
          |
          v
    processor task   ordering diagnostics, arrival stamping
          |                    |
          v                    v
    storage queue        broadcast queue
    (bounded)            (bounded)
          |                    |
          v                    v
    JSONL writer        observer connections

Backpressure policy
-------------------
Every queue is bounded, so memory cannot grow without limit.  When a
queue is full:

**Critical messages** wait for space, up to
``queues.operation_timeout_seconds``.  If space never appears, the
message is *not* dropped: the connection is failed with a
``queue_full`` protocol error, because losing one of these silently
would make the recording untrustworthy.  Critical messages are
``session_start``, ``session_end``, ``adaptation_command``,
``adaptation_acknowledgement``, ``protocol_error``, and any
``task_event`` carrying ``task_completed``.

**Non-critical messages** (``task_event`` in general, ``task_state``,
``telemetry``, ``heartbeat``, ``acknowledgement``) are dropped
immediately rather than blocking the process.  Every drop is:

- counted, per message type, in the session summary;
- written to ``dropped.jsonl`` with a reason and the queue that was full;
- warned about in the session log.

Dropping is never silent, and a reader of a recording can always tell
the difference between "nothing happened" and "something was dropped".
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

from engagevr._logging import get_logger
from engagevr.protocol.envelope import MessageEnvelope
from engagevr.protocol.messages import (
    AcknowledgementPayload,
    DisconnectReason,
    MessageType,
    ProtocolErrorCode,
)
from engagevr.protocol.validation import DecodedMessage, is_critical
from engagevr.storage.manifest import DropRecord, IngestionMetadata
from engagevr.storage.session_store import SessionRecorder
from engagevr.synchronization.clock import Clock, SystemClock, assess_arrival
from engagevr.synchronization.ordering import OrderingAnomaly, SequenceTracker

logger = get_logger(__name__)


class QueueFullError(RuntimeError):
    """A critical message could not be queued before the timeout expired."""

    error_code = ProtocolErrorCode.QUEUE_FULL


@dataclass(frozen=True, slots=True)
class BrokerSettings:
    """Bounded-queue capacities and timing tolerances for one session."""

    ingestion_capacity: int = 1024
    storage_capacity: int = 1024
    broadcast_capacity: int = 256
    operation_timeout_seconds: float = 2.0
    maximum_clock_skew_seconds: float = 5.0
    maximum_transport_delay_seconds: float = 2.0
    maximum_sequence_gap: int = 1000


@dataclass
class IngestionResult:
    """The outcome of ingesting one message."""

    stored: bool
    dropped: bool
    arrival_index: int | None
    anomalies: tuple[OrderingAnomaly, ...]
    acknowledgement: AcknowledgementPayload


@dataclass(frozen=True, slots=True)
class _StorageItem:
    envelope: MessageEnvelope
    ingestion: IngestionMetadata


class SessionBroker:
    """Owns one session's recorder, queues, and worker tasks.

    Start with :meth:`start` and always stop with :meth:`stop`, which
    drains what it can, cancels the workers, closes the recorder, and
    writes ``summary.json``.
    """

    def __init__(
        self,
        *,
        session_id: str,
        recorder: SessionRecorder,
        settings: BrokerSettings | None = None,
        clock: Clock | None = None,
        broadcast: object | None = None,
    ) -> None:
        self._session_id = session_id
        self._recorder = recorder
        self._settings = settings if settings is not None else BrokerSettings()
        self._clock = clock if clock is not None else SystemClock()
        self._broadcast_sink = broadcast

        self._storage_queue: asyncio.Queue[_StorageItem | None] = asyncio.Queue(
            maxsize=self._settings.storage_capacity
        )
        self._broadcast_queue: asyncio.Queue[MessageEnvelope | None] = asyncio.Queue(
            maxsize=self._settings.broadcast_capacity
        )
        self._tracker = SequenceTracker(
            maximum_sequence_gap=self._settings.maximum_sequence_gap
        )
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._dropped_count = 0
        self._sequence = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def recorder(self) -> SessionRecorder:
        return self._recorder

    @property
    def dropped_count(self) -> int:
        """How many non-critical messages backpressure has discarded."""
        return self._dropped_count

    @property
    def storage_queue_depth(self) -> int:
        return self._storage_queue.qsize()

    @property
    def broadcast_queue_depth(self) -> int:
        return self._broadcast_queue.qsize()

    def next_server_sequence(self) -> int:
        """Reserve a sequence number for a backend-originated message."""
        value = self._sequence
        self._sequence += 1
        return value

    async def start(self) -> None:
        """Launch the storage and broadcast worker tasks."""
        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(
                self._storage_worker(), name=f"storage:{self._session_id}"
            ),
            asyncio.create_task(
                self._broadcast_worker(), name=f"broadcast:{self._session_id}"
            ),
        ]

    async def stop(
        self, *, disconnect_reason: DisconnectReason = DisconnectReason.ORDERLY
    ) -> None:
        """Drain what can be drained, then close everything.

        The shutdown sentinel is enqueued with ``put_nowait``.  A blocking
        ``put`` here would deadlock whenever the queue is already full and
        its worker is wedged — which is precisely the situation shutdown
        exists to escape.  If the sentinel does not fit, or a worker does
        not finish within the operation timeout, the worker is cancelled.
        Records still sitting in the queue at that point are lost, which
        is why the timeout is a bounded, deliberate cost rather than an
        indefinite wait.
        """
        if not self._running:
            return
        self._running = False

        for queue in (self._storage_queue, self._broadcast_queue):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                logger.warning(
                    "session %s: a queue was still full at shutdown; its worker "
                    "will be cancelled and any records left in it are lost",
                    self._session_id,
                )

        for worker in self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.shield(worker),
                    timeout=self._settings.operation_timeout_seconds,
                )
            except (TimeoutError, asyncio.CancelledError):
                pass
            except Exception:  # pragma: no cover - worker already logged it
                logger.exception("worker %s failed during shutdown", worker.get_name())
            if not worker.done():
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await worker

        self._workers.clear()
        self._recorder.close(disconnect_reason=disconnect_reason)

    async def ingest(
        self,
        message: DecodedMessage,
        *,
        client_id: str | None = None,
        client_role: str | None = None,
        transport: str = "websocket",
        same_process: bool = False,
    ) -> IngestionResult:
        """Stamp, diagnose, and enqueue one validated message.

        Raises
        ------
        QueueFullError
            When a critical message could not be queued in time.
        """
        envelope = message.envelope
        received_utc = self._clock.utc_now()
        received_monotonic = self._clock.monotonic()

        observation = self._tracker.observe(envelope)
        timing = assess_arrival(
            sent_at_utc=envelope.sent_at_utc,
            server_received_at_utc=received_utc,
            server_monotonic_seconds=received_monotonic,
            same_process=same_process,
            maximum_clock_skew_seconds=self._settings.maximum_clock_skew_seconds,
            maximum_transport_delay_seconds=(
                self._settings.maximum_transport_delay_seconds
            ),
        )

        anomalies = list(observation.anomalies)
        if timing.future_timestamp:
            anomalies.append(OrderingAnomaly.FUTURE_TIMESTAMP)
        if timing.excessive_delay:
            anomalies.append(OrderingAnomaly.EXCESSIVE_TRANSPORT_DELAY)

        critical = is_critical(envelope, message.payload)
        arrival_index = self._recorder.next_arrival_index()
        ingestion = IngestionMetadata(
            arrival_index=arrival_index,
            server_received_at_utc=received_utc,
            server_monotonic_seconds=received_monotonic,
            transport=transport,
            client_id=client_id,
            client_role=client_role,
            anomalies=anomalies,
            anomaly_detail=observation.detail,
            expected_sequence_number=observation.expected_sequence_number,
            apparent_transport_delay_seconds=timing.apparent_transport_delay_seconds,
            delay_unavailable_reason=timing.delay_unavailable_reason,
        )

        if anomalies:
            logger.warning(
                "ordering anomaly on session %s from %s seq=%d: %s",
                self._session_id,
                envelope.source.value,
                envelope.sequence_number,
                ", ".join(a.value for a in anomalies),
            )

        stored = await self._enqueue_storage(envelope, ingestion, critical=critical)
        if not stored:
            self._record_drop(envelope, queue="storage")

        await self._enqueue_broadcast(envelope, critical=critical)

        acknowledgement = AcknowledgementPayload(
            acknowledged_message_id=envelope.message_id,
            acknowledged_message_type=envelope.message_type,
            acknowledged_sequence_number=envelope.sequence_number,
            server_received_at_utc=received_utc,
            stored=stored,
            dropped=not stored,
        )
        return IngestionResult(
            stored=stored,
            dropped=not stored,
            arrival_index=arrival_index if stored else None,
            anomalies=tuple(anomalies),
            acknowledgement=acknowledgement,
        )

    async def _enqueue_storage(
        self,
        envelope: MessageEnvelope,
        ingestion: IngestionMetadata,
        *,
        critical: bool,
    ) -> bool:
        item = _StorageItem(envelope=envelope, ingestion=ingestion)
        try:
            self._storage_queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            pass
        if not critical:
            return False
        try:
            await asyncio.wait_for(
                self._storage_queue.put(item),
                timeout=self._settings.operation_timeout_seconds,
            )
        except TimeoutError as exc:
            raise QueueFullError(
                f"storage queue (capacity {self._settings.storage_capacity}) stayed "
                f"full for {self._settings.operation_timeout_seconds}s while "
                f"enqueuing critical message {envelope.message_type.value} "
                f"({envelope.message_id}); refusing to drop it"
            ) from exc
        return True

    async def _enqueue_broadcast(
        self, envelope: MessageEnvelope, *, critical: bool
    ) -> None:
        """Enqueue for observers.

        A full broadcast queue never blocks ingestion and never fails a
        critical message: observers are read-only monitors, and losing a
        monitoring frame must not cost a recorded event.  The drop is
        still counted and warned about.
        """
        try:
            self._broadcast_queue.put_nowait(envelope)
        except asyncio.QueueFull:
            logger.warning(
                "broadcast queue full on session %s (capacity %d); observer copy "
                "of %s %s was dropped. The message itself was %s.",
                self._session_id,
                self._settings.broadcast_capacity,
                envelope.message_type.value,
                envelope.message_id,
                "still stored" if critical else "handled by the storage policy",
            )

    def _record_drop(self, envelope: MessageEnvelope, *, queue: str) -> None:
        reason = (
            f"{queue} queue was full and {envelope.message_type.value} is not a "
            "critical message type; it was dropped rather than blocking ingestion"
        )
        logger.warning(
            "dropped non-critical message on session %s: %s %s (seq=%d): %s",
            self._session_id,
            envelope.message_type.value,
            envelope.message_id,
            envelope.sequence_number,
            reason,
        )
        self._recorder.record_drop(
            DropRecord(
                message_id=envelope.message_id,
                message_type=envelope.message_type,
                source=envelope.source,
                sequence_number=envelope.sequence_number,
                dropped_at_utc=self._clock.utc_now(),
                queue=queue,
                reason=reason,
            )
        )
        self._dropped_count += 1

    async def _storage_worker(self) -> None:
        while True:
            item = await self._storage_queue.get()
            if item is None:
                return
            try:
                self._recorder.append(item.envelope, item.ingestion)
            except Exception:  # pragma: no cover - disk failure
                logger.exception(
                    "failed to append %s to session %s",
                    item.envelope.message_id,
                    self._session_id,
                )

    async def _broadcast_worker(self) -> None:
        while True:
            envelope = await self._broadcast_queue.get()
            if envelope is None:
                return
            sink = self._broadcast_sink
            if sink is None:
                continue
            try:
                await sink(self._session_id, envelope)  # type: ignore[operator]
            except Exception:  # pragma: no cover - observer send failure
                logger.exception(
                    "broadcast of %s failed on session %s",
                    envelope.message_id,
                    self._session_id,
                )


#: Message types that are broadcast to observers.  Everything a task
#: client sends is observable; backend acknowledgements are not, to keep
#: observers from seeing a duplicate of every message.
OBSERVABLE_MESSAGE_TYPES: frozenset[MessageType] = frozenset(
    {
        MessageType.SESSION_START,
        MessageType.SESSION_END,
        MessageType.TASK_EVENT,
        MessageType.TASK_STATE,
        MessageType.TELEMETRY,
        MessageType.ADAPTATION_COMMAND,
        MessageType.ADAPTATION_ACKNOWLEDGEMENT,
        MessageType.PROTOCOL_ERROR,
    }
)
