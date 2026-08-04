"""Application state shared by the HTTP routes and the WebSocket bridge.

Held on ``app.state`` and created by the lifespan context, so nothing is
constructed at import time and every resource has an owner responsible
for closing it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from engagevr._logging import get_logger
from engagevr.api.broker import OBSERVABLE_MESSAGE_TYPES, BrokerSettings, SessionBroker
from engagevr.api.connections import ConnectionRegistry
from engagevr.config import EngageVRConfig
from engagevr.protocol.envelope import MessageEnvelope
from engagevr.protocol.messages import DisconnectReason
from engagevr.storage.session_store import SessionStore
from engagevr.synchronization.clock import Clock, SystemClock

logger = get_logger("api.state")


class ApplicationState:
    """Owns the session store, the connection registry, and live brokers.

    Single-process by construction; see
    :mod:`engagevr.api.connections` for why that matters and what is
    consequently unsupported.
    """

    def __init__(
        self,
        config: EngageVRConfig,
        *,
        session_root: Path | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.clock = clock if clock is not None else SystemClock()
        self.store = SessionStore(
            session_root
            if session_root is not None
            else Path(config.sessions.root_directory)
        )
        self.registry = ConnectionRegistry()
        self._brokers: dict[str, SessionBroker] = {}
        self._broker_lock = asyncio.Lock()

    @property
    def broker_settings(self) -> BrokerSettings:
        queues = self.config.queues
        protocol = self.config.protocol
        return BrokerSettings(
            ingestion_capacity=queues.ingestion_capacity,
            storage_capacity=queues.storage_capacity,
            broadcast_capacity=queues.broadcast_capacity,
            operation_timeout_seconds=queues.operation_timeout_seconds,
            maximum_clock_skew_seconds=protocol.maximum_clock_skew_seconds,
            maximum_transport_delay_seconds=protocol.maximum_transport_delay_seconds,
            maximum_sequence_gap=protocol.maximum_sequence_gap,
        )

    def get_broker(self, session_id: str) -> SessionBroker | None:
        return self._brokers.get(session_id)

    def live_sessions(self) -> list[str]:
        return sorted(self._brokers)

    async def acquire_broker(self, session_id: str) -> SessionBroker:
        """Return the session's broker, creating and starting it if needed."""
        async with self._broker_lock:
            existing = self._brokers.get(session_id)
            if existing is not None:
                return existing
            recorder = self.store.open_recorder(
                session_id,
                configuration=self._configuration_snapshot(),
                flush_every=self.config.sessions.flush_every_events,
                engagevr_version=self.config.project.version,
                created_at_utc=self.clock.utc_now(),
            )
            broker = SessionBroker(
                session_id=session_id,
                recorder=recorder,
                settings=self.broker_settings,
                clock=self.clock,
                broadcast=self._broadcast,
            )
            await broker.start()
            self._brokers[session_id] = broker
            logger.info("opened session %s at %s", session_id, recorder.directory)
            return broker

    async def release_broker(
        self,
        session_id: str,
        *,
        disconnect_reason: DisconnectReason = DisconnectReason.ORDERLY,
    ) -> None:
        """Stop a session's broker once no client remains on it."""
        async with self._broker_lock:
            if self.registry.for_session(session_id):
                return
            broker = self._brokers.pop(session_id, None)
        if broker is None:
            return
        await broker.stop(disconnect_reason=disconnect_reason)
        logger.info(
            "closed session %s (%d events, %d dropped)",
            session_id,
            broker.recorder.event_count,
            broker.dropped_count,
        )

    async def shutdown(self) -> None:
        """Close every broker and mark every connection as shut down."""
        async with self._broker_lock:
            brokers = list(self._brokers.items())
            self._brokers.clear()
        for session_id, broker in brokers:
            await broker.stop(disconnect_reason=DisconnectReason.SERVER_SHUTDOWN)
            logger.info("shut down session %s", session_id)
        for connection in list(self.registry.all_connections()):
            self.registry.remove(
                connection.client_id, reason=DisconnectReason.SERVER_SHUTDOWN
            )

    async def _broadcast(self, session_id: str, envelope: MessageEnvelope) -> None:
        """Fan one message out to the session's observer clients."""
        if envelope.message_type not in OBSERVABLE_MESSAGE_TYPES:
            return
        for observer in self.registry.observers(session_id):
            if observer.closed:
                continue
            try:
                await observer.send(envelope)
            except Exception:
                logger.warning(
                    "observer %s could not be reached; removing it",
                    observer.client_id,
                )
                self.registry.remove(
                    observer.client_id, reason=DisconnectReason.INTERNAL_FAILURE
                )

    def _configuration_snapshot(self) -> dict[str, object]:
        """The settings a session ran under, recorded in its manifest."""
        return {
            "protocol": self.config.protocol.model_dump(mode="json"),
            "server": self.config.server.model_dump(
                mode="json", exclude={"host", "port"}
            ),
            "queues": self.config.queues.model_dump(mode="json"),
            "sessions": self.config.sessions.model_dump(mode="json"),
            "task": self.config.task.model_dump(mode="json"),
            "replay": self.config.replay.model_dump(mode="json"),
        }
