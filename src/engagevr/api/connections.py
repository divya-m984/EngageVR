"""In-process connection registry.

Explicitly single-process
-------------------------
This registry lives in one Python process's memory.  Running the server
under multiple uvicorn workers, or across multiple machines, would give
each worker a *different* registry: a command routed by worker A would
never reach a client connected to worker B, and an observer on one
worker would see only that worker's traffic.

Multi-worker and distributed operation are therefore **not supported**
in Milestone 4.  The ``serve`` command runs a single worker and does not
expose a workers option.  Making this work across processes needs a
shared broker, which is deliberately out of scope here.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from engagevr.protocol.envelope import MessageEnvelope
from engagevr.protocol.messages import ClientRole, DisconnectReason, MessageSource


class ConnectionClosedError(RuntimeError):
    """A send was attempted on a connection that is no longer open."""


@dataclass
class ClientConnection:
    """One connected client.

    ``send`` is serialized by a lock: concurrent writers on a single
    WebSocket would interleave frames, which the protocol has no way to
    recover from.
    """

    client_id: str
    session_id: str
    role: ClientRole
    source: MessageSource
    client_name: str
    connected_at_utc: datetime
    sender: object  # a WebSocket-like object with an async send_text
    handshake_complete: bool = False
    last_seen_monotonic: float = 0.0
    outbound_count: int = 0
    inbound_count: int = 0
    disconnect_reason: DisconnectReason | None = None
    _closed: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def is_task_client(self) -> bool:
        """Whether this connection may receive adaptation commands."""
        return self.role in (ClientRole.SIMULATOR, ClientRole.UNITY)

    @property
    def is_observer(self) -> bool:
        return self.role is ClientRole.OBSERVER

    @property
    def closed(self) -> bool:
        return self._closed

    async def send(self, envelope: MessageEnvelope) -> None:
        """Send one envelope as a JSON text frame."""
        if self._closed:
            raise ConnectionClosedError(
                f"connection {self.client_id} is closed; cannot send "
                f"{envelope.message_type.value}"
            )
        import json

        frame = json.dumps(envelope.to_json_dict(), separators=(",", ":"))
        async with self._lock:
            await self.sender.send_text(frame)  # type: ignore[attr-defined]
        self.outbound_count += 1

    def mark_closed(self, reason: DisconnectReason) -> None:
        """Record why this connection ended. Idempotent."""
        if not self._closed:
            self._closed = True
            self.disconnect_reason = reason


class ConnectionRegistry:
    """All connections in this process, indexed by session.

    Removal is unconditional: a disconnected client is taken out of the
    registry on every exit path, including protocol failures and
    internal errors, so a dead connection can never be selected as a
    routing target.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, ClientConnection] = {}
        self._by_session: dict[str, set[str]] = {}

    def __len__(self) -> int:
        return len(self._by_id)

    def register(
        self,
        *,
        session_id: str,
        role: ClientRole,
        source: MessageSource,
        client_name: str,
        sender: object,
        connected_at_utc: datetime,
        client_id: str | None = None,
    ) -> ClientConnection:
        """Add a connection and return its record."""
        assigned = client_id if client_id is not None else uuid.uuid4().hex
        connection = ClientConnection(
            client_id=assigned,
            session_id=session_id,
            role=role,
            source=source,
            client_name=client_name,
            connected_at_utc=connected_at_utc,
            sender=sender,
        )
        self._by_id[assigned] = connection
        self._by_session.setdefault(session_id, set()).add(assigned)
        return connection

    def remove(
        self, client_id: str, *, reason: DisconnectReason
    ) -> ClientConnection | None:
        """Remove a connection, recording why it ended."""
        connection = self._by_id.pop(client_id, None)
        if connection is None:
            return None
        connection.mark_closed(reason)
        peers = self._by_session.get(connection.session_id)
        if peers is not None:
            peers.discard(client_id)
            if not peers:
                del self._by_session[connection.session_id]
        return connection

    def get(self, client_id: str) -> ClientConnection | None:
        return self._by_id.get(client_id)

    def for_session(self, session_id: str) -> list[ClientConnection]:
        """Every live connection on a session, in registration order."""
        ids = self._by_session.get(session_id, set())
        return [self._by_id[i] for i in sorted(ids, key=self._registration_key)]

    def observers(self, session_id: str) -> list[ClientConnection]:
        return [c for c in self.for_session(session_id) if c.is_observer]

    def task_clients(self, session_id: str) -> list[ClientConnection]:
        return [c for c in self.for_session(session_id) if c.is_task_client]

    def sessions(self) -> list[str]:
        return sorted(self._by_session)

    def _registration_key(self, client_id: str) -> tuple[datetime, str]:
        connection = self._by_id[client_id]
        return (connection.connected_at_utc, client_id)

    def select_command_targets(
        self,
        session_id: str,
        *,
        target_role: ClientRole,
        target_client_id: str | None,
    ) -> list[ClientConnection]:
        """Pick which clients an adaptation command is routed to.

        A command with an explicit ``target_client_id`` goes only to that
        client, and only if it is on this session and is a task client.
        Otherwise every task client on the session with the requested
        role receives it.  Observers never receive commands.
        """
        candidates = self.task_clients(session_id)
        if target_client_id is not None:
            return [c for c in candidates if c.client_id == target_client_id]
        return [c for c in candidates if c.role is target_role]

    def all_connections(self) -> Iterable[ClientConnection]:
        return tuple(self._by_id.values())
