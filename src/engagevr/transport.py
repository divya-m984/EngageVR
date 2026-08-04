"""Message transports shared by the task simulator and the replay player.

A transport moves already-validated :class:`MessageEnvelope` objects.
It does not know what a task is, and it does not interpret payloads.

Three implementations are provided:

``InProcessTransport``
    Delivers to an in-process broker with no sockets and no event-loop
    scheduling subtleties.  This is what the tests use, so the whole
    simulator/backend/replay path is exercisable without a server.

``JsonlFileTransport``
    Writes a session recording directly, for offline runs that never
    contact a backend.

``WebSocketTransport``
    A real client connection to the FastAPI bridge.

All three are async context managers with the same three methods, so a
caller is written once and runs in every mode.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from types import TracebackType
from typing import Protocol

from engagevr.protocol.envelope import MessageEnvelope
from engagevr.protocol.validation import (
    DEFAULT_MAXIMUM_MESSAGE_BYTES,
    DecodedMessage,
    ProtocolValidationError,
    decode_message,
)
from engagevr.storage.manifest import IngestionMetadata
from engagevr.storage.session_store import SessionRecorder
from engagevr.synchronization.clock import Clock, SystemClock

#: Floor applied to a zero poll timeout on a real socket, so an
#: already-arrived frame gets one scheduler turn to be delivered.
_MINIMUM_POLL_SECONDS = 0.001


class TransportError(RuntimeError):
    """A transport failed to connect, send, or receive."""


class MessageTransport(Protocol):
    """What the simulator and replay player require of a transport."""

    async def connect(self) -> None:
        """Establish the transport. Idempotent."""
        ...

    async def send(self, envelope: MessageEnvelope) -> None:
        """Deliver one message."""
        ...

    async def receive(self, timeout: float | None = None) -> DecodedMessage | None:
        """Return the next inbound message, or None on timeout/close."""
        ...

    async def close(self) -> None:
        """Release the transport. Idempotent."""
        ...


class InProcessTransport:
    """Delivers messages to a callback, with an inbound queue for replies.

    Used by tests and by the in-process broker path.  The outbound
    callback may be async or sync; replies are pushed back with
    :meth:`deliver`.
    """

    def __init__(
        self,
        on_send: Callable[[MessageEnvelope], Awaitable[None] | None] | None = None,
        *,
        inbound_capacity: int = 256,
    ) -> None:
        self._on_send = on_send
        self._inbound: asyncio.Queue[DecodedMessage | None] = asyncio.Queue(
            maxsize=inbound_capacity
        )
        self.sent: list[MessageEnvelope] = []
        self._closed = False

    async def connect(self) -> None:
        self._closed = False

    async def send(self, envelope: MessageEnvelope) -> None:
        if self._closed:
            raise TransportError("transport is closed")
        self.sent.append(envelope)
        if self._on_send is not None:
            result = self._on_send(envelope)
            if asyncio.iscoroutine(result):
                await result

    async def deliver(self, message: DecodedMessage) -> None:
        """Push an inbound message towards the local receiver."""
        await self._inbound.put(message)

    async def receive(self, timeout: float | None = None) -> DecodedMessage | None:
        if self._closed and self._inbound.empty():
            return None
        if timeout is not None and timeout <= 0.0:
            # A zero timeout means "take what is already here". Routing
            # that through wait_for would cancel the get() before it ran
            # and lose an item that was in fact available.
            try:
                return self._inbound.get_nowait()
            except asyncio.QueueEmpty:
                return None
        try:
            if timeout is None:
                return await self._inbound.get()
            return await asyncio.wait_for(self._inbound.get(), timeout)
        except (TimeoutError, asyncio.CancelledError):
            return None

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            with_sentinel = self._inbound
            try:
                with_sentinel.put_nowait(None)
            except asyncio.QueueFull:  # pragma: no cover - best effort wake-up
                pass

    async def __aenter__(self) -> InProcessTransport:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


class JsonlFileTransport:
    """Records messages straight into a session directory.

    Used when the simulator runs with ``--output`` and no server.  The
    ingestion metadata records ``transport="file"`` and the writer's own
    clock, which is honest: there was no network hop and no second
    machine involved.
    """

    def __init__(
        self, recorder: SessionRecorder, *, clock: Clock | None = None
    ) -> None:
        self._recorder = recorder
        self._clock = clock if clock is not None else SystemClock()
        self._closed = False

    @property
    def recorder(self) -> SessionRecorder:
        return self._recorder

    async def connect(self) -> None:
        self._closed = False

    async def send(self, envelope: MessageEnvelope) -> None:
        if self._closed:
            raise TransportError("transport is closed")
        ingestion = IngestionMetadata(
            arrival_index=self._recorder.next_arrival_index(),
            server_received_at_utc=self._clock.utc_now(),
            server_monotonic_seconds=self._clock.monotonic(),
            transport="file",
            client_id=None,
            client_role=None,
        )
        self._recorder.append(envelope, ingestion)

    async def receive(self, timeout: float | None = None) -> DecodedMessage | None:
        """Always None: a file has nothing to say back."""
        return None

    async def close(self) -> None:
        self._closed = True

    async def __aenter__(self) -> JsonlFileTransport:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


class WebSocketTransport:
    """A real WebSocket client connection to the local backend.

    Built on the ``websockets`` library.  Inbound frames are validated
    through the same pipeline the server uses, so a malformed reply from
    the server is rejected rather than acted upon.
    """

    def __init__(
        self,
        url: str,
        *,
        maximum_message_bytes: int = DEFAULT_MAXIMUM_MESSAGE_BYTES,
        open_timeout: float = 10.0,
    ) -> None:
        self._url = url
        self._maximum_message_bytes = maximum_message_bytes
        self._open_timeout = open_timeout
        self._connection: object | None = None
        self._closed = False

    @property
    def url(self) -> str:
        return self._url

    async def connect(self) -> None:
        if self._connection is not None:
            return
        try:
            from websockets.asyncio.client import connect as ws_connect
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise TransportError(
                "the 'websockets' package is required for WebSocket transport"
            ) from exc
        try:
            self._connection = await ws_connect(
                self._url,
                max_size=self._maximum_message_bytes,
                open_timeout=self._open_timeout,
            )
        except OSError as exc:
            raise TransportError(f"cannot connect to {self._url}: {exc}") from exc
        except Exception as exc:  # websockets raises its own exception tree
            raise TransportError(f"cannot connect to {self._url}: {exc}") from exc
        self._closed = False

    async def send(self, envelope: MessageEnvelope) -> None:
        connection = self._require_connection()
        frame = json.dumps(envelope.to_json_dict(), separators=(",", ":"))
        try:
            await connection.send(frame)  # type: ignore[attr-defined]
        except Exception as exc:
            raise TransportError(f"send failed on {self._url}: {exc}") from exc

    async def receive(self, timeout: float | None = None) -> DecodedMessage | None:
        connection = self._require_connection()
        try:
            if timeout is None:
                raw = await connection.recv()  # type: ignore[attr-defined]
            else:
                # A zero timeout still needs one scheduler turn, otherwise
                # a frame that has already arrived would be reported as
                # "nothing waiting".
                raw = await asyncio.wait_for(
                    connection.recv(),  # type: ignore[attr-defined]
                    max(timeout, _MINIMUM_POLL_SECONDS),
                )
        except TimeoutError:
            return None
        except Exception:
            # A closed connection is a normal end of stream, not an error
            # the caller must distinguish here; close() reports the state.
            self._closed = True
            return None
        try:
            return decode_message(
                raw, maximum_message_bytes=self._maximum_message_bytes
            )
        except ProtocolValidationError as exc:
            raise TransportError(
                f"server sent a message this client rejected: {exc.detail}"
            ) from exc

    async def close(self) -> None:
        self._closed = True
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                await connection.close()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - closing a dead socket
                pass

    def _require_connection(self) -> object:
        if self._connection is None:
            raise TransportError("WebSocket transport is not connected")
        return self._connection

    async def __aenter__(self) -> WebSocketTransport:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


async def drain(
    transport: MessageTransport, *, timeout: float = 0.05
) -> AsyncIterator[DecodedMessage]:
    """Yield every message currently available, then stop.

    Used to collect acknowledgements without blocking a send loop.
    """
    while True:
        message = await transport.receive(timeout=timeout)
        if message is None:
            return
        yield message


__all__ = [
    "InProcessTransport",
    "JsonlFileTransport",
    "MessageTransport",
    "TransportError",
    "WebSocketTransport",
    "drain",
]
