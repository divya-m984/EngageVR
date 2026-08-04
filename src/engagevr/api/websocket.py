"""The WebSocket bridge: ``/ws/v1/sessions/{session_id}``.

Conversation
------------
1. Client connects.  Nothing is accepted until the handshake completes.
2. Client sends ``client_hello``.  Any other first message is rejected
   with ``handshake_required`` and the connection is closed.
3. Backend replies ``server_hello`` with the assigned client id, the
   heartbeat interval, and the message-size limit.
4. Client sends task messages; backend replies ``acknowledgement`` per
   message, or ``protocol_error`` on rejection.
5. Backend routes ``adaptation_command`` to task clients and broadcasts
   observable messages to observers.
6. Either side closes; the backend records why.

Every inbound frame is validated against the shared protocol schema
before anything is done with it.  Text JSON frames only: a binary frame
is rejected, because the protocol has no binary representation and
accepting one would mean guessing.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from engagevr._logging import get_logger
from engagevr.api.broker import QueueFullError, SessionBroker
from engagevr.api.connections import ClientConnection
from engagevr.api.errors import (
    BACKEND_PROVENANCE,
    build_protocol_error,
    protocol_error_from_validation,
)
from engagevr.api.state import ApplicationState
from engagevr.protocol.envelope import build_envelope
from engagevr.protocol.messages import (
    AdaptationAcknowledgementPayload,
    ClientHelloPayload,
    ClientRole,
    DisconnectReason,
    HeartbeatAcknowledgementPayload,
    HeartbeatPayload,
    MessageSource,
    MessageType,
    ProtocolErrorCode,
    ServerHelloPayload,
)
from engagevr.protocol.validation import (
    DecodedMessage,
    ProtocolValidationError,
    decode_message,
)
from engagevr.protocol.version import PROTOCOL_VERSION
from engagevr.storage.session_store import InvalidSessionIdError, validate_session_id

logger = get_logger("api.websocket")

SERVER_NAME = "engagevr-backend"

#: WebSocket close codes.  1000 is a normal closure; 1008 is a policy
#: violation, which is what a protocol rejection is.
_CLOSE_NORMAL = 1000
_CLOSE_POLICY_VIOLATION = 1008

#: Roles a client may claim, mapped to the source they must declare.
_ROLE_SOURCES: dict[ClientRole, tuple[MessageSource, ...]] = {
    ClientRole.SIMULATOR: (MessageSource.PYTHON_SIMULATOR, MessageSource.TEST_FIXTURE),
    ClientRole.UNITY: (MessageSource.UNITY_CLIENT, MessageSource.TEST_FIXTURE),
    ClientRole.OBSERVER: (
        MessageSource.BACKEND,
        MessageSource.TEST_FIXTURE,
        MessageSource.PYTHON_SIMULATOR,
    ),
    ClientRole.REPLAY: (
        MessageSource.REPLAY,
        MessageSource.PYTHON_SIMULATOR,
        MessageSource.UNITY_CLIENT,
        MessageSource.TEST_FIXTURE,
    ),
}

#: Message types an observer is permitted to send.  An observer is
#: read-only: it may not inject task telemetry into a session.
_OBSERVER_PERMITTED: frozenset[MessageType] = frozenset(
    {MessageType.CLIENT_HELLO, MessageType.HEARTBEAT, MessageType.REPLAY_CONTROL}
)


class _ServerSequence:
    """Monotonic per-connection sequence counter for backend messages."""

    def __init__(self) -> None:
        self._value = 0

    def next(self) -> int:
        value = self._value
        self._value += 1
        return value


async def session_websocket(
    websocket: WebSocket, session_id: str, state: ApplicationState
) -> None:
    """Serve one WebSocket connection for ``session_id``."""
    try:
        validate_session_id(session_id)
    except InvalidSessionIdError as exc:
        await websocket.close(code=_CLOSE_POLICY_VIOLATION, reason="invalid session id")
        logger.warning("rejected WebSocket for invalid session id: %s", exc)
        return

    await websocket.accept()
    sequence = _ServerSequence()
    maximum_bytes = state.config.server.maximum_message_bytes

    connection: ClientConnection | None = None
    broker: SessionBroker | None = None
    reason = DisconnectReason.ORDERLY

    try:
        handshake = await _await_handshake(
            websocket,
            session_id=session_id,
            sequence=sequence,
            maximum_bytes=maximum_bytes,
            timeout=state.config.server.connection_timeout_seconds,
        )
        if handshake is None:
            reason = DisconnectReason.INVALID_PROTOCOL
            return

        hello_message, hello = handshake
        broker = await state.acquire_broker(session_id)
        connection = state.registry.register(
            session_id=session_id,
            role=hello.role,
            source=hello_message.envelope.source,
            client_name=hello.client_name,
            sender=websocket,
            connected_at_utc=state.clock.utc_now(),
        )
        connection.handshake_complete = True
        connection.last_seen_monotonic = state.clock.monotonic()

        await broker.ingest(
            hello_message,
            client_id=connection.client_id,
            client_role=connection.role.value,
            transport="websocket",
        )
        await _send_server_hello(
            connection,
            state=state,
            sequence=sequence,
            session_id=session_id,
            correlation_id=hello_message.message_id,
        )
        logger.info(
            "client %s (%s, role=%s) joined session %s",
            connection.client_id,
            hello.client_name,
            hello.role.value,
            session_id,
        )

        reason = await _serve(
            websocket,
            connection=connection,
            broker=broker,
            state=state,
            sequence=sequence,
        )
    except WebSocketDisconnect:
        reason = DisconnectReason.CLIENT_DISCONNECT
    except asyncio.CancelledError:
        reason = DisconnectReason.SERVER_SHUTDOWN
        raise
    except Exception:
        reason = DisconnectReason.INTERNAL_FAILURE
        logger.exception("internal failure on session %s", session_id)
    finally:
        if connection is not None:
            state.registry.remove(connection.client_id, reason=reason)
        with contextlib.suppress(Exception):
            if websocket.client_state is WebSocketState.CONNECTED:
                await websocket.close(
                    code=(
                        _CLOSE_NORMAL
                        if reason
                        in (
                            DisconnectReason.ORDERLY,
                            DisconnectReason.CLIENT_DISCONNECT,
                        )
                        else _CLOSE_POLICY_VIOLATION
                    )
                )
        if broker is not None:
            await state.release_broker(session_id, disconnect_reason=reason)


async def _await_handshake(
    websocket: WebSocket,
    *,
    session_id: str,
    sequence: _ServerSequence,
    maximum_bytes: int,
    timeout: float,
) -> tuple[DecodedMessage, ClientHelloPayload] | None:
    """Read and validate the client's first message.

    Returns None when the handshake failed; the caller closes.
    """
    try:
        raw = await asyncio.wait_for(websocket.receive(), timeout=timeout)
    except TimeoutError:
        await _send_raw_error(
            websocket,
            session_id=session_id,
            sequence=sequence,
            error_code=ProtocolErrorCode.HANDSHAKE_REQUIRED,
            detail=f"no client_hello arrived within {timeout} seconds",
        )
        return None

    text = _extract_text(raw)
    if text is None:
        await _send_raw_error(
            websocket,
            session_id=session_id,
            sequence=sequence,
            error_code=ProtocolErrorCode.INVALID_JSON,
            detail=(
                "only JSON text frames are accepted; the protocol has no "
                "binary representation"
            ),
        )
        return None

    try:
        message = decode_message(text, maximum_message_bytes=maximum_bytes)
    except ProtocolValidationError as exc:
        await _send_error(
            websocket,
            protocol_error_from_validation(
                exc, session_id=session_id, sequence_number=sequence.next()
            ),
        )
        return None

    if message.message_type is not MessageType.CLIENT_HELLO:
        await _send_raw_error(
            websocket,
            session_id=session_id,
            sequence=sequence,
            error_code=ProtocolErrorCode.HANDSHAKE_REQUIRED,
            detail=(
                f"first message must be client_hello, got "
                f"{message.message_type.value!r}"
            ),
            offending_message_id=message.message_id,
        )
        return None

    hello = message.payload
    assert isinstance(hello, ClientHelloPayload)

    if hello.protocol_version != message.envelope.protocol_version:
        await _send_raw_error(
            websocket,
            session_id=session_id,
            sequence=sequence,
            error_code=ProtocolErrorCode.HANDSHAKE_REJECTED,
            detail=(
                f"client_hello payload declares protocol {hello.protocol_version!r} "
                f"but its envelope declares {message.envelope.protocol_version!r}"
            ),
            offending_message_id=message.message_id,
        )
        return None

    if not _session_matches(message, session_id):
        await _send_raw_error(
            websocket,
            session_id=session_id,
            sequence=sequence,
            error_code=ProtocolErrorCode.SESSION_MISMATCH,
            detail=(
                f"client_hello is for session {message.envelope.session_id!r} "
                f"but the connection is on {session_id!r}"
            ),
            offending_message_id=message.message_id,
        )
        return None

    permitted = _ROLE_SOURCES[hello.role]
    if message.envelope.source not in permitted:
        allowed = ", ".join(s.value for s in permitted)
        await _send_raw_error(
            websocket,
            session_id=session_id,
            sequence=sequence,
            error_code=ProtocolErrorCode.HANDSHAKE_REJECTED,
            detail=(
                f"role {hello.role.value!r} may not declare source "
                f"{message.envelope.source.value!r}; permitted: {allowed}"
            ),
            offending_message_id=message.message_id,
        )
        return None

    return message, hello


async def _send_server_hello(
    connection: ClientConnection,
    *,
    state: ApplicationState,
    sequence: _ServerSequence,
    session_id: str,
    correlation_id: str,
) -> None:
    server = state.config.server
    payload = ServerHelloPayload(
        accepted=True,
        server_name=SERVER_NAME,
        server_version=state.config.project.version,
        protocol_version=PROTOCOL_VERSION,
        session_id=session_id,
        assigned_client_id=connection.client_id,
        server_time_utc=state.clock.utc_now(),
        heartbeat_interval_seconds=server.heartbeat_interval_seconds,
        connection_timeout_seconds=server.connection_timeout_seconds,
        maximum_message_bytes=server.maximum_message_bytes,
    )
    await connection.send(
        build_envelope(
            message_type=MessageType.SERVER_HELLO,
            session_id=session_id,
            source=MessageSource.BACKEND,
            sequence_number=sequence.next(),
            payload=payload,
            provenance=BACKEND_PROVENANCE,
            correlation_id=correlation_id,
            sent_at_utc=state.clock.utc_now(),
            sent_at_monotonic_seconds=state.clock.monotonic(),
        )
    )


async def _serve(
    websocket: WebSocket,
    *,
    connection: ClientConnection,
    broker: SessionBroker,
    state: ApplicationState,
    sequence: _ServerSequence,
) -> DisconnectReason:
    """The post-handshake receive loop."""
    server = state.config.server
    maximum_bytes = server.maximum_message_bytes
    session_id = connection.session_id

    while True:
        try:
            raw = await asyncio.wait_for(
                websocket.receive(), timeout=server.connection_timeout_seconds
            )
        except TimeoutError:
            await _try_send(
                connection,
                build_protocol_error(
                    session_id=session_id,
                    sequence_number=sequence.next(),
                    error_code=ProtocolErrorCode.INTERNAL_ERROR,
                    detail=(
                        f"no message received for "
                        f"{server.connection_timeout_seconds} seconds; the "
                        f"heartbeat interval is "
                        f"{server.heartbeat_interval_seconds} seconds"
                    ),
                    fatal=True,
                ),
            )
            return DisconnectReason.TIMEOUT

        if raw.get("type") == "websocket.disconnect":
            return DisconnectReason.CLIENT_DISCONNECT

        connection.last_seen_monotonic = state.clock.monotonic()
        text = _extract_text(raw)
        if text is None:
            await _try_send(
                connection,
                build_protocol_error(
                    session_id=session_id,
                    sequence_number=sequence.next(),
                    error_code=ProtocolErrorCode.INVALID_JSON,
                    detail="only JSON text frames are accepted",
                ),
            )
            continue

        try:
            message = decode_message(text, maximum_message_bytes=maximum_bytes)
        except ProtocolValidationError as exc:
            await _try_send(
                connection,
                protocol_error_from_validation(
                    exc, session_id=session_id, sequence_number=sequence.next()
                ),
            )
            if exc.fatal:
                return DisconnectReason.INVALID_PROTOCOL
            continue

        connection.inbound_count += 1
        outcome = await _handle_message(
            message,
            connection=connection,
            broker=broker,
            state=state,
            sequence=sequence,
        )
        if outcome is not None:
            return outcome


async def _handle_message(
    message: DecodedMessage,
    *,
    connection: ClientConnection,
    broker: SessionBroker,
    state: ApplicationState,
    sequence: _ServerSequence,
) -> DisconnectReason | None:
    """Process one validated message. Returns a reason to disconnect, or None."""
    session_id = connection.session_id
    envelope = message.envelope

    if not _session_matches(message, session_id):
        await _try_send(
            connection,
            build_protocol_error(
                session_id=session_id,
                sequence_number=sequence.next(),
                error_code=ProtocolErrorCode.SESSION_MISMATCH,
                detail=(
                    f"message declares session {envelope.session_id!r} but the "
                    f"connection is on {session_id!r}"
                ),
                offending_message_id=envelope.message_id,
                offending_message_type=envelope.message_type.value,
                offending_sequence_number=envelope.sequence_number,
            ),
        )
        return None

    if connection.is_observer and envelope.message_type not in _OBSERVER_PERMITTED:
        allowed = ", ".join(sorted(t.value for t in _OBSERVER_PERMITTED))
        await _try_send(
            connection,
            build_protocol_error(
                session_id=session_id,
                sequence_number=sequence.next(),
                error_code=ProtocolErrorCode.ROLE_NOT_PERMITTED,
                detail=(
                    f"an observer may not send {envelope.message_type.value!r}; "
                    f"permitted: {allowed}"
                ),
                offending_message_id=envelope.message_id,
                offending_message_type=envelope.message_type.value,
            ),
        )
        return None

    if message.message_type is MessageType.HEARTBEAT:
        await _answer_heartbeat(
            message, connection=connection, state=state, sequence=sequence
        )
        return None

    try:
        result = await broker.ingest(
            message,
            client_id=connection.client_id,
            client_role=connection.role.value,
            transport="websocket",
        )
    except QueueFullError as exc:
        await _try_send(
            connection,
            build_protocol_error(
                session_id=session_id,
                sequence_number=sequence.next(),
                error_code=ProtocolErrorCode.QUEUE_FULL,
                detail=str(exc),
                offending_message_id=envelope.message_id,
                offending_message_type=envelope.message_type.value,
                fatal=True,
            ),
        )
        return DisconnectReason.INTERNAL_FAILURE

    await _try_send(
        connection,
        build_envelope(
            message_type=MessageType.ACKNOWLEDGEMENT,
            session_id=session_id,
            source=MessageSource.BACKEND,
            sequence_number=sequence.next(),
            payload=result.acknowledgement,
            provenance=BACKEND_PROVENANCE,
            correlation_id=envelope.message_id,
            sent_at_utc=state.clock.utc_now(),
            sent_at_monotonic_seconds=state.clock.monotonic(),
        ),
    )

    if message.message_type is MessageType.ADAPTATION_ACKNOWLEDGEMENT:
        payload = message.payload
        if isinstance(payload, AdaptationAcknowledgementPayload):
            logger.info(
                "session %s: command %s %s by %s",
                session_id,
                payload.command_id,
                "accepted" if payload.accepted else "rejected",
                connection.client_id,
            )
    return None


async def _answer_heartbeat(
    message: DecodedMessage,
    *,
    connection: ClientConnection,
    state: ApplicationState,
    sequence: _ServerSequence,
) -> None:
    """Reply to a heartbeat, echoing the client's monotonic reading.

    The client's own monotonic value is returned verbatim.  The backend
    does not translate it onto its own timeline, because the two
    monotonic clocks have unrelated origins.
    """
    payload = message.payload
    assert isinstance(payload, HeartbeatPayload)
    received = state.clock.utc_now()
    reply = HeartbeatAcknowledgementPayload(
        heartbeat_id=payload.heartbeat_id,
        client_monotonic_seconds=payload.client_monotonic_seconds,
        server_received_at_utc=received,
        server_sent_at_utc=state.clock.utc_now(),
    )
    await _try_send(
        connection,
        build_envelope(
            message_type=MessageType.HEARTBEAT_ACKNOWLEDGEMENT,
            session_id=connection.session_id,
            source=MessageSource.BACKEND,
            sequence_number=sequence.next(),
            payload=reply,
            provenance=BACKEND_PROVENANCE,
            correlation_id=message.message_id,
            sent_at_utc=state.clock.utc_now(),
            sent_at_monotonic_seconds=state.clock.monotonic(),
        ),
    )


async def _try_send(connection: ClientConnection, envelope: Any) -> bool:
    """Send, treating a dead peer as a normal condition rather than a crash."""
    try:
        await connection.send(envelope)
        return True
    except Exception:
        logger.debug(
            "could not send %s to %s; the peer is gone",
            envelope.message_type.value,
            connection.client_id,
        )
        return False


async def _send_error(websocket: WebSocket, envelope: Any) -> None:
    import json

    with contextlib.suppress(Exception):
        await websocket.send_text(
            json.dumps(envelope.to_json_dict(), separators=(",", ":"))
        )


async def _send_raw_error(
    websocket: WebSocket,
    *,
    session_id: str,
    sequence: _ServerSequence,
    error_code: ProtocolErrorCode,
    detail: str,
    offending_message_id: str | None = None,
) -> None:
    await _send_error(
        websocket,
        build_protocol_error(
            session_id=session_id,
            sequence_number=sequence.next(),
            error_code=error_code,
            detail=detail,
            offending_message_id=offending_message_id,
            fatal=True,
        ),
    )


def _session_matches(message: DecodedMessage, session_id: str) -> bool:
    """Whether a message belongs on this connection's session.

    A live message must declare the connection's session id.  A
    **replayed** message keeps the session id it was recorded under —
    rewriting it would falsify the recording — so it is matched on its
    ``replay.replay_session_id`` instead, which is the session the
    replay is deliberately publishing under.
    """
    envelope = message.envelope
    if envelope.session_id == session_id:
        return True
    return envelope.replay is not None and (
        envelope.replay.replay_session_id == session_id
    )


def _extract_text(raw: Mapping[str, Any]) -> str | None:
    """Pull the text of a Starlette WebSocket receive event.

    Binary frames return None: the protocol is JSON text only.
    """
    text = raw.get("text")
    return text if isinstance(text, str) else None
