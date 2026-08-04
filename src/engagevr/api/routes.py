"""HTTP endpoints for health, protocol metadata, and session inspection.

Everything here is read-only except ``POST /sessions/{id}/commands``,
which is the manual entry point for adaptation-command transport.  That
endpoint exists because Milestone 4 implements command *transport*: a
command has to come from somewhere, and in this milestone that
somewhere is a human or a test script, never a policy.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query, Request
from pydantic import BaseModel, Field

from engagevr.api.errors import BACKEND_PROVENANCE
from engagevr.api.state import ApplicationState
from engagevr.protocol.envelope import build_envelope
from engagevr.protocol.json_schema import build_protocol_json_schema
from engagevr.protocol.messages import (
    AdaptationCommandPayload,
    MessageSource,
    MessageType,
)
from engagevr.protocol.version import ACCEPTED_MAJOR_VERSIONS, PROTOCOL_VERSION
from engagevr.storage.manifest import RECORDING_DISCLAIMER
from engagevr.storage.session_store import (
    InvalidSessionIdError,
    SessionStoreError,
)

router = APIRouter()

#: Repeated on every response that describes the system's capabilities,
#: so a consumer of the API cannot infer more than the system does.
SCOPE_NOTE = (
    "This backend transports task telemetry and adaptation commands. It "
    "produces no engagement estimate, no cognitive-load estimate, and no "
    "psychological, clinical, or scientific conclusion. It implements no "
    "adaptation policy: every command is issued manually or by a test "
    "script."
)

SECURITY_NOTE = (
    "Development prototype. No authentication, no authorization, no "
    "transport encryption, no rate limiting. Bind to loopback only."
)


def _state(request: Request) -> ApplicationState:
    state: ApplicationState = request.app.state.engagevr
    return state


class CommandRequest(BaseModel):
    """Body of a manual adaptation-command request."""

    model_config = {"extra": "forbid"}

    command: AdaptationCommandPayload = Field(
        description="The command to route. Validated as a protocol payload."
    )


class CommandResponse(BaseModel):
    """Result of routing a manual adaptation command."""

    command_id: str
    session_id: str
    delivered_to: list[str]
    message_id: str
    note: str = (
        "Command transport only. No policy generated this command and no "
        "claim is made that it improves engagement or any other outcome."
    )


@router.get("/health", summary="Liveness probe")
async def health(request: Request) -> dict[str, Any]:
    state = _state(request)
    return {
        "status": "ok",
        "live_sessions": state.live_sessions(),
        "connections": len(state.registry),
    }


@router.get("/version", summary="Build and protocol versions")
async def version(request: Request) -> dict[str, Any]:
    state = _state(request)
    return {
        "name": state.config.project.name,
        "version": state.config.project.version,
        "protocol_version": PROTOCOL_VERSION,
        "session_root": str(state.store.root),
        "milestone": "4 -- task environment, simulator, real-time bridge, replay",
        "scope": SCOPE_NOTE,
        "security": SECURITY_NOTE,
    }


@router.get("/protocol", summary="Protocol metadata and JSON Schema")
async def protocol(
    include_schema: Annotated[bool, Query()] = False,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "accepted_major_versions": list(ACCEPTED_MAJOR_VERSIONS),
        "message_types": sorted(t.value for t in MessageType),
        "sources": sorted(s.value for s in MessageSource),
        "websocket_path": "/ws/v1/sessions/{session_id}",
        "scope": SCOPE_NOTE,
    }
    if include_schema:
        document["json_schema"] = build_protocol_json_schema()
    return document


@router.get("/sessions", summary="List recorded sessions")
async def list_sessions(request: Request) -> dict[str, Any]:
    state = _state(request)
    return {
        "session_root": str(state.store.root),
        "sessions": state.store.list_sessions(),
        "live_sessions": state.live_sessions(),
    }


@router.get("/sessions/{session_id}", summary="Session manifest and status")
async def get_session(request: Request, session_id: str) -> dict[str, Any]:
    state = _state(request)
    manifest = state.store.read_manifest(session_id)
    summary = state.store.read_summary(session_id)
    broker = state.get_broker(session_id)
    return {
        "manifest": manifest.model_dump(mode="json"),
        "summary": summary.model_dump(mode="json") if summary is not None else None,
        "live": broker is not None,
        "connected_clients": [
            {
                "client_id": c.client_id,
                "role": c.role.value,
                "client_name": c.client_name,
                "connected_at_utc": c.connected_at_utc.isoformat(),
            }
            for c in state.registry.for_session(session_id)
        ],
    }


@router.get("/sessions/{session_id}/events", summary="Recorded event stream")
async def get_session_events(
    request: Request,
    session_id: str,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 1000,
    offset: Annotated[int, Query(ge=0)] = 0,
    message_type: Annotated[str | None, Query()] = None,
    source: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    """Return recorded messages in their original arrival order.

    Arrival order is never re-sorted by sequence number: the two orders
    are recorded separately and both are present on every message.
    """
    state = _state(request)
    events: list[dict[str, Any]] = []
    total = 0
    for stored in state.store.iter_messages(session_id):
        if (
            message_type is not None
            and stored.envelope.message_type.value != message_type
        ):
            continue
        if source is not None and stored.envelope.source.value != source:
            continue
        total += 1
        if total <= offset:
            continue
        if len(events) >= limit:
            continue
        events.append(stored.model_dump(mode="json"))
    return {
        "session_id": session_id,
        "ordering": "original arrival order; never re-sorted by sequence number",
        "matched": total,
        "returned": len(events),
        "offset": offset,
        "limit": limit,
        "events": events,
        "disclaimer": RECORDING_DISCLAIMER,
    }


@router.get("/sessions/{session_id}/summary", summary="Session summary")
async def get_session_summary(request: Request, session_id: str) -> dict[str, Any]:
    """Return ``summary.json``, recovering one if the session never closed."""
    state = _state(request)
    summary = state.store.read_summary(session_id)
    recovered = False
    if summary is None:
        if not state.config.sessions.recover_incomplete_sessions:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"session {session_id!r} has no summary and recovery is "
                    "disabled by sessions.recover_incomplete_sessions"
                ),
            )
        summary = state.store.recover(session_id)
        recovered = True
    return {
        "summary": summary.model_dump(mode="json"),
        "recovered_from_incomplete_session": recovered,
    }


@router.post(
    "/sessions/{session_id}/commands",
    summary="Route a manual adaptation command to a task client",
)
async def post_command(
    request: Request,
    session_id: str,
    body: Annotated[CommandRequest, Body()],
) -> CommandResponse:
    """Send one manually-issued adaptation command to the session.

    Rejects a command whose ``session_id`` does not match the path, a
    command that has already expired, and a command with no live target.
    Both the command and every acknowledgement are persisted.
    """
    state = _state(request)
    broker = state.get_broker(session_id)
    if broker is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"session {session_id!r} has no live broker; a command can only "
                "be routed to a connected client"
            ),
        )

    command = body.command
    now = state.clock.utc_now()
    if command.expires_at_utc is not None and now > command.expires_at_utc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"command {command.command_id!r} expired at "
                f"{command.expires_at_utc.isoformat()}; it is now {now.isoformat()}"
            ),
        )

    targets = state.registry.select_command_targets(
        session_id,
        target_role=command.target_role,
        target_client_id=command.target_client_id,
    )
    if not targets:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no live task client on session {session_id!r} matches "
                f"target_role={command.target_role.value!r} "
                f"target_client_id={command.target_client_id!r}"
            ),
        )

    envelope = build_envelope(
        message_type=MessageType.ADAPTATION_COMMAND,
        session_id=session_id,
        source=MessageSource.BACKEND,
        sequence_number=broker.next_server_sequence(),
        payload=command,
        provenance=BACKEND_PROVENANCE,
        sent_at_utc=now,
        sent_at_monotonic_seconds=state.clock.monotonic(),
    )

    from engagevr.protocol.validation import DecodedMessage

    await broker.ingest(
        DecodedMessage(envelope=envelope, payload=command),
        client_id=None,
        client_role="backend",
        transport="in_process",
        same_process=True,
    )

    delivered: list[str] = []
    for target in targets:
        try:
            await target.send(envelope)
            delivered.append(target.client_id)
        except Exception:  # pragma: no cover - peer vanished mid-request
            continue

    if not delivered:
        raise HTTPException(
            status_code=502,
            detail="every matching client became unreachable while sending",
        )

    return CommandResponse(
        command_id=command.command_id,
        session_id=session_id,
        delivered_to=delivered,
        message_id=envelope.message_id,
    )


__all__ = [
    "SCOPE_NOTE",
    "SECURITY_NOTE",
    "CommandRequest",
    "CommandResponse",
    "InvalidSessionIdError",
    "SessionStoreError",
    "router",
]
