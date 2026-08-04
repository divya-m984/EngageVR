"""Typed error construction and HTTP error handling.

Every rejection the backend issues is a typed ``protocol_error`` message
on the WebSocket, or a JSON body with the same ``error_code`` vocabulary
over HTTP.  There is no path that returns an untyped or unexplained
failure.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from engagevr.protocol.envelope import (
    MessageEnvelope,
    MessageProvenance,
    build_envelope,
)
from engagevr.protocol.messages import (
    MessageSource,
    MessageType,
    ProtocolErrorCode,
    ProtocolErrorPayload,
)
from engagevr.protocol.validation import ProtocolValidationError
from engagevr.schemas.session import DataSource
from engagevr.storage.session_store import InvalidSessionIdError, SessionStoreError

#: The backend is a live participant, not a simulator: its own messages
#: are not synthetic content.
BACKEND_PROVENANCE = MessageProvenance(
    data_source=DataSource.LIVE,
    synthetic_label=None,
    producer="engagevr.api",
)


def build_protocol_error(
    *,
    session_id: str,
    sequence_number: int,
    error_code: ProtocolErrorCode,
    detail: str,
    offending_message_id: str | None = None,
    offending_message_type: str | None = None,
    offending_sequence_number: int | None = None,
    fatal: bool = False,
    sent_at_utc: datetime | None = None,
    sent_at_monotonic_seconds: float | None = None,
) -> MessageEnvelope:
    """Build a ``protocol_error`` envelope from the backend."""
    payload = ProtocolErrorPayload(
        error_code=error_code,
        detail=detail,
        offending_message_id=offending_message_id,
        offending_message_type=offending_message_type,
        offending_sequence_number=offending_sequence_number,
        fatal=fatal,
    )
    return build_envelope(
        message_type=MessageType.PROTOCOL_ERROR,
        session_id=session_id,
        source=MessageSource.BACKEND,
        sequence_number=sequence_number,
        payload=payload,
        provenance=BACKEND_PROVENANCE,
        correlation_id=offending_message_id,
        sent_at_utc=sent_at_utc,
        sent_at_monotonic_seconds=sent_at_monotonic_seconds,
    )


def protocol_error_from_validation(
    exc: ProtocolValidationError,
    *,
    session_id: str,
    sequence_number: int,
    sent_at_utc: datetime | None = None,
    sent_at_monotonic_seconds: float | None = None,
) -> MessageEnvelope:
    """Turn a validation failure into a wire ``protocol_error``.

    The rejection reason is carried through verbatim; nothing is
    summarized away, because the reason is the whole diagnostic value.
    """
    return build_protocol_error(
        session_id=session_id,
        sequence_number=sequence_number,
        error_code=exc.error_code,
        detail=exc.detail,
        offending_message_id=exc.message_id,
        offending_message_type=exc.message_type,
        offending_sequence_number=exc.sequence_number,
        fatal=exc.fatal,
        sent_at_utc=sent_at_utc,
        sent_at_monotonic_seconds=sent_at_monotonic_seconds,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install JSON error handlers for the storage-layer exceptions."""

    @app.exception_handler(InvalidSessionIdError)
    async def _invalid_session_id(
        _request: Request, exc: InvalidSessionIdError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error_code": "invalid_session_id",
                "detail": str(exc),
            },
        )

    @app.exception_handler(SessionStoreError)
    async def _session_store_error(
        _request: Request, exc: SessionStoreError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error_code": "session_unavailable",
                "detail": str(exc),
            },
        )
