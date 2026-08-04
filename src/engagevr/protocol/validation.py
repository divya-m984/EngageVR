"""Decoding and validation of inbound protocol messages.

Nothing in this package trusts an inbound message.  A message becomes a
:class:`DecodedMessage` only after passing, in order:

1. Size limit (bytes on the wire, before parsing).
2. JSON well-formedness.
3. Protocol major-version acceptance.
4. Known ``message_type``.
5. Envelope schema.
6. Payload schema for that specific message type.

Any failure raises :class:`ProtocolValidationError`, which carries a
machine-readable :class:`~engagevr.protocol.messages.ProtocolErrorCode`
and a human-readable detail.  The detail is preserved verbatim into the
``protocol_error`` message and into the session recording, so a
rejection reason is never lost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from engagevr.protocol.envelope import MessageEnvelope
from engagevr.protocol.messages import (
    CRITICAL_MESSAGE_TYPES,
    PAYLOAD_MODELS,
    MessageType,
    ProtocolErrorCode,
    TaskEventPayload,
)
from engagevr.protocol.version import (
    PROTOCOL_VERSION,
    ProtocolVersionError,
    require_supported_version,
)
from engagevr.schemas.events import EventType

#: Default ceiling on a single inbound frame.  Overridden by
#: ``server.maximum_message_bytes`` in configuration.
DEFAULT_MAXIMUM_MESSAGE_BYTES = 262_144


class ProtocolValidationError(Exception):
    """An inbound message was rejected.

    Attributes
    ----------
    error_code:
        Machine-readable reason.
    detail:
        Human-readable explanation, preserved verbatim downstream.
    message_id / message_type / sequence_number:
        Recovered from the raw message when possible, so a rejection can
        still be correlated even though the message was not accepted.
    """

    def __init__(
        self,
        error_code: ProtocolErrorCode,
        detail: str,
        *,
        message_id: str | None = None,
        message_type: str | None = None,
        sequence_number: int | None = None,
        fatal: bool = False,
    ) -> None:
        super().__init__(f"{error_code.value}: {detail}")
        self.error_code = error_code
        self.detail = detail
        self.message_id = message_id
        self.message_type = message_type
        self.sequence_number = sequence_number
        self.fatal = fatal


@dataclass(frozen=True, slots=True)
class DecodedMessage:
    """A fully validated message: envelope plus its typed payload."""

    envelope: MessageEnvelope
    payload: BaseModel

    @property
    def message_type(self) -> MessageType:
        return self.envelope.message_type

    @property
    def message_id(self) -> str:
        return self.envelope.message_id


def _string_or_none(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    return value if isinstance(value, str) else None


def _int_or_none(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def decode_json(
    data: str | bytes,
    *,
    maximum_message_bytes: int = DEFAULT_MAXIMUM_MESSAGE_BYTES,
) -> dict[str, Any]:
    """Parse a wire frame into a JSON object, enforcing the size limit.

    Raises
    ------
    ProtocolValidationError
        With ``MESSAGE_TOO_LARGE`` or ``INVALID_JSON``.
    """
    payload_bytes = data.encode("utf-8") if isinstance(data, str) else data
    size = len(payload_bytes)
    if size > maximum_message_bytes:
        raise ProtocolValidationError(
            ProtocolErrorCode.MESSAGE_TOO_LARGE,
            f"message of {size} bytes exceeds the limit of "
            f"{maximum_message_bytes} bytes",
            fatal=True,
        )
    try:
        parsed = json.loads(payload_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProtocolValidationError(
            ProtocolErrorCode.INVALID_JSON,
            f"frame is not valid JSON: {exc}",
        ) from exc
    if not isinstance(parsed, dict):
        raise ProtocolValidationError(
            ProtocolErrorCode.INVALID_ENVELOPE,
            f"top-level JSON value must be an object, got {type(parsed).__name__}",
        )
    return parsed


def decode_envelope(raw: dict[str, Any]) -> MessageEnvelope:
    """Validate the envelope of an already-parsed JSON object.

    The version and message-type checks run *before* full envelope
    validation so that an unsupported version is reported as an
    unsupported version rather than as a generic schema failure.
    """
    message_id = _string_or_none(raw, "message_id")
    raw_type = _string_or_none(raw, "message_type")
    sequence_number = _int_or_none(raw, "sequence_number")

    version = raw.get("protocol_version")
    if not isinstance(version, str):
        raise ProtocolValidationError(
            ProtocolErrorCode.INVALID_ENVELOPE,
            "protocol_version is missing or is not a string",
            message_id=message_id,
            message_type=raw_type,
            sequence_number=sequence_number,
        )
    try:
        require_supported_version(version)
    except ProtocolVersionError as exc:
        raise ProtocolValidationError(
            ProtocolErrorCode.UNSUPPORTED_PROTOCOL_VERSION,
            f"{exc} (this build speaks {PROTOCOL_VERSION})",
            message_id=message_id,
            message_type=raw_type,
            sequence_number=sequence_number,
            fatal=True,
        ) from exc

    if raw_type is None:
        raise ProtocolValidationError(
            ProtocolErrorCode.INVALID_ENVELOPE,
            "message_type is missing or is not a string",
            message_id=message_id,
            sequence_number=sequence_number,
        )
    if raw_type not in MessageType.__members__.values():
        known = ", ".join(sorted(t.value for t in MessageType))
        raise ProtocolValidationError(
            ProtocolErrorCode.UNKNOWN_MESSAGE_TYPE,
            f"unknown message_type {raw_type!r}; known types: {known}",
            message_id=message_id,
            message_type=raw_type,
            sequence_number=sequence_number,
        )

    try:
        return MessageEnvelope.model_validate(raw)
    except ValidationError as exc:
        raise ProtocolValidationError(
            ProtocolErrorCode.INVALID_ENVELOPE,
            f"envelope failed validation: {_summarize(exc)}",
            message_id=message_id,
            message_type=raw_type,
            sequence_number=sequence_number,
        ) from exc


def decode_payload(envelope: MessageEnvelope) -> BaseModel:
    """Validate an envelope's payload against its registered model."""
    model = PAYLOAD_MODELS[envelope.message_type]
    try:
        return model.model_validate(envelope.payload)
    except ValidationError as exc:
        raise ProtocolValidationError(
            ProtocolErrorCode.INVALID_PAYLOAD,
            f"payload for {envelope.message_type.value!r} failed validation: "
            f"{_summarize(exc)}",
            message_id=envelope.message_id,
            message_type=envelope.message_type.value,
            sequence_number=envelope.sequence_number,
        ) from exc


def decode_message(
    data: str | bytes,
    *,
    maximum_message_bytes: int = DEFAULT_MAXIMUM_MESSAGE_BYTES,
) -> DecodedMessage:
    """Run the full inbound pipeline on one wire frame."""
    raw = decode_json(data, maximum_message_bytes=maximum_message_bytes)
    envelope = decode_envelope(raw)
    payload = decode_payload(envelope)
    return DecodedMessage(envelope=envelope, payload=payload)


def decode_stored_message(raw: dict[str, Any]) -> DecodedMessage:
    """Validate a message read back from a recording.

    Same schema checks as the live path, minus the wire size limit,
    which does not apply to a file already on disk.
    """
    envelope = decode_envelope(raw)
    payload = decode_payload(envelope)
    return DecodedMessage(envelope=envelope, payload=payload)


def is_critical(envelope: MessageEnvelope, payload: BaseModel | None = None) -> bool:
    """Whether this message may never be silently dropped.

    Always critical: ``session_start``, ``session_end``,
    ``adaptation_command``, ``adaptation_acknowledgement``,
    ``protocol_error``.

    Conditionally critical: a ``task_event`` whose event is
    ``task_completed``.  Losing the completion marker would make a
    truncated recording indistinguishable from a finished one.
    """
    if envelope.message_type in CRITICAL_MESSAGE_TYPES:
        return True
    if envelope.message_type is MessageType.TASK_EVENT:
        if isinstance(payload, TaskEventPayload):
            return payload.event.event_type is EventType.TASK_COMPLETED
        event = envelope.payload.get("event")
        if isinstance(event, dict):
            return event.get("event_type") == EventType.TASK_COMPLETED.value
    return False


def _summarize(exc: ValidationError, limit: int = 5) -> str:
    """Render a Pydantic error compactly and deterministically."""
    parts: list[str] = []
    for error in exc.errors()[:limit]:
        location = ".".join(str(item) for item in error["loc"]) or "<root>"
        parts.append(f"{location}: {error['msg']}")
    remaining = len(exc.errors()) - limit
    if remaining > 0:
        parts.append(f"(+{remaining} more)")
    return "; ".join(parts)
