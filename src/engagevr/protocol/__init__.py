"""The EngageVR versioned real-time protocol.

One protocol definition is shared by the FastAPI backend, the Python
task simulator, the session replay player, and the Unity desktop client.
There is no second, client-specific message format.

Scope
-----
This protocol carries task telemetry, session lifecycle, adaptation
**command transport**, and connection diagnostics.  It carries no
engagement estimate, no cognitive-load estimate, no behavioural or
physiological measurement, and no image data.  Those are separate
schemas with separate storage, and the separation is enforced by the
payload models being closed (``extra="forbid"``).
"""

from engagevr.protocol.envelope import (
    REPLAY_LABEL,
    SYNTHETIC_LABEL,
    MessageEnvelope,
    MessageProvenance,
    ReplayMetadata,
    build_envelope,
    new_message_id,
)
from engagevr.protocol.json_schema import (
    SCHEMA_RELATIVE_PATH,
    build_protocol_json_schema,
    render_protocol_json_schema,
    write_protocol_json_schema,
)
from engagevr.protocol.messages import (
    CRITICAL_MESSAGE_TYPES,
    PAYLOAD_MODELS,
    TASK_CLIENT_ROLES,
    AcknowledgementPayload,
    AdaptationAcknowledgementPayload,
    AdaptationCommandName,
    AdaptationCommandPayload,
    ClientHelloPayload,
    ClientRole,
    DisconnectReason,
    HeartbeatAcknowledgementPayload,
    HeartbeatPayload,
    MessageSource,
    MessageType,
    ProtocolErrorCode,
    ProtocolErrorPayload,
    ReplayAction,
    ReplayControlPayload,
    ServerHelloPayload,
    SessionEndPayload,
    SessionStartPayload,
    TaskEventPayload,
    TaskState,
    TaskStatePayload,
    TelemetryPayload,
)
from engagevr.protocol.validation import (
    DEFAULT_MAXIMUM_MESSAGE_BYTES,
    DecodedMessage,
    ProtocolValidationError,
    decode_envelope,
    decode_json,
    decode_message,
    decode_payload,
    decode_stored_message,
    is_critical,
)
from engagevr.protocol.version import (
    ACCEPTED_MAJOR_VERSIONS,
    PROTOCOL_VERSION,
    ProtocolVersionError,
    is_supported_version,
    parse_protocol_version,
    require_supported_version,
)

__all__ = [
    "ACCEPTED_MAJOR_VERSIONS",
    "CRITICAL_MESSAGE_TYPES",
    "DEFAULT_MAXIMUM_MESSAGE_BYTES",
    "PAYLOAD_MODELS",
    "PROTOCOL_VERSION",
    "REPLAY_LABEL",
    "SCHEMA_RELATIVE_PATH",
    "SYNTHETIC_LABEL",
    "TASK_CLIENT_ROLES",
    "AcknowledgementPayload",
    "AdaptationAcknowledgementPayload",
    "AdaptationCommandName",
    "AdaptationCommandPayload",
    "ClientHelloPayload",
    "ClientRole",
    "DecodedMessage",
    "DisconnectReason",
    "HeartbeatAcknowledgementPayload",
    "HeartbeatPayload",
    "MessageEnvelope",
    "MessageProvenance",
    "MessageSource",
    "MessageType",
    "ProtocolErrorCode",
    "ProtocolErrorPayload",
    "ProtocolValidationError",
    "ProtocolVersionError",
    "ReplayAction",
    "ReplayControlPayload",
    "ReplayMetadata",
    "ServerHelloPayload",
    "SessionEndPayload",
    "SessionStartPayload",
    "TaskEventPayload",
    "TaskState",
    "TaskStatePayload",
    "TelemetryPayload",
    "build_envelope",
    "build_protocol_json_schema",
    "decode_envelope",
    "decode_json",
    "decode_message",
    "decode_payload",
    "decode_stored_message",
    "is_critical",
    "is_supported_version",
    "new_message_id",
    "parse_protocol_version",
    "render_protocol_json_schema",
    "require_supported_version",
    "write_protocol_json_schema",
]
