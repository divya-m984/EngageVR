"""Protocol data contracts, re-exported into the schemas namespace.

The models themselves live in :mod:`engagevr.protocol`, next to the
version rules and validation pipeline that give them meaning.  This
module exists so that ``engagevr.schemas`` remains the single place to
look for every data contract in the project, without creating a second,
divergent definition of any model.

Nothing is redefined here.  Every name below is the same object as the
one exported by :mod:`engagevr.protocol`.
"""

from __future__ import annotations

from engagevr.protocol.envelope import (
    REPLAY_LABEL,
    SYNTHETIC_LABEL,
    MessageEnvelope,
    MessageProvenance,
    ReplayMetadata,
)
from engagevr.protocol.messages import (
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
from engagevr.protocol.version import ACCEPTED_MAJOR_VERSIONS, PROTOCOL_VERSION

__all__ = [
    "ACCEPTED_MAJOR_VERSIONS",
    "PROTOCOL_VERSION",
    "REPLAY_LABEL",
    "SYNTHETIC_LABEL",
    "AcknowledgementPayload",
    "AdaptationAcknowledgementPayload",
    "AdaptationCommandName",
    "AdaptationCommandPayload",
    "ClientHelloPayload",
    "ClientRole",
    "DisconnectReason",
    "HeartbeatAcknowledgementPayload",
    "HeartbeatPayload",
    "MessageEnvelope",
    "MessageProvenance",
    "MessageSource",
    "MessageType",
    "ProtocolErrorCode",
    "ProtocolErrorPayload",
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
]
