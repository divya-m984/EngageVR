"""Message types, sources, roles, and typed payload models.

Every message that crosses the WebSocket boundary or is written to a
session recording is one of the payload models defined here, wrapped in
a :class:`~engagevr.protocol.envelope.MessageEnvelope`.

Design constraints
------------------
- Payloads contain JSON scalars, lists, and nested models only.  No
  arbitrary Python object is ever placed in a payload; ``extra`` is
  forbidden on every model so unknown fields are rejected rather than
  silently carried.
- Payloads carry **task and transport telemetry only**.  Behavioural,
  physiological, engagement, and cognitive-load information travels in
  its own schemas and is deliberately not representable here.
- Field names are the wire contract shared with the Unity client and
  must not be renamed without a protocol major-version bump.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, model_validator

from engagevr.schemas.events import TaskEventDetail
from engagevr.utils.timestamps import require_utc


class MessageType(enum.StrEnum):
    """Every message category the protocol defines.

    A message whose ``message_type`` is not a member of this enum is
    rejected with :attr:`ProtocolErrorCode.UNKNOWN_MESSAGE_TYPE`.
    """

    CLIENT_HELLO = "client_hello"
    SERVER_HELLO = "server_hello"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TASK_EVENT = "task_event"
    TASK_STATE = "task_state"
    TELEMETRY = "telemetry"
    ADAPTATION_COMMAND = "adaptation_command"
    ADAPTATION_ACKNOWLEDGEMENT = "adaptation_acknowledgement"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACKNOWLEDGEMENT = "heartbeat_acknowledgement"
    REPLAY_CONTROL = "replay_control"
    ACKNOWLEDGEMENT = "acknowledgement"
    PROTOCOL_ERROR = "protocol_error"


class MessageSource(enum.StrEnum):
    """Which program produced a message.

    ``source`` describes the **originator**.  A replayed message keeps
    its original source and gains separate replay metadata; it is never
    relabelled as though the replay player had produced it.
    """

    PYTHON_SIMULATOR = "python_simulator"
    UNITY_CLIENT = "unity_client"
    BACKEND = "backend"
    REPLAY = "replay"
    TEST_FIXTURE = "test_fixture"


class ClientRole(enum.StrEnum):
    """The role a connected client claims during the handshake.

    - ``simulator`` / ``unity``: task clients.  They emit task events and
      are eligible targets for adaptation commands.
    - ``observer``: read-only.  Receives broadcast telemetry, may not
      emit task events.
    - ``replay``: injects a previously recorded session.
    """

    SIMULATOR = "simulator"
    UNITY = "unity"
    OBSERVER = "observer"
    REPLAY = "replay"


#: Roles that a backend adaptation command may be routed to.
TASK_CLIENT_ROLES: frozenset[ClientRole] = frozenset(
    {ClientRole.SIMULATOR, ClientRole.UNITY}
)


class ProtocolErrorCode(enum.StrEnum):
    """Machine-readable reasons a message was rejected.

    Rejection reasons are preserved verbatim in the session recording so
    that a rejected message can be diagnosed after the fact.
    """

    INVALID_JSON = "invalid_json"
    MESSAGE_TOO_LARGE = "message_too_large"
    UNSUPPORTED_PROTOCOL_VERSION = "unsupported_protocol_version"
    UNKNOWN_MESSAGE_TYPE = "unknown_message_type"
    INVALID_ENVELOPE = "invalid_envelope"
    INVALID_PAYLOAD = "invalid_payload"
    DUPLICATE_MESSAGE_ID = "duplicate_message_id"
    DUPLICATE_SEQUENCE_NUMBER = "duplicate_sequence_number"
    SEQUENCE_REVERSAL = "sequence_reversal"
    SESSION_MISMATCH = "session_mismatch"
    HANDSHAKE_REQUIRED = "handshake_required"
    HANDSHAKE_REJECTED = "handshake_rejected"
    ROLE_NOT_PERMITTED = "role_not_permitted"
    EXPIRED_COMMAND = "expired_command"
    QUEUE_FULL = "queue_full"
    INTERNAL_ERROR = "internal_error"


class TaskState(enum.StrEnum):
    """Coarse lifecycle state of a task client."""

    IDLE = "idle"
    LOADED = "loaded"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


class AdaptationCommandName(enum.StrEnum):
    """The visually harmless commands implemented in Milestone 4.

    Milestone 4 implements **transport only**.  No policy, cooldown,
    hysteresis, or personalization logic derives these commands; every
    command in this milestone is issued manually or by a test script.
    """

    SET_DIFFICULTY = "set_difficulty"
    SET_STIMULUS_INTERVAL = "set_stimulus_interval"
    PAUSE_TASK = "pause_task"
    RESUME_TASK = "resume_task"


class ReplayAction(enum.StrEnum):
    """Control actions understood by a replay-capable receiver."""

    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STEP = "step"
    STOP = "stop"


class DisconnectReason(enum.StrEnum):
    """How a connection ended.

    Kept distinct so that an orderly shutdown is never reported as a
    protocol failure, and a protocol failure is never reported as a
    clean close.
    """

    ORDERLY = "orderly"
    CLIENT_DISCONNECT = "client_disconnect"
    TIMEOUT = "timeout"
    INVALID_PROTOCOL = "invalid_protocol"
    INTERNAL_FAILURE = "internal_failure"
    SERVER_SHUTDOWN = "server_shutdown"


#: JSON-representable scalar types permitted inside open-ended maps.
ProtocolScalar = float | int | bool | str | None


class _Payload(BaseModel):
    """Base for every payload model: strict, closed, JSON-only."""

    model_config = {"extra": "forbid"}


# --------------------------------------------------------------------------
# Handshake
# --------------------------------------------------------------------------


class ClientHelloPayload(_Payload):
    """First message a client sends after the socket opens."""

    role: ClientRole
    client_name: str = Field(min_length=1, max_length=128)
    client_version: str = Field(min_length=1, max_length=64)
    protocol_version: str = Field(
        description="Repeated inside the payload so a rejected handshake "
        "can still report what the client believed it spoke."
    )
    capabilities: list[str] = Field(default_factory=list, max_length=64)


class ServerHelloPayload(_Payload):
    """Backend response to a ``client_hello``."""

    accepted: bool
    server_name: str
    server_version: str
    protocol_version: str
    session_id: str
    assigned_client_id: str
    server_time_utc: datetime
    heartbeat_interval_seconds: float = Field(gt=0.0)
    connection_timeout_seconds: float = Field(gt=0.0)
    maximum_message_bytes: int = Field(gt=0)
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        require_utc(self.server_time_utc, "server_time_utc")
        if not self.accepted and self.rejection_reason is None:
            raise ValueError("a rejected server_hello must carry a rejection_reason")
        return self


# --------------------------------------------------------------------------
# Session lifecycle
# --------------------------------------------------------------------------


class SessionStartPayload(_Payload):
    """Declares the start of a task session.

    ``participant_id`` is a pseudonymous label.  No name, email, or
    other identifying information is representable here.
    """

    participant_id: str = Field(
        min_length=1,
        max_length=128,
        description="Pseudonymous identifier only. Never a real-world identity.",
    )
    task_id: str = Field(min_length=1, max_length=128)
    started_at_utc: datetime
    blocks: int = Field(ge=0)
    trials_per_block: int = Field(ge=0)
    difficulty_level: int = Field(ge=0)
    configuration: dict[str, ProtocolScalar] = Field(
        default_factory=dict,
        description="Flat snapshot of the task settings actually in force.",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        require_utc(self.started_at_utc, "started_at_utc")
        return self


class SessionEndPayload(_Payload):
    """Declares the end of a task session."""

    ended_at_utc: datetime
    completed: bool = Field(
        description="True only when the task ran to its planned end."
    )
    reason: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _check(self) -> Self:
        require_utc(self.ended_at_utc, "ended_at_utc")
        return self


# --------------------------------------------------------------------------
# Task telemetry
# --------------------------------------------------------------------------


class TaskEventPayload(_Payload):
    """One task event.

    The event body is :class:`~engagevr.schemas.events.TaskEventDetail`,
    reused directly rather than restated, so the wire format and the
    stored schema cannot diverge.
    """

    event: TaskEventDetail


class TaskStatePayload(_Payload):
    """Current coarse state of a task client."""

    state: TaskState
    task_id: str | None = None
    block_id: int | None = Field(default=None, ge=0)
    trial_id: int | None = Field(default=None, ge=0)
    difficulty_level: int | None = Field(default=None, ge=0)
    stimulus_interval_ms: float | None = Field(default=None, gt=0.0)


class TelemetryPayload(_Payload):
    """Software/runtime telemetry from a client.

    This carries frame rates, queue depths, dropped-message counts, and
    similar **program** measurements.  It deliberately cannot carry
    behavioural, physiological, engagement, or cognitive-load values:
    those have their own schemas and their own storage path, and mixing
    them into a task telemetry channel would blur the boundary this
    milestone is required to keep.
    """

    component: str = Field(min_length=1, max_length=128)
    metrics: dict[str, ProtocolScalar] = Field(default_factory=dict, max_length=64)


# --------------------------------------------------------------------------
# Adaptation transport
# --------------------------------------------------------------------------


class AdaptationCommandPayload(_Payload):
    """A command routed from the backend to a task client.

    Milestone 4 transports commands; it does not decide them.  ``reason``
    is a free-text audit note supplied by whoever issued the command, not
    a model-derived justification, and no field here asserts that the
    command improves engagement or any other outcome.
    """

    command_id: str = Field(min_length=1, max_length=128)
    command: AdaptationCommandName
    value: float | int | bool | str | None = None
    reason: str = Field(min_length=1, max_length=512)
    issued_at_utc: datetime
    expires_at_utc: datetime | None = None
    target_role: ClientRole = ClientRole.SIMULATOR
    target_client_id: str | None = Field(
        default=None,
        description="When set, only this client receives the command.",
    )
    is_manual: bool = Field(
        default=True,
        description=(
            "Milestone 4 issues manual or scripted commands only. "
            "No automatic policy exists in this milestone."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        require_utc(self.issued_at_utc, "issued_at_utc")
        if self.expires_at_utc is not None:
            require_utc(self.expires_at_utc, "expires_at_utc")
            if self.expires_at_utc <= self.issued_at_utc:
                raise ValueError("expires_at_utc must be after issued_at_utc")
        if self.target_role not in TASK_CLIENT_ROLES:
            allowed = ", ".join(sorted(r.value for r in TASK_CLIENT_ROLES))
            raise ValueError(
                f"target_role must be a task client role ({allowed}), "
                f"got {self.target_role.value!r}"
            )
        if self.command is AdaptationCommandName.SET_DIFFICULTY:
            if not isinstance(self.value, int) or isinstance(self.value, bool):
                raise ValueError("set_difficulty requires an integer value")
            if self.value < 0:
                raise ValueError("set_difficulty value must be non-negative")
        elif self.command is AdaptationCommandName.SET_STIMULUS_INTERVAL:
            if isinstance(self.value, bool) or not isinstance(self.value, int | float):
                raise ValueError("set_stimulus_interval requires a numeric value")
            if self.value <= 0:
                raise ValueError("set_stimulus_interval value must be positive")
        elif self.value is not None:
            raise ValueError(f"{self.command.value} takes no value")
        return self


class AdaptationAcknowledgementPayload(_Payload):
    """A task client's response to an adaptation command."""

    command_id: str = Field(min_length=1, max_length=128)
    accepted: bool
    applied_at_utc: datetime | None = None
    rejection_reason: str | None = Field(default=None, max_length=512)
    duplicate: bool = Field(
        default=False,
        description=(
            "True when this command_id had already been applied and the "
            "repeat was absorbed idempotently."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.applied_at_utc is not None:
            require_utc(self.applied_at_utc, "applied_at_utc")
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("an accepted command must not carry a rejection_reason")
        if not self.accepted:
            if self.rejection_reason is None:
                raise ValueError("a rejected command must carry a rejection_reason")
            if self.applied_at_utc is not None:
                raise ValueError("a rejected command must not carry applied_at_utc")
        return self


# --------------------------------------------------------------------------
# Liveness
# --------------------------------------------------------------------------


class HeartbeatPayload(_Payload):
    """Liveness probe.

    ``client_monotonic_seconds`` is the sender's own monotonic clock.  It
    is echoed back unchanged so the sender can compute a round-trip time
    against a clock it actually owns.
    """

    heartbeat_id: str = Field(min_length=1, max_length=128)
    client_monotonic_seconds: float


class HeartbeatAcknowledgementPayload(_Payload):
    """Response to a heartbeat, carrying both server timestamps.

    The two server timestamps bracket the server's handling of the
    probe.  They are UTC wall-clock values from a *different* machine's
    clock in the general case, so they support an offset **estimate**
    with an explicit RTT bound, never an assertion of synchronization.
    """

    heartbeat_id: str = Field(min_length=1, max_length=128)
    client_monotonic_seconds: float = Field(
        description="Echoed verbatim from the heartbeat. Never rewritten."
    )
    server_received_at_utc: datetime
    server_sent_at_utc: datetime

    @model_validator(mode="after")
    def _check(self) -> Self:
        require_utc(self.server_received_at_utc, "server_received_at_utc")
        require_utc(self.server_sent_at_utc, "server_sent_at_utc")
        return self


# --------------------------------------------------------------------------
# Replay control and generic responses
# --------------------------------------------------------------------------


class ReplayControlPayload(_Payload):
    """Control message for a replay-capable receiver."""

    action: ReplayAction
    source_session_id: str = Field(min_length=1, max_length=128)
    speed: float | None = Field(
        default=None,
        ge=0.0,
        description="0 means immediate (no sleeping). None leaves it unchanged.",
    )


class AcknowledgementPayload(_Payload):
    """Backend confirmation that one message was accepted and handled."""

    acknowledged_message_id: str
    acknowledged_message_type: MessageType
    acknowledged_sequence_number: int = Field(ge=0)
    server_received_at_utc: datetime
    stored: bool = Field(description="True when the message reached the session store.")
    dropped: bool = Field(
        default=False,
        description="True when a non-critical message was dropped under backpressure.",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        require_utc(self.server_received_at_utc, "server_received_at_utc")
        if self.stored and self.dropped:
            raise ValueError("a message cannot be both stored and dropped")
        return self


class ProtocolErrorPayload(_Payload):
    """Typed rejection notice.

    ``detail`` preserves the original rejection reason verbatim so it
    survives into the session recording.
    """

    error_code: ProtocolErrorCode
    detail: str = Field(min_length=1, max_length=2048)
    offending_message_id: str | None = None
    offending_message_type: str | None = Field(
        default=None,
        description="Kept as a raw string: the value may not be a known type.",
    )
    offending_sequence_number: int | None = None
    fatal: bool = Field(
        default=False, description="True when the connection is being closed."
    )


#: The single mapping from message type to payload model.  Validation,
#: JSON Schema generation, and the Unity contract fixtures all derive
#: from this table, so a new message type cannot be half-added.
PAYLOAD_MODELS: dict[MessageType, type[_Payload]] = {
    MessageType.CLIENT_HELLO: ClientHelloPayload,
    MessageType.SERVER_HELLO: ServerHelloPayload,
    MessageType.SESSION_START: SessionStartPayload,
    MessageType.SESSION_END: SessionEndPayload,
    MessageType.TASK_EVENT: TaskEventPayload,
    MessageType.TASK_STATE: TaskStatePayload,
    MessageType.TELEMETRY: TelemetryPayload,
    MessageType.ADAPTATION_COMMAND: AdaptationCommandPayload,
    MessageType.ADAPTATION_ACKNOWLEDGEMENT: AdaptationAcknowledgementPayload,
    MessageType.HEARTBEAT: HeartbeatPayload,
    MessageType.HEARTBEAT_ACKNOWLEDGEMENT: HeartbeatAcknowledgementPayload,
    MessageType.REPLAY_CONTROL: ReplayControlPayload,
    MessageType.ACKNOWLEDGEMENT: AcknowledgementPayload,
    MessageType.PROTOCOL_ERROR: ProtocolErrorPayload,
}


#: Message types that must never be silently dropped under backpressure.
#: ``task_event`` is conditionally critical: see
#: :func:`engagevr.protocol.validation.is_critical`.
CRITICAL_MESSAGE_TYPES: frozenset[MessageType] = frozenset(
    {
        MessageType.SESSION_START,
        MessageType.SESSION_END,
        MessageType.ADAPTATION_COMMAND,
        MessageType.ADAPTATION_ACKNOWLEDGEMENT,
        MessageType.PROTOCOL_ERROR,
    }
)
