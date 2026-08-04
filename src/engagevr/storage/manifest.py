"""Session manifest, ingestion metadata, and session summary schemas.

What a session recording contains
--------------------------------
Protocol envelopes and the ingestion metadata the backend added.  That
is all.

What it never contains
----------------------
Raw webcam frames, encoded video, image data of any kind, MediaPipe
objects, landmark arrays, engagement estimates, cognitive-load
estimates, model predictions, secrets, or any real-world identity.  The
payload models are closed (``extra="forbid"``), so none of these is
even representable on the wire, and the store writes nothing that did
not arrive through that wire format.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, model_validator

from engagevr.protocol.envelope import MessageEnvelope
from engagevr.protocol.messages import DisconnectReason, MessageSource, MessageType
from engagevr.synchronization.ordering import OrderingAnomaly
from engagevr.utils.timestamps import require_utc

#: Bumped whenever the on-disk layout changes incompatibly.
SESSION_FORMAT_VERSION = "1.0"

#: Repeated into every manifest and summary so a recording carries its
#: own scope statement even when read in isolation.
RECORDING_DISCLAIMER = (
    "This recording contains task and transport telemetry only. It "
    "contains no engagement estimate, no cognitive-load estimate, no "
    "psychological or clinical conclusion, no behavioural or "
    "physiological measurement, no image or video data, and no "
    "real-world identity. Simulated content is permanently labelled "
    "SYNTHETIC and must never be presented as participant data."
)


class IngestionMetadata(BaseModel):
    """What the receiver observed about one message, added at ingestion.

    This is kept strictly separate from the envelope: the envelope is
    what the sender said, this is what the receiver saw.  Neither is
    rewritten in terms of the other.
    """

    model_config = {"extra": "forbid"}

    arrival_index: int = Field(
        ge=0,
        description=(
            "Position in receiver arrival order. This ordering is "
            "authoritative for the file and is never re-sorted."
        ),
    )
    server_received_at_utc: datetime = Field(
        description="Receiver's wall clock at ingestion."
    )
    server_monotonic_seconds: float = Field(
        description="Receiver's own monotonic clock at ingestion."
    )
    transport: str = Field(
        description="How the message arrived: websocket | in_process | replay | file."
    )
    client_id: str | None = Field(
        default=None, description="Backend-assigned connection identifier."
    )
    client_role: str | None = None
    anomalies: list[OrderingAnomaly] = Field(
        default_factory=list,
        description="Ordering irregularities detected. Never repaired.",
    )
    anomaly_detail: str | None = None
    expected_sequence_number: int | None = Field(
        default=None,
        description="Sequence number the receiver expected next from this source.",
    )
    apparent_transport_delay_seconds: float | None = Field(
        default=None,
        description=(
            "Populated only when sender and receiver share one clock. "
            "Across processes this is left null rather than reporting a "
            "clock-offset artefact as a delay."
        ),
    )
    delay_unavailable_reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        require_utc(self.server_received_at_utc, "server_received_at_utc")
        return self


class StoredMessage(BaseModel):
    """One line of ``events.jsonl``: the message plus how it arrived."""

    model_config = {"extra": "forbid"}

    envelope: MessageEnvelope
    ingestion: IngestionMetadata


class DropRecord(BaseModel):
    """A non-critical message that backpressure discarded.

    A drop is never silent: it is counted, warned about in the session
    log, and recorded here so the gap in the event stream is explained
    rather than merely absent.
    """

    model_config = {"extra": "forbid"}

    message_id: str
    message_type: MessageType
    source: MessageSource
    sequence_number: int = Field(ge=0)
    dropped_at_utc: datetime
    queue: str = Field(description="Which bounded queue was full.")
    reason: str

    @model_validator(mode="after")
    def _check(self) -> Self:
        require_utc(self.dropped_at_utc, "dropped_at_utc")
        return self


class SessionManifest(BaseModel):
    """``manifest.json`` -- written once when the session directory opens."""

    model_config = {"extra": "forbid"}

    session_format_version: str = SESSION_FORMAT_VERSION
    protocol_version: str
    session_id: str
    created_at_utc: datetime
    engagevr_version: str
    configuration: dict[str, object] = Field(
        default_factory=dict,
        description="Snapshot of the settings this session ran under.",
    )
    disclaimer: str = RECORDING_DISCLAIMER

    @model_validator(mode="after")
    def _check(self) -> Self:
        require_utc(self.created_at_utc, "created_at_utc")
        return self


class SessionSummary(BaseModel):
    """``summary.json`` -- written atomically when the session closes.

    A summary is also derivable from a partially-written ``events.jsonl``
    by :meth:`engagevr.storage.session_store.SessionStore.recover`, in
    which case ``completed`` is False and ``recovered`` is True.
    """

    model_config = {"extra": "forbid"}

    session_format_version: str = SESSION_FORMAT_VERSION
    protocol_version: str
    session_id: str

    event_count: int = Field(ge=0)
    message_type_counts: dict[str, int] = Field(default_factory=dict)
    source_counts: dict[str, int] = Field(default_factory=dict)
    anomaly_counts: dict[str, int] = Field(default_factory=dict)

    dropped_message_count: int = Field(default=0, ge=0)
    dropped_message_types: dict[str, int] = Field(default_factory=dict)

    first_message_sent_at_utc: datetime | None = None
    last_message_sent_at_utc: datetime | None = None
    first_received_at_utc: datetime | None = None
    last_received_at_utc: datetime | None = None

    completed: bool = Field(
        description="True only when a session_end message was recorded."
    )
    disconnect_reason: DisconnectReason | None = None
    recovered: bool = Field(
        default=False,
        description="True when this summary was rebuilt from an interrupted run.",
    )
    malformed_line_numbers: list[int] = Field(
        default_factory=list,
        description="1-based lines that could not be parsed during recovery.",
    )

    synthetic_message_count: int = Field(default=0, ge=0)
    replay_message_count: int = Field(default=0, ge=0)

    disclaimer: str = RECORDING_DISCLAIMER

    @model_validator(mode="after")
    def _check(self) -> Self:
        for name in (
            "first_message_sent_at_utc",
            "last_message_sent_at_utc",
            "first_received_at_utc",
            "last_received_at_utc",
        ):
            value = getattr(self, name)
            if value is not None:
                require_utc(value, name)
        return self
