"""Typed presentation models for the dashboard's live and replay modes.

These are deliberately **separate** from the Milestone 5-8 run models in
:mod:`engagevr.schemas.dashboard`.  A recorded Milestone 4 session and an
experiment run are different objects with different provenance
contracts, and giving them one shared type would be the first step to
presenting a task recording as an experiment result.

Three properties are structural rather than procedural.

A session recording declares no scientific eligibility.
    The Milestone 4 session format has no ``scientific_evaluation_eligible``
    field, so :class:`DashboardSessionProvenance` presents every session as
    ineligible and states why.  ``data_source = live`` does not change
    that: *live* describes where bytes came from, not whether a study was
    designed, labelled, or validated.

A mode is a value, not a heading.
    :class:`DashboardSessionMode` travels with every view and report.  A
    live view cannot be rendered from a replay context and a replay view
    cannot be rendered from a live one, because the page checks the value
    rather than trusting the title it printed.

A replay position cannot leave the recording.
    :class:`DashboardReplayState` is frozen and its navigation methods
    return new instances clamped to the recorded range, so stepping past
    the end is not an error state to handle on a page — it is not
    representable.
"""

from __future__ import annotations

import enum
from typing import Self

from pydantic import BaseModel, Field, model_validator

from engagevr.schemas.dashboard import (
    DASHBOARD_DISCLAIMER,
    SYNTHETIC_BANNER,
    DashboardError,
    DashboardWarning,
)

#: Bumped whenever the exported session report's shape changes.
SESSION_REPORT_SCHEMA_VERSION = "1.0"

#: Why a session recording is never presented as scientifically eligible.
SESSION_ELIGIBILITY_NOTE = (
    "The session recording format records no scientific-eligibility "
    "declaration, so this session is presented as ineligible. A live data "
    "source is not automatically eligible: 'live' says where the bytes came "
    "from, not that a study was designed, labelled, approved, or validated."
)

#: Stated on every live view, where a moving number is most persuasive.
LIVE_OBSERVATION_NOTE = (
    "MODE: LIVE OBSERVATION. This view re-reads records that a session "
    "recorder already persisted. It runs no model, opens no camera, "
    "produces no estimate, sends nothing, and changes nothing in the "
    "recording it is reading."
)

#: Stated on every replay view.
REPLAY_PRESENTATION_NOTE = (
    "MODE: REPLAY. This is a read-only presentation of records that were "
    "persisted earlier. Nothing is re-emitted, re-simulated, re-inferred, "
    "or repaired, and a record that was SYNTHETIC when written stays "
    "SYNTHETIC here."
)

#: Stated wherever a session carries no measurement of a given kind.
SESSION_CONTENT_NOTE = (
    "A session recording carries task and transport telemetry only. It "
    "contains no engagement estimate, no cognitive-load estimate, no "
    "behavioural or physiological measurement, and no image data, because "
    "the protocol payloads cannot represent them. Their absence here is a "
    "property of the format, not a failed measurement."
)


class DashboardSessionMode(enum.StrEnum):
    """Which evidence source a view is rendering.

    Kept as three distinct values rather than one ``is_live`` flag so
    that a page can assert the mode it was written for, and so a report
    records which mode produced it.
    """

    #: The Milestone 5-8 experiment-artifact observatory.
    ARTIFACT = "artifact"
    #: Read-only observation of a session recording as it is written.
    LIVE = "live"
    #: Read-only navigation through an already-recorded session.
    REPLAY = "replay"


class DashboardSessionStatus(enum.StrEnum):
    """What the session catalogue could establish about a recording.

    ``ACTIVE_OR_INCOMPLETE`` is one state on purpose.  From the outside,
    a session still being written and a session that was interrupted look
    identical: neither has a ``summary.json``.  Reporting that honestly is
    better than guessing, and neither is a failure.
    """

    #: ``summary.json`` records a session that ran to its planned end.
    COMPLETED = "completed"
    #: ``summary.json`` exists but records ``completed=false``.
    INTERRUPTED = "interrupted"
    #: No summary yet: the session may still be running, or it stopped.
    ACTIVE_OR_INCOMPLETE = "active_or_incomplete"
    #: The summary explicitly records an internal or protocol failure.
    FAILED = "failed"
    #: The manifest is absent or unreadable.
    UNREADABLE = "unreadable"
    #: The event stream is absent.
    STREAM_UNAVAILABLE = "stream_unavailable"


class SessionRecordProblem(enum.StrEnum):
    """Why one line of a recording could not be presented as a message.

    These are kept apart from the experiment catalogue's ``corrupt``
    status because their consequences differ.  A torn final line while a
    writer is active is transient and expected; a malformed interior line
    is permanent and needs looking at.
    """

    #: The line is not JSON, or not a JSON object.
    MALFORMED_JSON = "malformed_json"
    #: The line parses but is not a stored message.
    INVALID_STRUCTURE = "invalid_structure"
    #: The envelope or payload is invalid under the protocol.
    PROTOCOL_INVALID = "protocol_invalid"
    #: The ingestion metadata is invalid.
    INGESTION_INVALID = "ingestion_invalid"


class DashboardSessionProvenance(BaseModel):
    """Where a presented session's records came from, and what they are not.

    Synthetic and replayed message counts are stored rather than reduced
    to one flag, because a recording can hold a mixture and the reader
    needs the composition, not a verdict.
    """

    model_config = {"extra": "forbid", "frozen": True}

    #: Identifier the recording's own manifest declares.
    session_id: str = Field(min_length=1)
    #: Directory the recording actually lives in. Kept separate from
    #: ``session_id`` because the two can disagree — a recording copied
    #: for comparison keeps its recorded id under a new folder name — and
    #: because the directory is what a selector must address. The
    #: recorded id is provenance; the directory name never is.
    directory_name: str = Field(min_length=1)
    session_directory: str = Field(min_length=1)
    mode: DashboardSessionMode

    session_format_version: str | None = None
    protocol_version: str | None = None
    engagevr_version: str | None = None
    created_at_utc: str | None = None

    #: ``data_source`` values actually recorded on the messages read so
    #: far, sorted. Empty when no record has been read.
    data_sources: tuple[str, ...] = ()
    synthetic_message_count: int = Field(default=0, ge=0)
    non_synthetic_message_count: int = Field(default=0, ge=0)
    replayed_message_count: int = Field(default=0, ge=0)

    scientific_evaluation_eligible: bool = False
    eligibility_reason: str = SESSION_ELIGIBILITY_NOTE

    status: DashboardSessionStatus
    status_reason: str | None = None
    warnings: tuple[DashboardWarning, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.mode is DashboardSessionMode.ARTIFACT:
            raise DashboardError(
                "a session provenance cannot claim the artifact mode; an "
                "experiment run and a session recording are different objects "
                "with different provenance contracts"
            )
        if self.scientific_evaluation_eligible:
            raise DashboardError(
                f"session {self.session_id!r} was marked scientifically "
                "eligible. " + SESSION_ELIGIBILITY_NOTE
            )
        if (
            self.status
            in (
                DashboardSessionStatus.UNREADABLE,
                DashboardSessionStatus.STREAM_UNAVAILABLE,
                DashboardSessionStatus.FAILED,
            )
            and not self.status_reason
        ):
            raise DashboardError(
                f"session {self.session_id!r} has status "
                f"{self.status.value!r} and must state why"
            )
        return self

    @property
    def is_synthetic(self) -> bool:
        """Whether any record read so far was permanently labelled synthetic."""
        return self.synthetic_message_count > 0

    @property
    def identifier_matches_directory(self) -> bool:
        """Whether the recorded id and the directory name agree."""
        return self.session_id == self.directory_name

    @property
    def provenance_established(self) -> bool:
        """Whether any record has been read at all.

        With no record read, nothing is known about provenance.  That is
        not the same as *not synthetic*, and the page says so.
        """
        return bool(self.data_sources)

    @property
    def banners(self) -> tuple[str, ...]:
        """Every banner this provenance obliges a view to render."""
        if self.is_synthetic:
            return (SYNTHETIC_BANNER, DASHBOARD_DISCLAIMER)
        return (DASHBOARD_DISCLAIMER,)


class DashboardSessionRecord(BaseModel):
    """One line of a recording, on its way to a page.

    A line that could not be decoded is still a record here.  It keeps
    its line number and states its problem, because a malformed interior
    line that vanished from the view would be a gap the reader never
    learned about.
    """

    model_config = {"extra": "forbid", "frozen": True}

    line_number: int = Field(ge=1)
    problem: SessionRecordProblem | None = None
    problem_detail: str | None = None

    arrival_index: int | None = Field(default=None, ge=0)
    sequence_number: int | None = Field(default=None, ge=0)
    message_type: str | None = None
    source: str | None = None
    message_id: str | None = None
    correlation_id: str | None = None

    sent_at_utc: str | None = None
    server_received_at_utc: str | None = None
    transport: str | None = None

    data_source: str | None = None
    synthetic_label: str | None = None
    producer: str | None = None
    replay_label: str | None = None
    replay_source_session_id: str | None = None

    #: Anomalies the receiver recorded on this message. Never repaired.
    recorded_anomalies: tuple[str, ...] = ()
    anomaly_detail: str | None = None
    expected_sequence_number: int | None = Field(default=None, ge=0)

    #: Flat, already-stringified payload fields chosen for display.
    payload_summary: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.problem is not None and not self.problem_detail:
            raise DashboardError(
                f"line {self.line_number} has problem {self.problem.value!r} "
                "and must state the detail; 'corrupt' is not an actionable "
                "message"
            )
        if self.problem is None and self.message_type is None:
            raise DashboardError(
                f"line {self.line_number} decoded without a problem but "
                "carries no message type"
            )
        return self

    @property
    def decoded(self) -> bool:
        """Whether this line decoded into a protocol message."""
        return self.problem is None

    @property
    def is_synthetic(self) -> bool:
        """Whether this record carries the permanent synthetic label."""
        return self.synthetic_label is not None

    @property
    def is_replayed(self) -> bool:
        """Whether this record was already a replay when it was written."""
        return self.replay_label is not None


class SessionSequenceObservation(BaseModel):
    """A sequence irregularity visible in the recorded numbers.

    Derived from ``sequence_number`` values that were actually recorded.
    Nothing is reordered, renumbered, filled in, or dropped; this states
    what the recording shows and stops there.
    """

    model_config = {"extra": "forbid", "frozen": True}

    source: str = Field(min_length=1)
    line_number: int = Field(ge=1)
    kind: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class SessionAdaptationCounts(BaseModel):
    """Adaptation messages present in one recording.

    Deliberately **not**
    :class:`~engagevr.schemas.dashboard.AdaptationLifecycleCounts`.  That
    model describes a Milestone 8 policy run, whose proposals and built
    commands are recorded artifacts.  A session recording contains
    neither: it shows commands that crossed the wire and whatever the
    task client said back.  Reusing the policy model here would invent a
    policy proposal behind every manually issued command.
    """

    model_config = {"extra": "forbid", "frozen": True}

    commands_recorded: int = Field(default=0, ge=0)
    acknowledgements_recorded: int = Field(default=0, ge=0)
    accepted_recorded: int = Field(default=0, ge=0)
    rejected_recorded: int = Field(default=0, ge=0)
    applied_timestamp_recorded: int = Field(default=0, ge=0)
    note: str = (
        "A session recording carries transported commands and the replies "
        "to them. It records no policy proposal and no built-but-unsent "
        "command, so those are not shown here rather than being inferred."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.acknowledgements_recorded > self.commands_recorded:
            raise DashboardError(
                f"{self.acknowledgements_recorded} acknowledgements were "
                f"recorded but only {self.commands_recorded} commands were. An "
                "acknowledgement requires a command that reached a client."
            )
        replies = self.accepted_recorded + self.rejected_recorded
        if replies > self.acknowledgements_recorded:
            raise DashboardError(
                f"{replies} accept/reject outcomes exceed "
                f"{self.acknowledgements_recorded} acknowledgements"
            )
        if self.applied_timestamp_recorded > self.accepted_recorded:
            raise DashboardError(
                "an applied timestamp requires an accepted acknowledgement"
            )
        return self


class DashboardSessionSummary(BaseModel):
    """What the catalogue and the pages know about one recording.

    ``partial_trailing_line`` is a first-class field.  A final line
    without its terminating newline is the normal appearance of a file
    being written right now, and calling that corruption would make every
    live session look broken.
    """

    model_config = {"extra": "forbid", "frozen": True}

    provenance: DashboardSessionProvenance

    #: Complete, newline-terminated lines present in ``events.jsonl``.
    complete_record_count: int = Field(default=0, ge=0)
    #: Complete lines this pass skipped, having presented them already.
    parse_start_line: int = Field(default=0, ge=0)
    #: Complete lines this pass actually decoded or tried to.
    parsed_record_count: int = Field(default=0, ge=0)
    decoded_record_count: int = Field(default=0, ge=0)
    malformed_record_count: int = Field(default=0, ge=0)
    malformed_line_numbers: tuple[int, ...] = ()

    partial_trailing_line: bool = False
    partial_trailing_note: str | None = None

    message_type_counts: tuple[tuple[str, int], ...] = ()
    source_counts: tuple[tuple[str, int], ...] = ()
    recorded_anomaly_counts: tuple[tuple[str, int], ...] = ()
    sequence_observations: tuple[SessionSequenceObservation, ...] = ()

    dropped_message_count: int | None = Field(default=None, ge=0)
    first_sent_at_utc: str | None = None
    last_sent_at_utc: str | None = None
    first_received_at_utc: str | None = None
    last_received_at_utc: str | None = None

    session_end_recorded: bool = False
    disconnect_reason: str | None = None
    summary_recovered: bool | None = None

    task_state: str | None = None
    current_difficulty_level: int | None = Field(default=None, ge=0)
    adaptation: SessionAdaptationCounts = Field(default_factory=SessionAdaptationCounts)

    #: Statements about what this format cannot carry, so an absent
    #: quantity is explained rather than merely missing.
    unavailable_statements: tuple[str, ...] = ()
    warnings: tuple[DashboardWarning, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        total = self.decoded_record_count + self.malformed_record_count
        if total != self.parsed_record_count:
            raise DashboardError(
                f"{self.decoded_record_count} decoded + "
                f"{self.malformed_record_count} malformed = {total}, but "
                f"{self.parsed_record_count} records were parsed. Every "
                "parsed line is exactly one of the two."
            )
        consumed = self.parse_start_line + self.parsed_record_count
        if consumed > self.complete_record_count:
            raise DashboardError(
                f"this pass accounts for {consumed} records but only "
                f"{self.complete_record_count} complete records exist"
            )
        if len(self.malformed_line_numbers) != self.malformed_record_count:
            raise DashboardError(
                f"{self.malformed_record_count} malformed records were "
                f"counted but {len(self.malformed_line_numbers)} line numbers "
                "were listed"
            )
        if self.partial_trailing_line and not self.partial_trailing_note:
            raise DashboardError(
                "a partial trailing line must be explained; it is a transient "
                "state, not corruption, and must not read like one"
            )
        return self

    @property
    def fully_parsed(self) -> bool:
        """Whether this pass decoded every complete record in the file."""
        return (
            self.parse_start_line == 0
            and self.parsed_record_count == self.complete_record_count
        )

    @property
    def unparsed_record_count(self) -> int:
        """Complete records this pass did not present."""
        return self.complete_record_count - (
            self.parse_start_line + self.parsed_record_count
        )


class DashboardSessionCatalogue(BaseModel):
    """Every recording under one session root.

    Kept apart from :class:`~engagevr.schemas.dashboard.DashboardCatalogue`.
    A recorded session is not an experiment run and is never listed as
    one.
    """

    model_config = {"extra": "forbid", "frozen": True}

    session_root: str = Field(min_length=1)
    root_exists: bool
    sessions: tuple[DashboardSessionSummary, ...] = ()
    warnings: tuple[DashboardWarning, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Whether the root holds no readable session directory."""
        return not self.sessions

    def session_ids(self) -> tuple[str, ...]:
        """Recorded identifiers of every listed session, in catalogue order."""
        return tuple(s.provenance.session_id for s in self.sessions)

    def directory_names(self) -> tuple[str, ...]:
        """Directory names of every listed session, in catalogue order.

        These, not the recorded identifiers, are what addresses a
        recording: two directories can legitimately hold recordings that
        declare the same ``session_id``.
        """
        return tuple(s.provenance.directory_name for s in self.sessions)

    def find(self, directory_name: str) -> DashboardSessionSummary | None:
        """The listed session in this directory, or ``None``."""
        for session in self.sessions:
            if session.provenance.directory_name == directory_name:
                return session
        return None


class DashboardReplayState(BaseModel):
    """A cursor into an already-parsed recording.

    Frozen, with navigation methods returning new instances clamped to
    the recorded range.  Stepping past either end is therefore not an
    error a page has to handle; it simply stays where it is.
    """

    model_config = {"extra": "forbid", "frozen": True}

    total: int = Field(ge=0)
    position: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.total == 0:
            if self.position != 0:
                raise DashboardError("an empty recording has no position other than 0")
            return self
        if self.position >= self.total:
            raise DashboardError(
                f"replay position {self.position} is outside a recording of "
                f"{self.total} record(s)"
            )
        return self

    @property
    def is_empty(self) -> bool:
        """Whether there is anything to navigate."""
        return self.total == 0

    @property
    def at_first(self) -> bool:
        """Whether the cursor is on the first recorded position."""
        return self.position == 0

    @property
    def at_last(self) -> bool:
        """Whether the cursor is on the last recorded position."""
        return self.total == 0 or self.position == self.total - 1

    @property
    def human_position(self) -> str:
        """One-based position, for display."""
        if self.total == 0:
            return "0 of 0"
        return f"{self.position + 1} of {self.total}"

    def _moved(self, position: int) -> DashboardReplayState:
        if self.total == 0:
            return self
        clamped = max(0, min(position, self.total - 1))
        return self.model_copy(update={"position": clamped})

    def first(self) -> DashboardReplayState:
        """Jump to the first recorded message."""
        return self._moved(0)

    def last(self) -> DashboardReplayState:
        """Jump to the last recorded message."""
        return self._moved(self.total - 1)

    def step_forward(self, steps: int = 1) -> DashboardReplayState:
        """Advance, stopping at the last recorded message."""
        return self._moved(self.position + steps)

    def step_backward(self, steps: int = 1) -> DashboardReplayState:
        """Go back, stopping at the first recorded message."""
        return self._moved(self.position - steps)

    def resized(self, total: int) -> DashboardReplayState:
        """Rebuild for a recording of a different length, keeping the place."""
        if total <= 0:
            return DashboardReplayState(total=0, position=0)
        return DashboardReplayState(
            total=total, position=max(0, min(self.position, total - 1))
        )


class DashboardSessionReport(BaseModel):
    """A deterministic, exportable report about one recording.

    A presentation artifact, not an experiment result.  It restates what
    a recording contains, with the provenance that recording carried, and
    it cannot be produced without that provenance: the fields are
    required and the validator refuses a synthetic report that omits the
    software-self-check banner.

    ``exported_at_utc`` is explicitly outside the report's identity.  The
    fingerprint is computed over the content, so the same recording
    inspected twice yields the same fingerprint no matter when.
    """

    model_config = {"extra": "forbid", "frozen": True}

    report_schema_version: str = SESSION_REPORT_SCHEMA_VERSION
    generated_by: str = "engagevr.dashboard.session_report"

    source_mode: DashboardSessionMode
    session_id: str = Field(min_length=1)
    session_directory: str = Field(min_length=1)
    session_format_version: str | None = None
    protocol_version: str | None = None
    engagevr_version: str | None = None

    data_sources: tuple[str, ...] = ()
    is_synthetic: bool
    scientific_evaluation_eligible: bool = False
    eligibility_reason: str = SESSION_ELIGIBILITY_NOTE
    synthetic_message_count: int = Field(default=0, ge=0)
    non_synthetic_message_count: int = Field(default=0, ge=0)
    replayed_message_count: int = Field(default=0, ge=0)

    disclaimer: str = DASHBOARD_DISCLAIMER
    synthetic_banner: str | None = None
    content_note: str = SESSION_CONTENT_NOTE

    status: DashboardSessionStatus
    status_reason: str | None = None
    session_end_recorded: bool = False
    disconnect_reason: str | None = None
    summary_recovered: bool | None = None

    created_at_utc: str | None = None
    first_sent_at_utc: str | None = None
    last_sent_at_utc: str | None = None
    first_received_at_utc: str | None = None
    last_received_at_utc: str | None = None

    complete_record_count: int = Field(ge=0)
    decoded_record_count: int = Field(ge=0)
    malformed_record_count: int = Field(ge=0)
    malformed_line_numbers: tuple[int, ...] = ()
    partial_trailing_line: bool = False
    dropped_message_count: int | None = Field(default=None, ge=0)

    message_type_counts: tuple[tuple[str, int], ...] = ()
    source_counts: tuple[tuple[str, int], ...] = ()
    recorded_anomaly_counts: tuple[tuple[str, int], ...] = ()
    sequence_observation_count: int = Field(default=0, ge=0)
    sequence_observation_reasons: tuple[str, ...] = ()

    task_state: str | None = None
    current_difficulty_level: int | None = Field(default=None, ge=0)
    adaptation: SessionAdaptationCounts = Field(default_factory=SessionAdaptationCounts)

    #: Quantities this format cannot carry, stated rather than omitted.
    unavailable_statements: tuple[str, ...] = ()

    #: Paths a reader can open to audit this report.
    source_paths: tuple[str, ...] = ()
    #: ``(filename, sha256)`` of every source file that was present.
    source_checksums: tuple[tuple[str, str], ...] = ()

    #: SHA-256 over the canonical content, excluding ``exported_at_utc``.
    report_fingerprint: str = Field(min_length=1)
    #: Display metadata only. Never part of the report's identity.
    exported_at_utc: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.source_mode is DashboardSessionMode.ARTIFACT:
            raise DashboardError(
                "a session report records a live or replay observation; an "
                "experiment run is reported by the artifact views"
            )
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise DashboardError(
                f"session {self.session_id!r} is synthetic and cannot be "
                "scientifically eligible"
            )
        if self.scientific_evaluation_eligible:
            raise DashboardError(
                f"session report {self.session_id!r} claims scientific "
                "eligibility. " + SESSION_ELIGIBILITY_NOTE
            )
        if self.is_synthetic and self.synthetic_banner != SYNTHETIC_BANNER:
            raise DashboardError(
                "a synthetic session report must permanently carry the banner "
                f"{SYNTHETIC_BANNER!r}. There is no export path that strips it."
            )
        if not self.is_synthetic and self.synthetic_banner is not None:
            raise DashboardError(
                "a report with no synthetic record must not carry the "
                "software-self-check banner"
            )
        if self.disclaimer != DASHBOARD_DISCLAIMER:
            raise DashboardError(
                "a session report carries the standing disclaimer verbatim"
            )
        total = self.decoded_record_count + self.malformed_record_count
        if total != self.complete_record_count:
            raise DashboardError(
                f"{self.decoded_record_count} decoded + "
                f"{self.malformed_record_count} malformed = {total}, but the "
                f"report states {self.complete_record_count} complete records"
            )
        if len(self.malformed_line_numbers) != self.malformed_record_count:
            raise DashboardError(
                "the malformed line numbers do not match the malformed count"
            )
        counted = sum(count for _, count in self.message_type_counts)
        if self.message_type_counts and counted != self.decoded_record_count:
            raise DashboardError(
                f"the message-type counts total {counted} but "
                f"{self.decoded_record_count} records decoded"
            )
        return self


__all__ = [
    "LIVE_OBSERVATION_NOTE",
    "REPLAY_PRESENTATION_NOTE",
    "SESSION_CONTENT_NOTE",
    "SESSION_ELIGIBILITY_NOTE",
    "SESSION_REPORT_SCHEMA_VERSION",
    "DashboardReplayState",
    "DashboardSessionCatalogue",
    "DashboardSessionMode",
    "DashboardSessionProvenance",
    "DashboardSessionRecord",
    "DashboardSessionReport",
    "DashboardSessionStatus",
    "DashboardSessionSummary",
    "SessionAdaptationCounts",
    "SessionRecordProblem",
    "SessionSequenceObservation",
]
