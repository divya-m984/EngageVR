"""Common event schemas for task telemetry and system events.

Scope note
----------
A task event is a **software telemetry record**.  Accuracy, reaction
time, and timeout counts describe what the task program observed.  They
are not engagement measurements, not attention measurements, not
cognitive-load measurements, and not fatigue measurements.  This task
has not been experimentally designed, piloted, or approved, so nothing
derived from it may be described as measuring a psychological
construct.
"""

from __future__ import annotations

import enum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class EventType(enum.StrEnum):
    """Standard event types for task and system telemetry.

    Members added in Milestone 4 extend the Milestone 1 vocabulary; no
    existing member was renamed or removed.
    """

    # --- Milestone 1 members (unchanged) ---
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    TRIAL_STARTED = "trial_started"
    TRIAL_ENDED = "trial_ended"
    STIMULUS_PRESENTED = "stimulus_presented"
    RESPONSE_RECEIVED = "response_received"
    DIFFICULTY_CHANGED = "difficulty_changed"
    BREAK_TRIGGERED = "break_triggered"
    ADAPTATION_APPLIED = "adaptation_applied"

    # --- Milestone 4 additions: neutral reaction-task vocabulary ---
    TASK_LOADED = "task_loaded"
    TASK_STARTED = "task_started"
    BLOCK_STARTED = "block_started"
    RESPONSE_REGISTERED = "response_registered"
    RESPONSE_TIMEOUT = "response_timeout"
    TRIAL_COMPLETED = "trial_completed"
    BLOCK_COMPLETED = "block_completed"
    TASK_PAUSED = "task_paused"
    TASK_RESUMED = "task_resumed"
    TASK_COMPLETED = "task_completed"
    TASK_ABORTED = "task_aborted"


#: The subset of :class:`EventType` that a task client may emit inside a
#: ``task_event`` protocol message.  Session lifecycle and adaptation
#: outcomes travel as their own message types, not as task events.
TASK_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.TASK_LOADED,
        EventType.TASK_STARTED,
        EventType.BLOCK_STARTED,
        EventType.TRIAL_STARTED,
        EventType.STIMULUS_PRESENTED,
        EventType.RESPONSE_REGISTERED,
        EventType.RESPONSE_TIMEOUT,
        EventType.TRIAL_COMPLETED,
        EventType.BLOCK_COMPLETED,
        EventType.TASK_PAUSED,
        EventType.TASK_RESUMED,
        EventType.TASK_COMPLETED,
        EventType.TASK_ABORTED,
    }
)


class ResponseOutcome(enum.StrEnum):
    """How a trial's response slot resolved.

    ``TIMEOUT`` is deliberately distinct from ``INCORRECT``: no response
    was produced at all, which is a different observation from a wrong
    response and must never be silently folded into an error rate.
    """

    CORRECT = "correct"
    INCORRECT = "incorrect"
    TIMEOUT = "timeout"


class TaskEventDetail(BaseModel):
    """The wire payload of one task event.

    Every field except ``event_type`` is optional because the vocabulary
    spans task-level, block-level, trial-level, and response-level
    events; a ``block_started`` event has no ``reaction_time_ms`` and
    must not invent one.

    Missing-data rules
    ------------------
    - A missing response leaves ``observed_response``,
      ``reaction_time_ms``, and ``response_correct`` as ``None``.  They
      are never set to ``""``, ``0``, or ``False`` to stand in for
      "no response".
    - ``response_outcome`` is ``timeout`` for a missed response, which
      is distinct from ``incorrect``.
    - ``reaction_time_ms`` may not be negative.

    Identifier fields (``task_id``, ``block_id``, ``trial_id``) are
    retained on every trial-scoped event so that events can later be
    joined into synchronized feature windows without re-deriving
    structure from ordering alone.
    """

    model_config = {"extra": "forbid"}

    event_type: EventType = Field(
        description="Task-event name; must be a member of TASK_EVENT_TYPES."
    )

    task_id: str | None = Field(
        default=None, description="Identifier of the task definition in use."
    )
    block_id: int | None = Field(
        default=None, ge=0, description="Zero-based block index within the task."
    )
    trial_id: int | None = Field(
        default=None, ge=0, description="Zero-based trial index within the block."
    )

    stimulus_id: str | None = Field(
        default=None, description="Identifier of the presented stimulus."
    )
    stimulus_category: str | None = Field(
        default=None, description="Neutral category label of the stimulus."
    )

    expected_response: str | None = Field(
        default=None, description="Response key the task defined as matching."
    )
    observed_response: str | None = Field(
        default=None,
        description="Response key actually registered. None = no response.",
    )
    response_correct: bool | None = Field(
        default=None,
        description=(
            "Whether the observed response matched the expected response. "
            "None when no response was registered."
        ),
    )
    response_outcome: ResponseOutcome | None = Field(
        default=None,
        description="Resolution of the response slot; timeout is not incorrect.",
    )

    reaction_time_ms: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Milliseconds from stimulus onset to registered response. "
            "None when no response was registered. Never negative."
        ),
    )

    difficulty_level: int | None = Field(
        default=None, ge=0, description="Task difficulty setting in force."
    )

    task_elapsed_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Milliseconds since task start, measured by the client.",
    )
    trial_elapsed_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Milliseconds since the current trial started.",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.event_type not in TASK_EVENT_TYPES:
            allowed = ", ".join(sorted(e.value for e in TASK_EVENT_TYPES))
            raise ValueError(
                f"event_type {self.event_type.value!r} is not a task event; "
                f"allowed: {allowed}"
            )
        if self.event_type is EventType.RESPONSE_TIMEOUT:
            if self.observed_response is not None:
                raise ValueError("response_timeout must not carry observed_response")
            if self.reaction_time_ms is not None:
                raise ValueError("response_timeout must not carry reaction_time_ms")
            if self.response_outcome not in (None, ResponseOutcome.TIMEOUT):
                raise ValueError("response_timeout requires response_outcome=timeout")
        if (
            self.response_outcome is ResponseOutcome.TIMEOUT
            and self.response_correct is not None
        ):
            raise ValueError(
                "a timeout has no correctness; leave response_correct unset"
            )
        return self


class BaseEvent(BaseModel):
    """Timestamped event base used by all telemetry producers."""

    session_id: str
    event_type: EventType
    monotonic_timestamp: float = Field(
        description="Session-local monotonic clock value."
    )
    data: dict[str, object] = Field(default_factory=dict)
    data_source: str = "synthetic"


class TaskEvent(BaseEvent):
    """Task-performance-specific event with trial metadata.

    The Milestone 1 fields (``trial_index``, ``difficulty_level``,
    ``correct``, ``reaction_time_ms``) are retained unchanged.  The
    Milestone 4 vocabulary is carried in ``detail``, which is the single
    authoritative representation of a task event on the wire; it is not
    duplicated field-by-field onto this model.
    """

    trial_index: int | None = None
    difficulty_level: int | None = None
    correct: bool | None = None
    reaction_time_ms: float | None = Field(
        default=None, ge=0.0, description="Reaction time in milliseconds."
    )
    detail: TaskEventDetail | None = Field(
        default=None,
        description="Milestone 4 task-event payload, when one is available.",
    )

    @classmethod
    def from_detail(
        cls,
        *,
        session_id: str,
        monotonic_timestamp: float,
        detail: TaskEventDetail,
        data_source: str = "synthetic",
    ) -> TaskEvent:
        """Build a :class:`TaskEvent` record from a wire ``TaskEventDetail``.

        This is the only mapping between the two representations, so the
        legacy and Milestone 4 views cannot drift apart.
        """
        return cls(
            session_id=session_id,
            event_type=detail.event_type,
            monotonic_timestamp=monotonic_timestamp,
            data_source=data_source,
            trial_index=detail.trial_id,
            difficulty_level=detail.difficulty_level,
            correct=detail.response_correct,
            reaction_time_ms=detail.reaction_time_ms,
            detail=detail,
        )
