"""The task client's state machine and adaptation-command handling.

Kept separate from the simulator loop so that the Unity client's
behaviour has an executable Python reference: both clients must accept
and reject the same commands in the same situations, and that rule lives
here rather than being restated twice.

Milestone 4 boundary
--------------------
This module *applies* commands that arrive.  It never *decides* them.
There is no policy, no cooldown, no hysteresis, and no derivation of a
command from task performance anywhere in this milestone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from engagevr.protocol.messages import (
    AdaptationAcknowledgementPayload,
    AdaptationCommandName,
    AdaptationCommandPayload,
    TaskState,
)


@dataclass(frozen=True, slots=True)
class AppliedCommand:
    """The record of one command that this client accepted."""

    command_id: str
    command: AdaptationCommandName
    applied_at_utc: datetime


@dataclass
class TaskRuntimeState:
    """Mutable state of a running task client."""

    task_id: str
    state: TaskState = TaskState.IDLE
    block_id: int | None = None
    trial_id: int | None = None
    difficulty_level: int = 1
    stimulus_interval_ms: float = 500.0

    #: Commands already applied, keyed by command id, so a repeat is
    #: absorbed idempotently instead of being applied twice.
    applied: dict[str, AppliedCommand] = field(default_factory=dict)

    def apply_command(
        self,
        command: AdaptationCommandPayload,
        *,
        session_id: str,
        now_utc: datetime,
    ) -> AdaptationAcknowledgementPayload:
        """Apply one adaptation command and build its acknowledgement.

        Rejection cases, all of which produce ``accepted=False`` with a
        stated reason rather than a silent no-op:

        - the command expired before it was handled;
        - ``pause_task`` when the task is not running;
        - ``resume_task`` when the task is not paused.

        A repeated ``command_id`` is accepted again with
        ``duplicate=True`` and is **not** re-applied, so a retransmitted
        command cannot double-step the difficulty.
        """
        previous = self.applied.get(command.command_id)
        if previous is not None:
            return AdaptationAcknowledgementPayload(
                command_id=command.command_id,
                accepted=True,
                applied_at_utc=previous.applied_at_utc,
                duplicate=True,
            )

        if command.expires_at_utc is not None and now_utc > command.expires_at_utc:
            return AdaptationAcknowledgementPayload(
                command_id=command.command_id,
                accepted=False,
                rejection_reason=(
                    f"command expired at {command.expires_at_utc.isoformat()}; "
                    f"it reached the client at {now_utc.isoformat()}"
                ),
            )

        rejection = self._rejection_for(command)
        if rejection is not None:
            return AdaptationAcknowledgementPayload(
                command_id=command.command_id,
                accepted=False,
                rejection_reason=rejection,
            )

        self._mutate(command)
        self.applied[command.command_id] = AppliedCommand(
            command_id=command.command_id,
            command=command.command,
            applied_at_utc=now_utc,
        )
        return AdaptationAcknowledgementPayload(
            command_id=command.command_id,
            accepted=True,
            applied_at_utc=now_utc,
        )

    def _rejection_for(self, command: AdaptationCommandPayload) -> str | None:
        if command.command is AdaptationCommandName.PAUSE_TASK:
            if self.state is not TaskState.RUNNING:
                return (
                    f"pause_task requires state 'running'; "
                    f"the task is {self.state.value!r}"
                )
        elif command.command is AdaptationCommandName.RESUME_TASK:
            if self.state is not TaskState.PAUSED:
                return (
                    f"resume_task requires state 'paused'; "
                    f"the task is {self.state.value!r}"
                )
        return None

    def _mutate(self, command: AdaptationCommandPayload) -> None:
        match command.command:
            case AdaptationCommandName.SET_DIFFICULTY:
                assert isinstance(command.value, int)
                self.difficulty_level = command.value
            case AdaptationCommandName.SET_STIMULUS_INTERVAL:
                assert isinstance(command.value, int | float)
                self.stimulus_interval_ms = float(command.value)
            case AdaptationCommandName.PAUSE_TASK:
                self.state = TaskState.PAUSED
            case AdaptationCommandName.RESUME_TASK:
                self.state = TaskState.RUNNING
