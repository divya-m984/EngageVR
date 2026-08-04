"""Runtime settings for one simulator run.

:class:`~engagevr.config.TaskConfig` holds the defaults from
``configs/defaults.yaml``.  This module adds the per-run knobs that only
make sense for a simulated run: the seed, the speed multiplier, and the
scripted failure scenarios.
"""

from __future__ import annotations

import enum
from typing import Self

from pydantic import BaseModel, Field, model_validator

from engagevr.config import TaskConfig

#: Attached to every simulator artefact and printed by the CLI.
SYNTHETIC_DISCLAIMER = (
    "SYNTHETIC DATA. Every response, reaction time, and timeout in this "
    "session was fabricated by software from a fixed seed. No person "
    "performed this task. This is not participant data, not experimental "
    "evidence, and not a measurement of anything."
)


class SimulatorSpeed(enum.StrEnum):
    """How the simulator relates to wall-clock time."""

    REALTIME = "realtime"
    ACCELERATED = "accelerated"
    IMMEDIATE = "immediate"


class ScenarioKind(enum.StrEnum):
    """A scripted deviation injected at a chosen trial.

    These exist so the backend's handling of pauses, disconnects, and
    missing responses can be tested deterministically rather than hoped
    for.
    """

    PAUSE = "pause"
    DISCONNECT = "disconnect"
    ABORT = "abort"


class Scenario(BaseModel):
    """One scripted deviation, anchored to a specific trial."""

    model_config = {"extra": "forbid"}

    kind: ScenarioKind
    block_id: int = Field(ge=0)
    trial_id: int = Field(ge=0)
    duration_ms: float = Field(
        default=1000.0,
        ge=0.0,
        description="Simulated pause length. Ignored for disconnect and abort.",
    )


class SimulatorConfig(BaseModel):
    """Everything one simulator run needs.

    ``speed`` is a multiplier on simulated time: ``1.0`` runs in real
    time, ``10.0`` runs ten times faster, and ``0`` runs immediately
    with no sleeping at all.  The **event content is identical in all
    three modes** for a given seed; only the pacing changes.  That is
    what makes the accelerated mode a valid stand-in for the real-time
    mode in tests.
    """

    model_config = {"extra": "forbid"}

    task: TaskConfig = Field(default_factory=TaskConfig)
    seed: int = Field(default=42, ge=0)
    speed: float = Field(
        default=1.0,
        ge=0.0,
        description="Time multiplier. 0 means immediate (no sleeping).",
    )
    participant_id: str = Field(
        default="synthetic_participant",
        min_length=1,
        max_length=128,
        description="Pseudonymous label for a SYNTHETIC participant that does "
        "not exist.",
    )
    task_id: str = Field(default="reaction_task_v1", min_length=1, max_length=128)
    scenarios: list[Scenario] = Field(default_factory=list, max_length=64)
    emit_task_state: bool = Field(
        default=True, description="Emit task_state alongside task_event messages."
    )

    @property
    def mode(self) -> SimulatorSpeed:
        """Which pacing mode ``speed`` selects."""
        if self.speed == 0.0:
            return SimulatorSpeed.IMMEDIATE
        if self.speed == 1.0:
            return SimulatorSpeed.REALTIME
        return SimulatorSpeed.ACCELERATED

    @property
    def total_trials(self) -> int:
        """Planned trial count across all blocks."""
        return self.task.blocks * self.task.trials_per_block

    @model_validator(mode="after")
    def _check(self) -> Self:
        for scenario in self.scenarios:
            if scenario.block_id >= self.task.blocks:
                raise ValueError(
                    f"scenario {scenario.kind.value} targets block "
                    f"{scenario.block_id}, but the task has only "
                    f"{self.task.blocks} block(s)"
                )
            if scenario.trial_id >= self.task.trials_per_block:
                raise ValueError(
                    f"scenario {scenario.kind.value} targets trial "
                    f"{scenario.trial_id}, but each block has only "
                    f"{self.task.trials_per_block} trial(s)"
                )
        return self
