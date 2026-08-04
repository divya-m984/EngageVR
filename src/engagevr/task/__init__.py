"""The deterministic Python task simulator.

Emits exactly the protocol messages the Unity desktop task emits, so the
backend can be developed, tested, and demonstrated with no Unity, no
webcam, no model asset, no display server, and no network.

Everything this package produces is SYNTHETIC and is permanently
labelled as such.  It is software test data.  It is not participant
data, not experimental evidence, and not a measurement of engagement,
attention, cognitive load, or fatigue.
"""

from engagevr.task.config import (
    SYNTHETIC_DISCLAIMER,
    Scenario,
    ScenarioKind,
    SimulatorConfig,
    SimulatorSpeed,
)
from engagevr.task.generator import (
    RESPONSE_KEYS,
    STIMULUS_CATEGORIES,
    PlannedTrial,
    TrialPlan,
    generate_trial_plan,
)
from engagevr.task.simulator import (
    PRODUCER,
    SimulatorResult,
    TaskSimulator,
)
from engagevr.task.state import AppliedCommand, TaskRuntimeState

__all__ = [
    "PRODUCER",
    "RESPONSE_KEYS",
    "STIMULUS_CATEGORIES",
    "SYNTHETIC_DISCLAIMER",
    "AppliedCommand",
    "PlannedTrial",
    "Scenario",
    "ScenarioKind",
    "SimulatorConfig",
    "SimulatorResult",
    "SimulatorSpeed",
    "TaskRuntimeState",
    "TaskSimulator",
    "TrialPlan",
    "generate_trial_plan",
]
