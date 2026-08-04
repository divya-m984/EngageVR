"""Deterministic trial generation from a seed.

Generation is separated from execution so that the entire trial plan —
stimuli, expected responses, fabricated outcomes, and fabricated
reaction times — can be produced, inspected, and compared without
running a session or touching a clock.

Determinism
-----------
Everything is drawn from a single :class:`random.Random` seeded once.
Draw order is fixed and does not depend on wall-clock time, dictionary
iteration order, or the pacing mode.  The same seed and the same
:class:`~engagevr.task.config.SimulatorConfig` therefore produce a
byte-identical plan on any machine.

Provenance
----------
Every outcome here is **fabricated**.  ``response_outcome`` is drawn
from a probability, not observed.  Nothing in this module measures
anything.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from engagevr.schemas.events import ResponseOutcome
from engagevr.task.config import SimulatorConfig

#: The neutral stimulus vocabulary.  Deliberately abstract: these are
#: shapes and colours with no semantic, emotional, or clinical content.
STIMULUS_CATEGORIES: tuple[str, ...] = ("square", "circle", "triangle")

#: Response keys the task defines, one per category, in the same order.
RESPONSE_KEYS: tuple[str, ...] = ("j", "k", "l")


@dataclass(frozen=True, slots=True)
class PlannedTrial:
    """One fully pre-determined trial.

    Attributes
    ----------
    response_outcome:
        ``None`` never occurs here: every trial resolves to correct,
        incorrect, or timeout.  A timeout carries no reaction time and
        no observed response, which is what keeps "no response" distinct
        from "wrong response" all the way through the pipeline.
    """

    block_id: int
    trial_id: int
    stimulus_id: str
    stimulus_category: str
    expected_response: str
    difficulty_level: int
    response_outcome: ResponseOutcome
    observed_response: str | None
    reaction_time_ms: float | None

    @property
    def timed_out(self) -> bool:
        return self.response_outcome is ResponseOutcome.TIMEOUT

    @property
    def response_correct(self) -> bool | None:
        """Correctness, or None for a timeout which has no correctness."""
        if self.timed_out:
            return None
        return self.response_outcome is ResponseOutcome.CORRECT


@dataclass(frozen=True, slots=True)
class TrialPlan:
    """The complete, ordered trial plan for one simulated session."""

    trials: tuple[PlannedTrial, ...]
    seed: int
    blocks: int
    trials_per_block: int

    @property
    def timeout_count(self) -> int:
        return sum(1 for trial in self.trials if trial.timed_out)

    @property
    def incorrect_count(self) -> int:
        return sum(
            1
            for trial in self.trials
            if trial.response_outcome is ResponseOutcome.INCORRECT
        )

    @property
    def correct_count(self) -> int:
        return sum(
            1
            for trial in self.trials
            if trial.response_outcome is ResponseOutcome.CORRECT
        )

    def for_block(self, block_id: int) -> tuple[PlannedTrial, ...]:
        return tuple(trial for trial in self.trials if trial.block_id == block_id)


def _draw_reaction_time(rng: random.Random, mean_ms: float, ceiling_ms: float) -> float:
    """Draw a fabricated reaction time in ``(0, ceiling_ms)``.

    A lognormal shape is used only because it is positive and
    right-skewed, which keeps the fabricated values from looking like a
    symmetric artefact.  **It is not a model of human reaction times**
    and no parameter here was fitted to any data.
    """
    value = rng.lognormvariate(0.0, 0.35) * mean_ms
    upper = ceiling_ms * 0.98
    return float(min(max(value, 1.0), upper))


def generate_trial_plan(config: SimulatorConfig) -> TrialPlan:
    """Build the complete deterministic trial plan for ``config``."""
    rng = random.Random(config.seed)
    task = config.task
    trials: list[PlannedTrial] = []

    for block_id in range(task.blocks):
        for trial_id in range(task.trials_per_block):
            category_index = rng.randrange(len(STIMULUS_CATEGORIES))
            category = STIMULUS_CATEGORIES[category_index]
            expected = RESPONSE_KEYS[category_index]
            stimulus_id = f"{category}-b{block_id}t{trial_id}"

            roll = rng.random()
            if roll < task.synthetic_timeout_rate:
                outcome = ResponseOutcome.TIMEOUT
                observed: str | None = None
                reaction: float | None = None
            elif roll < task.synthetic_timeout_rate + task.synthetic_error_rate:
                outcome = ResponseOutcome.INCORRECT
                wrong = [key for key in RESPONSE_KEYS if key != expected]
                observed = wrong[rng.randrange(len(wrong))]
                reaction = _draw_reaction_time(
                    rng,
                    task.synthetic_response_latency_ms,
                    task.response_timeout_ms,
                )
            else:
                outcome = ResponseOutcome.CORRECT
                observed = expected
                reaction = _draw_reaction_time(
                    rng,
                    task.synthetic_response_latency_ms,
                    task.response_timeout_ms,
                )

            trials.append(
                PlannedTrial(
                    block_id=block_id,
                    trial_id=trial_id,
                    stimulus_id=stimulus_id,
                    stimulus_category=category,
                    expected_response=expected,
                    difficulty_level=task.default_difficulty,
                    response_outcome=outcome,
                    observed_response=observed,
                    reaction_time_ms=reaction,
                )
            )

    return TrialPlan(
        trials=tuple(trials),
        seed=config.seed,
        blocks=task.blocks,
        trials_per_block=task.trials_per_block,
    )
