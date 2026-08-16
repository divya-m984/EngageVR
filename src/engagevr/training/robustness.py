"""Missing-modality robustness scenarios and deterministic modality dropout.

A scenario removes whole modality groups from the *availability* record and
nothing else.  It never rewrites a measurement, never zero-fills a feature,
and never touches a target value: a missing modality is an absence of
evidence, and this module represents it as one.

Two distinct things live here:

1. **Deterministic evaluation scenarios** — named, fixed patterns such as
   ``missing_rppg`` or ``only_task``, evaluated on the same outer folds as
   every other result so the comparison is like for like.
2. **Synthetic modality dropout** — an optional, seeded software-robustness
   check that drops whole modality groups at random.  It is a *software*
   check: it fabricates an availability pattern that no measurement
   produced.  It is refused in scientific mode, where the availability
   pattern must be the one actually recorded.

Nothing measured here is a real-world robustness result.  On synthetic data
it describes how this code behaves when told a modality is absent; it says
nothing about how the system would behave when a real camera signal failed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from engagevr.schemas.fusion import FusionModality, MissingModalityScenario

_ALL = tuple(FusionModality)


def _others(*present: FusionModality) -> tuple[FusionModality, ...]:
    keep = set(present)
    return tuple(m for m in _ALL if m not in keep)


#: The deterministic missing-modality scenarios evaluated by a fusion run.
#: Order is part of the contract: it fixes the order of ``robustness.json``.
SCENARIOS: tuple[MissingModalityScenario, ...] = (
    MissingModalityScenario(
        name="all_modalities",
        absent_modalities=(),
        description=(
            "Every configured modality is available as recorded. The "
            "reference condition."
        ),
    ),
    MissingModalityScenario(
        name="missing_behavioural",
        absent_modalities=(FusionModality.BEHAVIOURAL,),
        description="Facial behavioural proxies are unavailable.",
    ),
    MissingModalityScenario(
        name="missing_head_pose",
        absent_modalities=(FusionModality.HEAD_POSE,),
        description="Head-pose geometry is unavailable.",
    ),
    MissingModalityScenario(
        name="missing_rppg",
        absent_modalities=(FusionModality.RPPG,),
        description="Camera-based pulse estimates are unavailable.",
    ),
    MissingModalityScenario(
        name="missing_task",
        absent_modalities=(FusionModality.TASK,),
        description="Task telemetry is unavailable.",
    ),
    MissingModalityScenario(
        name="missing_behavioural_and_rppg",
        absent_modalities=(FusionModality.BEHAVIOURAL, FusionModality.RPPG),
        description=(
            "Both camera-derived modalities are unavailable — the pattern a "
            "failed or occluded camera would produce."
        ),
    ),
    MissingModalityScenario(
        name="only_task",
        absent_modalities=_others(FusionModality.TASK),
        description="Only task telemetry remains.",
    ),
    MissingModalityScenario(
        name="only_behavioural",
        absent_modalities=_others(FusionModality.BEHAVIOURAL),
        description="Only facial behavioural proxies remain.",
    ),
    MissingModalityScenario(
        name="only_rppg",
        absent_modalities=_others(FusionModality.RPPG),
        description="Only camera-based pulse estimates remain.",
    ),
    MissingModalityScenario(
        name="only_head_pose",
        absent_modalities=_others(FusionModality.HEAD_POSE),
        description="Only head-pose geometry remains.",
    ),
)

#: Scenario evaluated when robustness analysis is switched off.
REFERENCE_SCENARIO = SCENARIOS[0]

#: Refusal message used when synthetic dropout is requested in scientific mode.
SCIENTIFIC_DROPOUT_REFUSAL = (
    "synthetic modality dropout is refused in scientific mode: it fabricates "
    "an availability pattern that no measurement produced. A scientific "
    "evaluation must use the availability pattern actually recorded for each "
    "window."
)


class RobustnessError(ValueError):
    """A robustness scenario cannot be evaluated as requested."""


def scenario_by_name(name: str) -> MissingModalityScenario:
    """Return the scenario called ``name``.

    Raises
    ------
    RobustnessError
        If no such scenario is defined.
    """
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    available = ", ".join(s.name for s in SCENARIOS)
    raise RobustnessError(
        f"unknown missing-modality scenario {name!r}; defined scenarios: {available}"
    )


def resolve_scenarios(
    names: Sequence[str] | None,
) -> tuple[MissingModalityScenario, ...]:
    """Resolve requested scenario names, defaulting to every defined scenario.

    Raises
    ------
    RobustnessError
        On an unknown or duplicated name.
    """
    if not names:
        return SCENARIOS
    seen: list[str] = []
    for name in names:
        if name in seen:
            raise RobustnessError(f"scenario {name!r} was requested more than once")
        seen.append(name)
    return tuple(scenario_by_name(name) for name in seen)


def apply_scenario(
    availability: Mapping[FusionModality, np.ndarray],
    scenario: MissingModalityScenario,
) -> dict[FusionModality, np.ndarray]:
    """Availability with the scenario's modalities forced absent.

    A scenario only ever *removes* availability.  It cannot make a modality
    that contributed nothing appear to have contributed something: a
    scenario describes a loss of evidence, never a gain.
    """
    absent = set(scenario.absent_modalities)
    return {
        modality: (
            np.zeros_like(np.asarray(mask, dtype=bool))
            if modality in absent
            else np.asarray(mask, dtype=bool).copy()
        )
        for modality, mask in availability.items()
    }


def synthetic_modality_dropout(
    availability: Mapping[FusionModality, np.ndarray],
    *,
    window_ids: Sequence[str],
    seed: int,
    probability: float,
) -> dict[FusionModality, np.ndarray]:
    """Deterministically drop whole modality groups for a software check.

    The decision for one (window, modality) pair is a pure function of
    ``seed``, the window id, and the modality name, so it is reproducible
    regardless of row order, fold assignment, or how many rows are
    processed.  A dropped modality loses availability for that entire
    window — modalities are dropped coherently, never feature by feature.

    Targets are untouched.  Only availability changes, which is what a
    modality outage actually is.

    Raises
    ------
    RobustnessError
        If ``probability`` is outside ``[0, 1)`` or the row counts disagree.
    """
    if not 0.0 <= probability < 1.0:
        raise RobustnessError(
            f"synthetic dropout probability must be in [0, 1); got {probability!r}"
        )
    dropped: dict[FusionModality, np.ndarray] = {}
    for modality, mask in availability.items():
        current = np.asarray(mask, dtype=bool)
        if len(current) != len(window_ids):
            raise RobustnessError(
                f"availability for modality {modality.value!r} has "
                f"{len(current)} rows but {len(window_ids)} window ids were "
                "supplied"
            )
        keep = np.ones(len(current), dtype=bool)
        for index, window in enumerate(window_ids):
            if _uniform(seed, modality.value, str(window)) < probability:
                keep[index] = False
        dropped[modality] = current & keep
    return dropped


def _uniform(seed: int, modality: str, window_id: str) -> float:
    """A deterministic value in [0, 1) for one (seed, modality, window)."""
    digest = hashlib.blake2b(
        f"{seed}|{modality}|{window_id}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def coverage(fused_count: int, sample_count: int) -> float | None:
    """Fused fraction of the evaluated windows, or ``None`` when none were."""
    if sample_count <= 0:
        return None
    return float(fused_count) / float(sample_count)


__all__ = [
    "REFERENCE_SCENARIO",
    "SCENARIOS",
    "SCIENTIFIC_DROPOUT_REFUSAL",
    "RobustnessError",
    "apply_scenario",
    "coverage",
    "resolve_scenarios",
    "scenario_by_name",
    "synthetic_modality_dropout",
]
