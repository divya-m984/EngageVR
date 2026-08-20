"""State-to-direction mapping: the engineering demonstration rule.

Everything here is pure, total, and deterministic.  Given two ordinal
states it returns one direction and one reason, with no state carried
between calls.

Where the rule comes from
-------------------------
``docs/PROJECT_SPECIFICATION.md`` states an example policy.  Three of its
rows are expressible with the Milestone 4 protocol's ``set_difficulty``
action and are implemented verbatim:

- "High engagement + low load → increase difficulty slightly"
- "Moderate engagement + moderate load → maintain current state"
- "High load + declining performance → reduce difficulty"

Two are **not** implemented, and their absence is deliberate:

- "Declining engagement + low or moderate load → feedback or introduce
  variation."  Neither feedback nor stimulus variation is an action this
  protocol can express, and "declining" is a temporal derivative of a
  quantity this project has never validated.  The rule therefore holds
  and records ``no_expressible_action`` rather than substituting a
  difficulty change for the response the specification actually named.
- "Sustained fatigue indicators → a break."  No fatigue estimator exists
  in this repository.  Inferring fatigue from blink proxies or heart-rate
  estimates is exactly the measurement-to-construct leap that
  :func:`engagevr.schemas.targets.reject_automatic_derivation` refuses, so
  ``pause_task`` is never issued by this policy even though the protocol
  supports it.

The rule as two principles
--------------------------
The 3x3 table below is generated from two stated principles plus a
default, rather than being nine independent choices:

**P1 — overload protection.** Cognitive load ``HIGH`` suggests
``DECREASE``.  This is the only protective direction and it is the only
rule that may fire on one signal's state alone.

**P2 — engagement headroom.** Engagement ``HIGH`` suggests ``INCREASE``,
and an increase is proposed only when cognitive load is affirmatively
``LOW``.  A ``MEDIUM`` load does not permit an increase: it is the absence
of a reading that supports one, not a reading that supports it.

**P3 — default hold.** Everything else holds.

|                | load low   | load medium | load high  |
|----------------|------------|-------------|------------|
| **eng low**    | hold       | hold        | decrease   |
| **eng medium** | hold       | hold        | decrease   |
| **eng high**   | increase   | hold        | conflict   |

The one conflict cell — high engagement with high cognitive load — is
resolved by configuration: ``hold`` (the conservative default) or
``prefer_decrease`` (protection wins).  There is deliberately no
``prefer_increase``.

None of this is a validated interpretation of human state.
"""

from __future__ import annotations

import math

from engagevr.schemas.adaptation_policy import (
    ORDINAL_CLASSIFICATION_TARGETS,
    ORDINAL_STATE_BY_LABEL,
    AdaptationDirection,
    AdaptationPolicyError,
    AdaptationPolicyReason,
    ConflictResolution,
    OrdinalState,
    RegressionBand,
)
from engagevr.schemas.targets import TargetName, TaskType, get_target_spec

#: The resolved direction for each state pair, before conflict resolution
#: and before any temporal guard.  Present as data so a reader can check
#: the table against the documentation without following control flow.
MAPPING_TABLE: dict[
    tuple[OrdinalState, OrdinalState],
    tuple[AdaptationDirection, AdaptationPolicyReason],
] = {
    (OrdinalState.LOW, OrdinalState.LOW): (
        AdaptationDirection.HOLD,
        AdaptationPolicyReason.NO_EXPRESSIBLE_ACTION,
    ),
    (OrdinalState.LOW, OrdinalState.MEDIUM): (
        AdaptationDirection.HOLD,
        AdaptationPolicyReason.NO_EXPRESSIBLE_ACTION,
    ),
    (OrdinalState.LOW, OrdinalState.HIGH): (
        AdaptationDirection.DECREASE,
        AdaptationPolicyReason.PROPOSAL_ELIGIBLE,
    ),
    (OrdinalState.MEDIUM, OrdinalState.LOW): (
        AdaptationDirection.HOLD,
        AdaptationPolicyReason.ADAPTATION_NOT_NEEDED,
    ),
    (OrdinalState.MEDIUM, OrdinalState.MEDIUM): (
        AdaptationDirection.HOLD,
        AdaptationPolicyReason.TARGET_IN_DEADBAND,
    ),
    (OrdinalState.MEDIUM, OrdinalState.HIGH): (
        AdaptationDirection.DECREASE,
        AdaptationPolicyReason.PROPOSAL_ELIGIBLE,
    ),
    (OrdinalState.HIGH, OrdinalState.LOW): (
        AdaptationDirection.INCREASE,
        AdaptationPolicyReason.PROPOSAL_ELIGIBLE,
    ),
    (OrdinalState.HIGH, OrdinalState.MEDIUM): (
        AdaptationDirection.HOLD,
        AdaptationPolicyReason.ADAPTATION_NOT_NEEDED,
    ),
    (OrdinalState.HIGH, OrdinalState.HIGH): (
        AdaptationDirection.HOLD,
        AdaptationPolicyReason.DIRECTION_CONFLICT,
    ),
}

#: Human-readable justification recorded on every decision.
RESOLUTION_NOTES: dict[AdaptationPolicyReason, str] = {
    AdaptationPolicyReason.NO_EXPRESSIBLE_ACTION: (
        "The specification's response for low engagement at low or moderate "
        "load is feedback or stimulus variation, which this protocol cannot "
        "express. A difficulty change is not substituted for it."
    ),
    AdaptationPolicyReason.ADAPTATION_NOT_NEEDED: (
        "No principle fires: an increase requires high engagement AND low "
        "cognitive load, and a decrease requires high cognitive load."
    ),
    AdaptationPolicyReason.TARGET_IN_DEADBAND: (
        "Both estimated states are the neutral class, which is the deadband. "
        "Adapting here would act on the absence of a signal."
    ),
    AdaptationPolicyReason.DIRECTION_CONFLICT: (
        "Engagement suggests increase and cognitive load suggests decrease. "
        "The two readings disagree, so the conservative resolution applies."
    ),
    AdaptationPolicyReason.PROPOSAL_ELIGIBLE: (
        "One principle fires and the other signal does not contradict it."
    ),
}


class MappingResult:
    """The mapping's verdict for one state pair.

    A tiny value object rather than a tuple, so callers read
    ``result.direction`` instead of ``result[0]``.
    """

    __slots__ = ("conflict", "direction", "note", "reason", "resolution")

    def __init__(
        self,
        *,
        direction: AdaptationDirection,
        reason: AdaptationPolicyReason,
        conflict: bool,
        resolution: ConflictResolution | None,
        note: str,
    ) -> None:
        self.direction = direction
        self.reason = reason
        self.conflict = conflict
        self.resolution = resolution
        self.note = note

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"MappingResult(direction={self.direction.value!r}, "
            f"reason={self.reason.value!r}, conflict={self.conflict!r})"
        )


def ordinal_state_from_class(target_name: TargetName, class_label: str) -> OrdinalState:
    """Map a class label onto its ordered state.

    The ordering comes from two independent declarations that must agree:
    the target schema's
    :attr:`~engagevr.schemas.targets.TargetSpec.class_order_is_ordinal`
    flag, and this module's explicit
    :data:`~engagevr.schemas.adaptation_policy.ORDINAL_CLASSIFICATION_TARGETS`
    vocabulary.  Neither array position nor alphabetical order is
    consulted, so a vocabulary that was reordered, extended, or renamed
    stops the policy instead of quietly changing what "high" means.

    Raises
    ------
    AdaptationPolicyError
        If the target is not declared ordinal, if its vocabulary is not
        the one this policy knows, or if the label is not in it.
    """
    spec = get_target_spec(target_name)
    if spec.task_type is not TaskType.CLASSIFICATION:
        raise AdaptationPolicyError(
            f"target {target_name.value!r} is a {spec.task_type.value} target; "
            "a class label cannot be read from it"
        )
    if not spec.class_order_is_ordinal:
        raise AdaptationPolicyError(
            f"target {target_name.value!r} does not declare "
            "class_order_is_ordinal, so its class vocabulary carries no "
            "ordering. This policy refuses to infer one from array position "
            "or alphabetical order"
        )
    declared = ORDINAL_CLASSIFICATION_TARGETS.get(target_name)
    if declared is None:
        raise AdaptationPolicyError(
            f"no ordinal vocabulary is declared for {target_name.value!r} in "
            "this policy; add one explicitly before mapping it"
        )
    if tuple(spec.class_vocabulary or ()) != declared:
        raise AdaptationPolicyError(
            f"the vocabulary of {target_name.value!r} is "
            f"{list(spec.class_vocabulary or ())}, but this policy was written "
            f"against {list(declared)}. The mapping is refused rather than "
            "re-derived, because a changed vocabulary changes what each state "
            "means"
        )
    state = ORDINAL_STATE_BY_LABEL.get(class_label)
    if state is None or class_label not in declared:
        raise AdaptationPolicyError(
            f"class label {class_label!r} is not a state this policy knows for "
            f"{target_name.value!r}; known labels: {list(declared)}"
        )
    return state


def ordinal_state_from_value(
    band: RegressionBand,
    value: float,
    *,
    interval_lower_bound: float | None = None,
    interval_upper_bound: float | None = None,
    require_interval_inside_band: bool = True,
) -> OrdinalState:
    """Map a continuous estimate onto an ordered state via an explicit band.

    The band's boundaries are configuration.  There is no default and no
    quantile of the observed data: a threshold chosen by looking at the
    values would make the deadband a function of the run.

    ``value < low_below`` is ``LOW``, ``value > high_above`` is ``HIGH``,
    and everything between — inclusive of both boundaries — is the neutral
    ``MEDIUM`` region.  When ``require_interval_inside_band`` is set and a
    Milestone 7 prediction interval is supplied, the *whole* interval must
    lie in the same region; an interval straddling a boundary reads as
    neutral.  That is a use of the interval's width as a deadband, not a
    re-derivation of Milestone 7's acceptance rule.

    Raises
    ------
    AdaptationPolicyError
        If the value or interval is not finite, or the interval does not
        contain the point estimate.
    """
    if not math.isfinite(value):
        raise AdaptationPolicyError(
            f"the estimate for {band.target_name.value!r} is not finite"
        )
    bounds = (interval_lower_bound, interval_upper_bound)
    if (bounds[0] is None) != (bounds[1] is None):
        raise AdaptationPolicyError(
            "interval bounds are supplied as a pair or not at all"
        )
    if bounds[0] is not None and bounds[1] is not None:
        if not math.isfinite(bounds[0]) or not math.isfinite(bounds[1]):
            raise AdaptationPolicyError("a prediction interval bound is not finite")
        if not bounds[0] <= value <= bounds[1]:
            raise AdaptationPolicyError(
                f"the point estimate {value} lies outside its interval "
                f"[{bounds[0]}, {bounds[1]}]"
            )

    def region(point: float) -> OrdinalState:
        if point < band.low_below:
            return OrdinalState.LOW
        if point > band.high_above:
            return OrdinalState.HIGH
        return OrdinalState.MEDIUM

    state = region(value)
    if require_interval_inside_band and bounds[0] is not None and bounds[1] is not None:
        if region(bounds[0]) is not state or region(bounds[1]) is not state:
            return OrdinalState.MEDIUM
    return state


def engagement_suggestion(state: OrdinalState) -> AdaptationDirection:
    """What the engagement reading alone suggests (principle P2).

    Only ``HIGH`` suggests anything.  ``LOW`` deliberately does not suggest
    an increase: "low engagement therefore make it harder" is a
    psychological assumption, not a reading of the evidence, and the
    specification's own response for that state is one this protocol
    cannot express.
    """
    return (
        AdaptationDirection.INCREASE
        if state is OrdinalState.HIGH
        else AdaptationDirection.HOLD
    )


def cognitive_load_suggestion(state: OrdinalState) -> AdaptationDirection:
    """What the cognitive-load reading alone suggests (principle P1).

    Only ``HIGH`` suggests anything.  A ``LOW`` load does not suggest an
    increase on its own: low measured load is not evidence that a person
    wants more to do.
    """
    return (
        AdaptationDirection.DECREASE
        if state is OrdinalState.HIGH
        else AdaptationDirection.HOLD
    )


def resolve_direction(
    engagement: OrdinalState,
    cognitive_load: OrdinalState,
    *,
    conflict_resolution: ConflictResolution = ConflictResolution.HOLD,
) -> MappingResult:
    """The direction the mapping recommends for one state pair.

    Pure and total: every one of the nine pairs has an entry in
    :data:`MAPPING_TABLE`, so there is no fall-through case and no
    implicit default.
    """
    direction, reason = MAPPING_TABLE[(engagement, cognitive_load)]
    conflict = reason is AdaptationPolicyReason.DIRECTION_CONFLICT
    resolution = conflict_resolution if conflict else None
    note = RESOLUTION_NOTES[reason]

    if conflict and conflict_resolution is ConflictResolution.PREFER_DECREASE:
        return MappingResult(
            direction=AdaptationDirection.DECREASE,
            reason=AdaptationPolicyReason.PROPOSAL_ELIGIBLE,
            conflict=True,
            resolution=resolution,
            note=(
                note + " Configuration selects prefer_decrease, so the "
                "protective direction is taken. The conflict is still recorded."
            ),
        )
    return MappingResult(
        direction=direction,
        reason=reason,
        conflict=conflict,
        resolution=resolution,
        note=note,
    )


__all__ = [
    "MAPPING_TABLE",
    "RESOLUTION_NOTES",
    "MappingResult",
    "cognitive_load_suggestion",
    "engagement_suggestion",
    "ordinal_state_from_class",
    "ordinal_state_from_value",
    "resolve_direction",
]
