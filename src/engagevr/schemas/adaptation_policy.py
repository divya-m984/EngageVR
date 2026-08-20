"""Adaptation-policy schemas (Milestone 8).

Milestone 7 answers *may* an already-chosen action be acted upon.  This
module is the vocabulary for the next question and only the next question::

    Given an ELIGIBLE prediction and the current task state, should a
    conservative adaptation be PROPOSED?

What this module encodes structurally rather than by convention
---------------------------------------------------------------
1. *A policy decision is not a network message.*
   :class:`AdaptationPolicyDecision` carries no command id, no wire
   payload, no address, and no send method.  Translating an approved
   :class:`AdaptationProposal` into the existing Milestone 4
   ``adaptation_command`` payload is a separate, optional, pure step in
   :mod:`engagevr.adaptation.command`.
2. *HOLD is a decision, not an error and not a missing value.*  A hold
   states at least one :class:`AdaptationPolicyReason` and carries no
   proposal.  A proposal states exactly
   :attr:`AdaptationPolicyReason.PROPOSAL_ELIGIBLE` and nothing else.
3. *The Milestone 7 gate cannot be bypassed.*  A proposal embeds the
   :class:`~engagevr.schemas.uncertainty.AdaptationGateRecord` of every
   target it used, and :class:`AdaptationPolicyDecision` refuses to
   validate a proposal whose gates are not all
   :attr:`~engagevr.schemas.uncertainty.AdaptationGateDecision.ELIGIBLE`.
   There is no override flag anywhere in this module.
4. *Signal quality cannot choose a direction.*  Quality reaches the policy
   only inside a Milestone 7 gate record, as provenance.  No field here
   maps a quality value onto an :class:`AdaptationDirection`, and
   :class:`AdaptationTargetSuggestion` derives its direction solely from a
   declared-ordinal class label or an explicitly configured regression
   band.
5. *Confidence cannot choose a step size.*  :class:`AdaptationProposal`
   records ``step`` from configuration and carries no confidence field.
   Confidence decided, in Milestone 7, whether the window may be acted on
   at all; it is not a control gain.
6. *A proposal is not an applied adaptation.*
   :class:`AdaptationLifecycleStatus` keeps ``proposed``,
   ``command_built``, ``dispatched``, ``acknowledged``, and ``applied``
   distinct, and an acknowledgement status can only be recorded from a
   real Milestone 4 acknowledgement payload.
7. *Synthetic can never be scientifically eligible*, the same contract as
   :mod:`engagevr.schemas.experiments`, :mod:`engagevr.schemas.fusion`,
   :mod:`engagevr.schemas.personalization`, and
   :mod:`engagevr.schemas.uncertainty`.

Scientific status
-----------------
The mapping from an estimated state to an adaptation direction in this
project is an **engineering demonstration rule**.  No adaptation rule here
is psychologically validated, pedagogically optimal, therapeutic, safe, or
demonstrated to benefit any person, and no field in this module may be
read as asserting otherwise.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, model_validator

from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER, EvaluationMode
from engagevr.schemas.targets import TargetName, TaskType, get_target_spec
from engagevr.schemas.uncertainty import (
    AbstentionDecision,
    AbstentionReason,
    AdaptationGateDecision,
    AdaptationGateRecord,
)

#: Attached to every policy document, decision trace, and proposal.
ADAPTATION_POLICY_NOTE = (
    "The mapping from an estimated engagement or cognitive-load state to an "
    "adaptation direction is an ENGINEERING DEMONSTRATION RULE. It is not a "
    "validated interpretation of human state, not psychologically validated, "
    "not pedagogically optimal, not therapeutic, and not demonstrated to "
    "benefit any person. No policy in this repository has been evaluated "
    "with human participants."
)

#: Attached to every proposal, so a proposal is never read as an outcome.
ADAPTATION_PROPOSAL_NOTE = (
    "A proposal is a recommendation produced by a deterministic rule. It is "
    "not a sent message, not an applied change, and not evidence that the "
    "change helped anyone. Whether a command was built, dispatched, "
    "acknowledged, or applied is recorded separately."
)

#: Attached wherever a controller metric is reported.
CONTROLLER_METRIC_NOTE = (
    "The counts and spacings reported here describe the behaviour of a "
    "software controller on a fixed input sequence. They are NOT engagement "
    "improvement, NOT cognitive-load reduction, NOT learning improvement, "
    "NOT comfort, NOT adaptation effectiveness, and NOT evidence about any "
    "person."
)

#: Why partial evidence is refused. Stated once, referenced by the policy.
BOTH_TARGETS_REQUIRED_NOTE = (
    "Both ordinal targets are required because neither rule in the mapping "
    "can fire without the other signal's affirmative state: an increase "
    "requires high engagement AND low cognitive load, and a decrease "
    "requires high cognitive load. A single target therefore cannot select "
    "any direction, and inventing one would be an assumption rather than a "
    "reading of the evidence."
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AdaptationPolicyMode(enum.StrEnum):
    """Implemented policy families.

    There is exactly one.  A learned or reinforcement-learning policy is
    deliberately absent: Milestone 8 begins with a small deterministic
    rule whose every decision can be reconstructed from its inputs.
    """

    CONSERVATIVE_RULE_BASED = "conservative_rule_based"


class ExperimentMode(enum.StrEnum):
    """Whether the environment may adapt at all during a session.

    The two modes are the static-versus-adaptive experimental conditions.
    They are kept as an explicit mode rather than as "adaptation happened
    to produce no proposals", because a static condition that silently
    depended on the policy holding would not be a static condition.
    """

    #: The environment never adapts. Every window holds, by condition.
    STATIC = "static"
    #: The policy may propose adaptations, subject to every guard.
    ADAPTIVE = "adaptive"


class AdaptationDirection(enum.StrEnum):
    """What the policy recommends doing to task difficulty.

    ``HOLD`` is a first-class member, not a null.
    """

    INCREASE = "increase"
    DECREASE = "decrease"
    HOLD = "hold"


#: Directions that name an actual change of state.
ACTING_DIRECTIONS: frozenset[AdaptationDirection] = frozenset(
    {AdaptationDirection.INCREASE, AdaptationDirection.DECREASE}
)


def opposite_direction(direction: AdaptationDirection) -> AdaptationDirection:
    """The reverse of an acting direction; ``HOLD`` has no reverse."""
    if direction is AdaptationDirection.INCREASE:
        return AdaptationDirection.DECREASE
    if direction is AdaptationDirection.DECREASE:
        return AdaptationDirection.INCREASE
    return AdaptationDirection.HOLD


class AdaptationDecisionKind(enum.StrEnum):
    """The two outcomes one policy evaluation can have."""

    HOLD = "hold"
    PROPOSE_ADAPTATION = "propose_adaptation"


class OrdinalState(enum.StrEnum):
    """The three ordered states the policy reasons about.

    ``LOW < MEDIUM < HIGH``.  The order is a property of this enum, not of
    any array position: a class label becomes one of these only through
    :data:`ORDINAL_STATE_BY_LABEL`, and a label with no entry there is
    refused rather than guessed.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


#: The only class labels this policy will read, and what each one means.
#: A vocabulary containing anything else is refused: ordering by array
#: position or alphabetically would silently invent an ordinal scale.
ORDINAL_STATE_BY_LABEL: dict[str, OrdinalState] = {
    "low": OrdinalState.LOW,
    "medium": OrdinalState.MEDIUM,
    "high": OrdinalState.HIGH,
}

#: Targets whose class vocabulary this policy is allowed to interpret as
#: an ordered scale, with the exact vocabulary each must declare.
ORDINAL_CLASSIFICATION_TARGETS: dict[TargetName, tuple[str, ...]] = {
    TargetName.ENGAGEMENT_CLASS: ("low", "medium", "high"),
    TargetName.COGNITIVE_LOAD_CLASS: ("low", "medium", "high"),
}

#: The regression counterpart of each classification target, used only
#: when regression mapping is explicitly enabled.
REGRESSION_TARGETS: dict[TargetName, TargetName] = {
    TargetName.ENGAGEMENT_CLASS: TargetName.ENGAGEMENT_SCORE,
    TargetName.COGNITIVE_LOAD_CLASS: TargetName.COGNITIVE_LOAD_SCORE,
}

#: Which role a target plays in the mapping.
ENGAGEMENT_TARGETS: frozenset[TargetName] = frozenset(
    {TargetName.ENGAGEMENT_CLASS, TargetName.ENGAGEMENT_SCORE}
)
COGNITIVE_LOAD_TARGETS: frozenset[TargetName] = frozenset(
    {TargetName.COGNITIVE_LOAD_CLASS, TargetName.COGNITIVE_LOAD_SCORE}
)


class TargetStateSource(enum.StrEnum):
    """How an :class:`OrdinalState` was obtained for one target."""

    #: From a declared-ordinal class label.
    CLASSIFICATION_CLASS = "classification_class"
    #: From an explicitly configured regression band.
    REGRESSION_BAND = "regression_band"


class ConflictResolution(enum.StrEnum):
    """What to do when the two targets suggest opposite directions."""

    #: Propose nothing. The conservative default.
    HOLD = "hold"
    #: Take the protective direction. Never the permissive one — there is
    #: deliberately no ``prefer_increase`` member.
    PREFER_DECREASE = "prefer_decrease"


class AdaptationPolicyReason(enum.StrEnum):
    """Why one policy evaluation ended the way it did.

    Declaration order is the canonical reporting order, so two runs of one
    configuration produce identical documents.  The order runs from
    conditions that stop evaluation earliest to conditions discovered
    last: condition and stream problems, then Milestone 7's verdict, then
    evidence, then the mapping, then the temporal guards, then bounds.
    """

    #: ``adaptation.enabled`` is false. The experimenter lock is on.
    ADAPTATION_DISABLED = "adaptation_disabled"
    #: The session runs in the static experimental condition.
    STATIC_EXPERIMENT_MODE = "static_experiment_mode"
    #: This window was already evaluated; the repeat was absorbed.
    DUPLICATE_WINDOW = "duplicate_window"
    #: Milestone 7 recorded no model prediction for a required target.
    PREDICTION_UNAVAILABLE = "prediction_unavailable"
    #: Milestone 7's selective layer abstained on a required target.
    PREDICTION_ABSTAINED = "prediction_abstained"
    #: Milestone 7's adaptation gate blocked a required target.
    GATE_BLOCKED = "gate_blocked"
    #: A required target contributed no evidence record at all.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    #: No mapping is configured for the supplied target's task type.
    NO_POLICY_FOR_TARGET = "no_policy_for_target"
    #: The state pair indicates a response this protocol cannot express.
    NO_EXPRESSIBLE_ACTION = "no_expressible_action"
    #: Every contributing state fell in the neutral region.
    TARGET_IN_DEADBAND = "target_in_deadband"
    #: The mapping resolved to hold for this state pair.
    ADAPTATION_NOT_NEEDED = "adaptation_not_needed"
    #: The two targets suggested opposite directions.
    DIRECTION_CONFLICT = "direction_conflict"
    #: Fewer consecutive supporting windows than the dwell requirement.
    INSUFFICIENT_PERSISTENCE = "insufficient_persistence"
    #: This window's direction differs from the direction being counted.
    DIRECTION_CHANGE_BLOCKED = "direction_change_blocked"
    #: A previous proposal's cooldown has not elapsed.
    COOLDOWN_ACTIVE = "cooldown_active"
    #: The session's adaptation budget is spent.
    SESSION_ADAPTATION_BUDGET_EXHAUSTED = "session_adaptation_budget_exhausted"
    #: The environment reported no current difficulty.
    CURRENT_STATE_UNAVAILABLE = "current_state_unavailable"
    #: A decrease was indicated but difficulty is already at its minimum.
    ALREADY_AT_MINIMUM = "already_at_minimum"
    #: An increase was indicated but difficulty is already at its maximum.
    ALREADY_AT_MAXIMUM = "already_at_maximum"
    #: Every guard was satisfied and a proposal was produced.
    PROPOSAL_ELIGIBLE = "proposal_eligible"


#: Canonical reporting order for policy reason codes.
POLICY_REASON_ORDER: tuple[AdaptationPolicyReason, ...] = tuple(AdaptationPolicyReason)


class AdaptationLifecycleStatus(enum.StrEnum):
    """How far one proposal has actually travelled.

    These are never collapsed.  A proposal that produced a command object
    has not been dispatched; a dispatched command has not been
    acknowledged; and an acknowledged command was only *applied* if the
    task client said so.
    """

    PROPOSED = "proposed"
    COMMAND_BUILT = "command_built"
    DISPATCHED = "dispatched"
    ACKNOWLEDGED = "acknowledged"
    APPLIED = "applied"
    REJECTED = "rejected"


#: Statuses that may only be reached from a real Milestone 4
#: acknowledgement payload, never asserted by the policy itself.
ACKNOWLEDGEMENT_DERIVED_STATUSES: frozenset[AdaptationLifecycleStatus] = frozenset(
    {
        AdaptationLifecycleStatus.ACKNOWLEDGED,
        AdaptationLifecycleStatus.APPLIED,
        AdaptationLifecycleStatus.REJECTED,
    }
)


class AdaptationPolicyError(ValueError):
    """A policy input, state, or configuration is invalid.

    Raised for malformed input — an unknown class label, an out-of-order
    window, a current difficulty outside the configured bounds, another
    session's state.  It is deliberately **not** raised for the legitimate
    *absence* of evidence, which is a hold with a reason.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class DifficultyBounds(BaseModel):
    """The legal difficulty range and the one-evaluation step size.

    A step is a configured constant.  It is never scaled by confidence: a
    confidence score decided in Milestone 7 whether this window may be
    acted on at all, and reusing it as a control gain would make a barely
    admissible estimate move the environment further than a clear one.
    """

    model_config = {"extra": "forbid", "frozen": True}

    minimum: int = Field(ge=0, description="Inclusive. The protocol forbids negatives.")
    maximum: int = Field(ge=0, description="Inclusive.")
    step: int = Field(gt=0, description="Levels moved per proposal. Not a gain.")

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.minimum > self.maximum:
            raise ValueError(
                f"difficulty.minimum ({self.minimum}) must not exceed "
                f"difficulty.maximum ({self.maximum})"
            )
        if self.step > self.maximum - self.minimum and self.maximum > self.minimum:
            raise ValueError(
                f"difficulty.step ({self.step}) exceeds the whole configured "
                f"range [{self.minimum}, {self.maximum}]; every proposal would "
                "be clamped to a bound"
            )
        return self

    def contains(self, value: int) -> bool:
        """Whether ``value`` is a legal difficulty level."""
        return self.minimum <= value <= self.maximum


class RegressionBand(BaseModel):
    """Explicit low/high boundaries with a neutral region between them.

    Both boundaries are required when regression mapping is enabled.  There
    is no default, because a default would be an invented threshold on a
    scale this repository has never measured.
    """

    model_config = {"extra": "forbid", "frozen": True}

    target_name: TargetName
    low_below: float = Field(description="State is LOW strictly below this value.")
    high_above: float = Field(description="State is HIGH strictly above this value.")
    unit: str = Field(
        min_length=1,
        description="Unit of the boundaries, taken from the target spec.",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        spec = get_target_spec(self.target_name)
        if spec.task_type is not TaskType.REGRESSION:
            raise ValueError(
                f"{self.target_name.value!r} is not a regression target; a "
                "regression band cannot be declared for it"
            )
        for name, value in (
            ("low_below", self.low_below),
            ("high_above", self.high_above),
        ):
            if not math.isfinite(value):
                raise ValueError(f"regression band {name} is not finite")
        if self.low_below >= self.high_above:
            raise ValueError(
                f"regression band for {self.target_name.value!r}: low_below "
                f"({self.low_below}) must be strictly below high_above "
                f"({self.high_above}); otherwise there is no neutral region "
                "and the band is not a deadband"
            )
        low, high = spec.value_minimum, spec.value_maximum
        assert low is not None and high is not None  # guaranteed by TargetSpec
        for name, value in (
            ("low_below", self.low_below),
            ("high_above", self.high_above),
        ):
            if not low <= value <= high:
                raise ValueError(
                    f"regression band {name} ({value}) lies outside the declared "
                    f"range [{low}, {high}] of {self.target_name.value!r}"
                )
        return self


class AdaptationPolicyConfiguration(BaseModel):
    """The resolved, validated settings one policy run was evaluated under.

    Every numeric value here is an ENGINEERING DEFAULT.  None was selected
    by looking at a result, none is empirically optimal, none is validated,
    and none is a production setting.
    """

    model_config = {"extra": "forbid", "frozen": True}

    enabled: bool = True
    experiment_mode: ExperimentMode = ExperimentMode.ADAPTIVE
    mode: AdaptationPolicyMode = AdaptationPolicyMode.CONSERVATIVE_RULE_BASED

    minimum_persistence_windows: int = Field(
        default=3,
        ge=1,
        description=(
            "Consecutive policy-evaluation windows that must resolve to the "
            "same acting direction before a proposal. An engineering default, "
            "not a validated dwell time."
        ),
    )
    cooldown_windows: int = Field(
        default=6,
        ge=0,
        description=(
            "Policy-evaluation windows that must pass after a proposal before "
            "another may be made. Counted in windows, not seconds, so an "
            "offline replay is reproducible."
        ),
    )
    difficulty: DifficultyBounds
    max_adaptations_per_session: int | None = Field(
        default=10,
        ge=0,
        description="Proposals permitted per session. None means unlimited.",
    )
    conflict_resolution: ConflictResolution = ConflictResolution.HOLD

    regression_mapping_enabled: bool = False
    regression_bands: tuple[RegressionBand, ...] = ()
    require_interval_inside_band: bool = Field(
        default=True,
        description=(
            "When regression mapping is enabled, require the whole Milestone 7 "
            "prediction interval to fall inside a band before that band's "
            "state is used. A point estimate straddling a boundary reads as "
            "the neutral state instead."
        ),
    )

    note: str = ADAPTATION_POLICY_NOTE
    partial_evidence_note: str = BOTH_TARGETS_REQUIRED_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.regression_mapping_enabled:
            named = {band.target_name for band in self.regression_bands}
            required = set(REGRESSION_TARGETS.values())
            missing = required - named
            if missing:
                raise ValueError(
                    "regression mapping is enabled but no band is configured "
                    f"for {sorted(t.value for t in missing)}; a band has no "
                    "default because inventing one would presume a scale this "
                    "repository has not measured"
                )
        if len({band.target_name for band in self.regression_bands}) != len(
            self.regression_bands
        ):
            raise ValueError("a regression band is declared twice")
        return self

    def band_for(self, target_name: TargetName) -> RegressionBand | None:
        """The configured band for ``target_name``, if any."""
        for band in self.regression_bands:
            if band.target_name is target_name:
                return band
        return None

    def fingerprint(self) -> str:
        """Stable digest of every setting that can change a decision.

        Used to derive deterministic proposal identifiers, so two runs of
        one configuration produce identical identifiers and two different
        configurations cannot collide.
        """
        payload = self.model_dump(
            mode="json", exclude={"note", "partial_evidence_note"}
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class AdaptationTargetEvidence(BaseModel):
    """One target's Milestone 7 output, carried whole.

    The abstention decision and the gate record are embedded rather than
    summarised, so no Milestone 8 field can disagree with the Milestone 7
    record it came from, and every Milestone 7 reason survives into the
    Milestone 8 trace.
    """

    model_config = {"extra": "forbid", "frozen": True}

    target_name: TargetName
    decision: AbstentionDecision
    gate: AdaptationGateRecord

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.decision.target_name != self.target_name.value:
            raise ValueError(
                f"evidence declares target {self.target_name.value!r} but its "
                f"Milestone 7 decision names {self.decision.target_name!r}"
            )
        for field in ("window_id", "subject_id", "session_id", "source_prediction_id"):
            left = getattr(self.decision, field)
            right = getattr(self.gate, field)
            if left != right:
                raise ValueError(
                    f"the abstention decision and the gate record disagree on "
                    f"{field}: {left!r} vs {right!r}; they do not describe the "
                    "same window"
                )
        if self.gate.decision is AdaptationGateDecision.ELIGIBLE and (
            self.decision.abstained or not self.decision.prediction_available
        ):
            raise ValueError(
                "the gate reports eligible for a window whose prediction "
                "abstained or is unavailable"
            )
        return self

    @property
    def eligible(self) -> bool:
        """Whether Milestone 7 declared this window's action eligible."""
        return self.gate.decision is AdaptationGateDecision.ELIGIBLE

    @property
    def task_type(self) -> TaskType:
        """The task type of the embedded decision."""
        return self.decision.task_type


class AdaptationInput(BaseModel):
    """Everything one policy evaluation is allowed to see.

    There is no raw frame, no landmark, no name, and no email here, and
    the model forbids extra fields, so none can be smuggled in.
    """

    model_config = {"extra": "forbid", "frozen": True}

    session_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1, description="Pseudonymous. Never a name.")
    window_id: str = Field(min_length=1)
    window_order: int = Field(
        ge=0,
        description=(
            "Monotonically increasing index of this window within the "
            "session. Ordering is the policy's only notion of time."
        ),
    )

    current_difficulty: int | None = Field(
        default=None,
        description=(
            "Difficulty the environment reports it is currently at. None "
            "means unknown, which holds; it is never read as zero."
        ),
    )

    engagement: AdaptationTargetEvidence | None = None
    cognitive_load: AdaptationTargetEvidence | None = None

    scenario_id: str | None = Field(
        default=None, description="Name of the offline scenario, when replaying one."
    )
    is_synthetic: bool
    scientific_evaluation_eligible: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"window {self.window_id!r}"
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                f"{where}: a synthetic input can never be scientifically eligible"
            )
        for name, evidence, permitted in (
            ("engagement", self.engagement, ENGAGEMENT_TARGETS),
            ("cognitive_load", self.cognitive_load, COGNITIVE_LOAD_TARGETS),
        ):
            if evidence is None:
                continue
            if evidence.target_name not in permitted:
                raise ValueError(
                    f"{where}: the {name} slot holds evidence for "
                    f"{evidence.target_name.value!r}, which is not an "
                    f"{name} target"
                )
            for field in ("window_id", "subject_id", "session_id"):
                left = getattr(evidence.decision, field)
                if left != getattr(self, field):
                    raise ValueError(
                        f"{where}: the {name} evidence names {field}={left!r}, "
                        f"but the input names {getattr(self, field)!r}"
                    )
            if evidence.decision.is_synthetic != self.is_synthetic:
                raise ValueError(
                    f"{where}: the {name} evidence and the input disagree on "
                    "whether the data is synthetic"
                )
        return self


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AdaptationPolicyState(BaseModel):
    """The policy's whole memory for one session.

    Frozen, so an evaluation cannot mutate the state it was handed: every
    transition returns a new object, and the pair (before, after) is
    recorded on the decision.  There is no module-level singleton anywhere
    in :mod:`engagevr.adaptation`, and a state carrying one session id is
    refused for another.
    """

    model_config = {"extra": "forbid", "frozen": True}

    session_id: str = Field(min_length=1)

    last_window_id: str | None = None
    last_window_order: int | None = Field(default=None, ge=0)

    current_difficulty: int | None = Field(
        default=None,
        description=(
            "The last difficulty the environment REPORTED. Never the last "
            "difficulty this policy proposed: a proposal is not an applied "
            "change."
        ),
    )

    pending_direction: AdaptationDirection | None = Field(
        default=None, description="The acting direction currently being counted."
    )
    persistence_count: int = Field(default=0, ge=0)
    cooldown_remaining: int = Field(default=0, ge=0)

    last_applied_direction: AdaptationDirection | None = Field(
        default=None, description="Direction of the most recent PROPOSAL."
    )
    last_adaptation_window_order: int | None = Field(default=None, ge=0)
    adaptation_count: int = Field(
        default=0, ge=0, description="Proposals made this session, not changes applied."
    )
    evaluated_window_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.pending_direction is AdaptationDirection.HOLD:
            raise ValueError(
                "pending_direction records an acting direction or nothing; "
                "'hold' is not something to persist toward"
            )
        if self.last_applied_direction is AdaptationDirection.HOLD:
            raise ValueError("last_applied_direction cannot be 'hold'")
        if self.pending_direction is None and self.persistence_count != 0:
            raise ValueError(
                f"persistence_count is {self.persistence_count} with no pending "
                "direction; a count with nothing to count toward is not a state"
            )
        if self.pending_direction is not None and self.persistence_count == 0:
            raise ValueError(
                "a pending direction with a zero count is not a state; clear "
                "the direction instead"
            )
        if (self.last_applied_direction is None) != (
            self.last_adaptation_window_order is None
        ):
            raise ValueError(
                "last_applied_direction and last_adaptation_window_order are "
                "recorded as a pair"
            )
        if self.adaptation_count == 0 and self.last_applied_direction is not None:
            raise ValueError("a direction was applied but adaptation_count is zero")
        return self

    @classmethod
    def for_session(cls, session_id: str) -> Self:
        """A cold-start state for ``session_id``.

        Cold start is an explicit condition, not an absence: cooldown is
        zero because no proposal has been made in this session, and that
        fact is recorded rather than inferred from missing history.
        """
        return cls(session_id=session_id)


# ---------------------------------------------------------------------------
# Suggestions, proposals, decisions
# ---------------------------------------------------------------------------


class AdaptationTargetSuggestion(BaseModel):
    """What one target contributed, and what it alone would suggest.

    Recorded for every evaluation, including the ones that hold, so a
    reader can see each signal's own reading rather than only the
    resolved outcome.
    """

    model_config = {"extra": "forbid", "frozen": True}

    target_name: TargetName
    evidence_available: bool
    gate_decision: AdaptationGateDecision | None = None
    gate_reasons: tuple[AbstentionReason, ...] = Field(
        default=(),
        description="Milestone 7's reasons, preserved verbatim. Never rewritten.",
    )
    prediction_available: bool | None = None
    prediction_abstained: bool | None = None

    state: OrdinalState | None = None
    state_source: TargetStateSource | None = None
    predicted_class: str | None = None
    predicted_value: float | None = None
    interval_lower_bound: float | None = None
    interval_upper_bound: float | None = None

    suggested_direction: AdaptationDirection = AdaptationDirection.HOLD
    unavailable_reason: AdaptationPolicyReason | None = None

    #: Provenance only. Never an input to the direction.
    minimum_recorded_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Provenance only. Never a step multiplier.
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    source_prediction_id: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.evidence_available:
            if self.state is not None:
                raise ValueError(
                    "no evidence is available but an ordinal state is recorded"
                )
            if self.unavailable_reason is None:
                raise ValueError(
                    "an unavailable target must state why; absence with no "
                    "reason is indistinguishable from an unread field"
                )
        if self.state is None and self.suggested_direction is not (
            AdaptationDirection.HOLD
        ):
            raise ValueError(
                "a direction was suggested without an ordinal state; quality, "
                "confidence, and availability cannot choose a direction"
            )
        if (self.state is None) != (self.state_source is None):
            raise ValueError("state and state_source are recorded as a pair")
        return self


class AdaptationProposal(BaseModel):
    """A conservative adaptation the policy recommends.

    A proposal exists only when every guard was satisfied.  It embeds the
    Milestone 7 gate record of every target it used, so its eligibility is
    provable from the object rather than asserted by it.
    """

    model_config = {"extra": "forbid", "frozen": True}

    proposal_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    window_order: int = Field(ge=0)

    direction: AdaptationDirection
    current_difficulty: int = Field(ge=0)
    requested_difficulty: int = Field(
        description=(
            "current +/- step, before any bound was applied. Deliberately "
            "unbounded: a configured step can ask for a level outside the "
            "range, and recording what was asked for is the point of keeping "
            "it beside the resolved value. Only proposed_difficulty is "
            "constrained, because only it can become a command."
        )
    )
    proposed_difficulty: int = Field(ge=0, description="The in-bounds value.")
    step: int = Field(gt=0)
    clamping_applied: bool = Field(
        default=False,
        description=(
            "True when requested and proposed differ. Both values are kept, so "
            "a clamped proposal is never reported as the move that was asked "
            "for."
        ),
    )
    minimum_difficulty: int = Field(ge=0)
    maximum_difficulty: int = Field(ge=0)

    persistence_count: int = Field(ge=1)
    required_persistence_windows: int = Field(ge=1)
    adaptation_index: int = Field(
        ge=1, description="Which proposal of this session this is."
    )

    engagement_gate: AdaptationGateRecord
    cognitive_load_gate: AdaptationGateRecord

    policy_mode: AdaptationPolicyMode
    configuration_fingerprint: str = Field(min_length=1)

    is_synthetic: bool
    scientific_evaluation_eligible: bool = False

    note: str = ADAPTATION_PROPOSAL_NOTE
    policy_note: str = ADAPTATION_POLICY_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"proposal {self.proposal_id!r} (window {self.window_id!r})"
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                f"{where}: a synthetic proposal can never be scientifically eligible"
            )
        if self.direction not in ACTING_DIRECTIONS:
            raise ValueError(
                f"{where}: a proposal names an acting direction; got "
                f"{self.direction.value!r}"
            )
        if self.minimum_difficulty > self.maximum_difficulty:
            raise ValueError(f"{where}: the difficulty bounds are inverted")
        for name, value in (
            ("current_difficulty", self.current_difficulty),
            ("proposed_difficulty", self.proposed_difficulty),
        ):
            if not self.minimum_difficulty <= value <= self.maximum_difficulty:
                raise ValueError(
                    f"{where}: {name} ({value}) lies outside the configured "
                    f"bounds [{self.minimum_difficulty}, {self.maximum_difficulty}]"
                )
        expected = (
            self.current_difficulty + self.step
            if self.direction is AdaptationDirection.INCREASE
            else self.current_difficulty - self.step
        )
        if self.requested_difficulty != expected:
            raise ValueError(
                f"{where}: requested_difficulty ({self.requested_difficulty}) is "
                f"not current {self.direction.value} step ({expected}); a "
                "proposal's arithmetic is fixed by its configured step"
            )
        if self.clamping_applied != (
            self.requested_difficulty != (self.proposed_difficulty)
        ):
            raise ValueError(
                f"{where}: clamping_applied disagrees with the recorded values"
            )
        if self.proposed_difficulty == self.current_difficulty:
            raise ValueError(
                f"{where}: the proposal does not change difficulty; a no-op is "
                "a hold, and holds carry no proposal"
            )
        moved_up = self.proposed_difficulty > self.current_difficulty
        if moved_up != (self.direction is AdaptationDirection.INCREASE):
            raise ValueError(
                f"{where}: the proposed difficulty moves in the opposite "
                f"direction to {self.direction.value!r}"
            )
        if self.persistence_count < self.required_persistence_windows:
            raise ValueError(
                f"{where}: persistence {self.persistence_count} is below the "
                f"required {self.required_persistence_windows}; the dwell "
                "requirement is not something a proposal may record as unmet"
            )
        for name, gate in (
            ("engagement", self.engagement_gate),
            ("cognitive_load", self.cognitive_load_gate),
        ):
            if gate.decision is not AdaptationGateDecision.ELIGIBLE:
                raise ValueError(
                    f"{where}: the Milestone 7 gate for {name} is "
                    f"{gate.decision.value!r}; a blocked gate cannot produce a "
                    "proposal and there is no override"
                )
            if gate.window_id != self.window_id:
                raise ValueError(
                    f"{where}: the {name} gate describes window {gate.window_id!r}"
                )
            if gate.is_synthetic != self.is_synthetic:
                raise ValueError(
                    f"{where}: the {name} gate and the proposal disagree on "
                    "whether the data is synthetic"
                )
        return self


class AdaptationPolicyDecision(BaseModel):
    """The outcome of one policy evaluation, with its whole provenance.

    A reader of one record can answer, without re-running anything: was
    Milestone 7 eligible; what each target contributed; what each target
    alone suggested; whether they conflicted; whether persistence was
    satisfied; whether cooldown was active; whether the state was in
    bounds; whether budget remained; what was decided; and, if a change
    was proposed, from what to what.
    """

    model_config = {"extra": "forbid", "frozen": True}

    kind: AdaptationDecisionKind
    session_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    window_order: int = Field(ge=0)
    scenario_id: str | None = None

    reasons: tuple[AdaptationPolicyReason, ...]

    engagement: AdaptationTargetSuggestion
    cognitive_load: AdaptationTargetSuggestion
    conflict: bool = False
    conflict_resolution: ConflictResolution | None = None
    resolved_direction: AdaptationDirection = AdaptationDirection.HOLD
    resolution_note: str = Field(
        min_length=1, description="Why the resolved direction is what it is."
    )

    current_difficulty: int | None = None
    cooldown_remaining_before: int = Field(ge=0)
    cooldown_remaining_after: int = Field(ge=0)
    persistence_count_before: int = Field(ge=0)
    persistence_count_after: int = Field(ge=0)
    pending_direction_before: AdaptationDirection | None = None
    pending_direction_after: AdaptationDirection | None = None
    adaptation_budget_used: int = Field(ge=0)
    adaptation_budget_total: int | None = Field(default=None, ge=0)

    proposal: AdaptationProposal | None = None

    state_before: AdaptationPolicyState
    state_after: AdaptationPolicyState

    experiment_mode: ExperimentMode
    policy_mode: AdaptationPolicyMode
    configuration_fingerprint: str = Field(min_length=1)

    is_synthetic: bool
    scientific_evaluation_eligible: bool = False

    note: str = ADAPTATION_POLICY_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"window {self.window_id!r}"
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                f"{where}: a synthetic decision can never be scientifically eligible"
            )
        if not self.reasons:
            raise ValueError(f"{where}: a policy decision must state a reason")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError(f"{where}: duplicate policy reasons")
        ordered = tuple(r for r in POLICY_REASON_ORDER if r in set(self.reasons))
        if self.reasons != ordered:
            raise ValueError(
                f"{where}: policy reasons must be recorded in the canonical "
                f"order {[r.value for r in ordered]} so two runs of one "
                "configuration produce identical documents"
            )

        proposing = self.kind is AdaptationDecisionKind.PROPOSE_ADAPTATION
        eligible_recorded = AdaptationPolicyReason.PROPOSAL_ELIGIBLE in self.reasons
        if proposing:
            if self.proposal is None:
                raise ValueError(
                    f"{where}: a proposing decision must carry its proposal"
                )
            if self.reasons != (AdaptationPolicyReason.PROPOSAL_ELIGIBLE,):
                raise ValueError(
                    f"{where}: a proposing decision states exactly "
                    "'proposal_eligible'; a proposal that also records a "
                    "blocking reason is a contradiction"
                )
            if self.resolved_direction not in ACTING_DIRECTIONS:
                raise ValueError(
                    f"{where}: a proposing decision resolved to "
                    f"{self.resolved_direction.value!r}"
                )
            if self.proposal.direction is not self.resolved_direction:
                raise ValueError(
                    f"{where}: the proposal's direction disagrees with the "
                    "resolved direction"
                )
            if self.cooldown_remaining_before != 0:
                raise ValueError(
                    f"{where}: a proposal was made with "
                    f"{self.cooldown_remaining_before} cooldown window(s) left"
                )
            if self.adaptation_budget_total is not None and (
                self.adaptation_budget_used > self.adaptation_budget_total
            ):
                raise ValueError(f"{where}: the session adaptation budget was exceeded")
            for name, suggestion in (
                ("engagement", self.engagement),
                ("cognitive_load", self.cognitive_load),
            ):
                if suggestion.gate_decision is not AdaptationGateDecision.ELIGIBLE:
                    got = (
                        suggestion.gate_decision.value
                        if suggestion.gate_decision is not None
                        else None
                    )
                    raise ValueError(
                        f"{where}: a proposal requires an eligible Milestone 7 "
                        f"gate for {name}; got {got!r}"
                    )
            if self.current_difficulty is None:
                raise ValueError(
                    f"{where}: a proposal requires a known current difficulty"
                )
        else:
            if self.proposal is not None:
                raise ValueError(
                    f"{where}: a hold carries no proposal and therefore no "
                    "command payload"
                )
            if eligible_recorded:
                raise ValueError(f"{where}: a hold cannot record 'proposal_eligible'")
        if self.conflict and self.conflict_resolution is None:
            raise ValueError(f"{where}: a conflict must record how it was resolved")
        for name, snapshot in (
            ("state_before", self.state_before),
            ("state_after", self.state_after),
        ):
            if snapshot.session_id != self.session_id:
                raise ValueError(
                    f"{where}: {name} belongs to session "
                    f"{snapshot.session_id!r}, not {self.session_id!r}"
                )
        if self.state_before.cooldown_remaining != self.cooldown_remaining_before:
            raise ValueError(f"{where}: the recorded cooldown disagrees with the state")
        if self.state_after.cooldown_remaining != self.cooldown_remaining_after:
            raise ValueError(f"{where}: the recorded cooldown disagrees with the state")
        if self.state_before.persistence_count != self.persistence_count_before:
            raise ValueError(
                f"{where}: the recorded persistence disagrees with the state"
            )
        if self.state_after.persistence_count != self.persistence_count_after:
            raise ValueError(
                f"{where}: the recorded persistence disagrees with the state"
            )
        if self.state_after.adaptation_count != self.adaptation_budget_used:
            raise ValueError(
                f"{where}: the recorded budget use disagrees with the state"
            )
        return self

    @property
    def held(self) -> bool:
        """Whether this evaluation produced no proposal."""
        return self.kind is AdaptationDecisionKind.HOLD

    def primary_reason(self) -> AdaptationPolicyReason:
        """The first reason in canonical order."""
        return self.reasons[0]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class AdaptationHistoryEntry(BaseModel):
    """How far one proposal actually travelled.

    ``status`` starts at ``proposed`` and only advances through a step
    that really happened.  Nothing in :mod:`engagevr.adaptation` sets
    ``dispatched``, ``acknowledged``, ``applied``, or ``rejected`` on its
    own: the first requires a caller that sent something, and the rest
    require a Milestone 4 acknowledgement payload.
    """

    model_config = {"extra": "forbid", "frozen": True}

    proposal_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    window_order: int = Field(ge=0)

    direction: AdaptationDirection
    expected_command: str = Field(
        min_length=1, description="The existing protocol action name."
    )
    expected_value: int = Field(ge=0, description="The difficulty the command carries.")

    status: AdaptationLifecycleStatus = AdaptationLifecycleStatus.PROPOSED
    command_id: str | None = None
    dispatched_at_utc: datetime | None = None
    acknowledged: bool | None = None
    acknowledgement_duplicate: bool | None = None
    applied_at_utc: datetime | None = None
    rejection_reason: str | None = None

    is_synthetic: bool
    scientific_evaluation_eligible: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        where = f"proposal {self.proposal_id!r}"
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                f"{where}: a synthetic history entry can never be "
                "scientifically eligible"
            )
        if self.direction not in ACTING_DIRECTIONS:
            raise ValueError(f"{where}: a history entry names an acting direction")
        if self.status is AdaptationLifecycleStatus.PROPOSED:
            if self.command_id is not None:
                raise ValueError(
                    f"{where}: a merely proposed adaptation has no command id"
                )
        elif self.command_id is None:
            raise ValueError(
                f"{where}: status {self.status.value!r} requires a command id"
            )
        if self.status is AdaptationLifecycleStatus.COMMAND_BUILT and (
            self.dispatched_at_utc is not None
        ):
            raise ValueError(
                f"{where}: a built command that was never sent has no dispatch time"
            )
        if (
            self.status in ACKNOWLEDGEMENT_DERIVED_STATUSES
            and self.acknowledged is None
        ):
            raise ValueError(
                f"{where}: status {self.status.value!r} may only be recorded "
                "from a real acknowledgement; none is present"
            )
        if self.status is AdaptationLifecycleStatus.APPLIED:
            if self.acknowledged is not True or self.applied_at_utc is None:
                raise ValueError(
                    f"{where}: 'applied' requires an accepted acknowledgement "
                    "carrying the instant it was applied"
                )
        if self.status is AdaptationLifecycleStatus.REJECTED:
            if self.acknowledged is not False or not self.rejection_reason:
                raise ValueError(
                    f"{where}: 'rejected' requires a rejected acknowledgement "
                    "and its stated reason"
                )
        if self.acknowledged is False and self.applied_at_utc is not None:
            raise ValueError(f"{where}: a rejected command was not applied")
        return self


# ---------------------------------------------------------------------------
# Run-level reporting
# ---------------------------------------------------------------------------


class AdaptationControllerMetrics(BaseModel):
    """Mechanical behaviour of the controller on a fixed input sequence.

    Every field counts something the software did.  None of them measures
    engagement, cognitive load, learning, comfort, or benefit, and
    :data:`CONTROLLER_METRIC_NOTE` is carried on the document so a number
    lifted out of it keeps its meaning.
    """

    model_config = {"extra": "forbid", "frozen": True}

    evaluated_windows: int = Field(ge=0)
    gate_eligible_windows: int = Field(ge=0)
    gate_blocked_windows: int = Field(ge=0)

    hold_decisions: int = Field(ge=0)
    adaptation_proposals: int = Field(ge=0)
    increases: int = Field(ge=0)
    decreases: int = Field(ge=0)

    hold_reason_counts: dict[str, int] = Field(default_factory=dict)

    direction_reversals: int = Field(ge=0)
    minimum_proposal_spacing_windows: int | None = Field(default=None, ge=0)
    longest_same_direction_streak: int = Field(ge=0)
    eligible_window_adaptation_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    blocked_oscillation_attempts: int = Field(
        ge=0,
        description=(
            "Held windows whose resolved direction reversed the session's "
            "most recent proposed direction. Each one is a reversal the "
            "guards prevented; the reason it was held is in the trace."
        ),
    )

    final_difficulty_by_session: dict[str, int] = Field(default_factory=dict)
    proposals_by_session: dict[str, int] = Field(default_factory=dict)

    note: str = CONTROLLER_METRIC_NOTE

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.hold_decisions + self.adaptation_proposals != self.evaluated_windows:
            raise ValueError(
                "holds and proposals must account for every evaluated window: "
                f"{self.hold_decisions} + {self.adaptation_proposals} != "
                f"{self.evaluated_windows}"
            )
        if self.increases + self.decreases != self.adaptation_proposals:
            raise ValueError(
                "every proposal is an increase or a decrease: "
                f"{self.increases} + {self.decreases} != {self.adaptation_proposals}"
            )
        if self.gate_eligible_windows + self.gate_blocked_windows != (
            self.evaluated_windows
        ):
            raise ValueError(
                "every evaluated window is gate-eligible or gate-blocked: "
                f"{self.gate_eligible_windows} + {self.gate_blocked_windows} != "
                f"{self.evaluated_windows}"
            )
        if self.adaptation_proposals > self.gate_eligible_windows:
            raise ValueError(
                "more proposals than gate-eligible windows; a blocked window "
                "cannot have produced a proposal"
            )
        total_holds = sum(self.hold_reason_counts.values())
        if self.hold_reason_counts and total_holds < self.hold_decisions:
            raise ValueError("the hold reason counts do not account for every hold")
        return self


class AdaptationRunSummary(BaseModel):
    """The document written at the end of one offline policy run."""

    model_config = {"extra": "forbid"}

    run_id: str = Field(min_length=1)
    engagevr_version: str
    python_version: str

    evaluation_mode: EvaluationMode
    scientific_evaluation_eligible: bool
    is_synthetic: bool
    data_source: str = Field(min_length=1)

    configuration: AdaptationPolicyConfiguration
    configuration_fingerprint: str = Field(min_length=1)
    scenario_names: tuple[str, ...] = ()
    session_ids: tuple[str, ...] = ()

    metrics: AdaptationControllerMetrics
    naive_comparison: AdaptationControllerMetrics | None = Field(
        default=None,
        description=(
            "The same input sequence through a controller with no dwell "
            "requirement, no cooldown, and no budget. A SOFTWARE CONTROLLER "
            "comparison showing that the temporal guards reduce action "
            "frequency. It is not a claim that either controller is better "
            "for a person."
        ),
    )

    started_at_utc: datetime
    finished_at_utc: datetime
    disclaimers: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.is_synthetic and self.scientific_evaluation_eligible:
            raise ValueError(
                "a synthetic adaptation run can never be scientifically eligible"
            )
        if not self.disclaimers:
            raise ValueError("an adaptation run summary must carry a disclaimer")
        if self.evaluation_mode is EvaluationMode.SOFTWARE_SELF_CHECK:
            if self.scientific_evaluation_eligible:
                raise ValueError(
                    "a software self-check can never be scientifically eligible"
                )
            if not any(SOFTWARE_SELF_CHECK_BANNER in d for d in self.disclaimers):
                raise ValueError(
                    "a software self-check document must carry the banner "
                    f"{SOFTWARE_SELF_CHECK_BANNER!r}"
                )
        if not any(ADAPTATION_POLICY_NOTE in d for d in self.disclaimers):
            raise ValueError(
                "an adaptation run summary must carry the demonstration-rule note"
            )
        return self


__all__ = [
    "ACKNOWLEDGEMENT_DERIVED_STATUSES",
    "ACTING_DIRECTIONS",
    "ADAPTATION_POLICY_NOTE",
    "ADAPTATION_PROPOSAL_NOTE",
    "BOTH_TARGETS_REQUIRED_NOTE",
    "COGNITIVE_LOAD_TARGETS",
    "CONTROLLER_METRIC_NOTE",
    "ENGAGEMENT_TARGETS",
    "ORDINAL_CLASSIFICATION_TARGETS",
    "ORDINAL_STATE_BY_LABEL",
    "POLICY_REASON_ORDER",
    "REGRESSION_TARGETS",
    "AdaptationControllerMetrics",
    "AdaptationDecisionKind",
    "AdaptationDirection",
    "AdaptationHistoryEntry",
    "AdaptationInput",
    "AdaptationLifecycleStatus",
    "AdaptationPolicyConfiguration",
    "AdaptationPolicyDecision",
    "AdaptationPolicyError",
    "AdaptationPolicyMode",
    "AdaptationPolicyReason",
    "AdaptationPolicyState",
    "AdaptationProposal",
    "AdaptationRunSummary",
    "AdaptationTargetEvidence",
    "AdaptationTargetSuggestion",
    "ConflictResolution",
    "DifficultyBounds",
    "ExperimentMode",
    "OrdinalState",
    "RegressionBand",
    "TargetStateSource",
    "opposite_direction",
]
