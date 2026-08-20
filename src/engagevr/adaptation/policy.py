"""The conservative adaptation policy (Milestone 8).

This module answers one question::

    Given an ELIGIBLE Milestone 7 prediction and the current task state,
    should a conservative adaptation be PROPOSED?

and, like the Milestone 7 gate it depends on, it is bounded as much by
what is absent as by what is present.  There is no transport client here,
no broker, no socket, no task simulator, and no Milestone 7 inference
code.  A reviewer can establish that this module cannot send anything, and
cannot recompute a confidence, a threshold, or a gate verdict, by reading
its imports: it imports two schema modules and its own mapping table.

Everything is pure.  :func:`evaluate_policy` takes an input, a state, and
a configuration, and returns a decision carrying the state it produced.
It mutates nothing, reads no clock, draws no random number, and holds no
module-level state, so the same triple always yields the same decision.

Precedence
----------
Guards are applied in a fixed order, and the first that fires decides:

1. the experimenter lock (``adaptation.enabled``);
2. the static experimental condition;
3. stream integrity — duplicates absorbed, out-of-order refused;
4. the **Milestone 7 gate**, which cannot be overridden;
5. evidence availability;
6. the mapping — overload protection, then engagement headroom;
7. conflict resolution;
8. persistence (dwell);
9. cooldown;
10. the session adaptation budget;
11. current state and difficulty bounds;
12. the proposal.

Invalid versus unavailable
--------------------------
A malformed input raises
:class:`~engagevr.schemas.adaptation_policy.AdaptationPolicyError`:
an unknown class label, an out-of-order window, a current difficulty
outside the configured bounds, one session's state applied to another.
The legitimate *absence* of evidence is not an error — it is a hold with a
stated reason.  Nothing here turns a missing difficulty into zero, a
missing target into "medium", a missing gate into eligible, or a missing
history into "no cooldown" without recording the cold-start condition.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from engagevr.adaptation.mapping import (
    cognitive_load_suggestion,
    engagement_suggestion,
    ordinal_state_from_class,
    ordinal_state_from_value,
    resolve_direction,
)
from engagevr.schemas.adaptation_policy import (
    ACTING_DIRECTIONS,
    POLICY_REASON_ORDER,
    AdaptationDecisionKind,
    AdaptationDirection,
    AdaptationInput,
    AdaptationPolicyConfiguration,
    AdaptationPolicyDecision,
    AdaptationPolicyError,
    AdaptationPolicyReason,
    AdaptationPolicyState,
    AdaptationProposal,
    AdaptationTargetEvidence,
    AdaptationTargetSuggestion,
    ConflictResolution,
    ExperimentMode,
    OrdinalState,
    TargetStateSource,
)
from engagevr.schemas.targets import TargetName, TaskType
from engagevr.schemas.uncertainty import AdaptationGateDecision, AdaptationGateRecord

#: Recorded when a guard stopped evaluation before the mapping ran.
_NOT_REACHED_NOTE = (
    "The mapping was not consulted: an earlier guard decided this window."
)


def evaluate_policy(
    policy_input: AdaptationInput,
    state: AdaptationPolicyState,
    configuration: AdaptationPolicyConfiguration,
) -> AdaptationPolicyDecision:
    """Evaluate one window and return the decision plus the new state.

    Parameters
    ----------
    policy_input:
        Everything the policy is permitted to see for this window.
    state:
        The session's policy state entering this window.  It is not
        mutated; :attr:`AdaptationPolicyDecision.state_after` carries the
        state produced.
    configuration:
        The resolved, validated policy settings.

    Raises
    ------
    AdaptationPolicyError
        If the state belongs to another session, the window arrives out
        of order, the reported difficulty is outside the configured
        bounds, or a target reports a class label this policy does not
        know.
    """
    _check_session(policy_input, state)
    duplicate = _check_ordering(policy_input, state)
    if duplicate:
        return _duplicate_decision(policy_input, state, configuration)
    _check_current_difficulty(policy_input, configuration)

    fingerprint = configuration.fingerprint()

    # --- 1 and 2: experimenter lock and experimental condition ----------
    if not configuration.enabled:
        return _short_circuit(
            policy_input,
            state,
            configuration,
            fingerprint,
            reasons={AdaptationPolicyReason.ADAPTATION_DISABLED},
            note=(
                "Adaptation is disabled for this run. The experimenter lock "
                "holds every window regardless of the evidence."
            ),
        )
    if configuration.experiment_mode is ExperimentMode.STATIC:
        return _short_circuit(
            policy_input,
            state,
            configuration,
            fingerprint,
            reasons={AdaptationPolicyReason.STATIC_EXPERIMENT_MODE},
            note=(
                "This session runs in the static experimental condition. The "
                "environment does not adapt, by condition rather than by "
                "outcome."
            ),
        )

    # --- 4 and 5: the Milestone 7 gate, then evidence availability ------
    engagement, engagement_reasons = _suggestion(
        policy_input.engagement,
        default_target=TargetName.ENGAGEMENT_CLASS,
        suggest=engagement_suggestion,
        configuration=configuration,
    )
    cognitive_load, load_reasons = _suggestion(
        policy_input.cognitive_load,
        default_target=TargetName.COGNITIVE_LOAD_CLASS,
        suggest=cognitive_load_suggestion,
        configuration=configuration,
    )
    blocking = engagement_reasons | load_reasons

    if blocking or engagement.state is None or cognitive_load.state is None:
        if not blocking:  # pragma: no cover - defensive; a state-less target
            blocking = {AdaptationPolicyReason.INSUFFICIENT_EVIDENCE}
        return _short_circuit(
            policy_input,
            state,
            configuration,
            fingerprint,
            reasons=blocking,
            note=_NOT_REACHED_NOTE,
            engagement=engagement,
            cognitive_load=cognitive_load,
        )

    # --- 6 and 7: the mapping and conflict resolution -------------------
    mapping = resolve_direction(
        engagement.state,
        cognitive_load.state,
        conflict_resolution=configuration.conflict_resolution,
    )
    resolved = mapping.direction

    state_after = _advance(
        state,
        policy_input,
        configuration,
        resolved=resolved,
        proposed=False,
    )

    if resolved is AdaptationDirection.HOLD:
        return _build_decision(
            policy_input,
            state,
            state_after,
            configuration,
            fingerprint,
            kind=AdaptationDecisionKind.HOLD,
            reasons={mapping.reason},
            engagement=engagement,
            cognitive_load=cognitive_load,
            conflict=mapping.conflict,
            resolution=mapping.resolution,
            resolved=resolved,
            note=mapping.note,
            proposal=None,
        )

    # --- 8: persistence -------------------------------------------------
    required = configuration.minimum_persistence_windows
    changed_direction = (
        state.pending_direction is not None and state.pending_direction is not resolved
    )
    if state_after.persistence_count < required:
        reason = (
            AdaptationPolicyReason.DIRECTION_CHANGE_BLOCKED
            if changed_direction
            else AdaptationPolicyReason.INSUFFICIENT_PERSISTENCE
        )
        note = (
            f"This window resolved to {resolved.value!r} while "
            f"{state.pending_direction.value if state.pending_direction else None!r} "
            "was being counted, so the dwell count restarted at 1."
            if changed_direction
            else (
                f"{state_after.persistence_count} of {required} consecutive "
                f"windows support {resolved.value!r}."
            )
        )
        return _build_decision(
            policy_input,
            state,
            state_after,
            configuration,
            fingerprint,
            kind=AdaptationDecisionKind.HOLD,
            reasons={reason},
            engagement=engagement,
            cognitive_load=cognitive_load,
            conflict=mapping.conflict,
            resolution=mapping.resolution,
            resolved=resolved,
            note=note,
            proposal=None,
        )

    # --- 9: cooldown ----------------------------------------------------
    if state.cooldown_remaining > 0:
        return _build_decision(
            policy_input,
            state,
            state_after,
            configuration,
            fingerprint,
            kind=AdaptationDecisionKind.HOLD,
            reasons={AdaptationPolicyReason.COOLDOWN_ACTIVE},
            engagement=engagement,
            cognitive_load=cognitive_load,
            conflict=mapping.conflict,
            resolution=mapping.resolution,
            resolved=resolved,
            note=(
                f"{state.cooldown_remaining} cooldown window(s) remain after "
                f"the proposal at window order "
                f"{state.last_adaptation_window_order}."
            ),
            proposal=None,
        )

    # --- 10: session adaptation budget ----------------------------------
    budget = configuration.max_adaptations_per_session
    if budget is not None and state.adaptation_count >= budget:
        return _build_decision(
            policy_input,
            state,
            state_after,
            configuration,
            fingerprint,
            kind=AdaptationDecisionKind.HOLD,
            reasons={AdaptationPolicyReason.SESSION_ADAPTATION_BUDGET_EXHAUSTED},
            engagement=engagement,
            cognitive_load=cognitive_load,
            conflict=mapping.conflict,
            resolution=mapping.resolution,
            resolved=resolved,
            note=(
                f"{state.adaptation_count} of {budget} permitted proposals "
                "have already been made in this session."
            ),
            proposal=None,
        )

    # --- 11: current state and bounds -----------------------------------
    current = policy_input.current_difficulty
    if current is None:
        return _build_decision(
            policy_input,
            state,
            state_after,
            configuration,
            fingerprint,
            kind=AdaptationDecisionKind.HOLD,
            reasons={AdaptationPolicyReason.CURRENT_STATE_UNAVAILABLE},
            engagement=engagement,
            cognitive_load=cognitive_load,
            conflict=mapping.conflict,
            resolution=mapping.resolution,
            resolved=resolved,
            note=(
                "The environment reported no current difficulty. An unknown "
                "level is not level zero, so no target level can be computed."
            ),
            proposal=None,
        )

    bounds = configuration.difficulty
    at_bound = (
        current >= bounds.maximum
        if resolved is AdaptationDirection.INCREASE
        else current <= bounds.minimum
    )
    if at_bound:
        reason = (
            AdaptationPolicyReason.ALREADY_AT_MAXIMUM
            if resolved is AdaptationDirection.INCREASE
            else AdaptationPolicyReason.ALREADY_AT_MINIMUM
        )
        return _build_decision(
            policy_input,
            state,
            state_after,
            configuration,
            fingerprint,
            kind=AdaptationDecisionKind.HOLD,
            reasons={reason},
            engagement=engagement,
            cognitive_load=cognitive_load,
            conflict=mapping.conflict,
            resolution=mapping.resolution,
            resolved=resolved,
            note=(
                f"Difficulty {current} is already at the configured bound "
                f"[{bounds.minimum}, {bounds.maximum}]. The policy holds "
                "rather than emitting a command that would change nothing."
            ),
            proposal=None,
        )

    # --- 12: the proposal -----------------------------------------------
    requested = (
        current + bounds.step
        if resolved is AdaptationDirection.INCREASE
        else current - bounds.step
    )
    proposed = min(max(requested, bounds.minimum), bounds.maximum)
    proposal_state = _advance(
        state,
        policy_input,
        configuration,
        resolved=resolved,
        proposed=True,
    )
    proposal = AdaptationProposal(
        proposal_id=_proposal_id(
            policy_input, resolved, current, proposed, fingerprint
        ),
        session_id=policy_input.session_id,
        subject_id=policy_input.subject_id,
        window_id=policy_input.window_id,
        window_order=policy_input.window_order,
        direction=resolved,
        current_difficulty=current,
        requested_difficulty=requested,
        proposed_difficulty=proposed,
        step=bounds.step,
        clamping_applied=requested != proposed,
        minimum_difficulty=bounds.minimum,
        maximum_difficulty=bounds.maximum,
        persistence_count=state_after.persistence_count,
        required_persistence_windows=required,
        adaptation_index=proposal_state.adaptation_count,
        engagement_gate=_gate_of(policy_input.engagement),
        cognitive_load_gate=_gate_of(policy_input.cognitive_load),
        policy_mode=configuration.mode,
        configuration_fingerprint=fingerprint,
        is_synthetic=policy_input.is_synthetic,
        scientific_evaluation_eligible=False,
    )
    return _build_decision(
        policy_input,
        state,
        proposal_state,
        configuration,
        fingerprint,
        kind=AdaptationDecisionKind.PROPOSE_ADAPTATION,
        reasons={AdaptationPolicyReason.PROPOSAL_ELIGIBLE},
        engagement=engagement,
        cognitive_load=cognitive_load,
        conflict=mapping.conflict,
        resolution=mapping.resolution,
        resolved=resolved,
        note=mapping.note,
        proposal=proposal,
    )


# ---------------------------------------------------------------------------
# Input integrity
# ---------------------------------------------------------------------------


def _check_session(policy_input: AdaptationInput, state: AdaptationPolicyState) -> None:
    if state.session_id != policy_input.session_id:
        raise AdaptationPolicyError(
            f"policy state belongs to session {state.session_id!r} but the "
            f"window belongs to {policy_input.session_id!r}. Persistence, "
            "cooldown, and budget are session-scoped and are never carried "
            "across a session boundary; start a fresh state instead"
        )


def _check_ordering(
    policy_input: AdaptationInput, state: AdaptationPolicyState
) -> bool:
    """Whether this window is an absorbed duplicate.

    A window whose order is not strictly greater than the last one is
    either the same window again — absorbed idempotently, changing
    nothing — or a genuinely out-of-order arrival, which is refused.  The
    state remembers only the last window, which is sufficient because
    ``window_order`` is contractually monotonic within a session: an
    earlier window re-presented later always carries a lower order.
    """
    last_order = state.last_window_order
    if last_order is None:
        return False
    if policy_input.window_order > last_order:
        if policy_input.window_id == state.last_window_id:
            raise AdaptationPolicyError(
                f"window {policy_input.window_id!r} was already evaluated at "
                f"order {last_order} and now claims order "
                f"{policy_input.window_order}; one window cannot occupy two "
                "positions in the stream"
            )
        return False
    if (
        policy_input.window_order == last_order
        and policy_input.window_id == state.last_window_id
    ):
        return True
    raise AdaptationPolicyError(
        f"window {policy_input.window_id!r} arrived at order "
        f"{policy_input.window_order}, at or before the last evaluated order "
        f"{last_order} (window {state.last_window_id!r}). Out-of-order windows "
        "are refused rather than reordered: the dwell count, the cooldown, and "
        "the budget are all defined over the evaluation sequence"
    )


def _check_current_difficulty(
    policy_input: AdaptationInput, configuration: AdaptationPolicyConfiguration
) -> None:
    current = policy_input.current_difficulty
    if current is None:
        return
    bounds = configuration.difficulty
    if not bounds.contains(current):
        raise AdaptationPolicyError(
            f"the environment reports difficulty {current}, outside the "
            f"configured bounds [{bounds.minimum}, {bounds.maximum}]. This is "
            "a configuration or data error, not a state to adapt from; it is "
            "refused rather than clamped"
        )


# ---------------------------------------------------------------------------
# Evidence to suggestion
# ---------------------------------------------------------------------------


def _suggestion(
    evidence: AdaptationTargetEvidence | None,
    *,
    default_target: TargetName,
    suggest: Callable[[OrdinalState], AdaptationDirection],
    configuration: AdaptationPolicyConfiguration,
) -> tuple[AdaptationTargetSuggestion, set[AdaptationPolicyReason]]:
    """One target's contribution, and the reasons it blocked, if any."""
    if evidence is None:
        return (
            AdaptationTargetSuggestion(
                target_name=default_target,
                evidence_available=False,
                unavailable_reason=AdaptationPolicyReason.INSUFFICIENT_EVIDENCE,
            ),
            {AdaptationPolicyReason.INSUFFICIENT_EVIDENCE},
        )

    decision = evidence.decision
    gate = evidence.gate
    quality = decision.signal_quality
    common = {
        "target_name": evidence.target_name,
        "evidence_available": True,
        "gate_decision": gate.decision,
        # Milestone 7's reasons, verbatim and in Milestone 7's order.
        "gate_reasons": gate.reasons,
        "prediction_available": decision.prediction_available,
        "prediction_abstained": decision.abstained,
        "predicted_class": decision.predicted_class,
        "predicted_value": decision.predicted_value,
        "interval_lower_bound": decision.interval_lower_bound,
        "interval_upper_bound": decision.interval_upper_bound,
        # Provenance only. Neither field can reach a direction or a step.
        "minimum_recorded_quality": (
            quality.minimum_recorded_quality if quality is not None else None
        ),
        "confidence_score": decision.confidence_score,
        "source_prediction_id": decision.source_prediction_id,
    }

    if gate.decision is not AdaptationGateDecision.ELIGIBLE:
        reasons = {AdaptationPolicyReason.GATE_BLOCKED}
        if not decision.prediction_available:
            reasons.add(AdaptationPolicyReason.PREDICTION_UNAVAILABLE)
        elif decision.abstained:
            reasons.add(AdaptationPolicyReason.PREDICTION_ABSTAINED)
        primary = min(reasons, key=POLICY_REASON_ORDER.index)
        return (
            AdaptationTargetSuggestion(**common, unavailable_reason=primary),
            reasons,
        )

    if evidence.task_type is TaskType.CLASSIFICATION:
        assert decision.predicted_class is not None  # guaranteed when eligible
        state = ordinal_state_from_class(evidence.target_name, decision.predicted_class)
        source = TargetStateSource.CLASSIFICATION_CLASS
    else:
        if not configuration.regression_mapping_enabled:
            return (
                AdaptationTargetSuggestion(
                    **common,
                    unavailable_reason=AdaptationPolicyReason.NO_POLICY_FOR_TARGET,
                ),
                {AdaptationPolicyReason.NO_POLICY_FOR_TARGET},
            )
        band = configuration.band_for(evidence.target_name)
        if band is None:  # pragma: no cover - forbidden by the configuration
            return (
                AdaptationTargetSuggestion(
                    **common,
                    unavailable_reason=AdaptationPolicyReason.NO_POLICY_FOR_TARGET,
                ),
                {AdaptationPolicyReason.NO_POLICY_FOR_TARGET},
            )
        assert decision.predicted_value is not None  # guaranteed when eligible
        state = ordinal_state_from_value(
            band,
            decision.predicted_value,
            interval_lower_bound=decision.interval_lower_bound,
            interval_upper_bound=decision.interval_upper_bound,
            require_interval_inside_band=configuration.require_interval_inside_band,
        )
        source = TargetStateSource.REGRESSION_BAND

    return (
        AdaptationTargetSuggestion(
            **common,
            state=state,
            state_source=source,
            suggested_direction=suggest(state),
        ),
        set(),
    )


def _gate_of(evidence: AdaptationTargetEvidence | None) -> AdaptationGateRecord:
    """The gate record of a target that reached the proposal branch.

    Both targets are guaranteed present and eligible by then: a missing or
    blocked target short-circuits well before this point.
    """
    if evidence is None:  # pragma: no cover - unreachable by construction
        raise AdaptationPolicyError(
            "a proposal was reached without evidence for both targets"
        )
    return evidence.gate


# ---------------------------------------------------------------------------
# State transition
# ---------------------------------------------------------------------------


def _advance(
    state: AdaptationPolicyState,
    policy_input: AdaptationInput,
    configuration: AdaptationPolicyConfiguration,
    *,
    resolved: AdaptationDirection,
    proposed: bool,
) -> AdaptationPolicyState:
    """The state this window produces.

    Persistence rule.  A window that resolves to an acting direction
    either extends the count for that direction or restarts it at 1 for
    the new one.  A window that resolves to ``HOLD`` **resets** the count
    to zero and clears the pending direction; it does not decay it.  A
    decay would let evidence separated by contradicting windows accumulate
    into a dwell requirement that was never actually met, and a window
    blocked by Milestone 7 resolves to ``HOLD``, so a blocked window can
    never count as supporting evidence.

    Cooldown rule.  ``cooldown_remaining`` counts windows that must still
    pass, and it decreases by one on every evaluated window whether or not
    that window carried usable evidence.  Time passing is a property of
    the stream, not of the evidence in it.  A proposal sets it to
    ``cooldown_windows``, so the next proposal can occur at the earliest
    ``cooldown_windows + 1`` evaluation windows later.
    """
    if proposed and resolved not in ACTING_DIRECTIONS:  # pragma: no cover
        raise AdaptationPolicyError("a proposal requires an acting direction")

    if resolved in ACTING_DIRECTIONS:
        if state.pending_direction is resolved:
            persistence = state.persistence_count + 1
        else:
            persistence = 1
        pending: AdaptationDirection | None = resolved
    else:
        persistence = 0
        pending = None

    if proposed:
        # Fresh evidence is required for the next proposal: the dwell count
        # restarts rather than carrying over into the cooldown period.
        persistence = 0
        pending = None
        cooldown = configuration.cooldown_windows
        adaptation_count = state.adaptation_count + 1
        last_direction: AdaptationDirection | None = resolved
        last_window: int | None = policy_input.window_order
    else:
        cooldown = max(state.cooldown_remaining - 1, 0)
        adaptation_count = state.adaptation_count
        last_direction = state.last_applied_direction
        last_window = state.last_adaptation_window_order

    return AdaptationPolicyState(
        session_id=state.session_id,
        last_window_id=policy_input.window_id,
        last_window_order=policy_input.window_order,
        # What the environment REPORTED, never what this policy proposed.
        current_difficulty=policy_input.current_difficulty,
        pending_direction=pending,
        persistence_count=persistence,
        cooldown_remaining=cooldown,
        last_applied_direction=last_direction,
        last_adaptation_window_order=last_window,
        adaptation_count=adaptation_count,
        evaluated_window_count=state.evaluated_window_count + 1,
    )


# ---------------------------------------------------------------------------
# Decision construction
# ---------------------------------------------------------------------------


def _short_circuit(
    policy_input: AdaptationInput,
    state: AdaptationPolicyState,
    configuration: AdaptationPolicyConfiguration,
    fingerprint: str,
    *,
    reasons: set[AdaptationPolicyReason],
    note: str,
    engagement: AdaptationTargetSuggestion | None = None,
    cognitive_load: AdaptationTargetSuggestion | None = None,
) -> AdaptationPolicyDecision:
    """A hold decided before the mapping was consulted.

    The state still advances: the window was evaluated, so it counts
    toward cooldown expiry and it resets the dwell count.
    """
    state_after = _advance(
        state,
        policy_input,
        configuration,
        resolved=AdaptationDirection.HOLD,
        proposed=False,
    )
    return _build_decision(
        policy_input,
        state,
        state_after,
        configuration,
        fingerprint,
        kind=AdaptationDecisionKind.HOLD,
        reasons=reasons,
        engagement=engagement
        or AdaptationTargetSuggestion(
            target_name=TargetName.ENGAGEMENT_CLASS,
            evidence_available=False,
            unavailable_reason=AdaptationPolicyReason.INSUFFICIENT_EVIDENCE,
        ),
        cognitive_load=cognitive_load
        or AdaptationTargetSuggestion(
            target_name=TargetName.COGNITIVE_LOAD_CLASS,
            evidence_available=False,
            unavailable_reason=AdaptationPolicyReason.INSUFFICIENT_EVIDENCE,
        ),
        conflict=False,
        resolution=None,
        resolved=AdaptationDirection.HOLD,
        note=note,
        proposal=None,
    )


def _duplicate_decision(
    policy_input: AdaptationInput,
    state: AdaptationPolicyState,
    configuration: AdaptationPolicyConfiguration,
) -> AdaptationPolicyDecision:
    """A repeat of the previous window: absorbed, state unchanged.

    The dwell count does not advance, the cooldown does not tick down, and
    the budget is untouched.  A retransmitted window must not be able to
    manufacture evidence or expire a guard.
    """
    return _build_decision(
        policy_input,
        state,
        state,
        configuration,
        configuration.fingerprint(),
        kind=AdaptationDecisionKind.HOLD,
        reasons={AdaptationPolicyReason.DUPLICATE_WINDOW},
        engagement=AdaptationTargetSuggestion(
            target_name=TargetName.ENGAGEMENT_CLASS,
            evidence_available=False,
            unavailable_reason=AdaptationPolicyReason.DUPLICATE_WINDOW,
        ),
        cognitive_load=AdaptationTargetSuggestion(
            target_name=TargetName.COGNITIVE_LOAD_CLASS,
            evidence_available=False,
            unavailable_reason=AdaptationPolicyReason.DUPLICATE_WINDOW,
        ),
        conflict=False,
        resolution=None,
        resolved=AdaptationDirection.HOLD,
        note=(
            f"Window {policy_input.window_id!r} was already evaluated at order "
            f"{policy_input.window_order}. The repeat is absorbed: no count "
            "advances and no guard expires."
        ),
        proposal=None,
    )


def _build_decision(
    policy_input: AdaptationInput,
    state_before: AdaptationPolicyState,
    state_after: AdaptationPolicyState,
    configuration: AdaptationPolicyConfiguration,
    fingerprint: str,
    *,
    kind: AdaptationDecisionKind,
    reasons: set[AdaptationPolicyReason],
    engagement: AdaptationTargetSuggestion,
    cognitive_load: AdaptationTargetSuggestion,
    conflict: bool,
    resolution: ConflictResolution | None,
    resolved: AdaptationDirection,
    note: str,
    proposal: AdaptationProposal | None,
) -> AdaptationPolicyDecision:
    ordered = tuple(r for r in POLICY_REASON_ORDER if r in reasons)
    return AdaptationPolicyDecision(
        kind=kind,
        session_id=policy_input.session_id,
        subject_id=policy_input.subject_id,
        window_id=policy_input.window_id,
        window_order=policy_input.window_order,
        scenario_id=policy_input.scenario_id,
        reasons=ordered,
        engagement=engagement,
        cognitive_load=cognitive_load,
        conflict=conflict,
        conflict_resolution=resolution,
        resolved_direction=resolved,
        resolution_note=note,
        current_difficulty=policy_input.current_difficulty,
        cooldown_remaining_before=state_before.cooldown_remaining,
        cooldown_remaining_after=state_after.cooldown_remaining,
        persistence_count_before=state_before.persistence_count,
        persistence_count_after=state_after.persistence_count,
        pending_direction_before=state_before.pending_direction,
        pending_direction_after=state_after.pending_direction,
        adaptation_budget_used=state_after.adaptation_count,
        adaptation_budget_total=configuration.max_adaptations_per_session,
        proposal=proposal,
        state_before=state_before,
        state_after=state_after,
        experiment_mode=configuration.experiment_mode,
        policy_mode=configuration.mode,
        configuration_fingerprint=fingerprint,
        is_synthetic=policy_input.is_synthetic,
        scientific_evaluation_eligible=False,
    )


def _proposal_id(
    policy_input: AdaptationInput,
    direction: AdaptationDirection,
    current: int,
    proposed: int,
    fingerprint: str,
) -> str:
    """A deterministic identifier for one proposal.

    Derived from the session, the window, the move, and the configuration
    that produced it.  No clock and no random component participates, so
    re-running one sequence reproduces the same identifiers rather than
    making two traces look like different runs.
    """
    payload = "|".join(
        (
            policy_input.session_id,
            policy_input.window_id,
            str(policy_input.window_order),
            direction.value,
            str(current),
            str(proposed),
            fingerprint,
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"prop-{digest}"


__all__ = ["evaluate_policy"]
