"""Builders shared by the Milestone 8 tests.

Evidence is built through the real Milestone 7 gate function, so a test
cannot hand the policy an eligible gate for a window that abstained: the
gate refuses, exactly as it does in a real run.  That is deliberate — a
stubbed gate would let a test assert an invariant the production path
does not actually hold.
"""

from __future__ import annotations

from typing import Any

from engagevr.adaptation.scenarios import (
    BLOCKED_BY_CONFIDENCE,
    ELIGIBLE,
    build_evidence,
)
from engagevr.schemas.adaptation_policy import (
    AdaptationDecisionKind,
    AdaptationDirection,
    AdaptationInput,
    AdaptationPolicyConfiguration,
    AdaptationPolicyDecision,
    AdaptationPolicyMode,
    AdaptationPolicyReason,
    AdaptationPolicyState,
    AdaptationProposal,
    AdaptationTargetEvidence,
    AdaptationTargetSuggestion,
    ConflictResolution,
    DifficultyBounds,
    ExperimentMode,
    OrdinalState,
    TargetStateSource,
)
from engagevr.schemas.targets import TargetName
from engagevr.schemas.uncertainty import AdaptationGateDecision, AdaptationGateRecord

SESSION = "test-session"
SUBJECT = "test-subject"


def make_configuration(**overrides: Any) -> AdaptationPolicyConfiguration:
    """A policy configuration with the repository defaults, overridable."""
    settings: dict[str, Any] = {
        "enabled": True,
        "experiment_mode": ExperimentMode.ADAPTIVE,
        "mode": AdaptationPolicyMode.CONSERVATIVE_RULE_BASED,
        "minimum_persistence_windows": 3,
        "cooldown_windows": 6,
        "difficulty": DifficultyBounds(minimum=1, maximum=5, step=1),
        "max_adaptations_per_session": 10,
        "conflict_resolution": ConflictResolution.HOLD,
        "regression_mapping_enabled": False,
    }
    settings.update(overrides)
    return AdaptationPolicyConfiguration(**settings)


def make_evidence(
    target_name: TargetName,
    class_label: str | None,
    *,
    eligible: bool = True,
    status: str | None = None,
    window_id: str = "w000",
    session_id: str = SESSION,
    subject_id: str = SUBJECT,
) -> AdaptationTargetEvidence:
    """Milestone 7 evidence for one target of one window."""
    resolved = status or (ELIGIBLE if eligible else BLOCKED_BY_CONFIDENCE)
    return build_evidence(
        target_name=target_name,
        class_label=class_label,
        status=resolved,
        window_id=window_id,
        subject_id=subject_id,
        session_id=session_id,
    )


def make_input(
    *,
    engagement: str | None = "medium",
    cognitive_load: str | None = "medium",
    engagement_status: str = ELIGIBLE,
    cognitive_load_status: str = ELIGIBLE,
    window_order: int = 0,
    window_id: str | None = None,
    session_id: str = SESSION,
    subject_id: str = SUBJECT,
    current_difficulty: int | None = 3,
    omit_engagement: bool = False,
    omit_cognitive_load: bool = False,
) -> AdaptationInput:
    """One policy input, with either target optionally omitted entirely."""
    identifier = window_id or f"w{window_order:03d}"
    return AdaptationInput(
        session_id=session_id,
        subject_id=subject_id,
        window_id=identifier,
        window_order=window_order,
        current_difficulty=current_difficulty,
        engagement=(
            None
            if omit_engagement
            else make_evidence(
                TargetName.ENGAGEMENT_CLASS,
                engagement,
                status=engagement_status,
                window_id=identifier,
                session_id=session_id,
                subject_id=subject_id,
            )
        ),
        cognitive_load=(
            None
            if omit_cognitive_load
            else make_evidence(
                TargetName.COGNITIVE_LOAD_CLASS,
                cognitive_load,
                status=cognitive_load_status,
                window_id=identifier,
                session_id=session_id,
                subject_id=subject_id,
            )
        ),
        is_synthetic=True,
        scientific_evaluation_eligible=False,
    )


def run_sequence(
    inputs: list[AdaptationInput],
    configuration: AdaptationPolicyConfiguration | None = None,
    state: AdaptationPolicyState | None = None,
) -> list[AdaptationPolicyDecision]:
    """Feed a list of inputs through the policy, threading the state."""
    from engagevr.adaptation.policy import evaluate_policy

    config = configuration or make_configuration()
    current = state or AdaptationPolicyState.for_session(inputs[0].session_id)
    decisions: list[AdaptationPolicyDecision] = []
    for policy_input in inputs:
        decision = evaluate_policy(policy_input, current, config)
        current = decision.state_after
        decisions.append(decision)
    return decisions


def steady_inputs(
    engagement: str,
    cognitive_load: str,
    count: int,
    *,
    start: int = 0,
    difficulty: int | None = 3,
    session_id: str = SESSION,
) -> list[AdaptationInput]:
    """``count`` consecutive windows all reading the same pair of states."""
    return [
        make_input(
            engagement=engagement,
            cognitive_load=cognitive_load,
            window_order=start + index,
            current_difficulty=difficulty,
            session_id=session_id,
        )
        for index in range(count)
    ]


def build_proposal(**overrides: Any) -> AdaptationProposal:
    """A structurally valid decrease proposal, overridable field by field."""
    engagement_gate = make_evidence(TargetName.ENGAGEMENT_CLASS, "medium").gate
    load_gate = make_evidence(TargetName.COGNITIVE_LOAD_CLASS, "high").gate
    fields: dict[str, Any] = {
        "proposal_id": "prop-test",
        "session_id": SESSION,
        "subject_id": SUBJECT,
        "window_id": "w000",
        "window_order": 0,
        "direction": AdaptationDirection.DECREASE,
        "current_difficulty": 3,
        "requested_difficulty": 2,
        "proposed_difficulty": 2,
        "step": 1,
        "clamping_applied": False,
        "minimum_difficulty": 1,
        "maximum_difficulty": 5,
        "persistence_count": 3,
        "required_persistence_windows": 3,
        "adaptation_index": 1,
        "engagement_gate": engagement_gate,
        "cognitive_load_gate": load_gate,
        "policy_mode": AdaptationPolicyMode.CONSERVATIVE_RULE_BASED,
        "configuration_fingerprint": "fp",
        "is_synthetic": True,
        "scientific_evaluation_eligible": False,
    }
    fields.update(overrides)
    return AdaptationProposal(**fields)


def _suggestion(
    target_name: TargetName,
    state: OrdinalState | None,
    gate_decision: AdaptationGateDecision | None,
    direction: AdaptationDirection,
) -> AdaptationTargetSuggestion:
    return AdaptationTargetSuggestion(
        target_name=target_name,
        evidence_available=gate_decision is not None,
        gate_decision=gate_decision,
        state=state,
        state_source=(
            TargetStateSource.CLASSIFICATION_CLASS if state is not None else None
        ),
        suggested_direction=direction,
        unavailable_reason=(
            None
            if gate_decision is not None
            else AdaptationPolicyReason.INSUFFICIENT_EVIDENCE
        ),
    )


def build_decision(**overrides: Any) -> AdaptationPolicyDecision:
    """A structurally valid decision, overridable field by field."""
    state = AdaptationPolicyState.for_session(SESSION)
    fields: dict[str, Any] = {
        "kind": AdaptationDecisionKind.HOLD,
        "session_id": SESSION,
        "subject_id": SUBJECT,
        "window_id": "w000",
        "window_order": 0,
        "reasons": (AdaptationPolicyReason.COOLDOWN_ACTIVE,),
        "engagement": _suggestion(
            TargetName.ENGAGEMENT_CLASS,
            OrdinalState.MEDIUM,
            AdaptationGateDecision.ELIGIBLE,
            AdaptationDirection.HOLD,
        ),
        "cognitive_load": _suggestion(
            TargetName.COGNITIVE_LOAD_CLASS,
            OrdinalState.HIGH,
            AdaptationGateDecision.ELIGIBLE,
            AdaptationDirection.DECREASE,
        ),
        "conflict": False,
        "conflict_resolution": None,
        "resolved_direction": AdaptationDirection.HOLD,
        "resolution_note": "test",
        "current_difficulty": 3,
        "cooldown_remaining_before": 0,
        "cooldown_remaining_after": 0,
        "persistence_count_before": 0,
        "persistence_count_after": 0,
        "adaptation_budget_used": 0,
        "adaptation_budget_total": 10,
        "proposal": None,
        "state_before": state,
        "state_after": state,
        "experiment_mode": ExperimentMode.ADAPTIVE,
        "policy_mode": AdaptationPolicyMode.CONSERVATIVE_RULE_BASED,
        "configuration_fingerprint": "fp",
        "is_synthetic": True,
        "scientific_evaluation_eligible": False,
    }
    fields.update(overrides)
    return AdaptationPolicyDecision(**fields)


def gate_of(evidence: AdaptationTargetEvidence) -> AdaptationGateRecord:
    """The gate record of a piece of evidence."""
    return evidence.gate


__all__ = [
    "SESSION",
    "SUBJECT",
    "build_decision",
    "build_proposal",
    "gate_of",
    "make_configuration",
    "make_evidence",
    "make_input",
    "run_sequence",
    "steady_inputs",
]
