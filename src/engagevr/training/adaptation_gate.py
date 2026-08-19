"""The confidence-aware adaptation gate (Milestone 7).

This module answers exactly one question::

    May an ALREADY-CHOSEN adaptation action be acted upon, given the
    current prediction and evidence state?

It does **not** answer::

    Which adaptation should be chosen?

That distinction is the whole point of the module, so it is enforced by
what is absent as much as by what is present.  There is no action type
here, no action registry, no difficulty level, no scene identifier, no
ordering over actions, no reward, no cooldown, no hysteresis, no state
carried between windows, and no transport client.  A reviewer can
establish that this module cannot choose an adaptation by reading its
imports: it imports no policy, no configuration beyond thresholds it is
handed, no network module, and nothing from :mod:`engagevr.transport`,
:mod:`engagevr.api`, or :mod:`engagevr.task`.

Adaptation *policy* — choosing what to do, and when to stop doing it — is
Milestone 8.

Everything here is pure and deterministic.  The gate consumes
already-computed information: an abstention decision that has already been
made, an evidence-gate result that has already been evaluated, and
thresholds that have already been resolved.  It recomputes none of them,
so a gate decision can never disagree with the decision it is gating.
"""

from __future__ import annotations

from engagevr.schemas.targets import TaskType
from engagevr.schemas.uncertainty import (
    EVIDENCE_REASONS,
    REASON_ORDER,
    AbstentionDecision,
    AbstentionReason,
    AdaptationGateDecision,
    AdaptationGateRecord,
)

#: Stated scope, persisted on every record and in every artifact.
ADAPTATION_GATE_SCOPE = (
    "This gate answers only whether an ALREADY-CHOSEN action may be acted "
    "upon. It does not choose an adaptation, does not rank adaptations, "
    "does not set difficulty or scene content, does not issue any transport "
    "message, does not learn, and has no cooldown, hysteresis, or reward. "
    "Adaptation policy is Milestone 8."
)


def evaluate_adaptation_gate(
    decision: AbstentionDecision,
    *,
    applied_confidence_threshold: float | None = None,
    maximum_interval_width: float | None = None,
    enabled: bool = True,
) -> AdaptationGateRecord:
    """Whether an already-chosen action may be acted upon for this window.

    The gate blocks whenever the selective layer declined to act, and it
    blocks for the *same* reasons that layer recorded, in the same
    canonical order.  It never invents a reason of its own and never
    overrides an abstention: a window the model declined to act on cannot
    become actionable because a gate looked at it a second time.

    ``enabled=False`` does not make everything eligible.  A disabled gate
    still refuses to declare an abstained or unavailable window eligible,
    because that would be a false statement rather than a relaxed policy;
    it simply stops applying any *additional* requirement of its own.

    Parameters
    ----------
    decision:
        The abstention decision already computed for this window.
    applied_confidence_threshold:
        The threshold that decision was taken against, recorded for
        provenance.  It is not re-applied here.
    maximum_interval_width:
        The width policy that decision was taken against, recorded for
        provenance.  It is not re-applied here.
    enabled:
        Whether the gate is active in this run.
    """
    reasons: set[AbstentionReason] = set(decision.reasons)

    confidence_satisfied: bool | None = None
    interval_satisfied: bool | None = None
    if decision.task_type is TaskType.CLASSIFICATION:
        confidence_satisfied = (
            AbstentionReason.BELOW_CONFIDENCE_THRESHOLD not in reasons
            and AbstentionReason.PROBABILITY_CALIBRATION_UNAVAILABLE not in reasons
            and decision.prediction_available
        )
    else:
        interval_satisfied = (
            AbstentionReason.INTERVAL_TOO_WIDE not in reasons
            and AbstentionReason.PREDICTION_INTERVAL_UNAVAILABLE not in reasons
            and decision.prediction_available
        )

    evidence_passed = (
        decision.evidence_gate_passed
        if decision.evidence_gate_passed is not None
        else not any(
            reason in reasons
            for reason in (
                AbstentionReason.INSUFFICIENT_MEASUREMENT_EVIDENCE,
                AbstentionReason.REQUIRED_MODALITY_UNAVAILABLE,
                AbstentionReason.SIGNAL_QUALITY_BELOW_GATE,
            )
        )
    )

    if not decision.prediction_available:
        reasons.add(AbstentionReason.MODEL_PREDICTION_UNAVAILABLE)
    if not evidence_passed and not (set(EVIDENCE_REASONS) & reasons):
        # The evidence gate failed but the decision recorded no evidence
        # reason of its own. Naming the general condition is better than
        # reporting a block with no evidence reason attached to it.
        reasons.add(AbstentionReason.INSUFFICIENT_MEASUREMENT_EVIDENCE)

    blocked = bool(reasons) or decision.abstained or not decision.prediction_available
    if blocked and not reasons:  # pragma: no cover - abstention always has a reason
        reasons.add(AbstentionReason.MODEL_PREDICTION_UNAVAILABLE)

    ordered = tuple(r for r in REASON_ORDER if r in reasons)
    return AdaptationGateRecord(
        window_id=decision.window_id,
        subject_id=decision.subject_id,
        session_id=decision.session_id,
        fold_index=decision.fold_index,
        source_prediction_id=decision.source_prediction_id,
        decision=(
            AdaptationGateDecision.BLOCKED
            if blocked
            else AdaptationGateDecision.ELIGIBLE
        ),
        reasons=ordered,
        prediction_available=decision.prediction_available,
        prediction_abstained=decision.abstained,
        evidence_gate_passed=bool(evidence_passed),
        confidence_requirement_satisfied=confidence_satisfied,
        interval_requirement_satisfied=interval_satisfied,
        applied_confidence_threshold=(
            applied_confidence_threshold if enabled else None
        ),
        maximum_interval_width=maximum_interval_width if enabled else None,
        data_source=decision.data_source,
        is_synthetic=decision.is_synthetic,
        scientific_evaluation_eligible=decision.scientific_evaluation_eligible,
    )


__all__ = ["ADAPTATION_GATE_SCOPE", "evaluate_adaptation_gate"]
