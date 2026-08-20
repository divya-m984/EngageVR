"""Translating an approved proposal into an existing protocol command.

This module is the boundary between *what the policy recommends* and
*what could be put on the wire*.  It builds an object and returns it.  It
opens no socket, imports no broker, and has no ``send``: everything it
touches is a pure data structure, so a unit test can exercise the whole
path from prediction to command payload with no server, no WebSocket, no
Unity, and no task simulator running.

Protocol scope
--------------
No wire vocabulary is added.  The policy reasons internally about a
:class:`~engagevr.schemas.adaptation_policy.AdaptationDirection`, and this
module resolves that direction into the difficulty level the proposal
already computed, then builds the **existing** Milestone 4
:attr:`~engagevr.protocol.messages.AdaptationCommandName.SET_DIFFICULTY`
command.  ``pause_task`` and ``resume_task`` exist in the protocol and are
deliberately not used: the only specification rule that would call for a
break is a fatigue rule, and no fatigue estimator exists in this project.

Provenance
----------
``AdaptationCommandPayload`` has no free-form provenance field beyond
``reason`` (512 characters), and this milestone does not extend it.  The
proposal id and the Milestone 7 source prediction ids are therefore
written into ``reason`` in a compact, bounded form, and the full
provenance lives in the adaptation trace, which is where an auditor reads
it.  ``is_manual`` is set to ``False``, which is what distinguishes a
policy-derived command from the manual and scripted commands Milestone 4
issues.
"""

from __future__ import annotations

from datetime import datetime

from engagevr.protocol.messages import (
    TASK_CLIENT_ROLES,
    AdaptationCommandName,
    AdaptationCommandPayload,
    ClientRole,
)
from engagevr.schemas.adaptation_policy import (
    ACTING_DIRECTIONS,
    AdaptationDecisionKind,
    AdaptationPolicyDecision,
    AdaptationProposal,
)
from engagevr.schemas.uncertainty import AdaptationGateDecision

#: Longest ``reason`` the protocol accepts.
_MAXIMUM_REASON_LENGTH = 512


class AdaptationCommandBuildError(ValueError):
    """A command cannot be built from what was supplied.

    Raised rather than returning ``None`` so that a caller cannot treat a
    refusal as an empty command and send nothing while believing it sent
    something.
    """


def default_command_id(proposal: AdaptationProposal) -> str:
    """A deterministic command id derived from the proposal id.

    No clock and no random component participates, so replaying one
    sequence produces the same command ids.  The task client's idempotency
    rule keys on ``command_id``, so a retransmitted command for the same
    proposal is absorbed rather than double-stepping the difficulty.
    """
    return f"cmd-{proposal.proposal_id}"


def build_adaptation_command(
    proposal: AdaptationProposal,
    *,
    issued_at_utc: datetime,
    command_id: str | None = None,
    target_role: ClientRole = ClientRole.SIMULATOR,
    target_client_id: str | None = None,
    expires_at_utc: datetime | None = None,
) -> AdaptationCommandPayload:
    """Build the existing ``set_difficulty`` command for one proposal.

    Nothing is sent.  The returned payload is an ordinary value that a
    caller may inspect, store, or discard.

    Parameters
    ----------
    proposal:
        A proposal that already survived the Milestone 7 gate, the
        mapping, persistence, cooldown, the budget, and the bounds.
    issued_at_utc:
        The issue instant, supplied by the caller.  The policy reads no
        clock, so this is the only place a wall-clock value enters.
    command_id:
        Overrides :func:`default_command_id`.
    expires_at_utc:
        Optional expiry, using the protocol's own semantics: the task
        client rejects a command that reaches it after this instant.

    Raises
    ------
    AdaptationCommandBuildError
        If the proposal names no acting direction, if either Milestone 7
        gate is not eligible, if the target difficulty is out of bounds,
        or if the target role is not a task client role.
    """
    if proposal.direction not in ACTING_DIRECTIONS:  # pragma: no cover - schema-barred
        raise AdaptationCommandBuildError(
            f"proposal {proposal.proposal_id!r} names direction "
            f"{proposal.direction.value!r}, which is not a change to command"
        )
    for name, gate in (
        ("engagement", proposal.engagement_gate),
        ("cognitive_load", proposal.cognitive_load_gate),
    ):
        if gate.decision is not AdaptationGateDecision.ELIGIBLE:
            raise AdaptationCommandBuildError(
                f"proposal {proposal.proposal_id!r} carries a "
                f"{gate.decision.value!r} Milestone 7 gate for {name}; a "
                "blocked window has no command form and there is no override"
            )
    if not (
        proposal.minimum_difficulty
        <= proposal.proposed_difficulty
        <= proposal.maximum_difficulty
    ):  # pragma: no cover - schema-barred
        raise AdaptationCommandBuildError(
            f"proposal {proposal.proposal_id!r} targets difficulty "
            f"{proposal.proposed_difficulty}, outside its own bounds"
        )
    if target_role not in TASK_CLIENT_ROLES:
        allowed = ", ".join(sorted(r.value for r in TASK_CLIENT_ROLES))
        raise AdaptationCommandBuildError(
            f"target_role must be a task client role ({allowed}); got "
            f"{target_role.value!r}"
        )

    return AdaptationCommandPayload(
        command_id=command_id or default_command_id(proposal),
        command=AdaptationCommandName.SET_DIFFICULTY,
        value=proposal.proposed_difficulty,
        reason=_reason_text(proposal),
        issued_at_utc=issued_at_utc,
        expires_at_utc=expires_at_utc,
        target_role=target_role,
        target_client_id=target_client_id,
        # Milestone 8 commands are policy-derived, not manual or scripted.
        is_manual=False,
    )


def build_command_for_decision(
    decision: AdaptationPolicyDecision,
    *,
    issued_at_utc: datetime,
    command_id: str | None = None,
    target_role: ClientRole = ClientRole.SIMULATOR,
    target_client_id: str | None = None,
    expires_at_utc: datetime | None = None,
) -> AdaptationCommandPayload:
    """Build the command for a decision, refusing every hold.

    A hold is a decision, not a failure — but it is a decision *not* to
    change anything, so it has no command form.  Refusing here is what
    keeps "the policy held" from becoming "a command with no effect was
    sent".
    """
    if decision.kind is not AdaptationDecisionKind.PROPOSE_ADAPTATION:
        reasons = ", ".join(r.value for r in decision.reasons)
        raise AdaptationCommandBuildError(
            f"window {decision.window_id!r} produced a hold ({reasons}); a "
            "hold carries no command payload and none can be built for it"
        )
    assert decision.proposal is not None  # guaranteed by the decision schema
    return build_adaptation_command(
        decision.proposal,
        issued_at_utc=issued_at_utc,
        command_id=command_id,
        target_role=target_role,
        target_client_id=target_client_id,
        expires_at_utc=expires_at_utc,
    )


def _reason_text(proposal: AdaptationProposal) -> str:
    """The bounded audit note carried on the wire.

    It states what changed, on what evidence, and under which rule.  It
    makes no claim that the change improves engagement, reduces cognitive
    load, or benefits anyone.
    """
    text = (
        f"conservative_rule_based policy proposed {proposal.direction.value} "
        f"{proposal.current_difficulty}->{proposal.proposed_difficulty} after "
        f"{proposal.persistence_count} consecutive supporting window(s); "
        f"proposal={proposal.proposal_id}; "
        f"engagement_gate={proposal.engagement_gate.source_prediction_id}; "
        f"load_gate={proposal.cognitive_load_gate.source_prediction_id}; "
        "ENGINEERING DEMONSTRATION RULE, not validated with participants"
    )
    if len(text) > _MAXIMUM_REASON_LENGTH:
        text = text[: _MAXIMUM_REASON_LENGTH - 3] + "..."
    return text


__all__ = [
    "AdaptationCommandBuildError",
    "build_adaptation_command",
    "build_command_for_decision",
    "default_command_id",
]
