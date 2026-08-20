"""Proposal lifecycle: five states that are never collapsed into one.

A proposal has been *recommended*.  A built command exists as an object.
A dispatched command left this process.  An acknowledged command reached
a task client and the client answered.  An applied command is one the
client said it applied.  These are five different facts, and treating any
of them as evidence of a later one would let "the policy recommended
something" be reported as "the environment changed".

Every transition here is a pure function from one
:class:`~engagevr.schemas.adaptation_policy.AdaptationHistoryEntry` to the
next.  In particular :func:`record_acknowledgement` requires a real
Milestone 4 :class:`~engagevr.protocol.messages.AdaptationAcknowledgementPayload`;
there is no way to assert ``applied`` without one.
"""

from __future__ import annotations

from datetime import datetime

from engagevr.protocol.messages import (
    AdaptationAcknowledgementPayload,
    AdaptationCommandName,
    AdaptationCommandPayload,
)
from engagevr.schemas.adaptation_policy import (
    AdaptationHistoryEntry,
    AdaptationLifecycleStatus,
    AdaptationPolicyError,
    AdaptationProposal,
)


def start_history_entry(proposal: AdaptationProposal) -> AdaptationHistoryEntry:
    """The history entry for a proposal that has only been recommended.

    ``expected_command`` and ``expected_value`` record what a command
    *would* carry.  They are an expectation, not a sent message: the entry
    starts at :attr:`AdaptationLifecycleStatus.PROPOSED` and carries no
    command id.
    """
    return AdaptationHistoryEntry(
        proposal_id=proposal.proposal_id,
        session_id=proposal.session_id,
        window_id=proposal.window_id,
        window_order=proposal.window_order,
        direction=proposal.direction,
        expected_command=AdaptationCommandName.SET_DIFFICULTY.value,
        expected_value=proposal.proposed_difficulty,
        status=AdaptationLifecycleStatus.PROPOSED,
        is_synthetic=proposal.is_synthetic,
        scientific_evaluation_eligible=False,
    )


def record_command_built(
    entry: AdaptationHistoryEntry, command: AdaptationCommandPayload
) -> AdaptationHistoryEntry:
    """Advance an entry to ``command_built``.

    Building a command is not sending one, so no dispatch time is
    recorded here and the status stops short of ``dispatched``.
    """
    if entry.status is not AdaptationLifecycleStatus.PROPOSED:
        raise AdaptationPolicyError(
            f"proposal {entry.proposal_id!r} is already {entry.status.value!r}; "
            "a command is built once"
        )
    if command.command is not AdaptationCommandName.SET_DIFFICULTY:
        raise AdaptationPolicyError(
            f"proposal {entry.proposal_id!r} expects "
            f"{entry.expected_command!r} but the command is "
            f"{command.command.value!r}"
        )
    if command.value != entry.expected_value:
        raise AdaptationPolicyError(
            f"proposal {entry.proposal_id!r} expects difficulty "
            f"{entry.expected_value} but the command carries {command.value!r}"
        )
    return entry.model_copy(
        update={
            "status": AdaptationLifecycleStatus.COMMAND_BUILT,
            "command_id": command.command_id,
        }
    )


def record_dispatch(
    entry: AdaptationHistoryEntry, *, dispatched_at_utc: datetime
) -> AdaptationHistoryEntry:
    """Advance an entry to ``dispatched``.

    Called only by a caller that actually sent the command.  Nothing in
    :mod:`engagevr.adaptation` calls it: this package builds commands and
    stops.
    """
    if entry.status is not AdaptationLifecycleStatus.COMMAND_BUILT:
        raise AdaptationPolicyError(
            f"proposal {entry.proposal_id!r} is {entry.status.value!r}; only a "
            "built command can be dispatched"
        )
    return entry.model_copy(
        update={
            "status": AdaptationLifecycleStatus.DISPATCHED,
            "dispatched_at_utc": dispatched_at_utc,
        }
    )


def record_acknowledgement(
    entry: AdaptationHistoryEntry,
    acknowledgement: AdaptationAcknowledgementPayload,
) -> AdaptationHistoryEntry:
    """Record a task client's answer, and only what it actually said.

    An accepted acknowledgement carrying ``applied_at_utc`` becomes
    ``applied``; an accepted acknowledgement without one stops at
    ``acknowledged``; a rejection becomes ``rejected`` with the client's
    own stated reason.  No branch here infers that a change took effect.
    """
    if entry.status not in (
        AdaptationLifecycleStatus.COMMAND_BUILT,
        AdaptationLifecycleStatus.DISPATCHED,
    ):
        raise AdaptationPolicyError(
            f"proposal {entry.proposal_id!r} is {entry.status.value!r}; only a "
            "built or dispatched command can be acknowledged"
        )
    if entry.command_id != acknowledgement.command_id:
        raise AdaptationPolicyError(
            f"acknowledgement names command "
            f"{acknowledgement.command_id!r}, not {entry.command_id!r}; an "
            "acknowledgement is never attached to a different command"
        )

    if not acknowledgement.accepted:
        return entry.model_copy(
            update={
                "status": AdaptationLifecycleStatus.REJECTED,
                "acknowledged": False,
                "acknowledgement_duplicate": acknowledgement.duplicate,
                "rejection_reason": acknowledgement.rejection_reason,
            }
        )
    status = (
        AdaptationLifecycleStatus.APPLIED
        if acknowledgement.applied_at_utc is not None
        else AdaptationLifecycleStatus.ACKNOWLEDGED
    )
    return entry.model_copy(
        update={
            "status": status,
            "acknowledged": True,
            "acknowledgement_duplicate": acknowledgement.duplicate,
            "applied_at_utc": acknowledgement.applied_at_utc,
        }
    )


__all__ = [
    "record_acknowledgement",
    "record_command_built",
    "record_dispatch",
    "start_history_entry",
]
