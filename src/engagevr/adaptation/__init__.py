"""Adaptation policy (Milestone 8).

Milestone 7 decides whether an already-chosen action *may* be acted upon.
This package decides, for an ELIGIBLE prediction, whether a conservative
adaptation *should* be proposed — and stops there.

The boundary between the two questions is kept by what each module is
allowed to import.  :mod:`engagevr.adaptation.policy` imports schemas and
:mod:`engagevr.adaptation.mapping` only: no transport, no API, no broker,
no WebSocket, no task simulator, and no Milestone 7 inference code.  It
therefore cannot send anything, and it cannot recompute a confidence,
a threshold, or a gate verdict that Milestone 7 already recorded.

Translating an approved proposal into the **existing** Milestone 4
``adaptation_command`` payload is a separate pure function in
:mod:`engagevr.adaptation.command`, which builds an object and returns
it.  Nothing in this package dispatches.

Nothing here is validated with human participants.  See
``docs/ADAPTIVE_ENVIRONMENT.md``.
"""

from __future__ import annotations

from engagevr.adaptation.command import (
    AdaptationCommandBuildError,
    build_adaptation_command,
    build_command_for_decision,
    default_command_id,
)
from engagevr.adaptation.lifecycle import (
    record_acknowledgement,
    record_command_built,
    record_dispatch,
    start_history_entry,
)
from engagevr.adaptation.mapping import (
    MAPPING_TABLE,
    cognitive_load_suggestion,
    engagement_suggestion,
    ordinal_state_from_class,
    ordinal_state_from_value,
    resolve_direction,
)
from engagevr.adaptation.policy import evaluate_policy

__all__ = [
    "MAPPING_TABLE",
    "AdaptationCommandBuildError",
    "build_adaptation_command",
    "build_command_for_decision",
    "cognitive_load_suggestion",
    "default_command_id",
    "engagement_suggestion",
    "evaluate_policy",
    "ordinal_state_from_class",
    "ordinal_state_from_value",
    "record_acknowledgement",
    "record_command_built",
    "record_dispatch",
    "resolve_direction",
    "start_history_entry",
]
