"""Interpretable baseline modelling for EngageVR (Milestone 5).

This package trains and evaluates classical models on windowed feature
datasets.  It contains no fusion architecture, no temporal model, no
personalisation, no online inference, and no abstention policy; those are
later milestones.

Every evaluation is **grouped**.  When several windows share a
participant or a session, an ungrouped split puts the same person on both
sides of the boundary and the resulting score measures memorisation.  The
splitters here refuse to fall back to row-level shuffling: they fail with
an actionable error instead.
"""

from __future__ import annotations

from engagevr.training.splits import (
    GroupOverlapError,
    SplitConfigurationError,
    audit_split,
    build_splits,
    choose_group_field,
)

__all__ = [
    "GroupOverlapError",
    "SplitConfigurationError",
    "audit_split",
    "build_splits",
    "choose_group_field",
]
