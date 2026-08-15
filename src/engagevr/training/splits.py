"""Leakage-safe grouped splitting.

Grouping priority
-----------------
1. ``subject_id`` whenever two or more distinct subjects are present.
   This is the strongest available grouping: it keeps every session of a
   person on one side of every boundary.
2. ``session_id`` when subject identifiers are absent or constant.
3. Refusal, when neither field yields at least two distinct groups.  A
   dataset with one group cannot be split into independent train and test
   portions, and pretending otherwise produces a number that means
   nothing.

There is deliberately **no** row-level fallback.  ``KFold``,
``StratifiedKFold``, and ``train_test_split`` are never reachable from
this module: with several windows per session they place the same
session in both portions, and the resulting score measures memorisation
of that session rather than generalisation to a new one.

Calibration groups are carved out of the **training** groups of each
outer fold.  The outer test groups are never touched before the final
evaluation of that fold.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from engagevr.schemas.experiments import (
    FoldAssignment,
    GroupField,
    SplitManifest,
    SplitStrategy,
)
from engagevr.schemas.targets import TaskType


class SplitConfigurationError(ValueError):
    """A dataset cannot be split defensibly under the requested settings."""


class GroupOverlapError(ValueError):
    """A split places one group on both sides of a boundary."""


def choose_group_field(
    subject_ids: Sequence[str] | None,
    session_ids: Sequence[str],
) -> tuple[GroupField, str]:
    """Choose the grouping column and state why.

    Raises
    ------
    SplitConfigurationError
        If neither field yields at least two independent groups.
    """
    distinct_sessions = sorted(set(session_ids))
    if subject_ids is not None:
        distinct_subjects = sorted(set(subject_ids))
        if len(distinct_subjects) >= 2:
            sessions_per_subject: dict[str, set[str]] = {}
            for subject, session in zip(subject_ids, session_ids, strict=True):
                sessions_per_subject.setdefault(subject, set()).add(session)
            repeated = sum(
                1 for sessions in sessions_per_subject.values() if len(sessions) > 1
            )
            if repeated:
                reason = (
                    f"{repeated} of {len(distinct_subjects)} subjects contribute "
                    "more than one session, so grouping by session would place "
                    "the same subject in both training and test portions"
                )
            else:
                reason = (
                    f"{len(distinct_subjects)} subjects contribute one session "
                    "each; subject grouping is used because it remains correct "
                    "if further sessions are added later"
                )
            return GroupField.SUBJECT_ID, reason

    if len(distinct_sessions) >= 2:
        return (
            GroupField.SESSION_ID,
            "no usable subject identifier is present, so sessions are the "
            "strongest defensible grouping available",
        )

    raise SplitConfigurationError(
        "cannot split this dataset: it has "
        f"{len(set(subject_ids)) if subject_ids is not None else 0} distinct "
        f"subject(s) and {len(distinct_sessions)} distinct session(s). At "
        "least two independent groups are required. Row-level splitting is "
        "not offered: with several windows per session it would place the "
        "same session in both portions and the resulting score would measure "
        "memorisation rather than generalisation."
    )


def _stratification_feasible(
    groups: Sequence[str], labels: Sequence[str], n_splits: int
) -> tuple[bool, str]:
    """Whether every class has enough distinct groups to be stratified."""
    groups_per_class: dict[str, set[str]] = {}
    for group, label in zip(groups, labels, strict=True):
        groups_per_class.setdefault(label, set()).add(group)
    thin = {
        label: len(members)
        for label, members in groups_per_class.items()
        if len(members) < n_splits
    }
    if thin:
        detail = ", ".join(
            f"{label!r} appears in {count} group(s)"
            for label, count in sorted(thin.items())
        )
        return False, (
            "stratified grouped splitting is not feasible because "
            f"{detail}, which is fewer than the {n_splits} requested folds; "
            "a non-stratified grouped splitter is used instead"
        )
    return True, (
        "every class appears in at least as many independent groups as there "
        "are folds, so the class distribution can be preserved across folds "
        "without breaking grouping"
    )


def build_splits(
    *,
    group_values: Sequence[str],
    session_ids: Sequence[str],
    task_type: TaskType,
    group_field: GroupField,
    group_field_reason: str,
    n_splits: int,
    random_seed: int,
    class_labels: Sequence[str] | None = None,
    numeric_targets: Sequence[float] | None = None,
    calibration_group_fraction: float = 0.0,
) -> SplitManifest:
    """Build the outer folds and, optionally, per-fold calibration groups.

    Raises
    ------
    SplitConfigurationError
        If there are fewer independent groups than requested folds, or the
        target information required for the task type is missing.
    """
    if n_splits < 2:
        raise SplitConfigurationError(
            f"n_splits must be at least 2; got {n_splits}. A single fold has "
            "no held-out portion."
        )
    distinct_groups = sorted(set(group_values))
    if len(distinct_groups) < n_splits:
        raise SplitConfigurationError(
            f"{len(distinct_groups)} independent {group_field.value} group(s) "
            f"cannot support {n_splits} folds. Reduce --folds to at most "
            f"{len(distinct_groups)}, or collect more groups. The split will "
            "not be weakened to fit the requested fold count."
        )
    if task_type is TaskType.CLASSIFICATION and class_labels is None:
        raise SplitConfigurationError("classification splitting requires class labels")
    if task_type is TaskType.REGRESSION and numeric_targets is None:
        raise SplitConfigurationError(
            "regression splitting requires numeric target values"
        )

    groups = np.asarray(group_values, dtype=object)
    indices = np.arange(len(group_values))

    if task_type is TaskType.CLASSIFICATION:
        assert class_labels is not None  # narrowed above
        feasible, strategy_reason = _stratification_feasible(
            group_values, class_labels, n_splits
        )
        if feasible:
            strategy = SplitStrategy.STRATIFIED_GROUP_K_FOLD
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_seed
            )
            targets = np.asarray(class_labels, dtype=object)
        else:
            strategy = SplitStrategy.GROUP_K_FOLD
            splitter = GroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_seed
            )
            targets = np.zeros(len(indices))
    else:
        # Regression uses grouped splitting without stratification bins.
        # Deterministic bins are permitted by the design but are not used
        # here: choosing bin edges is an undocumented modelling decision
        # unless the edges themselves are justified, and no such
        # justification exists for an unvalidated target.
        strategy = SplitStrategy.GROUP_K_FOLD
        strategy_reason = (
            "grouped k-fold without stratification bins: the target is "
            "continuous and no documented, defensible bin edges exist for it"
        )
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        targets = np.asarray(numeric_targets, dtype=float)

    all_classes = sorted(set(class_labels)) if class_labels is not None else []
    folds: list[FoldAssignment] = []

    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(indices.reshape(-1, 1), targets, groups)
    ):
        train_groups = sorted({str(groups[i]) for i in train_idx})
        test_groups = sorted({str(groups[i]) for i in test_idx})
        calibration_groups = _select_calibration_groups(
            train_groups,
            fraction=calibration_group_fraction,
            random_seed=random_seed,
            fold_index=fold_index,
        )

        valid = True
        invalid_reason: str | None = None
        warnings: list[str] = []
        train_distribution: dict[str, int] | None = None
        test_distribution: dict[str, int] | None = None
        train_summary: dict[str, float] | None = None
        test_summary: dict[str, float] | None = None

        if len(test_idx) == 0:
            valid = False
            invalid_reason = "the test portion of this fold is empty"

        if class_labels is not None:
            train_distribution = _distribution(class_labels, train_idx, all_classes)
            test_distribution = _distribution(class_labels, test_idx, all_classes)
            missing_train = [c for c in all_classes if train_distribution[c] == 0]
            missing_test = [c for c in all_classes if test_distribution[c] == 0]
            if missing_train:
                valid = False
                invalid_reason = (
                    "the training portion of this fold contains no example of "
                    f"class(es) {missing_train}; a model fitted here could not "
                    "predict them and the fold is excluded from aggregates"
                )
            if missing_test:
                warnings.append(
                    f"class(es) {missing_test} are absent from the test portion; "
                    "per-class metrics for them are unavailable in this fold"
                )
        if numeric_targets is not None:
            train_summary = _numeric_summary(numeric_targets, train_idx)
            test_summary = _numeric_summary(numeric_targets, test_idx)

        folds.append(
            FoldAssignment(
                fold_index=fold_index,
                train_groups=tuple(train_groups),
                calibration_groups=tuple(calibration_groups),
                test_groups=tuple(test_groups),
                train_row_count=len(train_idx),
                calibration_row_count=int(
                    sum(
                        1
                        for i in train_idx
                        if str(groups[i]) in set(calibration_groups)
                    )
                ),
                test_row_count=len(test_idx),
                train_target_distribution=train_distribution,
                test_target_distribution=test_distribution,
                train_target_summary=train_summary,
                test_target_summary=test_summary,
                valid=valid,
                invalid_reason=invalid_reason,
                warnings=tuple(warnings),
            )
        )

    manifest = SplitManifest(
        strategy=strategy,
        strategy_reason=strategy_reason,
        group_field=group_field,
        group_field_reason=group_field_reason,
        group_count=len(distinct_groups),
        n_splits=n_splits,
        random_seed=random_seed,
        calibration_group_fraction=calibration_group_fraction,
        folds=tuple(folds),
    )
    audit_split(manifest, group_values=group_values, session_ids=session_ids)
    return manifest.model_copy(
        update={
            "audit_passed": True,
            "audit_notes": (
                "no group appears in both the training and test portion of any fold",
                "no calibration group appears in any test portion",
                "every session's rows fall in exactly one test fold",
            ),
        }
    )


def _select_calibration_groups(
    train_groups: Sequence[str],
    *,
    fraction: float,
    random_seed: int,
    fold_index: int,
) -> tuple[str, ...]:
    """Deterministically reserve a fraction of training groups for calibration."""
    if fraction <= 0.0 or len(train_groups) < 2:
        return ()
    count = max(1, math.floor(fraction * len(train_groups)))
    count = min(count, len(train_groups) - 1)
    ordered = sorted(train_groups)
    rng = np.random.default_rng(np.random.PCG64(random_seed + 1_000_003 * fold_index))
    chosen = rng.permutation(len(ordered))[:count]
    return tuple(sorted(ordered[int(i)] for i in chosen))


def _distribution(
    labels: Sequence[str], indices: Sequence[int] | np.ndarray, classes: Sequence[str]
) -> dict[str, int]:
    counter = Counter(labels[int(i)] for i in indices)
    return {label: int(counter.get(label, 0)) for label in classes}


def _numeric_summary(
    values: Sequence[float], indices: Sequence[int] | np.ndarray
) -> dict[str, float]:
    selected = np.asarray([values[int(i)] for i in indices], dtype=float)
    if selected.size == 0:
        return {"count": 0.0}
    return {
        "count": float(selected.size),
        "mean": float(selected.mean()),
        "std": float(selected.std(ddof=0)),
        "minimum": float(selected.min()),
        "maximum": float(selected.max()),
    }


def audit_split(
    manifest: SplitManifest,
    *,
    group_values: Sequence[str],
    session_ids: Sequence[str],
) -> None:
    """Fail if any fold leaks a group, a calibration set, or a session.

    Raises
    ------
    GroupOverlapError
        On the first detected contamination, naming the offending groups.
    """
    session_test_folds: dict[str, set[int]] = {}
    for fold in manifest.folds:
        train = set(fold.train_groups)
        calibration = set(fold.calibration_groups)
        test = set(fold.test_groups)

        overlap = train & test
        if overlap:
            raise GroupOverlapError(
                f"fold {fold.fold_index}: {len(overlap)} group(s) appear in both "
                f"the training and test portions: {sorted(overlap)[:5]}"
            )
        calibration_overlap = calibration & test
        if calibration_overlap:
            raise GroupOverlapError(
                f"fold {fold.fold_index}: calibration groups also appear in the "
                f"test portion: {sorted(calibration_overlap)[:5]}. Calibrating "
                "on the outer test fold makes the calibration metrics "
                "meaningless."
            )
        stray = calibration - train
        if stray:
            raise GroupOverlapError(
                f"fold {fold.fold_index}: calibration groups are not a subset of "
                f"the training groups: {sorted(stray)[:5]}"
            )

        for group, session in zip(group_values, session_ids, strict=True):
            if group in test:
                session_test_folds.setdefault(session, set()).add(fold.fold_index)

    split_sessions = {
        session: sorted(folds)
        for session, folds in session_test_folds.items()
        if len(folds) > 1
    }
    if split_sessions:
        first = next(iter(sorted(split_sessions)))
        raise GroupOverlapError(
            f"session {first!r} has rows in the test portion of folds "
            f"{split_sessions[first]}. Overlapping windows from one session "
            "must stay together, otherwise adjacent windows sharing evidence "
            "land on both sides of the boundary."
        )
