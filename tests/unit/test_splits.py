"""Grouped-splitting tests.

The property that matters: no participant and no session may appear on
both sides of any fold boundary, and the splitter must fail rather than
weaken the split when that is impossible.
"""

from __future__ import annotations

import pytest

from engagevr.schemas.experiments import GroupField, SplitStrategy
from engagevr.schemas.targets import TaskType
from engagevr.training.splits import (
    GroupOverlapError,
    SplitConfigurationError,
    audit_split,
    build_splits,
    choose_group_field,
)


def population(
    subjects: int = 10,
    sessions_per_subject: int = 2,
    windows: int = 4,
    classes: tuple[str, ...] = ("low", "medium", "high"),
) -> tuple[list[str], list[str], list[str], list[float]]:
    subject_ids: list[str] = []
    session_ids: list[str] = []
    labels: list[str] = []
    numeric: list[float] = []
    counter = 0
    for subject in range(subjects):
        for session in range(sessions_per_subject):
            for _window in range(windows):
                subject_ids.append(f"synthetic-subject-{subject:04d}")
                session_ids.append(f"synthetic-session-{subject:04d}-{session:02d}")
                labels.append(classes[counter % len(classes)])
                numeric.append(0.1 * (counter % 10))
                counter += 1
    return subject_ids, session_ids, labels, numeric


def make_splits(**overrides: object):  # type: ignore[no-untyped-def]
    subjects, sessions, labels, numeric = population(
        **{
            k: v
            for k, v in overrides.items()
            if k in {"subjects", "sessions_per_subject", "windows", "classes"}
        }  # type: ignore[arg-type]
    )
    field, reason = choose_group_field(subjects, sessions)
    groups = subjects if field is GroupField.SUBJECT_ID else sessions
    kwargs: dict[str, object] = {
        "group_values": groups,
        "session_ids": sessions,
        "task_type": TaskType.CLASSIFICATION,
        "group_field": field,
        "group_field_reason": reason,
        "n_splits": 3,
        "random_seed": 42,
        "class_labels": labels,
        "calibration_group_fraction": 0.25,
    }
    kwargs.update(
        {
            k: v
            for k, v in overrides.items()
            if k not in {"subjects", "sessions_per_subject", "windows", "classes"}
        }
    )
    manifest = build_splits(**kwargs)  # type: ignore[arg-type]
    return manifest, groups, sessions, labels, numeric


class TestGroupFieldChoice:
    def test_subject_grouping_is_preferred_when_subjects_repeat(self) -> None:
        subjects, sessions, _labels, _numeric = population()
        field, reason = choose_group_field(subjects, sessions)
        assert field is GroupField.SUBJECT_ID
        assert "more than one session" in reason

    def test_subject_grouping_is_still_used_with_one_session_each(self) -> None:
        subjects, sessions, _labels, _numeric = population(sessions_per_subject=1)
        field, reason = choose_group_field(subjects, sessions)
        assert field is GroupField.SUBJECT_ID
        assert "one session" in reason

    def test_session_grouping_when_no_subject_identifier_exists(self) -> None:
        _subjects, sessions, _labels, _numeric = population()
        field, reason = choose_group_field(None, sessions)
        assert field is GroupField.SESSION_ID
        assert "no usable subject identifier" in reason

    def test_session_grouping_when_every_row_shares_one_subject(self) -> None:
        _subjects, sessions, _labels, _numeric = population()
        field, _reason = choose_group_field(["only-subject"] * len(sessions), sessions)
        assert field is GroupField.SESSION_ID

    def test_a_single_group_is_refused_with_no_random_fallback(self) -> None:
        with pytest.raises(SplitConfigurationError) as excinfo:
            choose_group_field(["s"] * 10, ["one-session"] * 10)
        message = str(excinfo.value)
        assert "At least two independent groups" in message
        assert "Row-level splitting is not offered" in message


class TestNoOverlap:
    def test_no_subject_appears_in_train_and_test(self) -> None:
        manifest, _groups, _sessions, _labels, _numeric = make_splits()
        for fold in manifest.folds:
            assert not set(fold.train_groups) & set(fold.test_groups)

    def test_no_session_is_split_across_test_folds(self) -> None:
        manifest, groups, sessions, _labels, _numeric = make_splits()
        placement: dict[str, set[int]] = {}
        for fold in manifest.folds:
            members = set(fold.test_groups)
            for group, session in zip(groups, sessions, strict=True):
                if group in members:
                    placement.setdefault(session, set()).add(fold.fold_index)
        assert all(len(folds) == 1 for folds in placement.values())

    def test_overlapping_windows_of_one_session_stay_together(self) -> None:
        manifest, groups, sessions, _labels, _numeric = make_splits(windows=8)
        for fold in manifest.folds:
            members = set(fold.test_groups)
            for session in set(sessions):
                indices = [i for i, s in enumerate(sessions) if s == session]
                in_test = {groups[i] in members for i in indices}
                assert len(in_test) == 1, session

    def test_calibration_groups_are_a_training_subset(self) -> None:
        manifest, _groups, _sessions, _labels, _numeric = make_splits()
        for fold in manifest.folds:
            assert set(fold.calibration_groups) <= set(fold.train_groups)
            assert not set(fold.calibration_groups) & set(fold.test_groups)

    def test_fit_groups_exclude_calibration_groups(self) -> None:
        manifest, _groups, _sessions, _labels, _numeric = make_splits()
        for fold in manifest.folds:
            assert not set(fold.fit_groups()) & set(fold.calibration_groups)
            assert set(fold.fit_groups()) | set(fold.calibration_groups) == set(
                fold.train_groups
            )

    def test_every_group_is_tested_exactly_once(self) -> None:
        manifest, groups, _sessions, _labels, _numeric = make_splits()
        tested = [g for fold in manifest.folds for g in fold.test_groups]
        assert sorted(tested) == sorted(set(groups))

    def test_the_audit_passes_and_is_recorded(self) -> None:
        manifest, _groups, _sessions, _labels, _numeric = make_splits()
        assert manifest.audit_passed is True
        assert manifest.audit_notes


class TestDeterminism:
    def test_identical_runs_produce_identical_manifests(self) -> None:
        first, *_ = make_splits()
        second, *_ = make_splits()
        assert first.model_dump() == second.model_dump()

    def test_a_different_seed_changes_the_assignment(self) -> None:
        first, *_ = make_splits()
        second, *_ = make_splits(random_seed=7)
        assert first.model_dump() != second.model_dump()

    def test_calibration_groups_are_deterministic(self) -> None:
        first, *_ = make_splits()
        second, *_ = make_splits()
        assert [f.calibration_groups for f in first.folds] == [
            f.calibration_groups for f in second.folds
        ]


class TestStrategySelection:
    def test_stratification_is_used_when_feasible(self) -> None:
        manifest, *_ = make_splits(subjects=12)
        assert manifest.strategy is SplitStrategy.STRATIFIED_GROUP_K_FOLD
        assert "every class appears" in manifest.strategy_reason

    def test_a_thin_class_forces_the_non_stratified_splitter(self) -> None:
        subjects, sessions, labels, _numeric = population(subjects=8)
        # Make "high" appear in exactly one group.
        labels = ["low" if v == "high" else v for v in labels]
        labels[0] = "high"
        field, reason = choose_group_field(subjects, sessions)
        manifest = build_splits(
            group_values=subjects,
            session_ids=sessions,
            task_type=TaskType.CLASSIFICATION,
            group_field=field,
            group_field_reason=reason,
            n_splits=3,
            random_seed=42,
            class_labels=labels,
        )
        assert manifest.strategy is SplitStrategy.GROUP_K_FOLD
        assert "not feasible" in manifest.strategy_reason
        assert "fewer than the 3 requested folds" in manifest.strategy_reason

    def test_regression_uses_grouped_splitting_without_bins(self) -> None:
        subjects, sessions, _labels, numeric = population()
        field, reason = choose_group_field(subjects, sessions)
        manifest = build_splits(
            group_values=subjects,
            session_ids=sessions,
            task_type=TaskType.REGRESSION,
            group_field=field,
            group_field_reason=reason,
            n_splits=3,
            random_seed=42,
            numeric_targets=numeric,
        )
        assert manifest.strategy is SplitStrategy.GROUP_K_FOLD
        assert "without stratification bins" in manifest.strategy_reason
        for fold in manifest.folds:
            assert fold.train_target_summary is not None
            assert fold.test_target_summary is not None


class TestRefusals:
    def test_too_few_groups_for_the_requested_folds_is_refused(self) -> None:
        with pytest.raises(SplitConfigurationError) as excinfo:
            make_splits(subjects=3, n_splits=5)
        message = str(excinfo.value)
        assert "cannot support 5 folds" in message
        assert "will not be weakened" in message

    def test_fewer_than_two_folds_is_refused(self) -> None:
        with pytest.raises(SplitConfigurationError, match="at least 2"):
            make_splits(n_splits=1)

    def test_classification_without_labels_is_refused(self) -> None:
        subjects, sessions, _labels, _numeric = population()
        field, reason = choose_group_field(subjects, sessions)
        with pytest.raises(SplitConfigurationError, match="requires class labels"):
            build_splits(
                group_values=subjects,
                session_ids=sessions,
                task_type=TaskType.CLASSIFICATION,
                group_field=field,
                group_field_reason=reason,
                n_splits=3,
                random_seed=42,
            )

    def test_regression_without_targets_is_refused(self) -> None:
        subjects, sessions, _labels, _numeric = population()
        field, reason = choose_group_field(subjects, sessions)
        with pytest.raises(
            SplitConfigurationError, match="requires numeric target values"
        ):
            build_splits(
                group_values=subjects,
                session_ids=sessions,
                task_type=TaskType.REGRESSION,
                group_field=field,
                group_field_reason=reason,
                n_splits=3,
                random_seed=42,
            )


class TestFoldValidity:
    def test_a_fold_missing_a_training_class_is_marked_invalid(self) -> None:
        subjects, sessions, labels, _numeric = population(subjects=6, windows=2)
        # One subject holds every "high" example, so the fold that tests it
        # has no "high" in training.
        labels = ["low" if v == "high" else v for v in labels]
        for index, subject in enumerate(subjects):
            if subject == "synthetic-subject-0000":
                labels[index] = "high"
        field, reason = choose_group_field(subjects, sessions)
        manifest = build_splits(
            group_values=subjects,
            session_ids=sessions,
            task_type=TaskType.CLASSIFICATION,
            group_field=field,
            group_field_reason=reason,
            n_splits=3,
            random_seed=42,
            class_labels=labels,
        )
        invalid = [fold for fold in manifest.folds if not fold.valid]
        assert invalid
        assert "no example of class" in (invalid[0].invalid_reason or "")

    def test_fold_sizes_and_distributions_are_recorded(self) -> None:
        manifest, *_ = make_splits()
        for fold in manifest.folds:
            assert fold.train_row_count > 0
            assert fold.test_row_count > 0
            assert fold.train_target_distribution is not None
            assert fold.test_target_distribution is not None

    def test_a_missing_test_class_is_a_warning_not_a_failure(self) -> None:
        manifest, *_ = make_splits(subjects=6, windows=2)
        for fold in manifest.folds:
            if fold.warnings:
                assert "absent from the test portion" in fold.warnings[0]


class TestAuditFailsOnContamination:
    def test_the_audit_detects_a_train_test_overlap(self) -> None:
        manifest, groups, sessions, _labels, _numeric = make_splits()
        fold = manifest.folds[0]
        contaminated = fold.model_copy(
            update={"train_groups": (*fold.train_groups, fold.test_groups[0])}
        )
        polluted = manifest.model_copy(
            update={"folds": (contaminated, *manifest.folds[1:])}
        )
        with pytest.raises(GroupOverlapError, match="both"):
            audit_split(polluted, group_values=groups, session_ids=sessions)

    def test_the_audit_detects_calibration_on_the_test_fold(self) -> None:
        manifest, groups, sessions, _labels, _numeric = make_splits()
        fold = manifest.folds[0]
        contaminated = fold.model_copy(
            update={
                "train_groups": (*fold.train_groups, fold.test_groups[0]),
                "calibration_groups": (fold.test_groups[0],),
            }
        )
        polluted = manifest.model_copy(
            update={"folds": (contaminated, *manifest.folds[1:])}
        )
        with pytest.raises(GroupOverlapError):
            audit_split(polluted, group_values=groups, session_ids=sessions)

    def test_the_fold_schema_itself_rejects_an_overlap(self) -> None:
        manifest, *_ = make_splits()
        fold = manifest.folds[0]
        with pytest.raises(ValueError, match="both train and test"):
            fold.model_copy(
                update={"train_groups": (*fold.train_groups, fold.test_groups[0])}
            ).model_validate(
                {
                    **fold.model_dump(),
                    "train_groups": [*list(fold.train_groups), fold.test_groups[0]],
                }
            )

    def test_the_audit_detects_a_session_split_across_folds(self) -> None:
        manifest, groups, sessions, _labels, _numeric = make_splits()
        # Relabel every row as one session while keeping distinct groups:
        # that session now has rows in every test fold.
        one_session = ["shared-session"] * len(sessions)
        with pytest.raises(GroupOverlapError, match="must stay together"):
            audit_split(manifest, group_values=groups, session_ids=one_session)
