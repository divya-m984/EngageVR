"""Synthetic feature-generator tests.

The generator's job is to exercise the pipeline, not to make any model
look good. These tests assert that it is deterministic, that its hidden
latent variables never appear as predictor columns, that missingness and
quality failures are actually produced, and that every row it emits is
permanently labelled SYNTHETIC.
"""

from __future__ import annotations

from collections import Counter

import pytest

from engagevr.features.catalog import FEATURE_CATALOG
from engagevr.features.synthetic import (
    SYNTHETIC_INSTRUMENT,
    SyntheticDatasetConfig,
    generate_synthetic_dataset,
    session_identifier,
    subject_identifier,
)
from engagevr.features.validation import validate_feature_windows
from engagevr.schemas.features import (
    SYNTHETIC_LABEL,
    FeatureModality,
    FeatureWindow,
    SubjectKind,
)
from engagevr.schemas.session import DataSource
from engagevr.schemas.targets import TargetName, TargetSourceType

SMALL = SyntheticDatasetConfig(
    seed=7, subjects=6, sessions_per_subject=2, windows_per_session=4
)


class TestDeterminism:
    def test_the_same_seed_produces_identical_rows(self) -> None:
        first = generate_synthetic_dataset(SMALL)
        second = generate_synthetic_dataset(SMALL)
        assert [r.model_dump() for r in first] == [r.model_dump() for r in second]

    def test_a_different_seed_produces_different_rows(self) -> None:
        other = SMALL.model_copy(update={"seed": 8})
        assert [r.model_dump() for r in generate_synthetic_dataset(SMALL)] != [
            r.model_dump() for r in generate_synthetic_dataset(other)
        ]

    def test_row_count_matches_the_configuration(self) -> None:
        rows = generate_synthetic_dataset(SMALL)
        assert len(rows) == (
            SMALL.subjects * SMALL.sessions_per_subject * SMALL.windows_per_session
        )

    def test_generated_rows_pass_dataset_validation(self) -> None:
        report = validate_feature_windows(
            generate_synthetic_dataset(SMALL), FEATURE_CATALOG
        )
        assert report.subject_count == SMALL.subjects
        assert report.session_count == SMALL.subjects * SMALL.sessions_per_subject


class TestIdentifiers:
    def test_subjects_are_named_as_synthetic_stand_ins(self) -> None:
        assert subject_identifier(0) == "synthetic-subject-0001"
        assert subject_identifier(41) == "synthetic-subject-0042"

    def test_sessions_are_named_deterministically(self) -> None:
        assert session_identifier(0, 0) == "synthetic-session-0001-01"

    def test_no_identifier_reads_as_a_person(self) -> None:
        rows = generate_synthetic_dataset(SMALL)
        for row in rows:
            assert row.subject_id.startswith("synthetic-subject-")
            assert "participant" not in row.subject_id.lower()
            assert row.subject_kind is SubjectKind.SYNTHETIC_SUBJECT


class TestPermanentLabelling:
    def test_every_row_is_labelled_synthetic(self) -> None:
        for row in generate_synthetic_dataset(SMALL):
            assert row.data_source is DataSource.SYNTHETIC
            assert row.synthetic_label == SYNTHETIC_LABEL

    def test_every_target_is_labelled_and_prohibited(self) -> None:
        for row in generate_synthetic_dataset(SMALL):
            assert set(row.targets) == {t.value for t in TargetName}
            for value in row.targets.values():
                assert value.synthetic_label == SYNTHETIC_LABEL
                assert value.scientific_evaluation_permitted is False
                assert value.source_type == TargetSourceType.SYNTHETIC_GENERATOR.value
                assert value.source_instrument == SYNTHETIC_INSTRUMENT
                assert "SYNTHETIC" in value.provenance_notes.upper()

    def test_target_intervals_match_their_window(self) -> None:
        for row in generate_synthetic_dataset(SMALL):
            for value in row.targets.values():
                assert value.interval_start_utc == row.window_start_utc
                assert value.interval_end_utc == row.window_end_utc


class TestNoLatentLeakage:
    #: Names the hidden data-generating process uses internally. None may
    #: appear as a dataset column, or the modelling task becomes trivial.
    LATENT_TOKENS = (
        "latent",
        "e_raw",
        "l_raw",
        "subject_effect",
        "session_effect",
        "drift",
        "ar_e",
        "ar_l",
        "true_engagement",
        "true_load",
        "ground_truth",
    )

    def test_no_feature_column_exposes_a_latent_variable(self) -> None:
        row = generate_synthetic_dataset(SMALL)[0]
        names = set(row.features) | set(row.categorical_features)
        for name in names:
            for token in self.LATENT_TOKENS:
                assert token not in name.lower(), name

    def test_generated_features_are_exactly_the_catalog(self) -> None:
        row = generate_synthetic_dataset(SMALL)[0]
        produced = set(row.features) | set(row.categorical_features)
        assert produced == set(FEATURE_CATALOG.names())

    def test_no_feature_is_a_near_perfect_copy_of_a_target(self) -> None:
        """No feature may be a re-encoding of the answer.

        A single coincidental equality proves nothing; leakage shows up as
        a feature that tracks the target across the whole dataset. The
        check is therefore on the correlation, not on individual values.
        """
        import numpy as np

        rows = generate_synthetic_dataset(
            SMALL.model_copy(update={"subjects": 20, "windows_per_session": 8})
        )
        for target_name in ("engagement_score", "cognitive_load_score"):
            targets = np.asarray(
                [float(r.targets[target_name].numeric_value or 0.0) for r in rows]
            )
            for name in FEATURE_CATALOG.names():
                values = np.asarray(
                    [
                        r.features.get(name)
                        if r.features.get(name) is not None
                        else np.nan
                        for r in rows
                    ],
                    dtype=float,
                )
                mask = np.isfinite(values)
                if mask.sum() < 10 or np.std(values[mask]) == 0.0:
                    continue
                correlation = float(np.corrcoef(values[mask], targets[mask])[0, 1])
                assert abs(correlation) < 0.99, (name, target_name, correlation)

    def test_the_class_targets_are_not_present_as_categorical_features(self) -> None:
        row = generate_synthetic_dataset(SMALL)[0]
        assert set(row.categorical_features) == {"rppg_method"}


class TestStructure:
    def test_the_generator_produces_missing_modalities(self) -> None:
        rows = generate_synthetic_dataset(
            SMALL.model_copy(update={"modality_dropout_probability": 0.3})
        )
        dropped = [row for row in rows if not all(row.modality_available.values())]
        assert dropped, "expected at least one window with a dropped modality"
        for row in dropped:
            for modality, available in row.modality_available.items():
                if available:
                    continue
                assert row.modality_quality[modality] is None
                for entry in FEATURE_CATALOG.by_modality(FeatureModality(modality)):
                    name = entry.canonical_name
                    if name == "window_missing_feature_pct":
                        continue
                    assert row.features.get(name) is None
                    assert row.feature_available[name] is False

    def test_capture_quality_is_never_dropped(self) -> None:
        rows = generate_synthetic_dataset(
            SMALL.model_copy(update={"modality_dropout_probability": 0.9})
        )
        assert all(row.modality_available["quality"] for row in rows)

    def test_poor_quality_windows_lose_the_pulse_estimate(self) -> None:
        rows = generate_synthetic_dataset(
            SMALL.model_copy(update={"poor_quality_probability": 0.9})
        )
        gated = [
            row
            for row in rows
            if row.modality_available["rppg"]
            and row.features["rppg_heart_rate_bpm"] is None
        ]
        assert gated, "expected at least one quality-gated rPPG window"
        for row in gated:
            assert row.features["rppg_unavailable_window_pct"] == pytest.approx(100.0)
            # The diagnostics survive: they are how the rejection is explained.
            assert row.features["rppg_quality_score"] is not None

    def test_missing_feature_percentage_is_always_available(self) -> None:
        for row in generate_synthetic_dataset(SMALL):
            assert row.feature_available["window_missing_feature_pct"] is True
            value = row.features["window_missing_feature_pct"]
            assert value is not None
            assert 0.0 <= value <= 100.0

    def test_response_proportions_sum_to_one(self) -> None:
        for row in generate_synthetic_dataset(SMALL):
            parts = [
                row.features["task_correct_proportion"],
                row.features["task_incorrect_proportion"],
                row.features["task_timeout_proportion"],
            ]
            if any(p is None for p in parts):
                continue
            assert sum(p for p in parts if p is not None) == pytest.approx(1.0)

    def test_order_statistics_are_ordered(self) -> None:
        for row in generate_synthetic_dataset(SMALL):
            low = row.features["task_reaction_time_min_ms"]
            mean = row.features["task_reaction_time_mean_ms"]
            high = row.features["task_reaction_time_max_ms"]
            if None in (low, mean, high):
                continue
            assert low <= mean <= high  # type: ignore[operator]

    def test_values_respect_catalog_ranges(self) -> None:
        for row in generate_synthetic_dataset(SMALL):
            for name, value in row.features.items():
                if value is None:
                    continue
                entry = FEATURE_CATALOG.get(name)
                if entry.value_minimum is not None:
                    assert value >= entry.value_minimum - 1e-9, name
                if entry.value_maximum is not None:
                    assert value <= entry.value_maximum + 1e-9, name


class TestTargetDistribution:
    def test_default_thresholds_produce_all_three_classes(self) -> None:
        rows = generate_synthetic_dataset(
            SMALL.model_copy(update={"subjects": 20, "windows_per_session": 8})
        )
        counts = Counter(r.targets["engagement_class"].class_value for r in rows)
        assert set(counts) == {"low", "medium", "high"}

    def test_imbalance_can_be_requested(self) -> None:
        balanced = generate_synthetic_dataset(
            SMALL.model_copy(update={"subjects": 20, "windows_per_session": 8})
        )
        imbalanced = generate_synthetic_dataset(
            SMALL.model_copy(
                update={
                    "subjects": 20,
                    "windows_per_session": 8,
                    "engagement_class_thresholds": (0.15, 0.30),
                }
            )
        )
        balanced_high = sum(
            1 for r in balanced if r.targets["engagement_class"].class_value == "high"
        )
        imbalanced_high = sum(
            1 for r in imbalanced if r.targets["engagement_class"].class_value == "high"
        )
        assert imbalanced_high > balanced_high

    def test_scores_stay_inside_their_declared_range(self) -> None:
        for row in generate_synthetic_dataset(SMALL):
            for name in ("engagement_score", "cognitive_load_score"):
                value = row.targets[name].numeric_value
                assert value is not None
                assert 0.0 <= value <= 1.0

    def test_labels_carry_observation_noise(self) -> None:
        """A target is a noisy view of the latent, so a perfect fit is impossible.

        Asserted indirectly: two windows with identical rounded feature
        vectors would otherwise always share a label. Here the check is
        that the score is not a step function of the class thresholds.
        """
        rows = generate_synthetic_dataset(
            SMALL.model_copy(update={"subjects": 20, "windows_per_session": 8})
        )
        scores = [r.targets["engagement_score"].numeric_value for r in rows]
        assert len({round(float(s or 0.0), 3) for s in scores}) > len(rows) // 2


class TestConfigurationValidation:
    def test_step_larger_than_duration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            SyntheticDatasetConfig(
                window_duration_seconds=5.0, window_step_seconds=10.0
            )

    def test_unordered_class_thresholds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="0 < low < high < 1"):
            SyntheticDatasetConfig(engagement_class_thresholds=(0.8, 0.2))

    def test_a_single_subject_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            SyntheticDatasetConfig(subjects=1)

    def test_overlap_flag_follows_the_geometry(self) -> None:
        assert SyntheticDatasetConfig(
            window_duration_seconds=10.0, window_step_seconds=5.0
        ).windows_overlap
        assert not SyntheticDatasetConfig(
            window_duration_seconds=10.0, window_step_seconds=10.0
        ).windows_overlap


class TestGroupAndSessionEffects:
    def test_windows_carry_their_subject_and_session(self) -> None:
        rows = generate_synthetic_dataset(SMALL)
        by_session: dict[str, set[str]] = {}
        for row in rows:
            by_session.setdefault(row.session_id, set()).add(row.subject_id)
        assert all(len(subjects) == 1 for subjects in by_session.values())

    def test_each_subject_contributes_multiple_sessions(self) -> None:
        rows = generate_synthetic_dataset(SMALL)
        sessions: dict[str, set[str]] = {}
        for row in rows:
            sessions.setdefault(row.subject_id, set()).add(row.session_id)
        assert all(
            len(values) == SMALL.sessions_per_subject for values in sessions.values()
        )

    def test_subject_means_differ(self) -> None:
        """A group effect must actually be present, or grouped CV proves nothing."""
        rows: tuple[FeatureWindow, ...] = generate_synthetic_dataset(
            SMALL.model_copy(update={"subjects": 20, "windows_per_session": 8})
        )
        by_subject: dict[str, list[float]] = {}
        for row in rows:
            value = row.targets["engagement_score"].numeric_value
            if value is not None:
                by_subject.setdefault(row.subject_id, []).append(value)
        means = [sum(v) / len(v) for v in by_subject.values()]
        assert max(means) - min(means) > 0.1
