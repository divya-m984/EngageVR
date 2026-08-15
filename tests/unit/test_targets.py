"""Target schema and provenance tests.

The rules under test are the ones that keep an unvalidated project honest:
a target value must state where it came from, a synthetic target can never
be used for scientific evaluation, and a measurement is never silently
promoted into a label.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from engagevr.schemas.session import DataSource
from engagevr.schemas.targets import (
    COGNITIVE_LOAD_CLASSES,
    ENGAGEMENT_CLASSES,
    SYNTHETIC_LABEL,
    TARGET_SPECS,
    TargetName,
    TargetObservation,
    TargetProvenanceError,
    TargetSourceType,
    TargetSpec,
    TaskType,
    UnsupportedTargetError,
    get_target_spec,
    reject_automatic_derivation,
)

START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
END = START + timedelta(seconds=10)


def observation(**overrides: object) -> TargetObservation:
    fields: dict[str, object] = {
        "target_name": TargetName.ENGAGEMENT_CLASS,
        "task_type": TaskType.CLASSIFICATION,
        "class_value": "medium",
        "class_vocabulary": ENGAGEMENT_CLASSES,
        "source_type": TargetSourceType.SUBJECTIVE_SELF_REPORT,
        "source_instrument": "example-instrument-v1",
        "observed_at_utc": END,
        "interval_start_utc": START,
        "interval_end_utc": END,
        "subject_id": "participant-0001",
        "session_id": "session-0001",
        "data_source": DataSource.LIVE,
        "provenance_notes": "recorded immediately after the window closed",
        "scientific_evaluation_permitted": True,
    }
    fields.update(overrides)
    return TargetObservation(**fields)  # type: ignore[arg-type]


class TestTargetSpecs:
    def test_all_four_targets_are_declared(self) -> None:
        assert set(TARGET_SPECS) == set(TargetName)

    def test_classification_targets_declare_ordered_vocabularies(self) -> None:
        assert TARGET_SPECS[TargetName.ENGAGEMENT_CLASS].class_vocabulary == (
            ENGAGEMENT_CLASSES
        )
        assert TARGET_SPECS[TargetName.COGNITIVE_LOAD_CLASS].class_vocabulary == (
            COGNITIVE_LOAD_CLASSES
        )

    def test_regression_targets_declare_a_range(self) -> None:
        spec = TARGET_SPECS[TargetName.ENGAGEMENT_SCORE]
        assert spec.value_minimum == 0.0
        assert spec.value_maximum == 1.0

    def test_classification_spec_requires_a_vocabulary(self) -> None:
        with pytest.raises(ValueError, match="must declare a class_vocabulary"):
            TargetSpec(
                target_name=TargetName.ENGAGEMENT_CLASS,
                task_type=TaskType.CLASSIFICATION,
                description="d",
            )

    def test_regression_spec_requires_a_range(self) -> None:
        with pytest.raises(ValueError, match="value_minimum and value_maximum"):
            TargetSpec(
                target_name=TargetName.ENGAGEMENT_SCORE,
                task_type=TaskType.REGRESSION,
                description="d",
            )

    def test_unknown_target_is_refused(self) -> None:
        with pytest.raises(UnsupportedTargetError, match="unsupported target"):
            get_target_spec("mood_class")


class TestValidObservations:
    def test_valid_classification_label(self) -> None:
        record = observation()
        assert record.class_value == "medium"
        assert record.numeric_value is None
        assert record.scientific_evaluation_permitted is True

    def test_valid_regression_label(self) -> None:
        record = observation(
            target_name=TargetName.ENGAGEMENT_SCORE,
            task_type=TaskType.REGRESSION,
            class_value=None,
            class_vocabulary=None,
            numeric_value=0.42,
            value_minimum=0.0,
            value_maximum=1.0,
        )
        assert record.numeric_value == pytest.approx(0.42)
        assert record.class_value is None


class TestInvalidObservations:
    def test_class_outside_the_vocabulary_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="is not in the vocabulary"):
            observation(class_value="extremely_high")

    def test_score_outside_the_declared_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside the declared range"):
            observation(
                target_name=TargetName.ENGAGEMENT_SCORE,
                task_type=TaskType.REGRESSION,
                class_value=None,
                class_vocabulary=None,
                numeric_value=1.7,
                value_minimum=0.0,
                value_maximum=1.0,
            )

    def test_missing_source_instrument_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            observation(source_instrument="")

    def test_missing_provenance_notes_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            observation(provenance_notes="")

    def test_mismatched_task_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="is a classification target"):
            observation(task_type=TaskType.REGRESSION, numeric_value=0.5)

    def test_classification_with_a_numeric_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not carry numeric_value"):
            observation(numeric_value=0.5)

    def test_reversed_label_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not precede"):
            observation(interval_start_utc=END, interval_end_utc=START)

    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            observation(engagement_is_definitely_high=True)


class TestSyntheticProhibition:
    def test_a_synthetic_target_must_carry_the_label(self) -> None:
        with pytest.raises(ValueError, match="must carry synthetic_label"):
            observation(
                source_type=TargetSourceType.SYNTHETIC_GENERATOR,
                data_source=DataSource.SYNTHETIC,
                scientific_evaluation_permitted=False,
            )

    def test_a_synthetic_target_can_never_permit_scientific_evaluation(self) -> None:
        with pytest.raises(ValueError, match="can never set"):
            observation(
                source_type=TargetSourceType.SYNTHETIC_GENERATOR,
                data_source=DataSource.SYNTHETIC,
                synthetic_label=SYNTHETIC_LABEL,
                scientific_evaluation_permitted=True,
            )

    def test_a_synthetic_data_source_alone_triggers_the_prohibition(self) -> None:
        with pytest.raises(ValueError, match="can never set"):
            observation(
                data_source=DataSource.SYNTHETIC,
                synthetic_label=SYNTHETIC_LABEL,
                scientific_evaluation_permitted=True,
            )

    def test_a_valid_synthetic_target_is_accepted(self) -> None:
        record = observation(
            source_type=TargetSourceType.SYNTHETIC_GENERATOR,
            data_source=DataSource.SYNTHETIC,
            synthetic_label=SYNTHETIC_LABEL,
            scientific_evaluation_permitted=False,
            subject_id="synthetic-subject-0001",
        )
        assert record.synthetic_label == SYNTHETIC_LABEL
        assert record.scientific_evaluation_permitted is False

    def test_a_non_synthetic_target_must_not_carry_the_label(self) -> None:
        with pytest.raises(ValueError, match="must leave synthetic_label unset"):
            observation(synthetic_label=SYNTHETIC_LABEL)


class TestMeasurementToLabelGuard:
    def test_task_accuracy_is_not_automatically_engagement(self) -> None:
        with pytest.raises(TargetProvenanceError) as excinfo:
            reject_automatic_derivation(
                source_group="task", target_name=TargetName.ENGAGEMENT_CLASS
            )
        message = str(excinfo.value)
        assert "engagement_class" in message
        assert "software measurement" in message

    def test_difficulty_is_not_automatically_cognitive_load(self) -> None:
        with pytest.raises(TargetProvenanceError) as excinfo:
            reject_automatic_derivation(
                source_group="task_difficulty",
                target_name=TargetName.COGNITIVE_LOAD_SCORE,
            )
        assert "experimental manipulation" in str(excinfo.value)

    def test_heart_rate_is_not_automatically_a_label(self) -> None:
        with pytest.raises(TargetProvenanceError, match="not engagement"):
            reject_automatic_derivation(
                source_group="rppg", target_name=TargetName.COGNITIVE_LOAD_CLASS
            )

    def test_signal_quality_is_never_a_label(self) -> None:
        with pytest.raises(TargetProvenanceError, match="never be rendered"):
            reject_automatic_derivation(
                source_group="quality", target_name=TargetName.ENGAGEMENT_SCORE
            )

    @pytest.mark.parametrize(
        "group",
        ["task", "task_difficulty", "rppg", "behavioural", "head_pose", "quality"],
    )
    def test_every_measurement_group_is_refused(self, group: str) -> None:
        with pytest.raises(TargetProvenanceError):
            reject_automatic_derivation(
                source_group=group, target_name=TargetName.ENGAGEMENT_CLASS
            )

    def test_an_unknown_group_is_also_refused(self) -> None:
        with pytest.raises(TargetProvenanceError, match="no documented label"):
            reject_automatic_derivation(
                source_group="tea_leaves", target_name=TargetName.ENGAGEMENT_CLASS
            )
