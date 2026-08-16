"""Leakage-safe stacking tests.

The central assertions here are the negative ones: a meta-model fitted on
in-sample expert predictions, or on predictions produced by experts that
saw an outer-test group, must be detected and refused.  A stacker that
silently trains on either produces weights that look fine and mean nothing.

No test here needs a webcam, a model asset, a display server, a network,
Unity, a public dataset, or participant data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engagevr.features.catalog import FEATURE_CATALOG
from engagevr.schemas.fusion import FusionModality, ModalityPrediction
from engagevr.schemas.targets import TargetName, TaskType
from engagevr.training.experts import modality_availability, modality_quality
from engagevr.training.models import get_model_spec
from engagevr.training.preprocessing import ModellingFrame, load_modelling_frame
from engagevr.training.splits import build_splits, choose_group_field
from engagevr.training.stacking import (
    META_COLUMN_PREFIX,
    OutOfFoldProvenance,
    StackingLeakageError,
    assert_out_of_fold,
    build_out_of_fold_matrix,
    fit_stacked_meta_model,
    meta_feature_columns,
    meta_feature_row,
    stacked_predictions,
)

ALL = tuple(FusionModality)
LABELS = ("low", "medium", "high")


def _available(modality: FusionModality) -> ModalityPrediction:
    return ModalityPrediction(
        modality=modality,
        available=True,
        predicted_class="low",
        class_vocabulary=LABELS,
        probabilities=(0.8, 0.1, 0.1),
    )


def _unavailable(modality: FusionModality) -> ModalityPrediction:
    return ModalityPrediction(
        modality=modality, available=False, unavailable_reason="no evidence"
    )


def _provenance(
    row_index: int, group: str, sources: tuple[str, ...]
) -> OutOfFoldProvenance:
    return OutOfFoldProvenance(
        row_index=row_index,
        row_group=group,
        inner_fold_index=0,
        source_groups=sources,
    )


class TestMetaFeatures:
    def test_columns_cover_every_modality_and_class(self) -> None:
        columns = meta_feature_columns(ALL, LABELS, TaskType.CLASSIFICATION)
        assert len(columns) == len(ALL) * (len(LABELS) + 1)
        assert all(c.startswith(META_COLUMN_PREFIX) for c in columns)
        for modality in ALL:
            assert f"{META_COLUMN_PREFIX}{modality.value}__available" in columns

    def test_regression_columns_carry_one_prediction_per_modality(self) -> None:
        columns = meta_feature_columns(ALL, (), TaskType.REGRESSION)
        assert len(columns) == len(ALL) * 2
        assert f"{META_COLUMN_PREFIX}task__prediction" in columns

    def test_an_unavailable_expert_contributes_missing_not_zero(self) -> None:
        predictions = {
            FusionModality.TASK: _available(FusionModality.TASK),
            FusionModality.RPPG: _unavailable(FusionModality.RPPG),
            FusionModality.BEHAVIOURAL: _unavailable(FusionModality.BEHAVIOURAL),
            FusionModality.HEAD_POSE: _unavailable(FusionModality.HEAD_POSE),
        }
        row = meta_feature_row(predictions, ALL, LABELS, TaskType.CLASSIFICATION)
        assert row[f"{META_COLUMN_PREFIX}rppg__probability__low"] is None
        assert row[f"{META_COLUMN_PREFIX}rppg__available"] == 0.0
        assert row[f"{META_COLUMN_PREFIX}task__probability__low"] == pytest.approx(0.8)
        assert row[f"{META_COLUMN_PREFIX}task__available"] == 1.0


class TestLeakageDetection:
    def test_an_out_of_fold_matrix_passes(self) -> None:
        assert_out_of_fold(
            [_provenance(0, "s1", ("s2", "s3"))],
            outer_train_groups=["s1", "s2", "s3"],
            outer_test_groups=["s4"],
        )

    def test_in_sample_meta_training_is_detected(self) -> None:
        with pytest.raises(StackingLeakageError, match="in-sample expert predictions"):
            assert_out_of_fold(
                [_provenance(0, "s1", ("s1", "s2"))],
                outer_train_groups=["s1", "s2", "s3"],
                outer_test_groups=["s4"],
            )

    def test_outer_test_contamination_is_detected(self) -> None:
        with pytest.raises(StackingLeakageError, match="outer-test group"):
            assert_out_of_fold(
                [_provenance(0, "s1", ("s2", "s4"))],
                outer_train_groups=["s1", "s2", "s3"],
                outer_test_groups=["s4"],
            )

    def test_a_row_outside_the_training_portion_is_detected(self) -> None:
        with pytest.raises(StackingLeakageError, match="not an outer-training group"):
            assert_out_of_fold(
                [_provenance(0, "s9", ("s2", "s3"))],
                outer_train_groups=["s1", "s2", "s3"],
                outer_test_groups=["s4"],
            )

    def test_a_source_outside_the_training_portion_is_detected(self) -> None:
        with pytest.raises(StackingLeakageError, match="outside the outer training"):
            assert_out_of_fold(
                [_provenance(0, "s1", ("s2", "s7"))],
                outer_train_groups=["s1", "s2", "s3"],
                outer_test_groups=["s4"],
            )

    def test_the_meta_model_refuses_to_fit_on_in_sample_predictions(self) -> None:
        import pandas as pd

        columns = meta_feature_columns(ALL, LABELS, TaskType.CLASSIFICATION)
        features = pd.DataFrame(
            [dict.fromkeys(columns, 0.5) for _ in range(40)], dtype=float
        )
        with pytest.raises(StackingLeakageError):
            fit_stacked_meta_model(
                features=features,
                row_indices=np.arange(40),
                provenance=[_provenance(i, "s1", ("s1",)) for i in range(40)],
                target_values=np.asarray(["low"] * 20 + ["high"] * 20, dtype=object),
                task_type=TaskType.CLASSIFICATION,
                labels=LABELS,
                outer_train_groups=["s1", "s2"],
                outer_test_groups=["s3"],
                group_values=["s1"] * 40,
                inner_fold_count=3,
                random_seed=42,
            )


class TestOutOfFoldConstruction:
    @pytest.fixture(scope="class")
    def frame(self, m5_dataset: Path) -> ModellingFrame:
        return load_modelling_frame(
            m5_dataset,
            target_name=TargetName.ENGAGEMENT_SCORE,
            catalog=FEATURE_CATALOG,
        )

    def test_it_produces_genuinely_out_of_fold_rows(
        self, frame: ModellingFrame
    ) -> None:
        group_field, reason = choose_group_field(frame.subject_ids, frame.session_ids)
        groups = (
            frame.subject_ids
            if group_field.value == "subject_id"
            else frame.session_ids
        )
        splits = build_splits(
            group_values=groups,
            session_ids=frame.session_ids,
            task_type=TaskType.REGRESSION,
            group_field=group_field,
            group_field_reason=reason,
            n_splits=3,
            random_seed=42,
            numeric_targets=list(frame.numeric_targets()),
            calibration_group_fraction=0.25,
        )
        fold = splits.folds[0]
        members = set(fold.train_groups)
        train_indices = np.asarray(
            [i for i, g in enumerate(groups) if g in members], dtype=int
        )
        availability = {
            m: modality_availability(frame.predictors, m, FEATURE_CATALOG) for m in ALL
        }
        quality = {m: modality_quality(frame.predictors, m) for m in ALL}

        matrix = build_out_of_fold_matrix(
            modalities=ALL,
            predictors=frame.predictors,
            catalog=FEATURE_CATALOG,
            target_values=frame.target_values,
            task_type=TaskType.REGRESSION,
            model_spec=get_model_spec("ridge", TaskType.REGRESSION),
            train_indices=train_indices,
            group_values=groups,
            availability=availability,
            quality=quality,
            labels=(),
            fold_index=0,
            random_seed=42,
            inner_folds=3,
        )
        assert matrix.available
        assert_out_of_fold(
            matrix.provenance,
            outer_train_groups=fold.train_groups,
            outer_test_groups=fold.test_groups,
        )
        assert set(matrix.row_indices.tolist()) == set(train_indices.tolist())
        for record in matrix.provenance:
            assert record.row_group not in record.source_groups

        model, reason_text = fit_stacked_meta_model(
            features=matrix.features,
            row_indices=matrix.row_indices,
            provenance=matrix.provenance,
            target_values=frame.target_values,
            task_type=TaskType.REGRESSION,
            labels=(),
            outer_train_groups=fold.train_groups,
            outer_test_groups=fold.test_groups,
            group_values=groups,
            inner_fold_count=3,
            random_seed=42,
        )
        assert reason_text is None
        assert model is not None
        assert model.meta_model_name == "ridge"
        assert model.probabilities_are_calibrated is False

        predicted, probabilities = stacked_predictions(
            model, matrix.features, TaskType.REGRESSION
        )
        assert probabilities is None
        assert np.isfinite(np.asarray(predicted, dtype=float)).all()

    def test_the_inner_split_is_deterministic(self, frame: ModellingFrame) -> None:
        groups = frame.subject_ids
        members = sorted(set(groups))[:8]
        train_indices = np.asarray(
            [i for i, g in enumerate(groups) if g in set(members)], dtype=int
        )
        availability = {
            m: modality_availability(frame.predictors, m, FEATURE_CATALOG) for m in ALL
        }
        quality = {m: modality_quality(frame.predictors, m) for m in ALL}
        arguments = {
            "modalities": ALL,
            "predictors": frame.predictors,
            "catalog": FEATURE_CATALOG,
            "target_values": frame.target_values,
            "task_type": TaskType.REGRESSION,
            "model_spec": get_model_spec("ridge", TaskType.REGRESSION),
            "train_indices": train_indices,
            "group_values": groups,
            "availability": availability,
            "quality": quality,
            "labels": (),
            "fold_index": 0,
            "random_seed": 42,
            "inner_folds": 3,
        }
        first = build_out_of_fold_matrix(**arguments)  # type: ignore[arg-type]
        second = build_out_of_fold_matrix(**arguments)  # type: ignore[arg-type]
        assert first.provenance == second.provenance
        assert first.features.equals(second.features)

    def test_too_few_groups_refuses_rather_than_leaking(
        self, frame: ModellingFrame
    ) -> None:
        groups = frame.subject_ids
        single = sorted(set(groups))[:1]
        train_indices = np.asarray(
            [i for i, g in enumerate(groups) if g in set(single)], dtype=int
        )
        matrix = build_out_of_fold_matrix(
            modalities=ALL,
            predictors=frame.predictors,
            catalog=FEATURE_CATALOG,
            target_values=frame.target_values,
            task_type=TaskType.REGRESSION,
            model_spec=get_model_spec("ridge", TaskType.REGRESSION),
            train_indices=train_indices,
            group_values=groups,
            availability={
                m: modality_availability(frame.predictors, m, FEATURE_CATALOG)
                for m in ALL
            },
            quality={m: modality_quality(frame.predictors, m) for m in ALL},
            labels=(),
            fold_index=0,
            random_seed=42,
            inner_folds=3,
        )
        assert not matrix.available
        assert "at least 2 independent groups" in (matrix.unavailable_reason or "")


class TestStackedClassificationProbabilities:
    def test_stacked_probabilities_normalise(self) -> None:
        # The stacked regression path is exercised end to end by the shared
        # run fixture; this covers the classification path directly.
        import pandas as pd

        columns = meta_feature_columns(ALL, LABELS, TaskType.CLASSIFICATION)
        rows = []
        targets = []
        for index in range(60):
            fraction = index / 60.0
            row = dict.fromkeys(columns, 0.0)
            for modality in ALL:
                prefix = f"{META_COLUMN_PREFIX}{modality.value}"
                row[f"{prefix}__probability__low"] = 1.0 - fraction
                row[f"{prefix}__probability__medium"] = 0.0
                row[f"{prefix}__probability__high"] = fraction
                row[f"{prefix}__available"] = 1.0
            rows.append(row)
            targets.append("low" if fraction < 0.5 else "high")
        features = pd.DataFrame(rows, columns=list(columns), dtype=float)
        groups = [f"g{index % 6}" for index in range(60)]
        provenance = [
            OutOfFoldProvenance(
                row_index=index,
                row_group=groups[index],
                inner_fold_index=0,
                source_groups=tuple(sorted({g for g in groups if g != groups[index]})),
            )
            for index in range(60)
        ]
        model, reason = fit_stacked_meta_model(
            features=features,
            row_indices=np.arange(60),
            provenance=provenance,
            target_values=np.asarray(targets, dtype=object),
            task_type=TaskType.CLASSIFICATION,
            labels=LABELS,
            outer_train_groups=sorted(set(groups)),
            outer_test_groups=["g9"],
            group_values=groups,
            inner_fold_count=3,
            random_seed=42,
        )
        assert reason is None
        assert model is not None
        predicted, probabilities = stacked_predictions(
            model, features, TaskType.CLASSIFICATION
        )
        assert probabilities is not None
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.isfinite(probabilities).all()
        assert set(predicted.tolist()) <= set(LABELS)
