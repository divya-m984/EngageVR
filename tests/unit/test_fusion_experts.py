"""Modality-expert tests.

An expert must see only its own modality, must refuse to train on evidence
too thin to support it, must never predict for a window its modality did
not observe, and must never be fitted on outer-test rows.

No test here needs a webcam, a model asset, a display server, a network,
Unity, a public dataset, or participant data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engagevr.features.catalog import FEATURE_CATALOG
from engagevr.schemas.fusion import FusionModality
from engagevr.schemas.targets import TargetName, TaskType
from engagevr.training.calibration import CalibrationMethod
from engagevr.training.experts import (
    MINIMUM_EXPERT_FIT_ROWS,
    ExpertFit,
    expert_predictions,
    fit_modality_expert,
    modality_availability,
    modality_quality,
)
from engagevr.training.fusion import FEATURE_MODALITY_OF
from engagevr.training.models import get_model_spec
from engagevr.training.preprocessing import ModellingFrame, load_modelling_frame
from engagevr.training.splits import build_splits, choose_group_field

LABELS = ("low", "medium", "high")


@pytest.fixture(scope="module")
def frame(m5_dataset: Path) -> ModellingFrame:
    return load_modelling_frame(
        m5_dataset,
        target_name=TargetName.ENGAGEMENT_CLASS,
        catalog=FEATURE_CATALOG,
    )


@pytest.fixture(scope="module")
def fold_indices(frame: ModellingFrame) -> dict[str, object]:
    group_field, reason = choose_group_field(frame.subject_ids, frame.session_ids)
    groups = (
        frame.subject_ids if group_field.value == "subject_id" else frame.session_ids
    )
    splits = build_splits(
        group_values=groups,
        session_ids=frame.session_ids,
        task_type=TaskType.CLASSIFICATION,
        group_field=group_field,
        group_field_reason=reason,
        n_splits=3,
        random_seed=42,
        class_labels=list(frame.class_labels()),
        calibration_group_fraction=0.25,
    )
    fold = splits.folds[0]

    def rows(wanted: tuple[str, ...]) -> np.ndarray:
        members = set(wanted)
        return np.asarray([i for i, g in enumerate(groups) if g in members], dtype=int)

    return {
        "groups": groups,
        "fold": fold,
        "fit": rows(fold.fit_groups()),
        "calibration": rows(fold.calibration_groups),
        "test": rows(fold.test_groups),
    }


def _fit(
    frame: ModellingFrame,
    fold_indices: dict[str, object],
    modality: FusionModality,
    **overrides: object,
) -> ExpertFit:
    fold = fold_indices["fold"]
    parameters: dict[str, object] = {
        "modality": modality,
        "predictors": frame.predictors,
        "catalog": FEATURE_CATALOG,
        "target_values": frame.target_values,
        "task_type": TaskType.CLASSIFICATION,
        "model_spec": get_model_spec("logistic_regression", TaskType.CLASSIFICATION),
        "fit_indices": fold_indices["fit"],
        "calibration_indices": fold_indices["calibration"],
        "fit_groups": fold.fit_groups(),  # type: ignore[attr-defined]
        "calibration_groups": fold.calibration_groups,  # type: ignore[attr-defined]
        "test_groups": fold.test_groups,  # type: ignore[attr-defined]
        "group_values": fold_indices["groups"],
        "labels": LABELS,
        "fold_index": 0,
        "random_seed": 42,
        "calibration_method": CalibrationMethod.SIGMOID,
        "use_calibrated": True,
    }
    parameters.update(overrides)
    return fit_modality_expert(**parameters)  # type: ignore[arg-type]


class TestModalityAvailabilityAndQuality:
    def test_availability_is_read_from_the_dataset(self, frame: ModellingFrame) -> None:
        for modality in FusionModality:
            mask = modality_availability(frame.predictors, modality, FEATURE_CATALOG)
            assert mask.dtype == bool
            assert len(mask) == frame.row_count
            assert mask.any()

    def test_quality_is_a_separate_array(self, frame: ModellingFrame) -> None:
        quality = modality_quality(frame.predictors, FusionModality.RPPG)
        assert len(quality) == frame.row_count
        finite = quality[np.isfinite(quality)]
        assert finite.min() >= 0.0
        assert finite.max() <= 1.0

    def test_an_unavailable_modality_records_no_quality(
        self, frame: ModellingFrame
    ) -> None:
        mask = modality_availability(
            frame.predictors, FusionModality.RPPG, FEATURE_CATALOG
        )
        quality = modality_quality(frame.predictors, FusionModality.RPPG)
        assert not np.isfinite(quality[~mask]).any()


class TestExpertFitting:
    def test_an_expert_sees_only_its_own_modality(
        self, frame: ModellingFrame, fold_indices: dict[str, object]
    ) -> None:
        for modality in FusionModality:
            expert = _fit(frame, fold_indices, modality)
            assert expert.trained
            for column in expert.columns:
                if column.startswith(("feat__", "avail__")):
                    name = column.split("__", 1)[1]
                    assert (
                        FEATURE_CATALOG.get(name).modality
                        is FEATURE_MODALITY_OF[modality]
                    )
                else:
                    assert column == f"modality_available__{modality.value}"

    def test_the_record_states_the_features_it_saw(
        self, frame: ModellingFrame, fold_indices: dict[str, object]
    ) -> None:
        expert = _fit(frame, fold_indices, FusionModality.TASK)
        assert expert.record.feature_names == expert.columns
        assert expert.record.fit_row_count > 0
        assert expert.record.fit_group_count >= 2

    def test_an_expert_is_never_fitted_on_outer_test_rows(
        self, frame: ModellingFrame, fold_indices: dict[str, object]
    ) -> None:
        fold = fold_indices["fold"]
        expert = _fit(frame, fold_indices, FusionModality.TASK)
        fit_rows = fold_indices["fit"]
        groups = fold_indices["groups"]
        used = {str(groups[int(i)]) for i in fit_rows}  # type: ignore[index]
        assert not used & set(fold.test_groups)  # type: ignore[attr-defined]
        assert not used & set(fold.calibration_groups)  # type: ignore[attr-defined]
        assert used <= set(fold.fit_groups())  # type: ignore[attr-defined]
        assert 0 < expert.record.fit_group_count <= len(used)
        assert expert.record.calibration_group_count <= len(
            fold.calibration_groups  # type: ignore[attr-defined]
        )

    def test_too_few_rows_refuses_rather_than_fitting(
        self, frame: ModellingFrame, fold_indices: dict[str, object]
    ) -> None:
        tiny = np.asarray(fold_indices["fit"], dtype=int)[:3]  # type: ignore[arg-type]
        expert = _fit(frame, fold_indices, FusionModality.TASK, fit_indices=tiny)
        assert not expert.trained
        assert str(MINIMUM_EXPERT_FIT_ROWS) in (expert.record.unavailable_reason or "")

    def test_a_refusal_produces_no_prediction(
        self, frame: ModellingFrame, fold_indices: dict[str, object]
    ) -> None:
        tiny = np.asarray(fold_indices["fit"], dtype=int)[:3]  # type: ignore[arg-type]
        expert = _fit(frame, fold_indices, FusionModality.TASK, fit_indices=tiny)
        test = np.asarray(fold_indices["test"], dtype=int)  # type: ignore[arg-type]
        predictions = expert_predictions(
            expert,
            predictors=frame.predictors,
            row_indices=test,
            availability=np.ones(len(test), dtype=bool),
            quality=np.full(len(test), np.nan),
            labels=LABELS,
            task_type=TaskType.CLASSIFICATION,
        )
        assert all(not p.available for p in predictions)
        assert all(p.predicted_class is None for p in predictions)
        assert all(p.unavailable_reason for p in predictions)


class TestExpertPredictions:
    def test_calibrated_probabilities_normalise(
        self, frame: ModellingFrame, fold_indices: dict[str, object]
    ) -> None:
        expert = _fit(frame, fold_indices, FusionModality.TASK)
        test = np.asarray(fold_indices["test"], dtype=int)  # type: ignore[arg-type]
        availability = modality_availability(
            frame.predictors, FusionModality.TASK, FEATURE_CATALOG
        )[test]
        predictions = expert_predictions(
            expert,
            predictors=frame.predictors,
            row_indices=test,
            availability=availability,
            quality=modality_quality(frame.predictors, FusionModality.TASK)[test],
            labels=LABELS,
            task_type=TaskType.CLASSIFICATION,
        )
        available = [p for p in predictions if p.available]
        assert available
        for prediction in available:
            assert sum(prediction.probabilities) == pytest.approx(1.0)
            assert prediction.class_vocabulary == LABELS

    def test_the_class_vocabulary_is_the_ordered_run_vocabulary(
        self, frame: ModellingFrame, fold_indices: dict[str, object]
    ) -> None:
        expert = _fit(frame, fold_indices, FusionModality.RPPG)
        test = np.asarray(fold_indices["test"], dtype=int)  # type: ignore[arg-type]
        availability = modality_availability(
            frame.predictors, FusionModality.RPPG, FEATURE_CATALOG
        )[test]
        predictions = expert_predictions(
            expert,
            predictors=frame.predictors,
            row_indices=test,
            availability=availability,
            quality=modality_quality(frame.predictors, FusionModality.RPPG)[test],
            labels=LABELS,
            task_type=TaskType.CLASSIFICATION,
        )
        for prediction in predictions:
            if prediction.available:
                assert prediction.class_vocabulary == LABELS
                assert prediction.predicted_class in LABELS

    def test_an_unavailable_modality_produces_no_prediction(
        self, frame: ModellingFrame, fold_indices: dict[str, object]
    ) -> None:
        expert = _fit(frame, fold_indices, FusionModality.RPPG)
        test = np.asarray(fold_indices["test"], dtype=int)  # type: ignore[arg-type]
        predictions = expert_predictions(
            expert,
            predictors=frame.predictors,
            row_indices=test,
            availability=np.zeros(len(test), dtype=bool),
            quality=np.full(len(test), np.nan),
            labels=LABELS,
            task_type=TaskType.CLASSIFICATION,
        )
        assert all(not p.available for p in predictions)
        assert all(p.probabilities == () for p in predictions)
        assert all(
            "never as a zero-valued measurement" in (p.unavailable_reason or "")
            for p in predictions
        )

    def test_quality_travels_beside_the_prediction_not_inside_it(
        self, frame: ModellingFrame, fold_indices: dict[str, object]
    ) -> None:
        expert = _fit(frame, fold_indices, FusionModality.RPPG)
        test = np.asarray(fold_indices["test"], dtype=int)  # type: ignore[arg-type]
        availability = modality_availability(
            frame.predictors, FusionModality.RPPG, FEATURE_CATALOG
        )[test]
        quality = modality_quality(frame.predictors, FusionModality.RPPG)[test]
        predictions = expert_predictions(
            expert,
            predictors=frame.predictors,
            row_indices=test,
            availability=availability,
            quality=quality,
            labels=LABELS,
            task_type=TaskType.CLASSIFICATION,
        )
        for position, prediction in enumerate(predictions):
            if prediction.available and np.isfinite(quality[position]):
                assert prediction.quality == pytest.approx(quality[position])
                assert prediction.quality not in prediction.probabilities


class TestRegressionExperts:
    def test_a_regression_expert_produces_finite_predictions(
        self, m5_dataset: Path, fold_indices: dict[str, object]
    ) -> None:
        regression_frame = load_modelling_frame(
            m5_dataset,
            target_name=TargetName.ENGAGEMENT_SCORE,
            catalog=FEATURE_CATALOG,
        )
        expert = _fit(
            regression_frame,
            fold_indices,
            FusionModality.TASK,
            predictors=regression_frame.predictors,
            target_values=regression_frame.target_values,
            task_type=TaskType.REGRESSION,
            model_spec=get_model_spec("ridge", TaskType.REGRESSION),
            labels=(),
        )
        assert expert.trained
        test = np.asarray(fold_indices["test"], dtype=int)  # type: ignore[arg-type]
        availability = modality_availability(
            regression_frame.predictors, FusionModality.TASK, FEATURE_CATALOG
        )[test]
        predictions = expert_predictions(
            expert,
            predictors=regression_frame.predictors,
            row_indices=test,
            availability=availability,
            quality=np.full(len(test), np.nan),
            labels=(),
            task_type=TaskType.REGRESSION,
        )
        available = [p for p in predictions if p.available]
        assert available
        assert all(np.isfinite(p.predicted_value) for p in available)
        assert all(p.probabilities == () for p in available)
