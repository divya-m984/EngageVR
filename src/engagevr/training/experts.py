"""Modality-specific experts for late fusion.

One estimator per modality, fitted on that modality's features only, inside
the outer fold's fit groups.  Nothing here ever sees the outer test groups
during fitting or calibration: the fit rows, the calibration rows, and the
test rows are selected from disjoint group sets chosen by
:mod:`engagevr.training.splits`, and
:func:`engagevr.training.calibration.assert_calibration_disjoint` re-checks
that before a calibrator is fitted.

Which rows an expert learns from
--------------------------------
An expert is fitted on the fit-group rows **in which its own modality
contributed evidence**.  Rows where the modality is absent carry no
measurement from it: including them would teach the expert the fold's
imputation median rather than anything about the signal, and would then let
it emit a confident prediction for a window it never observed.  The row
counts, before and after that restriction, are recorded on the expert.

Refusing rather than faking
---------------------------
An expert that cannot be trained on defensible evidence returns
``unavailable`` with a stated reason.  It never returns a fabricated
prediction, a class prior dressed up as a prediction, or a uniform
probability vector.  Likewise, an expert never predicts for a window in
which its modality contributed nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from engagevr.schemas.features import (
    FEATURE_PREFIX,
    FeatureCatalog,
    modality_available_column,
    modality_quality_column,
)
from engagevr.schemas.fusion import (
    ExpertRecord,
    FusionModality,
    ModalityPrediction,
)
from engagevr.schemas.targets import TaskType
from engagevr.training.calibration import (
    CalibrationMethod,
    aligned_probabilities,
    calibrate_classifier,
)
from engagevr.training.fusion import FEATURE_MODALITY_OF, modality_expert_columns
from engagevr.training.models import ModelSpec, build_pipeline

#: Fewest fit rows carrying evidence from a modality before its expert is
#: trained at all.  Below this, a fold-local imputer and a penalised linear
#: model are describing a handful of points, and the resulting predictions
#: would carry a confidence the evidence cannot support.
MINIMUM_EXPERT_FIT_ROWS = 10

#: Fewest independent groups contributing evidence before an expert is
#: trained.  One group means the expert has seen one participant.
MINIMUM_EXPERT_FIT_GROUPS = 2


class ExpertError(ValueError):
    """A modality expert cannot be constructed as requested."""


@dataclass(frozen=True, slots=True)
class ExpertFit:
    """A fitted (or deliberately unfitted) modality expert."""

    modality: FusionModality
    columns: tuple[str, ...]
    estimator: BaseEstimator | None
    calibrated_estimator: BaseEstimator | None
    record: ExpertRecord

    @property
    def trained(self) -> bool:
        """Whether an estimator was fitted for this modality."""
        return self.estimator is not None


def modality_availability(
    predictors: pd.DataFrame,
    modality: FusionModality,
    catalog: FeatureCatalog,
) -> np.ndarray:
    """Per-row boolean: did this modality contribute evidence to the window?

    Uses the dataset's ``modality_available__<modality>`` column when it is
    present.  When it is not, availability falls back to "at least one
    measured feature of this modality is non-null", which is the same
    question asked of the data directly.
    """
    column = modality_available_column(modality.value)
    if column in predictors.columns:
        values = pd.to_numeric(predictors[column], errors="coerce").to_numpy(
            dtype=float
        )
        return np.asarray(values > 0.5, dtype=bool)
    measured = [
        name
        for name in predictors.columns
        if str(name).startswith(FEATURE_PREFIX)
        and _feature_modality(str(name), catalog) is modality
    ]
    if not measured:
        return np.zeros(len(predictors), dtype=bool)
    frame = predictors.loc[:, measured]
    return np.asarray(frame.notna().any(axis=1).to_numpy(), dtype=bool)


def modality_quality(
    predictors: pd.DataFrame,
    modality: FusionModality,
) -> np.ndarray:
    """Per-row modality signal quality, ``NaN`` where none was recorded.

    Quality describes the measurement, never the person.  It is returned as
    a separate array and never merged into a feature value or a prediction.
    """
    column = modality_quality_column(modality.value)
    if column not in predictors.columns:
        return np.full(len(predictors), np.nan, dtype=float)
    values = pd.to_numeric(predictors[column], errors="coerce").to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def _feature_modality(column: str, catalog: FeatureCatalog) -> FusionModality | None:
    name = column.removeprefix(FEATURE_PREFIX)
    try:
        entry = catalog.get(name)
    except KeyError:  # pragma: no cover - refused earlier by assert_no_leakage
        return None
    for fusion_modality, feature_modality in FEATURE_MODALITY_OF.items():
        if entry.modality is feature_modality:
            return fusion_modality
    return None


def fit_modality_expert(
    *,
    modality: FusionModality,
    predictors: pd.DataFrame,
    catalog: FeatureCatalog,
    target_values: np.ndarray,
    task_type: TaskType,
    model_spec: ModelSpec,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    fit_groups: Sequence[str],
    calibration_groups: Sequence[str],
    test_groups: Sequence[str],
    group_values: Sequence[str],
    labels: tuple[str, ...],
    fold_index: int,
    random_seed: int,
    calibration_method: CalibrationMethod,
    use_calibrated: bool,
    include_modality_quality: bool = False,
    minimum_fit_rows: int = MINIMUM_EXPERT_FIT_ROWS,
    minimum_fit_groups: int = MINIMUM_EXPERT_FIT_GROUPS,
) -> ExpertFit:
    """Fit one modality expert inside one outer fold, or refuse with a reason.

    Raises
    ------
    ExpertError
        If the supplied index arrays are inconsistent with the frame.
    """
    if fit_indices.size and int(fit_indices.max()) >= len(predictors):
        raise ExpertError("a fit index falls outside the modelling frame")

    columns = modality_expert_columns(
        modality,
        [str(c) for c in predictors.columns],
        catalog,
        include_modality_quality=include_modality_quality,
    )
    available = modality_availability(predictors, modality, catalog)

    def _refuse(reason: str, *, feature_names: tuple[str, ...] = ()) -> ExpertFit:
        return ExpertFit(
            modality=modality,
            columns=feature_names,
            estimator=None,
            calibrated_estimator=None,
            record=ExpertRecord(
                modality=modality,
                fold_index=fold_index,
                model_name=model_spec.name,
                trained=False,
                unavailable_reason=reason,
                feature_names=feature_names,
                class_vocabulary=labels if task_type is TaskType.CLASSIFICATION else (),
            ),
        )

    if not columns:
        return _refuse(
            f"modality {modality.value!r} contributes no permitted predictor "
            "column to this dataset, so no expert can be trained for it"
        )

    fit_available = (
        fit_indices[available[fit_indices]] if fit_indices.size else fit_indices
    )
    if fit_available.size < minimum_fit_rows:
        return _refuse(
            f"modality {modality.value!r} carries evidence in only "
            f"{int(fit_available.size)} of {int(fit_indices.size)} training "
            f"rows in fold {fold_index}, which is below the minimum of "
            f"{minimum_fit_rows}; the expert is reported unavailable rather "
            "than fitted on evidence too thin to support it",
            feature_names=columns,
        )
    fit_group_names = {str(group_values[int(i)]) for i in fit_available}
    if len(fit_group_names) < minimum_fit_groups:
        return _refuse(
            f"modality {modality.value!r} carries evidence from only "
            f"{len(fit_group_names)} independent group(s) in fold "
            f"{fold_index}, which is below the minimum of "
            f"{minimum_fit_groups}",
            feature_names=columns,
        )

    y_fit = target_values[fit_available]
    if task_type is TaskType.CLASSIFICATION:
        present = sorted({str(v) for v in y_fit})
        if len(present) < 2:
            return _refuse(
                f"modality {modality.value!r} has only class(es) {present} in "
                f"the rows carrying its evidence in fold {fold_index}; a "
                "classifier fitted there could not predict the others",
                feature_names=columns,
            )

    X = predictors.loc[:, list(columns)]
    X_fit = X.iloc[fit_available]
    calibration_available = (
        calibration_indices[available[calibration_indices]]
        if calibration_indices.size
        else calibration_indices
    )
    X_calibration = X.iloc[calibration_available]
    y_calibration = target_values[calibration_available]

    pipeline = build_pipeline(model_spec, columns, random_seed=random_seed)
    calibrated_estimator: BaseEstimator | None = None
    calibration_reason: str | None = None
    method_used: str | None = None

    if task_type is TaskType.CLASSIFICATION and use_calibrated:
        base, outcome = calibrate_classifier(
            pipeline,
            X_fit=X_fit,
            y_fit=[str(v) for v in y_fit],
            X_calibration=X_calibration,
            y_calibration=[str(v) for v in y_calibration],
            method=calibration_method,
            fit_groups=fit_groups,
            calibration_groups=calibration_groups,
            test_groups=test_groups,
        )
        calibrated_estimator = outcome.calibrated_estimator
        calibration_reason = outcome.unavailable_reason
        method_used = outcome.method.value
    else:
        base = clone(pipeline)
        base.fit(X_fit, list(y_fit) if task_type is TaskType.CLASSIFICATION else y_fit)
        if task_type is TaskType.CLASSIFICATION:
            calibration_reason = "calibrated experts were not requested"

    return ExpertFit(
        modality=modality,
        columns=columns,
        estimator=base,
        calibrated_estimator=calibrated_estimator,
        record=ExpertRecord(
            modality=modality,
            fold_index=fold_index,
            model_name=model_spec.name,
            trained=True,
            feature_names=columns,
            fit_row_count=int(fit_available.size),
            fit_group_count=len(fit_group_names),
            calibration_row_count=int(calibration_available.size),
            calibration_group_count=len(
                {str(group_values[int(i)]) for i in calibration_available}
            ),
            calibrated=calibrated_estimator is not None,
            calibration_method=method_used,
            calibration_unavailable_reason=calibration_reason,
            class_vocabulary=labels if task_type is TaskType.CLASSIFICATION else (),
            available_row_count=int(fit_available.size),
        ),
    )


def expert_predictions(
    expert: ExpertFit,
    *,
    predictors: pd.DataFrame,
    row_indices: np.ndarray,
    availability: np.ndarray,
    quality: np.ndarray,
    labels: tuple[str, ...],
    task_type: TaskType,
    prefer_calibrated: bool = True,
) -> tuple[ModalityPrediction, ...]:
    """One :class:`ModalityPrediction` per requested row, in order.

    A row whose modality contributed nothing receives ``available=False``
    and a reason.  It never receives an imputed prediction: the expert saw
    no evidence from this modality in that window, so it has nothing to say
    about it.
    """
    name = expert.modality.value
    count = int(row_indices.size)
    if not expert.trained:
        reason = expert.record.unavailable_reason or (
            f"no expert was trained for modality {name!r}"
        )
        return tuple(
            ModalityPrediction(
                modality=expert.modality, available=False, unavailable_reason=reason
            )
            for _ in range(count)
        )

    usable = np.asarray(availability, dtype=bool)
    selected = row_indices[usable]
    results: list[ModalityPrediction | None] = [None] * count

    if selected.size:
        X = predictors.loc[:, list(expert.columns)].iloc[selected]
        estimator = expert.estimator
        assert estimator is not None  # narrowed by expert.trained
        probability_source = estimator
        calibrated = False
        if (
            prefer_calibrated
            and expert.calibrated_estimator is not None
            and task_type is TaskType.CLASSIFICATION
        ):
            probability_source = expert.calibrated_estimator
            calibrated = True
        raw_predictions = np.asarray(probability_source.predict(X))
        probabilities = (
            aligned_probabilities(probability_source, X, labels)
            if task_type is TaskType.CLASSIFICATION
            else None
        )
        position = 0
        for index in range(count):
            if not usable[index]:
                continue
            quality_value = float(quality[index]) if quality.size else float("nan")
            recorded = None if not np.isfinite(quality_value) else quality_value
            if recorded is not None:
                recorded = float(min(max(recorded, 0.0), 1.0))
            if task_type is TaskType.CLASSIFICATION:
                vector = (
                    tuple(float(v) for v in probabilities[position])
                    if probabilities is not None
                    else ()
                )
                results[index] = ModalityPrediction(
                    modality=expert.modality,
                    available=True,
                    predicted_class=str(raw_predictions[position]),
                    class_vocabulary=labels,
                    probabilities=vector,
                    probabilities_are_calibrated=calibrated,
                    quality=recorded,
                )
            else:
                results[index] = ModalityPrediction(
                    modality=expert.modality,
                    available=True,
                    predicted_value=float(raw_predictions[position]),
                    quality=recorded,
                )
            position += 1

    absent_reason = (
        f"modality {name!r} contributed no evidence to this window, so its "
        "expert produced no prediction; the absence is represented through "
        "availability and never as a zero-valued measurement"
    )
    return tuple(
        result
        if result is not None
        else ModalityPrediction(
            modality=expert.modality,
            available=False,
            unavailable_reason=absent_reason,
        )
        for result in results
    )


__all__ = [
    "MINIMUM_EXPERT_FIT_GROUPS",
    "MINIMUM_EXPERT_FIT_ROWS",
    "ExpertError",
    "ExpertFit",
    "expert_predictions",
    "fit_modality_expert",
    "modality_availability",
    "modality_quality",
]
