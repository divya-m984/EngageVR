"""Leakage-safe stacked fusion.

A stacker learns how much to trust each modality expert.  If it is fitted
on predictions the experts made about their own training rows, it learns to
trust memorised predictions, and the weights it derives do not transfer.
So, inside each outer fold:

1. the outer-training groups are split into grouped inner folds;
2. modality experts are fitted on the inner-training groups only;
3. those experts predict **only** the held-out inner groups;
4. the meta-model is fitted on that out-of-fold matrix and nothing else;
5. experts are refitted on the whole outer-training portion;
6. those experts predict the untouched outer-test groups;
7. the already-fitted meta-model is applied to those predictions.

:func:`assert_out_of_fold` re-checks the property independently before the
meta-model is fitted and raises on the first violation, so a future change
that quietly reintroduces in-sample meta-training fails loudly.

Calibration placement
---------------------
The stacker consumes **uncalibrated** expert probabilities at both
meta-training and meta-inference time.  Fitting a meta-model on
uncalibrated inputs and then applying it to calibrated ones would change
the input distribution between fitting and use, which is a silent error
that no metric would reveal.  Per-expert calibration remains available for
the other late-fusion strategies, where the combination is a fixed weighted
average rather than a learned function.

Only interpretable meta-models are offered: multinomial logistic regression
for classification and ridge for regression.  There is no neural stacker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from engagevr.schemas.features import FeatureCatalog
from engagevr.schemas.fusion import FusionModality, ModalityPrediction
from engagevr.schemas.targets import TaskType
from engagevr.training.calibration import CalibrationMethod, aligned_probabilities
from engagevr.training.experts import expert_predictions, fit_modality_expert
from engagevr.training.models import ModelSpec, build_pipeline, get_model_spec

#: Prefix of every meta-model input column.
META_COLUMN_PREFIX = "expert__"

#: Fewest out-of-fold rows before a meta-model is fitted at all.
MINIMUM_META_TRAINING_ROWS = 20


class StackingError(ValueError):
    """A stacked fusion model cannot be built as requested."""


class StackingLeakageError(StackingError):
    """A meta-model would be fitted on predictions it must never see."""


@dataclass(frozen=True, slots=True)
class OutOfFoldProvenance:
    """Where one meta-training row's expert predictions came from."""

    row_index: int
    row_group: str
    inner_fold_index: int
    source_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutOfFoldMatrix:
    """Out-of-fold expert predictions for one outer training portion."""

    features: pd.DataFrame
    row_indices: np.ndarray
    provenance: tuple[OutOfFoldProvenance, ...]
    modality_predictions: dict[FusionModality, tuple[ModalityPrediction, ...]]
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        """Whether an out-of-fold matrix was produced."""
        return self.unavailable_reason is None and not self.features.empty


@dataclass(frozen=True, slots=True)
class StackedFusionModel:
    """A fitted meta-model plus the provenance of its training rows."""

    meta_estimator: BaseEstimator
    meta_model_name: str
    columns: tuple[str, ...]
    class_vocabulary: tuple[str, ...]
    provenance: tuple[OutOfFoldProvenance, ...]
    inner_fold_count: int
    training_row_count: int
    training_group_count: int
    probabilities_are_calibrated: bool = False


def meta_feature_columns(
    modalities: Sequence[FusionModality],
    labels: Sequence[str],
    task_type: TaskType,
) -> tuple[str, ...]:
    """Meta-model input columns, in a deterministic order."""
    columns: list[str] = []
    for modality in modalities:
        name = modality.value
        if task_type is TaskType.CLASSIFICATION:
            columns.extend(
                f"{META_COLUMN_PREFIX}{name}__probability__{label}" for label in labels
            )
        else:
            columns.append(f"{META_COLUMN_PREFIX}{name}__prediction")
        columns.append(f"{META_COLUMN_PREFIX}{name}__available")
    return tuple(columns)


def meta_feature_row(
    predictions: Mapping[FusionModality, ModalityPrediction],
    modalities: Sequence[FusionModality],
    labels: Sequence[str],
    task_type: TaskType,
) -> dict[str, float | None]:
    """One meta-model input row.

    An unavailable expert contributes ``None`` — a missing value — never a
    zero and never a uniform probability vector.  The availability flag
    beside it states the absence explicitly, and the meta-model's fold-local
    imputer adds its own missingness indicator, so "the expert said nothing"
    is never mistaken for "the expert said zero".
    """
    row: dict[str, float | None] = {}
    for modality in modalities:
        name = modality.value
        prediction = predictions.get(modality)
        available = prediction is not None and prediction.available
        if task_type is TaskType.CLASSIFICATION:
            for index, label in enumerate(labels):
                key = f"{META_COLUMN_PREFIX}{name}__probability__{label}"
                row[key] = (
                    float(prediction.probabilities[index])
                    if available
                    and prediction is not None
                    and index < len(prediction.probabilities)
                    else None
                )
        else:
            key = f"{META_COLUMN_PREFIX}{name}__prediction"
            row[key] = (
                float(prediction.predicted_value)
                if available
                and prediction is not None
                and prediction.predicted_value is not None
                else None
            )
        row[f"{META_COLUMN_PREFIX}{name}__available"] = 1.0 if available else 0.0
    return row


def assert_out_of_fold(
    provenance: Sequence[OutOfFoldProvenance],
    *,
    outer_train_groups: Sequence[str],
    outer_test_groups: Sequence[str],
) -> None:
    """Assert every meta-training row is genuinely out of fold.

    Three distinct violations are checked, each a different way for a
    stacker to become meaningless:

    1. a row predicted by experts fitted on its own group — in-sample
       meta-training;
    2. a row predicted by experts fitted on an outer-test group — the outer
       test fold influencing the meta-model;
    3. a meta-training row drawn from outside the outer-training groups.

    Raises
    ------
    StackingLeakageError
        On the first violation found, naming the offending groups.
    """
    train = set(outer_train_groups)
    test = set(outer_test_groups)
    for record in provenance:
        sources = set(record.source_groups)
        if record.row_group in sources:
            raise StackingLeakageError(
                f"meta-training row {record.row_index} belongs to group "
                f"{record.row_group!r}, which also trained the experts that "
                "predicted it. A meta-model fitted on in-sample expert "
                "predictions learns to trust memorised predictions, and the "
                "weights it derives do not transfer."
            )
        contaminated = sources & test
        if contaminated:
            raise StackingLeakageError(
                f"meta-training row {record.row_index} was predicted by "
                f"experts fitted on outer-test group(s) "
                f"{sorted(contaminated)[:5]}. The outer test fold must not "
                "influence meta-model fitting."
            )
        if record.row_group not in train:
            raise StackingLeakageError(
                f"meta-training row {record.row_index} belongs to group "
                f"{record.row_group!r}, which is not an outer-training group. "
                "The meta-model may only be fitted inside the outer training "
                "portion."
            )
        stray = sources - train
        if stray:
            raise StackingLeakageError(
                f"meta-training row {record.row_index} was predicted by "
                f"experts fitted on group(s) {sorted(stray)[:5]} outside the "
                "outer training portion"
            )


def build_out_of_fold_matrix(
    *,
    modalities: Sequence[FusionModality],
    predictors: pd.DataFrame,
    catalog: FeatureCatalog,
    target_values: np.ndarray,
    task_type: TaskType,
    model_spec: ModelSpec,
    train_indices: np.ndarray,
    group_values: Sequence[str],
    availability: Mapping[FusionModality, np.ndarray],
    quality: Mapping[FusionModality, np.ndarray],
    labels: tuple[str, ...],
    fold_index: int,
    random_seed: int,
    inner_folds: int,
    include_modality_quality: bool = False,
) -> OutOfFoldMatrix:
    """Generate out-of-fold expert predictions inside the outer training rows.

    Returns the meta-feature frame, the row indices it describes, the
    provenance of each row, the per-modality predictions those features were
    built from, and — when the matrix could not be built — the reason.

    The per-modality predictions are returned as well because
    validation-derived fusion weights must be estimated from exactly this
    out-of-fold evidence: an inner-validation estimate and a meta-training
    matrix are two uses of one honest pass over the training portion, and
    computing them separately would double the cost and risk the two
    disagreeing.
    """
    columns = meta_feature_columns(modalities, labels, task_type)
    empty = pd.DataFrame(columns=list(columns), dtype=float)
    groups = [str(group_values[int(i)]) for i in train_indices]
    distinct = sorted(set(groups))
    splits = min(inner_folds, len(distinct))
    if splits < 2:
        return OutOfFoldMatrix(
            features=empty,
            row_indices=np.asarray([], dtype=int),
            provenance=(),
            modality_predictions={},
            unavailable_reason=(
                "out-of-fold construction needs at least 2 independent groups "
                f"inside the outer training portion of fold {fold_index}; "
                f"{len(distinct)} are available"
            ),
        )

    inner_targets = target_values[train_indices]
    if task_type is TaskType.CLASSIFICATION:
        feasible = _stratification_feasible(
            groups, [str(v) for v in inner_targets], splits
        )
        splitter = (
            StratifiedGroupKFold(
                n_splits=splits, shuffle=True, random_state=random_seed
            )
            if feasible
            else GroupKFold(n_splits=splits, shuffle=True, random_state=random_seed)
        )
        stratify_on: np.ndarray = np.asarray(
            [str(v) for v in inner_targets], dtype=object
        )
    else:
        splitter = GroupKFold(n_splits=splits, shuffle=True, random_state=random_seed)
        stratify_on = np.zeros(len(train_indices))

    rows: list[dict[str, float | None]] = []
    row_indices: list[int] = []
    provenance: list[OutOfFoldProvenance] = []
    collected: dict[FusionModality, list[ModalityPrediction]] = {
        modality: [] for modality in modalities
    }
    positions = np.arange(len(train_indices))

    for inner_index, (inner_train, inner_holdout) in enumerate(
        splitter.split(
            positions.reshape(-1, 1), stratify_on, np.asarray(groups, dtype=object)
        )
    ):
        inner_fit = train_indices[inner_train]
        inner_test = train_indices[inner_holdout]
        source_groups = tuple(sorted({str(group_values[int(i)]) for i in inner_fit}))

        fitted = {
            modality: fit_modality_expert(
                modality=modality,
                predictors=predictors,
                catalog=catalog,
                target_values=target_values,
                task_type=task_type,
                model_spec=model_spec,
                fit_indices=inner_fit,
                calibration_indices=np.asarray([], dtype=int),
                fit_groups=source_groups,
                calibration_groups=(),
                test_groups=tuple(
                    sorted({str(group_values[int(i)]) for i in inner_test})
                ),
                group_values=group_values,
                labels=labels,
                fold_index=fold_index,
                random_seed=random_seed,
                calibration_method=CalibrationMethod.NONE,
                use_calibrated=False,
                include_modality_quality=include_modality_quality,
            )
            for modality in modalities
        }

        per_modality = {
            modality: expert_predictions(
                expert,
                predictors=predictors,
                row_indices=inner_test,
                availability=np.asarray(
                    [bool(availability[modality][int(i)]) for i in inner_test],
                    dtype=bool,
                ),
                quality=np.asarray(
                    [float(quality[modality][int(i)]) for i in inner_test], dtype=float
                ),
                labels=labels,
                task_type=task_type,
                prefer_calibrated=False,
            )
            for modality, expert in fitted.items()
        }

        for offset, row_index in enumerate(inner_test):
            predictions = {
                modality: values[offset] for modality, values in per_modality.items()
            }
            for modality, prediction in predictions.items():
                collected[modality].append(prediction)
            rows.append(meta_feature_row(predictions, modalities, labels, task_type))
            row_indices.append(int(row_index))
            provenance.append(
                OutOfFoldProvenance(
                    row_index=int(row_index),
                    row_group=str(group_values[int(row_index)]),
                    inner_fold_index=inner_index,
                    source_groups=source_groups,
                )
            )

    if not rows:  # pragma: no cover - a grouped splitter always yields holdouts
        return OutOfFoldMatrix(
            features=empty,
            row_indices=np.asarray([], dtype=int),
            provenance=(),
            modality_predictions={},
            unavailable_reason="the inner grouped splitter produced no held-out rows",
        )
    return OutOfFoldMatrix(
        features=pd.DataFrame(rows, columns=list(columns), dtype=float),
        row_indices=np.asarray(row_indices, dtype=int),
        provenance=tuple(provenance),
        modality_predictions={
            modality: tuple(values) for modality, values in collected.items()
        },
        unavailable_reason=None,
    )


def _stratification_feasible(
    groups: Sequence[str], labels: Sequence[str], n_splits: int
) -> bool:
    per_class: dict[str, set[str]] = {}
    for group, label in zip(groups, labels, strict=True):
        per_class.setdefault(label, set()).add(group)
    return all(len(members) >= n_splits for members in per_class.values())


def fit_stacked_meta_model(
    *,
    features: pd.DataFrame,
    row_indices: np.ndarray,
    provenance: Sequence[OutOfFoldProvenance],
    target_values: np.ndarray,
    task_type: TaskType,
    labels: tuple[str, ...],
    outer_train_groups: Sequence[str],
    outer_test_groups: Sequence[str],
    group_values: Sequence[str],
    inner_fold_count: int,
    random_seed: int,
    meta_model_classification: str = "logistic_regression",
    meta_model_regression: str = "ridge",
    minimum_rows: int = MINIMUM_META_TRAINING_ROWS,
    minimum_available_experts: int = 1,
) -> tuple[StackedFusionModel | None, str | None]:
    """Fit the meta-model on out-of-fold expert predictions only.

    Rows in which fewer than ``minimum_available_experts`` experts produced
    a prediction are dropped from meta-training: their inputs would be
    entirely missing, and a meta-model fitted on imputed placeholders would
    be learning the imputer.

    Raises
    ------
    StackingLeakageError
        If any supplied row is not genuinely out of fold.
    """
    assert_out_of_fold(
        provenance,
        outer_train_groups=outer_train_groups,
        outer_test_groups=outer_test_groups,
    )
    if features.empty:
        return None, "no out-of-fold expert predictions are available"

    available_columns = [
        column for column in features.columns if str(column).endswith("__available")
    ]
    counts = features.loc[:, available_columns].sum(axis=1).to_numpy(dtype=float)
    keep = counts >= float(minimum_available_experts)
    usable = features.loc[keep]
    kept_indices = row_indices[keep]
    if len(usable) < minimum_rows:
        return None, (
            f"only {len(usable)} out-of-fold row(s) carry at least "
            f"{minimum_available_experts} expert prediction(s), which is below "
            f"the minimum of {minimum_rows} for fitting a meta-model"
        )

    y = target_values[kept_indices]
    if task_type is TaskType.CLASSIFICATION:
        present = sorted({str(v) for v in y})
        if len(present) < 2:
            return None, (
                f"the out-of-fold meta-training rows carry only class(es) "
                f"{present}; a meta-classifier fitted there could not predict "
                "the others"
            )
        spec = get_model_spec(meta_model_classification, TaskType.CLASSIFICATION)
    else:
        spec = get_model_spec(meta_model_regression, TaskType.REGRESSION)

    pipeline = build_pipeline(
        spec, [str(c) for c in usable.columns], random_seed=random_seed
    )
    pipeline.fit(
        usable, [str(v) for v in y] if task_type is TaskType.CLASSIFICATION else y
    )
    return (
        StackedFusionModel(
            meta_estimator=pipeline,
            meta_model_name=spec.name,
            columns=tuple(str(c) for c in usable.columns),
            class_vocabulary=labels if task_type is TaskType.CLASSIFICATION else (),
            provenance=tuple(provenance),
            inner_fold_count=inner_fold_count,
            training_row_count=len(usable),
            training_group_count=len({str(group_values[int(i)]) for i in kept_indices}),
        ),
        None,
    )


def stacked_predictions(
    model: StackedFusionModel,
    features: pd.DataFrame,
    task_type: TaskType,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply a fitted meta-model to expert predictions for held-out rows."""
    ordered = features.loc[:, list(model.columns)]
    predicted = np.asarray(model.meta_estimator.predict(ordered))
    if task_type is not TaskType.CLASSIFICATION:
        return predicted, None
    probabilities = aligned_probabilities(
        model.meta_estimator, ordered, model.class_vocabulary
    )
    return predicted, probabilities


__all__ = [
    "META_COLUMN_PREFIX",
    "MINIMUM_META_TRAINING_ROWS",
    "OutOfFoldMatrix",
    "OutOfFoldProvenance",
    "StackedFusionModel",
    "StackingError",
    "StackingLeakageError",
    "assert_out_of_fold",
    "build_out_of_fold_matrix",
    "fit_stacked_meta_model",
    "meta_feature_columns",
    "meta_feature_row",
    "stacked_predictions",
]
