"""Fold-local preprocessing: loading, column selection, imputation, scaling.

Everything in this module is designed so that **no statistic crosses a
fold boundary**.  The transformers are returned unfitted and are fitted
inside a :class:`~sklearn.pipeline.Pipeline` on the training rows of one
fold only.  Nothing is fitted on the full dataset, and nothing is fitted
on a test fold.

Imputation and signal quality
-----------------------------
Median imputation puts a plausible number where a measurement failed.
That is unavoidable for estimators which cannot accept ``NaN``, but it
must not erase the difference between "measured" and "guessed", so:

- every imputed numeric column is accompanied by a missingness indicator,
  which is a first-class input;
- the availability and modality-quality columns remain in the matrix
  untouched, so the failure itself is visible to the model;
- estimators with native ``NaN`` support (scikit-learn's histogram
  gradient boosting) receive the values unimputed.

A quality failure is therefore never silently converted into a
physiological value: the value is marked as absent in three separate
places.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from engagevr.features.assembly import read_dataset_metadata, read_dataset_table
from engagevr.features.validation import (
    DatasetValidationError,
    assert_no_identifier_values,
    assert_no_identity_columns,
    assert_no_leakage,
    select_predictor_columns,
)
from engagevr.schemas.features import (
    AVAILABILITY_PREFIX,
    MODALITY_AVAILABLE_PREFIX,
    DatasetMetadata,
    FeatureCatalog,
    target_column,
    target_meta_column,
)
from engagevr.schemas.targets import TargetName, TaskType, get_target_spec


class PreprocessingError(ValueError):
    """A predictor matrix could not be prepared."""


class ImputationStrategy(enum.StrEnum):
    """How a model wants missing numeric values handled."""

    MEDIAN_WITH_INDICATOR = "median_with_indicator"
    NATIVE_NAN = "native_nan"


class ModellingFrame:
    """A loaded dataset split into predictors, target, and grouping columns.

    The predictor matrix is a :class:`pandas.DataFrame` rather than a bare
    array so that feature names survive every transformation and can be
    reported alongside coefficients and importances.
    """

    def __init__(
        self,
        *,
        predictors: pd.DataFrame,
        target_values: np.ndarray,
        target_name: TargetName,
        task_type: TaskType,
        subject_ids: tuple[str, ...],
        session_ids: tuple[str, ...],
        window_ids: tuple[str, ...],
        data_sources: tuple[str, ...],
        target_source_types: tuple[str, ...],
        target_scientific_permitted: tuple[bool, ...],
        metadata: DatasetMetadata,
    ) -> None:
        self.predictors = predictors
        self.target_values = target_values
        self.target_name = target_name
        self.task_type = task_type
        self.subject_ids = subject_ids
        self.session_ids = session_ids
        self.window_ids = window_ids
        self.data_sources = data_sources
        self.target_source_types = target_source_types
        self.target_scientific_permitted = target_scientific_permitted
        self.metadata = metadata

    @property
    def row_count(self) -> int:
        """Number of labelled rows in the frame."""
        return len(self.predictors)

    @property
    def predictor_columns(self) -> tuple[str, ...]:
        """Predictor column names, in dataset order."""
        return tuple(str(c) for c in self.predictors.columns)

    def class_labels(self) -> tuple[str, ...]:
        """Target values as strings. Classification only."""
        if self.task_type is not TaskType.CLASSIFICATION:
            raise PreprocessingError(
                "class labels are only defined for a classification target"
            )
        return tuple(str(v) for v in self.target_values)

    def numeric_targets(self) -> tuple[float, ...]:
        """Target values as floats. Regression only."""
        if self.task_type is not TaskType.REGRESSION:
            raise PreprocessingError(
                "numeric targets are only defined for a regression target"
            )
        return tuple(float(v) for v in self.target_values)


def load_modelling_frame(
    dataset_path: Path,
    *,
    target_name: TargetName,
    catalog: FeatureCatalog,
    feature_names: Sequence[str] | None = None,
) -> ModellingFrame:
    """Load a Parquet dataset into a leakage-audited modelling frame.

    ``feature_names`` restricts the matrix to a feature subset (used by
    ablations).  Whatever the subset, the resulting column list is audited
    for target, identifier, and post-window leakage before it is returned.

    Raises
    ------
    DatasetValidationError
        If the dataset carries a forbidden column or an identifier value.
    PreprocessingError
        If the target column is absent, or no row carries a target value.
    """
    table = read_dataset_table(dataset_path)
    metadata = read_dataset_metadata(dataset_path)
    frame = table.to_pandas()

    assert_no_identity_columns(str(c) for c in frame.columns)

    value_column = target_column(target_name)
    if value_column not in frame.columns:
        available = sorted(
            str(c).removeprefix("target__")
            for c in frame.columns
            if str(c).startswith("target__")
        )
        raise PreprocessingError(
            f"dataset {dataset_path} has no column {value_column!r}. Targets "
            f"present: {available or 'none'}."
        )

    labelled = frame[frame[value_column].notna()].reset_index(drop=True)
    if labelled.empty:
        raise PreprocessingError(
            f"dataset {dataset_path} has no row with a value for target "
            f"{target_name.value!r}; there is nothing to evaluate against"
        )

    predictor_columns = select_predictor_columns(
        [str(c) for c in labelled.columns], catalog
    )
    if feature_names is not None:
        wanted = set(feature_names)
        predictor_columns = tuple(
            column
            for column in predictor_columns
            if _feature_of(column) is None or _feature_of(column) in wanted
        )
    if not predictor_columns:
        raise PreprocessingError(
            "no permitted predictor column survives the requested feature "
            "subset; there is nothing to fit"
        )
    assert_no_leakage(predictor_columns, catalog, target_name=target_name)

    predictors = labelled.loc[:, list(predictor_columns)].copy()
    for column in predictors.columns:
        if predictors[column].dtype == "bool" or predictors[column].dtype == "boolean":
            predictors[column] = predictors[column].astype("float64")
        else:
            predictors[column] = pd.to_numeric(predictors[column], errors="coerce")

    spec = get_target_spec(target_name)
    if spec.task_type is TaskType.CLASSIFICATION:
        target_values = labelled[value_column].astype(str).to_numpy()
    else:
        target_values = labelled[value_column].astype("float64").to_numpy()
        if not np.isfinite(target_values).all():
            raise PreprocessingError(
                f"target {target_name.value!r} contains a non-finite value; a "
                "regression target must be finite"
            )

    for column in ("subject_id", "session_id", "window_id"):
        if column not in labelled.columns:
            raise PreprocessingError(
                f"dataset {dataset_path} has no {column!r} column; grouping and "
                "provenance cannot be established"
            )
    assert_no_identifier_values(labelled["subject_id"].tolist())
    assert_no_identifier_values(labelled["session_id"].tolist())

    permitted_column = target_meta_column(
        target_name, "scientific_evaluation_permitted"
    )
    source_column = target_meta_column(target_name, "source_type")
    permitted = (
        tuple(bool(v) for v in labelled[permitted_column].fillna(False))
        if permitted_column in labelled.columns
        else tuple(False for _ in range(len(labelled)))
    )
    sources = (
        tuple(str(v) for v in labelled[source_column].fillna("unstated"))
        if source_column in labelled.columns
        else tuple("unstated" for _ in range(len(labelled)))
    )

    return ModellingFrame(
        predictors=predictors,
        target_values=target_values,
        target_name=target_name,
        task_type=spec.task_type,
        subject_ids=tuple(str(v) for v in labelled["subject_id"]),
        session_ids=tuple(str(v) for v in labelled["session_id"]),
        window_ids=tuple(str(v) for v in labelled["window_id"]),
        data_sources=tuple(str(v) for v in labelled["data_source"]),
        target_source_types=sources,
        target_scientific_permitted=permitted,
        metadata=metadata,
    )


def _feature_of(column: str) -> str | None:
    """Catalog feature a predictor column belongs to, or ``None``."""
    if column.startswith("feat__"):
        return column.removeprefix("feat__")
    if column.startswith(AVAILABILITY_PREFIX):
        return column.removeprefix(AVAILABILITY_PREFIX)
    return None


def split_column_kinds(
    columns: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition predictor columns into measured values and flags.

    Availability and modality-availability columns are binary indicators
    that are never missing; scaling or imputing them would be meaningless.
    """
    flags = tuple(
        c
        for c in columns
        if c.startswith((AVAILABILITY_PREFIX, MODALITY_AVAILABLE_PREFIX))
    )
    measured = tuple(c for c in columns if c not in set(flags))
    return measured, flags


def build_preprocessor(
    columns: Sequence[str],
    *,
    strategy: ImputationStrategy,
    scale: bool,
) -> ColumnTransformer:
    """Build an unfitted preprocessor for ``columns``.

    Parameters
    ----------
    strategy:
        ``MEDIAN_WITH_INDICATOR`` imputes numeric columns with the training
        fold's median and appends a missingness indicator.  ``NATIVE_NAN``
        leaves values untouched for estimators that handle ``NaN``
        directly.
    scale:
        Standardise measured columns.  Required for penalised linear
        models, whose coefficients are otherwise governed by unit choice;
        omitted for tree ensembles, which are invariant to monotone
        rescaling and whose coefficients are easier to read unscaled.

    Raises
    ------
    PreprocessingError
        If ``columns`` is empty.
    """
    if not columns:
        raise PreprocessingError("cannot build a preprocessor with no columns")
    measured, flags = split_column_kinds(columns)

    steps: list[tuple[str, Any]] = []
    if strategy is ImputationStrategy.MEDIAN_WITH_INDICATOR:
        steps.append(
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            )
        )
    else:
        steps.append(("identity", FunctionTransformer(feature_names_out="one-to-one")))
    if scale:
        steps.append(("scale", StandardScaler()))
    measured_pipeline = Pipeline(steps)

    transformers: list[tuple[str, Any, list[str]]] = []
    if measured:
        transformers.append(("measured", measured_pipeline, list(measured)))
    if flags:
        transformers.append(
            (
                "flags",
                FunctionTransformer(feature_names_out="one-to-one"),
                list(flags),
            )
        )
    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_feature_names(preprocessor: ColumnTransformer) -> tuple[str, ...]:
    """Feature names produced by a **fitted** preprocessor.

    Raises
    ------
    PreprocessingError
        If the preprocessor has not been fitted.
    """
    try:
        names = preprocessor.get_feature_names_out()
    except Exception as exc:  # sklearn raises NotFittedError and ValueError
        raise PreprocessingError(
            "cannot read feature names from an unfitted preprocessor"
        ) from exc
    return tuple(str(name) for name in names)


def imputation_parameters(preprocessor: ColumnTransformer) -> dict[str, object]:
    """Extract fitted imputation and scaling parameters for the run artifact.

    Recording these makes a stored model auditable: a reader can see the
    exact median substituted for each column and the exact scaling applied,
    without executing the estimator file.
    """
    recorded: dict[str, object] = {}
    for name, transformer, columns in getattr(preprocessor, "transformers_", []):
        if name != "measured" or not hasattr(transformer, "named_steps"):
            continue
        steps = transformer.named_steps
        imputer = steps.get("impute")
        if imputer is not None and hasattr(imputer, "statistics_"):
            recorded["imputation_strategy"] = "median"
            recorded["imputation_statistics"] = {
                str(column): _finite_or_none(value)
                for column, value in zip(columns, imputer.statistics_, strict=True)
            }
        scaler = steps.get("scale")
        if scaler is not None and hasattr(scaler, "mean_"):
            recorded["scaling"] = "standard_score"
            recorded["scaling_mean"] = [float(v) for v in scaler.mean_]
            recorded["scaling_scale"] = [float(v) for v in scaler.scale_]
    return recorded


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def assert_columns_declared(columns: Sequence[str], catalog: FeatureCatalog) -> None:
    """Assert every column is declared by ``catalog`` and permitted.

    Raises
    ------
    DatasetValidationError
        On the first undeclared or non-permitted column.
    """
    try:
        assert_no_leakage(columns, catalog)
    except Exception as exc:
        raise DatasetValidationError(str(exc)) from exc
