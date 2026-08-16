"""Orchestration of a grouped cross-validated baseline evaluation.

The control flow is deliberately explicit rather than hidden behind a
scikit-learn convenience wrapper, because the ordering constraints are the
substance of this milestone:

1. Load and audit the dataset; refuse it in scientific mode if any
   predictor or target is synthetic, or if target provenance is missing.
2. Choose the grouping field and build the outer folds **once**.  Every
   model and every ablation reuses those exact folds.
3. Inside each fold: reserve calibration groups from the training groups,
   fit the pipeline on the fit groups only, fit the calibrator on the
   calibration groups only, and touch the test groups only to score.
4. Compute interpretation on held-out fold data, store it per fold, and
   aggregate afterwards.
5. Write every artifact, then the manifest, atomically and last.

No "champion" is selected.  A synthetic self-check cannot rank models for
any purpose that matters, and this code will not pretend otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold

from engagevr.features.catalog import FEATURE_CATALOG_VERSION, get_catalog
from engagevr.features.windowing import utc_now
from engagevr.schemas.experiments import (
    SELF_CHECK_DISCLAIMER,
    SOFTWARE_SELF_CHECK_BANNER,
    AblationDocument,
    AblationResult,
    AggregateMetric,
    ClassificationMetrics,
    EvaluationMode,
    FoldAssignment,
    MetricsDocument,
    ModelResult,
    RegressionMetrics,
    RunManifest,
    RunStatus,
    SplitManifest,
)
from engagevr.schemas.features import FeatureCatalog
from engagevr.schemas.targets import (
    TARGET_DISCLAIMER,
    TargetName,
    TaskType,
    get_target_spec,
)
from engagevr.training.ablation import ABLATION_SPECS, resolve_ablation_features
from engagevr.training.artifacts import (
    ExperimentRun,
    build_run_id,
    dependency_versions,
    engagevr_version,
    importance_table,
    predictions_table,
    runtime_environment,
)
from engagevr.training.calibration import (
    CalibrationMethod,
    aligned_probabilities,
    calibrate_classifier,
)
from engagevr.training.metrics import (
    CLASSIFICATION_AGGREGATE_FIELDS,
    DEFAULT_CALIBRATION_BINS,
    REGRESSION_AGGREGATE_FIELDS,
    aggregate_calibration_metrics,
    aggregate_fold_metrics,
    calibration_metrics,
    classification_metrics,
    regression_metrics,
    sum_confusion_matrices,
)
from engagevr.training.models import (
    ModelKind,
    ModelSpec,
    build_pipeline,
    describe_parameters,
    get_model_spec,
    registry_for,
)
from engagevr.training.preprocessing import (
    ModellingFrame,
    imputation_parameters,
    load_modelling_frame,
    transformed_feature_names,
)
from engagevr.training.splits import build_splits, choose_group_field

#: Feature each rule baseline thresholds, per target.
#:
#: These are **arbitrary implementation choices** made so the rule baseline
#: has something to threshold. Neither is a validated indicator: task
#: accuracy is not engagement, and reaction time is not cognitive load.
RULE_FEATURE_BY_TARGET: dict[TargetName, str] = {
    TargetName.ENGAGEMENT_CLASS: "feat__task_correct_proportion",
    TargetName.ENGAGEMENT_SCORE: "feat__task_correct_proportion",
    TargetName.COGNITIVE_LOAD_CLASS: "feat__task_reaction_time_mean_ms",
    TargetName.COGNITIVE_LOAD_SCORE: "feat__task_reaction_time_mean_ms",
}

#: Model used for the ablation comparison. One model is used across every
#: subset so that the comparison isolates the feature set.
ABLATION_MODEL_BY_TASK: dict[TaskType, str] = {
    TaskType.CLASSIFICATION: "logistic_regression",
    TaskType.REGRESSION: "ridge",
}


class EvaluationError(ValueError):
    """A run cannot proceed as configured."""


class ScientificModeError(EvaluationError):
    """A dataset is ineligible for scientific evaluation."""


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    """Everything that defines one evaluation run."""

    dataset_path: Path
    target_name: TargetName
    output_directory: Path
    evaluation_mode: EvaluationMode = EvaluationMode.SOFTWARE_SELF_CHECK
    n_splits: int = 5
    random_seed: int = 42
    model_names: tuple[str, ...] = ()
    calibration_method: CalibrationMethod = CalibrationMethod.SIGMOID
    calibration_group_fraction: float = 0.25
    calibration_bins: int = DEFAULT_CALIBRATION_BINS
    permutation_repeats: int = 5
    run_ablations: bool = True
    tune_hyperparameters: bool = False
    catalog_version: str = FEATURE_CATALOG_VERSION


@dataclass(slots=True)
class RunResult:
    """What a completed run produced."""

    run_id: str
    directory: Path
    metrics: MetricsDocument
    splits: SplitManifest
    manifest: RunManifest
    ablations: AblationDocument | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def assert_scientific_eligibility(frame: ModellingFrame) -> None:
    """Refuse a scientific evaluation on synthetic or unattributed data.

    Raises
    ------
    ScientificModeError
        If any row is synthetic, any target forbids scientific evaluation,
        or any target lacks a stated source type.
    """
    synthetic = sorted({s for s in frame.data_sources if s == "synthetic"})
    if synthetic:
        count = sum(1 for s in frame.data_sources if s == "synthetic")
        raise ScientificModeError(
            f"scientific evaluation refused: {count} of {frame.row_count} rows "
            "carry data_source='synthetic'. Synthetic data verifies software; "
            "it is never evidence about a person. Use the software self-check "
            "mode instead."
        )
    forbidden = sum(1 for p in frame.target_scientific_permitted if not p)
    if forbidden:
        raise ScientificModeError(
            f"scientific evaluation refused: {forbidden} of {frame.row_count} "
            f"target values for {frame.target_name.value!r} set "
            "scientific_evaluation_permitted=false"
        )
    unstated = sorted(
        {s for s in frame.target_source_types if s in ("", "unstated", "None")}
    )
    if unstated:
        raise ScientificModeError(
            "scientific evaluation refused: target provenance is missing. "
            "Every target value must state a documented source type "
            "(subjective_self_report, experiment_condition, expert_annotation, "
            "or public_dataset_annotation)."
        )


def run_baselines(config: RunConfiguration) -> RunResult:
    """Execute a complete grouped cross-validated baseline evaluation."""
    started = utc_now()
    catalog = get_catalog(config.catalog_version)
    spec = get_target_spec(config.target_name)

    frame = load_modelling_frame(
        config.dataset_path, target_name=config.target_name, catalog=catalog
    )
    if config.evaluation_mode is EvaluationMode.SCIENTIFIC:
        assert_scientific_eligibility(frame)

    model_names = config.model_names or tuple(sorted(registry_for(spec.task_type)))
    specs = [get_model_spec(name, spec.task_type) for name in model_names]

    group_field, group_reason = choose_group_field(frame.subject_ids, frame.session_ids)
    group_values = (
        frame.subject_ids if group_field.value == "subject_id" else frame.session_ids
    )
    class_labels = (
        list(frame.class_labels())
        if spec.task_type is TaskType.CLASSIFICATION
        else None
    )
    numeric_targets = (
        list(frame.numeric_targets()) if spec.task_type is TaskType.REGRESSION else None
    )
    splits = build_splits(
        group_values=group_values,
        session_ids=frame.session_ids,
        task_type=spec.task_type,
        group_field=group_field,
        group_field_reason=group_reason,
        n_splits=config.n_splits,
        random_seed=config.random_seed,
        class_labels=class_labels,
        numeric_targets=numeric_targets,
        calibration_group_fraction=config.calibration_group_fraction,
    )

    labels = tuple(spec.class_vocabulary or ())
    if spec.task_type is TaskType.CLASSIFICATION:
        observed = sorted(set(class_labels or []))
        labels = tuple(label for label in labels if label in set(observed))
        if len(labels) < 2:
            raise EvaluationError(
                f"target {config.target_name.value!r} has {len(labels)} class(es) "
                "present in the dataset; classification requires at least two"
            )

    run_id = build_run_id(
        target_name=config.target_name.value,
        evaluation_mode=config.evaluation_mode.value,
        dataset_fingerprint=frame.metadata.dataset_fingerprint,
        random_seed=config.random_seed,
        fold_count=config.n_splits,
        model_names=model_names,
        feature_set=frame.predictor_columns,
        calibration_method=config.calibration_method.value,
    )
    run = ExperimentRun(config.output_directory, run_id)

    disclaimers = _disclaimers(config.evaluation_mode)
    warnings: list[str] = []
    prediction_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    results: list[ModelResult] = []

    try:
        for model_spec in specs:
            result = _evaluate_model(
                model_spec,
                config=config,
                frame=frame,
                splits=splits,
                group_values=group_values,
                labels=labels,
                run=run,
                prediction_rows=prediction_rows,
                importance_rows=importance_rows,
                calibration_records=calibration_records,
                warnings=warnings,
            )
            results.append(result)

        metrics = MetricsDocument(
            run_id=run_id,
            evaluation_mode=config.evaluation_mode,
            scientific_evaluation_eligible=(
                config.evaluation_mode is EvaluationMode.SCIENTIFIC
            ),
            target_name=config.target_name.value,
            task_type=spec.task_type.value,
            dataset_fingerprint=frame.metadata.dataset_fingerprint,
            group_field=group_field.value,
            group_count=splits.group_count,
            fold_count=splits.n_splits,
            random_seed=config.random_seed,
            results=tuple(results),
            disclaimers=disclaimers,
        )

        ablations: AblationDocument | None = None
        if config.run_ablations:
            ablations = _run_ablations(
                config=config,
                catalog=catalog,
                frame=frame,
                splits=splits,
                group_values=group_values,
                labels=labels,
                disclaimers=disclaimers,
                run_id=run_id,
            )

        run.write_json("dataset.json", frame.metadata.model_dump(mode="json"))
        run.write_json("feature_catalog.json", catalog.model_dump(mode="json"))
        run.write_json("splits.json", splits.model_dump(mode="json"))
        run.write_json("metrics.json", metrics.model_dump(mode="json"))
        run.write_json(
            "calibration.json",
            {
                "run_id": run_id,
                "requested_method": config.calibration_method.value,
                "calibration_group_fraction": config.calibration_group_fraction,
                "ece_bin_count": config.calibration_bins,
                "design": (
                    "The base estimator is fitted on the fit groups; the "
                    "calibrator is fitted on calibration groups drawn from the "
                    "training groups and disjoint from them; the outer test "
                    "groups are never used to fit anything."
                ),
                "note": (
                    "A calibrated probability is not certainty and is not "
                    "signal quality."
                ),
                "folds": calibration_records,
                "disclaimers": list(disclaimers),
            },
        )
        run.write_json(
            "ablations.json",
            (
                ablations.model_dump(mode="json")
                if ablations is not None
                else {
                    "run_id": run_id,
                    "results": [],
                    "note": "ablations were not requested for this run",
                    "disclaimers": list(disclaimers),
                }
            ),
        )
        run.write_table(
            "predictions.parquet", _predictions_table(prediction_rows, labels)
        )
        run.write_table("feature_importance.parquet", importance_table(importance_rows))
        run.write_model_warning()

        manifest = _manifest(
            run_id=run_id,
            config=config,
            frame=frame,
            splits=splits,
            specs=specs,
            started=started,
            finished=utc_now(),
            status=RunStatus.COMPLETED,
            failure_reason=None,
            disclaimers=disclaimers,
        )
        run.finalize(manifest)
    except Exception as exc:
        manifest = _manifest(
            run_id=run_id,
            config=config,
            frame=frame,
            splits=splits,
            specs=specs,
            started=started,
            finished=utc_now(),
            status=RunStatus.FAILED,
            failure_reason=f"{type(exc).__name__}: {exc}",
            disclaimers=disclaimers,
        )
        run.finalize(manifest)
        raise

    return RunResult(
        run_id=run_id,
        directory=run.directory,
        metrics=metrics,
        splits=splits,
        manifest=manifest,
        ablations=ablations,
        warnings=tuple(warnings),
    )


def _disclaimers(mode: EvaluationMode) -> tuple[str, ...]:
    if mode is EvaluationMode.SOFTWARE_SELF_CHECK:
        return (SELF_CHECK_DISCLAIMER, TARGET_DISCLAIMER)
    return (
        TARGET_DISCLAIMER,
        "This run was executed in scientific mode. Eligibility was checked "
        "against data source and target provenance; eligibility is not "
        "validity, and no claim of experimental validation follows from it.",
    )


def _rows_for_groups(group_values: Sequence[str], wanted: Sequence[str]) -> np.ndarray:
    members = set(wanted)
    return np.asarray(
        [index for index, group in enumerate(group_values) if group in members],
        dtype=int,
    )


def _evaluate_model(
    model_spec: ModelSpec,
    *,
    config: RunConfiguration,
    frame: ModellingFrame,
    splits: SplitManifest,
    group_values: Sequence[str],
    labels: tuple[str, ...],
    run: ExperimentRun | None,
    prediction_rows: list[dict[str, Any]],
    importance_rows: list[dict[str, Any]],
    calibration_records: list[dict[str, Any]],
    warnings: list[str],
    feature_columns: Sequence[str] | None = None,
    record_artifacts: bool = True,
) -> ModelResult:
    """Evaluate one model across every valid outer fold."""
    columns = list(feature_columns or frame.predictor_columns)
    classification = frame.task_type is TaskType.CLASSIFICATION
    fold_classification: list[ClassificationMetrics] = []
    fold_regression: list[RegressionMetrics] = []
    fold_calibrations: list[tuple[Any, ...]] = []
    failed: dict[int, str] = {}
    notes: list[str] = list(model_spec.notes)
    matrices = []

    for fold in splits.folds:
        if not fold.valid:
            failed[fold.fold_index] = fold.invalid_reason or "fold marked invalid"
            continue
        try:
            outcome = _evaluate_fold(
                model_spec,
                config=config,
                frame=frame,
                fold=fold,
                group_values=group_values,
                labels=labels,
                columns=columns,
                run=run if record_artifacts else None,
                prediction_rows=prediction_rows if record_artifacts else None,
                importance_rows=importance_rows if record_artifacts else None,
                calibration_records=(calibration_records if record_artifacts else None),
                warnings=warnings,
            )
        except Exception as exc:
            failed[fold.fold_index] = f"{type(exc).__name__}: {exc}"
            continue
        if classification:
            fold_classification.append(outcome["metrics"])
            fold_calibrations.append(outcome["metrics"].calibration)
            matrices.append(outcome["metrics"].confusion_matrix)
        else:
            fold_regression.append(outcome["metrics"])

    total_folds = splits.n_splits
    if classification:
        aggregate = list(
            aggregate_fold_metrics(
                fold_classification,
                CLASSIFICATION_AGGREGATE_FIELDS,
                total_fold_count=total_folds,
            )
        )
        for label in ("uncalibrated", config.calibration_method.value):
            aggregate.extend(
                aggregate_calibration_metrics(
                    fold_calibrations, label=label, total_fold_count=total_folds
                )
            )
        aggregate_matrix = sum_confusion_matrices(
            [m for m in matrices if m is not None]
        )
    else:
        aggregate = list(
            aggregate_fold_metrics(
                fold_regression,
                REGRESSION_AGGREGATE_FIELDS,
                total_fold_count=total_folds,
            )
        )
        aggregate_matrix = None

    if failed:
        notes.append(
            f"{len(failed)} of {total_folds} folds produced no metrics; see "
            "failed_folds. Aggregates are means over the remaining folds only."
        )

    return ModelResult(
        model_name=model_spec.name,
        model_kind=model_spec.kind.value,
        is_software_check_baseline=model_spec.is_software_check_baseline,
        parameters={
            "description": model_spec.description,
            "imputation": model_spec.imputation.value,
            "standardised": model_spec.scale,
            "random_seed": config.random_seed,
        },
        predictor_columns=tuple(columns),
        fold_classification_metrics=tuple(fold_classification),
        fold_regression_metrics=tuple(fold_regression),
        aggregate=tuple(aggregate),
        aggregate_confusion_matrix=aggregate_matrix,
        failed_folds=failed,
        notes=tuple(notes),
    )


def _evaluate_fold(
    model_spec: ModelSpec,
    *,
    config: RunConfiguration,
    frame: ModellingFrame,
    fold: FoldAssignment,
    group_values: Sequence[str],
    labels: tuple[str, ...],
    columns: Sequence[str],
    run: ExperimentRun | None,
    prediction_rows: list[dict[str, Any]] | None,
    importance_rows: list[dict[str, Any]] | None,
    calibration_records: list[dict[str, Any]] | None,
    warnings: list[str],
) -> dict[str, Any]:
    """Fit and score one model on one outer fold."""
    classification = frame.task_type is TaskType.CLASSIFICATION
    fit_groups = fold.fit_groups()
    calibration_groups = fold.calibration_groups
    test_groups = fold.test_groups

    fit_idx = _rows_for_groups(group_values, fit_groups)
    calibration_idx = _rows_for_groups(group_values, calibration_groups)
    test_idx = _rows_for_groups(group_values, test_groups)
    if fit_idx.size == 0 or test_idx.size == 0:
        raise EvaluationError(
            f"fold {fold.fold_index}: the fit or test portion is empty"
        )

    X = frame.predictors.loc[:, list(columns)]
    X_fit = X.iloc[fit_idx]
    X_calibration = X.iloc[calibration_idx]
    X_test = X.iloc[test_idx]
    y = frame.target_values
    y_fit = y[fit_idx]
    y_calibration = y[calibration_idx]
    y_test = y[test_idx]

    pipeline = build_pipeline(
        model_spec,
        columns,
        random_seed=config.random_seed,
        rule_feature=RULE_FEATURE_BY_TARGET.get(frame.target_name),
        rule_class_order=labels or None,
    )
    if config.tune_hyperparameters and model_spec.parameter_grid:
        pipeline = _tune(
            pipeline,
            model_spec,
            X_fit,
            y_fit,
            groups=[group_values[i] for i in fit_idx],
            classification=classification,
            random_seed=config.random_seed,
        )

    calibration_entries: list[Any] = []
    calibrated_estimator: BaseEstimator | None = None

    if classification:
        base, outcome = calibrate_classifier(
            pipeline,
            X_fit=X_fit,
            y_fit=[str(v) for v in y_fit],
            X_calibration=X_calibration,
            y_calibration=[str(v) for v in y_calibration],
            method=config.calibration_method,
            fit_groups=fit_groups,
            calibration_groups=calibration_groups,
            test_groups=test_groups,
        )
        calibrated_estimator = outcome.calibrated_estimator
        if calibration_records is not None:
            calibration_records.append(
                {
                    "model_name": model_spec.name,
                    "fold_index": fold.fold_index,
                    "method": outcome.method.value,
                    "calibrated": outcome.available,
                    "fit_row_count": outcome.fit_row_count,
                    "calibration_row_count": outcome.calibration_row_count,
                    "fit_group_count": outcome.fit_group_count,
                    "calibration_group_count": outcome.calibration_group_count,
                    "fit_groups": list(fit_groups),
                    "calibration_groups": list(calibration_groups),
                    "test_groups": list(test_groups),
                    "unavailable_reason": outcome.unavailable_reason,
                }
            )
    else:
        base = clone(pipeline)
        base.fit(X_fit, y_fit)

    predictions = base.predict(X_test)
    uncalibrated_proba = (
        aligned_probabilities(base, X_test, labels) if classification else None
    )
    calibrated_proba = (
        aligned_probabilities(calibrated_estimator, X_test, labels)
        if classification and calibrated_estimator is not None
        else None
    )

    if classification:
        calibration_entries = [
            calibration_metrics(
                label="uncalibrated",
                probabilities=uncalibrated_proba,
                y_true=[str(v) for v in y_test],
                labels=labels,
                bin_count=config.calibration_bins,
            ),
            calibration_metrics(
                label=config.calibration_method.value,
                probabilities=calibrated_proba,
                y_true=[str(v) for v in y_test],
                labels=labels,
                bin_count=config.calibration_bins,
            ),
        ]
        metrics: Any = classification_metrics(
            y_true=[str(v) for v in y_test],
            y_predicted=[str(v) for v in predictions],
            labels=labels,
            group_ids=[group_values[i] for i in test_idx],
            calibration=calibration_entries,
        )
    else:
        metrics = regression_metrics(
            y_true=[float(v) for v in y_test],
            y_predicted=[float(v) for v in predictions],
            group_ids=[group_values[i] for i in test_idx],
        )

    if prediction_rows is not None:
        for position, row_index in enumerate(test_idx):
            prediction_rows.append(
                {
                    "window_id": frame.window_ids[row_index],
                    "subject_id": frame.subject_ids[row_index],
                    "session_id": frame.session_ids[row_index],
                    "fold_index": fold.fold_index,
                    "model_name": model_spec.name,
                    "true_value": y_test[position],
                    "predicted_value": predictions[position],
                    "uncalibrated": (
                        None
                        if uncalibrated_proba is None
                        else uncalibrated_proba[position].tolist()
                    ),
                    "calibrated": (
                        None
                        if calibrated_proba is None
                        else calibrated_proba[position].tolist()
                    ),
                }
            )

    if importance_rows is not None:
        importance_rows.extend(
            _interpretation(
                model_spec,
                base,
                X_test=X_test,
                y_test=y_test,
                fold_index=fold.fold_index,
                metrics=metrics,
                classification=classification,
                config=config,
                warnings=warnings,
            )
        )

    if run is not None and fold.fold_index == 0:
        run.save_model(f"{model_spec.name}-fold0", base)
        if calibrated_estimator is not None:
            run.save_model(f"{model_spec.name}-fold0-calibrated", calibrated_estimator)

    return {"metrics": metrics, "estimator": base}


def _tune(
    pipeline: Any,
    model_spec: ModelSpec,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    *,
    groups: Sequence[str],
    classification: bool,
    random_seed: int,
) -> Any:
    """Select hyperparameters with a group-aware inner search.

    The inner search never sees the outer test groups: it runs entirely
    within the fold's fit groups.  Falls back to the fixed-parameter
    pipeline when there are too few inner groups to split.
    """
    distinct = sorted(set(groups))
    inner_splits = min(3, len(distinct))
    if inner_splits < 2:
        return pipeline
    inner = (
        StratifiedGroupKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_seed
        )
        if classification
        else GroupKFold(n_splits=inner_splits, shuffle=True, random_state=random_seed)
    )
    search = GridSearchCV(
        pipeline,
        param_grid=model_spec.parameter_grid,
        cv=inner,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X_fit, y_fit, groups=list(groups))
    return search.best_estimator_


def _interpretation(
    model_spec: ModelSpec,
    estimator: Any,
    *,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    fold_index: int,
    metrics: Any,
    classification: bool,
    config: RunConfiguration,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Fold-level interpretation records, computed on held-out data.

    Association is not causation, and a feature a model leans on is not a
    measurement of the construct being modelled. Correlated features share
    credit arbitrarily: with two near-duplicate inputs, a linear model may
    split a coefficient between them and a permutation test may score both
    as unimportant because either alone substitutes for the other.
    """
    records: list[dict[str, Any]] = []
    warning = _chance_level_warning(metrics, classification)
    if warning is not None:
        warnings.append(
            f"{model_spec.name} fold {fold_index}: {warning} Interpretation "
            "data is still recorded, but reading importances from a model that "
            "does not beat chance describes noise."
        )

    if model_spec.kind is ModelKind.LINEAR:
        records.extend(_linear_records(model_spec, estimator, fold_index, warning))
    elif model_spec.kind is ModelKind.TREE:
        records.extend(
            _permutation_records(
                model_spec,
                estimator,
                X_test,
                y_test,
                fold_index,
                config,
                warning,
            )
        )
    return records


def linear_interpretation_records(
    model_spec: ModelSpec,
    estimator: Any,
    fold_index: int,
    warning: str | None = None,
) -> list[dict[str, Any]]:
    """Fold-level linear-coefficient records for a fitted pipeline.

    Exposed so the Milestone 6 fusion runner can record the same
    interpretation data for its early-fusion model and its linear modality
    experts without duplicating the extraction logic.  Coefficients are read
    from the fitted estimator only; nothing is refitted.
    """
    return _linear_records(model_spec, estimator, fold_index, warning)


def _chance_level_warning(metrics: Any, classification: bool) -> str | None:
    if classification:
        support = getattr(metrics, "class_support", {})
        total = sum(support.values())
        accuracy = getattr(metrics, "accuracy", None)
        if not total or accuracy is None:
            return None
        majority = max(support.values()) / total
        if accuracy <= majority + 1e-12:
            return (
                f"accuracy {accuracy:.3f} does not exceed the majority-class "
                f"rate {majority:.3f}."
            )
        return None
    r_squared = getattr(metrics, "r_squared", None)
    if r_squared is not None and r_squared <= 0.0:
        return (
            f"R-squared {r_squared:.3f} is at or below zero, so the model does "
            "not beat predicting the mean."
        )
    return None


def _linear_records(
    model_spec: ModelSpec,
    estimator: Any,
    fold_index: int,
    warning: str | None,
) -> list[dict[str, Any]]:
    final = estimator.named_steps["estimator"]
    preprocessor = estimator.named_steps["preprocess"]
    try:
        names = transformed_feature_names(preprocessor)
    except Exception:  # pragma: no cover - fitted pipelines always expose names
        return []
    coefficients = np.asarray(getattr(final, "coef_", []), dtype=float)
    if coefficients.size == 0:
        return []
    scaling = (
        "features standardised on the training fold; a coefficient is the "
        "effect of a one-standard-deviation change"
        if model_spec.scale
        else "features on their original scale; coefficients are not comparable "
        "across features with different units"
    )
    records: list[dict[str, Any]] = []
    rows: list[tuple[str | None, Any]]
    if coefficients.ndim == 1:
        rows = [(None, coefficients)]
    else:
        classes = [str(c) for c in getattr(final, "classes_", [])]
        rows = [
            (classes[index] if index < len(classes) else str(index), row)
            for index, row in enumerate(coefficients)
        ]
    for class_label, row in rows:
        for name, value in zip(names, row, strict=False):
            number = float(value)
            records.append(
                {
                    "model_name": model_spec.name,
                    "fold_index": fold_index,
                    "feature_name": name,
                    "kind": "linear_coefficient",
                    "value": number,
                    "absolute_value": abs(number),
                    "sign": int(np.sign(number)),
                    "standard_deviation": None,
                    "repeat_count": None,
                    "scaling_context": scaling,
                    "class_label": class_label,
                    "warning": warning,
                }
            )
    return records


def _permutation_records(
    model_spec: ModelSpec,
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    fold_index: int,
    config: RunConfiguration,
    warning: str | None,
) -> list[dict[str, Any]]:
    if len(X_test) < 2:
        return []
    # The preprocessor is applied once and the permutation is then run
    # against the final estimator's own input matrix. Re-running the
    # ColumnTransformer for every one of the (features x repeats)
    # permutations costs far more than the fit itself and changes nothing:
    # the transformer is already fitted and deterministic. Missingness
    # indicator columns are permuted in their own right, which is why they
    # appear as named features in the output.
    preprocessor = estimator.named_steps["preprocess"]
    final = estimator.named_steps["estimator"]
    transformed = preprocessor.transform(X_test)
    if not isinstance(transformed, pd.DataFrame):  # pragma: no cover - pandas output
        transformed = pd.DataFrame(
            transformed, columns=list(transformed_feature_names(preprocessor))
        )
    result = permutation_importance(
        final,
        transformed,
        y_test,
        n_repeats=config.permutation_repeats,
        random_state=config.random_seed,
        n_jobs=1,
    )
    context = (
        "permutation importance computed on held-out fold data, against the "
        "estimator's post-preprocessing input matrix; impurity-based "
        "importance is not used because it is biased toward high-cardinality "
        "features and is computed on training data"
    )
    records: list[dict[str, Any]] = []
    for index, name in enumerate(transformed.columns):
        mean = float(result.importances_mean[index])
        records.append(
            {
                "model_name": model_spec.name,
                "fold_index": fold_index,
                "feature_name": str(name),
                "kind": "permutation_importance",
                "value": mean,
                "absolute_value": abs(mean),
                "sign": int(np.sign(mean)),
                "standard_deviation": float(result.importances_std[index]),
                "repeat_count": int(config.permutation_repeats),
                "scaling_context": context,
                "class_label": None,
                "warning": warning,
            }
        )
    return records


def _run_ablations(
    *,
    config: RunConfiguration,
    catalog: FeatureCatalog,
    frame: ModellingFrame,
    splits: SplitManifest,
    group_values: Sequence[str],
    labels: tuple[str, ...],
    disclaimers: tuple[str, ...],
    run_id: str,
) -> AblationDocument:
    """Evaluate every ablation on the same outer folds."""
    model_name = ABLATION_MODEL_BY_TASK[frame.task_type]
    model_spec = get_model_spec(model_name, frame.task_type)
    available = frozenset(
        column.removeprefix("feat__")
        for column in frame.predictor_columns
        if column.startswith("feat__")
    )
    fields = (
        CLASSIFICATION_AGGREGATE_FIELDS
        if frame.task_type is TaskType.CLASSIFICATION
        else REGRESSION_AGGREGATE_FIELDS
    )

    results: list[AblationResult] = []
    for ablation in ABLATION_SPECS:
        features, reason = resolve_ablation_features(ablation, catalog, available)
        if reason is not None:
            results.append(
                AblationResult(
                    ablation_name=ablation.name,
                    included_modalities=tuple(m.value for m in ablation.modalities),
                    feature_names=(),
                    available=False,
                    unavailable_reason=reason,
                )
            )
            continue
        wanted = set(features)
        columns = tuple(
            column
            for column in frame.predictor_columns
            if _column_feature(column) is None or _column_feature(column) in wanted
        )
        subset = _evaluate_model(
            model_spec,
            config=config,
            frame=frame,
            splits=splits,
            group_values=group_values,
            labels=labels,
            run=None,
            prediction_rows=[],
            importance_rows=[],
            calibration_records=[],
            warnings=[],
            feature_columns=columns,
            record_artifacts=False,
        )
        aggregate: tuple[AggregateMetric, ...] = tuple(
            entry for entry in subset.aggregate if entry.name in set(fields)
        )
        results.append(
            AblationResult(
                ablation_name=ablation.name,
                included_modalities=tuple(m.value for m in ablation.modalities),
                feature_names=features,
                available=True,
                model_name=model_name,
                aggregate=aggregate,
            )
        )

    return AblationDocument(
        run_id=run_id,
        evaluation_mode=config.evaluation_mode,
        target_name=config.target_name.value,
        shared_fold_count=splits.n_splits,
        split_strategy=splits.strategy.value,
        results=tuple(results),
        disclaimers=disclaimers,
    )


def _column_feature(column: str) -> str | None:
    if column.startswith("feat__"):
        return column.removeprefix("feat__")
    if column.startswith("avail__"):
        return column.removeprefix("avail__")
    return None


def _predictions_table(rows: Sequence[dict[str, Any]], labels: Sequence[str]) -> Any:
    if not rows:
        return predictions_table(
            window_ids=[],
            subject_ids=[],
            session_ids=[],
            fold_indices=[],
            model_names=[],
            true_values=[],
            predicted_values=[],
            probability_labels=labels,
            uncalibrated=[],
            calibrated=[],
        )
    return predictions_table(
        window_ids=[str(r["window_id"]) for r in rows],
        subject_ids=[str(r["subject_id"]) for r in rows],
        session_ids=[str(r["session_id"]) for r in rows],
        fold_indices=[int(r["fold_index"]) for r in rows],
        model_names=[str(r["model_name"]) for r in rows],
        true_values=[_scalar(r["true_value"]) for r in rows],
        predicted_values=[_scalar(r["predicted_value"]) for r in rows],
        probability_labels=labels,
        uncalibrated=[r["uncalibrated"] for r in rows],
        calibrated=[r["calibrated"] for r in rows],
    )


def _scalar(value: Any) -> str | float:
    if isinstance(value, str | np.str_):
        return str(value)
    return float(value)


def _manifest(
    *,
    run_id: str,
    config: RunConfiguration,
    frame: ModellingFrame,
    splits: SplitManifest,
    specs: Sequence[ModelSpec],
    started: datetime,
    finished: datetime,
    status: RunStatus,
    failure_reason: str | None,
    disclaimers: tuple[str, ...],
) -> RunManifest:
    environment = runtime_environment()
    parameters: dict[str, dict[str, object]] = {}
    for model_spec in specs:
        pipeline = build_pipeline(
            model_spec,
            frame.predictor_columns,
            random_seed=config.random_seed,
            rule_feature=RULE_FEATURE_BY_TARGET.get(frame.target_name),
        )
        described = describe_parameters(pipeline)
        described["imputation"] = model_spec.imputation.value
        described["standardised"] = model_spec.scale
        described["parameter_grid"] = {
            key: list(values) for key, values in model_spec.parameter_grid.items()
        }
        parameters[model_spec.name] = described

    return RunManifest(
        run_id=run_id,
        engagevr_version=engagevr_version(),
        python_version=environment["python_version"],
        dependency_versions=dependency_versions(),
        evaluation_mode=config.evaluation_mode,
        scientific_evaluation_eligible=(
            config.evaluation_mode is EvaluationMode.SCIENTIFIC
        ),
        dataset_path=str(config.dataset_path),
        dataset_fingerprint=frame.metadata.dataset_fingerprint,
        feature_catalog_version=frame.metadata.feature_catalog_version,
        target_name=config.target_name.value,
        task_type=frame.task_type.value,
        feature_set=frame.predictor_columns,
        model_names=tuple(s.name for s in specs),
        model_parameters=parameters,
        split_strategy=splits.strategy.value,
        group_field=splits.group_field.value,
        group_count=splits.group_count,
        fold_count=splits.n_splits,
        fold_assignments={
            str(fold.fold_index): {
                "train": fold.fit_groups(),
                "calibration": fold.calibration_groups,
                "test": fold.test_groups,
            }
            for fold in splits.folds
        },
        calibration_method=config.calibration_method.value,
        configuration={
            "n_splits": config.n_splits,
            "random_seed": config.random_seed,
            "calibration_group_fraction": config.calibration_group_fraction,
            "calibration_bins": config.calibration_bins,
            "permutation_repeats": config.permutation_repeats,
            "run_ablations": config.run_ablations,
            "tune_hyperparameters": config.tune_hyperparameters,
            "catalog_version": config.catalog_version,
        },
        random_seed=config.random_seed,
        started_at_utc=started,
        finished_at_utc=finished,
        status=status,
        failure_reason=failure_reason,
        disclaimers=disclaimers,
    )


def imputation_snapshot(estimator: Any) -> dict[str, object]:
    """Fitted imputation and scaling parameters of a pipeline, for artifacts."""
    return imputation_parameters(estimator.named_steps["preprocess"])


__all__ = [
    "ABLATION_MODEL_BY_TASK",
    "RULE_FEATURE_BY_TARGET",
    "SOFTWARE_SELF_CHECK_BANNER",
    "EvaluationError",
    "RunConfiguration",
    "RunResult",
    "ScientificModeError",
    "assert_scientific_eligibility",
    "imputation_snapshot",
    "linear_interpretation_records",
    "run_baselines",
]
