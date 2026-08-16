"""Orchestration of a leakage-safe personalized-versus-population evaluation.

The ordering constraints are the substance of this pass, so the control
flow is explicit:

1. Load and audit the dataset; refuse it in scientific mode if any
   predictor or target is synthetic, or if target provenance is missing.
2. Refuse unless the dataset groups by ``subject_id``.  Personalization is
   defined per person; a session-grouped dataset has no person to
   personalise to.
3. Build the outer folds **once**, with the same call the Milestone 5
   baseline runner and the Milestone 6 fusion runner make, and fingerprint
   the manifest so fold reuse is checkable.
4. Inside each fold, fit the **population reference model** on the fit
   groups only.  The held-out subjects appear in no training portion, and
   their calibration windows are not training data either.
5. For each held-out subject, cut their windows chronologically into a
   calibration region and a strictly later evaluation region, estimate the
   personal baseline and the correction from the calibration region alone,
   and apply them to the evaluation region.
6. Score the population model and the personalized model over **exactly the
   same evaluation windows**, then write every artifact and the manifest,
   atomically and last.

Two models, one set of rows
---------------------------
The population reference model is fitted on raw features.  When the
requested method normalises against a personal baseline, a second
estimator is fitted on personally-normalised features — each training
subject normalised by their own training windows, each held-out subject by
their calibration windows only.  Both are evaluated on the same rows, and
the population prediction is retained unchanged on every record.  A
personalized prediction is an addition to it, never a replacement.

Nothing here abstains.  Personalized confidence thresholds and selective
prediction are Milestone 7.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.base import BaseEstimator, clone

from engagevr.features.catalog import FEATURE_CATALOG_VERSION, get_catalog
from engagevr.features.windowing import utc_now
from engagevr.schemas.experiments import (
    SELF_CHECK_DISCLAIMER,
    EvaluationMode,
    FoldAssignment,
    GroupField,
    MetricsDocument,
    ModelResult,
    RunManifest,
    RunStatus,
    SplitManifest,
)
from engagevr.schemas.features import FeatureCatalog
from engagevr.schemas.personalization import (
    BASELINE_METHODS,
    CALIBRATION_MEANING_NOTE,
    CORRECTION_METHODS,
    PERSONALIZATION_NOTE,
    PersonalBaselineDocument,
    PersonalBaselineStatistics,
    PersonalCalibrationSplit,
    PersonalizationConfiguration,
    PersonalizationCorrection,
    PersonalizationEvaluation,
    PersonalizationExperimentManifest,
    PersonalizationFoldResult,
    PersonalizationMethod,
    PersonalizedPrediction,
)
from engagevr.schemas.targets import (
    TARGET_DISCLAIMER,
    TargetName,
    TaskType,
    get_target_spec,
)
from engagevr.training.artifacts import (
    ExperimentRun,
    dependency_versions,
    engagevr_version,
    runtime_environment,
)
from engagevr.training.calibration import (
    CalibrationMethod,
    aligned_probabilities,
    calibrate_classifier,
)
from engagevr.training.fusion import early_fusion_columns
from engagevr.training.fusion_artifacts import split_manifest_fingerprint
from engagevr.training.metrics import (
    CLASSIFICATION_AGGREGATE_FIELDS,
    DEFAULT_CALIBRATION_BINS,
    REGRESSION_AGGREGATE_FIELDS,
    aggregate_fold_metrics,
    classification_metrics,
    regression_metrics,
)
from engagevr.training.models import build_pipeline, describe_parameters, get_model_spec
from engagevr.training.personalization import (
    PersonalizationError,
    SubjectWindow,
    apply_classification_correction,
    apply_personal_baseline,
    apply_regression_correction,
    build_calibration_split,
    build_personalization_run_id,
    classification_correction,
    personal_baseline_statistics,
    personalizable_columns,
    regression_correction,
    subject_windows,
)
from engagevr.training.preprocessing import ModellingFrame, load_modelling_frame
from engagevr.training.runner import (
    EvaluationError,
    ScientificModeError,
    assert_scientific_eligibility,
)
from engagevr.training.splits import build_splits, choose_group_field

#: Artifacts a completed personalization run must contain.
PERSONALIZATION_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "dataset.json",
    "feature_catalog.json",
    "splits.json",
    "personalization_config.json",
    "personalization.json",
    "personal_baselines.json",
    "metrics.json",
)

#: ``model_name`` of the two separately reported results.
POPULATION_MODEL_NAME = "population"
PERSONALIZED_MODEL_NAME = "personalized"

#: Note attached to both reported results.
SEPARATE_REPORTING_NOTE = (
    "Population and personalized software-check results are reported "
    "separately over identical evaluation windows. Synthetic differences "
    "are not evidence of a personalization benefit."
)


class PersonalizationConfigurationError(EvaluationError):
    """A personalization run cannot proceed as configured."""


@dataclass(frozen=True, slots=True)
class PersonalizationRunConfiguration:
    """Everything that defines one personalization evaluation run."""

    dataset_path: Path
    target_name: TargetName
    output_directory: Path
    personalization: PersonalizationConfiguration
    evaluation_mode: EvaluationMode = EvaluationMode.SOFTWARE_SELF_CHECK
    n_splits: int = 5
    random_seed: int = 42
    calibration_method: CalibrationMethod = CalibrationMethod.SIGMOID
    calibration_group_fraction: float = 0.25
    calibration_bins: int = DEFAULT_CALIBRATION_BINS
    catalog_version: str = FEATURE_CATALOG_VERSION


@dataclass(slots=True)
class PersonalizationRunResult:
    """What a completed personalization run produced."""

    run_id: str
    directory: Path
    metrics: MetricsDocument
    personalization: PersonalizationEvaluation
    baselines: PersonalBaselineDocument
    splits: SplitManifest
    manifest: RunManifest
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class _FoldOutcome:
    """One fold's predictions, splits, corrections, and baselines."""

    result: PersonalizationFoldResult
    predictions: tuple[PersonalizedPrediction, ...]
    baselines: tuple[PersonalBaselineStatistics, ...]


def _disclaimers(mode: EvaluationMode) -> tuple[str, ...]:
    if mode is EvaluationMode.SOFTWARE_SELF_CHECK:
        return (SELF_CHECK_DISCLAIMER, TARGET_DISCLAIMER, PERSONALIZATION_NOTE)
    return (
        TARGET_DISCLAIMER,
        PERSONALIZATION_NOTE,
        "This personalization run was executed in scientific mode. "
        "Eligibility was checked against data source and target provenance; "
        "eligibility is not validity, and no claim of personalization "
        "benefit follows from a run completing.",
    )


def _required_regression_value(
    value: float | None,
    *,
    kind: str,
    prediction: PersonalizedPrediction,
) -> float:
    """The regression prediction, or a stated invariant violation.

    ``PersonalizedPrediction`` already refuses a regression record whose
    predicted values are absent or non-finite, so neither branch below is
    reachable from a validated record. They are kept as an assertion rather
    than as a fallback because the alternative — substituting ``0.0`` — would
    silently turn an unavailable prediction into a confident mid-scale one and
    score it against a real label. A missing estimate is not zero. A genuine
    ``0.0`` is a value like any other and passes through untouched.
    """
    if value is None:
        raise PersonalizationError(
            f"the {kind} regression prediction for window "
            f"{prediction.window_id} (subject {prediction.subject_id}, fold "
            f"{prediction.fold_index}) is missing, and a missing prediction is "
            "not zero. Regression scoring requires a value for every evaluated "
            "window; this record should have been refused when it was built."
        )
    numeric = float(value)
    if not np.isfinite(numeric):
        raise PersonalizationError(
            f"the {kind} regression prediction for window "
            f"{prediction.window_id} (subject {prediction.subject_id}, fold "
            f"{prediction.fold_index}) is {numeric}, which is not a finite "
            "estimate. Regression scoring requires finite values; this record "
            "should have been refused when it was built."
        )
    return numeric


def _rows_for_groups(group_values: Sequence[str], wanted: Sequence[str]) -> np.ndarray:
    members = set(wanted)
    return np.asarray(
        [index for index, group in enumerate(group_values) if group in members],
        dtype=int,
    )


def run_personalization(
    config: PersonalizationRunConfiguration,
) -> PersonalizationRunResult:
    """Execute a complete population-versus-personalized evaluation."""
    started = utc_now()
    catalog = get_catalog(config.catalog_version)
    spec = get_target_spec(config.target_name)
    settings = config.personalization

    frame = load_modelling_frame(
        config.dataset_path, target_name=config.target_name, catalog=catalog
    )
    if config.evaluation_mode is EvaluationMode.SCIENTIFIC:
        assert_scientific_eligibility(frame)

    predictor_columns = early_fusion_columns(
        settings.modalities,
        frame.predictor_columns,
        catalog,
        include_modality_quality=settings.include_modality_quality,
    )
    personalized_columns = personalizable_columns(
        settings.modalities, frame.predictor_columns, catalog
    )
    if settings.method in BASELINE_METHODS and not personalized_columns:
        raise PersonalizationConfigurationError(
            f"method {settings.method.value!r} normalises features against a "
            "personal baseline, but no catalogued measured feature of the "
            f"configured modality groups is present in {config.dataset_path}. "
            "The absence is reported rather than silently reduced to a "
            "population-only run."
        )

    group_field, group_reason = choose_group_field(frame.subject_ids, frame.session_ids)
    if group_field is not GroupField.SUBJECT_ID:
        raise PersonalizationConfigurationError(
            f"personalization requires subject grouping, but this dataset is "
            f"grouped by {group_field.value!r}: {group_reason}. A personal "
            "baseline is defined per person; without a subject identifier "
            "there is no person to personalise to, and the run is refused "
            "rather than personalising to a session."
        )
    group_values = frame.subject_ids

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

    labels: tuple[str, ...] = ()
    if spec.task_type is TaskType.CLASSIFICATION:
        observed = set(class_labels or [])
        labels = tuple(
            label for label in (spec.class_vocabulary or ()) if label in observed
        )
        if len(labels) < 2:
            raise PersonalizationConfigurationError(
                f"target {config.target_name.value!r} has {len(labels)} class(es) "
                "present in the dataset; classification requires at least two"
            )

    fingerprint = split_manifest_fingerprint(splits)
    run_id = build_personalization_run_id(
        target_name=config.target_name.value,
        task_type=spec.task_type.value,
        evaluation_mode=config.evaluation_mode.value,
        dataset_fingerprint=frame.metadata.dataset_fingerprint,
        split_manifest_fingerprint=fingerprint,
        random_seed=config.random_seed,
        configuration=settings,
        calibration_method=config.calibration_method.value,
        engagevr_version=engagevr_version(),
    )
    run = ExperimentRun(
        config.output_directory,
        run_id,
        required_artifacts=PERSONALIZATION_REQUIRED_ARTIFACTS,
    )
    disclaimers = _disclaimers(config.evaluation_mode)

    model_name = (
        settings.population_model_classification
        if spec.task_type is TaskType.CLASSIFICATION
        else settings.population_model_regression
    )
    model_spec = get_model_spec(model_name, spec.task_type)

    outcomes: list[_FoldOutcome] = []
    warnings: list[str] = []
    calibration_records: list[dict[str, Any]] = []

    try:
        for fold in splits.folds:
            if not fold.valid:
                outcomes.append(
                    _FoldOutcome(
                        result=PersonalizationFoldResult(
                            fold_index=fold.fold_index,
                            evaluated=False,
                            unavailable_reason=(
                                fold.invalid_reason or "fold marked invalid"
                            ),
                        ),
                        predictions=(),
                        baselines=(),
                    )
                )
                continue
            outcomes.append(
                _evaluate_fold(
                    fold,
                    config=config,
                    frame=frame,
                    catalog=catalog,
                    model_spec=model_spec,
                    group_values=group_values,
                    labels=labels,
                    task_type=spec.task_type,
                    predictor_columns=predictor_columns,
                    personalized_columns=personalized_columns,
                    calibration_records=calibration_records,
                    warnings=warnings,
                )
            )

        evaluation = _build_evaluation(
            run_id=run_id,
            config=config,
            frame=frame,
            splits=splits,
            fingerprint=fingerprint,
            group_field=group_field,
            task_type=spec.task_type,
            outcomes=outcomes,
            predictor_columns=predictor_columns,
            personalized_columns=personalized_columns,
            disclaimers=disclaimers,
        )
        metrics = _metrics_document(
            run_id=run_id,
            config=config,
            frame=frame,
            splits=splits,
            group_field=group_field,
            task_type=spec.task_type,
            outcomes=outcomes,
            evaluation=evaluation,
            predictor_columns=predictor_columns,
            model_name=model_name,
            disclaimers=disclaimers,
        )
        baselines = PersonalBaselineDocument(
            run_id=run_id,
            evaluation_mode=config.evaluation_mode,
            target_name=config.target_name.value,
            method=settings.method,
            statistics=tuple(
                record for outcome in outcomes for record in outcome.baselines
            ),
            disclaimers=disclaimers,
        )
        personalization_manifest = PersonalizationExperimentManifest(
            run_id=run_id,
            evaluation_mode=config.evaluation_mode,
            scientific_evaluation_eligible=(
                config.evaluation_mode is EvaluationMode.SCIENTIFIC
            ),
            target_name=config.target_name.value,
            task_type=spec.task_type,
            dataset_path=str(config.dataset_path),
            dataset_fingerprint=frame.metadata.dataset_fingerprint,
            feature_catalog_version=frame.metadata.feature_catalog_version,
            split_manifest_fingerprint=fingerprint,
            split_strategy=splits.strategy.value,
            group_field=group_field.value,
            group_count=splits.group_count,
            fold_count=splits.n_splits,
            random_seed=config.random_seed,
            configuration=settings,
            calibration_method=config.calibration_method.value,
            calibration_group_fraction=config.calibration_group_fraction,
            predictor_columns=predictor_columns,
            personalized_columns=personalized_columns,
            disclaimers=disclaimers,
        )

        predictions = [
            prediction for outcome in outcomes for prediction in outcome.predictions
        ]
        true_by_window = {
            frame.window_ids[index]: frame.target_values[index]
            for index in range(frame.row_count)
        }

        run.write_json("dataset.json", frame.metadata.model_dump(mode="json"))
        run.write_json("feature_catalog.json", catalog.model_dump(mode="json"))
        run.write_json("splits.json", splits.model_dump(mode="json"))
        run.write_json(
            "personalization_config.json",
            personalization_manifest.model_dump(mode="json"),
        )
        run.write_json("personalization.json", evaluation.model_dump(mode="json"))
        run.write_json("personal_baselines.json", baselines.model_dump(mode="json"))
        run.write_json("metrics.json", metrics.model_dump(mode="json"))
        run.write_json(
            "calibration.json",
            {
                "run_id": run_id,
                "requested_method": config.calibration_method.value,
                "calibration_group_fraction": config.calibration_group_fraction,
                "design": (
                    "Probability calibration of the POPULATION model, fitted "
                    "on calibration groups drawn from the training groups and "
                    "disjoint from both the fit groups and the outer test "
                    "groups. It is a separate step from per-subject "
                    "personalization."
                ),
                "note": CALIBRATION_MEANING_NOTE,
                "folds": calibration_records,
                "disclaimers": list(disclaimers),
            },
        )
        run.write_table(
            "predictions.parquet",
            personalized_predictions_table(
                predictions,
                labels,
                spec.task_type,
                [true_by_window.get(p.window_id) for p in predictions],
            ),
        )
        run.write_model_warning()

        manifest = _manifest(
            run_id=run_id,
            config=config,
            frame=frame,
            splits=splits,
            fingerprint=fingerprint,
            predictor_columns=predictor_columns,
            model_spec=model_spec,
            model_name=model_name,
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
            fingerprint=fingerprint,
            predictor_columns=predictor_columns,
            model_spec=model_spec,
            model_name=model_name,
            started=started,
            finished=utc_now(),
            status=RunStatus.FAILED,
            failure_reason=f"{type(exc).__name__}: {exc}",
            disclaimers=disclaimers,
        )
        run.finalize(manifest)
        raise

    return PersonalizationRunResult(
        run_id=run_id,
        directory=run.directory,
        metrics=metrics,
        personalization=evaluation,
        baselines=baselines,
        splits=splits,
        manifest=manifest,
        warnings=tuple(warnings),
    )


def _evaluate_fold(
    fold: FoldAssignment,
    *,
    config: PersonalizationRunConfiguration,
    frame: ModellingFrame,
    catalog: FeatureCatalog,
    model_spec: Any,
    group_values: Sequence[str],
    labels: tuple[str, ...],
    task_type: TaskType,
    predictor_columns: tuple[str, ...],
    personalized_columns: tuple[str, ...],
    calibration_records: list[dict[str, Any]],
    warnings: list[str],
) -> _FoldOutcome:
    """Fit the population model and personalise every held-out subject."""
    settings = config.personalization
    classification = task_type is TaskType.CLASSIFICATION
    fit_groups = fold.fit_groups()
    calibration_groups = fold.calibration_groups
    test_groups = fold.test_groups

    fit_idx = _rows_for_groups(group_values, fit_groups)
    calibration_idx = _rows_for_groups(group_values, calibration_groups)
    if fit_idx.size == 0:
        return _FoldOutcome(
            result=PersonalizationFoldResult(
                fold_index=fold.fold_index,
                evaluated=False,
                unavailable_reason="the fit portion of this fold is empty",
            ),
            predictions=(),
            baselines=(),
        )

    # The population model never sees a held-out subject: the fit and
    # calibration rows are drawn from the training groups only, and
    # ``audit_split`` has already established those are disjoint from the
    # test groups.
    raw = frame.predictors.loc[:, list(predictor_columns)]
    y = frame.target_values

    # Cut every held-out subject into a calibration region and a strictly
    # later evaluation region before anything is fitted, so the two regions
    # are a property of the data and not of the model.
    splits_by_subject: dict[str, PersonalCalibrationSplit] = {}
    calibration_windows_by_subject: dict[str, tuple[SubjectWindow, ...]] = {}
    evaluation_windows_by_subject: dict[str, tuple[SubjectWindow, ...]] = {}
    for subject in sorted(set(test_groups)):
        rows = [index for index, group in enumerate(group_values) if group == subject]
        windows = subject_windows(
            row_indices=rows,
            window_ids=frame.window_ids,
            session_ids=frame.session_ids,
            window_start_utc=frame.window_start_utc,
            window_end_utc=frame.window_end_utc,
            window_indices=frame.window_indices,
        )
        split, calibration, evaluation = build_calibration_split(
            windows,
            subject_id=subject,
            fold_index=fold.fold_index,
            calibration_windows=settings.calibration_windows,
            minimum_evaluation_windows=settings.minimum_evaluation_windows,
            windows_overlap=frame.windows_overlap,
        )
        splits_by_subject[subject] = split
        calibration_windows_by_subject[subject] = calibration
        evaluation_windows_by_subject[subject] = evaluation

    baseline_records: list[PersonalBaselineStatistics] = []
    personal_matrix: pd.DataFrame | None = None
    if settings.method in BASELINE_METHODS:
        personal_matrix, baseline_records = _personalised_matrix(
            frame=frame,
            catalog=catalog,
            raw=raw,
            group_values=group_values,
            fit_idx=fit_idx,
            calibration_idx=calibration_idx,
            splits_by_subject=splits_by_subject,
            calibration_windows_by_subject=calibration_windows_by_subject,
            personalized_columns=personalized_columns,
            fold_index=fold.fold_index,
            settings=settings,
        )

    population_estimator, population_calibrated = _fit_population_model(
        model_spec,
        config=config,
        matrix=raw,
        y=y,
        fit_idx=fit_idx,
        calibration_idx=calibration_idx,
        fit_groups=fit_groups,
        calibration_groups=calibration_groups,
        test_groups=test_groups,
        classification=classification,
        label=POPULATION_MODEL_NAME,
        fold_index=fold.fold_index,
        calibration_records=calibration_records,
    )
    if personal_matrix is None:
        personal_estimator, personal_calibrated = (
            population_estimator,
            population_calibrated,
        )
    else:
        personal_estimator, personal_calibrated = _fit_population_model(
            model_spec,
            config=config,
            matrix=personal_matrix,
            y=y,
            fit_idx=fit_idx,
            calibration_idx=calibration_idx,
            fit_groups=fit_groups,
            calibration_groups=calibration_groups,
            test_groups=test_groups,
            classification=classification,
            label=f"{PERSONALIZED_MODEL_NAME}_baseline",
            fold_index=fold.fold_index,
            calibration_records=calibration_records,
        )

    predictions: list[PersonalizedPrediction] = []
    corrections: list[PersonalizationCorrection] = []
    row_of_window = {frame.window_ids[index]: index for index in range(frame.row_count)}
    cold_start_subjects = 0
    personalized_subjects = 0
    unavailable_subjects = 0
    calibration_window_total = 0
    evaluation_window_total = 0
    excluded_total = 0

    for subject in sorted(splits_by_subject):
        split = splits_by_subject[subject]
        excluded_total += len(split.excluded_overlap_window_ids)
        if not split.available:
            unavailable_subjects += 1
            warnings.append(
                f"fold {fold.fold_index}: subject {subject!r} was not "
                f"evaluated: {split.unavailable_reason}"
            )
            continue
        calibration_window_total += len(split.calibration_window_ids)
        evaluation_window_total += len(split.evaluation_window_ids)

        calibration_rows = [
            row_of_window[w.window_id] for w in calibration_windows_by_subject[subject]
        ]
        evaluation_rows = [
            row_of_window[w.window_id] for w in evaluation_windows_by_subject[subject]
        ]

        correction = _subject_correction(
            subject=subject,
            fold_index=fold.fold_index,
            settings=settings,
            task_type=task_type,
            labels=labels,
            estimator=population_estimator,
            calibrated=population_calibrated,
            matrix=raw,
            y=y,
            calibration_rows=calibration_rows,
            split=split,
        )
        if correction is not None:
            corrections.append(correction)

        subject_records, applied = _subject_predictions(
            subject=subject,
            fold_index=fold.fold_index,
            config=config,
            frame=frame,
            task_type=task_type,
            labels=labels,
            split=split,
            evaluation_rows=evaluation_rows,
            population_estimator=population_estimator,
            population_calibrated=population_calibrated,
            personal_estimator=personal_estimator,
            personal_calibrated=personal_calibrated,
            raw=raw,
            personal_matrix=personal_matrix,
            correction=correction,
            baseline_records=baseline_records,
        )
        predictions.extend(subject_records)
        if applied:
            personalized_subjects += 1
        else:
            cold_start_subjects += 1

    population_classification, personalized_classification = (None, None)
    population_regression, personalized_regression = (None, None)
    if predictions:
        truth = [frame.target_values[row_of_window[p.window_id]] for p in predictions]
        groups = [p.subject_id for p in predictions]
        if classification:
            population_classification = classification_metrics(
                y_true=[str(v) for v in truth],
                y_predicted=[str(p.population_predicted_class) for p in predictions],
                labels=labels,
                group_ids=groups,
            )
            personalized_classification = classification_metrics(
                y_true=[str(v) for v in truth],
                y_predicted=[str(p.personalized_predicted_class) for p in predictions],
                labels=labels,
                group_ids=groups,
            )
        else:
            population_regression = regression_metrics(
                y_true=[float(v) for v in truth],
                y_predicted=[
                    _required_regression_value(
                        p.population_predicted_value,
                        kind="population",
                        prediction=p,
                    )
                    for p in predictions
                ],
                group_ids=groups,
            )
            personalized_regression = regression_metrics(
                y_true=[float(v) for v in truth],
                y_predicted=[
                    _required_regression_value(
                        p.personalized_predicted_value,
                        kind="personalized",
                        prediction=p,
                    )
                    for p in predictions
                ],
                group_ids=groups,
            )

    result = PersonalizationFoldResult(
        fold_index=fold.fold_index,
        evaluated=bool(predictions),
        unavailable_reason=(
            None
            if predictions
            else (
                "no held-out subject in this fold could be cut into a "
                "calibration region and a strictly later evaluation region"
            )
        ),
        population_training_subject_count=len(set(fit_groups)),
        held_out_subject_count=len(set(test_groups)),
        evaluated_subject_count=personalized_subjects + cold_start_subjects,
        personalized_subject_count=personalized_subjects,
        cold_start_subject_count=cold_start_subjects,
        unavailable_subject_count=unavailable_subjects,
        calibration_window_count=calibration_window_total,
        evaluation_window_count=evaluation_window_total,
        excluded_overlap_window_count=excluded_total,
        population_classification_metrics=population_classification,
        personalized_classification_metrics=personalized_classification,
        population_regression_metrics=population_regression,
        personalized_regression_metrics=personalized_regression,
        splits=tuple(splits_by_subject[s] for s in sorted(splits_by_subject)),
        corrections=tuple(corrections),
    )
    return _FoldOutcome(
        result=result,
        predictions=tuple(predictions),
        baselines=tuple(baseline_records),
    )


def _personalised_matrix(
    *,
    frame: ModellingFrame,
    catalog: FeatureCatalog,
    raw: pd.DataFrame,
    group_values: Sequence[str],
    fit_idx: np.ndarray,
    calibration_idx: np.ndarray,
    splits_by_subject: dict[str, PersonalCalibrationSplit],
    calibration_windows_by_subject: dict[str, tuple[SubjectWindow, ...]],
    personalized_columns: tuple[str, ...],
    fold_index: int,
    settings: PersonalizationConfiguration,
) -> tuple[pd.DataFrame, list[PersonalBaselineStatistics]]:
    """Build the personally-normalised predictor matrix for one fold.

    A **held-out** subject's baseline uses their calibration windows only,
    and only those baselines are persisted, because only they have a
    boundary worth auditing.

    A **training** subject has no evaluation region to protect — every one
    of their windows is training data — but their baseline is nonetheless
    estimated from their own earliest ``calibration_windows`` windows, under
    exactly the same chronological rule.  Using all of their windows would
    be leakage-free and still wrong: it would estimate the training scale
    from more evidence than any held-out subject ever gets, and the
    estimator would then meet a differently-scaled matrix at evaluation
    time.  The normalisation a model is fitted through must be the
    normalisation it is deployed through.
    """
    matrix = raw.copy()
    persisted: list[PersonalBaselineStatistics] = []
    row_of_window = {frame.window_ids[index]: index for index in range(frame.row_count)}

    training_rows: dict[str, list[int]] = {}
    for index in list(fit_idx) + list(calibration_idx):
        training_rows.setdefault(str(group_values[int(index)]), []).append(int(index))

    for subject in sorted(training_rows):
        rows = training_rows[subject]
        windows = subject_windows(
            row_indices=rows,
            window_ids=frame.window_ids,
            session_ids=frame.session_ids,
            window_start_utc=frame.window_start_utc,
            window_end_utc=frame.window_end_utc,
            window_indices=frame.window_indices,
        )
        earliest = sorted(windows, key=SubjectWindow.sort_key)[
            : settings.calibration_windows
        ]
        baseline_rows = [row_of_window[w.window_id] for w in earliest]
        statistics = personal_baseline_statistics(
            raw.iloc[baseline_rows],
            subject_id=subject,
            fold_index=fold_index,
            columns=personalized_columns,
            catalog=catalog,
            source_window_ids=[w.window_id for w in earliest],
            minimum_samples=settings.minimum_baseline_samples,
            zero_variance_epsilon=settings.zero_variance_epsilon,
        )
        matrix.iloc[rows] = apply_personal_baseline(
            raw.iloc[rows], statistics
        ).to_numpy()

    for subject in sorted(splits_by_subject):
        split = splits_by_subject[subject]
        if not split.available:
            continue
        calibration_rows = [
            row_of_window[w.window_id] for w in calibration_windows_by_subject[subject]
        ]
        subject_rows = [
            row_of_window[window_id]
            for window_id in (
                *split.calibration_window_ids,
                *split.evaluation_window_ids,
                *split.excluded_overlap_window_ids,
            )
        ]
        statistics = personal_baseline_statistics(
            raw.iloc[calibration_rows],
            subject_id=subject,
            fold_index=fold_index,
            columns=personalized_columns,
            catalog=catalog,
            source_window_ids=list(split.calibration_window_ids),
            minimum_samples=settings.minimum_baseline_samples,
            zero_variance_epsilon=settings.zero_variance_epsilon,
        )
        persisted.extend(statistics)
        if subject_rows:
            matrix.iloc[subject_rows] = apply_personal_baseline(
                raw.iloc[subject_rows], statistics
            ).to_numpy()
    return matrix, persisted


def _fit_population_model(
    model_spec: Any,
    *,
    config: PersonalizationRunConfiguration,
    matrix: pd.DataFrame,
    y: np.ndarray,
    fit_idx: np.ndarray,
    calibration_idx: np.ndarray,
    fit_groups: Sequence[str],
    calibration_groups: Sequence[str],
    test_groups: Sequence[str],
    classification: bool,
    label: str,
    fold_index: int,
    calibration_records: list[dict[str, Any]],
) -> tuple[BaseEstimator, BaseEstimator | None]:
    """Fit one estimator, and its calibrator, on the training portion only."""
    pipeline = build_pipeline(
        model_spec,
        [str(c) for c in matrix.columns],
        random_seed=config.random_seed,
    )
    X_fit = matrix.iloc[fit_idx]
    X_calibration = matrix.iloc[calibration_idx]
    if not classification:
        base = clone(pipeline)
        base.fit(X_fit, y[fit_idx])
        return base, None

    base, outcome = calibrate_classifier(
        pipeline,
        X_fit=X_fit,
        y_fit=[str(v) for v in y[fit_idx]],
        X_calibration=X_calibration,
        y_calibration=[str(v) for v in y[calibration_idx]],
        method=config.calibration_method,
        fit_groups=fit_groups,
        calibration_groups=calibration_groups,
        test_groups=test_groups,
    )
    calibration_records.append(
        {
            "model_name": label,
            "fold_index": fold_index,
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
    calibrated = (
        outcome.calibrated_estimator
        if config.personalization.use_calibrated_population_model
        else None
    )
    return base, calibrated


def _probabilities(
    estimator: BaseEstimator,
    calibrated: BaseEstimator | None,
    X: pd.DataFrame,
    labels: Sequence[str],
) -> tuple[np.ndarray, bool]:
    """Probabilities from the calibrated estimator when one exists."""
    if calibrated is not None:
        aligned = aligned_probabilities(calibrated, X, labels)
        if aligned is not None:
            return aligned, True
    aligned = aligned_probabilities(estimator, X, labels)
    if aligned is None:  # pragma: no cover - every registered classifier has proba
        raise PersonalizationError(
            "the population classifier produces no class probabilities, so no "
            "per-subject probability correction can be applied to it"
        )
    return aligned, False


def _subject_correction(
    *,
    subject: str,
    fold_index: int,
    settings: PersonalizationConfiguration,
    task_type: TaskType,
    labels: tuple[str, ...],
    estimator: BaseEstimator,
    calibrated: BaseEstimator | None,
    matrix: pd.DataFrame,
    y: np.ndarray,
    calibration_rows: Sequence[int],
    split: PersonalCalibrationSplit,
) -> PersonalizationCorrection | None:
    """Fit the per-subject correction from the calibration region alone."""
    if settings.method not in CORRECTION_METHODS:
        return None
    if not calibration_rows:
        return PersonalizationCorrection(
            subject_id=subject,
            fold_index=fold_index,
            method=settings.method,
            task_type=task_type,
            available=False,
            unavailable_reason=(
                split.cold_start_reason
                or "no calibration window is available for this subject"
            ),
            calibration_sample_count=0,
        )
    X = matrix.iloc[list(calibration_rows)]
    if task_type is TaskType.CLASSIFICATION:
        probabilities, _ = _probabilities(estimator, calibrated, X, labels)
        return classification_correction(
            subject_id=subject,
            fold_index=fold_index,
            method=settings.method,
            calibration_window_ids=split.calibration_window_ids,
            calibration_labels=[str(y[row]) for row in calibration_rows],
            population_probabilities=probabilities,
            vocabulary=labels,
            smoothing=settings.classification_smoothing,
            shrinkage_constant=settings.classification_shrinkage_constant,
            minimum_windows=settings.minimum_calibration_windows,
            minimum_classes=settings.minimum_calibration_classes,
        )
    predictions = np.asarray(estimator.predict(X), dtype=float)
    return regression_correction(
        subject_id=subject,
        fold_index=fold_index,
        method=settings.method,
        calibration_window_ids=split.calibration_window_ids,
        calibration_targets=[float(y[row]) for row in calibration_rows],
        population_predictions=[float(v) for v in predictions],
        minimum_windows=settings.minimum_calibration_windows,
    )


def _subject_predictions(
    *,
    subject: str,
    fold_index: int,
    config: PersonalizationRunConfiguration,
    frame: ModellingFrame,
    task_type: TaskType,
    labels: tuple[str, ...],
    split: PersonalCalibrationSplit,
    evaluation_rows: Sequence[int],
    population_estimator: BaseEstimator,
    population_calibrated: BaseEstimator | None,
    personal_estimator: BaseEstimator,
    personal_calibrated: BaseEstimator | None,
    raw: pd.DataFrame,
    personal_matrix: pd.DataFrame | None,
    correction: PersonalizationCorrection | None,
    baseline_records: Sequence[PersonalBaselineStatistics],
) -> tuple[list[PersonalizedPrediction], bool]:
    """Score one subject's evaluation windows under both models."""
    settings = config.personalization
    rows = list(evaluation_rows)
    X_population = raw.iloc[rows]
    X_personal = (
        personal_matrix.iloc[rows] if personal_matrix is not None else X_population
    )

    normalized = tuple(
        record
        for record in baseline_records
        if record.subject_id == subject and record.normalized
    )
    baseline_applied = personal_matrix is not None and bool(normalized)
    correction_applied = correction is not None and correction.available

    reasons: list[str] = []
    if settings.method is PersonalizationMethod.POPULATION_ONLY:
        reasons.append(
            "the population-only method was requested; no personal evidence is used"
        )
    else:
        if settings.method in BASELINE_METHODS and not baseline_applied:
            reasons.append(
                "no feature could be normalised against a personal baseline "
                "from this subject's calibration windows"
            )
        if settings.method in CORRECTION_METHODS and not correction_applied:
            reasons.append(
                correction.unavailable_reason
                if correction is not None and correction.unavailable_reason
                else "no supervised correction could be fitted for this subject"
            )
    applied = not reasons
    cold_start = split.cold_start or not applied
    cold_start_reason = (
        split.cold_start_reason
        if split.cold_start and split.cold_start_reason
        else ("; ".join(reasons) if reasons else None)
    )
    unavailable_reason = "; ".join(reasons) if reasons else None

    records: list[PersonalizedPrediction] = []
    if task_type is TaskType.CLASSIFICATION:
        population_probabilities, calibrated_flag = _probabilities(
            population_estimator, population_calibrated, X_population, labels
        )
        if baseline_applied:
            personalized_probabilities, _ = _probabilities(
                personal_estimator, personal_calibrated, X_personal, labels
            )
        else:
            personalized_probabilities = population_probabilities
        if correction_applied and correction is not None:
            personalized_probabilities = apply_classification_correction(
                personalized_probabilities, correction.log_odds_shift, labels
            )
        if not applied:
            personalized_probabilities = population_probabilities
        for position, row in enumerate(rows):
            records.append(
                _classification_record(
                    row=row,
                    position=position,
                    subject=subject,
                    fold_index=fold_index,
                    config=config,
                    frame=frame,
                    labels=labels,
                    split=split,
                    population=population_probabilities,
                    personalized=personalized_probabilities,
                    calibrated_flag=calibrated_flag,
                    applied=applied,
                    unavailable_reason=unavailable_reason,
                    cold_start=cold_start,
                    cold_start_reason=cold_start_reason,
                    baseline_applied=baseline_applied,
                    correction_applied=correction_applied,
                    normalized_count=len(normalized),
                    correction=correction,
                )
            )
        return records, applied

    population_values = np.asarray(
        population_estimator.predict(X_population), dtype=float
    )
    personalized_values = (
        np.asarray(personal_estimator.predict(X_personal), dtype=float)
        if baseline_applied
        else population_values.copy()
    )
    if correction_applied and correction is not None and correction.bias is not None:
        personalized_values = np.asarray(
            [
                apply_regression_correction(v, correction.bias)
                for v in personalized_values
            ],
            dtype=float,
        )
    if not applied:
        personalized_values = population_values.copy()
    if (
        not np.isfinite(population_values).all()
        or not np.isfinite(personalized_values).all()
    ):
        raise PersonalizationError(
            f"subject {subject!r} produced a non-finite regression prediction; "
            "a model that cannot produce a finite value must fail rather than "
            "emit one"
        )
    for position, row in enumerate(rows):
        records.append(
            PersonalizedPrediction(
                window_id=frame.window_ids[row],
                subject_id=subject,
                session_id=frame.session_ids[row],
                target_name=config.target_name.value,
                task_type=task_type,
                fold_index=fold_index,
                method=(
                    settings.method if applied else PersonalizationMethod.COLD_START
                ),
                population_predicted_value=float(population_values[position]),
                personalized_predicted_value=float(personalized_values[position]),
                personalization_applied=applied,
                unavailable_reason=unavailable_reason,
                cold_start=cold_start,
                cold_start_reason=cold_start_reason,
                baseline_normalized=baseline_applied,
                supervised_correction_applied=correction_applied,
                normalized_feature_count=len(normalized),
                calibration_window_ids=split.calibration_window_ids,
                calibration_sample_count=(
                    correction.calibration_sample_count
                    if correction is not None
                    else len(split.calibration_window_ids)
                ),
                data_source=frame.data_sources[row],
                is_synthetic=frame.data_sources[row] == "synthetic",
                scientific_evaluation_eligible=(
                    config.evaluation_mode is EvaluationMode.SCIENTIFIC
                ),
            )
        )
    return records, applied


def _classification_record(
    *,
    row: int,
    position: int,
    subject: str,
    fold_index: int,
    config: PersonalizationRunConfiguration,
    frame: ModellingFrame,
    labels: tuple[str, ...],
    split: PersonalCalibrationSplit,
    population: np.ndarray,
    personalized: np.ndarray,
    calibrated_flag: bool,
    applied: bool,
    unavailable_reason: str | None,
    cold_start: bool,
    cold_start_reason: str | None,
    baseline_applied: bool,
    correction_applied: bool,
    normalized_count: int,
    correction: PersonalizationCorrection | None,
) -> PersonalizedPrediction:
    population_row = _normalised_row(population[position])
    personalized_row = _normalised_row(personalized[position])
    return PersonalizedPrediction(
        window_id=frame.window_ids[row],
        subject_id=subject,
        session_id=frame.session_ids[row],
        target_name=config.target_name.value,
        task_type=TaskType.CLASSIFICATION,
        fold_index=fold_index,
        method=(
            config.personalization.method
            if applied
            else PersonalizationMethod.COLD_START
        ),
        population_predicted_class=labels[int(np.argmax(population_row))],
        population_probabilities=tuple(float(v) for v in population_row),
        personalized_predicted_class=labels[int(np.argmax(personalized_row))],
        personalized_probabilities=tuple(float(v) for v in personalized_row),
        class_vocabulary=labels,
        probabilities_are_calibrated=calibrated_flag,
        personalization_applied=applied,
        unavailable_reason=unavailable_reason,
        cold_start=cold_start,
        cold_start_reason=cold_start_reason,
        baseline_normalized=baseline_applied,
        supervised_correction_applied=correction_applied,
        normalized_feature_count=normalized_count,
        calibration_window_ids=split.calibration_window_ids,
        calibration_sample_count=(
            correction.calibration_sample_count
            if correction is not None
            else len(split.calibration_window_ids)
        ),
        data_source=frame.data_sources[row],
        is_synthetic=frame.data_sources[row] == "synthetic",
        scientific_evaluation_eligible=(
            config.evaluation_mode is EvaluationMode.SCIENTIFIC
        ),
    )


def _normalised_row(values: np.ndarray) -> np.ndarray:
    """Re-normalise one probability row so it sums to exactly one.

    ``aligned_probabilities`` already renormalises, but a subsequent
    multiplication and division in the correction leaves an error of order
    1e-16.  Repairing it here keeps the schema's distribution check about
    real defects rather than about floating-point residue.
    """
    row = np.asarray(values, dtype=float)
    total = float(row.sum())
    if not np.isfinite(total) or total <= 0.0:  # pragma: no cover - guarded upstream
        raise PersonalizationError(
            "a probability row could not be renormalised; a vector that does "
            "not sum to one is not a distribution and is never emitted"
        )
    return row / total


def _build_evaluation(
    *,
    run_id: str,
    config: PersonalizationRunConfiguration,
    frame: ModellingFrame,
    splits: SplitManifest,
    fingerprint: str,
    group_field: GroupField,
    task_type: TaskType,
    outcomes: Sequence[_FoldOutcome],
    predictor_columns: tuple[str, ...],
    personalized_columns: tuple[str, ...],
    disclaimers: tuple[str, ...],
) -> PersonalizationEvaluation:
    """Assemble the personalization document from the per-fold outcomes."""
    results = [outcome.result for outcome in outcomes]
    classification = task_type is TaskType.CLASSIFICATION
    fields = (
        CLASSIFICATION_AGGREGATE_FIELDS
        if classification
        else REGRESSION_AGGREGATE_FIELDS
    )
    population_metrics = _fold_metrics(results, classification, personalized=False)
    personalized_metrics = _fold_metrics(results, classification, personalized=True)

    personalized_subjects = sum(r.personalized_subject_count for r in results)
    cold_start_subjects = sum(r.cold_start_subject_count for r in results)
    denominator = personalized_subjects + cold_start_subjects
    return PersonalizationEvaluation(
        run_id=run_id,
        evaluation_mode=config.evaluation_mode,
        scientific_evaluation_eligible=(
            config.evaluation_mode is EvaluationMode.SCIENTIFIC
        ),
        target_name=config.target_name.value,
        task_type=task_type,
        dataset_fingerprint=frame.metadata.dataset_fingerprint,
        split_manifest_fingerprint=fingerprint,
        group_field=group_field.value,
        group_count=splits.group_count,
        fold_count=splits.n_splits,
        random_seed=config.random_seed,
        configuration=config.personalization,
        predictor_columns=predictor_columns,
        personalized_columns=personalized_columns,
        folds=tuple(results),
        population_aggregate=aggregate_fold_metrics(
            population_metrics, fields, total_fold_count=splits.n_splits
        ),
        personalized_aggregate=aggregate_fold_metrics(
            personalized_metrics, fields, total_fold_count=splits.n_splits
        ),
        total_calibration_window_count=sum(r.calibration_window_count for r in results),
        total_evaluation_window_count=sum(r.evaluation_window_count for r in results),
        total_excluded_overlap_window_count=sum(
            r.excluded_overlap_window_count for r in results
        ),
        cold_start_subject_count=cold_start_subjects,
        personalized_subject_count=personalized_subjects,
        unavailable_personalization_count=sum(
            r.unavailable_subject_count for r in results
        ),
        personalization_coverage=(
            personalized_subjects / denominator if denominator else None
        ),
        disclaimers=disclaimers,
    )


def _fold_metrics(
    results: Sequence[PersonalizationFoldResult],
    classification: bool,
    *,
    personalized: bool,
) -> list[Any]:
    metrics: list[Any] = []
    for result in results:
        if classification:
            entry: Any = (
                result.personalized_classification_metrics
                if personalized
                else result.population_classification_metrics
            )
        else:
            entry = (
                result.personalized_regression_metrics
                if personalized
                else result.population_regression_metrics
            )
        if entry is not None:
            metrics.append(entry)
    return metrics


def _metrics_document(
    *,
    run_id: str,
    config: PersonalizationRunConfiguration,
    frame: ModellingFrame,
    splits: SplitManifest,
    group_field: GroupField,
    task_type: TaskType,
    outcomes: Sequence[_FoldOutcome],
    evaluation: PersonalizationEvaluation,
    predictor_columns: tuple[str, ...],
    model_name: str,
    disclaimers: tuple[str, ...],
) -> MetricsDocument:
    """Two `ModelResult` entries: population and personalized, same rows."""
    results = [outcome.result for outcome in outcomes]
    classification = task_type is TaskType.CLASSIFICATION
    failed = {
        r.fold_index: r.unavailable_reason or "fold not evaluated"
        for r in results
        if not r.evaluated
    }
    entries: list[ModelResult] = []
    for name, kind, aggregate, personalized in (
        (
            POPULATION_MODEL_NAME,
            "population",
            evaluation.population_aggregate,
            False,
        ),
        (
            PERSONALIZED_MODEL_NAME,
            "personalized",
            evaluation.personalized_aggregate,
            True,
        ),
    ):
        fold_metrics = _fold_metrics(results, classification, personalized=personalized)
        entries.append(
            ModelResult(
                model_name=name,
                model_kind=kind,
                parameters={
                    "description": (
                        "The population reference model: early fusion over the "
                        "configured modality groups, fitted on other subjects "
                        "only."
                        if not personalized
                        else (
                            "The population model adapted to each held-out "
                            "subject from that subject's own earlier windows."
                        )
                    ),
                    "estimator": model_name,
                    "method": (
                        PersonalizationMethod.POPULATION_ONLY.value
                        if not personalized
                        else config.personalization.method.value
                    ),
                    "random_seed": config.random_seed,
                },
                predictor_columns=predictor_columns,
                fold_classification_metrics=(
                    tuple(fold_metrics) if classification else ()
                ),
                fold_regression_metrics=(() if classification else tuple(fold_metrics)),
                aggregate=aggregate,
                failed_folds=failed,
                notes=(SEPARATE_REPORTING_NOTE, CALIBRATION_MEANING_NOTE),
            )
        )

    return MetricsDocument(
        run_id=run_id,
        evaluation_mode=config.evaluation_mode,
        scientific_evaluation_eligible=(
            config.evaluation_mode is EvaluationMode.SCIENTIFIC
        ),
        target_name=config.target_name.value,
        task_type=task_type.value,
        dataset_fingerprint=frame.metadata.dataset_fingerprint,
        group_field=group_field.value,
        group_count=splits.group_count,
        fold_count=splits.n_splits,
        random_seed=config.random_seed,
        results=tuple(entries),
        disclaimers=disclaimers,
    )


def personalized_predictions_table(
    predictions: Sequence[PersonalizedPrediction],
    labels: Sequence[str],
    task_type: TaskType,
    true_values: Sequence[Any],
) -> pa.Table:
    """Per-window population and personalized outputs, side by side.

    Both are stored; neither overwrites the other, so a reader can compute
    the difference without re-running anything.
    """
    columns: dict[str, list[Any]] = {
        "window_id": [p.window_id for p in predictions],
        "subject_id": [p.subject_id for p in predictions],
        "session_id": [p.session_id for p in predictions],
        "fold_index": [int(p.fold_index) for p in predictions],
        "method": [p.method.value for p in predictions],
        "personalization_applied": [p.personalization_applied for p in predictions],
        "cold_start": [p.cold_start for p in predictions],
        "cold_start_reason": [p.cold_start_reason for p in predictions],
        "unavailable_reason": [p.unavailable_reason for p in predictions],
        "baseline_normalized": [p.baseline_normalized for p in predictions],
        "supervised_correction_applied": [
            p.supervised_correction_applied for p in predictions
        ],
        "normalized_feature_count": [
            int(p.normalized_feature_count) for p in predictions
        ],
        "calibration_sample_count": [
            int(p.calibration_sample_count) for p in predictions
        ],
        "calibration_window_ids": [list(p.calibration_window_ids) for p in predictions],
        "is_synthetic": [p.is_synthetic for p in predictions],
        "scientific_evaluation_eligible": [
            p.scientific_evaluation_eligible for p in predictions
        ],
    }
    if task_type is TaskType.CLASSIFICATION:
        columns["true_value"] = [
            None if value is None else str(value) for value in true_values
        ]
        columns["population_predicted_class"] = [
            p.population_predicted_class for p in predictions
        ]
        columns["personalized_predicted_class"] = [
            p.personalized_predicted_class for p in predictions
        ]
        for index, label in enumerate(labels):
            columns[f"population_probability__{label}"] = [
                float(p.population_probabilities[index]) for p in predictions
            ]
            columns[f"personalized_probability__{label}"] = [
                float(p.personalized_probabilities[index]) for p in predictions
            ]
    else:
        columns["true_value"] = [
            None if value is None else float(value) for value in true_values
        ]
        columns["population_predicted_value"] = [
            p.population_predicted_value for p in predictions
        ]
        columns["personalized_predicted_value"] = [
            p.personalized_predicted_value for p in predictions
        ]
    if not predictions:
        schema = pa.schema(
            [
                (name, pa.list_(pa.string()) if name.endswith("_ids") else pa.string())
                for name in columns
            ]
        )
        return pa.table({name: [] for name in columns}, schema=schema)
    return pa.table(columns)


def _manifest(
    *,
    run_id: str,
    config: PersonalizationRunConfiguration,
    frame: ModellingFrame,
    splits: SplitManifest,
    fingerprint: str,
    predictor_columns: tuple[str, ...],
    model_spec: Any,
    model_name: str,
    started: datetime,
    finished: datetime,
    status: RunStatus,
    failure_reason: str | None,
    disclaimers: tuple[str, ...],
) -> RunManifest:
    environment = runtime_environment()
    described = describe_parameters(
        build_pipeline(model_spec, predictor_columns, random_seed=config.random_seed)
    )
    described["imputation"] = model_spec.imputation.value
    described["standardised"] = model_spec.scale
    settings = config.personalization
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
        feature_set=predictor_columns,
        model_names=(POPULATION_MODEL_NAME, PERSONALIZED_MODEL_NAME),
        model_parameters={model_name: described},
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
            "kind": "personalization",
            "n_splits": config.n_splits,
            "random_seed": config.random_seed,
            "calibration_group_fraction": config.calibration_group_fraction,
            "calibration_bins": config.calibration_bins,
            "catalog_version": config.catalog_version,
            "split_manifest_fingerprint": fingerprint,
            "personalization": settings.model_dump(mode="json"),
        },
        random_seed=config.random_seed,
        started_at_utc=started,
        finished_at_utc=finished,
        status=status,
        failure_reason=failure_reason,
        disclaimers=disclaimers,
    )


__all__ = [
    "PERSONALIZATION_REQUIRED_ARTIFACTS",
    "PERSONALIZED_MODEL_NAME",
    "POPULATION_MODEL_NAME",
    "SEPARATE_REPORTING_NOTE",
    "PersonalizationConfigurationError",
    "PersonalizationRunConfiguration",
    "PersonalizationRunResult",
    "ScientificModeError",
    "personalized_predictions_table",
    "run_personalization",
]
