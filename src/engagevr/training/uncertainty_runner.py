"""Orchestration of an uncertainty-aware, selective-prediction evaluation.

The ordering constraints are the substance of this milestone, so the
control flow is explicit:

1. Load and audit the dataset; refuse it in scientific mode if any
   predictor or target is synthetic, or if target provenance is missing.
2. Build the outer folds **once**, with the same ``build_splits`` call the
   Milestone 5 baseline runner, the Milestone 6 fusion runner, and the
   Milestone 6 personalization runner make, and fingerprint the manifest so
   fold reuse is checkable.
3. Inside each fold, fit the base estimator on the **fit groups**, fit the
   probability calibrator (classification) or the conformal residual
   distribution (regression) on the **calibration groups**, and — when
   threshold estimation is enabled — choose the threshold on those same
   calibration groups.  Four group sets are recorded per fold: fit,
   probability calibration, threshold selection, conformal calibration.
   The outer-test groups appear in none of them.
4. Score the outer-test rows, derive confidence or an interval, take an
   abstention decision at the applied threshold, and evaluate the
   adaptation gate on the decision that was actually taken.
5. Sweep the configured threshold grid over the *same* outer-test
   predictions to produce the coverage curve, then write every artifact and
   the manifest, atomically and last.

What the outer-test fold never does
-----------------------------------
It never fits the model, never fits a probability calibrator, never fits a
conformal residual distribution, never chooses a confidence threshold,
never chooses an interval-width threshold, never tunes an abstention rule,
and never tunes a personalized threshold.  ``UncertaintyFoldResult``
re-checks the group sets and refuses to validate if any of that were
violated, so the claim is enforced by the persisted document rather than
only by this comment.

Two things this runner deliberately does not do
-----------------------------------------------
It does not calibrate a *fused* probability vector.  Milestone 6 calibrates
each modality expert and the early-fusion estimator but never the fused
output, so a late-fusion maximum is not calibrated confidence and is not
offered as a Milestone 7 confidence source.  And it does not choose an
adaptation: the gate here reports whether an already-chosen action may be
acted upon, and nothing more.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
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
from engagevr.schemas.targets import (
    TARGET_DISCLAIMER,
    TargetName,
    TaskType,
    get_target_spec,
)
from engagevr.schemas.uncertainty import (
    ABSTENTION_MEANING_NOTE,
    COVERAGE_AXIS_UNITS,
    SELECTIVE_PREDICTION_NOTE,
    UNCERTAINTY_NOTE,
    AbstentionDecision,
    AbstentionReason,
    AdaptationGateDecision,
    AdaptationGateRecord,
    ClassificationConfidence,
    CoverageAxis,
    CoverageCurve,
    EnsembleDisagreementReference,
    PersonalThresholdRecord,
    PredictionSource,
    ProbabilityCalibrationStatus,
    RegressionPredictionInterval,
    SelectiveMetrics,
    SelectivePredictionConfiguration,
    SignalQualitySummary,
    ThresholdSource,
    UncertaintyEvaluation,
    UncertaintyExperimentManifest,
    UncertaintyFoldResult,
)
from engagevr.training.adaptation_gate import evaluate_adaptation_gate
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
from engagevr.training.experts import modality_availability, modality_quality
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
    build_calibration_split,
    subject_windows,
)
from engagevr.training.preprocessing import ModellingFrame, load_modelling_frame
from engagevr.training.runner import (
    EvaluationError,
    ScientificModeError,
    assert_scientific_eligibility,
)
from engagevr.training.splits import build_splits, choose_group_field
from engagevr.training.uncertainty import (
    UncertaintyError,
    absolute_residuals,
    acceptance_rule_for,
    accepts_at_threshold,
    accepts_interval_width,
    area_under_risk_coverage,
    build_uncertainty_run_id,
    confidence_components,
    confidence_method,
    conformal_interval,
    coverage_axis_for,
    coverage_is_monotonic,
    evaluate_evidence_gate,
    expected_monotonic_direction,
    fit_conformal_quantile,
    interval_contains,
    personal_confidence_threshold,
    project_interval_to_range,
    reason_counts,
    risk_coverage_points,
    select_population_threshold,
    selective_classification_metrics,
    selective_regression_metrics,
)

#: Artifacts a completed uncertainty run must contain.
UNCERTAINTY_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "dataset.json",
    "feature_catalog.json",
    "splits.json",
    "uncertainty_config.json",
    "uncertainty.json",
    "thresholds.json",
    "selective_metrics.json",
    "coverage_curve.json",
    "metrics.json",
)

#: ``model_name`` of the two separately reported metric results.
ALL_WINDOWS_MODEL_NAME = "all_windows"
ACCEPTED_MODEL_NAME = "accepted_at_applied_threshold"


class UncertaintyConfigurationError(EvaluationError):
    """An uncertainty run cannot proceed as configured."""


@dataclass(frozen=True, slots=True)
class UncertaintyRunConfiguration:
    """Everything that defines one uncertainty-aware evaluation run."""

    dataset_path: Path
    target_name: TargetName
    output_directory: Path
    selective: SelectivePredictionConfiguration
    evaluation_mode: EvaluationMode = EvaluationMode.SOFTWARE_SELF_CHECK
    n_splits: int = 5
    random_seed: int = 42
    calibration_method: CalibrationMethod = CalibrationMethod.SIGMOID
    calibration_group_fraction: float = 0.25
    calibration_bins: int = DEFAULT_CALIBRATION_BINS
    catalog_version: str = FEATURE_CATALOG_VERSION


@dataclass(slots=True)
class UncertaintyRunResult:
    """What a completed uncertainty run produced."""

    run_id: str
    directory: Path
    metrics: MetricsDocument
    uncertainty: UncertaintyEvaluation
    coverage_curve: CoverageCurve
    splits: SplitManifest
    manifest: RunManifest
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class _FoldOutcome:
    """One fold's records, decisions, and gate outcomes."""

    result: UncertaintyFoldResult
    confidences: tuple[ClassificationConfidence, ...] = ()
    intervals: tuple[RegressionPredictionInterval, ...] = ()
    decisions: tuple[AbstentionDecision, ...] = ()
    gates: tuple[AdaptationGateRecord, ...] = ()
    selective: SelectiveMetrics | None = None
    curve_points: tuple[SelectiveMetrics, ...] = ()
    all_window_classification: Any = None
    all_window_regression: Any = None


def _disclaimers(mode: EvaluationMode) -> tuple[str, ...]:
    if mode is EvaluationMode.SOFTWARE_SELF_CHECK:
        return (
            SELF_CHECK_DISCLAIMER,
            TARGET_DISCLAIMER,
            UNCERTAINTY_NOTE,
            SELECTIVE_PREDICTION_NOTE,
            ABSTENTION_MEANING_NOTE,
        )
    return (
        TARGET_DISCLAIMER,
        UNCERTAINTY_NOTE,
        SELECTIVE_PREDICTION_NOTE,
        ABSTENTION_MEANING_NOTE,
        "This uncertainty run was executed in scientific mode. Eligibility "
        "was checked against data source and target provenance; eligibility "
        "is not validity. No claim of calibration quality, selective-"
        "prediction reliability, or safety follows from a run completing.",
    )


def _rows_for_groups(group_values: Sequence[str], wanted: Sequence[str]) -> np.ndarray:
    members = set(wanted)
    return np.asarray(
        [index for index, group in enumerate(group_values) if group in members],
        dtype=int,
    )


def run_uncertainty(config: UncertaintyRunConfiguration) -> UncertaintyRunResult:
    """Execute a complete uncertainty-aware selective-prediction evaluation."""
    started = utc_now()
    catalog = get_catalog(config.catalog_version)
    spec = get_target_spec(config.target_name)
    settings = config.selective

    frame = load_modelling_frame(
        config.dataset_path, target_name=config.target_name, catalog=catalog
    )
    if config.evaluation_mode is EvaluationMode.SCIENTIFIC:
        assert_scientific_eligibility(frame)

    predictor_columns = _predictor_columns(settings, frame, catalog)
    if not predictor_columns:
        raise UncertaintyConfigurationError(
            f"prediction source {settings.prediction_source.value!r} resolves to "
            f"no permitted predictor column in {config.dataset_path}. The "
            "absence is reported rather than silently reduced to whatever "
            "happened to survive."
        )

    group_field, group_reason = choose_group_field(frame.subject_ids, frame.session_ids)
    group_values = (
        frame.subject_ids if group_field is GroupField.SUBJECT_ID else frame.session_ids
    )
    if settings.personalized_thresholds_enabled and (
        group_field is not GroupField.SUBJECT_ID
    ):
        raise UncertaintyConfigurationError(
            "personalized confidence thresholds require subject grouping, but "
            f"this dataset is grouped by {group_field.value!r}: {group_reason}. "
            "A personal threshold is defined per person; without a subject "
            "identifier there is no person to personalise to, and the run is "
            "refused rather than personalising to a session."
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

    labels: tuple[str, ...] = ()
    if spec.task_type is TaskType.CLASSIFICATION:
        observed = set(class_labels or [])
        labels = tuple(
            label for label in (spec.class_vocabulary or ()) if label in observed
        )
        if len(labels) < 2:
            raise UncertaintyConfigurationError(
                f"target {config.target_name.value!r} has {len(labels)} class(es) "
                "present in the dataset; a top-two margin and a selective "
                "classification rule both require at least two"
            )

    fingerprint = split_manifest_fingerprint(splits)
    run_id = build_uncertainty_run_id(
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
        required_artifacts=UNCERTAINTY_REQUIRED_ARTIFACTS,
    )
    disclaimers = _disclaimers(config.evaluation_mode)

    model_name = (
        settings.model_classification
        if spec.task_type is TaskType.CLASSIFICATION
        else settings.model_regression
    )
    model_spec = get_model_spec(model_name, spec.task_type)

    availability = {
        modality: modality_availability(frame.predictors, modality, catalog)
        for modality in settings.modalities
    }
    quality = {
        modality: modality_quality(frame.predictors, modality)
        for modality in settings.modalities
    }

    outcomes: list[_FoldOutcome] = []
    warnings: list[str] = []
    calibration_records: list[dict[str, Any]] = []

    try:
        for fold in splits.folds:
            if not fold.valid:
                outcomes.append(
                    _FoldOutcome(
                        result=UncertaintyFoldResult(
                            fold_index=fold.fold_index,
                            evaluated=False,
                            unavailable_reason=(
                                fold.invalid_reason or "fold marked invalid"
                            ),
                        )
                    )
                )
                continue
            outcomes.append(
                _evaluate_fold(
                    fold,
                    config=config,
                    run_id=run_id,
                    frame=frame,
                    model_spec=model_spec,
                    group_values=group_values,
                    group_field=group_field,
                    labels=labels,
                    task_type=spec.task_type,
                    predictor_columns=predictor_columns,
                    availability=availability,
                    quality=quality,
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
            labels=labels,
            outcomes=outcomes,
            predictor_columns=predictor_columns,
            disclaimers=disclaimers,
        )
        curve = _pooled_curve(
            config=config, task_type=spec.task_type, outcomes=outcomes
        )
        metrics = _metrics_document(
            run_id=run_id,
            config=config,
            frame=frame,
            splits=splits,
            group_field=group_field,
            task_type=spec.task_type,
            outcomes=outcomes,
            predictor_columns=predictor_columns,
            model_name=model_name,
            disclaimers=disclaimers,
        )
        experiment_manifest = UncertaintyExperimentManifest(
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
            probability_calibration_method=config.calibration_method.value,
            calibration_group_fraction=config.calibration_group_fraction,
            predictor_columns=predictor_columns,
            disclaimers=disclaimers,
        )

        run.write_json("dataset.json", frame.metadata.model_dump(mode="json"))
        run.write_json("feature_catalog.json", catalog.model_dump(mode="json"))
        run.write_json("splits.json", splits.model_dump(mode="json"))
        run.write_json(
            "uncertainty_config.json", experiment_manifest.model_dump(mode="json")
        )
        run.write_json("uncertainty.json", evaluation.model_dump(mode="json"))
        run.write_json(
            "thresholds.json", _thresholds_document(run_id, evaluation, disclaimers)
        )
        run.write_json(
            "selective_metrics.json",
            _selective_document(run_id, evaluation, outcomes, disclaimers),
        )
        run.write_json(
            "coverage_curve.json",
            {
                "run_id": run_id,
                "evaluation_mode": config.evaluation_mode.value,
                "scientific_evaluation_eligible": False
                if config.evaluation_mode is EvaluationMode.SOFTWARE_SELF_CHECK
                else True,
                "x_axis": curve.axis.value,
                "x_axis_units": curve.axis_units,
                "monotonicity_rule": curve.monotonicity_rule,
                "curve": curve.model_dump(mode="json"),
                "disclaimers": list(disclaimers),
            },
        )
        run.write_json("metrics.json", metrics.model_dump(mode="json"))
        run.write_json(
            "calibration.json",
            _calibration_document(
                run_id, config, calibration_records, outcomes, disclaimers
            ),
        )
        run.write_table(
            "predictions.parquet",
            _predictions_table(outcomes, labels, spec.task_type, frame),
        )
        run.write_table(
            "selective_predictions.parquet",
            _selective_predictions_table(outcomes, labels, spec.task_type, frame),
        )
        run.write_table("adaptation_gate.parquet", _gate_table(outcomes))
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

    return UncertaintyRunResult(
        run_id=run_id,
        directory=run.directory,
        metrics=metrics,
        uncertainty=evaluation,
        coverage_curve=curve,
        splits=splits,
        manifest=manifest,
        warnings=tuple(warnings),
    )


def _predictor_columns(
    settings: SelectivePredictionConfiguration,
    frame: ModellingFrame,
    catalog: FeatureCatalog,
) -> tuple[str, ...]:
    """Columns the configured prediction source is allowed to see."""
    if settings.prediction_source is PredictionSource.EARLY_FUSION:
        return early_fusion_columns(
            settings.modalities,
            frame.predictor_columns,
            catalog,
            include_modality_quality=False,
        )
    return frame.predictor_columns


def _evaluate_fold(
    fold: FoldAssignment,
    *,
    config: UncertaintyRunConfiguration,
    run_id: str,
    frame: ModellingFrame,
    model_spec: Any,
    group_values: Sequence[str],
    group_field: GroupField,
    labels: tuple[str, ...],
    task_type: TaskType,
    predictor_columns: tuple[str, ...],
    availability: dict[Any, np.ndarray],
    quality: dict[Any, np.ndarray],
    calibration_records: list[dict[str, Any]],
    warnings: list[str],
) -> _FoldOutcome:
    """Fit, calibrate, threshold, decide, and gate one outer fold."""
    classification = task_type is TaskType.CLASSIFICATION
    fit_groups = fold.fit_groups()
    calibration_groups = fold.calibration_groups
    test_groups = fold.test_groups

    fit_idx = _rows_for_groups(group_values, fit_groups)
    calibration_idx = _rows_for_groups(group_values, calibration_groups)
    test_idx = _rows_for_groups(group_values, test_groups)
    if fit_idx.size == 0 or test_idx.size == 0:
        return _FoldOutcome(
            result=UncertaintyFoldResult(
                fold_index=fold.fold_index,
                evaluated=False,
                unavailable_reason=(
                    "the fit portion of this fold is empty"
                    if fit_idx.size == 0
                    else "the test portion of this fold is empty"
                ),
            )
        )

    matrix = frame.predictors.loc[:, list(predictor_columns)]
    y = frame.target_values
    pipeline = build_pipeline(
        model_spec, [str(c) for c in matrix.columns], random_seed=config.random_seed
    )

    if classification:
        return _evaluate_classification_fold(
            fold,
            config=config,
            run_id=run_id,
            frame=frame,
            pipeline=pipeline,
            matrix=matrix,
            y=y,
            fit_idx=fit_idx,
            calibration_idx=calibration_idx,
            test_idx=test_idx,
            fit_groups=fit_groups,
            calibration_groups=calibration_groups,
            test_groups=test_groups,
            group_values=group_values,
            group_field=group_field,
            labels=labels,
            availability=availability,
            quality=quality,
            calibration_records=calibration_records,
            warnings=warnings,
        )
    return _evaluate_regression_fold(
        fold,
        config=config,
        run_id=run_id,
        frame=frame,
        pipeline=pipeline,
        matrix=matrix,
        y=y,
        fit_idx=fit_idx,
        calibration_idx=calibration_idx,
        test_idx=test_idx,
        fit_groups=fit_groups,
        calibration_groups=calibration_groups,
        test_groups=test_groups,
        group_values=group_values,
        availability=availability,
        quality=quality,
        warnings=warnings,
    )


def _signal_quality(
    *,
    row: int,
    settings: SelectivePredictionConfiguration,
    availability: dict[Any, np.ndarray],
    quality: dict[Any, np.ndarray],
) -> SignalQualitySummary:
    """Per-window measurement quality, assembled beside the prediction."""
    available = tuple(m for m in settings.modalities if bool(availability[m][row]))
    unavailable = tuple(
        m for m in settings.modalities if not bool(availability[m][row])
    )
    recorded: dict[str, float | None] = {}
    for modality in settings.modalities:
        value = float(quality[modality][row])
        recorded[modality.value] = value if np.isfinite(value) else None
    present = [v for v in recorded.values() if v is not None]
    return SignalQualitySummary(
        available_modalities=available,
        unavailable_modalities=unavailable,
        modality_quality=recorded,
        minimum_recorded_quality=(min(present) if present else None),
    )


def _prediction_id(run_id: str, fold_index: int, window_id: str, source: str) -> str:
    return f"{run_id}|{fold_index}|{window_id}|{source}"


def _evaluate_classification_fold(
    fold: FoldAssignment,
    *,
    config: UncertaintyRunConfiguration,
    run_id: str,
    frame: ModellingFrame,
    pipeline: BaseEstimator,
    matrix: pd.DataFrame,
    y: np.ndarray,
    fit_idx: np.ndarray,
    calibration_idx: np.ndarray,
    test_idx: np.ndarray,
    fit_groups: Sequence[str],
    calibration_groups: Sequence[str],
    test_groups: Sequence[str],
    group_values: Sequence[str],
    group_field: GroupField,
    labels: tuple[str, ...],
    availability: dict[Any, np.ndarray],
    quality: dict[Any, np.ndarray],
    calibration_records: list[dict[str, Any]],
    warnings: list[str],
) -> _FoldOutcome:
    settings = config.selective
    base, outcome = calibrate_classifier(
        pipeline,
        X_fit=matrix.iloc[fit_idx],
        y_fit=[str(v) for v in y[fit_idx]],
        X_calibration=matrix.iloc[calibration_idx],
        y_calibration=[str(v) for v in y[calibration_idx]],
        method=config.calibration_method,
        fit_groups=fit_groups,
        calibration_groups=calibration_groups,
        test_groups=test_groups,
    )
    calibration_records.append(
        {
            "model_name": settings.model_classification,
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

    estimator = outcome.calibrated_estimator or base
    status = (
        ProbabilityCalibrationStatus.CALIBRATED
        if outcome.available
        else ProbabilityCalibrationStatus.UNCALIBRATED
    )
    calibration_reason = (
        None
        if outcome.available
        else (
            outcome.unavailable_reason
            or "no probability calibrator was fitted for this fold"
        )
    )
    if not outcome.available:
        warnings.append(
            f"fold {fold.fold_index}: probabilities are UNCALIBRATED — "
            f"{calibration_reason}. Their maximum is recorded as a selection "
            "score, not as calibrated confidence."
        )

    test_probabilities = aligned_probabilities(estimator, matrix.iloc[test_idx], labels)
    if (
        test_probabilities is None
    ):  # pragma: no cover - registered models all support it
        raise UncertaintyError(
            f"fold {fold.fold_index}: the estimator produces no class "
            "probabilities, so no confidence representation is defined for it"
        )

    # --- population threshold: configured, or estimated on calibration rows
    estimated = None
    applied_population = settings.population_confidence_threshold
    threshold_source = ThresholdSource.CONFIGURED_POPULATION
    if settings.estimate_population_threshold and calibration_idx.size:
        calibration_probabilities = aligned_probabilities(
            estimator, matrix.iloc[calibration_idx], labels
        )
        if calibration_probabilities is not None:
            scores = calibration_probabilities.max(axis=1)
            predicted = [
                labels[int(index)] for index in calibration_probabilities.argmax(axis=1)
            ]
            correct = [
                str(y[int(row)]) == label
                for row, label in zip(calibration_idx, predicted, strict=True)
            ]
            estimated = select_population_threshold(
                scores=[float(v) for v in scores],
                correct=correct,
                group_ids=[str(group_values[int(r)]) for r in calibration_idx],
                grid=settings.threshold_grid,
                objective=settings.threshold_objective,
                target=settings.threshold_objective_target,
                minimum_samples=settings.minimum_threshold_selection_samples,
                minimum_groups=settings.minimum_threshold_selection_groups,
                fold_index=fold.fold_index,
                calibration_group_ids=calibration_groups,
                outer_test_group_ids=test_groups,
            )
            if estimated.available and estimated.selected_threshold is not None:
                applied_population = estimated.selected_threshold
                threshold_source = ThresholdSource.ESTIMATED_POPULATION
            else:
                warnings.append(
                    f"fold {fold.fold_index}: threshold estimation was "
                    f"unavailable — {estimated.unavailable_reason} "
                )

    row_position = {int(row): position for position, row in enumerate(test_idx)}

    # --- personal thresholds, from each subject's EARLIER windows only
    personal: dict[str, PersonalThresholdRecord] = {}
    threshold_by_row: dict[int, tuple[float, ThresholdSource]] = {
        int(row): (applied_population, threshold_source) for row in test_idx
    }
    if settings.personalized_thresholds_enabled:
        personal = _personal_thresholds(
            fold=fold,
            frame=frame,
            settings=settings,
            group_values=group_values,
            group_field=group_field,
            test_groups=test_groups,
            test_idx=test_idx,
            probabilities=test_probabilities,
            row_position=row_position,
            population_threshold=applied_population,
        )
        for row in test_idx:
            record = personal.get(str(group_values[int(row)]))
            if record is None:
                continue
            threshold_by_row[int(row)] = (
                record.applied_threshold,
                record.threshold_source,
            )
        # A window used to derive its subject's own threshold is never also
        # scored under it: the calibration windows are excluded from the
        # evaluated set, exactly as Milestone 6 excludes them from the
        # personalized report.
        calibration_windows = {
            window
            for record in personal.values()
            for window in record.calibration_window_ids
        }
        if calibration_windows:
            keep = [
                row
                for row in test_idx
                if frame.window_ids[int(row)] not in calibration_windows
            ]
            test_idx = np.asarray(keep, dtype=int)
            if test_idx.size == 0:
                return _FoldOutcome(
                    result=UncertaintyFoldResult(
                        fold_index=fold.fold_index,
                        evaluated=False,
                        unavailable_reason=(
                            "every outer-test window of this fold was consumed as "
                            "personal threshold calibration evidence, leaving "
                            "nothing to evaluate"
                        ),
                        fit_group_ids=tuple(sorted(set(fit_groups))),
                        probability_calibration_group_ids=tuple(
                            sorted(set(calibration_groups))
                        ),
                        outer_test_group_ids=tuple(sorted(set(test_groups))),
                    )
                )

    # --- per-window records, decisions, and gate outcomes
    confidences: list[ClassificationConfidence] = []
    decisions: list[AbstentionDecision] = []
    gates: list[AdaptationGateRecord] = []
    for row in test_idx:
        index = int(row)
        position = row_position[index]
        window_id = frame.window_ids[index]
        vector = tuple(float(v) for v in test_probabilities[position])
        parts = confidence_components(vector, labels, context=f"window {window_id!r}")
        signal = _signal_quality(
            row=index, settings=settings, availability=availability, quality=quality
        )
        prediction_id = _prediction_id(
            run_id, fold.fold_index, window_id, settings.prediction_source.value
        )
        calibrated = status is ProbabilityCalibrationStatus.CALIBRATED
        confidence = ClassificationConfidence(
            window_id=window_id,
            subject_id=frame.subject_ids[index],
            session_id=frame.session_ids[index],
            target_name=config.target_name.value,
            fold_index=fold.fold_index,
            source_model=settings.prediction_source,
            source_model_name=settings.model_classification,
            source_prediction_id=prediction_id,
            class_vocabulary=labels,
            probabilities=vector,
            probability_calibration_status=status,
            probability_calibration_method=config.calibration_method.value,
            probability_calibration_group_count=len(set(calibration_groups)),
            probability_calibration_unavailable_reason=calibration_reason,
            predicted_class=parts.predicted_class,
            maximum_probability=parts.maximum_probability,
            maximum_probability_class=parts.maximum_probability_class,
            method=confidence_method(status),
            confidence_score=parts.maximum_probability if calibrated else None,
            selection_score=None if calibrated else parts.maximum_probability,
            entropy=parts.entropy,
            normalized_entropy=parts.normalized_entropy,
            margin=parts.margin,
            signal_quality=signal,
            disagreement=EnsembleDisagreementReference(
                ensemble_disagreement=None,
                expert_count=0,
                source_strategy=None,
            ),
            data_source=frame.data_sources[index],
            is_synthetic=frame.data_sources[index] == "synthetic",
            scientific_evaluation_eligible=(
                config.evaluation_mode is EvaluationMode.SCIENTIFIC
            ),
        )
        confidences.append(confidence)

        threshold, source = threshold_by_row[index]
        decision = _classification_decision(
            record=confidence,
            threshold=threshold,
            threshold_source=source,
            settings=settings,
        )
        decisions.append(decision)
        gates.append(
            evaluate_adaptation_gate(
                decision,
                applied_confidence_threshold=threshold,
                maximum_interval_width=None,
                enabled=settings.adaptation_gate_enabled,
            )
        )

    truth = [str(y[int(row)]) for row in test_idx]
    groups = [str(group_values[int(row)]) for row in test_idx]
    predicted = [d.predicted_class or "" for d in decisions]
    probability_rows = [list(c.probabilities) for c in confidences]
    unavailable_flags = [not d.prediction_available for d in decisions]

    # The label on the applied-metrics record is the POPULATION threshold.
    # When personalization is on, different rows cleared different
    # thresholds; averaging them would produce a number no window was
    # actually judged against. The per-row values live on the decision
    # records and in ``selective_predictions.parquet``.
    applied_metrics = selective_classification_metrics(
        threshold=applied_population,
        y_true=truth,
        y_predicted=predicted,
        probabilities=probability_rows,
        labels=labels,
        group_ids=groups,
        accepted=[d.accepted for d in decisions],
        unavailable=unavailable_flags,
        calibration_bins=config.calibration_bins,
    )
    all_windows = classification_metrics(
        y_true=truth,
        y_predicted=predicted,
        labels=list(labels),
        group_ids=groups,
    )

    curve_points = _classification_curve(
        config=config,
        confidences=confidences,
        decisions=decisions,
        truth=truth,
        groups=groups,
        labels=labels,
        settings=settings,
    )
    curve = _fold_curve(TaskType.CLASSIFICATION, settings, curve_points)

    result = UncertaintyFoldResult(
        fold_index=fold.fold_index,
        evaluated=True,
        fit_group_ids=tuple(sorted(set(fit_groups))),
        probability_calibration_group_ids=tuple(sorted(set(calibration_groups))),
        threshold_selection_group_ids=(
            tuple(sorted(set(calibration_groups))) if estimated is not None else ()
        ),
        outer_test_group_ids=tuple(sorted(set(test_groups))),
        probability_calibration_status=status,
        probability_calibration_method=config.calibration_method.value,
        probability_calibration_unavailable_reason=calibration_reason,
        estimated_threshold=estimated,
        applied_population_threshold=applied_population,
        applied_population_threshold_source=threshold_source,
        personal_thresholds=tuple(personal[k] for k in sorted(personal)),
        total_window_count=len(decisions),
        accepted_count=sum(1 for d in decisions if d.accepted),
        abstained_count=sum(1 for d in decisions if d.abstained),
        unavailable_count=sum(1 for flag in unavailable_flags if flag),
        abstention_reason_counts=reason_counts([d.reasons for d in decisions]),
        applied_selective_metrics=applied_metrics,
        coverage_curve=curve,
        adaptation_gate_eligible_count=sum(
            1 for g in gates if g.decision is AdaptationGateDecision.ELIGIBLE
        ),
        adaptation_gate_blocked_count=sum(
            1 for g in gates if g.decision is AdaptationGateDecision.BLOCKED
        ),
    )
    return _FoldOutcome(
        result=result,
        confidences=tuple(confidences),
        decisions=tuple(decisions),
        gates=tuple(gates),
        selective=applied_metrics,
        curve_points=curve_points,
        all_window_classification=all_windows,
    )


def _classification_decision(
    *,
    record: ClassificationConfidence,
    threshold: float,
    threshold_source: ThresholdSource,
    settings: SelectivePredictionConfiguration,
) -> AbstentionDecision:
    """Take one selective-classification decision, preserving the prediction."""
    calibrated = (
        record.probability_calibration_status is ProbabilityCalibrationStatus.CALIBRATED
    )
    signal = record.signal_quality
    passed, evidence_reasons = evaluate_evidence_gate(
        configuration=settings.evidence_gate,
        prediction_available=True,
        available_modalities=[
            m.value for m in (signal.available_modalities if signal else ())
        ],
        modality_quality=(signal.modality_quality if signal else {}),
        probability_calibrated=calibrated,
    )
    reasons = set(evidence_reasons)
    if not accepts_at_threshold(record.score(), threshold):
        reasons.add(AbstentionReason.BELOW_CONFIDENCE_THRESHOLD)

    ordered = tuple(r for r in AbstentionReason if r in reasons)
    accepted = not ordered
    return AbstentionDecision(
        window_id=record.window_id,
        subject_id=record.subject_id,
        session_id=record.session_id,
        target_name=record.target_name,
        task_type=TaskType.CLASSIFICATION,
        fold_index=record.fold_index,
        source_prediction_id=record.source_prediction_id,
        prediction_available=True,
        accepted=accepted,
        abstained=not accepted,
        reasons=ordered,
        predicted_class=record.predicted_class,
        class_vocabulary=record.class_vocabulary,
        probabilities=record.probabilities,
        probability_calibration_status=record.probability_calibration_status,
        confidence_score=record.confidence_score,
        selection_score=record.selection_score,
        entropy=record.entropy,
        margin=record.margin,
        applied_threshold=threshold,
        threshold_source=threshold_source,
        signal_quality=signal,
        disagreement=record.disagreement,
        evidence_gate_passed=passed,
        data_source=record.data_source,
        is_synthetic=record.is_synthetic,
        scientific_evaluation_eligible=record.scientific_evaluation_eligible,
        acceptance_rule=acceptance_rule_for(TaskType.CLASSIFICATION),
    )


def _personal_thresholds(
    *,
    fold: FoldAssignment,
    frame: ModellingFrame,
    settings: SelectivePredictionConfiguration,
    group_values: Sequence[str],
    group_field: GroupField,
    test_groups: Sequence[str],
    test_idx: np.ndarray,
    probabilities: np.ndarray,
    row_position: dict[int, int],
    population_threshold: float,
) -> dict[str, PersonalThresholdRecord]:
    """One threshold per held-out subject, from their earliest windows.

    The temporal split is Milestone 6's, unchanged: the subject's earliest
    ``personal_calibration_windows`` windows form the calibration region and
    everything starting at or after that region's end forms the evaluation
    region.  Only the *confidence scores* of the calibration windows are
    read — no label of any kind participates.
    """
    records: dict[str, PersonalThresholdRecord] = {}
    for subject in sorted(set(test_groups)):
        rows = [int(r) for r in test_idx if str(group_values[int(r)]) == subject]
        if not rows:
            continue
        windows = subject_windows(
            row_indices=rows,
            window_ids=frame.window_ids,
            session_ids=frame.session_ids,
            window_start_utc=frame.window_start_utc,
            window_end_utc=frame.window_end_utc,
            window_indices=frame.window_indices,
        )
        try:
            split, calibration, _evaluation = build_calibration_split(
                windows,
                subject_id=subject,
                fold_index=fold.fold_index,
                calibration_windows=settings.personal_calibration_windows,
                minimum_evaluation_windows=settings.minimum_evaluation_windows,
                windows_overlap=frame.windows_overlap,
            )
        except PersonalizationError as exc:
            records[subject] = personal_confidence_threshold(
                subject_id=subject,
                fold_index=fold.fold_index,
                calibration_scores=(),
                calibration_window_ids=(),
                evaluation_window_ids=(),
                population_threshold=population_threshold,
                configuration=settings,
                unavailable_reason=str(exc),
            )
            continue

        scores = [
            float(probabilities[row_position[window.row_index]].max())
            for window in calibration
        ]
        records[subject] = personal_confidence_threshold(
            subject_id=subject,
            fold_index=fold.fold_index,
            calibration_scores=scores,
            calibration_window_ids=split.calibration_window_ids,
            evaluation_window_ids=split.evaluation_window_ids,
            population_threshold=population_threshold,
            configuration=settings,
            calibration_start_utc=(
                split.calibration_start_utc.isoformat()
                if split.calibration_start_utc
                else None
            ),
            calibration_end_utc=(
                split.calibration_end_utc.isoformat()
                if split.calibration_end_utc
                else None
            ),
            evaluation_start_utc=(
                split.evaluation_start_utc.isoformat()
                if split.evaluation_start_utc
                else None
            ),
            temporal_order_verified=split.temporal_order_verified,
            unavailable_reason=(
                None if split.available else (split.unavailable_reason or "")
            ),
        )
    return records


def _classification_curve(
    *,
    config: UncertaintyRunConfiguration,
    confidences: Sequence[ClassificationConfidence],
    decisions: Sequence[AbstentionDecision],
    truth: Sequence[str],
    groups: Sequence[str],
    labels: tuple[str, ...],
    settings: SelectivePredictionConfiguration,
) -> tuple[SelectiveMetrics, ...]:
    """Sweep the configured grid over the SAME outer-test predictions.

    The evidence gate is applied at every grid point, so a window blocked
    for missing evidence is blocked at every threshold and coverage remains
    monotonically non-increasing in the threshold.
    """
    predicted = [d.predicted_class or "" for d in decisions]
    probability_rows = [list(c.probabilities) for c in confidences]
    gate_blocked = [
        any(reason in set(d.reasons) for reason in _EVIDENCE_AND_CALIBRATION)
        for d in decisions
    ]
    scores = [c.score() for c in confidences]
    unavailable = [not d.prediction_available for d in decisions]

    points: list[SelectiveMetrics] = []
    for threshold in settings.threshold_grid:
        accepted = [
            (not blocked)
            and available
            and accepts_at_threshold(score, float(threshold))
            for score, blocked, available in zip(
                scores, gate_blocked, [not u for u in unavailable], strict=True
            )
        ]
        points.append(
            selective_classification_metrics(
                threshold=float(threshold),
                y_true=truth,
                y_predicted=predicted,
                probabilities=probability_rows,
                labels=labels,
                group_ids=groups,
                accepted=accepted,
                unavailable=unavailable,
                calibration_bins=config.calibration_bins,
            )
        )
    return tuple(points)


#: Reasons that block a window regardless of which threshold is applied.
_EVIDENCE_AND_CALIBRATION: tuple[AbstentionReason, ...] = (
    AbstentionReason.MODEL_PREDICTION_UNAVAILABLE,
    AbstentionReason.INSUFFICIENT_MEASUREMENT_EVIDENCE,
    AbstentionReason.REQUIRED_MODALITY_UNAVAILABLE,
    AbstentionReason.SIGNAL_QUALITY_BELOW_GATE,
    AbstentionReason.PROBABILITY_CALIBRATION_UNAVAILABLE,
    AbstentionReason.PREDICTION_INTERVAL_UNAVAILABLE,
)


def _evaluate_regression_fold(
    fold: FoldAssignment,
    *,
    config: UncertaintyRunConfiguration,
    run_id: str,
    frame: ModellingFrame,
    pipeline: BaseEstimator,
    matrix: pd.DataFrame,
    y: np.ndarray,
    fit_idx: np.ndarray,
    calibration_idx: np.ndarray,
    test_idx: np.ndarray,
    fit_groups: Sequence[str],
    calibration_groups: Sequence[str],
    test_groups: Sequence[str],
    group_values: Sequence[str],
    availability: dict[Any, np.ndarray],
    quality: dict[Any, np.ndarray],
    warnings: list[str],
) -> _FoldOutcome:
    settings = config.selective
    spec = get_target_spec(config.target_name)
    if settings.personalized_thresholds_enabled and fold.fold_index == 0:
        warnings.append(
            "personalized confidence thresholds are enabled but this is a "
            "regression target, and a confidence threshold has no meaning for "
            "a point prediction. Every window is judged by the interval-width "
            "policy instead. Subject-conditional conformal intervals are NOT "
            "implemented: estimating a per-subject residual distribution from "
            "a handful of calibration windows would overfit, and doing it with "
            "labels would put the subject's own outcomes into their interval."
        )
    estimator = clone(pipeline)
    estimator.fit(matrix.iloc[fit_idx], y[fit_idx].astype(float))

    # Conformal calibration uses the fold's calibration groups, which are
    # disjoint from the rows that fitted the estimator and from the
    # outer-test rows. Residuals from rows the model memorised would
    # understate the interval.
    calibration_predictions = (
        np.asarray(estimator.predict(matrix.iloc[calibration_idx]), dtype=float)
        if calibration_idx.size
        else np.asarray([], dtype=float)
    )
    residuals = absolute_residuals(
        [float(v) for v in y[calibration_idx].astype(float)],
        [float(v) for v in calibration_predictions],
    )
    fit_result = fit_conformal_quantile(residuals, alpha=settings.alpha)
    if not fit_result.available:
        warnings.append(
            f"fold {fold.fold_index}: no conformal interval — "
            f"{fit_result.unavailable_reason}"
        )

    predictions = np.asarray(estimator.predict(matrix.iloc[test_idx]), dtype=float)
    if predictions.size and not np.isfinite(predictions).all():
        raise UncertaintyError(
            f"fold {fold.fold_index}: the regression estimator produced a "
            "non-finite prediction; a model that cannot produce a finite "
            "prediction must fail rather than emit one"
        )

    intervals: list[RegressionPredictionInterval] = []
    decisions: list[AbstentionDecision] = []
    gates: list[AdaptationGateRecord] = []
    for position, row in enumerate(test_idx):
        index = int(row)
        window_id = frame.window_ids[index]
        value = float(predictions[position])
        signal = _signal_quality(
            row=index, settings=settings, availability=availability, quality=quality
        )
        prediction_id = _prediction_id(
            run_id, fold.fold_index, window_id, settings.prediction_source.value
        )

        bounds: tuple[float, float, float] | None = None
        clipped: tuple[float, float] | None = None
        if fit_result.available and fit_result.quantile is not None:
            bounds = conformal_interval(value, fit_result.quantile)
            if settings.clip_interval_to_target_range:
                clipped = project_interval_to_range(
                    bounds[0],
                    bounds[1],
                    minimum=float(spec.value_minimum or 0.0),
                    maximum=float(spec.value_maximum or 1.0),
                )

        record = RegressionPredictionInterval(
            window_id=window_id,
            subject_id=frame.subject_ids[index],
            session_id=frame.session_ids[index],
            target_name=config.target_name.value,
            fold_index=fold.fold_index,
            source_model=settings.prediction_source,
            source_model_name=settings.model_regression,
            source_prediction_id=prediction_id,
            predicted_value=value,
            interval_method=settings.interval_method,
            calibration_succeeded=fit_result.available,
            unavailable_reason=fit_result.unavailable_reason,
            lower_bound=bounds[0] if bounds else None,
            upper_bound=bounds[1] if bounds else None,
            interval_width=bounds[2] if bounds else None,
            conformal_quantile=fit_result.quantile,
            conformal_order_statistic=fit_result.order_statistic,
            clipped_lower_bound=clipped[0] if clipped else None,
            clipped_upper_bound=clipped[1] if clipped else None,
            clipping_note=(
                (
                    "PRESENTATION PROJECTION ONLY. The raw conformal bounds "
                    "remain the interval of record and empirical interval "
                    "coverage is computed on them; clipping narrows an interval "
                    "with no statistical justification for doing so."
                )
                if clipped
                else None
            ),
            alpha=settings.alpha,
            nominal_coverage=1.0 - settings.alpha,
            calibration_sample_count=fit_result.sample_count,
            calibration_group_count=len(set(calibration_groups)),
            calibration_group_ids=tuple(sorted(set(calibration_groups))),
            signal_quality=signal,
            disagreement=EnsembleDisagreementReference(
                ensemble_disagreement=None, expert_count=0, source_strategy=None
            ),
            data_source=frame.data_sources[index],
            is_synthetic=frame.data_sources[index] == "synthetic",
            scientific_evaluation_eligible=(
                config.evaluation_mode is EvaluationMode.SCIENTIFIC
            ),
        )
        intervals.append(record)

        decision = _regression_decision(record=record, settings=settings)
        decisions.append(decision)
        gates.append(
            evaluate_adaptation_gate(
                decision,
                applied_confidence_threshold=None,
                maximum_interval_width=settings.maximum_interval_width,
                enabled=settings.adaptation_gate_enabled,
            )
        )

    truth = [float(v) for v in y[test_idx].astype(float)]
    groups = [str(group_values[int(row)]) for row in test_idx]
    lower = [r.lower_bound for r in intervals]
    upper = [r.upper_bound for r in intervals]
    widths = [r.interval_width for r in intervals]
    unavailable = [not d.prediction_available for d in decisions]

    applied_metrics = selective_regression_metrics(
        maximum_interval_width=settings.maximum_interval_width,
        no_maximum_reason=(
            None
            if settings.maximum_interval_width is not None
            else "no maximum_interval_width is configured, so no width policy "
            "was applied at this operating point. That is not a maximum width "
            "of zero, which would abstain on every window; every window whose "
            "evidence gate passed and whose interval exists was accepted."
        ),
        y_true=truth,
        y_predicted=[r.predicted_value for r in intervals],
        group_ids=groups,
        accepted=[d.accepted for d in decisions],
        unavailable=unavailable,
        interval_lower=lower,
        interval_upper=upper,
        interval_width=widths,
    )
    all_windows = regression_metrics(
        y_true=truth,
        y_predicted=[r.predicted_value for r in intervals],
        group_ids=groups,
    )
    curve_points = _regression_curve(
        settings=settings,
        intervals=intervals,
        decisions=decisions,
        truth=truth,
        groups=groups,
    )
    curve = _fold_curve(TaskType.REGRESSION, settings, curve_points)

    result = UncertaintyFoldResult(
        fold_index=fold.fold_index,
        evaluated=True,
        fit_group_ids=tuple(sorted(set(fit_groups))),
        conformal_calibration_group_ids=tuple(sorted(set(calibration_groups))),
        outer_test_group_ids=tuple(sorted(set(test_groups))),
        probability_calibration_status=ProbabilityCalibrationStatus.UNAVAILABLE,
        probability_calibration_unavailable_reason=(
            "a regression target has no class probabilities to calibrate; its "
            "uncertainty representation is a conformal prediction interval"
        ),
        conformal_quantile=fit_result.quantile,
        conformal_order_statistic=fit_result.order_statistic,
        conformal_calibration_sample_count=fit_result.sample_count,
        conformal_available=fit_result.available,
        conformal_unavailable_reason=fit_result.unavailable_reason,
        total_window_count=len(decisions),
        accepted_count=sum(1 for d in decisions if d.accepted),
        abstained_count=sum(1 for d in decisions if d.abstained),
        unavailable_count=sum(1 for flag in unavailable if flag),
        abstention_reason_counts=reason_counts([d.reasons for d in decisions]),
        applied_selective_metrics=applied_metrics,
        coverage_curve=curve,
        adaptation_gate_eligible_count=sum(
            1 for g in gates if g.decision is AdaptationGateDecision.ELIGIBLE
        ),
        adaptation_gate_blocked_count=sum(
            1 for g in gates if g.decision is AdaptationGateDecision.BLOCKED
        ),
    )
    return _FoldOutcome(
        result=result,
        intervals=tuple(intervals),
        decisions=tuple(decisions),
        gates=tuple(gates),
        selective=applied_metrics,
        curve_points=curve_points,
        all_window_regression=all_windows,
    )


def _regression_decision(
    *,
    record: RegressionPredictionInterval,
    settings: SelectivePredictionConfiguration,
) -> AbstentionDecision:
    """Take one selective-regression decision, preserving the prediction."""
    signal = record.signal_quality
    passed, evidence_reasons = evaluate_evidence_gate(
        configuration=settings.evidence_gate,
        prediction_available=True,
        available_modalities=[
            m.value for m in (signal.available_modalities if signal else ())
        ],
        modality_quality=(signal.modality_quality if signal else {}),
        probability_calibrated=None,
    )
    reasons = set(evidence_reasons)
    if record.interval_width is None:
        reasons.add(AbstentionReason.PREDICTION_INTERVAL_UNAVAILABLE)
    elif not accepts_interval_width(
        record.interval_width, settings.maximum_interval_width
    ):
        reasons.add(AbstentionReason.INTERVAL_TOO_WIDE)

    ordered = tuple(r for r in AbstentionReason if r in reasons)
    accepted = not ordered
    return AbstentionDecision(
        window_id=record.window_id,
        subject_id=record.subject_id,
        session_id=record.session_id,
        target_name=record.target_name,
        task_type=TaskType.REGRESSION,
        fold_index=record.fold_index,
        source_prediction_id=record.source_prediction_id,
        prediction_available=True,
        accepted=accepted,
        abstained=not accepted,
        reasons=ordered,
        predicted_value=record.predicted_value,
        interval_lower_bound=record.lower_bound,
        interval_upper_bound=record.upper_bound,
        interval_width=record.interval_width,
        maximum_interval_width=settings.maximum_interval_width,
        signal_quality=signal,
        disagreement=record.disagreement,
        evidence_gate_passed=passed,
        data_source=record.data_source,
        is_synthetic=record.is_synthetic,
        scientific_evaluation_eligible=record.scientific_evaluation_eligible,
        acceptance_rule=acceptance_rule_for(TaskType.REGRESSION),
    )


def _regression_curve(
    *,
    settings: SelectivePredictionConfiguration,
    intervals: Sequence[RegressionPredictionInterval],
    decisions: Sequence[AbstentionDecision],
    truth: Sequence[float],
    groups: Sequence[str],
) -> tuple[SelectiveMetrics, ...]:
    """Sweep the configured interval-width grid, in the target's own units.

    The rule is the same one the run applies:
    ``accept if interval_width <= W_max``.  Each grid value **is** a
    ``W_max``; it is compared against the original width and is neither
    normalised into ``[0, 1]`` nor inverted into a confidence score.
    Raising ``W_max`` is therefore *more permissive*, and coverage is
    monotonically **non-decreasing** along this axis — the opposite
    direction from the classification confidence axis.

    When no width grid is configured this returns no points at all rather
    than manufacturing a curve out of the classification grid, whose values
    are probabilities and carry no width units.

    Split conformal produces one quantile per fold, so every window in a
    fold shares one width and this sweep is a step: coverage moves from 0
    to 1 at the grid value that first reaches that shared width, rather
    than tracing a gradual curve.  That is a property of the method and is
    stated in the documents rather than smoothed over by manufacturing
    per-window variation it does not produce.
    """
    grid = settings.interval_width_grid
    if not grid:
        return ()

    gate_blocked = [
        any(reason in set(d.reasons) for reason in _EVIDENCE_AND_CALIBRATION)
        for d in decisions
    ]
    widths = [r.interval_width for r in intervals]
    unavailable = [not d.prediction_available for d in decisions]

    points: list[SelectiveMetrics] = []
    for maximum in grid:
        accepted = [
            (not blocked)
            and available
            and accepts_interval_width(width, float(maximum))
            for width, blocked, available in zip(
                widths, gate_blocked, [not u for u in unavailable], strict=True
            )
        ]
        points.append(
            selective_regression_metrics(
                maximum_interval_width=float(maximum),
                y_true=truth,
                y_predicted=[r.predicted_value for r in intervals],
                group_ids=groups,
                accepted=accepted,
                unavailable=unavailable,
                interval_lower=[r.lower_bound for r in intervals],
                interval_upper=[r.upper_bound for r in intervals],
                interval_width=widths,
            )
        )
    return tuple(points)


def _axis_values(
    task_type: TaskType, settings: SelectivePredictionConfiguration
) -> tuple[float, ...]:
    """The grid swept for a task type, in that axis's own units."""
    if task_type is TaskType.CLASSIFICATION:
        return settings.threshold_grid
    return settings.interval_width_grid or ()


def _no_width_grid_reason(settings: SelectivePredictionConfiguration) -> str:
    """Why a regression run reports no width coverage curve."""
    applied = settings.maximum_interval_width
    operating = (
        f"maximum_interval_width = {applied}"
        if applied is not None
        else "no width policy at all, so every otherwise-available prediction "
        "is accepted"
    )
    return (
        "no regression interval-width grid is configured "
        "(uncertainty.regression.interval_width_grid is null), so no width "
        "sweep was evaluated and no coverage curve is reported. A width "
        "threshold is measured in the target's own units, so the "
        "classification confidence grid cannot stand in for one: its values "
        "are probabilities, a general regression target need not live in "
        f"[0, 1], and 1 - width is not a confidence. The run reports its "
        f"operating point only, at {operating}."
    )


def _fold_curve(
    task_type: TaskType,
    settings: SelectivePredictionConfiguration,
    points: Sequence[SelectiveMetrics],
) -> CoverageCurve:
    axis = coverage_axis_for(task_type)
    direction = expected_monotonic_direction(axis)
    if not points:
        return CoverageCurve(
            task_type=task_type,
            axis=axis,
            axis_values=(),
            expected_monotonic_direction=direction,
            points_unavailable_reason=_no_width_grid_reason(settings),
            area_under_risk_coverage=None,
            area_under_risk_coverage_unavailable_reason=(
                "no coverage curve was evaluated, so there is no area under one"
            ),
        )
    risk = risk_coverage_points(points)
    area, reason = area_under_risk_coverage(risk)
    return CoverageCurve(
        task_type=task_type,
        axis=axis,
        axis_values=_axis_values(task_type, settings),
        points=tuple(points),
        risk_coverage=risk,
        area_under_risk_coverage=area,
        area_under_risk_coverage_unavailable_reason=reason,
        expected_monotonic_direction=direction,
        coverage_is_monotonic=coverage_is_monotonic(
            [p.coverage_point for p in points], direction=direction
        ),
    )


def _pooled_curve(
    *,
    config: UncertaintyRunConfiguration,
    task_type: TaskType,
    outcomes: Sequence[_FoldOutcome],
) -> CoverageCurve:
    """The run-level curve: fold curves summed point-by-point.

    Counts add across folds, and accepted-set metrics are re-derived from
    the summed counts rather than averaged, so every point of the run curve
    describes exactly the same evaluated windows.
    """
    settings = config.selective
    axis = coverage_axis_for(task_type)
    direction = expected_monotonic_direction(axis)
    values = _axis_values(task_type, settings)
    evaluated = [o for o in outcomes if o.curve_points]
    if not evaluated:
        unswept = (
            _no_width_grid_reason(settings)
            if task_type is TaskType.REGRESSION and not values
            else "no fold produced a coverage curve, so there is nothing to pool"
        )
        return CoverageCurve(
            task_type=task_type,
            axis=axis,
            axis_values=(),
            expected_monotonic_direction=direction,
            points_unavailable_reason=unswept,
            area_under_risk_coverage=None,
            area_under_risk_coverage_unavailable_reason=(
                "no coverage curve was pooled, so there is no area under one"
            ),
        )

    from engagevr.training.uncertainty import coverage_point as build_point

    points: list[SelectiveMetrics] = []
    for position, threshold in enumerate(values):
        accepted = sum(
            o.curve_points[position].coverage_point.accepted_count for o in evaluated
        )
        abstained = sum(
            o.curve_points[position].coverage_point.abstained_count for o in evaluated
        )
        missing = sum(
            o.curve_points[position].coverage_point.unavailable_count for o in evaluated
        )
        point = build_point(
            threshold=float(threshold),
            accepted_count=accepted,
            abstained_count=abstained,
            unavailable_count=missing,
            axis=axis,
        )
        risk: float | None = None
        unavailable: dict[str, str] = {}
        if task_type is TaskType.CLASSIFICATION:
            correct = 0
            total = 0
            for outcome in evaluated:
                metrics = outcome.curve_points[position].accepted_classification
                if metrics is None or metrics.accuracy is None:
                    continue
                correct += round(metrics.accuracy * metrics.sample_count)
                total += metrics.sample_count
            if total:
                risk = float(1.0 - (correct / total))
            else:
                unavailable["empirical_risk"] = (
                    f"no window was accepted at threshold {threshold!r} in any "
                    "fold, so accepted accuracy is undefined"
                )
        else:
            unavailable["empirical_risk"] = (
                "empirical risk is defined here as 1 - accepted accuracy, which "
                "is a classification quantity; regression selectivity is "
                "reported through accepted error and interval coverage instead"
            )
        points.append(
            SelectiveMetrics(
                axis=axis,
                threshold=float(threshold),
                coverage_point=point,
                empirical_risk=None,
                unavailable_metrics=unavailable
                | (
                    {}
                    if risk is None
                    else {
                        "empirical_risk_note": (
                            "pooled risk is recorded on the risk-coverage points "
                            "rather than here, because SelectiveMetrics ties "
                            "empirical_risk to an accepted-set ClassificationMetrics "
                            "document and the pooled point has none"
                        )
                    }
                ),
            )
        )

    risk_points = []
    from engagevr.schemas.uncertainty import RiskCoveragePoint

    for position, threshold in enumerate(values):
        point = points[position].coverage_point
        value: float | None = None
        reason: str | None = None
        if task_type is TaskType.CLASSIFICATION:
            correct = 0
            total = 0
            for outcome in evaluated:
                metrics = outcome.curve_points[position].accepted_classification
                if metrics is None or metrics.accuracy is None:
                    continue
                correct += round(metrics.accuracy * metrics.sample_count)
                total += metrics.sample_count
            if total:
                value = float(1.0 - (correct / total))
            else:
                reason = (
                    f"no window was accepted at threshold {threshold!r} in any "
                    "fold, so accepted accuracy is undefined"
                )
        else:
            reason = (
                "empirical risk is 1 - accepted accuracy, a classification "
                "quantity; it is undefined for a regression target"
            )
        risk_points.append(
            RiskCoveragePoint(
                axis=axis,
                threshold=float(threshold),
                coverage=point.coverage,
                empirical_risk=value,
                accepted_count=point.accepted_count,
                unavailable_reason=reason,
            )
        )

    area, reason = area_under_risk_coverage(tuple(risk_points))
    return CoverageCurve(
        task_type=task_type,
        axis=axis,
        axis_values=values,
        points=tuple(points),
        risk_coverage=tuple(risk_points),
        area_under_risk_coverage=area,
        area_under_risk_coverage_unavailable_reason=reason,
        expected_monotonic_direction=direction,
        coverage_is_monotonic=coverage_is_monotonic(
            [p.coverage_point for p in points], direction=direction
        ),
    )


def _build_evaluation(
    *,
    run_id: str,
    config: UncertaintyRunConfiguration,
    frame: ModellingFrame,
    splits: SplitManifest,
    fingerprint: str,
    group_field: GroupField,
    task_type: TaskType,
    labels: tuple[str, ...],
    outcomes: Sequence[_FoldOutcome],
    predictor_columns: tuple[str, ...],
    disclaimers: tuple[str, ...],
) -> UncertaintyEvaluation:
    folds = tuple(outcome.result for outcome in outcomes)
    total = sum(f.total_window_count for f in folds)
    accepted = sum(f.accepted_count for f in folds)
    abstained = sum(f.abstained_count for f in folds)
    missing = sum(f.unavailable_count for f in folds)
    personal = [record for f in folds for record in f.personal_thresholds]
    return UncertaintyEvaluation(
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
        configuration=config.selective,
        predictor_columns=predictor_columns,
        class_vocabulary=labels,
        folds=folds,
        total_window_count=total,
        accepted_count=accepted,
        abstained_count=abstained,
        unavailable_count=missing,
        coverage=(accepted / total) if total else None,
        abstention_rate=(abstained / total) if total else None,
        abstention_reason_counts=reason_counts(
            [d.reasons for outcome in outcomes for d in outcome.decisions]
        ),
        personalized_subject_count=sum(
            1 for record in personal if record.personalization_applied
        ),
        population_fallback_subject_count=sum(
            1 for record in personal if record.fallback_to_population
        ),
        adaptation_gate_eligible_count=sum(
            f.adaptation_gate_eligible_count for f in folds
        ),
        adaptation_gate_blocked_count=sum(
            f.adaptation_gate_blocked_count for f in folds
        ),
        disclaimers=disclaimers,
    )


def _metrics_document(
    *,
    run_id: str,
    config: UncertaintyRunConfiguration,
    frame: ModellingFrame,
    splits: SplitManifest,
    group_field: GroupField,
    task_type: TaskType,
    outcomes: Sequence[_FoldOutcome],
    predictor_columns: tuple[str, ...],
    model_name: str,
    disclaimers: tuple[str, ...],
) -> MetricsDocument:
    """Two results over the same folds: all windows, and accepted windows.

    Reusing ``MetricsDocument`` rather than defining a parallel one means
    the undefined-stays-null rule, the documented macro / Brier / ECE
    conventions, and the equal-weight fold aggregation are the Milestone 5
    ones, unchanged.
    """
    classification = task_type is TaskType.CLASSIFICATION
    failed = {
        outcome.result.fold_index: outcome.result.unavailable_reason or "not evaluated"
        for outcome in outcomes
        if not outcome.result.evaluated
    }
    note = (
        "The two results below cover the SAME outer folds. 'all_windows' scores "
        "every evaluated window; 'accepted_at_applied_threshold' scores only the "
        "windows the selective rule accepted, and its coverage is recorded in "
        "uncertainty.json and selective_metrics.json. An accepted-set score is "
        "not comparable to a whole-set score without that coverage."
    )

    results: list[ModelResult] = []
    if classification:
        all_folds = [
            outcome.all_window_classification
            for outcome in outcomes
            if outcome.all_window_classification is not None
        ]
        accepted_folds = [
            outcome.selective.accepted_classification
            for outcome in outcomes
            if outcome.selective is not None
            and outcome.selective.accepted_classification is not None
        ]
        results.append(
            ModelResult(
                model_name=ALL_WINDOWS_MODEL_NAME,
                model_kind="all_windows",
                parameters={"model": model_name},
                predictor_columns=predictor_columns,
                fold_classification_metrics=tuple(all_folds),
                aggregate=aggregate_fold_metrics(
                    all_folds,
                    CLASSIFICATION_AGGREGATE_FIELDS,
                    total_fold_count=splits.n_splits,
                ),
                failed_folds=failed,
                notes=(note, UNCERTAINTY_NOTE),
            )
        )
        results.append(
            ModelResult(
                model_name=ACCEPTED_MODEL_NAME,
                model_kind="selective",
                parameters={
                    "model": model_name,
                    "acceptance_rule": acceptance_rule_for(task_type),
                },
                predictor_columns=predictor_columns,
                fold_classification_metrics=tuple(accepted_folds),
                aggregate=aggregate_fold_metrics(
                    accepted_folds,
                    CLASSIFICATION_AGGREGATE_FIELDS,
                    total_fold_count=splits.n_splits,
                ),
                failed_folds=failed,
                notes=(note, SELECTIVE_PREDICTION_NOTE),
            )
        )
    else:
        all_folds_r = [
            outcome.all_window_regression
            for outcome in outcomes
            if outcome.all_window_regression is not None
        ]
        accepted_folds_r = [
            outcome.selective.accepted_regression
            for outcome in outcomes
            if outcome.selective is not None
            and outcome.selective.accepted_regression is not None
        ]
        results.append(
            ModelResult(
                model_name=ALL_WINDOWS_MODEL_NAME,
                model_kind="all_windows",
                parameters={"model": model_name},
                predictor_columns=predictor_columns,
                fold_regression_metrics=tuple(all_folds_r),
                aggregate=aggregate_fold_metrics(
                    all_folds_r,
                    REGRESSION_AGGREGATE_FIELDS,
                    total_fold_count=splits.n_splits,
                ),
                failed_folds=failed,
                notes=(note, UNCERTAINTY_NOTE),
            )
        )
        results.append(
            ModelResult(
                model_name=ACCEPTED_MODEL_NAME,
                model_kind="selective",
                parameters={
                    "model": model_name,
                    "acceptance_rule": acceptance_rule_for(task_type),
                },
                predictor_columns=predictor_columns,
                fold_regression_metrics=tuple(accepted_folds_r),
                aggregate=aggregate_fold_metrics(
                    accepted_folds_r,
                    REGRESSION_AGGREGATE_FIELDS,
                    total_fold_count=splits.n_splits,
                ),
                failed_folds=failed,
                notes=(note, SELECTIVE_PREDICTION_NOTE),
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
        results=tuple(results),
        disclaimers=disclaimers,
    )


def _thresholds_document(
    run_id: str,
    evaluation: UncertaintyEvaluation,
    disclaimers: tuple[str, ...],
) -> dict[str, Any]:
    settings = evaluation.configuration
    return {
        "run_id": run_id,
        "evaluation_mode": evaluation.evaluation_mode.value,
        "scientific_evaluation_eligible": evaluation.scientific_evaluation_eligible,
        "population_confidence_threshold": settings.population_confidence_threshold,
        "population_threshold_provenance": (
            "An ENGINEERING DEFAULT read from configuration. It was not chosen "
            "by looking at any result, it is not empirically optimal, it is not "
            "validated, and it is not a production threshold."
        ),
        "confidence_threshold_grid": list(settings.threshold_grid),
        "confidence_threshold_grid_units": COVERAGE_AXIS_UNITS[
            CoverageAxis.CONFIDENCE_THRESHOLD
        ],
        "maximum_interval_width": settings.maximum_interval_width,
        "interval_width_grid": (
            None
            if settings.interval_width_grid is None
            else list(settings.interval_width_grid)
        ),
        "interval_width_grid_units": COVERAGE_AXIS_UNITS[
            CoverageAxis.MAXIMUM_INTERVAL_WIDTH
        ],
        "grids_are_not_interchangeable": (
            "The confidence grid and the interval-width grid are separate "
            "surfaces sweeping opposite directions. A confidence value is a "
            "probability and raising it is stricter; a width is a distance in "
            "the target's units and raising it is more permissive. Neither is "
            "derived from the other, and 1 - width is not a confidence."
        ),
        "acceptance_rule": settings.classification_acceptance_rule,
        "regression_acceptance_rule": settings.regression_acceptance_rule,
        "estimation_enabled": settings.estimate_population_threshold,
        "estimation_objective": settings.threshold_objective.value,
        "estimation_target": settings.threshold_objective_target,
        "personalized_thresholds_enabled": settings.personalized_thresholds_enabled,
        "personalized_threshold_rule": (
            "tau_s = (1 - lambda) * tau_population + lambda * "
            "quantile(subject calibration confidence, 1 - target_coverage), "
            "lambda = n / (n + kappa). It reads only the subject's own EARLIER "
            "calibration windows and NO LABEL of any kind, so an evaluation "
            "label cannot influence it by any path."
        ),
        "leakage_rules": [
            "the outer-test fold never fits the model",
            "the outer-test fold never fits a probability calibrator",
            "the outer-test fold never fits a conformal residual distribution",
            "the outer-test fold never chooses a confidence threshold",
            "the outer-test fold never chooses an interval-width threshold",
            "the outer-test fold never tunes a personalized threshold",
            "no threshold is read off the reported coverage curve",
        ],
        "folds": [
            {
                "fold_index": fold.fold_index,
                "applied_population_threshold": fold.applied_population_threshold,
                "applied_population_threshold_source": (
                    fold.applied_population_threshold_source.value
                    if fold.applied_population_threshold_source
                    else None
                ),
                "fit_group_ids": list(fold.fit_group_ids),
                "probability_calibration_group_ids": list(
                    fold.probability_calibration_group_ids
                ),
                "threshold_selection_group_ids": list(
                    fold.threshold_selection_group_ids
                ),
                "conformal_calibration_group_ids": list(
                    fold.conformal_calibration_group_ids
                ),
                "outer_test_group_ids": list(fold.outer_test_group_ids),
                "estimated_threshold": (
                    fold.estimated_threshold.model_dump(mode="json")
                    if fold.estimated_threshold
                    else None
                ),
                "conformal_quantile": fold.conformal_quantile,
                "conformal_order_statistic": fold.conformal_order_statistic,
                "conformal_calibration_sample_count": (
                    fold.conformal_calibration_sample_count
                ),
                "conformal_unavailable_reason": fold.conformal_unavailable_reason,
                "personal_thresholds": [
                    record.model_dump(mode="json")
                    for record in fold.personal_thresholds
                ],
            }
            for fold in evaluation.folds
        ],
        "disclaimers": list(disclaimers),
    }


def _selective_document(
    run_id: str,
    evaluation: UncertaintyEvaluation,
    outcomes: Sequence[_FoldOutcome],
    disclaimers: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "evaluation_mode": evaluation.evaluation_mode.value,
        "scientific_evaluation_eligible": evaluation.scientific_evaluation_eligible,
        "coverage_equation": evaluation.configuration.coverage_equation,
        "total_window_count": evaluation.total_window_count,
        "accepted_count": evaluation.accepted_count,
        "abstained_count": evaluation.abstained_count,
        "unavailable_count": evaluation.unavailable_count,
        "coverage": evaluation.coverage,
        "abstention_rate": evaluation.abstention_rate,
        "abstention_reason_counts": dict(evaluation.abstention_reason_counts),
        "folds": [
            {
                "fold_index": outcome.result.fold_index,
                "applied": (
                    outcome.selective.model_dump(mode="json")
                    if outcome.selective
                    else None
                ),
            }
            for outcome in outcomes
        ],
        "note": SELECTIVE_PREDICTION_NOTE,
        "abstention_note": ABSTENTION_MEANING_NOTE,
        "disclaimers": list(disclaimers),
    }


def _calibration_document(
    run_id: str,
    config: UncertaintyRunConfiguration,
    records: list[dict[str, Any]],
    outcomes: Sequence[_FoldOutcome],
    disclaimers: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "requested_method": config.calibration_method.value,
        "calibration_group_fraction": config.calibration_group_fraction,
        "design": (
            "Probability calibration of the source model, fitted on calibration "
            "groups drawn from the training groups and disjoint from both the "
            "fit groups and the outer-test groups. For a regression target no "
            "probability is calibrated; the same calibration groups supply the "
            "split-conformal residual distribution instead."
        ),
        "note": (
            "A calibrated probability is not certainty and is not signal "
            "quality. A maximum taken from an UNCALIBRATED vector is recorded "
            "as a selection score, never as calibrated confidence."
        ),
        "conformal": [
            {
                "fold_index": outcome.result.fold_index,
                "available": outcome.result.conformal_available,
                "quantile": outcome.result.conformal_quantile,
                "order_statistic": outcome.result.conformal_order_statistic,
                "calibration_sample_count": (
                    outcome.result.conformal_calibration_sample_count
                ),
                "calibration_group_ids": list(
                    outcome.result.conformal_calibration_group_ids
                ),
                "unavailable_reason": outcome.result.conformal_unavailable_reason,
            }
            for outcome in outcomes
        ],
        "folds": records,
        "disclaimers": list(disclaimers),
    }


def _predictions_table(
    outcomes: Sequence[_FoldOutcome],
    labels: tuple[str, ...],
    task_type: TaskType,
    frame: ModellingFrame,
) -> pa.Table:
    """The original prediction, its probabilities, and every diagnostic.

    This table is the *unselected* record: it carries what the model said
    before any threshold was applied, so the effect of the selective layer
    is computable from the artifacts without re-running anything.
    """
    truth = {
        frame.window_ids[index]: frame.target_values[index]
        for index in range(frame.row_count)
    }
    columns: dict[str, list[Any]] = {
        key: []
        for key in (
            "window_id",
            "subject_id",
            "session_id",
            "fold_index",
            "source_model",
            "source_prediction_id",
            "true_value",
            "predicted_class",
            "predicted_value",
            "probability_calibration_status",
            "confidence_score",
            "selection_score",
            "entropy",
            "normalized_entropy",
            "margin",
            "interval_lower_bound",
            "interval_upper_bound",
            "interval_width",
            "conformal_quantile",
            "minimum_recorded_quality",
            "available_modality_count",
            "ensemble_disagreement",
            "data_source",
            "is_synthetic",
            "scientific_evaluation_eligible",
        )
    }
    for label in labels:
        columns[f"probability__{label}"] = []

    for outcome in outcomes:
        for confidence in outcome.confidences:
            columns["window_id"].append(confidence.window_id)
            columns["subject_id"].append(confidence.subject_id)
            columns["session_id"].append(confidence.session_id)
            columns["fold_index"].append(confidence.fold_index)
            columns["source_model"].append(confidence.source_model.value)
            columns["source_prediction_id"].append(confidence.source_prediction_id)
            columns["true_value"].append(str(truth.get(confidence.window_id)))
            columns["predicted_class"].append(confidence.predicted_class)
            columns["predicted_value"].append(None)
            columns["probability_calibration_status"].append(
                confidence.probability_calibration_status.value
            )
            columns["confidence_score"].append(confidence.confidence_score)
            columns["selection_score"].append(confidence.selection_score)
            columns["entropy"].append(confidence.entropy)
            columns["normalized_entropy"].append(confidence.normalized_entropy)
            columns["margin"].append(confidence.margin)
            columns["interval_lower_bound"].append(None)
            columns["interval_upper_bound"].append(None)
            columns["interval_width"].append(None)
            columns["conformal_quantile"].append(None)
            quality = confidence.signal_quality
            columns["minimum_recorded_quality"].append(
                quality.minimum_recorded_quality if quality else None
            )
            columns["available_modality_count"].append(
                len(quality.available_modalities) if quality else 0
            )
            columns["ensemble_disagreement"].append(
                confidence.disagreement.ensemble_disagreement
                if confidence.disagreement
                else None
            )
            columns["data_source"].append(confidence.data_source)
            columns["is_synthetic"].append(confidence.is_synthetic)
            columns["scientific_evaluation_eligible"].append(
                confidence.scientific_evaluation_eligible
            )
            for index, label in enumerate(labels):
                columns[f"probability__{label}"].append(
                    float(confidence.probabilities[index])
                )
        for record in outcome.intervals:
            columns["window_id"].append(record.window_id)
            columns["subject_id"].append(record.subject_id)
            columns["session_id"].append(record.session_id)
            columns["fold_index"].append(record.fold_index)
            columns["source_model"].append(record.source_model.value)
            columns["source_prediction_id"].append(record.source_prediction_id)
            columns["true_value"].append(str(truth.get(record.window_id)))
            columns["predicted_class"].append(None)
            columns["predicted_value"].append(record.predicted_value)
            columns["probability_calibration_status"].append(
                ProbabilityCalibrationStatus.UNAVAILABLE.value
            )
            columns["confidence_score"].append(None)
            columns["selection_score"].append(None)
            columns["entropy"].append(None)
            columns["normalized_entropy"].append(None)
            columns["margin"].append(None)
            columns["interval_lower_bound"].append(record.lower_bound)
            columns["interval_upper_bound"].append(record.upper_bound)
            columns["interval_width"].append(record.interval_width)
            columns["conformal_quantile"].append(record.conformal_quantile)
            quality = record.signal_quality
            columns["minimum_recorded_quality"].append(
                quality.minimum_recorded_quality if quality else None
            )
            columns["available_modality_count"].append(
                len(quality.available_modalities) if quality else 0
            )
            columns["ensemble_disagreement"].append(
                record.disagreement.ensemble_disagreement
                if record.disagreement
                else None
            )
            columns["data_source"].append(record.data_source)
            columns["is_synthetic"].append(record.is_synthetic)
            columns["scientific_evaluation_eligible"].append(
                record.scientific_evaluation_eligible
            )
            for label in labels:
                columns[f"probability__{label}"].append(None)

    _ = task_type
    return pa.table(columns)


def _selective_predictions_table(
    outcomes: Sequence[_FoldOutcome],
    labels: tuple[str, ...],
    task_type: TaskType,
    frame: ModellingFrame,
) -> pa.Table:
    """The decision layer: what was accepted, what abstained, and why.

    The original prediction is repeated here beside the decision so a
    reader never has to trust that an abstained row still holds one.
    """
    _ = labels, task_type
    truth = {
        frame.window_ids[index]: frame.target_values[index]
        for index in range(frame.row_count)
    }
    columns: dict[str, list[Any]] = {
        key: []
        for key in (
            "window_id",
            "subject_id",
            "session_id",
            "fold_index",
            "source_prediction_id",
            "true_value",
            "prediction_available",
            "accepted",
            "abstained",
            "primary_abstention_reason",
            "abstention_reasons",
            "predicted_class",
            "predicted_value",
            "confidence_score",
            "selection_score",
            "probability_calibration_status",
            "applied_threshold",
            "threshold_source",
            "interval_lower_bound",
            "interval_upper_bound",
            "interval_width",
            "maximum_interval_width",
            "evidence_gate_passed",
            "minimum_recorded_quality",
            "ensemble_disagreement",
            "data_source",
            "is_synthetic",
            "scientific_evaluation_eligible",
        )
    }
    for outcome in outcomes:
        for decision in outcome.decisions:
            primary = decision.primary_reason()
            quality = decision.signal_quality
            columns["window_id"].append(decision.window_id)
            columns["subject_id"].append(decision.subject_id)
            columns["session_id"].append(decision.session_id)
            columns["fold_index"].append(decision.fold_index)
            columns["source_prediction_id"].append(decision.source_prediction_id)
            columns["true_value"].append(str(truth.get(decision.window_id)))
            columns["prediction_available"].append(decision.prediction_available)
            columns["accepted"].append(decision.accepted)
            columns["abstained"].append(decision.abstained)
            columns["primary_abstention_reason"].append(
                primary.value if primary else None
            )
            columns["abstention_reasons"].append([r.value for r in decision.reasons])
            columns["predicted_class"].append(decision.predicted_class)
            columns["predicted_value"].append(decision.predicted_value)
            columns["confidence_score"].append(decision.confidence_score)
            columns["selection_score"].append(decision.selection_score)
            columns["probability_calibration_status"].append(
                decision.probability_calibration_status.value
                if decision.probability_calibration_status
                else None
            )
            columns["applied_threshold"].append(decision.applied_threshold)
            columns["threshold_source"].append(
                decision.threshold_source.value if decision.threshold_source else None
            )
            columns["interval_lower_bound"].append(decision.interval_lower_bound)
            columns["interval_upper_bound"].append(decision.interval_upper_bound)
            columns["interval_width"].append(decision.interval_width)
            columns["maximum_interval_width"].append(decision.maximum_interval_width)
            columns["evidence_gate_passed"].append(decision.evidence_gate_passed)
            columns["minimum_recorded_quality"].append(
                quality.minimum_recorded_quality if quality else None
            )
            columns["ensemble_disagreement"].append(
                decision.disagreement.ensemble_disagreement
                if decision.disagreement
                else None
            )
            columns["data_source"].append(decision.data_source)
            columns["is_synthetic"].append(decision.is_synthetic)
            columns["scientific_evaluation_eligible"].append(
                decision.scientific_evaluation_eligible
            )
    return pa.table(columns)


def _gate_table(outcomes: Sequence[_FoldOutcome]) -> pa.Table:
    """The adaptation-gate decisions. No column names an action."""
    columns: dict[str, list[Any]] = {
        key: []
        for key in (
            "window_id",
            "subject_id",
            "session_id",
            "fold_index",
            "source_prediction_id",
            "decision",
            "reasons",
            "prediction_available",
            "prediction_abstained",
            "evidence_gate_passed",
            "confidence_requirement_satisfied",
            "interval_requirement_satisfied",
            "applied_confidence_threshold",
            "maximum_interval_width",
            "data_source",
            "is_synthetic",
            "scientific_evaluation_eligible",
        )
    }
    for outcome in outcomes:
        for record in outcome.gates:
            columns["window_id"].append(record.window_id)
            columns["subject_id"].append(record.subject_id)
            columns["session_id"].append(record.session_id)
            columns["fold_index"].append(record.fold_index)
            columns["source_prediction_id"].append(record.source_prediction_id)
            columns["decision"].append(record.decision.value)
            columns["reasons"].append([r.value for r in record.reasons])
            columns["prediction_available"].append(record.prediction_available)
            columns["prediction_abstained"].append(record.prediction_abstained)
            columns["evidence_gate_passed"].append(record.evidence_gate_passed)
            columns["confidence_requirement_satisfied"].append(
                record.confidence_requirement_satisfied
            )
            columns["interval_requirement_satisfied"].append(
                record.interval_requirement_satisfied
            )
            columns["applied_confidence_threshold"].append(
                record.applied_confidence_threshold
            )
            columns["maximum_interval_width"].append(record.maximum_interval_width)
            columns["data_source"].append(record.data_source)
            columns["is_synthetic"].append(record.is_synthetic)
            columns["scientific_evaluation_eligible"].append(
                record.scientific_evaluation_eligible
            )
    return pa.table(columns)


def _manifest(
    *,
    run_id: str,
    config: UncertaintyRunConfiguration,
    frame: ModellingFrame,
    splits: SplitManifest,
    fingerprint: str,
    predictor_columns: tuple[str, ...],
    model_spec: Any,
    model_name: str,
    started: Any,
    finished: Any,
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
    settings = config.selective
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
        model_names=(model_name,),
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
            "split_manifest_fingerprint": fingerprint,
            "calibration_group_fraction": config.calibration_group_fraction,
            "calibration_bins": config.calibration_bins,
            "selective_prediction": settings.model_dump(mode="json"),
        },
        random_seed=config.random_seed,
        started_at_utc=started,
        finished_at_utc=finished,
        status=status,
        failure_reason=failure_reason,
        disclaimers=disclaimers,
    )


__all__ = [
    "ACCEPTED_MODEL_NAME",
    "ALL_WINDOWS_MODEL_NAME",
    "UNCERTAINTY_REQUIRED_ARTIFACTS",
    "ScientificModeError",
    "UncertaintyConfigurationError",
    "UncertaintyRunConfiguration",
    "UncertaintyRunResult",
    "interval_contains",
    "run_uncertainty",
]
