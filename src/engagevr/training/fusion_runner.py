"""Orchestration of a grouped cross-validated multimodal-fusion evaluation.

The ordering constraints are the substance of this milestone, so the control
flow is explicit rather than hidden behind a convenience wrapper:

1. Load and audit the dataset; refuse it in scientific mode if any predictor
   or target is synthetic, or if target provenance is missing.
2. Choose the grouping field and build the outer folds **once**, with the
   same call the Milestone 5 baseline runner makes.  Every strategy, every
   expert, and every missing-modality scenario reuses those exact folds, and
   the split manifest is fingerprinted so that reuse is checkable.
3. Inside each fold: fit one expert per modality on the fit groups, fit the
   early-fusion estimator on the same fit groups, calibrate on the
   calibration groups, and touch the test groups only to score.
4. Where a stacker or validation-derived weights are requested, build
   out-of-fold expert predictions inside the outer training portion, assert
   they are genuinely out of fold, and derive the meta-model or the weights
   from those alone.
5. Evaluate every strategy under every missing-modality scenario on the same
   fitted models, then write every artifact and the manifest, atomically and
   last.

Two things are deliberately kept apart
--------------------------------------
*Synthetic modality dropout* changes the dataset's recorded availability
before folding, so it affects training as well as evaluation: it describes a
dataset in which those modalities were never captured.  It is refused in
scientific mode.

*Missing-modality scenarios* are applied at evaluation time only.  The
models are trained once on the recorded availability and then met with each
deterministic availability pattern, which is the question a robustness
scenario actually asks: this system, as trained, encounters a window with no
rPPG — what does it do?

No champion is selected.  A synthetic self-check cannot rank fusion
architectures for any purpose that matters, and this code will not pretend
otherwise.
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

from engagevr.features.catalog import FEATURE_CATALOG_VERSION, get_catalog
from engagevr.features.windowing import utc_now
from engagevr.schemas.experiments import (
    SELF_CHECK_DISCLAIMER,
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
from engagevr.schemas.features import (
    AVAILABILITY_PREFIX,
    FEATURE_PREFIX,
    FeatureCatalog,
    modality_available_column,
    modality_quality_column,
)
from engagevr.schemas.fusion import (
    ExpertDocument,
    ExpertRecord,
    FusionConfiguration,
    FusionDiagnostics,
    FusionEvaluation,
    FusionExperimentManifest,
    FusionFoldResult,
    FusionModality,
    FusionPrediction,
    FusionStrategy,
    FusionStrategyResult,
    MissingModalityScenario,
    ModalityPrediction,
    ModalityWeight,
    RobustnessDocument,
    RobustnessResult,
    StackingProvenanceRecord,
    UnimodalControlResult,
    ValidationWeightRecord,
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
    importance_table,
    runtime_environment,
)
from engagevr.training.calibration import (
    CalibrationMethod,
    aligned_probabilities,
    calibrate_classifier,
)
from engagevr.training.experts import (
    ExpertFit,
    expert_predictions,
    fit_modality_expert,
    modality_availability,
    modality_quality,
)
from engagevr.training.fusion import (
    STRATEGY_DESCRIPTIONS,
    FusionError,
    build_fusion_weights,
    early_fusion_columns,
    fuse_class_probabilities,
    fuse_regression_predictions,
    modality_expert_columns,
    resolve_base_weights,
)
from engagevr.training.fusion_artifacts import (
    FUSION_REQUIRED_ARTIFACTS,
    build_fusion_run_id,
    expert_predictions_table,
    fusion_predictions_table,
    fusion_weights_table,
    split_manifest_fingerprint,
)
from engagevr.training.fusion_metrics import (
    aggregate_fusion_diagnostics,
    fusion_diagnostics,
    pool_diagnostics,
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
)
from engagevr.training.models import build_pipeline, describe_parameters, get_model_spec
from engagevr.training.preprocessing import ModellingFrame, load_modelling_frame
from engagevr.training.robustness import (
    REFERENCE_SCENARIO,
    SCIENTIFIC_DROPOUT_REFUSAL,
    apply_scenario,
    resolve_scenarios,
    synthetic_modality_dropout,
)
from engagevr.training.runner import (
    EvaluationError,
    ScientificModeError,
    assert_scientific_eligibility,
    linear_interpretation_records,
)
from engagevr.training.splits import build_splits, choose_group_field
from engagevr.training.stacking import (
    StackedFusionModel,
    build_out_of_fold_matrix,
    fit_stacked_meta_model,
    meta_feature_columns,
    meta_feature_row,
    stacked_predictions,
)

#: Metric each unimodal descriptive control is ranked by, per task type.
CONTROL_METRIC: dict[TaskType, tuple[str, bool]] = {
    TaskType.CLASSIFICATION: ("balanced_accuracy", True),
    TaskType.REGRESSION: ("mean_absolute_error", False),
}

#: Note attached to every unimodal-expert result.
UNIMODAL_EXPERT_NOTE = (
    "A unimodal expert scores only the windows in which its own modality "
    "contributed evidence, so its sample count and its coverage differ from "
    "a fusion result's. It is recorded as a descriptive control, never as a "
    "competitor with a comparable denominator."
)


class FusionConfigurationError(EvaluationError):
    """A fusion run cannot proceed as configured."""


@dataclass(frozen=True, slots=True)
class FusionRunConfiguration:
    """Everything that defines one fusion evaluation run."""

    dataset_path: Path
    target_name: TargetName
    output_directory: Path
    fusion: FusionConfiguration
    evaluation_mode: EvaluationMode = EvaluationMode.SOFTWARE_SELF_CHECK
    n_splits: int = 5
    random_seed: int = 42
    calibration_method: CalibrationMethod = CalibrationMethod.SIGMOID
    calibration_group_fraction: float = 0.25
    calibration_bins: int = DEFAULT_CALIBRATION_BINS
    catalog_version: str = FEATURE_CATALOG_VERSION


@dataclass(slots=True)
class FusionRunResult:
    """What a completed fusion run produced."""

    run_id: str
    directory: Path
    metrics: MetricsDocument
    fusion_metrics: FusionEvaluation
    splits: SplitManifest
    manifest: RunManifest
    experts: ExpertDocument
    robustness: RobustnessDocument
    scenarios: tuple[MissingModalityScenario, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class _StrategyFold:
    """One strategy's outcome on one fold under one scenario."""

    fold_index: int
    strategy: FusionStrategy
    scenario: str
    predictions: tuple[FusionPrediction, ...]
    classification: ClassificationMetrics | None
    regression: RegressionMetrics | None
    diagnostics: FusionDiagnostics


def _disclaimers(mode: EvaluationMode) -> tuple[str, ...]:
    if mode is EvaluationMode.SOFTWARE_SELF_CHECK:
        return (SELF_CHECK_DISCLAIMER, TARGET_DISCLAIMER)
    return (
        TARGET_DISCLAIMER,
        "This fusion run was executed in scientific mode. Eligibility was "
        "checked against data source and target provenance; eligibility is "
        "not validity, and no claim of experimental validation follows from "
        "a run completing.",
    )


def _rows_for_groups(group_values: Sequence[str], wanted: Sequence[str]) -> np.ndarray:
    members = set(wanted)
    return np.asarray(
        [index for index, group in enumerate(group_values) if group in members],
        dtype=int,
    )


def _columns_by_modality(
    modalities: Sequence[FusionModality],
    predictor_columns: Sequence[str],
    catalog: FeatureCatalog,
) -> dict[FusionModality, tuple[str, ...]]:
    return {
        modality: modality_expert_columns(
            modality, predictor_columns, catalog, include_modality_quality=True
        )
        for modality in modalities
    }


def _apply_availability_mask(
    predictors: pd.DataFrame,
    availability: dict[FusionModality, np.ndarray],
    columns_by_modality: dict[FusionModality, tuple[str, ...]],
) -> pd.DataFrame:
    """Represent an availability pattern in the predictor matrix itself.

    A modality marked absent for a row loses its measured values (they
    become missing, never zero), its per-feature availability flags become
    0, its modality-availability flag becomes 0, and its modality-quality
    value becomes missing.  That is exactly the shape a window with no
    evidence from that modality has in a real dataset, so an early-fusion
    estimator meets the same input it would meet in the field.
    """
    masked = predictors.copy()
    for modality, mask in availability.items():
        absent = ~np.asarray(mask, dtype=bool)
        if not absent.any():
            continue
        rows = masked.index[absent]
        for column in columns_by_modality.get(modality, ()):
            name = str(column)
            if name.startswith(FEATURE_PREFIX):
                masked.loc[rows, name] = np.nan
            elif name.startswith(AVAILABILITY_PREFIX):
                masked.loc[rows, name] = 0.0
            elif name == modality_available_column(modality.value):
                masked.loc[rows, name] = 0.0
            elif name == modality_quality_column(modality.value):
                masked.loc[rows, name] = np.nan
    return masked


def run_fusion(config: FusionRunConfiguration) -> FusionRunResult:
    """Execute a complete grouped cross-validated fusion evaluation."""
    started = utc_now()
    catalog = get_catalog(config.catalog_version)
    spec = get_target_spec(config.target_name)
    fusion = config.fusion
    modalities = tuple(fusion.modalities)

    frame = load_modelling_frame(
        config.dataset_path, target_name=config.target_name, catalog=catalog
    )
    if config.evaluation_mode is EvaluationMode.SCIENTIFIC:
        assert_scientific_eligibility(frame)

    columns_by_modality = _columns_by_modality(
        modalities, frame.predictor_columns, catalog
    )
    missing = sorted(
        m.value for m, columns in columns_by_modality.items() if not columns
    )
    if missing:
        raise FusionConfigurationError(
            f"modality group(s) {missing} contribute no permitted predictor to "
            f"{config.dataset_path}. Fusion over the requested groups is "
            "reported as unavailable rather than silently reduced to whatever "
            "happened to survive."
        )

    availability = {
        modality: modality_availability(frame.predictors, modality, catalog)
        for modality in modalities
    }
    quality = {
        modality: modality_quality(frame.predictors, modality)
        for modality in modalities
    }

    dropout_applied = False
    if fusion.robustness.synthetic_dropout_enabled:
        if config.evaluation_mode is EvaluationMode.SCIENTIFIC:
            raise ScientificModeError(SCIENTIFIC_DROPOUT_REFUSAL)
        availability = synthetic_modality_dropout(
            availability,
            window_ids=frame.window_ids,
            seed=fusion.robustness.synthetic_dropout_seed,
            probability=fusion.robustness.synthetic_dropout_probability,
        )
        dropout_applied = True
        for modality in modalities:
            absent = ~availability[modality]
            quality[modality] = np.where(absent, np.nan, quality[modality])

    predictors = (
        _apply_availability_mask(frame.predictors, availability, columns_by_modality)
        if dropout_applied
        else frame.predictors
    )

    early_columns = early_fusion_columns(
        modalities,
        frame.predictor_columns,
        catalog,
        include_modality_quality=fusion.include_modality_quality_in_early_fusion,
    )

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

    labels: tuple[str, ...] = ()
    if spec.task_type is TaskType.CLASSIFICATION:
        observed = set(class_labels or [])
        labels = tuple(
            label for label in (spec.class_vocabulary or ()) if label in observed
        )
        if len(labels) < 2:
            raise FusionConfigurationError(
                f"target {config.target_name.value!r} has {len(labels)} class(es) "
                "present in the dataset; classification requires at least two"
            )

    scenarios = (
        resolve_scenarios(fusion.robustness.scenarios or None)
        if fusion.robustness.enabled
        else (REFERENCE_SCENARIO,)
    )
    if REFERENCE_SCENARIO not in scenarios:
        scenarios = (REFERENCE_SCENARIO, *scenarios)

    fingerprint = split_manifest_fingerprint(splits)
    run_id = build_fusion_run_id(
        target_name=config.target_name.value,
        task_type=spec.task_type.value,
        evaluation_mode=config.evaluation_mode.value,
        dataset_fingerprint=frame.metadata.dataset_fingerprint,
        split_manifest_fingerprint=fingerprint,
        random_seed=config.random_seed,
        fusion=fusion,
        calibration_method=config.calibration_method.value,
        scenario_names=[s.name for s in scenarios],
    )
    run = ExperimentRun(
        config.output_directory, run_id, required_artifacts=FUSION_REQUIRED_ARTIFACTS
    )
    disclaimers = _disclaimers(config.evaluation_mode)

    expert_model_name = (
        fusion.expert_model_classification
        if spec.task_type is TaskType.CLASSIFICATION
        else fusion.expert_model_regression
    )
    model_spec = get_model_spec(expert_model_name, spec.task_type)
    base_weights = resolve_base_weights(modalities, fusion.quality.base_weights)

    outcomes: dict[tuple[FusionStrategy, str], list[_StrategyFold]] = {
        (strategy, scenario.name): []
        for strategy in fusion.strategies
        for scenario in scenarios
    }
    expert_records: list[ExpertRecord] = []
    expert_fold_metrics: dict[FusionModality, list[Any]] = {m: [] for m in modalities}
    expert_reference_predictions: list[FusionPrediction] = []
    importance_rows: list[dict[str, Any]] = []
    validation_records: list[ValidationWeightRecord] = []
    stacking_records: list[StackingProvenanceRecord] = []
    failed_folds: dict[int, str] = {}
    warnings: list[str] = []

    try:
        for fold in splits.folds:
            if not fold.valid:
                failed_folds[fold.fold_index] = (
                    fold.invalid_reason or "fold marked invalid"
                )
                continue
            _evaluate_fold(
                fold,
                config=config,
                frame=frame,
                predictors=predictors,
                catalog=catalog,
                model_spec=model_spec,
                group_values=group_values,
                labels=labels,
                task_type=spec.task_type,
                modalities=modalities,
                availability=availability,
                quality=quality,
                columns_by_modality=columns_by_modality,
                early_columns=early_columns,
                base_weights=base_weights,
                scenarios=scenarios,
                run=run,
                outcomes=outcomes,
                expert_records=expert_records,
                expert_fold_metrics=expert_fold_metrics,
                expert_reference_predictions=expert_reference_predictions,
                importance_rows=importance_rows,
                validation_records=validation_records,
                stacking_records=stacking_records,
                warnings=warnings,
            )

        strategy_results = _build_strategy_results(
            config=config,
            outcomes=outcomes,
            modalities=modalities,
            task_type=spec.task_type,
            expert_model_name=expert_model_name,
            validation_records=validation_records,
            stacking_records=stacking_records,
            failed_folds=failed_folds,
            total_fold_count=splits.n_splits,
        )
        unimodal = _unimodal_control(
            expert_fold_metrics, spec.task_type, total_fold_count=splits.n_splits
        )
        model_results = _model_results(
            config=config,
            outcomes=outcomes,
            strategy_results=strategy_results,
            expert_fold_metrics=expert_fold_metrics,
            modalities=modalities,
            task_type=spec.task_type,
            early_columns=early_columns,
            columns_by_modality=columns_by_modality,
            expert_model_name=expert_model_name,
            total_fold_count=splits.n_splits,
        )

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
            results=tuple(model_results),
            disclaimers=disclaimers,
        )
        fusion_evaluation = FusionEvaluation(
            run_id=run_id,
            evaluation_mode=config.evaluation_mode,
            scientific_evaluation_eligible=(
                config.evaluation_mode is EvaluationMode.SCIENTIFIC
            ),
            target_name=config.target_name.value,
            task_type=spec.task_type,
            dataset_fingerprint=frame.metadata.dataset_fingerprint,
            split_manifest_fingerprint=fingerprint,
            group_field=group_field.value,
            group_count=splits.group_count,
            fold_count=splits.n_splits,
            random_seed=config.random_seed,
            modalities=modalities,
            strategies=tuple(strategy_results),
            unimodal_control=unimodal,
            disclaimers=disclaimers,
        )
        robustness = _robustness_document(
            run_id=run_id,
            config=config,
            outcomes=outcomes,
            scenarios=scenarios,
            modalities=modalities,
            task_type=spec.task_type,
            total_fold_count=splits.n_splits,
            dropout_applied=dropout_applied,
            disclaimers=disclaimers,
        )
        experts_document = ExpertDocument(
            run_id=run_id,
            evaluation_mode=config.evaluation_mode,
            target_name=config.target_name.value,
            task_type=spec.task_type,
            experts=tuple(expert_records),
            disclaimers=disclaimers,
        )
        fusion_manifest = FusionExperimentManifest(
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
            fusion=fusion,
            calibration_method=config.calibration_method.value,
            calibration_group_fraction=config.calibration_group_fraction,
            early_fusion_columns=early_columns,
            modality_columns={
                modality.value: columns
                for modality, columns in columns_by_modality.items()
            },
            scenarios=scenarios,
            disclaimers=disclaimers,
        )

        all_predictions = [
            prediction
            for entries in outcomes.values()
            for entry in entries
            for prediction in entry.predictions
        ]
        true_by_window = {
            frame.window_ids[index]: frame.target_values[index]
            for index in range(frame.row_count)
        }

        run.write_json("dataset.json", frame.metadata.model_dump(mode="json"))
        run.write_json("feature_catalog.json", catalog.model_dump(mode="json"))
        run.write_json("splits.json", splits.model_dump(mode="json"))
        run.write_json("fusion_config.json", fusion_manifest.model_dump(mode="json"))
        run.write_json("experts.json", experts_document.model_dump(mode="json"))
        run.write_json("metrics.json", metrics.model_dump(mode="json"))
        run.write_json("fusion_metrics.json", fusion_evaluation.model_dump(mode="json"))
        run.write_json("robustness.json", robustness.model_dump(mode="json"))
        run.write_json(
            "calibration.json",
            _calibration_document(
                run_id=run_id,
                config=config,
                expert_records=expert_records,
                disclaimers=disclaimers,
            ),
        )
        run.write_table(
            "predictions.parquet",
            fusion_predictions_table(
                all_predictions,
                labels,
                spec.task_type,
                [true_by_window.get(p.window_id) for p in all_predictions],
            ),
        )
        run.write_table(
            "expert_predictions.parquet",
            expert_predictions_table(
                expert_reference_predictions, labels, spec.task_type
            ),
        )
        run.write_table("fusion_weights.parquet", fusion_weights_table(all_predictions))
        run.write_table("feature_importance.parquet", importance_table(importance_rows))
        run.write_model_warning()

        manifest = _manifest(
            run_id=run_id,
            config=config,
            frame=frame,
            splits=splits,
            early_columns=early_columns,
            expert_model_name=expert_model_name,
            scenarios=scenarios,
            fingerprint=fingerprint,
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
            early_columns=early_columns,
            expert_model_name=expert_model_name,
            scenarios=scenarios,
            fingerprint=fingerprint,
            started=started,
            finished=utc_now(),
            status=RunStatus.FAILED,
            failure_reason=f"{type(exc).__name__}: {exc}",
            disclaimers=disclaimers,
        )
        run.finalize(manifest)
        raise

    return FusionRunResult(
        run_id=run_id,
        directory=run.directory,
        metrics=metrics,
        fusion_metrics=fusion_evaluation,
        splits=splits,
        manifest=manifest,
        experts=experts_document,
        robustness=robustness,
        scenarios=scenarios,
        warnings=tuple(warnings),
    )


def _evaluate_fold(
    fold: FoldAssignment,
    *,
    config: FusionRunConfiguration,
    frame: ModellingFrame,
    predictors: pd.DataFrame,
    catalog: FeatureCatalog,
    model_spec: Any,
    group_values: Sequence[str],
    labels: tuple[str, ...],
    task_type: TaskType,
    modalities: tuple[FusionModality, ...],
    availability: dict[FusionModality, np.ndarray],
    quality: dict[FusionModality, np.ndarray],
    columns_by_modality: dict[FusionModality, tuple[str, ...]],
    early_columns: tuple[str, ...],
    base_weights: dict[FusionModality, float],
    scenarios: tuple[MissingModalityScenario, ...],
    run: ExperimentRun,
    outcomes: dict[tuple[FusionStrategy, str], list[_StrategyFold]],
    expert_records: list[ExpertRecord],
    expert_fold_metrics: dict[FusionModality, list[Any]],
    expert_reference_predictions: list[FusionPrediction],
    importance_rows: list[dict[str, Any]],
    validation_records: list[ValidationWeightRecord],
    stacking_records: list[StackingProvenanceRecord],
    warnings: list[str],
) -> None:
    """Fit every model for one outer fold and score every strategy/scenario."""
    fusion = config.fusion
    fit_groups = fold.fit_groups()
    fit_idx = _rows_for_groups(group_values, fit_groups)
    calibration_idx = _rows_for_groups(group_values, fold.calibration_groups)
    test_idx = _rows_for_groups(group_values, fold.test_groups)
    if fit_idx.size == 0 or test_idx.size == 0:
        raise FusionConfigurationError(
            f"fold {fold.fold_index}: the fit or test portion is empty"
        )
    train_idx = _rows_for_groups(group_values, fold.train_groups)

    experts: dict[FusionModality, ExpertFit] = {}
    for modality in modalities:
        expert = fit_modality_expert(
            modality=modality,
            predictors=predictors,
            catalog=catalog,
            target_values=frame.target_values,
            task_type=task_type,
            model_spec=model_spec,
            fit_indices=fit_idx,
            calibration_indices=calibration_idx,
            fit_groups=fit_groups,
            calibration_groups=fold.calibration_groups,
            test_groups=fold.test_groups,
            group_values=group_values,
            labels=labels,
            fold_index=fold.fold_index,
            random_seed=config.random_seed,
            calibration_method=config.calibration_method,
            use_calibrated=fusion.use_calibrated_experts,
            include_modality_quality=fusion.include_modality_quality_in_experts,
        )
        experts[modality] = expert
        expert_records.append(expert.record)
        if not expert.trained:
            warnings.append(
                f"fold {fold.fold_index}: no expert for modality "
                f"{modality.value!r} — {expert.record.unavailable_reason}"
            )
        elif expert.estimator is not None:
            if model_spec.kind.value == "linear":
                for record in linear_interpretation_records(
                    model_spec, expert.estimator, fold.fold_index, None
                ):
                    record["model_name"] = f"expert_{modality.value}"
                    importance_rows.append(record)
            if fold.fold_index == 0:
                run.save_model(f"expert-{modality.value}-fold0", expert.estimator)

    early_model, early_calibrated = _fit_early_model(
        predictors=predictors,
        early_columns=early_columns,
        target_values=frame.target_values,
        task_type=task_type,
        model_spec=model_spec,
        fit_idx=fit_idx,
        calibration_idx=calibration_idx,
        fit_groups=fit_groups,
        calibration_groups=fold.calibration_groups,
        test_groups=fold.test_groups,
        random_seed=config.random_seed,
        calibration_method=config.calibration_method,
        use_calibrated=fusion.use_calibrated_experts,
    )
    if model_spec.kind.value == "linear":
        for record in linear_interpretation_records(
            model_spec, early_model, fold.fold_index, None
        ):
            record["model_name"] = "early_fusion"
            importance_rows.append(record)
    if fold.fold_index == 0:
        run.save_model("early-fusion-fold0", early_model)
        if early_calibrated is not None:
            run.save_model("early-fusion-fold0-calibrated", early_calibrated)

    needs_out_of_fold = (
        FusionStrategy.STACKED_LATE in fusion.strategies
        or FusionStrategy.VALIDATION_WEIGHTED_LATE in fusion.strategies
    )
    stacked_model: StackedFusionModel | None = None
    validation_weights: dict[FusionModality, float] | None = None

    if needs_out_of_fold:
        out_of_fold = build_out_of_fold_matrix(
            modalities=modalities,
            predictors=predictors,
            catalog=catalog,
            target_values=frame.target_values,
            task_type=task_type,
            model_spec=model_spec,
            train_indices=train_idx,
            group_values=group_values,
            availability=availability,
            quality=quality,
            labels=labels,
            fold_index=fold.fold_index,
            random_seed=config.random_seed,
            inner_folds=fusion.stacking.inner_folds,
            include_modality_quality=fusion.include_modality_quality_in_experts,
        )
        if FusionStrategy.VALIDATION_WEIGHTED_LATE in fusion.strategies:
            weight_record, validation_weights = _validation_weights(
                out_of_fold=out_of_fold,
                modalities=modalities,
                target_values=frame.target_values,
                task_type=task_type,
                labels=labels,
                group_values=group_values,
                fold_index=fold.fold_index,
            )
            validation_records.append(weight_record)
        if FusionStrategy.STACKED_LATE in fusion.strategies:
            if not out_of_fold.available:
                stacking_records.append(
                    StackingProvenanceRecord(
                        fold_index=fold.fold_index,
                        available=False,
                        unavailable_reason=(
                            out_of_fold.unavailable_reason
                            or "no out-of-fold matrix could be built"
                        ),
                        outer_train_group_count=len(fold.train_groups),
                    )
                )
            else:
                stacked_model, reason = fit_stacked_meta_model(
                    features=out_of_fold.features,
                    row_indices=out_of_fold.row_indices,
                    provenance=out_of_fold.provenance,
                    target_values=frame.target_values,
                    task_type=task_type,
                    labels=labels,
                    outer_train_groups=fold.train_groups,
                    outer_test_groups=fold.test_groups,
                    group_values=group_values,
                    inner_fold_count=fusion.stacking.inner_folds,
                    random_seed=config.random_seed,
                    meta_model_classification=(
                        fusion.stacking.meta_model_classification
                    ),
                    meta_model_regression=fusion.stacking.meta_model_regression,
                    minimum_available_experts=fusion.minimum_modalities,
                )
                stacking_records.append(
                    StackingProvenanceRecord(
                        fold_index=fold.fold_index,
                        available=stacked_model is not None,
                        unavailable_reason=reason,
                        meta_model_name=(
                            None
                            if stacked_model is None
                            else stacked_model.meta_model_name
                        ),
                        inner_fold_count=fusion.stacking.inner_folds,
                        out_of_fold_row_count=int(out_of_fold.row_indices.size),
                        meta_training_row_count=(
                            0
                            if stacked_model is None
                            else stacked_model.training_row_count
                        ),
                        meta_training_group_count=(
                            0
                            if stacked_model is None
                            else stacked_model.training_group_count
                        ),
                        outer_train_group_count=len(fold.train_groups),
                        probabilities_are_calibrated=False,
                        leakage_checks_passed=(
                            "no meta-training row was predicted by experts fitted "
                            "on its own group",
                            "no meta-training row was predicted by experts fitted "
                            "on an outer-test group",
                            "every meta-training row lies inside the outer "
                            "training groups",
                        ),
                    )
                )
                if stacked_model is None and reason:
                    warnings.append(
                        f"fold {fold.fold_index}: stacked fusion unavailable — {reason}"
                    )

    row_context = [
        (
            frame.window_ids[int(index)],
            frame.subject_ids[int(index)],
            frame.session_ids[int(index)],
            frame.data_sources[int(index)],
        )
        for index in test_idx
    ]

    for scenario in scenarios:
        scenario_availability = apply_scenario(
            {m: availability[m][test_idx] for m in modalities}, scenario
        )
        scenario_quality = {
            m: np.where(scenario_availability[m], quality[m][test_idx], np.nan)
            for m in modalities
        }
        calibrated_predictions = {
            modality: expert_predictions(
                experts[modality],
                predictors=predictors,
                row_indices=test_idx,
                availability=scenario_availability[modality],
                quality=scenario_quality[modality],
                labels=labels,
                task_type=task_type,
                prefer_calibrated=fusion.use_calibrated_experts,
            )
            for modality in modalities
        }
        uncalibrated_predictions = (
            {
                modality: expert_predictions(
                    experts[modality],
                    predictors=predictors,
                    row_indices=test_idx,
                    availability=scenario_availability[modality],
                    quality=scenario_quality[modality],
                    labels=labels,
                    task_type=task_type,
                    prefer_calibrated=False,
                )
                for modality in modalities
            }
            if fusion.use_calibrated_experts and stacked_model is not None
            else calibrated_predictions
        )

        if scenario.name == REFERENCE_SCENARIO.name:
            _record_expert_metrics(
                calibrated_predictions,
                modalities=modalities,
                test_idx=test_idx,
                frame=frame,
                group_values=group_values,
                labels=labels,
                task_type=task_type,
                calibration_bins=config.calibration_bins,
                expert_fold_metrics=expert_fold_metrics,
            )

        early_predicted, early_probabilities = _early_predictions(
            model=early_model,
            calibrated=early_calibrated if fusion.use_calibrated_experts else None,
            predictors=predictors,
            early_columns=early_columns,
            test_idx=test_idx,
            scenario_availability=scenario_availability,
            columns_by_modality=columns_by_modality,
            labels=labels,
            task_type=task_type,
        )

        for strategy in config.fusion.strategies:
            predictions = _strategy_predictions(
                strategy=strategy,
                config=config,
                fold_index=fold.fold_index,
                scenario=scenario,
                row_context=row_context,
                modalities=modalities,
                scenario_availability=scenario_availability,
                scenario_quality=scenario_quality,
                calibrated_predictions=calibrated_predictions,
                uncalibrated_predictions=uncalibrated_predictions,
                base_weights=base_weights,
                labels=labels,
                task_type=task_type,
                early_predicted=early_predicted,
                early_probabilities=early_probabilities,
                validation_weights=validation_weights,
                stacked_model=stacked_model,
            )
            metrics_classification, metrics_regression = _score(
                predictions,
                frame=frame,
                test_idx=test_idx,
                group_values=group_values,
                labels=labels,
                task_type=task_type,
                calibration_bins=config.calibration_bins,
            )
            outcomes[(strategy, scenario.name)].append(
                _StrategyFold(
                    fold_index=fold.fold_index,
                    strategy=strategy,
                    scenario=scenario.name,
                    predictions=predictions,
                    classification=metrics_classification,
                    regression=metrics_regression,
                    diagnostics=fusion_diagnostics(predictions, modalities, task_type),
                )
            )
            if (
                scenario.name == REFERENCE_SCENARIO.name
                and strategy is _first_late_strategy(config.fusion.strategies)
            ):
                expert_reference_predictions.extend(predictions)


def _first_late_strategy(
    strategies: Sequence[FusionStrategy],
) -> FusionStrategy | None:
    """The strategy whose expert outputs are written to the expert table."""
    for strategy in strategies:
        if strategy is not FusionStrategy.EARLY:
            return strategy
    return None


def _fit_early_model(
    *,
    predictors: pd.DataFrame,
    early_columns: tuple[str, ...],
    target_values: np.ndarray,
    task_type: TaskType,
    model_spec: Any,
    fit_idx: np.ndarray,
    calibration_idx: np.ndarray,
    fit_groups: Sequence[str],
    calibration_groups: Sequence[str],
    test_groups: Sequence[str],
    random_seed: int,
    calibration_method: CalibrationMethod,
    use_calibrated: bool,
) -> tuple[BaseEstimator, BaseEstimator | None]:
    """Fit the early-fusion estimator on the fold's fit groups only."""
    X = predictors.loc[:, list(early_columns)]
    X_fit = X.iloc[fit_idx]
    y_fit = target_values[fit_idx]
    pipeline = build_pipeline(model_spec, early_columns, random_seed=random_seed)
    if task_type is TaskType.CLASSIFICATION and use_calibrated:
        base, outcome = calibrate_classifier(
            pipeline,
            X_fit=X_fit,
            y_fit=[str(v) for v in y_fit],
            X_calibration=X.iloc[calibration_idx],
            y_calibration=[str(v) for v in target_values[calibration_idx]],
            method=calibration_method,
            fit_groups=fit_groups,
            calibration_groups=calibration_groups,
            test_groups=test_groups,
        )
        return base, outcome.calibrated_estimator
    base = clone(pipeline)
    base.fit(
        X_fit,
        [str(v) for v in y_fit] if task_type is TaskType.CLASSIFICATION else y_fit,
    )
    return base, None


def _early_predictions(
    *,
    model: BaseEstimator,
    calibrated: BaseEstimator | None,
    predictors: pd.DataFrame,
    early_columns: tuple[str, ...],
    test_idx: np.ndarray,
    scenario_availability: dict[FusionModality, np.ndarray],
    columns_by_modality: dict[FusionModality, tuple[str, ...]],
    labels: tuple[str, ...],
    task_type: TaskType,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Early-fusion predictions for the test rows under one scenario."""
    X_test = predictors.iloc[test_idx]
    masked = _apply_availability_mask(
        X_test, scenario_availability, columns_by_modality
    )
    matrix = masked.loc[:, list(early_columns)]
    estimator = calibrated if calibrated is not None else model
    predicted = np.asarray(estimator.predict(matrix))
    probabilities = (
        aligned_probabilities(estimator, matrix, labels)
        if task_type is TaskType.CLASSIFICATION
        else None
    )
    return predicted, probabilities


def _strategy_predictions(
    *,
    strategy: FusionStrategy,
    config: FusionRunConfiguration,
    fold_index: int,
    scenario: MissingModalityScenario,
    row_context: Sequence[tuple[str, str, str, str]],
    modalities: tuple[FusionModality, ...],
    scenario_availability: dict[FusionModality, np.ndarray],
    scenario_quality: dict[FusionModality, np.ndarray],
    calibrated_predictions: dict[FusionModality, tuple[ModalityPrediction, ...]],
    uncalibrated_predictions: dict[FusionModality, tuple[ModalityPrediction, ...]],
    base_weights: dict[FusionModality, float],
    labels: tuple[str, ...],
    task_type: TaskType,
    early_predicted: np.ndarray,
    early_probabilities: np.ndarray | None,
    validation_weights: dict[FusionModality, float] | None,
    stacked_model: StackedFusionModel | None,
) -> tuple[FusionPrediction, ...]:
    """Fused predictions for one strategy on one fold under one scenario."""
    fusion = config.fusion
    classification = task_type is TaskType.CLASSIFICATION
    is_synthetic_run = config.evaluation_mode is EvaluationMode.SOFTWARE_SELF_CHECK
    eligible = config.evaluation_mode is EvaluationMode.SCIENTIFIC
    minimum = fusion.minimum_modalities

    source = (
        uncalibrated_predictions
        if strategy is FusionStrategy.STACKED_LATE
        else calibrated_predictions
    )
    stacked_predicted: np.ndarray | None = None
    stacked_probabilities: np.ndarray | None = None
    if strategy is FusionStrategy.STACKED_LATE and stacked_model is not None:
        rows = [
            meta_feature_row(
                {m: source[m][position] for m in modalities},
                modalities,
                labels,
                task_type,
            )
            for position in range(len(row_context))
        ]
        features = pd.DataFrame(
            rows,
            columns=list(meta_feature_columns(modalities, labels, task_type)),
            dtype=float,
        )
        stacked_predicted, stacked_probabilities = stacked_predictions(
            stacked_model, features, task_type
        )

    results: list[FusionPrediction] = []
    for position, (window_id, subject_id, session_id, data_source) in enumerate(
        row_context
    ):
        per_modality = {m: source[m][position] for m in modalities}
        available = tuple(m for m in modalities if per_modality[m].available)
        unavailable = tuple(m for m in modalities if not per_modality[m].available)
        quality_map = {
            m.value: (
                float(scenario_quality[m][position])
                if np.isfinite(scenario_quality[m][position])
                else None
            )
            for m in modalities
        }
        common: dict[str, Any] = {
            "window_id": window_id,
            "subject_id": subject_id,
            "session_id": session_id,
            "target_name": config.target_name.value,
            "task_type": task_type,
            "fold_index": fold_index,
            "strategy": strategy,
            "scenario": scenario.name,
            "participating_modalities": modalities,
            "modality_quality": quality_map,
            "data_source": data_source,
            "is_synthetic": is_synthetic_run or data_source == "synthetic",
            "scientific_evaluation_eligible": eligible and data_source != "synthetic",
        }

        if strategy is FusionStrategy.EARLY:
            present = tuple(
                m for m in modalities if bool(scenario_availability[m][position])
            )
            absent = tuple(m for m in modalities if m not in set(present))
            if len(present) < minimum:
                results.append(
                    FusionPrediction(
                        **common,
                        available_modalities=present,
                        unavailable_modalities=absent,
                        fused=False,
                        unavailable_reason=(
                            f"{len(present)} modality group(s) contributed "
                            f"evidence to this window; the configured minimum "
                            f"is {minimum}"
                        ),
                    )
                )
                continue
            results.append(
                FusionPrediction(
                    **common,
                    available_modalities=present,
                    unavailable_modalities=absent,
                    fused=True,
                    predicted_class=(
                        str(early_predicted[position]) if classification else None
                    ),
                    class_vocabulary=labels if classification else (),
                    probabilities=(
                        tuple(float(v) for v in early_probabilities[position])
                        if classification and early_probabilities is not None
                        else ()
                    ),
                    predicted_value=(
                        None if classification else float(early_predicted[position])
                    ),
                )
            )
            continue

        if strategy is FusionStrategy.STACKED_LATE:
            if stacked_model is None or stacked_predicted is None:
                results.append(
                    FusionPrediction(
                        **common,
                        available_modalities=available,
                        unavailable_modalities=unavailable,
                        modality_predictions=tuple(per_modality[m] for m in modalities),
                        fused=False,
                        unavailable_reason=(
                            "no leakage-safe stacked meta-model could be fitted "
                            "for this fold"
                        ),
                    )
                )
                continue
            if len(available) < minimum:
                results.append(
                    FusionPrediction(
                        **common,
                        available_modalities=available,
                        unavailable_modalities=unavailable,
                        modality_predictions=tuple(per_modality[m] for m in modalities),
                        fused=False,
                        unavailable_reason=(
                            f"{len(available)} modality expert(s) produced a "
                            f"prediction; the configured minimum is {minimum}"
                        ),
                    )
                )
                continue
            results.append(
                FusionPrediction(
                    **common,
                    available_modalities=available,
                    unavailable_modalities=unavailable,
                    modality_predictions=tuple(per_modality[m] for m in modalities),
                    fused=True,
                    predicted_class=(
                        str(stacked_predicted[position]) if classification else None
                    ),
                    class_vocabulary=labels if classification else (),
                    probabilities=(
                        tuple(float(v) for v in stacked_probabilities[position])
                        if classification and stacked_probabilities is not None
                        else ()
                    ),
                    predicted_value=(
                        None if classification else float(stacked_predicted[position])
                    ),
                )
            )
            continue

        weights = _weights_for(
            strategy=strategy,
            modalities=modalities,
            per_modality=per_modality,
            base_weights=base_weights,
            quality_map={
                m: (
                    float(scenario_quality[m][position])
                    if np.isfinite(scenario_quality[m][position])
                    else None
                )
                for m in modalities
            },
            fusion=fusion,
            validation_weights=validation_weights,
        )
        contributors = [w for w in weights if w.contributed]
        if len(available) < minimum:
            reason = (
                f"{len(available)} modality expert(s) produced a prediction; "
                f"the configured minimum is {minimum}"
            )
        elif len(contributors) < minimum:
            excluded = "; ".join(
                f"{w.modality.value}: {w.exclusion_reason}"
                for w in weights
                if not w.contributed and w.modality in set(available)
            )
            reason = (
                f"{len(contributors)} modality expert(s) carried a non-zero "
                f"fusion weight; the configured minimum is {minimum}"
                + (f". Excluded: {excluded}" if excluded else "")
            )
        else:
            reason = None

        if reason is not None:
            results.append(
                FusionPrediction(
                    **common,
                    available_modalities=available,
                    unavailable_modalities=unavailable,
                    modality_predictions=tuple(per_modality[m] for m in modalities),
                    fusion_weights=(),
                    fused=False,
                    unavailable_reason=reason,
                )
            )
            continue

        if classification:
            contributions = [
                (w.normalized_weight, per_modality[w.modality].probabilities)
                for w in contributors
            ]
            fused_vector = fuse_class_probabilities(contributions, labels)
            predicted_class = labels[int(np.argmax(np.asarray(fused_vector)))]
            results.append(
                FusionPrediction(
                    **common,
                    available_modalities=available,
                    unavailable_modalities=unavailable,
                    modality_predictions=tuple(per_modality[m] for m in modalities),
                    fusion_weights=weights,
                    fused=True,
                    predicted_class=predicted_class,
                    class_vocabulary=labels,
                    probabilities=fused_vector,
                )
            )
        else:
            numeric: list[tuple[float, float]] = []
            for weight in contributors:
                value = per_modality[weight.modality].predicted_value
                if value is not None:
                    numeric.append((weight.normalized_weight, float(value)))
            results.append(
                FusionPrediction(
                    **common,
                    available_modalities=available,
                    unavailable_modalities=unavailable,
                    modality_predictions=tuple(per_modality[m] for m in modalities),
                    fusion_weights=weights,
                    fused=True,
                    predicted_value=fuse_regression_predictions(numeric),
                )
            )
    return tuple(results)


def _weights_for(
    *,
    strategy: FusionStrategy,
    modalities: tuple[FusionModality, ...],
    per_modality: dict[FusionModality, ModalityPrediction],
    base_weights: dict[FusionModality, float],
    quality_map: dict[FusionModality, float | None],
    fusion: FusionConfiguration,
    validation_weights: dict[FusionModality, float] | None,
) -> tuple[ModalityWeight, ...]:
    if strategy is FusionStrategy.UNIFORM_LATE:
        return build_fusion_weights(
            modalities=modalities,
            predictions=per_modality,
            base_weights={m: 1.0 for m in modalities},
            quality=None,
            quality_config=None,
        )
    if strategy is FusionStrategy.QUALITY_LATE:
        return build_fusion_weights(
            modalities=modalities,
            predictions=per_modality,
            base_weights=base_weights,
            quality=quality_map,
            quality_config=fusion.quality,
        )
    if strategy is FusionStrategy.VALIDATION_WEIGHTED_LATE:
        weights = validation_weights or {m: 1.0 for m in modalities}
        return build_fusion_weights(
            modalities=modalities,
            predictions=per_modality,
            base_weights=weights,
            quality=None,
            quality_config=None,
        )
    raise FusionError(  # pragma: no cover - guarded by the caller
        f"strategy {strategy.value!r} does not use per-window weights"
    )


def _validation_weights(
    *,
    out_of_fold: Any,
    modalities: tuple[FusionModality, ...],
    target_values: np.ndarray,
    task_type: TaskType,
    labels: tuple[str, ...],
    group_values: Sequence[str],
    fold_index: int,
) -> tuple[ValidationWeightRecord, dict[FusionModality, float]]:
    """Derive late-fusion weights from inner validation groups only.

    Classification uses ``max(0, (balanced_accuracy - chance) / (1 - chance))``
    and regression uses ``max(0, 1 - MAE / MAE_of_predicting_the_mean)``.
    Both are bounded in ``[0, 1]``, deterministic, and scale-free, and
    neither can diverge when an expert happens to score perfectly — which
    is why a reciprocal-error rule is not used.  When every weight is zero
    the run falls back to equal weights and records that it did.

    The outer test fold contributes nothing to any of this: every score is
    computed on out-of-fold rows drawn from the outer training portion.
    """
    metric_name = (
        "balanced_accuracy" if task_type is TaskType.CLASSIFICATION else "mae_skill"
    )
    definition = (
        "weight_m = max(0, (balanced_accuracy_m - 1/n_classes) / (1 - 1/n_classes)) "
        "over out-of-fold rows in which modality m produced a prediction"
        if task_type is TaskType.CLASSIFICATION
        else (
            "weight_m = max(0, 1 - MAE_m / MAE_mean_predictor) over out-of-fold "
            "rows in which modality m produced a prediction, where "
            "MAE_mean_predictor is the mean absolute error of predicting the "
            "out-of-fold target mean"
        )
    )
    groups = tuple(sorted({str(group_values[int(i)]) for i in out_of_fold.row_indices}))
    raw: dict[str, float | None] = {}
    weights: dict[FusionModality, float] = {}

    if not out_of_fold.available:
        return (
            ValidationWeightRecord(
                fold_index=fold_index,
                metric_name=metric_name,
                metric_definition=definition,
                groups_used=groups,
                raw_scores={m.value: None for m in modalities},
                weights={m.value: 1.0 for m in modalities},
                fallback_applied=True,
                fallback_reason=(
                    out_of_fold.unavailable_reason
                    or "no out-of-fold evidence was available"
                ),
            ),
            {m: 1.0 for m in modalities},
        )

    truth = target_values[out_of_fold.row_indices]
    for modality in modalities:
        predictions = out_of_fold.modality_predictions.get(modality, ())
        mask = [p.available for p in predictions]
        if not any(mask):
            raw[modality.value] = None
            weights[modality] = 0.0
            continue
        selected_truth = [truth[i] for i, keep in enumerate(mask) if keep]
        if task_type is TaskType.CLASSIFICATION:
            predicted = [
                p.predicted_class
                for p, keep in zip(predictions, mask, strict=True)
                if keep
            ]
            metrics = classification_metrics(
                y_true=[str(v) for v in selected_truth],
                y_predicted=[str(v) for v in predicted],
                labels=labels,
                group_ids=[],
            )
            score = metrics.balanced_accuracy
            chance = 1.0 / max(1, len(labels))
            raw[modality.value] = score
            weights[modality] = (
                0.0
                if score is None or chance >= 1.0
                else max(0.0, (float(score) - chance) / (1.0 - chance))
            )
        else:
            predicted = [
                p.predicted_value
                for p, keep in zip(predictions, mask, strict=True)
                if keep
            ]
            truth_array = np.asarray([float(v) for v in selected_truth], dtype=float)
            error = float(
                np.mean(
                    np.abs(
                        truth_array
                        - np.asarray([float(v) for v in predicted], dtype=float)
                    )
                )
            )
            reference = float(np.mean(np.abs(truth_array - truth_array.mean())))
            raw[modality.value] = error
            weights[modality] = (
                0.0 if reference <= 0.0 else max(0.0, 1.0 - error / reference)
            )

    fallback = not any(weight > 0.0 for weight in weights.values())
    if fallback:
        weights = {m: 1.0 for m in modalities}
    # A zero weight would be refused as a base weight, so a modality that
    # scored at or below chance is given the smallest positive base weight
    # and is then excluded per window by its own availability.
    positive = {m: (w if w > 0.0 else 1e-6) for m, w in weights.items()}
    return (
        ValidationWeightRecord(
            fold_index=fold_index,
            metric_name=metric_name,
            metric_definition=definition,
            groups_used=groups,
            raw_scores=raw,
            weights={m.value: float(w) for m, w in positive.items()},
            fallback_applied=fallback,
            fallback_reason=(
                "no modality scored above the reference level on the inner "
                "validation groups, so deterministic equal weights are used"
                if fallback
                else None
            ),
        ),
        positive,
    )


def _score(
    predictions: Sequence[FusionPrediction],
    *,
    frame: ModellingFrame,
    test_idx: np.ndarray,
    group_values: Sequence[str],
    labels: tuple[str, ...],
    task_type: TaskType,
    calibration_bins: int,
) -> tuple[ClassificationMetrics | None, RegressionMetrics | None]:
    """Score the windows for which a strategy produced a prediction.

    Metrics describe the covered windows only; ``coverage`` in the fusion
    diagnostics states how many that was.  A metric computed over a subset
    is not comparable to one computed over the whole fold unless the reader
    can see both numbers, so both are always recorded.
    """
    fused = [
        (position, prediction)
        for position, prediction in enumerate(predictions)
        if prediction.fused
    ]
    truth = [frame.target_values[int(test_idx[position])] for position, _p in fused]
    groups = [str(group_values[int(test_idx[position])]) for position, _p in fused]

    if task_type is TaskType.CLASSIFICATION:
        predicted = [str(p.predicted_class) for _position, p in fused]
        probabilities = (
            np.asarray([p.probabilities for _position, p in fused], dtype=float)
            if fused
            else None
        )
        calibration = [
            calibration_metrics(
                label="fused",
                probabilities=probabilities,
                y_true=[str(v) for v in truth],
                labels=labels,
                bin_count=calibration_bins,
            )
        ]
        return (
            classification_metrics(
                y_true=[str(v) for v in truth],
                y_predicted=predicted,
                labels=labels,
                group_ids=groups,
                calibration=calibration,
            ),
            None,
        )
    return (
        None,
        regression_metrics(
            y_true=[float(v) for v in truth],
            y_predicted=[float(p.predicted_value or 0.0) for _position, p in fused],
            group_ids=groups,
        ),
    )


def _record_expert_metrics(
    per_modality: dict[FusionModality, tuple[ModalityPrediction, ...]],
    *,
    modalities: tuple[FusionModality, ...],
    test_idx: np.ndarray,
    frame: ModellingFrame,
    group_values: Sequence[str],
    labels: tuple[str, ...],
    task_type: TaskType,
    calibration_bins: int,
    expert_fold_metrics: dict[FusionModality, list[Any]],
) -> None:
    """Score each modality expert on the windows it actually predicted."""
    for modality in modalities:
        predictions = per_modality[modality]
        selected = [
            (position, prediction)
            for position, prediction in enumerate(predictions)
            if prediction.available
        ]
        truth = [frame.target_values[int(test_idx[p])] for p, _q in selected]
        groups = [str(group_values[int(test_idx[p])]) for p, _q in selected]
        if task_type is TaskType.CLASSIFICATION:
            probabilities = (
                np.asarray([q.probabilities for _p, q in selected], dtype=float)
                if selected
                else None
            )
            expert_fold_metrics[modality].append(
                classification_metrics(
                    y_true=[str(v) for v in truth],
                    y_predicted=[str(q.predicted_class) for _p, q in selected],
                    labels=labels,
                    group_ids=groups,
                    calibration=[
                        calibration_metrics(
                            label="fused",
                            probabilities=probabilities,
                            y_true=[str(v) for v in truth],
                            labels=labels,
                            bin_count=calibration_bins,
                        )
                    ],
                )
            )
        else:
            expert_fold_metrics[modality].append(
                regression_metrics(
                    y_true=[float(v) for v in truth],
                    y_predicted=[float(q.predicted_value or 0.0) for _p, q in selected],
                    group_ids=groups,
                )
            )


def _build_strategy_results(
    *,
    config: FusionRunConfiguration,
    outcomes: dict[tuple[FusionStrategy, str], list[_StrategyFold]],
    modalities: tuple[FusionModality, ...],
    task_type: TaskType,
    expert_model_name: str,
    validation_records: Sequence[ValidationWeightRecord],
    stacking_records: Sequence[StackingProvenanceRecord],
    failed_folds: dict[int, str],
    total_fold_count: int,
) -> list[FusionStrategyResult]:
    """Assemble one result per strategy from its reference-scenario folds."""
    results: list[FusionStrategyResult] = []
    classification = task_type is TaskType.CLASSIFICATION
    fields = (
        CLASSIFICATION_AGGREGATE_FIELDS
        if classification
        else REGRESSION_AGGREGATE_FIELDS
    )
    for strategy in config.fusion.strategies:
        entries = outcomes[(strategy, REFERENCE_SCENARIO.name)]
        fold_results: list[FusionFoldResult] = []
        metric_entries: list[Any] = []
        calibrations: list[tuple[Any, ...]] = []
        strategy_failures = dict(failed_folds)
        for entry in entries:
            metrics = entry.classification if classification else entry.regression
            evaluated = bool(metrics is not None and metrics.sample_count > 0)
            if not evaluated:
                strategy_failures[entry.fold_index] = (
                    "no window in this fold produced a fused prediction"
                )
            else:
                metric_entries.append(metrics)
                if classification and entry.classification is not None:
                    calibrations.append(entry.classification.calibration)
            fold_results.append(
                FusionFoldResult(
                    fold_index=entry.fold_index,
                    strategy=strategy,
                    scenario=entry.scenario,
                    evaluated=evaluated,
                    unavailable_reason=(
                        None
                        if evaluated
                        else "no window in this fold produced a fused prediction"
                    ),
                    classification_metrics=entry.classification,
                    regression_metrics=entry.regression,
                    diagnostics=entry.diagnostics,
                )
            )
        aggregate = list(
            aggregate_fold_metrics(
                metric_entries, fields, total_fold_count=total_fold_count
            )
        )
        if classification:
            aggregate.extend(
                aggregate_calibration_metrics(
                    calibrations, label="fused", total_fold_count=total_fold_count
                )
            )
        fusion_aggregate = aggregate_fusion_diagnostics(
            [entry.diagnostics for entry in entries],
            modalities,
            task_type,
            total_fold_count=total_fold_count,
        )
        notes = [
            "Metrics describe the windows this strategy produced a prediction "
            "for; fusion.coverage states what fraction of the evaluated "
            "windows that was.",
        ]
        if strategy is FusionStrategy.EARLY:
            notes.append(
                "Early fusion concatenates feature groups before a single "
                "estimator is fitted. It is a fusion architecture, not the "
                "Milestone 5 'all_available' ablation: modality membership, "
                "availability, quality, and missing-modality patterns are "
                "tracked and recorded here, and none of them was in the "
                "ablation."
            )
        results.append(
            FusionStrategyResult(
                strategy=strategy,
                description=STRATEGY_DESCRIPTIONS[strategy],
                modalities=modalities,
                expert_model_name=(
                    None if strategy is FusionStrategy.EARLY else expert_model_name
                ),
                calibrated_experts=(
                    config.fusion.use_calibrated_experts
                    and strategy is not FusionStrategy.STACKED_LATE
                    and classification
                ),
                folds=tuple(fold_results),
                aggregate=tuple(aggregate),
                fusion_aggregate=fusion_aggregate,
                valid_fold_count=len(metric_entries),
                total_fold_count=total_fold_count,
                failed_folds=strategy_failures,
                validation_weights=(
                    tuple(validation_records)
                    if strategy is FusionStrategy.VALIDATION_WEIGHTED_LATE
                    else ()
                ),
                stacking_provenance=(
                    tuple(stacking_records)
                    if strategy is FusionStrategy.STACKED_LATE
                    else ()
                ),
                notes=tuple(notes),
            )
        )
    return results


def _unimodal_control(
    expert_fold_metrics: dict[FusionModality, list[Any]],
    task_type: TaskType,
    *,
    total_fold_count: int,
) -> UnimodalControlResult | None:
    """The strongest single-modality expert per fold, as a descriptive control."""
    metric_name, higher_is_better = CONTROL_METRIC[task_type]
    per_fold_modality: dict[str, str] = {}
    per_fold_value: dict[str, float | None] = {}
    fold_values: list[float | None] = []

    fold_count = max((len(v) for v in expert_fold_metrics.values()), default=0)
    if not fold_count:
        return None
    for fold_index in range(fold_count):
        best_value: float | None = None
        best_modality: str | None = None
        for modality, entries in expert_fold_metrics.items():
            if fold_index >= len(entries):
                continue
            value = getattr(entries[fold_index], metric_name, None)
            if value is None:
                continue
            number = float(value)
            if (
                best_value is None
                or (higher_is_better and number > best_value)
                or (not higher_is_better and number < best_value)
            ):
                best_value = number
                best_modality = modality.value
        per_fold_value[str(fold_index)] = best_value
        fold_values.append(best_value)
        if best_modality is not None:
            per_fold_modality[str(fold_index)] = best_modality

    defined = [v for v in fold_values if v is not None]
    return UnimodalControlResult(
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        per_fold_modality=per_fold_modality,
        per_fold_value=per_fold_value,
        aggregate=AggregateMetric(
            name=f"best_unimodal_expert.{metric_name}",
            mean=float(np.mean(defined)) if defined else None,
            standard_deviation=float(np.std(defined, ddof=0)) if defined else None,
            valid_fold_count=len(defined),
            total_fold_count=total_fold_count,
            fold_values=tuple(fold_values),
            unavailable_reason=(
                None
                if defined
                else "no modality expert produced a defined metric in any fold"
            ),
        ),
    )


def _model_results(
    *,
    config: FusionRunConfiguration,
    outcomes: dict[tuple[FusionStrategy, str], list[_StrategyFold]],
    strategy_results: Sequence[FusionStrategyResult],
    expert_fold_metrics: dict[FusionModality, list[Any]],
    modalities: tuple[FusionModality, ...],
    task_type: TaskType,
    early_columns: tuple[str, ...],
    columns_by_modality: dict[FusionModality, tuple[str, ...]],
    expert_model_name: str,
    total_fold_count: int,
) -> list[ModelResult]:
    """Render fusion and unimodal results into the Milestone 5 metrics shape."""
    classification = task_type is TaskType.CLASSIFICATION
    fields = (
        CLASSIFICATION_AGGREGATE_FIELDS
        if classification
        else REGRESSION_AGGREGATE_FIELDS
    )
    results: list[ModelResult] = []
    for strategy_result in strategy_results:
        strategy = strategy_result.strategy
        entries = outcomes[(strategy, REFERENCE_SCENARIO.name)]
        fold_classification = tuple(
            entry.classification
            for entry in entries
            if entry.classification is not None and entry.classification.sample_count
        )
        fold_regression = tuple(
            entry.regression
            for entry in entries
            if entry.regression is not None and entry.regression.sample_count
        )
        predictor_columns = (
            early_columns
            if strategy is FusionStrategy.EARLY
            else tuple(
                column
                for modality in modalities
                for column in columns_by_modality[modality]
            )
        )
        results.append(
            ModelResult(
                model_name=strategy.value,
                model_kind="fusion",
                parameters={
                    "description": STRATEGY_DESCRIPTIONS[strategy],
                    "modalities": [m.value for m in modalities],
                    "expert_model": (
                        None if strategy is FusionStrategy.EARLY else expert_model_name
                    ),
                    "calibrated_experts": strategy_result.calibrated_experts,
                    "minimum_modalities": config.fusion.minimum_modalities,
                    "random_seed": config.random_seed,
                },
                predictor_columns=predictor_columns,
                fold_classification_metrics=fold_classification,
                fold_regression_metrics=fold_regression,
                aggregate=strategy_result.aggregate + strategy_result.fusion_aggregate,
                failed_folds=strategy_result.failed_folds,
                notes=strategy_result.notes,
            )
        )

    for modality in modalities:
        expert_entries = expert_fold_metrics[modality]
        defined = [e for e in expert_entries if e.sample_count]
        results.append(
            ModelResult(
                model_name=f"unimodal_{modality.value}",
                model_kind="unimodal_expert",
                parameters={
                    "description": (
                        f"Single-modality {modality.value} expert, evaluated on "
                        "the windows in which that modality contributed evidence."
                    ),
                    "expert_model": expert_model_name,
                    "random_seed": config.random_seed,
                },
                predictor_columns=columns_by_modality[modality],
                fold_classification_metrics=tuple(defined) if classification else (),
                fold_regression_metrics=() if classification else tuple(defined),
                aggregate=aggregate_fold_metrics(
                    defined, fields, total_fold_count=total_fold_count
                ),
                notes=(UNIMODAL_EXPERT_NOTE,),
            )
        )
    return results


def _robustness_document(
    *,
    run_id: str,
    config: FusionRunConfiguration,
    outcomes: dict[tuple[FusionStrategy, str], list[_StrategyFold]],
    scenarios: tuple[MissingModalityScenario, ...],
    modalities: tuple[FusionModality, ...],
    task_type: TaskType,
    total_fold_count: int,
    dropout_applied: bool,
    disclaimers: tuple[str, ...],
) -> RobustnessDocument:
    """Pool every strategy's behaviour under every missing-modality scenario."""
    classification = task_type is TaskType.CLASSIFICATION
    fields = (
        CLASSIFICATION_AGGREGATE_FIELDS
        if classification
        else REGRESSION_AGGREGATE_FIELDS
    )
    results: list[RobustnessResult] = []
    for scenario in scenarios:
        present = scenario.present(modalities)
        absent = tuple(m for m in modalities if m not in set(present))
        for strategy in config.fusion.strategies:
            if not present:
                results.append(
                    RobustnessResult(
                        scenario_name=scenario.name,
                        scenario_description=scenario.description,
                        strategy=strategy,
                        present_modalities=(),
                        absent_modalities=absent,
                        evaluated=False,
                        unavailable_reason=(
                            f"scenario {scenario.name!r} leaves no configured "
                            "modality present, so there is no evidence to "
                            "evaluate it against"
                        ),
                    )
                )
                continue
            entries = outcomes[(strategy, scenario.name)]
            metric_entries = [
                (entry.classification if classification else entry.regression)
                for entry in entries
            ]
            usable = [m for m in metric_entries if m is not None and m.sample_count]
            diagnostics = [entry.diagnostics for entry in entries]
            pooled = pool_diagnostics(diagnostics, modalities, task_type)
            evaluated_windows = sum(d.sample_count for d in diagnostics)
            fused_windows = sum(d.fused_count for d in diagnostics)
            results.append(
                RobustnessResult(
                    scenario_name=scenario.name,
                    scenario_description=scenario.description,
                    strategy=strategy,
                    present_modalities=present,
                    absent_modalities=absent,
                    evaluated=bool(entries),
                    unavailable_reason=(
                        None
                        if entries
                        else "no valid outer fold was evaluated for this scenario"
                    ),
                    evaluated_window_count=evaluated_windows,
                    fused_window_count=fused_windows,
                    unavailable_fusion_count=evaluated_windows - fused_windows,
                    coverage=(
                        float(fused_windows) / float(evaluated_windows)
                        if evaluated_windows
                        else None
                    ),
                    valid_fold_count=len(usable),
                    aggregate=aggregate_fold_metrics(
                        usable, fields, total_fold_count=total_fold_count
                    )
                    + aggregate_fusion_diagnostics(
                        diagnostics,
                        modalities,
                        task_type,
                        total_fold_count=total_fold_count,
                    ),
                    diagnostics=pooled,
                )
            )
    return RobustnessDocument(
        run_id=run_id,
        evaluation_mode=config.evaluation_mode,
        scientific_evaluation_eligible=(
            config.evaluation_mode is EvaluationMode.SCIENTIFIC
        ),
        target_name=config.target_name.value,
        task_type=task_type,
        synthetic_dropout_applied=dropout_applied,
        synthetic_dropout_seed=(
            config.fusion.robustness.synthetic_dropout_seed if dropout_applied else None
        ),
        synthetic_dropout_probability=(
            config.fusion.robustness.synthetic_dropout_probability
            if dropout_applied
            else None
        ),
        results=tuple(results),
        disclaimers=disclaimers,
    )


def _calibration_document(
    *,
    run_id: str,
    config: FusionRunConfiguration,
    expert_records: Sequence[ExpertRecord],
    disclaimers: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "requested_method": config.calibration_method.value,
        "calibration_group_fraction": config.calibration_group_fraction,
        "ece_bin_count": config.calibration_bins,
        "use_calibrated_experts": config.fusion.use_calibrated_experts,
        "placement": (
            "Calibration is applied PER EXPERT, BEFORE fusion, and to the "
            "early-fusion estimator. No post-fusion calibrator is fitted: "
            "calibrating twice would make the reported probability the output "
            "of two corrections with no way to attribute either. The stacked "
            "strategy consumes uncalibrated expert probabilities at both "
            "meta-training and meta-inference time, so that the meta-model is "
            "applied to the same input distribution it was fitted on."
        ),
        "design": (
            "Each expert's base estimator is fitted on the fold's fit groups; "
            "its calibrator is fitted on the fold's calibration groups, which "
            "are drawn from the training groups and are disjoint from them; "
            "the outer test groups are never used to fit anything."
        ),
        "note": (
            "A calibrated probability is not certainty and is not signal "
            "quality. Fusion weights derived from signal quality are a "
            "separate quantity and are recorded in separate fields."
        ),
        "experts": [
            {
                "modality": record.modality.value,
                "fold_index": record.fold_index,
                "trained": record.trained,
                "calibrated": record.calibrated,
                "method": record.calibration_method,
                "calibration_row_count": record.calibration_row_count,
                "calibration_group_count": record.calibration_group_count,
                "unavailable_reason": record.calibration_unavailable_reason,
            }
            for record in expert_records
        ],
        "disclaimers": list(disclaimers),
    }


def _manifest(
    *,
    run_id: str,
    config: FusionRunConfiguration,
    frame: ModellingFrame,
    splits: SplitManifest,
    early_columns: tuple[str, ...],
    expert_model_name: str,
    scenarios: tuple[MissingModalityScenario, ...],
    fingerprint: str,
    started: datetime,
    finished: datetime,
    status: RunStatus,
    failure_reason: str | None,
    disclaimers: tuple[str, ...],
) -> RunManifest:
    environment = runtime_environment()
    model_spec = get_model_spec(expert_model_name, frame.task_type)
    described = describe_parameters(
        build_pipeline(model_spec, early_columns, random_seed=config.random_seed)
    )
    described["imputation"] = model_spec.imputation.value
    described["standardised"] = model_spec.scale
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
        feature_set=early_columns,
        model_names=tuple(s.value for s in config.fusion.strategies),
        model_parameters={expert_model_name: described},
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
            "milestone": 6,
            "n_splits": config.n_splits,
            "random_seed": config.random_seed,
            "calibration_group_fraction": config.calibration_group_fraction,
            "calibration_bins": config.calibration_bins,
            "catalog_version": config.catalog_version,
            "split_manifest_fingerprint": fingerprint,
            "fusion": config.fusion.model_dump(mode="json"),
            "scenarios": [scenario.name for scenario in scenarios],
        },
        random_seed=config.random_seed,
        started_at_utc=started,
        finished_at_utc=finished,
        status=status,
        failure_reason=failure_reason,
        disclaimers=disclaimers,
    )


__all__ = [
    "CONTROL_METRIC",
    "UNIMODAL_EXPERT_NOTE",
    "FusionConfigurationError",
    "FusionRunConfiguration",
    "FusionRunResult",
    "run_fusion",
]
