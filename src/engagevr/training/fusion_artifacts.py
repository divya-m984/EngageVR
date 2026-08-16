"""Fusion run identity and the fusion Parquet artifacts.

The Milestone 5 experiment format is **extended, not replaced**: a fusion
run writes the same ``manifest.json``, ``dataset.json``,
``feature_catalog.json``, ``splits.json``, ``metrics.json``, and
``checksums.json``, and adds ``fusion_config.json``, ``experts.json``,
``fusion_metrics.json``, ``robustness.json``, ``expert_predictions.parquet``,
and ``fusion_weights.parquet``.

Run identity is deterministic.  Two equivalent configurations on the same
data produce the same ``run_id``; no wall clock and no random component
participates, so re-running a configuration reproduces its identifier
rather than accumulating near-duplicate directories.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import pyarrow as pa

from engagevr.schemas.experiments import SplitManifest
from engagevr.schemas.fusion import (
    FusionConfiguration,
    FusionPrediction,
    QualityWeightingConfiguration,
)
from engagevr.schemas.targets import TaskType
from engagevr.training.artifacts import engagevr_version

#: Artifacts a completed fusion run must contain.
FUSION_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "dataset.json",
    "feature_catalog.json",
    "splits.json",
    "fusion_config.json",
    "experts.json",
    "metrics.json",
    "fusion_metrics.json",
    "robustness.json",
)


def split_manifest_fingerprint(manifest: SplitManifest) -> str:
    """SHA-256 over the canonical rendering of a split manifest.

    The manifest carries no wall clock, so two runs producing the same folds
    fingerprint identically.  Pinning the fingerprint in the fusion run
    identity is what makes "these strategies were compared on exactly the
    same folds" checkable after the fact.
    """
    encoded = json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _quality_identity(quality: QualityWeightingConfiguration) -> dict[str, Any]:
    return {
        "enabled": quality.enabled,
        "missing_quality_policy": quality.missing_quality_policy.value,
        "missing_quality_fallback": quality.missing_quality_fallback,
        "minimum_quality": quality.minimum_quality,
        "minimum_effective_weight": quality.minimum_effective_weight,
        "base_weights": dict(sorted(quality.base_weights.items())),
    }


def build_fusion_run_id(
    *,
    target_name: str,
    task_type: str,
    evaluation_mode: str,
    dataset_fingerprint: str,
    split_manifest_fingerprint: str,
    random_seed: int,
    fusion: FusionConfiguration,
    calibration_method: str,
    scenario_names: Sequence[str],
) -> str:
    """Deterministic, collision-resistant identifier for a fusion run.

    The hash covers the dataset fingerprint, the target and task type, the
    seed, the split-manifest fingerprint, the enabled strategies, the
    modality groups, the expert model types, the calibration setting, the
    quality-weighting configuration, the missing-modality scenarios, and the
    EngageVR version.  It is insensitive to the *order* in which strategies,
    modalities, or scenarios were requested, because that does not change
    what was run.
    """
    payload = {
        "target_name": target_name,
        "task_type": task_type,
        "evaluation_mode": evaluation_mode,
        "dataset_fingerprint": dataset_fingerprint,
        "split_manifest_fingerprint": split_manifest_fingerprint,
        "random_seed": random_seed,
        "strategies": sorted(s.value for s in fusion.strategies),
        "modalities": sorted(m.value for m in fusion.modalities),
        "minimum_modalities": fusion.minimum_modalities,
        "expert_model_classification": fusion.expert_model_classification,
        "expert_model_regression": fusion.expert_model_regression,
        "use_calibrated_experts": fusion.use_calibrated_experts,
        "include_modality_quality_in_experts": (
            fusion.include_modality_quality_in_experts
        ),
        "include_modality_quality_in_early_fusion": (
            fusion.include_modality_quality_in_early_fusion
        ),
        "calibration_method": calibration_method,
        "quality": _quality_identity(fusion.quality),
        "stacking": {
            "enabled": fusion.stacking.enabled,
            "inner_folds": fusion.stacking.inner_folds,
            "meta_model_classification": fusion.stacking.meta_model_classification,
            "meta_model_regression": fusion.stacking.meta_model_regression,
        },
        "robustness": {
            "enabled": fusion.robustness.enabled,
            "synthetic_dropout_enabled": fusion.robustness.synthetic_dropout_enabled,
            "synthetic_dropout_seed": fusion.robustness.synthetic_dropout_seed,
            "synthetic_dropout_probability": (
                fusion.robustness.synthetic_dropout_probability
            ),
        },
        "scenarios": sorted(scenario_names),
        "engagevr_version": engagevr_version(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    short_mode = "sci" if evaluation_mode == "scientific" else "selfcheck"
    return f"{target_name}-fusion-{short_mode}-{digest}"


def fusion_predictions_table(
    predictions: Sequence[FusionPrediction],
    labels: Sequence[str],
    task_type: TaskType,
    true_values: Sequence[str | float | None],
) -> pa.Table:
    """Per-window fused predictions for every strategy and scenario."""
    classification = task_type is TaskType.CLASSIFICATION
    columns: dict[str, list[Any]] = {
        "window_id": [p.window_id for p in predictions],
        "subject_id": [p.subject_id for p in predictions],
        "session_id": [p.session_id for p in predictions],
        "fold_index": [int(p.fold_index) for p in predictions],
        "strategy": [p.strategy.value for p in predictions],
        "scenario": [p.scenario for p in predictions],
        "fused": [bool(p.fused) for p in predictions],
        "unavailable_reason": [p.unavailable_reason for p in predictions],
        "available_expert_count": [len(p.available_modalities) for p in predictions],
        "available_modalities": [
            "+".join(m.value for m in p.available_modalities) or "none"
            for p in predictions
        ],
        "unavailable_modalities": [
            "+".join(m.value for m in p.unavailable_modalities) or "none"
            for p in predictions
        ],
        "data_source": [p.data_source for p in predictions],
        "is_synthetic": [bool(p.is_synthetic) for p in predictions],
        "scientific_evaluation_eligible": [
            bool(p.scientific_evaluation_eligible) for p in predictions
        ],
    }
    if classification:
        columns["true_class"] = [
            None if value is None else str(value) for value in true_values
        ]
        columns["predicted_class"] = [p.predicted_class for p in predictions]
        for index, label in enumerate(labels):
            columns[f"probability__{label}"] = [
                float(p.probabilities[index]) if p.probabilities else None
                for p in predictions
            ]
    else:
        columns["true_value"] = [
            None if value is None else float(value) for value in true_values
        ]
        columns["predicted_value"] = [p.predicted_value for p in predictions]
    return pa.table(columns)


def expert_predictions_table(
    predictions: Sequence[FusionPrediction],
    labels: Sequence[str],
    task_type: TaskType,
) -> pa.Table:
    """Per-modality expert outputs, one row per (window, modality).

    Written for the reference scenario only.  A missing-modality scenario
    does not change what an expert computed; it changes which experts were
    allowed to contribute, and that is recorded in ``fusion_weights.parquet``
    for every scenario.
    """
    classification = task_type is TaskType.CLASSIFICATION
    window: list[str] = []
    subject: list[str] = []
    session: list[str] = []
    fold: list[int] = []
    strategy: list[str] = []
    scenario: list[str] = []
    modality: list[str] = []
    available: list[bool] = []
    reason: list[str | None] = []
    predicted_class: list[str | None] = []
    predicted_value: list[float | None] = []
    calibrated: list[bool] = []
    quality: list[float | None] = []
    probability_columns: dict[str, list[float | None]] = {
        f"probability__{label}": [] for label in labels
    }

    for prediction in predictions:
        for expert in prediction.modality_predictions:
            window.append(prediction.window_id)
            subject.append(prediction.subject_id)
            session.append(prediction.session_id)
            fold.append(int(prediction.fold_index))
            strategy.append(prediction.strategy.value)
            scenario.append(prediction.scenario)
            modality.append(expert.modality.value)
            available.append(bool(expert.available))
            reason.append(expert.unavailable_reason)
            predicted_class.append(expert.predicted_class)
            predicted_value.append(expert.predicted_value)
            calibrated.append(bool(expert.probabilities_are_calibrated))
            quality.append(expert.quality)
            for index, label in enumerate(labels):
                probability_columns[f"probability__{label}"].append(
                    float(expert.probabilities[index]) if expert.probabilities else None
                )

    columns: dict[str, list[Any]] = {
        "window_id": window,
        "subject_id": subject,
        "session_id": session,
        "fold_index": fold,
        "strategy": strategy,
        "scenario": scenario,
        "modality": modality,
        "available": available,
        "unavailable_reason": reason,
        "modality_quality": quality,
        "probabilities_are_calibrated": calibrated,
    }
    if classification:
        columns["predicted_class"] = predicted_class
        columns.update(probability_columns)
    else:
        columns["predicted_value"] = predicted_value
    return pa.table(columns)


def fusion_weights_table(predictions: Sequence[FusionPrediction]) -> pa.Table:
    """Per-modality fusion weights, one row per (window, strategy, modality).

    Raw and normalised weights are stored side by side with the quality
    value that produced them and where that value came from, so the
    arithmetic is reconstructible without re-running anything.
    """
    columns: dict[str, list[Any]] = {
        "window_id": [],
        "subject_id": [],
        "session_id": [],
        "fold_index": [],
        "strategy": [],
        "scenario": [],
        "modality": [],
        "base_weight": [],
        "availability": [],
        "quality_used": [],
        "quality_source": [],
        "normalized_quality": [],
        "raw_effective_weight": [],
        "normalized_weight": [],
        "contributed": [],
        "exclusion_reason": [],
    }
    for prediction in predictions:
        for weight in prediction.fusion_weights:
            columns["window_id"].append(prediction.window_id)
            columns["subject_id"].append(prediction.subject_id)
            columns["session_id"].append(prediction.session_id)
            columns["fold_index"].append(int(prediction.fold_index))
            columns["strategy"].append(prediction.strategy.value)
            columns["scenario"].append(prediction.scenario)
            columns["modality"].append(weight.modality.value)
            columns["base_weight"].append(float(weight.base_weight))
            columns["availability"].append(float(weight.availability))
            columns["quality_used"].append(weight.quality_used)
            columns["quality_source"].append(weight.quality_source.value)
            columns["normalized_quality"].append(weight.normalized_quality)
            columns["raw_effective_weight"].append(float(weight.raw_effective_weight))
            columns["normalized_weight"].append(float(weight.normalized_weight))
            columns["contributed"].append(bool(weight.contributed))
            columns["exclusion_reason"].append(weight.exclusion_reason)
    return pa.table(columns)


__all__ = [
    "FUSION_REQUIRED_ARTIFACTS",
    "build_fusion_run_id",
    "expert_predictions_table",
    "fusion_predictions_table",
    "fusion_weights_table",
    "split_manifest_fingerprint",
]
