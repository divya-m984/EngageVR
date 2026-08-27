"""Minimal run directories for the Milestone 9 dashboard tests.

Every fixture writes the documents a real runner writes, cut down to the
fields the dashboard reads.  Building them here rather than depending on
``artifacts/`` means the tests need no prior milestone to have been run,
no dataset, and no model file.

All fixture data is obviously synthetic.  Subject identifiers are of the
form ``synthetic-subject-0001``; there is no name, no email address, no
image, and nothing that could be mistaken for a real participant.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from engagevr.dashboard.catalogue import build_catalogue
from engagevr.schemas.dashboard import DashboardRunSummary
from engagevr.schemas.experiments import SELF_CHECK_DISCLAIMER

#: Attached to every fixture document, as a real run does.
DISCLAIMERS: list[str] = [SELF_CHECK_DISCLAIMER]

#: Pseudonymous, obviously software-generated subject labels.
SUBJECTS: tuple[str, ...] = (
    "synthetic-subject-0001",
    "synthetic-subject-0002",
    "synthetic-subject-0003",
)

#: The three ordinal classes used throughout this repository.
CLASSES: tuple[str, ...] = ("low", "medium", "high")


def write_json(path: Path, document: Mapping[str, Any]) -> Path:
    """Write one JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def write_parquet(path: Path, columns: Mapping[str, Sequence[Any]]) -> Path:
    """Write one Parquet artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(dict(columns)), path)
    return path


def dataset_document(
    *, data_source: str = "synthetic", eligible: bool = False
) -> dict[str, Any]:
    """A ``dataset.json`` carrying the fields the dashboard reads."""
    return {
        "dataset_schema_version": "1.0",
        "feature_catalog_version": "1.0",
        "row_count": 60,
        "feature_count": 4,
        "column_order": [
            "window_id",
            "session_id",
            "subject_id",
            "data_source",
            "modality_available__behavioural",
            "modality_quality__behavioural",
            "modality_available__task",
            "modality_quality__task",
        ],
        "subject_count": len(SUBJECTS),
        "session_count": len(SUBJECTS) * 2,
        "data_source_counts": {data_source: 60},
        "window_duration_seconds": 10.0,
        "window_step_seconds": 10.0,
        "windows_overlap": False,
        "missingness": [
            {"feature_name": "eye_openness", "missing_count": 6, "missing_pct": 10.0},
            {"feature_name": "task_accuracy", "missing_count": 0, "missing_pct": 0.0},
        ],
        "overall_missing_pct": 5.0,
        "targets": [
            {
                "target_name": "engagement_class",
                "task_type": "classification",
                "labelled_row_count": 60,
                "class_distribution": {"low": 20, "medium": 20, "high": 20},
            }
        ],
        "random_seed": 42,
        "dataset_fingerprint": "f" * 64,
        "fingerprint_algorithm": "sha256",
        "created_at_utc": "2026-01-01T00:00:00Z",
        "scientific_evaluation_eligible": eligible,
        "disclaimers": DISCLAIMERS,
    }


def splits_document() -> dict[str, Any]:
    """A ``splits.json`` with two grouped folds and a passing audit."""
    return {
        "strategy": "stratified_group_k_fold",
        "strategy_reason": "windows share a subject",
        "group_field": "subject_id",
        "group_field_reason": "one subject contributes many windows",
        "group_count": len(SUBJECTS),
        "n_splits": 2,
        "random_seed": 42,
        "calibration_group_fraction": 0.0,
        "folds": [
            {
                "fold_index": 0,
                "train_groups": [SUBJECTS[0], SUBJECTS[1]],
                "calibration_groups": [],
                "test_groups": [SUBJECTS[2]],
                "train_row_count": 40,
                "calibration_row_count": 0,
                "test_row_count": 20,
                "valid": True,
                "invalid_reason": None,
                "warnings": [],
            },
            {
                "fold_index": 1,
                "train_groups": [SUBJECTS[1], SUBJECTS[2]],
                "calibration_groups": [],
                "test_groups": [SUBJECTS[0]],
                "train_row_count": 40,
                "calibration_row_count": 0,
                "test_row_count": 20,
                "valid": True,
                "invalid_reason": None,
                "warnings": [],
            },
        ],
        "audit_passed": True,
        "audit_notes": [],
    }


def manifest_document(
    *,
    run_id: str,
    target_name: str = "engagement_class",
    task_type: str = "classification",
    status: str = "completed",
    failure_reason: str | None = None,
    eligible: bool = False,
    evaluation_mode: str = "software_self_check",
    configuration: Mapping[str, Any] | None = None,
    feature_catalog_version: str = "1.0",
) -> dict[str, Any]:
    """A ``manifest.json`` carrying the fields the catalogue reads."""
    return {
        "run_id": run_id,
        "engagevr_version": "0.1.0",
        "python_version": "3.12.13",
        "dependency_versions": {"numpy": "2.5.1"},
        "evaluation_mode": evaluation_mode,
        "scientific_evaluation_eligible": eligible,
        "dataset_path": "artifacts/datasets/fixture.parquet",
        "dataset_fingerprint": "f" * 64,
        "feature_catalog_version": feature_catalog_version,
        "target_name": target_name,
        "task_type": task_type,
        "feature_set": ["feat__eye_openness", "feat__task_accuracy"],
        "model_names": ["logistic_regression"],
        "model_parameters": {},
        "split_strategy": "stratified_group_k_fold",
        "group_field": "subject_id",
        "group_count": len(SUBJECTS),
        "fold_count": 2,
        "fold_assignments": {},
        "calibration_method": "sigmoid",
        "configuration": dict(
            configuration or {"split_manifest_fingerprint": "a" * 64}
        ),
        "random_seed": 42,
        "started_at_utc": "2026-01-01T00:00:00Z",
        "finished_at_utc": "2026-01-01T00:01:00Z",
        "status": status,
        "failure_reason": failure_reason,
        "artifact_checksums": {},
        "disclaimers": DISCLAIMERS,
    }


def classification_metrics(
    *, run_id: str, model_name: str = "logistic_regression"
) -> dict[str, Any]:
    """A ``metrics.json`` for a classification run."""
    return {
        "run_id": run_id,
        "evaluation_mode": "software_self_check",
        "scientific_evaluation_eligible": False,
        "target_name": "engagement_class",
        "task_type": "classification",
        "dataset_fingerprint": "f" * 64,
        "group_field": "subject_id",
        "group_count": len(SUBJECTS),
        "fold_count": 2,
        "random_seed": 42,
        "results": [
            {
                "model_name": model_name,
                "model_kind": "linear",
                "is_software_check_baseline": False,
                "parameters": {},
                "predictor_columns": [],
                "fold_classification_metrics": [
                    {
                        "sample_count": 20,
                        "independent_group_count": 1,
                        "class_support": {"low": 7, "medium": 7, "high": 6},
                        "accuracy": 0.6,
                        "balanced_accuracy": 0.58,
                        "macro_precision": 0.57,
                        "macro_recall": 0.58,
                        "macro_f1": 0.56,
                        "weighted_f1": 0.57,
                        "per_class": [
                            {
                                "label": "low",
                                "support": 7,
                                "precision": 0.6,
                                "recall": 0.7,
                                "f1": 0.65,
                            }
                        ],
                        "confusion_matrix": {
                            "labels": list(CLASSES),
                            "rows_are": "true_label",
                            "columns_are": "predicted_label",
                            "counts": [[5, 1, 1], [2, 4, 1], [1, 2, 3]],
                        },
                        "calibration": [
                            {
                                "label": "sigmoid",
                                "sample_count": 20,
                                "brier_score": 0.42,
                                "log_loss": 0.9,
                                "expected_calibration_error": 0.07,
                                "ece_bin_count": 2,
                                "bins": [
                                    {
                                        "bin_index": 0,
                                        "lower_edge": 0.0,
                                        "upper_edge": 0.5,
                                        "count": 8,
                                        "mean_confidence": 0.42,
                                        "empirical_accuracy": 0.5,
                                    },
                                    {
                                        "bin_index": 1,
                                        "lower_edge": 0.5,
                                        "upper_edge": 1.0,
                                        "count": 12,
                                        "mean_confidence": 0.71,
                                        "empirical_accuracy": 0.66,
                                    },
                                ],
                                "unavailable_reason": None,
                            }
                        ],
                        "unavailable_metrics": {},
                    }
                ],
                "fold_regression_metrics": [],
                "aggregate": [
                    {
                        "name": "accuracy",
                        "aggregation": "unweighted_mean_over_valid_folds",
                        "mean": 0.6,
                        "standard_deviation": 0.02,
                        "valid_fold_count": 2,
                        "total_fold_count": 2,
                        "fold_values": [0.58, 0.62],
                        "unavailable_reason": None,
                    },
                    {
                        "name": "macro_f1",
                        "aggregation": "unweighted_mean_over_valid_folds",
                        "mean": 0.56,
                        "standard_deviation": 0.01,
                        "valid_fold_count": 2,
                        "total_fold_count": 2,
                        "fold_values": [0.55, 0.57],
                        "unavailable_reason": None,
                    },
                    {
                        "name": "balanced_accuracy",
                        "aggregation": "unweighted_mean_over_valid_folds",
                        "mean": None,
                        "standard_deviation": None,
                        "valid_fold_count": 0,
                        "total_fold_count": 2,
                        "fold_values": [None, None],
                        "unavailable_reason": "no fold produced a defined value",
                    },
                    {
                        "name": "sigmoid.brier_score",
                        "aggregation": "unweighted_mean_over_valid_folds",
                        "mean": 0.42,
                        "standard_deviation": 0.0,
                        "valid_fold_count": 2,
                        "total_fold_count": 2,
                        "fold_values": [0.42, 0.42],
                        "unavailable_reason": None,
                    },
                ],
                "aggregate_confusion_matrix": {
                    "labels": list(CLASSES),
                    "rows_are": "true_label",
                    "columns_are": "predicted_label",
                    "counts": [[10, 2, 2], [4, 8, 2], [2, 4, 6]],
                },
                "failed_folds": {},
                "notes": [],
            }
        ],
        "disclaimers": DISCLAIMERS,
    }


def regression_metrics(*, run_id: str) -> dict[str, Any]:
    """A ``metrics.json`` for a regression run."""
    return {
        "run_id": run_id,
        "evaluation_mode": "software_self_check",
        "scientific_evaluation_eligible": False,
        "target_name": "engagement_score",
        "task_type": "regression",
        "dataset_fingerprint": "f" * 64,
        "group_field": "subject_id",
        "group_count": len(SUBJECTS),
        "fold_count": 2,
        "random_seed": 42,
        "results": [
            {
                "model_name": "ridge",
                "model_kind": "linear",
                "is_software_check_baseline": False,
                "parameters": {},
                "predictor_columns": [],
                "fold_classification_metrics": [],
                "fold_regression_metrics": [
                    {
                        "sample_count": 20,
                        "independent_group_count": 1,
                        "mean_absolute_error": 0.12,
                        "root_mean_squared_error": 0.15,
                        "median_absolute_error": 0.09,
                        "r_squared": 0.55,
                        "target_mean": 0.44,
                        "target_std": 0.22,
                        "unavailable_metrics": {},
                    }
                ],
                "aggregate": [
                    {
                        "name": "mean_absolute_error",
                        "aggregation": "unweighted_mean_over_valid_folds",
                        "mean": 0.12,
                        "standard_deviation": 0.01,
                        "valid_fold_count": 2,
                        "total_fold_count": 2,
                        "fold_values": [0.11, 0.13],
                        "unavailable_reason": None,
                    },
                    {
                        "name": "r_squared",
                        "aggregation": "unweighted_mean_over_valid_folds",
                        "mean": None,
                        "standard_deviation": None,
                        "valid_fold_count": 0,
                        "total_fold_count": 2,
                        "fold_values": [None, None],
                        "unavailable_reason": "variance of the target was zero",
                    },
                ],
                "aggregate_confusion_matrix": None,
                "failed_folds": {},
                "notes": [],
            }
        ],
        "disclaimers": DISCLAIMERS,
    }


def make_baseline_run(
    root: Path,
    name: str = "fixture-baseline",
    *,
    task_type: str = "classification",
    status: str = "completed",
    failure_reason: str | None = None,
    with_predictions: bool = True,
    with_checksums: bool = True,
    feature_catalog_version: str = "1.0",
    data_source: str = "synthetic",
    evaluation_mode: str = "software_self_check",
    eligible: bool = False,
) -> Path:
    """A Milestone 5 baseline run directory.

    ``data_source`` names one of the project's own
    :class:`~engagevr.schemas.session.DataSource` members, so a test can
    build the ``public_dataset`` provenance no run in this repository
    currently has.  A non-synthetic run needs an ``evaluation_mode``
    other than ``software_self_check``, because the catalogue reads that
    field as the synthetic statement for the training families.
    ``eligible`` stays false by default: this repository has validated
    nothing, and a public corpus does not change that.
    """
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    run_id = f"{name}-selfcheck-0001"
    write_json(directory / "dataset.json", dataset_document(data_source=data_source))
    write_json(directory / "feature_catalog.json", {"version": "1.0", "features": []})
    write_json(directory / "splits.json", splits_document())
    if task_type == "classification":
        write_json(directory / "metrics.json", classification_metrics(run_id=run_id))
        target = "engagement_class"
    else:
        write_json(directory / "metrics.json", regression_metrics(run_id=run_id))
        target = "engagement_score"
    if with_predictions:
        write_parquet(
            directory / "predictions.parquet", _predictions(task_type=task_type)
        )
    write_json(
        directory / "manifest.json",
        manifest_document(
            run_id=run_id,
            target_name=target,
            task_type=task_type,
            status=status,
            failure_reason=failure_reason,
            feature_catalog_version=feature_catalog_version,
            evaluation_mode=evaluation_mode,
            eligible=eligible,
        ),
    )
    if with_checksums:
        write_checksums(directory)
    return directory


def _predictions(*, task_type: str) -> dict[str, list[Any]]:
    windows = [f"synthetic-session-000{i // 4 + 1}:w{i:06d}" for i in range(12)]
    subjects = [SUBJECTS[i % len(SUBJECTS)] for i in range(12)]
    sessions = [f"synthetic-session-000{i // 4 + 1}" for i in range(12)]
    if task_type == "classification":
        truth = [CLASSES[i % 3] for i in range(12)]
        predicted = [CLASSES[(i + 1) % 3] for i in range(12)]
        columns: dict[str, list[Any]] = {
            "window_id": windows,
            "subject_id": subjects,
            "session_id": sessions,
            "fold_index": [i % 2 for i in range(12)],
            "model_name": ["logistic_regression"] * 12,
            "true_value": truth,
            "predicted_value": predicted,
        }
        for index, label in enumerate(CLASSES):
            columns[f"probability_calibrated__{label}"] = [
                0.2 + 0.1 * ((i + index) % 5) for i in range(12)
            ]
        return columns
    observed = [0.1 * i for i in range(12)]
    predicted_values: list[float | None] = [0.1 * i + 0.02 for i in range(12)]
    predicted_values[3] = None
    return {
        "window_id": windows,
        "subject_id": subjects,
        "session_id": sessions,
        "fold_index": [i % 2 for i in range(12)],
        "model_name": ["ridge"] * 12,
        "true_value": observed,
        "predicted_value": predicted_values,
    }


def make_fusion_run(root: Path, name: str = "fixture-fusion") -> Path:
    """A Milestone 6 fusion run directory."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    run_id = f"{name}-fusion-selfcheck-0001"
    write_json(directory / "dataset.json", dataset_document())
    write_json(directory / "feature_catalog.json", {"version": "1.0", "features": []})
    write_json(directory / "splits.json", splits_document())
    write_json(directory / "metrics.json", classification_metrics(run_id=run_id))
    write_json(
        directory / "fusion_config.json",
        {"run_id": run_id, "fusion": {"strategies": ["early", "quality_late"]}},
    )
    write_json(
        directory / "experts.json",
        {
            "run_id": run_id,
            "evaluation_mode": "software_self_check",
            "target_name": "engagement_class",
            "task_type": "classification",
            "experts": [
                {
                    "modality": "behavioural",
                    "fold_index": 0,
                    "model_name": "logistic_regression",
                    "trained": True,
                    "unavailable_reason": None,
                    "calibrated": True,
                    "calibration_method": "sigmoid",
                    "fit_row_count": 40,
                    "fit_group_count": 2,
                },
                {
                    "modality": "task",
                    "fold_index": 0,
                    "model_name": "logistic_regression",
                    "trained": False,
                    "unavailable_reason": "no usable window for this modality",
                    "calibrated": False,
                    "calibration_method": None,
                    "fit_row_count": 0,
                    "fit_group_count": 0,
                },
            ],
            "disclaimers": DISCLAIMERS,
        },
    )
    write_json(directory / "fusion_metrics.json", fusion_metrics(run_id=run_id))
    write_json(directory / "robustness.json", robustness_document(run_id=run_id))
    write_json(
        directory / "manifest.json",
        manifest_document(
            run_id=run_id,
            configuration={"milestone": 6, "split_manifest_fingerprint": "a" * 64},
        ),
    )
    write_checksums(directory)
    return directory


def fusion_metrics(*, run_id: str) -> dict[str, Any]:
    """A ``fusion_metrics.json`` with weights and disagreement diagnostics."""

    def aggregate(name: str, mean: float | None) -> dict[str, Any]:
        return {
            "name": name,
            "aggregation": "unweighted_mean_over_valid_folds",
            "mean": mean,
            "standard_deviation": 0.01 if mean is not None else None,
            "valid_fold_count": 2 if mean is not None else 0,
            "total_fold_count": 2,
            "fold_values": [mean, mean],
            "unavailable_reason": None if mean is not None else "not computable",
        }

    return {
        "run_id": run_id,
        "evaluation_mode": "software_self_check",
        "scientific_evaluation_eligible": False,
        "target_name": "engagement_class",
        "task_type": "classification",
        "dataset_fingerprint": "f" * 64,
        "split_manifest_fingerprint": "a" * 64,
        "group_field": "subject_id",
        "group_count": len(SUBJECTS),
        "fold_count": 2,
        "random_seed": 42,
        "modalities": ["behavioural", "task"],
        "strategies": [
            {
                "strategy": "early",
                "description": "early feature fusion",
                "modalities": ["behavioural", "task"],
                "expert_model_name": None,
                "calibrated_experts": True,
                "folds": [],
                "aggregate": [
                    aggregate("accuracy", 0.61),
                    aggregate("balanced_accuracy", 0.6),
                    aggregate("macro_f1", 0.58),
                    aggregate("weighted_f1", 0.59),
                ],
                "fusion_aggregate": [
                    aggregate("fusion.coverage", 1.0),
                    aggregate("fusion.mean_normalized_weight.behavioural", 0.55),
                    aggregate("fusion.mean_normalized_weight.task", 0.45),
                    aggregate("fusion.missing_modality_rate.behavioural", 0.1),
                    aggregate("fusion.missing_modality_rate.task", 0.05),
                    aggregate("fusion.disagreement.unanimous_fraction", 0.7),
                    aggregate("fusion.disagreement.disagreement_fraction", 0.3),
                ],
                "valid_fold_count": 2,
                "total_fold_count": 2,
                "failed_folds": {},
                "validation_weights": [],
                "stacking_provenance": [],
                "notes": [],
            }
        ],
        "unimodal_control": {},
        "disclaimers": DISCLAIMERS,
        "comparison_note": "identical grouped outer folds",
    }


def robustness_document(*, run_id: str) -> dict[str, Any]:
    """A ``robustness.json`` with two missing-modality scenarios."""

    def scenario(name: str, absent: list[str], metric: float | None) -> dict[str, Any]:
        return {
            "scenario_name": name,
            "scenario_description": f"scenario {name}",
            "strategy": "early",
            "present_modalities": [
                m for m in ("behavioural", "task") if m not in absent
            ],
            "absent_modalities": absent,
            "evaluated": metric is not None,
            "unavailable_reason": None if metric is not None else "no expert available",
            "evaluated_window_count": 60,
            "fused_window_count": 60 if metric is not None else 0,
            "unavailable_fusion_count": 0 if metric is not None else 60,
            "coverage": 1.0 if metric is not None else 0.0,
            "valid_fold_count": 2,
            "aggregate": [
                {
                    "name": "accuracy",
                    "aggregation": "unweighted_mean_over_valid_folds",
                    "mean": metric,
                    "standard_deviation": None,
                    "valid_fold_count": 2 if metric is not None else 0,
                    "total_fold_count": 2,
                    "fold_values": [metric, metric],
                    "unavailable_reason": None if metric is not None else "no fusion",
                },
                {
                    "name": "fusion.coverage",
                    "aggregation": "unweighted_mean_over_valid_folds",
                    "mean": 1.0 if metric is not None else 0.0,
                    "standard_deviation": None,
                    "valid_fold_count": 2,
                    "total_fold_count": 2,
                    "fold_values": [1.0, 1.0],
                    "unavailable_reason": None,
                },
            ],
            "diagnostics": {},
        }

    return {
        "run_id": run_id,
        "evaluation_mode": "software_self_check",
        "scientific_evaluation_eligible": False,
        "target_name": "engagement_class",
        "task_type": "classification",
        "synthetic_dropout_applied": False,
        "results": [
            scenario("all_modalities", [], 0.61),
            scenario("missing_task", ["task"], 0.55),
            scenario("only_task", ["behavioural"], None),
        ],
        "disclaimers": DISCLAIMERS,
    }


def make_personalization_run(root: Path, name: str = "fixture-personalization") -> Path:
    """A Milestone 6 personalization run whose personalized arm is worse."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    run_id = f"{name}-personalization-selfcheck-0001"
    write_json(directory / "dataset.json", dataset_document())
    write_json(directory / "feature_catalog.json", {"version": "1.0", "features": []})
    write_json(directory / "splits.json", splits_document())
    write_json(directory / "metrics.json", classification_metrics(run_id=run_id))
    write_json(
        directory / "personalization_config.json",
        {"run_id": run_id, "personalization": {"calibration_windows": 5}},
    )
    write_json(
        directory / "personal_baselines.json", {"run_id": run_id, "subjects": []}
    )
    write_json(
        directory / "personalization.json", personalization_document(run_id=run_id)
    )
    write_json(
        directory / "manifest.json",
        manifest_document(
            run_id=run_id,
            configuration={
                "kind": "personalization",
                "split_manifest_fingerprint": "a" * 64,
            },
        ),
    )
    write_checksums(directory)
    return directory


def personalization_document(*, run_id: str) -> dict[str, Any]:
    """A ``personalization.json`` whose personalized arm scores lower."""

    def aggregate(name: str, mean: float) -> dict[str, Any]:
        return {
            "name": name,
            "aggregation": "unweighted_mean_over_valid_folds",
            "mean": mean,
            "standard_deviation": 0.02,
            "valid_fold_count": 2,
            "total_fold_count": 2,
            "fold_values": [mean, mean],
            "unavailable_reason": None,
        }

    return {
        "run_id": run_id,
        "evaluation_mode": "software_self_check",
        "scientific_evaluation_eligible": False,
        "target_name": "engagement_class",
        "task_type": "classification",
        "dataset_fingerprint": "f" * 64,
        "group_field": "subject_id",
        "group_count": len(SUBJECTS),
        "fold_count": 2,
        "random_seed": 42,
        "folds": [
            {
                "fold_index": 0,
                "evaluated": True,
                "unavailable_reason": None,
                "population_training_subject_count": 2,
                "held_out_subject_count": 1,
                "evaluated_subject_count": 1,
                "personalized_subject_count": 1,
                "cold_start_subject_count": 0,
                "unavailable_subject_count": 0,
                "calibration_window_count": 5,
                "evaluation_window_count": 20,
                "excluded_overlap_window_count": 0,
                "corrections": [
                    {
                        "subject_id": SUBJECTS[2],
                        "applied": True,
                        "cold_start": False,
                        "calibration_sample_count": 5,
                        "unavailable_reason": None,
                    }
                ],
            },
            {
                "fold_index": 1,
                "evaluated": True,
                "unavailable_reason": None,
                "population_training_subject_count": 2,
                "held_out_subject_count": 1,
                "evaluated_subject_count": 1,
                "personalized_subject_count": 0,
                "cold_start_subject_count": 1,
                "unavailable_subject_count": 0,
                "calibration_window_count": 2,
                "evaluation_window_count": 20,
                "excluded_overlap_window_count": 0,
                "corrections": [
                    {
                        "subject_id": SUBJECTS[0],
                        "applied": False,
                        "cold_start": True,
                        "calibration_sample_count": 2,
                        "unavailable_reason": "too few calibration windows",
                    }
                ],
            },
        ],
        "population_aggregate": [
            aggregate("accuracy", 0.67),
            aggregate("macro_f1", 0.6),
        ],
        "personalized_aggregate": [
            aggregate("accuracy", 0.51),
            aggregate("macro_f1", 0.44),
        ],
        "total_calibration_window_count": 7,
        "total_evaluation_window_count": 40,
        "total_excluded_overlap_window_count": 0,
        "cold_start_subject_count": 1,
        "personalized_subject_count": 1,
        "unavailable_personalization_count": 0,
        "personalization_coverage": 0.5,
        "disclaimers": DISCLAIMERS,
    }


def make_uncertainty_run(
    root: Path,
    name: str = "fixture-uncertainty",
    *,
    task_type: str = "classification",
    record_axis: bool = True,
    reconciling: bool = True,
    with_coverage_curve: bool = True,
) -> Path:
    """A Milestone 7 uncertainty run directory."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    run_id = f"{name}-uncertainty-selfcheck-0001"
    target = "engagement_class" if task_type == "classification" else "engagement_score"
    write_json(directory / "dataset.json", dataset_document())
    write_json(directory / "feature_catalog.json", {"version": "1.0", "features": []})
    write_json(directory / "splits.json", splits_document())
    write_json(
        directory / "metrics.json",
        classification_metrics(run_id=run_id)
        if task_type == "classification"
        else regression_metrics(run_id=run_id),
    )
    write_json(
        directory / "uncertainty_config.json",
        {"run_id": run_id, "selective_prediction": {"alpha": 0.1}},
    )
    write_json(
        directory / "uncertainty.json",
        uncertainty_document(
            run_id=run_id, task_type=task_type, reconciling=reconciling
        ),
    )
    write_json(
        directory / "thresholds.json",
        {
            "run_id": run_id,
            "population_confidence_threshold": 0.7,
            "population_threshold_provenance": "an engineering default",
            "acceptance_rule": "accept if score >= tau",
            "estimation_enabled": False,
            "personalized_thresholds_enabled": True,
            "personalized_threshold_rule": "shrunken subject quantile",
            "disclaimers": DISCLAIMERS,
        },
    )
    write_json(
        directory / "selective_metrics.json",
        {"run_id": run_id, "coverage": 0.6, "disclaimers": DISCLAIMERS},
    )
    if with_coverage_curve:
        write_json(
            directory / "coverage_curve.json",
            coverage_curve_document(
                run_id=run_id, task_type=task_type, record_axis=record_axis
            ),
        )
    write_parquet(
        directory / "selective_predictions.parquet",
        _selective_predictions(task_type=task_type),
    )
    write_parquet(
        directory / "predictions.parquet", _unselected_predictions(task_type=task_type)
    )
    write_json(
        directory / "manifest.json",
        manifest_document(run_id=run_id, target_name=target, task_type=task_type),
    )
    write_checksums(directory)
    return directory


def uncertainty_document(
    *, run_id: str, task_type: str, reconciling: bool
) -> dict[str, Any]:
    """An ``uncertainty.json`` with the selective counts the page reads."""
    accepted = 6
    abstained = 3 if reconciling else 4
    return {
        "run_id": run_id,
        "evaluation_mode": "software_self_check",
        "scientific_evaluation_eligible": False,
        "target_name": (
            "engagement_class" if task_type == "classification" else "engagement_score"
        ),
        "task_type": task_type,
        "dataset_fingerprint": "f" * 64,
        "group_field": "subject_id",
        "group_count": len(SUBJECTS),
        "fold_count": 2,
        "random_seed": 42,
        "configuration": {
            "maximum_interval_width": 0.5 if task_type == "regression" else None,
        },
        "class_vocabulary": list(CLASSES) if task_type == "classification" else [],
        "folds": [
            {
                "fold_index": 0,
                "probability_calibration_status": (
                    "calibrated" if task_type == "classification" else "not_applicable"
                ),
                "applied_selective_metrics": {
                    "coverage_point": {"accepted_count": accepted},
                    "empirical_interval_coverage": (
                        0.9 if task_type == "regression" else None
                    ),
                },
            }
        ],
        "total_window_count": 10,
        "accepted_count": accepted,
        "abstained_count": abstained,
        "unavailable_count": 1,
        "coverage": accepted / 10,
        "abstention_rate": abstained / 10,
        "abstention_reason_counts": (
            {"below_confidence_threshold": abstained}
            if task_type == "classification"
            else {"interval_too_wide": abstained}
        ),
        "disclaimers": DISCLAIMERS,
    }


def coverage_curve_document(
    *, run_id: str, task_type: str, record_axis: bool
) -> dict[str, Any]:
    """A ``coverage_curve.json``, optionally missing the ``axis`` field.

    Omitting the axis reproduces the pre-DEC-072 artifacts that shared one
    grid between the two coverage axes; the dashboard must refuse to
    display such a curve rather than guessing which axis it was.
    """
    classification = task_type == "classification"
    axis_values = [0.0, 0.5, 0.9] if classification else [0.1, 0.4, 1.0]
    coverages = [1.0, 0.7, 0.3] if classification else [0.3, 0.7, 1.0]
    curve: dict[str, Any] = {
        "task_type": task_type,
        "axis_values": axis_values,
        "points": [
            {
                "threshold": value,
                "coverage_point": {
                    "total_window_count": 10,
                    "accepted_count": int(coverage * 10),
                    "abstained_count": 10 - int(coverage * 10),
                    "unavailable_count": 0,
                    "coverage": coverage,
                    "abstention_rate": 1.0 - coverage,
                },
            }
            for value, coverage in zip(axis_values, coverages, strict=True)
        ],
        "risk_coverage": [
            {
                "threshold": value,
                "coverage": coverage,
                "empirical_risk": (0.4 - 0.1 * index) if classification else None,
                "accepted_count": int(coverage * 10),
                "unavailable_reason": (
                    None
                    if classification
                    else "empirical risk is undefined for a regression target"
                ),
            }
            for index, (value, coverage) in enumerate(
                zip(axis_values, coverages, strict=True)
            )
        ],
        "area_under_risk_coverage": 0.3 if classification else None,
        "area_under_risk_coverage_unavailable_reason": (
            None if classification else "no point has a defined risk"
        ),
        "expected_monotonic_direction": (
            "non_increasing" if classification else "non_decreasing"
        ),
        "coverage_is_monotonic": True,
    }
    if record_axis:
        curve["axis"] = (
            "confidence_threshold" if classification else "maximum_interval_width"
        )
    return {
        "run_id": run_id,
        "evaluation_mode": "software_self_check",
        "scientific_evaluation_eligible": False,
        "curve": curve,
        "disclaimers": DISCLAIMERS,
    }


def _selective_predictions(*, task_type: str) -> dict[str, list[Any]]:
    windows = [f"synthetic-session-0001:w{i:06d}" for i in range(10)]
    classification = task_type == "classification"
    return {
        "window_id": windows,
        "subject_id": [SUBJECTS[i % len(SUBJECTS)] for i in range(10)],
        "session_id": ["synthetic-session-0001"] * 10,
        "fold_index": [0] * 10,
        "accepted": [i < 6 for i in range(10)],
        "abstained": [6 <= i < 9 for i in range(10)],
        "confidence_score": (
            [0.5 + 0.04 * i for i in range(10)] if classification else [None] * 10
        ),
        "interval_width": (
            [None] * 10 if classification else [0.2 + 0.05 * i for i in range(10)]
        ),
        "interval_lower_bound": (
            [None] * 10 if classification else [0.1 * i for i in range(10)]
        ),
        "interval_upper_bound": (
            [None] * 10 if classification else [0.1 * i + 0.4 for i in range(10)]
        ),
    }


def _unselected_predictions(*, task_type: str) -> dict[str, list[Any]]:
    classification = task_type == "classification"
    return {
        "window_id": [f"synthetic-session-0001:w{i:06d}" for i in range(10)],
        "entropy": (
            [0.4 + 0.03 * i for i in range(10)] if classification else [None] * 10
        ),
        "margin": (
            [0.1 + 0.05 * i for i in range(10)] if classification else [None] * 10
        ),
    }


def make_adaptation_run(
    root: Path,
    name: str = "fixture-adaptation",
    *,
    experiment_mode: str = "adaptive",
    proposals: int = 2,
) -> Path:
    """A Milestone 8 adaptation run directory."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    run_id = f"adaptation-selfcheck-{name}"
    windows = 8
    holds = windows - proposals
    write_json(
        directory / "adaptation_policy_config.json",
        {"run_id": run_id, "mode": "conservative_rule_based"},
    )
    write_json(
        directory / "adaptation_summary.json",
        {
            "run_id": run_id,
            "engagevr_version": "0.1.0",
            "python_version": "3.12.13",
            "evaluation_mode": "software_self_check",
            "scientific_evaluation_eligible": False,
            "is_synthetic": True,
            "data_source": "synthetic",
            "configuration": {
                "enabled": experiment_mode != "static",
                "experiment_mode": experiment_mode,
                "mode": "conservative_rule_based",
                "minimum_persistence_windows": 3,
                "cooldown_windows": 6,
                "difficulty": {"minimum": 1, "maximum": 5, "step": 1},
                "max_adaptations_per_session": 10,
                "conflict_resolution": "hold",
                "regression_mapping_enabled": False,
            },
            "configuration_fingerprint": "0123456789abcdef",
            "scenario_names": ["fixture-scenario"],
            "session_ids": ["scn-fixture"],
            "metrics": {
                "evaluated_windows": windows,
                "gate_eligible_windows": windows - 1,
                "gate_blocked_windows": 1,
                "hold_decisions": holds,
                "adaptation_proposals": proposals,
                "increases": 0,
                "decreases": proposals,
                "hold_reason_counts": {
                    "insufficient_persistence": holds - 1,
                    "gate_blocked": 1,
                },
                "direction_reversals": 0,
                "minimum_proposal_spacing_windows": 7,
                "longest_same_direction_streak": 3,
                "eligible_window_adaptation_fraction": proposals / (windows - 1),
                "blocked_oscillation_attempts": 0,
                "final_difficulty_by_session": {"scn-fixture": 2},
                "proposals_by_session": {"scn-fixture": proposals},
            },
            "naive_comparison": {
                "evaluated_windows": windows,
                "hold_decisions": 2,
                "adaptation_proposals": 6,
                "direction_reversals": 1,
                "minimum_proposal_spacing_windows": 1,
                "eligible_window_adaptation_fraction": 6 / (windows - 1),
            },
            "started_at_utc": "2026-01-01T00:00:00Z",
            "finished_at_utc": "2026-01-01T00:00:01Z",
            "disclaimers": DISCLAIMERS,
        },
    )
    write_json(
        directory / "scenarios.json",
        {
            "disclaimer": "Deterministic controller tests.",
            "scenarios": [
                {
                    "name": "fixture-scenario",
                    "session_id": "scn-fixture",
                    "subject_id": "synthetic_subject",
                    "window_count": windows,
                    "description": "every window reads high cognitive load",
                    "expectation": "decrease after the dwell requirement",
                }
            ],
        },
    )
    write_parquet(
        directory / "adaptation_trace.parquet",
        _adaptation_trace(run_id=run_id, windows=windows, proposals=proposals),
    )
    write_checksums(directory)
    return directory


def _adaptation_trace(
    *, run_id: str, windows: int, proposals: int
) -> dict[str, list[Any]]:
    proposal_indices = {windows - 1 - i for i in range(proposals)}
    return {
        "run_id": [run_id] * windows,
        "scenario_id": ["fixture-scenario"] * windows,
        "session_id": ["scn-fixture"] * windows,
        "subject_id": ["synthetic_subject"] * windows,
        "window_id": [f"scn-fixture:w{i:06d}" for i in range(windows)],
        "window_order": list(range(windows)),
        "current_difficulty": [3 if i < 4 else 2 for i in range(windows)],
        "proposed_difficulty": [
            2 if i in proposal_indices else 3 for i in range(windows)
        ],
        "decision_kind": [
            "propose_adaptation" if i in proposal_indices else "hold"
            for i in range(windows)
        ],
        "resolved_direction": [
            "decrease" if i in proposal_indices else "hold" for i in range(windows)
        ],
        "policy_reasons": [
            ["proposal_eligible"]
            if i in proposal_indices
            else ["insufficient_persistence"]
            for i in range(windows)
        ],
        "command_built": [i in proposal_indices for i in range(windows)],
        "lifecycle_status": [
            "command_built" if i in proposal_indices else None for i in range(windows)
        ],
        "cooldown_after": [0] * windows,
        "persistence_after": [min(i, 3) for i in range(windows)],
        "is_synthetic": [True] * windows,
        "scientific_evaluation_eligible": [False] * windows,
    }


def write_checksums(directory: Path) -> Path:
    """Record the SHA-256 of every JSON and Parquet artifact in a run."""
    from engagevr.training.artifacts import sha256_file

    checksums = {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
        and path.suffix in (".json", ".parquet")
        and path.name != "checksums.json"
    }
    return write_json(directory / "checksums.json", checksums)


def corrupt(path: Path) -> Path:
    """Replace a JSON artifact with something that is not JSON."""
    path.write_text("{ this is not valid json", encoding="utf-8")
    return path


def summary_for(
    root: Path, name: str, *, validate: bool = False
) -> DashboardRunSummary:
    """The catalogue entry for one fixture run."""
    catalogue = build_catalogue(root, validate_checksums=validate)
    summary = catalogue.find(name)
    assert summary is not None, f"{name} was not discovered under {root}"
    return summary


__all__ = [
    "CLASSES",
    "DISCLAIMERS",
    "SUBJECTS",
    "classification_metrics",
    "corrupt",
    "coverage_curve_document",
    "dataset_document",
    "fusion_metrics",
    "make_adaptation_run",
    "make_baseline_run",
    "make_fusion_run",
    "make_personalization_run",
    "make_uncertainty_run",
    "manifest_document",
    "personalization_document",
    "regression_metrics",
    "robustness_document",
    "splits_document",
    "summary_for",
    "uncertainty_document",
    "write_checksums",
    "write_json",
    "write_parquet",
]
