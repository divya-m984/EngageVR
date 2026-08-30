"""Distribution-shift diagnostics between two explicitly named datasets.

What this is
------------
A small, deterministic, interpretable set of statistics describing how
far one dataset's columns moved relative to another's.  Five methods,
chosen because each answers a different question in a way a reader can
check by hand:

``missingness_rate_difference``
    ``P_current(missing) - P_reference(missing)``.  Whether a measurement
    stopped arriving.  **Missingness is a measurement-availability fact.
    It is never disengagement**, and this repository never reads it as
    one.

``standardized_mean_difference``
    ``(mean_current - mean_reference) / sqrt((var_ref + var_cur) / 2)``,
    the pooled-standard-deviation form of Cohen's *d*.  Whether the
    centre moved, in units that survive rescaling.

``kolmogorov_smirnov_statistic``
    ``sup_x |F_current(x) - F_reference(x)|``, the two-sample KS
    statistic.  Whether the *shape* moved, without assuming normality.
    The statistic only; no p-value is reported, because a p-value here
    would be a hypothesis test nobody specified in advance and would grow
    significant with sample size alone.

``population_stability_index``
    ``sum_i (c_i - r_i) * ln(c_i / r_i)`` over bins whose edges are the
    reference distribution's quantiles.  A single legible number for
    "how much mass moved between bins".

``categorical_total_variation_distance``
    ``0.5 * sum_k |c_k - r_k|`` over category shares.  The categorical
    analogue, bounded in ``[0, 1]``.

What this is not
----------------
This is **not** drift detection about a person, and **not** concept
drift.  Concept drift is a change in the relationship between features
and labels; establishing one needs labels from both periods, and no
validated participant-provided engagement or cognitive-load label exists
in this repository.  A prediction-side comparison is therefore named
``prediction_distribution_shift`` and nothing stronger.

A threshold crossing is not a failure.  Every default is an ENGINEERING
DIAGNOSTIC DEFAULT (see ``mlops.drift`` in ``configs/defaults.yaml``),
chosen for interpretability rather than calibrated against an outcome,
and the report has no field an "the model failed" claim could occupy.

Unavailability is never zero
----------------------------
A column missing on one side, present but all-null, or too thin to
support a statistic is reported ``unavailable`` with a reason.  It is
never reported as zero shift: zero is a legitimate answer meaning "these
distributions agree", and collapsing the two would let an absent
measurement read as a healthy one.

Target columns take no part.  ``target__*`` and ``target_meta__*`` are
excluded by construction, so a shift statistic can never be computed
from a label.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from engagevr.config import DriftThresholdSettings
from engagevr.mlops.fingerprints import repository_relative, sha256_payload
from engagevr.schemas.experiments import SELF_CHECK_DISCLAIMER
from engagevr.schemas.features import (
    AVAILABILITY_PREFIX,
    FEATURE_PREFIX,
    MODALITY_AVAILABLE_PREFIX,
    MODALITY_QUALITY_PREFIX,
    TARGET_META_PREFIX,
    TARGET_PREFIX,
)
from engagevr.schemas.mlops import (
    DRIFT_INTERPRETATION_NOTE,
    MLOPS_DISCLAIMER,
    DriftDatasetReference,
    DriftMethod,
    DriftReport,
    DriftReportKind,
    DriftStatistic,
    DriftStatus,
    FeatureDriftResult,
)

#: Columns compared as continuous quantities.
NUMERIC_PREFIXES: tuple[str, ...] = (FEATURE_PREFIX, MODALITY_QUALITY_PREFIX)

#: Columns compared as category shares.
CATEGORICAL_PREFIXES: tuple[str, ...] = (
    AVAILABILITY_PREFIX,
    MODALITY_AVAILABLE_PREFIX,
)

#: Columns excluded from every comparison, with the reason recorded in
#: the report so a reader never has to guess why a column is absent.
EXCLUSION_REASONS: dict[str, str] = {
    "target": (
        "a target column. A distribution-shift statistic computed from a "
        "label would be a leakage path, and no shift diagnostic in this "
        "repository is permitted to read one."
    ),
    "identity": (
        "an identity or window-geometry column. Comparing window ids or "
        "timestamps measures how the datasets were assembled, not how any "
        "measurement moved."
    ),
    "provenance": (
        "a provenance column. Its distribution is reported as dataset "
        "provenance on both sides of this report rather than as a shift "
        "statistic."
    ),
    "schema": "a schema-version column, constant by construction.",
}

_IDENTITY_COLUMNS: frozenset[str] = frozenset(
    {
        "window_id",
        "session_id",
        "subject_id",
        "window_index",
        "window_start_utc",
        "window_end_utc",
        "window_start_monotonic_seconds",
        "window_end_monotonic_seconds",
        "window_duration_seconds",
        "window_step_seconds",
        "windows_overlap",
    }
)

_PROVENANCE_COLUMNS: frozenset[str] = frozenset(
    {"subject_kind", "experiment_condition", "data_source", "synthetic_label"}
)

_SCHEMA_COLUMNS: frozenset[str] = frozenset(
    {"feature_schema_version", "feature_catalog_version"}
)

#: Epsilon substituted for an empty PSI bin.
#:
#: ``ln(0)`` is undefined and ``ln(c/0)`` is infinite, so an empty bin on
#: either side would make the whole index non-finite. A small floor keeps
#: the number finite and is the conventional treatment; it means a PSI
#: computed over a bin nobody landed in is a floor artefact rather than a
#: measurement, which is why the bin count is recorded in the report.
PSI_EPSILON = 1e-6


class DriftError(ValueError):
    """A distribution-shift comparison could not be set up."""


def _column_role(name: str) -> str | None:
    """Which comparison a column takes part in, or ``None`` if excluded."""
    if name.startswith((TARGET_PREFIX, TARGET_META_PREFIX)):
        return None
    if name.startswith(NUMERIC_PREFIXES):
        return "numeric"
    if name.startswith(CATEGORICAL_PREFIXES):
        return "categorical"
    return None


def _exclusion_reason(name: str) -> str:
    if name.startswith((TARGET_PREFIX, TARGET_META_PREFIX)):
        return EXCLUSION_REASONS["target"]
    if name in _IDENTITY_COLUMNS:
        return EXCLUSION_REASONS["identity"]
    if name in _PROVENANCE_COLUMNS:
        return EXCLUSION_REASONS["provenance"]
    if name in _SCHEMA_COLUMNS:
        return EXCLUSION_REASONS["schema"]
    return (
        "not a comparable measurement column. Only feature, availability, "
        "modality-availability, and modality-quality columns are compared."
    )


def _values(table: pa.Table, name: str) -> list[Any]:
    return list(table.column(name).to_pylist())


def _numeric(values: Sequence[Any]) -> np.ndarray:
    """Present, finite values as a float array.

    ``None`` and NaN both mean *not measured* here, and a non-finite
    value is not a measurement, so all three are dropped before any
    statistic is computed rather than propagating into it.
    """
    kept: list[float] = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            kept.append(number)
    return np.asarray(kept, dtype=np.float64)


def _categories(values: Sequence[Any]) -> list[str]:
    return [str(value) for value in values if value is not None]


def _missing_count(values: Sequence[Any], *, numeric: bool) -> int:
    missing = 0
    for value in values:
        if value is None:
            missing += 1
        elif numeric and not isinstance(value, bool):
            try:
                if not math.isfinite(float(value)):
                    missing += 1
            except (TypeError, ValueError):
                missing += 1
    return missing


def _unavailable(
    method: DriftMethod, status: DriftStatus, reason: str
) -> DriftStatistic:
    return DriftStatistic(
        method=method,
        status=status,
        unavailable_reason=reason,
        interpretation=(
            "Not computable from this pair of columns. This is not zero "
            "shift; it is an absence of evidence about shift."
        ),
    )


def _statistic(
    method: DriftMethod,
    value: float,
    threshold: float,
    interpretation: str,
) -> DriftStatistic:
    return DriftStatistic(
        method=method,
        statistic=float(value),
        threshold=float(threshold),
        exceeded=bool(abs(value) > threshold),
        status=DriftStatus.COMPUTED,
        interpretation=interpretation,
    )


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, *, bins: int
) -> float | None:
    """PSI over quantile bins of the reference distribution.

    Returns ``None`` when the reference is constant, in which case every
    quantile edge collapses and the index would describe the binning
    rather than the data.
    """
    if reference.size == 0 or current.size == 0:
        return None
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        return None
    interior = edges[1:-1]
    reference_counts = np.bincount(
        np.searchsorted(interior, reference, side="right"), minlength=edges.size - 1
    ).astype(np.float64)
    current_counts = np.bincount(
        np.searchsorted(interior, current, side="right"), minlength=edges.size - 1
    ).astype(np.float64)
    reference_share = np.maximum(reference_counts / reference.size, PSI_EPSILON)
    current_share = np.maximum(current_counts / current.size, PSI_EPSILON)
    return float(
        np.sum(
            (current_share - reference_share) * np.log(current_share / reference_share)
        )
    )


def total_variation_distance(
    reference: Sequence[str], current: Sequence[str]
) -> float | None:
    """Half the L1 distance between two category-share vectors."""
    if not reference or not current:
        return None
    labels = sorted(set(reference) | set(current))
    reference_total = float(len(reference))
    current_total = float(len(current))
    total = 0.0
    for label in labels:
        reference_share = reference.count(label) / reference_total
        current_share = current.count(label) / current_total
        total += abs(current_share - reference_share)
    return float(0.5 * total)


def standardized_mean_difference(
    reference: np.ndarray, current: np.ndarray
) -> float | None:
    """Cohen's *d* with a pooled standard deviation, or ``None``."""
    if reference.size < 2 or current.size < 2:
        return None
    pooled = math.sqrt(
        (float(np.var(reference, ddof=1)) + float(np.var(current, ddof=1))) / 2.0
    )
    if pooled <= 0.0:
        return None
    return float((float(np.mean(current)) - float(np.mean(reference))) / pooled)


def _compare_numeric(
    name: str,
    reference: Sequence[Any],
    current: Sequence[Any],
    thresholds: DriftThresholdSettings,
) -> FeatureDriftResult:
    reference_values = _numeric(reference)
    current_values = _numeric(current)
    reference_missing = _missing_count(reference, numeric=True)
    current_missing = _missing_count(current, numeric=True)

    statistics: list[DriftStatistic] = [
        _missingness_statistic(
            len(reference), len(current), reference_missing, current_missing, thresholds
        )
    ]

    if reference_values.size == 0 or current_values.size == 0:
        reason = (
            f"{name} has no present, finite value on at least one side "
            f"(reference {reference_values.size}, current {current_values.size}). "
            "Distribution statistics need values to compare."
        )
        for method in (
            DriftMethod.STANDARDIZED_MEAN_DIFFERENCE,
            DriftMethod.KOLMOGOROV_SMIRNOV,
            DriftMethod.POPULATION_STABILITY_INDEX,
        ):
            statistics.append(
                _unavailable(method, DriftStatus.UNAVAILABLE_ALL_VALUES_MISSING, reason)
            )
    elif (
        reference_values.size < thresholds.minimum_samples
        or current_values.size < thresholds.minimum_samples
    ):
        reason = (
            f"{name} has {reference_values.size} reference and "
            f"{current_values.size} current present values; "
            f"mlops.drift.minimum_samples is {thresholds.minimum_samples}. A "
            "statistic from fewer samples would describe the sample, not the "
            "distribution."
        )
        for method in (
            DriftMethod.STANDARDIZED_MEAN_DIFFERENCE,
            DriftMethod.KOLMOGOROV_SMIRNOV,
            DriftMethod.POPULATION_STABILITY_INDEX,
        ):
            statistics.append(
                _unavailable(
                    method, DriftStatus.UNAVAILABLE_INSUFFICIENT_SAMPLES, reason
                )
            )
    else:
        smd = standardized_mean_difference(reference_values, current_values)
        if smd is None:
            statistics.append(
                _unavailable(
                    DriftMethod.STANDARDIZED_MEAN_DIFFERENCE,
                    DriftStatus.UNAVAILABLE_ZERO_VARIANCE,
                    f"{name} has zero pooled variance, so a standardized mean "
                    "difference would divide by zero. The two columns may still "
                    "differ in level; compare the raw means.",
                )
            )
        else:
            statistics.append(
                _statistic(
                    DriftMethod.STANDARDIZED_MEAN_DIFFERENCE,
                    smd,
                    thresholds.standardized_mean_difference,
                    "Mean difference in pooled standard deviations. Positive "
                    "means the current dataset sits higher. A shift in a "
                    "feature's centre is not a change in any person's state.",
                )
            )
        ks = float(stats.ks_2samp(reference_values, current_values).statistic)
        statistics.append(
            _statistic(
                DriftMethod.KOLMOGOROV_SMIRNOV,
                ks,
                thresholds.kolmogorov_smirnov,
                "Maximum absolute distance between the two empirical CDFs. "
                "The statistic only: no p-value is reported, because it would "
                "grow significant with sample size alone.",
            )
        )
        psi = population_stability_index(
            reference_values, current_values, bins=thresholds.histogram_bins
        )
        if psi is None:
            statistics.append(
                _unavailable(
                    DriftMethod.POPULATION_STABILITY_INDEX,
                    DriftStatus.UNAVAILABLE_ZERO_VARIANCE,
                    f"{name} is constant in the reference dataset, so every "
                    "quantile edge collapses into one bin and the index would "
                    "describe the binning rather than the data.",
                )
            )
        else:
            statistics.append(
                _statistic(
                    DriftMethod.POPULATION_STABILITY_INDEX,
                    psi,
                    thresholds.population_stability_index,
                    f"Mass moved between {thresholds.histogram_bins} quantile "
                    "bins of the reference. An engineering summary, not a test.",
                )
            )

    return _assemble(
        name,
        "numeric",
        reference,
        current,
        reference_missing,
        current_missing,
        statistics,
    )


def _compare_categorical(
    name: str,
    reference: Sequence[Any],
    current: Sequence[Any],
    thresholds: DriftThresholdSettings,
) -> FeatureDriftResult:
    reference_values = _categories(reference)
    current_values = _categories(current)
    reference_missing = _missing_count(reference, numeric=False)
    current_missing = _missing_count(current, numeric=False)

    statistics: list[DriftStatistic] = [
        _missingness_statistic(
            len(reference), len(current), reference_missing, current_missing, thresholds
        )
    ]
    if not reference_values or not current_values:
        statistics.append(
            _unavailable(
                DriftMethod.CATEGORICAL_TOTAL_VARIATION,
                DriftStatus.UNAVAILABLE_ALL_VALUES_MISSING,
                f"{name} has no present value on at least one side.",
            )
        )
    elif (
        len(reference_values) < thresholds.minimum_samples
        or len(current_values) < thresholds.minimum_samples
    ):
        statistics.append(
            _unavailable(
                DriftMethod.CATEGORICAL_TOTAL_VARIATION,
                DriftStatus.UNAVAILABLE_INSUFFICIENT_SAMPLES,
                f"{name} has {len(reference_values)} reference and "
                f"{len(current_values)} current present values; "
                f"mlops.drift.minimum_samples is {thresholds.minimum_samples}.",
            )
        )
    else:
        distance = total_variation_distance(reference_values, current_values)
        assert distance is not None
        statistics.append(
            _statistic(
                DriftMethod.CATEGORICAL_TOTAL_VARIATION,
                distance,
                thresholds.categorical_total_variation,
                "Half the summed absolute change in category share, bounded "
                "in [0, 1]. New and vanished categories both contribute.",
            )
        )
    return _assemble(
        name,
        "categorical",
        reference,
        current,
        reference_missing,
        current_missing,
        statistics,
    )


def _missingness_statistic(
    reference_rows: int,
    current_rows: int,
    reference_missing: int,
    current_missing: int,
    thresholds: DriftThresholdSettings,
) -> DriftStatistic:
    if reference_rows == 0 or current_rows == 0:
        return _unavailable(
            DriftMethod.MISSINGNESS_RATE_DIFFERENCE,
            DriftStatus.UNAVAILABLE_INSUFFICIENT_SAMPLES,
            "one side has no rows, so a missingness rate is undefined.",
        )
    difference = (current_missing / current_rows) - (reference_missing / reference_rows)
    return _statistic(
        DriftMethod.MISSINGNESS_RATE_DIFFERENCE,
        difference,
        thresholds.missingness_rate_difference,
        "Change in how often this measurement was unavailable. Positive "
        "means it arrived less often. Missingness is a measurement-"
        "availability fact and is NEVER disengagement.",
    )


def _assemble(
    name: str,
    kind: str,
    reference: Sequence[Any],
    current: Sequence[Any],
    reference_missing: int,
    current_missing: int,
    statistics: Sequence[DriftStatistic],
) -> FeatureDriftResult:
    computed = [s for s in statistics if s.status is DriftStatus.COMPUTED]
    status = DriftStatus.COMPUTED
    reason: str | None = None
    if not computed:
        status = statistics[0].status if statistics else DriftStatus.COMPUTED
        reason = statistics[0].unavailable_reason if statistics else None
    return FeatureDriftResult(
        feature_name=name,
        value_kind=kind,
        reference_row_count=len(reference),
        current_row_count=len(current),
        reference_present_count=len(reference) - reference_missing,
        current_present_count=len(current) - current_missing,
        reference_missing_rate=(
            reference_missing / len(reference) if reference else None
        ),
        current_missing_rate=(current_missing / len(current) if current else None),
        statistics=tuple(statistics),
        status=status,
        unavailable_reason=reason,
        exceeded_methods=tuple(s.method for s in statistics if s.exceeded),
    )


def _absent(
    name: str,
    kind: str,
    reference_rows: int,
    current_rows: int,
    status: DriftStatus,
    reason: str,
) -> FeatureDriftResult:
    return FeatureDriftResult(
        feature_name=name,
        value_kind=kind,
        reference_row_count=reference_rows,
        current_row_count=current_rows,
        reference_present_count=0,
        current_present_count=0,
        statistics=(),
        status=status,
        unavailable_reason=reason,
    )


def _dataset_reference(role: str, path: Path, table: pa.Table) -> DriftDatasetReference:
    fingerprint: str | None = None
    counts: dict[str, int] = {}
    subject_count: int | None = None
    eligible = False
    metadata_path = path.with_name(f"{path.stem}.metadata.json")
    if metadata_path.is_file():
        import json

        document = json.loads(metadata_path.read_text(encoding="utf-8"))
        fingerprint = document.get("dataset_fingerprint")
        counts = {
            str(k): int(v) for k, v in document.get("data_source_counts", {}).items()
        }
        subject_count = document.get("subject_count")
        eligible = bool(document.get("scientific_evaluation_eligible", False))
    elif "data_source" in table.column_names:
        for value in table.column("data_source").to_pylist():
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
    synthetic = bool(counts) and set(counts) == {"synthetic"}
    if not counts:
        synthetic = not eligible
    return DriftDatasetReference(
        role=role,
        path=repository_relative(path),
        dataset_fingerprint=fingerprint,
        row_count=table.num_rows,
        subject_count=int(subject_count) if subject_count is not None else None,
        data_source_counts=counts,
        is_synthetic=synthetic,
        scientific_evaluation_eligible=eligible and not synthetic,
    )


def compare_datasets(
    reference_path: Path,
    current_path: Path,
    *,
    thresholds: DriftThresholdSettings,
) -> DriftReport:
    """Compare two windowed-feature datasets, feature by feature."""
    reference_path = Path(reference_path)
    current_path = Path(current_path)
    for label, path in (("reference", reference_path), ("current", current_path)):
        if not path.is_file():
            raise DriftError(
                f"the {label} dataset {path} does not exist. Both sides of a "
                "distribution-shift comparison must be named explicitly; "
                "nothing here guesses which directories to compare."
            )
    reference_table = pq.read_table(reference_path)
    current_table = pq.read_table(current_path)
    return _build_report(
        DriftReportKind.FEATURE_DISTRIBUTION_SHIFT,
        _dataset_reference("reference", reference_path, reference_table),
        _dataset_reference("current", current_path, current_table),
        reference_table,
        current_table,
        thresholds=thresholds,
        column_roles={
            name: role
            for name in sorted(
                set(reference_table.column_names) | set(current_table.column_names)
            )
            if (role := _column_role(name)) is not None
        },
        excluded={
            name: _exclusion_reason(name)
            for name in sorted(
                set(reference_table.column_names) | set(current_table.column_names)
            )
            if _column_role(name) is None
        },
    )


def compare_predictions(
    reference_path: Path,
    current_path: Path,
    *,
    thresholds: DriftThresholdSettings,
    column: str = "predicted_value",
) -> DriftReport:
    """Compare the predicted-value column of two prediction tables.

    Named ``prediction_distribution_shift`` and nothing stronger.  A
    change in what a model predicts is a change in what a model predicts.
    Calling it concept drift would assert something about the relationship
    between features and labels, which cannot be assessed without labels
    from both periods.
    """
    reference_path = Path(reference_path)
    current_path = Path(current_path)
    for label, path in (("reference", reference_path), ("current", current_path)):
        if not path.is_file():
            raise DriftError(f"the {label} prediction table {path} does not exist")
    reference_table = pq.read_table(reference_path)
    current_table = pq.read_table(current_path)
    for label, table in (("reference", reference_table), ("current", current_table)):
        if column not in table.column_names:
            raise DriftError(
                f"the {label} prediction table has no {column!r} column; it "
                f"has {sorted(table.column_names)}"
            )
    values = reference_table.column(column).to_pylist()
    role = (
        "categorical"
        if any(isinstance(v, str) for v in values if v is not None)
        else "numeric"
    )
    return _build_report(
        DriftReportKind.PREDICTION_DISTRIBUTION_SHIFT,
        _dataset_reference("reference", reference_path, reference_table),
        _dataset_reference("current", current_path, current_table),
        reference_table,
        current_table,
        thresholds=thresholds,
        column_roles={column: role},
        excluded={
            name: (
                "not the compared prediction column. A prediction-shift "
                "report compares one column, named explicitly."
            )
            for name in sorted(
                set(reference_table.column_names) | set(current_table.column_names)
            )
            if name != column
        },
    )


def _build_report(
    kind: DriftReportKind,
    reference_reference: DriftDatasetReference,
    current_reference: DriftDatasetReference,
    reference_table: pa.Table,
    current_table: pa.Table,
    *,
    thresholds: DriftThresholdSettings,
    column_roles: Mapping[str, str],
    excluded: Mapping[str, str],
) -> DriftReport:
    results: list[FeatureDriftResult] = []
    for name in sorted(column_roles):
        role = column_roles[name]
        in_reference = name in reference_table.column_names
        in_current = name in current_table.column_names
        if not in_reference:
            results.append(
                _absent(
                    name,
                    role,
                    reference_table.num_rows,
                    current_table.num_rows,
                    DriftStatus.UNAVAILABLE_MISSING_IN_REFERENCE,
                    f"{name} is absent from the reference dataset. There is "
                    "nothing to compare against; this is not zero shift.",
                )
            )
            continue
        if not in_current:
            results.append(
                _absent(
                    name,
                    role,
                    reference_table.num_rows,
                    current_table.num_rows,
                    DriftStatus.UNAVAILABLE_MISSING_IN_CURRENT,
                    f"{name} is absent from the current dataset. There is "
                    "nothing to compare; this is not zero shift.",
                )
            )
            continue
        reference_values = _values(reference_table, name)
        current_values = _values(current_table, name)
        if role == "numeric" and _looks_categorical(reference_values, current_values):
            results.append(
                _absent(
                    name,
                    role,
                    reference_table.num_rows,
                    current_table.num_rows,
                    DriftStatus.UNAVAILABLE_TYPE_MISMATCH,
                    f"{name} holds non-numeric values on at least one side, so "
                    "the two columns are not the same kind of quantity.",
                )
            )
            continue
        if role == "numeric":
            results.append(
                _compare_numeric(name, reference_values, current_values, thresholds)
            )
        else:
            results.append(
                _compare_categorical(name, reference_values, current_values, thresholds)
            )

    compared = tuple(
        r.feature_name for r in results if r.status is DriftStatus.COMPUTED
    )
    unavailable = tuple(
        r.feature_name for r in results if r.status is not DriftStatus.COMPUTED
    )
    is_synthetic = reference_reference.is_synthetic and current_reference.is_synthetic
    payload = {
        "report_kind": kind.value,
        "reference": {
            "fingerprint": reference_reference.dataset_fingerprint,
            "row_count": reference_reference.row_count,
        },
        "current": {
            "fingerprint": current_reference.dataset_fingerprint,
            "row_count": current_reference.row_count,
        },
        "compared_features": list(compared),
        "unavailable_features": list(unavailable),
        "thresholds": thresholds.as_mapping(),
        "minimum_samples": thresholds.minimum_samples,
        "histogram_bin_count": thresholds.histogram_bins,
        "results": [
            {
                "feature": result.feature_name,
                "statistics": [
                    {
                        "method": statistic.method.value,
                        "status": statistic.status.value,
                        "statistic": statistic.statistic,
                        "threshold": statistic.threshold,
                        "exceeded": statistic.exceeded,
                    }
                    for statistic in result.statistics
                ],
            }
            for result in results
        ],
    }
    return DriftReport(
        report_kind=kind,
        reference=reference_reference,
        current=current_reference,
        compared_features=compared,
        excluded_features=dict(excluded),
        unavailable_features=unavailable,
        thresholds=thresholds.as_mapping(),
        minimum_samples=thresholds.minimum_samples,
        histogram_bin_count=thresholds.histogram_bins,
        results=tuple(results),
        features_compared_count=len(compared),
        features_exceeding_count=sum(1 for r in results if r.exceeded_methods),
        features_unavailable_count=len(unavailable),
        report_fingerprint=sha256_payload(payload),
        is_synthetic=is_synthetic,
        scientific_evaluation_eligible=False,
        interpretation=DRIFT_INTERPRETATION_NOTE,
        disclaimers=(
            SELF_CHECK_DISCLAIMER if is_synthetic else DRIFT_INTERPRETATION_NOTE,
            MLOPS_DISCLAIMER,
        ),
    )


def _looks_categorical(reference: Sequence[Any], current: Sequence[Any]) -> bool:
    for values in (reference, current):
        for value in values:
            if value is None:
                continue
            if isinstance(value, str):
                return True
    return False


def exceeding_summary(report: DriftReport) -> tuple[str, ...]:
    """One line per feature-method pair whose statistic crossed a default."""
    lines: list[str] = []
    for result in report.results:
        for statistic in result.statistics:
            if statistic.exceeded:
                lines.append(
                    f"{result.feature_name}: {statistic.method.value}="
                    f"{statistic.statistic:.4f} exceeds the engineering "
                    f"diagnostic default {statistic.threshold:.4f}"
                )
    return tuple(lines)


__all__ = [
    "CATEGORICAL_PREFIXES",
    "EXCLUSION_REASONS",
    "NUMERIC_PREFIXES",
    "PSI_EPSILON",
    "DriftError",
    "compare_datasets",
    "compare_predictions",
    "exceeding_summary",
    "population_stability_index",
    "standardized_mean_difference",
    "total_variation_distance",
]
