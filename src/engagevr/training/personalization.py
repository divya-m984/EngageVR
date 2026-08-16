"""Personalization algebra: chronological splits, personal baselines, corrections.

This module holds the deterministic, dependency-free core of the Milestone
6 personalization pass: which columns may be personalised, how a subject's
windows are cut into a calibration region and a later evaluation region,
how a personal baseline is estimated and applied, and how a population
prediction is corrected for one subject.  Everything here is pure: no
estimator is fitted, no fold is built, and no artifact is written.

The one ordering rule
---------------------
A personal statistic may only be estimated from windows that **end before**
the evaluation region **begins**.  With overlapping windows a positional
split is not enough: a calibration window whose interval extends past the
boundary shares evidence with the first evaluation window.  So the split is
made in wall-clock time and any window straddling the boundary is dropped
from both regions and listed.  Nothing is ever mixed at random.

What is never personalised
--------------------------
Identifiers, timestamps, targets, target provenance, split metadata,
availability flags, modality-availability flags, modality-quality columns,
and categorical provenance fields.  Only catalogued measured features of
the configured measurement modalities are.  A poor-quality reading stays a
missing measurement; it never becomes a normalised physiological value, and
a low quality score is never converted into low engagement.

Personalized *calibration* here means adapting a population model to one
subject.  It is not uncertainty calibration and nothing here abstains;
confidence thresholds and selective prediction are Milestone 7.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from engagevr.schemas.features import (
    FEATURE_PREFIX,
    FeatureCatalog,
)
from engagevr.schemas.fusion import FusionModality
from engagevr.schemas.personalization import (
    CLASSIFICATION_CORRECTION_EQUATION,
    REGRESSION_CORRECTION_EQUATION,
    PersonalBaselineStatistics,
    PersonalCalibrationSplit,
    PersonalizationConfiguration,
    PersonalizationCorrection,
    PersonalizationMethod,
)
from engagevr.schemas.targets import TaskType
from engagevr.training.fusion import FEATURE_MODALITY_OF

#: Column-name prefixes that may never be personalised, with the reason.
FORBIDDEN_COLUMN_PREFIXES: dict[str, str] = {
    "avail__": (
        "a per-feature availability flag is a binary indicator of whether a "
        "measurement exists; z-scoring it would turn 'absent' into a number"
    ),
    "modality_available__": (
        "a modality-availability flag is a binary indicator, not a measurement"
    ),
    "modality_quality__": (
        "signal quality describes the MEASUREMENT, never the person; "
        "normalising it against a personal baseline would present it as a "
        "personal physiological value"
    ),
    "target__": "a target may never be transformed as if it were a predictor",
    "target_meta__": "target provenance is metadata, not a measurement",
    "window_": "window timing is provenance, not a measurement",
}

#: Exact column names that may never be personalised.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "subject_id",
        "session_id",
        "window_id",
        "subject_kind",
        "experiment_condition",
        "data_source",
        "synthetic_label",
        "feature_schema_version",
        "feature_catalog_version",
        "windows_overlap",
    }
)


class PersonalizationError(ValueError):
    """A personalization operation cannot be performed as requested."""


@dataclass(frozen=True, slots=True)
class SubjectWindow:
    """One of a subject's windows, with everything the split needs."""

    row_index: int
    window_id: str
    session_id: str
    start_utc: datetime | None
    end_utc: datetime | None
    window_index: int

    def sort_key(self) -> tuple[float, float, int, str]:
        """Deterministic ordering key: time first, then index, then id."""
        start = _epoch(self.start_utc)
        end = _epoch(self.end_utc)
        return (start, end, self.window_index, self.window_id)


def _epoch(moment: datetime | None) -> float:
    return math.inf if moment is None else moment.timestamp()


def parse_method(name: str) -> PersonalizationMethod:
    """Parse a personalization method, accepting hyphens as well as underscores.

    Raises
    ------
    PersonalizationError
        If ``name`` is not an implemented method.
    """
    normalised = name.strip().lower().replace("-", "_")
    try:
        return PersonalizationMethod(normalised)
    except ValueError as exc:
        valid = ", ".join(
            m.value.replace("_", "-")
            for m in PersonalizationMethod
            if m is not PersonalizationMethod.COLD_START
        )
        raise PersonalizationError(
            f"unknown personalization method {name!r}; implemented methods: "
            f"{valid}. 'cold-start' is an outcome, not a method to request: "
            "ask for it with --calibration-windows 0."
        ) from exc


def personalizable_columns(
    modalities: Sequence[FusionModality],
    predictor_columns: Sequence[str],
    catalog: FeatureCatalog,
) -> tuple[str, ...]:
    """Columns a personal baseline may normalise, in catalogue order.

    Only catalogued measured features (``feat__``) of the configured
    measurement modalities qualify.  Availability flags, modality flags,
    modality-quality columns, identifiers, timestamps, targets, and target
    provenance are all excluded — see :data:`FORBIDDEN_COLUMN_PREFIXES`.
    """
    wanted = {FEATURE_MODALITY_OF[modality] for modality in modalities}
    selected: list[str] = []
    for column in predictor_columns:
        if not column.startswith(FEATURE_PREFIX):
            continue
        try:
            entry = catalog.get(column.removeprefix(FEATURE_PREFIX))
        except KeyError:  # pragma: no cover - refused earlier by assert_no_leakage
            continue
        if entry.modality in wanted:
            selected.append(column)
    return tuple(selected)


def assert_personalizable(columns: Sequence[str]) -> None:
    """Assert every column may legitimately carry a personal baseline.

    Raises
    ------
    PersonalizationError
        On the first column that must not be personalised, naming why.
    """
    for column in columns:
        name = str(column)
        if name in FORBIDDEN_COLUMNS:
            raise PersonalizationError(
                f"column {name!r} may never be personalised: it is an "
                "identifier or provenance field, not a measurement"
            )
        for prefix, reason in FORBIDDEN_COLUMN_PREFIXES.items():
            if name.startswith(prefix):
                raise PersonalizationError(
                    f"column {name!r} may never be personalised: {reason}"
                )
        if not name.startswith(FEATURE_PREFIX):
            raise PersonalizationError(
                f"column {name!r} is not a catalogued measured feature "
                f"(expected the {FEATURE_PREFIX!r} prefix); only measured "
                "features carry a personal baseline"
            )


def subject_windows(
    *,
    row_indices: Sequence[int],
    window_ids: Sequence[str],
    session_ids: Sequence[str],
    window_start_utc: Sequence[datetime | None],
    window_end_utc: Sequence[datetime | None],
    window_indices: Sequence[int],
) -> tuple[SubjectWindow, ...]:
    """Assemble one subject's windows from parallel frame columns."""
    windows: list[SubjectWindow] = []
    for row in row_indices:
        index = int(row)
        windows.append(
            SubjectWindow(
                row_index=index,
                window_id=str(window_ids[index]),
                session_id=str(session_ids[index]),
                start_utc=(
                    window_start_utc[index] if index < len(window_start_utc) else None
                ),
                end_utc=(
                    window_end_utc[index] if index < len(window_end_utc) else None
                ),
                window_index=(
                    int(window_indices[index]) if index < len(window_indices) else index
                ),
            )
        )
    return tuple(windows)


def build_calibration_split(
    windows: Sequence[SubjectWindow],
    *,
    subject_id: str,
    fold_index: int,
    calibration_windows: int,
    minimum_evaluation_windows: int,
    windows_overlap: bool = False,
) -> tuple[
    PersonalCalibrationSplit, tuple[SubjectWindow, ...], tuple[SubjectWindow, ...]
]:
    """Cut one subject's windows into a calibration region and a later one.

    Returns the recorded split plus the calibration and evaluation windows
    themselves.  ``calibration_windows=0`` requests cold-start mode: no
    calibration evidence is taken and every window is evaluated under the
    population model.

    Raises
    ------
    PersonalizationError
        If a window carries no start timestamp, so the two regions could not
        be ordered in time.
    """
    ordered = tuple(sorted(windows, key=SubjectWindow.sort_key))
    sessions = tuple(sorted({w.session_id for w in ordered}))
    total = len(ordered)

    # Every window is paired with its concrete start and end here so the rest
    # of the function works with real instants. A window without both is a
    # refusal, not something to order by row position: row order is not a
    # temporal order, and the whole point of this split is that it is one.
    timed: list[tuple[SubjectWindow, datetime, datetime]] = []
    for window in ordered:
        if window.start_utc is None or window.end_utc is None:
            raise PersonalizationError(
                f"subject {subject_id!r} has window(s) with no start or end "
                "timestamp, so a calibration region cannot be proved to "
                "precede an evaluation region. Personalization is refused "
                "rather than ordered by row position, which would not be a "
                "temporal split."
            )
        timed.append((window, window.start_utc, window.end_utc))

    if calibration_windows == 0:
        if total < minimum_evaluation_windows:
            return (
                _unavailable_split(
                    subject_id=subject_id,
                    fold_index=fold_index,
                    sessions=sessions,
                    total=total,
                    reason=(
                        f"subject {subject_id!r} has {total} window(s), fewer "
                        f"than the {minimum_evaluation_windows} required for "
                        "evaluation"
                    ),
                    windows_overlap=windows_overlap,
                ),
                (),
                (),
            )
        split = PersonalCalibrationSplit(
            subject_id=subject_id,
            fold_index=fold_index,
            session_ids=sessions,
            total_window_count=total,
            available=True,
            cold_start=True,
            cold_start_reason=(
                "cold-start mode was requested with calibration_windows=0: no "
                "personal evidence is taken, and the population model is used "
                "unchanged for every window"
            ),
            evaluation_window_ids=tuple(w.window_id for w, _s, _e in timed),
            evaluation_start_utc=timed[0][1],
            evaluation_end_utc=max(end for _w, _s, end in timed),
            windows_overlap=windows_overlap,
        )
        return split, (), ordered

    if total <= calibration_windows:
        return (
            _unavailable_split(
                subject_id=subject_id,
                fold_index=fold_index,
                sessions=sessions,
                total=total,
                reason=(
                    f"subject {subject_id!r} has {total} window(s), which "
                    f"cannot supply {calibration_windows} calibration "
                    "window(s) and still leave a later evaluation region"
                ),
                windows_overlap=windows_overlap,
            ),
            (),
            (),
        )

    calibration_timed = timed[:calibration_windows]
    calibration = tuple(w for w, _s, _e in calibration_timed)
    boundary = max(end for _w, _s, end in calibration_timed)
    remainder = timed[calibration_windows:]
    evaluation_timed = [entry for entry in remainder if entry[1] >= boundary]
    evaluation = tuple(w for w, _s, _e in evaluation_timed)
    excluded = tuple(w for w, start, _e in remainder if start < boundary)

    if len(evaluation) < minimum_evaluation_windows:
        return (
            _unavailable_split(
                subject_id=subject_id,
                fold_index=fold_index,
                sessions=sessions,
                total=total,
                reason=(
                    f"subject {subject_id!r} has {len(evaluation)} window(s) "
                    f"starting at or after the calibration boundary "
                    f"{boundary.isoformat()}, fewer than the "
                    f"{minimum_evaluation_windows} required; "
                    f"{len(excluded)} window(s) straddled the boundary and "
                    "were excluded rather than moved"
                ),
                windows_overlap=windows_overlap,
            ),
            (),
            (),
        )

    split = PersonalCalibrationSplit(
        subject_id=subject_id,
        fold_index=fold_index,
        session_ids=sessions,
        total_window_count=total,
        available=True,
        calibration_window_ids=tuple(w.window_id for w in calibration),
        evaluation_window_ids=tuple(w.window_id for w in evaluation),
        excluded_overlap_window_ids=tuple(w.window_id for w in excluded),
        calibration_start_utc=calibration_timed[0][1],
        calibration_end_utc=boundary,
        evaluation_start_utc=evaluation_timed[0][1],
        evaluation_end_utc=max(end for _w, _s, end in evaluation_timed),
        windows_overlap=windows_overlap,
        temporal_order_verified=True,
    )
    return split, calibration, evaluation


def _unavailable_split(
    *,
    subject_id: str,
    fold_index: int,
    sessions: tuple[str, ...],
    total: int,
    reason: str,
    windows_overlap: bool,
) -> PersonalCalibrationSplit:
    return PersonalCalibrationSplit(
        subject_id=subject_id,
        fold_index=fold_index,
        session_ids=sessions,
        total_window_count=total,
        available=False,
        unavailable_reason=reason,
        windows_overlap=windows_overlap,
    )


def personal_baseline_statistics(
    values: pd.DataFrame,
    *,
    subject_id: str,
    fold_index: int,
    columns: Sequence[str],
    catalog: FeatureCatalog,
    source_window_ids: Sequence[str],
    minimum_samples: int,
    zero_variance_epsilon: float,
) -> tuple[PersonalBaselineStatistics, ...]:
    """Estimate one subject's personal baseline from the supplied rows only.

    ``values`` must already be restricted to the subject's permitted
    calibration windows: this function has no way to tell which rows it was
    handed, so the caller carries the ordering obligation and
    ``source_window_ids`` records what it used.

    Raises
    ------
    PersonalizationError
        If any column must not be personalised.
    """
    assert_personalizable(columns)
    records: list[PersonalBaselineStatistics] = []
    sample_count = len(values)
    for column in columns:
        entry = catalog.get(str(column).removeprefix(FEATURE_PREFIX))
        modality = _fusion_modality_of(entry.modality)
        if modality is None:  # pragma: no cover - filtered by personalizable_columns
            continue
        series = (
            pd.to_numeric(values[column], errors="coerce")
            if column in values.columns
            else pd.Series(dtype="float64")
        )
        finite = series.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        observed = float(np.std(finite, ddof=0)) if finite.size else None

        normalized = True
        reason: str | None = None
        mean = 0.0
        scale = 1.0
        scale_source = "calibration_standard_deviation"
        if finite.size < minimum_samples:
            normalized = False
            scale_source = "identity_insufficient_evidence"
            reason = (
                f"{finite.size} finite calibration value(s) are available for "
                f"{column!r}, fewer than the {minimum_samples} required; the "
                "feature is passed through unchanged rather than centred on "
                "one reading"
            )
        elif observed is not None and observed <= zero_variance_epsilon:
            mean = float(np.mean(finite))
            scale_source = "unit_scale_zero_variance"
        else:
            mean = float(np.mean(finite))
            scale = float(observed if observed is not None else 1.0)

        records.append(
            PersonalBaselineStatistics(
                subject_id=subject_id,
                fold_index=fold_index,
                column=str(column),
                feature_name=entry.canonical_name,
                modality=modality,
                unit=entry.unit,
                normalized=normalized,
                unavailable_reason=reason,
                calibration_sample_count=sample_count,
                finite_sample_count=int(finite.size),
                mean=mean,
                scale=scale,
                observed_standard_deviation=observed,
                scale_source=scale_source,
                source_window_ids=tuple(str(v) for v in source_window_ids),
            )
        )
    return tuple(records)


def _fusion_modality_of(modality: object) -> FusionModality | None:
    for fusion_modality, feature_modality in FEATURE_MODALITY_OF.items():
        if feature_modality is modality:
            return fusion_modality
    return None


def apply_personal_baseline(
    frame: pd.DataFrame,
    statistics: Sequence[PersonalBaselineStatistics],
) -> pd.DataFrame:
    """Apply ``z = (x - mean) / scale`` to the recorded columns.

    A missing measurement stays missing.  A column with no recorded
    statistic is left untouched, which is how availability flags, modality
    flags, and quality columns pass through unchanged.
    """
    transformed = frame.copy()
    for record in statistics:
        column = record.column
        if column not in transformed.columns:
            continue
        series = pd.to_numeric(transformed[column], errors="coerce")
        transformed[column] = (series - record.mean) / record.scale
    return transformed


def regression_correction(
    *,
    subject_id: str,
    fold_index: int,
    method: PersonalizationMethod,
    calibration_window_ids: Sequence[str],
    calibration_targets: Sequence[float],
    population_predictions: Sequence[float],
    minimum_windows: int,
) -> PersonalizationCorrection:
    """Fit the documented per-subject bias correction.

    ``b_s = mean(y_calibration - y_population_prediction)`` and
    ``y_personalized = y_population_prediction + b_s``.  Only calibration
    labels take part; no evaluation label is visible to this function.
    """
    targets = np.asarray(list(calibration_targets), dtype=float)
    predictions = np.asarray(list(population_predictions), dtype=float)
    usable = np.isfinite(targets) & np.isfinite(predictions)
    count = int(usable.sum())
    window_ids = tuple(str(v) for v in calibration_window_ids)
    recorded = {
        window_ids[index]: f"{float(targets[index]):.12g}"
        for index in range(min(len(window_ids), targets.size))
        if np.isfinite(targets[index])
    }

    if count < minimum_windows:
        return PersonalizationCorrection(
            subject_id=subject_id,
            fold_index=fold_index,
            method=method,
            task_type=TaskType.REGRESSION,
            available=False,
            unavailable_reason=(
                f"{count} labelled calibration window(s) with a finite "
                f"population prediction are available, fewer than the "
                f"{minimum_windows} required; the subject falls back to the "
                "population model rather than being corrected from thin "
                "evidence"
            ),
            calibration_sample_count=count,
            calibration_window_ids=window_ids,
            calibration_targets=recorded,
        )

    bias = float(np.mean(targets[usable] - predictions[usable]))
    if not math.isfinite(bias):  # pragma: no cover - guarded by the finite mask
        return PersonalizationCorrection(
            subject_id=subject_id,
            fold_index=fold_index,
            method=method,
            task_type=TaskType.REGRESSION,
            available=False,
            unavailable_reason="the estimated bias correction is not finite",
            calibration_sample_count=count,
            calibration_window_ids=window_ids,
            calibration_targets=recorded,
        )
    return PersonalizationCorrection(
        subject_id=subject_id,
        fold_index=fold_index,
        method=method,
        task_type=TaskType.REGRESSION,
        available=True,
        supervised=True,
        calibration_sample_count=count,
        calibration_window_ids=window_ids,
        calibration_targets=recorded,
        bias=bias,
        equation=REGRESSION_CORRECTION_EQUATION,
    )


def apply_regression_correction(value: float, bias: float) -> float:
    """``y_personalized = y_population_prediction + b_s``."""
    return float(value) + float(bias)


def classification_correction(
    *,
    subject_id: str,
    fold_index: int,
    method: PersonalizationMethod,
    calibration_window_ids: Sequence[str],
    calibration_labels: Sequence[str],
    population_probabilities: np.ndarray,
    vocabulary: Sequence[str],
    smoothing: float,
    shrinkage_constant: float,
    minimum_windows: int,
    minimum_classes: int,
) -> PersonalizationCorrection:
    """Fit the documented regularised per-subject log-odds shift.

    See :data:`~engagevr.schemas.personalization.CLASSIFICATION_CORRECTION_EQUATION`
    for the exact arithmetic.  Only calibration labels take part; no
    evaluation label is visible to this function.  The shift is exactly zero
    when the subject's calibration labels match what the population model
    predicted on average, and it is shrunk toward zero in proportion to how
    little calibration evidence there is.
    """
    labels = [str(v) for v in calibration_labels]
    window_ids = tuple(str(v) for v in calibration_window_ids)
    vocabulary = tuple(str(v) for v in vocabulary)
    recorded = {
        window_ids[index]: labels[index]
        for index in range(min(len(window_ids), len(labels)))
    }
    support = {label: labels.count(label) for label in vocabulary}
    count = len(labels)

    def _refuse(reason: str) -> PersonalizationCorrection:
        return PersonalizationCorrection(
            subject_id=subject_id,
            fold_index=fold_index,
            method=method,
            task_type=TaskType.CLASSIFICATION,
            available=False,
            unavailable_reason=reason,
            calibration_sample_count=count,
            calibration_window_ids=window_ids,
            calibration_targets=recorded,
            calibration_class_support=support,
        )

    if count < minimum_windows:
        return _refuse(
            f"{count} labelled calibration window(s) are available, fewer than "
            f"the {minimum_windows} required; the subject falls back to the "
            "population model rather than being corrected from thin evidence"
        )
    unknown = sorted({label for label in labels if label not in set(vocabulary)})
    if unknown:
        return _refuse(
            f"calibration label(s) {unknown} are outside the declared class "
            f"vocabulary {list(vocabulary)}"
        )
    present = sorted(label for label, n in support.items() if n > 0)
    if len(present) < minimum_classes:
        return _refuse(
            f"the calibration labels contain {len(present)} distinct class(es) "
            f"({present}), fewer than the {minimum_classes} required. With too "
            "few classes the shift would be driven entirely by the absence of "
            "the others, so the subject falls back to the population model"
        )
    probabilities = np.asarray(population_probabilities, dtype=float)
    if probabilities.shape != (count, len(vocabulary)):
        return _refuse(
            f"the population probability matrix has shape "
            f"{probabilities.shape}, expected {(count, len(vocabulary))}"
        )
    if not np.isfinite(probabilities).all():
        return _refuse("the population probability matrix contains a non-finite value")

    k = len(vocabulary)
    denominator = count + smoothing * k
    shrinkage = count / (count + shrinkage_constant)
    shift: dict[str, float] = {}
    for index, label in enumerate(vocabulary):
        observed = (support[label] + smoothing) / denominator
        expected = (float(probabilities[:, index].sum()) + smoothing) / denominator
        delta = shrinkage * (math.log(observed) - math.log(expected))
        if not math.isfinite(delta):  # pragma: no cover - both terms are positive
            return _refuse(f"the log-odds shift for class {label!r} is not finite")
        shift[label] = float(delta)

    return PersonalizationCorrection(
        subject_id=subject_id,
        fold_index=fold_index,
        method=method,
        task_type=TaskType.CLASSIFICATION,
        available=True,
        supervised=True,
        calibration_sample_count=count,
        calibration_window_ids=window_ids,
        calibration_targets=recorded,
        calibration_class_support=support,
        log_odds_shift=shift,
        shrinkage=float(shrinkage),
        smoothing=float(smoothing),
        equation=CLASSIFICATION_CORRECTION_EQUATION,
    )


def apply_classification_correction(
    probabilities: np.ndarray,
    shift: Mapping[str, float],
    vocabulary: Sequence[str],
) -> np.ndarray:
    """Apply the per-class log-odds shift and renormalise each row.

    Raises
    ------
    PersonalizationError
        If a row cannot be renormalised, which would mean emitting something
        that is not a distribution.
    """
    matrix = np.asarray(probabilities, dtype=float)
    deltas = np.asarray([float(shift.get(str(label), 0.0)) for label in vocabulary])
    # Subtracting the maximum before exponentiating keeps the factors in a
    # representable range; it cancels in the renormalisation below.
    factors = np.exp(deltas - deltas.max())
    weighted = matrix * factors
    totals = weighted.sum(axis=1, keepdims=True)
    if not np.isfinite(weighted).all() or (totals <= 0.0).any():
        raise PersonalizationError(
            "a personalized probability row could not be renormalised; a "
            "vector that does not sum to one is not a distribution and is "
            "never emitted"
        )
    return np.asarray(weighted / totals, dtype=float)


def build_personalization_run_id(
    *,
    target_name: str,
    task_type: str,
    evaluation_mode: str,
    dataset_fingerprint: str,
    split_manifest_fingerprint: str,
    random_seed: int,
    configuration: PersonalizationConfiguration,
    calibration_method: str,
    engagevr_version: str,
) -> str:
    """Deterministic identifier for one personalization run.

    Every input that changes what the run computes participates.  No wall
    clock and no random component does, so re-running an identical
    configuration reproduces the identifier instead of accumulating
    near-duplicate directories.
    """
    payload = {
        "kind": "personalization",
        "target_name": target_name,
        "task_type": task_type,
        "evaluation_mode": evaluation_mode,
        "dataset_fingerprint": dataset_fingerprint,
        "split_manifest_fingerprint": split_manifest_fingerprint,
        "random_seed": random_seed,
        "calibration_method": calibration_method,
        "method": configuration.method.value,
        "modalities": [m.value for m in configuration.modalities],
        "calibration_windows": configuration.calibration_windows,
        "minimum_calibration_windows": configuration.minimum_calibration_windows,
        "minimum_evaluation_windows": configuration.minimum_evaluation_windows,
        "minimum_calibration_classes": configuration.minimum_calibration_classes,
        "minimum_baseline_samples": configuration.minimum_baseline_samples,
        "zero_variance_epsilon": configuration.zero_variance_epsilon,
        "classification_smoothing": configuration.classification_smoothing,
        "classification_shrinkage_constant": (
            configuration.classification_shrinkage_constant
        ),
        "population_model_classification": (
            configuration.population_model_classification
        ),
        "population_model_regression": configuration.population_model_regression,
        "use_calibrated_population_model": (
            configuration.use_calibrated_population_model
        ),
        "include_modality_quality": configuration.include_modality_quality,
        "engagevr_version": engagevr_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    short_mode = "sci" if evaluation_mode == "scientific" else "selfcheck"
    return f"{target_name}-personalization-{short_mode}-{digest}"


__all__ = [
    "FORBIDDEN_COLUMNS",
    "FORBIDDEN_COLUMN_PREFIXES",
    "PersonalizationError",
    "SubjectWindow",
    "apply_classification_correction",
    "apply_personal_baseline",
    "apply_regression_correction",
    "assert_personalizable",
    "build_calibration_split",
    "build_personalization_run_id",
    "classification_correction",
    "parse_method",
    "personal_baseline_statistics",
    "personalizable_columns",
    "regression_correction",
    "subject_windows",
]
