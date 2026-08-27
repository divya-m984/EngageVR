"""Dataset provenance and measurement-quality views.

Two pages are built here and they answer different questions.

*Dataset and provenance* answers "where did these rows come from, and how
were they split".  Every field is read from ``dataset.json`` or
``splits.json``.  A count this repository never recorded is *Unavailable*;
it is not recovered from a directory name and it is not estimated.

*Signal and feature quality* answers "could the measurement be taken at
all".  Nothing on that page is an engagement or cognitive-load quantity,
and nothing on it is a confidence score.  A window with a low
availability percentage is a window that was hard to measure.  That is
the only thing it is, and the page says so beside every chart.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from engagevr.dashboard import formatting as fmt
from engagevr.dashboard.aggregation import histogram_series
from engagevr.dashboard.loaders import (
    DEFAULT_MAX_TABLE_ROWS,
    missing_artifact_warning,
    try_document,
    unreadable_artifact_warning,
)
from engagevr.dashboard.presentation import SIGNAL_QUALITY_NOTE
from engagevr.schemas.dashboard import (
    DashboardRunFamily,
    DashboardRunSummary,
    DashboardWarning,
    DashboardWarningLevel,
    DatasetProvenanceDashboardData,
    LabelledChart,
    MetricKind,
    SignalQualityDashboardData,
)

#: Families that write ``dataset.json`` and ``splits.json``.
_DATASET_FAMILIES = (
    DashboardRunFamily.BASELINE,
    DashboardRunFamily.FUSION,
    DashboardRunFamily.PERSONALIZATION,
    DashboardRunFamily.UNCERTAINTY,
)

#: Why an adaptation run has no dataset page.
_NO_DATASET = (
    "A Milestone 8 adaptation run evaluates a deterministic controller over "
    "hand-written scenario windows. It reads no feature dataset and writes "
    "no dataset.json, so there is no dataset provenance to display. The "
    "scenarios are on the Adaptive environment page."
)


def load_dataset_provenance(
    run: DashboardRunSummary, *, max_rows: int = DEFAULT_MAX_TABLE_ROWS
) -> DatasetProvenanceDashboardData:
    """Build the dataset-and-provenance view for one run."""
    provenance = run.provenance
    warnings: list[DashboardWarning] = []

    if run.provenance.family not in _DATASET_FAMILIES:
        return DatasetProvenanceDashboardData(
            provenance=provenance, unavailable_reason=_NO_DATASET
        )

    dataset, dataset_error = try_document(run, "dataset.json")
    if dataset is None:
        return DatasetProvenanceDashboardData(
            provenance=provenance,
            warnings=(
                unreadable_artifact_warning(run, "dataset.json", str(dataset_error)),
            ),
            unavailable_reason=(
                f"dataset.json is unavailable for this run: {dataset_error}"
            ),
        )

    rows = [
        ("Dataset schema version", fmt.text(dataset.get("dataset_schema_version"))),
        (
            "Feature catalog version",
            fmt.text(dataset.get("feature_catalog_version")),
        ),
        ("Dataset fingerprint", fmt.text(dataset.get("dataset_fingerprint"))),
        ("Fingerprint algorithm", fmt.text(dataset.get("fingerprint_algorithm"))),
        ("Window (row) count", fmt.text(dataset.get("row_count"))),
        ("Feature count", fmt.text(dataset.get("feature_count"))),
        ("Subject count", fmt.text(dataset.get("subject_count"))),
        ("Session count", fmt.text(dataset.get("session_count"))),
        (
            "Window duration (s)",
            fmt.text(dataset.get("window_duration_seconds")),
        ),
        ("Window step (s)", fmt.text(dataset.get("window_step_seconds"))),
        ("Windows overlap", fmt.text(dataset.get("windows_overlap"))),
        (
            "Scientifically eligible",
            fmt.text(dataset.get("scientific_evaluation_eligible")),
        ),
        ("Created (display only)", fmt.text(dataset.get("created_at_utc"))),
    ]
    dataset_table = fmt.build_table(
        title="Dataset provenance",
        columns=("field", "value"),
        rows=rows,
        source_artifact="dataset.json",
        caption=(
            "Every value is read from dataset.json. A field the artifact does "
            "not record is shown as Unavailable and is never derived from a "
            "file or directory name."
        ),
    )

    target_table = None
    targets = dataset.get("targets")
    if isinstance(targets, list) and targets:
        target_rows = []
        for entry in targets:
            if not isinstance(entry, dict):
                continue
            distribution = entry.get("class_distribution")
            target_rows.append(
                (
                    fmt.text(entry.get("target_name")),
                    fmt.text(entry.get("task_type")),
                    fmt.text(entry.get("labelled_row_count")),
                    fmt.text(
                        ", ".join(f"{k}={v}" for k, v in sorted(distribution.items()))
                        if isinstance(distribution, dict) and distribution
                        else None
                    ),
                )
            )
        target_table = fmt.build_table(
            title="Targets recorded in this dataset",
            columns=(
                "target",
                "task type",
                "labelled windows",
                "class distribution",
            ),
            rows=target_rows,
            source_artifact="dataset.json",
            caption=(
                "These targets were generated by this repository. No "
                "validated participant label, questionnaire score, or expert "
                "annotation exists in this project."
            ),
        )

    split_table, fold_table, audit_passed, audit_notes, split_warning = _splits(
        run, max_rows=max_rows
    )
    if split_warning is not None:
        warnings.append(split_warning)

    counts = dataset.get("data_source_counts")
    source_counts: tuple[tuple[str, int], ...] = ()
    if isinstance(counts, dict):
        source_counts = tuple(
            (str(k), int(v)) for k, v in sorted(counts.items()) if isinstance(v, int)
        )

    return DatasetProvenanceDashboardData(
        provenance=provenance,
        dataset_table=dataset_table,
        target_table=target_table,
        split_table=split_table,
        fold_table=fold_table,
        data_source_counts=source_counts,
        split_audit_passed=audit_passed,
        split_audit_notes=audit_notes,
        warnings=tuple(warnings),
    )


def _splits(
    run: DashboardRunSummary, *, max_rows: int
) -> tuple[Any, Any, bool | None, tuple[str, ...], DashboardWarning | None]:
    """Split strategy, fold table, and the recorded leakage-audit result."""
    document, error = try_document(run, "splits.json")
    if document is None:
        return (
            None,
            None,
            None,
            (),
            unreadable_artifact_warning(run, "splits.json", str(error)),
        )
    audit_passed = document.get("audit_passed")
    notes = document.get("audit_notes")
    audit_notes = tuple(str(note) for note in notes) if isinstance(notes, list) else ()
    split_table = fmt.build_table(
        title="Split strategy",
        columns=("field", "value"),
        rows=[
            ("Strategy", fmt.text(document.get("strategy"))),
            ("Why this strategy", fmt.text(document.get("strategy_reason"))),
            ("Group field", fmt.text(document.get("group_field"))),
            ("Why this group field", fmt.text(document.get("group_field_reason"))),
            ("Independent groups", fmt.text(document.get("group_count"))),
            ("Folds", fmt.text(document.get("n_splits"))),
            ("Random seed", fmt.text(document.get("random_seed"))),
            (
                "Calibration group fraction",
                fmt.text(document.get("calibration_group_fraction")),
            ),
            (
                "Overlap / leakage audit passed",
                fmt.text(audit_passed),
            ),
        ],
        source_artifact="splits.json",
        caption=(
            "The audit result is the one the run recorded. It is not recomputed here."
        ),
    )

    folds = document.get("folds")
    fold_table = None
    if isinstance(folds, list) and folds:
        fold_rows = []
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            fold_rows.append(
                (
                    fmt.text(fold.get("fold_index")),
                    fmt.text(len(fold.get("train_groups") or ())),
                    fmt.text(len(fold.get("calibration_groups") or ())),
                    fmt.text(len(fold.get("test_groups") or ())),
                    fmt.text(fold.get("train_row_count")),
                    fmt.text(fold.get("calibration_row_count")),
                    fmt.text(fold.get("test_row_count")),
                    fmt.text(fold.get("valid")),
                    fmt.text(fold.get("invalid_reason")),
                )
            )
        fold_table = fmt.build_table(
            title="Fold assignments",
            columns=(
                "fold",
                "train groups",
                "calibration groups",
                "test groups",
                "train windows",
                "calibration windows",
                "test windows",
                "valid",
                "invalid reason",
            ),
            rows=fold_rows,
            source_artifact="splits.json",
            max_rows=max_rows,
            caption=(
                "Grouped folds. Calibration groups are carved out of the "
                "training groups, never out of the test groups."
            ),
        )
    return (
        split_table,
        fold_table,
        bool(audit_passed) if audit_passed is not None else None,
        audit_notes,
        None,
    )


def load_signal_quality(
    run: DashboardRunSummary, *, max_rows: int = DEFAULT_MAX_TABLE_ROWS
) -> SignalQualityDashboardData:
    """Build the measurement-quality view for one run.

    Availability and missingness are read from ``dataset.json``.  Nothing
    is derived from a confidence score: if the run recorded no quality
    information, the page says so rather than inventing it from
    something that happens to be a number in the same range.
    """
    provenance = run.provenance
    if run.provenance.family not in _DATASET_FAMILIES:
        return SignalQualityDashboardData(
            provenance=provenance,
            unavailable_reason=(
                "This run family records no feature dataset, so it records no "
                "measurement quality. " + SIGNAL_QUALITY_NOTE
            ),
        )

    dataset, error = try_document(run, "dataset.json")
    if dataset is None:
        return SignalQualityDashboardData(
            provenance=provenance,
            warnings=(unreadable_artifact_warning(run, "dataset.json", str(error)),),
            unavailable_reason=(
                f"dataset.json is unavailable, so no measurement-quality "
                f"information can be shown for this run: {error}"
            ),
        )

    missingness = dataset.get("missingness")
    if not isinstance(missingness, list) or not missingness:
        return SignalQualityDashboardData(
            provenance=provenance,
            warnings=(missing_artifact_warning(run, "dataset.json missingness"),),
            unavailable_reason=(
                "This run's dataset.json records no per-feature missingness, "
                "so feature availability is unavailable. It has not been "
                "reconstructed from anything else. " + SIGNAL_QUALITY_NOTE
            ),
        )

    rows = []
    percentages: list[float] = []
    for entry in missingness:
        if not isinstance(entry, dict):
            continue
        percent = entry.get("missing_pct")
        if isinstance(percent, int | float) and not isinstance(percent, bool):
            percentages.append(float(percent))
        rows.append(
            (
                fmt.text(entry.get("feature_name")),
                fmt.text(entry.get("missing_count")),
                fmt.optional_percentage(percent, already_percent=True),
            )
        )
    rows.sort(key=lambda row: row[0])

    missing_table = fmt.build_table(
        title="Per-feature missingness",
        columns=("feature", "windows with no value", "missing %"),
        rows=rows,
        source_artifact="dataset.json",
        max_rows=max_rows,
        caption=(
            "The fraction of windows in which this feature could not be "
            "computed. " + SIGNAL_QUALITY_NOTE
        ),
    )

    modality_table = _modality_availability(dataset)

    series = histogram_series("features", percentages, bins=20, lower=0.0)
    chart = LabelledChart(
        title="Distribution of per-feature missingness",
        subtitle="Measurement availability only",
        x_axis_label="missing windows (%)",
        y_axis_label="number of features",
        series=(series,) if series is not None else (),
        x_axis_note=SIGNAL_QUALITY_NOTE,
        source_artifact="dataset.json",
        unavailable_reason=(
            None
            if series is not None
            else "no per-feature missingness percentage was recorded"
        ),
    )

    overall = fmt.metric(
        "Overall missing feature values",
        _percent_fraction(dataset.get("overall_missing_pct")),
        kind=MetricKind.PERCENTAGE,
        source_artifact="dataset.json",
    )

    return SignalQualityDashboardData(
        provenance=provenance,
        modality_availability_table=modality_table,
        missing_feature_table=missing_table,
        missingness_chart=chart,
        overall_missing_percentage=overall,
        warnings=(),
    )


def _modality_availability(dataset: Mapping[str, Any]) -> Any:
    """Which modalities the dataset's columns cover.

    Derived from the recorded column order, which lists a
    ``modality_available__<name>`` column for every modality the feature
    catalog defines.  This is a statement about columns, not about how
    good any measurement was.
    """
    columns = dataset.get("column_order")
    if not isinstance(columns, list):
        return None
    prefix = "modality_available__"
    modalities = sorted(
        str(column)[len(prefix) :]
        for column in columns
        if isinstance(column, str) and column.startswith(prefix)
    )
    if not modalities:
        return None
    quality_prefix = "modality_quality__"
    recorded_quality = {
        str(column)[len(quality_prefix) :]
        for column in columns
        if isinstance(column, str) and column.startswith(quality_prefix)
    }
    return fmt.build_table(
        title="Modality coverage in this dataset",
        columns=("modality", "availability column", "quality column"),
        rows=[
            (
                modality,
                "present",
                "present" if modality in recorded_quality else "Unavailable",
            )
            for modality in modalities
        ],
        source_artifact="dataset.json",
        caption=(
            "Availability says whether a modality produced usable values in a "
            "window. Quality says how reliable they were. Neither is "
            "engagement, cognitive load, or model confidence."
        ),
    )


def _percent_fraction(value: object) -> float | None:
    """Convert a stored 0-100 percentage to a 0-1 fraction."""
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) / 100.0


def dataset_source_warning(
    data: DatasetProvenanceDashboardData,
) -> DashboardWarning | None:
    """Warn when a dataset mixes data sources.

    A mixed dataset cannot be presented under a single provenance
    heading, and the strictest reading is the one that applies.
    """
    if len(data.data_source_counts) <= 1:
        return None
    names = ", ".join(name for name, _ in data.data_source_counts)
    return DashboardWarning(
        level=DashboardWarningLevel.WARNING,
        message=(
            f"this dataset mixes the data sources {names}. Any view derived "
            "from it inherits the strictest provenance present."
        ),
        subject=data.provenance.run_directory,
    )


__all__ = [
    "dataset_source_warning",
    "load_dataset_provenance",
    "load_signal_quality",
]
