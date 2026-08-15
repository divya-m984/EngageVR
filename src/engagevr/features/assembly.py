"""Assemble feature windows into a Parquet dataset with a deterministic fingerprint.

Layout written for a dataset at ``<stem>.parquet``::

    <stem>.parquet               the table
    <stem>.metadata.json         provenance, fingerprint, disclaimers
    <stem>.feature_catalog.json  the catalog the dataset was built against

The metadata and catalog are plain JSON on purpose: the origin of a
dataset must be inspectable without opening the table, and certainly
without loading a model file.

Fingerprint
-----------
``dataset_fingerprint`` is a SHA-256 over a canonical UTF-8 rendering of:
the schema versions, the catalog version, the exact column order, the
window geometry, and every row's values in that column order, with rows
sorted by ``(session_id, window_index, window_id)``.

Wall-clock values are **excluded** from that rendering.  Two builds of
equivalent data must fingerprint identically, and a creation timestamp
would guarantee they never do.  Anything that changes row content, the
schema, the column order, the target set, or the window geometry changes
the fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from engagevr.features.catalog import FEATURE_CATALOG_VERSION, get_catalog
from engagevr.features.windowing import utc_now
from engagevr.schemas.features import (
    FEATURE_SCHEMA_VERSION,
    SYNTHETIC_DATASET_DISCLAIMER,
    DatasetMetadata,
    FeatureCatalog,
    FeatureDtype,
    FeatureWindow,
    MissingnessSummary,
    TargetSummary,
    availability_column,
    feature_column,
    modality_available_column,
    modality_quality_column,
    target_column,
    target_meta_column,
)
from engagevr.schemas.session import DataSource
from engagevr.schemas.targets import (
    TARGET_DISCLAIMER,
    TargetName,
    TaskType,
    get_target_spec,
)

#: Timestamp unit used for every temporal column.
_TIMESTAMP_TYPE = pa.timestamp("us", tz="UTC")


class DatasetAssemblyError(ValueError):
    """A dataset could not be assembled from the supplied rows."""


def dataset_columns(
    catalog: FeatureCatalog,
    target_names: Sequence[TargetName],
) -> tuple[str, ...]:
    """Canonical column order for a dataset.

    Order is deterministic and is part of the fingerprint, so a
    reordering is a detectable schema change rather than a silent one.
    """
    columns: list[str] = [
        "window_id",
        "session_id",
        "subject_id",
        "subject_kind",
        "experiment_condition",
        "data_source",
        "synthetic_label",
        "window_index",
        "window_start_utc",
        "window_end_utc",
        "window_start_monotonic_seconds",
        "window_end_monotonic_seconds",
        "window_duration_seconds",
        "window_step_seconds",
        "windows_overlap",
        "feature_schema_version",
        "feature_catalog_version",
    ]
    columns.extend(feature_column(name) for name in catalog.names())
    columns.extend(availability_column(name) for name in catalog.names())
    for modality in catalog.modalities():
        columns.append(modality_available_column(modality.value))
    for modality in catalog.modalities():
        columns.append(modality_quality_column(modality.value))
    for target in target_names:
        columns.append(target_column(target))
        columns.extend(
            target_meta_column(target, field)
            for field in (
                "task_type",
                "source_type",
                "source_instrument",
                "observed_at_utc",
                "interval_start_utc",
                "interval_end_utc",
                "synthetic_label",
                "provenance_notes",
                "scientific_evaluation_permitted",
            )
        )
    return tuple(columns)


def _arrow_schema(
    catalog: FeatureCatalog,
    target_names: Sequence[TargetName],
) -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("window_id", pa.string(), nullable=False),
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("subject_id", pa.string(), nullable=False),
        pa.field("subject_kind", pa.string(), nullable=False),
        pa.field("experiment_condition", pa.string(), nullable=False),
        pa.field("data_source", pa.string(), nullable=False),
        pa.field("synthetic_label", pa.string(), nullable=True),
        pa.field("window_index", pa.int64(), nullable=False),
        pa.field("window_start_utc", _TIMESTAMP_TYPE, nullable=False),
        pa.field("window_end_utc", _TIMESTAMP_TYPE, nullable=False),
        pa.field("window_start_monotonic_seconds", pa.float64(), nullable=True),
        pa.field("window_end_monotonic_seconds", pa.float64(), nullable=True),
        pa.field("window_duration_seconds", pa.float64(), nullable=False),
        pa.field("window_step_seconds", pa.float64(), nullable=False),
        pa.field("windows_overlap", pa.bool_(), nullable=False),
        pa.field("feature_schema_version", pa.string(), nullable=False),
        pa.field("feature_catalog_version", pa.string(), nullable=False),
    ]
    for entry in catalog.entries:
        # Numeric features are stored as nullable float64 regardless of the
        # catalog's semantic dtype: a count that could not be observed must
        # be null, and an integer column with a null is not representable
        # without either a mask or a sentinel. The catalog remains the
        # authority on what the value means.
        arrow_type = (
            pa.string() if entry.dtype is FeatureDtype.CATEGORY else pa.float64()
        )
        fields.append(
            pa.field(feature_column(entry.canonical_name), arrow_type, nullable=True)
        )
    for entry in catalog.entries:
        fields.append(
            pa.field(
                availability_column(entry.canonical_name), pa.bool_(), nullable=False
            )
        )
    for modality in catalog.modalities():
        fields.append(
            pa.field(
                modality_available_column(modality.value), pa.bool_(), nullable=False
            )
        )
    for modality in catalog.modalities():
        fields.append(
            pa.field(
                modality_quality_column(modality.value), pa.float64(), nullable=True
            )
        )
    for target in target_names:
        spec = get_target_spec(target)
        value_type = (
            pa.string() if spec.task_type is TaskType.CLASSIFICATION else pa.float64()
        )
        fields.append(pa.field(target_column(target), value_type, nullable=True))
        # Field order must match dataset_columns() exactly: the column order
        # is part of the dataset fingerprint.
        meta_types = {
            "task_type": pa.string(),
            "source_type": pa.string(),
            "source_instrument": pa.string(),
            "observed_at_utc": _TIMESTAMP_TYPE,
            "interval_start_utc": _TIMESTAMP_TYPE,
            "interval_end_utc": _TIMESTAMP_TYPE,
            "synthetic_label": pa.string(),
            "provenance_notes": pa.string(),
            "scientific_evaluation_permitted": pa.bool_(),
        }
        for field_name, field_type in meta_types.items():
            fields.append(
                pa.field(
                    target_meta_column(target, field_name),
                    field_type,
                    nullable=True,
                )
            )
    schema = pa.schema(fields)
    ordered = dataset_columns(catalog, target_names)
    if tuple(schema.names) != ordered:
        raise DatasetAssemblyError(  # pragma: no cover - guards a coding error
            "arrow schema column order does not match dataset_columns()"
        )
    return schema


def sort_rows(rows: Sequence[FeatureWindow]) -> tuple[FeatureWindow, ...]:
    """Rows in canonical order: session, then window index, then window id."""
    return tuple(
        sorted(rows, key=lambda r: (r.session_id, r.window_index, r.window_id))
    )


def _row_cells(
    row: FeatureWindow,
    catalog: FeatureCatalog,
    target_names: Sequence[TargetName],
) -> dict[str, Any]:
    cells: dict[str, Any] = {
        "window_id": row.window_id,
        "session_id": row.session_id,
        "subject_id": row.subject_id,
        "subject_kind": row.subject_kind.value,
        "experiment_condition": row.experiment_condition.value,
        "data_source": row.data_source.value,
        "synthetic_label": row.synthetic_label,
        "window_index": row.window_index,
        "window_start_utc": row.window_start_utc,
        "window_end_utc": row.window_end_utc,
        "window_start_monotonic_seconds": row.window_start_monotonic_seconds,
        "window_end_monotonic_seconds": row.window_end_monotonic_seconds,
        "window_duration_seconds": row.window_duration_seconds,
        "window_step_seconds": row.window_step_seconds,
        "windows_overlap": row.windows_overlap,
        "feature_schema_version": row.feature_schema_version,
        "feature_catalog_version": row.feature_catalog_version,
    }
    for entry in catalog.entries:
        name = entry.canonical_name
        if entry.dtype is FeatureDtype.CATEGORY:
            cells[feature_column(name)] = row.categorical_features.get(name)
        else:
            cells[feature_column(name)] = row.features.get(name)
        cells[availability_column(name)] = bool(row.feature_available.get(name, False))
    for modality in catalog.modalities():
        cells[modality_available_column(modality.value)] = bool(
            row.modality_available.get(modality.value, False)
        )
        cells[modality_quality_column(modality.value)] = row.modality_quality.get(
            modality.value
        )
    for target in target_names:
        value = row.targets.get(target.value)
        spec = get_target_spec(target)
        if value is None:
            cells[target_column(target)] = None
            for field_name in (
                "task_type",
                "source_type",
                "source_instrument",
                "observed_at_utc",
                "interval_start_utc",
                "interval_end_utc",
                "synthetic_label",
                "provenance_notes",
                "scientific_evaluation_permitted",
            ):
                cells[target_meta_column(target, field_name)] = None
            continue
        cells[target_column(target)] = (
            value.class_value
            if spec.task_type is TaskType.CLASSIFICATION
            else value.numeric_value
        )
        cells[target_meta_column(target, "task_type")] = value.task_type.value
        cells[target_meta_column(target, "source_type")] = value.source_type
        cells[target_meta_column(target, "source_instrument")] = value.source_instrument
        cells[target_meta_column(target, "observed_at_utc")] = value.observed_at_utc
        cells[target_meta_column(target, "interval_start_utc")] = (
            value.interval_start_utc
        )
        cells[target_meta_column(target, "interval_end_utc")] = value.interval_end_utc
        cells[target_meta_column(target, "synthetic_label")] = value.synthetic_label
        cells[target_meta_column(target, "provenance_notes")] = value.provenance_notes
        cells[target_meta_column(target, "scientific_evaluation_permitted")] = (
            value.scientific_evaluation_permitted
        )
    return cells


def rows_to_table(
    rows: Sequence[FeatureWindow],
    catalog: FeatureCatalog,
    target_names: Sequence[TargetName],
) -> pa.Table:
    """Build an Arrow table from feature windows in canonical order."""
    schema = _arrow_schema(catalog, target_names)
    ordered_rows = sort_rows(rows)
    cells = [_row_cells(row, catalog, target_names) for row in ordered_rows]
    columns = {
        name: [cell.get(name) for cell in cells]
        for name in dataset_columns(catalog, target_names)
    }
    return pa.Table.from_pydict(columns, schema=schema)


def _canonical_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        # repr round-trips exactly for IEEE-754 doubles, so a value that
        # differs in the last bit produces a different fingerprint.
        return repr(value)
    if isinstance(value, int):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def compute_fingerprint(
    rows: Sequence[FeatureWindow],
    catalog: FeatureCatalog,
    target_names: Sequence[TargetName],
    *,
    window_duration_seconds: float,
    window_step_seconds: float,
    windows_overlap: bool,
) -> str:
    """SHA-256 fingerprint over canonical dataset content.

    Excludes every wall-clock creation value, so two equivalent
    deterministic builds fingerprint identically.
    """
    columns = dataset_columns(catalog, target_names)
    payload = {
        "dataset_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_catalog_version": catalog.version,
        "column_order": list(columns),
        "window_duration_seconds": repr(float(window_duration_seconds)),
        "window_step_seconds": repr(float(window_step_seconds)),
        "windows_overlap": windows_overlap,
        "rows": [
            [_canonical_value(cells[column]) for column in columns]
            for cells in (
                _row_cells(row, catalog, target_names) for row in sort_rows(rows)
            )
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=False, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_dataset_metadata(
    rows: Sequence[FeatureWindow],
    catalog: FeatureCatalog,
    target_names: Sequence[TargetName],
    *,
    window_duration_seconds: float,
    window_step_seconds: float,
    windows_overlap: bool,
    creation_configuration: dict[str, object] | None = None,
    random_seed: int | None = None,
    input_session_fingerprints: dict[str, str] | None = None,
    created_at_utc: datetime | None = None,
) -> DatasetMetadata:
    """Produce the deterministic provenance document for ``rows``."""
    ordered = sort_rows(rows)
    fingerprint = compute_fingerprint(
        ordered,
        catalog,
        target_names,
        window_duration_seconds=window_duration_seconds,
        window_step_seconds=window_step_seconds,
        windows_overlap=windows_overlap,
    )

    data_source_counts: dict[str, int] = {}
    subject_kind_counts: dict[str, int] = {}
    condition_counts: dict[str, int] = {}
    for row in ordered:
        data_source_counts[row.data_source.value] = (
            data_source_counts.get(row.data_source.value, 0) + 1
        )
        subject_kind_counts[row.subject_kind.value] = (
            subject_kind_counts.get(row.subject_kind.value, 0) + 1
        )
        condition_counts[row.experiment_condition.value] = (
            condition_counts.get(row.experiment_condition.value, 0) + 1
        )

    missingness: list[MissingnessSummary] = []
    total_slots = 0
    total_missing = 0
    for entry in catalog.entries:
        name = entry.canonical_name
        missing = sum(
            1 for row in ordered if not row.feature_available.get(name, False)
        )
        total_slots += len(ordered)
        total_missing += missing
        missingness.append(
            MissingnessSummary(
                feature_name=name,
                missing_count=missing,
                missing_pct=(100.0 * missing / len(ordered)) if ordered else 0.0,
            )
        )

    targets = tuple(_summarise_target(ordered, target) for target in target_names)
    eligible = bool(ordered) and all(
        row.data_source is not DataSource.SYNTHETIC for row in ordered
    )
    if targets:
        eligible = eligible and all(
            summary.scientific_evaluation_permitted for summary in targets
        )

    disclaimers = [TARGET_DISCLAIMER]
    if any(row.data_source is DataSource.SYNTHETIC for row in ordered):
        disclaimers.insert(0, SYNTHETIC_DATASET_DISCLAIMER)

    return DatasetMetadata(
        dataset_schema_version=FEATURE_SCHEMA_VERSION,
        feature_catalog_version=catalog.version,
        row_count=len(ordered),
        feature_count=len(catalog.entries),
        column_order=dataset_columns(catalog, target_names),
        subject_count=len({row.subject_id for row in ordered}),
        session_count=len({row.session_id for row in ordered}),
        data_source_counts=data_source_counts,
        subject_kind_counts=subject_kind_counts,
        experiment_condition_counts=condition_counts,
        window_duration_seconds=window_duration_seconds,
        window_step_seconds=window_step_seconds,
        windows_overlap=windows_overlap,
        missingness=tuple(missingness),
        overall_missing_pct=(
            100.0 * total_missing / total_slots if total_slots else 0.0
        ),
        targets=targets,
        creation_configuration=dict(creation_configuration or {}),
        random_seed=random_seed,
        input_session_fingerprints=dict(input_session_fingerprints or {}),
        dataset_fingerprint=fingerprint,
        created_at_utc=created_at_utc if created_at_utc is not None else utc_now(),
        scientific_evaluation_eligible=eligible,
        disclaimers=tuple(disclaimers),
    )


def _summarise_target(
    rows: Sequence[FeatureWindow], target: TargetName
) -> TargetSummary:
    spec = get_target_spec(target)
    values = [row.targets.get(target.value) for row in rows]
    present = [v for v in values if v is not None]
    source_types: dict[str, int] = {}
    for value in present:
        source_types[value.source_type] = source_types.get(value.source_type, 0) + 1

    distribution: dict[str, int] | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    if spec.task_type is TaskType.CLASSIFICATION:
        distribution = {}
        for label in spec.class_vocabulary or ():
            distribution[label] = sum(1 for v in present if v.class_value == label)
    else:
        numbers = [v.numeric_value for v in present if v.numeric_value is not None]
        if numbers:
            minimum = float(min(numbers))
            maximum = float(max(numbers))
            mean = float(sum(numbers) / len(numbers))

    permitted = bool(present) and all(
        v.scientific_evaluation_permitted for v in present
    )
    return TargetSummary(
        target_name=target.value,
        task_type=spec.task_type.value,
        labelled_row_count=len(present),
        class_distribution=distribution,
        value_minimum=minimum,
        value_maximum=maximum,
        value_mean=mean,
        source_types=source_types,
        scientific_evaluation_permitted=permitted,
    )


def metadata_path(dataset_path: Path) -> Path:
    """Path of the metadata document belonging to ``dataset_path``."""
    return dataset_path.with_name(f"{dataset_path.stem}.metadata.json")


def catalog_path(dataset_path: Path) -> Path:
    """Path of the catalog snapshot belonging to ``dataset_path``."""
    return dataset_path.with_name(f"{dataset_path.stem}.feature_catalog.json")


def write_parquet_atomic(table: pa.Table, path: Path) -> Path:
    """Write ``table`` to ``path`` via a temporary file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression="snappy")
    with temporary.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def write_dataset(
    rows: Sequence[FeatureWindow],
    path: Path,
    *,
    target_names: Sequence[TargetName],
    window_duration_seconds: float,
    window_step_seconds: float,
    windows_overlap: bool,
    catalog: FeatureCatalog | None = None,
    creation_configuration: dict[str, object] | None = None,
    random_seed: int | None = None,
    input_session_fingerprints: dict[str, str] | None = None,
    created_at_utc: datetime | None = None,
) -> DatasetMetadata:
    """Write the dataset, its metadata, and its catalog snapshot.

    Raises
    ------
    DatasetAssemblyError
        If ``rows`` is empty or a row was built against a different catalog.
    """
    active = catalog if catalog is not None else get_catalog(FEATURE_CATALOG_VERSION)
    if not rows:
        raise DatasetAssemblyError("cannot write a dataset with no rows")
    mismatched = sorted(
        {
            r.feature_catalog_version
            for r in rows
            if r.feature_catalog_version != active.version
        }
    )
    if mismatched:
        raise DatasetAssemblyError(
            f"rows were built against feature catalog version(s) {mismatched} but "
            f"the dataset is being written against {active.version!r}"
        )

    table = rows_to_table(rows, active, target_names)
    metadata = build_dataset_metadata(
        rows,
        active,
        target_names,
        window_duration_seconds=window_duration_seconds,
        window_step_seconds=window_step_seconds,
        windows_overlap=windows_overlap,
        creation_configuration=creation_configuration,
        random_seed=random_seed,
        input_session_fingerprints=input_session_fingerprints,
        created_at_utc=created_at_utc,
    )

    write_parquet_atomic(table, path)
    _write_json_atomic(metadata_path(path), metadata.model_dump(mode="json"))
    _write_json_atomic(catalog_path(path), active.model_dump(mode="json"))
    return metadata


def _write_json_atomic(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    text = json.dumps(data, indent=2, default=str) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def read_dataset_table(path: Path) -> pa.Table:
    """Read a dataset table from Parquet."""
    return pq.read_table(path)


def read_dataset_metadata(path: Path) -> DatasetMetadata:
    """Read the metadata document belonging to the dataset at ``path``.

    Raises
    ------
    DatasetAssemblyError
        If the metadata document is missing. A dataset without provenance
        is not usable: nothing could establish where its rows came from.
    """
    document = metadata_path(path)
    if not document.exists():
        raise DatasetAssemblyError(
            f"dataset {path} has no metadata document at {document}. A dataset "
            "with no recorded provenance cannot be evaluated."
        )
    with document.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return DatasetMetadata.model_validate(raw)


__all__ = [
    "DatasetAssemblyError",
    "build_dataset_metadata",
    "catalog_path",
    "compute_fingerprint",
    "dataset_columns",
    "metadata_path",
    "read_dataset_metadata",
    "read_dataset_table",
    "rows_to_table",
    "sort_rows",
    "write_dataset",
    "write_parquet_atomic",
]
