"""Typed, read-only access to the artifacts of a selected run.

Everything here opens a file for reading and closes it.  There is no
write path, no append path, and no delete path in this module or in
anything that imports it, and the absence is checked by a test rather
than left to discipline.

Three deliberate choices are worth recording.

**Only JSON and Parquet are read.**  ``models/*.joblib`` are pickles and
loading one executes code contained in it.  Nothing needed to display a
run is inside an estimator, so the dashboard never opens one.

**Columns are selected, not whole tables.**  A predictions table holds
thousands of rows across many columns; a residual plot needs two of
them.  Reading the two is not an optimisation so much as a way of
keeping a page from depending on columns it does not display.

**Absence is a value.**  Every loader returns a view with an
``unavailable_reason`` rather than raising when an optional artifact is
missing.  One absent file degrades one chart and says which file it was;
it never blanks a page and never becomes a zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from engagevr.dashboard.catalogue import (
    ArtifactReadError,
    read_json_mapping,
)
from engagevr.schemas.dashboard import (
    DashboardProvenance,
    DashboardRunSummary,
    DashboardWarning,
    DashboardWarningLevel,
)

#: Rows a single table view renders before it reports truncation.
DEFAULT_MAX_TABLE_ROWS = 1000


class ArtifactColumnError(ArtifactReadError):
    """A Parquet artifact does not carry a column a view needs."""


def load_document(run: DashboardRunSummary, name: str) -> dict[str, Any]:
    """Read one JSON artifact of a run.

    Raises :class:`ArtifactReadError` when the file is absent or
    unparseable; callers convert that into a stated unavailable reason.
    """
    return read_json_mapping(Path(run.absolute_path) / name)


def try_document(
    run: DashboardRunSummary, name: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Read a JSON artifact, returning the reason instead of raising."""
    try:
        return load_document(run, name), None
    except ArtifactReadError as exc:
        return None, str(exc)


def parquet_columns(run: DashboardRunSummary, name: str) -> tuple[str, ...]:
    """Column names of a Parquet artifact, without reading any rows."""
    path = Path(run.absolute_path) / name
    if not path.is_file():
        raise ArtifactReadError(f"{name} is not present in {run.directory_name}")
    try:
        return tuple(pq.read_schema(path).names)
    except OSError as exc:
        raise ArtifactReadError(f"{name} could not be read: {exc}") from exc
    except Exception as exc:  # pyarrow raises its own hierarchy
        raise ArtifactReadError(
            f"{name} is not a readable Parquet file: {exc}"
        ) from exc


def parquet_row_count(run: DashboardRunSummary, name: str) -> int:
    """Row count of a Parquet artifact, from its footer only."""
    path = Path(run.absolute_path) / name
    if not path.is_file():
        raise ArtifactReadError(f"{name} is not present in {run.directory_name}")
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        raise ArtifactReadError(
            f"{name} is not a readable Parquet file: {exc}"
        ) from exc


def read_parquet(
    run: DashboardRunSummary,
    name: str,
    columns: Sequence[str],
    *,
    optional_columns: Sequence[str] = (),
) -> dict[str, list[Any]]:
    """Read the named columns of a Parquet artifact into plain lists.

    ``columns`` must all be present; a missing one is an error, because a
    view that asked for it cannot honestly render without it.
    ``optional_columns`` are read when present and omitted from the
    result when not, which is how a view degrades one series rather than
    failing outright.
    """
    path = Path(run.absolute_path) / name
    if not path.is_file():
        raise ArtifactReadError(f"{name} is not present in {run.directory_name}")
    available = set(parquet_columns(run, name))
    missing = [column for column in columns if column not in available]
    if missing:
        raise ArtifactColumnError(
            f"{name} in {run.directory_name} has no column(s) {missing}. This "
            "artifact was written by a version of the pipeline this dashboard "
            "does not know how to read."
        )
    wanted = list(columns) + [c for c in optional_columns if c in available]
    try:
        table = pq.read_table(path, columns=wanted)
    except Exception as exc:
        raise ArtifactReadError(f"{name} could not be read: {exc}") from exc
    return {column: table.column(column).to_pylist() for column in wanted}


def filter_rows(
    data: Mapping[str, Sequence[Any]], column: str, value: Any
) -> dict[str, list[Any]]:
    """Keep the rows of a column-oriented table where ``column == value``."""
    if column not in data:
        raise ArtifactColumnError(f"cannot filter on absent column {column!r}")
    keep = [index for index, entry in enumerate(data[column]) if entry == value]
    return {name: [values[i] for i in keep] for name, values in data.items()}


def distinct(values: Sequence[Any]) -> tuple[str, ...]:
    """Distinct non-null entries of a column, sorted for a stable control."""
    return tuple(sorted({str(v) for v in values if v is not None}))


def warning(
    level: DashboardWarningLevel, message: str, subject: str
) -> DashboardWarning:
    """Build a dashboard warning."""
    return DashboardWarning(level=level, message=message, subject=subject)


def missing_artifact_warning(run: DashboardRunSummary, name: str) -> DashboardWarning:
    """The standard warning for an optional artifact that is not present."""
    return warning(
        DashboardWarningLevel.INFORMATION,
        f"{name} was not written by this run, so the views that read it are "
        "unavailable. No value has been substituted for it.",
        run.directory_name,
    )


def unreadable_artifact_warning(
    run: DashboardRunSummary, name: str, reason: str
) -> DashboardWarning:
    """The standard warning for an artifact that exists but cannot be read."""
    return warning(
        DashboardWarningLevel.ERROR,
        f"{name} could not be read, so the views that depend on it are "
        f"unavailable: {reason}",
        run.directory_name,
    )


def provenance_of(run: DashboardRunSummary) -> DashboardProvenance:
    """The provenance a view must carry, taken from the catalogue entry.

    Views call this rather than constructing their own provenance, so a
    derived view cannot lose the synthetic flag between the catalogue and
    the page.
    """
    return run.provenance


def numeric_list(values: Sequence[Any]) -> list[float | None]:
    """Coerce a Parquet column to floats, keeping nulls as ``None``."""
    result: list[float | None] = []
    for value in values:
        if value is None or isinstance(value, bool):
            result.append(None)
        elif isinstance(value, int | float):
            result.append(float(value))
        else:
            result.append(None)
    return result


def string_list(values: Sequence[Any]) -> list[str | None]:
    """Coerce a Parquet column to strings, keeping nulls as ``None``."""
    return [None if value is None else str(value) for value in values]


__all__ = [
    "DEFAULT_MAX_TABLE_ROWS",
    "ArtifactColumnError",
    "ArtifactReadError",
    "distinct",
    "filter_rows",
    "load_document",
    "missing_artifact_warning",
    "numeric_list",
    "parquet_columns",
    "parquet_row_count",
    "provenance_of",
    "read_parquet",
    "string_list",
    "try_document",
    "unreadable_artifact_warning",
    "warning",
]
