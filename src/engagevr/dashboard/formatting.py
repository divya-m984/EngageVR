"""Centralised display formatting for the dashboard.

Every number a page shows passes through this module.  The reason is one
rule that is easy to state and easy to break accidentally:

    An absent value is never displayed as zero.

``float(value or 0)`` and its relatives turn *not computable* into *very
bad*, and on a page reporting model metrics those two readings are
opposite.  So there is exactly one place that decides what ``None``
looks like, one place that refuses ``NaN``, and one place that fixes the
precision, and it is here.

The second rule is that a display type is not a formatting preference.
A probability, a percentage, a count, and an interval width are
different quantities.  An interval width carries the regression target's
own units, is not confined to ``[0, 1]``, and is never rendered with a
percent sign, because doing so would present it as a probability.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from engagevr.schemas.dashboard import (
    UNAVAILABLE_TEXT,
    DashboardError,
    LabelledTable,
    MetricDisplayValue,
    MetricKind,
)

#: Decimal places used for a real-valued metric.
REAL_PRECISION = 4
#: Decimal places used for a probability or proportion.
PROBABILITY_PRECISION = 4
#: Decimal places used for a percentage, before the ``%`` sign.
PERCENTAGE_PRECISION = 2
#: Decimal places used for an interval width in target units.
INTERVAL_WIDTH_PRECISION = 4

#: Reason attached when an artifact simply did not record a field.
NOT_RECORDED = "not recorded by this artifact"
#: Reason attached when a stored value was non-finite.
NON_FINITE = (
    "the artifact recorded a non-finite value; a non-finite number is not "
    "displayed as a measurement"
)


def format_value(metric: MetricDisplayValue) -> str:
    """Render one metric, or ``"Unavailable"`` when it has no value."""
    if metric.value is None:
        return UNAVAILABLE_TEXT
    return _format_number(metric.value, metric.kind, metric.units)


def _format_number(value: float, kind: MetricKind, units: str | None) -> str:
    if not math.isfinite(value):  # pragma: no cover - refused at construction
        return UNAVAILABLE_TEXT
    if kind is MetricKind.COUNT:
        return f"{int(value):d}"
    if kind is MetricKind.PERCENTAGE:
        return f"{value * 100.0:.{PERCENTAGE_PRECISION}f}%"
    if kind is MetricKind.PROBABILITY:
        return f"{value:.{PROBABILITY_PRECISION}f}"
    if kind is MetricKind.INTERVAL_WIDTH:
        rendered = f"{value:.{INTERVAL_WIDTH_PRECISION}f}"
        return f"{rendered} {units}" if units else rendered
    rendered = f"{value:.{REAL_PRECISION}f}"
    return f"{rendered} {units}" if units else rendered


def metric(
    name: str,
    value: object,
    *,
    kind: MetricKind = MetricKind.REAL,
    units: str | None = None,
    source_artifact: str | None = None,
    unavailable_reason: str | None = None,
) -> MetricDisplayValue:
    """Build a metric from a value read out of an artifact.

    ``value`` arrives as whatever JSON produced, which is why this
    function exists: it maps ``None`` and non-finite floats onto an
    explicit unavailable state instead of letting them reach a page.
    A genuine ``0.0`` survives as ``0.0``.
    """
    reason = unavailable_reason
    number: float | None = None
    if value is None:
        reason = reason or NOT_RECORDED
    elif isinstance(value, bool):
        raise DashboardError(
            f"metric {name!r} was given the boolean {value!r}; a flag is not a metric"
        )
    elif isinstance(value, int | float):
        candidate = float(value)
        if math.isfinite(candidate):
            number = candidate
            reason = None
        else:
            reason = reason or NON_FINITE
    else:
        raise DashboardError(
            f"metric {name!r} was given {type(value).__name__} {value!r}, "
            "which is not a number"
        )
    return MetricDisplayValue(
        name=name,
        kind=kind,
        value=number,
        unavailable_reason=reason,
        units=units,
        source_artifact=source_artifact,
    )


def count(
    name: str,
    value: object,
    *,
    source_artifact: str | None = None,
    unavailable_reason: str | None = None,
) -> MetricDisplayValue:
    """Build an integer-valued metric.

    A count that arrives as a float with a fractional part is refused
    rather than rounded: it means the caller read the wrong field.
    """
    if value is not None and not isinstance(value, bool) and isinstance(value, float):
        if math.isfinite(value) and value != int(value):
            raise DashboardError(
                f"count {name!r} was given the fractional value {value!r}"
            )
    return metric(
        name,
        value,
        kind=MetricKind.COUNT,
        source_artifact=source_artifact,
        unavailable_reason=unavailable_reason,
    )


def text(value: object) -> str:
    """Render a non-numeric field, or ``"Unavailable"`` when absent.

    An empty string is treated as absent.  A stored empty string is
    indistinguishable on a page from a missing one, so it is shown the
    honest way round rather than as a blank cell.
    """
    if value is None:
        return UNAVAILABLE_TEXT
    if isinstance(value, bool):
        return "Yes" if value else "No"
    rendered = str(value).strip()
    return rendered if rendered else UNAVAILABLE_TEXT


def optional_percentage(value: object, *, already_percent: bool = False) -> str:
    """Render a proportion as a percentage, or ``"Unavailable"``.

    ``already_percent`` is for artifacts that stored 0 to 100
    rather than 0 to 1; the two are never guessed apart by looking
    at the magnitude, because ``0.5`` is a legitimate value of both.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return UNAVAILABLE_TEXT
    number = float(value)
    if not math.isfinite(number):
        return UNAVAILABLE_TEXT
    fraction = number / 100.0 if already_percent else number
    return f"{fraction * 100.0:.{PERCENTAGE_PRECISION}f}%"


def build_table(
    *,
    title: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[str]],
    source_artifact: str | None = None,
    caption: str | None = None,
    max_rows: int | None = None,
) -> LabelledTable:
    """Assemble a table, stating how many rows a display limit dropped.

    Truncation is recorded rather than silent.  A table that quietly
    stops at row 1000 reads as a complete table, which is the one thing
    a research view must not do.
    """
    materialised = [tuple(str(cell) for cell in row) for row in rows]
    truncated = 0
    if max_rows is not None:
        if max_rows < 1:
            raise DashboardError(f"max_rows must be positive, got {max_rows!r}")
        if len(materialised) > max_rows:
            truncated = len(materialised) - max_rows
            materialised = materialised[:max_rows]
    return LabelledTable(
        title=title,
        columns=tuple(columns),
        rows=tuple(materialised),
        source_artifact=source_artifact,
        caption=caption,
        truncated_row_count=truncated,
    )


def metric_rows(
    metrics: Sequence[MetricDisplayValue],
) -> tuple[tuple[str, str], ...]:
    """Render metrics as ``(name, formatted)`` pairs."""
    return tuple((m.name, format_value(m)) for m in metrics)


def counts_table(
    *,
    title: str,
    counts: Mapping[str, int],
    key_column: str,
    source_artifact: str | None = None,
    caption: str | None = None,
) -> LabelledTable:
    """Render a mapping of names to counts, sorted by count then name.

    Sorting is descending by count so the reader sees the dominant
    category first.  This is a display order and nothing more; it does
    not rank, score, or prefer anything.
    """
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return build_table(
        title=title,
        columns=(key_column, "count"),
        rows=[(name, str(value)) for name, value in ordered],
        source_artifact=source_artifact,
        caption=caption,
    )


def truncation_note(table: LabelledTable) -> str | None:
    """Sentence stating what a display limit removed, if anything."""
    if not table.truncated_row_count:
        return None
    return (
        f"{table.truncated_row_count} further row(s) are not shown because of "
        "the configured display limit. The artifact on disk is complete; only "
        "this view is truncated."
    )


__all__ = [
    "INTERVAL_WIDTH_PRECISION",
    "NON_FINITE",
    "NOT_RECORDED",
    "PERCENTAGE_PRECISION",
    "PROBABILITY_PRECISION",
    "REAL_PRECISION",
    "build_table",
    "count",
    "counts_table",
    "format_value",
    "metric",
    "metric_rows",
    "optional_percentage",
    "text",
    "truncation_note",
]
