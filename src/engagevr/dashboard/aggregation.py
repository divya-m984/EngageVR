"""Display-only aggregation of already-persisted records.

The line this module sits on is worth stating precisely, because it is
the difference between a dashboard and a second modelling pipeline.

**Allowed here** — arithmetic whose only purpose is to draw a picture of
records that already exist: histogram bins, group counts, a mean for a
summary row, confusion-matrix totals, residuals computed from stored
observed and predicted values, and the arrays a coverage curve is
plotted from.

**Not allowed anywhere in the dashboard** — anything that produces a new
scientific output: a retrained prediction, a fresh probability
calibration, a new conformal quantile, a new personalization correction,
or a new adaptation decision.  Those belong to the milestone that owns
them, and re-deriving one here would mean the dashboard was reporting a
number no run ever recorded.

A histogram of stored confidences is a picture of stored confidences.
A recalibration of them would be a claim.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence

from engagevr.schemas.dashboard import ChartSeries, DashboardError

#: Bins used for every distribution chart unless a caller overrides it.
DEFAULT_BIN_COUNT = 20


def finite_values(values: Iterable[object]) -> tuple[float, ...]:
    """Keep the finite numbers from a column, dropping nothing silently.

    Non-numeric and non-finite entries are removed here, and every caller
    reports how many were removed.  A chart drawn over a filtered column
    without saying so misrepresents its own sample size.
    """
    kept: list[float] = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if not isinstance(value, int | float):
            continue
        number = float(value)
        if math.isfinite(number):
            kept.append(number)
    return tuple(kept)


def dropped_count(values: Sequence[object]) -> int:
    """How many entries :func:`finite_values` would remove."""
    return len(values) - len(finite_values(values))


def histogram(
    values: Sequence[float],
    *,
    bins: int = DEFAULT_BIN_COUNT,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Bin values into ``(bin_centre, count)`` arrays.

    Returns empty arrays when there is nothing to bin, rather than a
    single bin holding zero: an empty histogram must render as
    *unavailable*, not as a bar of height nought.
    """
    if bins < 1:
        raise DashboardError(f"a histogram needs at least one bin, got {bins!r}")
    finite = finite_values(values)
    if not finite:
        return (), ()
    low = min(finite) if lower is None else lower
    high = max(finite) if upper is None else upper
    if not (math.isfinite(low) and math.isfinite(high)):
        return (), ()
    if high <= low:
        # A degenerate column is one bar at the single observed value.
        return (low,), (len(finite),)
    width = (high - low) / bins
    counts = [0] * bins
    for value in finite:
        index = int((value - low) / width)
        if index >= bins:
            index = bins - 1
        elif index < 0:
            index = 0
        counts[index] += 1
    centres = tuple(low + width * (index + 0.5) for index in range(bins))
    return centres, tuple(counts)


def histogram_series(
    name: str,
    values: Sequence[float],
    *,
    bins: int = DEFAULT_BIN_COUNT,
    lower: float | None = None,
    upper: float | None = None,
) -> ChartSeries | None:
    """A histogram as a chart series, or ``None`` when there is no data."""
    centres, counts = histogram(values, bins=bins, lower=lower, upper=upper)
    if not centres:
        return None
    return ChartSeries(
        name=name,
        x_values=centres,
        y_values=tuple(float(count) for count in counts),
    )


def group_counts(values: Iterable[object]) -> dict[str, int]:
    """Count occurrences of each distinct value, as display strings.

    ``None`` becomes the explicit key ``"unavailable"`` rather than being
    dropped, because the number of rows with no value recorded is itself
    something a reader needs.
    """
    counts: dict[str, int] = {}
    for value in values:
        key = "unavailable" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def mean(values: Sequence[object]) -> float | None:
    """Arithmetic mean of the finite entries, or ``None`` if there are none."""
    finite = finite_values(values)
    if not finite:
        return None
    return sum(finite) / len(finite)


def median(values: Sequence[object]) -> float | None:
    """Median of the finite entries, or ``None`` if there are none."""
    finite = sorted(finite_values(values))
    if not finite:
        return None
    middle = len(finite) // 2
    if len(finite) % 2:
        return finite[middle]
    return (finite[middle - 1] + finite[middle]) / 2.0


def residuals(
    observed: Sequence[object], predicted: Sequence[object]
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    """Paired ``(predicted, residual)`` arrays from stored values.

    A residual is ``observed - predicted`` computed from two values the
    run already persisted.  Nothing is re-predicted.  Rows where either
    side is absent are dropped and counted, and the count is reported to
    the reader.
    """
    if len(observed) != len(predicted):
        raise DashboardError(
            f"{len(observed)} observed values and {len(predicted)} predicted "
            "values cannot be paired"
        )
    kept_predicted: list[float] = []
    kept_residual: list[float] = []
    dropped = 0
    for actual, estimate in zip(observed, predicted, strict=True):
        pair = finite_values((actual, estimate))
        if len(pair) != 2:
            dropped += 1
            continue
        kept_predicted.append(pair[1])
        kept_residual.append(pair[0] - pair[1])
    return tuple(kept_predicted), tuple(kept_residual), dropped


def paired_series(
    name: str, x_values: Sequence[object], y_values: Sequence[object]
) -> tuple[ChartSeries | None, int]:
    """A scatter series over two stored columns, plus the dropped-row count."""
    if len(x_values) != len(y_values):
        raise DashboardError(
            f"{len(x_values)} x values and {len(y_values)} y values cannot be paired"
        )
    kept_x: list[float] = []
    kept_y: list[float] = []
    dropped = 0
    for left, right in zip(x_values, y_values, strict=True):
        pair = finite_values((left, right))
        if len(pair) != 2:
            dropped += 1
            continue
        kept_x.append(pair[0])
        kept_y.append(pair[1])
    if not kept_x:
        return None, dropped
    return (
        ChartSeries(name=name, x_values=tuple(kept_x), y_values=tuple(kept_y)),
        dropped,
    )


def curve_series(
    name: str,
    points: Sequence[Mapping[str, object]],
    *,
    x_key: str,
    y_key: str,
) -> ChartSeries | None:
    """A series read out of a recorded curve document.

    Points whose y value was recorded as unavailable are kept with a
    ``None`` y, so a gap in the curve renders as a gap rather than as a
    dip to zero.
    """
    x_values: list[float] = []
    y_values: list[float | None] = []
    for point in points:
        raw_x = point.get(x_key)
        if raw_x is None or isinstance(raw_x, bool):
            continue
        if not isinstance(raw_x, int | float) or not math.isfinite(float(raw_x)):
            continue
        raw_y = point.get(y_key)
        value: float | None = None
        if (
            raw_y is not None
            and not isinstance(raw_y, bool)
            and isinstance(raw_y, int | float)
            and math.isfinite(float(raw_y))
        ):
            value = float(raw_y)
        x_values.append(float(raw_x))
        y_values.append(value)
    if not x_values:
        return None
    if all(value is None for value in y_values):
        return None
    return ChartSeries(name=name, x_values=tuple(x_values), y_values=tuple(y_values))


def is_monotonic(
    values: Sequence[float | None], *, non_increasing: bool
) -> bool | None:
    """Whether a recorded curve moves in the direction its axis requires.

    Returns ``None`` when fewer than two points have values, because a
    direction is not a property a single point can have.
    """
    present = [value for value in values if value is not None]
    if len(present) < 2:
        return None
    pairs = itertools.pairwise(present)
    if non_increasing:
        return all(later <= earlier + 1e-12 for earlier, later in pairs)
    return all(later >= earlier - 1e-12 for earlier, later in pairs)


def confusion_totals(
    counts: Sequence[Sequence[int]],
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Grand total, row totals, and column totals of a confusion matrix."""
    if not counts:
        return 0, (), ()
    width = len(counts[0])
    row_totals = tuple(sum(row) for row in counts)
    column_totals = tuple(sum(row[index] for row in counts) for index in range(width))
    return sum(row_totals), row_totals, column_totals


__all__ = [
    "DEFAULT_BIN_COUNT",
    "confusion_totals",
    "curve_series",
    "dropped_count",
    "finite_values",
    "group_counts",
    "histogram",
    "histogram_series",
    "is_monotonic",
    "mean",
    "median",
    "paired_series",
    "residuals",
]
