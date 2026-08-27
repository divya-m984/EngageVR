"""Display formatting and display-only aggregation.

The rule these tests defend is one line long: an absent value is never
displayed as zero.  On a page reporting model metrics, *not computable*
and *very bad* are opposite readings, and ``float(value or 0)`` merges
them.
"""

from __future__ import annotations

import math

import pytest

from engagevr.dashboard import aggregation as agg
from engagevr.dashboard import formatting as fmt
from engagevr.schemas.dashboard import (
    UNAVAILABLE_TEXT,
    DashboardError,
    MetricKind,
)


class TestAbsentValues:
    def test_none_renders_as_unavailable(self) -> None:
        assert fmt.format_value(fmt.metric("accuracy", None)) == UNAVAILABLE_TEXT

    def test_none_never_renders_as_zero(self) -> None:
        assert fmt.format_value(fmt.metric("accuracy", None)) != "0.0000"
        assert "0" not in fmt.format_value(fmt.metric("accuracy", None))

    def test_a_genuine_zero_survives(self) -> None:
        entry = fmt.metric("accuracy", 0.0)
        assert entry.value == 0.0
        assert fmt.format_value(entry) == "0.0000"

    def test_nan_becomes_unavailable_not_a_number(self) -> None:
        entry = fmt.metric("accuracy", math.nan)
        assert not entry.available
        assert fmt.format_value(entry) == UNAVAILABLE_TEXT

    def test_nan_records_why_it_is_unavailable(self) -> None:
        entry = fmt.metric("accuracy", math.nan)
        assert entry.unavailable_reason == fmt.NON_FINITE

    def test_positive_infinity_becomes_unavailable(self) -> None:
        assert not fmt.metric("accuracy", math.inf).available

    def test_negative_infinity_becomes_unavailable(self) -> None:
        assert not fmt.metric("accuracy", -math.inf).available

    def test_an_absent_value_states_that_it_was_not_recorded(self) -> None:
        assert fmt.metric("accuracy", None).unavailable_reason == fmt.NOT_RECORDED

    def test_a_boolean_is_not_a_metric(self) -> None:
        with pytest.raises(DashboardError, match="not a metric"):
            fmt.metric("accuracy", True)

    def test_a_string_is_not_a_metric(self) -> None:
        with pytest.raises(DashboardError, match="not a number"):
            fmt.metric("accuracy", "0.5")


class TestKinds:
    def test_a_probability_keeps_its_scale(self) -> None:
        entry = fmt.metric("p", 0.5, kind=MetricKind.PROBABILITY)
        assert fmt.format_value(entry) == "0.5000"

    def test_a_percentage_is_marked_as_one(self) -> None:
        entry = fmt.metric("share", 0.5, kind=MetricKind.PERCENTAGE)
        assert fmt.format_value(entry) == "50.00%"

    def test_a_probability_is_distinguishable_from_a_percentage(self) -> None:
        probability = fmt.metric("p", 0.5, kind=MetricKind.PROBABILITY)
        percentage = fmt.metric("p", 0.5, kind=MetricKind.PERCENTAGE)
        assert fmt.format_value(probability) != fmt.format_value(percentage)

    def test_a_count_has_no_decimals(self) -> None:
        assert fmt.format_value(fmt.count("windows", 42)) == "42"

    def test_a_count_from_a_whole_float_is_accepted(self) -> None:
        assert fmt.format_value(fmt.count("windows", 42.0)) == "42"

    def test_a_fractional_count_is_refused(self) -> None:
        with pytest.raises(DashboardError, match="fractional"):
            fmt.count("windows", 1.5)

    def test_an_interval_width_carries_its_units(self) -> None:
        entry = fmt.metric(
            "width", 0.44, kind=MetricKind.INTERVAL_WIDTH, units="target units"
        )
        assert fmt.format_value(entry) == "0.4400 target units"

    def test_an_interval_width_is_never_a_percentage(self) -> None:
        entry = fmt.metric("width", 1.8, kind=MetricKind.INTERVAL_WIDTH)
        assert "%" not in fmt.format_value(entry)

    def test_an_interval_width_may_exceed_one(self) -> None:
        entry = fmt.metric("width", 3.5, kind=MetricKind.INTERVAL_WIDTH)
        assert entry.available
        assert fmt.format_value(entry).startswith("3.5")

    def test_precision_is_consistent_across_metrics(self) -> None:
        first = fmt.format_value(fmt.metric("a", 0.1))
        second = fmt.format_value(fmt.metric("b", 0.987654321))
        assert len(first.split(".")[1]) == len(second.split(".")[1])


class TestText:
    def test_none_becomes_unavailable(self) -> None:
        assert fmt.text(None) == UNAVAILABLE_TEXT

    def test_an_empty_string_becomes_unavailable(self) -> None:
        assert fmt.text("   ") == UNAVAILABLE_TEXT

    def test_a_boolean_reads_as_words(self) -> None:
        assert fmt.text(True) == "Yes"
        assert fmt.text(False) == "No"

    def test_zero_is_not_treated_as_absent(self) -> None:
        assert fmt.text(0) == "0"

    def test_an_absent_percentage_is_unavailable(self) -> None:
        assert fmt.optional_percentage(None) == UNAVAILABLE_TEXT

    def test_a_stored_percentage_is_not_multiplied_twice(self) -> None:
        assert fmt.optional_percentage(10.0, already_percent=True) == "10.00%"

    def test_a_fraction_becomes_a_percentage(self) -> None:
        assert fmt.optional_percentage(0.1) == "10.00%"

    def test_a_non_finite_percentage_is_unavailable(self) -> None:
        assert fmt.optional_percentage(math.nan) == UNAVAILABLE_TEXT


class TestTables:
    def test_truncation_is_recorded(self) -> None:
        table = fmt.build_table(
            title="t",
            columns=("a",),
            rows=[("1",), ("2",), ("3",)],
            max_rows=2,
        )
        assert len(table.rows) == 2
        assert table.truncated_row_count == 1

    def test_truncation_produces_a_visible_note(self) -> None:
        table = fmt.build_table(
            title="t", columns=("a",), rows=[("1",), ("2",)], max_rows=1
        )
        note = fmt.truncation_note(table)
        assert note is not None
        assert "complete" in note

    def test_an_untruncated_table_has_no_note(self) -> None:
        table = fmt.build_table(title="t", columns=("a",), rows=[("1",)])
        assert fmt.truncation_note(table) is None

    def test_a_zero_row_limit_is_refused(self) -> None:
        with pytest.raises(DashboardError, match="positive"):
            fmt.build_table(title="t", columns=("a",), rows=[], max_rows=0)

    def test_counts_are_ordered_by_size_then_name(self) -> None:
        table = fmt.counts_table(
            title="t",
            counts={"b": 1, "a": 3, "c": 1},
            key_column="reason",
        )
        assert [row[0] for row in table.rows] == ["a", "b", "c"]


class TestAggregation:
    def test_an_empty_histogram_is_empty_not_a_zero_bar(self) -> None:
        centres, counts = agg.histogram([])
        assert centres == ()
        assert counts == ()

    def test_a_histogram_series_is_none_when_there_is_nothing_to_bin(self) -> None:
        assert agg.histogram_series("s", []) is None

    def test_a_histogram_counts_every_finite_value(self) -> None:
        _centres, counts = agg.histogram([0.1, 0.2, 0.9], bins=2, lower=0.0, upper=1.0)
        assert sum(counts) == 3

    def test_a_degenerate_column_becomes_one_bar(self) -> None:
        centres, counts = agg.histogram([0.5, 0.5, 0.5])
        assert centres == (0.5,)
        assert counts == (3,)

    def test_non_finite_values_are_dropped(self) -> None:
        assert agg.finite_values([1.0, math.nan, math.inf, None, "x"]) == (1.0,)

    def test_the_dropped_count_is_reportable(self) -> None:
        assert agg.dropped_count([1.0, None, math.nan]) == 2

    def test_group_counts_keep_the_unavailable_rows(self) -> None:
        assert agg.group_counts(["a", None, "a"]) == {"a": 2, "unavailable": 1}

    def test_a_mean_of_nothing_is_none_not_zero(self) -> None:
        assert agg.mean([]) is None
        assert agg.mean([None, math.nan]) is None

    def test_a_median_of_nothing_is_none_not_zero(self) -> None:
        assert agg.median([None]) is None

    def test_residuals_come_from_stored_values_only(self) -> None:
        predicted, residual, dropped = agg.residuals([1.0, 2.0], [0.5, 1.5])
        assert predicted == (0.5, 1.5)
        assert residual == (0.5, 0.5)
        assert dropped == 0

    def test_residuals_drop_and_count_incomplete_rows(self) -> None:
        _predicted, _residual, dropped = agg.residuals([1.0, None], [0.5, 1.5])
        assert dropped == 1

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(DashboardError, match="cannot be paired"):
            agg.residuals([1.0], [1.0, 2.0])

    def test_a_curve_keeps_a_gap_as_a_gap(self) -> None:
        series = agg.curve_series(
            "s",
            [{"x": 0.0, "y": 1.0}, {"x": 1.0, "y": None}],
            x_key="x",
            y_key="y",
        )
        assert series is not None
        assert series.y_values == (1.0, None)

    def test_a_curve_with_no_values_at_all_is_none(self) -> None:
        assert (
            agg.curve_series("s", [{"x": 0.0, "y": None}], x_key="x", y_key="y") is None
        )

    def test_non_increasing_monotonicity_is_checked(self) -> None:
        assert agg.is_monotonic([1.0, 0.7, 0.3], non_increasing=True)
        assert not agg.is_monotonic([1.0, 0.7, 0.9], non_increasing=True)

    def test_non_decreasing_monotonicity_is_checked(self) -> None:
        assert agg.is_monotonic([0.3, 0.7, 1.0], non_increasing=False)
        assert not agg.is_monotonic([0.3, 0.2], non_increasing=False)

    def test_a_single_point_has_no_direction(self) -> None:
        assert agg.is_monotonic([1.0], non_increasing=True) is None

    def test_confusion_totals_reconcile(self) -> None:
        total, rows, columns = agg.confusion_totals([[1, 2], [3, 4]])
        assert total == 10
        assert rows == (3, 7)
        assert columns == (4, 6)
        assert sum(rows) == sum(columns) == total
