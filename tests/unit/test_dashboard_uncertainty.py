"""The Milestone 7 uncertainty view.

Classification and regression are selective on different axes that move
in opposite directions, and neither axis is a relabelling of the other.
These tests check the axes, their monotonicity wording, the selective
accounting, and the absence of every control that belongs to the other
task type.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engagevr.dashboard.views_uncertainty import (
    AXIS_RULE,
    AXIS_UNITS,
    CONFIDENCE_AXIS,
    INTERVAL_WIDTH_AXIS,
    load_uncertainty,
)
from engagevr.schemas.dashboard import DashboardWarningLevel
from tests.unit import dashboard_fixtures as fx


@pytest.fixture
def classification(tmp_path: Path):  # type: ignore[no-untyped-def]
    fx.make_uncertainty_run(tmp_path, "u-cls")
    return load_uncertainty(fx.summary_for(tmp_path, "u-cls"))


@pytest.fixture
def regression(tmp_path: Path):  # type: ignore[no-untyped-def]
    fx.make_uncertainty_run(tmp_path, "u-reg", task_type="regression")
    return load_uncertainty(fx.summary_for(tmp_path, "u-reg"))


class TestSelectiveAccounting:
    def test_the_three_outcomes_reconcile(self, classification) -> None:  # type: ignore[no-untyped-def]
        accounting = classification.accounting
        assert accounting is not None
        assert accounting.reconciles
        assert (
            accounting.accepted_count
            + accounting.abstained_count
            + accounting.unavailable_count
            == accounting.evaluated_window_count
        )

    def test_abstained_and_unavailable_stay_distinct(self, classification) -> None:  # type: ignore[no-untyped-def]
        accounting = classification.accounting
        assert accounting.abstained_count == 3
        assert accounting.unavailable_count == 1

    def test_a_mismatch_becomes_a_validation_error(self, tmp_path: Path) -> None:
        fx.make_uncertainty_run(tmp_path, "broken", reconciling=False)
        data = load_uncertainty(fx.summary_for(tmp_path, "broken"))
        assert data.accounting is not None
        assert not data.accounting.reconciles
        assert "ARTIFACT VALIDATION ERROR" in str(data.accounting.reconciliation_error)

    def test_a_mismatch_is_not_normalised_away(self, tmp_path: Path) -> None:
        fx.make_uncertainty_run(tmp_path, "broken", reconciling=False)
        data = load_uncertainty(fx.summary_for(tmp_path, "broken"))
        # The counts are shown as recorded, not adjusted to add up.
        assert data.accounting.accepted_count == 6
        assert data.accounting.abstained_count == 4
        assert data.accounting.unavailable_count == 1

    def test_a_mismatch_raises_an_error_level_warning(self, tmp_path: Path) -> None:
        fx.make_uncertainty_run(tmp_path, "broken", reconciling=False)
        data = load_uncertainty(fx.summary_for(tmp_path, "broken"))
        assert DashboardWarningLevel.ERROR in {w.level for w in data.warnings}

    def test_abstention_is_not_counted_as_error(self, classification) -> None:  # type: ignore[no-untyped-def]
        caption = (classification.abstention_reason_table.caption or "").lower()
        assert "not an error" in caption
        assert "never counted as one" in caption

    def test_the_reason_counts_are_the_recorded_ones(self, classification) -> None:  # type: ignore[no-untyped-def]
        rows = dict(classification.abstention_reason_table.rows)
        assert rows["below_confidence_threshold"] == "3"


class TestClassificationSemantics:
    def test_the_axis_is_the_confidence_threshold(self, classification) -> None:  # type: ignore[no-untyped-def]
        assert classification.coverage_axis == CONFIDENCE_AXIS

    def test_the_axis_units_are_stated(self, classification) -> None:  # type: ignore[no-untyped-def]
        assert classification.coverage_axis_units == AXIS_UNITS[CONFIDENCE_AXIS]
        assert "probability in [0, 1]" in classification.coverage_axis_units

    def test_the_monotonic_direction_is_stated(self, classification) -> None:  # type: ignore[no-untyped-def]
        assert classification.coverage_monotonicity_rule == AXIS_RULE[CONFIDENCE_AXIS]
        assert "non-increasing" in classification.coverage_monotonicity_rule

    def test_no_interval_control_appears(self, classification) -> None:  # type: ignore[no-untyped-def]
        assert classification.interval_width_histogram is None
        assert classification.width_coverage_curve is None
        assert classification.empirical_interval_coverage is None
        assert classification.configured_maximum_interval_width is None
        assert classification.interval_table is None

    def test_the_curve_names_its_axis(self, classification) -> None:  # type: ignore[no-untyped-def]
        chart = classification.confidence_coverage_curve
        assert "confidence threshold" in chart.x_axis_label
        assert "uncertainty threshold" not in chart.x_axis_label.lower()

    def test_the_curve_is_non_increasing_as_recorded(self, classification) -> None:  # type: ignore[no-untyped-def]
        series = classification.confidence_coverage_curve.series[0]
        values = [v for v in series.y_values if v is not None]
        assert values == sorted(values, reverse=True)

    def test_the_confidence_chart_names_the_quantity_exactly(
        self, classification
    ) -> None:  # type: ignore[no-untyped-def]
        chart = classification.calibrated_confidence_histogram
        assert chart.title == "Calibrated classification confidence"
        assert chart.title != "Confidence"

    def test_the_confidence_chart_denies_the_quality_reading(
        self, classification
    ) -> None:  # type: ignore[no-untyped-def]
        note = (
            classification.calibrated_confidence_histogram.x_axis_note or ""
        ).lower()
        assert "not signal quality" in note
        assert "not certainty" in note

    def test_entropy_and_margin_have_their_own_charts(self, classification) -> None:  # type: ignore[no-untyped-def]
        assert classification.predictive_entropy_histogram.available
        assert classification.probability_margin_histogram.available
        assert (
            classification.predictive_entropy_histogram.title
            != classification.probability_margin_histogram.title
        )

    def test_entropy_is_not_called_a_confidence_score(self, classification) -> None:  # type: ignore[no-untyped-def]
        note = (classification.predictive_entropy_histogram.x_axis_note or "").lower()
        assert "not a confidence score" in note

    def test_the_calibration_status_is_reported(self, classification) -> None:  # type: ignore[no-untyped-def]
        assert classification.probability_calibration_status == "calibrated"

    def test_the_risk_coverage_curve_is_drawn(self, classification) -> None:  # type: ignore[no-untyped-def]
        assert classification.risk_coverage_curve.available
        note = (classification.risk_coverage_curve.x_axis_note or "").lower()
        assert "not a bound" in note
        assert "not a guarantee" in note


class TestRegressionSemantics:
    def test_the_axis_is_the_maximum_interval_width(self, regression) -> None:  # type: ignore[no-untyped-def]
        assert regression.coverage_axis == INTERVAL_WIDTH_AXIS

    def test_the_axis_units_are_target_units_not_a_probability(
        self, regression
    ) -> None:  # type: ignore[no-untyped-def]
        units = regression.coverage_axis_units
        assert "target's own units" in units
        assert "NOT a probability" in units

    def test_the_monotonic_direction_is_the_opposite_one(self, regression) -> None:  # type: ignore[no-untyped-def]
        assert "non-decreasing" in regression.coverage_monotonicity_rule
        assert regression.coverage_monotonicity_rule != AXIS_RULE[CONFIDENCE_AXIS]

    def test_no_confidence_control_appears(self, regression) -> None:  # type: ignore[no-untyped-def]
        assert regression.calibrated_confidence_histogram is None
        assert regression.probability_margin_histogram is None
        assert regression.confidence_coverage_curve is None
        assert regression.probability_calibration_status is None

    def test_the_curve_names_the_width_axis(self, regression) -> None:  # type: ignore[no-untyped-def]
        chart = regression.width_coverage_curve
        assert "maximum interval width" in chart.x_axis_label
        assert "confidence" not in chart.x_axis_label.lower()

    def test_the_curve_is_non_decreasing_as_recorded(self, regression) -> None:  # type: ignore[no-untyped-def]
        series = regression.width_coverage_curve.series[0]
        values = [v for v in series.y_values if v is not None]
        assert values == sorted(values)

    def test_the_interval_width_is_in_target_units(self, regression) -> None:  # type: ignore[no-untyped-def]
        assert "target units" in regression.interval_width_histogram.x_axis_label

    def test_the_width_is_never_shown_as_a_probability(self, regression) -> None:  # type: ignore[no-untyped-def]
        note = (regression.interval_width_histogram.x_axis_note or "").lower()
        assert "not a probability" in note
        assert "not convertible into a confidence score" in note
        assert "never displayed as 1 - width" in note

    def test_the_configured_maximum_width_carries_its_units(self, regression) -> None:  # type: ignore[no-untyped-def]
        entry = regression.configured_maximum_interval_width
        assert entry.units == "target units"
        assert entry.value == pytest.approx(0.5)

    def test_the_empirical_interval_coverage_is_reported(self, regression) -> None:  # type: ignore[no-untyped-def]
        assert regression.empirical_interval_coverage.value == pytest.approx(0.9)

    def test_empirical_interval_coverage_differs_from_selective_coverage(
        self, regression
    ) -> None:  # type: ignore[no-untyped-def]
        assert regression.accounting is not None
        assert regression.empirical_interval_coverage.value != pytest.approx(
            regression.accounting.coverage
        )

    def test_the_risk_curve_is_unavailable_rather_than_fabricated(
        self, regression
    ) -> None:  # type: ignore[no-untyped-def]
        assert not regression.risk_coverage_curve.available
        assert "undefined for a regression target" in str(
            regression.risk_coverage_curve.unavailable_reason
        )

    def test_the_interval_summary_uses_target_units(self, regression) -> None:  # type: ignore[no-untyped-def]
        assert "regression target units" in regression.interval_table.columns[1]


class TestUnswptAndLegacyCurves:
    def test_a_curve_with_no_axis_field_is_refused(self, tmp_path: Path) -> None:
        fx.make_uncertainty_run(tmp_path, "legacy", record_axis=False)
        data = load_uncertainty(fx.summary_for(tmp_path, "legacy"))
        assert data.confidence_coverage_curve is None
        messages = " ".join(w.message for w in data.warnings)
        assert "records no 'axis' field" in messages
        assert "must not be guessed" in messages

    def test_an_absent_curve_document_is_stated_not_fabricated(
        self, tmp_path: Path
    ) -> None:
        directory = fx.make_uncertainty_run(tmp_path, "no-curve")
        (directory / "coverage_curve.json").unlink()
        data = load_uncertainty(fx.summary_for(tmp_path, "no-curve"))
        assert data.confidence_coverage_curve is None
        messages = " ".join(w.message for w in data.warnings)
        assert "coverage_curve.json" in messages

    def test_an_unswept_curve_is_unavailable_with_a_reason(
        self, tmp_path: Path
    ) -> None:
        directory = fx.make_uncertainty_run(tmp_path, "empty-curve")
        document = fx.coverage_curve_document(
            run_id="x", task_type="classification", record_axis=True
        )
        document["curve"]["points"] = []
        document["curve"]["points_unavailable_reason"] = (
            "no interval-width grid was configured"
        )
        fx.write_json(directory / "coverage_curve.json", document)
        data = load_uncertainty(fx.summary_for(tmp_path, "empty-curve"))
        chart = data.confidence_coverage_curve
        assert not chart.available
        assert "no interval-width grid" in str(chart.unavailable_reason)

    def test_a_curve_swept_over_the_wrong_axis_is_refused(self, tmp_path: Path) -> None:
        directory = fx.make_uncertainty_run(tmp_path, "wrong-axis")
        document = fx.coverage_curve_document(
            run_id="x", task_type="classification", record_axis=True
        )
        document["curve"]["axis"] = "maximum_interval_width"
        fx.write_json(directory / "coverage_curve.json", document)
        data = load_uncertainty(fx.summary_for(tmp_path, "wrong-axis"))
        assert data.confidence_coverage_curve is None
        messages = " ".join(w.message for w in data.warnings)
        assert "selective on" in messages


class TestDegradation:
    def test_a_missing_selective_table_degrades_one_chart_only(
        self, tmp_path: Path
    ) -> None:
        directory = fx.make_uncertainty_run(tmp_path, "no-parquet")
        (directory / "selective_predictions.parquet").unlink()
        data = load_uncertainty(fx.summary_for(tmp_path, "no-parquet"))
        assert not data.calibrated_confidence_histogram.available
        assert data.accounting is not None
        assert data.confidence_coverage_curve.available

    def test_a_missing_column_says_it_was_not_derived(self, tmp_path: Path) -> None:
        directory = fx.make_uncertainty_run(tmp_path, "no-entropy")
        (directory / "predictions.parquet").unlink()
        data = load_uncertainty(fx.summary_for(tmp_path, "no-entropy"))
        chart = data.predictive_entropy_histogram
        assert not chart.available
        assert "not been derived from another column" in str(chart.unavailable_reason)

    def test_an_unreadable_uncertainty_document_states_the_reason(
        self, tmp_path: Path
    ) -> None:
        directory = fx.make_uncertainty_run(tmp_path, "corrupt")
        fx.corrupt(directory / "uncertainty.json")
        data = load_uncertainty(fx.summary_for(tmp_path, "corrupt"))
        assert data.unavailable_reason is not None
        assert "uncertainty.json" in data.unavailable_reason
