"""Structural invariants of the dashboard view models.

These are the guarantees the pages rely on and therefore never have to
re-check.  Where a rule is enforced here, a page cannot violate it: the
model refuses to be constructed.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from engagevr.schemas.dashboard import (
    DASHBOARD_DISCLAIMER,
    SYNTHETIC_BANNER,
    AdaptationDashboardData,
    AdaptationLifecycleCounts,
    ChartSeries,
    ConfusionMatrixView,
    DashboardArtifactAvailability,
    DashboardError,
    DashboardProvenance,
    DashboardRunFamily,
    DashboardRunStatus,
    LabelledChart,
    LabelledTable,
    MetricDisplayValue,
    MetricKind,
    SelectiveAccounting,
    UncertaintyDashboardData,
)


def provenance(**overrides: object) -> DashboardProvenance:
    fields: dict[str, object] = {
        "run_id": "fixture-run",
        "run_directory": "fixture",
        "family": DashboardRunFamily.BASELINE,
        "status": DashboardRunStatus.COMPLETED,
        "is_synthetic": True,
        "scientific_evaluation_eligible": False,
    }
    fields.update(overrides)
    return DashboardProvenance.model_validate(fields)


class TestProvenance:
    def test_a_synthetic_run_cannot_be_eligible(self) -> None:
        with pytest.raises(ValidationError, match="cannot be"):
            provenance(is_synthetic=True, scientific_evaluation_eligible=True)

    def test_a_synthetic_run_requires_the_banner(self) -> None:
        assert provenance().requires_synthetic_banner
        assert SYNTHETIC_BANNER in provenance().banners

    def test_every_run_carries_the_standing_disclaimer(self) -> None:
        assert DASHBOARD_DISCLAIMER in provenance().banners
        assert DASHBOARD_DISCLAIMER in provenance(is_synthetic=False).banners

    def test_a_non_synthetic_run_may_still_be_ineligible(self) -> None:
        entry = provenance(is_synthetic=False, scientific_evaluation_eligible=False)
        assert not entry.scientific_evaluation_eligible
        assert not entry.requires_synthetic_banner

    def test_a_failed_run_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="failure reason"):
            provenance(status=DashboardRunStatus.FAILED)

    def test_a_derived_view_may_not_change_the_synthetic_flag(self) -> None:
        with pytest.raises(DashboardError, match="may not change"):
            provenance().derive(is_synthetic=False)

    def test_a_derived_view_may_not_grant_eligibility(self) -> None:
        with pytest.raises(DashboardError, match="may not change"):
            provenance().derive(scientific_evaluation_eligible=True)

    def test_a_derived_view_keeps_the_provenance(self) -> None:
        derived = provenance().derive(target_name="engagement_class")
        assert derived.is_synthetic
        assert not derived.scientific_evaluation_eligible
        assert derived.target_name == "engagement_class"

    def test_provenance_is_frozen(self) -> None:
        with pytest.raises(ValidationError):
            provenance().run_id = "something-else"  # type: ignore[misc]

    def test_the_model_forbids_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            provenance(marked_as_validated=True)


class TestMetricDisplayValue:
    def test_zero_is_a_real_value(self) -> None:
        entry = MetricDisplayValue(name="accuracy", value=0.0)
        assert entry.available
        assert entry.value == 0.0

    def test_an_absent_value_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state why"):
            MetricDisplayValue(name="accuracy", value=None)

    def test_nan_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="non-finite"):
            MetricDisplayValue(name="accuracy", value=math.nan)

    def test_infinity_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="non-finite"):
            MetricDisplayValue(name="accuracy", value=math.inf)

    def test_a_value_and_a_reason_cannot_coexist(self) -> None:
        with pytest.raises(ValidationError, match="both"):
            MetricDisplayValue(name="a", value=1.0, unavailable_reason="missing")

    def test_a_fractional_count_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="fractional"):
            MetricDisplayValue(name="windows", value=1.5, kind=MetricKind.COUNT)

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="negative"):
            MetricDisplayValue(name="windows", value=-1.0, kind=MetricKind.COUNT)


class TestSelectiveAccounting:
    def test_reconciling_counts_are_accepted(self) -> None:
        accounting = SelectiveAccounting(
            evaluated_window_count=10,
            accepted_count=6,
            abstained_count=3,
            unavailable_count=1,
        )
        assert accounting.reconciles
        assert accounting.coverage == pytest.approx(0.6)

    def test_a_mismatch_cannot_be_silently_accepted(self) -> None:
        with pytest.raises(ValidationError, match="never"):
            SelectiveAccounting(
                evaluated_window_count=10,
                accepted_count=6,
                abstained_count=3,
                unavailable_count=0,
            )

    def test_a_declared_mismatch_must_state_the_error(self) -> None:
        with pytest.raises(ValidationError, match="must state"):
            SelectiveAccounting(
                evaluated_window_count=10,
                accepted_count=6,
                abstained_count=3,
                unavailable_count=0,
                reconciles=False,
            )

    def test_a_declared_mismatch_with_an_error_is_representable(self) -> None:
        accounting = SelectiveAccounting(
            evaluated_window_count=10,
            accepted_count=6,
            abstained_count=3,
            unavailable_count=0,
            reconciles=False,
            reconciliation_error="the artifact does not add up",
        )
        assert not accounting.reconciles

    def test_reconciling_counts_may_not_claim_a_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="but the view says"):
            SelectiveAccounting(
                evaluated_window_count=10,
                accepted_count=6,
                abstained_count=3,
                unavailable_count=1,
                reconciles=False,
                reconciliation_error="wrong",
            )

    def test_abstained_and_unavailable_are_separate_fields(self) -> None:
        accounting = SelectiveAccounting(
            evaluated_window_count=4,
            accepted_count=1,
            abstained_count=2,
            unavailable_count=1,
        )
        assert accounting.abstained_count != accounting.unavailable_count


class TestLifecycleCounts:
    def test_the_current_milestone_8_shape_is_representable(self) -> None:
        counts = AdaptationLifecycleCounts(
            proposals=19,
            commands_built=19,
            commands_dispatched=0,
            acknowledgements_recorded=0,
        )
        assert counts.commands_dispatched == 0
        assert counts.acknowledgements_recorded == 0

    def test_a_command_cannot_exist_without_a_proposal(self) -> None:
        with pytest.raises(ValidationError, match="cannot exist"):
            AdaptationLifecycleCounts(
                proposals=0,
                commands_built=1,
                commands_dispatched=0,
                acknowledgements_recorded=0,
            )

    def test_a_dispatch_cannot_exceed_the_commands_built(self) -> None:
        with pytest.raises(ValidationError, match="dispatched but"):
            AdaptationLifecycleCounts(
                proposals=2,
                commands_built=1,
                commands_dispatched=2,
                acknowledgements_recorded=0,
            )

    def test_an_acknowledgement_requires_a_dispatch(self) -> None:
        with pytest.raises(ValidationError, match="environment reply"):
            AdaptationLifecycleCounts(
                proposals=2,
                commands_built=2,
                commands_dispatched=0,
                acknowledgements_recorded=1,
            )

    def test_applied_requires_an_acknowledgement(self) -> None:
        with pytest.raises(ValidationError, match="requires an acknowledgement"):
            AdaptationLifecycleCounts(
                proposals=2,
                commands_built=2,
                commands_dispatched=2,
                acknowledgements_recorded=0,
                applied_confirmed=1,
            )


class TestAdaptationView:
    def test_holds_and_proposals_must_partition_the_windows(self) -> None:
        with pytest.raises(ValidationError, match="evaluated windows"):
            AdaptationDashboardData(
                provenance=provenance(family=DashboardRunFamily.ADAPTATION),
                evaluated_windows=10,
                hold_decisions=8,
                lifecycle=AdaptationLifecycleCounts(
                    proposals=1,
                    commands_built=1,
                    commands_dispatched=0,
                    acknowledgements_recorded=0,
                ),
            )

    def test_eligible_and_blocked_must_partition_the_windows(self) -> None:
        with pytest.raises(ValidationError, match="evaluated windows"):
            AdaptationDashboardData(
                provenance=provenance(family=DashboardRunFamily.ADAPTATION),
                evaluated_windows=10,
                gate_eligible_windows=8,
                gate_blocked_windows=1,
            )

    def test_the_view_has_no_effectiveness_field(self) -> None:
        fields = set(AdaptationDashboardData.model_fields)
        for banned in ("effectiveness", "benefit", "improvement", "success_rate"):
            assert not any(banned in name for name in fields)

    def test_an_effectiveness_field_cannot_be_added_by_a_caller(self) -> None:
        with pytest.raises(ValidationError):
            AdaptationDashboardData(
                provenance=provenance(family=DashboardRunFamily.ADAPTATION),
                adaptation_effectiveness=0.9,
            )


class TestUncertaintyView:
    def test_a_regression_view_cannot_carry_calibrated_confidence(self) -> None:
        chart = LabelledChart(
            title="c",
            x_axis_label="x",
            y_axis_label="y",
            series=(),
            unavailable_reason="none",
        )
        with pytest.raises(ValidationError, match="no class probability"):
            UncertaintyDashboardData(
                provenance=provenance(family=DashboardRunFamily.UNCERTAINTY),
                task_type="regression",
                calibrated_confidence_histogram=chart,
            )

    def test_a_regression_view_cannot_carry_a_confidence_curve(self) -> None:
        chart = LabelledChart(
            title="c",
            x_axis_label="x",
            y_axis_label="y",
            series=(),
            unavailable_reason="none",
        )
        with pytest.raises(ValidationError, match="no class probability"):
            UncertaintyDashboardData(
                provenance=provenance(family=DashboardRunFamily.UNCERTAINTY),
                task_type="regression",
                confidence_coverage_curve=chart,
            )

    def test_a_regression_view_cannot_carry_a_calibration_status(self) -> None:
        with pytest.raises(ValidationError, match="no class probability"):
            UncertaintyDashboardData(
                provenance=provenance(family=DashboardRunFamily.UNCERTAINTY),
                task_type="regression",
                probability_calibration_status="calibrated",
            )

    def test_a_classification_view_cannot_carry_an_interval(self) -> None:
        with pytest.raises(ValidationError, match="no prediction interval"):
            UncertaintyDashboardData(
                provenance=provenance(family=DashboardRunFamily.UNCERTAINTY),
                task_type="classification",
                empirical_interval_coverage=MetricDisplayValue(
                    name="coverage", value=0.9
                ),
            )

    def test_a_classification_view_cannot_carry_a_width_curve(self) -> None:
        chart = LabelledChart(
            title="c",
            x_axis_label="x",
            y_axis_label="y",
            series=(),
            unavailable_reason="none",
        )
        with pytest.raises(ValidationError, match="no prediction interval"):
            UncertaintyDashboardData(
                provenance=provenance(family=DashboardRunFamily.UNCERTAINTY),
                task_type="classification",
                width_coverage_curve=chart,
            )

    def test_there_is_no_single_combined_uncertainty_score_field(self) -> None:
        fields = set(UncertaintyDashboardData.model_fields)
        assert "uncertainty_score" not in fields
        assert "combined_uncertainty" not in fields


class TestChartsAndTables:
    def test_a_series_needs_matching_axis_lengths(self) -> None:
        with pytest.raises(ValidationError, match="x values"):
            ChartSeries(name="s", x_values=(0.0, 1.0), y_values=(0.0,))

    def test_a_series_refuses_a_non_finite_x(self) -> None:
        with pytest.raises(ValidationError, match="non-finite x"):
            ChartSeries(name="s", x_values=(math.nan,), y_values=(0.0,))

    def test_a_series_refuses_a_non_finite_y(self) -> None:
        with pytest.raises(ValidationError, match="non-finite y"):
            ChartSeries(name="s", x_values=(0.0,), y_values=(math.inf,))

    def test_a_series_may_carry_a_gap(self) -> None:
        series = ChartSeries(name="s", x_values=(0.0, 1.0), y_values=(0.5, None))
        assert series.y_values[1] is None

    def test_an_empty_chart_must_state_why(self) -> None:
        with pytest.raises(ValidationError, match="must state why"):
            LabelledChart(title="t", x_axis_label="x", y_axis_label="y", series=())

    def test_a_table_row_must_match_its_columns(self) -> None:
        with pytest.raises(ValidationError, match="cells"):
            LabelledTable(title="t", columns=("a", "b"), rows=(("1",),))

    def test_a_chart_always_names_both_axes(self) -> None:
        with pytest.raises(ValidationError):
            LabelledChart(title="t", x_axis_label="", y_axis_label="y", series=())


class TestConfusionMatrixView:
    def test_counts_reconcile_with_the_labels(self) -> None:
        matrix = ConfusionMatrixView(
            labels=("low", "medium"),
            counts=((3, 1), (2, 4)),
            row_axis_label="observed synthetic label",
            column_axis_label="predicted class",
        )
        assert matrix.total == 10
        assert matrix.row_totals() == (4, 6)

    def test_a_ragged_matrix_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="entries"):
            ConfusionMatrixView(
                labels=("low", "medium"),
                counts=((3, 1), (2,)),
                row_axis_label="observed synthetic label",
                column_axis_label="predicted class",
            )

    def test_a_matrix_without_labels_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="class labels"):
            ConfusionMatrixView(
                labels=(),
                counts=(),
                row_axis_label="observed synthetic label",
                column_axis_label="predicted class",
            )

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="negative"):
            ConfusionMatrixView(
                labels=("low",),
                counts=((-1,),),
                row_axis_label="observed synthetic label",
                column_axis_label="predicted class",
            )


class TestArtifactAvailability:
    def test_an_absent_artifact_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state a reason"):
            DashboardArtifactAvailability(
                name="metrics.json", present=False, required=True
            )

    def test_a_present_artifact_must_record_its_size(self) -> None:
        with pytest.raises(ValidationError, match="records no size"):
            DashboardArtifactAvailability(
                name="metrics.json", present=True, required=True
            )
