"""The Milestone 8 adaptation view.

The single misreading this page exists to prevent is "the system made
N adaptations".  It did not.  It proposed N changes under a rule nobody
has validated, built N payloads, sent none of them, and heard back from
nothing.  These tests check that those four numbers stay four numbers,
and that no wording anywhere turns action frequency into effectiveness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engagevr.dashboard.views_adaptation import hold_reason_detail, load_adaptation
from engagevr.schemas.dashboard import AdaptationDashboardData
from tests.unit import dashboard_fixtures as fx


def all_text(data: AdaptationDashboardData, *, captions: bool = True) -> str:
    """Every rendered string of the view, lowercased.

    ``captions=False`` skips the explanatory captions. Several of them
    deny a reading in that reading's own words ("not a claim that either
    controller is ... more effective"), so a banned-word sweep has to
    look at what the reader takes away as a fact, and the denials are
    asserted separately.
    """
    parts: list[str] = []
    for table in (
        data.hold_reason_table,
        data.guard_table,
        data.spacing_table,
        data.scenario_table,
        data.session_table,
        data.lifecycle_table,
        data.action_frequency_comparison_table,
    ):
        if table is None:
            continue
        parts.extend([table.title, *table.columns])
        if captions:
            parts.append(table.caption or "")
        for row in table.rows:
            parts.extend(row)
    if data.difficulty_trace is not None:
        parts.extend(
            [
                data.difficulty_trace.title,
                data.difficulty_trace.subtitle or "",
                data.difficulty_trace.x_axis_label,
                data.difficulty_trace.y_axis_label,
                data.difficulty_trace.x_axis_note or "" if captions else "",
            ]
        )
    return " ".join(parts).lower()


@pytest.fixture
def data(tmp_path: Path) -> AdaptationDashboardData:
    fx.make_adaptation_run(tmp_path)
    return load_adaptation(fx.summary_for(tmp_path, "fixture-adaptation"))


class TestControllerCounts:
    def test_holds_and_proposals_partition_the_windows(
        self, data: AdaptationDashboardData
    ) -> None:
        assert data.lifecycle is not None
        assert data.hold_decisions + data.lifecycle.proposals == data.evaluated_windows

    def test_gate_eligible_and_blocked_partition_the_windows(
        self, data: AdaptationDashboardData
    ) -> None:
        assert (
            data.gate_eligible_windows + data.gate_blocked_windows
            == data.evaluated_windows
        )

    def test_the_hold_reasons_sum_to_the_hold_count(
        self, data: AdaptationDashboardData
    ) -> None:
        total = sum(int(row[1]) for row in data.hold_reason_table.rows)
        assert total == data.hold_decisions

    def test_the_hold_reason_counts_are_the_recorded_ones(
        self, data: AdaptationDashboardData
    ) -> None:
        rows = dict(data.hold_reason_table.rows)
        assert rows["gate_blocked"] == "1"


class TestLifecycleStaysSeparate:
    def test_proposals_above_zero_with_dispatch_at_zero_is_valid(
        self, data: AdaptationDashboardData
    ) -> None:
        assert data.lifecycle.proposals > 0
        assert data.lifecycle.commands_built > 0
        assert data.lifecycle.commands_dispatched == 0
        assert data.lifecycle.acknowledgements_recorded == 0

    def test_nothing_was_applied(self, data: AdaptationDashboardData) -> None:
        assert data.lifecycle.applied_confirmed == 0

    def test_the_five_states_are_listed_separately(
        self, data: AdaptationDashboardData
    ) -> None:
        stages = [row[0] for row in data.lifecycle_table.rows]
        assert stages == [
            "Proposal",
            "Command built",
            "Dispatched",
            "Acknowledged",
            "Applied",
        ]

    def test_the_table_denies_the_collapsed_reading(
        self, data: AdaptationDashboardData
    ) -> None:
        caption = (data.lifecycle_table.caption or "").lower()
        assert "never added together" in caption
        assert "never collapsed" in caption

    def test_dispatch_states_that_milestone_8_sends_nothing(
        self, data: AdaptationDashboardData
    ) -> None:
        row = next(r for r in data.lifecycle_table.rows if r[0] == "Dispatched")
        assert "sends nothing" in row[2].lower()

    def test_a_hold_window_contributes_no_lifecycle_status(
        self, tmp_path: Path
    ) -> None:
        fx.make_adaptation_run(tmp_path, "one-proposal", proposals=1)
        view = load_adaptation(fx.summary_for(tmp_path, "one-proposal"))
        assert view.lifecycle.proposals == 1
        assert view.lifecycle.commands_built == 1


class TestNoEffectivenessClaim:
    def test_no_effectiveness_wording_appears(
        self, data: AdaptationDashboardData
    ) -> None:
        text = all_text(data, captions=False)
        for banned in (
            "effectiveness",
            "effective",
            "benefit",
            "improved",
            "improvement",
            "helped",
            "optimal",
            "champion",
        ):
            assert banned not in text

    def test_the_page_says_frequency_is_not_quality(
        self, data: AdaptationDashboardData
    ) -> None:
        caption = (data.spacing_table.caption or "").lower()
        assert "how often a controller acts is not how well it works" in caption

    def test_the_comparison_is_labelled_action_frequency(
        self, data: AdaptationDashboardData
    ) -> None:
        assert (
            data.action_frequency_comparison_table.title
            == "Software-controller action-frequency comparison"
        )

    def test_the_comparison_denies_a_quality_reading(
        self, data: AdaptationDashboardData
    ) -> None:
        caption = (data.action_frequency_comparison_table.caption or "").lower()
        assert "not a claim that either controller is better" in caption
        assert "safer" in caption

    def test_the_comparison_reports_both_controllers(
        self, data: AdaptationDashboardData
    ) -> None:
        assert data.action_frequency_comparison_table.columns == (
            "diagnostic",
            "conservative policy",
            "guard-free controller",
        )
        rows = dict(
            (row[0], (row[1], row[2]))
            for row in data.action_frequency_comparison_table.rows
        )
        assert rows["Proposals"] == ("2", "6")

    def test_the_view_model_has_no_effectiveness_field(self) -> None:
        assert "effectiveness" not in " ".join(AdaptationDashboardData.model_fields)


class TestTraceAndScenarios:
    def test_the_difficulty_trace_uses_recorded_values(
        self, data: AdaptationDashboardData
    ) -> None:
        assert data.difficulty_trace.available
        series = data.difficulty_trace.series[0]
        assert series.y_values[0] == 3.0
        assert series.y_values[-1] == 2.0

    def test_a_synthetic_trace_says_so_in_its_subtitle(
        self, data: AdaptationDashboardData
    ) -> None:
        assert data.difficulty_trace.subtitle == (
            "Synthetic controller scenario — software diagnostic only"
        )

    def test_the_trace_is_not_a_participant_response(
        self, data: AdaptationDashboardData
    ) -> None:
        note = (data.difficulty_trace.x_axis_note or "").lower()
        assert "not a participant's response" in note

    def test_a_flat_trace_is_described_as_ordinary(
        self, data: AdaptationDashboardData
    ) -> None:
        note = (data.difficulty_trace.x_axis_note or "").lower()
        assert "the ordinary outcome" in note

    def test_the_scenarios_are_listed_with_their_expectations(
        self, data: AdaptationDashboardData
    ) -> None:
        assert data.scenario_table is not None
        assert "expectation" in data.scenario_table.columns

    def test_the_scenarios_are_labelled_as_controller_tests(
        self, data: AdaptationDashboardData
    ) -> None:
        assert "controller tests" in (data.scenario_table.caption or "").lower()

    def test_the_per_window_detail_is_read_not_recomputed(self, tmp_path: Path) -> None:
        fx.make_adaptation_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-adaptation")
        table = hold_reason_detail(run, "scn-fixture")
        assert table is not None
        assert "no decision is recomputed" in (table.caption or "").lower()
        assert len(table.rows) == 8

    def test_an_unknown_session_yields_nothing(self, tmp_path: Path) -> None:
        fx.make_adaptation_run(tmp_path)
        run = fx.summary_for(tmp_path, "fixture-adaptation")
        assert hold_reason_detail(run, "no-such-session") is None


class TestExperimentMode:
    def test_the_adaptive_condition_is_reported(
        self, data: AdaptationDashboardData
    ) -> None:
        assert data.experiment_mode == "adaptive"
        assert data.adaptation_enabled is True

    def test_a_static_run_is_a_legitimate_condition(self, tmp_path: Path) -> None:
        fx.make_adaptation_run(
            tmp_path, "static-run", experiment_mode="static", proposals=0
        )
        view = load_adaptation(fx.summary_for(tmp_path, "static-run"))
        assert view.experiment_mode == "static"
        assert view.lifecycle.proposals == 0
        assert view.hold_decisions == view.evaluated_windows

    def test_a_static_run_is_not_described_as_a_malfunction(
        self, tmp_path: Path
    ) -> None:
        fx.make_adaptation_run(
            tmp_path, "static-run", experiment_mode="static", proposals=0
        )
        view = load_adaptation(fx.summary_for(tmp_path, "static-run"))
        messages = " ".join(w.message for w in view.warnings).lower()
        assert "legitimate experimental control condition" in messages
        assert "not a malfunction" in messages
        assert "not low engagement" in messages


class TestGuards:
    def test_the_configured_guards_are_shown(
        self, data: AdaptationDashboardData
    ) -> None:
        rows = dict((row[0], row[1]) for row in data.guard_table.rows)
        assert rows["Minimum persistence (dwell)"] == "3"
        assert rows["Cooldown"] == "6"
        assert rows["Difficulty bounds"] == "1 to 5"

    def test_the_guards_are_labelled_engineering_defaults(
        self, data: AdaptationDashboardData
    ) -> None:
        caption = (data.guard_table.caption or "").lower()
        assert "engineering default" in caption
        assert "none is psychologically validated" in caption
        assert "none was derived from evidence" in caption
        assert "therapeutic" in caption

    def test_the_step_is_never_scaled_by_confidence(
        self, data: AdaptationDashboardData
    ) -> None:
        row = next(r for r in data.guard_table.rows if r[0] == "Step size")
        assert "never scaled by confidence" in row[2].lower()


class TestDegradation:
    def test_a_missing_trace_degrades_the_chart_only(self, tmp_path: Path) -> None:
        directory = fx.make_adaptation_run(tmp_path)
        (directory / "adaptation_trace.parquet").unlink()
        view = load_adaptation(fx.summary_for(tmp_path, "fixture-adaptation"))
        assert view.lifecycle is None
        assert view.hold_reason_table is not None
        assert not view.difficulty_trace.available

    def test_an_unreadable_summary_states_the_reason(self, tmp_path: Path) -> None:
        directory = fx.make_adaptation_run(tmp_path)
        fx.corrupt(directory / "adaptation_summary.json")
        view = load_adaptation(fx.summary_for(tmp_path, "fixture-adaptation"))
        assert view.unavailable_reason is not None
        assert "adaptation_summary.json" in view.unavailable_reason
