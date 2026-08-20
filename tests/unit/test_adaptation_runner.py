"""The offline adaptation run: artifacts, metrics, privacy, determinism.

Every test writes into a temporary directory.  Nothing here needs a
network, a dataset, a model file, MLflow, DVC, or Docker.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from engagevr.adaptation.runner import (
    ADAPTATION_REQUIRED_ARTIFACTS,
    NAIVE_COMPARISON_NOTE,
    AdaptationRunConfiguration,
    AdaptationRunError,
    build_run_id,
    controller_metrics,
    evaluate_sequence,
    naive_configuration,
    run_adaptation,
    scenario_inputs,
    trace_table,
)
from engagevr.adaptation.scenarios import SCENARIOS, get_scenario
from engagevr.schemas.adaptation_policy import (
    ADAPTATION_POLICY_NOTE,
    CONTROLLER_METRIC_NOTE,
    AdaptationDecisionKind,
    AdaptationDirection,
    ExperimentMode,
)
from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER, EvaluationMode
from tests.unit.adaptation_helpers import make_configuration

#: A small subset, so most tests stay fast.
_SMALL = (
    get_scenario("persistent-decrease"),
    get_scenario("gate-blocked"),
    get_scenario("stable-neutral"),
)


def _run(tmp_path: Path, **overrides: object):
    settings: dict[str, object] = {
        "output_directory": tmp_path / "run",
        "policy": make_configuration(),
        "compare_naive": True,
    }
    settings.update(overrides)
    config = AdaptationRunConfiguration(**settings)  # type: ignore[arg-type]
    return run_adaptation(config, scenarios=_SMALL)


class TestArtifacts:
    def test_every_required_artifact_is_written(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        for name in ADAPTATION_REQUIRED_ARTIFACTS:
            assert (result.directory / name).exists()
        assert (result.directory / "checksums.json").exists()
        assert (result.directory / "scenarios.json").exists()

    def test_the_trace_has_one_row_per_evaluation(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        table = pq.read_table(result.directory / "adaptation_trace.parquet")
        assert table.num_rows == len(result.decisions)
        assert table.num_rows == result.metrics.evaluated_windows

    def test_the_checksums_match_the_files(self, tmp_path: Path) -> None:
        from engagevr.training.artifacts import sha256_file

        result = _run(tmp_path)
        recorded = json.loads((result.directory / "checksums.json").read_text())
        for name, digest in recorded.items():
            assert sha256_file(result.directory / name) == digest

    def test_the_summary_carries_its_disclaimers(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        document = json.loads(
            (result.directory / "adaptation_summary.json").read_text()
        )
        joined = " ".join(document["disclaimers"])
        assert SOFTWARE_SELF_CHECK_BANNER in joined
        assert ADAPTATION_POLICY_NOTE in joined
        assert CONTROLLER_METRIC_NOTE in joined
        assert NAIVE_COMPARISON_NOTE in joined

    def test_the_scenario_document_states_each_expectation(
        self, tmp_path: Path
    ) -> None:
        result = _run(tmp_path)
        document = json.loads((result.directory / "scenarios.json").read_text())
        assert "CONTROLLER TESTS" in document["disclaimer"]
        assert len(document["scenarios"]) == len(_SMALL)
        for entry in document["scenarios"]:
            assert entry["description"] and entry["expectation"]


class TestTraceInvariants:
    def test_no_blocked_row_proposes(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        for row in pq.read_table(
            result.directory / "adaptation_trace.parquet"
        ).to_pylist():
            if "gate_blocked" in row["policy_reasons"]:
                assert row["decision_kind"] == "hold"
                assert row["proposal_id"] is None

    def test_no_abstained_row_proposes(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        for row in pq.read_table(
            result.directory / "adaptation_trace.parquet"
        ).to_pylist():
            if "prediction_abstained" in row["policy_reasons"]:
                assert row["decision_kind"] == "hold"

    def test_a_hold_row_carries_no_command(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        for row in pq.read_table(
            result.directory / "adaptation_trace.parquet"
        ).to_pylist():
            if row["decision_kind"] == "hold":
                assert row["proposal_id"] is None
                assert row["command_built"] is False
                assert row["lifecycle_status"] is None

    def test_every_proposal_is_in_bounds_and_satisfied_every_guard(
        self, tmp_path: Path
    ) -> None:
        result = _run(tmp_path)
        policy = result.summary.configuration
        for row in pq.read_table(
            result.directory / "adaptation_trace.parquet"
        ).to_pylist():
            if row["decision_kind"] != "propose_adaptation":
                continue
            assert row["engagement_gate_decision"] == "eligible"
            assert row["cognitive_load_gate_decision"] == "eligible"
            assert row["cooldown_before"] == 0
            assert (
                policy.difficulty.minimum
                <= row["proposed_difficulty"]
                <= policy.difficulty.maximum
            )
            assert row["persistence_before"] + 1 >= policy.minimum_persistence_windows
            assert row["adaptation_budget_used"] <= (
                row["adaptation_budget_total"] or row["adaptation_budget_used"]
            )
            assert row["policy_reasons"] == ["proposal_eligible"]

    def test_the_trace_records_both_targets_suggestions(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        columns = pq.read_table(
            result.directory / "adaptation_trace.parquet"
        ).column_names
        for name in (
            "engagement_state",
            "cognitive_load_state",
            "engagement_suggestion",
            "cognitive_load_suggestion",
            "conflict",
            "resolved_direction",
            "policy_reasons",
            "persistence_before",
            "persistence_after",
            "cooldown_before",
            "cooldown_after",
        ):
            assert name in columns


class TestPrivacy:
    def test_the_trace_carries_no_media_or_biometric_column(
        self, tmp_path: Path
    ) -> None:
        result = _run(tmp_path)
        columns = pq.read_table(
            result.directory / "adaptation_trace.parquet"
        ).column_names
        for forbidden in (
            "frame",
            "image",
            "landmark",
            "crop",
            "rgb",
            "heart_rate",
            "bpm",
            "name",
            "email",
            "token",
            "secret",
        ):
            assert not [c for c in columns if forbidden in c], forbidden

    def test_no_artifact_contains_an_address_or_a_secret(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        for path in result.directory.glob("*.json"):
            text = path.read_text(encoding="utf-8").lower()
            assert "@" not in text
            assert "password" not in text
            assert "api_key" not in text
            assert "secret" not in text

    def test_identifiers_are_pseudonymous(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        for row in pq.read_table(
            result.directory / "adaptation_trace.parquet"
        ).to_pylist():
            assert row["subject_id"].startswith("synthetic")
            assert row["session_id"].startswith("scn-")

    def test_the_synthetic_flag_persists_into_every_row(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        for row in pq.read_table(
            result.directory / "adaptation_trace.parquet"
        ).to_pylist():
            assert row["is_synthetic"] is True
            assert row["scientific_evaluation_eligible"] is False


class TestScientificEligibility:
    def test_a_synthetic_run_is_never_eligible(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        assert result.summary.scientific_evaluation_eligible is False
        assert result.summary.is_synthetic is True

    def test_scientific_mode_refuses_synthetic_inputs(self, tmp_path: Path) -> None:
        with pytest.raises(AdaptationRunError, match="refuses synthetic policy inputs"):
            _run(tmp_path, evaluation_mode=EvaluationMode.SCIENTIFIC)

    def test_an_empty_sequence_is_refused(self, tmp_path: Path) -> None:
        config = AdaptationRunConfiguration(
            output_directory=tmp_path / "run", policy=make_configuration()
        )
        with pytest.raises(AdaptationRunError, match="at least one window"):
            run_adaptation(config, inputs=[])


class TestMetrics:
    def test_the_counts_reconcile_with_the_trace(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        rows = pq.read_table(result.directory / "adaptation_trace.parquet").to_pylist()
        holds = sum(1 for r in rows if r["decision_kind"] == "hold")
        proposals = sum(1 for r in rows if r["decision_kind"] == "propose_adaptation")
        assert result.metrics.hold_decisions == holds
        assert result.metrics.adaptation_proposals == proposals
        assert result.metrics.evaluated_windows == len(rows)
        assert result.metrics.increases + result.metrics.decreases == proposals

    def test_a_command_is_built_for_every_proposal_and_none_is_sent(
        self, tmp_path: Path
    ) -> None:
        result = _run(tmp_path)
        assert len(result.history) == result.metrics.adaptation_proposals
        assert all(e.status.value == "command_built" for e in result.history)
        assert all(e.dispatched_at_utc is None for e in result.history)
        assert all(e.acknowledged is None for e in result.history)

    def test_the_metrics_name_no_human_outcome(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        fields = set(type(result.metrics).model_fields)
        for forbidden in (
            "improvement",
            "benefit",
            "effectiveness",
            "comfort",
            "learning",
            "engagement_gain",
        ):
            assert not [f for f in fields if forbidden in f]

    def test_the_naive_controller_acts_more_often(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        naive = result.summary.naive_comparison
        assert naive is not None
        assert naive.adaptation_proposals > result.metrics.adaptation_proposals

    def test_the_naive_comparison_can_be_skipped(self, tmp_path: Path) -> None:
        result = _run(tmp_path, compare_naive=False)
        assert result.summary.naive_comparison is None

    def test_the_naive_controller_drops_every_guard(self) -> None:
        naive = naive_configuration(make_configuration())
        assert naive.minimum_persistence_windows == 1
        assert naive.cooldown_windows == 0
        assert naive.max_adaptations_per_session is None

    def test_final_difficulty_is_the_reported_level(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        for session, difficulty in result.metrics.final_difficulty_by_session.items():
            assert result.final_states[session].current_difficulty == difficulty


class TestDeterminism:
    def test_two_runs_produce_an_identical_trace(self, tmp_path: Path) -> None:
        from engagevr.training.artifacts import sha256_file

        first = _run(tmp_path / "a", output_directory=tmp_path / "a")
        second = _run(tmp_path / "b", output_directory=tmp_path / "b")
        assert first.run_id == second.run_id
        assert sha256_file(first.directory / "adaptation_trace.parquet") == sha256_file(
            second.directory / "adaptation_trace.parquet"
        )
        assert sha256_file(
            first.directory / "adaptation_policy_config.json"
        ) == sha256_file(second.directory / "adaptation_policy_config.json")

    def test_the_trace_carries_no_wall_clock_column(self, tmp_path: Path) -> None:
        result = _run(tmp_path)
        columns = pq.read_table(
            result.directory / "adaptation_trace.parquet"
        ).column_names
        assert not [c for c in columns if "time" in c or "utc" in c or "clock" in c]

    def test_the_run_id_is_a_function_of_configuration_and_inputs(self) -> None:
        inputs = scenario_inputs(_SMALL)
        mode = EvaluationMode.SOFTWARE_SELF_CHECK
        base = build_run_id(make_configuration(), inputs, mode)
        assert base == build_run_id(make_configuration(), inputs, mode)
        assert base != build_run_id(
            make_configuration(cooldown_windows=1), inputs, mode
        )
        assert base != build_run_id(
            make_configuration(), scenario_inputs(SCENARIOS), mode
        )

    def test_trace_construction_is_pure(self) -> None:
        inputs = scenario_inputs(_SMALL)
        decisions, _ = evaluate_sequence(inputs, make_configuration())
        first = trace_table(decisions, (), run_id="r")
        second = trace_table(decisions, (), run_id="r")
        assert first.to_pylist() == second.to_pylist()


class TestExperimentModes:
    def test_the_static_condition_produces_no_proposal(self, tmp_path: Path) -> None:
        result = _run(
            tmp_path,
            policy=make_configuration(experiment_mode=ExperimentMode.STATIC),
        )
        assert result.metrics.adaptation_proposals == 0
        assert len(result.history) == 0

    def test_the_experimenter_lock_produces_no_proposal(self, tmp_path: Path) -> None:
        result = _run(tmp_path, policy=make_configuration(enabled=False))
        assert result.metrics.adaptation_proposals == 0

    def test_the_two_conditions_are_separately_recorded(self, tmp_path: Path) -> None:
        static = _run(
            tmp_path / "s",
            output_directory=tmp_path / "s",
            policy=make_configuration(experiment_mode=ExperimentMode.STATIC),
        )
        adaptive = _run(tmp_path / "a", output_directory=tmp_path / "a")
        assert static.summary.configuration.experiment_mode is ExperimentMode.STATIC
        assert adaptive.summary.configuration.experiment_mode is ExperimentMode.ADAPTIVE
        assert static.run_id != adaptive.run_id


class TestSessionScoping:
    def test_each_session_gets_its_own_state(self) -> None:
        inputs = scenario_inputs((get_scenario("session-change"),))
        _decisions, states = evaluate_sequence(inputs, make_configuration())
        assert len(states) == 2
        for session, state in states.items():
            assert state.session_id == session

    def test_no_identifier_leaks_between_sessions(self) -> None:
        inputs = scenario_inputs((get_scenario("session-change"),))
        decisions, _ = evaluate_sequence(inputs, make_configuration())
        for decision in decisions:
            assert decision.state_before.session_id == decision.session_id
            assert decision.state_after.session_id == decision.session_id

    def test_metrics_are_reported_per_session(self) -> None:
        inputs = scenario_inputs((get_scenario("session-change"),))
        decisions, states = evaluate_sequence(inputs, make_configuration())
        metrics = controller_metrics(decisions, states)
        assert len(metrics.proposals_by_session) == 2
        assert sum(metrics.proposals_by_session.values()) == (
            metrics.adaptation_proposals
        )

    def test_spacing_is_measured_within_a_session(self) -> None:
        inputs = scenario_inputs((get_scenario("cooldown-suppression"),))
        decisions, states = evaluate_sequence(inputs, make_configuration())
        metrics = controller_metrics(decisions, states)
        assert metrics.minimum_proposal_spacing_windows == 7


class TestFullSuite:
    def test_the_whole_suite_runs_and_reconciles(self, tmp_path: Path) -> None:
        config = AdaptationRunConfiguration(
            output_directory=tmp_path / "full", policy=make_configuration()
        )
        result = run_adaptation(config)
        assert result.summary.scenario_names == tuple(s.name for s in SCENARIOS)
        assert result.metrics.evaluated_windows == len(result.decisions)
        proposals = [
            d
            for d in result.decisions
            if d.kind is AdaptationDecisionKind.PROPOSE_ADAPTATION
        ]
        assert result.metrics.adaptation_proposals == len(proposals)
        assert result.metrics.direction_reversals >= 1
        assert result.metrics.blocked_oscillation_attempts >= 1
        assert all(
            d.resolved_direction
            in (AdaptationDirection.INCREASE, AdaptationDirection.DECREASE)
            for d in proposals
        )
