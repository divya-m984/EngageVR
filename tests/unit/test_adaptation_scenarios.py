"""The deterministic controller-scenario suite.

Each scenario exercises one guard.  These are controller tests: no
window state here describes a person, and nothing in this module is a
simulated participant response.
"""

from __future__ import annotations

import pytest

from engagevr.adaptation.runner import evaluate_sequence
from engagevr.adaptation.scenarios import (
    SCENARIO_DISCLAIMER,
    SCENARIO_NAMES,
    SCENARIOS,
    build_inputs,
    get_scenario,
)
from engagevr.schemas.adaptation_policy import (
    AdaptationDecisionKind,
    AdaptationDirection,
    AdaptationPolicyReason,
)
from engagevr.schemas.uncertainty import AbstentionReason, AdaptationGateDecision
from tests.unit.adaptation_helpers import make_configuration


def _run(name: str) -> list:
    decisions, _ = evaluate_sequence(
        build_inputs(get_scenario(name)), make_configuration()
    )
    return list(decisions)


def _proposals(decisions: list) -> list:
    return [d for d in decisions if d.kind is AdaptationDecisionKind.PROPOSE_ADAPTATION]


class TestSuiteShape:
    def test_the_suite_covers_fifteen_named_scenarios(self) -> None:
        assert len(SCENARIOS) == 15
        assert len(set(SCENARIO_NAMES)) == 15

    def test_every_scenario_states_what_it_exercises(self) -> None:
        for scenario in SCENARIOS:
            assert scenario.description
            assert scenario.expectation
            assert scenario.windows

    def test_an_unknown_scenario_is_refused(self) -> None:
        with pytest.raises(KeyError, match="unknown scenario"):
            get_scenario("no-such-scenario")

    def test_the_disclaimer_denies_the_participant_reading(self) -> None:
        assert "CONTROLLER TESTS" in SCENARIO_DISCLAIMER
        assert "do not simulate a participant" in SCENARIO_DISCLAIMER


class TestScenarioBehaviour:
    def test_1_stable_neutral_state_holds_throughout(self) -> None:
        decisions = _run("stable-neutral")
        assert not _proposals(decisions)
        assert all(
            AdaptationPolicyReason.TARGET_IN_DEADBAND in d.reasons for d in decisions
        )

    def test_2_persistent_increase_evidence_adapts_once_after_the_dwell(self) -> None:
        decisions = _run("persistent-increase")
        proposals = _proposals(decisions)
        assert len(proposals) == 1
        assert proposals[0].window_order == 2
        assert proposals[0].resolved_direction is AdaptationDirection.INCREASE
        assert proposals[0].proposal is not None
        assert proposals[0].proposal.proposed_difficulty == 4

    def test_3_persistent_decrease_evidence_adapts_once_after_the_dwell(self) -> None:
        decisions = _run("persistent-decrease")
        proposals = _proposals(decisions)
        assert len(proposals) == 1
        assert proposals[0].window_order == 2
        assert proposals[0].resolved_direction is AdaptationDirection.DECREASE

    def test_4_a_one_window_spike_never_adapts(self) -> None:
        decisions = _run("single-window-spike")
        assert not _proposals(decisions)
        spike = decisions[3]
        assert spike.resolved_direction is AdaptationDirection.DECREASE
        assert AdaptationPolicyReason.INSUFFICIENT_PERSISTENCE in spike.reasons

    def test_5_conflicting_evidence_holds(self) -> None:
        decisions = _run("conflicting-evidence")
        assert not _proposals(decisions)
        assert all(d.conflict for d in decisions)
        assert all(
            AdaptationPolicyReason.DIRECTION_CONFLICT in d.reasons for d in decisions
        )

    def test_6_a_blocked_gate_holds_and_keeps_its_milestone_seven_reason(self) -> None:
        decisions = _run("gate-blocked")
        assert not _proposals(decisions)
        for decision in decisions:
            assert AdaptationPolicyReason.GATE_BLOCKED in decision.reasons
            assert (
                AbstentionReason.SIGNAL_QUALITY_BELOW_GATE
                in decision.cognitive_load.gate_reasons
            )
            # Quality never became a direction.
            assert decision.resolved_direction is AdaptationDirection.HOLD

    def test_7_an_abstained_prediction_holds(self) -> None:
        decisions = _run("prediction-abstained")
        assert not _proposals(decisions)
        for decision in decisions:
            assert AdaptationPolicyReason.PREDICTION_ABSTAINED in decision.reasons
            assert (
                AbstentionReason.BELOW_CONFIDENCE_THRESHOLD
                in decision.cognitive_load.gate_reasons
            )

    def test_8_cooldown_suppresses_repeated_action(self) -> None:
        decisions = _run("cooldown-suppression")
        proposals = _proposals(decisions)
        assert len(proposals) == 2
        assert proposals[1].window_order - proposals[0].window_order == 7
        held = [
            d for d in decisions if AdaptationPolicyReason.COOLDOWN_ACTIVE in d.reasons
        ]
        assert held

    def test_9_a_direction_reversal_requires_fresh_evidence(self) -> None:
        decisions = _run("direction-reversal")
        proposals = _proposals(decisions)
        assert len(proposals) == 2
        assert proposals[0].resolved_direction is AdaptationDirection.DECREASE
        assert proposals[1].resolved_direction is AdaptationDirection.INCREASE
        assert proposals[1].window_order - proposals[0].window_order >= 7
        assert proposals[1].proposal is not None
        assert proposals[1].proposal.persistence_count >= 3

    def test_10_the_minimum_bound_holds(self) -> None:
        decisions = _run("minimum-bound")
        assert not _proposals(decisions)
        assert any(
            AdaptationPolicyReason.ALREADY_AT_MINIMUM in d.reasons for d in decisions
        )

    def test_11_the_maximum_bound_holds(self) -> None:
        decisions = _run("maximum-bound")
        assert not _proposals(decisions)
        assert any(
            AdaptationPolicyReason.ALREADY_AT_MAXIMUM in d.reasons for d in decisions
        )

    def test_12_the_session_budget_stops_further_proposals(self) -> None:
        decisions = _run("budget-exhausted")
        proposals = _proposals(decisions)
        assert len(proposals) == 10
        assert any(
            AdaptationPolicyReason.SESSION_ADAPTATION_BUDGET_EXHAUSTED in d.reasons
            for d in decisions
        )
        assert decisions[-1].adaptation_budget_used == 10

    def test_13_a_session_change_resets_the_temporal_state(self) -> None:
        decisions = _run("session-change")
        sessions = {d.session_id for d in decisions}
        assert len(sessions) == 2
        second = [d for d in decisions if d.session_id.endswith("-b")]
        assert second[0].cooldown_remaining_before == 0
        assert second[0].persistence_count_before == 0
        assert second[0].state_before.last_applied_direction is None
        assert second[0].state_before.adaptation_count == 0
        assert _proposals(second)

    def test_14_a_duplicate_window_is_absorbed(self) -> None:
        decisions = _run("duplicate-window")
        duplicate = decisions[1]
        assert AdaptationPolicyReason.DUPLICATE_WINDOW in duplicate.reasons
        assert duplicate.state_after == duplicate.state_before
        # Three DISTINCT supporting windows were still required.
        assert _proposals(decisions)[0].window_id == "dup-w002"

    def test_15_no_usable_target_holds(self) -> None:
        decisions = _run("no-usable-target")
        assert not _proposals(decisions)
        for decision in decisions:
            assert AdaptationPolicyReason.INSUFFICIENT_EVIDENCE in decision.reasons
            assert decision.cognitive_load.evidence_available is False
            assert decision.cognitive_load.gate_decision is None


class TestGateIsRealInScenarios:
    def test_a_blocked_window_cannot_be_dressed_as_eligible(self) -> None:
        inputs = build_inputs(get_scenario("prediction-abstained"))
        for policy_input in inputs:
            assert policy_input.cognitive_load is not None
            assert (
                policy_input.cognitive_load.gate.decision
                is AdaptationGateDecision.BLOCKED
            )

    def test_every_scenario_input_is_permanently_synthetic(self) -> None:
        for scenario in SCENARIOS:
            for policy_input in build_inputs(scenario):
                assert policy_input.is_synthetic is True
                assert policy_input.scientific_evaluation_eligible is False
                for evidence in (policy_input.engagement, policy_input.cognitive_load):
                    if evidence is None:
                        continue
                    assert evidence.decision.is_synthetic is True
                    assert evidence.decision.scientific_evaluation_eligible is False
                    assert evidence.gate.scientific_evaluation_eligible is False


class TestDeterminism:
    def test_building_a_scenario_twice_yields_identical_inputs(self) -> None:
        for scenario in SCENARIOS:
            first = [i.model_dump(mode="json") for i in build_inputs(scenario)]
            second = [i.model_dump(mode="json") for i in build_inputs(scenario)]
            assert first == second

    def test_running_the_suite_twice_yields_identical_decisions(self) -> None:
        config = make_configuration()
        inputs = [i for s in SCENARIOS for i in build_inputs(s)]
        first, first_states = evaluate_sequence(inputs, config)
        second, second_states = evaluate_sequence(inputs, config)
        assert [d.model_dump(mode="json") for d in first] == [
            d.model_dump(mode="json") for d in second
        ]
        assert first_states == second_states

    def test_no_scenario_carries_a_personal_identifier(self) -> None:
        for scenario in SCENARIOS:
            assert "@" not in scenario.subject_id
            assert scenario.subject_id.startswith("synthetic")
            for policy_input in build_inputs(scenario):
                text = policy_input.model_dump_json()
                assert "@" not in text
                assert "password" not in text.lower()
                assert "landmark" not in text.lower()
