"""The conservative adaptation policy: gate, mapping, guards, and state.

Every test here runs the real policy against real Milestone 7 records.
No server, no WebSocket, no Unity, no task simulator, no webcam, no
MediaPipe asset, no network, and no external dataset is involved.
"""

from __future__ import annotations

import ast
from itertools import pairwise
from pathlib import Path

import pytest

from engagevr.adaptation.mapping import (
    MAPPING_TABLE,
    cognitive_load_suggestion,
    engagement_suggestion,
    ordinal_state_from_class,
    ordinal_state_from_value,
    resolve_direction,
)
from engagevr.adaptation.policy import evaluate_policy
from engagevr.adaptation.scenarios import (
    BLOCKED_BY_CONFIDENCE,
    BLOCKED_BY_QUALITY,
    EVIDENCE_MISSING,
    PREDICTION_MISSING,
)
from engagevr.schemas.adaptation_policy import (
    AdaptationDecisionKind,
    AdaptationDirection,
    AdaptationPolicyError,
    AdaptationPolicyReason,
    AdaptationPolicyState,
    ConflictResolution,
    DifficultyBounds,
    ExperimentMode,
    OrdinalState,
    RegressionBand,
)
from engagevr.schemas.targets import TargetName
from engagevr.schemas.uncertainty import AbstentionReason, AdaptationGateDecision
from tests.unit.adaptation_helpers import (
    SESSION,
    make_configuration,
    make_input,
    run_sequence,
    steady_inputs,
)


def _propose_at(decisions: list) -> list[int]:
    return [
        index
        for index, decision in enumerate(decisions)
        if decision.kind is AdaptationDecisionKind.PROPOSE_ADAPTATION
    ]


class TestGateIntegration:
    def test_an_eligible_gate_reaches_the_mapping(self) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 3))
        assert decisions[-1].kind is AdaptationDecisionKind.PROPOSE_ADAPTATION

    def test_a_blocked_gate_always_holds(self) -> None:
        decisions = run_sequence(
            [
                make_input(
                    engagement="medium",
                    cognitive_load="high",
                    cognitive_load_status=BLOCKED_BY_CONFIDENCE,
                    window_order=index,
                )
                for index in range(10)
            ]
        )
        assert all(d.held for d in decisions)
        assert all(AdaptationPolicyReason.GATE_BLOCKED in d.reasons for d in decisions)

    def test_a_blocked_gate_retains_the_milestone_seven_reasons(self) -> None:
        decision = run_sequence(
            [
                make_input(
                    engagement="medium",
                    cognitive_load="high",
                    cognitive_load_status=BLOCKED_BY_QUALITY,
                )
            ]
        )[0]
        assert (
            AbstentionReason.SIGNAL_QUALITY_BELOW_GATE
            in decision.cognitive_load.gate_reasons
        )
        assert AdaptationPolicyReason.GATE_BLOCKED in decision.reasons

    def test_an_abstained_prediction_always_holds(self) -> None:
        decisions = run_sequence(
            [
                make_input(
                    engagement="medium",
                    cognitive_load="high",
                    cognitive_load_status=BLOCKED_BY_CONFIDENCE,
                    window_order=index,
                )
                for index in range(6)
            ]
        )
        assert all(
            AdaptationPolicyReason.PREDICTION_ABSTAINED in d.reasons for d in decisions
        )
        assert not _propose_at(decisions)

    def test_an_unavailable_prediction_always_holds(self) -> None:
        decisions = run_sequence(
            [
                make_input(
                    engagement="medium",
                    cognitive_load=None,
                    cognitive_load_status=PREDICTION_MISSING,
                    window_order=index,
                )
                for index in range(6)
            ]
        )
        assert all(
            AdaptationPolicyReason.PREDICTION_UNAVAILABLE in d.reasons
            for d in decisions
        )
        assert not _propose_at(decisions)

    def test_missing_evidence_holds_without_becoming_an_error(self) -> None:
        decisions = run_sequence(
            [
                make_input(omit_cognitive_load=True, window_order=index)
                for index in range(6)
            ]
        )
        assert all(
            AdaptationPolicyReason.INSUFFICIENT_EVIDENCE in d.reasons for d in decisions
        )

    def test_an_evidence_gate_failure_holds(self) -> None:
        decision = run_sequence(
            [
                make_input(
                    engagement="medium",
                    cognitive_load="high",
                    cognitive_load_status=EVIDENCE_MISSING,
                )
            ]
        )[0]
        assert decision.held
        assert (
            AbstentionReason.REQUIRED_MODALITY_UNAVAILABLE
            in decision.cognitive_load.gate_reasons
        )

    def test_no_module_can_convert_blocked_into_eligible(self) -> None:
        # A structural check rather than a behavioural one: no Milestone 8
        # module may name the eligible member on the left of an assignment,
        # construct a gate record, or call the gate evaluator with an
        # override. The gate is consumed, never re-derived.
        package = Path("src/engagevr/adaptation")
        for path in sorted(package.glob("*.py")):
            if path.name == "scenarios.py":
                continue  # scenarios build evidence THROUGH the real gate
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id != "AdaptationGateRecord", (
                        f"{path.name} constructs a gate record; Milestone 8 "
                        "consumes Milestone 7's verdict and never mints one"
                    )
                    assert node.func.id != "evaluate_adaptation_gate", (
                        f"{path.name} re-runs the Milestone 7 gate"
                    )

    def test_the_policy_imports_no_transport_or_inference_module(self) -> None:
        tree = ast.parse(
            Path("src/engagevr/adaptation/policy.py").read_text(encoding="utf-8")
        )
        forbidden = (
            "engagevr.api",
            "engagevr.transport",
            "engagevr.task",
            "engagevr.training",
        )
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        for module in modules:
            assert not any(module.startswith(f) for f in forbidden), (
                f"policy.py imports {module}; it must be unable to send "
                "anything or to recompute a Milestone 7 quantity"
            )


class TestMapping:
    @pytest.mark.parametrize(
        ("engagement", "load", "direction"),
        [
            ("low", "low", AdaptationDirection.HOLD),
            ("low", "medium", AdaptationDirection.HOLD),
            ("low", "high", AdaptationDirection.DECREASE),
            ("medium", "low", AdaptationDirection.HOLD),
            ("medium", "medium", AdaptationDirection.HOLD),
            ("medium", "high", AdaptationDirection.DECREASE),
            ("high", "low", AdaptationDirection.INCREASE),
            ("high", "medium", AdaptationDirection.HOLD),
            ("high", "high", AdaptationDirection.HOLD),
        ],
    )
    def test_every_state_pair_maps_as_documented(
        self, engagement: str, load: str, direction: AdaptationDirection
    ) -> None:
        result = resolve_direction(OrdinalState(engagement), OrdinalState(load))
        assert result.direction is direction

    def test_the_table_is_total(self) -> None:
        assert len(MAPPING_TABLE) == 9
        for engagement in OrdinalState:
            for load in OrdinalState:
                assert (engagement, load) in MAPPING_TABLE

    def test_the_neutral_pair_records_the_deadband(self) -> None:
        result = resolve_direction(OrdinalState.MEDIUM, OrdinalState.MEDIUM)
        assert result.reason is AdaptationPolicyReason.TARGET_IN_DEADBAND

    def test_low_engagement_records_an_inexpressible_action(self) -> None:
        result = resolve_direction(OrdinalState.LOW, OrdinalState.LOW)
        assert result.reason is AdaptationPolicyReason.NO_EXPRESSIBLE_ACTION

    def test_conflicting_directions_hold_by_default(self) -> None:
        result = resolve_direction(OrdinalState.HIGH, OrdinalState.HIGH)
        assert result.direction is AdaptationDirection.HOLD
        assert result.conflict is True
        assert result.resolution is ConflictResolution.HOLD

    def test_prefer_decrease_takes_the_protective_direction(self) -> None:
        result = resolve_direction(
            OrdinalState.HIGH,
            OrdinalState.HIGH,
            conflict_resolution=ConflictResolution.PREFER_DECREASE,
        )
        assert result.direction is AdaptationDirection.DECREASE
        assert result.conflict is True

    def test_single_signal_suggestions_never_reach_for_an_increase(self) -> None:
        assert cognitive_load_suggestion(OrdinalState.LOW) is AdaptationDirection.HOLD
        assert engagement_suggestion(OrdinalState.LOW) is AdaptationDirection.HOLD
        assert engagement_suggestion(OrdinalState.HIGH) is AdaptationDirection.INCREASE
        assert (
            cognitive_load_suggestion(OrdinalState.HIGH) is AdaptationDirection.DECREASE
        )

    def test_an_unknown_class_label_is_refused(self) -> None:
        with pytest.raises(
            AdaptationPolicyError, match="not a state this policy knows"
        ):
            ordinal_state_from_class(TargetName.ENGAGEMENT_CLASS, "very_high")

    def test_a_non_ordinal_target_is_refused(self) -> None:
        with pytest.raises(AdaptationPolicyError, match="is a regression target"):
            ordinal_state_from_class(TargetName.ENGAGEMENT_SCORE, "high")

    def test_mapping_is_deterministic(self) -> None:
        first = [
            resolve_direction(e, load).direction
            for e in OrdinalState
            for load in OrdinalState
        ]
        second = [
            resolve_direction(e, load).direction
            for e in OrdinalState
            for load in OrdinalState
        ]
        assert first == second

    def test_quality_cannot_produce_a_direction(self) -> None:
        # Two windows with identical states but very different recorded
        # quality resolve identically: quality reaches the policy only as
        # Milestone 7 provenance.
        good = run_sequence(steady_inputs("high", "low", 1))[0]
        assert good.engagement.minimum_recorded_quality is not None
        assert good.resolved_direction is AdaptationDirection.INCREASE
        blocked = run_sequence(
            [
                make_input(
                    engagement="high",
                    cognitive_load="low",
                    cognitive_load_status=BLOCKED_BY_QUALITY,
                )
            ]
        )[0]
        # Poor quality holds; it does not become a decrease.
        assert blocked.resolved_direction is AdaptationDirection.HOLD
        assert AdaptationPolicyReason.GATE_BLOCKED in blocked.reasons


class TestRegressionMapping:
    def _band(self) -> RegressionBand:
        return RegressionBand(
            target_name=TargetName.ENGAGEMENT_SCORE,
            low_below=0.3,
            high_above=0.7,
            unit="dimensionless",
        )

    def test_values_map_to_their_regions(self) -> None:
        band = self._band()
        assert ordinal_state_from_value(band, 0.1) is OrdinalState.LOW
        assert ordinal_state_from_value(band, 0.5) is OrdinalState.MEDIUM
        assert ordinal_state_from_value(band, 0.9) is OrdinalState.HIGH

    def test_the_boundaries_themselves_are_neutral(self) -> None:
        band = self._band()
        assert ordinal_state_from_value(band, 0.3) is OrdinalState.MEDIUM
        assert ordinal_state_from_value(band, 0.7) is OrdinalState.MEDIUM

    def test_just_inside_a_boundary_is_not_neutral(self) -> None:
        band = self._band()
        assert ordinal_state_from_value(band, 0.3 - 1e-9) is OrdinalState.LOW
        assert ordinal_state_from_value(band, 0.7 + 1e-9) is OrdinalState.HIGH

    def test_an_interval_straddling_a_boundary_reads_as_neutral(self) -> None:
        band = self._band()
        assert (
            ordinal_state_from_value(
                band,
                0.75,
                interval_lower_bound=0.65,
                interval_upper_bound=0.85,
            )
            is OrdinalState.MEDIUM
        )

    def test_an_interval_inside_the_band_keeps_its_state(self) -> None:
        band = self._band()
        assert (
            ordinal_state_from_value(
                band,
                0.85,
                interval_lower_bound=0.75,
                interval_upper_bound=0.95,
            )
            is OrdinalState.HIGH
        )

    def test_a_non_finite_estimate_is_refused(self) -> None:
        with pytest.raises(AdaptationPolicyError, match="not finite"):
            ordinal_state_from_value(self._band(), float("inf"))

    def test_a_point_outside_its_own_interval_is_refused(self) -> None:
        with pytest.raises(AdaptationPolicyError, match="outside its interval"):
            ordinal_state_from_value(
                self._band(),
                0.9,
                interval_lower_bound=0.1,
                interval_upper_bound=0.5,
            )


class TestBounds:
    def test_a_decrease_moves_exactly_one_step(self) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 3, difficulty=4))
        proposal = decisions[2].proposal
        assert proposal is not None
        assert proposal.current_difficulty == 4
        assert proposal.proposed_difficulty == 3
        assert proposal.clamping_applied is False

    def test_an_increase_moves_exactly_one_step(self) -> None:
        decisions = run_sequence(steady_inputs("high", "low", 3, difficulty=2))
        proposal = decisions[2].proposal
        assert proposal is not None
        assert proposal.proposed_difficulty == 3

    def test_the_minimum_boundary_holds(self) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 5, difficulty=1))
        assert not _propose_at(decisions)
        assert AdaptationPolicyReason.ALREADY_AT_MINIMUM in decisions[-1].reasons

    def test_the_maximum_boundary_holds(self) -> None:
        decisions = run_sequence(steady_inputs("high", "low", 5, difficulty=5))
        assert not _propose_at(decisions)
        assert AdaptationPolicyReason.ALREADY_AT_MAXIMUM in decisions[-1].reasons

    def test_an_unknown_current_difficulty_holds_rather_than_becoming_zero(
        self,
    ) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 4, difficulty=None))
        assert not _propose_at(decisions)
        assert AdaptationPolicyReason.CURRENT_STATE_UNAVAILABLE in decisions[-1].reasons
        assert decisions[-1].current_difficulty is None

    def test_an_out_of_bounds_current_difficulty_is_refused(self) -> None:
        with pytest.raises(
            AdaptationPolicyError, match="outside the configured bounds"
        ):
            run_sequence([make_input(current_difficulty=99)])

    def test_a_step_larger_than_one_records_its_clamping(self) -> None:
        config = make_configuration(
            difficulty=DifficultyBounds(minimum=1, maximum=5, step=3)
        )
        decisions = run_sequence(
            steady_inputs("medium", "high", 3, difficulty=2), config
        )
        proposal = decisions[2].proposal
        assert proposal is not None
        assert proposal.requested_difficulty == -1
        assert proposal.proposed_difficulty == 1
        assert proposal.clamping_applied is True


class TestPersistence:
    def test_the_first_supporting_window_holds(self) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 1))
        assert decisions[0].held
        assert AdaptationPolicyReason.INSUFFICIENT_PERSISTENCE in decisions[0].reasons
        assert decisions[0].persistence_count_after == 1

    def test_the_nth_consecutive_window_may_adapt(self) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 3))
        assert _propose_at(decisions) == [2]

    def test_a_configured_dwell_of_one_adapts_immediately(self) -> None:
        decisions = run_sequence(
            steady_inputs("medium", "high", 1),
            make_configuration(minimum_persistence_windows=1),
        )
        assert _propose_at(decisions) == [0]

    def test_a_direction_change_resets_the_count(self) -> None:
        inputs = [
            *steady_inputs("medium", "high", 2, difficulty=3),
            make_input(
                engagement="high",
                cognitive_load="low",
                window_order=2,
                current_difficulty=3,
            ),
        ]
        decisions = run_sequence(inputs)
        assert decisions[2].persistence_count_after == 1
        assert AdaptationPolicyReason.DIRECTION_CHANGE_BLOCKED in decisions[2].reasons

    def test_a_hold_resets_the_count_rather_than_decaying_it(self) -> None:
        inputs = [
            *steady_inputs("medium", "high", 2, difficulty=3),
            make_input(
                engagement="medium",
                cognitive_load="medium",
                window_order=2,
                current_difficulty=3,
            ),
            *steady_inputs("medium", "high", 2, start=3, difficulty=3),
        ]
        decisions = run_sequence(inputs)
        assert decisions[2].persistence_count_after == 0
        assert decisions[3].persistence_count_after == 1
        assert decisions[4].persistence_count_after == 2
        assert not _propose_at(decisions)

    def test_a_blocked_window_does_not_count_as_supporting_evidence(self) -> None:
        inputs = [
            *steady_inputs("medium", "high", 2, difficulty=3),
            make_input(
                engagement="medium",
                cognitive_load="high",
                cognitive_load_status=BLOCKED_BY_CONFIDENCE,
                window_order=2,
                current_difficulty=3,
            ),
            *steady_inputs("medium", "high", 2, start=3, difficulty=3),
        ]
        decisions = run_sequence(inputs)
        assert decisions[2].persistence_count_after == 0
        assert not _propose_at(decisions)

    def test_a_duplicate_window_does_not_increase_the_count(self) -> None:
        first = make_input(engagement="medium", cognitive_load="high", window_order=0)
        decisions = run_sequence([first, first, first])
        assert decisions[0].persistence_count_after == 1
        assert decisions[1].persistence_count_after == 1
        assert AdaptationPolicyReason.DUPLICATE_WINDOW in decisions[1].reasons
        assert decisions[1].state_after == decisions[0].state_after

    def test_an_out_of_order_window_is_refused(self) -> None:
        inputs = [
            make_input(engagement="medium", cognitive_load="high", window_order=5),
            make_input(engagement="medium", cognitive_load="high", window_order=2),
        ]
        with pytest.raises(AdaptationPolicyError, match="Out-of-order windows"):
            run_sequence(inputs)

    def test_one_window_id_cannot_occupy_two_positions(self) -> None:
        inputs = [
            make_input(window_id="w", window_order=0),
            make_input(window_id="w", window_order=1),
        ]
        with pytest.raises(AdaptationPolicyError, match="two positions"):
            run_sequence(inputs)


class TestCooldown:
    def test_a_proposal_starts_the_cooldown(self) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 3))
        assert decisions[2].cooldown_remaining_after == 6

    def test_adaptation_during_cooldown_holds(self) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 8))
        assert _propose_at(decisions) == [2]
        assert all(
            AdaptationPolicyReason.COOLDOWN_ACTIVE in d.reasons for d in decisions[5:8]
        )

    def test_the_remaining_cooldown_counts_down_deterministically(self) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 9))
        remaining = [d.cooldown_remaining_after for d in decisions]
        assert remaining == [0, 0, 6, 5, 4, 3, 2, 1, 0]

    def test_the_cooldown_eventually_expires_and_a_second_proposal_follows(
        self,
    ) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 12))
        assert _propose_at(decisions) == [2, 9]

    def test_the_minimum_spacing_is_cooldown_plus_one(self) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 20))
        orders = [decisions[i].window_order for i in _propose_at(decisions)]
        spacings = [b - a for a, b in pairwise(orders)]
        assert spacings and all(s == 7 for s in spacings)

    def test_a_blocked_window_still_advances_the_cooldown(self) -> None:
        inputs = [
            *steady_inputs("medium", "high", 3, difficulty=3),
            *[
                make_input(
                    engagement="medium",
                    cognitive_load="high",
                    cognitive_load_status=BLOCKED_BY_CONFIDENCE,
                    window_order=3 + index,
                    current_difficulty=3,
                )
                for index in range(3)
            ],
        ]
        decisions = run_sequence(inputs)
        # Time passing is a property of the stream, not of the evidence.
        assert decisions[5].cooldown_remaining_after == 3

    def test_a_new_session_resets_the_cooldown(self) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 3))
        assert decisions[2].state_after.cooldown_remaining == 6
        fresh = AdaptationPolicyState.for_session("other-session")
        assert fresh.cooldown_remaining == 0

    def test_a_zero_cooldown_permits_consecutive_proposals(self) -> None:
        decisions = run_sequence(
            steady_inputs("medium", "high", 6),
            make_configuration(cooldown_windows=0, minimum_persistence_windows=1),
        )
        assert len(_propose_at(decisions)) == 6


class TestHysteresis:
    def test_an_immediate_reversal_is_blocked(self) -> None:
        inputs = [
            *steady_inputs("medium", "high", 3, difficulty=3),
            make_input(
                engagement="high",
                cognitive_load="low",
                window_order=3,
                current_difficulty=2,
            ),
        ]
        decisions = run_sequence(inputs)
        assert _propose_at(decisions) == [2]
        assert decisions[3].held

    def test_an_isolated_opposite_signal_does_not_reverse(self) -> None:
        inputs = [
            *steady_inputs("medium", "high", 3, difficulty=3),
            make_input(
                engagement="high",
                cognitive_load="low",
                window_order=3,
                current_difficulty=2,
            ),
            *steady_inputs("medium", "medium", 8, start=4, difficulty=2),
        ]
        decisions = run_sequence(inputs)
        assert _propose_at(decisions) == [2]

    def test_fresh_persistent_opposite_evidence_eventually_reverses(self) -> None:
        inputs = [
            *steady_inputs("medium", "high", 3, difficulty=3),
            *[
                make_input(
                    engagement="high",
                    cognitive_load="low",
                    window_order=3 + index,
                    current_difficulty=2,
                )
                for index in range(10)
            ],
        ]
        decisions = run_sequence(inputs)
        proposals = _propose_at(decisions)
        assert len(proposals) == 2
        first, second = proposals
        assert decisions[first].resolved_direction is AdaptationDirection.DECREASE
        assert decisions[second].resolved_direction is AdaptationDirection.INCREASE
        # The reversal needed both a fresh dwell and the cooldown to expire.
        assert second - first >= 7

    def test_hysteresis_needs_no_dedicated_knob(self) -> None:
        from engagevr.schemas.adaptation_policy import AdaptationPolicyConfiguration

        assert not [
            name
            for name in AdaptationPolicyConfiguration.model_fields
            if "hysteresis" in name
        ]


class TestBudget:
    def test_adaptations_consume_the_budget(self) -> None:
        decisions = run_sequence(
            steady_inputs("medium", "high", 12),
            make_configuration(max_adaptations_per_session=2),
        )
        assert decisions[-1].adaptation_budget_used == 2

    def test_holds_do_not_consume_the_budget(self) -> None:
        decisions = run_sequence(steady_inputs("medium", "medium", 8))
        assert all(d.adaptation_budget_used == 0 for d in decisions)

    def test_exhaustion_produces_a_hold(self) -> None:
        decisions = run_sequence(
            steady_inputs("medium", "high", 20),
            make_configuration(max_adaptations_per_session=1),
        )
        assert len(_propose_at(decisions)) == 1
        assert (
            AdaptationPolicyReason.SESSION_ADAPTATION_BUDGET_EXHAUSTED
            in decisions[-1].reasons
        )

    def test_a_zero_budget_permits_nothing(self) -> None:
        decisions = run_sequence(
            steady_inputs("medium", "high", 6),
            make_configuration(max_adaptations_per_session=0),
        )
        assert not _propose_at(decisions)

    def test_a_null_budget_is_unlimited(self) -> None:
        decisions = run_sequence(
            steady_inputs("medium", "high", 30),
            make_configuration(
                max_adaptations_per_session=None,
                cooldown_windows=0,
                minimum_persistence_windows=1,
            ),
        )
        assert len(_propose_at(decisions)) == 30

    def test_a_new_session_resets_the_budget(self) -> None:
        assert AdaptationPolicyState.for_session("fresh").adaptation_count == 0


class TestExperimenterControls:
    def test_disabling_adaptation_holds_every_window(self) -> None:
        decisions = run_sequence(
            steady_inputs("medium", "high", 10), make_configuration(enabled=False)
        )
        assert not _propose_at(decisions)
        assert all(
            AdaptationPolicyReason.ADAPTATION_DISABLED in d.reasons for d in decisions
        )

    def test_the_static_condition_holds_every_window(self) -> None:
        decisions = run_sequence(
            steady_inputs("medium", "high", 10),
            make_configuration(experiment_mode=ExperimentMode.STATIC),
        )
        assert not _propose_at(decisions)
        assert all(
            AdaptationPolicyReason.STATIC_EXPERIMENT_MODE in d.reasons
            for d in decisions
        )

    def test_the_lock_and_the_condition_are_distinguishable(self) -> None:
        locked = run_sequence(
            steady_inputs("medium", "high", 1), make_configuration(enabled=False)
        )[0]
        static = run_sequence(
            steady_inputs("medium", "high", 1),
            make_configuration(experiment_mode=ExperimentMode.STATIC),
        )[0]
        assert locked.reasons != static.reasons
        assert static.experiment_mode is ExperimentMode.STATIC


class TestState:
    def test_the_policy_module_holds_no_mutable_global(self) -> None:
        tree = ast.parse(
            Path("src/engagevr/adaptation/policy.py").read_text(encoding="utf-8")
        )
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    assert isinstance(target, ast.Name)
                    assert target.id.isupper() or target.id.startswith("_"), (
                        f"module-level assignment to {target.id!r}"
                    )
                assert isinstance(node.value, ast.Constant | ast.JoinedStr | ast.List)

    def test_the_input_state_is_not_mutated(self) -> None:
        state = AdaptationPolicyState.for_session(SESSION)
        snapshot = state.model_dump()
        evaluate_policy(
            make_input(engagement="medium", cognitive_load="high"),
            state,
            make_configuration(),
        )
        assert state.model_dump() == snapshot

    def test_the_same_triple_produces_the_same_output(self) -> None:
        policy_input = make_input(engagement="medium", cognitive_load="high")
        state = AdaptationPolicyState.for_session(SESSION)
        config = make_configuration()
        first = evaluate_policy(policy_input, state, config)
        second = evaluate_policy(policy_input, state, config)
        assert first.model_dump() == second.model_dump()
        assert first.state_after == second.state_after

    def test_one_sessions_state_is_refused_for_another(self) -> None:
        state = AdaptationPolicyState.for_session("session-a")
        with pytest.raises(AdaptationPolicyError, match="belongs to session"):
            evaluate_policy(
                make_input(session_id="session-b"), state, make_configuration()
            )

    def test_the_state_records_reported_difficulty_not_proposed(self) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 3, difficulty=4))
        proposal = decisions[2].proposal
        assert proposal is not None and proposal.proposed_difficulty == 3
        # A proposal is not an applied change: the state still records what
        # the environment reported.
        assert decisions[2].state_after.current_difficulty == 4

    def test_state_serialization_round_trips_through_a_run(self) -> None:
        decisions = run_sequence(steady_inputs("medium", "high", 5))
        final = decisions[-1].state_after
        restored = AdaptationPolicyState.model_validate_json(final.model_dump_json())
        assert restored == final


class TestProvenance:
    def test_a_decision_answers_every_provenance_question(self) -> None:
        decision = run_sequence(steady_inputs("medium", "high", 3))[2]
        assert decision.engagement.gate_decision is AdaptationGateDecision.ELIGIBLE
        assert decision.cognitive_load.gate_decision is AdaptationGateDecision.ELIGIBLE
        assert decision.engagement.state is OrdinalState.MEDIUM
        assert decision.cognitive_load.state is OrdinalState.HIGH
        assert decision.engagement.suggested_direction is AdaptationDirection.HOLD
        assert (
            decision.cognitive_load.suggested_direction is AdaptationDirection.DECREASE
        )
        assert decision.conflict is False
        assert decision.persistence_count_before == 2
        # A proposal restarts the dwell count: the next one needs fresh
        # evidence rather than inheriting this one's.
        assert decision.persistence_count_after == 0
        assert decision.cooldown_remaining_before == 0
        assert decision.adaptation_budget_total == 10
        assert decision.proposal is not None
        assert decision.proposal.current_difficulty == 3
        assert decision.proposal.proposed_difficulty == 2
        assert decision.proposal.persistence_count == 3
        assert decision.resolution_note

    def test_a_hold_states_why(self) -> None:
        decision = run_sequence(steady_inputs("medium", "medium", 1))[0]
        assert decision.reasons == (AdaptationPolicyReason.TARGET_IN_DEADBAND,)
        assert decision.resolution_note

    def test_a_proposal_id_is_deterministic(self) -> None:
        first = run_sequence(steady_inputs("medium", "high", 3))[2].proposal
        second = run_sequence(steady_inputs("medium", "high", 3))[2].proposal
        assert first is not None and second is not None
        assert first.proposal_id == second.proposal_id

    def test_a_proposal_id_changes_with_the_configuration(self) -> None:
        first = run_sequence(steady_inputs("medium", "high", 3))[2].proposal
        second = run_sequence(
            steady_inputs("medium", "high", 3),
            make_configuration(cooldown_windows=2),
        )[2].proposal
        assert first is not None and second is not None
        assert first.proposal_id != second.proposal_id

    def test_confidence_is_recorded_but_never_scales_the_step(self) -> None:
        decision = run_sequence(steady_inputs("medium", "high", 3))[2]
        assert decision.cognitive_load.confidence_score is not None
        proposal = decision.proposal
        assert proposal is not None
        assert proposal.step == 1
        assert abs(proposal.current_difficulty - proposal.proposed_difficulty) == 1
