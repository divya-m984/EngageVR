"""Milestone 8 schema invariants.

These tests assert what the schema layer refuses.  A great deal of the
policy's safety is structural rather than procedural: a proposal that
carries a blocked gate, a hold that carries a command, or a synthetic
record that claims scientific eligibility cannot be constructed at all,
so no code path can produce one by accident.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from engagevr.schemas.adaptation_policy import (
    ADAPTATION_POLICY_NOTE,
    CONTROLLER_METRIC_NOTE,
    POLICY_REASON_ORDER,
    AdaptationControllerMetrics,
    AdaptationDecisionKind,
    AdaptationDirection,
    AdaptationHistoryEntry,
    AdaptationLifecycleStatus,
    AdaptationPolicyConfiguration,
    AdaptationPolicyMode,
    AdaptationPolicyReason,
    AdaptationPolicyState,
    AdaptationRunSummary,
    ConflictResolution,
    DifficultyBounds,
    ExperimentMode,
    OrdinalState,
    RegressionBand,
    opposite_direction,
)
from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER, EvaluationMode
from engagevr.schemas.targets import TARGET_SPECS, TargetName
from engagevr.utils.timestamps import utc_now
from tests.unit.adaptation_helpers import (
    build_decision,
    build_proposal,
    make_configuration,
    make_evidence,
)


class TestOrdinalDeclaration:
    def test_class_targets_declare_their_order(self) -> None:
        for name in (TargetName.ENGAGEMENT_CLASS, TargetName.COGNITIVE_LOAD_CLASS):
            assert TARGET_SPECS[name].class_order_is_ordinal is True

    def test_score_targets_declare_no_class_order(self) -> None:
        for name in (TargetName.ENGAGEMENT_SCORE, TargetName.COGNITIVE_LOAD_SCORE):
            assert TARGET_SPECS[name].class_order_is_ordinal is False

    def test_ordinal_states_are_ordered_low_medium_high(self) -> None:
        assert list(OrdinalState) == [
            OrdinalState.LOW,
            OrdinalState.MEDIUM,
            OrdinalState.HIGH,
        ]

    def test_direction_reversal_is_symmetric(self) -> None:
        assert (
            opposite_direction(AdaptationDirection.INCREASE)
            is AdaptationDirection.DECREASE
        )
        assert (
            opposite_direction(AdaptationDirection.DECREASE)
            is AdaptationDirection.INCREASE
        )
        assert opposite_direction(AdaptationDirection.HOLD) is AdaptationDirection.HOLD


class TestConfiguration:
    def test_inverted_bounds_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="must not exceed"):
            DifficultyBounds(minimum=5, maximum=1, step=1)

    def test_zero_step_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            DifficultyBounds(minimum=1, maximum=5, step=0)

    def test_step_larger_than_the_range_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="exceeds the whole configured range"):
            DifficultyBounds(minimum=1, maximum=3, step=5)

    def test_persistence_below_one_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            make_configuration(minimum_persistence_windows=0)

    def test_negative_cooldown_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            make_configuration(cooldown_windows=-1)

    def test_negative_budget_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            make_configuration(max_adaptations_per_session=-1)

    def test_null_budget_is_permitted(self) -> None:
        assert (
            make_configuration(
                max_adaptations_per_session=None
            ).max_adaptations_per_session
            is None
        )

    def test_regression_mapping_without_bands_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="no band is configured"):
            make_configuration(regression_mapping_enabled=True)

    def test_regression_band_needs_a_neutral_region(self) -> None:
        with pytest.raises(ValidationError, match="strictly below"):
            RegressionBand(
                target_name=TargetName.ENGAGEMENT_SCORE,
                low_below=0.7,
                high_above=0.3,
                unit="dimensionless",
            )

    def test_regression_band_must_lie_inside_the_target_range(self) -> None:
        with pytest.raises(ValidationError, match="outside the declared range"):
            RegressionBand(
                target_name=TargetName.ENGAGEMENT_SCORE,
                low_below=0.3,
                high_above=1.9,
                unit="dimensionless",
            )

    def test_a_classification_target_cannot_carry_a_band(self) -> None:
        with pytest.raises(ValidationError, match="not a regression target"):
            RegressionBand(
                target_name=TargetName.ENGAGEMENT_CLASS,
                low_below=0.3,
                high_above=0.7,
                unit="class",
            )

    def test_non_finite_band_boundaries_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            RegressionBand(
                target_name=TargetName.ENGAGEMENT_SCORE,
                low_below=float("nan"),
                high_above=0.7,
                unit="dimensionless",
            )

    def test_fingerprint_is_stable_and_discriminating(self) -> None:
        left = make_configuration()
        assert left.fingerprint() == make_configuration().fingerprint()
        assert (
            left.fingerprint() != make_configuration(cooldown_windows=1).fingerprint()
        )

    def test_fingerprint_ignores_the_prose_notes(self) -> None:
        left = make_configuration()
        right = left.model_copy(update={"note": "a different note"})
        assert left.fingerprint() == right.fingerprint()

    def test_only_one_policy_mode_is_implemented(self) -> None:
        assert list(AdaptationPolicyMode) == [
            AdaptationPolicyMode.CONSERVATIVE_RULE_BASED
        ]

    def test_there_is_no_prefer_increase_resolution(self) -> None:
        assert set(ConflictResolution) == {
            ConflictResolution.HOLD,
            ConflictResolution.PREFER_DECREASE,
        }


class TestPolicyState:
    def test_cold_start_is_explicit(self) -> None:
        state = AdaptationPolicyState.for_session("s1")
        assert state.cooldown_remaining == 0
        assert state.adaptation_count == 0
        assert state.pending_direction is None
        assert state.last_applied_direction is None
        assert state.current_difficulty is None

    def test_state_is_frozen(self) -> None:
        state = AdaptationPolicyState.for_session("s1")
        with pytest.raises(ValidationError):
            state.persistence_count = 3  # type: ignore[misc]

    def test_hold_cannot_be_a_pending_direction(self) -> None:
        with pytest.raises(ValidationError, match="not something to persist toward"):
            AdaptationPolicyState(
                session_id="s1",
                pending_direction=AdaptationDirection.HOLD,
                persistence_count=1,
            )

    def test_a_count_without_a_direction_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="nothing to count toward"):
            AdaptationPolicyState(session_id="s1", persistence_count=2)

    def test_a_direction_without_a_count_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="zero count is not a state"):
            AdaptationPolicyState(
                session_id="s1", pending_direction=AdaptationDirection.INCREASE
            )

    def test_applied_direction_and_window_are_a_pair(self) -> None:
        with pytest.raises(ValidationError, match="recorded as a pair"):
            AdaptationPolicyState(
                session_id="s1",
                last_applied_direction=AdaptationDirection.INCREASE,
                adaptation_count=1,
            )

    def test_serialization_round_trips(self) -> None:
        state = AdaptationPolicyState(
            session_id="s1",
            last_window_id="w3",
            last_window_order=3,
            current_difficulty=2,
            pending_direction=AdaptationDirection.DECREASE,
            persistence_count=2,
            cooldown_remaining=4,
            last_applied_direction=AdaptationDirection.INCREASE,
            last_adaptation_window_order=1,
            adaptation_count=1,
            evaluated_window_count=4,
        )
        restored = AdaptationPolicyState.model_validate_json(state.model_dump_json())
        assert restored == state


class TestProposal:
    def test_a_proposal_records_current_and_desired_state(self) -> None:
        proposal = build_proposal()
        assert proposal.current_difficulty == 3
        assert proposal.proposed_difficulty == 2
        assert proposal.requested_difficulty == 2
        assert proposal.clamping_applied is False

    def test_a_blocked_gate_cannot_reach_a_proposal(self) -> None:
        blocked = make_evidence(
            TargetName.COGNITIVE_LOAD_CLASS, "high", eligible=False
        ).gate
        with pytest.raises(ValidationError, match="blocked gate cannot produce"):
            build_proposal(cognitive_load_gate=blocked)

    def test_a_proposal_clamped_back_to_where_it_started_is_refused(self) -> None:
        # A decrease from the minimum would clamp to the minimum, which is a
        # command that changes nothing. That is a hold, not a proposal.
        with pytest.raises(ValidationError, match="does not change difficulty"):
            build_proposal(
                current_difficulty=1,
                requested_difficulty=0,
                proposed_difficulty=1,
                clamping_applied=True,
            )

    def test_the_proposed_level_must_be_in_bounds(self) -> None:
        with pytest.raises(ValidationError, match="outside the configured bounds"):
            build_proposal(proposed_difficulty=9, requested_difficulty=9)

    def test_arithmetic_must_match_the_configured_step(self) -> None:
        with pytest.raises(ValidationError, match="fixed by its configured step"):
            build_proposal(requested_difficulty=1, proposed_difficulty=1)

    def test_direction_must_match_the_movement(self) -> None:
        with pytest.raises(ValidationError, match="opposite direction"):
            build_proposal(
                direction=AdaptationDirection.INCREASE,
                requested_difficulty=4,
                proposed_difficulty=2,
                clamping_applied=True,
            )

    def test_unmet_persistence_cannot_be_recorded_on_a_proposal(self) -> None:
        with pytest.raises(ValidationError, match="below the required"):
            build_proposal(persistence_count=1, required_persistence_windows=3)

    def test_a_synthetic_proposal_is_never_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically eligible"):
            build_proposal(scientific_evaluation_eligible=True)

    def test_a_proposal_carries_no_confidence_or_quality_field(self) -> None:
        from engagevr.schemas.adaptation_policy import AdaptationProposal

        assert not [
            name
            for name in AdaptationProposal.model_fields
            if "confidence" in name or "quality" in name
        ]


class TestPolicyDecision:
    def test_a_hold_carries_no_proposal(self) -> None:
        with pytest.raises(ValidationError, match="hold carries no proposal"):
            build_decision(
                kind=AdaptationDecisionKind.HOLD,
                reasons=(AdaptationPolicyReason.COOLDOWN_ACTIVE,),
                proposal=build_proposal(),
            )

    def test_a_proposing_decision_must_carry_its_proposal(self) -> None:
        with pytest.raises(ValidationError, match="must carry its proposal"):
            build_decision(
                kind=AdaptationDecisionKind.PROPOSE_ADAPTATION,
                reasons=(AdaptationPolicyReason.PROPOSAL_ELIGIBLE,),
                proposal=None,
            )

    def test_a_hold_cannot_claim_eligibility(self) -> None:
        with pytest.raises(ValidationError, match="cannot record 'proposal_eligible'"):
            build_decision(
                kind=AdaptationDecisionKind.HOLD,
                reasons=(AdaptationPolicyReason.PROPOSAL_ELIGIBLE,),
            )

    def test_a_proposal_states_exactly_one_reason(self) -> None:
        with pytest.raises(ValidationError, match="states exactly"):
            build_decision(
                kind=AdaptationDecisionKind.PROPOSE_ADAPTATION,
                reasons=(
                    AdaptationPolicyReason.COOLDOWN_ACTIVE,
                    AdaptationPolicyReason.PROPOSAL_ELIGIBLE,
                ),
                proposal=build_proposal(),
                resolved_direction=AdaptationDirection.DECREASE,
            )

    def test_reasons_must_be_in_canonical_order(self) -> None:
        with pytest.raises(ValidationError, match="canonical order"):
            build_decision(
                kind=AdaptationDecisionKind.HOLD,
                reasons=(
                    AdaptationPolicyReason.COOLDOWN_ACTIVE,
                    AdaptationPolicyReason.GATE_BLOCKED,
                ),
            )

    def test_a_decision_must_state_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="must state a reason"):
            build_decision(kind=AdaptationDecisionKind.HOLD, reasons=())

    def test_a_conflict_must_record_its_resolution(self) -> None:
        with pytest.raises(ValidationError, match="how it was resolved"):
            build_decision(
                kind=AdaptationDecisionKind.HOLD,
                reasons=(AdaptationPolicyReason.DIRECTION_CONFLICT,),
                conflict=True,
                conflict_resolution=None,
            )

    def test_a_proposal_cannot_be_made_during_cooldown(self) -> None:
        with pytest.raises(ValidationError, match="cooldown window"):
            build_decision(
                kind=AdaptationDecisionKind.PROPOSE_ADAPTATION,
                reasons=(AdaptationPolicyReason.PROPOSAL_ELIGIBLE,),
                proposal=build_proposal(),
                resolved_direction=AdaptationDirection.DECREASE,
                cooldown_remaining_before=2,
            )

    def test_states_must_belong_to_the_decision_session(self) -> None:
        with pytest.raises(ValidationError, match="belongs to session"):
            build_decision(
                kind=AdaptationDecisionKind.HOLD,
                reasons=(AdaptationPolicyReason.COOLDOWN_ACTIVE,),
                state_before=AdaptationPolicyState.for_session("other"),
            )

    def test_canonical_order_covers_every_reason(self) -> None:
        assert set(POLICY_REASON_ORDER) == set(AdaptationPolicyReason)
        assert len(POLICY_REASON_ORDER) == len(AdaptationPolicyReason)


class TestLifecycleEntry:
    def test_a_proposed_entry_has_no_command_id(self) -> None:
        with pytest.raises(ValidationError, match="has no command id"):
            AdaptationHistoryEntry(
                proposal_id="p1",
                session_id="s1",
                window_id="w1",
                window_order=0,
                direction=AdaptationDirection.INCREASE,
                expected_command="set_difficulty",
                expected_value=2,
                status=AdaptationLifecycleStatus.PROPOSED,
                command_id="c1",
                is_synthetic=True,
            )

    def test_applied_requires_an_acknowledgement(self) -> None:
        with pytest.raises(ValidationError, match="may only be recorded from a real"):
            AdaptationHistoryEntry(
                proposal_id="p1",
                session_id="s1",
                window_id="w1",
                window_order=0,
                direction=AdaptationDirection.INCREASE,
                expected_command="set_difficulty",
                expected_value=2,
                status=AdaptationLifecycleStatus.APPLIED,
                command_id="c1",
                is_synthetic=True,
            )

    def test_rejected_requires_a_stated_reason(self) -> None:
        with pytest.raises(ValidationError, match="requires a rejected"):
            AdaptationHistoryEntry(
                proposal_id="p1",
                session_id="s1",
                window_id="w1",
                window_order=0,
                direction=AdaptationDirection.INCREASE,
                expected_command="set_difficulty",
                expected_value=2,
                status=AdaptationLifecycleStatus.REJECTED,
                command_id="c1",
                acknowledged=False,
                is_synthetic=True,
            )

    def test_the_five_lifecycle_states_are_distinct(self) -> None:
        assert len({s.value for s in AdaptationLifecycleStatus}) == 6


class TestControllerMetrics:
    def _metrics(self, **overrides: object) -> AdaptationControllerMetrics:
        base: dict[str, object] = {
            "evaluated_windows": 10,
            "gate_eligible_windows": 8,
            "gate_blocked_windows": 2,
            "hold_decisions": 9,
            "adaptation_proposals": 1,
            "increases": 1,
            "decreases": 0,
            "direction_reversals": 0,
            "longest_same_direction_streak": 1,
            "blocked_oscillation_attempts": 0,
        }
        base.update(overrides)
        return AdaptationControllerMetrics(**base)  # type: ignore[arg-type]

    def test_counts_reconcile(self) -> None:
        assert self._metrics().evaluated_windows == 10

    def test_holds_and_proposals_must_account_for_every_window(self) -> None:
        with pytest.raises(ValidationError, match="account for every evaluated window"):
            self._metrics(hold_decisions=5)

    def test_every_proposal_is_an_increase_or_a_decrease(self) -> None:
        with pytest.raises(ValidationError, match="increase or a decrease"):
            self._metrics(increases=0)

    def test_a_blocked_window_cannot_have_proposed(self) -> None:
        with pytest.raises(ValidationError, match="cannot have produced a proposal"):
            self._metrics(
                gate_eligible_windows=0,
                gate_blocked_windows=10,
                hold_decisions=9,
                adaptation_proposals=1,
            )

    def test_metrics_carry_the_controller_note(self) -> None:
        note = self._metrics().note
        assert CONTROLLER_METRIC_NOTE in note
        # The note exists to deny the benefit reading, so it must name it.
        assert "NOT engagement improvement" in note
        assert "NOT evidence about any" in note


class TestRunSummary:
    def _summary(self, **overrides: object) -> AdaptationRunSummary:
        configuration = make_configuration()
        base: dict[str, object] = {
            "run_id": "r1",
            "engagevr_version": "0.0.0",
            "python_version": "3.12.0",
            "evaluation_mode": EvaluationMode.SOFTWARE_SELF_CHECK,
            "scientific_evaluation_eligible": False,
            "is_synthetic": True,
            "data_source": "synthetic",
            "configuration": configuration,
            "configuration_fingerprint": configuration.fingerprint(),
            "metrics": AdaptationControllerMetrics(
                evaluated_windows=0,
                gate_eligible_windows=0,
                gate_blocked_windows=0,
                hold_decisions=0,
                adaptation_proposals=0,
                increases=0,
                decreases=0,
                direction_reversals=0,
                longest_same_direction_streak=0,
                blocked_oscillation_attempts=0,
            ),
            "started_at_utc": utc_now(),
            "finished_at_utc": utc_now(),
            "disclaimers": (
                SOFTWARE_SELF_CHECK_BANNER,
                ADAPTATION_POLICY_NOTE,
            ),
        }
        base.update(overrides)
        return AdaptationRunSummary(**base)  # type: ignore[arg-type]

    def test_a_synthetic_summary_is_never_scientifically_eligible(self) -> None:
        with pytest.raises(ValidationError, match="never be scientifically eligible"):
            self._summary(scientific_evaluation_eligible=True)

    def test_a_self_check_must_carry_the_banner(self) -> None:
        with pytest.raises(ValidationError, match="must carry the banner"):
            self._summary(disclaimers=(ADAPTATION_POLICY_NOTE,))

    def test_a_summary_must_carry_the_demonstration_rule_note(self) -> None:
        with pytest.raises(ValidationError, match="demonstration-rule note"):
            self._summary(disclaimers=(SOFTWARE_SELF_CHECK_BANNER,))


class TestExperimentModeSeparation:
    def test_static_and_adaptive_are_distinct_modes(self) -> None:
        assert {m.value for m in ExperimentMode} == {"static", "adaptive"}

    def test_configuration_records_which_condition_applied(self) -> None:
        static = make_configuration(experiment_mode=ExperimentMode.STATIC)
        adaptive = make_configuration(experiment_mode=ExperimentMode.ADAPTIVE)
        assert static.fingerprint() != adaptive.fingerprint()


class TestConfigurationSurface:
    def test_defaults_yaml_resolves(self) -> None:
        from engagevr.config import load_config

        resolved = load_config().adaptation.resolve()
        assert isinstance(resolved, AdaptationPolicyConfiguration)
        assert resolved.mode is AdaptationPolicyMode.CONSERVATIVE_RULE_BASED
        assert resolved.minimum_persistence_windows >= 1
        assert resolved.cooldown_windows >= 0
        assert resolved.regression_mapping_enabled is False

    def test_an_unknown_policy_mode_is_refused(self) -> None:
        from engagevr.config import AdaptationPolicySettings

        with pytest.raises(ValidationError, match="not an implemented policy"):
            AdaptationPolicySettings(mode="deep_q_learning")

    def test_an_unknown_conflict_resolution_is_refused(self) -> None:
        from engagevr.config import AdaptationPolicySettings

        with pytest.raises(ValidationError, match="prefer_increase"):
            AdaptationPolicySettings(conflict_resolution="prefer_increase")

    def test_an_unknown_experiment_mode_is_refused(self) -> None:
        from engagevr.config import AdaptationConfig

        with pytest.raises(ValidationError, match="experiment_mode"):
            AdaptationConfig(experiment_mode="semi_adaptive")

    def test_bounds_must_contain_the_tasks_starting_difficulty(self) -> None:
        from engagevr.config import EngageVRConfig

        with pytest.raises(ValidationError, match="lies outside"):
            EngageVRConfig.model_validate(
                {
                    "task": {"default_difficulty": 9},
                    "adaptation": {
                        "policy": {"difficulty": {"minimum": 1, "maximum": 5}}
                    },
                }
            )

    def test_regression_mapping_without_boundaries_is_refused(self) -> None:
        from engagevr.config import AdaptationConfig

        settings = AdaptationConfig.model_validate(
            {"policy": {"regression_mapping": {"enabled": True}}}
        )
        with pytest.raises(ValueError, match="invented threshold"):
            settings.resolve()

    def test_regression_mapping_with_boundaries_resolves(self) -> None:
        from engagevr.config import AdaptationConfig

        settings = AdaptationConfig.model_validate(
            {
                "policy": {
                    "regression_mapping": {
                        "enabled": True,
                        "engagement_score": {"low_below": 0.3, "high_above": 0.7},
                        "cognitive_load_score": {"low_below": 0.3, "high_above": 0.7},
                    }
                }
            }
        )
        resolved = settings.resolve()
        assert resolved.regression_mapping_enabled is True
        assert len(resolved.regression_bands) == 2

    def test_no_confidence_or_quality_threshold_lives_in_this_section(self) -> None:
        from engagevr.config import AdaptationConfig, AdaptationPolicySettings

        names = set(AdaptationConfig.model_fields) | set(
            AdaptationPolicySettings.model_fields
        )
        assert not [n for n in names if "confidence" in n or "quality" in n]
