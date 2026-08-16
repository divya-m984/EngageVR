"""Missing-modality scenario and synthetic-dropout tests.

A scenario removes availability and nothing else.  Synthetic dropout is a
deterministic software check that fabricates an availability pattern no
measurement produced, so it is refused in scientific mode.

No test here needs a webcam, a model asset, a display server, a network,
Unity, a public dataset, or participant data.
"""

from __future__ import annotations

import numpy as np
import pytest

from engagevr.schemas.fusion import FusionModality
from engagevr.training.robustness import (
    REFERENCE_SCENARIO,
    SCENARIOS,
    SCIENTIFIC_DROPOUT_REFUSAL,
    RobustnessError,
    apply_scenario,
    coverage,
    resolve_scenarios,
    scenario_by_name,
    synthetic_modality_dropout,
)

ALL = tuple(FusionModality)


def _availability(count: int = 8) -> dict[FusionModality, np.ndarray]:
    return {modality: np.ones(count, dtype=bool) for modality in ALL}


def _windows(count: int = 8) -> list[str]:
    return [f"synthetic-session-{index:04d}-w{index:03d}" for index in range(count)]


class TestScenarioDefinitions:
    def test_every_required_scenario_is_declared(self) -> None:
        assert {scenario.name for scenario in SCENARIOS} == {
            "all_modalities",
            "missing_behavioural",
            "missing_head_pose",
            "missing_rppg",
            "missing_task",
            "missing_behavioural_and_rppg",
            "only_task",
            "only_behavioural",
            "only_rppg",
            "only_head_pose",
        }

    def test_the_reference_scenario_removes_nothing(self) -> None:
        assert REFERENCE_SCENARIO.name == "all_modalities"
        assert REFERENCE_SCENARIO.absent_modalities == ()
        assert REFERENCE_SCENARIO.present(ALL) == ALL

    def test_single_missing_scenarios_remove_exactly_one_group(self) -> None:
        for scenario in SCENARIOS:
            if scenario.name.startswith("missing_") and "_and_" not in scenario.name:
                assert len(scenario.absent_modalities) == 1

    def test_only_scenarios_leave_exactly_one_group(self) -> None:
        for scenario in SCENARIOS:
            if scenario.name.startswith("only_"):
                assert len(scenario.present(ALL)) == 1

    def test_the_multi_missing_scenario_removes_both_camera_modalities(self) -> None:
        scenario = scenario_by_name("missing_behavioural_and_rppg")
        assert set(scenario.absent_modalities) == {
            FusionModality.BEHAVIOURAL,
            FusionModality.RPPG,
        }

    def test_an_unknown_scenario_is_refused(self) -> None:
        with pytest.raises(RobustnessError, match="unknown missing-modality scenario"):
            scenario_by_name("missing_eeg")

    def test_resolving_defaults_to_every_scenario(self) -> None:
        assert resolve_scenarios(None) == SCENARIOS
        assert resolve_scenarios([]) == SCENARIOS

    def test_a_duplicate_request_is_refused(self) -> None:
        with pytest.raises(RobustnessError, match="more than once"):
            resolve_scenarios(["only_task", "only_task"])


class TestScenarioApplication:
    def test_a_scenario_clears_the_named_modalities(self) -> None:
        masked = apply_scenario(_availability(), scenario_by_name("missing_rppg"))
        assert not masked[FusionModality.RPPG].any()
        assert masked[FusionModality.TASK].all()

    def test_only_task_clears_everything_else(self) -> None:
        masked = apply_scenario(_availability(), scenario_by_name("only_task"))
        assert masked[FusionModality.TASK].all()
        for modality in ALL:
            if modality is not FusionModality.TASK:
                assert not masked[modality].any()

    def test_a_scenario_never_restores_missing_evidence(self) -> None:
        availability = _availability()
        availability[FusionModality.TASK][0] = False
        masked = apply_scenario(availability, scenario_by_name("missing_rppg"))
        assert masked[FusionModality.TASK][0] == np.False_

    def test_the_source_arrays_are_not_mutated(self) -> None:
        availability = _availability()
        apply_scenario(availability, scenario_by_name("only_rppg"))
        assert availability[FusionModality.TASK].all()


class TestSyntheticDropout:
    def test_it_is_deterministic_for_one_seed(self) -> None:
        windows = _windows(64)
        first = synthetic_modality_dropout(
            _availability(64), window_ids=windows, seed=42, probability=0.3
        )
        second = synthetic_modality_dropout(
            _availability(64), window_ids=windows, seed=42, probability=0.3
        )
        for modality in ALL:
            assert np.array_equal(first[modality], second[modality])

    def test_a_different_seed_produces_a_different_pattern(self) -> None:
        windows = _windows(64)
        first = synthetic_modality_dropout(
            _availability(64), window_ids=windows, seed=42, probability=0.3
        )
        second = synthetic_modality_dropout(
            _availability(64), window_ids=windows, seed=7, probability=0.3
        )
        assert any(
            not np.array_equal(first[modality], second[modality]) for modality in ALL
        )

    def test_the_decision_does_not_depend_on_row_order(self) -> None:
        windows = _windows(32)
        forward = synthetic_modality_dropout(
            _availability(32), window_ids=windows, seed=42, probability=0.4
        )
        reversed_windows = list(reversed(windows))
        backward = synthetic_modality_dropout(
            _availability(32), window_ids=reversed_windows, seed=42, probability=0.4
        )
        for modality in ALL:
            assert np.array_equal(forward[modality], backward[modality][::-1])

    def test_it_drops_whole_modalities_coherently(self) -> None:
        windows = _windows(32)
        dropped = synthetic_modality_dropout(
            _availability(32), window_ids=windows, seed=42, probability=0.5
        )
        for modality in ALL:
            # A dropped window loses the entire modality, never part of it.
            assert dropped[modality].dtype == bool
        assert any(not dropped[modality].all() for modality in ALL)

    def test_zero_probability_drops_nothing(self) -> None:
        dropped = synthetic_modality_dropout(
            _availability(16), window_ids=_windows(16), seed=42, probability=0.0
        )
        assert all(dropped[modality].all() for modality in ALL)

    def test_it_never_restores_availability(self) -> None:
        availability = _availability(16)
        availability[FusionModality.RPPG][:] = False
        dropped = synthetic_modality_dropout(
            availability, window_ids=_windows(16), seed=42, probability=0.9
        )
        assert not dropped[FusionModality.RPPG].any()

    def test_an_invalid_probability_is_refused(self) -> None:
        with pytest.raises(RobustnessError, match=r"must be in \[0, 1\)"):
            synthetic_modality_dropout(
                _availability(4), window_ids=_windows(4), seed=42, probability=1.0
            )

    def test_mismatched_row_counts_are_refused(self) -> None:
        with pytest.raises(RobustnessError, match="window ids were supplied"):
            synthetic_modality_dropout(
                _availability(8), window_ids=_windows(4), seed=42, probability=0.1
            )

    def test_the_scientific_refusal_message_explains_itself(self) -> None:
        assert "fabricates an availability pattern" in SCIENTIFIC_DROPOUT_REFUSAL


class TestCoverage:
    def test_coverage_is_the_fused_fraction(self) -> None:
        assert coverage(3, 4) == pytest.approx(0.75)

    def test_coverage_is_unavailable_with_no_windows(self) -> None:
        assert coverage(0, 0) is None
