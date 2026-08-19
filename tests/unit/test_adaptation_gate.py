"""The Milestone 7 adaptation gate: what it decides, and what it cannot.

The gate answers only whether an ALREADY-CHOSEN action may be acted upon.
A large part of this module tests the *absence* of behaviour: that the gate
cannot name an action, cannot rank actions, cannot send anything anywhere,
and cannot turn an abstention into permission.

No test needs a webcam, a model asset, a display server, a network, Unity,
a public dataset, or participant data.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from engagevr.schemas.targets import TaskType
from engagevr.schemas.uncertainty import (
    AbstentionDecision,
    AbstentionReason,
    AdaptationGateDecision,
    ProbabilityCalibrationStatus,
    SignalQualitySummary,
    ThresholdSource,
)
from engagevr.training import adaptation_gate as gate_module
from engagevr.training.adaptation_gate import (
    ADAPTATION_GATE_SCOPE,
    evaluate_adaptation_gate,
)

VOCABULARY = ("low", "medium", "high")


def _decision(**overrides: object) -> AbstentionDecision:
    payload: dict[str, object] = {
        "window_id": "w01",
        "subject_id": "synthetic-subject-01",
        "session_id": "sess-a",
        "target_name": "engagement_class",
        "task_type": TaskType.CLASSIFICATION,
        "fold_index": 0,
        "source_prediction_id": "run|0|w01|baseline_model",
        "prediction_available": True,
        "accepted": True,
        "abstained": False,
        "predicted_class": "low",
        "class_vocabulary": VOCABULARY,
        "probabilities": (0.9, 0.05, 0.05),
        "probability_calibration_status": ProbabilityCalibrationStatus.CALIBRATED,
        "confidence_score": 0.9,
        "applied_threshold": 0.7,
        "threshold_source": ThresholdSource.CONFIGURED_POPULATION,
        "evidence_gate_passed": True,
        "data_source": "synthetic",
        "is_synthetic": True,
        "scientific_evaluation_eligible": False,
        "acceptance_rule": "accept if score >= tau",
    }
    payload.update(overrides)
    return AbstentionDecision(**payload)  # type: ignore[arg-type]


def _abstained(*reasons: AbstentionReason, **overrides: object) -> AbstentionDecision:
    ordered = tuple(r for r in AbstentionReason if r in set(reasons))
    payload: dict[str, object] = {
        "accepted": False,
        "abstained": True,
        "reasons": ordered,
    }
    payload.update(overrides)
    return _decision(**payload)


def _regression(**overrides: object) -> AbstentionDecision:
    payload: dict[str, object] = {
        "task_type": TaskType.REGRESSION,
        "target_name": "engagement_score",
        "predicted_class": None,
        "class_vocabulary": (),
        "probabilities": (),
        "probability_calibration_status": None,
        "confidence_score": None,
        "applied_threshold": None,
        "threshold_source": None,
        "predicted_value": 0.5,
        "interval_lower_bound": 0.4,
        "interval_upper_bound": 0.6,
        "interval_width": 0.2,
        "maximum_interval_width": 0.5,
        "acceptance_rule": "accept if interval_width <= maximum",
    }
    payload.update(overrides)
    return _decision(**payload)


class TestScope:
    """The gate must be unable to choose an adaptation."""

    def test_the_module_imports_nothing_that_could_send_a_message(self) -> None:
        # Parsed rather than grepped: the module docstring NAMES these
        # modules in the sentence that denies importing them, so a textual
        # search would fail on the very prose that documents the rule.
        tree = ast.parse(inspect.getsource(gate_module))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for forbidden in (
            "engagevr.transport",
            "engagevr.api",
            "engagevr.task",
            "websockets",
            "requests",
            "httpx",
            "socket",
        ):
            assert not any(name.startswith(forbidden) for name in imported), (
                f"the gate imports {forbidden}"
            )

    def test_the_module_imports_only_schemas(self) -> None:
        tree = ast.parse(inspect.getsource(gate_module))
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert modules == {
            "__future__",
            "engagevr.schemas.targets",
            "engagevr.schemas.uncertainty",
        }

    def test_the_module_defines_no_action_vocabulary(self) -> None:
        source = inspect.getsource(gate_module).lower()
        # The words appear only inside the prose that DENIES the behaviour,
        # never as an identifier the gate could act on.
        for name in dir(gate_module):
            if name.startswith("_"):
                continue
            for token in ("difficulty", "scene", "reward", "policy", "cooldown"):
                assert token not in name.lower()
        assert "does not choose an adaptation" in source

    def test_the_scope_note_is_persisted_on_every_record(self) -> None:
        record = evaluate_adaptation_gate(_decision())
        assert "Milestone 8" in record.scope_note
        assert "does not choose an adaptation" in ADAPTATION_GATE_SCOPE

    def test_the_gate_returns_only_eligible_or_blocked(self) -> None:
        record = evaluate_adaptation_gate(_decision())
        assert record.decision in set(AdaptationGateDecision)

    def test_the_gate_never_recomputes_the_decision_it_gates(self) -> None:
        # A decision that abstained for a reason the gate does not itself
        # evaluate is still blocked, and blocked for that same reason.
        decision = _abstained(AbstentionReason.SIGNAL_QUALITY_BELOW_GATE)
        record = evaluate_adaptation_gate(decision)
        assert record.decision is AdaptationGateDecision.BLOCKED
        assert AbstentionReason.SIGNAL_QUALITY_BELOW_GATE in record.reasons


class TestClassificationGating:
    def test_an_accepted_high_confidence_prediction_is_eligible(self) -> None:
        record = evaluate_adaptation_gate(_decision(), applied_confidence_threshold=0.7)
        assert record.decision is AdaptationGateDecision.ELIGIBLE
        assert record.reasons == ()
        assert record.confidence_requirement_satisfied is True

    def test_an_abstained_prediction_blocks(self) -> None:
        record = evaluate_adaptation_gate(
            _abstained(AbstentionReason.BELOW_CONFIDENCE_THRESHOLD)
        )
        assert record.decision is AdaptationGateDecision.BLOCKED
        assert AbstentionReason.BELOW_CONFIDENCE_THRESHOLD in record.reasons
        assert record.confidence_requirement_satisfied is False

    def test_an_unavailable_prediction_blocks(self) -> None:
        record = evaluate_adaptation_gate(
            _abstained(
                AbstentionReason.MODEL_PREDICTION_UNAVAILABLE,
                prediction_available=False,
                predicted_class=None,
                class_vocabulary=(),
                probabilities=(),
                confidence_score=None,
                applied_threshold=None,
                threshold_source=None,
                evidence_gate_passed=False,
            )
        )
        assert record.decision is AdaptationGateDecision.BLOCKED
        assert AbstentionReason.MODEL_PREDICTION_UNAVAILABLE in record.reasons

    def test_inadequate_quality_blocks_independently_of_confidence(self) -> None:
        record = evaluate_adaptation_gate(
            _abstained(
                AbstentionReason.SIGNAL_QUALITY_BELOW_GATE,
                evidence_gate_passed=False,
                signal_quality=SignalQualitySummary(
                    modality_quality={"rppg": 0.05}, minimum_recorded_quality=0.05
                ),
            )
        )
        assert record.decision is AdaptationGateDecision.BLOCKED
        assert AbstentionReason.SIGNAL_QUALITY_BELOW_GATE in record.reasons
        # The confidence requirement was satisfied; quality alone blocked it.
        assert record.confidence_requirement_satisfied is True
        assert record.evidence_gate_passed is False

    def test_insufficient_modality_evidence_blocks_independently(self) -> None:
        record = evaluate_adaptation_gate(
            _abstained(
                AbstentionReason.INSUFFICIENT_MEASUREMENT_EVIDENCE,
                evidence_gate_passed=False,
            )
        )
        assert record.decision is AdaptationGateDecision.BLOCKED
        assert AbstentionReason.INSUFFICIENT_MEASUREMENT_EVIDENCE in record.reasons
        assert record.confidence_requirement_satisfied is True

    def test_uncalibrated_confidence_blocks_when_it_was_required(self) -> None:
        record = evaluate_adaptation_gate(
            _abstained(
                AbstentionReason.PROBABILITY_CALIBRATION_UNAVAILABLE,
                probability_calibration_status=(
                    ProbabilityCalibrationStatus.UNCALIBRATED
                ),
                confidence_score=None,
                selection_score=0.9,
                evidence_gate_passed=False,
            )
        )
        assert record.decision is AdaptationGateDecision.BLOCKED
        assert AbstentionReason.PROBABILITY_CALIBRATION_UNAVAILABLE in record.reasons
        assert record.confidence_requirement_satisfied is False


class TestRegressionGating:
    def test_a_narrow_interval_is_eligible(self) -> None:
        record = evaluate_adaptation_gate(_regression(), maximum_interval_width=0.5)
        assert record.decision is AdaptationGateDecision.ELIGIBLE
        assert record.interval_requirement_satisfied is True
        assert record.confidence_requirement_satisfied is None

    def test_a_too_wide_interval_blocks(self) -> None:
        record = evaluate_adaptation_gate(
            _regression(
                accepted=False,
                abstained=True,
                reasons=(AbstentionReason.INTERVAL_TOO_WIDE,),
                interval_width=0.9,
                interval_lower_bound=0.05,
                interval_upper_bound=0.95,
            )
        )
        assert record.decision is AdaptationGateDecision.BLOCKED
        assert AbstentionReason.INTERVAL_TOO_WIDE in record.reasons
        assert record.interval_requirement_satisfied is False

    def test_a_missing_interval_blocks(self) -> None:
        record = evaluate_adaptation_gate(
            _regression(
                accepted=False,
                abstained=True,
                reasons=(AbstentionReason.PREDICTION_INTERVAL_UNAVAILABLE,),
                interval_lower_bound=None,
                interval_upper_bound=None,
                interval_width=None,
            )
        )
        assert record.decision is AdaptationGateDecision.BLOCKED
        assert record.interval_requirement_satisfied is False

    def test_a_regression_record_reports_no_confidence_requirement(self) -> None:
        record = evaluate_adaptation_gate(_regression())
        assert record.confidence_requirement_satisfied is None


class TestDeterminism:
    def test_the_reason_list_is_deterministic_and_canonically_ordered(self) -> None:
        decision = _abstained(
            AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,
            AbstentionReason.SIGNAL_QUALITY_BELOW_GATE,
            AbstentionReason.INSUFFICIENT_MEASUREMENT_EVIDENCE,
            evidence_gate_passed=False,
        )
        first = evaluate_adaptation_gate(decision)
        second = evaluate_adaptation_gate(decision)
        assert first.reasons == second.reasons
        order = list(AbstentionReason)
        assert list(first.reasons) == sorted(first.reasons, key=order.index)

    def test_evidence_reasons_precede_model_reasons(self) -> None:
        record = evaluate_adaptation_gate(
            _abstained(
                AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,
                AbstentionReason.SIGNAL_QUALITY_BELOW_GATE,
                evidence_gate_passed=False,
            )
        )
        reasons = list(record.reasons)
        assert reasons.index(
            AbstentionReason.SIGNAL_QUALITY_BELOW_GATE
        ) < reasons.index(AbstentionReason.BELOW_CONFIDENCE_THRESHOLD)

    def test_the_same_decision_always_produces_the_same_record(self) -> None:
        decision = _decision()
        first = evaluate_adaptation_gate(decision, applied_confidence_threshold=0.7)
        second = evaluate_adaptation_gate(decision, applied_confidence_threshold=0.7)
        assert first == second


class TestDisabledGate:
    def test_disabling_the_gate_does_not_make_an_abstention_eligible(self) -> None:
        # A disabled gate stops applying its own additional requirement; it
        # does not license a false statement about a window that abstained.
        record = evaluate_adaptation_gate(
            _abstained(AbstentionReason.BELOW_CONFIDENCE_THRESHOLD), enabled=False
        )
        assert record.decision is AdaptationGateDecision.BLOCKED

    def test_a_disabled_gate_records_no_threshold_provenance(self) -> None:
        record = evaluate_adaptation_gate(
            _decision(), applied_confidence_threshold=0.7, enabled=False
        )
        assert record.applied_confidence_threshold is None

    def test_an_enabled_gate_records_the_threshold_it_was_given(self) -> None:
        record = evaluate_adaptation_gate(
            _decision(), applied_confidence_threshold=0.7, enabled=True
        )
        assert record.applied_confidence_threshold == pytest.approx(0.7)


class TestEvidenceInference:
    def test_an_absent_evidence_flag_is_inferred_from_the_reasons(self) -> None:
        record = evaluate_adaptation_gate(
            _abstained(
                AbstentionReason.SIGNAL_QUALITY_BELOW_GATE,
                evidence_gate_passed=None,
            )
        )
        assert record.evidence_gate_passed is False

    def test_a_clean_decision_with_no_flag_infers_a_passing_gate(self) -> None:
        record = evaluate_adaptation_gate(_decision(evidence_gate_passed=None))
        assert record.evidence_gate_passed is True
        assert record.decision is AdaptationGateDecision.ELIGIBLE

    def test_a_failed_gate_with_no_named_reason_gets_the_general_one(self) -> None:
        record = evaluate_adaptation_gate(
            _abstained(
                AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,
                evidence_gate_passed=False,
            )
        )
        assert AbstentionReason.INSUFFICIENT_MEASUREMENT_EVIDENCE in record.reasons
