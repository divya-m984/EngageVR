"""Deterministic controller scenarios for the adaptation policy.

These are **controller tests**, not participant simulations.  Each
scenario is a hand-written sequence of window states chosen to exercise
one guard, and nothing in this module models a person, a task, a
physiological process, or a response to an adaptation.  Calling any of
them a simulated participant would be false: the "engagement" and
"cognitive load" states here are inputs the author chose in order to make
a specific branch of the policy run.

Every evidence record is built by putting a real
:class:`~engagevr.schemas.uncertainty.AbstentionDecision` through the real
:func:`~engagevr.training.adaptation_gate.evaluate_adaptation_gate`, so
the Milestone 7 gate path is genuinely exercised rather than stubbed.  A
scenario cannot manufacture an eligible gate for an abstained window: the
gate refuses, exactly as it does in a real run.

Everything here is permanently marked synthetic and is never
scientifically eligible.
"""

from __future__ import annotations

from dataclasses import dataclass

from engagevr.schemas.adaptation_policy import (
    AdaptationInput,
    AdaptationTargetEvidence,
)
from engagevr.schemas.targets import TargetName, TaskType
from engagevr.schemas.uncertainty import (
    ABSTENTION_MEANING_NOTE,
    AbstentionDecision,
    AbstentionReason,
    ProbabilityCalibrationStatus,
    SignalQualitySummary,
    ThresholdSource,
)
from engagevr.training.adaptation_gate import evaluate_adaptation_gate

#: Printed and persisted wherever a scenario result appears.
SCENARIO_DISCLAIMER = (
    "These scenarios are DETERMINISTIC CONTROLLER TESTS. Each window state "
    "was chosen by the author to make one branch of the policy run. They do "
    "not simulate a participant, a task, a physiological process, or anyone's "
    "response to an adaptation, and no count derived from them describes a "
    "person."
)

#: The engineering default this policy applies when it accepts a window.
_ACCEPTANCE_RULE = (
    "accept if calibrated confidence >= applied threshold; this scenario "
    "fixes the confidence and the threshold directly"
)
_THRESHOLD = 0.6
_ACCEPTED_CONFIDENCE = 0.9
_REJECTED_CONFIDENCE = 0.4
_QUALITY = 0.8

#: How a scenario window declares what Milestone 7 concluded.
BLOCKED_BY_CONFIDENCE = "blocked_by_confidence"
BLOCKED_BY_QUALITY = "blocked_by_quality"
PREDICTION_MISSING = "prediction_missing"
EVIDENCE_MISSING = "evidence_missing"
ELIGIBLE = "eligible"


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """One window of a scenario, stated in the terms a reader checks.

    ``engagement`` and ``cognitive_load`` are class labels, or ``None`` to
    omit that target's evidence entirely.
    """

    engagement: str | None
    cognitive_load: str | None
    difficulty: int | None = 1
    engagement_status: str = ELIGIBLE
    cognitive_load_status: str = ELIGIBLE
    session_id: str | None = None
    window_id: str | None = None
    window_order: int | None = None


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named window sequence and what it is meant to exercise."""

    name: str
    description: str
    expectation: str
    windows: tuple[WindowSpec, ...]
    subject_id: str = "synthetic_subject"
    session_id: str = "synthetic_session"


def _probabilities(
    vocabulary: tuple[str, ...], predicted: str, top: float
) -> tuple[float, ...]:
    """A distribution whose maximum sits on ``predicted``.

    The remaining mass is split evenly, so the vector is a function of the
    label and the confidence alone and two runs produce identical numbers.
    """
    rest = (1.0 - top) / (len(vocabulary) - 1)
    values = [top if label == predicted else rest for label in vocabulary]
    # Absorb the float residue into the predicted entry so the vector sums
    # to exactly 1.0 rather than to 1.0 plus an epsilon.
    values[vocabulary.index(predicted)] = 1.0 - sum(
        v for i, v in enumerate(values) if vocabulary[i] != predicted
    )
    return tuple(values)


def build_evidence(
    *,
    target_name: TargetName,
    class_label: str | None,
    status: str,
    window_id: str,
    subject_id: str,
    session_id: str,
) -> AdaptationTargetEvidence:
    """Build one target's Milestone 7 records for a scenario window.

    The gate is computed by the real Milestone 7 function, so a window
    whose decision abstained cannot be handed to the policy as eligible.
    """
    spec_vocabulary = ("low", "medium", "high")
    prediction_id = f"scenario|{session_id}|{window_id}|{target_name.value}"

    if status == PREDICTION_MISSING:
        decision = AbstentionDecision(
            window_id=window_id,
            subject_id=subject_id,
            session_id=session_id,
            target_name=target_name.value,
            task_type=TaskType.CLASSIFICATION,
            fold_index=0,
            source_prediction_id=prediction_id,
            prediction_available=False,
            accepted=False,
            abstained=True,
            reasons=(AbstentionReason.MODEL_PREDICTION_UNAVAILABLE,),
            data_source="synthetic",
            is_synthetic=True,
            scientific_evaluation_eligible=False,
            acceptance_rule=_ACCEPTANCE_RULE,
            note=ABSTENTION_MEANING_NOTE,
        )
        return AdaptationTargetEvidence(
            target_name=target_name,
            decision=decision,
            gate=evaluate_adaptation_gate(
                decision, applied_confidence_threshold=_THRESHOLD
            ),
        )

    if class_label is None:  # pragma: no cover - guarded by the caller
        raise ValueError("a present prediction needs a class label")

    accepted = status == ELIGIBLE
    reasons: tuple[AbstentionReason, ...]
    if status == BLOCKED_BY_CONFIDENCE:
        reasons = (AbstentionReason.BELOW_CONFIDENCE_THRESHOLD,)
        confidence = _REJECTED_CONFIDENCE
        evidence_passed: bool | None = True
        quality = _QUALITY
    elif status == BLOCKED_BY_QUALITY:
        reasons = (AbstentionReason.SIGNAL_QUALITY_BELOW_GATE,)
        confidence = _ACCEPTED_CONFIDENCE
        evidence_passed = False
        quality = 0.1
    elif status == EVIDENCE_MISSING:
        reasons = (AbstentionReason.REQUIRED_MODALITY_UNAVAILABLE,)
        confidence = _ACCEPTED_CONFIDENCE
        evidence_passed = False
        quality = None
    else:
        reasons = ()
        confidence = _ACCEPTED_CONFIDENCE
        evidence_passed = True
        quality = _QUALITY

    decision = AbstentionDecision(
        window_id=window_id,
        subject_id=subject_id,
        session_id=session_id,
        target_name=target_name.value,
        task_type=TaskType.CLASSIFICATION,
        fold_index=0,
        source_prediction_id=prediction_id,
        prediction_available=True,
        accepted=accepted,
        abstained=not accepted,
        reasons=reasons,
        predicted_class=class_label,
        class_vocabulary=spec_vocabulary,
        probabilities=_probabilities(spec_vocabulary, class_label, confidence),
        probability_calibration_status=ProbabilityCalibrationStatus.CALIBRATED,
        confidence_score=confidence,
        applied_threshold=_THRESHOLD,
        threshold_source=ThresholdSource.CONFIGURED_POPULATION,
        signal_quality=SignalQualitySummary(minimum_recorded_quality=quality),
        evidence_gate_passed=evidence_passed,
        data_source="synthetic",
        is_synthetic=True,
        scientific_evaluation_eligible=False,
        acceptance_rule=_ACCEPTANCE_RULE,
        note=ABSTENTION_MEANING_NOTE,
    )
    return AdaptationTargetEvidence(
        target_name=target_name,
        decision=decision,
        gate=evaluate_adaptation_gate(
            decision, applied_confidence_threshold=_THRESHOLD
        ),
    )


def _evidence_or_none(
    *,
    target_name: TargetName,
    class_label: str | None,
    status: str,
    window_id: str,
    subject_id: str,
    session_id: str,
) -> AdaptationTargetEvidence | None:
    """Evidence for one target, or nothing when the scenario omits it.

    A label of ``None`` means the target contributed no record at all,
    which is different from a record whose prediction was unavailable:
    the first is a missing input, the second is Milestone 7 stating that
    it had no prediction.
    """
    if class_label is None and status != PREDICTION_MISSING:
        return None
    return build_evidence(
        target_name=target_name,
        class_label=class_label,
        status=status,
        window_id=window_id,
        subject_id=subject_id,
        session_id=session_id,
    )


def build_inputs(scenario: Scenario) -> tuple[AdaptationInput, ...]:
    """Materialise one scenario into the policy's input sequence."""
    inputs: list[AdaptationInput] = []
    for index, window in enumerate(scenario.windows):
        session_id = window.session_id or scenario.session_id
        window_id = window.window_id or f"{scenario.name}-w{index:03d}"
        order = window.window_order if window.window_order is not None else index
        engagement = _evidence_or_none(
            target_name=TargetName.ENGAGEMENT_CLASS,
            class_label=window.engagement,
            status=window.engagement_status,
            window_id=window_id,
            subject_id=scenario.subject_id,
            session_id=session_id,
        )
        cognitive_load = _evidence_or_none(
            target_name=TargetName.COGNITIVE_LOAD_CLASS,
            class_label=window.cognitive_load,
            status=window.cognitive_load_status,
            window_id=window_id,
            subject_id=scenario.subject_id,
            session_id=session_id,
        )
        inputs.append(
            AdaptationInput(
                session_id=session_id,
                subject_id=scenario.subject_id,
                window_id=window_id,
                window_order=order,
                current_difficulty=window.difficulty,
                engagement=engagement,
                cognitive_load=cognitive_load,
                scenario_id=scenario.name,
                is_synthetic=True,
                scientific_evaluation_eligible=False,
            )
        )
    return tuple(inputs)


def _steady(
    engagement: str, load: str, count: int, difficulty: int = 3
) -> tuple[WindowSpec, ...]:
    return tuple(
        WindowSpec(engagement=engagement, cognitive_load=load, difficulty=difficulty)
        for _ in range(count)
    )


#: The scenario suite.  Each entry names the guard it exercises.  The
#: sequences are sized against the default configuration: persistence 3,
#: cooldown 6, difficulty in [1, 5] step 1, budget 10.
SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="stable-neutral",
        description="Every window reads medium engagement and medium load.",
        expectation="Every window holds, in the deadband.",
        session_id="scn-stable-neutral",
        windows=_steady("medium", "medium", 8),
    ),
    Scenario(
        name="persistent-increase",
        description="High engagement with low load, sustained.",
        expectation=(
            "The first two windows hold for insufficient persistence; the "
            "third proposes one increase; the rest are held by cooldown."
        ),
        session_id="scn-persistent-increase",
        windows=_steady("high", "low", 8),
    ),
    Scenario(
        name="persistent-decrease",
        description="High cognitive load, sustained.",
        expectation=(
            "The first two windows hold; the third proposes one decrease; "
            "the rest are held by cooldown."
        ),
        session_id="scn-persistent-decrease",
        windows=_steady("medium", "high", 8),
    ),
    Scenario(
        name="single-window-spike",
        description="One high-load window inside an otherwise neutral run.",
        expectation="Every window holds: one window never satisfies the dwell.",
        session_id="scn-single-window-spike",
        windows=(
            *_steady("medium", "medium", 3),
            WindowSpec(engagement="medium", cognitive_load="high", difficulty=3),
            *_steady("medium", "medium", 4),
        ),
    ),
    Scenario(
        name="conflicting-evidence",
        description="High engagement together with high cognitive load.",
        expectation="Every window holds: the two readings disagree.",
        session_id="scn-conflicting-evidence",
        windows=_steady("high", "high", 6),
    ),
    Scenario(
        name="gate-blocked",
        description=(
            "Sustained high load, but Milestone 7 blocks every window on "
            "signal quality."
        ),
        expectation=(
            "Every window holds with gate_blocked, and Milestone 7's own "
            "signal_quality_below_gate reason is preserved on the record. "
            "Poor signal quality is a statement about the measurement, and it "
            "produces no direction of its own."
        ),
        session_id="scn-gate-blocked",
        windows=tuple(
            WindowSpec(
                engagement="medium",
                cognitive_load="high",
                difficulty=3,
                cognitive_load_status=BLOCKED_BY_QUALITY,
            )
            for _ in range(6)
        ),
    ),
    Scenario(
        name="prediction-abstained",
        description="Sustained high load, but the model abstained on confidence.",
        expectation=(
            "Every window holds with prediction_abstained, and Milestone 7's "
            "below_confidence_threshold reason is preserved. Milestone 8 never "
            "re-derives a confidence and never lowers a threshold."
        ),
        session_id="scn-prediction-abstained",
        windows=tuple(
            WindowSpec(
                engagement="medium",
                cognitive_load="high",
                difficulty=3,
                cognitive_load_status=BLOCKED_BY_CONFIDENCE,
            )
            for _ in range(6)
        ),
    ),
    Scenario(
        name="cooldown-suppression",
        description="Sustained high load for long enough to adapt twice over.",
        expectation=(
            "Exactly two proposals, separated by more than the cooldown; the "
            "windows between them hold with cooldown_active."
        ),
        session_id="scn-cooldown-suppression",
        windows=_steady("medium", "high", 16, difficulty=5),
    ),
    Scenario(
        name="direction-reversal",
        description=(
            "Sustained high load, then sustained high engagement with low load."
        ),
        expectation=(
            "One decrease, then the reversal waits for the cooldown to expire "
            "AND for fresh persistent evidence before an increase."
        ),
        session_id="scn-direction-reversal",
        windows=(
            *_steady("medium", "high", 4, difficulty=3),
            *_steady("high", "low", 12, difficulty=2),
        ),
    ),
    Scenario(
        name="minimum-bound",
        description="Sustained high load with difficulty already at the minimum.",
        expectation=(
            "Every window holds. Once the dwell requirement is met the reason "
            "becomes already_at_minimum: the policy holds at the bound rather "
            "than emitting a command clamped back to where it already is."
        ),
        session_id="scn-minimum-bound",
        windows=_steady("medium", "high", 8, difficulty=1),
    ),
    Scenario(
        name="maximum-bound",
        description=(
            "Sustained high engagement and low load with difficulty already "
            "at the maximum."
        ),
        expectation=(
            "Every window holds. Once the dwell requirement is met the reason "
            "becomes already_at_maximum."
        ),
        session_id="scn-maximum-bound",
        windows=_steady("high", "low", 8, difficulty=5),
    ),
    Scenario(
        name="budget-exhausted",
        description=(
            "A decrease-eligible run long enough to reach the session "
            "adaptation budget. The environment keeps reporting the same "
            "difficulty, because a proposal is not an applied change."
        ),
        expectation=(
            "Exactly max_adaptations_per_session proposals; every later "
            "window holds with session_adaptation_budget_exhausted."
        ),
        session_id="scn-budget-exhausted",
        windows=_steady("medium", "high", 76, difficulty=5),
    ),
    Scenario(
        name="session-change",
        description=("A decrease-eligible run that crosses into a second session id."),
        expectation=(
            "The second session starts cold: no cooldown, no dwell count, and "
            "no previous direction carried over."
        ),
        session_id="scn-session-change-a",
        windows=(
            *_steady("medium", "high", 4, difficulty=4),
            *tuple(
                WindowSpec(
                    engagement="medium",
                    cognitive_load="high",
                    difficulty=4,
                    session_id="scn-session-change-b",
                    window_order=index,
                    window_id=f"session-change-b-w{index:03d}",
                )
                for index in range(4)
            ),
        ),
    ),
    Scenario(
        name="duplicate-window",
        description="A decrease-eligible run in which one window repeats.",
        expectation=(
            "The repeat is absorbed with duplicate_window: it advances no "
            "count and expires no guard, so the proposal still needs three "
            "distinct supporting windows."
        ),
        session_id="scn-duplicate-window",
        windows=(
            WindowSpec(
                engagement="medium",
                cognitive_load="high",
                difficulty=3,
                window_id="dup-w000",
                window_order=0,
            ),
            WindowSpec(
                engagement="medium",
                cognitive_load="high",
                difficulty=3,
                window_id="dup-w000",
                window_order=0,
            ),
            WindowSpec(
                engagement="medium",
                cognitive_load="high",
                difficulty=3,
                window_id="dup-w001",
                window_order=1,
            ),
            WindowSpec(
                engagement="medium",
                cognitive_load="high",
                difficulty=3,
                window_id="dup-w002",
                window_order=2,
            ),
        ),
    ),
    Scenario(
        name="no-usable-target",
        description="Cognitive-load evidence is absent from every window.",
        expectation=(
            "Every window holds with insufficient_evidence. Partial evidence "
            "cannot select a direction; see the mapping."
        ),
        session_id="scn-no-usable-target",
        windows=tuple(
            WindowSpec(engagement="high", cognitive_load=None, difficulty=3)
            for _ in range(6)
        ),
    ),
)

#: Scenario names, in suite order.
SCENARIO_NAMES: tuple[str, ...] = tuple(s.name for s in SCENARIOS)


def get_scenario(name: str) -> Scenario:
    """The scenario called ``name``.

    Raises
    ------
    KeyError
        If no scenario has that name.
    """
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(f"unknown scenario {name!r}; available: {list(SCENARIO_NAMES)}")


__all__ = [
    "BLOCKED_BY_CONFIDENCE",
    "BLOCKED_BY_QUALITY",
    "ELIGIBLE",
    "EVIDENCE_MISSING",
    "PREDICTION_MISSING",
    "SCENARIOS",
    "SCENARIO_DISCLAIMER",
    "SCENARIO_NAMES",
    "Scenario",
    "WindowSpec",
    "build_evidence",
    "build_inputs",
    "get_scenario",
]
