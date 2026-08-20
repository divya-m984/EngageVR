"""Offline adaptation-policy simulation and its artifacts.

Runs a sequence of policy inputs through :func:`evaluate_policy`, one
session-scoped state per session id, and writes an auditable run
directory::

    artifacts/experiments/<run-name>/
        adaptation_policy_config.json  the resolved configuration
        scenarios.json                 what each scenario exercises
        adaptation_trace.parquet       one row per policy evaluation
        adaptation_summary.json        controller metrics and provenance
        checksums.json                 SHA-256 of every artifact above

The trace deliberately carries **no wall-clock column**.  Every value in
it is a function of the inputs, the configuration, and the initial state,
so two runs of one configuration produce byte-identical Parquet and a
determinism check is a checksum comparison rather than a tolerance.
Timestamps live in the summary, where they are provenance rather than
data.

Nothing here is a measurement of a person.  The metrics count what the
software did.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from engagevr.adaptation.command import build_command_for_decision
from engagevr.adaptation.policy import evaluate_policy
from engagevr.adaptation.scenarios import (
    SCENARIO_DISCLAIMER,
    SCENARIOS,
    Scenario,
    build_inputs,
)
from engagevr.features.assembly import write_parquet_atomic
from engagevr.schemas.adaptation_policy import (
    ACTING_DIRECTIONS,
    ADAPTATION_POLICY_NOTE,
    CONTROLLER_METRIC_NOTE,
    AdaptationControllerMetrics,
    AdaptationDecisionKind,
    AdaptationDirection,
    AdaptationHistoryEntry,
    AdaptationInput,
    AdaptationPolicyConfiguration,
    AdaptationPolicyDecision,
    AdaptationPolicyState,
    AdaptationRunSummary,
    opposite_direction,
)
from engagevr.schemas.experiments import SELF_CHECK_DISCLAIMER, EvaluationMode
from engagevr.schemas.uncertainty import AdaptationGateDecision
from engagevr.training.artifacts import (
    dependency_versions,
    engagevr_version,
    sha256_file,
    write_json_atomic,
)
from engagevr.utils.timestamps import utc_now

#: Artifacts a completed adaptation run must contain.
ADAPTATION_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "adaptation_policy_config.json",
    "adaptation_trace.parquet",
    "adaptation_summary.json",
)

#: The comparison controller: no dwell requirement, no cooldown, no budget.
NAIVE_COMPARISON_NOTE = (
    "SOFTWARE CONTROLLER COMPARISON. The naive controller is the same policy "
    "with the dwell requirement reduced to one window, the cooldown removed, "
    "and the session budget removed. The comparison shows only that the "
    "temporal guards mechanically reduce how often the controller acts. It is "
    "NOT a claim that either controller is better, safer, or more useful for "
    "any person, and the conservative policy was not tuned to win it."
)


class AdaptationRunError(RuntimeError):
    """An adaptation run could not be completed."""


@dataclass(frozen=True, slots=True)
class AdaptationRunConfiguration:
    """Everything that defines one offline policy run."""

    output_directory: Path
    policy: AdaptationPolicyConfiguration
    evaluation_mode: EvaluationMode = EvaluationMode.SOFTWARE_SELF_CHECK
    data_source: str = "synthetic"
    is_synthetic: bool = True
    build_commands: bool = True
    compare_naive: bool = True


@dataclass(slots=True)
class AdaptationRunResult:
    """What a completed offline policy run produced."""

    run_id: str
    directory: Path
    decisions: tuple[AdaptationPolicyDecision, ...]
    history: tuple[AdaptationHistoryEntry, ...]
    metrics: AdaptationControllerMetrics
    summary: AdaptationRunSummary
    final_states: dict[str, AdaptationPolicyState] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_sequence(
    inputs: Sequence[AdaptationInput],
    configuration: AdaptationPolicyConfiguration,
) -> tuple[tuple[AdaptationPolicyDecision, ...], dict[str, AdaptationPolicyState]]:
    """Run a whole input sequence, one session-scoped state per session.

    A new session id starts a cold state: the dwell count, the cooldown,
    the previous direction, and the budget do not cross a session
    boundary, and no identifier from the previous session leaks into the
    next one.  The policy itself refuses to apply one session's state to
    another; this function never asks it to.
    """
    states: dict[str, AdaptationPolicyState] = {}
    decisions: list[AdaptationPolicyDecision] = []
    for policy_input in inputs:
        state = states.get(policy_input.session_id)
        if state is None:
            state = AdaptationPolicyState.for_session(policy_input.session_id)
        decision = evaluate_policy(policy_input, state, configuration)
        states[policy_input.session_id] = decision.state_after
        decisions.append(decision)
    return tuple(decisions), states


def scenario_inputs(
    scenarios: Sequence[Scenario] = SCENARIOS,
) -> tuple[AdaptationInput, ...]:
    """The concatenated input sequence of a scenario suite."""
    inputs: list[AdaptationInput] = []
    for scenario in scenarios:
        inputs.extend(build_inputs(scenario))
    return tuple(inputs)


# ---------------------------------------------------------------------------
# Controller metrics
# ---------------------------------------------------------------------------


def controller_metrics(
    decisions: Sequence[AdaptationPolicyDecision],
    final_states: dict[str, AdaptationPolicyState],
) -> AdaptationControllerMetrics:
    """Mechanical behaviour of the controller on this input sequence.

    Every quantity counts something the software did.  None of them is
    engagement, cognitive load, learning, comfort, or benefit.
    """
    holds = [d for d in decisions if d.held]
    proposals = [d for d in decisions if not d.held]

    eligible = sum(
        1
        for d in decisions
        if d.engagement.gate_decision is AdaptationGateDecision.ELIGIBLE
        and d.cognitive_load.gate_decision is AdaptationGateDecision.ELIGIBLE
    )

    reason_counts: dict[str, int] = {}
    for decision in holds:
        for reason in decision.reasons:
            reason_counts[reason.value] = reason_counts.get(reason.value, 0) + 1

    reversals = 0
    spacings: list[int] = []
    streak = 0
    longest_streak = 0
    blocked_oscillations = 0
    previous_by_session: dict[str, tuple[AdaptationDirection, int]] = {}

    for decision in decisions:
        session = decision.session_id
        previous = previous_by_session.get(session)
        if decision.held:
            if (
                previous is not None
                and decision.resolved_direction in ACTING_DIRECTIONS
                and decision.resolved_direction is opposite_direction(previous[0])
            ):
                blocked_oscillations += 1
            continue
        assert decision.proposal is not None
        direction = decision.proposal.direction
        if previous is not None:
            spacings.append(decision.window_order - previous[1])
            if direction is not previous[0]:
                reversals += 1
                streak = 1
            else:
                streak += 1
        else:
            streak = 1
        longest_streak = max(longest_streak, streak)
        previous_by_session[session] = (direction, decision.window_order)

    return AdaptationControllerMetrics(
        evaluated_windows=len(decisions),
        gate_eligible_windows=eligible,
        gate_blocked_windows=len(decisions) - eligible,
        hold_decisions=len(holds),
        adaptation_proposals=len(proposals),
        increases=sum(
            1
            for d in proposals
            if d.proposal is not None
            and d.proposal.direction is AdaptationDirection.INCREASE
        ),
        decreases=sum(
            1
            for d in proposals
            if d.proposal is not None
            and d.proposal.direction is AdaptationDirection.DECREASE
        ),
        hold_reason_counts=dict(sorted(reason_counts.items())),
        direction_reversals=reversals,
        minimum_proposal_spacing_windows=min(spacings) if spacings else None,
        longest_same_direction_streak=longest_streak,
        eligible_window_adaptation_fraction=(
            len(proposals) / eligible if eligible else None
        ),
        blocked_oscillation_attempts=blocked_oscillations,
        final_difficulty_by_session={
            session: state.current_difficulty
            for session, state in sorted(final_states.items())
            if state.current_difficulty is not None
        },
        proposals_by_session={
            session: state.adaptation_count
            for session, state in sorted(final_states.items())
        },
    )


def naive_configuration(
    configuration: AdaptationPolicyConfiguration,
) -> AdaptationPolicyConfiguration:
    """The same policy with every temporal guard and the budget removed."""
    return configuration.model_copy(
        update={
            "minimum_persistence_windows": 1,
            "cooldown_windows": 0,
            "max_adaptations_per_session": None,
        }
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def build_run_id(
    configuration: AdaptationPolicyConfiguration,
    inputs: Sequence[AdaptationInput],
    evaluation_mode: EvaluationMode,
) -> str:
    """A deterministic identifier for one run.

    A hash of the configuration and of the input sequence's identity, so
    re-running an identical run reproduces the identifier rather than
    accumulating near-duplicate directories.  No clock and no random
    component participates.
    """
    payload = {
        "configuration": configuration.fingerprint(),
        "evaluation_mode": evaluation_mode.value,
        "windows": [
            f"{i.session_id}|{i.window_id}|{i.window_order}|{i.scenario_id}"
            for i in inputs
        ],
        "engagevr_version": engagevr_version(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    short = "sci" if evaluation_mode is EvaluationMode.SCIENTIFIC else "selfcheck"
    return f"adaptation-{short}-{digest}"


def run_adaptation(
    config: AdaptationRunConfiguration,
    inputs: Sequence[AdaptationInput] | None = None,
    *,
    scenarios: Sequence[Scenario] = SCENARIOS,
) -> AdaptationRunResult:
    """Evaluate a sequence offline and write the run directory.

    Raises
    ------
    AdaptationRunError
        If a scientific run is handed synthetic inputs.  Synthetic
        evidence can verify that the controller is wired correctly; it
        can never be scientific evidence about anyone.
    """
    started = utc_now()
    from_suite = inputs is None
    if inputs is None:
        inputs = scenario_inputs(scenarios)
    if not inputs:
        raise AdaptationRunError("an adaptation run needs at least one window")

    synthetic = any(i.is_synthetic for i in inputs)
    if config.evaluation_mode is EvaluationMode.SCIENTIFIC and synthetic:
        raise AdaptationRunError(
            "scientific mode refuses synthetic policy inputs. A synthetic "
            "adaptation trace verifies that the controller is wired together "
            "correctly; it is never evidence about a person, and no policy in "
            "this repository has been evaluated with participants in any case"
        )

    decisions, final_states = evaluate_sequence(inputs, config.policy)
    metrics = controller_metrics(decisions, final_states)

    history: list[AdaptationHistoryEntry] = []
    if config.build_commands:
        history = _build_history(decisions, started)

    naive: AdaptationControllerMetrics | None = None
    if config.compare_naive:
        naive_decisions, naive_states = evaluate_sequence(
            inputs, naive_configuration(config.policy)
        )
        naive = controller_metrics(naive_decisions, naive_states)

    run_id = build_run_id(config.policy, inputs, config.evaluation_mode)
    disclaimers = _disclaimers(config, synthetic)
    summary = AdaptationRunSummary(
        run_id=run_id,
        engagevr_version=engagevr_version(),
        python_version=platform.python_version(),
        evaluation_mode=config.evaluation_mode,
        scientific_evaluation_eligible=(
            config.evaluation_mode is EvaluationMode.SCIENTIFIC and not synthetic
        ),
        is_synthetic=synthetic,
        data_source=config.data_source,
        configuration=config.policy,
        configuration_fingerprint=config.policy.fingerprint(),
        scenario_names=tuple(s.name for s in scenarios) if from_suite else (),
        session_ids=tuple(sorted(final_states)),
        metrics=metrics,
        naive_comparison=naive,
        started_at_utc=started,
        finished_at_utc=utc_now(),
        disclaimers=disclaimers,
    )

    directory = _write_artifacts(
        config, summary, decisions, history, scenarios if from_suite else ()
    )
    return AdaptationRunResult(
        run_id=run_id,
        directory=directory,
        decisions=decisions,
        history=tuple(history),
        metrics=metrics,
        summary=summary,
        final_states=final_states,
    )


def _build_history(
    decisions: Sequence[AdaptationPolicyDecision], issued_at_utc: datetime
) -> list[AdaptationHistoryEntry]:
    """Build the command object for every proposal, and send none of them.

    The lifecycle stops at ``command_built``.  Nothing in this module
    dispatches, so no entry here can reach ``dispatched``,
    ``acknowledged``, or ``applied``.
    """
    from engagevr.adaptation.lifecycle import record_command_built, start_history_entry

    entries: list[AdaptationHistoryEntry] = []
    for decision in decisions:
        if decision.kind is not AdaptationDecisionKind.PROPOSE_ADAPTATION:
            continue
        assert decision.proposal is not None
        entry = start_history_entry(decision.proposal)
        command = build_command_for_decision(decision, issued_at_utc=issued_at_utc)
        entries.append(record_command_built(entry, command))
    return entries


def _disclaimers(
    config: AdaptationRunConfiguration, synthetic: bool
) -> tuple[str, ...]:
    disclaimers = [ADAPTATION_POLICY_NOTE, CONTROLLER_METRIC_NOTE]
    if config.evaluation_mode is EvaluationMode.SOFTWARE_SELF_CHECK:
        disclaimers.insert(0, SELF_CHECK_DISCLAIMER)
    if synthetic:
        disclaimers.append(SCENARIO_DISCLAIMER)
    if config.compare_naive:
        disclaimers.append(NAIVE_COMPARISON_NOTE)
    return tuple(disclaimers)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def trace_table(
    decisions: Sequence[AdaptationPolicyDecision],
    history: Sequence[AdaptationHistoryEntry],
    *,
    run_id: str,
) -> pa.Table:
    """One row per policy evaluation.

    Carries no frame, no landmark, no name, no email, and no timestamp.
    The pseudonymous subject, session, and window references are the ones
    already present in the Milestone 5-7 artifacts.
    """
    built = {entry.proposal_id for entry in history}
    columns: dict[str, list[Any]] = {
        key: []
        for key in (
            "run_id",
            "scenario_id",
            "session_id",
            "subject_id",
            "window_id",
            "window_order",
            "engagement_gate_decision",
            "engagement_gate_reasons",
            "cognitive_load_gate_decision",
            "cognitive_load_gate_reasons",
            "engagement_state",
            "cognitive_load_state",
            "engagement_predicted_class",
            "cognitive_load_predicted_class",
            "engagement_suggestion",
            "cognitive_load_suggestion",
            "conflict",
            "conflict_resolution",
            "resolved_direction",
            "pending_direction_before",
            "pending_direction_after",
            "persistence_before",
            "persistence_after",
            "cooldown_before",
            "cooldown_after",
            "current_difficulty",
            "requested_difficulty",
            "proposed_difficulty",
            "clamping_applied",
            "decision_kind",
            "policy_reasons",
            "adaptation_budget_used",
            "adaptation_budget_total",
            "proposal_id",
            "command_built",
            "lifecycle_status",
            "experiment_mode",
            "policy_mode",
            "configuration_fingerprint",
            "is_synthetic",
            "scientific_evaluation_eligible",
        )
    }
    for decision in decisions:
        proposal = decision.proposal
        columns["run_id"].append(run_id)
        columns["scenario_id"].append(decision.scenario_id)
        columns["session_id"].append(decision.session_id)
        columns["subject_id"].append(decision.subject_id)
        columns["window_id"].append(decision.window_id)
        columns["window_order"].append(decision.window_order)
        columns["engagement_gate_decision"].append(
            decision.engagement.gate_decision.value
            if decision.engagement.gate_decision
            else None
        )
        columns["engagement_gate_reasons"].append(
            [r.value for r in decision.engagement.gate_reasons]
        )
        columns["cognitive_load_gate_decision"].append(
            decision.cognitive_load.gate_decision.value
            if decision.cognitive_load.gate_decision
            else None
        )
        columns["cognitive_load_gate_reasons"].append(
            [r.value for r in decision.cognitive_load.gate_reasons]
        )
        columns["engagement_state"].append(
            decision.engagement.state.value if decision.engagement.state else None
        )
        columns["cognitive_load_state"].append(
            decision.cognitive_load.state.value
            if decision.cognitive_load.state
            else None
        )
        columns["engagement_predicted_class"].append(
            decision.engagement.predicted_class
        )
        columns["cognitive_load_predicted_class"].append(
            decision.cognitive_load.predicted_class
        )
        columns["engagement_suggestion"].append(
            decision.engagement.suggested_direction.value
        )
        columns["cognitive_load_suggestion"].append(
            decision.cognitive_load.suggested_direction.value
        )
        columns["conflict"].append(decision.conflict)
        columns["conflict_resolution"].append(
            decision.conflict_resolution.value if decision.conflict_resolution else None
        )
        columns["resolved_direction"].append(decision.resolved_direction.value)
        columns["pending_direction_before"].append(
            decision.pending_direction_before.value
            if decision.pending_direction_before
            else None
        )
        columns["pending_direction_after"].append(
            decision.pending_direction_after.value
            if decision.pending_direction_after
            else None
        )
        columns["persistence_before"].append(decision.persistence_count_before)
        columns["persistence_after"].append(decision.persistence_count_after)
        columns["cooldown_before"].append(decision.cooldown_remaining_before)
        columns["cooldown_after"].append(decision.cooldown_remaining_after)
        columns["current_difficulty"].append(decision.current_difficulty)
        columns["requested_difficulty"].append(
            proposal.requested_difficulty if proposal else None
        )
        columns["proposed_difficulty"].append(
            proposal.proposed_difficulty if proposal else None
        )
        columns["clamping_applied"].append(
            proposal.clamping_applied if proposal else None
        )
        columns["decision_kind"].append(decision.kind.value)
        columns["policy_reasons"].append([r.value for r in decision.reasons])
        columns["adaptation_budget_used"].append(decision.adaptation_budget_used)
        columns["adaptation_budget_total"].append(decision.adaptation_budget_total)
        columns["proposal_id"].append(proposal.proposal_id if proposal else None)
        columns["command_built"].append(
            proposal is not None and proposal.proposal_id in built
        )
        columns["lifecycle_status"].append(
            ("command_built" if proposal.proposal_id in built else "proposed")
            if proposal
            else None
        )
        columns["experiment_mode"].append(decision.experiment_mode.value)
        columns["policy_mode"].append(decision.policy_mode.value)
        columns["configuration_fingerprint"].append(decision.configuration_fingerprint)
        columns["is_synthetic"].append(decision.is_synthetic)
        columns["scientific_evaluation_eligible"].append(
            decision.scientific_evaluation_eligible
        )
    return pa.table(columns)


def _scenario_document(scenarios: Sequence[Scenario]) -> dict[str, Any]:
    return {
        "disclaimer": SCENARIO_DISCLAIMER,
        "scenarios": [
            {
                "name": scenario.name,
                "session_id": scenario.session_id,
                "subject_id": scenario.subject_id,
                "window_count": len(scenario.windows),
                "description": scenario.description,
                "expectation": scenario.expectation,
            }
            for scenario in scenarios
        ],
    }


def _write_artifacts(
    config: AdaptationRunConfiguration,
    summary: AdaptationRunSummary,
    decisions: Sequence[AdaptationPolicyDecision],
    history: Sequence[AdaptationHistoryEntry],
    scenarios: Sequence[Scenario],
) -> Path:
    directory = config.output_directory
    directory.mkdir(parents=True, exist_ok=True)

    write_json_atomic(
        directory / "adaptation_policy_config.json",
        {
            "run_id": summary.run_id,
            "configuration_fingerprint": summary.configuration_fingerprint,
            "configuration": config.policy.model_dump(mode="json"),
            "dependency_versions": dependency_versions(),
            "disclaimers": list(summary.disclaimers),
        },
    )
    write_json_atomic(directory / "scenarios.json", _scenario_document(scenarios))
    write_parquet_atomic(
        trace_table(decisions, history, run_id=summary.run_id),
        directory / "adaptation_trace.parquet",
    )
    write_json_atomic(
        directory / "adaptation_summary.json", summary.model_dump(mode="json")
    )

    missing = [
        name
        for name in ADAPTATION_REQUIRED_ARTIFACTS
        if not (directory / name).exists()
    ]
    if missing:  # pragma: no cover - every write above is atomic
        raise AdaptationRunError(
            f"run {summary.run_id!r} is missing required artifact(s): {missing}"
        )

    checksums = {
        name: sha256_file(directory / name)
        for name in (
            "adaptation_policy_config.json",
            "scenarios.json",
            "adaptation_trace.parquet",
            "adaptation_summary.json",
        )
    }
    write_json_atomic(directory / "checksums.json", checksums)
    return directory


__all__ = [
    "ADAPTATION_REQUIRED_ARTIFACTS",
    "NAIVE_COMPARISON_NOTE",
    "AdaptationRunConfiguration",
    "AdaptationRunError",
    "AdaptationRunResult",
    "build_run_id",
    "controller_metrics",
    "evaluate_sequence",
    "naive_configuration",
    "run_adaptation",
    "scenario_inputs",
    "trace_table",
]
