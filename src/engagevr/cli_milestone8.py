"""Milestone 8 CLI: the conservative adaptation policy.

``adaptation-demo`` runs the deterministic controller-scenario suite
through the policy and writes an auditable trace and a set of
controller-behaviour metrics.

Kept in its own module so ``__main__`` stays a thin dispatcher and so the
Milestone 8 command can be tested without importing the webcam, rPPG, or
WebSocket code paths.  Running the demo starts no server, opens no
socket, and sends nothing: it builds command objects and stops.

No command in this module prints "improved engagement", "reduced
cognitive load", "better policy", "optimal adaptation", "effective",
"safe", or "validated".  A synthetic controller trace shows that the
policy is wired together correctly; it is not evidence that any
adaptation helps any person, and the printed output says so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER, EvaluationMode

#: Printed by every command in this module.
_PERMANENT_DISCLAIMER = (
    "EngageVR produces ESTIMATES from software measurements. Nothing here is "
    "a medical, diagnostic, psychological, or clinical conclusion, and no "
    "number printed by these commands is experimental evidence about any "
    "person."
)

_SYNTHETIC_BANNER = (
    "=== SYNTHETIC CONTROLLER SCENARIOS ===\n"
    "Every window state below was chosen by the author to make one branch of\n"
    "the policy run. These are deterministic controller tests. They do not\n"
    "simulate a participant, a task, a physiological process, or anyone's\n"
    "response to an adaptation."
)

_POLICY_NOTE = (
    "The mapping from an estimated state to an adaptation direction is an "
    "ENGINEERING DEMONSTRATION RULE. It is not psychologically validated, not "
    "pedagogically optimal, not therapeutic, and not demonstrated to benefit "
    "anyone. Every threshold, dwell time, cooldown, and bound below is an "
    "engineering default."
)

_CONFIG_FILE = "adaptation_policy_config.json"
_TRACE_FILE = "adaptation_trace.parquet"

_BOUNDARY_NOTE = (
    "Milestone 7 decides whether a window may be acted upon; Milestone 8 "
    "decides whether a conservative change should be proposed. A blocked "
    "Milestone 7 gate always holds, and there is no override. A proposal is "
    "not a sent message and not an applied change."
)


def add_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the Milestone 8 subcommands on an existing subparser set."""
    demo = sub.add_parser(
        "adaptation-demo",
        help=(
            "Run the deterministic adaptation-policy scenario suite and write "
            "a controller trace. SYNTHETIC; sends nothing."
        ),
    )
    demo.add_argument(
        "--output", type=str, default="artifacts/experiments/m8-adaptation-demo"
    )
    demo.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run one named scenario instead of the whole suite.",
    )
    demo.add_argument(
        "--experiment-mode",
        type=str,
        default=None,
        choices=["adaptive", "static"],
        help=(
            "Override the configured experimental condition. 'static' is the "
            "non-adaptive control condition: the environment never adapts."
        ),
    )
    demo.add_argument(
        "--disable-adaptation",
        action="store_true",
        help="Apply the experimenter lock: hold every window.",
    )
    demo.add_argument(
        "--no-naive-comparison",
        action="store_true",
        help="Skip the software-controller comparison against a guard-free policy.",
    )
    demo.add_argument(
        "--dispatch",
        action="store_true",
        help=(
            "NOT IMPLEMENTED in Milestone 8 and refused if passed. Live "
            "transport of policy-derived commands is deferred; this flag "
            "exists so that asking for it produces a stated refusal rather "
            "than silently doing nothing."
        ),
    )
    demo.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print the scenario suite and exit.",
    )


def run_adaptation_demo(args: argparse.Namespace) -> int:
    """Run the controller-scenario suite as a software self-check."""
    from engagevr.adaptation.runner import (
        AdaptationRunConfiguration,
        AdaptationRunError,
        run_adaptation,
    )
    from engagevr.adaptation.scenarios import SCENARIOS, get_scenario
    from engagevr.config import load_config
    from engagevr.schemas.adaptation_policy import AdaptationPolicyError

    if getattr(args, "list_scenarios", False):
        for scenario in SCENARIOS:
            print(f"{scenario.name}")
            print(f"  windows:     {len(scenario.windows)}")
            print(f"  description: {scenario.description}")
            print(f"  expectation: {scenario.expectation}")
        return 0

    if getattr(args, "dispatch", False):
        print(
            "Error: --dispatch is not implemented. Milestone 8 constructs "
            "adaptation commands and stops there; no project requirement asks "
            "for live transport of a policy-derived command in this "
            "milestone, and turning one on by default would put an "
            "unvalidated rule in control of a running environment. Use "
            "'serve' and 'task-sim' to exercise the Milestone 4 transport "
            "with a manual command.",
            file=sys.stderr,
        )
        return 2

    scenarios = SCENARIOS
    if args.scenario is not None:
        try:
            scenarios = (get_scenario(args.scenario),)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    settings = load_config()
    adaptation = settings.adaptation
    if args.experiment_mode is not None:
        adaptation = adaptation.model_copy(
            update={"experiment_mode": args.experiment_mode}
        )
    if args.disable_adaptation:
        adaptation = adaptation.model_copy(update={"enabled": False})

    try:
        policy = adaptation.resolve()
    except (ValueError, AdaptationPolicyError) as exc:
        print(f"Error: invalid adaptation configuration: {exc}", file=sys.stderr)
        return 2

    output = Path(args.output)
    config = AdaptationRunConfiguration(
        output_directory=output,
        policy=policy,
        evaluation_mode=EvaluationMode.SOFTWARE_SELF_CHECK,
        data_source="synthetic",
        is_synthetic=True,
        compare_naive=not args.no_naive_comparison,
    )

    print(SOFTWARE_SELF_CHECK_BANNER)
    print(_SYNTHETIC_BANNER)
    print()

    try:
        result = run_adaptation(config, scenarios=scenarios)
    except (AdaptationRunError, AdaptationPolicyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_result(result, scenarios)
    return 0


def _print_result(result: object, scenarios: object) -> None:
    from engagevr.adaptation.runner import AdaptationRunResult
    from engagevr.adaptation.scenarios import Scenario

    assert isinstance(result, AdaptationRunResult)
    assert isinstance(scenarios, tuple)
    suite: tuple[Scenario, ...] = scenarios
    metrics = result.metrics
    policy = result.summary.configuration

    print(f"Run id:                        {result.run_id}")
    print(f"Experiment directory:          {result.directory}")
    print(f"Configuration fingerprint:     {result.summary.configuration_fingerprint}")
    print(f"Adaptation enabled:            {policy.enabled}")
    print(f"Experiment mode:               {policy.experiment_mode.value}")
    print(f"Policy mode:                   {policy.mode.value}")
    print(
        "Persistence requirement:       "
        f"{policy.minimum_persistence_windows} consecutive window(s) "
        "(ENGINEERING DEFAULT, unvalidated)"
    )
    print(
        "Cooldown:                      "
        f"{policy.cooldown_windows} window(s) blocked after a proposal "
        f"(minimum spacing {policy.cooldown_windows + 1} windows)"
    )
    print(
        "Difficulty bounds:             "
        f"[{policy.difficulty.minimum}, {policy.difficulty.maximum}] "
        f"step {policy.difficulty.step} (step is never scaled by confidence)"
    )
    print(
        "Session adaptation budget:     "
        + (
            str(policy.max_adaptations_per_session)
            if policy.max_adaptations_per_session is not None
            else "unlimited"
        )
    )
    print(f"Conflict resolution:           {policy.conflict_resolution.value}")
    print(
        "Regression mapping:            "
        + (
            "enabled"
            if policy.regression_mapping_enabled
            else "disabled; the ordinal class targets carry the deadband"
        )
    )
    print(f"Scenarios:                     {len(suite)}")
    print(f"Sessions:                      {len(result.summary.session_ids)}")
    print()

    print("--- Controller behaviour (SOFTWARE DIAGNOSTICS ONLY) ---")
    print(f"Evaluated policy windows:      {metrics.evaluated_windows}")
    print(f"Milestone 7 eligible:          {metrics.gate_eligible_windows}")
    print(f"Milestone 7 blocked:           {metrics.gate_blocked_windows}")
    print(f"Hold decisions:                {metrics.hold_decisions}")
    print(f"Adaptation proposals:          {metrics.adaptation_proposals}")
    print(f"  increases:                   {metrics.increases}")
    print(f"  decreases:                   {metrics.decreases}")
    print(f"Direction reversals:           {metrics.direction_reversals}")
    print(
        "Minimum proposal spacing:      "
        + (
            f"{metrics.minimum_proposal_spacing_windows} window(s)"
            if metrics.minimum_proposal_spacing_windows is not None
            else "n/a (fewer than two proposals in any session)"
        )
    )
    print(f"Longest same-direction streak: {metrics.longest_same_direction_streak}")
    print(f"Blocked oscillation attempts:  {metrics.blocked_oscillation_attempts}")
    print(
        "Eligible windows that adapted: "
        + (
            f"{metrics.eligible_window_adaptation_fraction:.4f}"
            if metrics.eligible_window_adaptation_fraction is not None
            else "n/a"
        )
    )
    print(f"Commands built:                {len(result.history)}")
    print("Commands dispatched:           0 (Milestone 8 sends nothing)")
    print("Acknowledgements recorded:     0 (no command was sent)")

    if metrics.hold_reason_counts:
        print("Hold reasons:")
        for reason, count in metrics.hold_reason_counts.items():
            print(f"  {reason:<40} {count}")

    if metrics.final_difficulty_by_session:
        print("Final reported difficulty per session:")
        for session, difficulty in metrics.final_difficulty_by_session.items():
            proposals = metrics.proposals_by_session.get(session, 0)
            print(f"  {session:<32} {difficulty}  ({proposals} proposal(s))")

    naive = result.summary.naive_comparison
    if naive is not None:
        print()
        print("--- Software controller comparison (NOT a benefit claim) ---")
        print(
            "A guard-free controller (dwell 1, no cooldown, no budget) on the "
            "same input sequence:"
        )
        print(
            f"  proposals: {naive.adaptation_proposals} vs "
            f"{metrics.adaptation_proposals} conservative"
        )
        print(
            f"  reversals: {naive.direction_reversals} vs "
            f"{metrics.direction_reversals} conservative"
        )
        print(
            "Neither controller has been shown to help anyone. This compares "
            "action frequency only."
        )

    print()
    print(f"Policy configuration:          {result.directory / _CONFIG_FILE}")
    print(f"Scenarios:                     {result.directory / 'scenarios.json'}")
    print(f"Adaptation trace:              {result.directory / _TRACE_FILE}")
    print(
        f"Summary:                       {result.directory / 'adaptation_summary.json'}"
    )
    print(f"Checksums:                     {result.directory / 'checksums.json'}")
    print(
        "scientific_evaluation_eligible="
        f"{str(result.summary.scientific_evaluation_eligible).lower()}"
    )

    print()
    print(_BOUNDARY_NOTE)
    print()
    print(_POLICY_NOTE)
    print()
    print(_PERMANENT_DISCLAIMER)
    print()
    print(SOFTWARE_SELF_CHECK_BANNER)


__all__ = ["add_parsers", "run_adaptation_demo"]
