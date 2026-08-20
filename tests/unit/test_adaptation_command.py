"""The proposal-to-command boundary and the proposal lifecycle.

Nothing in this module starts a server, opens a socket, or reaches a
broker.  That is the point: a policy proposal must be translatable into
the existing Milestone 4 payload as a pure value, and the translation
must refuse anything that did not survive every guard.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from engagevr.adaptation.command import (
    AdaptationCommandBuildError,
    build_adaptation_command,
    build_command_for_decision,
    default_command_id,
)
from engagevr.adaptation.lifecycle import (
    record_acknowledgement,
    record_command_built,
    record_dispatch,
    start_history_entry,
)
from engagevr.protocol.messages import (
    AdaptationAcknowledgementPayload,
    AdaptationCommandName,
    ClientRole,
    MessageType,
    TaskState,
)
from engagevr.schemas.adaptation_policy import (
    AdaptationDirection,
    AdaptationLifecycleStatus,
    AdaptationPolicyError,
)
from engagevr.schemas.targets import TargetName
from engagevr.task.state import TaskRuntimeState
from tests.unit.adaptation_helpers import (
    build_proposal,
    make_evidence,
    run_sequence,
    steady_inputs,
)

ISSUED = datetime(2026, 8, 19, 12, 0, 0, tzinfo=UTC)


class TestCommandVocabulary:
    def test_the_protocol_action_set_is_unchanged(self) -> None:
        assert {c.value for c in AdaptationCommandName} == {
            "set_difficulty",
            "set_stimulus_interval",
            "pause_task",
            "resume_task",
        }

    def test_the_builder_reuses_the_existing_action(self) -> None:
        command = build_adaptation_command(build_proposal(), issued_at_utc=ISSUED)
        assert command.command is AdaptationCommandName.SET_DIFFICULTY

    def test_the_policy_never_issues_pause_or_resume(self) -> None:
        # No fatigue estimator exists, and the specification's only rule
        # calling for a break is a fatigue rule.
        source = Path("src/engagevr/adaptation").rglob("*.py")
        for path in source:
            text = path.read_text(encoding="utf-8")
            assert "PAUSE_TASK" not in text, path.name
            assert "RESUME_TASK" not in text, path.name


class TestCommandBuilder:
    def test_the_target_level_is_translated(self) -> None:
        command = build_adaptation_command(
            build_proposal(
                direction=AdaptationDirection.INCREASE,
                current_difficulty=2,
                requested_difficulty=3,
                proposed_difficulty=3,
            ),
            issued_at_utc=ISSUED,
        )
        assert command.value == 3

    def test_a_policy_command_is_not_manual(self) -> None:
        command = build_adaptation_command(build_proposal(), issued_at_utc=ISSUED)
        assert command.is_manual is False

    def test_the_command_id_is_deterministic(self) -> None:
        proposal = build_proposal()
        assert default_command_id(proposal) == default_command_id(proposal)
        assert build_adaptation_command(
            proposal, issued_at_utc=ISSUED
        ).command_id == default_command_id(proposal)

    def test_provenance_survives_into_the_reason(self) -> None:
        proposal = build_proposal()
        command = build_adaptation_command(proposal, issued_at_utc=ISSUED)
        assert proposal.proposal_id in command.reason
        assert proposal.engagement_gate.source_prediction_id in command.reason
        assert "DEMONSTRATION RULE" in command.reason
        assert len(command.reason) <= 512

    def test_the_reason_makes_no_benefit_claim(self) -> None:
        reason = build_adaptation_command(
            build_proposal(), issued_at_utc=ISSUED
        ).reason.lower()
        for claim in ("improve", "optimal", "effective", "safe"):
            assert claim not in reason
        # "validated" may appear only in its denial.
        assert "not validated with participants" in reason
        assert reason.count("validated") == 1

    def test_expiry_uses_the_existing_protocol_semantics(self) -> None:
        expires = datetime(2026, 8, 19, 12, 0, 30, tzinfo=UTC)
        command = build_adaptation_command(
            build_proposal(), issued_at_utc=ISSUED, expires_at_utc=expires
        )
        assert command.expires_at_utc == expires

    def test_an_expiry_before_issue_is_refused_by_the_protocol(self) -> None:
        with pytest.raises(ValueError, match="must be after issued_at_utc"):
            build_adaptation_command(
                build_proposal(),
                issued_at_utc=ISSUED,
                expires_at_utc=datetime(2026, 8, 19, 11, 0, 0, tzinfo=UTC),
            )

    def test_a_hold_cannot_build_a_command(self) -> None:
        hold = run_sequence(steady_inputs("medium", "medium", 1))[0]
        assert hold.held
        with pytest.raises(AdaptationCommandBuildError, match="produced a hold"):
            build_command_for_decision(hold, issued_at_utc=ISSUED)

    def test_a_blocked_gate_cannot_build_a_command(self) -> None:
        blocked = make_evidence(
            TargetName.COGNITIVE_LOAD_CLASS, "high", eligible=False
        ).gate
        proposal = build_proposal().model_copy(update={"cognitive_load_gate": blocked})
        with pytest.raises(AdaptationCommandBuildError, match="no override"):
            build_adaptation_command(proposal, issued_at_utc=ISSUED)

    def test_an_out_of_bounds_target_is_refused(self) -> None:
        proposal = build_proposal().model_copy(update={"proposed_difficulty": 42})
        with pytest.raises(AdaptationCommandBuildError, match="outside its own bounds"):
            build_adaptation_command(proposal, issued_at_utc=ISSUED)

    def test_a_non_task_target_role_is_refused(self) -> None:
        with pytest.raises(AdaptationCommandBuildError, match="task client role"):
            build_adaptation_command(
                build_proposal(),
                issued_at_utc=ISSUED,
                target_role=ClientRole.OBSERVER,
            )

    def test_a_proposing_decision_builds_its_command(self) -> None:
        decision = run_sequence(steady_inputs("medium", "high", 3))[2]
        command = build_command_for_decision(decision, issued_at_utc=ISSUED)
        assert decision.proposal is not None
        assert command.value == decision.proposal.proposed_difficulty


class TestNoNetworkSideEffect:
    def test_the_command_module_imports_no_transport(self) -> None:
        tree = ast.parse(
            Path("src/engagevr/adaptation/command.py").read_text(encoding="utf-8")
        )
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        forbidden = (
            "engagevr.api",
            "engagevr.transport",
            "websockets",
            "httpx",
            "socket",
            "asyncio",
            "fastapi",
            "uvicorn",
        )
        for module in modules:
            assert not any(module.startswith(f) for f in forbidden), module

    def test_no_adaptation_module_calls_send_or_broadcast(self) -> None:
        for path in sorted(Path("src/engagevr/adaptation").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in {
                        "send",
                        "send_json",
                        "broadcast",
                        "publish",
                        "dispatch",
                    }, f"{path.name} calls {node.func.attr}"


class TestCommandIsAcceptedByTheTaskClient:
    def test_the_reference_client_applies_a_policy_command(self) -> None:
        # The Milestone 4 client is the executable reference for what the
        # Unity client must do. A policy command must be an ordinary
        # command to it, with no new handling.
        command = build_adaptation_command(build_proposal(), issued_at_utc=ISSUED)
        runtime = TaskRuntimeState(task_id="t", state=TaskState.RUNNING)
        ack = runtime.apply_command(command, session_id="s", now_utc=ISSUED)
        assert ack.accepted is True
        assert runtime.difficulty_level == 2

    def test_a_retransmitted_command_does_not_double_step(self) -> None:
        command = build_adaptation_command(build_proposal(), issued_at_utc=ISSUED)
        runtime = TaskRuntimeState(task_id="t", state=TaskState.RUNNING)
        runtime.apply_command(command, session_id="s", now_utc=ISSUED)
        again = runtime.apply_command(command, session_id="s", now_utc=ISSUED)
        assert again.duplicate is True
        assert runtime.difficulty_level == 2

    def test_the_command_is_a_critical_message_type(self) -> None:
        from engagevr.protocol.messages import CRITICAL_MESSAGE_TYPES

        assert MessageType.ADAPTATION_COMMAND in CRITICAL_MESSAGE_TYPES


class TestLifecycle:
    def _built(self) -> tuple[object, object]:
        proposal = build_proposal()
        entry = start_history_entry(proposal)
        command = build_adaptation_command(proposal, issued_at_utc=ISSUED)
        return record_command_built(entry, command), command

    def test_a_new_entry_is_only_proposed(self) -> None:
        entry = start_history_entry(build_proposal())
        assert entry.status is AdaptationLifecycleStatus.PROPOSED
        assert entry.command_id is None
        assert entry.expected_command == "set_difficulty"
        assert entry.expected_value == 2

    def test_proposed_is_not_dispatched(self) -> None:
        built, _ = self._built()
        assert built.status is AdaptationLifecycleStatus.COMMAND_BUILT  # type: ignore[attr-defined]
        assert built.dispatched_at_utc is None  # type: ignore[attr-defined]

    def test_dispatched_is_not_acknowledged(self) -> None:
        built, _ = self._built()
        dispatched = record_dispatch(built, dispatched_at_utc=ISSUED)  # type: ignore[arg-type]
        assert dispatched.status is AdaptationLifecycleStatus.DISPATCHED
        assert dispatched.acknowledged is None

    def test_applied_is_recorded_only_from_a_real_acknowledgement(self) -> None:
        built, command = self._built()
        applied_at = datetime(2026, 8, 19, 12, 0, 1, tzinfo=UTC)
        acknowledged = record_acknowledgement(
            built,  # type: ignore[arg-type]
            AdaptationAcknowledgementPayload(
                command_id=command.command_id,  # type: ignore[attr-defined]
                accepted=True,
                applied_at_utc=applied_at,
            ),
        )
        assert acknowledged.status is AdaptationLifecycleStatus.APPLIED
        assert acknowledged.applied_at_utc == applied_at

    def test_an_accepted_acknowledgement_without_an_instant_stops_short(self) -> None:
        built, command = self._built()
        acknowledged = record_acknowledgement(
            built,  # type: ignore[arg-type]
            AdaptationAcknowledgementPayload(
                command_id=command.command_id,  # type: ignore[attr-defined]
                accepted=True,
            ),
        )
        assert acknowledged.status is AdaptationLifecycleStatus.ACKNOWLEDGED
        assert acknowledged.applied_at_utc is None

    def test_a_rejection_keeps_the_clients_own_reason(self) -> None:
        built, command = self._built()
        rejected = record_acknowledgement(
            built,  # type: ignore[arg-type]
            AdaptationAcknowledgementPayload(
                command_id=command.command_id,  # type: ignore[attr-defined]
                accepted=False,
                rejection_reason="the task is not running",
            ),
        )
        assert rejected.status is AdaptationLifecycleStatus.REJECTED
        assert rejected.rejection_reason == "the task is not running"
        assert rejected.applied_at_utc is None

    def test_an_acknowledgement_for_another_command_is_refused(self) -> None:
        built, _ = self._built()
        with pytest.raises(
            AdaptationPolicyError, match="never attached to a different"
        ):
            record_acknowledgement(
                built,  # type: ignore[arg-type]
                AdaptationAcknowledgementPayload(
                    command_id="someone-else", accepted=True
                ),
            )

    def test_a_proposal_cannot_be_dispatched_before_a_command_exists(self) -> None:
        entry = start_history_entry(build_proposal())
        with pytest.raises(AdaptationPolicyError, match="only a built command"):
            record_dispatch(entry, dispatched_at_utc=ISSUED)

    def test_a_command_is_built_once(self) -> None:
        built, command = self._built()
        with pytest.raises(AdaptationPolicyError, match="built once"):
            record_command_built(built, command)  # type: ignore[arg-type]

    def test_a_mismatched_command_value_is_refused(self) -> None:
        proposal = build_proposal()
        entry = start_history_entry(proposal)
        other = build_adaptation_command(
            build_proposal(
                direction=AdaptationDirection.INCREASE,
                current_difficulty=2,
                requested_difficulty=3,
                proposed_difficulty=3,
            ),
            issued_at_utc=ISSUED,
        )
        with pytest.raises(AdaptationPolicyError, match="expects difficulty"):
            record_command_built(entry, other)
