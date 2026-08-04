"""Task simulator: determinism, event structure, provenance, scenarios.

Every test runs with an injected :class:`ManualClock` and a no-op sleep,
so no real time passes and results do not depend on machine speed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from engagevr.config import TaskConfig
from engagevr.protocol.envelope import SYNTHETIC_LABEL, MessageEnvelope
from engagevr.protocol.messages import (
    AdaptationCommandName,
    AdaptationCommandPayload,
    MessageSource,
    MessageType,
    TaskState,
)
from engagevr.protocol.validation import DecodedMessage, decode_message
from engagevr.schemas.events import EventType, ResponseOutcome
from engagevr.schemas.session import DataSource
from engagevr.storage import SessionStore
from engagevr.synchronization.clock import ManualClock
from engagevr.task import (
    Scenario,
    ScenarioKind,
    SimulatorConfig,
    SimulatorSpeed,
    TaskSimulator,
    generate_trial_plan,
)
from engagevr.task.simulator import PRODUCER
from engagevr.transport import InProcessTransport, JsonlFileTransport


def make_config(
    *,
    blocks: int = 2,
    trials_per_block: int = 5,
    seed: int = 42,
    speed: float = 0.0,
    scenarios: list[Scenario] | None = None,
    **task_overrides: object,
) -> SimulatorConfig:
    task = TaskConfig.model_validate(
        {
            "blocks": blocks,
            "trials_per_block": trials_per_block,
            **task_overrides,
        }
    )
    return SimulatorConfig(task=task, seed=seed, speed=speed, scenarios=scenarios or [])


async def run_simulator(
    config: SimulatorConfig,
    *,
    session_id: str = "sim-session",
    clock: ManualClock | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    transport: InProcessTransport | None = None,
) -> tuple[object, list[MessageEnvelope]]:
    """Run a simulator over an in-process transport and collect its output."""
    clock = clock if clock is not None else ManualClock()
    transport = transport if transport is not None else InProcessTransport()

    async def no_sleep(_seconds: float) -> None:
        return None

    simulator = TaskSimulator(
        session_id=session_id,
        config=config,
        transport=transport,
        clock=clock,
        sleep=sleep if sleep is not None else no_sleep,
        collect_envelopes=True,
    )
    result = await simulator.run()
    return result, list(transport.sent)


def task_events(envelopes: list[MessageEnvelope]) -> list[dict[str, object]]:
    return [
        e.payload["event"]  # type: ignore[index]
        for e in envelopes
        if e.message_type is MessageType.TASK_EVENT
    ]


# --- trial plan ------------------------------------------------------------


class TestTrialPlan:
    def test_geometry_matches_the_configuration(self) -> None:
        plan = generate_trial_plan(make_config(blocks=3, trials_per_block=7))
        assert len(plan.trials) == 21
        assert plan.blocks == 3
        assert {t.block_id for t in plan.trials} == {0, 1, 2}
        assert len(plan.for_block(1)) == 7

    def test_the_same_seed_gives_an_identical_plan(self) -> None:
        first = generate_trial_plan(make_config(seed=7))
        second = generate_trial_plan(make_config(seed=7))
        assert first.trials == second.trials

    def test_a_different_seed_gives_a_different_plan(self) -> None:
        first = generate_trial_plan(make_config(seed=1, trials_per_block=30))
        second = generate_trial_plan(make_config(seed=2, trials_per_block=30))
        assert first.trials != second.trials

    def test_timeouts_carry_no_response_and_no_reaction_time(self) -> None:
        plan = generate_trial_plan(
            make_config(trials_per_block=50, synthetic_timeout_rate=0.5)
        )
        timed_out = [t for t in plan.trials if t.timed_out]
        assert timed_out, "the configuration should produce timeouts"
        for trial in timed_out:
            assert trial.observed_response is None
            assert trial.reaction_time_ms is None
            assert trial.response_correct is None

    def test_incorrect_responses_differ_from_the_expected_key(self) -> None:
        plan = generate_trial_plan(
            make_config(trials_per_block=50, synthetic_error_rate=0.5)
        )
        incorrect = [
            t for t in plan.trials if t.response_outcome is ResponseOutcome.INCORRECT
        ]
        assert incorrect
        for trial in incorrect:
            assert trial.observed_response != trial.expected_response
            assert trial.response_correct is False

    def test_reaction_times_are_positive_and_below_the_timeout(self) -> None:
        config = make_config(trials_per_block=50)
        plan = generate_trial_plan(config)
        for trial in plan.trials:
            if trial.reaction_time_ms is None:
                continue
            assert 0.0 < trial.reaction_time_ms < config.task.response_timeout_ms

    def test_counts_partition_the_trials(self) -> None:
        plan = generate_trial_plan(make_config(trials_per_block=40))
        assert plan.correct_count + plan.incorrect_count + plan.timeout_count == len(
            plan.trials
        )


# --- simulator run ---------------------------------------------------------


class TestSimulatorRun:
    @pytest.mark.asyncio
    async def test_completes_and_reports_its_counts(self) -> None:
        config = make_config(blocks=2, trials_per_block=5)
        result, envelopes = await run_simulator(config)

        assert result.completed is True  # type: ignore[attr-defined]
        assert result.blocks == 2  # type: ignore[attr-defined]
        assert result.trials == 10  # type: ignore[attr-defined]
        plan = generate_trial_plan(config)
        assert result.timeout_count == plan.timeout_count  # type: ignore[attr-defined]
        assert (
            result.synthetic_response_count  # type: ignore[attr-defined]
            == plan.correct_count + plan.incorrect_count
        )
        assert len(envelopes) == result.emitted_message_count  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_output_is_deterministic_for_a_seed(self) -> None:
        config = make_config(seed=99)
        _first_result, first = await run_simulator(config)
        _second_result, second = await run_simulator(config)

        # message_id is a fresh uuid per run, so compare everything else.
        def comparable(envelopes: list[MessageEnvelope]) -> list[dict[str, object]]:
            return [
                e.model_dump(mode="json", exclude={"message_id", "correlation_id"})
                for e in envelopes
            ]

        assert comparable(first) == comparable(second)

    @pytest.mark.asyncio
    async def test_accelerated_and_immediate_produce_identical_events(self) -> None:
        slept: list[float] = []

        async def record_sleep(seconds: float) -> None:
            slept.append(seconds)

        immediate_result, immediate = await run_simulator(make_config(speed=0.0))
        accelerated_result, accelerated = await run_simulator(
            make_config(speed=10.0), sleep=record_sleep
        )

        assert [e.message_type for e in immediate] == [
            e.message_type for e in accelerated
        ]
        assert task_events(immediate) == task_events(accelerated)
        assert immediate_result.emitted_message_count == (  # type: ignore[attr-defined]
            accelerated_result.emitted_message_count  # type: ignore[attr-defined]
        )
        assert slept, "the accelerated mode must actually pace itself"
        assert all(delay >= 0.0 for delay in slept)

    @pytest.mark.asyncio
    async def test_immediate_mode_never_sleeps(self) -> None:
        slept: list[float] = []

        async def record_sleep(seconds: float) -> None:
            slept.append(seconds)

        await run_simulator(make_config(speed=0.0), sleep=record_sleep)
        assert slept == []

    def test_speed_selects_the_mode(self) -> None:
        assert make_config(speed=0.0).mode is SimulatorSpeed.IMMEDIATE
        assert make_config(speed=1.0).mode is SimulatorSpeed.REALTIME
        assert make_config(speed=10.0).mode is SimulatorSpeed.ACCELERATED

    @pytest.mark.asyncio
    async def test_sequence_numbers_are_contiguous_from_zero(self) -> None:
        _result, envelopes = await run_simulator(make_config())
        assert [e.sequence_number for e in envelopes] == list(range(len(envelopes)))

    @pytest.mark.asyncio
    async def test_message_ids_are_unique(self) -> None:
        _result, envelopes = await run_simulator(make_config())
        assert len({e.message_id for e in envelopes}) == len(envelopes)

    @pytest.mark.asyncio
    async def test_every_emitted_message_validates(self) -> None:
        import json

        _result, envelopes = await run_simulator(make_config())
        for envelope in envelopes:
            decode_message(json.dumps(envelope.to_json_dict()))


class TestEventOrdering:
    @pytest.mark.asyncio
    async def test_lifecycle_order_is_correct(self) -> None:
        _result, envelopes = await run_simulator(
            make_config(blocks=2, trials_per_block=2)
        )
        types = [e.message_type.value for e in envelopes]
        assert types[0] == "client_hello"
        assert types[1] == "session_start"
        assert types[-1] == "session_end"

        events = [str(e["event_type"]) for e in task_events(envelopes)]
        assert events[0] == EventType.TASK_LOADED.value
        assert events[1] == EventType.TASK_STARTED.value
        assert events[2] == EventType.BLOCK_STARTED.value
        assert events[-1] == EventType.TASK_COMPLETED.value
        assert events[-2] == EventType.BLOCK_COMPLETED.value

    @pytest.mark.asyncio
    async def test_each_trial_emits_start_stimulus_resolution_completion(self) -> None:
        _result, envelopes = await run_simulator(
            make_config(blocks=1, trials_per_block=3)
        )
        events = [str(e["event_type"]) for e in task_events(envelopes)]
        per_trial = [
            e
            for e in events
            if e
            in {
                EventType.TRIAL_STARTED.value,
                EventType.STIMULUS_PRESENTED.value,
                EventType.RESPONSE_REGISTERED.value,
                EventType.RESPONSE_TIMEOUT.value,
                EventType.TRIAL_COMPLETED.value,
            }
        ]
        assert len(per_trial) == 3 * 4
        for index in range(0, len(per_trial), 4):
            assert per_trial[index] == EventType.TRIAL_STARTED.value
            assert per_trial[index + 1] == EventType.STIMULUS_PRESENTED.value
            assert per_trial[index + 2] in {
                EventType.RESPONSE_REGISTERED.value,
                EventType.RESPONSE_TIMEOUT.value,
            }
            assert per_trial[index + 3] == EventType.TRIAL_COMPLETED.value

    @pytest.mark.asyncio
    async def test_block_counts_match_the_configuration(self) -> None:
        _result, envelopes = await run_simulator(
            make_config(blocks=3, trials_per_block=2)
        )
        events = [str(e["event_type"]) for e in task_events(envelopes)]
        assert events.count(EventType.BLOCK_STARTED.value) == 3
        assert events.count(EventType.BLOCK_COMPLETED.value) == 3

    @pytest.mark.asyncio
    async def test_task_elapsed_never_decreases(self) -> None:
        _result, envelopes = await run_simulator(make_config())
        elapsed = [
            float(e["task_elapsed_ms"])  # type: ignore[arg-type]
            for e in task_events(envelopes)
            if e.get("task_elapsed_ms") is not None
        ]
        assert elapsed == sorted(elapsed)


class TestSyntheticProvenance:
    @pytest.mark.asyncio
    async def test_every_message_is_labelled_synthetic(self) -> None:
        _result, envelopes = await run_simulator(make_config())
        for envelope in envelopes:
            assert envelope.provenance.data_source is DataSource.SYNTHETIC
            assert envelope.provenance.synthetic_label == SYNTHETIC_LABEL
            assert envelope.provenance.producer == PRODUCER
            assert envelope.source is MessageSource.PYTHON_SIMULATOR

    @pytest.mark.asyncio
    async def test_no_message_is_marked_as_a_replay(self) -> None:
        _result, envelopes = await run_simulator(make_config())
        assert all(e.replay is None for e in envelopes)

    @pytest.mark.asyncio
    async def test_session_start_records_the_synthetic_label(self) -> None:
        _result, envelopes = await run_simulator(make_config())
        start = next(
            e for e in envelopes if e.message_type is MessageType.SESSION_START
        )
        configuration = start.payload["configuration"]
        assert isinstance(configuration, dict)
        assert configuration["synthetic_label"] == SYNTHETIC_LABEL
        assert configuration["data_source"] == DataSource.SYNTHETIC.value

    @pytest.mark.asyncio
    async def test_result_reports_its_own_provenance(self) -> None:
        result, _envelopes = await run_simulator(make_config())
        assert result.data_source is DataSource.SYNTHETIC  # type: ignore[attr-defined]
        assert result.synthetic_label == SYNTHETIC_LABEL  # type: ignore[attr-defined]


class TestMissingResponses:
    @pytest.mark.asyncio
    async def test_a_timeout_emits_a_timeout_event_not_a_response(self) -> None:
        config = make_config(
            trials_per_block=6, synthetic_timeout_rate=1.0, synthetic_error_rate=0.0
        )
        result, envelopes = await run_simulator(config)

        events = task_events(envelopes)
        timeouts = [
            e for e in events if e["event_type"] == EventType.RESPONSE_TIMEOUT.value
        ]
        responses = [
            e for e in events if e["event_type"] == EventType.RESPONSE_REGISTERED.value
        ]
        assert responses == []
        assert len(timeouts) == config.total_trials
        assert result.synthetic_response_count == 0  # type: ignore[attr-defined]
        assert result.timeout_count == config.total_trials  # type: ignore[attr-defined]

        for event in timeouts:
            assert event["reaction_time_ms"] is None
            assert event["observed_response"] is None
            assert event["response_correct"] is None
            assert event["response_outcome"] == ResponseOutcome.TIMEOUT.value

    @pytest.mark.asyncio
    async def test_the_session_still_completes_with_no_responses(self) -> None:
        result, envelopes = await run_simulator(
            make_config(synthetic_timeout_rate=1.0, synthetic_error_rate=0.0)
        )
        assert result.completed is True  # type: ignore[attr-defined]
        assert envelopes[-1].message_type is MessageType.SESSION_END
        assert envelopes[-1].payload["completed"] is True

    @pytest.mark.asyncio
    async def test_incorrect_responses_are_recorded_as_responses(self) -> None:
        config = make_config(
            trials_per_block=6, synthetic_error_rate=1.0, synthetic_timeout_rate=0.0
        )
        result, envelopes = await run_simulator(config)

        assert result.incorrect_response_count == config.total_trials  # type: ignore[attr-defined]
        assert result.correct_response_count == 0  # type: ignore[attr-defined]
        for event in task_events(envelopes):
            if event["event_type"] != EventType.RESPONSE_REGISTERED.value:
                continue
            assert event["response_correct"] is False
            assert event["response_outcome"] == ResponseOutcome.INCORRECT.value
            assert isinstance(event["reaction_time_ms"], float)


class TestScenarios:
    @pytest.mark.asyncio
    async def test_pause_and_resume_are_emitted(self) -> None:
        config = make_config(
            blocks=1,
            trials_per_block=3,
            scenarios=[Scenario(kind=ScenarioKind.PAUSE, block_id=0, trial_id=1)],
        )
        _result, envelopes = await run_simulator(config)
        events = [str(e["event_type"]) for e in task_events(envelopes)]

        assert EventType.TASK_PAUSED.value in events
        assert EventType.TASK_RESUMED.value in events
        assert events.index(EventType.TASK_PAUSED.value) < events.index(
            EventType.TASK_RESUMED.value
        )

    @pytest.mark.asyncio
    async def test_disconnect_stops_without_a_session_end(self) -> None:
        config = make_config(
            blocks=1,
            trials_per_block=5,
            scenarios=[Scenario(kind=ScenarioKind.DISCONNECT, block_id=0, trial_id=2)],
        )
        result, envelopes = await run_simulator(config)

        assert result.disconnected is True  # type: ignore[attr-defined]
        assert result.completed is False  # type: ignore[attr-defined]
        assert all(e.message_type is not MessageType.SESSION_END for e in envelopes)

    @pytest.mark.asyncio
    async def test_abort_emits_a_task_aborted_event(self) -> None:
        config = make_config(
            blocks=1,
            trials_per_block=5,
            scenarios=[Scenario(kind=ScenarioKind.ABORT, block_id=0, trial_id=2)],
        )
        _result, envelopes = await run_simulator(config)
        events = [str(e["event_type"]) for e in task_events(envelopes)]
        assert events[-1] == EventType.TASK_ABORTED.value

    def test_a_scenario_outside_the_task_geometry_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="block"):
            make_config(
                blocks=1,
                scenarios=[Scenario(kind=ScenarioKind.PAUSE, block_id=5, trial_id=0)],
            )
        with pytest.raises(ValidationError, match="trial"):
            make_config(
                trials_per_block=2,
                scenarios=[Scenario(kind=ScenarioKind.PAUSE, block_id=0, trial_id=9)],
            )


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancellation_records_an_abort_and_an_incomplete_end(self) -> None:
        transport = InProcessTransport()
        clock = ManualClock()
        cancelled_after = 12

        async def sleep_then_cancel(_seconds: float) -> None:
            if len(transport.sent) >= cancelled_after:
                raise asyncio.CancelledError

        simulator = TaskSimulator(
            session_id="cancel-session",
            config=make_config(blocks=2, trials_per_block=10, speed=1.0),
            transport=transport,
            clock=clock,
            sleep=sleep_then_cancel,
        )

        with pytest.raises(asyncio.CancelledError):
            await simulator.run()

        types = [e.message_type for e in transport.sent]
        events = [
            str(e.payload["event"]["event_type"])  # type: ignore[index]
            for e in transport.sent
            if e.message_type is MessageType.TASK_EVENT
        ]
        assert EventType.TASK_ABORTED.value in events
        assert types[-1] is MessageType.SESSION_END
        assert transport.sent[-1].payload["completed"] is False
        assert transport.sent[-1].payload["reason"] == "cancelled"

    @pytest.mark.asyncio
    async def test_sequence_numbers_stay_contiguous_through_cancellation(self) -> None:
        transport = InProcessTransport()

        async def sleep_then_cancel(_seconds: float) -> None:
            if len(transport.sent) >= 8:
                raise asyncio.CancelledError

        simulator = TaskSimulator(
            session_id="cancel-session",
            config=make_config(speed=1.0),
            transport=transport,
            clock=ManualClock(),
            sleep=sleep_then_cancel,
        )
        with pytest.raises(asyncio.CancelledError):
            await simulator.run()

        numbers = [e.sequence_number for e in transport.sent]
        assert numbers == list(range(len(numbers)))


class TestAdaptationHandling:
    @pytest.mark.asyncio
    async def test_a_command_is_applied_and_acknowledged(self) -> None:
        transport = InProcessTransport()
        clock = ManualClock()
        command = AdaptationCommandPayload(
            command_id="cmd-1",
            command=AdaptationCommandName.SET_DIFFICULTY,
            value=4,
            reason="manual test",
            issued_at_utc=clock.utc_now(),
        )
        from engagevr.protocol.envelope import build_envelope

        envelope = build_envelope(
            message_type=MessageType.ADAPTATION_COMMAND,
            session_id="sim-session",
            source=MessageSource.BACKEND,
            sequence_number=0,
            payload=command,
            sent_at_utc=clock.utc_now(),
            sent_at_monotonic_seconds=0.0,
        )
        await transport.deliver(DecodedMessage(envelope=envelope, payload=command))

        async def no_sleep(_seconds: float) -> None:
            return None

        simulator = TaskSimulator(
            session_id="sim-session",
            config=make_config(blocks=1, trials_per_block=2),
            transport=transport,
            clock=clock,
            sleep=no_sleep,
        )
        result = await simulator.run()

        assert result.adaptation_commands_received == 1
        assert simulator.runtime_state.difficulty_level == 4
        acknowledgement = next(
            e
            for e in transport.sent
            if e.message_type is MessageType.ADAPTATION_ACKNOWLEDGEMENT
        )
        assert acknowledgement.payload["command_id"] == "cmd-1"
        assert acknowledgement.payload["accepted"] is True
        assert acknowledgement.correlation_id == envelope.message_id

    @pytest.mark.asyncio
    async def test_no_command_is_generated_without_one_arriving(self) -> None:
        """Milestone 4 has no policy: nothing derives a command."""
        result, envelopes = await run_simulator(make_config())
        assert result.adaptation_commands_received == 0  # type: ignore[attr-defined]
        assert all(
            e.message_type is not MessageType.ADAPTATION_COMMAND for e in envelopes
        )
        assert all(
            e.message_type is not MessageType.ADAPTATION_ACKNOWLEDGEMENT
            for e in envelopes
        )


class TestTaskState:
    @pytest.mark.asyncio
    async def test_final_state_is_completed(self) -> None:
        transport = InProcessTransport()

        async def no_sleep(_seconds: float) -> None:
            return None

        simulator = TaskSimulator(
            session_id="s",
            config=make_config(blocks=1, trials_per_block=2),
            transport=transport,
            clock=ManualClock(),
            sleep=no_sleep,
        )
        await simulator.run()
        assert simulator.runtime_state.state is TaskState.COMPLETED


class TestFileTransport:
    @pytest.mark.asyncio
    async def test_an_offline_run_produces_a_readable_recording(
        self, tmp_path: Path
    ) -> None:
        store = SessionStore(tmp_path)
        recorder = store.open_recorder("offline")
        clock = ManualClock()

        async def no_sleep(_seconds: float) -> None:
            return None

        simulator = TaskSimulator(
            session_id="offline",
            config=make_config(blocks=1, trials_per_block=3),
            transport=JsonlFileTransport(recorder, clock=clock),
            clock=clock,
            sleep=no_sleep,
        )
        result = await simulator.run()
        summary = recorder.close()

        assert summary.event_count == result.emitted_message_count
        assert summary.completed is True
        assert summary.synthetic_message_count == summary.event_count
        stored = list(store.iter_messages("offline"))
        assert [m.ingestion.arrival_index for m in stored] == list(range(len(stored)))
        assert all(m.ingestion.transport == "file" for m in stored)


class TestConfigurationValidation:
    def test_impossible_probability_mass_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="probability mass"):
            TaskConfig(synthetic_error_rate=0.7, synthetic_timeout_rate=0.5)

    def test_timeout_shorter_than_the_stimulus_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="response_timeout_ms"):
            TaskConfig(stimulus_duration_ms=1000.0, response_timeout_ms=500.0)

    def test_zero_blocks_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TaskConfig(blocks=0)

    def test_negative_speed_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SimulatorConfig(speed=-1.0)

    def test_out_of_range_probabilities_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TaskConfig(synthetic_error_rate=1.5)
        with pytest.raises(ValidationError):
            TaskConfig(synthetic_timeout_rate=-0.1)
