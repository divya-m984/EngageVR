"""The deterministic Python task simulator.

Runs the same protocol conversation the Unity desktop task runs, with no
Unity, no webcam, no model asset, no display, and — with an in-process
or file transport — no network.

Everything it reports is fabricated
-----------------------------------
Responses, reaction times, and timeouts come from
:mod:`engagevr.task.generator`, which draws them from a seeded RNG.  No
person performs this task.  Every message this simulator emits carries
``data_source="synthetic"`` and ``synthetic_label="SYNTHETIC"`` in its
provenance, permanently, and nothing downstream clears those markers.

Injected dependencies
---------------------
The clock and the RNG are both injected.  Nothing calls
``time.monotonic`` or ``random`` at module level, so a test can run a
whole session with a :class:`~engagevr.synchronization.clock.ManualClock`
and get identical output every time.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime

from engagevr.protocol.envelope import (
    SYNTHETIC_LABEL,
    MessageEnvelope,
    MessageProvenance,
    build_envelope,
)
from engagevr.protocol.messages import (
    AdaptationAcknowledgementPayload,
    AdaptationCommandPayload,
    ClientHelloPayload,
    ClientRole,
    MessageSource,
    MessageType,
    SessionEndPayload,
    SessionStartPayload,
    TaskEventPayload,
    TaskState,
    TaskStatePayload,
)
from engagevr.protocol.validation import DecodedMessage
from engagevr.protocol.version import PROTOCOL_VERSION
from engagevr.schemas.events import EventType, ResponseOutcome, TaskEventDetail
from engagevr.schemas.session import DataSource
from engagevr.synchronization.clock import Clock, SystemClock
from engagevr.task.config import (
    ScenarioKind,
    SimulatorConfig,
    SimulatorSpeed,
)
from engagevr.task.generator import PlannedTrial, TrialPlan, generate_trial_plan
from engagevr.task.state import TaskRuntimeState
from engagevr.transport import MessageTransport

#: Identifies the producer in every message's provenance.
PRODUCER = "engagevr.task.simulator"

CLIENT_NAME = "engagevr-python-simulator"
CLIENT_VERSION = "0.1.0"


@dataclass
class SimulatorResult:
    """What one simulator run produced.

    ``synthetic_response_count`` counts fabricated responses of any
    correctness.  ``timeout_count`` counts trials where no response was
    fabricated at all; those are excluded from the response count rather
    than counted as zero-latency responses.
    """

    session_id: str
    protocol_version: str = PROTOCOL_VERSION
    blocks: int = 0
    trials: int = 0
    emitted_message_count: int = 0
    task_event_count: int = 0
    synthetic_response_count: int = 0
    correct_response_count: int = 0
    incorrect_response_count: int = 0
    timeout_count: int = 0
    adaptation_commands_received: int = 0
    completed: bool = False
    cancelled: bool = False
    disconnected: bool = False
    data_source: DataSource = DataSource.SYNTHETIC
    synthetic_label: str = SYNTHETIC_LABEL
    envelopes: list[MessageEnvelope] = field(default_factory=list)


class TaskSimulator:
    """Runs one simulated task session over a transport.

    The instance is single-use: call :meth:`run` once.
    """

    def __init__(
        self,
        *,
        session_id: str,
        config: SimulatorConfig,
        transport: MessageTransport,
        clock: Clock | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        collect_envelopes: bool = False,
    ) -> None:
        self._session_id = session_id
        self._config = config
        self._transport = transport
        self._clock = clock if clock is not None else SystemClock()
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._collect = collect_envelopes

        self._plan: TrialPlan = generate_trial_plan(config)
        self._runtime = TaskRuntimeState(
            task_id=config.task_id,
            difficulty_level=config.task.default_difficulty,
            stimulus_interval_ms=config.task.inter_trial_interval_ms,
        )
        self._sequence = 0
        self._simulated_elapsed_ms = 0.0
        self._task_started_ms = 0.0
        self._result = SimulatorResult(
            session_id=session_id,
            blocks=config.task.blocks,
            trials=config.total_trials,
        )
        self._provenance = MessageProvenance(
            data_source=DataSource.SYNTHETIC,
            synthetic_label=SYNTHETIC_LABEL,
            producer=PRODUCER,
        )
        self._scenarios = {
            (scenario.block_id, scenario.trial_id): scenario
            for scenario in config.scenarios
        }
        self._stopped = False

    @property
    def plan(self) -> TrialPlan:
        """The deterministic trial plan this run will execute."""
        return self._plan

    @property
    def runtime_state(self) -> TaskRuntimeState:
        return self._runtime

    async def run(self) -> SimulatorResult:
        """Run the session to completion, cancellation, or disconnect.

        On :class:`asyncio.CancelledError` the simulator emits a
        ``task_aborted`` task event and a ``session_end`` marked
        ``completed=False``, then re-raises.  A cancelled session is
        therefore still a *complete recording of an incomplete run*,
        never a recording that simply stops mid-sentence.
        """
        try:
            await self._handshake()
            await self._session_start()
            await self._run_blocks()
        except asyncio.CancelledError:
            self._result.cancelled = True
            await self._emit_abort("cancelled")
            await self._session_end(completed=False, reason="cancelled")
            raise
        except _SimulatedDisconnect:
            self._result.disconnected = True
            # A disconnect is an abrupt loss of the transport. No
            # session_end is emitted, because the real failure mode this
            # models does not get to send one; recovery is the reader's
            # job, not a fabricated tidy ending.
            return self._result
        else:
            self._result.completed = True
            await self._session_end(completed=True, reason="task_completed")
            return self._result

    # -- message emission -------------------------------------------------

    def _next_sequence(self) -> int:
        value = self._sequence
        self._sequence += 1
        return value

    async def _emit(
        self,
        message_type: MessageType,
        payload: object,
        *,
        correlation_id: str | None = None,
    ) -> MessageEnvelope:
        envelope = build_envelope(
            message_type=message_type,
            session_id=self._session_id,
            source=MessageSource.PYTHON_SIMULATOR,
            sequence_number=self._next_sequence(),
            payload=payload,  # type: ignore[arg-type]
            provenance=self._provenance,
            correlation_id=correlation_id,
            sent_at_utc=self._clock.utc_now(),
            sent_at_monotonic_seconds=self._clock.monotonic(),
        )
        await self._transport.send(envelope)
        self._result.emitted_message_count += 1
        if self._collect:
            self._result.envelopes.append(envelope)
        return envelope

    async def _emit_task_event(self, detail: TaskEventDetail) -> None:
        await self._emit(MessageType.TASK_EVENT, TaskEventPayload(event=detail))
        self._result.task_event_count += 1

    async def _emit_state(self) -> None:
        if not self._config.emit_task_state:
            return
        await self._emit(
            MessageType.TASK_STATE,
            TaskStatePayload(
                state=self._runtime.state,
                task_id=self._runtime.task_id,
                block_id=self._runtime.block_id,
                trial_id=self._runtime.trial_id,
                difficulty_level=self._runtime.difficulty_level,
                stimulus_interval_ms=self._runtime.stimulus_interval_ms,
            ),
        )

    # -- lifecycle --------------------------------------------------------

    async def _handshake(self) -> None:
        await self._transport.connect()
        await self._emit(
            MessageType.CLIENT_HELLO,
            ClientHelloPayload(
                role=ClientRole.SIMULATOR,
                client_name=CLIENT_NAME,
                client_version=CLIENT_VERSION,
                protocol_version=PROTOCOL_VERSION,
                capabilities=["task_event", "adaptation_acknowledgement", "heartbeat"],
            ),
        )
        await self._pump_inbound()

    async def _session_start(self) -> None:
        task = self._config.task
        await self._emit(
            MessageType.SESSION_START,
            SessionStartPayload(
                participant_id=self._config.participant_id,
                task_id=self._config.task_id,
                started_at_utc=self._clock.utc_now(),
                blocks=task.blocks,
                trials_per_block=task.trials_per_block,
                difficulty_level=self._runtime.difficulty_level,
                configuration={
                    "seed": self._config.seed,
                    "speed": self._config.speed,
                    "mode": self._config.mode.value,
                    "stimulus_duration_ms": task.stimulus_duration_ms,
                    "response_timeout_ms": task.response_timeout_ms,
                    "inter_trial_interval_ms": task.inter_trial_interval_ms,
                    "synthetic_error_rate": task.synthetic_error_rate,
                    "synthetic_timeout_rate": task.synthetic_timeout_rate,
                    "data_source": DataSource.SYNTHETIC.value,
                    "synthetic_label": SYNTHETIC_LABEL,
                },
            ),
        )
        self._runtime.state = TaskState.LOADED
        await self._emit_task_event(
            TaskEventDetail(
                event_type=EventType.TASK_LOADED,
                task_id=self._config.task_id,
                difficulty_level=self._runtime.difficulty_level,
                task_elapsed_ms=0.0,
            )
        )
        self._runtime.state = TaskState.RUNNING
        await self._emit_task_event(
            TaskEventDetail(
                event_type=EventType.TASK_STARTED,
                task_id=self._config.task_id,
                difficulty_level=self._runtime.difficulty_level,
                task_elapsed_ms=0.0,
            )
        )
        await self._emit_state()

    async def _session_end(self, *, completed: bool, reason: str) -> None:
        try:
            await self._emit(
                MessageType.SESSION_END,
                SessionEndPayload(
                    ended_at_utc=self._clock.utc_now(),
                    completed=completed,
                    reason=reason,
                ),
            )
        except Exception:  # pragma: no cover - transport already gone
            pass

    async def _emit_abort(self, reason: str) -> None:
        self._runtime.state = TaskState.ABORTED
        try:
            await self._emit_task_event(
                TaskEventDetail(
                    event_type=EventType.TASK_ABORTED,
                    task_id=self._config.task_id,
                    block_id=self._runtime.block_id,
                    trial_id=self._runtime.trial_id,
                    difficulty_level=self._runtime.difficulty_level,
                    task_elapsed_ms=self._simulated_elapsed_ms,
                )
            )
            await self._emit_state()
        except Exception:  # pragma: no cover - transport already gone
            pass
        self._result.completed = False
        del reason

    # -- trial execution --------------------------------------------------

    async def _run_blocks(self) -> None:
        for block_id in range(self._config.task.blocks):
            self._runtime.block_id = block_id
            self._runtime.trial_id = None
            await self._emit_task_event(
                TaskEventDetail(
                    event_type=EventType.BLOCK_STARTED,
                    task_id=self._config.task_id,
                    block_id=block_id,
                    difficulty_level=self._runtime.difficulty_level,
                    task_elapsed_ms=self._simulated_elapsed_ms,
                )
            )
            for trial in self._plan.for_block(block_id):
                await self._run_trial(trial)
            await self._emit_task_event(
                TaskEventDetail(
                    event_type=EventType.BLOCK_COMPLETED,
                    task_id=self._config.task_id,
                    block_id=block_id,
                    difficulty_level=self._runtime.difficulty_level,
                    task_elapsed_ms=self._simulated_elapsed_ms,
                )
            )

        self._runtime.state = TaskState.COMPLETED
        await self._emit_task_event(
            TaskEventDetail(
                event_type=EventType.TASK_COMPLETED,
                task_id=self._config.task_id,
                difficulty_level=self._runtime.difficulty_level,
                task_elapsed_ms=self._simulated_elapsed_ms,
            )
        )
        await self._emit_state()

    async def _run_trial(self, trial: PlannedTrial) -> None:
        task = self._config.task
        self._runtime.trial_id = trial.trial_id
        trial_start_ms = self._simulated_elapsed_ms

        await self._handle_scenario(trial)
        await self._pump_inbound()

        await self._emit_task_event(
            TaskEventDetail(
                event_type=EventType.TRIAL_STARTED,
                task_id=self._config.task_id,
                block_id=trial.block_id,
                trial_id=trial.trial_id,
                difficulty_level=self._runtime.difficulty_level,
                task_elapsed_ms=self._simulated_elapsed_ms,
                trial_elapsed_ms=0.0,
            )
        )

        await self._advance(self._runtime.stimulus_interval_ms)
        await self._emit_task_event(
            TaskEventDetail(
                event_type=EventType.STIMULUS_PRESENTED,
                task_id=self._config.task_id,
                block_id=trial.block_id,
                trial_id=trial.trial_id,
                stimulus_id=trial.stimulus_id,
                stimulus_category=trial.stimulus_category,
                expected_response=trial.expected_response,
                difficulty_level=self._runtime.difficulty_level,
                task_elapsed_ms=self._simulated_elapsed_ms,
                trial_elapsed_ms=self._simulated_elapsed_ms - trial_start_ms,
            )
        )
        stimulus_onset_ms = self._simulated_elapsed_ms

        if trial.timed_out:
            await self._advance(task.response_timeout_ms)
            await self._emit_task_event(
                TaskEventDetail(
                    event_type=EventType.RESPONSE_TIMEOUT,
                    task_id=self._config.task_id,
                    block_id=trial.block_id,
                    trial_id=trial.trial_id,
                    stimulus_id=trial.stimulus_id,
                    stimulus_category=trial.stimulus_category,
                    expected_response=trial.expected_response,
                    response_outcome=ResponseOutcome.TIMEOUT,
                    difficulty_level=self._runtime.difficulty_level,
                    task_elapsed_ms=self._simulated_elapsed_ms,
                    trial_elapsed_ms=self._simulated_elapsed_ms - trial_start_ms,
                )
            )
            self._result.timeout_count += 1
        else:
            assert trial.reaction_time_ms is not None
            await self._advance(trial.reaction_time_ms)
            await self._emit_task_event(
                TaskEventDetail(
                    event_type=EventType.RESPONSE_REGISTERED,
                    task_id=self._config.task_id,
                    block_id=trial.block_id,
                    trial_id=trial.trial_id,
                    stimulus_id=trial.stimulus_id,
                    stimulus_category=trial.stimulus_category,
                    expected_response=trial.expected_response,
                    observed_response=trial.observed_response,
                    response_correct=trial.response_correct,
                    response_outcome=trial.response_outcome,
                    reaction_time_ms=self._simulated_elapsed_ms - stimulus_onset_ms,
                    difficulty_level=self._runtime.difficulty_level,
                    task_elapsed_ms=self._simulated_elapsed_ms,
                    trial_elapsed_ms=self._simulated_elapsed_ms - trial_start_ms,
                )
            )
            self._result.synthetic_response_count += 1
            if trial.response_outcome is ResponseOutcome.CORRECT:
                self._result.correct_response_count += 1
            else:
                self._result.incorrect_response_count += 1

        await self._emit_task_event(
            TaskEventDetail(
                event_type=EventType.TRIAL_COMPLETED,
                task_id=self._config.task_id,
                block_id=trial.block_id,
                trial_id=trial.trial_id,
                stimulus_id=trial.stimulus_id,
                stimulus_category=trial.stimulus_category,
                expected_response=trial.expected_response,
                observed_response=trial.observed_response,
                response_correct=trial.response_correct,
                response_outcome=trial.response_outcome,
                reaction_time_ms=trial.reaction_time_ms,
                difficulty_level=self._runtime.difficulty_level,
                task_elapsed_ms=self._simulated_elapsed_ms,
                trial_elapsed_ms=self._simulated_elapsed_ms - trial_start_ms,
            )
        )

    async def _handle_scenario(self, trial: PlannedTrial) -> None:
        scenario = self._scenarios.get((trial.block_id, trial.trial_id))
        if scenario is None:
            return
        if scenario.kind is ScenarioKind.DISCONNECT:
            await self._transport.close()
            raise _SimulatedDisconnect
        if scenario.kind is ScenarioKind.ABORT:
            await self._emit_abort("scripted_abort")
            raise _SimulatedDisconnect
        # PAUSE
        self._runtime.state = TaskState.PAUSED
        await self._emit_task_event(
            TaskEventDetail(
                event_type=EventType.TASK_PAUSED,
                task_id=self._config.task_id,
                block_id=trial.block_id,
                trial_id=trial.trial_id,
                difficulty_level=self._runtime.difficulty_level,
                task_elapsed_ms=self._simulated_elapsed_ms,
            )
        )
        await self._emit_state()
        await self._advance(scenario.duration_ms)
        self._runtime.state = TaskState.RUNNING
        await self._emit_task_event(
            TaskEventDetail(
                event_type=EventType.TASK_RESUMED,
                task_id=self._config.task_id,
                block_id=trial.block_id,
                trial_id=trial.trial_id,
                difficulty_level=self._runtime.difficulty_level,
                task_elapsed_ms=self._simulated_elapsed_ms,
            )
        )
        await self._emit_state()

    # -- pacing and inbound -----------------------------------------------

    async def _advance(self, simulated_ms: float) -> None:
        """Advance simulated time, sleeping only when the mode requires it.

        Simulated elapsed time advances identically in all three modes.
        Only the real sleeping differs, which is why an accelerated run
        and a real-time run of the same seed produce the same events with
        the same ``task_elapsed_ms`` values.
        """
        self._simulated_elapsed_ms += simulated_ms
        if self._config.mode is SimulatorSpeed.IMMEDIATE:
            return
        await self._sleep(simulated_ms / 1000.0 / self._config.speed)

    async def _pump_inbound(self) -> None:
        """Handle any inbound messages that are already waiting."""
        while True:
            message = await self._transport.receive(timeout=0.0)
            if message is None:
                return
            await self._handle_inbound(message)

    async def _handle_inbound(self, message: DecodedMessage) -> None:
        if message.message_type is not MessageType.ADAPTATION_COMMAND:
            return
        payload = message.payload
        if not isinstance(payload, AdaptationCommandPayload):  # pragma: no cover
            return
        self._result.adaptation_commands_received += 1
        acknowledgement = self._runtime.apply_command(
            payload,
            session_id=self._session_id,
            now_utc=self._clock.utc_now(),
        )
        await self._emit_adaptation_acknowledgement(
            acknowledgement, correlation_id=message.message_id
        )

    async def _emit_adaptation_acknowledgement(
        self,
        acknowledgement: AdaptationAcknowledgementPayload,
        *,
        correlation_id: str,
    ) -> None:
        await self._emit(
            MessageType.ADAPTATION_ACKNOWLEDGEMENT,
            acknowledgement,
            correlation_id=correlation_id,
        )

    def now_utc(self) -> datetime:
        """The clock reading this simulator would stamp right now."""
        return self._clock.utc_now()


class _SimulatedDisconnect(Exception):
    """Internal signal that a scripted disconnect scenario fired."""
