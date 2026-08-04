# Python Task Simulator

The simulator runs the same protocol conversation the Unity desktop task
runs, with **no Unity, no webcam, no model asset, no display server**, and —
with an offline output directory — **no network**.

## Everything it produces is SYNTHETIC

Responses, reaction times, and timeouts come from a seeded random number
generator. No person performs this task. Every message the simulator emits
carries `data_source: "synthetic"` and `synthetic_label: "SYNTHETIC"` in its
provenance, permanently, and nothing downstream clears those markers.

> Task accuracy, reaction time, and timeout counts are **software
> telemetry** from fabricated responses. They are not engagement, attention,
> cognitive-load, or fatigue measurements, they are not participant data, and
> they are not experimental evidence.

The reaction-time draw uses a lognormal shape only because it is positive and
right-skewed, so the fabricated values do not look like a symmetric artefact.
**It is not a model of human reaction times** and no parameter was fitted to
any data.

## Commands

Offline — write a session recording directly, no server:

```bash
uv run python -m engagevr task-sim \
  --seed 42 \
  --blocks 2 \
  --trials-per-block 10 \
  --speed 10 \
  --output artifacts/task-session
```

Over WebSocket — send to a running backend:

```bash
uv run python -m engagevr task-sim \
  --seed 42 \
  --blocks 2 \
  --trials-per-block 10 \
  --connect ws://127.0.0.1:8000/ws/v1/sessions/demo-session
```

### Options

| Option | Default | Meaning |
|--------|---------|---------|
| `--seed` | 42 | Seeds the trial plan. Same seed, same plan. |
| `--blocks` | from config | Number of blocks. |
| `--trials-per-block` | from config | Trials in each block. |
| `--speed` | 0 | Time multiplier. `0` = immediate (no sleeping), `1` = real time, `>1` = accelerated. |
| `--session-id` | generated | Pseudonymous session id. Validated for path safety. |
| `--participant-id` | `synthetic_participant` | Pseudonymous label for a participant who does not exist. |
| `--output` | — | Session-root directory for an offline run. |
| `--connect` | — | Backend WebSocket URL. Mutually exclusive with `--output`. |

### Printed output

Session id, protocol version, blocks, trials, emitted events, task events,
synthetic responses (correct / incorrect), timeouts, adaptation commands
received, completion status, the output path or WebSocket destination, the
data source and synthetic label, and a permanent SYNTHETIC disclaimer.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Transport failure (e.g. the backend is not reachable) |
| 2 | Invalid configuration (negative speed, unsafe session id, invalid task geometry, `--output` and `--connect` together) |
| 130 | Interrupted (Ctrl+C) |

## Determinism

The trial plan is generated from a single `random.Random` seeded once. Draw
order is fixed and does not depend on wall-clock time, dictionary iteration
order, or the pacing mode. The same seed and configuration produce an
identical plan on any machine.

The **event content is identical in all three pacing modes**; only the real
sleeping differs. That is what makes an accelerated run a valid stand-in for
a real-time run in tests, and it is asserted by a test.

The clock and the sleep function are both injected. Nothing calls
`time.monotonic` or `random` at module level, so a test can run a whole
session with a `ManualClock` and get identical output every time.

## Trial structure

Per block: `block_started` … per trial … `block_completed`.
Per trial: `trial_started` → `stimulus_presented` → (`response_registered`
**or** `response_timeout`) → `trial_completed`.

Around the whole run: `client_hello`, `session_start`, `task_loaded`,
`task_started`, …, `task_completed`, `session_end`.

Stimuli are three abstract shapes (`square`, `circle`, `triangle`) with one
response key each (`j`, `k`, `l`). Deliberately abstract: no semantic,
emotional, or clinical content.

## Missing responses

A timeout emits `response_timeout` with `observed_response`,
`reaction_time_ms`, and `response_correct` all `null`, and
`response_outcome: "timeout"`. It is never recorded as a zero-millisecond or
incorrect response.

A session in which **every** trial times out still completes and still emits
`session_end` with `completed: true`. Missing responses do not truncate a
recording.

## Scripted scenarios

`Scenario(kind=..., block_id=..., trial_id=...)` injects a deviation at a
chosen trial, so the backend's handling of these cases is tested
deterministically rather than hoped for:

| Kind | Effect |
|------|--------|
| `pause` | Emits `task_paused`, advances simulated time, emits `task_resumed`. |
| `disconnect` | Closes the transport abruptly. **No `session_end`** — the failure mode being modelled does not get to send one; recovery is the reader's job, not a fabricated tidy ending. |
| `abort` | Emits `task_aborted`, then stops. |

## Cancellation

On `asyncio.CancelledError` the simulator emits `task_aborted` and a
`session_end` marked `completed: false`, then re-raises. A cancelled session
is therefore still a **complete recording of an incomplete run**, never a
recording that simply stops mid-sentence. Sequence numbers stay contiguous
through cancellation.

`Ctrl+C` during a CLI run triggers exactly this path and exits 130.

## Adaptation commands

The simulator polls its transport between trials. An arriving
`adaptation_command` is applied through the same `TaskRuntimeState` logic the
Unity client uses, and answered with an `adaptation_acknowledgement`
correlated to the command's `message_id`.

Accept/reject rules (shared with Unity):

- `pause_task` requires state `running`; `resume_task` requires `paused`.
- An expired command is rejected with a stated reason.
- A repeated `command_id` is acknowledged with `duplicate: true` and is **not**
  re-applied.

The simulator **never generates a command on its own.** With no command sent
to it, a run emits zero `adaptation_command` and zero
`adaptation_acknowledgement` messages — asserted by a test.

## Transports

| Transport | Use |
|-----------|-----|
| `InProcessTransport` | Tests and the in-process broker. No sockets. |
| `JsonlFileTransport` | Offline runs; writes a session recording directly. |
| `WebSocketTransport` | A real client connection to the backend. |

All three are async context managers with the same interface, so the
simulator is written once and runs in every mode.

## Configuration

`configs/defaults.yaml`, `task:` section — blocks, trials per block, stimulus
duration, response timeout, inter-trial interval, default difficulty, and the
three synthetic-response parameters. Validation rejects a probability mass
above 1.0 (no room left for correct responses) and a response timeout shorter
than the stimulus duration (a trial that times out while its stimulus is
still on screen).
