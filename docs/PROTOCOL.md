# EngageVR Real-Time Protocol

**Protocol version: `1.0`**
Accepted major versions: `1`

One protocol definition is shared by the FastAPI backend, the Python task
simulator, the session replay player, and the Unity desktop client. There is
no second, client-specific message format.

## Scope

This protocol carries **task telemetry, session lifecycle, adaptation command
transport, and connection diagnostics**.

It carries no engagement estimate, no cognitive-load estimate, no behavioural
or physiological measurement, no model prediction, and no image data. Those
live in separate schemas with a separate storage path. The separation is
enforced, not merely documented: every payload model sets
`extra="forbid"`, so a field that is not part of the declared contract is
rejected rather than silently carried.

## Versioning

Version strings are `MAJOR.MINOR`.

| Case | Behaviour |
|------|-----------|
| Same major, same or lower minor | Accepted |
| Same major, higher minor | Accepted; unknown fields are rejected by the closed models, so a minor bump that adds fields requires the receiver to be updated before it can parse them |
| Different major | **Rejected** with `unsupported_protocol_version`, fatal |
| Malformed (`"1"`, `"one.two"`, `""`) | **Rejected** with `unsupported_protocol_version`, fatal |

Source of truth: `src/engagevr/protocol/version.py` and, on the C# side,
`ProtocolVersion` in `unity/EngageVR/Assets/Scripts/Protocol/ProtocolEnvelope.cs`.

## Envelope

Every message is one envelope wrapping one payload.

| Field | Type | Meaning |
|-------|------|---------|
| `protocol_version` | string | `"1.0"` |
| `message_id` | string | Unique per message. Duplicates are detected. |
| `message_type` | string | One of the 14 types below. |
| `session_id` | string | Session this message belongs to. |
| `source` | string | Originating program. |
| `sequence_number` | integer ≥ 0 | Per-source counter within a session. |
| `sent_at_utc` | ISO-8601 datetime | Sender's wall clock. **Must be timezone-aware.** |
| `sent_at_monotonic_seconds` | number | Sender's own monotonic clock. |
| `payload` | object | Body, validated against the model for `message_type`. |
| `correlation_id` | string or null | `message_id` this message answers. |
| `provenance` | object | Data source and permanent SYNTHETIC label. |
| `replay` | object or null | Present only on replayed messages. |

### `provenance`

```json
{"data_source": "synthetic", "synthetic_label": "SYNTHETIC", "producer": "engagevr.task.simulator"}
```

`data_source` is `synthetic`, `public_dataset`, or `live`. A `synthetic`
message **must** carry `synthetic_label: "SYNTHETIC"`; a non-synthetic
message **must** carry `synthetic_label: null`. Both rules are enforced by
the schema, so a simulated message cannot be laundered into looking live.

### `replay`

Added by the replay player and never present on a live message:

```json
{
  "replay_label": "REPLAY",
  "source_session_id": "demo-session",
  "replay_session_id": "replay-of-demo",
  "replay_index": 42,
  "replay_speed": 5.0,
  "replayed_at_utc": "2026-01-01T12:00:00Z",
  "original_arrival_index": 42
}
```

Replay is **additive**. The replayed envelope keeps its original `source`,
`sequence_number`, `message_id`, timestamps, and provenance. A replayed
synthetic message therefore carries `SYNTHETIC` *and* `REPLAY`.

## Clock model

Three timestamps mean three different things and are never conflated:

- `sent_at_utc` — the **sender's** wall clock. Recorded, not trusted.
- `sent_at_monotonic_seconds` — the **sender's own** monotonic clock. Its
  origin is arbitrary; only differences within that one sender are
  meaningful. The backend never translates it onto its own timeline.
- `ingestion.server_received_at_utc` — the **receiver's** wall clock, added
  at ingestion.

**Nothing in this protocol synchronizes clocks.** Transport delay is computed
only when sender and receiver share one clock (in-process). Across processes
the field is left `null` with a stated reason, because subtracting two
independent wall clocks measures clock offset, not delay.

Clock offset is estimated only from heartbeat round trips, and always with an
explicit uncertainty. See "Heartbeat flow" below.

## Message types

| Type | Direction | Purpose |
|------|-----------|---------|
| `client_hello` | client → backend | Opens the handshake; declares role and version. |
| `server_hello` | backend → client | Accepts or rejects; assigns a client id. |
| `session_start` | client → backend | Declares a task session and its configuration. |
| `session_end` | client → backend | Declares the end, with `completed` true/false. |
| `task_event` | client → backend | One task event (see below). |
| `task_state` | client → backend | Coarse lifecycle state snapshot. |
| `telemetry` | client → backend | Software/runtime metrics only. |
| `adaptation_command` | backend → client | A manually issued command. |
| `adaptation_acknowledgement` | client → backend | Accepted or rejected, with a reason. |
| `heartbeat` | client → backend | Liveness probe. |
| `heartbeat_acknowledgement` | backend → client | Echo plus both server timestamps. |
| `replay_control` | either | Start/pause/resume/step/stop a replay. |
| `acknowledgement` | backend → client | Per-message confirmation. |
| `protocol_error` | backend → client | Typed rejection. |

### Sources

`python_simulator`, `unity_client`, `backend`, `replay`, `test_fixture`.

### Roles

`simulator`, `unity` (task clients — may send task events and receive
commands), `observer` (read-only; may send only `client_hello`, `heartbeat`,
`replay_control`), `replay`.

## Task events

A task event is a **software telemetry record**. Accuracy, reaction time, and
timeout counts describe what the task program observed. They are not
engagement, attention, cognitive-load, or fatigue measurements, and the task
has not been experimentally designed, piloted, or approved.

Vocabulary (13 names, all members of `EventType`):

`task_loaded`, `task_started`, `block_started`, `trial_started`,
`stimulus_presented`, `response_registered`, `response_timeout`,
`trial_completed`, `block_completed`, `task_paused`, `task_resumed`,
`task_completed`, `task_aborted`.

Payload (`TaskEventDetail`, every field but `event_type` optional):

`task_id`, `block_id`, `trial_id`, `stimulus_id`, `stimulus_category`,
`expected_response`, `observed_response`, `response_correct`,
`response_outcome`, `reaction_time_ms`, `difficulty_level`,
`task_elapsed_ms`, `trial_elapsed_ms`.

**Missing-data rules, enforced by the schema:**

- A missing response leaves `observed_response`, `reaction_time_ms`, and
  `response_correct` as `null`. Never `""`, never `0`, never `false`.
- `response_outcome: "timeout"` is distinct from `"incorrect"`. A timeout
  has no correctness, so `response_correct` must stay `null`.
- A `response_timeout` event may not carry an `observed_response` or a
  `reaction_time_ms` at all.
- `reaction_time_ms` may not be negative. `0` is permitted, because it is a
  real (if implausible) value; "no response" is expressed as `null`.

Identifiers are repeated on every trial-scoped event so events can later be
joined into synchronized feature windows without re-deriving structure from
ordering alone.

## Handshake flow

```
client                                backend
  |  client_hello (role, versions)  ->   |
  |                                      |  validate: version, session, role/source
  |  <-  server_hello (client id,        |
  |      heartbeat interval, max bytes)  |
```

Rejections, all fatal: `handshake_required` (first message was not a
`client_hello`), `handshake_rejected` (payload/envelope version mismatch, or
a role claiming an unrelated source), `session_mismatch`,
`unsupported_protocol_version`.

## Acknowledgement flow

```
client  --  task_event  ->  backend
        <-  acknowledgement  --
```

The acknowledgement reports `stored` and `dropped`, so a client can tell the
difference between "recorded" and "discarded under backpressure". A message
can never be both.

On rejection the backend replies `protocol_error` instead, carrying the
machine-readable `error_code` and the rejection reason verbatim.

## Heartbeat flow

```
client  --  heartbeat (heartbeat_id, client_monotonic_seconds)  ->  backend
        <-  heartbeat_acknowledgement (echo + server_received + server_sent)  --
```

The client's monotonic reading is echoed **verbatim**. From `t0`/`t3` on the
client's own clock and `t1`/`t2` from the server:

```
rtt    = (t3 - t0) - (t2 - t1)
offset = ((t1 - t0) + (t2 - t3)) / 2      # an ESTIMATE
```

`rtt` is trustworthy — it is a single-clock difference. `offset` is reported
with `offset_uncertainty_seconds = rtt / 2`, valid only under an
**unverified** symmetric-delay assumption, which is recorded on the estimate
itself rather than left in prose.

A client that sends nothing for `server.connection_timeout_seconds` receives
a fatal `protocol_error` and is disconnected with reason `timeout`.

## Sequence and ordering semantics

Sequence numbers are per `(session, source)`, start at 0, must not repeat,
and must not decrease.

Two orders are recorded separately and never merged:

- **Arrival order** — what the receiver actually saw. This is the order of
  lines in `events.jsonl`. It is **never re-sorted**.
- **Source sequence order** — each source's own `sequence_number`.

Discrepancies are recorded as typed anomalies on the affected message, not
repaired: `duplicate_message_id`, `duplicate_sequence_number`,
`sequence_reversal`, `missing_sequence_range`, `excessive_sequence_gap`,
`future_timestamp`, `excessive_transport_delay`.

Duplicate `message_id` detection has a finite horizon: the tracker retains
the most recent 4096 ids per source so memory stays bounded. A duplicate
arriving after more than 4096 intervening messages from the same source will
not be detected.

## Error codes

`invalid_json`, `message_too_large`, `unsupported_protocol_version`,
`unknown_message_type`, `invalid_envelope`, `invalid_payload`,
`duplicate_message_id`, `duplicate_sequence_number`, `sequence_reversal`,
`session_mismatch`, `handshake_required`, `handshake_rejected`,
`role_not_permitted`, `expired_command`, `queue_full`, `internal_error`.

## Adaptation transport vs adaptation policy

Milestone 4 implements **transport only**.

- Commands are issued **manually** (`POST /sessions/{id}/commands`) or by a
  test script. Nothing derives a command from task performance.
- There is no policy, no cooldown, no hysteresis, and no personalization.
- No field or message asserts that applying a command improves engagement or
  any other outcome.
- Implemented commands are visually harmless: `set_difficulty`,
  `set_stimulus_interval`, `pause_task`, `resume_task`.
- A repeated `command_id` is acknowledged again with `duplicate: true` and is
  **not** re-applied, so a retransmission cannot double-step the difficulty.
- Expired, malformed, and session-mismatched commands are rejected with a
  stated reason.

## Example messages

**Task event — a registered response**

```json
{
  "protocol_version": "1.0",
  "message_id": "fixture-task-event-response-registered",
  "message_type": "task_event",
  "session_id": "fixture-session",
  "source": "unity_client",
  "sequence_number": 3,
  "sent_at_utc": "2026-01-01T12:00:00Z",
  "sent_at_monotonic_seconds": 1003.0,
  "payload": {
    "event": {
      "event_type": "response_registered",
      "task_id": "reaction_task_v1",
      "block_id": 0,
      "trial_id": 3,
      "stimulus_id": "square-b0t3",
      "stimulus_category": "square",
      "expected_response": "j",
      "observed_response": "j",
      "response_correct": true,
      "response_outcome": "correct",
      "reaction_time_ms": 412.5,
      "difficulty_level": 1,
      "task_elapsed_ms": 4612.5,
      "trial_elapsed_ms": 912.5
    }
  },
  "correlation_id": null,
  "provenance": {
    "data_source": "synthetic",
    "synthetic_label": "SYNTHETIC",
    "producer": "engagevr.protocol.fixtures"
  },
  "replay": null
}
```

**Task event — a timeout.** Note the nulls: no response, no reaction time,
no correctness.

```json
{
  "event": {
    "event_type": "response_timeout",
    "expected_response": "k",
    "observed_response": null,
    "response_correct": null,
    "response_outcome": "timeout",
    "reaction_time_ms": null
  }
}
```

**Adaptation command**

```json
{
  "command_id": "cmd-0001",
  "command": "set_difficulty",
  "value": 3,
  "reason": "manual operator request during a transport test",
  "issued_at_utc": "2026-01-01T12:00:00Z",
  "expires_at_utc": null,
  "target_role": "unity",
  "target_client_id": null,
  "is_manual": true
}
```

Complete, checked-in examples of every message type live in
`protocol/fixtures/valid/`.

## JSON Schema and contract fixtures

| Artefact | Path |
|----------|------|
| JSON Schema for version 1 | `protocol/engagevr-protocol-v1.schema.json` |
| Valid fixtures (19) | `protocol/fixtures/valid/` |
| Invalid fixtures (12), each with its expected error code | `protocol/fixtures/invalid/` |
| Index | `protocol/fixtures/index.json` |

Regenerate with:

```bash
uv run python scripts/generate_protocol_artifacts.py
```

The fixtures are the **shared contract**: the Python test suite
(`tests/unit/test_protocol.py`) and the Unity EditMode tests
(`unity/EngageVR/Assets/Tests/EditMode/ProtocolContractTests.cs`) parse the
same files. A field rename on either side fails the other side's tests. A
test also asserts the checked-in schema matches the Pydantic models, so the
artefacts cannot silently drift.

## Unity JSON restrictions

Unity's built-in `JsonUtility` is **not** used, for a correctness reason:

- It cannot represent `null`. A null string serializes as `""` and a nullable
  number as `0`. That would turn every missing response into a
  zero-millisecond response — exactly what the schema forbids.
- It cannot serialize dictionaries or top-level arrays.
- It cannot distinguish an absent field from a default-valued one.

Instead, `Assets/Scripts/Protocol/Json.cs` is a small dependency-free JSON
reader/writer in which `null` is a first-class kind. It also refuses to
serialize `NaN` or `Infinity` rather than substituting a placeholder, since
JSON cannot represent them and a substituted value would be fabricated data.

Fixtures are checked to contain no `NaN`, no `Infinity`, and no construct the
C# reader cannot represent.

## Security assumptions

Localhost only. **No authentication, no authorization, no transport
encryption, no rate limiting.** Binding to a non-loopback address requires
`server.allow_public_bind` to be set explicitly. Anyone who can reach the
port can read and inject session data.
