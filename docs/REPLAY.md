# Session Replay

Replay re-emits a recorded session, in recorded arrival order, at a chosen
pace, over any transport.

## What replay is not

A replay is **not a new participant session**, **not live data**, and **not
new evidence**. The replay player makes that inspectable rather than merely
documented:

- Every replayed envelope keeps its original `source`, `sequence_number`,
  `message_id`, `sent_at_utc`, `sent_at_monotonic_seconds`, and `provenance`.
- A **separate** `replay` block is *added*, carrying `replay_label: "REPLAY"`,
  the source and replay session ids, the position in the replay stream, the
  speed, and the original arrival index.
- A message that was `SYNTHETIC` when recorded stays `SYNTHETIC` **and**
  gains `REPLAY`. Both labels are present simultaneously.
- A message that was recorded as `live` does not acquire a synthetic label.

**The source recording is opened read-only and is never modified.** A test
asserts the session directory is byte-identical after replaying it twice.

## Commands

Inspect a recording first:

```bash
uv run python -m engagevr session-inspect artifacts/sessions/SESSION_ID
```

Replay immediately (no sleeping), reporting counts:

```bash
uv run python -m engagevr session-replay \
  artifacts/sessions/SESSION_ID \
  --speed 0
```

Replay into a running backend under a different session id:

```bash
uv run python -m engagevr session-replay \
  artifacts/sessions/SESSION_ID \
  --connect ws://127.0.0.1:8000/ws/v1/sessions/REPLAY_SESSION_ID \
  --speed 5
```

### Options

| Option | Meaning |
|--------|---------|
| `--speed` | `0` immediate, `1` original timing, `>1` accelerated. Defaults to `replay.default_speed`. |
| `--immediate` | Equivalent to `--speed 0`. |
| `--connect` | Backend WebSocket URL. Without it, replay runs through the in-process broker and reports counts. |
| `--replay-session-id` | Session id this replay publishes under. Defaults to the source id, or the id in `--connect`. |
| `--message-type` | Include only these types. Repeatable. |
| `--source` | Include only these sources. Repeatable. |
| `--json` | Machine-readable output. |

### Exit codes

`0` success · `1` unreadable or malformed recording, or transport failure ·
`2` invalid speed or unknown filter value · `130` interrupted.

## Modes

| Mode | Speed | Behaviour |
|------|-------|-----------|
| Immediate | `0` | No sleeping at all. Deterministic and instant. This is what tests use. |
| Original | `1` | Waits the recorded gap between consecutive messages. |
| Accelerated | any other positive | Waits the recorded gap divided by the speed. |
| Step | (API only) | `ReplayPlayer(step_mode=True)`; each `step()` releases exactly one message. |

All modes emit **the same ordered messages**. Only the pacing differs.

### Which timestamp defines "the gap"

`ingestion.server_received_at_utc` — the **receiver's** clock. It is the only
timeline on which messages from different sources are comparable. Using each
sender's own `sent_at_utc` would mix independent clocks and could produce
negative gaps. A negative gap from a clock adjustment during recording is
clamped to zero rather than rewinding.

The sleep function is injectable, so a test can replay original timing with a
fake clock and assert the exact sequence of requested delays without real
time passing.

## Determinism

Replay order is the recorded arrival order, read from `events.jsonl` line by
line. It is **not** re-derived from sequence numbers: a recording containing a
genuine sequence reversal replays that reversal, not a tidied-up version of
it. Two replays of the same recording emit byte-identical envelopes apart
from the `replayed_at_utc` stamp.

## Invalid speeds

Rejected with a stated reason: negative, `NaN`, `Infinity`, and anything
above `replay.maximum_speed`. Zero is valid and means immediate.

## Filtering

`--message-type` and `--source` combine conjunctively. Filtering **removes**
messages; it never renumbers or re-times the survivors, so the gap a filter
creates stays visible in the sequence numbers rather than being papered over.

## Replaying into the backend

A replayed message keeps the `session_id` it was recorded under — rewriting it
would falsify the recording. The backend therefore matches a replayed message
on `replay.replay_session_id` instead, which is the session the replay is
deliberately publishing under. A **live** message for the wrong session is
still rejected with `session_mismatch`.

The result is a second recording in which every line carries both
`SYNTHETIC` (in provenance) and `REPLAY` (in the replay block), and whose
summary reports `replay_message_count == event_count`.

## Interrupted recordings

A recording with no `summary.json` — an interrupted run — is recovered
automatically before replay. The recovered summary is marked
`recovered: true` and `completed: false`. Pass
`recover_incomplete=False` to `read_recorded_session` to refuse instead.

A malformed line aborts the read with its **1-based** line number and an
excerpt, rather than a generic "corrupt recording".
