# Session Recording Format

## Directory layout

```
artifacts/sessions/<session-id>/
    manifest.json     written once, when the session opens
    events.jsonl      append-only, one protocol message per line
    summary.json      written atomically when the session closes
    dropped.jsonl     one line per message discarded under backpressure
```

`artifacts/` is gitignored. The default root is
`sessions.root_directory` in `configs/defaults.yaml`
(`artifacts/sessions`).

Format version: **1.0** (`SESSION_FORMAT_VERSION`).

## What is persisted

- Protocol envelopes exactly as they arrived.
- Ingestion metadata the receiver added.
- A configuration snapshot and the protocol version, in the manifest.
- Counts, timestamps, and completion status, in the summary.

## What is never persisted

Raw webcam frames. Encoded video. Image data of any kind. MediaPipe objects.
Landmark arrays. Engagement estimates. Cognitive-load estimates. Model
predictions or confidences. Heart-rate values. Secrets, tokens, or
credentials. Names, emails, or any real-world identity.

None of these is *representable*, not merely omitted: the protocol payload
models are closed (`extra="forbid"`), and the store writes only what arrived
through that wire format. A test (`TestPrivacyInvariants`) walks a real
recording and asserts none of these tokens appears in any key or value.

Participants are identified by a **pseudonymous** `participant_id` only.
`SessionStartPayload` has no field that could hold a name, email, or date of
birth.

## `manifest.json`

```json
{
  "session_format_version": "1.0",
  "protocol_version": "1.0",
  "session_id": "demo-session",
  "created_at_utc": "2026-08-02T17:16:27.696176Z",
  "engagevr_version": "0.1.0",
  "configuration": { "...": "settings this session ran under" },
  "disclaimer": "This recording contains task and transport telemetry only. ..."
}
```

## `events.jsonl`

One JSON object per line, append-only. Each line is:

```json
{"envelope": { ... }, "ingestion": { ... }}
```

`envelope` is the protocol message verbatim (see `docs/PROTOCOL.md`).

`ingestion` is what the **receiver** observed, kept strictly separate from
what the sender said:

| Field | Meaning |
|-------|---------|
| `arrival_index` | Position in receiver arrival order. Authoritative for the file. |
| `server_received_at_utc` | Receiver's wall clock at ingestion. |
| `server_monotonic_seconds` | Receiver's own monotonic clock. |
| `transport` | `websocket`, `in_process`, `replay`, or `file`. |
| `client_id`, `client_role` | Which connection it arrived on. |
| `anomalies` | Ordering irregularities detected. **Never repaired.** |
| `anomaly_detail` | Human-readable explanation. |
| `expected_sequence_number` | What the receiver expected next from this source. |
| `apparent_transport_delay_seconds` | Populated **only** for same-process messages. |
| `delay_unavailable_reason` | Why the delay is null across processes. |

### Ordering

Lines appear in **arrival order** and are never re-sorted by sequence number.
Both orders are recoverable from every line: `ingestion.arrival_index` and
`envelope.sequence_number`. A recording that contains a genuine sequence
reversal preserves that reversal — repairing it would destroy the evidence of
what actually happened.

### Flushing

`sessions.flush_every_events` controls how many records are buffered before
flushing to the OS. The default of `1` flushes every event.

`fsync` is **not** called per record: that would dominate the cost of a
high-rate task stream. The guarantee this store offers is *"every line the OS
accepted is readable"*, not *"every line survives a power cut"*.
`summary.json` **is** fsynced, because it is written once.

## `summary.json`

Written atomically: to a temporary file in the same directory, flushed,
fsynced, then `os.replace`d over the target. A reader therefore sees either
the previous document or the complete new one, never a half-written file.
`sessions.atomic_summary` cannot be disabled — a non-atomic write can leave a
summary that reads as valid but is wrong.

Contents: `event_count`, `message_type_counts`, `source_counts`,
`anomaly_counts`, `dropped_message_count`, `dropped_message_types`,
`first_message_sent_at_utc`, `last_message_sent_at_utc`,
`first_received_at_utc`, `last_received_at_utc`, `completed`,
`disconnect_reason`, `recovered`, `malformed_line_numbers`,
`synthetic_message_count`, `replay_message_count`, and the disclaimer.

`completed` is `true` only when a `session_end` message was actually
recorded. A truncated recording is never indistinguishable from a finished
one.

## `dropped.jsonl`

One line per message discarded by backpressure, with the message id, type,
source, sequence number, the queue that was full, and the reason. A drop is
never silent: it is counted in the summary, warned about in the session log,
and recorded here, so the gap it leaves in `events.jsonl` is *explained*
rather than merely absent.

## Crash recovery

If the process dies before `close()`, there is no `summary.json`.
`SessionStore.recover(session_id)` rebuilds one:

- every line that parses is counted;
- lines that do not are listed by **1-based** line number in
  `malformed_line_numbers` (a torn final line affects only itself, because
  JSONL records are independent);
- the result is marked `recovered: true` and is `completed: true` only if a
  `session_end` was genuinely recorded before the interruption;
- **the source recording is not modified.**

`session-inspect` and `GET /sessions/{id}/summary` recover automatically when
`sessions.recover_incomplete_sessions` is true.

## Session-identifier safety

A session id becomes a directory name, so it is validated before touching the
filesystem:

- 1–128 characters of `[A-Za-z0-9._-]`, starting with a letter or digit;
- `.`, `..`, and reserved device names rejected;
- path separators, parent references, null bytes, and leading dots rejected;
- the resolved directory is additionally checked to be inside the resolved
  root, so a symlinked root cannot be used to escape it.

Invalid ids raise `InvalidSessionIdError` and produce HTTP 400.

## Malformed-line errors

`read_jsonl` raises `JsonlFormatError` naming the file, the **1-based** line
number, what was wrong, and an excerpt of the offending text:

```
artifacts/sessions/demo/events.jsonl:5: invalid JSON (Expecting value): '{ this is not js'
```

`iter_messages` additionally re-validates every line against the protocol on
the way out, so a recording corrupted after the fact cannot be replayed as
though it were sound.

## Concurrency

**Explicitly single-process.** Two processes writing the same session
directory concurrently is not supported and is not defended against. Multi-
worker and distributed operation are out of scope for Milestone 4; see
`src/engagevr/api/connections.py`.
