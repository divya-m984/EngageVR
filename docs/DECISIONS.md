# EngageVR -- Architectural Decisions

## Decision Log

### DEC-001: Python 3.12 as Target Runtime

**Date:** 2026-08-01
**Status:** Accepted

**Context:** The system Python is 3.14.6 but `.python-version` pins 3.12.
MediaPipe, OpenCV, and several ML libraries have best compatibility with 3.12.
The existing venv uses Python 3.12.13.

**Decision:** Target Python 3.12 as specified in `.python-version`. The venv
already uses 3.12.13 via uv.

**Consequence:** All dependencies must be compatible with Python 3.12.

---

### DEC-002: uv as Package Manager

**Date:** 2026-08-01
**Status:** Accepted

**Context:** The repository was initialized with `uv init`. uv provides fast
dependency resolution, lock files, and virtual environment management.

**Decision:** Use uv for all dependency management. The lock file (`uv.lock`)
ensures reproducible installs.

**Consequence:** Contributors must install uv. This is already standard for
the project owner.

---

### DEC-003: Desktop-First, Software-Only MVP

**Date:** 2026-08-01
**Status:** Accepted

**Context:** No VR headset or research-grade sensors are available. The project
must be fully functional on a standard laptop.

**Decision:** Build the entire MVP for desktop operation. Unity runs in monitor
mode. Sensors are represented as adapter interfaces. The first demo requires
only a webcam (optional) and Python.

**Consequence:** VR-specific features are deferred. All modules must work
without hardware beyond a laptop and webcam.

---

### DEC-004: Pydantic v2 for All Schemas

**Date:** 2026-08-01
**Status:** Accepted

**Context:** The system requires strict data validation at layer boundaries.

**Decision:** Use Pydantic v2 models for all inter-layer data contracts:
sessions, modality samples, predictions, adaptation events, and questionnaire
responses.

**Consequence:** Schema changes require model updates. Validation errors are
caught at ingestion, not at prediction time.

---

### DEC-005: Interpretable Baselines Before Deep Learning

**Date:** 2026-08-01
**Status:** Accepted

**Context:** The specification requires starting with interpretable models.
Deep temporal models (LSTM, GRU, TCN) require more data and are harder to
debug.

**Decision:** Implement logistic regression, random forest, and gradient
boosting first. Deep models are deferred to Milestone D (after Milestone 5
baselines are evaluated).

**Consequence:** PyTorch is not an immediate dependency.

---

### DEC-006: FastAPI WebSocket Bridge for Unity Communication

**Date:** 2026-08-01
**Status:** Accepted

**Context:** The Python backend must communicate bidirectionally with Unity
(or a Python simulator) in near-real-time.

**Decision:** Use a FastAPI WebSocket endpoint as the communication bridge.
Both Unity and the Python simulator connect as WebSocket clients.

**Consequence:** The protocol is JSON over WebSocket. Message schemas are
shared via documentation (Python Pydantic models and C# equivalents).

---

### DEC-007: Synthetic Data Permanently Labelled

**Date:** 2026-08-01
**Status:** Accepted

**Context:** Synthetic data is necessary for integration testing but must never
be confused with real experimental data.

**Decision:** Every synthetic record carries `data_source: "synthetic"`. The
dashboard displays a permanent visible label. Synthetic data is excluded from
scientific evaluation metrics.

**Consequence:** All data-generation code must set this field. Tests verify
the label is present and non-removable in the pipeline.

---

### DEC-008: YAML Configuration with Pydantic Validation

**Date:** 2026-08-01
**Status:** Accepted

**Context:** Runtime parameters (window sizes, thresholds, frame rates) must be
configurable without code changes.

**Decision:** Store configuration in YAML files under `configs/`. Load and
validate with Pydantic models at startup.

**Consequence:** Adding a new parameter requires updating both the YAML file
and the corresponding Pydantic config model.

---

### DEC-009: Monotonic Timestamps for All Modalities

**Date:** 2026-08-01
**Status:** Accepted

**Context:** Multimodal synchronization requires a common time reference.

**Decision:** Use `time.monotonic()` as the session-local clock for all
modality samples. Wall-clock time is recorded at session start for external
correlation.

**Consequence:** All capture modules must use the shared clock source. Replay
must preserve original timestamp ordering.

---

### DEC-010: Abstention as a First-Class Output

**Date:** 2026-08-01
**Status:** Accepted

**Context:** The system must distinguish between confident predictions and
situations where evidence is insufficient.

**Decision:** The prediction schema includes an `abstain` boolean and a
`reason` field. The adaptation policy must not act on abstained predictions.

**Consequence:** All downstream consumers (adaptation, dashboard, logging) must
handle abstained predictions correctly.

---

### DEC-011: Logging Module Named `_logging.py`

**Date:** 2026-08-01
**Status:** Accepted

**Context:** A module named `logging.py` inside the package could shadow the
stdlib `logging` module in certain import scenarios.

**Decision:** Name the module `_logging.py` to avoid any shadowing risk.
Public access is via `from engagevr._logging import setup_logging, get_logger`.

**Consequence:** The module name differs from what `ARCHITECTURE.md` lists as
`logging.py`. The architecture document describes the responsibility; the
actual file uses the underscore prefix for safety.

---

### DEC-012: Remove Unused mypy Test Override

**Date:** 2026-08-01
**Status:** Accepted

**Context:** The `[[tool.mypy.overrides]]` section for `module = "tests.*"`
was unused because `mypy` is invoked only on `src/`. This produced a
`pyproject.toml: note: unused section(s)` warning on every run.

**Decision:** Remove the override entirely rather than leaving a known
warning in the project. If test type-checking is added later, the override
can be reintroduced at that time.

**Consequence:** `uv run mypy src` produces clean output with no notes.

---

### DEC-013: Standard-Library CLI via `__main__.py`

**Date:** 2026-08-01
**Status:** Accepted

**Context:** The project needs a minimal CLI to generate and save a synthetic
demo session without adding new runtime dependencies.

**Decision:** Use `argparse` in `src/engagevr/__main__.py` so the package is
invocable as `python -m engagevr demo`. No click, typer, or other CLI library
is added.

**Consequence:** The CLI uses only the standard library. The `demo` subcommand
generates SYNTHETIC data and prints a summary. Adding future subcommands
requires extending the argparse parser.

---

### DEC-014: MediaPipe Tasks Vision API for Face Landmarks

**Date:** 2026-08-01
**Status:** Accepted

**Context:** Face landmark extraction requires a maintained, Python 3.12-
compatible library. MediaPipe provides a Tasks Vision API with a
FaceLandmarker model that runs in VIDEO mode.

**Decision:** Use `mediapipe>=0.10.14` with the `FaceLandmarker` Tasks API
in VIDEO mode. The legacy `mediapipe.solutions.face_mesh` API is not used.
The model asset (`face_landmarker.task`) is downloaded separately and stored
under `models/` (gitignored).

**Consequence:** The FaceLandmarker model must be downloaded before capture.
Tests use synthetic landmarks and do not require the model or internet.
MediaPipe-specific objects do not cross module boundaries.

---

### DEC-015: External Model Asset Not Committed to Git

**Date:** 2026-08-01
**Status:** Accepted

**Context:** The FaceLandmarker model file is ~4 MB and is a binary asset
from Google. Committing it to Git would bloat the repository.

**Decision:** Store the model under `models/` which is listed in `.gitignore`.
A reproducible download script (`scripts/download_models.py`) fetches it from
the official Google Cloud Storage endpoint. The script records source, license
(Apache 2.0), and SHA-256 checksum.

**Consequence:** Users must run the download script before using the `capture`
command. The CLI exits non-zero with a clear instruction when the model is
missing.

---

### DEC-016: Behavioural Outputs Are Proxies Only

**Date:** 2026-08-01
**Status:** Accepted

**Context:** Eye Aspect Ratio, blink detection, mouth movement, and head
pose are geometric measurements derived from facial landmarks. They are
associated with observable behaviours but do not directly measure
engagement, cognitive load, or emotional state.

**Decision:** All behavioural feature schemas and CLI outputs include
explicit disclaimers that these are proxies. The system does not label
facial expressions as emotions, does not infer engagement in this
milestone, and does not identify people. When landmarks are missing or
unreliable, feature values are set to `None` (unavailable) rather than
silently replaced with zero or a misleading default.

**Consequence:** Downstream consumers must treat these as input features,
not as conclusions. Unavailable values propagate as `None` through the
pipeline. The adaptation layer (future) must apply confidence and
signal-quality thresholds before acting on any derived estimates.

---

### DEC-017: Single OpenCV Distribution (opencv-contrib-python)

**Date:** 2026-08-01
**Status:** Accepted

**Context:** MediaPipe depends on `opencv-contrib-python`. Installing
`opencv-python-headless` alongside it creates a conflict because both
provide the `cv2` namespace. The capture CLI also supports an optional
preview window, which requires a desktop-capable OpenCV build.

**Decision:** Use `opencv-contrib-python` as the single OpenCV
distribution. Remove `opencv-python-headless` from `pyproject.toml`.

**Consequence:** One `cv2` package is installed. The preview window works
when a display server is available. On headless CI, `cv2.imshow` is never
called (preview is off by default), so no display server is required.

---

### DEC-018: SciPy for Classical Signal Processing

**Date:** 2026-08-02
**Status:** Accepted

**Context:** Milestone 3 requires Butterworth filter design, zero-phase
filtering, linear detrending, Welch power spectral density estimation,
and peak prominence. NumPy alone provides none of these.

**Decision:** Add `scipy>=1.15,<2` as a direct runtime dependency, and
`scipy-stubs` as a dev dependency so `mypy --strict` type-checks SciPy
calls rather than treating them as `Any`. SciPy 1.18.0 resolves for
Python 3.12; the lower bound 1.15 is the oldest release in the range
that ships cp312 wheels for the current dependency set.

No other dependency is added. PyTorch, TensorFlow, XGBoost,
scikit-learn, FastAPI, Streamlit, MLflow, and DVC remain out of scope.

**Consequence:** The dependency tree gains SciPy and its single
transitive requirement (NumPy, already present at the same version). No
duplicate or conflicting package results.

---

### DEC-019: Three Classical rPPG Methods, No Learned Method

**Date:** 2026-08-02
**Status:** Accepted

**Context:** The specification lists green-channel, CHROM, and POS as
candidate methods and requires interpretable signal processing before
deep learning.

**Decision:** Implement GREEN (Verkruysse et al., 2008), CHROM (de Haan
& Jeanne, 2013), and POS (Wang et al., 2017) from their primary
references, behind one shared typed interface. Every deviation from a
published algorithm is recorded in the function docstring and in
`docs/REFERENCES.md`. No learned method is implemented.

**Consequence:** All three methods can be run over the same window and
compared. The project makes **no claim** that any one of them is
superior; relative performance depends on illumination, motion, camera,
and subject, and nothing in this repository establishes a ranking.

---

### DEC-020: Unavailable Is a Value, Never NaN or a Clamp

**Date:** 2026-08-02
**Status:** Accepted

**Context:** A heart-rate estimate that cannot be justified must be
distinguishable from one that can. Filling with NaN, zero, the previous
value, or a clamped in-range number destroys that distinction.

**Decision:** Numeric helpers raise `RppgUnavailable` carrying a
machine-readable `UnavailableReason`; the orchestration layer converts
it into an `available=False` schema field with that reason attached.
Missing frames are recorded as invalid samples with no channel values,
never imputed. A BPM outside the configured band is never clamped into
it — the search is restricted to the band, and if no acceptable in-band
peak exists the estimate is withheld.

**Consequence:** `RppgUnavailable` lives in `src/engagevr/rppg/errors.py`,
one module beyond the filenames listed in the milestone brief. This
keeps the numeric helpers total and independently testable while
guaranteeing no NaN-filled value reaches a schema.

---

### DEC-021: rPPG Quality Uses Equal Weights Plus Hard Gates

**Date:** 2026-08-02
**Status:** Accepted

**Context:** An aggregate quality score needs a defensible combination
rule. Hand-picked weights across a dozen components would be an
arbitrary unexplained constant, which the project rules forbid.

**Decision:** `overall_quality` is the **unweighted arithmetic mean of
the components that could actually be computed** for that window.
Components that could not be computed are omitted, not imputed.
Separately, four components are **gates** — ROI availability, timestamp
monotonicity, filter viability, and window duration. A failed gate
forces `acceptable=False` regardless of the mean, because these
conditions make the estimate invalid rather than merely noisy.

Equal weighting is chosen because this repository has no validated
empirical basis for ranking the components against one another. If such
a basis is established later, the weights must be derived from it and
recorded here.

**Consequence:** The aggregation rule is stated in the schema docstring
and asserted by a test. A window with unacceptable quality returns
`unavailable` for heart rate. Poor quality is reported as poor quality,
never as low engagement.

---

### DEC-022: HRV and Inter-Beat Intervals Deferred

**Date:** 2026-08-02
**Status:** Accepted

**Context:** The specification lists HRV features (SDNN, RMSSD, pNN50)
under the physiological feature category, and the milestone brief
explicitly permits deferring them.

**Decision:** No HRV, IBI, SDNN, RMSSD, or pNN50 value is computed in
Milestone 3. Time-domain HRV requires beat-to-beat interval accuracy on
the order of milliseconds, which requires validated individual peak
detection on a waveform whose morphology is trustworthy. A spectral
pulse-rate estimate provides no per-beat timing whatsoever.

The prerequisites that must be established from primary literature
before HRV is implemented are listed in `docs/REFERENCES.md`.

**Consequence:** A test asserts that no HRV-shaped field exists on the
heart-rate schema, so the deferral cannot be silently undone.

---

### DEC-023: Datasets Are Never Downloaded Automatically

**Date:** 2026-08-02
**Status:** Accepted

**Context:** UBFC-rPPG is distributed from a university page whose terms
of use are not stated explicitly. Automatically fetching data whose
licence is unknown would commit the user to terms neither they nor this
project have read.

**Decision:** Dataset adapters contain no network code. Roots are
supplied through configuration or `--root`. `docs/DATASETS.md` records
the official source, citation, access procedure, and — for UBFC-rPPG —
that the licence **requires manual verification**, because no explicit
permitted-use statement was found on the official page. Absence of a
stated licence is recorded as absence, never as permission.

Facts that the official source does not state, such as the reference
oximeter's sampling rate, are left as `None` rather than guessed.

**Consequence:** A test asserts the adapter module contains no
networking imports. Public-dataset evaluation remains **pending** until
the data is obtained through the official channel and the pipeline is
actually run against it.

---

### DEC-024: Two Additional rPPG Modules Beyond the Listed Filenames

**Date:** 2026-08-02
**Status:** Accepted

**Context:** The Milestone 3 brief lists a specific set of module
filenames and permits adjustment for a documented architectural reason.

**Decision:** Two modules are added beyond that list:

- `src/engagevr/rppg/errors.py` — the typed failure signal described in
  DEC-020. Placing `RppgUnavailable` in any of the listed modules would
  force a circular import, since every one of them needs to raise it.
- `src/engagevr/rppg/evaluation.py` — dataset error metrics. These
  belong with the rPPG pipeline rather than in `datasets/`, which is
  responsible for reading data, not for judging estimator accuracy.

`ARCHITECTURE.md` previously listed `rppg/filtering.py`; the actual
module is `rppg/preprocessing.py`, because it covers detrending,
normalization, and resampling in addition to filtering.

**Consequence:** `docs/ARCHITECTURE.md` is updated to match the
implemented module layout.

---

### DEC-025: One Shared Versioned Protocol, Not Per-Client Formats

**Date:** 2026-08-02
**Status:** Accepted

**Context:** Milestone 4 requires that "the Python simulator and Unity use
the same versioned protocol". The tempting shortcut is a Python format plus
a hand-maintained C# mirror, which drifts silently the first time a field is
renamed.

**Decision:** A single protocol lives in `src/engagevr/protocol/`. Its JSON
Schema and a set of representative valid/invalid message fixtures are
generated from the Pydantic models and **checked in** under `protocol/`.
Both test suites — Python `tests/unit/test_protocol.py` and Unity
`Assets/Tests/EditMode/ProtocolContractTests.cs` — parse those same files.

**Consequence:** A field rename on either side fails the other side's tests.
A test also asserts the checked-in schema matches the models, so the
artefacts cannot drift from the code. Regeneration is one command:
`uv run python scripts/generate_protocol_artifacts.py`.

---

### DEC-026: Replay Adds Metadata Rather Than Rewriting Messages

**Date:** 2026-08-02
**Status:** Accepted

**Context:** A replayed message must be distinguishable from a live one. Two
approaches were available: rewrite the envelope's `source`/`session_id` to
say "replay", or add a separate metadata block.

**Decision:** Replay is **additive**. A replayed envelope keeps its original
`source`, `sequence_number`, `message_id`, timestamps, and provenance
untouched, and gains a `replay` block recording the source session, the
replay session, the position in the replay stream, and the speed.

Rewriting would have falsified the recording: the message really was
produced by that source, at that sequence number, at that time. The
distinguishing fact — that this *emission* is a replay — is a property of the
emission, not of the message, and is recorded as such.

**Consequence:** A replayed synthetic message carries `SYNTHETIC` in
provenance **and** `REPLAY` in the replay block, simultaneously. Because the
`session_id` is preserved, the backend matches a replayed message on
`replay.replay_session_id`; a *live* message for the wrong session is still
rejected with `session_mismatch`. Asserted by tests on both paths.

---

### DEC-027: Backpressure Drops Non-Critical Telemetry, Never Critical Messages

**Date:** 2026-08-02
**Status:** Accepted

**Context:** Bounded queues force a choice when full: block, drop, or fail.
Blocking indefinitely converts a slow disk into a hung process; dropping
everything makes a recording untrustworthy; unbounded queueing turns a fast
producer into an out-of-memory kill that surfaces with no diagnosis.

**Decision:** Every queue is bounded and configurable. On a full queue:

- **Critical messages** (`session_start`, `session_end`,
  `adaptation_command`, `adaptation_acknowledgement`, `protocol_error`, and
  any `task_event` carrying `task_completed`) wait up to
  `queues.operation_timeout_seconds`. If space never appears the connection
  is **failed** with a `queue_full` protocol error. They are never dropped.
- **Non-critical messages** are dropped immediately rather than blocking.
- A full **broadcast** queue never costs a stored event: observers are
  read-only monitors, so their copy is dropped and warned about while
  ingestion continues.

**Consequence:** No drop is silent. Every drop is counted per message type in
the summary, written to `dropped.jsonl` with the queue that was full and a
reason, and warned about in the session log — so a gap in `events.jsonl` is
always *explained* rather than merely absent. `task_completed` is treated as
critical specifically so that a truncated recording can never be mistaken for
a finished one.

---

### DEC-028: Arrival Order and Sequence Order Are Recorded Separately

**Date:** 2026-08-02
**Status:** Accepted

**Context:** Messages can arrive out of sequence order. The storage layer
could sort them into sequence order on write, producing a tidy file.

**Decision:** It does not. `events.jsonl` preserves **arrival order**, byte
for byte, append-only, and is never re-sorted. Each line carries both
`ingestion.arrival_index` and `envelope.sequence_number`, so either order is
recoverable. Discrepancies — duplicate ids, duplicate sequence numbers,
reversal, missing ranges, future timestamps, excessive delay — are recorded
as typed anomalies on the affected message and **not repaired**.

**Consequence:** A recording containing a genuine sequence reversal replays
that reversal. Sorting on write would have destroyed the evidence of what
actually happened, which is the one thing a recording exists to preserve.

---

### DEC-029: Clock Offset Is an Estimate With Stated Uncertainty; Delay Is Often Unavailable

**Date:** 2026-08-02
**Status:** Accepted

**Context:** It is tempting to report `server_received_at_utc - sent_at_utc`
as "transport latency". Across two machines that subtraction measures the
*clock offset* between them far more than it measures delay.

**Decision:**

- `apparent_transport_delay_seconds` is populated **only** when sender and
  receiver share one clock (in-process). Across processes it is `null`, with
  `delay_unavailable_reason` stating why.
- Clock offset is estimated only from heartbeat round trips, using the
  standard four-timestamp formula, and is always accompanied by
  `offset_uncertainty_seconds = rtt / 2` and an explicit
  `symmetric_delay_assumed = True` flag.
- The sender's monotonic clock is preserved verbatim and never translated
  onto the receiver's timeline; the two have unrelated origins.

**Consequence:** No output of this system claims that two independent
machines' clocks are synchronized. A `future_timestamp` anomaly is reported
as a *clock* observation and never causes a message to be rejected.

---

### DEC-030: Unity Uses a Hand-Written JSON Serializer, Not JsonUtility

**Date:** 2026-08-02
**Status:** Accepted

**Context:** Unity's built-in `JsonUtility` is the zero-dependency default
for JSON in Unity.

**Decision:** It is **not** used. `JsonUtility` cannot represent `null`: it
serializes a null string as `""` and a nullable number as `0`, and on
deserialization leaves absent fields at their default. The EngageVR protocol
depends on the difference between "no response" (`null`) and "a response of
0 ms". Using `JsonUtility` would silently convert every missed trial into a
zero-latency response — precisely the failure the Python schema forbids. It
also cannot serialize dictionaries or top-level arrays.

`Assets/Scripts/Protocol/Json.cs` is a small dependency-free reader/writer in
which `null` is a first-class kind. It refuses to serialize `NaN` or
`Infinity` rather than substituting a placeholder.

`com.unity.nuget.newtonsoft-json` would also have worked, but adding a
package dependency that has not been resolved or compiled in this repository
would be an unverified claim.

**Consequence:** A test asserts the fixtures contain no `NaN`/`Infinity` and
no construct the C# reader cannot represent, and a C# test asserts that
`null`, `""`, and `0` remain distinct through a round trip.

---

### DEC-031: Unity Uses System.Net.WebSockets, Not a Third-Party Package

**Date:** 2026-08-02
**Status:** Accepted

**Context:** Unity has no built-in WebSocket component, and several
third-party packages exist. The milestone brief forbids adding an unverified
third-party WebSocket package automatically.

**Decision:** `System.Net.WebSockets.ClientWebSocket` is used. It ships with
the .NET Standard 2.1 base class library that Unity's Mono and IL2CPP
backends expose, so it needs no package, no manifest entry, and no licence
review, and it works in the Editor and in Windows/macOS/Linux standalone
players.

**Compatibility rationale and limitation:** `ClientWebSocket` is *not*
supported on the WebGL player, where the browser owns the socket. EngageVR
targets a desktop player, so this is not a constraint; a WebGL build would
need a JavaScript interop bridge and is out of scope.

Send and receive run on background tasks that touch no Unity API; inbound
messages are dispatched on the main thread by `Poll()` from `Update`.

**Consequence:** No third-party dependency, no licence question, and no
package that this repository cannot verify. Recorded in
`docs/UNITY_SETUP.md`. The Unity project has **not** been compiled or run.

---

### DEC-032: The Unity Project Ships Source Only, With No Fabricated ProjectVersion

**Date:** 2026-08-02
**Status:** Accepted

**Context:** `unity/EngageVR/` was an empty directory. A "complete" Unity
project would normally include `ProjectSettings/ProjectVersion.txt` naming
an editor version.

**Decision:** No `ProjectVersion.txt` is written. This repository has no
Unity Editor installed and cannot verify which version the project opens
under; asserting one would be a claim it has not tested. Unity Hub asks on
first open. `Packages/manifest.json` and assembly definitions *are* checked
in, because those are declarations the repository can stand behind.

The demo scene is likewise **generated from an editor script** rather than
checked in as a serialized `.unity` asset, keeping the repository free of
binary-ish blobs that cannot be reviewed in a diff.

**Consequence:** `docs/UNITY_SETUP.md` states plainly that Unity compilation
and runtime validation are pending, and the Unity acceptance criteria are
recorded as **not met** in `docs/PROGRESS.md`.

---

### DEC-033: Task Telemetry Is Software Measurement, Never a Psychological Construct

**Date:** 2026-08-02
**Status:** Accepted

**Context:** A reaction task produces accuracy and reaction-time numbers that
look like the dependent variables of a cognitive experiment.

**Decision:** They are treated as **software telemetry** throughout. The
schema docstring, the wire documentation, the CLI output, the session
manifest, the session summary, and the Unity HUD all state that these are not
engagement, attention, cognitive-load, or fatigue measurements, and that the
task has not been experimentally designed, piloted, or approved.

The protocol payload models are closed (`extra="forbid"`), so an engagement
or cognitive-load value is not merely absent from a recording — it is not
*representable* in one.

**Consequence:** Tests assert that no recording, no fixture, and no Unity
message contains the tokens `engagement`, `cognitive_load`, `attention`, or
`fatigue`, and that the backend answers a run of ten consecutive timeouts
with acknowledgements rather than an adaptation command.

---

### DEC-034: Milestone 4 Implements Adaptation Transport Only

**Date:** 2026-08-02
**Status:** Accepted

**Context:** The adaptation schemas exist from Milestone 1, and the transport
needed to move a command from backend to client is in scope for Milestone 4.
Policy is Milestone 8.

**Decision:** Commands are **transported**, never **decided**. Every command
in this milestone is issued manually via `POST /sessions/{id}/commands` or by
a test script. There is no policy, no cooldown, no hysteresis, and no
personalization anywhere in the code. No field or message asserts that
applying a command improves engagement or any other outcome; `reason` is an
audit note supplied by whoever issued the command.

A repeated `command_id` is acknowledged with `duplicate: true` and is **not**
re-applied, so a retransmission cannot double-step the difficulty. The
accept/reject rules live in one place (`engagevr/task/state.py`) and are
mirrored by the Unity `AdaptationReceiver`, so both clients behave
identically.

**Consequence:** A test asserts that ten consecutive `response_timeout`
events produce zero commands, so the absence of a policy cannot be silently
undone.

---

### DEC-035: Two Modules Beyond the Milestone 4 Filename List

**Date:** 2026-08-02
**Status:** Accepted

**Context:** The Milestone 4 brief lists module filenames and permits
adjustment for a documented architectural reason.

**Decision:** Three modules exist beyond that list:

- `src/engagevr/api/broker.py` — the bounded-queue ingestion pipeline.
  Placing it in `connections.py` would mix connection identity with
  backpressure policy, which are separately configured and separately
  tested.
- `src/engagevr/api/state.py` — the lifespan-owned application state.
  Keeping it out of `app.py` lets the routes and the WebSocket handler share
  it without importing the application factory, which would be circular.
- `src/engagevr/transport.py` — the transport abstraction shared by the task
  simulator and the replay player. It belongs to neither package, and
  duplicating it in both was the alternative.

`src/engagevr/cli_milestone4.py` holds the four new CLI commands so
`__main__.py` stays a thin dispatcher and the new commands can be tested
without importing the webcam or rPPG code paths.

`src/engagevr/schemas/protocol.py` is a pure re-export of the models in
`engagevr.protocol`, so `engagevr.schemas` remains the single place to look
for every data contract without creating a second, divergent definition.

**Consequence:** `docs/ARCHITECTURE.md` is updated to match the implemented
module layout.

---

### DEC-036: httpx2 Rather Than httpx for the Test Client

**Date:** 2026-08-02
**Status:** Accepted

**Context:** The milestone brief anticipated `httpx` as the development
dependency that `fastapi.testclient` imports. Starlette 1.3.1 emits a
`StarletteDeprecationWarning` when used with `httpx` and declares
`httpx2>=2.0.0` alongside it.

**Decision:** `httpx2>=2.9,<3` is the declared development dependency.
Pinning the deprecated path would have meant importing a warning into every
test run.

`pytest-asyncio>=1.0,<2` is also added: the backend, simulator, and replay
player are async, and the alternative was to depend on the `anyio` pytest
plugin, which reaches this project only as an undeclared transitive
dependency of Starlette.

**Consequence:** Verified against Python 3.12.13 — fastapi 0.141.1,
uvicorn 0.52.1, websockets 17.0.1, httpx2 2.9.1, pytest-asyncio 1.4.0. No
Redis, PostgreSQL, MongoDB, Kafka, RabbitMQ, Celery, MLflow, DVC, Streamlit,
React, Node.js, or Docker was added. One OpenCV wheel variant remains.
