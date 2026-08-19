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

---

### DEC-037: Four Modelling Dependencies, No Second Gradient-Boosting Library

**Date:** 2026-08-04
**Status:** Accepted

**Context:** Milestone 5 needs tabular datasets, classical estimators, and
model persistence. The architecture document already anticipated pandas,
PyArrow, scikit-learn, and XGBoost.

**Decision:** Add exactly four runtime dependencies —
`scikit-learn>=1.6,<2`, `pandas>=2.2,<4`, `pyarrow>=17,<26`, and
`joblib>=1.4,<2`. **XGBoost is not added.**

`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`
implement the same histogram-based boosted-tree algorithm, handle missing
values natively, and arrive with scikit-learn. A second gradient-boosting
library would add a dependency, a build requirement, and a second
serialisation format without adding a capability.

`joblib` is declared explicitly rather than relied upon as a scikit-learn
transitive dependency, because EngageVR imports it directly for model
persistence.

**Consequence:** Resolved and verified on Python 3.12.13 — scikit-learn
1.9.0, pandas 3.0.5, pyarrow 25.0.0, joblib 1.5.3. NumPy 2.5.1 and SciPy
1.18.0 were unchanged by the resolution. `uv tree` shows no duplicate or
conflicting package and one OpenCV wheel variant. No PyTorch, TensorFlow,
LightGBM, CatBoost, Optuna, SHAP, MLflow, DVC, Streamlit, Docker, or
database was added.

None of the four ships a `py.typed` marker, so each is listed in the
existing `ignore_missing_imports` mypy override alongside mediapipe and
cv2. A second, narrower override sets `disallow_subclassing_any = false`
for `engagevr.training.models` only: the two rule baselines must subclass
scikit-learn's untyped `BaseEstimator` to be usable inside a `Pipeline`.

---

### DEC-038: A Feature Catalog Gates What May Become a Predictor

**Date:** 2026-08-04
**Status:** Accepted

**Context:** A modelling dataset accumulates columns. Without a gate,
whatever happens to be in the table becomes a model input, including
identifiers, provenance, and re-encodings of the answer.

**Decision:** Every feature is declared in a versioned, **ordered** feature
catalog carrying its canonical name, description, source modality, unit,
aggregation formula, minimum evidence, missing-value behaviour, quality
dependency, and a `permitted_predictor` flag. A feature with no entry
cannot enter a dataset; a column with no entry cannot reach a model.

Catalog order fixes dataset column order, which is part of the dataset
fingerprint, so reordering the catalog is a detectable change.

**Consequence:** `rppg_method` is catalogued for provenance but is **not**
a permitted predictor: the extraction method is a property of the pipeline
configuration, and a model using it would learn a processing artefact.
Categorical features are barred from being predictors in this milestone,
enforced by a schema validator.

The catalog is snapshotted beside every dataset as
`<stem>.feature_catalog.json`, so a stored dataset can be audited against
the contract it was built under rather than against whatever the code says
today.

---

### DEC-039: Unavailable Is Null; Missing Is Never Zero, and Quality Is Never a Value

**Date:** 2026-08-04
**Status:** Accepted

**Context:** DEC-020 established this for rPPG. The feature layer faces the
same choice at a different granularity, and a modelling table makes it
tempting to zero-fill so that estimators "just work".

**Decision:** Feature values, per-feature availability, modality
availability, modality quality, targets, and target provenance occupy
**separate columns with separate prefixes** and are never merged. A missing
measurement is null in the dataset. Non-finite values are rejected rather
than stored.

Three missing-value semantics are declared per feature:
`null_when_unavailable`, `null_when_evidence_insufficient`, and
`zero_when_no_events`. The last is reserved for counts where zero is a
genuine observation — a window in which a tracked face produced no blinks
really did contain no blinks — and is not applied to a window with no face
at all.

**Consequence:** Imputation becomes a modelling decision made inside a
training fold, never a dataset property. Where median imputation is used,
a missingness indicator accompanies it, the `avail__` and
`modality_quality__` columns stay in the matrix, and histogram gradient
boosting receives values unimputed. A quality failure is therefore marked
in three places and is never silently converted into a physiological
value.

---

### DEC-040: The Dataset Fingerprint Excludes Every Wall Clock

**Date:** 2026-08-04
**Status:** Accepted

**Context:** Milestone 5's acceptance criterion is that metrics are exactly
reproducible from the same data, configuration, and seed. That is only
checkable if "the same data" is decidable.

**Decision:** `dataset_fingerprint` is a SHA-256 over a canonical rendering
of the schema versions, the catalog version, the exact column order, the
window geometry, and every row's values in that order, with rows sorted by
`(session_id, window_index, window_id)`. Floats are rendered with `repr`,
which round-trips exactly for IEEE-754 doubles.

`created_at_utc` and every other wall-clock value are **excluded** from the
canonical content.

**Consequence:** Two equivalent deterministic builds fingerprint
identically, which is asserted by a test that builds the same metadata with
two different creation timestamps. The fingerprint changes on any change to
row content, schema, column order, feature order, target set, or window
geometry — each asserted separately — and does not change when rows are
supplied in a different order.

Run identifiers follow the same rule: `build_run_id` hashes only what
defines a run, so re-running one configuration reproduces its identifier
instead of accumulating near-duplicate directories.

---

### DEC-041: Grouped Splitting Only, With No Row-Level Fallback

**Date:** 2026-08-04
**Status:** Accepted

**Context:** Several windows come from one session and several sessions can
come from one person. `KFold`, `StratifiedKFold`, and `train_test_split`
place the same person on both sides of the boundary.

**Decision:** `engagevr.training.splits` offers grouped splitters only.
Grouping priority is `subject_id`, then `session_id`, then **refusal**.
There is no configuration flag that enables row-level splitting, and the
ungrouped scikit-learn splitters are not imported.

When a defensible split is impossible — fewer than two groups, or fewer
groups than requested folds — the splitter raises with an actionable
message rather than weakening the split.

Stratification feasibility is checked explicitly: when a class appears in
fewer distinct groups than there are folds, the run falls back to
non-stratified grouped k-fold and **records why** in the split manifest.

**Consequence:** A dataset with one participant cannot be evaluated by this
pipeline, which is correct: it has no independent held-out group.
`audit_split` independently re-checks every manifest for train/test
overlap, calibration/test overlap, calibration groups outside training, and
any session whose rows reach the test portion of more than one fold.

---

### DEC-042: Calibration Uses FrozenEstimator on Groups Disjoint From Fitting

**Date:** 2026-08-04
**Status:** Accepted

**Context:** `CalibratedClassifierCV(cv="prefit")` was deprecated in
scikit-learn 1.6 and removed in 1.8. The installed version is 1.9.0.

**Decision:** Calibration wraps the fitted pipeline in
`sklearn.frozen.FrozenEstimator`, which is the supported API and which
makes `ensemble="auto"` resolve to `False`, so the whole calibration set
fits one calibrator.

Within each outer fold, the base estimator is fitted on the *fit groups*,
the calibrator on *calibration groups* carved out of the training groups
and disjoint from them, and the outer test groups are used only to score.
`assert_calibration_disjoint` re-checks all three sets before anything is
fitted.

Isotonic calibration requires at least 50 calibration rows and at least 10
per class; below that it is skipped with a stated reason rather than
silently downgraded. Isotonic regression is non-parametric and will fit a
step function through a handful of points.

**Consequence:** scikit-learn still constructs a default CV splitter on the
way into `CalibratedClassifierCV` and warns about thin classes even though
a frozen estimator never splits. That one message is suppressed at the call
site with a comment stating why; every genuine thin-class condition is
caught by the explicit checks above it. A test asserts
`len(calibrated_classifiers_) == 1`, which is what proves no internal
cross-validation occurred.

Abstention, coverage-versus-performance, and online confidence policy are
**not** implemented here. They are Milestone 7.

---

### DEC-043: An Undefined Metric Is Unavailable, Never Zero

**Date:** 2026-08-04
**Status:** Accepted

**Context:** scikit-learn's `zero_division=0` turns an undefined precision
into `0.0`. Zero is a legitimate score, so a reader cannot distinguish "the
model scored zero" from "this was not computable".

**Decision:** Undefined metrics are `None` with a reason recorded in
`unavailable_metrics`. Per-class quantities use `zero_division=np.nan` and
are reported as null; macro means are taken over the classes for which the
quantity is *defined*, and the excluded classes are recorded. R² is null
when the true values have zero variance or there are fewer than two
samples. Log loss, Brier, and ECE are null without probabilities.

A non-finite *prediction*, by contrast, raises: a model that cannot produce
a finite prediction must fail rather than emit one.

Fold aggregation is the unweighted mean and population standard deviation
over the folds in which the metric was defined, with the valid-fold count
reported alongside. Folds are weighted equally so one large participant
cannot dominate the summary.

**Consequence:** Confusion matrices are stored as labelled data — the
vocabulary, an explicit statement that rows are true and columns predicted,
and the counts — never as a bare array. The multiclass Brier convention and
the exact ECE binning formula are stated in the schema fields that carry
the numbers, so a stored result is interpretable without the source.

---

### DEC-044: Evaluation Mode Is a Required Field, and Synthetic Can Never Be Scientific

**Date:** 2026-08-04
**Status:** Accepted

**Context:** DEC-007 requires synthetic data to be permanently labelled.
Milestone 5 introduces the first artefacts that carry *scores*, which are
exactly the artefacts most likely to be quoted out of context.

**Decision:** Every metrics document carries a required `evaluation_mode`
and a required non-empty `disclaimers` list. A `software_self_check`
document cannot set `scientific_evaluation_eligible: true` and must carry
the banner `SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION`; both are
schema-enforced.

Synthetic target observations must carry `synthetic_label: "SYNTHETIC"` and
must set `scientific_evaluation_permitted: false`, also schema-enforced.
`baseline-train --mode scientific` refuses any dataset with a synthetic
row, a prohibited target, or an unstated target source type.

**Consequence:** There is no field anywhere in the experiment schemas that
could hold a published or public-dataset score, so a synthetic number
cannot be recorded as one. Tests scan generated artefacts for
validity-claim phrases and fail if any appears.

---

### DEC-045: Measurements Are Never Automatically Promoted to Labels

**Date:** 2026-08-04
**Status:** Accepted

**Context:** With task accuracy, reaction time, difficulty level, and a
heart-rate estimate available, the shortest path to a "labelled" dataset is
to declare one of them the target. That step is a research claim, not a
data transformation.

**Decision:** `reject_automatic_derivation` refuses to derive any target
from any measurement group, with a stated reason per group. There is no
`allow` flag: producing a real label requires an external instrument and a
documented protocol.

Every target observation must state a `source_type` from a closed
vocabulary, a non-empty `source_instrument`, non-empty `provenance_notes`,
and the interval it describes. Only `synthetic_generator` is currently
populated; the other four categories are declared so real labels can be
ingested later, and so a reader can see which categories remain empty.

**Consequence:** Task telemetry and rPPG estimates remain *predictors or
outcome measurements* throughout, consistent with DEC-033 and DEC-020.
Difficulty level is a predictor and an experimental manipulation, never
cognitive load.

---

### DEC-046: Rule Baselines Are Software Checks and Are Labelled as Such

**Date:** 2026-08-04
**Status:** Accepted

**Context:** A learned model's score means little without something trivial
to compare it against. A rule baseline supplies that — and is also the
easiest artefact to misread as a validated indicator.

**Decision:** Two deterministic rule estimators are registered, one per
task type. Each thresholds or rescales **one** feature whose identity is an
arbitrary implementation choice recorded per target. Both are flagged
`is_software_check_baseline`, both carry
`RULE_BASELINE_DISCLAIMER` in their notes, and their probabilities are
documented as *not calibrated*.

When the preferred feature is absent — inside an ablation that removed it —
the estimator falls back to the first available measured column and records
which one it used. The substitution is never silent.

**Consequence:** No model is selected as a champion and none is labelled
production-ready. A synthetic self-check cannot rank models for any purpose
that matters, and tests assert that no registry entry, artefact, or CLI
line claims otherwise.

---

### DEC-047: MLflow Deferred; Local JSON and Parquet Run Records Instead

**Date:** 2026-08-04
**Status:** Accepted

**Context:** MLflow is a Milestone 10 deliverable. Milestone 5 needs
experiment records now.

**Decision:** Each run writes a self-describing directory of JSON and
Parquet under `artifacts/experiments/`, with `checksums.json` written
second-to-last and `manifest.json` written **last**, both atomically.

A directory with no manifest is an interrupted run. A manifest with
`status: failed` is a run that concluded in failure. A manifest claiming
completion is refused if a required artefact is missing.

**Consequence:** A run is auditable with no service running and without
loading any model file: provenance, metrics, splits, calibration design,
and disclaimers are all JSON. Model files are pickles and are labelled as
executable content in `models/README.txt`; the warning states they must
never be loaded from an untrusted source.

No Git command is run to obtain run metadata — version information comes
from installed package metadata.

---

### DEC-048: Two Modules Beyond the Milestone 5 Filename List

**Date:** 2026-08-04
**Status:** Accepted

**Context:** The milestone brief listed the modules to create under
`features/` and `training/`. Two pieces of behaviour had no listed home.

**Decision:** Dataset loading and predictor-matrix construction live in
`training/preprocessing.py` rather than in a new module: selecting columns,
auditing them for leakage, and preparing them for a transformer is one
concern, and splitting it would put the leakage audit at a distance from
the code that builds the matrix.

`cli_milestone5.py` follows the Milestone 4 precedent (DEC-013): the
modelling commands live in their own module so `__main__` stays a thin
dispatcher and so they can be tested without importing the webcam, rPPG, or
WebSocket code paths.

**Consequence:** `docs/ARCHITECTURE.md` is updated to match the implemented
module layout. No module listed in the brief was omitted.

---

### DEC-049: Quality Is a Support Signal, Not a Fusion Modality

**Date:** 2026-08-16
**Status:** Accepted

**Context:** The feature catalog has five modality groups, one of which is
`quality`. Milestone 6 fits one estimator per modality and weights their
outputs. Treating capture-quality diagnostics as a sixth voice would let
measurement conditions vote on a person's engagement.

**Decision:** `FusionModality` has exactly four members — behavioural,
head pose, rPPG, task. There is no `quality` member, so quality cannot
become a measurement modality by accident rather than by decision.

Capture-quality features, availability flags, and missingness indicators
are **support/context signals**. They may inform the explicitly named
quality-aware weighting strategy, and modality availability is always
carried in the early-fusion matrix because that is how a missing modality
is represented without fabricating a zero. `modality_quality__*` is
excluded from experts and from the early-fusion matrix by default.

**Consequence:** `parse_modality("quality")` raises with an explanation,
and `fusion.modalities: [quality]` is rejected at configuration load with
the same explanation. `FusionModality("quality")` is not constructible, so
no future code path can smuggle it in.

---

### DEC-050: A Missing Modality Is an Absence, Never a Zero

**Date:** 2026-08-16
**Status:** Accepted

**Context:** The obvious implementations of late fusion — substitute a
uniform probability vector, substitute the training mean, substitute zero —
all convert "we could not measure it" into a measurement, which is the
failure mode DEC-039 exists to prevent, moved one layer up.

**Decision:** A modality that produced no prediction is represented through
**availability**. Its effective weight is zero by construction
(`availability_m = 0` in the weight equation), the remaining weights are
renormalised over the contributors, and the exclusion is recorded with a
reason. A window that cannot meet `minimum_modalities` is recorded as
**unfused** with a stated reason and carries no prediction at all.

The schema enforces it: a `ModalityWeight` that did not contribute must
carry `normalized_weight == 0` and state why; a `FusionPrediction` may not
give weight to a modality that produced no prediction; an unfused
prediction may not carry a predicted class, a predicted value, or
probabilities.

**Consequence:** Coverage becomes a first-class result. A strategy that
scored well on a third of its windows is visibly different from one that
scored the same on all of them, and both numbers are always stored
together.

---

### DEC-051: The Quality-Aware Weight Equation Is Documented, Not Tuned

**Date:** 2026-08-16
**Status:** Accepted

**Context:** Quality-aware fusion needs a rule for turning a signal-quality
value into a weight. Any rule chosen by looking at a result on synthetic
data would be a fact about the generator.

**Decision:** One equation, stated in the schema, in the configuration
file, and in `docs/MULTIMODAL_FUSION.md`:

```
raw_effective_weight_m = base_weight_m * availability_m * normalised_quality_m
normalized_weight_m    = raw_effective_weight_m / sum over contributors
```

Base weights default to a deterministic 1.0 for every modality — the
control. **No optimised weight set is shipped**, and none was chosen after
looking at a score.

Missing quality is handled by a stated policy: `documented_fallback`
substitutes a neutral 0.5 (the midpoint of the range) and records
`quality_source=documented_fallback` on the weight; `exclude` drops the
modality and records the reason. Neither treats missing quality as perfect
quality, and no policy that does is offered. Task telemetry has no
signal-quality channel at all, so this is a normal condition rather than an
anomaly.

`minimum_quality` defaults to 0.0 because no empirically validated quality
cut-off exists for these signals; `minimum_effective_weight` defaults to
1e-9 as a numerical guard against normalising by ~0, not as a modelling
threshold. Both defaults are stated as such in the configuration file.

**Consequence:** Signal quality stays a statement about the measurement.
The weight it produces is recorded raw and normalised beside the quality
value and its source, so the arithmetic is reconstructible without
re-running anything, and low quality can never be read as low engagement.

---

### DEC-052: An Expert Learns Only From Windows Its Modality Observed

**Date:** 2026-08-16
**Status:** Accepted

**Context:** A modality expert fitted on every training row would be fitted
mostly on imputed medians for the rows where its modality was absent, and
would then emit confident predictions for windows it never observed.

**Decision:** Each expert is fitted on the fit-group rows in which its own
modality contributed evidence, and predicts only for test rows in which it
did. Below 10 such rows, or fewer than 2 independent groups, or fewer than
two classes, the expert returns **unavailable with a reason** rather than a
prediction. Row and group counts are recorded on every expert.

**Consequence:** An expert never speaks about a window it could not see.
The refusal is a recorded result, not a silent gap, and the two thresholds
are stated as engineering defaults rather than validated cut-offs.

---

### DEC-053: Calibration Is Per Expert, Before Fusion, and Only Once

**Date:** 2026-08-16
**Status:** Accepted

**Context:** Late-fusion classification can calibrate the experts, the
fused output, or both.

**Decision:** Calibration happens per expert before fusion, and to the
early-fusion estimator. **No post-fusion calibrator is fitted.**
Calibrating twice would make the reported probability the output of two
corrections with no way to attribute either, and there is no documented
reason in this project to prefer that. The fused probabilities are still
*evaluated* for calibration; nothing is fitted to them.

The stacked strategy is the exception in the other direction: it consumes
**uncalibrated** expert probabilities at both meta-training and
meta-inference time, because applying a meta-model to a different input
distribution from the one it was fitted on is a silent error no metric
would reveal.

Two shared-code consequences: `MINIMUM_CALIBRATION_SAMPLES_PER_CLASS = 5`
was added to `training/calibration.py`, because `CalibratedClassifierCV`
resolves `cv=None` to a 5-fold stratified splitter and cross-validates the
calibration set even over a `FrozenEstimator`; a class thinner than the
fold count cannot be split, and the pipeline now records an unavailable
calibrator with a reason instead of failing the fold.

**Consequence:** `calibration.json` states the placement in words. A
calibrated probability remains a statement about outcome frequency, kept in
separate fields from signal quality throughout.

---

### DEC-054: Stacking Is Implemented, Leakage-Checked, and Off by Default

**Date:** 2026-08-16
**Status:** Accepted

**Context:** A stacker trained on expert predictions about the experts' own
training rows learns to trust memorised predictions, and the weights it
derives do not transfer. The failure is invisible in the metrics.

**Decision:** Stacking is implemented with grouped out-of-fold construction
inside each outer training portion, and `assert_out_of_fold` re-checks the
property independently before the meta-model is fitted. Three violations
are detected and raised on: a row predicted by experts fitted on its own
group, a row predicted by experts fitted on an outer-test group, and a
meta-training row from outside the outer-training groups. Each has a test.

Meta-models are `LogisticRegression` and `Ridge`; a configuration naming
anything else is rejected. There is no neural stacker.

An unavailable expert contributes a **missing value** to the meta matrix,
never a zero, and the meta-model's own fold-local imputer adds a
missingness indicator.

`fusion.stacking.enabled` defaults to **false**: it costs an extra
inner-fold pass over every fold, and the three default strategies answer
the milestone's questions without it.

**Consequence:** A future change that reintroduces in-sample meta-training
fails loudly rather than producing a plausible number.

---

### DEC-055: Scenarios Are Evaluation-Time; Synthetic Dropout Is Dataset-Level

**Date:** 2026-08-16
**Status:** Accepted

**Context:** "What happens when rPPG is missing?" has two readings: a
system trained normally meets a degraded window, or a system is trained on
a dataset in which the modality was never captured. They are different
questions and need different mechanics.

**Decision:** Both are implemented and kept apart.

Missing-modality **scenarios** are applied at evaluation time only. Models
are trained once on the recorded availability and then met with each
deterministic availability pattern. A scenario removes availability and
nothing else: it never rewrites a measurement, never zero-fills a feature,
and never touches a target, and it cannot make an absent modality appear
present.

Synthetic modality **dropout** changes the dataset's availability before
folding, so it affects training as well. It is seeded, decided by a pure
function of (seed, window id, modality) so it is order-independent, drops
whole modality groups coherently, and records its configuration. **It is
refused in scientific mode**, because it fabricates an availability pattern
that no measurement produced.

**Consequence:** Ten named scenarios can be compared on one set of fitted
models and one set of folds, and a reader can tell which question a number
answers.

---

### DEC-056: Expert Disagreement Is a Diagnostic, Not Uncertainty

**Date:** 2026-08-16
**Status:** Accepted

**Context:** Ensemble spread is easy to compute and easy to mislabel.
Calling it uncertainty would pre-empt Milestone 7 and would imply a
calibration this quantity does not have.

**Decision:** Disagreement is recorded as an **ensemble-disagreement
diagnostic** with explicit definitions: distinct predicted classes,
unanimity, mean pairwise probability distance, fused-probability entropy,
prediction standard deviation, prediction range. Windows with fewer than
two available experts contribute to no summary and are counted separately.

Every stored summary carries a required `note` stating that it is not a
calibrated uncertainty estimate, not signal quality, not model confidence,
and that it does not trigger abstention. Nothing in this milestone gates a
prediction on it.

**Consequence:** Milestone 7 can introduce uncertainty-aware inference and
abstention without first having to retract a claim.

---

### DEC-057: Validation-Derived Weights Use Bounded Skill, Not Reciprocal Error

**Date:** 2026-08-16
**Status:** Accepted

**Context:** The common rule for performance-derived fusion weights,
`w = 1 / error`, diverges when an expert scores perfectly on a small
validation set. Clamping it requires a threshold with no justification.

**Decision:** Bounded, scale-free skill scores computed on inner validation
groups drawn only from the outer training portion:

- classification: `w = max(0, (balanced_accuracy − 1/K) / (1 − 1/K))`
- regression: `w = max(0, 1 − MAE / MAE_of_predicting_the_mean)`

Both lie in `[0, 1]` and cannot diverge. When every weight is zero the run
falls back to deterministic equal weights and records that it did, with the
reason. Each fold records the metric, its definition, the exact groups
used, the raw scores, and the resulting weights.

**Consequence:** The outer test fold contributes to no weight, and a test
asserts that the recorded groups are a subset of the fold's training
groups.

---

### DEC-058: The Fusion Run Extends the Milestone 5 Experiment Format

**Date:** 2026-08-16
**Status:** Accepted

**Context:** A fusion run needs several documents Milestone 5 has no place
for. Replacing the format would orphan the existing runs and the tooling
that reads them.

**Decision:** The Milestone 5 directory layout is kept and extended with
`fusion_config.json`, `experts.json`, `fusion_metrics.json`,
`robustness.json`, `expert_predictions.parquet`, and
`fusion_weights.parquet`. `metrics.json` is the same `MetricsDocument`, with
one `ModelResult` per fusion strategy (`model_kind: "fusion"`) and one per
unimodal expert (`model_kind: "unimodal_expert"`), so the Milestone 5
metric machinery is reused rather than duplicated.

`ExperimentRun` gained an optional `required_artifacts` parameter so a
fusion run can declare a **larger** required set. The Milestone 5 default is
unchanged.

`expert_predictions.parquet` is written for the reference scenario only: a
scenario does not change what an expert computed, only which experts were
allowed to contribute, and that is in `fusion_weights.parquet` for every
scenario.

**Consequence:** Existing readers of a Milestone 5 run directory keep
working. Still no MLflow, no DVC, and no Docker; tests assert none appears.

---

### DEC-059: Modules Beyond the Milestone 6 Filename List

**Date:** 2026-08-16
**Status:** Accepted

**Context:** `docs/ARCHITECTURE.md` listed one module, `training/fusion.py`,
for the whole milestone. The milestone contains several separable concerns
with different failure modes.

**Decision:** The fusion layer is split by concern:

| Module | Concern |
|---|---|
| `training/fusion.py` | pure algebra: modality columns, weights, combination, disagreement |
| `training/experts.py` | fitting one estimator per modality, and refusing to |
| `training/stacking.py` | out-of-fold construction and the leakage assertion |
| `training/robustness.py` | scenarios and deterministic synthetic dropout |
| `training/fusion_metrics.py` | coverage, contribution, disagreement summaries |
| `training/fusion_artifacts.py` | run identity, split fingerprint, Parquet tables |
| `training/fusion_runner.py` | fold orchestration and artifact assembly |
| `cli_milestone6.py` | `fusion-demo` / `fusion-train` |

`cli_milestone6.py` follows the Milestone 4 and 5 precedent (DEC-013,
DEC-048): `__main__` stays a thin dispatcher and the fusion commands can be
tested without importing the webcam, rPPG, or WebSocket code paths.

**Consequence:** `docs/ARCHITECTURE.md` is updated to match the implemented
layout. No behaviour listed for the milestone was omitted; the pure algebra
in `training/fusion.py` is testable without fitting anything, which is what
made the weight and combination rules cheap to pin down.

---

### DEC-060: Personalization Layers on the Fused Population Prediction

**Date:** 2026-08-16
**Status:** Accepted

**Context:** Milestone 6 acceptance criterion 3 requires personalized and
population baselines to be reported separately. Personalization could have
been built as its own model stack, or layered onto the existing fusion
output. A separate stack would duplicate the fold machinery and make the
two reports incomparable.

**Decision:** Personalization layers on top of a **population reference
model**, which is the early-fusion estimator over the configured modality
groups (`early_fusion_columns`, unchanged). The documented path is
`early-fusion population prediction -> subject calibration/correction ->
personalized prediction`.

Early fusion rather than a late-fusion strategy is used as the reference
because it yields exactly one population prediction per window with no
weighting step to disturb: a per-subject correction applied on top of a
re-weighted combination would confound two adjustments and neither would be
attributable.

No fusion weight is retuned by personalization, and the population
prediction is retained unchanged on every record — a personalized
prediction is an addition to it, never a replacement.

**Consequence:** `metrics.json` carries two `ModelResult` entries,
`population` and `personalized`, over identical evaluation windows. Tests
assert the row counts match and that the population column is always
populated.

---

### DEC-061: The Calibration/Evaluation Split Is Temporal, Not Positional

**Date:** 2026-08-16
**Status:** Accepted

**Context:** A personal baseline must come from windows that precede the
windows it is evaluated on. Taking the first *N* rows is not sufficient:
`configs/defaults.yaml` permits overlapping windows, and with a 30-second
window stepped every 15 seconds the window immediately after the
calibration region still shares evidence with it.

**Decision:** A held-out subject's windows are ordered by
`(window_start_utc, window_end_utc, window_index, window_id)`. The first
`calibration_windows` form the calibration region; the **boundary** is the
latest `window_end_utc` in it; a later window joins the evaluation region
only if its `window_start_utc` is at or after the boundary. A window
straddling the boundary is **excluded from both regions** and recorded in
`excluded_overlap_window_ids`.

`PersonalCalibrationSplit` refuses to validate when the calibration region
does not end before the evaluation region begins, so an unordered split
cannot be persisted. A window with no timestamp is refused outright rather
than ordered by row position, which is not a temporal order.

`ModellingFrame` gained optional `window_start_utc`, `window_end_utc`,
`window_indices`, and `windows_overlap` fields to carry the timing. They
are provenance, never predictors, and `assert_no_leakage` still refuses
them in any predictor matrix.

**Consequence:** A subject who cannot supply both regions is recorded as
unavailable with a reason and excluded from both reports. The protocol is
never weakened to keep a subject in the personalized report.

---

### DEC-062: Both Documented Corrections, and Why They Are Regularised

**Date:** 2026-08-16
**Status:** Accepted

**Context:** A few labelled calibration windows cannot support a
subject-specific model. They can support a small, interpretable correction
to a population model's output — provided it degrades gracefully when the
evidence is thin.

**Decision:** Two corrections, both fitted from calibration windows only.

*Regression* uses the bias correction exactly as specified:

    b_s = mean(y_calibration - y_population_prediction)
    y_personalized = y_population_prediction + b_s

*Classification* uses a regularised per-subject log-odds shift, with `K`
classes, `n` labelled calibration windows, smoothing `alpha`, and shrinkage
constant `kappa`:

    observed_c = (count_c + alpha) / (n + alpha*K)
    expected_c = (sum_w p_population_c(w) + alpha) / (n + alpha*K)
    lambda     = n / (n + kappa)
    delta_c    = lambda * (log(observed_c) - log(expected_c))
    p_personalized_c ∝ p_population_c * exp(delta_c)

Both terms are smoothed identically, which gives the property that matters:
`delta_c` is **exactly zero** when the subject's calibration labels match
what the population model predicted on average, so a subject the population
model already fits is left alone. `lambda` shrinks the shift toward zero in
proportion to how little evidence there is.

`kappa = 5.0` and `alpha = 1.0` are engineering defaults. No validated
value exists for either, and no attempt was made to tune them: tuning on
synthetic data would fit this repository's own generator.

**Consequence:** Corrected probabilities are finite, non-negative, and
renormalised; a row that cannot be renormalised raises rather than being
emitted. `calibration_targets` records the label used per calibration
window by id, and a test asserts no evaluation window id ever appears
there.

---

### DEC-063: Cold Start Is an Outcome, Not a Requestable Method

**Date:** 2026-08-16
**Status:** Accepted

**Context:** The specification lists cold-start mode alongside the other
personalization modes. Offering it as a method name would make an *outcome*
("this subject had no usable evidence") indistinguishable from an
*intention* ("evaluate without personal evidence").

**Decision:** `PersonalizationMethod.COLD_START` exists and is recorded on
every prediction that fell back, but it is not in `REQUESTABLE_METHODS`.
Both the typed configuration and `configs/defaults.yaml` reject it by name
and point the reader at `calibration_windows: 0`, which is how cold-start
behaviour is requested.

Falling back is always explicit: `personalization_applied=false`,
`cold_start=true`, a stated reason, and — enforced by the schema — a
personalized output that reproduces the population output **exactly**. No
run borrows another subject's baseline, substitutes a global statistic and
calls it personal, or fabricates one.

**Consequence:** Coverage is a first-class number.
`personalization_coverage` is the fraction of evaluated subject-folds that
were actually personalised, and on the shipped synthetic dataset it is
0.700 for the classification targets — nine of thirty subject-folds have
only one class in their five calibration windows and fall back, which is
the intended behaviour rather than a failure.

---

### DEC-064: Training Subjects Are Normalised Under the Same Chronological Rule

**Date:** 2026-08-16
**Status:** Accepted

**Context:** With personal-baseline normalization, a training subject's
baseline could legitimately use every one of their training windows — they
are training data, so there is no leakage. The first implementation did
exactly that, and the held-out subjects' regression predictions were wildly
out of range (RMSE 3.38 against a target range of `[0, 1]`).

**Decision:** A training subject's baseline is estimated from their own
earliest `calibration_windows` windows within the training portion, under
the identical chronological rule applied to held-out subjects.

The cause of the failure was not leakage but a **scale mismatch**:
estimating `sigma_s` from ten windows and then applying a `sigma_s`
estimated from five produced systematically larger z-scores at evaluation
time, and the estimator met a differently-scaled matrix than it was fitted
through. The normalisation a model is fitted through must be the
normalisation it is deployed through. After the change, the same run's RMSE
fell to 0.47.

Only held-out subjects' baselines are persisted to
`personal_baselines.json`: a training subject has no calibration/evaluation
boundary to audit, and recording thousands of them would add bulk without
adding evidence.

**Consequence:** The remaining gap between the population and personalized
regression results is a property of the generator — its targets track
absolute feature levels, which within-subject z-scoring removes — and is
reported as such rather than tuned away.

---

### DEC-065: Balanced Accuracy Is Computed From the Recall Vector

**Date:** 2026-08-16
**Status:** Accepted

**Context:** The complete test run emitted one warning: scikit-learn's
`UserWarning: y_pred contains classes not in y_true`, from
`balanced_accuracy_score` during a heavy missing-modality robustness case.
Suppressing warnings globally would hide genuine ones; suppressing this
message specifically would hide it in cases where it is informative.

**Decision:** Balanced accuracy is derived from the per-class recall vector
that `classification_metrics` already computes via
`precision_recall_fscore_support(..., zero_division=np.nan)`, rather than
by calling `balanced_accuracy_score`.

The two are the same quantity by definition: balanced accuracy is the
unweighted mean of recall over the classes present in `y_true`, and
`zero_division=np.nan` marks exactly the absent classes so `_macro` excludes
them. `balanced_accuracy_score` derives it from an *unlabelled* confusion
matrix, which is why a predicted-but-absent class becomes an all-zero row
and triggers the warning before being dropped.

Equivalence was checked over 200 randomly generated label/prediction pairs
before the change, and three regression tests now pin it, including the
exact heavy-dropout condition.

**Consequence:** The normal test run completes with **no warnings**. The
Milestone 5 rule is untouched: a metric whose prerequisites are unmet is
still `None` with a stated reason, never zero.

---

### DEC-066: Only an Already-Calibrated Source May Produce "Confidence"

**Date:** 2026-08-16
**Status:** Accepted

**Context:** Milestone 7 needs a number to threshold. The obvious candidate
is the maximum predicted class probability. But Milestone 6 calibrates each
modality expert and the early-fusion estimator and **never the fused
probability vector** (`docs/MODEL_EVALUATION.md`, "Calibration is placed
once"), so a late-fusion maximum has not been fitted against observed
outcomes at all.

**Decision:** `PredictionSource` offers exactly two members —
`baseline_model` and `early_fusion` — both of which pass through the
Milestone 5 calibration step. A late-fusion source is not offered, and
`UncertaintyConfig` refuses one with a message naming the reason.

When a fold's calibrator is nevertheless unavailable — usually a class with
fewer than `MINIMUM_CALIBRATION_SAMPLES_PER_CLASS` rows — the identical
number is recorded as `selection_score` under
`max_uncalibrated_probability`, with a stated reason.
`ClassificationConfidence` refuses to validate a record that populates
`confidence_score` without the contract, so the mistake cannot be
persisted.

By default the evidence gate then refuses confidence-based abstention for
that fold rather than thresholding an uncalibrated number as though it were
calibrated. `require_probability_calibration_for_classification_confidence:
false` opts into an explicitly named uncalibrated selection-score policy.

**Alternatives rejected:** *Calibrate the fused vector post hoc.* That
would be double calibration on a set that had already served as a
calibration set for the experts, with no independent third split available.
*Call the fused maximum confidence anyway.* That is the exact overstatement
this milestone exists to prevent.

**Consequence:** The shared test fixture is deliberately sized so that some
folds calibrate and at least one does not, so both branches are exercised
end to end rather than only in unit tests.

---

### DEC-067: The Split-Conformal Quantile Rule, and Refusal Below It

**Date:** 2026-08-16
**Status:** Accepted

**Context:** A regression target has no class probability, so it needs a
different uncertainty representation. `1 - interval_width` would be a
probability-shaped number with no probabilistic meaning, and an ensemble
spread is not a calibrated interval.

**Decision:** Split conformal absolute-residual intervals, with the exact
finite-sample convention recorded on every record:

```
r_i = |y_i - yhat_i|                     on the fold's calibration groups
k   = ceil((n + 1) * (1 - alpha))
q   = the k-th smallest residual         (1-indexed)
interval(x) = [yhat(x) - q, yhat(x) + q]
```

Residuals come from the calibration groups, which are disjoint from the fit
groups and from the outer-test groups. Residuals from rows the model
memorised would understate the interval, and `UncertaintyFoldResult` refuses
to validate a fold where they overlap.

**When `k > n` the interval is UNAVAILABLE with a reason.** It is not
widened to infinity, and it is not fabricated. The rule first holds at
`n = ceil(1/alpha) - 1`.

**Assumption, stated rather than assumed away:** marginal coverage of at
least `1 - alpha` holds under *exchangeability* of calibration and test
points. Under grouped cross-validation those rows come from **different
people**, so exchangeability is an assumption about between-subject
variation that this repository has never tested — and one there is good
reason to expect to fail. On the 30-subject synthetic dataset, empirical
interval coverage varies between **0.846 and 0.963 per fold** against a
nominal 0.90, with a cross-fold mean near nominal. The wide per-fold
dispersion is what a violated exchangeability assumption produces; a mean
that happens to land near nominal is not the guarantee holding. No
conformal coverage guarantee is claimed for real EngageVR data.

**Consequence:** Split conformal produces one quantile per fold, so the
width sweep is all-or-none rather than a gradual curve. That is documented
as a property of the method rather than smoothed over.

---

### DEC-068: The Personalized Threshold Reads No Labels At All

**Date:** 2026-08-16
**Status:** Accepted

**Context:** Milestone 6 deferred personalized confidence thresholds to
Milestone 7. Fitting a per-subject rule from five calibration windows is an
overfitting risk, and any rule that consumes labels is a leakage risk.

**Decision:** A per-subject threshold that consumes **no label of any
kind** — only the confidence scores the population model assigned to the
subject's own earlier windows:

```
tau_raw = quantile(subject calibration confidence, 1 - target_coverage)
lambda  = n / (n + kappa)
tau_s   = (1 - lambda) * tau_population + lambda * tau_raw
```

clipped to `[0, 1]`, with numpy's `"lower"` quantile method so `tau_raw` is
an *observed* confidence value rather than an interpolated one.

This is a stronger safety statement than "we were careful not to pass a
label": an evaluation label cannot influence the threshold by *any* path,
because the function has no argument through which one could arrive.
`PersonalThresholdRecord.uses_labels` is recorded as `false` and the
validator refuses a record claiming otherwise.

The temporal split is Milestone 6's `build_calibration_split`, unchanged,
so the wall-clock calibration-before-evaluation boundary and the
straddling-window exclusion are inherited rather than reimplemented. The
calibration windows are removed from the fold's evaluated set, so a window
never both derives a threshold and is scored under it.

Below `minimum_personal_calibration_windows` the subject falls back to the
population threshold with a stated reason.
`fallback_to_population_threshold` cannot be disabled.

**Alternatives rejected:** *Fit a per-subject calibrator or a per-subject
classifier.* Both need labels and both would overfit a handful of windows.
*Target accepted accuracy per subject.* Needs labels.

**Consequence:** The rule addresses a real failure mode — a subject the
model is uniformly less confident about being abstained on entirely, which
is a measurement artefact presented as a property of that person — without
any label-driven mechanism. It was not tuned to make synthetic
personalization look better.

---

### DEC-069: Signal Quality Gates Separately, Never Multiplicatively

**Date:** 2026-08-16
**Status:** Accepted

**Context:** It is tempting to fold signal quality into the confidence
score — `confidence * quality` — so that one number governs everything.

**Decision:** The evidence gate is a **separate** component with its own
reason codes, and quality is never multiplied into a model probability.

No probabilistic model in this repository justifies treating a camera
diagnostic as a likelihood term. Multiplying them would also destroy the
distinction the whole milestone exists to preserve: a window blocked
because the camera signal was poor is a different event from a window
blocked because the model was unsure, and a single product cannot say which
happened. Worse, it would make a measurement problem readable as a
statement about a person.

`evaluate_evidence_gate` returns `(passed, reasons)` with reasons in
canonical order; evidence reasons precede model-confidence reasons, because
an estimate built on absent evidence should not be discussed in terms of
its confidence.

Absence of a recorded quality is **not** a low quality: a modality with no
quality column does not fail the gate unless
`treat_missing_quality_as_failure` is set.

**Consequence:** `signal_quality_below_gate`,
`insufficient_measurement_evidence`, `required_modality_unavailable`,
`probability_calibration_unavailable`, `below_confidence_threshold`,
`prediction_interval_unavailable`, and `interval_too_wide` are seven
distinct codes, and a run's `abstention_reason_counts` shows which fired.

---

### DEC-070: One Grid, Two Task Types, Higher Is Always Stricter

**Date:** 2026-08-16
**Status:** **Superseded by DEC-072 (2026-08-18)**

**Context:** A classification threshold rises toward *stricter*; an
interval-width maximum falls toward stricter. Using the same configured
grid for both naively would make coverage non-increasing in one task type
and non-decreasing in the other, and the monotonicity check — which is a
correctness check, not a finding — would mean opposite things.

**Decision (superseded):** One grid, with the invariant that **higher is
stricter**. For classification, `accept if score >= g`. For regression,
grid point `g` mapped to `maximum_width = (1 - g) * widest_observed_width`.

**Why this was wrong.** The mapping made the reported regression curve
incompatible with the acceptance rule the run actually applies. Its x-axis
was neither a confidence nor a width: it was a dimensionless fraction of an
observed maximum, so a grid point had no meaning outside the fold it was
computed in, the axis moved when the data moved, and the curve was labelled
"non-increasing" while the documented rule
`accept if interval_width <= W_max` is non-*decreasing* in `W_max`. Sharing
one configuration surface was not worth reporting a curve that contradicted
the rule. Superseded by DEC-072.

**Retained from this decision:** `accepts_interval_width` still accepts a
`maximum` of exactly zero, so a configured width sweep may include its
strictest endpoint. A zero configured as a run's operating policy is still
refused separately, because a policy that abstains on every window is a
configuration mistake rather than a curve endpoint. The evidence gate is
still applied at every grid point, so a window blocked for missing evidence
is blocked at every axis value and the direction contract holds regardless
of the gate's configuration.

---

### DEC-071: The Adaptation Gate Is Bounded by What It Cannot Import

**Date:** 2026-08-16
**Status:** Accepted

**Context:** Milestone 7 acceptance criterion 3 in `docs/PROJECT_PLAN.md`
reads "adaptation policy respects both thresholds". That wording predates
the M7/M8 split, and reading it as licence to implement a policy here would
pull Milestone 8 forward.

**Decision:** Implement the **gate only**. It answers "may an
already-chosen action be acted upon?" and never "which action?".

The boundary is enforced structurally rather than by intention:
`adaptation_gate.py` imports nothing but `engagevr.schemas.targets` and
`engagevr.schemas.uncertainty`, and a test parses the module's AST and
asserts exactly that import set. A reviewer can establish that the module
cannot send a message, choose a difficulty, or learn a policy by reading
what it is allowed to import.

The gate consumes already-computed information and recomputes nothing, so a
gate decision can never disagree with the decision it gates. A disabled
gate stops applying any additional requirement of its own but still refuses
to declare an abstained or unavailable window eligible — that would be a
false statement, not a relaxed policy.

**Consequence:** Cooldown, hysteresis, manual override, static-versus-
adaptive modes, and every reward or action vocabulary remain Milestone 8.
The acceptance table records criterion 3 as met *for the gate*, and says
explicitly that the policy is not implemented.

---

### DEC-072: Two Coverage Axes, in Their Own Units, in Their Own Directions

**Date:** 2026-08-18
**Status:** Accepted

**Context:** DEC-070 forced one grid to serve both task types by rescaling
interval widths into fractions of the widest observed width. That produced
a regression coverage curve whose x-axis was not the quantity the
acceptance rule compares against, and whose declared monotonicity direction
was the opposite of the rule's.

**Decision:** Selective prediction has **two distinct axes**, and a curve
records which one it was swept over.

| | Classification | Regression |
|---|---|---|
| axis | `confidence_threshold` | `maximum_interval_width` |
| units | probability in [0, 1] | the target's own units |
| rule | `accept if score >= tau` | `accept if interval_width <= W_max` |
| raising it | stricter | more permissive |
| coverage | non-increasing | non-decreasing |

`CoverageAxis` and `MonotonicDirection` are enums; `CoverageCurve` carries
both and validates the axis against its `task_type`, so a regression curve
cannot be indexed by a classification confidence score and a classification
curve cannot be indexed by a width. `coverage_is_monotonic` takes the
direction as a required argument and refuses to check an axis in the
direction that is not its own — the two checks are different assertions,
not one assertion with a sign.

The width grid is a **separate configuration surface**,
`uncertainty.regression.interval_width_grid`, holding widths in the
target's own units. It is validated finite and non-negative but **not**
bounded to [0, 1], because a regression target need not live there. Each
grid value is compared against the original interval width by the same rule
the run applies. No width is normalised into [0, 1] to share the confidence
grid, and none is inverted into `1 - width`.

The grid defaults to `null`, because the right widths depend on a target
scale this repository has not measured and inventing one would presume a
scale. When it is null the run **manufactures no curve**: it records its
operating point and marks the width curve unavailable with a reason that
names the configuration key. `CoverageCurve` enforces this — a curve with
no points must state why, and must report `coverage_is_monotonic: null`
rather than a vacuously true claim.

An operating point with no configured `maximum_interval_width` records
`threshold: null` with a stated reason, never `0.0`. An absent width policy
accepts every otherwise-available prediction; a width policy of `0.0` would
accept none. Recording the first as the second would invert the reported
meaning of the run.

**Consequence:** The regression curve is now readable as the rule that
produced it. Two configuration surfaces exist where there was one, and
`CoverageCurve.threshold_grid` became `axis_values` because the values are
no longer always thresholds on a probability. The synthetic regression
curve is a rising step rather than a falling one — that is the corrected
shape of the same data, not a new result.
