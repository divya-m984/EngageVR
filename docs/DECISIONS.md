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

---

### DEC-073: Milestone 8 Chooses; It Does Not Send

**Date:** 2026-08-19
**Status:** Accepted

**Context:** Milestone 7 answers whether an already-chosen action *may* be
acted upon. Milestone 8 is the first layer in this project that chooses
anything. The obvious shape — a policy that decides and then puts the
resulting command on the Milestone 4 WebSocket — would have made "the rule
recommended a change" and "the environment changed" the same event, at the
moment the project has no evidence that any adaptation helps anyone.

**Decision:** A **policy decision is not a network transmission**, and the two
are different objects.

`AdaptationPolicyDecision` carries no command id, no wire payload, no address,
and no `send`. Translating an approved `AdaptationProposal` into the existing
Milestone 4 `adaptation_command` payload is a separate pure function in
`adaptation/command.py` that builds an object and returns it.

The boundary is enforced by imports rather than by convention.
`adaptation/policy.py` imports two schema modules and its own mapping table;
it imports nothing from `engagevr.api`, `engagevr.transport`, `engagevr.task`,
or `engagevr.training`. Tests parse the AST of every module in
`engagevr.adaptation` and assert that none imports a transport module and that
none calls `send`, `send_json`, `broadcast`, `publish`, or `dispatch`.

Milestone 8 therefore stops at `command_built`. The `--dispatch` flag exists
only so that asking for live transport produces a stated refusal rather than
silently doing nothing.

**Consequence:** The whole path — prediction, gate, policy, proposal, command
object — is unit-testable with no server, no WebSocket, no Unity, and no task
simulator. The cost is that an integrated live loop does not exist and will
need a deliberate decision in a later milestone.

---

### DEC-074: The Adaptation Mapping Is a Demonstration Rule Built From Two Principles

**Date:** 2026-08-19
**Status:** Accepted

**Context:** `docs/PROJECT_SPECIFICATION.md` gives an example policy. Three of
its rows are expressible with the protocol's `set_difficulty` action; two are
not. Writing nine independent table cells would have hidden which cells came
from the specification and which were invented, and the obvious invention —
"low engagement, therefore make it harder" — is a psychological assumption
rather than a reading of the evidence.

**Decision:** The 3x3 table is **generated from two stated principles plus a
default**, and it is present as data (`MAPPING_TABLE`) so it can be checked
against the documentation without following control flow.

- **P1, overload protection:** cognitive load `HIGH` suggests `DECREASE`. The
  only protective direction, and the only rule that fires on one signal alone.
- **P2, engagement headroom:** engagement `HIGH` suggests `INCREASE`, and an
  increase is proposed only when cognitive load is affirmatively `LOW`. A
  `MEDIUM` load is the absence of a reading that supports an increase, not a
  reading that supports one. This is the specification's row 1 verbatim.
- **P3:** everything else holds.

| eng \ load | low | medium | high |
|---|---|---|---|
| low | hold | hold | **decrease** |
| medium | hold | hold (deadband) | **decrease** |
| high | **increase** | hold | hold (conflict) |

Two specification rows are deliberately **not** implemented. "Declining
engagement + low or moderate load -> feedback or introduce variation" names a
response this protocol cannot express, so the policy holds with
`no_expressible_action` rather than substituting a difficulty change for the
response the specification actually named. "Sustained fatigue -> a break" has
no fatigue estimator behind it; inferring fatigue from blink proxies or
heart-rate estimates is the measurement-to-construct leap
`reject_automatic_derivation` refuses, so `pause_task` is never issued even
though the protocol supports it.

Every decision additionally records what **each signal alone** suggested,
whether they conflicted, and how the conflict was resolved. Conflicts are
never hidden behind the resolved direction.

**Consequence:** The policy is conservative by construction: seven of nine
cells hold. It is labelled an ENGINEERING DEMONSTRATION RULE in the
configuration, in every persisted document, in the CLI output, and in the
`reason` field of every command it builds. It is not a validated
interpretation of human state, and no part of this repository claims it is.

---

### DEC-075: Both Ordinal Targets Are Required, Because the Mapping Says So

**Date:** 2026-08-19
**Status:** Accepted

**Context:** A policy could act on engagement alone, on cognitive load alone,
or on both. Accepting partial evidence is more permissive and would let the
demo produce more adaptations.

**Decision:** Both `engagement_class` and `cognitive_load_class` must be
present and Milestone 7-eligible. A missing target holds with
`insufficient_evidence`.

This is **derived from the mapping rather than imposed on it**. Under DEC-074,
an increase requires high engagement *and* low cognitive load, and a decrease
requires high cognitive load. A single target therefore cannot select any
direction. Accepting one target would mean supplying the missing signal's
state from somewhere, and the only available sources are a default (an
assumption) or the measurement itself (the leap `reject_automatic_derivation`
refuses).

**Consequence:** An offline run consumes two Milestone 7 targets. The
scenario suite exercises the policy without Milestone 7 runs at all, by
building real `AbstentionDecision` records and putting them through the real
gate function. Single-target operation would need a documented single-signal
rule first, which would be a new research claim rather than a configuration
change.

---

### DEC-076: The Dwell Count Resets on Hold; Cooldown Is Counted in Windows

**Date:** 2026-08-19
**Status:** Accepted

**Context:** Two temporal choices needed fixing, and the plausible-looking
alternative was wrong in each case.

*Persistence.* A decaying count (hold subtracts one instead of clearing)
looks gentler. It lets evidence separated by contradicting windows accumulate
into a dwell requirement that was never actually met: three supporting
windows interleaved with three holds would eventually reach a threshold that
"three consecutive supporting windows" was written to express.

*Cooldown.* A seconds-based cooldown matches the specification's wording
("at least 20-30 seconds") but makes an offline replay depend on wall-clock
timestamps, so a re-run is only approximately reproducible.

**Decision:**

1. A window resolving to `HOLD` **resets** the dwell count to zero and clears
   the pending direction. It does not decay it. A direction change restarts
   the count at 1. A Milestone 7-blocked window resolves to `HOLD`, so a
   blocked window can never count as supporting evidence.
2. Cooldown is counted in **evaluation windows**, one primitive
   (`cooldown_windows`), with no `cooldown_seconds` alongside it. The default
   of 6 is the specification's 30 s at the default
   `windowing.model_inference_seconds` of 5 s, and the derivation is written
   in the configuration file. A proposal sets the counter to
   `cooldown_windows`, so the minimum spacing between proposals is
   `cooldown_windows + 1` evaluation windows.
3. **Window passage is defined independently of evidence.** The cooldown
   decreases on every evaluated window, blocked or not. A blocked window is
   not evidence, but it is still a window; making time depend on evidence
   would mean a run of unusable windows freezes the cooldown indefinitely.
4. A **duplicate** window (same id at the same order as the previous
   evaluation) is absorbed idempotently: no count advances and no guard
   expires, so a retransmission cannot manufacture evidence or expire a
   cooldown. An **out-of-order** window raises rather than being reordered.

**Consequence:** The trace is byte-reproducible and every guard is auditable
from `persistence_count_before/after` and `cooldown_remaining_before/after`.
The wall-clock meaning of the cooldown depends on the stream's window cadence,
which is stated as a limitation rather than hidden.

---

### DEC-077: Hysteresis Is Emergent; Confidence Is Not a Control Gain

**Date:** 2026-08-19
**Status:** Accepted

**Context:** The specification lists hysteresis as a required safety control,
and the natural reading is a dedicated parameter. Separately, a confidence
score is a tempting step multiplier: act more decisively when the model is
more certain.

**Decision:** Neither knob exists.

*Hysteresis.* It is implemented through mechanisms that already had to exist:
the neutral-class deadband, the dwell requirement, the direction-change reset,
and the cooldown. A proposal additionally **restarts** the dwell count at
zero, so the next proposal in either direction needs fresh evidence rather
than inheriting the previous window's. With the defaults a reversal therefore
requires at least 7 windows, of which at least 3 consecutive must support the
new direction. Adding a `hysteresis_*` parameter would be a redundant knob
that could disagree with the mechanisms already enforcing the behaviour. A
test asserts that no such field exists.

*Step size.* `step` is a configured constant and is never multiplied by
confidence. Confidence decided, in Milestone 7, whether this window may be
acted on **at all**; reusing it as a gain would let a barely admissible
estimate move the environment further than a clear one, which inverts what the
threshold was for. `AdaptationProposal` carries no confidence field, and a
test asserts that.

The same reasoning bars signal quality from choosing a direction.
`AdaptationTargetSuggestion` refuses to record a direction without an ordinal
state, and an ordinal state comes only from a declared-ordinal class label or
an explicitly configured regression band — never from a quality value.

**Consequence:** Fewer parameters, and the ones that remain each mean exactly
one thing. The behaviour the specification asked for is demonstrated by the
`direction-reversal` scenario rather than asserted by a setting.

---

### DEC-078: An Ordinal Class Order Must Be Declared Twice, and the Two Must Agree

**Date:** 2026-08-19
**Status:** Accepted

**Context:** The policy needs `low < medium < high`. The vocabulary is
already an ordered tuple on `TargetSpec`, so reading `vocabulary.index(label)`
would work today. It would also silently acquire a meaning it never had if
the vocabulary were ever reordered, extended, or renamed — a nominal
vocabulary listed in some arbitrary order would become an ordinal scale.

**Decision:** The ordering is declared **twice**, and the two declarations
must agree before the policy will map anything.

1. `TargetSpec.class_order_is_ordinal` (new field, default `False`) states
   whether a vocabulary is an ordered scale at all. Set `True` for
   `engagement_class` and `cognitive_load_class`; a regression target may not
   set it.
2. `ORDINAL_CLASSIFICATION_TARGETS` in the Milestone 8 schema names the exact
   vocabulary each target must declare.

`ordinal_state_from_class` refuses with `AdaptationPolicyError` if the target
is not declared ordinal, if its vocabulary is not the one the policy was
written against, or if the label is not in it. Neither array position nor
alphabetical order is consulted anywhere.

**Consequence:** A vocabulary change stops the policy instead of quietly
changing what "high" means. The cost is a small duplication that a test keeps
in sync.

---

### DEC-079: Regression Mapping Exists and Is Disabled, With No Default Boundary

**Date:** 2026-08-19
**Status:** Accepted

**Context:** The project has four targets: two ordinal classes and two
continuous scores. Symmetry argues for driving the policy from either. But
thresholding a continuous score needs boundaries, and this repository has
never measured either score on a person, so any boundary would be invented.
DEC-072 established the same point for interval widths.

**Decision:** Regression mapping is implemented, fully tested, and **disabled
by default**. The ordinal class targets carry a neutral class that already
acts as a deadband, so classification is the first policy input and nothing is
forced into a continuous form for symmetry. A regression target supplied while
the mapping is off holds with `no_policy_for_target`.

When enabled, `low_below` and `high_above` must both be stated for both score
targets. Both default to `null`, and enabling the mapping without stating them
is a configuration error naming the key, not a silent fallback. Boundaries are
validated finite, ordered, and inside the target's declared range; units come
from the target spec. `value < low_below` is `LOW`, `value > high_above` is
`HIGH`, and everything between — **inclusive of both boundaries** — is the
neutral region.

`require_interval_inside_band` (default `true`) additionally requires the
whole Milestone 7 prediction interval to lie in one region before that
region's state is used; an interval straddling a boundary reads as neutral.
That is a use of the interval's *width* as a deadband, not a re-derivation of
Milestone 7's acceptance rule, which the policy never recomputes.

**Consequence:** The continuous path is ready and its boundary behaviour is
tested, but no number in it was chosen by looking at synthetic output, and
turning it on is an explicit act that requires stating a scale.

---

### DEC-080: The Experimenter Lock and the Static Condition Are Different Things

**Date:** 2026-08-19
**Status:** Accepted

**Context:** The project plan requires both "experimenter can disable
adaptation" and "static and adaptive modes are clearly separated". One boolean
could serve both.

**Decision:** Two settings, two hold reasons, both recorded on every decision.

- `adaptation.enabled` is the **experimenter lock**: hold every window
  regardless of evidence. Reason: `adaptation_disabled`.
- `adaptation.experiment_mode` is the **experimental condition**, `static` or
  `adaptive`. Reason: `static_experiment_mode`.

A static condition that silently depended on the policy happening to propose
nothing would not be a static condition — it would be an adaptive condition
that produced no adaptations, which is a different thing and would be analysed
differently. The mode participates in the configuration fingerprint, so a
static run and an adaptive run get different run ids and cannot be confused in
the artifact directory.

**Consequence:** A static-versus-adaptive experimental design has the
separation it needs at the software level. It does not have participants,
approval, or any evidence of benefit, and Milestone 8 makes no claim that it
does.

---

### DEC-081: The Adaptation Trace Carries No Wall Clock

**Date:** 2026-08-19
**Status:** Accepted

**Context:** Every other run directory in this repository records timestamps
in its tables. Doing the same here would have made the trace differ between
two runs of one configuration, which turns a determinism check into a
field-by-field comparison with timestamp exclusions.

**Decision:** `adaptation_trace.parquet` has **no wall-clock column**. Every
value in it is a function of the inputs, the configuration, and the initial
state. Timestamps live in `adaptation_summary.json`, where they are
provenance rather than data.

Identifiers follow: `proposal_id` is a digest of the session, window, order,
direction, current and proposed level, and the configuration fingerprint;
`command_id` is derived from it; `run_id` is a digest of the configuration
fingerprint, the evaluation mode, the input sequence's identity, and the
package version. No clock and no random component participates in any of them.
The command's `issued_at_utc` is supplied by the caller, because the policy
reads no clock at all.

**Consequence:** Two runs produce byte-identical Parquet, so the determinism
requirement is checked with `sha256sum`. A deterministic `command_id` also
means a retransmitted command is absorbed by the task client's existing
idempotency rule instead of double-stepping the difficulty.

---

### DEC-082: Milestone 8's Metrics Describe a Controller, Not a Person

**Date:** 2026-08-19
**Status:** Accepted

**Context:** An adaptation layer invites outcome language. "Adaptations per
session", "reversal rate", and "action frequency" are one sentence away from
"the policy stabilised engagement", and a synthetic run would supply the
numbers to say it.

**Decision:** Every reported quantity counts something the **software** did,
and `AdaptationControllerMetrics` carries a note saying so that travels with
any number lifted out of it. The schema validates that the counts reconcile —
holds plus proposals equal evaluated windows, increases plus decreases equal
proposals, eligible plus blocked equal evaluated windows, and proposals never
exceed eligible windows — so an inconsistent summary cannot be written.

Improved engagement, reduced cognitive load, learning improvement, comfort,
therapeutic effect, and adaptation effectiveness are **not** reported, and a
test asserts that no metric field name contains them.

The optional comparison against a guard-free controller (dwell 1, no cooldown,
no budget) is labelled a SOFTWARE CONTROLLER comparison. It shows that the
temporal guards mechanically reduce action frequency and nothing else. The
conservative policy was not tuned to win it, and neither controller is
described as better.

**Consequence:** Milestone 8 produces a defensible statement — the controller
behaves as specified on a fixed input sequence — and no statement at all about
whether adapting helps anyone. That question needs participants, and the
completion wording says so.

---

### DEC-083: The Milestone 9 Dashboard Is Read-Only Observability, Not Live Monitoring

**Date:** 2026-08-22
**Status:** Partially superseded by DEC-090 (2026-08-25)

**What still stands:** the dashboard is read-only, loads no model, opens no
camera, runs no inference, and holds no transport client. That is the whole
point of the milestone and is unchanged.

**What was wrong:** the conclusion that the plan's real-time and replay
modes therefore could not be delivered. It conflated *live monitoring of a
person* with *live observability of a recording*. The second needs no model,
no camera, and no inference, and the project plan required it. See DEC-090.

**Context:** `docs/PROJECT_PLAN.md` describes Milestone 9 as a "Streamlit
dashboard with all core monitoring pages, real-time and replay modes", and
`docs/PROJECT_SPECIFICATION.md` lists a "Live session" page. A live page
would need an online inference path: a camera opened, features extracted, a
model loaded, an estimate produced, and a number shown beside a person's
face, updating every second.

**Decision:** Milestone 9 is **read-only research observability over
persisted artifacts**. It discovers experiment runs, reads their JSON and
Parquet documents, and renders what those documents recorded. There is no
live mode, no inference, no camera, no model loading, and no transport
client.

Two reasons. First, a live engagement readout is the single most misreadable
artefact this project could build: no validated participant-labelled dataset
exists, so a number updating beside a person would be a claim about that
person that nothing here supports. Second, the useful research question at
this point is not "what is the estimate now" but "what did that run actually
record, and how far can it be trusted" — which is a question about
artifacts.

The read-only boundary is structural rather than advisory. AST tests assert
that no dashboard module writes, deletes, retrains, recalibrates, dispatches
an adaptation, opens a model pickle, imports the transport or API layer, or
performs a Git operation.

**Consequence (as originally recorded):** the plan's live and replay pages
were not delivered. This consequence no longer holds; both are delivered as
read-only presentation modes, under DEC-090.

---

### DEC-084: Run Family Comes From Artifact Signatures, Never From Directory Names

**Date:** 2026-08-22
**Status:** Accepted

**Context:** The local artifact root holds directories named `m5-*`, `m6-*`,
`m7-*`, `m8-*`. Parsing that prefix would classify every run in one line.
It would also be wrong the first time somebody renamed a folder, copied one
for comparison, or created `m7-scratch` and left it empty — and the failure
would be silent, presenting one milestone's artifacts under another
milestone's semantics.

**Decision:** A run's family is determined by the artifacts it contains.
Each family declares the documents only it writes: `adaptation_summary.json`
plus `adaptation_policy_config.json` for adaptation, `uncertainty.json` plus
`uncertainty_config.json` for uncertainty, and so on. Baseline additionally
declares a *disqualifying* set, because every training family writes
`manifest.json`, `metrics.json`, and `splits.json`; "baseline" means "a
training run carrying none of the later milestones' documents".

Detection runs in two passes. The first requires every distinguishing
artifact. The second accepts any of them, so a fusion run interrupted before
`fusion_metrics.json` is reported as an **incomplete fusion run** rather
than as an unclassifiable directory — a reader needs to know which run
failed. A directory matching neither pass is `unknown`, which is a real
answer and better than a guess.

Where a manifest records `configuration.milestone` or `configuration.kind`,
it is cross-checked. A disagreement raises a visible error; the signature is
used and the conflict is not resolved silently.

Directory names are shown in the run selector as display metadata.
Filesystem modification time is never used as provenance.

**Consequence:** A renamed, copied, or empty directory is classified by what
is in it. `m5-scientific-refusal` and `m7-invalid-scientific` — two empty
directories in the local artifact root — are reported as `unknown`, which is
exactly what they are.

---

### DEC-085: One Visualization Library, and It Is the Framework's Own

**Date:** 2026-08-22
**Status:** Accepted

**Context:** `docs/ARCHITECTURE.md` previously listed "Streamlit, Plotly"
as the dashboard stack. Plotly would give interactive hover, zoom, and
selection.

**Decision:** Streamlit only. Charts use `st.line_chart`, `st.bar_chart`,
`st.scatter_chart`, and `st.dataframe`. Plotly, Altair, matplotlib, and
Bokeh are not added.

The plots this dashboard needs are line charts of recorded curves, scatter
plots of stored predictions, histograms of stored columns, and confusion
matrices — all of which the native charts draw. Adding a second charting
library would mean two sets of theming, two accessibility stories, and two
places for an axis label to be wrong, in exchange for hover text. Every
chart already carries a table of its plotted values beneath it, which serves
the same inspection need and is readable by assistive technology.

Streamlit ships Altair and pydeck as its own transitive dependencies. No
EngageVR module imports either.

**Consequence:** One dependency was added for Milestone 9: `streamlit`.
Interactive zoom is unavailable; the value table beneath each chart is the
substitute.

---

### DEC-086: Absence Crosses Into the Presentation Layer as a Typed State

**Date:** 2026-08-22
**Status:** Accepted

**Context:** Every metric in this repository can legitimately be `None`: a
fold whose prerequisites were unmet, a conformal quantile with too few
calibration points, a modality with no usable window. Rendering those as
`0` is one `float(value or 0)` away, and on a metrics page *not computable*
and *very bad* are opposite readings that would then be indistinguishable.

**Decision:** Every number reaching a page is a `MetricDisplayValue`, which
holds either a value or an `unavailable_reason` and refuses to hold both or
neither. A non-finite value is **refused at construction**: `NaN` printed as
a number reads as a measurement, and `inf` reads as a very large one. A
genuine `0.0` survives as `0.0`.

Formatting lives in exactly one module. Probabilities, percentages, counts,
and interval widths are separate display kinds: a percentage carries `%`, a
probability does not, a count has no decimals, and an interval width carries
the regression target's units and is never given a percent sign, because it
is not a probability and is not confined to `[0, 1]`.

Table truncation is recorded and stated, never silent. A table that quietly
stops at row 1000 reads as a complete table.

**Consequence:** A model whose folds all failed shows a column of
*Unavailable*, not a column of zeros that reads as the worst possible score.

---

### DEC-087: The Uncertainty View Cannot Hold the Other Task's Controls

**Date:** 2026-08-22
**Status:** Accepted

**Context:** DEC-072 established that classification and regression are
selective on different axes that move in opposite directions and never share
a grid. A dashboard is where that separation is most likely to erode: one
"uncertainty" page, one threshold slider, one coverage curve, and the
distinction is gone — with `1 - interval_width` as the obvious bridge.

**Decision:** `UncertaintyDashboardData` **refuses** to be constructed with
a calibrated-confidence, probability-margin, or confidence-curve field when
`task_type == "regression"`, and refuses interval fields when
`task_type == "classification"`. The page hides the other task's controls
rather than showing them disabled, because it structurally cannot carry
them.

Each axis is rendered with its own name, units, and monotonicity contract.
Neither is relabelled "uncertainty threshold". `1 - interval_width` is never
computed anywhere in the dashboard.

A recorded curve whose `axis` field disagrees with the task type is not
displayed, with the disagreement stated. A curve written before DEC-072 —
which carries no `axis` field at all, because the two axes then shared one
grid — is likewise refused, because which axis it was swept over cannot be
established and must not be guessed. The local `m7-*` artifacts predate
DEC-072 and are handled by exactly this path.

**Consequence:** A regression page shows no confidence control and a
classification page shows no interval control, and neither is a matter of
page-authoring discipline.

---

### DEC-088: Provenance Is Carried, Not Reconstructed

**Date:** 2026-08-22
**Status:** Accepted

**Context:** A dashboard derives: it filters, aggregates, bins, and plots.
Each derivation is an opportunity to build a fresh object describing the
result and to leave the synthetic flag behind — after which a chart of
synthetic numbers is a chart with no provenance at all.

**Decision:** `DashboardProvenance` is constructed once per run, in the
catalogue, and carried into every view. It refuses
`scientific_evaluation_eligible=True` when the artifact says the data is
synthetic, and its `derive()` method refuses to change either flag, so a
derived view cannot become eligible by being plotted.

Every result-bearing page renders the provenance banner **at the top** — not
in an expander, not in a footer. For a synthetic run it shows the
software-self-check banner, and it states the eligibility flag in words.
There is no green "validated" styling anywhere: software checks passing is
not validation.

No dashboard control can reach the provenance fields. There is no "Mark as
scientifically validated" action, no "Treat as real" switch, and
`DashboardConfig` carries no such key — its `extra="forbid"` means one
cannot be added by configuration either. Where two recorded sources
disagree, the run is shown as corrupt with the contradiction stated.

**Consequence:** Provenance survives every derivation the dashboard
performs, and a test suite checks that a page cannot omit the banner.

---

### DEC-089: The Adaptation Page Has No Field an Effectiveness Claim Could Occupy

**Date:** 2026-08-22
**Status:** Accepted

**Context:** DEC-082 established that Milestone 8's metrics describe a
controller. A dashboard reintroduces the temptation in a new form: a metric
card labelled "Adaptation effectiveness" needs no new computation, only a
new caption over numbers that already exist.

**Decision:** `AdaptationDashboardData` has no effectiveness, benefit,
improvement, or success-rate field, and `extra="forbid"` prevents adding
one. The page reports evaluated windows, gate outcomes, hold reasons,
proposals, spacing, guards, and the difficulty trace — all of them
statements about software.

The lifecycle is five separate counts, never summed: proposal, command
built, dispatched, acknowledged, applied. `AdaptationLifecycleCounts`
refuses an ordering that could not have happened — a command without a
proposal, a dispatch exceeding the commands built, an acknowledgement
without a dispatch. For the current Milestone 8 runs the page shows 19
proposals, 19 commands built, 0 dispatched, 0 acknowledged, and says in
words that nothing reached a running environment.

The guard-free comparison is titled "software-controller action-frequency
comparison" and captioned with the denial that either controller is better,
safer, or more effective.

**Consequence:** A reader can see how often the controller acted and which
guard stopped it when it did not, and can find nowhere on the page a
statement that acting helped anyone.

---

### DEC-090: Real-Time and Replay Are Read-Only Presentation Modes, Not Live Inference

**Date:** 2026-08-25
**Status:** Accepted
**Supersedes:** the delivery conclusion of DEC-083

**Context:** `docs/PROJECT_PLAN.md` §"Milestone 9" requires "real-time and
replay modes", and its acceptance criteria require that live data be
visually labelled and that a session report be reproducible. DEC-083
declined to build either, on the grounds that a live engagement readout
beside a person's face would be a claim nothing in this repository supports.

That reasoning was right about the danger and wrong about the requirement.
It read "real-time mode" as *live monitoring of a person* when the plan only
needs *live observability of what has been recorded*. Milestone 4 already
writes an append-only session recording; observing it needs no model, no
camera, no inference, and no second protocol.

**Decision:** Milestone 9 has three evidence modes, explicitly separated in
the UI and in the types.

- **Experiment artifacts** — the Milestone 5-8 observatory, unchanged and
  still primary.
- **Live session** — a read-only reader over an existing session recording,
  re-read on request, presenting records the recorder had already persisted.
- **Session replay** — read-only navigation through a recording already
  complete or interrupted.

The prohibitions of DEC-083 are unchanged and are extended to cover the new
code: no model runner, no calibration, no inference, no policy evaluation,
no command dispatch, no artifact or recording mutation, no simulator, no
replay transmitter, no transport client, and no second protocol. A live
display shows no engagement value, no cognitive-load value, no confidence,
and no abstention, because a session recording structurally cannot carry
one; each is stated as *Unavailable* with the reason.

A live data source is not evidence. `data_source = live` says where bytes
came from, not that a study was designed, labelled, approved, or validated,
and the session format declares no scientific eligibility at all — so every
session is presented as ineligible with that stated as the reason.

**Consequence:** the plan's Milestone 9 objectives are met without an online
inference path. The read-only boundary is enforced by extended AST tests
(`SessionRecorder`, `JsonlWriter`, `ReplayPlayer`, `publish`, `connect`,
`asyncio`, socket modules, the simulator, and the replay player are all
forbidden) and, behaviourally, by tests that digest every file of a
recording before and after a full inspection.

---

### DEC-091: A Recorded Session Is Not an Experiment Run, and Has Its Own Catalogue

**Date:** 2026-08-25
**Status:** Accepted

**Context:** the dashboard now discovers two kinds of thing: Milestone 5-8
experiment runs and Milestone 4 session recordings. It would have been less
code to put both in one catalogue with a `kind` field.

**Decision:** they are separate types (`DashboardSessionCatalogue`,
`DashboardSessionSummary`, `DashboardSessionProvenance`,
`DashboardReplayState`, `DashboardSessionReport`), scanned from separate
roots (`dashboard.artifact_root`, `dashboard.session_root`), by separate
modules.

Their provenance contracts genuinely differ. An experiment run declares
`scientific_evaluation_eligible`, a dataset fingerprint, a split manifest, a
target, and a task type. A session recording declares none of those and
instead carries per-message provenance, arrival ordering, and receiver
anomalies. One type covering both would need every field optional, at which
point neither contract is checkable — and a task recording could appear in
a run selector, which is the point at which a transport log starts reading
like an experiment result.

**Consequence:** more types and no shared catalogue code. In exchange, a
recording can never be listed as a run, and the eligibility rules of each
are enforced separately by their own validators.

---

### DEC-092: Tail Safety Is Its Own Taxonomy, Separate From "Corrupt"

**Date:** 2026-08-25
**Status:** Accepted

**Context:** the artifact catalogue calls an unparseable document *corrupt*
and stops displaying its run. Applied to a session being appended to, that
rule marks every live recording corrupt, because the final line usually has
no terminating newline yet.

**Decision:** the session reader distinguishes five states, and none of them
is the run catalogue's `corrupt`:

1. a complete line that decodes — presented;
2. a complete line that will not decode — **kept visible** with its 1-based
   line number, a problem code, and the reason;
3. an incomplete trailing line — reported as **transient**, not counted, not
   parsed, and explicitly not called corruption;
4. an interrupted session (no summary) — *active or incomplete*, fully
   inspectable;
5. an absent stream or a removed directory — its own status with a reason.

A final complete line that happens to lack its newline is indistinguishable
from a torn one, so it is reported as partial; that resolves on the next
read.

`SessionStore` was not reused for this. Its `iter_messages` raises on the
first bad line, which would blank a live view because a writer was
mid-flush, and its `recover` is whole-file and not incremental. The reader
does reuse the store's layout constants and its session-id allowlist — the
pure parts — and the protocol's own `decode_stored_message`, so a record
this dashboard calls sound is sound by the project's definition.

**Consequence:** a malformed interior line is never silently swallowed and a
transient tail never reads as damage. The cost is a second parser, kept
narrow and covered by tests that construct each of the five states.

---

### DEC-093: The Session Report Is a Pure Function With a Content Fingerprint

**Date:** 2026-08-25
**Status:** Accepted

**Context:** the project plan requires that a session report "can be
reproduced". A report containing its own export timestamp is unique on every
export, which makes reproduction unobservable.

**Decision:** `build_report` is a pure function of an already-completed
read. `report_fingerprint` is a SHA-256 over the canonical report content
with exactly two fields excluded: the fingerprint itself and
`exported_at_utc`. Wall-clock time is therefore explicitly outside the
report's logical identity, and the same recording reported twice yields
byte-identical JSON and Markdown.

A **partial** read cannot produce a report at all. Counts taken from half a
file read exactly like counts taken from all of it, so the builder refuses
rather than reporting.

Provenance is not removable: `is_synthetic`,
`scientific_evaluation_eligible` (always false, with its reason), the
standing disclaimer, and — for a synthetic recording — the
software-self-check banner are required fields whose validators refuse a
report that drops, rewords, or contradicts them. There is no "clean" export.

The report is **printed or downloaded, never saved by this repository**.
Writing it would be the one file operation Milestone 9 does not have. It
carries a SHA-256 of every source file so a reader can confirm afterwards
that inspecting a recording did not change it.

**Consequence:** reproduction is a testable property rather than a claim,
and a report cannot be laundered into looking like validated evidence.

---

### DEC-094: The Live View Refreshes Automatically, at a Conservative Interval

**Date:** 2026-08-25, revised 2026-08-27
**Status:** Accepted (revises the original "the live view does not poll")

**Context:** `docs/PROJECT_PLAN.md` §"Milestone 9" requires a **real-time**
mode. The original DEC-094 made refresh explicit — press *Read new records*
— which is a *current* view, not a real-time one. Three worries drove that:
filesystem traffic, a moving number beside a person's session, and the rule
that no automated test may depend on a long-running process.

The first is answered by a floor rather than by having no timer at all. The
second is a worry about *what* is displayed, not about *when* it is redrawn:
a page carrying no engagement value, no cognitive-load value, no confidence,
and no person-status indicator does not acquire one by being redrawn. The
third turned out to rest on a false assumption — the cadence can be verified
where it is configured, without waiting for a firing.

**Decision:** the live-observation page refreshes automatically, using
Streamlit's native `st.fragment(run_every=...)` — available in the installed
Streamlit 1.62, so no dependency was added for it. The interval is
`dashboard.live_refresh_seconds`, default 5s, floor 2s. Each firing re-reads
the recording with the read-only session reader and redraws what it finds.
The header states `Mode: LIVE OBSERVATION` and `Automatic refresh: every N
seconds`.

The manual *Read new records* button remains as an additional action; both
paths go through the same body.

**Scope, which is the whole point of the decision:**

- **Only the live page.** Replay never auto-advances — its cursor moves only
  when one of its own controls is used — and the artifact observatory never
  polls. The fragment is constructed inside `live_session_page` and nowhere
  else, and a test asserts `run_every` occurs exactly once in the package.
- **Real-time observation is not real-time inference.** What refreshes is a
  view of records another subsystem already wrote. No model is loaded, no
  camera opened, no estimate produced, nothing sent, nothing written, on any
  cadence. A session recording structurally cannot carry an engagement or
  cognitive-load value, so a faster refresh could not produce one.
- **The interval is validated, not clamped.** `live_refresh_interval` refuses
  a non-numeric, non-finite, zero, negative, or sub-2s value and the timer
  does not start, with the reason shown. Clamping would leave a page
  refreshing at a cadence nobody chose, which is the sort of unattributed
  behaviour a read-only observability tool must not have. The page still
  renders and its manual control still works.
- **Nothing is cached between passes.** The session catalogue and every
  session read stay uncached: the experiment catalogue's modification-time
  key is sound for runs, which are written once, but an append to a recording
  leaves its directory's modification time untouched, so the same key would go
  stale in exactly the mode that must not — and a cached read would give a
  live view that cannot show an appended record.

Nothing is interpolated or fabricated between two passes. A recording that
*shrank* between passes is still reported as an error, because an
append-only file cannot shrink on its own.

**Consequence:** the plan's *real-time mode* is met by a mode that actually
refreshes, and the honesty the original decision was protecting is now
carried by the scope statements and the read-only boundary rather than by
the absence of a timer. A researcher watching a session sees new records
appear without pressing anything, and still sees `Mode: LIVE OBSERVATION`
and the standing disclaimer above them.

---

### DEC-095: Synthetic, Public, and Live Are Labelled in Words, and Tested With Fixtures

**Date:** 2026-08-27
**Status:** Accepted

**Context:** `docs/PROJECT_PLAN.md` accepts Milestone 9 only when "synthetic,
public, and live data are visually labelled". The dashboard did display the
recorded `data_source`, but as the raw string — `public_dataset` in a
monospaced cell — and only the synthetic case had ever been rendered,
because no public dataset and no live participant recording exists in this
repository. "There is nothing to label" is not evidence that the labelling
works; it is evidence that it has never been exercised.

**Decision:** two things.

*The label is rendered in words, beside the recorded value.*
`presentation.data_source_label` produces `PUBLIC (recorded as
'public_dataset')`, `LIVE (recorded as 'live')`, `SYNTHETIC (recorded as
'synthetic')`, `MIXED`, `UNRECOGNISED (recorded as '…')`, or `Unavailable —
no data source is recorded`. The recorded value is never replaced, only
accompanied. `data_source_statement` renders what that source does and does
not establish, and both appear on the artifact banner, the session banner,
the session data-source table, the session catalogue, and the dataset page.

The vocabulary is the project's own `engagevr.schemas.session.DataSource`
enum. No provenance string is invented, and a test asserts every member has
a label and a statement.

*All three cases are exercised with temporary fixtures.*
`tests/unit/test_dashboard_provenance_labels.py` builds a synthetic run, a
`public_dataset` run, and a `live` recording, and renders each through the
real pages via `AppTest`.

**Neither public nor live implies scientific eligibility.** `data_source`
and `scientific_evaluation_eligible` are independent fields, and the tests
assert a public run and a live recording both stay visibly ineligible with
the reason stated, and that no provenance mode drops the standing
disclaimer.

**Consequence:** the acceptance criterion is met by demonstration rather
than by the absence of a counter-example, and the day a public corpus or a
real live recording arrives, the surface that must label it has already been
rendered.

---

### DEC-096: MLflow Is Opt-In, Local, and Skinny

**Date:** 2026-08-29
**Status:** Accepted

**Context:** `docs/PROJECT_PLAN.md` requires MLflow for Milestone 10.
Milestone 5 deferred it with a stated reason (`docs/EXPERIMENT_TRACKING.md`,
"Why not MLflow yet"), and three constraints survived into this milestone:
the layer must be local-first with no account and no server; running an
ordinary Milestone 5-8 command must not start writing to a tracking store as
a side effect; and the modelling code, written against pandas 3, must not be
disturbed to accommodate a bookkeeping layer.

Measured, not assumed: `uv add mlflow` resolves the full distribution, which
pins `pandas<3` and downgrades this project's pandas from 3.0.3 to 2.3.3 — a
major-version downgrade of the library the Milestone 5-9 code is written
against. It also pulls Flask, gunicorn, SQLAlchemy, Alembic, Graphene, the
Docker SDK, and matplotlib: a server stack for a milestone that must need no
server.

**Decision:** three things.

*The dependency is `mlflow-skinny>=3.15,<4`.* It is the same project's
tracking client without the server stack. `import mlflow` works,
`mlflow.tracking.MlflowClient` works, a local file store works. What it
cannot do — serve a UI, use a database backend — is not required. One
consequence is recorded rather than hidden: `mlflow-skinny` caps
`protobuf<7`, so protobuf moved 7.36.0 to 6.33.6. That was verified against
the whole existing suite before the dependency was kept (3241 passed, 1
skipped, unchanged); mediapipe declares no protobuf constraint and streamlit
allows `>=5.26.1,<8`.

*Tracking is opt-in.* `mlops.mlflow.enabled` is `false`. No Milestone 5-8
runner imports the adapter, calls it, or starts a store, and importing
`engagevr.mlops.mlflow_tracking` does not import `mlflow` — the client is
imported inside the functions that need it, and a test asserts this in a
subprocess. The Milestone 5 tests asserting no `mlruns`, `MLmodel`, or
`meta.yaml` appears in a run directory still pass unchanged.

*The store is `mlruns/`, not `artifacts/mlflow/`.* MLflow's file store
rejects any run directory with a path component named `artifacts` — a
path-traversal defence — and then reports the runs it just wrote as "not
found". `mlruns/` is MLflow's own convention and was already gitignored.
`assert_usable_store` raises a legible error rather than leaving a reader to
decode MLflow's. MLflow 3.15 also puts the file store in maintenance mode,
so the adapter sets `MLFLOW_ALLOW_FILE_STORE` and `MLFLOW_DISABLE_TELEMETRY`
for the duration of one client call and restores the previous values,
including their absence, on the way out.

**Consequence:** tracking works from a clean clone with no account, no
server, no database, and no network, and it happens only when somebody asks
for it. The `<4` bound is the guard: if MLflow 4 removes the file store,
this project does not follow it silently.

---

### DEC-097: A Model Version Is a Manifest, Not a Registry Entry

**Date:** 2026-08-29
**Status:** Accepted

**Context:** `docs/PROJECT_PLAN.md` requires "model artifact and
configuration are versioned". The obvious reading is MLflow's Model
Registry, which is built around a *stage* — `Staging`, `Production`,
`Archived` — or a named alias such as `champion`. Every one of those words
records a decision somebody made about a model. Nobody has made one here: no
model in this repository has been evaluated against a real participant
label, and a registry entry would read as an approval that does not exist.

**Decision:** version with an immutable, checksum-linked
`ModelVersionManifest` and register nothing.

The record answers three questions and refuses a fourth. *Where did this
come from?* — source run id and family, dataset fingerprint, split
fingerprint, feature-schema fingerprint, the embedded configuration version,
fold index, calibration state, library versions. *Which bytes is it?* —
SHA-256 and size of the `.joblib`, plus the run's own recorded digests for
the documents it depends on. *What may be said about it?* — evaluation mode,
synthetic status, scientific eligibility, disclaimers, and a limitation
paragraph. *Should it be used?* — **there is no field**, and a test asserts
that `stage`, `alias`, `status`, `promoted`, and `approved_by` are all
absent from the schema.

`model_version_id` is `mv-<target>-<model>-<sha256(...)[:12]>` over exactly
the content above. No wall clock and no random component participates, so
re-deriving a version from the same run reproduces the identifier.

Three safety properties are enforced rather than documented. Nothing is
unpickled: a `.joblib` is executable content, so the layer hashes bytes and
a test monkeypatches `joblib.load` to raise. The producing run is not
modified: a test hashes the whole directory before and after. Tampering is
refused at build time rather than certified.

`FORBIDDEN_STATUS_WORDS` — production, staging, champion, challenger,
approved, validated, certified, clinical, diagnostic — is rejected at the
schema boundary in identifiers, model names, experiment names, run names,
and every tag value that is not itself a disclaimer.

**Consequence:** a release can state exactly which bytes were fitted, from
which data, under which configuration, and verify it — without any record
implying that a person decided the model was fit for anything.

---

### DEC-098: Configuration Identity Is the Normalized Effective Configuration

**Date:** 2026-08-29
**Status:** Accepted

**Context:** `docs/PROJECT_PLAN.md` requires configuration versioning.
Recording the filename `configs/defaults.yaml` versions nothing: the file
can change between two runs that both name it, and the YAML omits every
default, so it is not even a complete description of what ran.

**Decision:** record the **normalized effective configuration** — the
resolved Pydantic model rendered in JSON mode, so every default is present —
together with a SHA-256 over its canonical form and a snapshot of the
sections that shape a run.

Three paths are removed before hashing, each with a reason stored in the
record itself, and `ConfigurationVersion` refuses to validate if an excluded
path has no stated reason:

- `rppg.datasets.ubfc_rppg_root` — an absolute path to a locally obtained
  public dataset, different on every machine, never fetched by this
  software, and taking no part in any synthetic pipeline;
- `logging.file` — a local log destination, which changes where diagnostics
  are written and nothing about what was computed;
- `capture.camera_index` — a device index identifying one webcam on one
  machine, read by no modelling, tracking, or packaging stage.

Nothing that can change a number is excluded. The exclusions exist so that
two identical runs on two machines agree, which is the fingerprint's only
job.

Two companions use the same convention. The **split fingerprint** covers the
strategy, group field, split count, seed, and each fold's *sorted* group
membership, and deliberately excludes row counts and target distributions:
those are consequences of the group assignment, and including them would
make a dataset that merely grew look like a different split. The
**feature-schema fingerprint** covers the catalog version and the *ordered*
predictor columns — order participates, because a linear model's
coefficients are read positionally.

**Consequence:** "which configuration produced this?" has an auditable
answer that survives being copied to another machine, and a settings change
that could alter a result changes the fingerprint.

---

### DEC-099: Drift Is a Distribution-Shift Diagnostic, Never a Diagnosis

**Date:** 2026-08-29
**Status:** Accepted

**Context:** `docs/PROJECT_PLAN.md` requires drift checks. In a project that
estimates engagement and cognitive load, "drift" is a word one slip away
from a claim about a person — participant drift, disengagement drift,
cognitive decline — and one slip away from a claim about a model, since a
threshold crossing reads naturally as a failure.

**Decision:** implement a small, deterministic, interpretable
distribution-shift layer, and build the refusals into the schema.

*Five methods, not fifteen*, each answering a different question a reader
can check by hand: missingness-rate difference, standardized mean
difference (pooled), the two-sample Kolmogorov-Smirnov statistic,
Population Stability Index over quantile bins of the reference, and
categorical total variation distance. No p-value accompanies the KS
statistic: it would be a hypothesis test nobody specified in advance and
would grow significant with sample size alone.

*Both sides are named.* Reference and current are explicit arguments;
nothing guesses which two directories to compare. Each side records its
path, fingerprint, row count, data-source counts, and eligibility.

*Unavailability is never zero.* A column missing on one side, all-null, too
thin, constant, or of a mismatched type is reported unavailable with a
reason, and the schema refuses to let an unavailable statistic carry a
`statistic` or an `exceeded` verdict. Zero means "these distributions
agree", and collapsing the two would let an absent measurement read as a
healthy one.

*Targets never take part.* `target__*` and `target_meta__*` are excluded by
construction with "leakage" as the stored reason, so no shift statistic can
be computed from a label.

*There is no failure field.* Counts and per-feature, per-method statistics
with their thresholds beside them — no overall pass/fail, and a test asserts
`model_failed`, `alarm`, `passed`, `verdict`, and `status` are absent from
the schema. `--fail-on-shift` exists as an opt-in build gate and says so
when it fires.

*The vocabulary is bounded by the enum.* `DriftReportKind` has exactly
`feature_distribution_shift` and `prediction_distribution_shift`. There is
deliberately no `concept_drift`: establishing concept drift needs labels
from both periods, and no validated participant-provided label exists here.

Every threshold is an **engineering diagnostic default**, chosen for
legibility, and the report says so in a required field.

**Consequence:** the milestone delivers the drift check the plan asks for,
and the record it produces cannot be quoted as evidence that a model
degraded or that anyone's state changed.

---

### DEC-100: DVC Outputs Are Not Cached, and `dvc.lock` Is Tracked and Stable

**Date:** 2026-08-29
**Status:** Accepted

**Context:** `docs/PROJECT_PLAN.md` accepts Milestone 10 only when a "clean
clone can reproduce the demo". DVC's usual answer is a cache plus a remote:
commit `dvc.lock`, run `dvc pull`, restore the outputs. This repository has
no remote, wants none, and gitignores every artifact the pipeline produces.

`dvc.lock` also records hashes of those outputs, and they embed wall-clock
fields — run manifests carry `started_at_utc`, dataset metadata carries
`created_at_utc` — so the lock cannot match across two correct executions or
two machines.

**Decision:** declare every stage output `cache: false`, track `dvc.lock`,
and make it **byte-stable** so that tracking it costs nothing.

*Outputs are not cached.* The demo is **regenerated from source**, never
restored. There is nothing to `dvc pull`, no remote to configure, no binary
in Git, and a clean clone reproduces by running the pipeline. `dvc.yaml` and
`params.yaml` are the committed, reviewable definition of what the pipeline
is; `.dvc/config` (telemetry off, auto-staging off, version check off),
`.dvc/.gitignore`, and `.dvcignore` are committed with them.
`autostage = false` matters here: DVC must never run `git add` on this
repository's behalf.

*`dvc.lock` is tracked and stable.* Gitignoring it was tried first and
rejected by DVC itself, which refuses to operate with an ignored lock
(`ERROR: 'dvc.lock' is git-ignored`). An earlier revision of this decision
then accepted a lock that churned on every fresh reproduction, on the
grounds that several declared outputs recorded their own creation time.
**That was wrong and has been corrected.** A tracked file that changes on
every run stops carrying information: "the lock changed" becomes noise, and
a reviewer loses the one signal that would have told them a pipeline output
genuinely moved.

The invariant now maintained:

```
clean source tree -> dvc repro -> dvc.lock byte-identical
```

given the same source, `uv.lock`, configuration, synthetic seed, and
parameters. The mechanism is the orchestration boundary in DEC-104, not the
removal of timestamps from anywhere they belong. Verified twice: two
consecutive fresh reproductions in the working tree produce the same lock,
and two independent source-only trees — each synced from `uv.lock` and
reproduced from scratch — produce that same lock as each other. A second
`dvc repro` in each skips all eight stages.

No digest is recorded in this decision. The lock depends on the source by
design: change a declared dependency and it changes. The invariant is that
two reproductions *from the same source* agree, which `make dvc-verify`
checks in one command.

Measured consequence: a second `dvc repro` still does no work — all eight
stages report "didn't change, skipping" and `dvc status` reports "up to
date" — because DVC compares dependencies, parameters, and commands, none of
which carries a timestamp. CI asserts it.

**Consequence:** reproduction needs a clone, a lock file for *dependencies*
(`uv.lock`), and one command. It never needs storage credentials, and it
never leaves a dirty working tree. A modified `dvc.lock` after a
reproduction now means something changed — which is what a tracked lock is
for.

---

### DEC-101: Reproducibility Is Logical, Not Byte-for-Byte

**Date:** 2026-08-29
**Status:** Accepted

**Context:** Two correct executions of the demo do not produce identical
bytes. Run manifests record `started_at_utc` and `finished_at_utc`; dataset
metadata records `created_at_utc`; model-version records and drift reports
each record when they were built. A reproducibility check that compared
whole directories would fail every time and teach a reader to ignore it.

**Decision:** `ReproducibilityManifest` separates two things and hashes only
one of them.

*Logical identity* — dataset fingerprints, run identifiers, model-version
identifiers, the drift report fingerprint, the catalogue digest, and the
SHA-256 of every output **declared byte-deterministic** — is covered by
`logical_fingerprint`. That is what must match.

*Volatile record* — the checksum of every other output — is stored as
information beside a required `non_determinism_reason`. Useful for spotting
a file that changed when it should not have; never folded into identity.

`excluded_from_identity` is a required field naming what never participates:
`created_at_utc`, absolute filesystem paths, wall-clock fields inside run
manifests and dataset metadata, MLflow run and experiment identifiers, and
the host platform. Output paths are recorded relative to the pipeline root
for the same reason: an absolute path is a fact about one machine.

`BYTE_DETERMINISTIC_NAMES` is short and checked. A file added to it that
turns out to vary makes the two-execution comparison fail loudly rather than
quietly weakening the guarantee.

Verified: two executions from clean output directories produce identical
dataset fingerprints, run identifiers, model-version identifiers, report
fingerprints, and deterministic checksums; `repro-manifest --compare` exits
zero. CI runs the same comparison.

**Consequence:** the project claims a property it can actually demonstrate,
and the timestamps that would defeat a naive check are excluded by
construction rather than tolerated by convention.

---

### DEC-102: The Pipeline Root Is Outside the Dashboard Artifact Root

**Date:** 2026-08-29
**Status:** Accepted

**Context:** The Milestone 10 pipeline produces run directories that look
exactly like Milestone 5-8 runs, because they are: the stages call
`baseline-demo` and `uncertainty-demo`. Writing them into
`dashboard.artifact_root` (`artifacts/experiments`) would mean every
`dvc repro` silently reshuffled the catalogue a reader is looking at, and
that a `make clean`-style operation on pipeline output could take
Milestone 5-9 evidence with it.

**Decision:** the pipeline writes under `mlops.pipeline_root`
(`artifacts/pipeline`) and the smoke check under `mlops.smoke_root`
(`artifacts/smoke`), both inside the gitignored `artifacts/` tree and both
outside `dashboard.artifact_root`. Configuration validation refuses to let
the two roots be equal.

The Milestone 9 dashboard still reads them, because the artifact root is an
argument: `dashboard --artifact-root artifacts/pipeline/experiments`, and
the `integrity` stage does exactly that through `dashboard-check`. Nothing
about Milestone 9 changed.

`make clean-mlops` removes `artifacts/pipeline`, `artifacts/smoke`, and
`mlruns`, and deliberately nothing else — it never touches
`artifacts/experiments`, `artifacts/sessions`, `artifacts/datasets`, or the
tracked `dvc.lock`.
There is no `git clean` anywhere in this repository.

**Consequence:** the demo can be regenerated as often as anyone likes
without disturbing the evidence somebody is reading, and cleaning up after
it is a scoped operation rather than a hopeful one.

---

### DEC-103: Docker Packages the Existing Backend and Dashboard, and Adds No API

**Date:** 2026-08-29
**Status:** Accepted

**Context:** `docs/PROJECT_PLAN.md` accepts Milestone 10 only when
"Dockerized backend and dashboard work". A packaging milestone invites a
model-serving container — a `/predict` endpoint, a scoring API — because
that is what MLOps tutorials contain. This project has no validated model to
serve, and an inference endpoint would be an interface promising a
capability that does not exist.

**Decision:** two images, each running code that already existed.
`Dockerfile.backend` runs `python -m engagevr serve`, the Milestone 4
FastAPI and WebSocket bridge. `Dockerfile.dashboard` runs
`python -m engagevr dashboard`, the Milestone 9 Streamlit application. No
third image, no new endpoint, no second dashboard. A test asserts the words
`predict`, `inference`, `model-server`, and `/invocations` appear in no
instruction of either Dockerfile.

Both images pin `python:3.12-slim-bookworm`, install with
`uv sync --locked --no-dev` and never `pip install`, run as a non-root user,
expose exactly one port, and health-check against the application's own
liveness route using `urllib` so the image needs no extra package to check
itself. Neither uses BuildKit cache mounts: they would make the images
buildable only by a daemon with `buildx`, and a packaging milestone whose
images cannot be built by a plain `docker build` has packaged nothing.

`docker-compose.yml` publishes **both ports to `127.0.0.1` only**. Neither
service has authentication, authorisation, or transport encryption;
publishing either to a routable interface would expose an unauthenticated
bridge and a filesystem browser for the artifact root. The dashboard mounts
both roots read-only, which makes its read-only property a filesystem fact
as well as a code property.

`.dockerignore` excludes every generated, local, and private path, and the
exclusion is verified after building rather than assumed: both images
contain only source, configuration, the locked environment, and empty mount
points.

**Consequence:** the acceptance criterion is met by packaging what the
project has, and the images cannot be mistaken for a deployment of a model
that has never been evaluated against a person.

---

### DEC-104: Deterministic DVC Outputs Are Separated From Volatile Execution Metadata

**Date:** 2026-08-29
**Status:** Accepted

**Context:** DEC-100 requires `dvc.lock` to be byte-stable across fresh
reproductions. Measured, twenty of the pipeline's fifty-six files were not:
run manifests record `started_at_utc` and `finished_at_utc`, dataset
metadata records `created_at_utc`, `checksums.json` digests those documents
and inherits their instability, and the Milestone 10 records each stamped
their own build time.

Two repairs were available and one of them is wrong. Stripping timestamps
from the Milestone 5-8 artifacts would satisfy the build tool by damaging
scientific provenance: a run *did* happen at a time, and that is a fact a
research repository keeps.

**Decision:** put a boundary between the two, at the Milestone 10
orchestration layer.

```
existing runner output          timestamped, intact, NOT DVC-declared
          |
deterministic stage record      engagevr stage-record
          |
DVC-declared output             byte-stable, hashed into dvc.lock
```

*The runner artifacts are untouched.* `manifest.json` still records both
timestamps. What changed is that the run directory is no longer a DVC
output.

*A stage record is declared in its place.* `DeterministicStageRecord` pins
the stage's logical identity — the run id, itself a hash of the run's
inputs — and checksums every byte-stable file the run produced, models
included. The timestamped documents are listed by path **with the reason
they vary and without a checksum**, so their contents cannot reach the lock
while a reader can still see what was excluded and why.

*A meaningful change still propagates.* Alter `metrics.json` and the
record's checksum for it changes, so the record's bytes change, so
`dvc.lock` changes and every downstream stage re-runs. What no longer
propagates is the clock. A test asserts both halves.

*Milestone 10's own documents carry no wall clock at all.* Model versions,
the drift report, the reproducibility manifest, and the stage records have
no `created_at_utc` field. When each was produced is written to a
`<name>.execution.json` sidecar beside it, which is never declared —
`ExecutionMetadata` says so in a required field. Nothing was discarded;
provenance was moved to where it cannot make a reproducible pipeline look
irreproducible.

Three smaller rules follow from the same principle:

- **Python is recorded as a series** (`3.12`), not a patch level. The
  compatibility contract is the series; recording `3.12.13` would put every
  interpreter upgrade into the identity of every document. The full version
  is in the sidecar.
- **Recorded commands are pipeline-relative.** `stage-record` normalises the
  pipeline root out of the command it stores, so the same demo under
  `/tmp/scratch` and under `artifacts/pipeline` produces the same record.
  Without this a temporary directory would leak into a tracked identity.
- **`dataset.json` is not among a model version's referenced checksums.** It
  copies the dataset metadata verbatim, creation time included. The dataset
  is pinned by `dataset_fingerprint` instead, which excludes the wall clock
  by construction.

*Classification is explicit and fails safe.* `VOLATILE_ARTIFACT_REASONS`
names the timestamped documents. A file this repository has not classified
is treated as **deterministic** and checksummed, so if it turns out to vary
the two-execution test fails loudly rather than the guarantee weakening in
silence.

**Consequence:** the pipeline satisfies a reproducibility property it can
demonstrate byte for byte, and it does so without any Milestone 5-8 artifact
losing a field. Forty-seven regression tests
(`tests/unit/test_dvc_determinism.py`) hold the boundary in place, and an
opt-in two-source-tree proof
(`tests/system/test_dvc_lock_stability.py`) checks it end to end.
