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
