# EngageVR -- Progress Tracker

## Current Milestone: 5 -- Windowed Feature Datasets and Interpretable Baselines

**Status:** Milestone 5 baseline-model pipeline implementation complete;
scientific evaluation on real participant-labelled data pending.

## Milestone History

### Milestone 0: Repository Audit and Plan

**Started:** 2026-08-01
**Completed:** 2026-08-01

**Deliverables:**
- [x] Inspected repository structure and current state
- [x] Checked environment: Python 3.12.13 (venv), uv 0.11.32, git 2.55.0,
      ffmpeg 8.1.2, Docker 29.6.2, v4l2-ctl available
- [x] Reviewed `pyproject.toml` (minimal, no dependencies, uv_build backend)
- [x] Reviewed `docs/PROJECT_SPECIFICATION.md` (complete specification)
- [x] Defined software-only MVP boundary
- [x] Identified deferred functionality
- [x] Created `docs/PROJECT_PLAN.md`
- [x] Created `docs/ARCHITECTURE.md`
- [x] Created `docs/RESEARCH_QUESTIONS.md`
- [x] Created `docs/LIMITATIONS.md`
- [x] Created `docs/DECISIONS.md`
- [x] Created `docs/PROGRESS.md` (this file)

**Repository state at audit:**
- Package: `engagevr` 0.1.0, empty `src/engagevr/__init__.py` (hello stub)
- No dependencies declared
- No tests, configs, Makefile, CI, or documentation beyond the specification
- `data/` directories exist with `.gitkeep` files (raw, interim, processed, synthetic)
- `.python-version` pins 3.12
- `.venv` functional with Python 3.12.13

**Decisions recorded:** DEC-001 through DEC-010 (see `docs/DECISIONS.md`)

---

### Milestone 1: Foundation

**Started:** 2026-08-01
**Completed:** 2026-08-01
**Status:** Complete

**Deliverables:**
- [x] Updated `pyproject.toml` with dependencies (pydantic, pyyaml) and dev
      tools (ruff, mypy, pytest, pytest-cov, pre-commit, types-PyYAML)
- [x] Created `configs/defaults.yaml` with all configurable parameters
- [x] Created `src/engagevr/config.py` -- YAML config loading with Pydantic validation
- [x] Created `src/engagevr/_logging.py` -- structured JSON/text logging
- [x] Created `src/engagevr/utils/timestamps.py` -- UTC and monotonic clock utilities
- [x] Created `src/engagevr/schemas/session.py` -- session metadata, DataSource, ExperimentCondition
- [x] Created `src/engagevr/schemas/modality.py` -- Modality enum, ModalitySample
- [x] Created `src/engagevr/schemas/signal_quality.py` -- ModalityQuality, SignalQualityReport
- [x] Created `src/engagevr/schemas/prediction.py` -- EngagementPrediction with abstention
- [x] Created `src/engagevr/schemas/adaptation.py` -- AdaptationCommand, AdaptationEvent
- [x] Created `src/engagevr/schemas/events.py` -- BaseEvent, TaskEvent, EventType
- [x] Created `src/engagevr/simulator/synthetic.py` -- deterministic synthetic session generator
- [x] Created `Makefile` with install, format, lint, typecheck, test, check targets
- [x] Created `.pre-commit-config.yaml`
- [x] Created `.github/workflows/ci.yml`
- [x] Created `tests/conftest.py` and 5 test modules
- [x] Updated `README.md` with setup and development commands
- [x] Updated `.gitignore`

**Verification:**
- `uv sync` -- clean install (29 packages)
- `make format` -- all files formatted
- `make lint` -- all checks passed
- `make typecheck` -- no issues in 14 source files (15 after validation pass)
- `make test` -- 65 passed initially; 74 after validation pass

**Decisions recorded:** DEC-011 (see `docs/DECISIONS.md`)

**Milestone 1 validation pass (2026-08-01):**
- [x] Reviewed all source and test files
- [x] Removed unused `[[tool.mypy.overrides]]` for `tests.*` (eliminated mypy warning)
- [x] Fixed missing `---` separator between M1 and M2 in this file
- [x] Fixed stale `(current)` label on M0 in `docs/PROJECT_PLAN.md`
- [x] Added CLI entry point (`src/engagevr/__main__.py`) with `demo` subcommand
- [x] Added unit and smoke tests for the CLI (10 tests)
- [x] Added demo command to `README.md`
- [x] Recorded DEC-012, DEC-013

---

### Milestone 2: Webcam Behavioural Capture

**Started:** 2026-08-01
**Completed:** 2026-08-01
**Status:** Milestone 2 implementation complete; physical-webcam validation pending.

**Dependencies added:** numpy>=2, opencv-contrib-python>=4.9, mediapipe>=0.10.14

**Deliverables:**
- [x] `src/engagevr/capture/webcam.py` -- webcam acquisition with fake backend
- [x] `src/engagevr/capture/frame.py` -- BGR-to-RGB/gray conversion
- [x] `src/engagevr/capture/quality.py` -- brightness, blur, motion quality metrics
- [x] `src/engagevr/face/landmarker.py` -- MediaPipe FaceLandmarker wrapper
- [x] `src/engagevr/face/features.py` -- EAR, blink tracker, mouth aspect ratio
- [x] `src/engagevr/head_pose/estimator.py` -- PnP head-pose estimation
- [x] `src/engagevr/head_pose/features.py` -- angular velocity, motion variability
- [x] `src/engagevr/schemas/capture.py` -- frame, landmark, behavioural, quality schemas
- [x] Extended `configs/defaults.yaml` with face, head_pose, quality sections
- [x] Extended `src/engagevr/config.py` with FaceConfig, HeadPoseConfig, QualityConfig
- [x] Extended CLI with `capture` subcommand
- [x] `scripts/download_models.py` -- FaceLandmarker model download
- [x] 6 new test modules (66 new tests, 140 total)
- [x] Updated README.md, pyproject.toml, docs

**Model asset:**
- Name: face_landmarker.task (float16)
- Source: Google MediaPipe
- License: Apache 2.0
- Location: models/face_landmarker.task (gitignored)
- Download: `uv run python scripts/download_models.py`

**Verification:**
- `uv lock --check` -- resolved
- `uv sync --locked` -- clean
- `uv run ruff format --check .` -- 50 files formatted
- `uv run ruff check .` -- all passed
- `uv run mypy src` -- no issues in 26 source files
- `uv run pytest` -- 140 passed in 1.60s
- `uv run pre-commit run --all-files` -- all passed
- `make check` -- all passed

**Decisions recorded:** DEC-014, DEC-015, DEC-016, DEC-017 (see `docs/DECISIONS.md`)

**Hardware check:** No physical V4L2 webcam detected. All tests use
`FakeCaptureBackend`. Hardware smoke test skipped (marker: `@pytest.mark.hardware`).

**Milestone 2 validation pass (2026-08-01):**
- [x] Resolved OpenCV package conflict: removed `opencv-python-headless`,
      using `opencv-contrib-python` as single OpenCV distribution
- [x] Verified mediapipe 1.0.0 FaceLandmarker construction with downloaded model
- [x] Corrected status to "implementation complete; physical-webcam validation pending"
- [x] Repaired DEC-016, recorded DEC-017
- [x] Added opt-in hardware smoke test (`tests/hardware/test_webcam_smoke.py`)
- [x] Fixed model download command to `uv run python scripts/download_models.py`

---

### Milestone 3: Interpretable rPPG Signal-Processing Pipeline

**Started:** 2026-08-02
**Completed (implementation):** 2026-08-02
**Status:** Milestone 3 implementation complete; physical-webcam and
public-dataset validation pending.

**Dependencies added:** `scipy>=1.15,<2` (runtime, resolved to 1.18.0);
`scipy-stubs>=1.15,<2` (dev, for strict type checking). No other
dependency added.

**Deliverables:**
- [x] `src/engagevr/schemas/rppg.py` -- 9 typed schemas plus
      `UnavailableReason`, `RoiRegion`, `RppgMethod`
- [x] `src/engagevr/rppg/errors.py` -- typed unavailable-with-reason
- [x] `src/engagevr/rppg/roi.py` -- forehead + both cheeks, inset,
      frame-clipped, valid-pixel accounting
- [x] `src/engagevr/rppg/trace.py` -- timestamped RGB trace, timing
      diagnostics, deterministic SYNTHETIC generator
- [x] `src/engagevr/rppg/preprocessing.py` -- detrend, normalize,
      resample, Butterworth SOS band-pass via `sosfiltfilt`
- [x] `src/engagevr/rppg/methods.py` -- GREEN, CHROM, POS
- [x] `src/engagevr/rppg/heart_rate.py` -- Welch PSD peak estimation
- [x] `src/engagevr/rppg/quality.py` -- ~13-component quality index
- [x] `src/engagevr/rppg/window.py` -- orchestration and quality gate
- [x] `src/engagevr/rppg/evaluation.py` -- reference-only error metrics
- [x] `src/engagevr/datasets/base.py` -- abstract adapter interface
- [x] `src/engagevr/datasets/ubfc_rppg.py` -- UBFC-rPPG adapter
- [x] Extended `configs/defaults.yaml` and `config.py` with a validated
      `rppg` section (7 sub-sections, cross-field validators)
- [x] Extended CLI with `rppg-demo` and `rppg-evaluate`
- [x] 7 new test modules (279 new tests, 420 total)
- [x] `docs/DATASETS.md`, `docs/REFERENCES.md` created
- [x] Updated README, ARCHITECTURE, LIMITATIONS, DECISIONS

**Methods implemented (from primary references):**

| Method | Reference | DOI |
|--------|-----------|-----|
| GREEN | Verkruysse, Svaasand & Nelson (2008) | 10.1364/OE.16.021434 |
| CHROM | de Haan & Jeanne (2013) | 10.1109/TBME.2013.2266196 |
| POS | Wang, den Brinker, Stuijk & de Haan (2017) | 10.1109/TBME.2016.2609282 |

Equations, windowing, overlap assumptions, and every deviation from the
published algorithms are recorded in `docs/REFERENCES.md` and in the
function docstrings.

**Verification:**
- `uv lock --check` -- resolved
- `uv sync --locked` -- clean (53 packages)
- `uv tree` -- no duplicate or conflicting packages; one OpenCV wheel
- `uv run ruff format --check .` -- 75 files formatted
- `uv run ruff check .` -- all checks passed
- `uv run mypy src` -- no issues in 40 source files
- `uv run pytest` -- 419 passed, 1 skipped
- `uv run pytest -m hardware` -- 1 skipped (no physical webcam)
- `uv run pre-commit run --all-files` -- all passed
- `make check` -- all passed

**Acceptance criteria against `docs/PROJECT_PLAN.md`:**

| Criterion | Status |
|-----------|--------|
| Pipeline processes a known public sample | **NOT MET -- pending.** UBFC-rPPG is not present locally and is not downloaded by this software. |
| Signal quality is reported per window | Met. ~13 named components with raw values, documented aggregation, explicit gates. |
| Unreliable windows return `unavailable` | Met. Asserted by tests across ROI, timing, filter, spectral, and quality-threshold failures. |
| No fabricated accuracy claims | Met. Error metrics are computed only against genuine reference signals; synthetic recovery is labelled a self-check, not validation. |

**Hardware check:** No physical V4L2 webcam detected. The rPPG pipeline
has never been run on live camera frames.

**Public-dataset check:** UBFC-rPPG not present. Adapter tested against
temporary deterministic fixtures only. No dataset metric exists.

**Decisions recorded:** DEC-018 through DEC-024 (see `docs/DECISIONS.md`)

**Remaining validation for this milestone:**
1. Run `capture` + rPPG on a physical webcam once hardware is available.
2. Obtain UBFC-rPPG through the official channel, resolve its licensing
   (currently unverified), confirm the reference sampling rate, then run
   a real evaluation and record the metrics with full provenance.

---

### Milestone 4: Task Environment and Simulator

**Status:** Backend, Python simulator, shared protocol, session storage, and
replay **COMPLETE** (2026-08-02). Unity compilation and runtime validation
**PENDING** (no Unity Editor installed).

**Delivered:**

- **Shared versioned protocol `1.0`** (`src/engagevr/protocol/`): typed
  envelope, 14 message types, 5 sources, closed payload models, full inbound
  validation pipeline, generated JSON Schema, 19 valid and 12 invalid
  contract fixtures shared with the Unity test suite (DEC-025).
- **Task-event schema extended, not replaced**: 11 new `EventType` members,
  `TASK_EVENT_TYPES`, `ResponseOutcome`, and `TaskEventDetail` carrying the
  full identifier and response field set. Timeout is distinct from
  incorrect; a missing response stays `None` (DEC-033).
- **Deterministic Python task simulator** (`src/engagevr/task/`): seeded
  trial plan, injected clock and RNG, real-time / accelerated / immediate
  modes with identical event content, pause / disconnect / abort scenarios,
  graceful cancellation, in-process / file / WebSocket transports.
- **Local FastAPI backend** (`src/engagevr/api/`): 7 HTTP endpoints plus a
  manual-command endpoint, `/ws/v1/sessions/{session_id}`, handshake, typed
  acknowledgements and protocol errors, heartbeat round trips, observer
  broadcast, targeted command routing, lifespan-owned resources.
- **Bounded-queue backpressure** (DEC-027): critical messages never dropped,
  non-critical drops counted, logged, and written to `dropped.jsonl`.
- **Clock and ordering diagnostics** (`src/engagevr/synchronization/`,
  DEC-028, DEC-029): RTT-bounded offset *estimates*, seven anomaly types
  recorded and never repaired, arrival order and sequence order kept
  separate.
- **Session storage** (`src/engagevr/storage/`): append-only JSONL, atomic
  summary, crash recovery with 1-based malformed-line numbers, path-traversal
  rejection.
- **Deterministic replay** (`src/engagevr/replay/`): four pacing modes,
  filtering, in-process and WebSocket output, additive replay metadata
  (DEC-026), source recording never modified.
- **CLI**: `serve`, `task-sim`, `session-inspect`, `session-replay`.
- **Unity desktop task** (`unity/EngageVR/`): 10 C# source files, 3 test
  files, asmdefs, editor scene generator, dependency-free JSON with
  first-class `null` (DEC-030), `ClientWebSocket` transport (DEC-031),
  offline mock transport. **Not compiled, not executed** (DEC-032).
- **Documentation**: `docs/PROTOCOL.md`, `docs/TASK_SIMULATOR.md`,
  `docs/SESSION_FORMAT.md`, `docs/REPLAY.md`, `docs/UNITY_SETUP.md`.

**Dependencies added:** fastapi 0.141.1, uvicorn 0.52.1, websockets 17.0.1
(runtime); httpx2 2.9.1, pytest-asyncio 1.4.0 (development). Verified against
Python 3.12.13. No database, broker, or container dependency (DEC-036).

**Verification:**
- `uv lock --check` -- resolved
- `uv sync --locked` -- clean
- `uv tree` -- no duplicate or conflicting packages; one OpenCV wheel
- `uv run ruff format --check .` -- all files formatted
- `uv run ruff check .` -- all checks passed
- `uv run mypy src` -- no issues in 73 source files
- `uv run pytest` -- **747 passed, 1 skipped**
- `uv run pytest -m hardware` -- 1 skipped (no physical webcam)
- `uv run pre-commit run --all-files` -- all passed
- `make check` -- all passed
- End-to-end demo (serve -> task-sim over WebSocket -> inspect -> replay) --
  executed successfully; also covered by an automated test that starts a
  loopback server in-process.

**Milestone 4 acceptance criteria:**

| Criterion | Status |
|-----------|--------|
| 1. The backend works without Unity | **Met.** Every test runs with no Unity. The full demo is Python-only. |
| 2. The Python simulator and Unity use the same versioned protocol | **Partially met.** One protocol definition, one generated schema, one fixture set parsed by both test suites. The Python half is verified; the Unity half is **unverified** because the C# has not been compiled or executed. |
| 3. A complete recorded session can be replayed deterministically | **Met.** Asserted by tests for ordering, byte-stable output, and an unmodified source recording. |
| 4. Every transmitted and persisted message is typed and validated | **Met.** Closed Pydantic models on the wire and re-validated on read from disk. |
| 5. No scientific, psychological, clinical, engagement, or cognitive-load conclusions | **Met.** Not representable in the payload models; asserted by tests over recordings, fixtures, and CLI output. |

**Not met / pending:**
1. Unity compilation, EditMode and PlayMode test execution, and any player
   build. No Unity Editor or Unity Hub is installed; Unity was not
   downloaded automatically.
2. Consequently, criterion 2 cannot be reported as fully met.

**Decisions recorded:** DEC-025 through DEC-036 (see `docs/DECISIONS.md`)

**Remaining validation for this milestone:**
1. Install a supported Unity LTS, open `unity/EngageVR`, compile, and run the
   EditMode and PlayMode suites; record the exact editor version and
   commands in `docs/UNITY_SETUP.md`.
2. Run the Unity client against the live backend and confirm the handshake,
   task telemetry, and adaptation acknowledgement paths end to end.

---

### Milestone 5: Windowed Feature Datasets and Interpretable Baseline Models

**Started:** 2026-08-04
**Completed (implementation):** 2026-08-04
**Status:** Milestone 5 baseline-model pipeline implementation complete;
scientific evaluation on real participant-labelled data pending.

**Dependencies added:** `scikit-learn>=1.6,<2` (resolved 1.9.0),
`pandas>=2.2,<4` (3.0.5), `pyarrow>=17,<26` (25.0.0), `joblib>=1.4,<2`
(1.5.3), all runtime. **XGBoost deliberately not added** (DEC-037): the
maintained scikit-learn histogram gradient boosting implements the same
algorithm, handles missing values natively, and is already installed.
NumPy 2.5.1 and SciPy 1.18.0 were unchanged by the resolution; one OpenCV
wheel variant remains.

**Delivered:**

- **Feature catalog** (`features/catalog.py`, DEC-038): 61 versioned,
  ordered entries across five modality groups, each declaring name,
  description, modality, unit, aggregation formula, minimum evidence,
  missing-value behaviour, quality dependency, and whether it is permitted
  as a predictor. `rppg_method` is catalogued for provenance and is not a
  permitted predictor.
- **Deterministic windowing** (`features/windowing.py`): half-open
  `[start, end)` boundaries computed from the session start rather than
  accumulated; overlap flag; `drop` / `keep_if_minimum` partial-window
  policy; session-containment assertion; a single window-selection
  primitive that makes "no future events" structural.
- **Aggregation** (`features/aggregation.py`): behavioural, head-pose,
  rPPG, task, and capture-quality aggregators with minimum-evidence gates;
  rejected rPPG windows contribute nothing to any physiological summary;
  a timeout contributes no reaction time; a bridge from the Milestone 3
  `RppgMethodResult` contract.
- **Dataset assembly** (`features/assembly.py`, DEC-040): Parquet writing,
  a provenance metadata document, a catalog snapshot, and a SHA-256
  fingerprint over canonical content that **excludes every wall clock**.
  All three written atomically.
- **Validation** (`features/validation.py`): duplicate-window, reversed-
  range, non-finite, catalog-conformance, session-containment, PII, and
  four-mode leakage checks.
- **Synthetic generator** (`features/synthetic.py`): two latent variables
  never emitted as columns, subject and session effects, drift, AR noise,
  configurable class imbalance, modality dropout, quality-gated rPPG
  windows, correlated and irrelevant features, and noisy labels so a
  perfect score is unattainable in principle. Every row and target
  permanently labelled SYNTHETIC.
- **Schemas**: `schemas/features.py`, `schemas/targets.py`,
  `schemas/experiments.py`. Synthetic targets are schema-forbidden from
  scientific evaluation (DEC-044); measurements are schema-refused as
  automatic labels (DEC-045).
- **Grouped splitting** (`training/splits.py`, DEC-041): subject → session
  → refusal; explicit stratification-feasibility check with a recorded
  reason; no row-level fallback anywhere; an independent `audit_split`.
- **Fold-local preprocessing** (`training/preprocessing.py`): imputation
  and scaling inside the `Pipeline`, missingness indicators, pandas output
  so feature names survive, undeclared-column refusal.
- **Baselines** (`training/models.py`, DEC-046): 5 classifiers and 5
  regressors including two clearly labelled rule-based software-check
  estimators.
- **Calibration** (`training/calibration.py`, DEC-042): `FrozenEstimator`
  on groups disjoint from fitting and from the outer test fold; isotonic
  refused below 50 rows or 10 per class, with a stated reason.
- **Metrics** (`training/metrics.py`, DEC-043): labelled confusion
  matrices, undefined-stays-null, documented macro / Brier / ECE / log-loss
  conventions, equal-weight fold aggregation with valid-fold counts.
- **Interpretation**: linear coefficients with scaling context;
  permutation importance on held-out fold data with per-repeat spread;
  fold-level records stored before aggregation; chance-level warnings.
- **Ablations** (`training/ablation.py`): nine feature subsets on
  identical folds, with an explicit denial that this is fusion.
- **Experiment records** (`training/artifacts.py`, DEC-047): a JSON +
  Parquet run directory, deterministic run ids, checksums, atomic
  manifest written last, failure recorded as failure. **No MLflow.**
- **CLI**: `features-demo`, `baseline-demo`, `baseline-train`.
- **Configuration**: new `features` and `training` sections in
  `configs/defaults.yaml` with validated Pydantic models.
- **Documentation**: `docs/FEATURE_DATASET.md`,
  `docs/BASELINE_MODELS.md`, `docs/MODEL_EVALUATION.md`,
  `docs/EXPERIMENT_TRACKING.md`; updates to README, ARCHITECTURE,
  LIMITATIONS, REFERENCES, DATASETS, DECISIONS.

**Verification:**
- `uv lock --check` -- resolved
- `uv sync --locked` -- clean (72 packages)
- `uv tree` -- no duplicate or conflicting packages; one OpenCV wheel
- `uv run ruff format --check .` -- 158 files formatted
- `uv run ruff check .` -- all checks passed
- `uv run mypy src` -- no issues in 93 source files
- `uv run pytest` -- **1195 passed, 1 skipped**
- `uv run pytest -m hardware` -- 1 skipped (no physical webcam)
- `uv run pre-commit run --all-files` -- all passed
- `uv run python scripts/generate_protocol_artifacts.py` -- no drift
- `make check` -- all passed

448 tests were added across 14 new modules.

**Milestone 5 acceptance criteria:**

| Criterion | Status |
|-----------|--------|
| 1. No data leakage between participant sessions | **Met for the implementation.** Grouped splitting with no row-level fallback, an independent split audit, fold-local preprocessing, calibration on disjoint groups, structural refusal of target, identifier, and post-window columns. Not exercisable against real participant sessions, because none exist. |
| 2. Metrics are reproducible (seeded) | **Met.** Deterministic dataset fingerprints, deterministic split manifests, explicit seeds; asserted by repeat-run tests over `metrics.json` and `splits.json`. |
| 3. Data origin is documented | **Met.** Dataset metadata, catalog snapshot, per-row target provenance, and a run manifest recording every dependency version — all inspectable as JSON without loading a model file. |
| 4. Synthetic data is excluded from scientific evaluation | **Met.** Schema-enforced on targets and re-checked by the scientific-mode gate. |
| 5. No synthetic number presented as real evidence | **Met.** Required disclaimers, schema-enforced self-check banner, no schema field capable of holding a published score, CLI output asserted by tests. |

**Not met / pending:**
1. **Scientific evaluation on real participant-labelled data.** No
   validated EngageVR participant dataset exists and no approved
   engagement or cognitive-load label exists. Every metric produced so far
   is a synthetic software self-check.
2. The feature-aggregation layer has never been run on a live webcam
   session (physical-webcam validation remains pending from Milestone 3).
3. No public dataset can currently supply a target; UBFC-rPPG carries no
   engagement annotation and its own validation remains pending.

**Decisions recorded:** DEC-037 through DEC-048 (see `docs/DECISIONS.md`)

**Remaining validation for this milestone:**
1. Obtain, or design and gain approval for, a real engagement or
   cognitive-load label instrument, then record labels with documented
   provenance.
2. Run the aggregation layer on real capture sessions.
3. Re-run `baseline-train --mode scientific` on that data and report the
   metrics with full provenance — the first numbers in this project that
   would describe anything other than software.

### Milestone 6: Multimodal Fusion

**Status:** Not started

### Milestone 7: Uncertainty-Aware Inference

**Status:** Not started

### Milestone 8: Adaptive Environment

**Status:** Not started

### Milestone 9: Dashboard

**Status:** Not started

### Milestone 10: MLOps and Packaging

**Status:** Not started

### Milestone 11: Research Documentation

**Status:** Not started
