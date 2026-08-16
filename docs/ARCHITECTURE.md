# EngageVR -- Software Architecture

## System Overview

EngageVR is organized as a set of independent but integrated layers. Each layer
communicates through well-defined typed interfaces (Pydantic models). The system
runs locally on a standard laptop; hardware integrations use adapter interfaces
so real sensors can be substituted later.

```
+------------------+     +---------------------+     +------------------+
|  Capture Layer   | --> | Signal-Processing   | --> |  Feature Layer   |
|  (webcam, task,  |     |  Layer              |     |  (behavioural,   |
|   questionnaire, |     |  (rPPG, filtering,  |     |   physiological, |
|   simulator,     |     |   quality scoring)  |     |   task, subj.)   |
|   future sensors)|     +---------------------+     +------------------+
+------------------+                                         |
                                                             v
+------------------+     +---------------------+     +------------------+
|  Unity / Task    | <-- | Adaptation Policy   | <-- |  ML / Inference  |
|  Environment     |     |  Layer              |     |  Layer           |
|  (desktop or VR) |     |  (rules, cooldown,  |     |  (fusion, unc.,  |
+------------------+     |   hysteresis, log)  |     |   personalization|
        ^                +---------------------+     +------------------+
        |                                                    |
        +--- WebSocket / HTTP (FastAPI) --------------------+
                                                             |
                                                    +------------------+
                                                    |  Dashboard       |
                                                    |  (Streamlit)     |
                                                    +------------------+
```

## Layer Descriptions

### 1. Capture Layer (`src/engagevr/capture/`, `face/`, `head_pose/`)

Acquires raw signals from available sources.

| Module | Responsibility |
|--------|---------------|
| `capture/webcam.py` | OpenCV webcam acquisition, frame timestamping |
| `face/detector.py` | Face detection (MediaPipe) |
| `face/landmarks.py` | 468-point facial landmark extraction |
| `face/features.py` | Blink rate, eye-closure, mouth movement, landmark motion |
| `head_pose/estimator.py` | Yaw, pitch, roll estimation from landmarks |
| `head_pose/features.py` | Velocity, variability, stability metrics |
| `task/simulator.py` | Deterministic SYNTHETIC task simulator (M4) |
| `questionnaires/collector.py` | Subjective feedback capture (generic, configurable) |
| `simulator/synthetic.py` | Deterministic synthetic stream generator |
| `capture/adapters.py` | Abstract adapter interface for future wearable sensors |

### 2. Signal-Processing Layer (`src/engagevr/rppg/`, `physiology/`)

Processes raw signals into physiological estimates with quality scores.

| Module | Responsibility | Status |
|--------|---------------|--------|
| `rppg/roi.py` | Skin-region-of-interest extraction | Implemented (M3) |
| `rppg/trace.py` | RGB trace extraction, timing diagnostics, synthetic traces | Implemented (M3) |
| `rppg/preprocessing.py` | Detrending, normalization, resampling, band-pass filtering | Implemented (M3) |
| `rppg/methods.py` | GREEN, CHROM, POS algorithms | Implemented (M3) |
| `rppg/heart_rate.py` | Spectral heart-rate estimation (Welch) | Implemented (M3) |
| `rppg/quality.py` | Interpretable signal-quality index | Implemented (M3) |
| `rppg/window.py` | Per-window pipeline orchestration and quality gate | Implemented (M3) |
| `rppg/evaluation.py` | Error metrics against real reference signals | Implemented (M3) |
| `rppg/errors.py` | Typed unavailable-with-reason signalling | Implemented (M3) |
| `physiology/peaks.py` | Peak detection, IBI extraction | **Deferred** (DEC-022) |
| `physiology/hrv.py` | HRV features (SDNN, RMSSD, pNN50) | **Deferred** (DEC-022) |
| `physiology/validation.py` | Minimum-duration and quality checks before HRV | **Deferred** (DEC-022) |

Heart-rate estimation lives in `rppg/heart_rate.py` rather than a
separate `physiology/` package: it operates directly on the rPPG
waveform and shares the pipeline's configuration and failure model.
A `physiology/` package will be created if and when HRV is implemented.

#### rPPG processing flow

```
frame + landmarks
      |
      v
  roi.py            forehead + both cheeks, inset, clipped to frame,
      |             non-clipped pixels pooled, spatially averaged
      v
  trace.py          timestamped RGB sample; missing ROI -> valid=False
      |             (never zero-filled). Windowed by time, not by count.
      v
preprocessing.py    validate timing -> resample -> detrend -> normalize
      |             Rejects duplicate/reversed/jittery timestamps rather
      |             than assuming uniform sampling.
      v
  methods.py        GREEN | CHROM | POS  ->  band-pass (Butterworth SOS,
      |             zero-phase sosfiltfilt)
      v
heart_rate.py       Welch PSD -> peak search restricted to the pulse band
      |             -> BPM = f * 60, with full spectral diagnostics
      v
  quality.py        ~13 independent components, equal-weight mean,
      |             plus hard gates (DEC-021)
      v
  window.py         GATE: quality unacceptable  ->  heart rate = unavailable
```

#### ROI strategy

Three regions — forehead and both cheeks — are derived from MediaPipe
Face Mesh landmark index sets.  Each region's bounding box is trimmed
inward by a configurable `inset_fraction`, which is what excludes the
hairline above the forehead box, the eyebrows and eyes below it, and the
nostrils and face outline beside the cheek boxes.  Boxes are clipped to
frame bounds; degenerate or out-of-frame boxes are rejected.

A pixel is *valid* when every channel lies strictly between the
configured clipping bounds — crushed-black and saturated pixels carry no
plethysmographic modulation.  Valid pixels from all available regions are
pooled before averaging, so each region contributes in proportion to its
valid-pixel count.

`left` and `right` name the **image** frame, not the subject's anatomy.

#### Failure and abstention conditions

A window returns `unavailable` when any of these holds:

| Condition | Reason code |
|-----------|-------------|
| Too few frames had a usable ROI | `too_few_valid_frames` |
| Duplicate timestamps | `duplicate_timestamps` |
| Reversed timestamps | `non_monotonic_timestamps` |
| Inter-sample jitter beyond tolerance | `excessive_timestamp_jitter` |
| Window shorter than the minimum | `window_too_short` |
| Window shorter than the filter's padding requirement | `filter_not_viable` |
| Constant or non-finite signal | `constant_signal` / `non_finite_values` |
| Band edge at or above Nyquist | `invalid_frequency_band` |
| No local maximum inside the pulse band | `no_spectral_peak` |
| Peak prominence below threshold | `peak_below_min_prominence` |
| Aggregate quality below threshold, or a gate failed | `insufficient_signal_quality` |

None of these is ever rendered as a low engagement or high cognitive-load
value.

### 3. Feature Layer (`src/engagevr/features/`)

Aggregates frame-level and event-level observations into windowed rows with
explicit availability and quality metadata. Implemented in Milestone 5.

| Module | Responsibility | Status |
|--------|---------------|--------|
| `features/catalog.py` | The versioned, ordered feature catalog (DEC-038) | Implemented (M5) |
| `features/windowing.py` | Deterministic half-open windows, containment, selection | Implemented (M5) |
| `features/aggregation.py` | Per-modality aggregation with minimum-evidence gates | Implemented (M5) |
| `features/assembly.py` | Parquet writing, dataset metadata, SHA-256 fingerprint | Implemented (M5) |
| `features/validation.py` | Structural, privacy, and four-mode leakage checks | Implemented (M5) |
| `features/synthetic.py` | Deterministic SYNTHETIC dataset generator | Implemented (M5) |
| `features/subjective.py` | Subjective response features | **Deferred** (no instrument) |

Behavioural, head-pose, rPPG, task, and capture-quality aggregation live as
functions inside `aggregation.py` rather than as one module per modality:
they share the window-selection primitive, the minimum-evidence
configuration, and the availability contract, and separating them would
duplicate all three. Missing-data handling is not a module either — it is
the `avail__` / `modality_available__` / `modality_quality__` column
convention plus per-feature `missing_behaviour` in the catalog (DEC-039).

See `docs/FEATURE_DATASET.md` for the row schema, formulas, units, and
fingerprint definition.

### 4. Data and Synchronization Layer (`src/engagevr/schemas/`, `storage/`, `synchronization/`)

Defines typed schemas, manages storage, and aligns timestamps.

| Module | Responsibility |
|--------|---------------|
| `schemas/session.py` | Session, participant, experiment condition |
| `schemas/modality.py` | Timestamped modality samples |
| `schemas/prediction.py` | Engagement/load estimate, confidence, abstention |
| `schemas/adaptation.py` | Adaptation event with full provenance |
| `schemas/rppg.py` | ROI, RGB trace, waveform, heart rate, rPPG quality, evaluation |
| `datasets/base.py` | Abstract dataset adapter interface |
| `datasets/ubfc_rppg.py` | UBFC-rPPG adapter (no download; see `docs/DATASETS.md`) |
| `schemas/questionnaire.py` | Subjective response schema |
| `schemas/events.py` | Task-event vocabulary and `TaskEventDetail` (extended in M4) |
| `synchronization/clock.py` | Clock abstraction; RTT-bounded offset *estimates* (M4) |
| `synchronization/ordering.py` | Sequence and message-id anomaly detection (M4) |
| `storage/session_store.py` | Session directories, recovery, path safety (M4) |
| `storage/jsonl.py` | Append-only JSON Lines and atomic JSON writes (M4) |
| `storage/manifest.py` | Manifest, ingestion metadata, summary, drop records (M4) |
| `features/assembly.py` | Parquet read/write for windowed feature datasets (M5) | Implemented (M5) |
| `storage/parquet.py` | Parquet read/write for raw time-series data | **Deferred** |
| `storage/session_db.py` | SQLite session summary storage | **Deferred** |
| `datasets/adapter.py` | Abstract dataset adapter interface |

#### Timing and ordering invariants (Milestone 4)

- Three timestamps are kept distinct and never conflated: the sender's wall
  clock, the sender's own monotonic clock, and the receiver's ingestion wall
  clock. Nothing translates one onto another's timeline.
- Transport delay is computed **only** within one process. Across processes
  the field is null with a stated reason, because subtracting two
  independent wall clocks measures clock offset, not delay (DEC-029).
- Clock offset is estimated only from heartbeat round trips and always
  carries `rtt/2` uncertainty and an explicit symmetric-delay assumption.
- Arrival order and source sequence order are recorded **separately** and the
  event stream is never re-sorted; anomalies are recorded, not repaired
  (DEC-028).

### 5. Machine-Learning Layer (`src/engagevr/training/`, `inference/`, `uncertainty/`, `personalization/`)

Staged modelling pipeline from deterministic demo through uncertainty-aware fusion.

| Stage | Module(s) | Description | Status |
|-------|-----------|-------------|--------|
| A: Demo | `inference/demo.py` | Deterministic synthetic predictions | Deferred |
| B: Baselines | `training/` (below) | Grouped CV over interpretable classical models | Implemented (M5) |
| C: Fusion | `training/fusion*.py`, `experts.py`, `stacking.py`, `robustness.py` | Early / late / quality-aware / stacked fusion, modality availability, quality weights | Implemented (M6) |
| D: Temporal | `training/temporal.py` | LSTM/GRU/TCN | **Deferred** (DEC-005) |
| E: Personal | `training/personalization*.py` | Per-user baseline, z-score, few-shot, cold-start, separate population reporting | Implemented (M6) |
| F: Uncertainty | `uncertainty/calibration.py` | Abstention, conformal, ensemble disagreement | **Deferred** (M7) |

#### Milestone 5 modules (`src/engagevr/training/`)

| Module | Responsibility |
|--------|---------------|
| `training/splits.py` | Grouped splitters, group-field choice, split audit (DEC-041) |
| `training/preprocessing.py` | Dataset loading, predictor selection, fold-local transformers |
| `training/models.py` | Classification and regression registries; rule software-check baselines (DEC-046) |
| `training/calibration.py` | `FrozenEstimator` calibration on disjoint groups (DEC-042) |
| `training/metrics.py` | Metrics with explicit unavailability and documented formulas (DEC-043) |
| `training/ablation.py` | Nine deterministic feature-subset definitions |
| `training/runner.py` | Fold orchestration, interpretation, artifact assembly |
| `training/artifacts.py` | Run directories, atomic manifests, checksums (DEC-047) |
| `cli_milestone5.py` | `features-demo` / `baseline-demo` / `baseline-train` |

The baseline registry uses scikit-learn's histogram gradient boosting
rather than XGBoost (DEC-037).

#### Milestone 6 modules (`src/engagevr/training/`)

| Module | Responsibility |
|--------|---------------|
| `training/fusion.py` | Modality columns, weight algebra, probability and regression combination, disagreement helpers (DEC-049, DEC-050, DEC-051) |
| `training/experts.py` | One estimator per modality, fitted only on windows that modality observed; refusal with a reason (DEC-052) |
| `training/stacking.py` | Grouped out-of-fold construction and the independent leakage assertion (DEC-054) |
| `training/robustness.py` | Ten missing-modality scenarios and deterministic synthetic dropout (DEC-055) |
| `training/fusion_metrics.py` | Coverage, contribution counts, mean weights, disagreement summaries (DEC-056) |
| `training/fusion_artifacts.py` | Deterministic run identity, split-manifest fingerprint, fusion Parquet tables (DEC-058) |
| `training/fusion_runner.py` | Fold orchestration, strategy evaluation, artifact assembly |
| `schemas/personalization.py` | Personalization records; a cold start must reproduce the population prediction (DEC-063) |
| `training/personalization.py` | Chronological calibration/evaluation split, personal baselines, corrections (DEC-061, DEC-062) |
| `training/personalization_runner.py` | Population reference model, per-subject adaptation, separate reporting (DEC-060, DEC-064) |
| `cli_milestone6.py` | `fusion-demo` / `fusion-train` (DEC-059), `personalization-demo` / `personalization-train` |

The fusion layer reuses the Milestone 5 splitter, preprocessing,
calibration, and metric machinery unchanged; the split manifest is built
once and fingerprinted so that "the same folds" is checkable. The
personalization layer sits on top of the early-fusion population
prediction rather than beside it, and never retunes a fusion weight. No
temporal model and no abstention policy exists in either, and no deep or
neural fusion is implemented.

See `docs/BASELINE_MODELS.md`, `docs/MODEL_EVALUATION.md`,
`docs/EXPERIMENT_TRACKING.md`, and `docs/MULTIMODAL_FUSION.md`.

### 6. Adaptation Policy Layer (`src/engagevr/adaptation/`)

Maps model outputs to environment-change commands with safety controls.

| Module | Responsibility |
|--------|---------------|
| `adaptation/policy.py` | Rule-based policy, confidence/quality gating |
| `adaptation/safety.py` | Cooldown, hysteresis, max-change limits, experimenter lock |
| `adaptation/logger.py` | Complete adaptation event log |

### 7. API Layer (`src/engagevr/api/`)

Local FastAPI application providing HTTP endpoints and a WebSocket bridge.
Implemented in Milestone 4.

| Module | Responsibility | Status |
|--------|---------------|--------|
| `api/app.py` | Application factory and lifespan-owned resources | Implemented (M4) |
| `api/routes.py` | HTTP endpoints; manual adaptation-command entry point | Implemented (M4) |
| `api/websocket.py` | `/ws/v1/sessions/{session_id}` bridge and handshake | Implemented (M4) |
| `api/connections.py` | Single-process connection registry and routing | Implemented (M4) |
| `api/broker.py` | Bounded ingestion/storage/broadcast queues (DEC-035) | Implemented (M4) |
| `api/state.py` | Session store, registry, and live brokers (DEC-035) | Implemented (M4) |
| `api/errors.py` | Typed protocol errors and HTTP error handlers | Implemented (M4) |

**Deployment scope:** one process, bound to loopback, with no
authentication, authorization, or transport encryption. The connection
registry lives in one process's memory, so multi-worker and distributed
operation are **not supported**; a command routed by one worker would never
reach a client connected to another.

### 7b. Protocol Layer (`src/engagevr/protocol/`)

The single versioned wire contract shared by the backend, the Python
simulator, the replay player, and the Unity client (DEC-025).

| Module | Responsibility |
|--------|---------------|
| `protocol/version.py` | Version constants, parsing, major-version acceptance |
| `protocol/messages.py` | 14 message types, 5 sources, closed payload models |
| `protocol/envelope.py` | Envelope, provenance, additive replay metadata |
| `protocol/validation.py` | Size, JSON, version, type, envelope, payload checks |
| `protocol/json_schema.py` | Deterministic JSON Schema generation |
| `schemas/protocol.py` | Pure re-export into the schemas namespace |
| `transport.py` | In-process, JSONL-file, and WebSocket transports (DEC-035) |

### 7c. Task Environment (`src/engagevr/task/`)

| Module | Responsibility |
|--------|---------------|
| `task/config.py` | Per-run simulator settings and scripted scenarios |
| `task/generator.py` | Deterministic seeded trial plan |
| `task/state.py` | Task state machine and adaptation-command application |
| `task/simulator.py` | The simulator loop; injected clock and RNG |
| `unity/EngageVR/` | Unity desktop task (source only; **not compiled**) |

### 7d. Replay (`src/engagevr/replay/`)

| Module | Responsibility |
|--------|---------------|
| `replay/reader.py` | Read-only recording access and filtering |
| `replay/clock.py` | Pacing: immediate, original, accelerated |
| `replay/player.py` | Emission with additive replay metadata (DEC-026) |

### 8. Dashboard (`src/engagevr/dashboard/`)

Streamlit multi-page dashboard for monitoring and analysis.

### 9. Cross-Cutting Concerns

| Module | Responsibility |
|--------|---------------|
| `config.py` | YAML configuration loading with validation |
| `logging.py` | Structured logging (JSON) |
| `utils/timestamps.py` | Timestamp utilities |

## Data Flow

1. **Capture:** Webcam frames, task events, and subjective responses are acquired
   with monotonic timestamps.
2. **Process:** Frames are processed for face detection, landmarks, head pose,
   rPPG ROI extraction, and RGB traces. Signal quality is assessed.
3. **Extract:** Frame-level measurements are aggregated into configurable windows.
   Missing data is masked, not imputed with misleading defaults.
4. **Predict:** Feature windows are passed to the ML layer. The model produces
   engagement and cognitive-load estimates with confidence scores. It may abstain.
5. **Adapt:** The adaptation policy evaluates the prediction against confidence
   and signal-quality thresholds. If criteria are met and cooldown has elapsed,
   an adaptation command is issued.
6. **Execute:** The command is sent to Unity (or the simulator) via WebSocket.
7. **Log:** Every step is logged with full provenance for replay and analysis.
8. **Display:** The dashboard shows live or replayed data with clear labels for
   data source (live, public dataset, synthetic) and reliability.

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Package manager | uv |
| Web framework | FastAPI 0.141 + uvicorn 0.52 |
| Data validation | Pydantic v2 |
| Computer vision | OpenCV, MediaPipe |
| Numerics | NumPy, SciPy |
| Data frames | pandas 3.0, PyArrow 25 |
| ML | scikit-learn 1.9 (histogram gradient boosting; **no XGBoost**, DEC-037) |
| Model persistence | joblib |
| Deep learning | PyTorch (deferred) |
| Dashboard | Streamlit, Plotly |
| Storage | JSON Lines (M4); Parquet, SQLite (deferred) |
| Experiment tracking | MLflow (incremental) |
| Data versioning | DVC (incremental) |
| Linting | Ruff |
| Type checking | mypy |
| Testing | pytest |
| CI | GitHub Actions |
| Containers | Docker (incremental) |
| Game engine | Unity (C#, desktop first) |
| Communication | WebSocket (websockets 17), protocol version 1.0 |

## Configuration

All runtime parameters are stored in YAML files under `configs/` and loaded
at startup. No configuration is hard-coded in source files. Sensitive values
(if any) are stored in `.env` files excluded from version control.

## Privacy Model

- Raw webcam video storage is **disabled by default**.
- Only extracted features are persisted.
- Participants are identified by pseudonymous IDs only.
- No names, emails, or unnecessary identifying information are stored.
- Webcam frames are never exposed outside the local machine.
- Configuration controls what is stored and what is discarded.

### rPPG-specific privacy behaviour

- ROI pixels exist in memory for the duration of one frame and are
  discarded immediately after spatial averaging.
- No ROI image is written to disk, logged, or transmitted. The persisted
  `RoiObservation` contains bounding coordinates, pixel counts, and mean
  brightness only.
- The `rppg-demo` artifact persists window-level summaries, quality
  components, and the heart-rate estimate. Per-sample RGB arrays and
  per-sample waveform values are excluded from the persisted output.
- Nothing in the rPPG pipeline infers skin tone, ethnicity, identity,
  emotion, engagement, or cognitive state from a ROI.
- No dataset is downloaded; no data leaves the local machine.

### Session-recording privacy behaviour (Milestone 4)

- A session recording contains protocol envelopes and receiver ingestion
  metadata only.
- Raw webcam frames, video, image data, MediaPipe objects, landmark arrays,
  engagement estimates, cognitive-load estimates, model predictions, heart
  rates, secrets, and real-world identities are **not representable** in a
  recording: the protocol payload models are closed (`extra="forbid"`) and
  the store writes only what arrived through that wire format.
- Participants appear as a pseudonymous `participant_id` only.
- Session identifiers are validated against a strict allowlist before
  becoming directory names, and the resolved path is checked to be inside
  the session root, so path traversal cannot escape it.
- `artifacts/` remains gitignored.

### Data-labelling invariants

- A `synthetic` message must carry `synthetic_label: "SYNTHETIC"`; a
  non-synthetic message must carry `null`. Both are schema-enforced.
- A replayed message keeps its original provenance and *gains* a `REPLAY`
  block, so a replayed synthetic message carries both labels at once
  (DEC-026).

## Synthetic Data Policy

All synthetic data is:
- Generated by `simulator/synthetic.py`
- Stored under `data/synthetic/`
- Tagged with `data_source: "synthetic"` in every record
- Displayed with a permanent visible label in the dashboard
- Excluded from any scientific evaluation or metric reporting

### Modelling-dataset privacy behaviour (Milestone 5)

- A windowed feature dataset contains pseudonymous identifiers, scalar
  feature values, availability flags, quality scores, targets, and target
  provenance. Nothing else is representable.
- There is no column for a name, an email, a frame, an image, a landmark
  array, a protocol payload blob, or a secret. `assert_no_identity_columns`
  and `assert_no_identifier_values` assert that none appears, and a test
  scans a real generated dataset.
- Synthetic subjects are named `synthetic-subject-0001`, never
  "participant 1".
- Experiment run directories carry the same guarantee: a test scans every
  JSON artifact for identifiers, secrets, and scientific-validity claims.
- `artifacts/` remains gitignored. Generated Parquet datasets, prediction
  tables, and model files are never committed.
- Model files are Python pickles and are labelled as executable content;
  auditing a run never requires loading one.
