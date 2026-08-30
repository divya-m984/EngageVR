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
                                                    |  READ-ONLY, over |
                                                    |  artifacts/ and  |
                                                    |  session         |
                                                    |  recordings      |
                                                    +------------------+
```

The dashboard reads persisted artifacts and persisted session recordings. It
is not in the capture, inference, or adaptation path, and it has no arrow
back into any of them. Its live mode is an arrow *from* the session store,
never into it: it re-reads a recording the recorder already wrote.

The arrow from the adaptation layer to the task environment is the **design
intent**, not the current behaviour. As of Milestone 8 the policy produces a
proposal and, optionally, a command object; nothing dispatches it. See
§6 and `docs/ADAPTIVE_ENVIRONMENT.md`.

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
| F: Uncertainty | `training/uncertainty*.py`, `adaptation_gate.py` | Calibrated confidence, entropy, margin, split-conformal intervals, selective prediction, abstention, coverage curves, adaptation gate | Implemented (M7) |

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
| `training/uncertainty.py` | Entropy, margin, confidence, evidence gate, conformal quantile, threshold selection, selective metrics, coverage curves (DEC-066, DEC-067, DEC-068, DEC-069, DEC-070) |
| `training/uncertainty_runner.py` | Four recorded group sets per fold, decisions, curves, artifacts |
| `training/adaptation_gate.py` | Gate only: may an already-chosen action be acted upon? Imports nothing but two schema modules (DEC-071) |
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

Decides, for an **already-eligible** Milestone 7 prediction, whether a
conservative environment change should be **proposed**. Implemented in
Milestone 8.

| Module | Responsibility | Status |
|--------|---------------|--------|
| `adaptation/mapping.py` | The ordinal state-to-direction demonstration rule | Implemented (M8) |
| `adaptation/policy.py` | Pure `evaluate_policy`; gate, evidence, mapping, dwell, cooldown, budget, bounds | Implemented (M8) |
| `adaptation/command.py` | Pure proposal -> existing `set_difficulty` payload; sends nothing | Implemented (M8) |
| `adaptation/lifecycle.py` | proposed / command_built / dispatched / acknowledged / applied | Implemented (M8) |
| `adaptation/scenarios.py` | 15 deterministic controller scenarios | Implemented (M8) |
| `adaptation/runner.py` | Offline simulation, trace Parquet, controller metrics | Implemented (M8) |

**Boundary, enforced by imports.** `adaptation/policy.py` imports two schema
modules and its own mapping table. It imports nothing from `engagevr.api`,
`engagevr.transport`, `engagevr.task`, or `engagevr.training`, so it cannot
send anything and cannot recompute a Milestone 7 confidence, threshold, or
gate verdict. A test asserts this by parsing the module's AST.

**The Milestone 7 gate cannot be bypassed.** A `BLOCKED`
`AdaptationGateRecord` forces `HOLD`, and `AdaptationProposal` embeds both
targets' gate records and refuses to validate unless both are `ELIGIBLE`.
There is no override flag.

**A policy decision is not a network message.** Milestone 8 builds command
objects and stops; nothing here dispatches. Signal quality can never choose a
direction and confidence can never scale a step.

No adaptation rule here is validated with human participants. See
`docs/ADAPTIVE_ENVIRONMENT.md`.

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

**READ-ONLY RESEARCH OBSERVABILITY** over the artifacts previous milestones
already wrote. It displays what a run recorded; it computes no new
scientific quantity, and it has no path from a click to a retrained model, a
re-run pipeline, or a dispatched adaptation.

| Module | Streamlit? | Responsibility |
|--------|-----------|---------------|
| `catalogue.py` | no | Run discovery, artifact-signature family detection, status, checksums |
| `loaders.py` | no | Read-only JSON and Parquet access with column selection |
| `formatting.py` | no | All display formatting; `None` never becomes `0` (DEC-086) |
| `aggregation.py` | no | Display-only aggregates: bins, counts, residuals |
| `presentation.py` | no | Terminology and limitations as typed data |
| `views_dataset.py` | no | Dataset provenance and measurement quality |
| `views_models.py` | no | Classification and regression results |
| `views_fusion.py` | no | Fusion and personalization |
| `views_uncertainty.py` | no | Selective prediction, two coverage axes (DEC-072) |
| `views_adaptation.py` | no | Controller behaviour; no effectiveness field exists |
| `session_reader.py` | no | Tail-safe read-only recording parsing (DEC-092) |
| `session_catalogue.py` | no | Session discovery, status, provenance (DEC-091) |
| `views_session.py` | no | Live and replay view models |
| `session_report.py` | no | The pure, fingerprinted session report (DEC-093) |
| `components.py` | yes | Provenance banners, tables, charts, metric cards |
| `pages.py` | yes | The ten artifact pages |
| `session_pages.py` | yes | The live and replay pages |
| `app.py` | yes | Entry point, evidence-mode selector, selectors, caching |
| `launch.py` | no | `streamlit run` argv construction |
| `schemas/dashboard.py` | no | Typed presentation models, `extra="forbid"` |
| `schemas/dashboard_session.py` | no | Typed session models, replay cursor, report |

**Three evidence modes** (DEC-090), separated in the UI and in the types:
persisted experiment artifacts (primary), read-only live observation of a
session recording, and read-only replay of one. *Live* means re-reading a
recording on a conservative timer (DEC-094), not inference: no model is
loaded, no camera opened, and no estimate produced anywhere in this
package, on any cadence. Only the live page refreshes on its own — replay
never auto-advances and the artifact observatory never polls.

**The layering is import-enforced.** Nothing below `components.py` or
`session_pages.py` may import Streamlit, so the unit tests need no browser,
no socket, and no server. AST tests also assert that no dashboard module
writes, deletes, retrains, recalibrates, dispatches, opens a model pickle,
constructs a `SessionRecorder`, `JsonlWriter`, or `ReplayPlayer`, imports the
simulator, the replay player, `asyncio`, or a socket module, or touches Git.

Run families are detected from artifact signatures, **never from directory
names** (DEC-084). A directory that exists is not a successful run: the
catalogue distinguishes completed, failed, incomplete, corrupt, unsupported,
and unknown. See `docs/DASHBOARD.md`.

### 9a. MLOps and Packaging Layer (`src/engagevr/mlops/`)

The operational layer added by Milestone 10. It **orchestrates and
records**; it never models. Every pipeline stage invokes a Milestone 5-9
subcommand through the public CLI, so there is no second training
pipeline, fusion implementation, uncertainty engine, adaptation
controller, or dashboard anywhere in it.

```
source + config + synthetic generators
              |
        reproducible DVC stages          dvc.yaml, params.yaml
              |
       persisted experiment artifacts    (Milestones 5-8, UNCHANGED)
              |
       MLflow experiment-tracking        mlops/mlflow_tracking.py
              |
       versioned model/artifact records  mlops/model_version.py
              |
        drift / system verification      mlops/drift.py, mlops/smoke.py
              |
       Docker / CI / release workflow
```

| Module | Responsibility |
|--------|---------------|
| `schemas/mlops.py` | Every persisted M10 record, versioned and strict |
| `mlops/fingerprints.py` | Canonical hashing for configuration, splits, feature schemas |
| `mlops/model_version.py` | Immutable, checksum-linked model versions |
| `mlops/mlflow_tracking.py` | The only module that knows MLflow exists |
| `mlops/drift.py` | Five interpretable distribution-shift diagnostics |
| `mlops/pipeline.py` | Stage definitions shared by `dvc.yaml` and `mlops-demo` |
| `mlops/reproducibility.py` | Logical identity across executions |
| `mlops/stage_record.py` | The deterministic, DVC-declared representation of a stage |
| `mlops/execution.py` | Volatile execution metadata, in a never-declared sidecar |
| `mlops/smoke.py` | The 13-check integrated software self-check |
| `cli_milestone10.py` | `mlops-demo`, `model-manifest`, `drift-check`, `mlflow-log`, `repro-manifest`, `system-smoke` |

Two boundaries carry the design.

**Nothing mutates an existing artifact.** Tracking and versioning read a
run directory and write separate records elsewhere; a run that has been
logged is byte-identical to one that has not, which is what makes the
checksums a version record carries mean anything.

**Nothing confers scientific status.** Reproducibility is not validity,
tracking is not validation, registration is not approval, packaging is not
production readiness, and a distribution-shift statistic is an engineering
diagnostic. The schemas refuse to record anything stronger: a synthetic
document cannot carry `scientific_evaluation_eligible=true`, and no record
may contain `production`, `staging`, `champion`, `approved`, or
`validated` as a status word.

Tracking is **opt-in** (`mlops.mlflow.enabled: false`) and local: a file
store at `mlruns/`, no server, no database, no account, no network.
Importing the adapter does not import `mlflow`. See `docs/MLOPS.md` and
DEC-096 through DEC-103.

Outputs land under `mlops.pipeline_root` (`artifacts/pipeline`),
deliberately outside `dashboard.artifact_root`, so rebuilding the demo
never reshuffles the run catalogue a reader is looking at (DEC-102).

**Deterministic outputs are separated from volatile execution metadata**
(DEC-104). The Milestone 5-8 runners keep writing timestamped provenance —
`started_at_utc`, `finished_at_utc`, `created_at_utc` — and those
documents are simply never DVC-declared. A deterministic stage record is
declared in their place, pinning the run id and checksumming only
byte-stable files, so `dvc.lock` is byte-identical across fresh
reproductions while a genuine change to a metric still propagates. When
each Milestone 10 document was produced lives in a `.execution.json`
sidecar that is never an output.

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
5. **Gate (M7):** The selective layer decides whether the prediction may be
   acted on at all, and records `eligible` or `blocked` with reasons.
6. **Adapt (M8):** For an eligible prediction, the deterministic policy maps
   the engagement and cognitive-load states to a direction, then applies the
   dwell, cooldown, budget, and bounds guards. Most windows **hold**, which is
   a normal outcome and not an error. An approved proposal is translated into
   the existing `set_difficulty` payload.
7. **Execute:** Sending that command to Unity or the simulator over the
   Milestone 4 WebSocket bridge is **not automatic**. Milestone 8 constructs
   commands and stops; no policy-derived command is dispatched.
8. **Log:** Every step is logged with full provenance for replay and analysis.
   Every policy evaluation -- hold or proposal -- becomes one auditable row.
9. **Display (M9):** The dashboard reads the persisted artifacts of steps
   4--8 and shows them with their recorded provenance: data source,
   synthetic status, and scientific eligibility on every result-bearing
   page. It never re-runs any step above, and a view derived from a
   synthetic artifact stays synthetic.
10. **Observe and replay (M9):** The same dashboard also reads the session
    recordings written in step 7, either as they are appended to (live
    observation, refreshed automatically at the configured interval) or
    after the fact (replay, navigated by hand). Both are **read-only
    presentation** (DEC-090, DEC-094): the reader re-reads a file the
    recorder already wrote. No arrow runs from the dashboard back into
    steps 1--8, and a recording carries no estimate from steps 4--6 to
    display.
11. **Report (M9):** A deterministic session report can be built from a
    complete read and downloaded. It is a presentation artifact with a
    content fingerprint (DEC-093), not a new experiment result, and it
    carries its source's provenance permanently.

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
| Dashboard | Streamlit 1.62 (native charts; **no Plotly**, DEC-085) |
| Storage | JSON Lines (M4); Parquet, SQLite (deferred) |
| Experiment tracking | mlflow-skinny 3.15 (LOCAL file store, opt-in; DEC-096) |
| Pipeline / data versioning | DVC 3.67 (no remote; outputs regenerated, DEC-100) |
| Linting | Ruff |
| Type checking | mypy |
| Testing | pytest |
| CI | GitHub Actions |
| Containers | Docker: the existing M4 backend and M9 dashboard, loopback only (DEC-103) |
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
