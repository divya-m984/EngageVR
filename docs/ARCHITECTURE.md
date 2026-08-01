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
| `task/events.py` | Task-performance event ingestion |
| `questionnaires/collector.py` | Subjective feedback capture (generic, configurable) |
| `simulator/synthetic.py` | Deterministic synthetic stream generator |
| `capture/adapters.py` | Abstract adapter interface for future wearable sensors |

### 2. Signal-Processing Layer (`src/engagevr/rppg/`, `physiology/`)

Processes raw signals into physiological estimates with quality scores.

| Module | Responsibility |
|--------|---------------|
| `rppg/roi.py` | Skin-region-of-interest extraction |
| `rppg/trace.py` | RGB trace extraction from ROI |
| `rppg/methods.py` | Green-channel baseline, CHROM, POS algorithms |
| `rppg/filtering.py` | Detrending, band-pass filtering, normalization |
| `rppg/quality.py` | Signal-quality index (SNR, motion artifacts) |
| `physiology/hr.py` | Heart-rate estimation from rPPG waveform |
| `physiology/peaks.py` | Peak detection, IBI extraction |
| `physiology/hrv.py` | HRV features (SDNN, RMSSD, pNN50) when valid |
| `physiology/validation.py` | Minimum-duration and quality checks before HRV |

### 3. Feature Layer (`src/engagevr/features/`)

Aggregates frame-level and window-level features with quality metadata.

| Module | Responsibility |
|--------|---------------|
| `features/behavioural.py` | Facial and head-motion feature aggregation |
| `features/physiological.py` | HR, HRV, rPPG quality feature aggregation |
| `features/task.py` | Task-performance feature aggregation |
| `features/subjective.py` | Subjective response features |
| `features/windowing.py` | Configurable window aggregation |
| `features/missing.py` | Missing-data handling and masks |

### 4. Data and Synchronization Layer (`src/engagevr/schemas/`, `storage/`, `synchronization/`)

Defines typed schemas, manages storage, and aligns timestamps.

| Module | Responsibility |
|--------|---------------|
| `schemas/session.py` | Session, participant, experiment condition |
| `schemas/modality.py` | Timestamped modality samples |
| `schemas/prediction.py` | Engagement/load estimate, confidence, abstention |
| `schemas/adaptation.py` | Adaptation event with full provenance |
| `schemas/questionnaire.py` | Subjective response schema |
| `synchronization/clock.py` | Common monotonic timestamp source |
| `storage/parquet.py` | Parquet read/write for time-series data |
| `storage/session_db.py` | SQLite session summary storage |
| `datasets/adapter.py` | Abstract dataset adapter interface |

### 5. Machine-Learning Layer (`src/engagevr/training/`, `inference/`, `uncertainty/`, `personalization/`)

Staged modelling pipeline from deterministic demo through uncertainty-aware fusion.

| Stage | Module(s) | Description |
|-------|-----------|-------------|
| A: Demo | `inference/demo.py` | Deterministic synthetic predictions |
| B: Baselines | `training/baselines.py` | Logistic regression, RF, XGBoost, rules |
| C: Fusion | `training/fusion.py` | Early/late fusion, modality masks, quality weights |
| D: Temporal | `training/temporal.py` | LSTM/GRU/TCN (deferred until baselines evaluated) |
| E: Personal | `personalization/calibration.py` | Per-user baseline, z-score, few-shot, cold-start |
| F: Uncertainty | `uncertainty/calibration.py` | Calibrated probabilities, ensemble disagreement, conformal |

### 6. Adaptation Policy Layer (`src/engagevr/adaptation/`)

Maps model outputs to environment-change commands with safety controls.

| Module | Responsibility |
|--------|---------------|
| `adaptation/policy.py` | Rule-based policy, confidence/quality gating |
| `adaptation/safety.py` | Cooldown, hysteresis, max-change limits, experimenter lock |
| `adaptation/logger.py` | Complete adaptation event log |

### 7. API Layer (`src/engagevr/api/`)

FastAPI application providing WebSocket and HTTP endpoints.

| Module | Responsibility |
|--------|---------------|
| `api/app.py` | FastAPI application factory |
| `api/websocket.py` | WebSocket bridge for Unity/simulator |
| `api/routes.py` | REST endpoints for session management, replay |

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
| Web framework | FastAPI |
| Data validation | Pydantic v2 |
| Computer vision | OpenCV, MediaPipe |
| Numerics | NumPy, SciPy |
| Data frames | pandas, PyArrow |
| ML | scikit-learn, XGBoost |
| Deep learning | PyTorch (deferred) |
| Dashboard | Streamlit, Plotly |
| Storage | Parquet, SQLite |
| Experiment tracking | MLflow (incremental) |
| Data versioning | DVC (incremental) |
| Linting | Ruff |
| Type checking | mypy |
| Testing | pytest |
| CI | GitHub Actions |
| Containers | Docker (incremental) |
| Game engine | Unity (C#, desktop first) |
| Communication | WebSocket (FastAPI) |

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

## Synthetic Data Policy

All synthetic data is:
- Generated by `simulator/synthetic.py`
- Stored under `data/synthetic/`
- Tagged with `data_source: "synthetic"` in every record
- Displayed with a permanent visible label in the dashboard
- Excluded from any scientific evaluation or metric reporting
