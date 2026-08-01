# EngageVR -- Project Plan

## Overview

EngageVR is a modular software prototype that estimates engagement and cognitive load
from multimodal behavioural, physiological, and task-performance signals and uses
those estimates, together with signal quality and model confidence, to adapt a
Unity-based virtual environment.

This plan defines a feasible, incremental implementation path that respects the
current hardware and institutional constraints documented in
`docs/LIMITATIONS.md`.

## Software-Only MVP Boundary

The MVP operates entirely on a standard laptop with a webcam. It does not require
a VR headset, wearable sensors, or a participant-testing room.

**In scope for MVP (Milestones 0--10):**

- Webcam-based facial-behaviour and head-motion feature extraction
- Camera-based remote photoplethysmography (rPPG) pipeline
- Heart-rate estimation and short-term HRV features (when signal quality permits)
- Task-performance telemetry via Python simulator and Unity desktop task
- Subjective questionnaire capture (generic, configurable)
- Interpretable ML baselines (logistic regression, random forest, gradient boosting)
- Multimodal fusion with missing-modality masks and quality-aware weighting
- Uncertainty estimation and model abstention
- Rule-based adaptation policy with safety controls
- FastAPI WebSocket bridge
- Streamlit monitoring dashboard
- Synthetic data generator (permanently labelled)
- Public-dataset adapters for component validation
- Docker packaging, CI, MLflow, DVC

**Deferred until laboratory access:**

- VR headset integration (tracked controllers, 6-DOF head tracking, stereoscopic
  rendering)
- Research-grade wearable sensors (PPG, ECG, EDA, respiration)
- Participant recruitment and human-subject experimentation
- Institutional ethics approval and consent procedures
- Clinical or experimental validation of engagement/cognitive-load estimates
- Deep temporal models (LSTM, GRU, TCN, Transformer) -- deferred until
  interpretable baselines are evaluated
- Reinforcement-learning-based adaptation policy
- Cloud deployment and multi-node infrastructure
- Eye-tracking (requires hardware beyond webcam gaze proxies)

## Milestone Schedule

### Milestone 0: Repository Audit and Plan

**Objective:** Inspect the repository, document the architecture, define research
questions, record limitations, and produce a milestone checklist.

**Deliverables:**
- `docs/PROJECT_PLAN.md` (this file)
- `docs/ARCHITECTURE.md`
- `docs/RESEARCH_QUESTIONS.md`
- `docs/LIMITATIONS.md`
- `docs/DECISIONS.md`
- `docs/PROGRESS.md`

**Acceptance criteria:**
- All six documents exist and are internally consistent
- The plan is aligned with `docs/PROJECT_SPECIFICATION.md`
- Deferred functionality is explicitly identified
- No implementation code is written

---

### Milestone 1: Foundation

**Objective:** Establish the Python package structure, development tooling,
configuration loading, logging, timestamped schemas, session metadata, a
synthetic event generator, and initial CI.

**Deliverables:**
- `pyproject.toml` with dev dependencies (ruff, mypy, pytest, pytest-cov)
- `configs/` directory with YAML configuration files
- `src/engagevr/config.py` -- configuration loading
- `src/engagevr/logging.py` -- structured logging
- `src/engagevr/schemas/` -- Pydantic models for session, timestamp, modality,
  prediction, adaptation event
- `src/engagevr/simulator/synthetic.py` -- synthetic event generator
- `tests/unit/` -- schema validation, config loading, synthetic generator tests
- `Makefile` with lint, typecheck, test, format targets
- `.github/workflows/ci.yml`

**Acceptance criteria:**
- `uv sync` installs cleanly
- `make lint` passes (ruff)
- `make typecheck` passes (mypy)
- `make test` passes (pytest)
- Synthetic session can be generated and serialized
- Schemas reject invalid data (tested)

---

### Milestone 2: Webcam Behavioural Capture

**Objective:** Webcam capture, face detection, facial landmarks, head-pose
estimation, blink/eye-closure proxies, head-motion features, and capture-quality
metrics.

**Acceptance criteria:**
- Live capture runs on a laptop at reported frame rate
- Missing-face conditions are handled gracefully
- Features are timestamped to the common clock
- Privacy-preserving mode stores features without video
- Unit tests exist for feature calculations

---

### Milestone 3: rPPG Pipeline

**Objective:** Skin-region extraction, RGB trace, motion/illumination diagnostics,
green-channel baseline, POS or CHROM, filtering, HR estimation, signal-quality
index, public-dataset evaluation adapter.

**Acceptance criteria:**
- Pipeline processes a known public sample
- Signal quality is reported per window
- Unreliable windows return `unavailable`
- No fabricated accuracy claims

---

### Milestone 4: Task Environment and Simulator

**Objective:** Task-event schema, Python task simulator, Unity desktop cognitive
task, WebSocket bridge, synchronized event history.

**Acceptance criteria:**
- Backend works without Unity
- Unity and simulator share the same protocol
- Full session replay is possible

---

### Milestone 5: Baseline Models

**Objective:** Windowed feature dataset, interpretable models, participant-aware
splitting, cross-validation, calibration, metrics, ablation, experiment tracking.

**Acceptance criteria:**
- No data leakage between participant sessions
- Metrics are reproducible (seeded)
- Data origin is documented
- Synthetic data is excluded from scientific evaluation

---

### Milestone 6: Multimodal Fusion

**Objective:** Modality masks, quality-aware fusion, early/late fusion comparison,
missing-modality tests, personalized calibration.

**Acceptance criteria:**
- System remains functional with missing signals
- Quality-aware fusion is compared with naive fusion
- Personalized and population baselines are separately reported

---

### Milestone 7: Uncertainty-Aware Inference

**Objective:** Calibrated probabilities, abstention logic, coverage-vs-performance
analysis, confidence in API, adaptation gating.

**Acceptance criteria:**
- Predictions can abstain
- Confidence is not confused with signal quality
- Adaptation policy respects both thresholds

---

### Milestone 8: Adaptive Environment

**Objective:** Rule-based adaptation engine, cooldown, hysteresis, manual control,
adaptation logging, static/adaptive experimental modes.

**Acceptance criteria:**
- No rapid oscillation
- Deterministic policy output for same input
- Experimenter can disable adaptation
- Static and adaptive modes are clearly separated

---

### Milestone 9: Dashboard

**Objective:** Streamlit dashboard with all core monitoring pages, real-time and
replay modes, signal-quality warnings, uncertainty display, adaptation history,
exportable session report.

**Acceptance criteria:**
- Dashboard runs locally
- Synthetic, public, and live data are visually labelled
- Missing or unreliable measurements are visible
- Session report can be reproduced

---

### Milestone 10: MLOps and Packaging

**Objective:** MLflow, DVC stages, Docker, GitHub Actions, model versioning,
drift checks, system smoke tests, release instructions.

**Acceptance criteria:**
- Clean clone can reproduce the demo
- CI passes
- Model artifact and configuration are versioned
- Dockerized backend and dashboard work

---

### Milestone 11: Research Documentation

**Objective:** Research proposal, hypotheses, experimental variables,
static-vs-adaptive design, data-collection protocol, consent template draft,
risk assessment, privacy strategy, statistical-analysis plan, dataset cards,
model cards, limitations, hardware-validation plan, future lab extension plan.

**Acceptance criteria:**
- All documents are drafted
- Human-subject documents are marked as drafts requiring institutional review
- No claim of ethical approval

## Dependency Graph

```
M0 --> M1 --> M2 --> M3
                \       \
                 M4 ----> M5 --> M6 --> M7 --> M8 --> M9 --> M10 --> M11
```

Milestones 2--4 can proceed in parallel after M1. Milestones 5+ require outputs
from preceding milestones.
