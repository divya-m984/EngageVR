# EngageVR -- Progress Tracker

## Current Milestone: 2 -- Webcam Behavioural Capture

**Status:** Complete

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

### Milestone 3: rPPG Pipeline

**Status:** Not started

### Milestone 4: Task Environment and Simulator

**Status:** Not started

### Milestone 5: Baseline Models

**Status:** Not started

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
