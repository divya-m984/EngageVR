# EngageVR

**An Uncertainty-Aware Multimodal Framework for Personalized Engagement Estimation and Adaptive Virtual Reality**

> EngageVR is a research software prototype. Its engagement and cognitive-load
> outputs are model estimates and must not be treated as medical, psychological,
> or diagnostic conclusions.

## Status

**Milestone 5 baseline-model pipeline implementation complete; scientific
evaluation on real participant-labelled data pending.**

Implemented: webcam capture, face landmarks (MediaPipe), behavioural proxy
features, capture quality, a classical rPPG pipeline (GREEN / CHROM / POS,
spectral heart rate, interpretable quality index, UBFC-rPPG adapter), a
shared versioned protocol (`1.0`), a deterministic task simulator, a local
FastAPI + WebSocket bridge with bounded-queue backpressure, append-only
JSONL session storage with crash recovery, deterministic session replay, a
Unity desktop task at source level, and — new in Milestone 5 — a windowed
feature dataset with a versioned feature catalog and SHA-256 fingerprint,
interpretable baseline models under grouped cross-validation, offline
probability calibration, feature-group ablations, and local experiment
records.

Not implemented: multimodal fusion, uncertainty-aware abstention,
personalization, online inference, adaptation *policy* (Milestone 4
implements command **transport** only), HRV, Streamlit, MLflow, DVC,
Docker, deep learning.

Pending validation: **Unity compilation and runtime** (no Unity Editor is
installed here), **physical-webcam capture**, **UBFC-rPPG evaluation**, and
**any evaluation on real participant labels** — none exist.

> **rPPG heart-rate values are signal-processing estimates from camera
> data.** They are not medical measurements, have not been validated
> against any reference device, and are not engagement or cognitive-load
> values.

> **Every model metric in this repository was computed from SYNTHETIC
> data.** Those numbers are software self-checks. They are not model
> accuracy, not engagement validity, not cognitive-load validity, and must
> never be compared with a published result on real data.

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run tests
make test

# Run all checks (format, lint, typecheck, test)
make check
```

## Development Commands

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies via uv |
| `make format` | Format code with ruff |
| `make lint` | Lint code with ruff |
| `make typecheck` | Type-check with mypy |
| `make test` | Run pytest |
| `make test-cov` | Run pytest with coverage |
| `make check` | Run all checks |
| `make clean` | Remove caches |

## Generate a Synthetic Demo Session

```bash
uv run python -m engagevr demo \
  --seed 42 \
  --output artifacts/demo-session.json
```

This produces a deterministic **SYNTHETIC** session for software testing only.
It is not experimental evidence.

## Webcam Capture

```bash
# 1. Download FaceLandmarker model (Apache 2.0, ~4 MB)
uv run python scripts/download_models.py

# 2. Run capture (requires webcam + model)
uv run python -m engagevr capture \
  --camera 0 \
  --duration 30 \
  --output artifacts/webcam-session.json
```

**Privacy:** Raw video is never stored by default. Only timestamped
behavioural proxy features are persisted. Frames are processed in memory
and never leave the local process.

**Outputs are behavioural proxies only.** They are NOT engagement,
psychological, clinical, or diagnostic conclusions.

## rPPG Synthetic Demo

Runs the full rPPG pipeline over a deterministic **SYNTHETIC** RGB trace
with a known pulse frequency. Requires no webcam, no dataset, and no
network access.

```bash
uv run python -m engagevr rppg-demo \
  --bpm 72 \
  --duration 30 \
  --fps 30 \
  --method pos \
  --seed 42 \
  --output artifacts/rppg-demo.json
```

`--method` accepts `green`, `chrom`, or `pos`.

> **This is a software self-check, not validation.** The reported error
> measures how well the pipeline recovers a frequency that the program
> itself inserted. It is not evidence about real physiological signals
> and must never be presented as rPPG accuracy.

BPM precision is bounded by the Welch frequency resolution, which every
result reports. At the default settings that bin is 0.125 Hz = 7.5 BPM,
so a recovered value within one bin of the requested value is the best
outcome available — not an error.

When signal quality is insufficient, the heart rate is reported as
`unavailable` with an explicit reason. Poor signal quality means the
camera signal is unreliable; it never means low engagement.

## rPPG Methods

| Method | Reference | DOI |
|--------|-----------|-----|
| GREEN | Verkruysse, Svaasand & Nelson (2008) | [10.1364/OE.16.021434](https://doi.org/10.1364/OE.16.021434) |
| CHROM | de Haan & Jeanne (2013) | [10.1109/TBME.2013.2266196](https://doi.org/10.1109/TBME.2013.2266196) |
| POS | Wang, den Brinker, Stuijk & de Haan (2017) | [10.1109/TBME.2016.2609282](https://doi.org/10.1109/TBME.2016.2609282) |

Equations, windowing, overlap assumptions, and every deviation from the
published algorithms are documented in
[docs/REFERENCES.md](docs/REFERENCES.md).

**No method is universally superior.** Relative performance depends on
illumination, motion, camera, and subject, and nothing in this
repository establishes a ranking between them.

## Public-Dataset Evaluation (UBFC-rPPG)

**This software never downloads datasets.** Obtain UBFC-rPPG through its
official channel and satisfy yourself that your use is permitted — see
[docs/DATASETS.md](docs/DATASETS.md), which records that the dataset's
licensing **requires manual verification**.

```bash
uv run python -m engagevr rppg-evaluate \
  --dataset ubfc-rppg \
  --root /path/to/UBFC-rPPG \
  --method pos \
  --output artifacts/ubfc-evaluation.json
```

Or set `rppg.datasets.ubfc_rppg_root` in `configs/defaults.yaml`.

**Public-dataset evaluation is PENDING.** UBFC-rPPG is not present in
this environment. No MAE, RMSE, bias, or coverage figure against any
public dataset exists in this repository, and error metrics are only
ever computed against genuine reference physiological signals.

## Task Environment, Backend, and Replay (Milestone 4)

Everything below runs locally with **no Unity, no webcam, no model asset, no
display server, and no internet access**.

> The local backend has **no authentication, no authorization, and no
> transport encryption.** It binds to loopback by default; binding elsewhere
> requires an explicit flag. Do not expose it to a network.

### End-to-end demonstration

Terminal 1 — start the backend:

```bash
uv run python -m engagevr serve \
  --host 127.0.0.1 \
  --port 8000 \
  --session-root artifacts/sessions
```

Terminal 2 — run the SYNTHETIC task simulator against it:

```bash
uv run python -m engagevr task-sim \
  --seed 42 \
  --blocks 2 \
  --trials-per-block 10 \
  --connect ws://127.0.0.1:8000/ws/v1/sessions/demo-session
```

Then inspect and replay the recording:

```bash
uv run python -m engagevr session-inspect artifacts/sessions/demo-session

uv run python -m engagevr session-replay \
  artifacts/sessions/demo-session \
  --speed 0
```

Fully offline, with no server at all:

```bash
uv run python -m engagevr task-sim \
  --seed 42 --blocks 2 --trials-per-block 10 --speed 10 \
  --output artifacts/task-session
```

### What the task telemetry is

> Task accuracy, reaction time, and timeout counts are **software
> telemetry**. They are NOT engagement, attention, cognitive-load, or
> fatigue measurements. The task has not been experimentally designed,
> piloted, or approved. Every simulated response is fabricated from a seed:
> no person performs the simulated task.

### Protocol

Protocol version **1.0**, shared by the backend, the Python simulator, the
replay player, and the Unity client. The JSON Schema and contract fixtures
are checked in under `protocol/` and are parsed by **both** the Python and
the Unity test suites, so the two cannot drift apart. See
[docs/PROTOCOL.md](docs/PROTOCOL.md).

```bash
uv run python scripts/generate_protocol_artifacts.py   # regenerate
```

### Session recordings

```
artifacts/sessions/<session-id>/
  manifest.json   events.jsonl   summary.json   dropped.jsonl
```

Append-only JSON Lines, atomic summary, crash recovery with 1-based
malformed-line numbers. A recording contains protocol envelopes and receiver
ingestion metadata **only** — no frames, no video, no landmarks, no
predictions, no engagement or cognitive-load values, no real-world identity.
See [docs/SESSION_FORMAT.md](docs/SESSION_FORMAT.md).

### Replay

Replay preserves every original message and **adds** separate replay
metadata. A replayed synthetic message carries `SYNTHETIC` *and* `REPLAY`.
The source recording is opened read-only and is never modified. See
[docs/REPLAY.md](docs/REPLAY.md).

### Adaptation

Milestone 4 implements command **transport** only — there is no policy, no
cooldown, no hysteresis, and no personalization. Commands are issued
manually or by a test script, and nothing claims that applying one improves
engagement or any other outcome.

### Unity desktop task

Source only, at `unity/EngageVR/`. **It has not been compiled or executed**:
no Unity Editor is installed in this environment. See
[docs/UNITY_SETUP.md](docs/UNITY_SETUP.md).


## Feature Datasets and Baseline Models (Milestone 5)

### Build a deterministic SYNTHETIC feature dataset

```bash
uv run python -m engagevr features-demo \
  --seed 42 \
  --subjects 30 \
  --sessions-per-subject 2 \
  --windows-per-session 20 \
  --output artifacts/datasets/m5-synthetic.parquet
```

Writes three files: the Parquet table, a `*.metadata.json` provenance
document carrying the SHA-256 dataset fingerprint, and a
`*.feature_catalog.json` snapshot of the catalog it was built against.
Running it twice with the same seed produces the **same fingerprint** —
wall-clock values are excluded from the canonical content on purpose.

Every row and every target is permanently labelled `SYNTHETIC` and sets
`scientific_evaluation_permitted: false`.

### Run baseline software verification

```bash
uv run python -m engagevr baseline-demo \
  --dataset artifacts/datasets/m5-synthetic.parquet \
  --target engagement_class \
  --folds 5 --seed 42 \
  --output artifacts/experiments/m5-engagement-demo
```

Prints `SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION` before and after
the results, along with the dataset fingerprint, data-source counts, the
target and task type, the grouping field and group count, the split
strategy and why it was chosen, the fold count, every model evaluated, and
the artifact locations.

Targets: `engagement_class`, `engagement_score`, `cognitive_load_class`,
`cognitive_load_score`.

### Generic training command

```bash
uv run python -m engagevr baseline-train \
  --dataset /path/to/windowed-features.parquet \
  --target engagement_class \
  --mode scientific \
  --output artifacts/experiments/run-name
```

`--mode scientific` **refuses** any dataset with a synthetic row, a target
that forbids scientific evaluation, or an unstated target source type, and
exits non-zero. It is expected to refuse every dataset this repository can
currently produce.

### What the modelling layer is

- **Windowed feature dataset** — fixed-duration, half-open windows; 61
  catalogued features across five modality groups; separate columns for
  values, availability, modality availability, modality quality, targets,
  and target provenance; missing measurements stay null.
- **Grouped cross-validation** — subject grouping, falling back to session
  grouping, and **refusing** when neither yields two independent groups.
  There is no row-level fallback.
- **Baselines** — dummy, logistic regression / ridge, random forest,
  scikit-learn histogram gradient boosting, and a clearly labelled
  deterministic rule-based software-check baseline.
- **Calibration** — sigmoid and isotonic, fitted on groups disjoint from
  those used to fit the base estimator and never on the outer test fold.
- **Ablations** — nine feature subsets on identical folds. These are
  feature-subset comparisons, **not** multimodal fusion.
- **Experiment records** — a directory of JSON and Parquet per run, with a
  checksum file and an atomically written manifest. No MLflow yet.

See [Feature Dataset](docs/FEATURE_DATASET.md),
[Baseline Models](docs/BASELINE_MODELS.md),
[Model Evaluation](docs/MODEL_EVALUATION.md), and
[Experiment Tracking](docs/EXPERIMENT_TRACKING.md).

### What it is not

No model here has been fitted to a real participant label, because none
exists. No number produced by these commands is model accuracy, engagement
validity, cognitive-load validity, generalisation evidence, or a
psychological, clinical, or experimental conclusion. No model is a
champion and none is production-ready.

## Privacy

- Raw video is never stored by default.
- ROI pixels exist in memory for one frame and are discarded immediately
  after spatial averaging. **No ROI image is written, logged, or
  transmitted.**
- Persisted rPPG artifacts contain window summaries, quality components,
  and estimates — never frames, per-sample pixel arrays, or identifiers.
- Nothing infers skin tone, ethnicity, identity, or emotion.
- No data leaves the local machine.
- Session recordings contain protocol messages and receiver metadata only.
  Frames, video, landmarks, model predictions, engagement values,
  cognitive-load values, secrets, and real-world identities are not
  representable in one.
- Session identifiers are pseudonymous and validated against path traversal.

## Project Structure

```
src/engagevr/         Python package
  config.py           Configuration loading (YAML + Pydantic)
  _logging.py         Structured JSON logging
  utils/              Timestamp and utility functions
  schemas/            Pydantic data contracts (including capture schemas)
  capture/            Webcam acquisition and frame quality
  face/               Face landmarks and behavioural features
  head_pose/          Head-pose estimation and motion features
  rppg/               ROI, RGB trace, GREEN/CHROM/POS, HR, quality
  datasets/           Public-dataset adapters (no downloading)
  simulator/          Synthetic data generation (Milestone 1)
  protocol/           Shared versioned wire protocol (M4)
  transport.py        In-process / JSONL / WebSocket transports (M4)
  task/               Deterministic SYNTHETIC task simulator (M4)
  api/                FastAPI backend and WebSocket bridge (M4)
  synchronization/    Clock and ordering diagnostics (M4)
  storage/            Append-only JSONL session recordings (M4)
  replay/             Deterministic session replay (M4)
  cli_milestone4.py   serve / task-sim / session-inspect / session-replay
  features/           Windowed feature datasets: catalog, windowing,
                      aggregation, assembly, validation, synthetic (M5)
  training/           Splits, preprocessing, models, calibration, metrics,
                      ablation, runner, artifacts (M5)
  cli_milestone5.py   features-demo / baseline-demo / baseline-train
configs/              YAML configuration files
protocol/             Checked-in JSON Schema and contract fixtures (M4)
scripts/              Model download, protocol-artefact generation
unity/EngageVR/       Unity desktop task, source only (NOT compiled)
tests/                pytest test suite
docs/                 Project documentation
```

## Documentation

- [Project Plan](docs/PROJECT_PLAN.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Research Questions](docs/RESEARCH_QUESTIONS.md)
- [Limitations](docs/LIMITATIONS.md)
- [Decisions](docs/DECISIONS.md)
- [Progress](docs/PROGRESS.md)
- [Datasets](docs/DATASETS.md)
- [Method References](docs/REFERENCES.md)
- [Protocol](docs/PROTOCOL.md)
- [Task Simulator](docs/TASK_SIMULATOR.md)
- [Session Format](docs/SESSION_FORMAT.md)
- [Replay](docs/REPLAY.md)
- [Unity Setup and Status](docs/UNITY_SETUP.md)
- [Feature Dataset](docs/FEATURE_DATASET.md)
- [Baseline Models](docs/BASELINE_MODELS.md)
- [Model Evaluation](docs/MODEL_EVALUATION.md)
- [Experiment Tracking](docs/EXPERIMENT_TRACKING.md)

## Disclaimer

EngageVR is research software. Its rPPG heart-rate values are
signal-processing estimates from camera data, not medical measurements,
and **must not be used for any medical, diagnostic, screening, or
monitoring purpose.** Its engagement and cognitive-load outputs are model
estimates and are not psychological or clinical conclusions. Its task
telemetry is a software measurement, not a measurement of attention,
cognition, workload, or fatigue.

No validated participant engagement or cognitive-load label exists in this
project. Every model metric here was computed from SYNTHETIC data and is a
software self-check, not evidence about any person.

## License

To be determined.
