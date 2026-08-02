# EngageVR

**An Uncertainty-Aware Multimodal Framework for Personalized Engagement Estimation and Adaptive Virtual Reality**

> EngageVR is a research software prototype. Its engagement and cognitive-load
> outputs are model estimates and must not be treated as medical, psychological,
> or diagnostic conclusions.

## Status

**Milestone 3 -- Interpretable rPPG Signal-Processing Pipeline**
implementation complete; physical-webcam and public-dataset validation
pending.

Implemented: webcam capture, face landmarks (MediaPipe), behavioural
proxy features (EAR, blink, mouth, head pose), capture quality, and a
classical rPPG pipeline — skin ROI extraction, RGB traces, GREEN /
CHROM / POS extraction, Butterworth band-pass filtering, spectral
heart-rate estimation, an interpretable signal-quality index, and a
UBFC-rPPG dataset adapter.

Not implemented: ML models, cognitive-load models, multimodal fusion,
personalization, HRV, FastAPI, Streamlit, Unity, MLflow, DVC, deep
learning.

> **rPPG heart-rate values are signal-processing estimates from camera
> data.** They are not medical measurements, have not been validated
> against any reference device, and are not engagement or cognitive-load
> values.

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

## Privacy

- Raw video is never stored by default.
- ROI pixels exist in memory for one frame and are discarded immediately
  after spatial averaging. **No ROI image is written, logged, or
  transmitted.**
- Persisted rPPG artifacts contain window summaries, quality components,
  and estimates — never frames, per-sample pixel arrays, or identifiers.
- Nothing infers skin tone, ethnicity, identity, or emotion.
- No data leaves the local machine.

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
  simulator/          Synthetic data generation
configs/              YAML configuration files
scripts/              Model download and setup scripts
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

## Disclaimer

EngageVR is research software. Its rPPG heart-rate values are
signal-processing estimates from camera data, not medical measurements,
and **must not be used for any medical, diagnostic, screening, or
monitoring purpose.** Its engagement and cognitive-load outputs are model
estimates and are not psychological or clinical conclusions.

## License

To be determined.
