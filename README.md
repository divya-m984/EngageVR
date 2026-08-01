# EngageVR

**An Uncertainty-Aware Multimodal Framework for Personalized Engagement Estimation and Adaptive Virtual Reality**

> EngageVR is a research software prototype. Its engagement and cognitive-load
> outputs are model estimates and must not be treated as medical, psychological,
> or diagnostic conclusions.

## Status

**Milestone 2 -- Webcam Behavioural Capture** implementation complete;
physical-webcam validation pending. Webcam capture, face landmarks
(MediaPipe), behavioural proxy features (EAR, blink, mouth, head pose),
and capture quality metrics are implemented. No rPPG, ML models, or Unity
integration yet.

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

## License

To be determined.
