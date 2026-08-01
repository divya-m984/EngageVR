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
