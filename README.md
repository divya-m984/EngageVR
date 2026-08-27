# EngageVR

**An Uncertainty-Aware Multimodal Framework for Personalized Engagement Estimation and Adaptive Virtual Reality**

> EngageVR is a research software prototype. Its engagement and cognitive-load
> outputs are model estimates and must not be treated as medical, psychological,
> or diagnostic conclusions.

## Status

**Milestone 9 research dashboard implementation complete; scientific
evaluation and human-subject validation remain pending.**

Implemented: webcam capture, face landmarks (MediaPipe), behavioural proxy
features, capture quality, a classical rPPG pipeline (GREEN / CHROM / POS,
spectral heart rate, interpretable quality index, UBFC-rPPG adapter), a
shared versioned protocol (`1.0`), a deterministic task simulator, a local
FastAPI + WebSocket bridge with bounded-queue backpressure, append-only
JSONL session storage with crash recovery, deterministic session replay, a
Unity desktop task at source level, a windowed feature dataset with a
versioned feature catalog and SHA-256 fingerprint, interpretable baseline
models under grouped cross-validation, offline probability calibration,
feature-group ablations, local experiment records, and — new in Milestone
6 — multimodal fusion: early feature fusion, late decision-level fusion,
quality-aware fusion, modality-specific experts, leakage-safe stacking,
missing-modality robustness scenarios, expert-disagreement diagnostics,
and personalized calibration reported separately from a population
baseline; Milestone 7 — calibrated confidence, split-conformal prediction
intervals, selective prediction with seven abstention reason codes, an
evidence gate, population and label-free personalized thresholds,
coverage and risk-coverage curves, and a confidence-aware adaptation
**gate**; and — new in Milestone 8 — a conservative, deterministic
adaptation **policy**: an ordinal state-to-direction demonstration rule,
conservative conflict resolution, a dwell requirement, a cooldown, session
adaptation bounds and budget, an explicit session-scoped policy state, a
pure proposal-to-command builder that sends nothing, and 15 deterministic
controller scenarios; and -- new in Milestone 9 -- a local **READ-ONLY**
research dashboard (Streamlit) that discovers experiment runs by artifact
signature, verifies their checksums, renders baseline, fusion,
personalization, uncertainty, and adaptation artifacts with their recorded
provenance on every page, observes and replays recorded task sessions
without writing to them, and exports a deterministic session report.

Not implemented: online inference, autonomous dispatch of a policy-derived
command, a learned or reinforcement-learning policy, stimulus-pacing or
scene-content adaptation, break triggering, fatigue estimation, HRV, MLflow,
DVC, Docker, deep learning. No deep or neural fusion exists in this
repository, and no per-subject model is trained from scratch. The
dashboard's live mode observes an existing **recording**; it performs no
inference and produces no estimate of its own.

Pending validation: **Unity compilation and runtime** (no Unity Editor is
installed here), **physical-webcam capture**, **UBFC-rPPG evaluation**,
**any evaluation on real participant labels** — none exist — and **any
human-subject evaluation of adaptation**. No adaptation proposed by this
policy has ever been shown to a person, and the dashboard has never been
used by anyone but its author.

> **rPPG heart-rate values are signal-processing estimates from camera
> data.** They are not medical measurements, have not been validated
> against any reference device, and are not engagement or cognitive-load
> values.

> **Every model and fusion metric in this repository was computed from
> SYNTHETIC data.** Those numbers are software self-checks. They are not
> model accuracy, not engagement validity, not cognitive-load validity, and
> must never be compared with a published result on real data. No fusion
> strategy here is a champion, and a comparison on synthetic data cannot
> select a best fusion architecture.

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

Milestone 4 implements command **transport** only. Commands here are issued
manually or by a test script, and nothing claims that applying one improves
engagement or any other outcome. The Milestone 8 *policy* that decides
whether to propose one is separate and dispatches nothing — see
[Adaptive Environment](#adaptive-environment-milestone-8) below.

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
  feature-subset comparisons, **not** multimodal fusion; fusion is
  Milestone 6, below.
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

## Multimodal Fusion (Milestone 6)

### Run fusion software verification

```bash
uv run python -m engagevr fusion-demo \
  --dataset artifacts/datasets/m5-synthetic.parquet \
  --target engagement_class \
  --folds 5 --seed 42 \
  --strategies early uniform-late quality-late \
  --output artifacts/experiments/m6-engagement-fusion
```

Regression targets work the same way (`--target engagement_score`).
`fusion-train --mode scientific` refuses synthetic data, unstated target
provenance, and synthetic modality dropout, and exits non-zero.

### What fusion is here

- **Four measurement modalities** — behavioural, head pose, rPPG, task.
  `quality` is **not** a modality: capture-quality diagnostics,
  availability flags, and missingness indicators are support signals that
  explain a measurement rather than being one, and naming `quality` as a
  modality is rejected.
- **Early feature fusion** — modality features concatenated into one
  matrix, in catalogue order, with availability carried separately and
  fold-local preprocessing. It is concatenation; it is not attention.
- **Late decision-level fusion** — one estimator per modality, combined by
  a weighted average over the experts that actually produced a prediction.
- **Quality-aware fusion** — the same combination with weights derived from
  modality availability and recorded signal quality, using one documented
  equation and deterministic equal base weights.
- **Optional** — validation-derived weights from inner groups only, and
  leakage-safe stacking with an independent out-of-fold assertion (both off
  by default).
- **Missing-modality robustness** — ten deterministic scenarios plus
  optional seeded synthetic dropout, with coverage recorded for each.
- **Expert disagreement** — an ensemble-disagreement diagnostic. It is not
  uncertainty, it is not signal quality, and it does not trigger
  abstention. Milestone 7 carries it beside its own output under this
  same name.

A missing modality is represented through **availability**, never through a
zero, a uniform probability vector, or the training mean. A window that
cannot meet the minimum-modality rule is recorded as unfused with a stated
reason.

See [Multimodal Fusion](docs/MULTIMODAL_FUSION.md).

### What fusion is not

Fusion does not make an estimate valid. No fused output has been evaluated
against a real participant label. Signal quality is a statement about the
measurement, never about the person, and a low quality value is never a low
engagement value.

## Personalization (Milestone 6)

### Run personalization software verification

```bash
uv run python -m engagevr personalization-demo \
  --dataset artifacts/datasets/m5-synthetic.parquet \
  --target engagement_class \
  --folds 5 --seed 42 \
  --calibration-windows 5 \
  --output artifacts/experiments/m6-personalization-class
```

Regression targets work the same way (`--target engagement_score`).
`--calibration-windows 0` requests cold-start mode.
`personalization-train --mode scientific` refuses synthetic data and
unstated target provenance, and exits non-zero.

### What personalization is here

The documented path is `early-fusion population prediction -> subject
calibration/correction -> personalized prediction`. Personalization layers
on the fused population model; it does not create a parallel model stack,
it retunes no fusion weight, and it never overwrites the population
prediction.

- **Population-only baseline** — the control.
- **Personal-baseline feature calibration** — `z_s(x) = (x - mu_s) /
  sigma_s`, with `mu_s` and `sigma_s` estimated from that subject's
  calibration windows only.
- **Few-shot correction** — for regression, `b_s = mean(y_calibration -
  y_population_prediction)` and `y_personalized = y_population_prediction +
  b_s`; for classification, a smoothed and shrunk per-subject log-odds
  shift, renormalised.
- **Population model plus user-specific correction** — both of the above,
  and the shipped default.
- **Cold start** — when a subject has no usable personal evidence, the
  population model is used, `personalization_applied` is false,
  `cold_start` is true, and the reason is stated. It is an outcome, never a
  method to request.

Each held-out subject is cut in **wall-clock time** into a calibration
region and a strictly later evaluation region; a window that straddles the
boundary is excluded from both and listed. The population model is fitted
on other subjects only. Population and personalized results are then scored
over **exactly the same evaluation windows** and written as two separate
results in `metrics.json`.

Identifiers, timestamps, targets, target provenance, availability flags,
modality flags, and modality-quality columns are **never** personalised.
Signal quality describes the measurement; normalising it against a personal
baseline would present it as a personal physiological value.

See [Multimodal Fusion](docs/MULTIMODAL_FUSION.md).

### What personalization is not

Personalized calibration here means adapting a population model to one
subject. It is **not** uncertainty calibration, not a confidence estimate,
and nothing abstains. Personalized confidence thresholds and selective
prediction are Milestone 7, below.

Population and personalized results are reported separately, which is what
Milestone 6 acceptance criterion 3 asks for. **A difference between them
computed on synthetic data is not evidence of a personalization benefit.**
On this repository's generator the personalized variants score *worse* than
the population baseline; that describes a generator whose targets track
absolute feature levels, not a person and not personalization. Whether
personalized baselines outperform population models is unanswered.

## Uncertainty, Selective Prediction, and Abstention (Milestone 7)

```bash
uv run python -m engagevr uncertainty-demo \
  --dataset artifacts/datasets/m5-synthetic.parquet \
  --target engagement_class --folds 5 --seed 42 \
  --output artifacts/experiments/m7-engagement-uncertainty

uv run python -m engagevr uncertainty-demo \
  --dataset artifacts/datasets/m5-synthetic.parquet \
  --target engagement_score --folds 5 --seed 42 \
  --interval-width-grid 0,0.1,0.25,0.5,0.75,1.0 \
  --output artifacts/experiments/m7-engagement-regression-uncertainty
```

`--interval-width-grid` sweeps maximum interval widths **in the target's
own units**. Omit it and the run reports its operating point only, rather
than manufacturing a curve from the classification confidence grid.

### Five things, five fields

Signal quality, predicted probability, probability calibration, model
confidence, and ensemble disagreement are **different concepts** and stay
in different fields. There is no field anywhere named merely
`uncertainty`. **Low signal quality is never low engagement**, and quality
is never multiplied into a model probability: the two gate an actionable
estimate independently, each with its own reason code.

### Classification

```
confidence = max_c p_calibrated(c | x)     accept if confidence >= tau
H(p)       = -sum_c p_c ln(p_c)            (nats)
margin     = p_(1) - p_(2)
```

The acceptance boundary is **inclusive**. When a fold produces no
calibrator, the identical maximum is recorded as a `selection_score` under
an explicitly uncalibrated policy and confidence-based abstention is
**refused** — the schema will not let an uncalibrated maximum be persisted
as calibrated confidence.

`0.70` is an **engineering default**, not an optimum and not a production
threshold. An optional leakage-safe estimator can choose a threshold from
each fold's calibration groups; it never reads an outer-test label, and it
reports *unavailable* rather than inventing one for an unreachable target.

### Regression

A point prediction has no class probability, so it gets a **split-conformal
interval**:

```
r_i = |y_i - yhat_i|            on calibration groups only
k   = ceil((n + 1) * (1 - alpha))
q   = the k-th smallest residual
interval(x) = [yhat(x) - q, yhat(x) + q]
```

When `k > n` the interval is **unavailable** with a reason — never widened
to infinity, never fabricated. `1 - interval_width` is not computed
anywhere; it would be a probability-shaped number with no probabilistic
meaning. A missing interval is never treated as width zero.

The conformal coverage guarantee assumes **exchangeability**, which grouped
cross-validation over different people does not satisfy. **No coverage
guarantee is claimed for real EngageVR data.**

### Personalized thresholds

A per-subject threshold shrunk toward the population value, derived from
the subject's own earlier windows under Milestone 6's wall-clock boundary.
**It reads no labels at all** — only the confidence scores the population
model assigned to those windows — so an evaluation label cannot influence
it by any path. Below the minimum evidence, the subject falls back to the
population threshold with a stated reason.

### Coverage is always reported with performance

```
coverage = accepted / total       accepted + abstained + unavailable = total
```

An abstained window reduces coverage and is **never** converted into an
incorrect prediction or a zero. It keeps its original prediction, its
probabilities, and its diagnostics. `metrics.json` carries an
`all_windows` result and an `accepted_at_applied_threshold` result over the
same folds, so an accepted-set score is never read as a whole-set score.

### Two coverage axes, in opposite directions

The two task types sweep different quantities, and each curve records which
one it was swept over:

| | Classification | Regression |
|---|---|---|
| x-axis | `confidence_threshold` | `maximum_interval_width` |
| units | probability in [0, 1] | the target's own units |
| rule | `accept if score >= tau` | `accept if interval_width <= W_max` |
| raising it | stricter | more permissive |
| coverage | non-increasing | non-decreasing |

A width is never normalised into [0, 1] to reuse the confidence grid, and
never inverted into `1 - width`. The width sweep has its own configuration
key, `uncertainty.regression.interval_width_grid`, which is `null` by
default — the right widths depend on the target's scale. With no grid
configured, **no curve is manufactured**: the run reports its operating
point and marks the width curve unavailable with a stated reason.

### The adaptation gate is not an adaptation policy

The gate answers only *"may an already-chosen action be acted upon?"*. It
cannot choose an action, set a difficulty, address a scene, send a message,
learn, or hold state between windows — `adaptation_gate.py` imports nothing
but two schema modules, and a test asserts that by parsing its AST.
Adaptation policy is Milestone 8, below, and it treats this gate as a hard
prerequisite with **no override**.

**No confidence value, threshold, curve, or interval here is validated.** A
synthetic coverage curve describes this repository's generator; it is not
evidence of real-world calibration, reliability, safety, or usefulness.

See [Uncertainty and Abstention](docs/UNCERTAINTY_AND_ABSTENTION.md).

## Adaptive Environment (Milestone 8)

```bash
# The whole deterministic controller-scenario suite
uv run python -m engagevr adaptation-demo \
  --output artifacts/experiments/m8-adaptation-demo

# One scenario, and what each scenario exercises
uv run python -m engagevr adaptation-demo --scenario direction-reversal
uv run python -m engagevr adaptation-demo --list-scenarios

# The two experimenter controls
uv run python -m engagevr adaptation-demo --experiment-mode static
uv run python -m engagevr adaptation-demo --disable-adaptation
```

This starts no server, opens no socket, and **sends nothing.**

### What the policy is

Given an **eligible** Milestone 7 prediction and the current task state, it
decides whether a conservative adaptation should be **proposed**. That is the
whole of it. It does not retrain a model, recalibrate a probability, redefine
uncertainty, or change the fusion layer.

| Milestone | Question |
|---|---|
| M5 / M6 | What is the estimated state? |
| M7 | Is there enough evidence and confidence to act on it at all? |
| **M8** | **Should a conservative change be proposed?** |
| M4 | How does an explicit command reach a task client? |

### The Milestone 7 gate cannot be bypassed

A `blocked` gate forces `HOLD`. **There is no override flag.** The dependency
is enforced three times over: in control flow; in `AdaptationProposal`, which
embeds both targets' gate records and refuses to validate unless both are
`eligible`; and in `AdaptationPolicyDecision`. Milestone 7's reasons are
preserved verbatim, in Milestone 7's canonical order. Milestone 8 never
recomputes `max(probabilities)`, never lowers a threshold, and never uses
entropy or margin to make a blocked window eligible.

### The mapping is an engineering demonstration rule

Two principles and a default, not nine independent choices. **P1, overload
protection:** high cognitive load suggests a decrease. **P2, engagement
headroom:** high engagement suggests an increase, and one is proposed only
when cognitive load is affirmatively *low*. **P3:** everything else holds.

| engagement \ cognitive load | low | medium | high |
|---|---|---|---|
| **low** | hold | hold | **decrease** |
| **medium** | hold | hold *(deadband)* | **decrease** |
| **high** | **increase** | hold | hold *(conflict)* |

Seven of nine cells hold. The policy deliberately does **not** implement "low
engagement, therefore make it harder": the specification's own response to
that state is feedback or stimulus variation, which the protocol cannot
express, so the policy holds and says why. It never issues `pause_task`
either — the only rule that would call for a break is a fatigue rule, and no
fatigue estimator exists here.

**This is not a validated interpretation of human state.**

### HOLD is a first-class decision

Holds are normal and common. Each states at least one of 20 exact reason
codes in a canonical order, and carries **no proposal and therefore no command
payload**. A proposal carries exactly `proposal_eligible` and nothing else;
the schema refuses any other combination.

### Guards

| Guard | Default | Notes |
|---|---|---|
| dwell (persistence) | 3 consecutive supporting windows | a hold **resets** the count, it does not decay it; a blocked window never counts as evidence |
| cooldown | 6 windows | counted in windows, not seconds, so a replay is reproducible; minimum spacing 7 windows |
| difficulty bounds | `[1, 5]`, step 1 | at a bound the policy **holds**; clamping, if any, is recorded with both values |
| session budget | 10 proposals | `null` means unlimited |
| conflict | `hold` | there is deliberately no `prefer_increase` |

**Hysteresis is emergent** from the deadband, the dwell reset, the direction
change reset, and the cooldown. No redundant knob exists, and a test asserts
that none does.

**Confidence never scales the step.** It decided, in Milestone 7, whether the
window may be acted on at all; reusing it as a control gain would let a barely
admissible estimate move the environment further than a clear one.
`AdaptationProposal` carries no confidence field.

**Signal quality can never choose a direction.** It reaches the policy only as
Milestone 7 gate provenance and as a diagnostic. A direction cannot be
recorded without an ordinal state, and an ordinal state comes only from a
declared-ordinal class label or an explicitly configured regression band.

### Proposal, command, dispatch, acknowledgement, applied

Five different facts, never collapsed. Milestone 8 stops at
`command_built`:

```
prediction -> M7 gate -> policy -> proposal -> command object
                                        (Milestone 8 stops here)
```

The builder reuses the **existing** `set_difficulty` action — the protocol did
not change — sets `is_manual=False`, and refuses holds, blocked gates,
out-of-bounds levels, and non-task target roles. `--dispatch` exists only so
that asking for live transport produces a stated refusal. Tests parse every
module's AST and assert that none imports a transport module or calls `send`,
`broadcast`, `publish`, or `dispatch`.

`applied` cannot be recorded without a real Milestone 4 acknowledgement
payload carrying the instant the client applied the change.

### Experimenter controls

`adaptation.enabled` is the **experimenter lock**. `adaptation.experiment_mode`
is the separate **static vs adaptive experimental condition**. They are kept
distinct so that a static condition is a condition rather than an adaptive
policy that happened to propose nothing, and the mode participates in the run
id.

### What the demo produces

`adaptation_trace.parquet` with **one row per policy evaluation, holds
included**; `adaptation_summary.json` with controller metrics;
`adaptation_policy_config.json`; `scenarios.json`; and `checksums.json`. The
trace carries **no wall-clock column**, so two runs produce byte-identical
Parquet and the determinism check is a checksum comparison.

The metrics count what the software did: windows, holds, proposals,
increases, decreases, hold reasons, reversals, proposal spacing, streaks, and
blocked oscillation attempts. **They are not engagement improvement, cognitive
load reduction, learning improvement, comfort, or adaptation effectiveness**,
and none of them describes a person. The optional comparison against a
guard-free controller shows only that the temporal guards reduce action
frequency; it is not a claim that either controller is better for anyone.

See [Adaptive Environment](docs/ADAPTIVE_ENVIRONMENT.md).

## Research Dashboard (Milestone 9)

**READ-ONLY RESEARCH OBSERVABILITY.** The dashboard makes what previous
milestones recorded inspectable. It displays what a run recorded; it computes
no new scientific quantity.

```bash
# Launch the local dashboard (binds to 127.0.0.1:8501, opens no browser)
uv run python -m engagevr dashboard

# Scan the artifact root and report every run, without starting a server
uv run python -m engagevr dashboard-check --artifact-root artifacts/experiments

# List recorded sessions, or build one session's report, without a server
uv run python -m engagevr dashboard-sessions
uv run python -m engagevr dashboard-sessions --session demo-session

# Directly, for development
uv run streamlit run src/engagevr/dashboard/app.py
```

### Three evidence modes

Chosen in the sidebar and never merged into one ambiguous state:

| Mode | Evidence |
|------|----------|
| **Experiment artifacts** (default) | Milestone 5--8 run directories |
| **Live session** | a session recording, re-read automatically every `dashboard.live_refresh_seconds` |
| **Session replay** | a recording already complete or interrupted, navigated by hand |

**Real-time here means observability, not inference.** The live mode
re-reads a recording the Milestone 4 recorder already wrote. It loads no
model, opens no camera, runs no inference, and produces no estimate. A
session recording structurally cannot carry an engagement value, a
cognitive-load value, a confidence, an interval, or an abstention, so each is
shown as *Unavailable* with the reason. Replay navigates records already on
disk; it re-runs no simulator and re-emits nothing.

The live page refreshes on its own -- `st.fragment(run_every=...)`, a native
Streamlit mechanism, at a conservative configured interval with a two-second
floor -- and states `Mode: LIVE OBSERVATION` and `Automatic refresh: every N
seconds` above the evidence. A manual *Read new records* control remains.
**Only that page refreshes:** replay never auto-advances and the artifact
observatory never polls.

### Synthetic, public, and live are labelled in words

Every recorded `data_source` is rendered as a label beside its raw value --
`SYNTHETIC (recorded as 'synthetic')`, `PUBLIC (recorded as
'public_dataset')`, `LIVE (recorded as 'live')` -- together with a statement
of what it does and does not establish. **Neither public nor live implies
scientific eligibility:** where the bytes came from is not a statement that a
study was designed, labelled, approved, or validated.

### Ten artifact pages, plus two session pages

Overview; Dataset and provenance; Signal and feature quality; Baseline
models; Multimodal fusion; Personalization; Uncertainty and abstention;
Adaptive environment; Run integrity; Limitations and scientific status. Plus
Live session observation and Session replay.

### Exportable session report

A deterministic JSON or Markdown report of one recording, downloadable in
the browser or printable from `dashboard-sessions`. Its fingerprint covers
the content and excludes export time, so the same recording reported twice
is byte-identical. Provenance cannot be exported away: `is_synthetic`,
`scientific_evaluation_eligible = false` with its reason, the standing
disclaimer, and the software-self-check banner are required fields.

### Runs are classified by what they contain

Family detection uses artifact signatures, **never directory names**. A
folder called `m7-trial` holding no Milestone 7 document is not a Milestone 7
run; it is `unknown`, which is a real answer and better than a guess. A
directory that exists is not a successful run: the catalogue distinguishes
`completed`, `failed`, `incomplete`, `corrupt`, `unsupported`, and `unknown`,
and a run missing a required artifact never has its numbers displayed as
though it had finished.

### Provenance is carried, not reconstructed

Every result-bearing page renders the provenance banner **at the top** --
not in an expander, not in a footer. For a synthetic run it shows
`SOFTWARE SELF-CHECK -- NOT SCIENTIFIC EVALUATION` and states
`scientific_evaluation_eligible = false` in words. `DashboardProvenance`
refuses to be constructed as eligible when the artifact says synthetic, and
refuses to let a derived view change either flag. There is no
"Mark as validated" control, no "Treat as real" switch, and no configuration
key that could reach those fields.

### The vocabulary stays separate

Signal quality, calibrated classification confidence, predictive entropy,
probability margin, ensemble disagreement, fusion support weight, regression
prediction-interval width, selective coverage, abstention rate, and empirical
interval coverage are ten different quantities. None is a synonym for
another, there is no card named simply "Confidence", and there is **no single
combined uncertainty score** anywhere.

### Classification and regression are not interchangeable

`UncertaintyDashboardData` refuses to hold a calibrated-confidence field on a
regression run and refuses to hold an interval field on a classification run,
so a page cannot show the wrong control. Classification sweeps
`confidence_threshold` (raising it is stricter: coverage non-increasing);
regression sweeps `maximum_interval_width` (raising it is more permissive:
coverage non-decreasing). Neither is relabelled "uncertainty threshold", and
`1 - interval_width` is never computed.

### The adaptation page reports controller behaviour

It has no effectiveness card, no benefit metric, and no field one could
occupy. Proposal, command built, dispatched, acknowledged, and applied are
five separate counts that are never summed: for the current Milestone 8 runs
that reads 19 proposals, 19 commands built, **0 dispatched, 0 acknowledged**,
and the page says in words that nothing reached a running environment.

### An absent value is never a zero

`None` renders as *Unavailable*; `NaN` and infinities are refused at
construction; `0.0` remains a legitimate zero. Counts are integers,
percentages carry `%`, probabilities do not, and an interval width carries
the regression target's own units and never a percent sign.

### Read-only, structurally

The dashboard cannot retrain, recalibrate, re-run, dispatch, acknowledge,
modify a manifest, delete a run, or touch Git. AST tests over its own source
assert the absence of every write, fit, predict, calibrate, and dispatch
call, and of every import of a runner, the transport layer, `joblib`,
`pickle`, and `sklearn`. It reads JSON and Parquet only: `models/*.joblib`
are Python pickles, and loading one executes code in it.

There is no "Apply adaptation" button and no "Run model" button.

See [docs/DASHBOARD.md](docs/DASHBOARD.md).

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
                      ablation, runner, artifacts (M5); fusion algebra,
                      modality experts, stacking, robustness, fusion
                      metrics, fusion artifacts, fusion runner,
                      personalization algebra and runner (M6);
                      uncertainty algebra, uncertainty runner,
                      adaptation gate (M7)
  adaptation/         Conservative adaptation POLICY: mapping, policy,
                      command builder, lifecycle, scenarios, runner (M8)
  cli_milestone5.py   features-demo / baseline-demo / baseline-train
  cli_milestone6.py   fusion-demo / fusion-train /
                      personalization-demo / personalization-train
  cli_milestone7.py   uncertainty-demo / uncertainty-train
  cli_milestone8.py   adaptation-demo
  dashboard/          READ-ONLY research dashboard: catalogue, loaders,
                      formatting, aggregation, presentation, per-family
                      views, session reader/catalogue/views/report,
                      Streamlit components/pages/session_pages/app,
                      launch (M9)
  cli_milestone9.py   dashboard / dashboard-check / dashboard-sessions
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
- [Multimodal Fusion](docs/MULTIMODAL_FUSION.md)
- [Uncertainty and Abstention](docs/UNCERTAINTY_AND_ABSTENTION.md)
- [Adaptive Environment](docs/ADAPTIVE_ENVIRONMENT.md)
- [Research Dashboard](docs/DASHBOARD.md)

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

No adaptation rule in this repository is psychologically validated,
pedagogically optimal, therapeutic, safe, or demonstrated to benefit any
person, and no adaptation it proposes has ever been shown to anyone.

The research dashboard displays evidence; it does not create any. Every run
it can currently show is a software self-check on synthetic data, and
selecting, filtering, aggregating, or plotting such a value does not make it
scientific evidence. Dashboard visualizations do not establish engagement,
cognitive load, psychological state, health status, safety, or adaptation
benefit.

## License

To be determined.
