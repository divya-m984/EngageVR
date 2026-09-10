# MLOps and Packaging (Milestone 10)

## 0. What this milestone is, and is not

Milestone 10 is the **operational layer** around Milestones 5–9. It makes
the existing work reproducible, trackable, versioned, diagnosable,
packaged, and releasable. It adds no modelling.

There is no new engagement model, no new cognitive-load model, no new
fusion strategy, no new personalization algorithm, no new uncertainty
method, and no second adaptation policy or dashboard here. Every pipeline
stage calls a Milestone 5–9 subcommand through the public CLI.

**The five sentences this milestone exists to prevent somebody from
forgetting:**

> Reproducibility is not validity.
> Tracking is not validation.
> Registration is not approval.
> Packaging is not production readiness.
> Drift alerts are engineering diagnostics.

Every structured record Milestone 10 writes carries them, and the schemas
refuse to record anything stronger. Specifically:

- A synthetic document can never carry
  `scientific_evaluation_eligible=true`. The Pydantic validator raises.
- No record may contain `production`, `staging`, `champion`,
  `challenger`, `approved`, `validated`, `certified`, `clinical`, or
  `diagnostic` as a status word. See
  `engagevr.schemas.mlops.FORBIDDEN_STATUS_WORDS`.
- Every synthetic run, manifest, report, and tracked run carries
  `SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION` verbatim.

The scientific position is unchanged from Milestone 9. No validated
participant-labelled engagement dataset exists. No validated
participant-labelled cognitive-load dataset exists. No human-subject
evaluation of adaptation exists. A synthetic run is a software check, not
evidence, and appearing in MLflow does not change that.

---

## 1. Architecture

```
source + config + synthetic generators        (Milestones 1, 5)
              |
        reproducible DVC stages               dvc.yaml, params.yaml
              |
       persisted experiment artifacts         (Milestones 5-8, UNCHANGED)
              |
       MLflow experiment-tracking metadata    mlops/mlflow_tracking.py
              |
       versioned model/artifact manifests     mlops/model_version.py
              |
        drift / system verification           mlops/drift.py, mlops/smoke.py
              |
       Docker / CI / release workflow         Dockerfile.*, ci.yml, RELEASE.md
```

Two boundaries carry the design.

**Nothing mutates an existing artifact.** Tracking and versioning *read*
a run directory and write separate records elsewhere. A run that has been
logged to MLflow is byte-identical to one that has not, which is what
makes the checksums a version record carries mean anything. Tests assert
this directly.

**Nothing confers scientific status.** Every Milestone 10 record is
operational bookkeeping.

### Module layout

| Module | Responsibility |
|---|---|
| `src/engagevr/schemas/mlops.py` | Every persisted M10 record, versioned and strict (`extra="forbid"`) |
| `src/engagevr/mlops/fingerprints.py` | Canonical hashing for configuration, splits, feature schemas |
| `src/engagevr/mlops/model_version.py` | Immutable **logical** model versions; each serialized instance's checksum goes to an execution sidecar |
| `src/engagevr/mlops/mlflow_tracking.py` | The one module that knows MLflow exists |
| `src/engagevr/mlops/drift.py` | Distribution-shift diagnostics |
| `src/engagevr/mlops/pipeline.py` | Stage definitions shared by `dvc.yaml` and `mlops-demo` |
| `src/engagevr/mlops/reproducibility.py` | Logical identity across executions |
| `src/engagevr/mlops/stage_record.py` | The deterministic, DVC-declared representation of a stage |
| `src/engagevr/mlops/execution.py` | Volatile execution metadata and execution-specific artifact checksums, in never-declared sidecars |
| `src/engagevr/mlops/smoke.py` | The integrated software self-check |
| `src/engagevr/cli_milestone10.py` | The six commands |

---

## 2. Dependencies added

Two, both local, neither cloud:

| Package | Version bound | Why |
|---|---|---|
| `mlflow-skinny` | `>=3.15,<4` | The MLflow tracking client. `dvc.yaml`, the CLI, and the tests need `mlflow.tracking.MlflowClient` and a local store. |
| `dvc` | `>=3.67,<4` | The pipeline DAG the PROJECT_PLAN requires. |

Nothing from AWS, GCP, Azure, Kubernetes, Terraform, Airflow, Kubeflow,
BentoML, Ray, Spark, or Kafka is installed, and the PROJECT_PLAN requires
none of them.

### Why `mlflow-skinny` and not `mlflow`

The full `mlflow` distribution pins **`pandas<3`**. Installing it
downgrades this project's pandas from 3.0.3 to 2.3.3 — a major-version
downgrade of the library the modelling code is written against, imposed
to satisfy a bookkeeping layer. That is the wrong trade, and it was
measured before it was rejected: `uv add mlflow` really does perform that
downgrade.

`mlflow-skinny` is the same project's tracking client without the server
stack (no Flask, no gunicorn, no SQLAlchemy, no Alembic, no Graphene, no
Docker SDK, no matplotlib). `import mlflow` works; `mlflow.tracking` works;
a local file store works. What it cannot do is serve a UI or use a
database backend, and neither is required: this milestone must work with
no account and no server.

### One consequence, stated plainly

`mlflow-skinny` caps `protobuf<7`, so adding it downgraded protobuf from
7.36.0 to 6.33.6. That was verified against the whole existing suite
before the dependency was kept: **3241 passed, 1 skipped**, unchanged.
`mediapipe` declares no protobuf constraint and `streamlit` allows
`>=5.26.1,<8`, so 6.33.6 satisfies both.

---

## 3. MLflow

### Tracking is opt-in

`mlops.mlflow.enabled` is `false` in `configs/defaults.yaml`, and
**nothing changes that implicitly**. No Milestone 5–8 command imports the
adapter, calls it, or starts a store. Importing
`engagevr.mlops.mlflow_tracking` does not even import `mlflow`: the client
is imported inside the functions that need it, and a test asserts this by
importing the module in a subprocess and checking `sys.modules`.

The Milestone 5 tests asserting that no `mlruns`, `MLmodel`, or
`meta.yaml` appears in a run directory still pass, unchanged.

Tracking happens when, and only when, you ask for it:

```bash
uv run python -m engagevr mlflow-log --run artifacts/pipeline/experiments/baseline-engagement_class
uv run python -m engagevr mlops-demo --mlflow
```

### The exact local tracking URI

Default configuration `mlops.mlflow.tracking_uri: "mlruns"` resolves to:

```
file://<absolute path of the repository>/mlruns
```

for example `file:///home/you/EngageVR/mlruns`. It is a **local MLflow
file store**. No server, no database, no account, no API key, no network.

Remote schemes (`http`, `https`, `databricks`, `s3`, …) and database
schemes (`sqlite`, `postgresql`, …) are refused at configuration load with
a stated reason, so a misconfiguration fails before anything is written.

`mlruns/` is gitignored. Nothing in it is source.

#### Why `mlruns/` and not `artifacts/mlflow/`

MLflow's file store rejects any run directory with a path component named
`artifacts` — a path-traversal defence added in MLflow 3 — and then
reports the runs it just wrote as *not found*. A store under this
project's own `artifacts/` tree is therefore unusable. `mlruns/` is
MLflow's own convention and was already gitignored.
`engagevr.mlops.mlflow_tracking.assert_usable_store` raises a clear error
rather than leaving a reader to decode MLflow's.

#### Why a database backend is not offered

MLflow 3.15 puts the file store in maintenance mode and raises unless
`MLFLOW_ALLOW_FILE_STORE` is set. Its suggested replacement — a SQLAlchemy
backend — is unavailable in `mlflow-skinny` and would require the full
distribution, i.e. the pandas downgrade above. So the adapter sets the
documented opt-out **for the duration of one client call** and restores
the previous value (including its absence) on the way out. It also sets
`MLFLOW_DISABLE_TELEMETRY` in the same scope, because a local-first
milestone should not make an outbound request to record that it ran.

The upper bound `mlflow-skinny<4` is the guard: if MLflow 4 removes the
file store, this project does not silently follow it.

### Run naming and structure

- **Experiment:** `mlops.mlflow.experiment_name`, default
  `engagevr-software-self-check`. Names containing a status word are
  refused at configuration load.
- **Run name:** `<family>-<engagevr run id>`, e.g.
  `baseline-engagement_class-selfcheck-e56665026705`.
- **Nothing is registered.** `MLOpsRunSummary.registered_model` is
  typed `None` and can hold nothing else. A registry entry carries a
  *stage*, and a stage is a decision somebody made about a model. Nobody
  has made one here.

### Required provenance tags

Every tracked run carries all eight of these
(`engagevr.schemas.mlops.REQUIRED_TRACKING_TAGS`), and
`MLOpsRunSummary` refuses to validate without them:

| Tag | Value for a synthetic run |
|---|---|
| `engagevr.data_source` | `synthetic` |
| `engagevr.is_synthetic` | `true` |
| `engagevr.scientific_evaluation_eligible` | `false` |
| `engagevr.evaluation_mode` | `software_self_check` |
| `engagevr.disclaimer` | `SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION` |
| `engagevr.run_family` | `baseline` / `fusion` / `personalization` / `uncertainty` / `adaptation` |
| `engagevr.run_id` | the EngageVR run identifier |
| `engagevr.version` | the installed EngageVR version |

Plus, always written: `engagevr.limitation` (the full no-inflation
disclaimer), `engagevr.source_directory`,
`engagevr.is_model_training_run` (`false` for an adaptation run — the
controller is a deterministic rule, not a training run), and
`engagevr.registered_model` = `none`.

A synthetic run whose tags said otherwise would fail validation. There is
no path by which tracking a run makes it eligible.

### Parameters logged

`target`, `task_type`, `estimator`, `feature_subset`, `feature_count`,
`seed`, `split_strategy`, `group_field`, `group_count`, `fold_count`,
`calibration_method`, `evaluation_mode`,
`scientific_evaluation_eligible`, `dataset_fingerprint`,
`split_fingerprint`, `engagevr.config_fingerprint`, `engagevr.run_family`,
plus `fusion_strategy` / `personalization_configuration` /
`uncertainty_configuration` when the run wrote one.

Fingerprints are never truncated (a test asserts all three are 64
characters).

### Metrics logged

**Only values the run already computed.** Nothing is recomputed, averaged,
renamed, or derived in the tracking layer, which is what keeps the tracked
number and the artifact number the same number.

- From `metrics.json`: `<model>/<aggregate>` = the recorded mean, plus
  `.standard_deviation` and `.valid_fold_count`.
- From `selective_metrics.json`: `selective/coverage`,
  `selective/abstention_rate`, and the four counts. Namespaced
  separately because **coverage and abstention describe which windows
  were answered, not predictive performance**, and they must never be
  read beside accuracy as though they were.
- From `personalization.json`: `personalization/*`, `population/*`,
  `personalized/*` — population and personalized results stay separate,
  as Milestone 6 requires.
- From `adaptation_summary.json`: `controller/*` — controller
  diagnostics, tagged as a non-training run. **Adaptation activity is not
  adaptation benefit.**

Classification and regression metric names come straight from the run and
are never merged.

An aggregate the run could not compute is recorded in `skipped_metrics`
with the run's own reason. **It never becomes a zero**, because zero is a
legitimate score and would be indistinguishable from "not computable".

### Artifacts logged

The run's JSON documents plus `checksums.json`
(`LOGGED_JSON_ARTIFACTS`), and `predictions.parquet` when it is under
`MAX_PARQUET_BYTES` (8 MiB) — above that it is skipped with a stated
reason, because a tracking store is metadata plus small evidence, not a
second copy of the dataset.

**Never logged, under any setting** (`NEVER_LOGGED`): `models/`,
`*.joblib`, `*.pkl`, `.env`, `secrets/`, `*.mp4`, `*.avi`, `*.png`,
`*.jpg`, `*.npy`, `events.jsonl`. No webcam frame, no face crop, no raw
participant media, and no credential can enter a tracking store.

---

## 4. Model versioning

`ModelVersionManifest` is an **immutable record of one fitted estimator**.
It is not a registry entry. It answers three questions and refuses a
fourth.

**Where did this come from?** `source_run_id`, `source_run_family`,
`source_run_directory`, `dataset_fingerprint`, `split_fingerprint`,
`feature_schema_fingerprint`, `estimator_parameters_fingerprint`,
`feature_catalog_version`, `feature_count`, the embedded
`ConfigurationVersion`, `fold_index`, `is_calibrated`,
`calibration_method`, `estimator_type`, `estimator_class`,
`engagevr_version`, `python_series`, `dependency_versions`.

**Which bytes is it?** `model_artifact_path`, `referenced_checksums` (the
run's recorded digests for `feature_catalog.json`, `splits.json`, and
`metrics.json`), `serialization_format` = `joblib-pickle`,
`serialization_library_version`, and
`model_artifact_integrity_document` — the name of the record that holds
the artifact's actual SHA-256. That digest is **not** in this manifest;
see below.

**What may be said about it?** `evaluation_mode`, `is_synthetic`,
`scientific_evaluation_eligible`, `data_source_counts`, `disclaimers`,
and `limitation`.

**Should it be used?** There is no field. No `stage`, no `alias`, no
`status`, no `promoted`, no `approved_by`. A test asserts none of those
names exists. `MODEL_VERSION_LIMITATION` says so in words on every record.

### One logical version, N serialized instances

The manifest is a DVC-declared output, so everything in it must be
portable. A `.joblib`'s SHA-256 is not — see §6, *Serialized model bytes
are execution-specific*, and DEC-105 — so the relation this layer models
is:

```
one logical model version   ->   N serialized artifact instances
```

The logical version is the *scientific and software* identity of a fitted
estimator: which run, which estimator with which hyperparameters, which
data, split, features, and configuration. Two machines that fit that same
model agree on it, even when their pickles differ.

Each concrete instance's digest lives in
`model_versions.artifact-integrity.execution.json`, beside the version
directory and never DVC-declared. It records, per file: the
pipeline-relative path, the actual SHA-256, the size in bytes, and why the
digest is excluded from portable identity — plus the interpreter, library
versions, and platform of the execution that produced it, because those
are the explanation.

**Artifact integrity is not scientific validity.** A matching SHA-256 says
the bytes on disk are the bytes that were written. It says nothing about
whether the estimator is correct or useful, and two environments
disagreeing is expected rather than alarming.

### The identifier

```
mv-<target>-<model name>-<sha256(...)[:12]>
```

The digest covers the source run id, target, task type, estimator type,
estimator class, the estimator-hyperparameter fingerprint, model name,
dataset fingerprint, split fingerprint, feature-schema fingerprint,
configuration fingerprint, serialization format, and the EngageVR version.
**No wall clock, no random component, and no serialized-artifact checksum
participates**, so re-deriving a version from the same run on any machine
reproduces the identifier rather than minting a new one.

It used to include the `.joblib` SHA-256, and that is exactly what renamed
every model version on the GitHub runner. See DEC-105.

### Safety properties

- **Nothing is loaded.** A `.joblib` is a pickle and loading one executes
  code in it. Versioning hashes the bytes; it never unpickles. A test
  monkeypatches `joblib.load` to raise and the build still succeeds.
- **The run is not modified.** A test hashes the whole run directory
  before and after and asserts equality.
- **Tampering is refused at build time.** If a model file's bytes no
  longer match the digest the run recorded in its own `checksums.json`,
  `build_model_versions` raises rather than certifying content the run
  never produced. Moving the digest out of the portable manifest did not
  change this.
- **`--verify` still re-hashes every model file** and fails on a mismatch,
  now against the integrity record.
- **A failed or interrupted run cannot be versioned.**
- **Generated binaries are never committed.** `artifacts/` and `models/`
  are gitignored, and a test walks `git ls-files` asserting no
  `.joblib`, `.pkl`, `.parquet`, or `.db` is tracked. Version *metadata*
  is what a release records; the binary is regenerated.

```bash
uv run python -m engagevr model-manifest \
  --run artifacts/pipeline/experiments/baseline-engagement_class \
  --output artifacts/pipeline/mlops/model_versions \
  --verify
```

---

## 5. Configuration versioning

A filename is not a configuration version. `configs/defaults.yaml` can
change between two runs that both name it.

`ConfigurationVersion` records the **normalized effective configuration**
— the resolved Pydantic model rendered in JSON mode, so every default is
present, not just what somebody typed — together with a SHA-256 over its
canonical form (`json.dumps(..., sort_keys=True, separators=(",", ":"))`),
plus a snapshot of the sections that shape a run (`project`, `features`,
`training`, `fusion`, `personalization`, `uncertainty`, `adaptation`,
`mlops`).

### Three paths are excluded, each with a recorded reason

These describe *where this machine keeps things*, not *what the pipeline
computed*. Including them would give two identical runs on two machines
different fingerprints, which destroys the fingerprint's only job. Nothing
that can change a number is excluded.

| Path | Reason |
|---|---|
| `rppg.datasets.ubfc_rppg_root` | An absolute path to a locally obtained public dataset. Differs on every machine; never fetched by this software; takes no part in any synthetic pipeline. |
| `logging.file` | A local log destination. Changes where diagnostics are written, nothing about what was computed. |
| `capture.camera_index` | A device index identifying one webcam on one machine. No modelling, tracking, or packaging stage reads it. |

`ConfigurationVersion` refuses to validate if an excluded path has no
stated reason.

### The other two fingerprints

- **Split fingerprint** — SHA-256 over the strategy, group field, split
  count, seed, and each fold's *sorted* group membership. Row counts and
  target distributions are excluded: they are consequences of the group
  assignment, and including them would make a dataset that merely grew
  look like a different split. A fold is a set, so group order does not
  participate.
- **Feature-schema fingerprint** — SHA-256 over the catalog version and
  the **ordered** predictor columns. Order *does* participate: a linear
  model's coefficients are read positionally.

---

## 6. DVC

### The pipeline

`dvc.yaml` defines eight stages; `params.yaml` holds their parameters and
is the single source of truth shared with `engagevr.mlops.pipeline`.

```
dataset-reference ─┬─> baseline ──────┬─> model-version ─┐
                   ├─> uncertainty ───┤                  │
dataset-current  ──┴─> drift          └─> integrity ─────┼─> reproducibility
```

| Stage | Commands | DVC-declared output |
|---|---|---|
| `dataset-reference` | `features-demo --seed 42 …` then `stage-record` | `datasets/reference.parquet`, `.feature_catalog.json`, `mlops/stages/dataset-reference.json` |
| `dataset-current` | `features-demo --seed 1337 …` then `stage-record` | `datasets/current.parquet`, `.feature_catalog.json`, `mlops/stages/dataset-current.json` |
| `baseline` | `baseline-demo` then `stage-record` | `mlops/stages/baseline.json` |
| `uncertainty` | `uncertainty-demo` then `stage-record` | `mlops/stages/uncertainty.json` |
| `model-version` | `model-manifest --verify` | `mlops/model_versions/` |
| `drift` | `drift-check` | `mlops/drift_report.json` |
| `integrity` | `dashboard-check --json` | `mlops/catalogue.json` |
| `reproducibility` | `repro-manifest` | `mlops/reproducibility.json` |

The four generating stages run two commands: the Milestone 5-8 subcommand,
then `stage-record`. What is *declared* is the record, not the run
directory — that is the boundary of DEC-104, and it is what keeps
`dvc.lock` byte-stable. The run directories and the timestamped dataset
metadata are still written, still intact, still readable by the Milestone 9
dashboard; they are simply not DVC outputs.

Everything lands under `artifacts/pipeline/`, which is inside the
gitignored `artifacts/` tree and deliberately **outside**
`dashboard.artifact_root`: rebuilding the pipeline must not silently
reshuffle the run catalogue a reader is looking at.

`baseline` performs training and evaluation in one stage because the
Milestone 5 runner performs them as one operation; splitting them would
mean re-fitting in order to score.

`fusion` and `personalization` are **not** stages. The DAG is a
representative deterministic demonstration of reproducibility, not an
exhaustive enumeration of Milestones 5–9; both remain reachable through
their own commands. A stage per target per family would quadruple the
runtime of a pipeline whose value is that people actually re-run it.

No stage starts a long-running server (no Streamlit, no uvicorn, no
MLflow UI), reaches the network, or touches a participant dataset. Tests
assert each of those.

### Stage principles, enforced by test

Every stage: invokes a real registered `engagevr` subcommand; declares
dependencies; declares outputs; declares its parameters where it has any;
writes only under the pipeline root; contains no inline Python and no
modelling logic.

### Outputs are not DVC-cached

Every `outs` entry is `cache: false`. The demo is **regenerated from
source**, never restored: there is no remote, nothing to `dvc pull`, and
no binary in Git. That is the deliberate answer to "how does a clean
clone reproduce without a remote?" — it runs the pipeline.

### `dvc.lock` is tracked and stable

The invariant this pipeline maintains:

```
clean source tree -> dvc repro -> dvc.lock byte-identical
```

given the same source, `uv.lock`, configuration, synthetic seed, and
parameters. `dvc.lock` is tracked, and a fresh reproduction leaves it
unchanged — no `git checkout -- dvc.lock` step, no "expected churn"
caveat.

Measured, twice over. Two consecutive fresh reproductions in the working
tree produce the same lock, and two **independent source-only trees** —
each `uv sync --locked` then `uv run dvc repro` from scratch — produce
that same lock as each other. A second `dvc repro` in each skips all eight
stages and leaves the digest unchanged.

No digest is quoted here on purpose. The lock legitimately depends on the
source: edit a file a stage declares as a dependency and the lock changes,
which is the mechanism working. What is invariant is that **two
reproductions from the same source agree**, and that is what
`make dvc-verify` and `tests/system/test_dvc_lock_stability.py` check.

Committed with it: `dvc.yaml`, `params.yaml`, `.dvcignore`,
`.dvc/config`, `.dvc/.gitignore`.

### How the stability is achieved: the orchestration boundary

Not by deleting timestamps from where they belong. The Milestone 5–8
runners still write `started_at_utc`, `finished_at_utc`, and
`created_at_utc` into their own provenance, and that is correct — a run
*did* happen at a time. Rewriting those semantics to please a build tool
would be the wrong repair.

Instead:

```
existing runner output          timestamped, intact, NOT DVC-declared
          |
deterministic stage record      engagevr stage-record
          |
DVC-declared output             byte-stable, hashed into dvc.lock
```

The `baseline` and `uncertainty` stages declare
`mlops/stages/<stage>.json` rather than their run directory. Each record
pins the stage's **logical identity** (the run id, itself a hash of the
run's inputs) and checksums every byte-stable file the run produced —
`metrics.json`, `splits.json`, `calibration.json`, and the Parquet
tables. The timestamped documents are listed by path with the reason they
vary, and **without a checksum**, so their contents cannot reach the lock.

The dataset stages do the same: `reference.parquet` and
`reference.feature_catalog.json` are declared, `reference.metadata.json`
is produced but not declared, and its `dataset_fingerprint` becomes the
stage's logical identity.

**A meaningful change still propagates.** Alter `metrics.json` and the
record's checksum for it changes, so the record's own bytes change, so
`dvc.lock` changes and every downstream stage re-runs. What no longer
propagates is the clock, or one machine's heap.

### Serialized model bytes are execution-specific

A stage record has **four** classifications, not two:

| classification | in the record | checksummed | reaches `dvc.lock` |
|---|---|---|---|
| `portable_deterministic` | path + SHA-256 + size | yes, raw bytes | yes |
| `cpu_dependent_numeric` | path + **structure** SHA-256 + reason + tolerance | structure only | yes, the structure digest |
| `execution_specific` | path + reason | no, not here | no |
| `volatile_provenance` | path + reason | no, anywhere | no |

The second row is DEC-106 and is described below. The third row is
DEC-105:

The middle row exists because of a defect GitHub Actions found and local
testing could not. `joblib.dump` writes scikit-learn's raw tree-node
buffer verbatim, and that C struct has **seven bytes of padding per node
that nothing ever initialises**. Measured on this repository's own
`baseline-engagement_class` run, each random-forest artifact carries
112,826 such bytes, around ten thousand of them non-zero heap residue —
and 191 of the 200 trees that are identical field-for-field between the
plain and the calibrated artifact **disagree in that padding**. Two
serializations of one model, in one process, already differ.

So:

> **Byte reproducibility of Python pickle/joblib artifacts is not assumed
> across execution environments unless proven.**
>
> **Logical reproducibility is not serialized binary byte identity.**

`.joblib`, `.pkl`, and `.pickle` are therefore execution-specific by name.
Their real digests go to `<name>.artifact-integrity.execution.json` beside
the stage record, which is never DVC-declared. The files themselves are
written exactly as before, still hashed, still tamper-checked. Nothing was
deleted; a checksum moved to where it is true.

The schema enforces it: `DeterministicStageRecord`, `ReproducibilityStage`,
and `ModelVersionManifest` all **refuse to be constructed** with a
serialized estimator in portable identity. Reintroducing one fails the unit
tests immediately rather than a runner three weeks later.

Classification is explicit, in
`engagevr.mlops.stage_record` — `VOLATILE_ARTIFACT_REASONS`,
`SERIALIZED_ESTIMATOR_SUFFIXES`, and `CPU_DEPENDENT_NUMERIC_REASONS`. A
file this repository has not classified is treated as **portable
deterministic** and checksummed — so if it turns out to vary, the
two-execution test fails loudly rather than the guarantee weakening in
silence. Exclusion is always a named decision.

See DEC-105 for the full investigation.

### Model-derived numbers are CPU-dependent

The question the previous section left to `docs/LIMITATIONS.md` is no
longer open. CI run 34403534631 answered it: on a GitHub runner, ten
artifacts differed from the committed lock, and seven of them belong to
the `uncertainty` stage, **which persists no `.joblib` at all**. That
cannot be pickle padding. It is arithmetic.

numpy and scipy ship OpenBLAS built `DYNAMIC_ARCH`, which selects a kernel
from the CPU it finds at run time. A different kernel accumulates a dot
product in a different order, so a fitted coefficient differs in its last
bits and everything derived from it follows. Forcing
`OPENBLAS_CORETYPE=HASWELL` on the development machine reproduced the
runner's `feature_importance.parquet` **byte for byte**, so this is
demonstrated rather than inferred.

What differs is **only the floating-point values**. Across three kernels
and 76,896 values the worst absolute deviation is **1.0712e-08**, and the
schema, column order, row order, dtypes, predicted labels, integer counts,
identifiers, and missing-value positions are identical every time.

So such an artifact is split rather than dropped:

- **structure digest** — SHA-256 over the schema, ordering, dtypes, every
  non-float value, and the positions of nulls and non-finite floats.
  Floats contribute a token naming their kind, never their value. Exact,
  portable, and what reaches `dvc.lock`.
- **raw digest** — the real SHA-256 and size, in
  `<name>.artifact-integrity.execution.json`, never DVC-declared, still
  detecting corruption and tampering.
- **numerical equivalence** — `engagevr numeric-check --reference R
  --candidate C`, holding every float to `|a-b| <= 1e-6 + 1e-6*|b|` with
  non-finite values compared exactly.

Eight file names are classified, each observed to move in run
34403534631: `metrics.json`, `predictions.parquet`,
`feature_importance.parquet`, `selective_metrics.json`,
`selective_predictions.parquet`, `thresholds.json`, `uncertainty.json`,
`adaptation_gate.parquet`. `splits.json`, `calibration.json`,
`ablations.json`, and `feature_catalog.json` are byte-identical on every
kernel measured and stay portable deterministic.

**Numerical identity is a comparison, not a digest,** and that is forced
rather than chosen. Quantising floats and hashing them would put values on
a grid that two one-ULP-apart values can straddle; with tens of thousands
of floats a straddle is nearly certain, and the fingerprint would differ
across machines for *some* runs. A hash has no notion of "close".

**The tolerance is an engineering portability allowance.** It is 93x the
worst measured deviation and no wider, because these artifacts report
probabilities and scores to a handful of decimals. It is **not** a
scientific uncertainty interval, not a confidence bound, and not evidence
about any model. Nothing in this repository has been evaluated against a
participant-provided label. No output file is rounded, quantised, or
rewritten: the pipeline still writes full precision.

Because the lock now carries no CPU-dependent value, `dvc.lock` is strict
again and `git diff --exit-code -- dvc.lock` remains in CI unchanged.
See DEC-106.

### The accepted numerical reference

Removing floats from identity also removes them from detection, and a
check that compares two executions of the *same* runner catches nothing:
both sides come from one CPU. Measured — mutating all 1,463 floats in
`baseline/metrics.json`, structure untouched, left `dvc.lock`
byte-identical and a same-runner comparison exiting zero. A result that
moved from `0.72` to `0.91` passed every gate.

So the accepted numbers are committed:

```
references/numeric/
  MANIFEST.json                     SHA-256 of every reference file
  experiments__baseline-engagement_class__metrics.json.reference.json
  ...                               10 files, 25,632 accepted values
```

Each reference holds the structure digest and every **finite** float at
full precision, in the canonical traversal order the structure digest
also uses. Nothing is rounded: the tolerance lives in the comparison, not
in the data.

```
engagevr numeric-reference            verify: present, unmodified, complete
engagevr numeric-reference --update   ACCEPT new numbers (review the diff)
engagevr numeric-check --accepted references/numeric --candidate <root>
```

`--accepted` is the **cross-environment** check: the numbers come from
the repository revision, so a runner whose CPU produces materially
different results fails no matter how self-consistent it is. `--reference
<other root>` additionally compares two executions, which is
same-environment agreement only, and the command says which claim it just
supported.

It fails closed, driven by the stage records rather than by the reference
directory: a newly classified artifact with no accepted reference is an
error, and so is a reference the pipeline no longer declares, one edited
without being re-accepted, one listed and missing, and one present but
unlisted.

**Classification is by exact pipeline-relative path**, never by file
name — ten paths, each observed to move in CI run 34403534631. A future
artifact called `metrics.json` under a different stage or target is
portable deterministic and fails loudly if it is not byte-stable.

**The accepted reference set is authoritative for the tolerance.** `atol`
and `rtol` are recorded as numbers in `MANIFEST.json`, in every
reference, and in every stage record; all three must agree, and the
running code's `DEFAULT_TOLERANCE` is checked against the manifest.
`--atol`/`--rtol` are accepted with `--accepted` only when exactly equal
to the committed values and refused otherwise, so the gate cannot be
widened from the command line. Changing the tolerance means editing the
code, re-running `numeric-reference --update`, and reviewing the diff.

**Model version identity does not certify numerical portability.**
`model_version_id` covers provenance, configuration, data, and code;
structure digests cover schema and ordering; neither covers values. Use
`verify_model_version_portability`, which invokes the reference
comparison, when the numbers matter. See DEC-107.

### Where the wall clock went: execution sidecars

Nothing was discarded. Every Milestone 10 deterministic document has a
`<name>.execution.json` beside it recording when it was produced, by
what, under which full interpreter version — and **that sidecar is never
a DVC-declared output**, precisely because its contents change on every
execution.

```
artifacts/pipeline/mlops/
    stages/baseline.json                            declared, portable
    stages/baseline.execution.json                  NOT declared, timestamp
    stages/baseline.artifact-integrity.execution.json
                                                    NOT declared, model SHA-256s
    drift_report.json                               declared, portable
    drift_report.execution.json                     NOT declared
    model_versions/                                 declared, portable
    model_versions.execution.json                   NOT declared
    model_versions.artifact-integrity.execution.json
                                                    NOT declared, model SHA-256s
    reproducibility.json                            declared, portable
    reproducibility.execution.json                  NOT declared
```

The integrity sidecar reuses the `.execution.json` suffix deliberately:
every rule that already kept an execution sidecar out of `dvc.yaml`, out
of the Docker images, and out of every identity applies to it unchanged,
because it *is* one. It is written even when a stage produced no
serialized estimator, so "this stage produced none" is an assertion on
disk rather than a missing file.

The Milestone 10 documents themselves — model versions, the drift report,
the reproducibility manifest, the stage records — now carry **no
`created_at_utc` field at all**. Tests assert that no wall-clock field
name and no ISO-8601 date appears anywhere in them.

Two smaller consequences of the same rule:

- **Python is recorded as a series, not a patch level.** A deterministic
  document says `3.12`; the full `3.12.13` is in the sidecar. The
  compatibility contract is the series, and recording the patch level
  would put every interpreter upgrade into the identity of every
  document.
- **Recorded commands are pipeline-relative.** `stage-record` normalises
  the pipeline root out of the command it stores, so running the demo
  under `/tmp/scratch` and under `artifacts/pipeline` produce the same
  record.
- **`dataset.json` is not among a model version's referenced
  checksums.** It copies the dataset metadata verbatim, creation time
  included, so its digest is volatile; the dataset is pinned by
  `dataset_fingerprint` instead, which excludes the wall clock by
  construction.

### Nothing may rewrite `dvc.lock`

DVC wraps long stage commands and leaves a trailing space on each
continued line — 22 of them in this lock. `pre-commit`'s
`trailing-whitespace` hook strips them, and a later `dvc repro` does not
put them back, because nothing changed. A stripped lock that gets
committed then fails `git diff --exit-code -- dvc.lock` on the next fresh
reproduction: the same CI step as DEC-105, for an entirely unrelated
reason.

`dvc.lock` is therefore excluded from `trailing-whitespace` and
`end-of-file-fixer`. A formatter must not rewrite a generated lock file,
and a unit test asserts both exclusions remain. If `pre-commit` ever
prints `Fixing dvc.lock`, restore it with `dvc repro` before committing.

### `.dvc/config`

Telemetry off (`analytics = false`), auto-staging off
(`autostage = false` — DVC must never run `git add` on this repository's
behalf), version check off (`check_update = false`). No remote is
configured and none is required.

### A second `dvc repro` does no work

Measured: eight stages, all `didn't change, skipping`, and
`dvc status` reports `Data and pipelines are up to date`. DVC re-runs a
stage when its dependencies, parameters, or command change, and none of
those carries a timestamp. CI asserts this.

---

## 7. Reproducibility

### Logical identity, not byte identity

Two correct executions do **not** produce identical bytes, and pretending
otherwise would make a reproducible pipeline look broken. Run manifests
record `started_at_utc` and `finished_at_utc`; dataset metadata records
`created_at_utc`; a model-version record and a drift report each record
when they were built.

So `ReproducibilityManifest` separates three things:

- **Logical identity** — dataset fingerprints, run identifiers,
  model-version identifiers, the drift report fingerprint, the catalogue
  digest, plus the pipeline-relative path and SHA-256 of every artifact
  declared portable deterministic. This is `logical_fingerprint`.
- **Execution-specific record** — the serialized estimators, listed by
  path with the reason their bytes belong to one execution and
  deliberately **without** a checksum here. Their real digests are in the
  artifact-integrity sidecars.
- **Volatile record** — the timestamped documents, listed by path with
  the reason they vary and deliberately **without** a checksum.

**Excluded from identity, always** (a required field of the manifest):
wall-clock time — which appears nowhere in the document at all — absolute
paths and temporary directories, the timestamped provenance documents the
runners write, the SHA-256 of every serialized estimator, MLflow run and
experiment identifiers, the host platform, and the process identifier.

Artifact paths are recorded relative to the pipeline root, never
absolute: an absolute path is a fact about one machine, and a temporary
directory is a fact about one execution.

The manifest is assembled from the stage records, never by walking the
filesystem: the stable-versus-timestamped classification has already been
made, and making it twice would invite the two answers to disagree. The
manifest is itself a DVC-declared output, so it obeys the rule it
describes and carries no wall clock. Everything else carries a stated
`non_determinism_reason`.

### Verifying it

The strongest check is the lock itself:

```bash
rm -rf artifacts/pipeline dvc.lock && uv run dvc repro && sha256sum dvc.lock
rm -rf artifacts/pipeline           && uv run dvc repro && sha256sum dvc.lock
# the two digests must match
```

The logical comparison is still available and says more when something
does differ:

```bash
uv run python -m engagevr mlops-demo
cp artifacts/pipeline/mlops/reproducibility.json first.json
rm -rf artifacts/pipeline
uv run python -m engagevr mlops-demo
uv run python -m engagevr repro-manifest --output second.json --compare first.json
```

Exits zero when the two agree and non-zero, listing each difference, when
they do not. CI runs both.

The two-source-tree proof is opt-in, because two full reproductions take
about six minutes:

```bash
ENGAGEVR_RUN_DVC_SYSTEM_TESTS=1 uv run pytest -m dvc_system
```

Skipping it is not passing it. The always-on regression coverage is
`tests/unit/test_dvc_determinism.py` (77 tests).

**Both are same-machine checks**, and that is their limit: two trees on one
machine cannot detect a difference between two machines. That is exactly
how the DEC-105 defect reached CI. The step is still worth running — it
catches a stage that stopped being deterministic at all — but the
cross-environment answer comes from the container check in
`docs/RELEASE.md` §4 and, definitively, from GitHub Actions.

---

## 8. Drift: a diagnostic, not a diagnosis

### The five methods

| Method | Formula | What it answers |
|---|---|---|
| `missingness_rate_difference` | `P_cur(missing) − P_ref(missing)` | Did a measurement stop arriving? |
| `standardized_mean_difference` | `(mean_cur − mean_ref) / sqrt((var_ref + var_cur)/2)` | Did the centre move, in scale-free units? |
| `kolmogorov_smirnov_statistic` | `sup_x \|F_cur(x) − F_ref(x)\|` | Did the shape move, without assuming normality? |
| `population_stability_index` | `Σ_i (c_i − r_i) · ln(c_i / r_i)` over quantile bins of the reference | How much mass moved between bins? |
| `categorical_total_variation_distance` | `0.5 · Σ_k \|c_k − r_k\|` | The categorical analogue, bounded in [0, 1]. |

Five, not fifteen. Each answers a different question in a way a reader can
check by hand.

Notes: **no p-value is reported** for the KS statistic — it would be a
hypothesis test nobody specified in advance and would grow significant
with sample size alone. PSI substitutes `1e-6` for an empty bin, because
`ln(0)` is undefined; a PSI computed over a bin nobody landed in is a
floor artefact, which is why the bin count is recorded in every report.

### Reference and current are always named

Both sides are explicit arguments. Nothing here guesses which two
directories to compare. Each side records its path, dataset fingerprint,
row count, subject count, data-source counts, synthetic status, and
scientific eligibility.

The report also records the compared feature list, every excluded feature
*with its reason*, every unavailable feature, all thresholds, the minimum
sample count, the histogram bin count, and a `report_fingerprint` that
excludes `created_at_utc` and both absolute paths.

### Unavailability is never zero

A column missing on one side, present but all-null, too thin, constant, or
of a mismatched type is reported **unavailable with a reason**. Its
`statistic` and `exceeded` are both `None` — the schema refuses to let an
unavailable statistic carry a verdict. Zero means "these distributions
agree", and collapsing the two would let an absent measurement read as a
healthy one.

### Target columns never take part

`target__*` and `target_meta__*` are excluded by construction, with
"leakage" as the recorded reason. No shift statistic can be computed from
a label. Identity, window-geometry, provenance, and schema-version columns
are excluded too, each with its own reason.

### Thresholds are engineering diagnostic defaults

| Threshold | Default | Where it comes from |
|---|---|---|
| `minimum_samples` | 30 | Below it the feature is unavailable, never zero shift |
| `histogram_bins` | 10 | PSI quantile bins of the reference |
| `missingness_rate_difference` | 0.10 | A legible round number |
| `standardized_mean_difference` | 0.20 | The conventional "small effect" landmark |
| `kolmogorov_smirnov` | 0.10 | A legible round number |
| `population_stability_index` | 0.20 | The conventional "investigate" landmark from credit-risk practice |
| `categorical_total_variation` | 0.10 | A legible round number |

**Not one was calibrated against an outcome, a participant, or an observed
failure.** Crossing one is an invitation to look at a feature, not a
verdict.

### There is no "MODEL FAILED" status

The report has counts (`features_compared_count`,
`features_exceeding_count`, `features_unavailable_count`) and per-feature,
per-method statistics with their thresholds beside them. There is no
overall pass/fail field and no field an "the model failed" claim could
occupy; a test asserts that `model_failed`, `alarm`, `passed`, `verdict`,
and `status` are all absent from the schema.

`drift-check --fail-on-shift` exists as an opt-in **build gate**, and says
so when it fires: "This is a build gate, not a scientific finding."

### The terminology it will not use

`DriftReportKind` has exactly two members:
`feature_distribution_shift` and `prediction_distribution_shift`. There
is deliberately **no `concept_drift`**: establishing concept drift needs
labels from both periods, and no validated participant-provided
engagement or cognitive-load label exists in this repository.

A test asserts no rendered report contains "participant drift",
"cognitive decline", "disengagement drift", "psychological change", or
"model failed". Missingness is described as measurement availability and
explicitly **never** as disengagement.

---

## 9. System smoke tests

Thirteen checks, `~12 seconds`, no webcam, no network, no Unity, no
browser, no display server, no external dataset, no participant data, no
server:

1. `package_imports` — nine modules across every milestone
2. `configuration_loads`
3. `synthetic_dataset_generated` — two SYNTHETIC datasets
4. `dataset_provenance_preserved` — all-synthetic, ineligible
5. `baseline_pipeline_ran` — the Milestone 5 pipeline end to end
6. `artifact_manifest_validated` — status, eligibility, every checksum
7. `model_version_manifest_validated` — build, write, re-read, verify
8. `mlflow_tracking_local` — a throwaway store, required tags present
9. `drift_diagnostic_ran`
10. `dashboard_catalogue_discovered_run` — Milestone 9 finds the run
11. `backend_application_created` — the FastAPI app is built and its
    `/health`, `/version`, and WebSocket routes exist. **No socket is
    bound and no server is started.**
12. `dashboard_module_imports` — the launch argv is built, headless. **No
    Streamlit process is started.**
13. `protocol_artifacts_current` — the checked-in schema matches the code

The smoke run fits two estimators with one permutation repeat and no
ablations. Statistical adequacy is irrelevant here: the question is
whether the wiring holds, and a check nobody waits for is a check nobody
runs.

Its MLflow store is a `tempfile.TemporaryDirectory` removed when the check
returns, so smoke never accumulates tracking state and never writes into
the project's `mlruns/`.

```bash
uv run python -m engagevr system-smoke          # table
uv run python -m engagevr system-smoke --json   # structured
make smoke
```

Exit code is 0 only when nothing failed. `--no-mlflow` records the
tracking check as **skipped**, and the skip reason says "Skipping is not
passing."

Every output carries `SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION`
and `scientific_evaluation_eligible=false`. A passing smoke check means
the software components interoperate. **It does not mean any model is
accurate, calibrated, useful, or validated.**

---

## 10. Docker

Two images, both packaging code that already existed.

| File | Packages | Entry point | Port |
|---|---|---|---|
| `Dockerfile.backend` | the Milestone 4 FastAPI + WebSocket bridge | `python -m engagevr serve` | 8000 |
| `Dockerfile.dashboard` | the Milestone 9 Streamlit dashboard | `python -m engagevr dashboard` | 8501 |

**No model-serving API was invented.** There is no inference endpoint,
because there is no validated model to serve. A test asserts the words
`predict`, `inference`, `model-server`, and `/invocations` appear in no
instruction of either Dockerfile.

Both images: `python:3.12-slim-bookworm`; install with
`uv sync --locked --no-dev` (never `pip install`); two-stage, so the
runtime carries the virtual environment without uv or its caches; run as
non-root `engagevr` (uid 10001); expose exactly one port; declare a
`HEALTHCHECK` against the application's own liveness route
(`/health` and `/_stcore/health`), implemented with `urllib` so no extra
package is needed to check the image.

No BuildKit cache mounts: they would make these images buildable only by a
daemon with `buildx`, and a packaging milestone whose images cannot be
built by a plain `docker build` has packaged nothing.

`libgl1` and `libglib2.0-0` are installed because OpenCV and MediaPipe are
declared project dependencies. Neither image opens a camera; there is no
device in either.

### `docker-compose.yml`

Two services, `backend` and `dashboard`. **Both ports are published to
`127.0.0.1` only** — neither service has authentication, authorisation, or
transport encryption, and publishing either to a routable interface would
expose an unauthenticated bridge and a filesystem browser for the artifact
root. A test asserts every published port starts with `127.0.0.1:`.

The dashboard mounts `artifacts/experiments` and `artifacts/sessions`
**read-only**, making its read-only property a filesystem fact as well as
a code property. The backend mounts only `artifacts/sessions`
read-write, which is the one thing it writes.

There is no `depends_on`: the dashboard reads files, never calls the
backend, and starting either alone is supported.

No secret appears in either Dockerfile or the compose file, and neither
service reads one. No `secrets:` block, no `env_file`.

### `.dockerignore`

Excludes `.git/`, `.venv/`, `artifacts/`, `models/`, `mlruns/`, `*.db`,
`*.joblib`, `*.pkl`, `data/{raw,interim,processed,synthetic}/`, every
media extension, `.env`, `.env.*`, `secrets/`, `*.pem`, `*.key`, `*.crt`,
`*.p12`, `.claude/`, `CLAUDE.local.md`, editor state, every cache,
`tests/`, `notebooks/`, `unity/`, `docs/`, `.github/`, and `dvc.lock`.

Verified after building: both images contain only `.venv`, `src`,
`configs`, `pyproject.toml`, `README.md` (plus `protocol/` in the
backend), and empty mount points. No `.joblib`, `.parquet`, `.jsonl`,
media file, `.env`, or `mlruns` directory is present. CI re-checks this
on every build.

### Usage

```bash
docker compose config          # validate without building
make docker-build              # or: docker compose build
make docker-up                 # docker compose up -d
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8501/_stcore/health
make docker-down
```

Images are ~2.18 GB each, dominated by MediaPipe, OpenCV, and
scikit-learn. That is the size of this project's declared dependency set,
not overhead the packaging added.

MLflow is **not** exposed as a service. Nothing requires a tracking
server.

---

## 11. GitHub Actions

`.github/workflows/ci.yml`, three jobs, Python 3.12 throughout, no
secrets, `permissions: contents: read`.

| Job | Steps |
|---|---|
| `check` | `uv lock --check`, `uv sync --locked`, `ruff format --check`, `ruff check`, `mypy src`, `pytest`, protocol-drift check |
| `smoke` | `uv sync --locked`, `system-smoke`, `dvc dag`, `dvc repro`, second `dvc repro` does no work, two-execution logical reproducibility, upload synthetic reports |
| `docker` | `docker compose config`, build both images, assert no generated or private state inside them, start the stack, wait for both health checks, curl both, `docker compose down` |

The existing `check` job was extended rather than duplicated.

**The hardware suite is never run.** `ENGAGEVR_RUN_HARDWARE_TESTS` is
never set, so `tests/hardware/` stays skipped. No step uses `pip install`,
fetches an external dataset, or reads `~/.claude/`, an existing
`artifacts/` directory, or any machine-specific path. Tests assert each of
these against the workflow file.

The one uploaded artifact is a set of **synthetic software reports** —
the smoke summary, the reproducibility manifest, the drift report, the
run catalogue — with a seven-day retention. No model binary, no dataset,
no session recording, and no media is uploaded.

GitHub-hosted Actions cannot be fully proven locally: this repository can
validate the workflow's *structure* and run each command by hand, but only
a push can prove the runner environment behaves. That limitation is real
and is not claimed away.

---

## 12. Clean-clone reproduction

```bash
git clone git@github.com:divya-m984/EngageVR.git
cd EngageVR
uv sync --locked
uv run dvc repro          # or: uv run python -m engagevr mlops-demo
make smoke
```

Nothing else is required. No hidden local state, no prior Milestone 5–9
artifact directory, no pre-existing MLflow store, no DVC remote, no
untracked configuration file, no network, and no account. The models
download script (`scripts/download_models.py`) is needed only for live
webcam capture, which no Milestone 10 command uses.

`uv run dvc repro` and `mlops-demo` produce the same eight stages; the
first runs them as a DAG, the second in one process.

### Verified

This was executed, not asserted. A source-only copy of the repository —
excluding `.git`, `.venv`, `artifacts/`, `mlruns/`, `dvc.lock`, every
cache, and every local file — was created in a temporary directory,
initialised as a fresh repository, and then:

```
uv sync --locked                        -> 0
uv run dvc repro                        -> 0, 8 stages ran
uv run python -m engagevr system-smoke  -> 0, 13 passed / 0 failed / 0 skipped
```

The `logical_fingerprint` produced in that clean tree was **identical** to
the one produced in the working tree, as was the configuration
fingerprint. Nothing local was required.

### Verified across environments, after the DEC-105 correction

The check above is a *same-machine* one, and that is precisely the class of
check that missed the DEC-105 defect. The pipeline was therefore also
reproduced inside a clean **Ubuntu 24.04 container** — a different libc
(glibc 2.39 against the development machine's 2.44), a different CPython
patch (3.12.14 against 3.12.13, both installed by `uv python install 3.12`
exactly as CI does), and four CPUs instead of eight:

```
uv sync --locked            -> 0
uv run dvc repro            -> 0, 8 stages ran
git diff -- dvc.lock        -> empty
```

Every declared output matched the development machine byte for byte. Only
the undeclared, timestamped documents and the execution sidecars differed,
which is what they are for.

**What this does not prove.** A container shares the host CPU, so it cannot
reproduce a difference in BLAS kernel dispatch, and it is not GitHub's
runner. The definitive test remains the next CI run.

---

## 13. Generated files, and where they go

| Path | Contents | Git |
|---|---|---|
| `artifacts/pipeline/` | Datasets, run directories, model versions, drift report, catalogue, reproducibility manifest, execution and artifact-integrity sidecars | ignored |
| `artifacts/smoke/` | Smoke scratch output and `smoke_report.json` | ignored |
| `mlruns/` | The local MLflow file store | ignored |
| `.dvc/cache/`, `.dvc/tmp/`, `.dvc/config.local` | DVC runtime state | ignored |
| `models/`, `data/{raw,interim,processed,synthetic}/` | Pre-existing | ignored |

Source-controlled and **not** ignored: `dvc.yaml`, `params.yaml`,
`dvc.lock` (tracked and byte-stable, DEC-100), `.dvcignore`,
`.dvc/config`, `.dvc/.gitignore`, `Dockerfile.backend`,
`Dockerfile.dashboard`, `docker-compose.yml`, `.dockerignore`, the
workflow, and everything under `src/`, `configs/`, `protocol/`, `tests/`,
and `docs/`. Tests assert both directions.

`make clean-mlops` removes `artifacts/pipeline`, `artifacts/smoke`, and
`mlruns` — and deliberately nothing else. It leaves `dvc.lock` alone,
because that file is tracked. It never touches
`artifacts/experiments`, `artifacts/sessions`, or `artifacts/datasets`,
which hold Milestone 5–9 evidence a reader may still be looking at. There
is no `git clean` anywhere in this repository.

---

## 14. Security and privacy

**No secret exists in this milestone and none is required.** No
Dockerfile, compose file, workflow, configuration file, or CLI reads a
credential. GitHub Actions needs no secret, and keeping it that way is a
property worth preserving.

Ignored so a stray local file cannot be committed or baked into an image:
`.env`, `.env.*`, `secrets/`, `*.pem`, `*.key`, `*.p12`, `*.crt`.

**Privacy.** Nothing in this milestone handles real session recordings,
raw media, or personal data — there is none in the repository. Every
example is obviously synthetic and permanently labelled. Raw webcam-video
storage remains disabled by default (`capture.store_raw_video: false`),
and the smoke check reports that setting so a reader sees it.

The tracking layer's `NEVER_LOGGED` list and the Docker context's
exclusions are the two places where "no media, no credentials, no
generated state leaves this machine" is enforced mechanically rather than
by convention.

Model files remain executable content. Nothing in Milestone 10 loads one.

---

## 15. Commands

| Command | What it does |
|---|---|
| `mlops-demo` | The whole deterministic SYNTHETIC pipeline, in one process |
| `model-manifest` | Immutable logical model versions from a run, plus each artifact's integrity record |
| `drift-check` | Distribution-shift diagnostic between two named datasets |
| `mlflow-log` | Log finished runs to a LOCAL MLflow store |
| `repro-manifest` | Build (and optionally compare) a reproducibility manifest |
| `stage-record` | Write one stage's deterministic, DVC-declared representation |
| `system-smoke` | The integrated software self-check |
| `--version` | The package version, from `engagevr.__version__` |

`repro-manifest` builds from the stage records, so it **refuses** a
pipeline that has not been executed rather than emitting a manifest of
absences — a manifest describing a pipeline nobody ran is not useful, and
the records are the only place the byte-stable-versus-timestamped
classification has been made. Pass `--stages` to describe part of a
pipeline deliberately.

Every one has `--help`, an explicit output path, prints its synthetic and
scientific status, and exits non-zero on a validation failure.

### Version identity

One source of truth: `engagevr.__version__`, which matches
`pyproject.toml`. It is **not** derived from Git state — a working tree is
not a release and a tag is not a build. It stays `0.1.0`: reaching
Milestone 10 does not make this a 1.0 research prototype. A test asserts
all three facts.

---

## 16. What Milestone 10 deliberately does not do

- Change the Milestone 4 wire protocol. It is unchanged; the drift check
  is part of CI and of the smoke suite.
- Register a model, promote one, or attach a stage or alias to one.
- Add automatic tracking to any Milestone 5–8 runner.
- Expose MLflow as a service, or require any server.
- Add a model-serving API.
- Add a second dashboard, training pipeline, fusion implementation,
  uncertainty engine, or adaptation controller.
- Commit a generated model binary, dataset, tracking store, or lock file
  full of volatile hashes.
- Turn any synthetic run into evidence.

---

## 17. Limitations

See `docs/LIMITATIONS.md` for the full statement. In short:

- Every number this milestone touches came from synthetic data. It is a
  software self-check, not evidence.
- The drift thresholds are engineering diagnostic defaults. None is
  validated.
- A drift diagnostic has never been run against real data, real drift, or
  a real deployment.
- **Serialized estimator bytes are not portable and are not claimed to
  be.** A `.joblib` embeds uninitialised C struct padding; its checksum is
  an integrity fact about one execution, not a portable identity. This was
  discovered by CI, not by design — see DEC-105.
- **Numerical portability across CPU microarchitectures is untested and
  not guaranteed.** Forcing a different BLAS kernel changes
  `metrics.json`, `predictions.parquet`, and `feature_importance.parquet`
  in the last bits. Those artifacts remain portable deterministic, so a
  runner whose CPU disagrees will fail the lock check loudly rather than
  silently — which is the intended behaviour, and an unresolved question,
  not a solved one.
- The two-execution reproducibility check has been run on one machine and
  in one Linux container that shares that machine's CPU. Reproducibility
  on macOS, on Windows, on ARM, and on a different x86-64 microarchitecture
  is untested.
- The Docker images have been built and health-checked locally; they have
  never been run under load, over time, or by anyone else.
- CI's behaviour on GitHub-hosted runners cannot be proven from this
  repository.
- `mlflow-skinny`'s file store is in maintenance mode upstream. The
  `<4` bound is the guard, and revisiting this is future work, not a
  present defect.
