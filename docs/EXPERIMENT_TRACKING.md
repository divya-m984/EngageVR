# Local Experiment Tracking (Milestone 5)

## Why not MLflow yet

MLflow is a **Milestone 10** deliverable. Adding a tracking server, a
backing store, and a UI now would introduce infrastructure before there is
anything to track: this milestone produces synthetic self-check runs only,
and no run in this repository has been fitted to a real label.

A directory of JSON and Parquet is:

- inspectable with no service running;
- diffable, so a change to a result is visible in a review;
- reproducible from the manifest alone;
- free of a schema migration when the pipeline changes.

When Milestone 10 introduces MLflow, these documents are what it will log.
Nothing here needs to be discarded to adopt it.

No MLflow file is produced by this milestone, and a test asserts that none
appears in a run directory.

## Run directory layout

```
artifacts/experiments/<run-name>/
    manifest.json               written atomically, last
    dataset.json                dataset provenance and fingerprint
    feature_catalog.json        the catalog the run was audited against
    splits.json                 fold assignments and the split audit
    metrics.json                fold and aggregate metrics
    calibration.json            calibration design and per-fold outcomes
    ablations.json              feature-subset comparisons on shared folds
    predictions.parquet         per-row predictions and probabilities
    feature_importance.parquet  fold-level interpretation, unaggregated
    models/
        README.txt              untrusted-pickle warning
        <model>-fold0.joblib
        <model>-fold0-calibrated.joblib
    checksums.json              SHA-256 of every artifact above
```

`artifacts/` is gitignored. **Datasets, predictions, and model files are
never committed.**

## Run identifiers

`build_run_id` produces a stable, collision-resistant identifier:

```
<target>-<selfcheck|sci>-<sha256(...)[:12]>
```

The hash covers the target, the evaluation mode, the dataset fingerprint,
the random seed, the fold count, the sorted model names, the feature set,
the calibration method, and the EngageVR version.

No wall clock and no random component participates. Two runs differing in
any of those inputs get different identifiers; re-running an identical
configuration reproduces the same identifier rather than accumulating
near-duplicate directories. The identifier is insensitive to the *order*
in which models were requested, because that does not change what was run.

The run *directory* is chosen by `--output`; the identifier is recorded
inside it.

## `manifest.json`

Records:

| Field | Contents |
|-------|----------|
| `run_id` | as above |
| `engagevr_version`, `python_version` | resolved at run time |
| `dependency_versions` | numpy, scipy, pandas, pyarrow, scikit-learn, joblib, pydantic |
| `evaluation_mode`, `scientific_evaluation_eligible` | self-check or scientific |
| `dataset_path`, `dataset_fingerprint`, `feature_catalog_version` | what was evaluated |
| `target_name`, `task_type`, `feature_set` | the prediction problem and every predictor column used |
| `model_names`, `model_parameters` | including the estimator class, its parameters, the imputation strategy, whether it was standardised, and its parameter grid |
| `split_strategy`, `group_field`, `group_count`, `fold_count` | the split design |
| `fold_assignments` | train / calibration / test group lists per fold |
| `calibration_method` | requested method |
| `configuration`, `random_seed` | everything needed to re-run |
| `started_at_utc`, `finished_at_utc` | when |
| `status`, `failure_reason` | `completed` or `failed` |
| `artifact_checksums` | SHA-256 per artifact |
| `disclaimers` | required, non-empty |

No Git command is run to obtain any of this. Version metadata comes from
installed package metadata, not from a repository query.

## Write ordering and atomicity

Every document is written to a temporary file in the same directory,
flushed, fsynced, then `os.replace`d over the target. `os.replace` is
atomic within a filesystem, so a reader sees either the previous document
or the complete new one, never a half-written file.

`checksums.json` is written second-to-last and `manifest.json` **last**.
The consequences are deliberate:

- a directory with **no manifest** is an interrupted run — it never
  reached a conclusion, and `read_manifest` says so rather than returning
  a partial result;
- a manifest with `status: failed` is a run that reached a conclusion and
  that conclusion was failure;
- a manifest with `status: completed` is refused if any required artifact
  is missing, so an incomplete artifact set can never read as a success.

Required artifacts for completion: `dataset.json`, `feature_catalog.json`,
`splits.json`, `metrics.json`.

A run that raises mid-way writes a `failed` manifest and re-raises. The
schema enforces that a failed manifest states a reason and that a
completed one does not.

No temporary file survives a successful write; a test asserts the
directory contains no `.tmp` leftovers.

## Checksums

`checksums.json` maps each artifact's relative path to the SHA-256 of its
bytes. `verify_checksums(directory)` returns the artifacts whose current
bytes disagree with the record, so tampering or corruption after the fact
is detectable without re-running anything.

## `predictions.parquet`

One row per (test row × model), carrying `window_id`, `subject_id`,
`session_id`, `fold_index`, `model_name`, the true value, the predicted
value, and — for classification — one column per class for the
**uncalibrated** and one for the **calibrated** probability. Keeping them
in separate columns means the effect of calibration is inspectable without
re-running the pipeline.

Probability rows sum to one: `aligned_probabilities` reorders columns to
the class vocabulary, fills classes the estimator never saw with zero, and
renormalises.

## `feature_importance.parquet`

Fold-level records **before** aggregation: `model_name`, `fold_index`,
`feature_name`, `kind` (`linear_coefficient` or
`permutation_importance`), `value`, `absolute_value`, `sign`,
`standard_deviation`, `repeat_count`, `scaling_context`, `class_label`,
and a `warning` set when the model did not beat chance in that fold.

Storing folds separately means a reader can see the spread rather than
only a mean, which is what tells them whether an "important" feature was
important consistently or once.

## Model files are executable content

`models/*.joblib` are Python pickles. **Loading a pickle executes code
contained in it. Never load a model file from an untrusted source.**

The warning is written to `models/README.txt` beside them.

Auditing a run never requires unpickling anything: the provenance, the
metrics, the splits, the calibration design, and the disclaimers are all
in the JSON documents. Fitted imputation medians and scaling parameters
are extracted into the run record for the same reason.

Only fold 0's estimators are persisted, per model, to keep run directories
small. The manifest records everything needed to refit any fold exactly.

## Privacy

A run directory contains pseudonymous subject and session identifiers,
feature column names, metrics, and configuration. It contains no name, no
email, no frame, no image, no landmark array, no secret, and no token.
Tests scan every JSON artifact for those and for scientific-validity
claims (`clinically validated`, `diagnostic accuracy`, `proven to
measure`, `production-ready`, `champion model`, `experimentally
validated`), and fail if any appears.

## Inspecting a run

```bash
# What was evaluated, and may it be cited?
python -c "import json;d=json.load(open('artifacts/experiments/run/manifest.json'));print(d['evaluation_mode'], d['scientific_evaluation_eligible'], d['dataset_fingerprint'])"

# Did any fold leak a group?
python -c "import json;print(json.load(open('artifacts/experiments/run/splits.json'))['audit_passed'])"

# Are the artifacts intact?
python -c "from pathlib import Path; from engagevr.training.artifacts import verify_checksums; print(verify_checksums(Path('artifacts/experiments/run')))"
```

None of these loads a model file.
