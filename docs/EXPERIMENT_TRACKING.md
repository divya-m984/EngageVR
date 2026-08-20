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

## Milestone 6: the fusion run directory

The format above is **extended, not replaced**. A fusion run writes every
document listed there and adds:

```
fusion_config.json          fusion configuration, resolved columns, scenarios
experts.json                per-fold modality-expert records and refusals
fusion_metrics.json         per-strategy folds, aggregates, diagnostics
robustness.json             every missing-modality scenario per strategy
expert_predictions.parquet  per-modality expert outputs (reference scenario)
fusion_weights.parquet      per-modality weights, raw and normalised
```

`metrics.json` is the same `MetricsDocument`, with one `ModelResult` per
fusion strategy (`model_kind: "fusion"`) and one per unimodal expert
(`model_kind: "unimodal_expert"`), so the metric machinery is reused rather
than duplicated.

`ExperimentRun` gained an optional `required_artifacts` parameter. A fusion
run declares a **larger** required set — `fusion_config.json`,
`experts.json`, `fusion_metrics.json`, and `robustness.json` in addition to
the Milestone 5 four — and a manifest claiming completion with any of them
missing is refused. The Milestone 5 default is unchanged.

`expert_predictions.parquet` is written for the reference scenario only. A
missing-modality scenario does not change what an expert computed; it
changes which experts were allowed to contribute, and that is recorded in
`fusion_weights.parquet` for every scenario.

### Fusion run identity

```
<target>-fusion-<selfcheck|sci>-<sha256(...)[:12]>
```

The hash covers the dataset fingerprint, the target, the task type, the
seed, the **split-manifest fingerprint**, the enabled strategies, the
modality groups, the minimum-modality count, the expert model types, the
calibration setting, the quality-weighting configuration, the stacking
configuration, the robustness configuration, the scenarios, and the
EngageVR version. No wall clock and no random component participates, and
the identifier is insensitive to the order in which strategies, modalities,
or scenarios were requested.

`split_manifest_fingerprint` is a SHA-256 over the canonical rendering of
the split manifest, which carries no wall clock. Pinning it in the run
identity is what makes "these strategies were compared on exactly the same
folds" checkable after the fact.

## Milestone 6: the personalization run directory

Also the format above, extended. A personalization run writes:

```
personalization_config.json  resolved configuration, every equation, run-id inputs
personalization.json         per-fold splits, corrections, both metric sets, coverage
personal_baselines.json      held-out subjects' per-feature baselines
```

plus `dataset.json`, `feature_catalog.json`, `splits.json`,
`calibration.json`, `metrics.json`, `predictions.parquet`,
`checksums.json`, and `manifest.json`. The seven documents through
`metrics.json` are its declared **required artifact set**, so a manifest
claiming completion without one of them is refused.

`metrics.json` is again the same `MetricsDocument`, with exactly two
`ModelResult` entries over identical evaluation windows:

| Pointer | `model_name` | `model_kind` |
|---|---|---|
| `/results/0` | `population` | `population` |
| `/results/1` | `personalized` | `personalized` |

`predictions.parquet` carries **both** predictions per window side by side
— `population_predicted_*` and `personalized_predicted_*`, with a
probability column per class for each — together with
`personalization_applied`, `cold_start`, `cold_start_reason`,
`baseline_normalized`, `supervised_correction_applied`,
`normalized_feature_count`, `calibration_sample_count`, and the
`calibration_window_ids` list. The population prediction is never
overwritten, so the difference between the two is computable from the
artifact without re-running anything.

`personal_baselines.json` records, per held-out subject per fold per
feature: the column, the catalogue feature name, its modality, its unit,
the calibration and finite sample counts, `mu_s`, `sigma_s`, the scale
source, whether it was normalised (and if not, why), and the exact source
window ids. Training subjects' baselines are not persisted — they have no
calibration/evaluation boundary to audit.

The audit trail for the leakage claim lives in `personalization.json`:
every split's calibration and evaluation window ids with both boundary
timestamps, and every correction's `calibration_targets` mapping window id
to the label used. A test asserts that mapping never names an evaluation
window.

### Personalization run identity

```
<target>-personalization-<selfcheck|sci>-<sha256(...)[:12]>
```

The hash covers the dataset fingerprint, the target, the task type, the
seed, the split-manifest fingerprint, the personalization method, the
modality groups, the calibration-window count, every minimum-evidence
constant, the zero-variance epsilon, the smoothing and shrinkage
constants, the population model types, the probability-calibration
setting, and the EngageVR version. No wall clock and no random component
participates.

## Milestone 7: the uncertainty run directory

Also the format above, extended. An uncertainty run writes:

```
uncertainty_config.json      resolved configuration, every equation, run-id inputs
uncertainty.json             per-fold group sets, thresholds, counts, coverage
thresholds.json              threshold provenance and the leakage rules
selective_metrics.json       accepted-set performance beside its coverage
coverage_curve.json          the deterministic axis sweep, with its x-axis named
selective_predictions.parquet   the decision layer: accepted, abstained, why
adaptation_gate.parquet      gate decisions; no column names an action
```

plus `dataset.json`, `feature_catalog.json`, `splits.json`,
`calibration.json`, `metrics.json`, `predictions.parquet`,
`checksums.json`, and `manifest.json`. The nine documents through
`metrics.json` are its declared **required artifact set**, so a manifest
claiming completion without one of them is refused.

`metrics.json` is again the same `MetricsDocument`, with exactly two
`ModelResult` entries over identical folds:

| `model_name` | `model_kind` | Scores |
|---|---|---|
| `all_windows` | `all_windows` | Every evaluated window |
| `accepted_at_applied_threshold` | `selective` | Only the accepted windows |

`predictions.parquet` is the **unselected** record: what the model said
before any threshold was applied. It carries the probability vector per
class, the calibration status, `confidence_score` *or* `selection_score`
(never both), `entropy`, `normalized_entropy`, `margin`, the interval
bounds and width, the conformal quantile, `minimum_recorded_quality`,
`available_modality_count`, and `ensemble_disagreement` — each in its own
column, so the five concepts stay distinguishable in the artifact.

`selective_predictions.parquet` is the **decision** layer, repeating the
original prediction beside `accepted`, `abstained`,
`primary_abstention_reason`, the full ordered `abstention_reasons` list,
`applied_threshold`, `threshold_source`, `maximum_interval_width`, and
`evidence_gate_passed`. Both tables carry the same
`source_prediction_id`, so the effect of the selective layer is computable
from the artifacts without re-running anything, and a reader never has to
trust that an abstained row still holds its prediction.

`thresholds.json` is the audit trail for the leakage claim: per fold, the
fit / probability-calibration / threshold-selection / conformal-calibration
/ outer-test group id lists, the estimated-threshold record with
`used_outer_test_labels: false`, the conformal quantile with its order
statistic and sample count, and every personal-threshold record with its
calibration window ids, both boundary timestamps, and
`uses_labels: false`. A test asserts no calibration group is also an
outer-test group.

`adaptation_gate.parquet` records the gate decision and its reasons. No
column in it names an action, a difficulty, a scene, a reward, or a policy;
a test asserts that.

`coverage_curve.json` names its x-axis explicitly — `x_axis`,
`x_axis_units`, and `monotonicity_rule` sit beside the curve — because the
two axes carry different units and move in opposite directions. A
classification curve is swept over `confidence_threshold` (a probability,
coverage non-increasing); a regression curve over
`maximum_interval_width` (the target's own units, coverage
non-decreasing). When no width grid is configured the document carries no
points and a `points_unavailable_reason` instead, rather than a curve
manufactured from the confidence grid. `thresholds.json` records both
grids, their units, and a statement that neither is derived from the
other.

### Uncertainty run identity

```
<target>-uncertainty-<selfcheck|sci>-<sha256(...)[:12]>
```

The hash covers the dataset fingerprint, the split-manifest fingerprint,
the target, the task type, the prediction source, the seed, the probability
calibration method, the confidence source and its definition, the
population threshold, the threshold grid, every threshold-estimation
setting, every personalized-threshold setting and the quantile method, the
interval method, alpha, the maximum interval width, the clipping setting,
the full evidence-gate configuration, the adaptation-gate setting, the
modality groups, the model names, and the EngageVR version. No wall clock
and no random component participates.

Still no MLflow, no DVC, and no Docker. Tests assert that none appears in a
fusion, a personalization, or an uncertainty run directory.

---

## Milestone 8: the adaptation run directory

An offline adaptation-policy run writes a directory of its own. It is not a
modelling run: it fits nothing, reads no dataset, splits no groups, and saves
no estimator, so it carries no `RunManifest`, no `splits.json`, and no
`models/`.

```
artifacts/experiments/<run-name>/
    adaptation_policy_config.json   the resolved configuration and its fingerprint
    scenarios.json                  what each scenario exercises
    adaptation_trace.parquet        ONE ROW PER POLICY EVALUATION
    adaptation_summary.json         controller metrics and provenance
    checksums.json                  SHA-256 of every artifact above
```

`ADAPTATION_REQUIRED_ARTIFACTS` names the three a completed run must contain;
a run missing one fails rather than reporting success.

### `adaptation_trace.parquet`

One row per policy evaluation — **holds included**, because a hold is a
decision and the reason it was taken is exactly what an auditor needs. Columns
cover the run, scenario, session, subject, and window references; the
Milestone 7 gate decision **and its reasons** for both targets; both ordinal
states and both predicted classes; each target's own suggested direction; the
conflict flag and its resolution; the resolved direction; the pending
direction, dwell count, and cooldown **before and after**; the current,
requested, and proposed difficulty with a clamping flag; the decision kind and
its policy reasons; the budget used and total; the proposal id; whether a
command object was built and the lifecycle status it reached; the experiment
mode; the configuration fingerprint; and the synthetic and
scientific-eligibility flags.

A reader of one row can answer, without re-running anything: was Milestone 7
eligible; what each target contributed; what each suggested; whether they
conflicted; whether the dwell requirement was satisfied; whether a cooldown
was active; whether the state was in bounds; whether budget remained; what was
decided; and, for a proposal, from what level to what level.

**The trace carries no wall-clock column** (DEC-081). Every value in it is a
function of the inputs, the configuration, and the initial state, so two runs
of one configuration produce byte-identical Parquet and the determinism check
is a checksum comparison. Timestamps live in the summary, where they are
provenance rather than data.

### `adaptation_summary.json`

The resolved configuration, its fingerprint, the scenario names, the session
ids, the controller metrics, and — when enabled — the guard-free controller
comparison. Every disclaimer travels with the document: the self-check banner,
the demonstration-rule note, the controller-metric note, the scenario note,
and the comparison note. `AdaptationRunSummary` refuses to validate a
self-check that omits the banner, a summary that omits the demonstration-rule
note, or a synthetic run claiming scientific eligibility.

### Adaptation run identity

```
adaptation-<selfcheck|sci>-<sha256(...)[:12]>
```

The hash covers the policy configuration fingerprint (every setting that can
change a decision, excluding the prose notes), the evaluation mode, the
identity of the input window sequence, and the EngageVR version. No wall clock
and no random component participates, so re-running an identical run
reproduces the identifier rather than accumulating near-duplicate directories,
and a static run and an adaptive run necessarily get different ids.

`proposal_id` and `command_id` are derived the same way, from the session,
window, order, direction, current and proposed level, and the configuration
fingerprint.

### Privacy

No raw frame, face crop, landmark, name, email, or secret appears in any
adaptation artifact. The subject, session, and window references are the
pseudonymous ones already present in the Milestone 5-7 artifacts. Tests assert
that no trace column name contains a media or biometric term and that no JSON
artifact contains an address or a credential.

Still no MLflow, no DVC, and no Docker.
