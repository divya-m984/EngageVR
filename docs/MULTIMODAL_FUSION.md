# Multimodal Fusion (Milestone 6)

## Status

**Milestone 6 multimodal-fusion implementation complete; scientific
evaluation on real participant-labelled multimodal data pending.**

Every number this pipeline has produced came from SYNTHETIC data. Those
numbers are software self-checks. They are not model accuracy, not
engagement validity, not cognitive-load validity, and not evidence about
any person. No fusion strategy in this repository is a champion, none is
validated, and none is production-ready. The same is true of every
personalized result below: **population and personalized software-check
results are reported separately, and synthetic differences between them are
not evidence of a personalization benefit.**

## Scope

In scope: early feature fusion, late decision-level fusion, quality-aware
late fusion, validation-derived late fusion, leakage-safe stacked fusion,
modality-specific experts, missing-modality robustness scenarios,
deterministic synthetic modality dropout, expert-disagreement diagnostics,
personalized calibration with separate population reporting, and fusion
experiment records.

**Explicitly out of scope for this milestone** (each belongs to a later
one): temporal neural networks, online inference, confidence-based
abstention, selective prediction, personalized confidence thresholds,
adaptation policy, dashboard pages, MLflow, DVC, Docker, and deployment. No
deep or neural fusion is implemented, and no per-subject model is trained
from scratch.

## Why this is not the Milestone 5 ablation

`docs/BASELINE_MODELS.md` describes nine feature-subset ablations —
`task_only`, `all_available`, `all_except_rppg`, and so on. Those are
**not** fusion architectures and are not renamed here. An ablation fits one
model on whatever columns survived a subset rule and records the subset. It
carries no notion of a modality producing or failing to produce evidence.

Milestone 6 adds, for every fused window:

| Tracked here | Present in the M5 ablation |
|---|---|
| modality membership of every column | only implicitly, via the subset name |
| per-window modality **availability** | no |
| per-window modality **quality** | no |
| missing-modality **patterns** | no |
| an explicit **fusion configuration** | no |
| per-modality **estimators** and their outputs | no |
| per-modality **weights**, raw and normalised | no |
| fusion **coverage** and unavailable-fusion counts | no |
| expert **disagreement** diagnostics | no |
| deterministic missing-modality **scenarios** | no |

`all_available` means "no feature group was removed". Early fusion means
"these named modality groups were combined into one matrix, and here is
what each contributed, when it was available, and how good its signal was".
The second is a fusion architecture; the first is a column filter.

## Modality groups

Four measurement modalities carry experts:

| Modality | Catalogue group | What it is |
|---|---|---|
| `behavioural` | `behavioural` | facial behavioural proxies (EAR, blink proxy, closure, tracking stability) |
| `head_pose` | `head_pose` | head-orientation geometry and angular motion |
| `rppg` | `rppg` | camera-based pulse estimates and their spectral diagnostics |
| `task` | `task` | task telemetry (counts, proportions, reaction times, pauses) |

### Quality is not a modality

`FusionModality` has exactly four members. There is no `quality` member, so
capture-quality diagnostics cannot become a fifth measurement modality by
accident. `parse_modality("quality")` raises with an explanation, and
`fusion.modalities: [quality, ...]` is rejected at configuration load.

Three kinds of signal are support/context rather than measurement:

- **quality features** (`capture_brightness_mean`, `capture_blur_score_mean`,
  `window_missing_feature_pct`, …) and `modality_quality__*`;
- **availability features** (`avail__*`, `modality_available__*`);
- **missingness indicators** produced by fold-local imputation.

They may participate in the explicitly named quality-aware strategy, and
availability is always carried in the early-fusion matrix because that is
how a missing modality is represented without fabricating a zero. They
never become another modality, and they never carry an expert.

What is never a predictor at all: identifiers, timestamps, targets, target
provenance, split metadata, session-completion fields, future events, and
the generator's latent variables. Those are refused structurally by
`assert_no_leakage` before fusion sees a column.

## Early feature fusion

The permitted features of two or more modality groups are concatenated into
one predictor matrix, and one estimator is fitted on it inside each training
fold.

- Columns are selected in **catalogue order**, not modality order, so
  reordering a configuration does not silently reorder the matrix.
- Per-feature availability flags (`avail__*`) and per-modality availability
  flags (`modality_available__*`) are always carried. Measured values and
  missingness stay in separate columns.
- `modality_quality__*` is **excluded by default**
  (`include_modality_quality_in_early_fusion: false`), so quality does not
  silently become a measurement.
- Preprocessing is fold-local: the imputer and the scaler live inside the
  scikit-learn `Pipeline`, so every statistic they learn is fitted on the
  fold's fit rows and never on the test rows.
- The estimator family is the same one the modality experts use, so early
  and late fusion differ in **architecture** and not in model choice.
- Every selected column, every modality, and the missing-modality pattern
  are recorded in `fusion_config.json`.

**Training-time missing values.** A window in which a modality contributed
nothing carries `null` for that modality's measured features. The fold-local
`SimpleImputer(strategy="median", add_indicator=True)` substitutes the
training fold's median **and** appends a missingness indicator, so the
estimator can distinguish "measured, and it was the median" from "not
measured". The absence is marked in three places: the value is imputed but
flagged, `avail__*` is 0, and `modality_available__*` is 0. Nothing is
zero-filled.

Concatenating feature groups is concatenation. It is not attention, and no
term from the neural literature is used for it anywhere in this milestone.

## Modality-specific experts

One estimator per modality, fitted on that modality's columns only.

- An expert sees the measured features of its modality, their availability
  flags, and its own `modality_available__*` column. It sees
  `modality_quality__*` only when `include_modality_quality_in_experts` is
  set, which is **off by default**.
- An expert is fitted on the fit-group rows **in which its own modality
  contributed evidence**. Rows where the modality is absent carry no
  measurement from it; including them would teach the expert the fold's
  imputation median and then let it speak confidently about windows it
  never observed.
- Preprocessing is fold-local, exactly as in Milestone 5.
- Experts are fitted on the **same outer fold assignment** as every other
  strategy — the split manifest is built once and fingerprinted.
- Class ordering is the run's ordered vocabulary; `aligned_probabilities`
  reorders every expert's output onto it and renormalises.
- Calibrated probabilities are produced when
  `use_calibrated_experts: true`, on the fold's calibration groups.

### When an expert refuses

| Condition | Result |
|---|---|
| the modality contributes no permitted predictor column | unavailable, with the reason |
| fewer than 10 fit rows carry the modality's evidence | unavailable, naming the count and the minimum |
| fewer than 2 independent groups carry it | unavailable, naming the count |
| a classification fold has fewer than 2 classes among those rows | unavailable, naming the classes |

A refusal is recorded in `experts.json` with its reason. It never produces a
fabricated prediction, a class prior dressed up as one, or a uniform
probability vector.

### When an expert declines a window

A window in which the expert's modality contributed nothing receives
`available=False` and a reason. It receives no prediction, no probability
vector, and no weight.

## Late / decision-level fusion

### The combination

**Classification** — a weighted average of the available experts'
probability vectors, aligned to one class vocabulary first, then
renormalised so the fused vector sums to one:

```
p_fused = normalise( sum_over_contributors w_m * p_m )
```

**Regression** — a weighted average of the available experts' numeric
estimates, with the weights normalised over the contributors:

```
y_fused = ( sum_over_contributors w_m * y_m ) / ( sum_over_contributors w_m )
```

An expert that produced nothing is simply absent from the sum. It is never
replaced with zero, never replaced with a uniform probability vector, and
never replaced with the training mean.

### Weights

```
raw_effective_weight_m = base_weight_m * availability_m * normalised_quality_m
normalized_weight_m    = raw_effective_weight_m / sum over contributors
```

`availability_m` is 1 when modality *m* produced a prediction for this
window and 0 otherwise, so an unavailable modality has zero effective weight
**by construction** rather than by a downstream filter.

| Strategy | `base_weight_m` | `normalised_quality_m` |
|---|---|---|
| `uniform_late` | 1.0 for every modality | fixed at 1.0 (`quality_source=not_used`) |
| `quality_late` | configured base weight, default 1.0 | the recorded modality quality, or the documented fallback |
| `validation_weighted_late` | the inner-validation weight (below) | fixed at 1.0 |

Refusals in the weight algebra: a negative weight, a non-finite weight, a
non-finite or out-of-range quality value, and an empty contributor set are
each an error with a stated reason, never a silent fallback.

### Minimum-modality policy

`minimum_modalities` (default 1) is applied to **every** strategy, so
coverage is comparable across them. A window is fused only when

1. at least `minimum_modalities` experts produced a prediction, **and**
2. at least `minimum_modalities` of them carried a non-zero weight.

The two are separate because quality-aware weighting can exclude an expert
that did produce a prediction, and the reason a window went unfused should
say which happened. A window that fails either check is recorded with
`fused: false` and the reason. It is never given a default prediction.

For early fusion the same rule is applied to modality **availability**: a
window in which fewer than `minimum_modalities` groups contributed evidence
is not fused, even though the single estimator could technically emit a
number from an all-imputed row.

## Quality-aware fusion

### The equation

The one above, with `normalised_quality_m` taken from the dataset's
`modality_quality__<m>` column, clipped to `[0, 1]`.

### Missing quality

Some modalities have no quality channel at all — task telemetry is a
software measurement with no signal-quality index — so "no quality
recorded" is a normal condition, not an anomaly.

| Policy | Behaviour |
|---|---|
| `documented_fallback` (default) | substitute `missing_quality_fallback` (default **0.5**, the midpoint of the range) and record `quality_source=documented_fallback` on the weight |
| `exclude` | drop the modality from quality-aware weighting and record `quality_source=unavailable` with the reason |

**Neither policy treats missing quality as perfect quality, and there is no
policy that does.** 0.5 is a neutral midpoint, not a tuned value, and every
weight that used it says so.

### Thresholds

| Setting | Default | Why |
|---|---|---|
| `minimum_quality` | `0.0` | No empirically validated quality cut-off exists for these signals, so no non-zero default could be justified. At 0.0 the quality gate excludes nothing; a user who sets a threshold gets an exclusion with a stated reason. |
| `minimum_effective_weight` | `1e-9` | A numerical guard against normalising by ~0, not a modelling threshold. It is what excludes a modality whose quality is exactly zero. |
| `base_weights` | `{}` | Empty means a deterministic equal base weight of 1.0 — the control. **No optimised weight set is shipped as a default**, and none was chosen after looking at a result. |

### What quality is not

Signal quality describes the **measurement**, never the person. A low
quality value means the camera or task signal was poor. It is never low
engagement and never high cognitive load, and nothing in this milestone
converts one into the other.

Quality is also not model confidence. A calibrated probability states how
often outcomes of this kind occurred at this predicted probability; a
quality value states whether the signal was usable. They live in separate
fields on every record — `quality_used` on a weight, `probabilities` on a
prediction — and are never merged.

## Validation-derived weighting

`validation_weighted_late` estimates one weight per modality per outer fold,
from **inner** grouped folds drawn entirely from that fold's outer-training
portion.

| Task | Rule |
|---|---|
| classification | `w_m = max(0, (balanced_accuracy_m − 1/K) / (1 − 1/K))` over the out-of-fold rows in which *m* predicted |
| regression | `w_m = max(0, 1 − MAE_m / MAE_of_predicting_the_out-of-fold_mean)` |

Both are bounded in `[0, 1]`, deterministic, and scale-free. A reciprocal-
error rule (`1 / MAE`) is deliberately **not** used: it diverges when an
expert happens to score perfectly on a small validation set, and clamping it
would need a threshold with no justification.

When every weight is zero — no modality beat the reference level — the run
falls back to deterministic equal weights and records
`fallback_applied: true` with the reason. A modality that scored at or below
the reference keeps the smallest positive base weight and is then excluded
per window by its own availability, because a base weight of exactly zero is
refused by the weight algebra.

Each fold records the metric name, the metric definition, the exact groups
used, the raw scores, and the resulting weights, in
`fusion_metrics.json`. The outer test groups appear in none of them, and a
test asserts it.

## Stacked fusion

Implemented, and **disabled by default** (`fusion.stacking.enabled: false`).

For each outer fold:

1. split the outer-**training** groups into grouped inner folds;
2. fit modality experts on the inner-training groups only;
3. predict **only** the held-out inner groups;
4. assemble the out-of-fold expert-prediction matrix;
5. fit the meta-model on that matrix and nothing else;
6. refit the experts on the whole outer-training portion;
7. predict the untouched outer-test groups with those experts;
8. apply the already-fitted meta-model.

`assert_out_of_fold` re-checks the property independently before the
meta-model is fitted, and raises `StackingLeakageError` on the first of
three violations: a row predicted by experts fitted on its own group
(in-sample meta-training), a row predicted by experts fitted on an
outer-test group, or a meta-training row from outside the outer-training
groups. Tests assert that each is detected.

**Meta-model inputs.** One column per (modality × class) probability for
classification, or one prediction column per modality for regression, plus
one availability column per modality. An unavailable expert contributes a
**missing value**, never a zero and never a uniform vector; the meta-model's
own fold-local median imputer fills it and appends a missingness indicator,
so "the expert said nothing" is never mistaken for "the expert said zero".
Rows carrying fewer than `minimum_modalities` expert predictions are dropped
from meta-training rather than fitted on entirely imputed inputs.

**Meta-models.** `LogisticRegression` for classification, `Ridge` for
regression. A configuration naming anything else is rejected. There is no
neural stacker.

**Calibration.** The stacker consumes **uncalibrated** expert probabilities
at both meta-training and meta-inference time. Fitting a meta-model on
uncalibrated inputs and then applying it to calibrated ones would change the
input distribution between fitting and use, which no metric would reveal.

`stacking_provenance` records, per fold: the meta-model, the inner-fold
count, the out-of-fold row count, the meta-training row and group counts,
and the three leakage checks that passed.

## Calibration placement

Calibration happens **per expert, before fusion**, and to the early-fusion
estimator. There is **no post-fusion calibrator**.

Calibrating twice would make the reported probability the output of two
corrections with no way to attribute either, and there is no documented
reason in this project to prefer that. The fused probabilities are still
*evaluated* for calibration — Brier score, log loss, ECE, and reliability
bins are computed on them — but nothing is fitted to them.

The Milestone 5 architecture is preserved unchanged: the base estimator is
fitted on the fold's fit groups, the calibrator on the fold's calibration
groups (disjoint from the fit groups and from the outer test groups), and
`assert_calibration_disjoint` re-checks all three sets before anything is
fitted.

One shared-code addition was needed: `CalibratedClassifierCV` resolves
`cv=None` to a 5-fold stratified splitter and cross-validates the
calibration set even over a `FrozenEstimator`, so a class with fewer than
five calibration rows cannot be split. `MINIMUM_CALIBRATION_SAMPLES_PER_CLASS`
now checks that first and records an unavailable calibrator with a reason
instead of failing the fold. It never falls back to calibrating on the
estimator's own training rows.

`calibration.json` records the placement in words, the design, the per-expert
outcome, and a note that a calibrated probability is neither certainty nor
signal quality.

**Abstention is not implemented here.** Coverage-versus-performance analysis
and any online confidence policy belong to Milestone 7.

## Expert disagreement

Interpretable diagnostics over the available experts, computed only for
windows with **at least two** available experts. Windows with fewer are
counted in `insufficient_expert_window_count` and contribute to no summary.

**Classification**

- `mean_distinct_predicted_classes` — how many different labels the experts
  chose;
- `unanimous_fraction` / `disagreement_fraction`;
- `mean_pairwise_probability_distance` — mean Euclidean distance between
  every pair of aligned probability vectors;
- `mean_fused_probability_entropy` — Shannon entropy, in nats, of the fused
  vector.

**Regression**

- `mean_prediction_standard_deviation` — population SD of the experts'
  numeric predictions;
- `mean_prediction_range` — max minus min.

### Disagreement is not uncertainty

This is an **ensemble-disagreement diagnostic**. It describes how far the
modality estimators differed from one another on a window. It is not a
calibrated uncertainty estimate, it is not signal quality, it is not model
confidence, and **it does not trigger abstention**. Formal uncertainty-aware
inference and abstention are Milestone 7, and nothing in this milestone
gates a prediction on disagreement. The distinction is carried as a required
`note` field on every stored summary.

## Missing-modality robustness

Ten deterministic scenarios, evaluated on the same outer folds as everything
else:

`all_modalities`, `missing_behavioural`, `missing_head_pose`,
`missing_rppg`, `missing_task`, `missing_behavioural_and_rppg`, `only_task`,
`only_behavioural`, `only_rppg`, `only_head_pose`.

A scenario removes **availability** and nothing else. It never rewrites a
measurement, never zero-fills a feature, and never touches a target. It can
only remove: a scenario cannot make a modality that contributed nothing
appear to have contributed something.

- **Late fusion** naturally renormalises over the surviving experts.
- **Quality-aware fusion** additionally excludes unusable modalities and
  records why.
- **Early fusion** meets the same input a real missing modality produces:
  the modality's measured columns become missing (not zero), its
  availability flags become 0, and its quality becomes missing. The
  fold-local imputer then does what it does in the field.

Scenarios are applied at **evaluation time only**. The models are trained
once on the recorded availability and then met with each pattern, which is
the question a robustness scenario asks: this system, as trained, meets a
window with no rPPG — what does it do? Training ten separate models would
answer a different question.

A scenario that leaves no configured modality present is **not evaluated**,
and says so.

Every scenario records: evaluated windows, fused windows, unavailable-fusion
count, coverage, valid fold count, aggregated metrics, per-modality
contribution counts, mean weights, and the pooled diagnostics.

**On synthetic data this is not a real-world robustness result.** It
describes how this code behaves when told a modality is absent. It says
nothing about how the system would behave when a real camera signal failed.

## Synthetic modality dropout

An optional deterministic software check, off by default.

- Seeded explicitly (`synthetic_dropout_seed`).
- Applied **after** dataset generation, to availability only. The hidden
  target construction is untouched and target values are unchanged.
- Whole modality groups are dropped coherently for a window — never feature
  by feature.
- The decision for one (window, modality) pair is a pure function of the
  seed, the window id, and the modality name, so it is reproducible
  regardless of row order or fold assignment.
- Availability is updated consistently, and the masked pattern is what both
  training and evaluation see: dropout describes a dataset in which those
  modalities were never captured.
- The configuration is recorded in `robustness.json`
  (`synthetic_dropout_applied`, seed, probability).

**Scientific mode refuses it**, because it fabricates an availability
pattern that no measurement produced. A scientific evaluation must use the
availability pattern actually recorded for each window.

## Evaluation

The same grouped outer folds as Milestone 5, built by the same
`build_splits` call with the same parameters and fingerprinted with
`split_manifest_fingerprint`. A test asserts that a baseline run and a
fusion run over the same dataset, target, fold count, and seed produce a
byte-identical split manifest.

Compared on those folds: **early fusion**, **uniform late fusion**,
**quality-aware late fusion**, and — as a descriptive control — the **best
available unimodal expert per fold**.

### Metrics

Classification and regression metrics are the Milestone 5 ones, unchanged:
accuracy, balanced accuracy, macro precision / recall / F1, weighted F1, a
labelled confusion matrix, log loss, Brier score, ECE and reliability bins;
MAE, RMSE, median absolute error, R². The conventions, the
undefined-stays-null rule, and the equal-weight fold aggregation are
documented in `docs/MODEL_EVALUATION.md`.

Metrics describe the windows a strategy produced a prediction for. Coverage
states what fraction of the evaluated windows that was, and both numbers are
always recorded together: a score over a subset is not comparable to a score
over the whole fold unless the reader can see both.

### Fusion diagnostics

`available-expert count`, `fusion coverage`, `missing-modality rate` per
modality, `modality contribution counts`, `mean normalised weight` per
modality, expert disagreement, and the `unavailable-fusion count`.

Early and stacked fusion carry no per-window weights — early fits one
estimator over concatenated features, and a stacker's coefficients are a
property of the meta-model rather than of a window — so their
`mean_normalized_weight` is **unavailable** rather than invented, and
contribution is read from availability.

Fold-level results are stored before any aggregate, and the valid fold count
is recorded beside every aggregate.

### The unimodal control

Reported per fold as the strongest single-modality expert on that fold's own
metric. This is **descriptive only**: the modality is selected using the
same outer fold it is scored on, so the value is optimistically biased and
is not a fair comparator. It is never used to choose a fusion strategy, and
the caveat is a required `note` field on the record.

### No strategy is selected

The outer folds are not used to pick a winner. Nothing in the artifacts, the
schemas, or the CLI names a best strategy, and tests assert it. A comparison
computed on synthetic data cannot select a fusion architecture for any
purpose that matters: it would describe the generator this repository wrote,
not any person.

## Experiment artifacts

The Milestone 5 format is **extended, not replaced**:

```
artifacts/experiments/<run-name>/
    manifest.json               written atomically, last
    dataset.json                dataset provenance and fingerprint
    feature_catalog.json        the catalog the run was audited against
    splits.json                 fold assignments and the split audit
    fusion_config.json          fusion configuration, columns, scenarios
    experts.json                per-fold expert records and refusals
    metrics.json                fold and aggregate metrics per strategy
    fusion_metrics.json         fusion diagnostics, weights, disagreement
    calibration.json            calibration placement and per-expert outcome
    robustness.json             every scenario for every strategy
    predictions.parquet         fused predictions, all strategies/scenarios
    expert_predictions.parquet  per-modality expert outputs
    fusion_weights.parquet      per-modality weights, raw and normalised
    feature_importance.parquet  fold-level linear coefficients
    models/
        README.txt              untrusted-pickle warning
        expert-<modality>-fold0.joblib
        early-fusion-fold0.joblib
    checksums.json              SHA-256 of every artifact above
```

`artifacts/` is gitignored. **Datasets, predictions, and model files are
never committed.**

A completed fusion run must contain `dataset.json`, `feature_catalog.json`,
`splits.json`, `fusion_config.json`, `experts.json`, `metrics.json`,
`fusion_metrics.json`, and `robustness.json`. A run claiming completion with
any of them missing is refused by `ExperimentRun.finalize`. A run that
raises mid-way writes a `failed` manifest and re-raises; a run that raises
before folds exist writes no manifest, which `read_manifest` reports as an
interrupted run rather than a success.

`expert_predictions.parquet` is written for the reference scenario only. A
missing-modality scenario does not change what an expert computed; it
changes which experts were allowed to contribute, and that is recorded in
`fusion_weights.parquet` for every scenario.

`models/*.joblib` are Python pickles. **Loading a pickle executes code
contained in it. Never load a model file from an untrusted source.**
Auditing a fusion run never requires unpickling anything.

## Run identity

```
<target>-fusion-<selfcheck|sci>-<sha256(...)[:12]>
```

The hash covers the dataset fingerprint, the target, the task type, the
random seed, the split-manifest fingerprint, the enabled strategies, the
modality groups, the minimum-modality count, the expert model types, the
calibration setting, the full quality-weighting configuration, the stacking
configuration, the robustness configuration, the missing-modality scenarios,
and the EngageVR version.

No wall clock and no random component participates. Equivalent
configurations reproduce the same logical run id; the identifier is
insensitive to the *order* in which strategies, modalities, or scenarios
were requested, because that does not change what was run.

## Configuration

See the `fusion` section of `configs/defaults.yaml`. It validates modality
names (rejecting `quality` with an explanation), strategy names, the minimum
modality count against the configured groups, quality ranges and
probabilities, base-weight positivity, duplicate strategies and modalities,
empty strategy sets, and stacking configurations that cannot satisfy the
grouped out-of-fold requirement.

## Commands

```bash
# Software verification on a synthetic dataset
uv run python -m engagevr fusion-demo \
  --dataset artifacts/datasets/m6-synthetic.parquet \
  --target engagement_class \
  --folds 5 --seed 42 \
  --strategies early uniform-late quality-late \
  --output artifacts/experiments/m6-engagement-fusion

# Generic command; scientific mode refuses synthetic data
uv run python -m engagevr fusion-train \
  --dataset /path/to/windowed-features.parquet \
  --target engagement_class \
  --mode scientific \
  --strategies early uniform-late quality-late \
  --output artifacts/experiments/run-name
```

Useful flags: `--modalities`, `--minimum-modalities`, `--calibration`,
`--no-calibrated-experts`, `--scenarios`, `--no-robustness`,
`--synthetic-dropout`, `--dropout-seed`.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid dataset, leakage, invalid folds, unsupported model, incomplete artifacts, failed evaluation |
| 2 | Bad arguments: missing dataset, fold count below 2, unsupported target, invalid fusion configuration |
| 3 | Scientific mode refused the dataset or the requested synthetic dropout |

## Personalization

Milestone 6 acceptance criterion 3 asks for personalized and population
baselines to be **reported separately**. That is what this layer does. It
adapts the population model to one subject using only that subject's own
earlier windows, and it reports both results over exactly the same
evaluation windows.

`src/engagevr/training/personalization.py` holds the pure algebra;
`personalization_runner.py` orchestrates the folds;
`src/engagevr/schemas/personalization.py` holds the typed records.

### Where personalization sits relative to fusion

The documented path is:

```
early-fusion population prediction
    -> subject calibration / correction
    -> personalized prediction
```

The population reference model is the **early-fusion estimator over the
configured modality groups** — the same `early_fusion_columns` the fusion
runner uses, unchanged. Personalization layers on top of its output. It is
not a parallel model stack, no fusion weight is retuned, and the population
prediction is retained on every record rather than overwritten.

Early fusion rather than a late-fusion strategy is used as the population
reference because it yields exactly one population prediction per window
with no weighting step to disturb: a per-subject correction applied on top
of a re-weighted combination would confound two adjustments.

### The five supported modes

| Mode | Method value | What it does |
|---|---|---|
| Population-only baseline | `population_only` | The control. The personalized output reproduces the population output exactly and `personalization_applied` is false everywhere. |
| Personal-baseline feature calibration | `personal_baseline` | Features are z-scored against the subject's own calibration windows. Unsupervised: no label is used. |
| Few-shot personalized calibration | `few_shot_correction` | The population prediction is corrected from the subject's few labelled calibration windows. |
| Population model + user-specific correction | `personal_baseline_and_correction` | Both of the above. The shipped default. |
| Cold start | `cold_start` (outcome only) | No valid personal evidence: the population model is used, `personalization_applied=false`, `cold_start=true`, and the reason is stated. Requested with `--calibration-windows 0`. |

`cold_start` is deliberately **not requestable**. It is what a run records
when a requested personalization could not be applied to a subject; offering
it as a method to ask for would make an outcome indistinguishable from an
intention. The configuration validator rejects it by name and says so.

No mode trains a subject-specific model from scratch. A handful of
calibration windows cannot support one, and a per-subject estimator fitted
on five windows describes those five windows.

### The calibration/evaluation split

For each outer fold, every held-out subject is cut in **wall-clock time**,
before anything is fitted:

1. The subject's windows are ordered by
   `(window_start_utc, window_end_utc, window_index, window_id)`.
2. The first `calibration_windows` of them form the calibration region.
3. The **boundary** is the latest `window_end_utc` in that region.
4. A later window joins the evaluation region only if its
   `window_start_utc` is at or after the boundary.
5. A window that straddles the boundary is **excluded from both regions**
   and listed in `excluded_overlap_window_ids`.

Step 5 is why the split is temporal rather than positional. With
overlapping windows, the window immediately after the calibration region
still shares evidence with it; moving it into the evaluation region would
put the personal baseline partly in the future of what it is evaluated on.
Windows are never mixed at random between the two regions.

The exact calibration and evaluation window ids, both boundary timestamps,
and `temporal_order_verified` are recorded per subject per fold in
`personalization.json`. A `PersonalCalibrationSplit` whose calibration
region does not end before its evaluation region begins fails schema
validation, so an unordered split cannot be persisted.

A subject who cannot supply both regions is recorded as **unavailable**
with a reason and is excluded from both reports. The protocol is never
weakened to keep them in.

### Personal-baseline normalization

For a numeric feature `x` and subject `s`:

```
z_s(x) = (x - mu_s) / sigma_s
```

`mu_s` and `sigma_s` are the mean and population standard deviation over
**that subject's calibration windows only**. Never over the whole subject
before splitting, never over an evaluation window, never over anyone else.

- A missing measurement stays missing. `mu_s` and `sigma_s` are computed
  over the finite values only.
- When `sigma_s <= zero_variance_epsilon` the value is **centred but not
  scaled** (`scale = 1.0`), recorded as
  `scale_source="unit_scale_zero_variance"`. It is never divided by ~0.
- When fewer than `minimum_baseline_samples` finite calibration values
  exist, the feature is passed through unchanged (`mu_s=0`, `sigma_s=1`),
  recorded as `normalized=false` with the reason. The schema rejects a
  non-normalised record that is not the identity transform, so "not
  normalised" cannot silently rescale anything.

`personal_baselines.json` records, per subject per fold per feature: the
column, the catalogue feature name, its modality, its **unit**, the
calibration sample count, the finite sample count, `mu_s`, `sigma_s`, the
scale source, and the exact source window ids.

**Training subjects are normalised under the same rule**, from their own
earliest `calibration_windows` windows within the training portion. Using
all of their windows would be leakage-free and still wrong: it would
estimate the training scale from more evidence than any held-out subject
ever gets, and the estimator would then meet a differently-scaled matrix at
evaluation time. Only held-out subjects' baselines are persisted — they are
the ones with a boundary worth auditing.

#### What is never personalised

Identifiers, timestamps, targets, target provenance, split metadata,
per-feature availability flags, modality-availability flags,
modality-quality columns, and categorical provenance fields. Only
catalogued measured `feat__` columns of the configured **measurement**
modalities are.

Quality is excluded for the same structural reason it is not a fusion
modality: signal quality describes the measurement, not the person.
Normalising it against a personal baseline would present it as a personal
physiological value. `assert_personalizable` raises on the first forbidden
column and names why.

### Few-shot correction

**Regression** — the exact documented equation:

```
b_s = mean over subject s's labelled calibration windows of
      (y_calibration - y_population_prediction)

y_personalized = y_population_prediction + b_s
```

**Classification** — a regularised per-subject log-odds shift. With `K`
classes, `n` labelled calibration windows, smoothing `alpha`, and shrinkage
constant `kappa`:

```
observed_c = (count_c + alpha) / (n + alpha*K)
expected_c = (sum_w p_population_c(w) + alpha) / (n + alpha*K)
lambda     = n / (n + kappa)
delta_c    = lambda * (log(observed_c) - log(expected_c))

p_personalized_c = p_population_c * exp(delta_c)
                   / sum_k p_population_k * exp(delta_k)
```

Both terms are smoothed identically, so `delta_c` is **exactly zero** when
the subject's calibration labels match what the population model predicted
on average. `lambda` grows with calibration evidence and is near zero when
there is almost none. The renormalisation guarantees a finite, non-negative
vector summing to one; a row that could not be renormalised raises rather
than being emitted.

Neither correction ever sees an evaluation label. `calibration_targets`
records the label used for each calibration window **by window id**, which
is the audit trail: a test asserts that no evaluation window id appears
there.

### Minimum evidence, and what happens below it

| Requirement | Default | Below it |
|---|---|---|
| labelled calibration windows | `minimum_calibration_windows: 3` | cold start, reason recorded |
| distinct calibration classes | `minimum_calibration_classes: 2` | cold start, reason recorded |
| finite values per feature | `minimum_baseline_samples: 3` | that feature is not normalised, reason recorded |
| windows after the boundary | `minimum_evaluation_windows: 1` | the subject is unavailable, reason recorded |

Every one of these is an **engineering default, not a validated
threshold**. None was tuned, and none could be: no real personalization
evidence exists in this project.

Below any of them the run falls back to the population model and says so.
It never borrows another subject's baseline, never substitutes a global
statistic and calls it personal, and never fabricates a personal baseline.
A cold-start prediction must reproduce the population prediction exactly —
the schema rejects one that does not.

### Population versus personalized reporting

Both results are computed over **exactly the same evaluation windows**.
`PersonalizationFoldResult` refuses to validate if the two metric documents
cover different row counts, so a comparison across different data cannot be
persisted.

`metrics.json` carries two `ModelResult` entries:

- `/results/0` — `model_name: "population"`, `model_kind: "population"`
- `/results/1` — `model_name: "personalized"`, `model_kind: "personalized"`

Classification reuses the Milestone 5 classification metrics; regression
reuses the regression metrics. `personalization.json` additionally records
the calibration window count, the evaluation window count, the excluded
boundary-window count, the cold-start count, the unavailable count, and

```
personalization_coverage = personalized_subject_count
                         / (personalized_subject_count + cold_start_subject_count)
```

over the subject-fold pairs that were evaluated at all.

**No document, message, or field declares personalization better.** The
wording used throughout is: *population and personalized software-check
results are reported separately; synthetic differences are not evidence of
a personalization benefit.*

### Personalized calibration is not uncertainty calibration

Two different things are called calibration in this repository, and they
are kept in separate files:

| | What it is | Where |
|---|---|---|
| Probability calibration | Correcting the **population** model's probabilities on disjoint calibration groups | `calibration.json` |
| Personalized calibration | Adapting the population model to **one subject** from that subject's earlier windows | `personalization.json` |

Neither is a confidence estimate and neither withholds a prediction.
Personalized confidence thresholds, selective prediction, and abstention
are **Milestone 7**, and `personalization.json` says so in a required
field. `docs/PROJECT_SPECIFICATION.md` Stage E lists "personalized
thresholds"; the only thresholds implemented here are the benign
minimum-evidence gates in the table above, which decide whether a
*correction is fitted* — never whether a *prediction is issued*.

### Personalization leakage risks, and the guard for each

| Risk | Guard |
|---|---|
| the population model sees the held-out subject | grouped outer folds; a test asserts no evaluated subject appears in any training group |
| a personal baseline estimated from the future | the temporal split; the schema refuses an unordered one; a test asserts every recorded source window is a calibration window |
| an evaluation label reaching the correction | `calibration_targets` records what was used, by window id; a test asserts it never intersects the evaluation windows |
| a window calibrating on itself | schema validator on `PersonalizedPrediction` |
| overlapping windows straddling the boundary | excluded from both regions and listed, never moved |
| a training/evaluation normalisation mismatch | training subjects are normalised under the identical chronological rule |
| the two reports covering different rows | `PersonalizationFoldResult` refuses mismatched sample counts |
| quality becoming a personalised physiological value | `assert_personalizable` refuses the column outright |

### Personalization commands

```bash
uv run python -m engagevr personalization-demo \
  --dataset artifacts/datasets/m6-synthetic.parquet \
  --target engagement_class \
  --folds 5 --seed 42 \
  --calibration-windows 5 \
  --output artifacts/experiments/m6-personalization-class

# Cold-start mode
uv run python -m engagevr personalization-demo \
  --calibration-windows 0 --method personal-baseline --output ...

# Generic command; scientific mode refuses synthetic data
uv run python -m engagevr personalization-train \
  --dataset /path/to/windowed-features.parquet \
  --mode scientific --output artifacts/experiments/run-name
```

Useful flags: `--method`, `--modalities`, `--calibration-windows`,
`--minimum-calibration-windows`, `--minimum-evaluation-windows`,
`--calibration`. Exit codes match the fusion commands.

### Personalization artifacts

The Milestone 5 directory layout again, with:

| File | Contents |
|---|---|
| `personalization_config.json` | the resolved configuration, every equation, and the run-identity inputs |
| `personalization.json` | per-fold splits, corrections, both metric sets, coverage counts |
| `personal_baselines.json` | held-out subjects' per-feature baselines with units and source windows |
| `metrics.json` | the population and personalized results as two `ModelResult` entries |
| `predictions.parquet` | both predictions per window, side by side, with the calibration window ids |

Run identity is deterministic over the dataset fingerprint, split-manifest
fingerprint, target, seed, method, modality groups, every minimum-evidence
and smoothing constant, the population model types, and the EngageVR
version. No wall clock participates.

## Leakage safeguards

| Risk | Guard |
|---|---|
| a group on both sides of a fold | `build_splits` + `audit_split`, reused unchanged |
| preprocessing statistics crossing a fold | structural: transformers live inside the `Pipeline` |
| a calibrator fitted on the estimator's own rows | `assert_calibration_disjoint` before anything is fitted |
| fusion weights tuned on the outer test fold | validation weights come only from inner groups; the groups used are recorded and asserted |
| a stacker trained on in-sample expert predictions | `assert_out_of_fold`, three checks, tested for each |
| a target or identifier reaching a predictor matrix | `assert_no_leakage` on every column set |
| evidence from after the predicted window | `POST_WINDOW_TOKENS` name check |
| a strategy comparison over different folds | one split manifest, fingerprinted and asserted equal |

## Acceptance criteria (`docs/PROJECT_PLAN.md`, Milestone 6)

| Criterion | Status |
|---|---|
| 1. System remains functional with missing signals | **Met for the implementation.** Ten deterministic scenarios, coverage and unavailable-fusion counts recorded per scenario and strategy, late fusion renormalising over survivors, early fusion meeting the real missing-modality input shape, and a refusal — never a fabricated prediction — when the minimum-modality rule is unmet. Never exercised on a real signal failure, because none exists. |
| 2. Quality-aware fusion is compared with naive fusion | **Met for the implementation.** `quality_late` and `uniform_late` are evaluated on identical folds with identical experts, and both are recorded with their weights, coverage, and diagnostics. The comparison is a software self-check on synthetic data; it establishes no superiority and none is claimed. |
| 3. Personalized and population baselines are separately reported | **Met for the software implementation.** For each held-out subject, a population-only result and a personalized result are produced over identical evaluation windows and written as two separate `ModelResult` entries in `metrics.json`, with per-fold splits, corrections, coverage, and cold-start counts in `personalization.json`. Per-participant baselines, z-scoring, few-shot correction, and an explicit cold-start path are implemented. **No personalization benefit is claimed**, and none could be: the comparison is a software self-check on synthetic data. |

**Pending for all of them:** every property above is a property of the
software. None has been exercised against real participant-labelled
multimodal data, because none exists.

## Limitations

- No fusion model here has been fitted to a real participant label.
- Every reported score is a self-check on data this repository generated.
- Which fusion architecture recovers a latent variable that this repository
  itself inserted is a fact about the generator, not about fusion.
- Missing-modality robustness has been measured only against deterministic
  scenarios and seeded synthetic dropout, never against a real signal
  failure.
- Modality quality has never been validated against anything external; the
  rPPG quality index in particular is an interpretable engineering
  construction (see `docs/LIMITATIONS.md`).
- The minimum-evidence gates on experts (10 rows, 2 groups) are engineering
  defaults, not validated thresholds. So are every personalization gate:
  3 calibration windows, 2 calibration classes, 3 finite values per
  feature, `kappa = 5.0`, `alpha = 1.0`.
- Expert disagreement is not uncertainty, and nothing here abstains.
- **No personalization result here is evidence of a personalization
  benefit.** On this repository's generator the personalized variants score
  *worse* than the population baseline on several targets; that is a fact
  about a generator whose targets track absolute feature levels, which
  within-subject z-scoring removes. It is neither a defect nor a finding
  about people, and it is reported rather than tuned away.
- Personalization has never been exercised on a real subject, a real
  session boundary, or a real label. `RQ2` in
  `docs/RESEARCH_QUESTIONS.md` — whether personalized baselines outperform
  population models — remains **unanswered**.
- No medical, diagnostic, psychological, clinical, or
  adaptive-effectiveness claim follows from anything in this document.
