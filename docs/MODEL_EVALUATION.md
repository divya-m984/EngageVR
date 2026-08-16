# Model Evaluation Design (Milestone 5)

## Status

**Milestone 5 baseline-model pipeline implementation complete; scientific
evaluation on real participant-labelled data pending.**

Everything below describes the evaluation *machinery*. It has been
exercised on deterministic SYNTHETIC data only.

## Participant and session grouping

Several windows come from one session, and several sessions can come from
one person. An ungrouped split therefore puts the same person on both
sides of the boundary, and the resulting score measures memorisation of
that person rather than generalisation to a new one.

### Grouping priority

1. **`subject_id`** whenever two or more distinct subjects are present.
   This keeps every session of a person on one side of every boundary.
2. **`session_id`** when subject identifiers are absent or constant.
3. **Refusal** when neither field yields at least two distinct groups.

The chosen field and the reason are recorded in the split manifest.

### No random fallback

`KFold`, `StratifiedKFold`, and `train_test_split` are **not reachable**
from `engagevr.training.splits`. There is no configuration flag that
enables row-level splitting. When a defensible split is impossible the
splitter raises with an actionable message rather than weakening the
split:

```
cannot split this dataset: it has 1 distinct subject(s) and 1 distinct
session(s). At least two independent groups are required. Row-level
splitting is not offered: with several windows per session it would place
the same session in both portions and the resulting score would measure
memorisation rather than generalisation.
```

### Overlapping windows

Because grouping is by subject or by session, and a session belongs to
exactly one group, every window of a session lands in the same fold.
Adjacent overlapping windows that share evidence therefore cannot straddle
a boundary. `audit_split` re-checks this independently and fails if any
session has rows in the test portion of more than one fold.

## Cross-validation design

### Outer folds

| Task | Strategy | When |
|------|----------|------|
| Classification | `StratifiedGroupKFold` | every class appears in at least `n_splits` distinct groups |
| Classification | `GroupKFold` | otherwise, with the reason recorded |
| Regression | `GroupKFold` | always |

Stratification feasibility is checked explicitly rather than left to the
splitter's behaviour. When it fails, the manifest records exactly which
classes were too thin:

```
stratified grouped splitting is not feasible because 'high' appears in 2
group(s), which is fewer than the 3 requested folds; a non-stratified
grouped splitter is used instead
```

Regression uses grouped k-fold **without** stratification bins.
Deterministic bins are permitted by the design but are not used: choosing
bin edges is an undocumented modelling decision unless the edges
themselves are justified, and no such justification exists for an
unvalidated target.

`shuffle=True` with an explicit `random_state` from the run seed. Repeated
identical runs produce byte-identical split manifests.

### Fold count

`n_splits` must be at least 2, and there must be at least `n_splits`
independent groups. Otherwise:

```
9 independent subject_id group(s) cannot support 50 folds. Reduce --folds
to at most 9, or collect more groups. The split will not be weakened to
fit the requested fold count.
```

### Fold validity

A fold is marked **invalid**, and excluded from every aggregate, when:

- its test portion is empty, or
- its training portion contains no example of a class present in the
  dataset — a model fitted there could not predict that class.

A class absent from the *test* portion is a recorded **warning** rather
than an invalidation: the fold still fits and still scores, but per-class
metrics for the missing class are unavailable in it.

Every fold records its group membership, row counts, and target
distribution (class counts for classification; count, mean, SD, min, max
for regression).

### Inner validation

When `--tune` is passed, hyperparameter selection uses a group-aware inner
splitter (`StratifiedGroupKFold` or `GroupKFold`, up to 3 folds) run
entirely within the outer fold's **fit groups**. The outer test groups are
untouched until that fold's final evaluation.

### Split manifest

`splits.json` records the strategy and why it was chosen, the group field
and why, the group count, the fold count, the random seed, the calibration
group fraction, and per fold: train / calibration / test group lists, row
counts, target distributions, validity, invalid reason, and warnings.

`audit_split` runs on every manifest before it is returned and raises
`GroupOverlapError` on: a group in both train and test, a calibration
group in the test portion, a calibration group outside the training
groups, or a session split across test folds. The manifest records
`audit_passed` and the specific checks that ran.

## Calibration design

### Group separation

Within each outer fold:

- the **base estimator** is fitted on the *fit groups*;
- the **calibrator** is fitted on the *calibration groups*, which are
  carved out of the training groups and are disjoint from the fit groups;
- the **outer test groups** are used only for the final evaluation and are
  never used to fit anything.

`calibration_group_fraction` (default 0.25) is the fraction of a fold's
training groups reserved for calibration, selected deterministically from
the run seed and the fold index. At least one training group always
remains for fitting.

`assert_calibration_disjoint` re-checks all three sets before anything is
fitted, and raises with the reason:

```
calibration groups overlap the groups used to fit the base estimator:
['synthetic-subject-0003']. A calibrator fitted on the estimator's own
training rows corrects memorised predictions.
```

### API

scikit-learn deprecated `CalibratedClassifierCV(cv="prefit")` in 1.6 and
removed it in 1.8. The supported way to calibrate an already-fitted
estimator is to wrap it in `sklearn.frozen.FrozenEstimator`, which is what
this pipeline does. With a frozen estimator, `ensemble="auto"` resolves to
`False` and all supplied data is used to fit a single calibrator
(`len(calibrated_classifiers_) == 1`), which is asserted by a test.

### Methods

| Method | Availability |
|--------|--------------|
| `none` | Always. Skips calibration; the base estimator is still fitted. |
| `sigmoid` | Whenever the calibration set is non-empty and contains every class the base estimator saw. |
| `isotonic` | Only when the calibration set has ≥50 rows **and** ≥10 rows per class. |

The isotonic minimum exists because isotonic regression is non-parametric
and will happily interpolate a step function through a handful of points,
producing a calibration curve that describes the calibration sample and
nothing else. When the minimum is unmet, calibration is skipped and the
reason is recorded — never silently downgraded.

Calibration is likewise skipped, with a reason, when the calibration set
is empty or is missing a class. It is never fitted on the estimator's own
training rows as a fallback.

### What is recorded

`calibration.json` records the requested method, the group fraction, the
bin count, the design in words, and per fold and model: the method,
whether a calibrator was produced, fit and calibration row counts, group
counts, the explicit fit / calibration / test group lists, and the
unavailable reason when there is one.

Both calibrated and uncalibrated probabilities are written to
`predictions.parquet` in separate columns, so a reader can compare them
without re-running anything.

### Calibration is not certainty and not signal quality

A calibrated probability states how often outcomes of this kind occurred
at this predicted probability across the evaluated folds. It says nothing
about whether the camera signal was usable. Model probability and signal
quality are kept in separate fields throughout, and the calibration
document says so in a required `note` field.

Abstention, coverage-versus-performance analysis, and any online
confidence policy are **not implemented here**. They belong to
Milestone 7.

## Metrics

### Classification

Sample count, independent-group count, per-class support, accuracy,
balanced accuracy, macro precision / recall / F1, weighted F1, per-class
precision / recall / F1, a labelled confusion matrix, log loss, Brier
score, expected calibration error, and reliability bins — at fold level
and as aggregates.

### Regression

Sample count, independent-group count, MAE, RMSE, median absolute error,
and R² — at fold level and as aggregates. Target mean and standard
deviation are recorded alongside, so an error is readable against the
scale it was measured on.

### Documented formulas

**Macro precision / recall / F1** — the unweighted mean over the classes
for which the quantity is *defined*. A class with no predicted instances
has undefined precision; it is excluded from the mean rather than counted
as zero, and its exclusion is recorded in `unavailable_metrics`.

**Balanced accuracy** — the unweighted mean of recall over the classes
that are *present in the true labels*. It is computed from the same
per-class recall vector as macro recall rather than through
`balanced_accuracy_score`, which derives it from an unlabelled confusion
matrix and warns whenever the predictions name a class the truth does not
contain — a condition a heavy missing-modality scenario produces routinely.
The two definitions agree exactly; equivalence was checked over 200
randomly generated label/prediction pairs and is pinned by three
regression tests, including that exact heavy-dropout case. Deriving it here
keeps a misleading warning out of the run log without suppressing any
warning (DEC-065).

**Multiclass Brier score** —
`mean_i || p_i - onehot(y_i) ||²`, the mean squared Euclidean distance
between the predicted probability vector and the one-hot true vector.
Range `[0, 2]`. The alternative convention (half this value) is not used;
the definition is stated in the schema field wherever the number is
stored.

**Expected calibration error** —

```
ECE = Σ_m (|B_m| / n) · | accuracy(B_m) − confidence(B_m) |
```

over `M` equal-width bins of the maximum predicted class probability.
`accuracy(B_m)` is the fraction of samples in bin `m` whose `argmax`
prediction is correct; `confidence(B_m)` is the mean maximum predicted
probability in bin `m`. Empty bins contribute nothing. `M` defaults to 10
and is recorded on every calibration document. Empty bins are **retained**
with `count = 0` and null summaries, so the binning is reconstructible
from the stored document.

**Log loss** — scikit-learn binarises labels in sorted order, so the
probability columns are reordered to match sorted labels before scoring.
Passing the ordered class vocabulary directly would silently pair the
wrong column with each class. A test asserts a confidently-correct
prediction scores near zero, which fails loudly if the pairing breaks.

**Fold aggregation** — the unweighted mean and population standard
deviation over the folds in which the metric was defined, plus the count
of such folds and the total fold count. Folds are weighted **equally**:
a size-weighted mean silently lets one large participant dominate, which
is the opposite of what grouped cross-validation is for.

### Undefined metrics

A metric whose prerequisites are unmet is `None` with a stated reason in
`unavailable_metrics`. It is never replaced with zero, because zero is a
legitimate score and a reader could not tell the two apart.

| Metric | Undefined when |
|--------|----------------|
| every metric | the evaluation set is empty |
| per-class precision | the class is never predicted |
| macro precision / recall / F1 | undefined for every class |
| R² | the true values have zero variance, or fewer than two samples |
| log loss, Brier, ECE | no probabilities, or a non-finite probability |

A non-finite *prediction* is an error, not a metric: a model that cannot
produce a finite prediction must fail rather than emit one.

### Confusion matrices

Stored as labelled data, never as a bare array: the label vocabulary, an
explicit statement that rows are true labels and columns are predicted
labels, and the counts. `as_labelled_cells()` renders them as
`{"true=low|predicted=medium": 3, …}`. Fold matrices are summed
element-wise into an aggregate only when they share a label ordering.

### Namespace separation

Synthetic and scientific results are kept in separate namespaces by the
`evaluation_mode` field, which is required on every metrics document. A
`software_self_check` document:

- can never set `scientific_evaluation_eligible: true` (schema-enforced);
- must carry the banner `SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION`
  in its disclaimers (schema-enforced).

There is no field anywhere in the schema that could hold a published or
public-dataset score, so a synthetic number cannot be recorded as one.

## Ablation design

Nine deterministic feature subsets, evaluated on the **same outer folds**
with fold-local preprocessing:

`task_only`, `behavioural_only`, `head_pose_only`, `rppg_only`,
`quality_only`, `all_available`, `all_except_task`, `all_except_rppg`,
`all_except_behavioural`.

One model is used across every subset (logistic regression for
classification, ridge for regression) so the only thing that differs
between two ablations is which features the model could see.

`quality_only` is included as a **control**: if capture-quality
diagnostics alone score well, the result is being driven by measurement
conditions rather than by anything about a person.

Each result records its included modalities and its exact feature names.
When a feature group is absent from a dataset, the ablation is marked
unavailable with a reason rather than silently reduced to whatever
happened to survive.

`all_available` means "no feature group was removed". It is **not** a
multimodal-fusion architecture: no modality mask, no quality-aware
weighting, and no early-versus-late fusion is implemented in this
milestone. The ablation document carries that statement as a required
field. Fusion is Milestone 6.

**No scientific conclusion may be drawn from an ablation over synthetic
data.** Which feature group helps a model recover a latent variable that
this repository itself inserted is a fact about the generator.

## Scientific-mode safeguards

`baseline-train --mode scientific` refuses to run when:

1. **any row is synthetic** —
   "scientific evaluation refused: N of M rows carry
   `data_source='synthetic'`. Synthetic data verifies software; it is
   never evidence about a person.";
2. **any target sets `scientific_evaluation_permitted: false`**;
3. **any target's source type is unstated** — the error names the four
   permitted real-label categories.

Grouping is required in every mode: `choose_group_field` refuses a dataset
that cannot be split defensibly, so a scientific run cannot proceed
without participant or session grouping.

Passing these checks establishes **eligibility, not validity**. The run
manifest says so explicitly. No claim of experimental validation follows
from a run completing.

## Reproducibility

Metrics are exactly reproducible from the same data, configuration, and
random seed:

- the dataset fingerprint pins the data;
- the run manifest pins the configuration, the seed, every dependency
  version, and the Python version;
- the split manifest pins every fold's group membership;
- every estimator has an explicit `random_state`;
- no wall clock participates in any fingerprint or identifier.

Tests assert that two runs of one configuration produce identical
`metrics.json` and identical `splits.json`.

## Acceptance criteria

| Criterion (`docs/PROJECT_PLAN.md`, Milestone 5) | Status |
|---|---|
| 1. No data leakage between participant sessions | **Met for the implementation.** Grouped splitting with no random fallback, an independent split audit, fold-local preprocessing, calibration on disjoint groups, and structural refusal of target, identifier, and post-window columns. Asserted by tests. Unverifiable against real participant sessions because none exist. |
| 2. Metrics are reproducible (seeded) | **Met.** Deterministic dataset fingerprints, deterministic split manifests, explicit seeds throughout; asserted by repeat-run tests. |
| 3. Data origin is documented | **Met.** Dataset metadata, feature catalog snapshot, target provenance per row, and a run manifest recording every dependency version — all inspectable as JSON without loading a model file. |
| 4. Synthetic data is excluded from scientific evaluation | **Met.** Schema-enforced on targets, enforced again by the scientific-mode gate, asserted by tests. |
| 5. No synthetic number presented as real evidence | **Met.** Required disclaimers, schema-enforced banner, no field capable of holding a published score, and CLI output tests. |

**Pending:** every one of these is a property of the software. None has
been exercised against real participant-labelled data, because none
exists.

## Reuse by the Milestone 6 fusion runner

The fusion runner reuses this design unchanged. It calls the same
`build_splits` with the same parameters, so a baseline run and a fusion run
over the same dataset, target, fold count, and seed produce a
byte-identical split manifest — asserted by a test — and it fingerprints
that manifest with `split_manifest_fingerprint` so "these strategies were
compared on exactly the same folds" is checkable after the fact.

The classification and regression metrics, the undefined-stays-null rule,
the documented macro / Brier / ECE / log-loss conventions, and the
equal-weight fold aggregation are the same functions. Fusion adds
diagnostics beside them — coverage, available-expert count,
missing-modality rate, modality contribution counts, mean normalised
weights, and expert disagreement — never in place of them.

Two evaluation properties are specific to fusion:

- **Metrics describe covered windows.** A strategy scores the windows it
  produced a prediction for; `fusion.coverage` states what fraction of the
  evaluated windows that was. Both numbers are always stored together,
  because a score over a subset is not comparable to a score over the whole
  fold unless the reader can see both.
- **Calibration is placed once.** Per expert, before fusion, and on the
  early-fusion estimator; never after fusion. Fused probabilities are
  *evaluated* for calibration but nothing is fitted to them. One shared
  addition was needed — `MINIMUM_CALIBRATION_SAMPLES_PER_CLASS = 5`,
  because `CalibratedClassifierCV` cross-validates the calibration set even
  over a `FrozenEstimator` — and an unmet minimum records an unavailable
  calibrator with a reason rather than failing the fold.

See `docs/MULTIMODAL_FUSION.md`.

## Reuse by the Milestone 6 personalization runner

The personalization runner reuses the same `build_splits` call, the same
fold-local preprocessing, the same probability calibration on disjoint
groups, and the same classification and regression metrics. It adds one
evaluation rule of its own.

**The two reports must cover the same rows.** Within each outer fold, every
held-out subject is cut in wall-clock time into a calibration region and a
strictly later evaluation region, and *both* the population-only model and
the personalized model are scored on that evaluation region — never on
different subsets. `PersonalizationFoldResult` refuses to validate when the
two metric documents disagree on sample count, so a comparison over
different data cannot be persisted. `metrics.json` carries the two results
as separate `ModelResult` entries (`model_kind: "population"` and
`"personalized"`), which is what Milestone 6 acceptance criterion 3 asks
for.

Beside them, `personalization.json` records the calibration window count,
the evaluation window count, the number of windows excluded for straddling
the boundary, the cold-start count, the unavailable count, and

```
personalization_coverage = personalized_subject_count
                         / (personalized_subject_count + cold_start_subject_count)
```

Coverage matters for the same reason `fusion.coverage` does: a personalized
score computed over the subjects that *could* be personalised is not
comparable to one over all of them unless the reader can see how many fell
back.

**No aggregate declares a winner.** The wording used in every document and
every printed line is: *population and personalized software-check results
are reported separately; synthetic differences are not evidence of a
personalization benefit.*

Two distinct things are called calibration and are kept in separate files:
probability calibration of the population model (`calibration.json`) and
per-subject adaptation (`personalization.json`). Neither is a confidence
estimate and neither withholds a prediction; abstention and selective
prediction are Milestone 7.
