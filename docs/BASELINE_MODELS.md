# Interpretable Baseline Models (Milestone 5)

## Status

**Milestone 5 baseline-model pipeline implementation complete; scientific
evaluation on real participant-labelled data pending.**

Every number this pipeline has produced so far came from SYNTHETIC data.
Those numbers are software self-checks. They are not model accuracy, not
engagement validity, not cognitive-load validity, and not evidence about
any person.

## Scope

In scope: windowed feature datasets, interpretable classical models,
grouped cross-validation, offline probability calibration, metrics,
feature interpretation, feature-group ablations, and local experiment
records.

**Explicitly out of scope for this milestone** (each belongs to a later
one): multimodal-fusion architectures, temporal neural networks,
personalisation, online inference, confidence-based abstention, adaptation
policy, dashboard pages, MLflow, DVC, Docker, and deployment.

## Model registries

Two registries, one per task type, because a classifier and a regressor
are not interchangeable and a single registry invites picking the wrong
one.

### Classification

| Name | Family | Preprocessing | Notes |
|------|--------|---------------|-------|
| `dummy` | dummy | median impute + indicator | Predicts the training class prior |
| `logistic_regression` | linear | median impute + indicator, standardised | `max_iter=2000`, `class_weight=None` |
| `random_forest` | tree | median impute + indicator | 200 trees, fixed `random_state` |
| `hist_gradient_boosting` | tree | **no imputation** (native NaN) | scikit-learn histogram boosting |
| `rule_software_check` | rule | none | Quantile thresholds on one feature |

### Regression

| Name | Family | Preprocessing | Notes |
|------|--------|---------------|-------|
| `dummy` | dummy | median impute + indicator | Predicts the training mean |
| `ridge` | linear | median impute + indicator, standardised | `alpha=1.0` |
| `random_forest` | tree | median impute + indicator | 200 trees, fixed `random_state` |
| `hist_gradient_boosting` | tree | **no imputation** (native NaN) | scikit-learn histogram boosting |
| `rule_software_check` | rule | none | Linear rescaling of one feature |

### Why Ridge rather than ElasticNet

Ridge has one hyperparameter and produces no sparsity pattern to
over-interpret. On a feature set with genuine collinearity — three
response proportions that sum to one, order statistics that are ordered by
construction — an L1 penalty picks one member of a correlated group
essentially arbitrarily, and a reader is then tempted to conclude the
others do not matter. Ridge spreads the coefficient instead, which is
honest about the ambiguity.

### Why XGBoost is not a dependency

`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`
implement the same histogram-based boosted-tree algorithm, handle missing
values natively, and are already installed as part of scikit-learn.
Adding a second gradient-boosting library would add a dependency, a build
requirement, and a second serialisation format without adding a
capability. If a later milestone finds a concrete need XGBoost meets and
scikit-learn does not, that is the point to add it — with the need stated.

### No neural or temporal models

None is registered, and a test asserts none appears. `DEC-005` defers deep
learning until interpretable baselines have been evaluated, and they have
not been evaluated on anything real yet.

## Hyperparameters

Grids are deliberately tiny and are documented in the registry:

| Model | Grid |
|-------|------|
| `logistic_regression` | `C ∈ {0.1, 1.0, 10.0}` |
| `random_forest` | `max_depth ∈ {None, 6}` |
| `hist_gradient_boosting` | `learning_rate ∈ {0.05, 0.1}` |
| `ridge` | `alpha ∈ {0.1, 1.0, 10.0}` |

Tuning is **off by default**. With `--tune`, a group-aware inner search
runs entirely within each outer fold's fit groups (3 inner folds, or fewer
when there are too few inner groups; falls back to fixed parameters when
fewer than two). The outer test groups are never seen by the search.

Every model has a fixed `random_state` derived from the run seed, so
repeated identical runs produce identical predictions.

### Class weighting

`class_weight` is left at `None`. Re-weighting changes what the fitted
probabilities mean, and there is no documented reason in this project to
prefer balanced errors over calibrated frequencies. No resampling is
performed anywhere. If resampling is added later it must happen **inside
training folds only**; oversampling before a split copies rows across the
boundary and the resulting score is meaningless.

## The rule-based software-check baselines

`RuleBasedThresholdClassifier` fits two quantiles of one feature on the
training rows and uses them as class boundaries.
`RuleBasedThresholdRegressor` rescales one feature linearly onto the
training target range. Nothing else is learned.

Which feature each one thresholds is an **arbitrary implementation
choice**, recorded per target:

| Target | Rule feature |
|--------|--------------|
| `engagement_class`, `engagement_score` | `feat__task_correct_proportion` |
| `cognitive_load_class`, `cognitive_load_score` | `feat__task_reaction_time_mean_ms` |

Task accuracy is not engagement and reaction time is not cognitive load.
These estimators exist so the harness can carry a non-learned estimator,
giving a "good" learned score something trivial to be compared against.
They are **not validated indicators**, their probabilities are **not
calibrated**, and they must never be described as scientifically
validated. Every result they produce is flagged
`is_software_check_baseline: true` and carries
`RULE_BASELINE_DISCLAIMER`.

If the preferred feature is absent — inside an ablation that removed it —
the estimator falls back to the first available measured column and
records which one it actually used in `resolved_feature_`. The
substitution is never silent.

## Preprocessing

Preprocessing lives **inside** the scikit-learn `Pipeline`, so every
statistic it learns is fitted on whatever rows the pipeline is fitted on,
which is always a training fold and never a test fold.

```
ColumnTransformer
├── "measured"  → [SimpleImputer(median, add_indicator) →] [StandardScaler]
└── "flags"     → passthrough  (avail__*, modality_available__*)
```

Availability and modality-availability columns are binary indicators that
are never missing; imputing or scaling them would be meaningless.

Pipelines use pandas output (`set_output(transform="pandas")`) so column
names survive every transformation and can be reported alongside
coefficients and importances.

### Imputation and signal quality

Median imputation puts a plausible number where a measurement failed.
That is unavoidable for estimators which cannot accept `NaN`, but it must
not erase the difference between "measured" and "guessed", so:

- every imputed numeric column is accompanied by a missingness indicator,
  which is a first-class model input;
- the `avail__` and `modality_quality__` columns remain in the matrix, so
  the failure itself is visible to the model;
- histogram gradient boosting receives values **unimputed** and routes
  missing values at each split.

A quality failure is therefore never silently converted into a
physiological value: the absence is marked in three separate places.

### Column admission

A column reaches the predictor matrix only if it is in the feature
catalog **and** the catalog permits it as a predictor. Availability and
modality-quality columns are admitted deliberately: whether a measurement
succeeded, and how good it was, are legitimate inputs and are kept
structurally distinct from the measurement itself.

Target, identifier, timestamp, provenance, schema, split, and post-outcome
columns are refused, each with a distinct error message naming the leakage
mode it would cause.

Fitted imputation medians and scaling parameters are extracted with
`imputation_parameters` and recorded in the run artifact, so a reader can
see exactly what was substituted without executing a model file.

## Feature interpretation

### Linear models

Coefficients, their sign, their absolute magnitude, the transformed
feature name, the class label (for multinomial models), and the scaling
context. Because linear models are fitted on standardised features, a
coefficient is the change in log-odds (or in the target) per one
**training-fold** standard deviation.

### Tree models

Permutation importance computed on **held-out fold data**, with the mean
and standard deviation across repeats and the repeat count. Impurity-based
importance is deliberately not used: it is biased toward high-cardinality
features and is computed on training data.

Permutation is applied to the estimator's post-preprocessing input matrix
rather than to the raw frame. The transformer is already fitted and
deterministic, so this changes nothing about the result while avoiding
re-running the `ColumnTransformer` for every (feature × repeat)
permutation. Missingness-indicator columns are permuted in their own
right and appear as named features in the output.

### Storage and caveats

Fold-level records are written to `feature_importance.parquet` **before**
any aggregation, so a reader can see the spread across folds rather than
only a mean.

- **Association is not causation.** A feature a model leans on is not a
  measurement of the construct being modelled.
- **Correlated features share credit arbitrarily.** With two
  near-duplicate inputs, a linear model may split a coefficient between
  them, and a permutation test may score both as unimportant because
  either alone substitutes for the other.
- **A chance-level model's importances describe noise.** When accuracy
  does not exceed the majority-class rate, or R² is at or below zero, a
  warning is attached to every interpretation record for that fold and
  surfaced in the CLI output. The data is still recorded — suppressing it
  would hide the fact that a model failed — but it is labelled.
- SHAP is not used in this milestone.

## No champion selection

This pipeline does not rank models, does not mark one "best", and does not
label anything production-ready. A synthetic self-check cannot rank models
for any purpose that matters, and there is no real evaluation to rank them
on. Tests assert that no registry entry, no artifact, and no CLI output
claims otherwise.

## Commands

```bash
# Software verification on a synthetic dataset
uv run python -m engagevr baseline-demo \
  --dataset artifacts/datasets/m5-synthetic.parquet \
  --target engagement_class \
  --folds 5 --seed 42 \
  --output artifacts/experiments/m5-engagement-demo

# Generic training command
uv run python -m engagevr baseline-train \
  --dataset /path/to/windowed-features.parquet \
  --target engagement_class \
  --mode scientific \
  --output artifacts/experiments/run-name
```

Useful flags: `--models a,b,c`, `--calibration none|sigmoid|isotonic`,
`--calibration-group-fraction`, `--calibration-bins`,
`--permutation-repeats`, `--no-ablations`, `--tune`.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid dataset, leakage, invalid folds, unsupported model, incomplete artifacts, failed evaluation |
| 2 | Bad arguments: missing dataset, fold count below 2, unsupported target |
| 3 | Scientific mode refused the dataset |

## Limitations

- No model here has been fitted to a real participant label.
- Every reported score is a self-check on data this repository generated.
- The feature set has never been validated against anything external.
- Hyperparameter grids are small by design and have not been searched
  broadly; nothing here is a claim about achievable performance.
- The rule baselines are software checks, not indicators.
- No medical, diagnostic, psychological, or adaptive-effectiveness claim
  follows from anything in this document.
