# EngageVR Model Cards

> **Research documentation**
>
> These model cards document the predictive-model families and associated
> inference layers currently implemented by EngageVR.
>
> They describe software behaviour, intended research use, provenance,
> limitations, and scientific status.
>
> No model or fusion strategy in this repository is currently designated as a
> scientifically validated champion.
>
> Current internal demonstrations use synthetic software-check data and do not
> establish performance on real participants.

## 1. Purpose

This document provides research-facing model cards for:

1. baseline classification models;
2. baseline regression models;
3. multimodal fusion models;
4. personalization;
5. classification calibration, confidence, and selective prediction;
6. regression conformal uncertainty and selective prediction.

Detailed implementation documentation remains in:

- `docs/BASELINE_MODELS.md`;
- `docs/MODEL_EVALUATION.md`;
- `docs/MULTIMODAL_FUSION.md`;
- `docs/UNCERTAINTY_AND_ABSTENTION.md`.

These cards summarize those systems without replacing their technical
contracts.

## 2. Global Scientific Status

The current model stack is implemented and tested as research software.

However:

- no real participant-labelled EngageVR dataset has been used to fit or
  evaluate the engagement/cognitive-load models;
- no baseline model is a champion;
- no fusion strategy is a champion;
- no personalization benefit has been demonstrated;
- no selective-prediction threshold has been scientifically validated for
  participant use;
- no conformal coverage guarantee has been established for real EngageVR
  participant data;
- synthetic results remain software self-checks.

Software correctness and reproducibility must not be interpreted as scientific
validation.

## 3. Shared Model-Evaluation Contract

Where applicable, EngageVR model evaluation uses:

- participant/session-aware grouping;
- no random row-level fallback when grouping is required;
- fold-local preprocessing;
- disjoint fit, calibration, and test groups;
- explicit split manifests;
- provenance-aware datasets;
- scientific-mode guards;
- reproducible run configuration.

Predictor matrices exclude inappropriate columns such as:

- targets;
- identifiers;
- timestamps;
- target provenance;
- split metadata;
- post-outcome fields.

No model card overrides those constraints.

## 4. Shared Missing-Data Rule

Missing measurement and missing prediction states must remain explicit.

Depending on the model:

- compatible estimators may use fold-local median imputation plus a missingness
  indicator;
- histogram gradient boosting may consume native `NaN`;
- unavailable modality predictions remain unavailable;
- late fusion never replaces an unavailable expert with zero;
- stacked fusion represents unavailable expert outputs as missing values before
  its own fold-local preprocessing.

Missingness must not be transformed into evidence the sensor or model did not
produce.

---

# Card 1 — Baseline Classification Models

## 5. Card Summary

**Task type:** classification

**Current role:** interpretable baseline comparison and software/model
evaluation framework.

**Scientific status:** not scientifically validated on real
participant-labelled EngageVR data.

## 6. Implemented Classification Registry

| Name | Family | Preprocessing | Key implementation note |
|---|---|---|---|
| `dummy` | Dummy | Median impute + indicator | Predicts training class prior |
| `logistic_regression` | Linear | Median impute + indicator, standardized | `max_iter=2000`, `class_weight=None` |
| `random_forest` | Tree ensemble | Median impute + indicator | 200 trees, fixed random state |
| `hist_gradient_boosting` | Tree boosting | Native missing-value handling | No imputation |
| `rule_software_check` | Rule | None | Quantile thresholds on one feature |

## 7. Intended Use

Appropriate uses include:

- establishing simple comparison baselines;
- testing grouped model evaluation;
- evaluating classification targets;
- comparing interpretable linear and tree-based model families;
- testing probability calibration;
- providing modality-specific experts for fusion where configured;
- validating reproducible experiment plumbing.

## 8. Hyperparameter Search

The implemented search grids are deliberately small.

| Model | Grid |
|---|---|
| `logistic_regression` | `C ∈ {0.1, 1.0, 10.0}` |
| `random_forest` | `max_depth ∈ {None, 6}` |
| `hist_gradient_boosting` | `learning_rate ∈ {0.05, 0.1}` |

Tuning is disabled by default.

When enabled, hyperparameter selection occurs within grouped inner folds drawn
only from the outer-training portion.

Outer-test observations must not participate in model selection.

## 9. Class Weighting

Classification models currently use:

```text
class_weight = None
```

No class-resampling procedure is used.

This preserves the current probability semantics rather than changing the class
distribution without a documented scientific reason.

## 10. Probability Calibration

Where configured and supported, classification probabilities may undergo
offline calibration using disjoint calibration groups.

Supported calibration modes include:

- none;
- sigmoid;
- isotonic where minimum evidence requirements are satisfied.

Probability calibration must not use the same groups used to fit the base
estimator.

A calibrated probability is not:

- certainty;
- signal quality;
- psychological confidence;
- proof that the prediction is correct.

## 11. Outputs

Potential outputs include:

- predicted class;
- class-probability vector;
- calibrated probability vector where available;
- calibration status;
- classification metrics;
- confusion matrix;
- calibration metrics;
- feature interpretation where supported.

## 12. Candidate Metrics

Current classification evaluation may include:

- balanced accuracy;
- macro F1;
- log loss;
- Brier score;
- expected calibration error;
- reliability bins.

Metric availability depends on the target and evaluation state.

## 13. Interpretability

Linear classification models permit coefficient inspection.

Tree-based models may use permutation-based interpretation where implemented.

Feature importance must not be interpreted causally.

A coefficient or permutation score does not establish that a feature causes
engagement or cognitive load.

## 14. Rule-Based Classification Baseline

`rule_software_check` exists primarily to verify that the evaluation harness can
carry a non-learned estimator.

It must not be described as a scientifically validated classifier.

Its purpose is software verification, not psychological inference.

## 15. Known Limitations

Current limitations include:

- no real participant-labelled EngageVR model evaluation;
- no champion classifier;
- no demonstrated generalization to a participant population;
- no validated engagement/cognitive-load construct validity;
- calibration depends on sufficient grouped calibration evidence;
- window-level observations may remain correlated within participants.

## 16. Unsupported Claims

This model family must not currently support claims such as:

- "EngageVR accurately detects engagement in humans";
- "EngageVR accurately measures cognitive load";
- "logistic regression is the best EngageVR model";
- "random forest is the selected production model";
- "a calibrated probability is certainty";
- "software-test accuracy demonstrates participant validity."

---

# Card 2 — Baseline Regression Models

## 17. Card Summary

**Task type:** regression

**Current role:** interpretable regression baselines and reusable population
estimators.

**Scientific status:** not scientifically validated on real
participant-labelled EngageVR data.

## 18. Implemented Regression Registry

| Name | Family | Preprocessing | Key implementation note |
|---|---|---|---|
| `dummy` | Dummy | Median impute + indicator | Predicts training mean |
| `ridge` | Linear | Median impute + indicator, standardized | `alpha=1.0` default |
| `random_forest` | Tree ensemble | Median impute + indicator | 200 trees, fixed random state |
| `hist_gradient_boosting` | Tree boosting | Native missing-value handling | No imputation |
| `rule_software_check` | Rule | None | Linear rescaling of one feature |

## 19. Why Ridge Is the Linear Baseline

Ridge is used instead of a sparse linear model because the feature set may
contain genuine correlated structures.

A sparse coefficient pattern could tempt readers to interpret one selected
member of a correlated group as uniquely important.

Ridge instead retains a simpler regularized linear baseline without implying
that correlated omitted coefficients are scientifically irrelevant.

## 20. Hyperparameter Search

| Model | Grid |
|---|---|
| `ridge` | `alpha ∈ {0.1, 1.0, 10.0}` |
| `random_forest` | `max_depth ∈ {None, 6}` |
| `hist_gradient_boosting` | `learning_rate ∈ {0.05, 0.1}` |

As with classification, tuning is:

- disabled by default;
- group-aware when enabled;
- confined to outer-training data.

## 21. Intended Use

Appropriate uses include:

- regression baseline comparison;
- continuous-target evaluation;
- software verification;
- early-fusion population estimation;
- modality-specific regression experts;
- personalization population reference;
- conformal-regression input.

## 22. Candidate Metrics

Current regression evaluation may include:

- MAE;
- RMSE;
- median absolute error;
- R².

No metric should be interpreted without:

- data provenance;
- participant/session grouping;
- target provenance;
- missingness;
- sample size.

## 23. Rule-Based Regression Baseline

`rule_software_check` rescales one selected feature onto the training target
range.

It is a software-check estimator.

It is not a scientifically validated regression model.

## 24. Known Limitations

Current limitations include:

- no real participant-labelled EngageVR regression evaluation;
- no champion regressor;
- no established cross-participant validity;
- no validated relationship between predicted values and psychological truth;
- standard point predictions do not represent uncertainty.

## 25. Unsupported Claims

These regression models must not currently be described as:

- validated measurements of engagement;
- validated measurements of cognitive load;
- clinical measurements;
- psychologically diagnostic models;
- production-ready participant estimators.

---

# Card 3 — Multimodal Fusion Models

## 26. Card Summary

**Purpose:** combine evidence from multiple modality-specific feature groups or
experts.

**Supported architecture families:**

- early feature fusion;
- uniform late fusion;
- quality-aware late fusion;
- validation-weighted late fusion;
- stacked fusion.

**Scientific status:** implementation validated using software checks; no
fusion strategy is scientifically selected as a champion.

## 27. Early Feature Fusion

Early fusion concatenates the available approved feature columns into one
predictor matrix and fits one estimator.

The estimator family is the same family used by the modality-specific experts
for the comparison.

Therefore, early-versus-late comparison changes fusion architecture rather than
silently changing the underlying estimator family.

## 28. Modality-Specific Experts

Late-fusion methods use one estimator per modality.

Each expert is fitted only on that modality's admitted predictor columns.

An expert may be unavailable for a window.

Unavailable experts must remain unavailable rather than being replaced by an
invented prediction.

## 29. Uniform Late Fusion

For classification:

```text
p_fused =
normalize(sum_over_contributors w_m * p_m)
```

For regression:

```text
y_fused =
sum_over_contributors(w_m * y_m)
/
sum_over_contributors(w_m)
```

For `uniform_late`, each available modality has equal base weight.

Weights are normalized across the experts that actually contributed.

## 30. Quality-Aware Late Fusion

Conceptually:

```text
raw_effective_weight_m =
base_weight_m * availability_m * normalized_quality_m
```

followed by normalization over contributing experts.

Signal quality describes the measurement.

It must not be interpreted as:

- engagement;
- cognitive load;
- model confidence.

### Missing quality

The default missing-quality policy uses a documented fallback of:

```text
0.5
```

which is the midpoint of the normalized quality range.

This is an engineering fallback, not an empirically optimized value.

The alternate configured behaviour may exclude a modality whose quality is
unavailable.

No policy treats missing quality as perfect quality.

## 31. Validation-Weighted Late Fusion

`validation_weighted_late` derives modality weights using grouped inner
validation data from the outer-training portion.

Outer-test outcomes do not determine the weights.

Classification weight construction is based on performance relative to
chance/reference classification performance.

Regression weighting is based on performance relative to predicting the
out-of-fold mean.

The resulting weights are bounded and recorded with their provenance.

## 32. Stacked Fusion

Stacked fusion is implemented but disabled by default.

Conceptually:

1. modality experts generate grouped out-of-fold predictions;
2. those predictions form a meta-training matrix;
3. a meta-model is fitted only on those out-of-fold predictions;
4. experts are refitted on the outer-training data;
5. the frozen meta-model combines predictions on untouched outer-test groups.

### Meta-models

Classification uses:

```text
LogisticRegression
```

Regression uses:

```text
Ridge
```

No neural stacker is implemented.

## 33. Stacking Missingness

An unavailable expert contributes a missing value to the stacking matrix.

It must not contribute:

- zero;
- a uniform probability vector;
- the target mean.

The stacker's own fold-local preprocessing handles that missingness and records
a missingness indicator.

## 34. Minimum-Modality Rule

Fusion occurs only when the configured minimum evidence requirement is met.

If too few modalities contribute:

```text
fused = false
```

with an explicit reason.

The system must not fabricate a fused prediction merely to preserve coverage.

## 35. Calibration Placement

Classification probability calibration occurs:

- per expert before late fusion where configured;
- on the early-fusion estimator where configured.

There is no general post-fusion calibrator.

Fused probability vectors may be evaluated for calibration, but evaluation does
not imply that a separate calibrator was fitted to them.

## 36. Expert Disagreement

Expert disagreement may be recorded as a fusion diagnostic.

It is not:

- calibrated uncertainty;
- signal quality;
- model confidence;
- participant state.

It must remain named and reported separately.

## 37. Missing-Modality Robustness

Fusion is designed to preserve real missingness.

Late fusion can renormalize across surviving experts.

Early fusion receives the corresponding missing-feature structure.

Synthetic modality dropout may test the software pathway, but does not establish
real-world robustness.

## 38. Model Selection Status

**No fusion strategy is selected as a champion.**

Metrics may compare strategies descriptively.

A strategy appearing first in a table or obtaining the strongest synthetic
metric does not make it scientifically selected.

## 39. Known Limitations

Current limitations include:

- no real multimodal participant evaluation;
- no empirically validated default quality thresholds;
- no optimized default base-weight set;
- stacked fusion disabled by default;
- quality-aware weighting depends on quality channels whose scientific validity
  may itself be limited;
- fusion performance does not establish construct validity.

---

# Card 4 — Personalization Layer

## 40. Card Summary

Personalization is a layer applied relative to a population reference model.

The population reference is the early-fusion estimator over the configured
modalities.

No mode trains a new participant-specific model from scratch.

## 41. Supported Personalization Modes

| Mode | Method value | Behaviour |
|---|---|---|
| Population-only baseline | `population_only` | Reproduces population output; no personalization applied |
| Personal-baseline feature calibration | `personal_baseline` | Z-scores eligible features using that subject's earlier calibration windows |
| Few-shot personalized calibration | `few_shot_correction` | Corrects population prediction using a few labelled calibration windows |
| Population model + user-specific correction | `personal_baseline_and_correction` | Applies both approaches; shipped default |
| Cold start | `cold_start` | Outcome state when personal evidence is insufficient |

## 42. Cold Start

`cold_start` is not a requestable personalization method.

It records that a requested personalization could not be applied because
sufficient personal evidence was unavailable.

In cold start:

- population prediction remains available;
- `personalization_applied` is false;
- `cold_start` is true;
- the reason is recorded.

## 43. Calibration/Evaluation Separation

Participant-specific information used for personalization must precede the
evaluation region.

Calibration windows and evaluation windows are temporally separated.

A window must not calibrate itself.

Evaluation labels must not enter the participant-specific correction.

## 44. Personal-Baseline Normalization

`personal_baseline` uses earlier participant calibration windows to estimate
participant-specific feature normalization.

It is unsupervised with respect to the target label.

This does not imply that normalization improves prediction.

## 45. Few-Shot Correction

Few-shot correction uses earlier labelled calibration observations to adjust the
population prediction.

It does not train an independent model from a handful of windows.

Minimum-evidence rules determine whether the correction can be applied.

## 46. Population Reporting

Population and personalized predictions are retained separately.

This permits direct comparison on the same evaluation observations.

A changed prediction is not automatically a better prediction.

## 47. Personalization Benefit Status

**No personalization benefit is currently claimed.**

Current personalization demonstrations use synthetic software-check data.

A synthetic difference between population and personalized predictions is not
evidence that real participants benefit.

Personalization may eventually:

- improve some participants;
- have little effect;
- worsen others.

All are scientifically valid possible outcomes.

## 48. Personalization vs Probability Calibration

These are separate concepts.

### Probability calibration

Adjusts population-model classification probabilities against observed
frequencies using disjoint calibration groups.

### Personalized calibration

Uses one subject's earlier observations to adapt that subject's prediction
process.

Neither is the same as uncertainty calibration.

## 49. Known Limitations

- personalization has not been exercised scientifically on real participants;
- default minimum-evidence gates are engineering settings rather than validated
  participant thresholds;
- a small personal calibration set may be unstable;
- personal baselines may be inappropriate under distribution shift;
- personalization is not automatically a fairness mechanism.

---

# Card 5 — Classification Calibration, Confidence, and Selective Prediction

## 50. Card Summary

This layer governs classification probability interpretation and whether an
actionable classification estimate should be emitted.

It is not a separate engagement classifier.

It operates on classification model outputs.

## 51. Calibrated Confidence

When the probability-calibration contract is satisfied:

```text
confidence =
max_c p_calibrated(c | x)
```

The class attaining this maximum is stored separately.

A value may only be persisted as calibrated confidence when a valid calibrator
was produced under the required disjoint-group procedure.

## 52. Uncalibrated Selection Score

When probability calibration is unavailable, the model may still possess an
uncalibrated probability vector.

Its maximum may be stored explicitly as:

```text
selection_score
```

under an uncalibrated name.

It must not be relabelled as calibrated confidence.

## 53. Default Refusal Behaviour

By default, classification confidence-based evidence gating requires valid
probability calibration.

When calibration is unavailable, the conservative default is to abstain rather
than overstate confidence.

## 54. Classification Diagnostics

Separate diagnostic quantities include:

```text
entropy
normalized_entropy
prediction_margin
maximum_probability
```

These remain separate fields.

They are not collapsed into an opaque general "uncertainty score."

## 55. Population Abstention Threshold

The configured default population confidence threshold is:

```text
0.70
```

This is an **engineering default**.

It is:

- not empirically optimal;
- not scientifically validated;
- not a production threshold.

## 56. Optional Threshold Estimation

Where enabled, population threshold estimation uses only appropriate calibration
groups from the outer-training portion.

The outer-test labels must not determine the threshold.

If the requested threshold objective cannot be supported, the system records
that it is unavailable rather than inventing a threshold.

## 57. Personalized Classification Threshold

A personalized classification threshold may be derived from the participant's
earlier confidence scores and shrunk toward the population threshold.

Conceptually:

```text
tau_raw =
quantile(subject_calibration_confidence, 1 - target_coverage)

lambda =
n / (n + kappa)

tau_subject =
(1 - lambda) * tau_population
+ lambda * tau_raw
```

This rule uses earlier confidence scores rather than evaluation labels.

Minimum-evidence rules apply.

When evidence is insufficient, it falls back explicitly to the population
threshold.

## 58. Classification Abstention Rule

Conceptually:

```text
accept  if score >= tau
abstain if score <  tau
```

The boundary is inclusive.

A rejected prediction is not described as an incorrect prediction.

The original prediction and diagnostics remain available for analysis.

## 59. Intended Use

Appropriate use includes:

- selective-prediction experiments;
- coverage-versus-performance analysis;
- uncertainty-aware evidence gating;
- conservative adaptation gating;
- comparison of population and personalized selection behaviour.

## 60. Unsupported Interpretations

Classification confidence is not:

- certainty;
- safety;
- signal quality;
- psychological confidence;
- probability that an adaptation is beneficial.

## 61. Scientific Status

No classification confidence threshold is currently validated for real
participant-facing use.

Synthetic coverage/performance curves are software demonstrations.

---

# Card 6 — Regression Conformal Uncertainty and Selective Prediction

## 62. Card Summary

Regression uncertainty uses split-conformal absolute-residual intervals.

It does not convert point predictions into classification-style confidence.

## 63. Conformal Calibration

Calibration residuals are calculated on groups separate from:

- estimator fit groups;
- outer-test groups.

For calibration residual:

```text
r_i = |y_i - yhat_i|
```

and nominal miscoverage `alpha`:

```text
k = ceil((n + 1) * (1 - alpha))
q = k-th smallest calibration residual

interval(x) =
[yhat(x) - q, yhat(x) + q]
```

## 64. Insufficient Calibration Evidence

When the required conformal order statistic cannot be defined:

```text
interval = unavailable
```

The interval is not:

- widened to infinity;
- replaced with zero width;
- fabricated.

## 65. Exchangeability Limitation

Split-conformal coverage depends on an exchangeability assumption between the
calibration and evaluation observations.

In EngageVR's grouped design, calibration and test observations may come from
different people.

Between-participant exchangeability has not been established.

Therefore:

> **No conformal coverage guarantee is currently claimed for real EngageVR
> participant data.**

## 66. Synthetic Coverage Results

Synthetic software checks may produce empirical coverage values.

Those results test implementation behaviour.

They do not establish the conformal guarantee for real participants.

## 67. Domain Projection

Optional interval clipping may project interval bounds into the target's
permitted range for presentation.

When used:

- raw bounds remain the interval of record;
- clipped bounds remain separately labelled;
- empirical conformal coverage is evaluated using the raw bounds.

Presentation clipping must not be represented as statistically justified
interval narrowing.

## 68. Regression Width-Based Abstention

Conceptually:

```text
accept  if interval_width <= maximum_interval_width
abstain otherwise
```

`maximum_interval_width` is unset by default.

When configured, it is an engineering threshold in the target's own units.

It is not a probability.

## 69. Direction of Selectivity

Increasing a classification confidence threshold makes classification
acceptance more restrictive.

Increasing a regression maximum-width threshold makes regression acceptance
more permissive.

Therefore, classification and regression threshold curves must remain separate
surfaces.

## 70. No Pseudo-Confidence Conversion

EngageVR does not compute:

```text
confidence = 1 - interval_width
```

Such a conversion would produce a probability-shaped value without valid
probability semantics.

## 71. Missing Interval Rule

An unavailable interval must not be treated as:

```text
interval_width = 0
```

because that would falsely imply perfect precision.

Instead, the prediction abstains with the appropriate unavailable state.

## 72. Personalized Regression Threshold

No personalized regression threshold is currently implemented.

A classification confidence threshold does not have valid semantics for a
regression point prediction.

Subject-specific conformal calibration from a very small number of personal
windows is also deliberately not implemented.

## 73. Expert Disagreement Boundary

Fusion expert disagreement must not be reused as a conformal prediction
interval.

The two concepts have different statistical semantics.

## 74. Scientific Status

No real EngageVR participant dataset has established:

- conformal coverage;
- valid maximum-width thresholds;
- participant-specific regression uncertainty;
- adaptation safety from conformal intervals.

---

# 75. Components That Are Not Predictive Models

Several EngageVR components interact with models but must not be misclassified
as predictive models.

| Component | What it actually is |
|---|---|
| Signal-quality logic | Measurement-usability assessment |
| Probability calibrator | Probability-calibration layer |
| Personalization correction | Participant-specific adaptation layer |
| Classification threshold | Selective-prediction decision rule |
| Conformal interval construction | Regression uncertainty procedure |
| Expert disagreement | Fusion diagnostic |
| Evidence gate | Decision gate |
| Adaptation policy | Conservative controller |
| `set_difficulty` | Environment command |
| Rule software-check estimators | Harness/software-test baselines |

## 76. Adaptation Policy Is Not a Model Card

The Milestone 8 adaptation controller consumes model/evidence outputs.

It is a policy/controller, not another engagement or cognitive-load predictor.

A policy decision such as:

```text
HOLD
increase difficulty
decrease difficulty
```

must not be interpreted as a new psychological prediction.

## 77. Signal Quality Is Not a Model

Signal quality describes whether a measurement is usable.

It is distinct from:

- engagement;
- cognitive load;
- model confidence;
- predictive uncertainty.

## 78. Expert Disagreement Is Not Calibrated Uncertainty

Differences among modality-specific experts can be useful diagnostics.

They are not a calibrated probability or prediction interval unless a future
method explicitly establishes such semantics.

## 79. Software-Check Baselines

The rule-based classifier and regressor exist so the training/evaluation
infrastructure can process deterministic non-standard estimators.

They must remain labelled as software checks rather than scientific reference
models.

---

# 80. Model Selection and Champion Policy

The current repository intentionally has **no champion model**.

This applies to:

- baseline classification models;
- baseline regression models;
- fusion strategies.

A table may be sorted by a metric for inspection.

Sorting is not model selection.

A scientifically defensible future champion would require:

- real appropriate data;
- predefined selection criteria;
- valid grouping;
- appropriate targets;
- leakage-safe evaluation;
- documented uncertainty;
- scientific justification.

## 81. Model-to-Hypothesis Mapping

| Model/system family | Primary hypothesis relevance |
|---|---|
| Baseline classification/regression | H1 reference models and general predictive evaluation |
| Multimodal fusion | H1 |
| Personalization | H2 |
| Classification selective prediction | H3 |
| Regression selective prediction | H3 |
| Adaptation controller | Consumes evidence for H4 experiment, but is not itself the H4 outcome |
| rPPG algorithms | H5 measurement component, documented separately from these predictive cards |

## 82. Data Requirements

A model result must identify the dataset that produced it.

Current possible data classes include:

- synthetic feature data;
- future appropriate public datasets;
- future participant data.

Synthetic data remain scientifically ineligible as evidence of human
engagement/cognitive-load model validity.

## 83. Reproducibility

Model artifacts should retain sufficient information to identify:

- dataset fingerprint;
- target;
- model family;
- hyperparameters;
- split configuration;
- seed;
- preprocessing;
- calibration;
- fusion configuration where applicable;
- personalization configuration where applicable;
- uncertainty/selectivity configuration where applicable;
- software environment.

Reproducibility establishes traceability.

It does not establish scientific validity.

## 84. Fairness and Generalization

No current model card claims equal performance across participant groups.

Future evaluation should report limitations of:

- participant population;
- acquisition environment;
- sensing hardware;
- missingness;
- signal quality;
- target construction.

Sensitive participant attributes should not be inferred merely to manufacture
subgroup labels.

## 85. Privacy

Models should operate on the minimum necessary research features.

Direct participant identifiers must not become predictor variables.

A trained artifact must not be assumed privacy-safe merely because direct names
were removed from the input dataset.

## 86. Human Oversight

No model documented here should currently make unsupervised high-stakes
participant decisions.

Participant-facing adaptive studies require:

- experimenter oversight;
- a stop/disable mechanism;
- uncertainty/availability handling;
- appropriate institutional review.

## 87. Reporting Requirements

Future scientific reporting should identify:

- exact model/configuration;
- data source;
- target;
- analysis population;
- grouping/split design;
- calibration status;
- metrics;
- uncertainty;
- missingness;
- scientific status.

Negative and null results must remain reportable.

## 88. Prohibited Current Claims

The current repository does not support claims that:

- any EngageVR model is scientifically validated for humans;
- any model is clinically validated;
- any model detects a participant's true mental state;
- any model is a production-ready psychological monitor;
- any fusion strategy is a champion;
- personalization is beneficial;
- abstention thresholds are clinically or scientifically optimal;
- conformal coverage is guaranteed for real EngageVR participants;
- adaptation benefit has been established.

## 89. Future Card Updates

These cards should be updated when any of the following materially changes:

- model registry;
- preprocessing;
- target definition;
- calibration procedure;
- fusion strategy;
- personalization mechanism;
- uncertainty procedure;
- selection/champion policy;
- scientific dataset;
- validated scientific status.

Historical scientific results should remain tied to the exact model and
configuration version that produced them.

## 90. Current Status

The current EngageVR model stack contains:

- interpretable classification baselines;
- interpretable regression baselines;
- grouped leakage-aware evaluation;
- probability calibration;
- early and late multimodal fusion;
- quality-aware and validation-weighted fusion;
- optional stacked fusion;
- population and personalized prediction paths;
- classification confidence/selective prediction;
- split-conformal regression intervals;
- uncertainty-aware abstention and evidence gating.

These capabilities are implemented as research software.

They have not yet established scientific validity on real participant-labelled
EngageVR engagement or cognitive-load data.

No champion model or fusion strategy currently exists.
