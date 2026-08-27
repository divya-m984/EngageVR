# Uncertainty-Aware Inference and Abstention (Milestone 7)

## Status

**Milestone 7 uncertainty-aware inference implementation complete;
scientific calibration and selective-prediction evaluation on real
participant-labelled data pending.**

Everything below describes the *machinery*. It has been exercised on
deterministic SYNTHETIC data only. A synthetic confidence value, coverage
curve, or interval-coverage number is **not** evidence of real-world
calibration, reliability, safety, or usefulness.

## Terminology

These concepts are kept distinct in schema, code, artifacts, and prose.
Nothing in this milestone merges them, and there is **no field anywhere
named merely `uncertainty`** — such a field would make the rows of this
table indistinguishable to a reader.

| Term | What it describes | What it is **not** |
|---|---|---|
| **Signal quality** | A property of the *measurement*: face-tracking quality, rPPG quality, modality availability, missing measurements | Not model confidence, not probability, not uncertainty, not engagement, not cognitive load |
| **Predicted probability** | The model's class-probability vector for one window | Not automatically calibrated |
| **Probability calibration** | Whether that vector was fitted against observed outcomes on disjoint groups | Not certainty; not a property of a single window |
| **Model confidence score** | `max_c p_calibrated(c \| x)`, with the class that attained it | Not epistemic uncertainty, not certainty, not psychological confidence, not safety |
| **Selection score** | The same maximum taken from an **uncalibrated** vector | Never called calibrated confidence |
| **Predictive entropy** | `H(p) = -Σ_c p_c log p_c`, nats | Not signal quality; not automatically calibrated uncertainty |
| **Prediction margin** | `p_(1) − p_(2)` | A ranking diagnostic, not calibrated confidence |
| **Ensemble disagreement** | The Milestone 6 diagnostic, carried beside M7 output unchanged | Never renamed uncertainty; does not by itself trigger abstention |
| **Regression prediction interval** | A bounded numeric interval from a documented calibration procedure | Not an arbitrary standard deviation and not an ensemble range |
| **Abstention** | A deliberate decision not to emit an actionable estimate because a declared rule was not satisfied | Not "modality unavailable", not "feature missing", not "model unavailable", not "calibration unavailable", not a runtime error |

**Low signal quality is never low engagement and never low cognitive load.**
Quality is not multiplied into a model probability anywhere: no
probabilistic model in this repository justifies treating a camera
diagnostic as a likelihood term, so quality and confidence gate an
actionable estimate *independently*, each with its own reason code.

## Relationship to existing abstention semantics

Milestone 7 **reuses** the semantics that were already in the repository
rather than defining a second, unrelated notion of abstention.

| Existing | Where | How M7 relates |
|---|---|---|
| `EngagementPrediction.abstain` / `.reason`, with `confidence` and `signal_quality` as separate fields | `src/engagevr/schemas/prediction.py` | The M1 runtime contract. M7's `AbstentionDecision` is the *offline modelling* form of the same idea and maps onto it field for field; the runtime schema is unchanged. |
| `DatasetEvaluationRecord.abstained` + `RppgEvaluationMetrics.coverage` | `src/engagevr/schemas/rppg.py` | The invariant M7 preserves: an abstained window **reduces coverage** and is never converted into an ordinary error. |
| `unavailable_metrics`, `unavailable_reason`, `UnavailableReason` | Milestones 3 and 5 | Kept distinct from abstention. "Unavailable" means a quantity could not be computed; "abstained" means one was computed and deliberately not acted upon. |

The invariant, stated once and enforced in three places (the pure metric
functions, the schema validators, and the tests):

> An abstained prediction contributes to coverage calculations but is never
> silently converted into an incorrect prediction or a zero-valued
> regression estimate.

## Classification

### Confidence

```
confidence = max_c p_calibrated(c | x)
```

together with `maximum_probability_class`, the class that attained it.

The **calibration contract** is the Milestone 5 one, unchanged: a
calibrator was fitted on calibration groups drawn from the training groups
and disjoint from both the fit groups and the outer-test groups, and it
produced an estimator. Anything else is `uncalibrated`.

When the contract is not satisfied — most often because a class has fewer
than `MINIMUM_CALIBRATION_SAMPLES_PER_CLASS = 5` calibration rows — the
record keeps its probability vector, sets
`probability_calibration_status = "uncalibrated"`, states why, and records
the identical number as **`selection_score`** under
`max_uncalibrated_probability`. `ClassificationConfidence` refuses to
validate a record that populates `confidence_score` without the contract,
so an uncalibrated maximum cannot be persisted as calibrated confidence.

By default the evidence gate then **refuses** confidence-based abstention
for that fold (`require_probability_calibration_for_classification_
confidence: true`), and every window in it abstains with
`probability_calibration_unavailable`. Refusal is preferred to overstating
calibration. Setting the flag to `false` opts into an explicitly named
uncalibrated selection-score policy; the number is still never called
confidence.

### Diagnostics, stored separately

```
H(p)      = -Σ_c p_c · ln(p_c)        with 0·ln 0 = 0, in nats
H_norm(p) = H(p) / ln(K)              K ≥ 2 classes, in [0, 1]
margin    = p_(1) − p_(2)
```

`H_norm` is `None` when `K < 2` rather than zero: zero would read as a
perfectly certain prediction, which is a different statement from "the
ratio is undefined". A top-two margin over a single-class vocabulary is
refused for the same reason.

The four numbers — maximum probability, entropy, normalised entropy,
margin — are stored in four columns. They are never collapsed into one
opaque number.

### Abstention rule and boundary convention

```
accept  if score >= tau
abstain if score <  tau
```

**The boundary is inclusive**: a score exactly equal to `tau` is accepted.
The convention is stated on `SelectivePredictionConfiguration`, recorded on
every `AbstentionDecision`, and pinned by a test, because an
off-by-one-epsilon disagreement between the rule and the curve would
silently shift every reported coverage.

Suggested reason code: `below_confidence_threshold`. A rejected prediction
is **not** called wrong — it is a prediction that exists and was not acted
upon. The original class, its probability vector, its calibration status,
its entropy, and its margin all remain on the decision record.

## Population threshold

```yaml
uncertainty:
  classification:
    population_confidence_threshold: 0.70
```

**0.70 is an ENGINEERING DEFAULT.** It was not selected by looking at any
result, it is not empirically optimal, it is not validated, and it is not a
production threshold. `thresholds.json` says so in a required field.

### Optional leakage-safe estimation

`estimate_population_threshold: true` selects the threshold per fold from
that fold's **calibration groups only** — disjoint from both the rows that
fitted the model and the outer-test rows.

| Property | Rule |
|---|---|
| Objective | `target_accepted_accuracy` (default), `target_empirical_risk`, or `target_coverage`; recorded explicitly |
| Tie-break | Among admissible grid points the **smallest** threshold wins, maximising coverage; the grid is walked ascending, so the choice is deterministic. For `target_coverage` the **largest** admissible wins, because coverage falls as `tau` rises. |
| Minimum evidence | `minimum_threshold_selection_samples` (30) and `minimum_threshold_selection_groups` (2) |
| Unreachable objective | `available: false` with a reason; the configured threshold applies. **No threshold is invented to satisfy an unreachable target.** |
| Leakage | `select_population_threshold` raises if handed a group that is also an outer-test group. The record carries `used_outer_test_labels: false` so the claim is auditable from the artifact. |

## Personalized threshold

Milestone 6 deferred personalized confidence thresholds to here. The
implementation is deliberately conservative.

```
tau_raw = quantile(subject calibration confidence, 1 − target_coverage)
lambda  = n / (n + kappa)
tau_s   = (1 − lambda) · tau_population + lambda · tau_raw
```

clipped to `[0, 1]`. `quantile` uses numpy's `"lower"` method, so `tau_raw`
is an **observed** confidence value rather than an interpolated one no
window produced.

**The rule reads no labels at all.** It consumes only the confidence scores
the population model assigned to the subject's own earlier windows. An
evaluation label therefore cannot influence it *by any path* — a stronger
statement than "we were careful not to pass one". `PersonalThresholdRecord`
carries `uses_labels: false` and refuses to validate if set otherwise.

The failure it addresses is real: a subject the model is uniformly less
confident about would otherwise be abstained on entirely by a population
threshold, which is a measurement artefact presented as a property of that
person. Shrinkage toward the population value keeps a thin calibration set
from moving the threshold far.

| Guarantee | Mechanism |
|---|---|
| Population model never trains on the held-out subject | The Milestone 5 grouped split, unchanged |
| Only earlier windows are read | Milestone 6's `build_calibration_split`, unchanged: the earliest `personal_calibration_windows` windows form the calibration region, everything starting at or after that region's **wall-clock end** forms the evaluation region, and a window straddling the boundary is excluded from both |
| A calibration window is never also scored | Those windows are removed from the fold's evaluated set before any decision is taken |
| Minimum evidence | `minimum_personal_calibration_windows` (5). Below it, explicit fallback. |
| Fallback | `threshold_source: population_fallback` with a stated `fallback_reason`. `fallback_to_population_threshold` cannot be disabled. |
| Recorded | population threshold, personalized threshold, applied threshold, source, calibration window ids, evaluation window ids, sample count, both boundary timestamps, selection method, shrinkage, raw quantile |

No elaborate per-subject uncertainty model is fitted from five windows, and
the rule was not tuned to make synthetic personalization look better.

### Not implemented, and why

**Regression has no personalized threshold.** A confidence threshold has no
meaning for a point prediction, and subject-conditional conformal intervals
would need a per-subject residual distribution estimated from a handful of
calibration windows — which would overfit, and which would put the
subject's own outcomes into their own interval. The run emits a warning
saying so rather than silently doing nothing.

## Regression

### Split conformal absolute-residual interval

Calibration residuals are computed on the fold's calibration groups, which
are disjoint from the rows that fitted the estimator and from the
outer-test rows:

```
r_i = |y_i − ŷ_i|
```

For nominal miscoverage `α` and `n` calibration residuals:

```
k = ceil((n + 1) · (1 − α))
q = the k-th smallest residual          (1-indexed order statistic)

interval(x) = [ŷ(x) − q,  ŷ(x) + q]
```

**When `k > n` the interval is UNAVAILABLE**, with a reason. It is never
widened to infinity and never fabricated. The rule first holds at
`n = ceil(1/α) − 1`: 9 residuals at `α = 0.10`, 19 at `α = 0.05`.

`q` is derived only from calibration residuals. The outer-test rows never
influence it, and `UncertaintyFoldResult` refuses to validate a fold whose
conformal calibration groups intersect either the fit groups or the
outer-test groups.

### Assumptions and limitations

Split conformal prediction gives marginal coverage of at least `1 − α`
**when the calibration and test points are exchangeable** (Vovk et al.
2005; Papadopoulos et al. 2002; Lei et al. 2018).

Under grouped cross-validation the calibration and test rows come from
**different people**. Exchangeability across subjects is therefore an
assumption about between-subject variation that this repository has never
tested, and there is good reason to expect it to fail: a physiological or
behavioural signal is not exchangeable between one person and another.

On the 30-subject synthetic dataset the empirical interval coverage varies
between **0.846 and 0.963 per fold** against a nominal 0.90, with a
cross-fold mean near nominal. Wide per-fold dispersion is what a violated
exchangeability assumption produces; a mean that happens to land near
nominal is not the guarantee holding, and on a different set of subjects it
need not.

**No conformal coverage guarantee is claimed for real EngageVR data.**

### Domain projection

`clip_interval_to_target_range` is off by default. When enabled, clipped
bounds are recorded **alongside** the raw ones and labelled
`PRESENTATION PROJECTION ONLY`; the raw bounds remain the interval of
record, and empirical interval coverage is always computed on them.
Clipping narrows an interval with no statistical justification for doing
so, so a clipped interval is never scored as though it were the conformal
one.

### Width-based abstention

```
accept  if interval_width <= maximum_interval_width
abstain otherwise
```

The boundary is inclusive. `maximum_interval_width` is `null` by default
(no width abstention). It is an engineering threshold, never derived from
outer-test outcomes, and it is a **width in the target's own units** — not
a probability, and not confined to [0, 1].

Raising `maximum_interval_width` is *more permissive*, which is the
opposite direction from raising a confidence threshold. The two are
therefore swept on separate axes with separate monotonicity contracts; see
"Coverage-versus-performance curve" below.

`confidence = 1 − interval_width` is **not** computed anywhere. It would be
a probability-shaped number with no probabilistic meaning.

**A missing interval is never treated as width zero**, which would read as
a perfectly certain prediction and would be accepted by every threshold.
It abstains with `prediction_interval_unavailable`.

Milestone 6 disagreement spread is never used as a calibrated interval.

## Evidence gate

Separate from model confidence, and never combined with it arithmetically.
An actionable estimate requires:

```
model_prediction_available
AND required_evidence_available
AND quality_requirement_satisfied
AND confidence_requirement_satisfied
```

Each failure keeps its own reason code:

| Reason | Fires when |
|---|---|
| `model_prediction_unavailable` | No prediction exists for the window at all |
| `insufficient_measurement_evidence` | Fewer modalities contributed than `minimum_available_modalities` |
| `required_modality_unavailable` | A modality named in `required_modalities` produced nothing |
| `signal_quality_below_gate` | A recorded modality quality fell below `minimum_signal_quality` |
| `probability_calibration_unavailable` | No calibrated probability exists and the policy requires one |
| `prediction_interval_unavailable` | No interval could be constructed |
| `below_confidence_threshold` | The score fell below the applied threshold |
| `interval_too_wide` | The interval was wider than the maximum |

Declaration order is the canonical reporting order, and evidence reasons
precede model-confidence reasons: an estimate built on absent evidence
should not be discussed in terms of its confidence.

**Absence of a recorded quality is not a low quality.** A modality with no
quality column does not fail the gate unless
`treat_missing_quality_as_failure` is explicitly set.

## Coverage and selective metrics

```
coverage        = accepted_count / total_window_count
abstention_rate = abstained_count / total_window_count
accepted + abstained + unavailable = total
```

The denominator is **every evaluated window**, including those with no
prediction — recorded as `coverage_denominator: "total_evaluated_windows"`.
Excluding them would let a run raise its reported coverage by producing
fewer predictions. `CoveragePoint` refuses to validate if the three counts
do not reconcile with the total.

Reported per threshold, classification: total / accepted / abstained /
unavailable counts, coverage, abstention rate, accepted accuracy, accepted
balanced accuracy, accepted macro precision / recall / F1, accepted
weighted F1, accepted log loss, accepted Brier score, accepted ECE, and
class support among accepted predictions.

Regression: total / accepted / abstained / unavailable counts, coverage,
accepted MAE, RMSE, median absolute error, R², empirical interval coverage
(on the **raw** bounds), and mean and median interval width.

Every metric is the Milestone 5 function, unchanged — so the
undefined-stays-null rule, the documented macro / Brier / ECE / log-loss
conventions, and the equal-weight fold aggregation are inherited rather
than reimplemented. An empty accepted set returns *unavailable with a
reason*, never zero. Fold-level results are stored before aggregates.

`metrics.json` carries two `ModelResult` entries over the same folds:

| `model_name` | `model_kind` | Scores |
|---|---|---|
| `all_windows` | `all_windows` | Every evaluated window |
| `accepted_at_applied_threshold` | `selective` | Only the accepted windows |

An accepted-set score is never comparable to a whole-set score without its
coverage, and the two are always written together.

## Coverage-versus-performance curve

A deterministic sweep over the **same** outer-test predictions. The two
task types sweep **different axes in different units and in opposite
directions**, so each curve records which axis it was swept over.

| | Classification | Regression |
|---|---|---|
| x-axis (`CoverageCurve.axis`) | `confidence_threshold` | `maximum_interval_width` |
| units | probability in [0, 1] | the target's own units |
| rule | `accept if score >= tau` | `accept if interval_width <= W_max` |
| raising the axis value | **stricter** | **more permissive** |
| coverage direction | non-increasing | non-decreasing |
| monotonicity check | `coverage[i+1] <= coverage[i]` | `coverage[i+1] >= coverage[i]` |
| configured by | `uncertainty.classification.threshold_grid` | `uncertainty.regression.interval_width_grid` |

The shipped confidence grid:

```
0.00 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95 0.99
```

### The two grids are separate surfaces

They are not one grid used twice, and neither is derived from the other:

- A confidence value is a **probability**. An interval width is a
  **distance in the target's units**; a general regression target need not
  live in [0, 1], so a width of 2.5 or 40.0 is an ordinary value and is
  accepted by the configuration.
- A width is **never normalised** into [0, 1] to share the confidence
  grid, and is **never inverted** into `confidence = 1 − width`
  (see "Width-based abstention" above).
- Each grid point is compared against the original quantity by the same
  rule the run applies at its operating point.

`interval_width_grid` is `null` by default. There is no scale-free default
sweep to ship, because the right widths depend on a target scale this
repository has not measured. When it is null, **no curve is manufactured**:
the run records its operating point and marks the width curve unavailable
with `points_unavailable_reason`, which names the configuration key and
states that a confidence grid cannot stand in for a width grid. The
operating point itself records `threshold: null` with a stated reason when
no `maximum_interval_width` is configured — an absent width policy accepts
everything, whereas a width policy of `0.0` would accept nothing, and the
two must never be recorded the same way.

The evidence gate is applied at every grid point, so a window blocked for
missing evidence is blocked at every axis value and the direction contract
holds under either axis.

Classification also stores a risk-coverage representation:

```
empirical_risk = 1 − accepted_accuracy
```

This is empirical classification risk **under this definition**. It is not
a bound, not a guarantee, and not a statement about any person. It is
undefined for a regression target, and recorded as unavailable with that
reason rather than as zero.

### Area under the risk-coverage curve

Optional and descriptive. Points are sorted by ascending coverage, the
trapezoidal rule is applied, and the area is divided by the covered
coverage span so the result reads as a mean risk rather than an
unnormalised area whose magnitude depends on how much of the coverage axis
the grid happened to span. At least two points with a defined risk and
distinct coverage are required; otherwise it is unavailable with a reason.

**A lower AURC on synthetic data establishes nothing about safety,
superiority, or usefulness.**

### Split conformal produces a step-shaped regression curve

Split conformal yields **one quantile per fold**, so every window in a fold
shares one interval width. The width sweep is therefore all-or-none within
a fold: coverage steps from 0 up to 1 at the first grid value that reaches
that shared width, rather than tracing a gradual curve. Pooled across folds
the step appears at each fold's own width, so the run curve rises in as
many steps as there are distinct fold widths. That is a property of the
method. It is stated here rather than smoothed over by manufacturing
per-window variation the method does not produce.

## Leakage prevention

For every outer fold, four group sets are recorded and re-checked by
`UncertaintyFoldResult`'s validator:

| Set | Field |
|---|---|
| Base-estimator fitting groups | `fit_group_ids` |
| Probability-calibration groups | `probability_calibration_group_ids` |
| Threshold-selection groups | `threshold_selection_group_ids` |
| Conformal-calibration groups | `conformal_calibration_group_ids` |
| Outer-test groups | `outer_test_group_ids` |

The outer-test fold **never**:

- fits the model
- fits a probability calibrator
- fits a conformal residual distribution
- chooses a confidence threshold
- chooses an interval-width threshold
- tunes an abstention rule
- tunes a personalized confidence threshold

and **no threshold anywhere in this repository is read off the reported
coverage curve.** The curve is a report, not a search;
`CoverageCurve.selection_note` says so.

A document that violated any of this cannot be persisted: the validator
raises, and the run writes a `failed` manifest.

## Adaptation gate

`src/engagevr/training/adaptation_gate.py` answers exactly one question:

> May an ALREADY-CHOSEN adaptation action be acted upon, given the current
> prediction and evidence state?

It does **not** answer "which adaptation should be chosen?".

| Decision | When |
|---|---|
| `eligible` | The prediction exists, the evidence gate passed, and the selective layer accepted it |
| `blocked` | Otherwise, with the same reason codes the selective layer recorded, in the same canonical order |

The gate consumes already-computed information and recomputes nothing, so a
gate decision can never disagree with the decision it is gating. A disabled
gate does not make everything eligible — it stops applying any *additional*
requirement of its own, but still refuses to declare an abstained or
unavailable window eligible, because that would be a false statement rather
than a relaxed policy.

**The gate is not an adaptation policy.** It has no action type, no action
registry, no difficulty level, no scene identifier, no ordering over
actions, no reward, no cooldown, no hysteresis, no state carried between
windows, and no transport client. A test parses the module's imports and
asserts it imports nothing but two schema modules — so the claim is
checkable rather than asserted. Adaptation policy is **Milestone 8**.

### How Milestone 8 consumes this gate

Milestone 8 (`src/engagevr/adaptation/`, `docs/ADAPTIVE_ENVIRONMENT.md`) is
the layer that chooses. It treats this gate as a **hard prerequisite with no
override**:

- an `AdaptationGateRecord` with `decision = blocked` forces the policy to
  `HOLD`;
- the Milestone 7 reasons are preserved **verbatim, in this module's
  canonical order**, on the Milestone 8 record. Milestone 8 may add its own
  reason (`gate_blocked`, `prediction_abstained`, `prediction_unavailable`)
  but never erases the underlying ones;
- `AdaptationProposal` **embeds** the gate record of every target it used and
  refuses to validate unless all are `eligible`, so a proposal's eligibility
  is provable from the object rather than asserted by it;
- Milestone 8 never recomputes `max(probabilities)`, never lowers a
  threshold, never overrides a personalized or population threshold decision,
  and never uses entropy or margin to turn a blocked window into an eligible
  one. A test parses every Milestone 8 module and asserts that none
  constructs an `AdaptationGateRecord` or calls `evaluate_adaptation_gate`.

Milestone 8 also carries the same separation forward: signal quality reaches
the policy only as this gate's provenance and as a diagnostic field, and no
rule there can turn a quality value into an adaptation direction. Confidence
decides whether a window may be acted on; it never scales how far the
environment moves.

## Transport and online inference

**No transport change was made and no new API route was added.** The
Milestone 4 protocol version is untouched.

`EngagementPrediction` (`src/engagevr/schemas/prediction.py`) already
carries `confidence`, `signal_quality`, `abstain`, and `reason` as separate
fields, with each of the first two documenting that it is not the other.
Milestone 7's records are the *offline modelling* form of the same
distinctions and map onto that contract without changing it.

Online inference — serving a confidence-aware prediction over the
WebSocket bridge — is deliberately **not** implemented here.

## Scientific mode

`uncertainty-train --mode scientific` inherits the Milestone 5 gate
unchanged and refuses any dataset whose rows are synthetic, whose targets
forbid scientific evaluation, or whose target provenance is unstated. In
addition:

- confidence evaluation requires explicit real target provenance;
- threshold estimation requiring labels refuses absent or unverified labels;
- conformal calibration requires explicit eligible target labels;
- synthetic modality dropout remains prohibited;
- synthetic threshold optimisation remains software-self-check only;
- synthetic confidence scores remain scientifically ineligible.

**No software self-check becomes scientifically eligible merely because its
probabilities were calibrated on synthetic labels.** Passing the gate
establishes *eligibility, not validity*.

## Run directory

See `docs/EXPERIMENT_TRACKING.md` for the full layout. The Milestone 5
format is extended, not replaced.

## What is not claimed

- No confidence value here is validated.
- No threshold here is optimal, validated, or a production threshold.
- No coverage curve here is evidence of real-world calibration.
- No abstention rule here has been shown to improve anything.
- No conformal coverage guarantee is claimed for real EngageVR data.
- Nothing here is a psychological, clinical, or diagnostic conclusion.
- Real participant-labelled calibration and selective-prediction evaluation
  remain **pending**, because no validated participant-labelled engagement
  or cognitive-load dataset exists in this project.

## How the Milestone 9 dashboard displays this

The dashboard's Uncertainty-and-abstention page preserves this document's
vocabulary structurally rather than by convention.

**The two task types cannot borrow each other's controls.**
`UncertaintyDashboardData` refuses to be constructed with a
calibrated-confidence, probability-margin, or confidence-curve field when
the task type is regression, and refuses interval fields when it is
classification. The page hides the other task's controls rather than showing
them disabled, because it structurally cannot carry them.

**The two coverage axes keep their own names, units, and directions.**
Classification is swept over `confidence_threshold` (a probability in
`[0, 1]`; raising it is stricter, so coverage is non-increasing). Regression
is swept over `maximum_interval_width` (the target's own units; raising it is
more permissive, so coverage is non-decreasing). Neither is relabelled
"uncertainty threshold", and `1 - interval_width` is never computed anywhere
in the dashboard.

**A curve whose recorded axis disagrees with its task type is not shown**,
with the disagreement stated. A curve written before DEC-072 carries no
`axis` field at all, because the two axes then shared one grid; such a curve
is refused, because which axis it was swept over cannot be established and
must not be guessed.

**The three selective outcomes reconcile.** `accepted + abstained +
unavailable = evaluated` is checked, and a mismatch is displayed as an
**artifact validation error** with the recorded counts unchanged — never
normalised away. An abstention is not an error and is not counted as one; an
unavailable window is a separate state again, because nothing was withheld.

**Calibrated confidence, predictive entropy, and probability margin have
three separate charts**, each captioned with what it is not. There is no
card named simply "Confidence" and no single combined uncertainty score.

See `docs/DASHBOARD.md`.
