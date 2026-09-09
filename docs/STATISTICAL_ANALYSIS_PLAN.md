# EngageVR Statistical Analysis Plan

> **Draft for research planning and institutional review**
>
> This statistical analysis plan (SAP) defines prospective analysis principles
> for future EngageVR research.
>
> No human-subject study described here has been conducted. No institutional or
> ethical approval is claimed. No participant sample size, primary outcome, or
> confirmatory statistical test is considered final until the study design is
> finalized and appropriately reviewed.
>
> This document must be finalized before confirmatory participant outcome
> analysis begins.

## 1. Purpose

This document defines the statistical-analysis framework for EngageVR.

It is intended to prevent:

- retrospective primary-outcome selection;
- inappropriate independence assumptions;
- participant/session leakage;
- selective metric reporting;
- silent exclusion of poor-quality observations;
- silent removal of abstentions;
- confusion between software diagnostics and scientific evidence;
- interpretation of statistical significance as participant benefit.

It should be read alongside:

- `docs/RESEARCH_PROPOSAL.md`;
- `docs/RESEARCH_QUESTIONS.md`;
- `docs/HYPOTHESES.md`;
- `docs/EXPERIMENTAL_VARIABLES.md`;
- `docs/EXPERIMENT_DESIGN.md`;
- `docs/DATA_COLLECTION_PROTOCOL.md`;
- `docs/ETHICS_AND_PRIVACY.md`;
- `docs/RISK_ASSESSMENT.md`.

## 2. Analysis Status

The current SAP is prospective.

At the time of drafting:

- no EngageVR participant dataset exists;
- no participant sample size has been justified;
- no final primary participant outcome has been selected;
- no final subjective instrument has been selected;
- no participant-facing adaptation experiment has been completed;
- current internal model demonstrations use synthetic data and are not
  scientifically eligible.

Therefore, this document defines analysis rules and candidate methods rather
than reporting study results.

## 3. Analysis Domains

EngageVR contains several scientifically distinct analysis domains.

They must not be collapsed into one omnibus analysis.

### 3.1 Predictive-model analysis

Relevant primarily to:

- H1 — multimodal fusion;
- H2 — personalization;
- H3 — selective prediction and abstention.

### 3.2 rPPG measurement analysis

Relevant primarily to:

- H5 — webcam-rPPG robustness.

### 3.3 Participant experimental analysis

Relevant primarily to:

- H4 — static versus adaptive task condition.

### 3.4 Exploratory analyses

May include:

- agreement between physiological estimates and self-report;
- missing-modality robustness;
- adaptation-appropriateness measures;
- temporal modelling;
- participant heterogeneity;
- adaptation exposure patterns.

Exploratory findings must remain labelled exploratory.

## 4. Units of Analysis

Different research questions require different analysis units.

Possible units include:

- participant;
- participant-condition;
- participant-session;
- task block;
- trial;
- feature window;
- prediction opportunity;
- rPPG window;
- adaptation opportunity;
- delivered adaptation.

The analysis must explicitly state its unit.

A dataset containing thousands of windows from ten participants does not have a
participant sample size of thousands.

Repeated windows from the same participant are not independent participants.

## 5. Participant Grouping

Participant identity is a statistical grouping variable.

Where multiple observations originate from one participant, analyses must
account for that dependency through an appropriate method such as:

- participant-level aggregation;
- paired analysis;
- repeated-measures analysis;
- mixed-effects modelling;
- grouped resampling.

The method must match the research question and outcome.

## 6. Session Grouping

Multiple sessions from the same participant are also dependent.

Session identifiers should remain available for:

- leakage prevention;
- hierarchical analysis;
- sensitivity analysis;
- protocol-deviation tracking.

Random row-level splitting must not separate highly related windows across
training and evaluation sets when doing so would cause leakage.

## 7. Analysis Populations

Future participant studies should define analysis populations before
confirmatory analysis.

Candidate populations may include:

### 7.1 Enrolled population

All participants who enter the study according to the approved enrolment
procedure.

### 7.2 Started-session population

Participants who begin at least one experimental session.

### 7.3 Completed-condition population

Participants with usable data from a specified condition.

### 7.4 Paired-comparison population

Participants with outcome data from both static and adaptive conditions where
the primary analysis requires a paired comparison.

### 7.5 Analysis-specific eligible population

Participants or observations satisfying predefined eligibility requirements
for a particular analysis.

A participant may be eligible for task-performance analysis while being
ineligible for rPPG analysis.

## 8. Primary Outcome

The primary outcome for the static-versus-adaptive participant study has not
yet been selected.

Candidate outcomes currently include:

- task accuracy;
- reaction time;
- completion time where scientifically appropriate.

The final SAP must specify exactly one primary outcome or a clearly defined
primary outcome family with an appropriate multiplicity strategy.

The primary outcome must be selected before confirmatory participant outcome
analysis.

## 9. Primary Outcome Selection Criteria

The selected outcome should:

- directly address the main experimental question;
- have a clear operational definition;
- be reliably recorded;
- have an interpretable analysis unit;
- remain interpretable under adaptive difficulty;
- avoid unnecessary multiplicity.

A metric should not be selected merely because preliminary data show a
favourable difference.

## 10. Secondary Outcomes

Potential secondary outcomes include:

- additional task-performance metrics;
- subjective engagement;
- perceived cognitive load or mental effort;
- adaptation acceptance;
- comfort;
- perceived disruption;
- prediction availability;
- abstention rate;
- signal-quality summaries;
- applied adaptation count.

Secondary outcomes must be distinguished from the confirmatory primary outcome.

## 11. Exploratory Outcomes

Exploratory outcomes may include:

- participant-specific response patterns;
- adaptation timing;
- relationships between signal quality and prediction availability;
- physiological/self-report associations;
- condition-order interactions;
- model disagreement;
- adaptation-density measures;
- subgroup patterns where scientifically and ethically justified.

Exploratory analyses should not be retrospectively promoted to confirmatory
status.

## 12. Static-versus-Adaptive Primary Comparison

The proposed design is currently a within-participant crossover design.

Conceptually, each participant may contribute:

```text
Y_static
Y_adaptive
```

for the predefined primary outcome.

The primary estimand should reflect the within-participant condition
difference.

Conceptually:

```text
condition_effect = Y_adaptive - Y_static
```

The sign and interpretation depend on the selected outcome.

For example:

- higher accuracy may represent better task correctness;
- lower reaction time may represent faster responding;
- neither automatically proves greater engagement.

## 13. Candidate Primary Statistical Model

The exact model will depend on the selected outcome and final dataset.

For an approximately continuous participant-condition outcome, candidate
methods include:

- paired-sample analysis;
- repeated-measures regression;
- linear mixed-effects modelling.

For trial-level binary correctness, a generalized mixed-effects model may be
more appropriate than averaging every trial into an independent observation.

The final method must be selected prospectively after the primary outcome and
data-generating structure are finalized.

## 14. Paired Analysis

If each participant contributes one valid summary value per condition, a paired
analysis may be appropriate.

Potential examples include:

- paired mean difference;
- paired median-oriented analysis where justified;
- paired confidence interval.

Choice of a parametric or non-parametric procedure should depend on the
estimand and distributional considerations, not an automatic normality-test
decision tree.

## 15. Mixed-Effects Analysis

Where trial- or block-level observations are retained, a mixed-effects model may
represent repeated participant observations more appropriately.

A conceptual structure may include:

```text
outcome ~ condition + period/order terms + participant random effect
```

Additional fixed or random effects should be included only when scientifically
justified.

Complexity should not exceed what the sample and design can support.

## 16. Condition Order and Period Effects

The proposed crossover design includes candidate sequences:

```text
Static -> Adaptive
Adaptive -> Static
```

The analysis should consider:

- condition;
- period;
- sequence/order.

Condition order must not be ignored automatically.

A difference caused primarily by practice or fatigue should not be attributed
uncritically to adaptation.

## 17. Carryover

Carryover should be considered because participants experience both conditions.

Possible sources include:

- learned task strategy;
- fatigue;
- familiarity;
- residual difficulty expectations.

The final protocol should minimize carryover through design.

Statistical adjustment is not a substitute for an obviously unsuitable study
design.

## 18. Task Difficulty as a Time-Varying Variable

Task difficulty is not identical to experimental condition.

Within the adaptive condition, difficulty may change over time.

Therefore, interpretation of outcomes such as accuracy or reaction time may
require consideration of:

- difficulty exposure;
- trial difficulty;
- difficulty transitions.

Higher accuracy under an adaptive condition cannot automatically be interpreted
as improved engagement if the adaptive controller primarily exposed the
participant to easier trials.

## 19. Adaptation Exposure

An adaptive-condition assignment does not guarantee exposure to an adaptation.

Candidate exposure variables include:

- number of proposals;
- commands dispatched;
- commands acknowledged;
- applied adaptations;
- time under each difficulty level.

Condition assignment should remain the primary experimental variable for the
main condition comparison.

Post-randomization adaptation exposure should not casually replace condition
assignment as though it were independently randomized.

## 20. Experimenter Disablement

If an adaptive-condition session has adaptation disabled by the experimenter,
the event must remain visible.

The SAP should eventually define whether such a session contributes to:

- primary intention-by-assignment analysis;
- per-protocol analysis;
- sensitivity analysis;
- no specific analysis.

The decision must be defined before inspecting condition outcomes.

## 21. Protocol Deviations

Deviations may include:

- wrong condition;
- incorrect configuration;
- participant interruption;
- experimenter disablement;
- task restart;
- sensor failure;
- missing questionnaire.

Each deviation should be categorized.

Exclusion should be analysis-specific rather than automatically deleting the
entire participant.

## 22. Missing Data

Missing data must retain its reason where known.

Relevant states include:

- not collected;
- technically unavailable;
- quality-gated unavailable;
- participant non-response;
- participant withdrawal;
- model abstention;
- protocol exclusion.

These states must not be replaced with zero.

## 23. Missing Participant Outcomes

The final SAP must define treatment of missing primary outcomes.

Possible approaches depend on:

- frequency of missingness;
- reason;
- design;
- sample size;
- estimand.

Complete-case analysis must not be treated as automatically unbiased.

Any imputation method would require explicit justification and documentation.

## 24. Physiological Missingness

Poor-quality rPPG should generally produce unavailable physiological output.

Participants with unusable rPPG data may still contribute valid:

- task;
- behavioural;
- questionnaire;

data.

Physiological missingness must not automatically exclude a participant from all
study analyses.

## 25. Model Abstention

Abstention is not ordinary missing data.

It is an explicit model decision.

Selective-prediction analysis should therefore report:

- number of eligible opportunities;
- accepted predictions;
- abstained predictions;
- unavailable inputs;
- prediction coverage.

## 26. Exclusions

Every confirmatory exclusion rule should be defined before analysis.

Possible categories include:

- participant-level eligibility exclusion;
- session corruption;
- insufficient primary-outcome exposure;
- predefined trial invalidity;
- analysis-specific signal-quality failure.

No observation should be excluded merely because its value is inconvenient.

## 27. Outliers

Outlier treatment must be defined by the measurement process and analysis goal.

Potential procedures may include:

- predefined physically or procedurally impossible ranges;
- robust summaries;
- sensitivity analyses.

Observations must not be deleted solely because they weaken a statistical
result.

## 28. Reaction-Time Data

Reaction-time analysis should specify:

- valid trial types;
- whether incorrect trials are included;
- omission treatment;
- anticipatory-response definition;
- extreme-response handling;
- participant-level aggregation or trial-level model.

Reaction-time distributions are often asymmetric, so the chosen analysis should
not assume symmetry without evaluation.

## 29. Accuracy Data

Accuracy may be represented as:

```text
correct_trials / valid_trials
```

or as trial-level binary correctness.

The denominator must remain explicit.

Accuracy must be interpreted jointly with task difficulty when difficulty can
adapt.

## 30. Completion-Time Data

Completion time should only be compared directly when task structure makes that
comparison meaningful.

If adaptation changes the number or nature of trials, raw completion time may
not represent the same construct across conditions.

## 31. Subjective Measures

Any subjective scale must be scored according to its defined instrument.

The SAP should record:

- instrument name;
- scoring procedure;
- missing-item rule;
- direction of scoring;
- analysis unit.

Different subjective constructs must not be combined arbitrarily into a single
"engagement score".

## 32. Descriptive Statistics

Before inferential analysis, report appropriate descriptive information.

Potential summaries include:

- participant count;
- observations per condition;
- mean;
- standard deviation;
- median;
- interquartile range;
- minimum/maximum where informative;
- proportions;
- missingness;
- quality eligibility;
- abstention.

Descriptive statistics must correspond to the actual analysis unit.

## 33. Effect Sizes

Statistical analysis should prioritize effect magnitude as well as uncertainty.

Candidate effect-size reporting may include:

- paired mean difference;
- standardized paired difference where useful;
- odds ratio for binary outcomes;
- regression coefficient;
- absolute risk/proportion difference;
- correlation coefficient for exploratory association analyses.

The effect-size definition must match the analysis.

## 34. Confidence Intervals

Where feasible, principal estimates should include confidence or compatibility
intervals.

Intervals should accompany, not be replaced by, p-values.

Confidence intervals must not be interpreted as probability statements about
the fixed parameter unless the statistical framework justifies that
interpretation.

## 35. Statistical Significance

If null-hypothesis significance testing is used, the nominal alpha level must
be specified before confirmatory analysis.

A statistically significant result does not automatically imply:

- practical importance;
- participant benefit;
- engagement improvement;
- clinical relevance.

A non-significant result does not prove that two conditions are identical.

## 36. Multiplicity

Multiple confirmatory tests increase false-positive risk.

The final SAP must define whether there is:

- one primary hypothesis/outcome;
- a hierarchy of confirmatory outcomes;
- a multiplicity-adjusted family.

Secondary and exploratory analyses should be labelled accordingly.

Multiplicity handling should not be chosen after viewing p-values.

## 37. Sample Size

No participant sample size is currently claimed.

The final sample-size justification should use the chosen:

- primary outcome;
- statistical model;
- within-participant design;
- effect-size or precision target;
- variability assumptions;
- attrition allowance where appropriate.

The assumptions and their sources must be documented.

## 38. Precision-Based Planning

If prior effect-size evidence is weak, a precision-based sample-size approach
may be preferable to assuming an unsupported effect.

For example, planning may target a desired width of the confidence interval for
the primary within-participant effect.

The selected method must be justified before recruitment.

## 39. Power Analysis

If formal statistical power is used, document:

- effect size assumed;
- variance/correlation assumptions;
- alpha;
- desired power;
- analysis method;
- attrition allowance;
- source of assumptions.

Synthetic EngageVR results must not be used as if they estimate real
participant effect sizes.

## 40. H1 — Multimodal Fusion Analysis

H1 concerns predictive models.

All compared configurations should use compatible:

- target;
- dataset;
- eligible observations;
- participant/session split;
- preprocessing;
- evaluation folds.

Candidate classification metrics include:

- balanced accuracy;
- macro F1;
- log loss;
- multiclass Brier score;
- calibration metrics.

Candidate regression metrics include:

- MAE;
- RMSE;
- target-appropriate calibration/uncertainty metrics.

The primary comparison metric should be predefined.

## 41. H1 Statistical Dependence

Predictions generated on the same participants/folds are paired.

Model comparison procedures should preserve that pairing rather than treating
all predictions as independent samples.

Participant-level or grouped resampling may be appropriate.

## 42. H2 — Personalization Analysis

H2 compares:

- population prediction;
- personalized prediction.

Both must be evaluated on the same eligible evaluation observations.

Calibration information must precede the evaluation observations.

Candidate quantities include:

```text
delta_metric =
personalized_metric - population_metric
```

with sign interpreted according to the metric.

For error metrics, a lower value may be better, so direction must be stated
explicitly.

## 43. H2 Heterogeneity

Personalization may:

- improve some participants;
- have little effect on others;
- worsen others.

Therefore, report participant-level distributions where sample size and privacy
allow.

An average positive result must not hide substantial negative participant
effects.

## 44. H3 — Selective Prediction Analysis

Selective prediction must report performance jointly with coverage.

For a given threshold:

```text
coverage =
accepted_predictions / eligible_prediction_opportunities
```

The analysis may examine:

- coverage;
- accepted-set performance;
- risk;
- calibration;
- false-confident predictions;
- unnecessary abstentions.

## 45. Risk-Coverage Analysis

A selective predictor can appear more accurate simply by making fewer
predictions.

Therefore, performance must be interpreted across coverage.

Candidate analyses include:

- risk-coverage curve;
- area under the risk-coverage curve where appropriate;
- predefined operating points.

Thresholds used for confirmatory comparison must not be selected solely after
viewing evaluation labels.

## 46. Classification Selectivity

Classification acceptance may use a calibrated confidence threshold.

The SAP should define:

- confidence quantity;
- acceptance threshold;
- target metric;
- coverage.

Model confidence is not certainty.

## 47. Regression Selectivity

Regression acceptance may use prediction-interval width or another appropriate
regression-specific criterion.

Classification confidence must not be substituted for regression uncertainty.

Candidate quantities include:

- empirical interval coverage;
- interval width;
- accepted-set MAE/RMSE;
- prediction coverage.

## 48. H4 — Static Versus Adaptive Analysis

H4 concerns participant outcomes under:

```text
static
```

versus:

```text
adaptive
```

The primary analysis should preserve the within-participant design.

The final model should consider condition and relevant period/order structure.

The analysis must not use synthetic adaptation diagnostics as participant
outcome evidence.

## 49. H4 Directionality

The current H4 is non-directional until the final primary outcome and
confirmatory hypothesis are finalized.

The SAP must determine before analysis whether the confirmatory alternative is:

- directional;
- non-directional.

This decision must not depend on the observed condition difference.

## 50. Adaptation Appropriateness Analysis

Adaptation appropriateness may be evaluated before adaptation benefit.

Possible outcomes include ratings of whether delivered adaptations were:

- appropriately timed;
- disruptive;
- comfortable;
- understandable;
- acceptable.

These analyses should distinguish:

- proposed adaptations;
- dispatched adaptations;
- acknowledged/applied adaptations.

Only actual participant exposure should be used when evaluating participant
response to delivered adaptation.

## 51. H5 — rPPG Robustness Analysis

H5 concerns measurement robustness under motion and illumination conditions.

Candidate factors include:

- rPPG algorithm;
- motion condition;
- illumination condition.

Candidate outcomes include:

- valid-window proportion;
- signal-quality index;
- HR availability;
- HR error against external reference;
- agreement with external reference.

## 52. rPPG Reference Requirement

Webcam rPPG must not be validated against itself.

Where heart-rate accuracy or agreement is assessed, an appropriate external
reference should be used.

Reference-device limitations should also be reported.

## 53. rPPG Repeated Measurements

Multiple rPPG windows from one participant are dependent.

Analysis should account for repeated measurements using:

- participant-level summaries;
- repeated-measures methods;
- mixed-effects modelling;

as appropriate.

Thousands of windows do not create thousands of independent participants.

## 54. Agreement Analysis

Correlation alone is insufficient for evaluating agreement between two
measurement methods.

Where reference agreement is a research objective, consider methods that
evaluate actual measurement differences and agreement, not only association.

The exact agreement method should be selected after the measurement protocol is
finalized.

## 55. Physiological and Subjective Associations

Associations between:

- physiological estimates;
- behavioural features;
- subjective responses;

should generally be treated cautiously and may initially be exploratory.

Correlation does not establish that a physiological feature directly measures
engagement or cognitive load.

## 56. Missing-Modality Analysis

Missing-modality experiments may deliberately remove or mask modalities.

For controlled ablations, compare performance under clearly defined modality
sets.

Report:

- exact modality availability;
- quality gating;
- prediction availability;
- performance;
- uncertainty/abstention.

Naturally missing modalities and experimentally ablated modalities should be
distinguished.

## 57. Temporal Modelling

Temporal modelling remains a later-stage research question.

If evaluated, temporal models and independent-window baselines must use
compatible:

- participant splits;
- targets;
- features;
- evaluation periods.

Temporal overlap must not leak neighbouring information across train/test
boundaries.

## 58. Model Calibration

Where probability calibration is evaluated, report both discrimination and
calibration.

Potential quantities include:

- Brier score;
- log loss;
- reliability analysis;
- expected calibration error where defined.

Calibration metrics should not replace predictive-performance reporting.

## 59. Calibration Data Separation

Calibration procedures must not use final evaluation labels in a way that
invalidates evaluation.

Training, calibration, and test roles must remain explicit.

## 60. Cross-Validation

Cross-validation should respect the relevant grouping structure.

Potential approaches include:

- participant-grouped folds;
- session-grouped folds.

Exact fold design depends on dataset availability.

Random row-level cross-validation is inappropriate when related windows from
one participant/session could appear in both training and validation folds.

## 61. Hyperparameter Selection

Hyperparameters should be selected without using held-out final evaluation data.

The final test set should not become an iterative model-development set.

## 62. Public Dataset Analysis

Every public-dataset result must report:

- dataset identity;
- sample size;
- participant count where known;
- modalities;
- labels;
- grouping strategy;
- license/usage constraints;
- important limitations.

A public-dataset component result must not be presented as validation of the
complete EngageVR system.

## 63. Synthetic Data Analysis

Synthetic data may be analysed for:

- software tests;
- reproducibility;
- pipeline integration;
- deterministic diagnostics.

Synthetic results must remain:

```text
scientific_evaluation_eligible = false
```

They must not contribute to participant-effect estimates or real-world model
validity claims.

## 64. Model Comparison Reporting

For every model comparison, report at minimum:

- model/configuration;
- target;
- data source;
- split;
- sample/observation counts;
- participant/session grouping where available;
- selected metrics;
- uncertainty where appropriate;
- missing/unavailable observations.

## 65. No Best-Result Cherry-Picking

If multiple models or configurations are evaluated, the reporting process must
not present only the best run while hiding the others.

Selection rules should be defined prospectively where possible.

## 66. Sensitivity Analyses

Potential sensitivity analyses may include:

- alternative valid aggregation choices;
- exclusion of protocol deviations;
- complete paired-case analysis;
- alternative robust estimators;
- condition-order adjustment;
- analyses restricted to adequate signal-quality observations.

Sensitivity analysis should evaluate robustness, not create a search for a
favourable result.

## 67. Per-Protocol Analysis

If defined, a per-protocol analysis may exclude sessions with specified major
deviations.

It should generally be secondary to the primary analysis defined by the final
study estimand.

The exact criteria must be predefined.

## 68. Intention-by-Assignment Principle

For a randomized/counterbalanced condition assignment, retaining participants
according to assigned condition structure can protect against post-assignment
selection bias.

The exact applicability of an intention-to-treat-style principle depends on the
final design.

This draft does not claim a finalized clinical-trial estimand.

## 69. Participant Withdrawal

Withdrawal must be handled according to the approved consent/data procedure.

The SAP should document:

- whether existing data remain eligible;
- which outcomes become missing;
- whether deletion occurred;
- which analyses are affected.

Withdrawal must never be converted silently into technical missingness.

## 70. Protocol Amendment

Any material change after data collection begins should be documented.

Examples include:

- primary-outcome change;
- exclusion-rule change;
- model/configuration change;
- threshold change;
- statistical-model change.

The documentation should state:

- what changed;
- why;
- when;
- whether data had already been inspected.

## 71. Analysis Blinding

If feasible, some analysis steps may use coded condition labels.

However, the study must not claim analyst blinding unless it was actually
implemented.

## 72. Software for Analysis

The final analysis environment should record:

- software/package versions;
- analysis code version;
- configuration;
- random seeds where applicable.

The analysis should be reproducible from versioned code and documented inputs.

## 73. Numerical Reproducibility

Exact byte identity across every hardware/software environment is not assumed
for floating-point analysis unless demonstrated.

Meaningful numerical differences must not be hidden merely to force identical
results.

Reproducibility claims should specify their actual scope.

## 74. Statistical Output Provenance

Each result should be traceable to:

- source dataset;
- dataset version/fingerprint where applicable;
- analysis code;
- configuration;
- model version;
- statistical method.

## 75. Tables and Figures

Tables and figures should state:

- analysis population;
- unit of analysis;
- condition/model;
- sample size;
- missingness where relevant;
- whether results are confirmatory or exploratory.

Graphs must not omit inconvenient observations without a documented rule.

## 76. Reporting Negative Results

Negative or null results are valid.

Examples include:

- multimodal fusion does not outperform the best unimodal model;
- personalization worsens performance;
- abstention reduces coverage without useful risk improvement;
- static and adaptive conditions do not differ;
- webcam rPPG becomes unusable under ordinary movement.

Such findings must not be hidden.

## 77. Reporting Unavailable Results

Sometimes an analysis cannot be performed.

Examples include:

- insufficient valid rPPG windows;
- insufficient sample size;
- missing reference device;
- unvalidated target labels.

The correct result may be:

```text
not evaluated
```

rather than a fabricated metric.

## 78. Confirmatory vs Exploratory Reporting

Every final analysis should be labelled as one of:

- confirmatory;
- secondary;
- exploratory;
- software self-check.

These categories must remain visible in reports.

## 79. Primary Participant Analysis Checklist

Before running the confirmatory H4 analysis:

- [ ] institutional requirements satisfied;
- [ ] participant dataset frozen;
- [ ] primary outcome frozen;
- [ ] analysis population defined;
- [ ] exclusion rules frozen;
- [ ] missing-data procedure frozen;
- [ ] statistical model frozen;
- [ ] condition-order handling frozen;
- [ ] alpha/multiplicity procedure frozen where applicable;
- [ ] effect-size reporting defined;
- [ ] confidence-interval procedure defined;
- [ ] software/configuration version recorded.

## 80. Predictive Model Analysis Checklist

Before confirmatory H1-H3 analysis:

- [ ] real scientifically appropriate dataset identified;
- [ ] target provenance documented;
- [ ] participant/session groups available where required;
- [ ] train/calibration/test roles frozen;
- [ ] primary metric defined;
- [ ] compared configurations frozen;
- [ ] hyperparameter-selection procedure defined;
- [ ] missing-modality treatment defined;
- [ ] uncertainty/selectivity threshold procedure defined;
- [ ] participant leakage audit complete.

## 81. rPPG Analysis Checklist

Before confirmatory H5 analysis:

- [ ] rPPG protocol finalized;
- [ ] motion conditions finalized;
- [ ] illumination conditions finalized;
- [ ] algorithms finalized;
- [ ] external reference selected where required;
- [ ] synchronization validated;
- [ ] quality criteria frozen;
- [ ] primary rPPG outcome defined;
- [ ] repeated-measure handling defined.

## 82. Decisions Still Required

Before this SAP becomes final, resolve:

1. participant population;
2. sample size;
3. final task;
4. final static/adaptive protocol;
5. primary H4 outcome;
6. final H4 statistical model;
7. secondary outcomes;
8. subjective instruments;
9. primary H1 metric;
10. primary H2 metric;
11. primary H3/selective-prediction operating comparison;
12. primary H5 outcome;
13. rPPG reference device if required;
14. exclusion criteria;
15. missing-data procedure;
16. multiplicity strategy;
17. condition-order analysis;
18. sensitivity analyses;
19. software analysis environment;
20. reporting format.

## 83. Current Status

This SAP is a prospective Milestone 11 planning document.

It does not claim:

- a finalized primary outcome;
- a justified participant sample size;
- a preregistered analysis;
- an approved participant protocol;
- participant data;
- statistically significant findings;
- scientific validation of EngageVR models;
- adaptation benefit.

The SAP must be updated and frozen before confirmatory analysis of future
participant data.
