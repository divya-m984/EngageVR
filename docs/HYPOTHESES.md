# EngageVR Research Hypotheses

> **Draft for research planning and institutional review**
>
> These hypotheses define prospective scientific questions for future
> evaluation of EngageVR. They have not been confirmed by the current software
> implementation, synthetic demonstrations, or engineering test suite.
>
> No human-subject study is claimed, and any participant-based hypothesis must
> be evaluated only after appropriate institutional review and approval.

## 1. Purpose

This document formalizes the hypotheses associated with the research questions
in `docs/RESEARCH_QUESTIONS.md` and the study framework described in
`docs/RESEARCH_PROPOSAL.md`.

The hypotheses are intentionally separated from software acceptance criteria.

A passing software test can establish that an algorithm, controller, schema, or
pipeline behaves as implemented. It cannot establish that:

- an engagement estimate is valid for a person;
- a cognitive-load estimate is valid for a person;
- personalization benefits a participant;
- an adaptation is appropriate;
- an adaptive environment improves an outcome.

Synthetic data may be used to verify analysis code and experimental plumbing,
but synthetic results are not evidence for or against these scientific
hypotheses.

## 2. Hypothesis Categories

The proposed research programme contains three broad hypothesis categories:

1. **model-level hypotheses**, concerning fusion, personalization, and
   selective prediction;
2. **measurement-level hypotheses**, concerning webcam-rPPG robustness;
3. **human-study hypotheses**, concerning adaptation appropriateness and
   static-versus-adaptive experimental conditions.

These categories may require different datasets and experimental procedures and
must not be treated as though one dataset can answer every research question.

## 3. H1 — Multimodal Fusion

### Research question

Does multimodal fusion provide better engagement and cognitive-load estimation
than individual modalities?

### Alternative hypothesis H1A

For a predefined prediction target and evaluation protocol, at least one
multimodal configuration will demonstrate better predictive performance and/or
calibration than the strongest corresponding single-modality baseline.

### Null hypothesis H1₀

Under the same evaluation protocol, multimodal configurations will not provide
a meaningful improvement over the strongest corresponding single-modality
baseline.

### Planned comparison

Candidate configurations include:

- behavioural-only;
- head-pose or movement features where analysed separately;
- rPPG/physiological-only where scientifically valid;
- task-performance-only;
- early feature concatenation;
- late multimodal fusion;
- quality-aware late fusion;
- validation-weighted fusion;
- other already-defined fusion configurations where applicable.

All configurations used in a comparison must use compatible targets, splits,
evaluation data, and leakage controls.

### Candidate outcomes

Depending on the task:

- balanced accuracy;
- macro F1;
- log loss;
- multiclass Brier score;
- calibration error;
- MAE;
- RMSE;
- prediction availability;
- other predefined target-appropriate metrics.

No single metric should be retrospectively selected solely because it makes one
method appear superior.

### Evidence required

This hypothesis cannot be established using the current synthetic software
checks alone.

Scientific evaluation requires suitable real data with valid target labels and
appropriate participant/session separation.

## 4. H2 — Personalized Calibration

### Research question

Does participant-specific calibration improve prediction relative to an
unmodified population model?

### Alternative hypothesis H2A

Participant-specific calibration will produce measurable differences in
prediction performance and/or calibration relative to the corresponding
population prediction.

Where a directional improvement hypothesis is used in a future confirmatory
study, the direction and primary metric must be defined before evaluation.

### Null hypothesis H2₀

Participant-specific calibration will not produce a meaningful difference from
the corresponding population prediction under the predefined evaluation
protocol.

### Planned comparison

The comparison should preserve both:

- the original population prediction; and
- the personalized prediction.

Potential approaches already represented in EngageVR include:

- participant baseline normalization;
- population model plus participant-specific correction;
- cold-start population inference where calibration evidence is insufficient.

### Temporal constraint

Personalization must use only information available before the evaluation
window.

Future participant data must preserve a clear chronological boundary between:

1. calibration observations; and
2. evaluation observations.

Using later evaluation observations to construct a participant baseline would
constitute leakage.

### Important interpretation rule

A personalized prediction is not assumed to be superior merely because it is
personalized.

Negative, null, and participant-specific heterogeneous effects are valid
outcomes.

## 5. H3 — Selective Prediction and Abstention

### Research question

Can uncertainty-aware selective prediction reduce prediction error among cases
for which EngageVR elects to produce an estimate?

### Alternative hypothesis H3A

Increasing selectivity by rejecting sufficiently uncertain predictions will be
associated with lower prediction error among accepted predictions, while
reducing prediction coverage.

### Null hypothesis H3₀

Selective prediction will not provide a meaningful accepted-prediction
performance advantage relative to the corresponding non-selective prediction
strategy.

### Classification evaluation

Classification analysis may consider:

- calibrated class probabilities;
- confidence thresholds;
- accepted-prediction performance;
- coverage;
- calibration;
- risk-coverage behaviour;
- false-confident predictions;
- unnecessary abstentions.

### Regression evaluation

Regression analysis must use regression-specific uncertainty semantics, such
as:

- prediction intervals;
- interval width;
- empirical interval coverage;
- prediction error;
- acceptance based on a predefined maximum interval width.

Regression uncertainty must not be transformed into classification confidence
merely to create a shared scale.

### Interpretation

Higher selectivity necessarily reduces the number of predictions available for
downstream use.

Therefore, an observed increase in accepted-set performance must always be
reported together with its corresponding coverage.

## 6. H4 — Adaptive Versus Static Condition

### Research question

Does the EngageVR adaptive condition produce measurable differences in
participant outcomes relative to an otherwise comparable static condition?

### Status

This is a future human-subject hypothesis.

It cannot be evaluated until:

- an appropriate participant protocol exists;
- institutional review requirements are satisfied;
- the task environment has been validated for experimental use;
- the relevant sensing and model outputs have sufficient scientific support;
- the adaptation path intended for the study has been technically validated.

### Alternative hypothesis H4A

One or more predefined participant outcomes will differ between the static and
adaptive conditions.

### Null hypothesis H4₀

The predefined primary participant outcome will not differ meaningfully between
the static and adaptive conditions.

### Candidate outcomes

Potential outcomes include:

- task accuracy;
- reaction time;
- completion time;
- subjective engagement;
- perceived cognitive load or mental effort;
- adaptation acceptance;
- comfort or disruption.

A future statistical-analysis plan must designate the primary outcome before
participant data are analysed.

### Initial adaptation scope

The first controlled comparison should be based on the adaptation mechanism
that is actually implemented and technically validated.

At present, EngageVR's conservative policy is centered on task-difficulty
adaptation through the existing `set_difficulty` command.

Broader adaptations such as automated breaks, audiovisual-intensity changes,
information-density changes, or other unimplemented policies must not be
silently included in the hypothesis as though they were existing experimental
conditions.

## 7. H5 — Webcam-rPPG Robustness

### Research question

How robust is webcam-based rPPG under motion and illumination variation relevant
to the intended use environment?

### Alternative hypothesis H5A

Increasing motion and/or adverse illumination will be associated with
measurable degradation in rPPG signal quality, availability, and/or agreement
with an appropriate reference measurement.

### Null hypothesis H5₀

Within the predefined experimental range, motion and illumination condition
will not produce a meaningful difference in the selected rPPG quality or
reference-agreement outcomes.

### Candidate comparisons

Evaluation may compare:

- green-channel baseline;
- CHROM;
- POS.

Candidate conditions may vary:

- participant/head movement;
- facial tracking stability;
- illumination level;
- illumination variability.

### Candidate outcomes

Potential measurement outcomes include:

- signal-quality index;
- valid-window proportion;
- estimated heart-rate availability;
- heart-rate error against an appropriate reference;
- agreement statistics where a suitable reference signal exists.

A webcam-rPPG estimate must not be treated as its own ground truth.

## 8. Adaptation Appropriateness Before Adaptation Benefit

Before interpreting a static-versus-adaptive comparison as a test of benefit,
the proposed adaptation mechanism should first be evaluated for basic
appropriateness and acceptability.

A preliminary human-study question may therefore ask:

> When an adaptation is proposed and delivered under the approved protocol, do
> participants and experimenters judge the adaptation to be acceptable and
> non-disruptive under the predefined criteria?

This is intentionally distinct from claiming that adaptation improves
engagement or performance.

A system can be technically functional without its adaptations being useful,
appropriate, comfortable, or beneficial.

## 9. Exploratory Research Questions

The following existing research questions are currently better treated as
exploratory or later-stage investigations rather than confirmatory hypotheses
for the first EngageVR study:

- agreement between physiological estimates and subjective reports;
- prediction behaviour under missing modalities;
- temporal modelling versus independent-window modelling;
- comparison among multiple adaptation strategies;
- sufficient-evidence detection beyond the primary selective-prediction
  analysis.

These may be promoted to confirmatory hypotheses only when an appropriate
dataset, study design, outcome definition, and analysis plan exist.

## 10. General Hypothesis-Testing Rules

Future evaluation must follow these rules.

### 10.1 Predefinition

Before confirmatory analysis, define:

- the hypothesis being tested;
- primary outcome;
- comparison groups or conditions;
- statistical model or test;
- effect-size measure;
- exclusion rules;
- missing-data procedure;
- multiplicity procedure where applicable.

### 10.2 Effect sizes and uncertainty

Results should not be reported using statistical significance alone.

Where appropriate, report:

- effect sizes;
- confidence or compatibility intervals;
- raw or descriptive distributions;
- sample size;
- missing/excluded observations.

### 10.3 Negative and null findings

Failure to demonstrate improvement is a valid result.

The analysis must not:

- discard negative personalization results;
- hide increased abstention;
- suppress poor signal quality;
- redefine the primary metric after seeing results;
- convert unavailable measurements into zero;
- reinterpret null results as evidence of benefit.

### 10.4 Provenance

Every reported result must state the source of its data.

At minimum, distinguish:

- synthetic software checks;
- public datasets;
- future participant data;
- future laboratory reference measurements.

Synthetic results must remain scientifically ineligible.

## 11. Hypothesis-to-Research-Question Mapping

| Hypothesis | Primary research question | Evidence class |
|---|---|---|
| H1 | RQ1 — Multimodal fusion | Real labelled dataset / future participant data |
| H2 | RQ2 — Personalization | Participant-labelled longitudinal/calibration data |
| H3 | RQ3 / RQ10 — Uncertainty and sufficient evidence | Real labelled evaluation data |
| H4 | RQ4 — Adaptive versus static environment | Approved controlled participant study |
| H5 | RQ5 — Webcam-rPPG robustness | Public/reference dataset and/or controlled hardware study |

## 12. Current Status

None of H1-H5 has been established scientifically by the current repository.

Existing synthetic experiments and deterministic scenarios may verify:

- implementation correctness;
- reproducibility;
- data contracts;
- model-training plumbing;
- abstention logic;
- adaptation-policy behaviour.

They do not establish the corresponding scientific hypotheses.

The hypotheses remain prospective until evaluated using data and experimental
procedures appropriate to each question.
