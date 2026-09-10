# EngageVR Experimental Variables

> **Draft for research planning and institutional review**
>
> This document defines candidate experimental variables for future EngageVR
> studies. No participant study described here has been conducted, and no
> institutional or ethical approval is claimed.
>
> Variables, primary outcomes, exclusion rules, and analysis decisions must be
> finalized before confirmatory participant data are collected or analysed.

## 1. Purpose

This document defines the variables required to evaluate the research questions
and hypotheses described in:

- `docs/RESEARCH_QUESTIONS.md`;
- `docs/RESEARCH_PROPOSAL.md`;
- `docs/HYPOTHESES.md`.

The purpose is to prevent ambiguity between:

- experimentally manipulated variables;
- participant outcomes;
- model outputs;
- sensing-quality variables;
- adaptation-controller state;
- control variables;
- covariates;
- provenance fields;
- exclusion or availability flags.

A software field is not automatically an experimental variable merely because
it exists in an EngageVR artifact.

Likewise, a model estimate must not be treated as ground truth for the mental
state it attempts to estimate.

## 2. Variable Families

EngageVR uses several conceptually different classes of variables.

### 2.1 Experimental variables

Variables deliberately manipulated or assigned as part of a study.

Examples:

- static versus adaptive experimental condition;
- condition order;
- controlled illumination condition in an rPPG validation experiment;
- controlled motion condition in an rPPG validation experiment.

### 2.2 Participant outcome variables

Measurements used to evaluate differences between experimental conditions.

Examples:

- task accuracy;
- reaction time;
- completion time;
- participant-reported engagement;
- perceived cognitive load;
- adaptation acceptance.

### 2.3 Model-output variables

Outputs produced by EngageVR's predictive pipeline.

Examples:

- engagement estimate;
- cognitive-load estimate;
- predicted class;
- classification confidence;
- regression prediction interval;
- abstention status.

These are model outputs, not direct measurements of a participant's internal
state.

### 2.4 Measurement-quality variables

Variables describing whether a sensing channel is usable.

Examples:

- rPPG signal quality;
- face-tracking stability;
- missing-feature fraction;
- illumination diagnostics;
- motion diagnostics.

Measurement quality must remain separate from engagement, cognitive load, and
predictive uncertainty.

### 2.5 Controller variables

Variables representing the state and behaviour of the adaptation policy.

Examples:

- current task difficulty;
- proposed difficulty;
- HOLD reason;
- persistence counter;
- cooldown state;
- adaptation count;
- command lifecycle state.

These describe controller behaviour and must not be interpreted directly as
participant benefit.

### 2.6 Provenance and grouping variables

Variables required to identify the origin and dependency structure of
observations.

Examples:

- participant pseudonym;
- session identifier;
- experimental condition;
- trial identifier;
- feature-window identifier;
- timestamp;
- data source;
- synthetic/scientific-eligibility state.

These variables are essential for leakage prevention and statistical grouping.

## 3. Primary Experimental Condition

### 3.1 Variable

**Name:** experimental condition

**Candidate values:**

- `static`;
- `adaptive`.

### 3.2 Static condition

In the static condition, the task environment does not alter task difficulty
in response to EngageVR's model estimates.

Sensing, recording, model inference, uncertainty computation, and observational
logging may still occur if the final protocol requires them.

The defining property of the static condition is therefore the absence of
model-driven environment adaptation, not necessarily the absence of sensing or
inference.

### 3.3 Adaptive condition

In the adaptive condition, the approved EngageVR adaptation policy may propose
a task-difficulty change when its predefined evidence requirements are
satisfied.

The current implemented policy is centered on the existing
`set_difficulty` command.

The adaptive condition does not imply that an adaptation must occur during
every session.

A participant may complete an adaptive-condition session with few or no
adaptations if:

- predictions abstain;
- signal evidence is insufficient;
- persistence requirements are not met;
- cooldown prevents a change;
- the current state lies in a policy deadband;
- a bound or budget prevents a change.

### 3.4 Important distinction: experimental condition vs experimenter lock

The experimental condition and the adaptation enable/disable safety control are
different variables.

Conceptually:

- `experiment_mode = static/adaptive` defines the intended study condition;
- `adaptation.enabled = true/false` is an experimenter safety/control lock.

An adaptive-condition session in which the experimenter disables adaptation
must not silently be analysed as an ordinary adaptive exposure.

Any such event must be recorded and addressed by the predefined analysis
protocol.

## 4. Condition Order

If the future study uses a within-participant design, condition order becomes an
experimental design variable.

Candidate sequences include:

- static → adaptive;
- adaptive → static.

Condition order should be recorded explicitly.

Potential order-related effects include:

- task learning;
- fatigue;
- familiarity;
- expectation;
- carryover from previous difficulty levels;
- habituation to the sensing setup.

Counterbalancing or another predefined order-control method should be used
where the final design permits.

Condition order may be used as:

- a design factor;
- a covariate;
- a sensitivity-analysis variable.

The exact treatment must be defined before confirmatory analysis.

## 5. Candidate Primary Outcome Variables

A future confirmatory static-versus-adaptive study must designate a primary
outcome before analysing participant data.

The project currently defines several candidates but has not selected one.

### 5.1 Task accuracy

Possible representation:

- proportion of correct responses;
- count of correct responses with an explicit denominator;
- trial-level correct/incorrect outcome.

Accuracy must be interpreted with task difficulty.

An increase in accuracy caused simply by exposure to easier trials is not
automatically evidence of improved engagement.

### 5.2 Reaction time

Possible representation:

- response latency per valid trial;
- participant-condition aggregate such as median reaction time.

Reaction-time analysis should define:

- valid response window;
- treatment of omissions;
- treatment of anticipatory responses;
- treatment of extreme values;
- whether incorrect trials are included.

These choices must be made before confirmatory analysis.

### 5.3 Completion time

Completion time may be relevant when the task has meaningful bounded episodes.

It may be inappropriate as a primary outcome if adaptive difficulty changes the
number, duration, or structure of trials.

### 5.4 Primary-outcome rule

Only one primary outcome, or a clearly predefined primary outcome family,
should be selected for a confirmatory static-versus-adaptive comparison unless
a multiplicity strategy is specified.

Other outcomes should be clearly labelled secondary or exploratory.

## 6. Candidate Subjective Outcome Variables

Participant-reported measures may complement performance and physiological
signals.

### 6.1 Subjective engagement

A future protocol may collect a participant-reported engagement measure.

The exact questionnaire or rating instrument must be chosen before deployment
and reviewed for:

- construct validity;
- appropriateness for the task;
- licensing or usage requirements;
- administration timing;
- scoring rules.

The current software does not establish a validated subjective-engagement
instrument.

### 6.2 Perceived cognitive load or mental effort

A future study may measure perceived cognitive load, workload, or mental effort.

The chosen construct must be named precisely.

Different questionnaires or rating scales must not be treated as
interchangeable merely because they are related to workload.

### 6.3 Adaptation acceptance

Potential measures may include participant ratings of whether an adaptation was:

- noticeable;
- appropriate;
- helpful;
- disruptive;
- comfortable.

The final instrument and scoring scheme remain to be defined.

### 6.4 Comfort and disruption

Potential supporting outcomes include:

- visual comfort;
- task frustration;
- perceived disruption;
- discomfort associated with difficulty changes.

These should not be merged into a single score unless the instrument defines
such a construct.

## 7. Task-Performance Variables

Potential task-performance variables include:

- trial correctness;
- reaction time;
- error count;
- error rate;
- completion time;
- hint usage where implemented;
- inactivity duration;
- retry count where implemented;
- current task difficulty.

All derived metrics must preserve sufficient provenance to identify:

- participant;
- session;
- condition;
- trial;
- relevant timestamp or feature window.

## 8. Engagement and Cognitive-Load Model Outputs

### 8.1 Engagement estimate

The engagement estimate is a model output.

It must not be labelled as:

- true engagement;
- measured engagement;
- psychological diagnosis;
- direct ground truth.

### 8.2 Cognitive-load estimate

The cognitive-load estimate is also a model output.

It must remain distinct from:

- task difficulty;
- subjective workload;
- performance deterioration;
- physiological signal quality.

### 8.3 Classification outputs

Where classification is used, variables may include:

- predicted class;
- calibrated class probabilities;
- maximum calibrated class probability;
- prediction accepted/abstained state.

### 8.4 Regression outputs

Where regression is used, variables may include:

- point prediction;
- lower prediction-interval bound;
- upper prediction-interval bound;
- interval width;
- prediction accepted/unavailable state.

Classification confidence and regression interval width must not be placed on a
shared pseudo-confidence scale unless a future method scientifically justifies
such a transformation.

## 9. Predictive-Uncertainty Variables

### 9.1 Classification confidence

For calibrated classification, confidence is conceptually:

using the calibrated class probabilities produced by the relevant model.

It is not certainty and is not signal quality.

### 9.2 Regression uncertainty

Regression uncertainty is represented using prediction intervals and their
width where applicable.

A wider interval represents different semantics from a lower classification
confidence.

### 9.3 Expert disagreement

Disagreement among modality-specific or ensemble predictions may be recorded as
a diagnostic variable.

Expert disagreement is not automatically calibrated predictive uncertainty.

### 9.4 Abstention

Candidate variables include:

- `abstain`;
- abstention reason;
- accepted prediction count;
- unavailable prediction count;
- prediction coverage.

Abstention must remain observable in analysis rather than being removed as
missing data without explanation.

## 10. Signal-Quality Variables

Signal quality asks whether a measurement is sufficiently usable.

It does not answer whether the participant is engaged.

Candidate quality variables may include:

### 10.1 Webcam and face quality

- face detected;
- face-detection stability;
- brightness;
- blur;
- motion;
- exposure-related diagnostics;
- missing behavioural features.

### 10.2 rPPG quality

Potential variables include:

- rPPG signal-quality index;
- valid-window state;
- peak-detection confidence where defined;
- motion diagnostic;
- illumination diagnostic;
- missing-signal percentage.

A low-quality rPPG window should generally become unavailable for physiological
analysis rather than being assigned an artificial physiological value.

## 11. Missing-Modality Variables

Missing modalities are meaningful system states.

Potential variables include:

- modality available/unavailable;
- missing-modality mask;
- reason for unavailability;
- number of contributing modalities;
- quality-gated modality exclusion.

Examples include:

- face absent;
- rPPG quality insufficient;
- task telemetry unavailable;
- required feature missing.

Missing data must not be silently converted to zero.

For fusion experiments, the missing-modality state may also become an
experimental or ablation factor.

## 12. Adaptation Variables

### 12.1 Current difficulty

Current task difficulty is a time-varying task-state variable.

It is not the primary experimental condition.

### 12.2 Proposed difficulty

The adaptation policy may propose a new difficulty level.

A proposal is not equivalent to:

- command dispatch;
- acknowledgement;
- application.

### 12.3 Adaptation direction

Possible policy outcomes include conceptually:

- increase;
- decrease;
- HOLD.

The exact representation should follow the existing adaptation schema.

### 12.4 HOLD reason

A HOLD is an explicit policy outcome.

Potential reasons include states such as:

- insufficient evidence;
- prediction abstained;
- gate blocked;
- insufficient persistence;
- cooldown active;
- target in deadband;
- direction conflict;
- difficulty bound reached;
- session adaptation budget exhausted.

The repository schema remains the canonical source for exact reason-code
values.

### 12.5 Adaptation lifecycle

The analysis must preserve the distinction between:

- proposed;
- command built;
- dispatched;
- acknowledged;
- applied;
- rejected where applicable.

Controller diagnostics must not infer later lifecycle states that were never
observed.

### 12.6 Adaptation count

Potential derived variables include:

- proposals per session;
- commands built per session;
- dispatched commands per session;
- acknowledged commands per session;
- applied adaptations per session.

None is a direct measure of adaptation effectiveness.

### 12.7 Time since previous adaptation

A time- or window-based spacing variable may be useful for evaluating:

- cooldown behaviour;
- adaptation density;
- participant exposure.

The definition must reflect the actual policy's window semantics.

## 13. Personalization Variables

Candidate variables include:

- personalization applied;
- cold-start state;
- number of calibration observations;
- participant baseline statistics;
- population prediction;
- personalized prediction;
- population-to-personalized prediction delta.

A prediction delta is not automatically an improvement.

Improvement requires comparison against an independently observed target in an
appropriate evaluation period.

## 14. Participant and Session Variables

Human-subject work should use pseudonymous identifiers.

Potential grouping variables include:

- participant pseudonym;
- session identifier;
- condition;
- condition order;
- trial identifier;
- feature-window identifier.

Direct identifiers such as participant name or email address must not enter the
modelling dataset unless a separately approved administrative process requires
them, in which case they should be kept outside the analysis dataset.

## 15. Time Variables

EngageVR measurements are time-dependent.

Relevant variables may include:

- event timestamp;
- trial start/end;
- response timestamp;
- feature-window start/end;
- adaptation proposal time;
- command time;
- acknowledgement time;
- subjective-response time.

A common time basis is necessary for multimodal synchronization.

Time variables used only for operational provenance must not accidentally enter
predictive features unless explicitly justified.

## 16. Candidate Control Variables

The following may need to be controlled or recorded in a participant study.

### 16.1 Environment

- ambient illumination;
- major illumination variation;
- webcam position;
- approximate participant-camera distance where relevant;
- monitor configuration;
- physical testing environment.

### 16.2 Task procedure

- initial difficulty;
- task duration;
- trial timing;
- task content;
- instructions;
- break schedule;
- condition order.

### 16.3 Sensing procedure

- camera device;
- camera resolution;
- target frame rate;
- rPPG method;
- preprocessing configuration;
- baseline/calibration duration.

### 16.4 Model configuration

Where the research question is not explicitly about model configuration, the
following should normally remain fixed within the relevant comparison:

- feature schema;
- trained model version;
- calibration method;
- uncertainty thresholds;
- adaptation-policy configuration.

Changing these during a study can confound experimental condition with software
version.

## 17. Candidate Covariates and Moderators

Depending on the approved study and sample size, candidate explanatory
variables may include:

- baseline task performance;
- baseline subjective rating;
- condition order;
- calibration-data quantity;
- signal-quality summary;
- adaptation exposure count.

Potential participant characteristics must be collected only when scientifically
justified and approved.

A variable should not be collected merely because it might become useful later.

## 18. rPPG Validation Variables

RQ5/H5 may require a separate component-validation experiment rather than the
main static-versus-adaptive study.

### 18.1 Candidate independent variables

Possible controlled factors include:

- motion condition;
- illumination condition;
- rPPG algorithm.

### 18.2 Motion condition

A future protocol may define levels such as:

- low-motion reference condition;
- controlled head movement;
- task-natural movement.

Exact motion conditions must be operationally defined before data collection.

### 18.3 Illumination condition

A future protocol may define controlled illumination levels or changes.

Exact intensity, source, placement, and variation should be measured where
possible rather than described only as "good" or "bad" lighting.

### 18.4 rPPG method

Candidate methods currently represented in EngageVR include:

- GREEN;
- CHROM;
- POS.

### 18.5 Candidate dependent variables

Potential outcomes include:

- signal-quality index;
- valid-window proportion;
- estimated-HR availability;
- HR error against an external reference;
- agreement with an appropriate reference signal.

Webcam rPPG must not serve as its own validation reference.

## 19. Derived Variables

Any derived variable must have a documented definition.

Examples may include:

```text
accuracy = correct_trials / valid_trials
```

```text
error_rate = incorrect_trials / valid_trials
```

```text
prediction_coverage =
accepted_predictions / eligible_prediction_windows
```

Possible participant-condition summaries may include:

- median reaction time;
- mean or median task accuracy across blocks;
- abstention proportion;
- valid-rPPG proportion;
- adaptation count.

The aggregation unit must be defined before analysis.

A row-level quantity and a participant-level aggregate are different variables
for statistical purposes.

## 20. Availability and Exclusion Variables

A future analysis dataset should distinguish:

- missing because not collected;
- unavailable because quality requirements failed;
- excluded by a predefined rule;
- participant withdrawal;
- technical failure;
- model abstention.

These states must not be collapsed into one generic null category if their
causes are known.

Potential flags include:

- measurement available;
- quality eligible;
- prediction available;
- trial valid;
- analysis eligible;
- exclusion reason.

## 21. Scientific Eligibility and Provenance

Every dataset or result used in analysis must retain provenance.

Relevant states include:

- synthetic;
- public dataset;
- future live/participant source;
- mixed where explicitly justified.

Synthetic observations are permanently ineligible as human-subject scientific
evidence.

Public data may support component-level scientific evaluation only to the
extent permitted by:

- its labels;
- modalities;
- protocol;
- population;
- license;
- measurement quality.

A public dataset must not be presented as a validation of the complete EngageVR
system merely because one component can be evaluated with it.

## 22. Variable-to-Hypothesis Mapping

| Hypothesis | Main independent/comparison variable | Candidate outcome family |
|---|---|---|
| H1 — Multimodal fusion | Modality/model configuration | Predictive performance and calibration |
| H2 — Personalization | Population vs personalized prediction | Predictive performance and calibration |
| H3 — Selective prediction | Acceptance/selectivity threshold | Coverage and accepted-set performance |
| H4 — Adaptive vs static | Experimental condition | Task and participant-reported outcomes |
| H5 — rPPG robustness | Motion / illumination / method | Signal quality, availability, reference agreement |

## 23. Variables That Must Not Be Conflated

The following distinctions are mandatory:

| Concept A | Concept B | Reason |
|---|---|---|
| Signal quality | Engagement | Poor measurement is not low engagement |
| Signal quality | Model confidence | Measurement usability and predictive uncertainty differ |
| Classification confidence | Certainty | A calibrated probability is not certainty |
| Regression interval width | Classification confidence | They have different semantics and units |
| Expert disagreement | Calibrated uncertainty | Disagreement is a diagnostic unless calibrated |
| Task difficulty | Cognitive load | Difficulty is an experimental/task state, not measured load |
| Model estimate | Ground truth | Predictions are not direct measurements of mental state |
| Personalization delta | Personalization benefit | A changed prediction is not necessarily a better prediction |
| Adaptation proposal | Applied adaptation | The command lifecycle contains distinct states |
| Adaptation count | Adaptation benefit | Frequency does not establish effectiveness |
| Static/adaptive condition | Experimenter enable lock | Experimental assignment and safety control are distinct |
| Synthetic performance | Human performance | Synthetic software checks are not participant evidence |

## 24. Decisions Required Before Participant Data Collection

Before a confirmatory study begins, the following must be finalized:

1. final experimental design;
2. primary outcome;
3. secondary outcomes;
4. subjective instruments;
5. operational definition of each condition;
6. condition-order procedure;
7. calibration procedure;
8. adaptation configuration;
9. exclusion criteria;
10. missing-data procedure;
11. analysis unit;
12. statistical model or test;
13. multiplicity strategy where required;
14. sample-size justification;
15. participant inclusion/exclusion criteria;
16. data-retention and privacy procedure.

These decisions must not be retrospectively chosen from the results.

## 25. Current Status

This document defines candidate variables for future research.

It does not assert that:

- the candidate primary outcomes are already finalized;
- the proposed within-participant design has been institutionally approved;
- subjective instruments have been selected;
- participant data exist;
- rPPG has been validated against a reference device;
- engagement or cognitive-load estimates have been scientifically validated;
- adaptation has demonstrated participant benefit.

Those questions remain part of Milestone 11 planning and future approved
research.
