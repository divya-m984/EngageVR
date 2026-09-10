# EngageVR Future Laboratory Extension Plan

> **Prospective research and engineering roadmap**
>
> This document describes possible future laboratory extensions for EngageVR.
>
> It is not an approved participant protocol, funding commitment, procurement
> plan, deployment authorization, ethics approval, or claim that the listed
> hardware and research capabilities currently exist.
>
> Human-subject research must proceed only through the appropriate
> institutional review and approved study procedures.

## 1. Purpose

EngageVR currently provides a software research framework for:

- webcam-derived behavioural features;
- webcam remote photoplethysmography (rPPG);
- task telemetry;
- feature-dataset construction;
- interpretable baseline models;
- multimodal fusion;
- personalization;
- uncertainty-aware inference;
- selective prediction and abstention;
- conservative task adaptation;
- research dashboards;
- reproducibility and MLOps infrastructure.

The next scientific stage requires moving gradually from software-only
verification toward controlled physical-hardware and, eventually, approved
participant research.

This document defines that future progression.

It should be read alongside:

- `docs/RESEARCH_PROPOSAL.md`;
- `docs/EXPERIMENT_DESIGN.md`;
- `docs/DATA_COLLECTION_PROTOCOL.md`;
- `docs/ETHICS_AND_PRIVACY.md`;
- `docs/RISK_ASSESSMENT.md`;
- `docs/STATISTICAL_ANALYSIS_PLAN.md`;
- `docs/HARDWARE_VALIDATION_PLAN.md`;
- `docs/DATASET_CARDS.md`;
- `docs/MODEL_CARDS.md`;
- `docs/LIMITATIONS.md`.

## 2. Current Boundary

At the current project state:

- no EngageVR participant-labelled dataset exists;
- no EngageVR human-subject experiment has been conducted;
- no institutional or ethical approval is claimed;
- physical webcam validation remains pending;
- real UBFC-rPPG evaluation remains pending;
- rPPG has not been compared against a locally acquired physiological
  reference;
- Unity has not been validated as the participant-facing runtime;
- no VR headset configuration has been validated;
- no participant-facing live adaptation study has been conducted;
- engagement and cognitive-load models remain scientifically unvalidated on
  real participant-labelled EngageVR data;
- no champion predictive model or fusion strategy exists.

The laboratory extension must preserve these boundaries until new evidence
actually changes them.

## 3. Laboratory Extension Principle

The project should advance through evidence gates rather than jumping directly
from synthetic software tests to a full adaptive participant study.

Conceptually:

```text
software verification
        |
        v
physical hardware bench validation
        |
        v
public/reference component validation
        |
        v
integrated laboratory dry runs
        |
        v
approved pilot participant work
        |
        v
scientific model/component evaluation
        |
        v
approved static-versus-adaptive study
        |
        v
replication / extension
```

A later stage should not be claimed because an earlier stage merely exists.

## 4. Proposed Extension Phases

The future programme is divided into:

1. Phase A — laboratory infrastructure;
2. Phase B — physical webcam validation;
3. Phase C — rPPG component validation;
4. Phase D — task and Unity runtime validation;
5. Phase E — integrated non-participant dry runs;
6. Phase F — approved participant pilot;
7. Phase G — participant-labelled model validation;
8. Phase H — static-versus-adaptive experiment;
9. Phase I — replication and advanced extensions.

The phase labels are planning constructs, not current project milestones.

## 5. Phase A — Laboratory Infrastructure

### Objective

Create a controlled environment in which hardware and participant-facing
software can be evaluated reproducibly.

### Candidate infrastructure

The laboratory may eventually require:

- participant workstation;
- validated webcam;
- controlled display;
- keyboard/controller;
- illumination measurement or control;
- physiological reference device;
- optional VR headset;
- secure research-data storage;
- experimenter monitoring interface.

Exact equipment remains to be selected.

## 6. Participant Workstation

The participant-facing workstation should have sufficient resources for:

- webcam capture;
- behavioural feature extraction;
- rPPG processing;
- task runtime;
- model inference;
- recording;
- optional Unity/VR rendering.

The final system should record relevant:

- CPU;
- GPU;
- RAM;
- operating system;
- software versions;
- device drivers;
- display configuration.

Hardware identity should be retained in study provenance where relevant.

## 7. Experimenter Workstation

A separate experimenter machine may be beneficial where the final laboratory
workflow requires:

- participant-session control;
- dashboard observation;
- recording verification;
- adaptation disablement;
- monitoring without obstructing the participant display.

Whether a second machine is required remains to be determined.

## 8. Network Architecture

EngageVR is currently local-first.

A future laboratory architecture should minimize network exposure.

Candidate designs include:

```text
single-machine local deployment
```

or:

```text
participant machine
        |
        | controlled local connection
        v
experimenter machine
```

No internet dependency should be introduced unless scientifically or
operationally necessary.

## 9. Security Boundary

A laboratory network does not automatically make an unauthenticated research
service safe.

Before any multi-machine deployment, evaluate:

- service binding;
- authentication where required;
- authorization;
- firewall rules;
- transport protection;
- participant-data paths;
- exposed artifact directories.

Development convenience must not define the security architecture.

## 10. Research Data Storage

Participant data should be stored outside ordinary source-control directories.

The final storage architecture must define:

- primary storage;
- filesystem permissions;
- authorized users;
- encryption where required;
- backups;
- retention;
- deletion;
- incident procedure.

Participant data must not be committed to Git.

## 11. Phase B — Physical Webcam Validation

### Objective

Move the webcam pipeline from synthetic/fixture verification to real device
operation.

### Candidate work

Validate:

- physical camera discovery;
- sustained capture;
- observed frame rate;
- timestamps;
- face detection;
- head pose;
- behavioural features;
- quality diagnostics;
- failure states.

This phase should follow `docs/HARDWARE_VALIDATION_PLAN.md`.

## 12. Webcam Selection

The final participant webcam should be selected prospectively.

Selection considerations include:

- resolution;
- frame rate;
- exposure control;
- low-light performance;
- driver/backend support;
- mounting stability;
- reproducibility between machines.

A webcam should not be selected solely because it is already available.

## 13. Camera Placement

A standardized participant-camera arrangement should eventually define:

- approximate viewing distance;
- camera height;
- angle;
- field of view;
- background constraints where required;
- allowable participant movement.

The setup should avoid requiring unnatural participant posture merely to
improve signal quality.

## 14. Illumination Infrastructure

Because both behavioural imaging and rPPG depend on illumination, the laboratory
may require:

- controllable light source;
- illuminance measurement;
- repeatable source placement;
- documentation of daylight/external-light effects.

The exact lighting protocol should be defined before confirmatory rPPG work.

## 15. Phase C — rPPG Component Validation

### Objective

Establish what the implemented webcam-rPPG pipeline can and cannot measure under
real conditions.

This phase is separate from engagement or cognitive-load validation.

## 16. Public-Dataset Validation First

Where permitted, UBFC-rPPG can provide a first real-data component test.

The future workflow should:

1. obtain the dataset through its authoritative source;
2. verify permitted use manually;
3. validate its local structure;
4. confirm reference sampling information;
5. run EngageVR rPPG methods;
6. report the predefined metrics;
7. retain dataset provenance.

UBFC-rPPG cannot supply engagement or cognitive-load validation.

## 17. Local Reference-Device Study

A later controlled study may compare webcam rPPG against an independent
physiological reference.

Candidate technologies include:

- contact PPG;
- ECG.

The exact reference device must be selected and validated before use.

## 18. rPPG Study Factors

A controlled component-validation study may examine:

- rPPG algorithm;
- illumination;
- motion;
- camera configuration.

Current algorithm candidates include:

```text
GREEN
CHROM
POS
```

The final design should avoid changing many factors simultaneously without a
clear analysis plan.

## 19. rPPG Outcomes

Potential outcomes include:

- valid-window proportion;
- estimated-HR availability;
- absolute HR error;
- signed error/bias;
- RMSE;
- agreement;
- signal-quality behaviour.

The final primary outcome should be selected prospectively.

## 20. rPPG Generalization

A successful laboratory rPPG result would remain conditional on:

- tested population;
- webcam;
- lighting;
- motion;
- reference device;
- preprocessing;
- algorithm.

It would not justify a claim of universal physiological accuracy.

## 21. Phase D — Task and Unity Runtime Validation

### Objective

Validate the exact participant-facing task environment.

If Unity becomes the participant runtime, the exact build used for research
must be compiled and exercised.

## 22. Unity Development Environment

A future laboratory setup may require:

- Unity Editor;
- documented Unity version;
- platform-specific build tooling;
- dependency/package lock information;
- reproducible project setup.

The selected version should be recorded before participant data collection.

## 23. Unity Task Validation

Validate:

- scene loading;
- task instructions;
- response capture;
- timing;
- task completion;
- difficulty state;
- telemetry;
- shutdown;
- restart/recovery.

A successful compilation alone is insufficient.

## 24. Python/Unity Integration

The complete communication path should be tested for:

- connection;
- protocol version;
- heartbeat;
- task events;
- model-side state;
- command creation;
- dispatch;
- receipt;
- acknowledgement;
- applied task state;
- disconnect handling.

## 25. Adaptation Lifecycle Validation

The future laboratory runtime should be capable of distinguishing:

```text
proposed
command built
dispatched
received
acknowledged
applied
rejected
```

where the actual implementation can observe those states.

A proposal must not be recorded as participant exposure unless the environment
actually applies it.

## 26. Experimenter Safety Control

Before adaptive participant use, validate an experimenter control capable of
stopping or disabling adaptation.

The control should remain usable during:

- normal operation;
- model failure;
- communications failure;
- unexpected adaptation behaviour.

## 27. Phase E — Integrated Non-Participant Dry Runs

### Objective

Exercise the complete laboratory workflow before participant recruitment.

A dry run may include:

- session creation;
- webcam capture;
- task execution;
- model inference;
- uncertainty gating;
- adaptation;
- recording;
- dashboard monitoring;
- session close;
- artifact verification.

Dry runs are engineering exercises.

They are not participant studies.

## 28. Dry-Run Scenarios

Candidate scenarios include:

### Normal session

All expected components remain available.

### Webcam degradation

Face or rPPG becomes unavailable.

Expected response:

```text
unavailable
abstain
HOLD
```

where applicable.

### Model calibration unavailable

The classification evidence gate should respond according to its documented
calibration requirement.

### Task communication failure

Adaptation should not silently become applied.

### Experimenter disablement

Adaptation should enter the defined disabled/safe state.

### Interrupted recording

The session should retain an explicit interrupted status.

## 29. Dry-Run Acceptance Gate

The project should not enter participant pilot work merely because one dry run
succeeded.

Candidate acceptance requirements include:

- repeated successful complete sessions;
- reproducible configuration;
- correct recording provenance;
- predictable failure handling;
- safe adaptation disablement;
- no unexpected raw-video retention;
- no participant-data path through Git;
- interpretable logs.

Exact engineering acceptance thresholds remain to be defined.

## 30. Phase F — Approved Participant Pilot

### Objective

Evaluate feasibility and participant-facing appropriateness before a main
confirmatory comparison.

This phase requires the appropriate institutional pathway.

## 31. Pilot Questions

The pilot may investigate:

- session duration;
- task comprehension;
- equipment comfort;
- webcam positioning;
- signal availability;
- rPPG feasibility;
- questionnaire burden;
- adaptation timing;
- adaptation acceptability;
- experimenter workflow;
- stop procedure.

## 32. Pilot Is Not the Main Experiment

Pilot observations should not automatically be pooled with a later
confirmatory dataset.

Before the pilot, define whether pilot participants could be included in later
analysis.

## 33. Pilot Adaptation Scope

The first participant-facing adaptation pilot should remain narrow.

The currently implemented candidate intervention is:

```text
set_difficulty
```

The pilot should not expand informally into additional interventions merely
because the study environment supports them technically.

## 34. Pilot Safety

The pilot should include:

- experimenter oversight;
- participant stop procedure;
- predefined safe state;
- adaptation disablement;
- technical-failure procedure;
- privacy incident procedure.

## 35. Pilot Exit Gate

Progress to a main study only if the approved pilot establishes that the
procedure is sufficiently:

- understandable;
- operationally stable;
- tolerable;
- technically observable;
- appropriately controlled.

Pilot success must not be equated with adaptation benefit.

## 36. Phase G — Participant-Labelled Model Validation

### Objective

Replace synthetic software checks with scientifically appropriate evaluation
for the predictive questions.

## 37. Label Strategy

The research programme must establish valid target sources before model claims
can be made.

Possible sources may include:

- participant-reported measures;
- task-derived outcomes;
- externally established experimental labels.

Model predictions cannot become their own ground truth.

## 38. Engagement Validation

The future engagement model evaluation should define:

- construct;
- target source;
- timing;
- participant grouping;
- primary metric;
- calibration procedure;
- limitations.

It must not assume engagement has one universally observable ground-truth
signal.

## 39. Cognitive-Load Validation

Likewise, future cognitive-load evaluation should define:

- construct;
- measurement/instrument;
- target timing;
- participant grouping;
- model target;
- analysis.

Physiological measurements alone must not automatically be treated as
cognitive-load ground truth.

## 40. Model Comparison

Future real-data evaluation may compare the existing interpretable model
families.

A champion should only be designated after predefined criteria exist.

Candidate comparison families include:

- logistic regression;
- Ridge;
- random forest;
- histogram gradient boosting;
- early fusion;
- late fusion;
- quality-aware fusion;
- validation-weighted fusion;
- stacked fusion.

No future champion is predetermined by this roadmap.

## 41. Calibration Validation

Probability calibration must eventually be evaluated on real appropriate data.

Future work should examine:

- reliability;
- calibration error;
- calibration sample requirements;
- stability across participants.

A calibration procedure passing synthetic tests does not establish empirical
calibration in people.

## 42. Regression-Uncertainty Validation

Future real participant evaluation should examine:

- empirical conformal coverage;
- interval width;
- participant heterogeneity;
- failures of cross-participant exchangeability.

The project must remain prepared for the conformal assumptions to perform
poorly.

## 43. Selective Prediction Validation

Future selective-prediction work should report:

```text
coverage
+
performance on accepted predictions
```

rather than reporting only improved accepted-set performance.

A system that achieves high performance by refusing nearly every case may not
be useful.

## 44. Personalization Validation

Real-data personalization research should compare population and personalized
predictions on identical held-out evaluation observations.

Important outcomes include:

- mean effect;
- participant-level distribution;
- number improved;
- number unchanged;
- number worsened;
- cold-start frequency.

Personalization must be allowed to fail scientifically.

## 45. Phase H — Static-versus-Adaptive Experiment

### Objective

Evaluate whether the adaptive condition produces measurable differences from a
comparable static condition.

The proposed design is currently a within-participant crossover study.

## 46. Experimental Conditions

The main comparison should retain:

```text
static
```

versus:

```text
adaptive
```

with condition order controlled prospectively.

The adaptive condition should use only the validated participant-facing
adaptation mechanism.

## 47. Main-Study Prerequisites

Before the main experiment:

- participant-facing runtime validated;
- participant data procedures approved;
- models appropriate for adaptation evaluated;
- uncertainty gate evaluated;
- adaptation transport validated;
- pilot completed where required;
- primary outcome selected;
- sample size justified;
- statistical-analysis plan frozen.

## 48. Main-Study Interpretation

Even if an adaptive condition differs statistically from a static condition,
the result should be interpreted according to the measured outcome.

For example:

```text
higher accuracy
```

does not automatically mean:

```text
higher engagement
```

unless the evidence actually establishes that relationship.

## 49. Adaptation Appropriateness

The laboratory programme should retain the distinction:

```text
adaptation technically worked
        !=
adaptation was appropriate
        !=
adaptation produced benefit
```

All three require different evidence.

## 50. Phase I — Replication and Extension

Only after foundational questions are answered should EngageVR expand to more
complex interventions or environments.

Possible future directions include:

- independent replication;
- larger participant samples;
- additional tasks;
- longitudinal sessions;
- richer temporal models;
- additional sensing modalities;
- broader adaptation actions;
- immersive VR experiments;
- cross-device validation.

None is required for the initial scientific validation programme.

## 51. Temporal Models

Future work may investigate temporal approaches after sufficient real
longitudinal data exist.

Possible model classes may include:

- state-space models;
- temporal boosting features;
- recurrent models;
- temporal convolution;
- transformer-based sequence models.

No temporal neural model should be added solely because it is more complex.

The comparison must include appropriate simpler baselines.

## 52. Additional Physiological Modalities

Potential future sensors may include:

- EDA;
- respiration;
- contact PPG;
- ECG;
- eye tracking.

Each new modality introduces:

- hardware burden;
- synchronization requirements;
- participant burden;
- privacy considerations;
- missingness;
- validation requirements.

More modalities do not automatically create a better scientific system.

## 53. Eye Tracking

If future VR hardware provides eye tracking, the project may investigate:

- gaze direction;
- fixation;
- blink information;
- pupil-related measures where technically valid.

Device-derived eye metrics must be validated before being treated as cognitive
or engagement measures.

## 54. EDA

Electrodermal activity may be considered as a future physiological modality.

EDA is not a direct measurement of engagement or cognitive load.

Any use would require:

- appropriate hardware;
- participant-contact procedure;
- preprocessing;
- quality assessment;
- scientific target interpretation.

## 55. Respiration

Future respiration measurement may support physiological characterization.

Again, respiration is not itself engagement.

Any derived relation must be evaluated empirically.

## 56. Multi-Session Research

Longitudinal research may eventually examine:

- within-person variability;
- baseline stability;
- personalization persistence;
- calibration decay;
- adaptation history;
- repeated exposure.

This requires participant IDs and session provenance capable of maintaining
longitudinal relationships without leaking identity into modelling features.

## 57. Personalization Across Sessions

Future work may test whether calibration from one session transfers to another.

Potential questions include:

- how long a personal baseline remains useful;
- whether recalibration is needed;
- whether hardware/environment changes invalidate it;
- whether personalization degrades.

No persistence duration should be assumed before evidence exists.

## 58. Cross-Device Generalization

Future experiments may compare models or rPPG across:

- webcams;
- displays;
- computers;
- VR headsets;
- reference sensors.

Device-specific performance should remain visible.

Pooling devices into one result can hide systematic failure.

## 59. Cross-Environment Generalization

Potential environments include:

- controlled laboratory;
- ordinary indoor workspace;
- different illumination;
- different task contexts.

A model validated in one environment should not automatically be assumed valid
in another.

## 60. Multi-Site Extension

A much later extension could evaluate EngageVR at more than one institution or
laboratory.

Multi-site work introduces:

- hardware heterogeneity;
- operator differences;
- protocol differences;
- governance;
- data transfer;
- institution-specific approvals.

This is not part of the initial laboratory extension.

## 61. Adaptive Action Expansion

Only after `set_difficulty` is evaluated should additional adaptations be
considered.

Possible future actions might include:

- pacing;
- information density;
- break suggestions;
- visual complexity;
- feedback timing.

Each action is a new intervention.

It requires its own:

- implementation;
- risk assessment;
- participant-facing description;
- validation;
- experimental justification.

## 62. No Automatic Clinical Expansion

Future availability of physiological sensors must not cause EngageVR to drift
into clinical claims.

The project remains a research framework unless an entirely separate body of
evidence and regulatory work establishes otherwise.

## 63. Laboratory Roles

A future laboratory study may require clearly separated responsibilities.

Possible roles include:

- principal investigator/supervisor;
- participant-facing experimenter;
- technical operator;
- data manager;
- analyst.

One person may hold multiple roles in a small study, but responsibilities should
still be explicit.

## 64. Standard Operating Procedures

Future laboratory operation may require SOPs for:

- participant setup;
- webcam setup;
- reference-sensor setup;
- equipment cleaning;
- task startup;
- session recording;
- adaptation stop control;
- incident handling;
- session closure;
- data transfer;
- backup;
- deletion.

SOPs should correspond to the actual approved laboratory configuration.

## 65. Training

Researchers operating participant-facing hardware should understand:

- study protocol;
- equipment;
- stop procedure;
- privacy requirements;
- data handling;
- system limitations.

A software README is not sufficient participant-study training.

## 66. Configuration Freeze

Before a confirmatory study, freeze relevant:

- source version;
- model version;
- feature schema;
- thresholds;
- personalization configuration;
- adaptation policy;
- Unity/task build;
- webcam configuration;
- reference sensor;
- questionnaires;
- analysis plan.

## 67. Change Control

If a laboratory component changes during data collection, record:

- what changed;
- why;
- when;
- affected sessions;
- whether institutional amendment/review is required;
- analysis consequences.

Changes must not disappear into ordinary development history.

## 68. Laboratory Artifact Provenance

Future laboratory artifacts should identify:

- study/protocol version;
- participant pseudonym where applicable;
- session;
- hardware configuration;
- software configuration;
- model configuration;
- experimental condition;
- timestamps.

## 69. Laboratory Validation Artifacts

Future engineering work may eventually produce records under a structure such
as:

```text
artifacts/lab_validation/
```

with possible files such as:

```text
hardware_manifest.json
environment_manifest.json
validation_results.json
timing_report.json
failure_log.jsonl
```

These are prospective examples.

They are not existing repository contracts.

## 70. Participant Study Artifacts

Approved participant recording should use the repository's validated session
format rather than inventing an unrelated format during the study.

Participant recordings must remain outside ordinary Git history.

## 71. Dataset Evolution

Future dataset cards should be updated as evidence progresses.

Possible progression:

```text
synthetic
        ->
public component dataset
        ->
pilot participant dataset
        ->
main participant dataset
        ->
replication dataset
```

These datasets should remain separate unless scientifically valid linkage
exists.

## 72. Model Card Evolution

Model cards should be updated when real evidence becomes available.

Future cards may eventually include:

- evaluated population;
- real performance;
- confidence intervals;
- calibration;
- known failure conditions;
- selection rationale;
- appropriate intended use.

A champion label must not be added automatically by the MLOps system.

## 73. Reproducibility

Future laboratory studies should preserve reproducible:

- software;
- configuration;
- analysis;
- dataset derivation;
- model fitting.

Physical measurements themselves cannot be recreated byte-for-byte.

Reproducibility means documenting enough of the acquisition and analysis process
for the work to be meaningfully repeated.

## 74. Replication

A successful first experiment should ideally be followed by replication rather
than immediate expansion of claims.

Replication may assess:

- effect stability;
- model stability;
- hardware sensitivity;
- participant heterogeneity;
- adaptation robustness.

## 75. Negative Findings

The laboratory extension must remain valuable even if:

- rPPG performs poorly;
- fusion provides no improvement;
- personalization degrades predictions;
- abstention is too frequent;
- adaptation provides no measurable benefit;
- participants dislike adaptive difficulty.

These outcomes should inform redesign rather than be suppressed.

## 76. Stop/Redesign Gates

A laboratory track should pause for redesign if evidence shows:

- unacceptable participant burden;
- unreliable task operation;
- unsafe adaptation behaviour;
- major privacy weakness;
- unusable physiological measurement;
- unresolvable synchronization error;
- insufficient target validity.

Progression through the roadmap is conditional, not automatic.

## 77. Proposed Laboratory Extension Matrix

| Stage | Primary question | Evidence required to advance |
|---|---|---|
| A — Infrastructure | Can the laboratory environment be configured reproducibly and securely? | Hardware/storage/network setup documented |
| B — Webcam | Does physical acquisition work reliably? | Stable physical capture and feature extraction |
| C — rPPG | Can webcam pulse estimates be evaluated against real reference data? | Public/reference evaluation with valid synchronization |
| D — Task/Unity | Does the participant-facing runtime behave as specified? | Compiled runtime and command/state validation |
| E — Dry run | Does the integrated system behave safely and observably? | Repeated complete non-participant dry runs |
| F — Pilot | Is the approved participant procedure feasible and acceptable? | Reviewed pilot and predefined exit criteria |
| G — Models | Do model outputs have scientific support on appropriate real labels? | Real participant-labelled evaluation |
| H — Adaptation | Does adaptive condition differ from static under the approved study? | Controlled participant comparison |
| I — Extension | Are findings sufficiently robust to justify broader research? | Replication and evidence-specific justification |

## 78. Dependencies Before Participant Research

The following remain mandatory dependencies before future participant work:

- appropriate institutional review;
- finalized participant information and consent;
- risk assessment;
- approved data-management procedure;
- participant storage/access plan;
- finalized task;
- hardware validation;
- stop procedure;
- analysis plan;
- study-specific sample-size justification.

## 79. Dependencies Before Adaptive Participant Research

Adaptive participant research additionally requires:

- participant-facing model evidence appropriate to the intended use;
- uncertainty/abstention configuration;
- validated command transport;
- validated applied-state recording;
- experimenter disable control;
- safe fallback;
- adaptation-appropriateness pilot where required.

## 80. Out-of-Scope Near-Term Extensions

The following should not distract from the initial validation programme:

- cloud-scale deployment;
- production SaaS operation;
- autonomous unsupervised participant monitoring;
- medical diagnosis;
- clinical decision support;
- emotion recognition claims;
- employment or academic assessment;
- identity recognition;
- high-stakes automated decisions.

## 81. Success Criteria for the Laboratory Programme

The laboratory extension should be considered successful when it produces
credible evidence about what EngageVR can and cannot do.

Success does not require every hypothesis to be supported.

Useful outcomes include:

- identifying measurement failure;
- rejecting an ineffective model;
- demonstrating insufficient calibration;
- finding personalization harms performance;
- showing adaptation has no benefit;
- discovering hardware constraints;
- narrowing future research questions.

## 82. Long-Term Research Vision

If the foundational validation succeeds, EngageVR may become a reproducible
platform for studying:

- multimodal human-computer interaction;
- uncertainty-aware adaptive systems;
- participant-specific calibration;
- missing-modality robustness;
- conservative intervention policies;
- adaptive VR/task environments.

This remains a research direction rather than a current validated capability.

## 83. Current Status

This document completes the planned Milestone 11 future-laboratory extension
roadmap.

It does not claim:

- laboratory hardware has been purchased or selected;
- physical hardware validation has been completed;
- participant recruitment is authorized;
- participant data exist;
- institutional approval exists;
- rPPG has been validated;
- engagement/cognitive-load models have been scientifically validated;
- personalization is beneficial;
- adaptive difficulty is beneficial;
- Unity or VR is participant-ready;
- a future laboratory phase is guaranteed to proceed.

Each future phase remains conditional on the evidence, engineering state, risk
controls, and institutional requirements that apply at that time.
