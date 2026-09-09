# EngageVR Data Collection Protocol

> **Draft requiring institutional review**
>
> This document describes a proposed future data-collection protocol for
> EngageVR research. No participant study described here has been conducted.
> No institutional or ethical approval is claimed.
>
> Participant recruitment or data collection must not begin until the final
> protocol, consent materials, risk controls, privacy procedures, and study
> design have undergone the appropriate institutional review.

## 1. Purpose

This document defines how data may be collected during future EngageVR
component-validation and participant studies.

It builds on:

- `docs/RESEARCH_PROPOSAL.md`;
- `docs/HYPOTHESES.md`;
- `docs/EXPERIMENTAL_VARIABLES.md`;
- `docs/EXPERIMENT_DESIGN.md`;
- `docs/SESSION_FORMAT.md`;
- `docs/FEATURE_DATASET.md`;
- `docs/PROTOCOL.md`.

The protocol is designed around:

- data minimization;
- explicit provenance;
- synchronized timestamps;
- quality-aware measurement;
- explicit missingness;
- pseudonymous participant identifiers;
- separation of administrative identity from research data;
- refusal to fabricate unavailable measurements.

## 2. Scope

This protocol covers candidate collection of:

- participant/session metadata;
- experimental-condition metadata;
- task telemetry;
- behavioural webcam-derived features;
- head-pose and movement features;
- webcam-rPPG-derived signals and estimates;
- signal-quality diagnostics;
- predictive model outputs;
- uncertainty and abstention information;
- adaptation decisions;
- adaptation lifecycle events;
- subjective participant responses;
- optional future reference physiological measurements.

It does not authorize participant data collection.

## 3. Study Types

EngageVR may require more than one study.

### 3.1 Component-validation studies

These may evaluate an individual subsystem such as:

- webcam behavioural capture;
- rPPG robustness;
- agreement with a physiological reference device;
- task telemetry;
- model calibration.

A component study must not be presented as validation of the complete EngageVR
system unless the evidence actually supports that conclusion.

### 3.2 Pilot study

A pilot may assess:

- task usability;
- procedure timing;
- signal availability;
- technical reliability;
- adaptation transport;
- adaptation appropriateness;
- participant burden;
- questionnaire timing.

Pilot results are primarily procedural unless the final protocol explicitly
defines another purpose.

### 3.3 Main controlled study

The proposed main study compares:

- static condition;
- adaptive condition.

The design is currently proposed as a within-participant crossover design,
subject to final statistical and institutional review.

## 4. Data-Minimization Principle

Only data required to answer a defined research question or to verify study
integrity should be collected.

The protocol must not collect a variable merely because:

- it may be interesting later;
- the software can record it;
- storage capacity is available;
- another study commonly collects it.

Every participant-data field should have a documented purpose.

## 5. Participant Identification

Research records should use pseudonymous participant identifiers.

Example conceptual form:

```text
participant_id = P0001
```

The exact identifier format may differ.

The analysis dataset must not use:

- participant name;
- email address;
- telephone number;
- student number;
- government identifier;
- social-media identity;

as modelling identifiers.

If administrative contact information is required for recruitment or consent,
it should be kept separately from research-analysis data according to the final
approved protocol.

## 6. Session Identification

Each recording should use a unique session identifier.

A session identifier must not encode unnecessary identifying information.

The recording directory and recorded `session_id` must remain distinguishable.

If they disagree, the mismatch should be reported rather than silently
rewritten.

## 7. Required Session Provenance

Each research session should record, where applicable:

- participant pseudonym;
- session identifier;
- study/protocol version;
- data-source classification;
- software version;
- configuration version or fingerprint;
- experimental condition;
- condition order;
- task version;
- model version;
- adaptation configuration;
- recording start/end state.

Provenance must remain attached to downstream analysis artifacts.

## 8. Scientific Eligibility

Every dataset or recording must state whether it is eligible for scientific
evaluation.

Synthetic software checks must retain:

```text
scientific_evaluation_eligible = false
```

Synthetic records must not later become participant evidence simply because
they were processed through the same pipeline.

Public datasets and future participant datasets must retain their own distinct
provenance.

## 9. Proposed Session Workflow

A future approved participant session may follow:

```text
Participant arrival
        |
        v
Identity handled by approved administrative process
        |
        v
Study information and consent
        |
        v
Eligibility / safety checks
        |
        v
Assign pseudonymous participant ID
        |
        v
Equipment setup
        |
        v
Signal-quality check
        |
        v
Baseline / calibration
        |
        v
Practice
        |
        v
Experimental condition 1
        |
        v
Condition questionnaire
        |
        v
Transition / recovery
        |
        v
Experimental condition 2
        |
        v
Condition questionnaire
        |
        v
Post-session assessment / debrief
        |
        v
Recording integrity check
        |
        v
Session closure
```

Exact timing remains to be finalized.

## 10. Pre-Collection Checks

Before starting a participant recording, the operator should verify:

- correct participant pseudonym;
- correct study configuration;
- correct experimental condition;
- correct condition order;
- expected software version;
- expected task version;
- webcam availability where required;
- storage destination;
- adequate disk space;
- synchronized system clock;
- required model/configuration availability;
- raw-video setting;
- adaptation enable state;
- emergency/experimenter stop control.

Failure should be corrected before participant exposure where possible.

## 11. Webcam Data

### 11.1 Default privacy behaviour

Raw webcam video storage should remain disabled by default.

Webcam frames may be processed locally/in memory for feature extraction.

### 11.2 Candidate derived webcam data

Potential derived data include:

- face-detection state;
- facial landmarks or derived geometry where permitted;
- eye-closure/blink proxies;
- mouth-geometry proxy;
- head yaw;
- head pitch;
- head roll;
- head-motion features;
- capture-quality metrics.

### 11.3 Raw image/video exception

Raw webcam images or video must not be enabled merely for convenience.

If a future approved study genuinely requires raw imagery, the protocol must
define:

- research necessity;
- consent wording;
- storage location;
- access controls;
- retention duration;
- deletion procedure;
- whether images leave the local machine.

This would be a deviation from the project's normal privacy-preserving default.

## 12. rPPG Data

Candidate rPPG-derived data may include:

- selected skin-region signal summaries;
- RGB traces where retained by the approved protocol;
- processed rPPG waveform;
- method identifier;
- estimated heart rate where valid;
- signal-quality index;
- motion diagnostics;
- illumination diagnostics;
- validity/unavailability reason.

Current candidate methods include:

- GREEN;
- CHROM;
- POS.

## 13. rPPG Validity Rule

An rPPG value must not be produced solely because a processing window exists.

If the signal does not satisfy the method's quality requirements, the result
should be recorded as unavailable with an explicit reason.

The system must not replace an unavailable physiological estimate with:

```text
0
```

or another fabricated default.

## 14. HR and HRV

Any rPPG-derived heart-rate value is a signal-processing estimate.

It is not a medical measurement.

HRV-related features should be collected only when:

- the required input signal exists;
- the recording duration is scientifically sufficient;
- beat/interval quality is acceptable;
- the selected HRV measure is appropriate to the window.

If these conditions are not satisfied, HRV should remain unavailable.

## 15. Reference Physiological Data

Future laboratory validation may include a reference device such as:

- PPG;
- ECG;

or another justified physiological reference.

If used, the protocol must record:

- device identity/model;
- sampling rate;
- acquisition configuration;
- synchronization method;
- placement procedure;
- quality criteria;
- preprocessing;
- clock alignment.

Reference measurements must not be assumed perfectly error-free.

## 16. Task Telemetry

Task events may include:

- session start;
- trial start;
- stimulus presentation;
- response;
- response correctness;
- reaction time;
- current difficulty;
- difficulty change;
- task pause/resume where applicable;
- trial completion;
- session completion.

The actual event vocabulary should follow the repository's canonical protocol.

## 17. Task Performance

Derived task-performance variables may include:

- accuracy;
- error rate;
- reaction time;
- completion time;
- inactivity;
- retries or hints where actually implemented;
- difficulty exposure.

Every derived quantity must retain enough provenance to identify the source
trials.

## 18. Subjective Responses

Candidate participant-reported constructs include:

- engagement;
- cognitive load or mental effort;
- comfort;
- adaptation acceptance;
- perceived disruption.

The exact instruments have not yet been finalized.

Before collection, each instrument must have documented:

- name;
- construct;
- version;
- source;
- permitted use;
- scoring method;
- administration timing.

Validated questionnaire wording must not be copied or modified without checking
its usage conditions.

## 19. Questionnaire Timing

Possible collection points include:

- after baseline where justified;
- after each experimental condition;
- after a delivered adaptation where appropriate;
- at final session completion.

Questionnaire frequency must balance measurement needs against:

- interruption;
- fatigue;
- demand characteristics;
- excessive participant burden.

## 20. Model Outputs

Where the approved study runs EngageVR inference, candidate stored outputs
include:

- prediction target;
- predicted class;
- predicted score where applicable;
- calibrated probabilities;
- classification confidence;
- regression prediction interval;
- regression interval width;
- model version;
- feature-schema version;
- prediction timestamp.

Model output is not ground truth.

## 21. Signal Quality vs Model Uncertainty

These must be stored as separate concepts.

### Signal quality

Answers:

> Is the measurement usable?

### Predictive uncertainty

Answers:

> How uncertain is the predictive model under its defined semantics?

Poor signal quality must not automatically be converted to:

- low engagement;
- high cognitive load;
- low model confidence.

## 22. Abstention

Each inference opportunity should preserve whether the system:

- produced a valid prediction;
- abstained;
- could not evaluate the input;
- was otherwise unavailable.

Where available, record:

- abstention status;
- abstention reason;
- evidence-gate state;
- relevant uncertainty measurement.

Abstained observations must not disappear silently from downstream analysis.

## 23. Missing Modalities

The collection layer should preserve:

- available modalities;
- unavailable modalities;
- reason for unavailability;
- quality-based exclusion where applicable.

Examples include:

- face not detected;
- rPPG insufficient quality;
- task telemetry missing;
- sensor disconnected;
- feature unavailable.

Missing modalities must not be filled with misleading zeros.

## 24. Adaptation Decision Data

Where adaptation evaluation runs, candidate records include:

- input prediction identifiers;
- evidence-gate result;
- policy decision;
- previous task difficulty;
- proposed task difficulty;
- HOLD/adaptation reason;
- policy-state information;
- timestamp.

These are controller records, not participant outcome measures.

## 25. Adaptation Lifecycle

Where participant-facing adaptation is enabled, the recording must distinguish:

```text
proposed
command built
dispatched
acknowledged
applied
rejected
```

where supported.

No later lifecycle state should be fabricated.

A proposal is not an applied adaptation.

## 26. Experimenter Intervention

Experimenter interventions should be recorded where relevant.

Possible fields include:

- timestamp;
- intervention category;
- resulting adaptation state;
- non-identifying reason.

Potential categories may include:

- participant request;
- discomfort;
- safety concern;
- technical failure;
- protocol deviation.

Free-text notes should be minimized where structured categories are sufficient.

## 27. Environmental Metadata

Environmental variables should be collected only when relevant.

Potential fields include:

- illumination condition;
- major illumination changes;
- webcam position;
- approximate camera-participant geometry where justified;
- device configuration;
- task display configuration.

For controlled rPPG studies, quantitative measurements are preferable to vague
labels such as:

```text
good lighting
bad lighting
```

where suitable measuring equipment is available.

## 28. Timestamping

All modalities should use the project's common timestamp system.

Each time-dependent record should identify its temporal position sufficiently
for synchronization.

Relevant events include:

- capture frames/features;
- rPPG windows;
- task events;
- predictions;
- adaptation decisions;
- commands;
- acknowledgements;
- questionnaire responses.

## 29. Synchronization

Synchronization must be explicit.

Analysis must not align unrelated observations simply because they occupy
similar row positions in different files.

Window-to-event association should use:

- timestamps;
- session identity;
- explicit window identifiers;
- protocol-defined relationships.

## 30. Recording Format

The existing EngageVR session infrastructure should remain the canonical basis
for session recording.

Conceptually, a session may contain:

```text
manifest.json
events.jsonl
summary.json
```

plus additional explicitly defined diagnostics where required.

The actual schema and filenames must follow the current repository contracts.

## 31. Append-Only Recording

During live acquisition, session event history should remain append-only.

Existing records should not be silently rewritten during the session to make a
recording appear complete.

Corrections or recovery states should be represented explicitly.

## 32. Partial Trailing Record

A partial trailing JSONL line may represent an in-progress append operation.

It should be treated differently from malformed data in the interior of the
recording.

Interior corruption must remain visible.

## 33. Interrupted Sessions

An interrupted session is not automatically a failed session.

The recording should distinguish states such as:

- active;
- completed;
- interrupted;
- corrupt;
- otherwise explicitly defined status.

The analysis plan must later determine which portions are eligible for each
analysis.

## 34. Data Quality Flags

Candidate quality/validity flags include:

- measurement available;
- measurement quality eligible;
- prediction available;
- trial valid;
- session complete;
- analysis eligible;
- exclusion reason.

Quality rules should be declared before confirmatory analysis.

## 35. Technical Failures

Potential technical failures include:

- webcam disconnection;
- task crash;
- storage failure;
- synchronization error;
- model loading failure;
- command-transport failure;
- corrupt output;
- insufficient disk space.

The system should log the failure where possible.

The operator should not manually invent replacement data.

## 36. Participant Withdrawal

If a participant withdraws, the procedure must follow the final approved
consent and institutional protocol.

The protocol must state:

- whether already collected data may be retained;
- whether deletion may be requested;
- how deletion is performed;
- which identifiers are required to locate the data;
- how withdrawal is distinguished from technical interruption.

This cannot be finalized before institutional review.

## 37. Participant Privacy

Research datasets should use pseudonymous identifiers.

Direct identity information must not be included in model-training or analysis
files unless explicitly required and approved.

Where administrative identity records exist, they should be logically and
operationally separated from research data.

## 38. Sensitive Data

Behavioural and physiological measurements may be sensitive even when they do
not contain a person's name.

The study documentation must avoid implying that pseudonymization removes all
privacy risk.

Access should be limited according to the final approved data-management plan.

## 39. Storage Location

The final participant protocol must define the approved storage location before
collection.

The development repository itself is not an approved participant-data store.

Participant data must not be committed to Git.

## 40. Git Exclusions

The following must remain outside ordinary source control:

- participant recordings;
- raw video;
- raw physiological files;
- participant questionnaires;
- generated model artifacts;
- credentials;
- `.env` secrets;
- private keys;
- local research identifiers capable of identifying participants.

`.gitignore` is not, by itself, a complete privacy control.

## 41. Access Control

Before participant data are collected, define:

- authorized users;
- storage permissions;
- device access;
- backup access;
- transfer procedures;
- incident handling.

Access should follow least-privilege principles.

## 42. Retention

No retention duration is asserted in this draft.

The final retention period must be determined according to:

- institutional requirements;
- consent language;
- research purpose;
- applicable policy;
- data sensitivity.

Retention periods must not be invented solely for this software document.

## 43. Deletion

The final protocol must define deletion procedures for:

- local primary data;
- backups;
- exported analysis copies;
- derived participant artifacts where applicable.

Deletion feasibility and limitations must be communicated accurately in the
approved participant information.

## 44. Backup

Any backup procedure must preserve:

- access controls;
- participant pseudonymization;
- integrity;
- deletion requirements;
- provenance.

Uncontrolled cloud synchronization must not be introduced implicitly.

## 45. Data Transfer

The default project architecture is local-first.

If future collaboration requires transferring participant data, the approved
protocol must define:

- destination;
- recipient;
- transfer mechanism;
- encryption where required;
- permitted data subset;
- legal/institutional basis;
- retention at the destination.

## 46. Public Datasets

Public datasets should be handled separately from participant recordings.

Before use, record:

- dataset title;
- authoritative source;
- license/terms;
- participant population;
- modalities;
- sampling rates;
- labels;
- known limitations;
- permitted use.

Public-dataset results must retain the dataset identity.

## 47. Synthetic Data

Synthetic data may be used for:

- unit tests;
- integration tests;
- schema validation;
- pipeline reproducibility;
- dashboard testing;
- deterministic software demonstrations.

Synthetic data must not be used as evidence that:

- an engagement model is scientifically accurate;
- a cognitive-load model is scientifically accurate;
- personalization benefits participants;
- adaptation benefits participants;
- rPPG is valid in humans.

## 48. Dataset Separation

Datasets with incompatible:

- participants;
- collection protocols;
- labels;
- modalities;
- sampling procedures;

must not be merged and described as one synchronized multimodal dataset unless
a scientifically valid linkage actually exists.

## 49. Data Integrity

Research artifacts should use integrity mechanisms where appropriate.

Potential mechanisms include:

- checksums;
- schema validation;
- immutable run metadata;
- versioned configuration;
- deterministic identifiers.

Artifact integrity confirms that data have not unexpectedly changed.

It does not establish scientific validity.

## 50. Schema Validation

Collected data should be validated against explicit schemas.

Validation should reject or identify:

- invalid identifiers;
- malformed timestamps;
- impossible enum values;
- missing required provenance;
- non-finite values where prohibited;
- incompatible record structure.

Validation must not silently repair scientifically meaningful errors.

## 51. Non-Finite Values

`NaN` and infinity should not be silently treated as ordinary valid values where
the schema forbids them.

The system should explicitly reject or classify these states.

## 52. Missing Value Rule

`None`, missing, unavailable, abstained, and zero are distinct concepts.

For example:

```text
heart_rate = 0
```

must not be used as a substitute for:

```text
heart_rate = unavailable
```

unless zero is a genuinely valid measurement for that field.

## 53. Data Collection During Static Condition

If the final study requires direct comparability between static and adaptive
conditions, the static condition should generally collect the same observational
measurements as the adaptive condition where scientifically and ethically
appropriate.

This may include:

- sensing;
- task telemetry;
- model inference;
- uncertainty;
- observational adaptation decisions.

However, model-driven changes must not be applied to the environment in the
static condition.

## 54. Shadow Adaptation Logging

A future design may optionally compute what the policy would have proposed in a
static condition without applying it.

If used, this must be explicitly identified as:

```text
shadow / observational policy output
```

and never confused with participant exposure.

Whether shadow evaluation is included should be finalized before data
collection.

## 55. Data Collection During Adaptive Condition

The adaptive condition may additionally record:

- adaptation proposal;
- command construction;
- dispatch;
- acknowledgement;
- application where observable;
- resulting difficulty;
- experimenter intervention.

Only actually observed lifecycle states may be reported.

## 56. Baseline Data

Baseline data must identify its purpose.

Possible purposes include:

- personal normalization;
- calibration;
- resting physiological reference;
- sensor-quality verification.

Baseline data must not automatically be used for every purpose simply because
it exists.

## 57. Calibration Data

Calibration observations used for personalization must precede evaluation
observations.

The protocol should identify:

- start/end of calibration;
- number of calibration windows;
- whether calibration succeeded;
- cold-start state;
- calibration exclusion reason where applicable.

## 58. Evaluation Data

Evaluation data must remain separate from information used to construct the
participant-specific baseline.

The analysis should be capable of proving the chronological boundary.

## 59. Practice Data

Practice observations should be marked as practice.

They should not enter the main confirmatory analysis unless the statistical
analysis plan explicitly permits this.

## 60. Analysis Dataset Construction

The final analysis dataset should be constructed reproducibly from the recorded
source data.

The transformation process should preserve:

- participant grouping;
- session grouping;
- condition;
- timing;
- provenance;
- quality;
- missingness;
- target source.

Row-level random splitting must not replace participant/session-aware splitting
where that would cause leakage.

## 61. Derived Dataset Provenance

Every derived dataset should record enough information to identify:

- source recording(s);
- processing configuration;
- feature schema;
- code/software version;
- transformation step;
- dataset fingerprint or integrity identifier where applicable.

## 62. Exclusion Logging

If an observation is excluded from an analysis, record:

- exclusion rule;
- stage at which exclusion occurred;
- affected analysis;
- count.

Exclusions must not be performed silently.

## 63. Analysis-Specific Eligibility

A participant or session may be eligible for one analysis and unavailable for
another.

For example:

- valid task data but unusable rPPG;
- completed static condition but interrupted adaptive condition;
- valid behavioural features but no subjective response.

The protocol should avoid unnecessary all-or-nothing deletion of a participant's
entire session.

## 64. Data Collection Checklist

Before participant collection:

- [ ] institutional requirements satisfied;
- [ ] final protocol approved where required;
- [ ] approved consent materials available;
- [ ] participant eligibility procedure defined;
- [ ] participant pseudonym procedure defined;
- [ ] storage location approved;
- [ ] retention procedure defined;
- [ ] deletion procedure defined;
- [ ] software/configuration frozen;
- [ ] participant-facing task validated;
- [ ] required sensing validated;
- [ ] adaptation transport validated if used;
- [ ] stop procedure tested;
- [ ] raw-video setting confirmed;
- [ ] questionnaire instruments finalized;
- [ ] primary outcome finalized;
- [ ] statistical-analysis plan finalized.

## 65. Per-Session Operator Checklist

Candidate operational checklist:

- [ ] correct participant pseudonym;
- [ ] correct session identifier;
- [ ] correct condition;
- [ ] correct condition order;
- [ ] correct software/task version;
- [ ] correct model/configuration;
- [ ] expected storage destination;
- [ ] webcam/sensors available;
- [ ] signal-quality check complete;
- [ ] raw-video setting verified;
- [ ] adaptation lock state verified;
- [ ] participant stop procedure understood;
- [ ] session successfully started.

## 66. Session Close Checklist

At the end of a session:

- [ ] participant procedure completed or interruption recorded;
- [ ] recording finalized where possible;
- [ ] session status recorded;
- [ ] condition questionnaires accounted for;
- [ ] technical failures recorded;
- [ ] interventions recorded;
- [ ] protocol deviations recorded;
- [ ] no direct identifiers entered analysis files;
- [ ] storage integrity checked;
- [ ] participant data not placed in Git.

## 67. Protocol Deviations

Possible deviations include:

- incorrect condition;
- wrong configuration;
- task restart;
- sensor substitution;
- participant interruption;
- experimenter disabling adaptation;
- missing questionnaire;
- unexpected software update.

Each deviation should be recorded explicitly.

A deviation does not automatically invalidate all collected data.

Its effect should be determined by predefined analysis rules.

## 68. Incident Handling

The final approved study procedure must define responses to:

- participant discomfort;
- privacy incident;
- data corruption;
- equipment failure;
- accidental raw-data recording;
- accidental unauthorized transfer;
- unexpected adaptation behaviour.

This software document does not substitute for institutional incident procedures.

## 69. Data Collection Boundaries

EngageVR must never infer that data collection was scientifically valid simply
because:

- files exist;
- schemas passed;
- checksums match;
- Docker ran;
- CI passed;
- the dashboard displayed the data;
- MLflow logged the run;
- DVC reproduced the pipeline.

Those mechanisms support engineering integrity and reproducibility.

They do not establish:

- participant consent;
- protocol compliance;
- measurement validity;
- ethical approval;
- scientific validity.

## 70. Current Status

This protocol is a Milestone 11 planning document.

It does not assert that:

- participants have been recruited;
- participant data exist;
- ethical approval exists;
- storage/retention policies are finalized;
- questionnaires have been finalized;
- reference physiological hardware is available;
- Unity is participant-ready;
- live adaptation transport is validated;
- model estimates are scientifically validated.

Those items remain prerequisites for future approved research.
