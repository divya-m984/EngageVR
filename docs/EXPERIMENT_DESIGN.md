# EngageVR Static-versus-Adaptive Experimental Design

> **Draft for research planning and institutional review**
>
> This document specifies a proposed future experimental design for EngageVR.
> No participant study described here has been conducted. No institutional or
> ethical approval is claimed.
>
> Participant recruitment, data collection, and experimental deployment must
> not begin until the final protocol has undergone the appropriate
> institutional review.

## 1. Purpose

This document defines the proposed experimental structure for evaluating a
static task environment against an EngageVR adaptive condition.

It builds on:

- `docs/RESEARCH_PROPOSAL.md`;
- `docs/RESEARCH_QUESTIONS.md`;
- `docs/HYPOTHESES.md`;
- `docs/EXPERIMENTAL_VARIABLES.md`;
- `docs/ADAPTIVE_ENVIRONMENT.md`.

The design is prospective.

It does not imply that:

- the current model estimates have already been scientifically validated;
- adaptation has been demonstrated to be appropriate for participants;
- adaptation improves engagement;
- adaptation improves task performance;
- Unity integration has already been validated for participant use.

## 2. Primary Experimental Question

The principal study-level question is:

> Does an EngageVR adaptive task condition produce measurable differences in
> predefined task-performance and participant-reported outcomes relative to an
> otherwise comparable static condition?

This corresponds primarily to RQ4 and H4.

The first controlled study should evaluate only adaptation mechanisms that have
been technically implemented and validated before participant exposure.

## 3. Experimental Scope

The current EngageVR adaptation policy is centered on the existing
`set_difficulty` task command.

Therefore, the initial study should compare:

- a **static task-difficulty condition**; and
- an **EngageVR-controlled adaptive task-difficulty condition**.

The first study must not describe the following as implemented adaptive
interventions unless they are separately developed and validated before the
study:

- automatic breaks;
- information-density adjustment;
- visual-complexity adjustment;
- audio-intensity adjustment;
- feedback-frequency adaptation;
- instruction-pacing adaptation;
- breathing or recovery prompts.

Keeping the initial intervention narrow reduces ambiguity about what caused an
observed effect.

## 4. Study Prerequisites

The controlled static-versus-adaptive comparison must not begin merely because
the software controller exists.

Before participant exposure, the following prerequisites should be satisfied.

### 4.1 Technical readiness

The exact participant-facing task path should have demonstrated:

- successful task execution;
- stable session recording;
- correct timestamped telemetry;
- reproducible condition assignment;
- correct difficulty-state tracking;
- validated `set_difficulty` handling;
- command dispatch where required;
- acknowledgement where required;
- correct adaptation lifecycle recording;
- experimenter disable control;
- recovery from expected technical failures.

### 4.2 Unity or task-environment readiness

If Unity is used in the study, the specific build should undergo:

- successful compilation;
- runtime testing;
- input testing;
- task-event validation;
- adaptation-command validation;
- session-completion testing;
- failure-mode testing.

A source tree that has not been compiled is not sufficient evidence of
participant-facing readiness.

### 4.3 Sensing readiness

The study should define which sensing modalities are required and which are
optional.

Each required modality should have:

- a documented acquisition procedure;
- quality criteria;
- an unavailable state;
- appropriate failure handling.

The study must remain safe and scientifically interpretable when an optional
modality becomes unavailable.

### 4.4 Model readiness

Any model used to drive participant-facing adaptation should have an evaluation
appropriate to its intended use.

Synthetic software checks alone are insufficient.

### 4.5 Adaptation appropriateness

Before testing whether adaptation provides benefit, a preliminary pilot may be
needed to determine whether the proposed difficulty changes are:

- understandable;
- tolerable;
- operationally correct;
- non-disruptive;
- reasonably acceptable to participants and experimenters.

Appropriateness and benefit are different questions.

## 5. Proposed Design Type

The initial candidate design is a **within-participant crossover design**.

Each participant would complete both:

1. a static condition; and
2. an adaptive condition.

This design is proposed because engagement, task performance, behavioural
signals, and physiological measurements can vary substantially between
individuals.

Having each participant experience both conditions permits the primary
comparison to be made within the same participant.

The design remains provisional until reviewed alongside the statistical
analysis plan and practical laboratory constraints.

## 6. Experimental Conditions

### 6.1 Static condition

In the static condition:

- EngageVR may collect permitted sensing and task data;
- model inference may run if required for observational comparison;
- uncertainty and abstention may be computed;
- adaptation decisions may be logged observationally if the protocol permits;
- task difficulty must not change because of model-driven adaptation.

The defining property is:

> the participant's environment does not receive model-driven difficulty
> changes.

A static condition should not be implemented simply by disabling all software
components if doing so creates systematic differences unrelated to adaptation.

### 6.2 Adaptive condition

In the adaptive condition:

- the same permitted sensing and task pipeline may operate;
- model estimates may be produced;
- signal-quality and uncertainty requirements remain active;
- predictions may abstain;
- the adaptation gate may block action;
- the conservative policy may propose a difficulty change;
- an approved and technically validated command may modify task difficulty.

The adaptive condition does not guarantee that adaptations occur.

A valid adaptive session may contain no adaptations if the policy's evidence
requirements are not met.

## 7. Experimental Condition vs Safety Lock

Two concepts must remain separate.

### Experimental assignment

The session is assigned to:

```text
static
```

or:

```text
adaptive
```

### Experimenter control

A separate safety control may enable or disable adaptation.

An adaptive session in which the experimenter disables adaptation is not
automatically equivalent to a normal static-condition session.

The event should be:

- timestamped;
- recorded;
- reason-coded where possible;
- handled according to the predefined analysis rules.

## 8. Condition Order

For a two-condition within-participant design, candidate orders are:

```text
Sequence A: Static -> Adaptive
Sequence B: Adaptive -> Static
```

Participants should be allocated to orders using a predefined counterbalancing
or randomization procedure where feasible.

Condition order must be recorded.

## 9. Why Counterbalancing Is Needed

Potential order effects include:

- practice;
- task learning;
- fatigue;
- habituation;
- expectation;
- familiarity with task controls;
- familiarity with webcam/sensing procedures;
- carryover from previous difficulty exposure.

Without order control, a difference between conditions could partly represent a
first-session/second-session effect rather than adaptation.

## 10. Session Structure

A candidate participant session may use the following sequence:

```text
Participant arrival / setup
        |
        v
Information and approved consent procedure
        |
        v
Eligibility / safety checks
        |
        v
Equipment and webcam setup
        |
        v
Baseline / calibration period
        |
        v
Practice block
        |
        v
Condition 1
        |
        v
Short recovery / transition period
        |
        v
Condition-specific questionnaire
        |
        v
Condition 2
        |
        v
Condition-specific questionnaire
        |
        v
Post-session questionnaire / debrief
        |
        v
Session close and data-integrity check
```

The exact timing remains to be finalized.

## 11. Baseline and Calibration Period

A baseline or calibration period may serve several distinct purposes.

These must not be conflated.

Potential purposes include:

- participant-specific feature normalization;
- model personalization;
- task familiarization;
- resting physiological reference;
- basic camera-quality verification.

The final protocol must state which purposes actually apply.

Calibration data used for personalization must precede the evaluation data to
which the personalization is applied.

Future evaluation data must not leak into baseline construction.

## 12. Practice Block

A practice period should be considered before experimental trials.

Its purpose may include:

- learning task controls;
- understanding response requirements;
- reducing early procedural errors;
- checking that the participant can comfortably complete the task.

Practice data should not automatically be mixed with confirmatory experimental
data.

Whether practice data are retained for diagnostic purposes must be defined in
the protocol.

## 13. Condition Duration

Condition duration has not yet been finalized.

It should be long enough to:

- produce sufficient task observations;
- permit model windows to accumulate;
- permit the adaptation policy to operate;
- obtain meaningful participant feedback;

while remaining short enough to reduce:

- fatigue;
- discomfort;
- excessive burden;
- deterioration in signal quality caused by prolonged participation.

The duration should be justified before data collection rather than selected
after inspecting results.

## 14. Starting Difficulty

The initial difficulty level should normally be controlled across conditions.

If static and adaptive conditions begin at different difficulty levels, the
condition comparison becomes confounded.

The protocol should therefore define:

- initial difficulty;
- valid difficulty range;
- whether practice performance affects the initial level;
- whether participant-specific initialization is allowed.

If participant-specific initialization is used, the same initialization logic
should apply consistently across relevant conditions.

## 15. Adaptive Difficulty Changes

The current adaptation policy is conservative.

Participant-facing use should preserve the implemented safeguards, including
where applicable:

- evidence gating;
- abstention;
- persistence;
- cooldown;
- difficulty bounds;
- direction-change control;
- adaptation budget;
- duplicate-window protection;
- experimenter disablement.

The policy configuration used in the study should be frozen and versioned.

Thresholds must not be changed during data collection simply because observed
participant behaviour is unexpected.

## 16. Adaptation Lifecycle

The experimental record should preserve the distinction between:

```text
proposed
command built
dispatched
acknowledged
applied
rejected
```

where supported by the actual runtime.

A proposal that was never delivered to the participant is not an applied
adaptation.

Analysis of participant exposure should use the latest lifecycle state actually
observed rather than infer that every proposal became an environmental change.

## 17. Candidate Primary Outcome

The primary outcome has not yet been selected.

Candidate performance outcomes include:

- task accuracy;
- reaction time;
- completion time where appropriate.

The final statistical-analysis plan must define the primary outcome before
confirmatory analysis.

Primary-outcome selection should consider whether adaptive difficulty changes
the interpretation of the metric.

For example, higher accuracy produced by substantially easier task exposure
cannot automatically be interpreted as improved engagement or superior
performance.

## 18. Secondary Outcomes

Potential secondary outcomes include:

- other task-performance metrics;
- subjective engagement;
- perceived cognitive load or mental effort;
- adaptation acceptance;
- comfort;
- disruption;
- prediction availability;
- abstention rate;
- number of applied adaptations;
- signal-quality summaries.

Secondary outcomes should be labelled as such before analysis.

## 19. Adaptation Appropriateness Outcomes

Before claiming benefit, the study may need to assess whether participants
consider delivered adaptations appropriate.

Candidate questions include whether a difficulty change was:

- noticeable;
- understandable;
- appropriately timed;
- too frequent;
- disruptive;
- helpful;
- uncomfortable.

The final questionnaire instrument remains to be selected.

## 20. Participant-Reported Measures

The experiment may include subjective measures of:

- engagement;
- cognitive load or mental effort;
- comfort;
- adaptation acceptance.

Instrument selection must consider:

- construct validity;
- administration burden;
- permitted use;
- scoring;
- timing;
- comparability with relevant literature.

Questionnaire wording should not be invented and presented as a validated scale.

## 21. Behavioural and Task Measures

Candidate task-derived measures include:

- correctness;
- reaction time;
- error rate;
- completion time;
- inactivity;
- current difficulty;
- difficulty transition history.

These measures should retain participant, condition, trial, and timestamp
provenance.

## 22. Physiological and Webcam Measures

Where technically and scientifically justified, supporting measurements may
include:

- rPPG-derived heart-rate estimate;
- rPPG signal quality;
- motion diagnostics;
- illumination diagnostics;
- behavioural features;
- head-pose or movement features.

A poor-quality physiological window should become unavailable rather than be
assigned a fabricated value.

Physiological variables should not be treated as direct ground truth for
engagement or cognitive load.

## 23. Reference Physiological Measurement

If physiological agreement is a study objective, an appropriate reference
device should be considered.

Webcam rPPG must not be treated as its own validation reference.

Reference-device validation may be conducted as a separate component study if
combining it with the main adaptation experiment would create excessive
participant burden or methodological complexity.

## 24. Signal Failure During a Session

The protocol must define how temporary or sustained sensing failure is handled.

Possible cases include:

- face not detected;
- rPPG quality insufficient;
- webcam unavailable;
- required task telemetry unavailable;
- model unable to produce a valid estimate.

The system should prefer:

```text
unavailable / abstain / HOLD
```

over fabricating evidence.

The participant-facing task should remain in a safe defined state.

## 25. Adaptation Failure During a Session

Possible failures include:

- proposal cannot be converted into a command;
- command cannot be dispatched;
- acknowledgement not received;
- requested level invalid;
- environment refuses the change.

Such failures must be recorded separately from:

- model abstention;
- policy HOLD;
- experimenter intervention.

The analysis must not treat a failed adaptation as an applied exposure.

## 26. Experimenter Intervention

An experimenter must be able to stop or disable adaptation where required by the
final safety procedure.

Potential intervention reasons may include:

- participant discomfort;
- participant request;
- technical malfunction;
- inappropriate repeated behaviour;
- protocol deviation;
- safety concern.

Experimenter intervention should be recorded without requiring unnecessary
personal information.

## 27. Participant Withdrawal

The final approved protocol must define withdrawal procedures.

Participants should be able to discontinue participation according to the
approved consent process.

The analysis plan must state how partial-session data are handled.

A partial or interrupted recording must not automatically be labelled a failed
participant session.

## 28. Randomization and Allocation

The exact allocation mechanism remains to be finalized.

For a two-sequence crossover design, an appropriate method should assign
participants between:

```text
Static -> Adaptive
```

and:

```text
Adaptive -> Static
```

The procedure should be documented and reproducible.

Allocation should not be changed after observing participant outcomes.

## 29. Blinding

Full participant blinding may be difficult because changes in difficulty can be
perceptible.

The final protocol should therefore state explicitly:

- what participants are told about adaptation;
- whether they know which condition is active;
- whether the experimenter knows the condition;
- whether outcome analysis can be performed using condition-coded data.

The study must not claim blinding that the design cannot realistically provide.

## 30. Carryover and Recovery

Because the proposed design exposes each participant to both conditions,
carryover should be considered.

Potential mitigation includes:

- short recovery periods;
- task reset;
- standardized starting difficulty;
- separate trial blocks;
- counterbalanced order.

The required washout or recovery duration, if any, has not yet been established.

## 31. Data Synchronization

All study data should use the project's common timestamp and session
infrastructure.

The analysis should be able to relate:

- task events;
- sensing windows;
- model predictions;
- abstention decisions;
- adaptation proposals;
- commands;
- acknowledgements;
- subjective responses.

Synchronization errors must not be silently corrected by arbitrary row
alignment.

## 32. Data Provenance

Every observation used in analysis should preserve enough provenance to identify:

- participant pseudonym;
- session;
- condition;
- trial or window;
- source modality;
- timestamp;
- data source;
- relevant software/model/configuration version.

Participant identity information should remain outside the modelling dataset.

## 33. Study Configuration Freeze

Before participant data collection begins, the study should freeze the relevant
software and configuration state.

This may include:

- EngageVR software version;
- task build;
- protocol version;
- feature schema;
- model version;
- model configuration;
- uncertainty thresholds;
- adaptation configuration;
- questionnaire version;
- data schema.

Any change during a study should be documented as a protocol or implementation
change rather than silently introduced.

## 34. Exclusion Criteria

Exact exclusion criteria remain to be defined.

Potential categories may include:

- participant withdrawal;
- predefined eligibility violation;
- irrecoverable task failure;
- corrupted session;
- insufficient task exposure;
- predefined minimum-quality failure for a specific analysis.

Low-quality physiological data should not necessarily cause exclusion from all
other analyses.

Exclusions should be analysis-specific where appropriate.

## 35. Missing Data

The analysis should distinguish at least:

- not collected;
- technically unavailable;
- quality-gated unavailable;
- model abstention;
- participant non-response;
- participant withdrawal;
- excluded by protocol.

Missing values must not be replaced with zero merely to simplify analysis.

## 36. Sample Size

No participant count is currently asserted.

Sample size should be justified before recruitment using the final:

- primary outcome;
- study design;
- statistical analysis;
- effect-size assumptions or precision target;
- feasibility constraints.

A convenient arbitrary number of participants should not be presented as a
statistically justified sample.

## 37. Statistical Analysis Dependency

This document defines study structure, not the final statistical model.

`docs/STATISTICAL_ANALYSIS_PLAN.md` must later specify:

- analysis population;
- primary outcome;
- statistical test/model;
- treatment of repeated measures;
- effect-size reporting;
- confidence intervals;
- missing-data handling;
- exclusions;
- multiplicity;
- sensitivity analyses.

The analysis plan should be finalized before confirmatory outcome analysis.

## 38. Preliminary Pilot Phase

A small pilot phase may be appropriate before the main controlled comparison.

Its purpose would be engineering and procedural validation, for example:

- task duration feasibility;
- participant comprehension;
- webcam setup;
- telemetry integrity;
- live adaptation transport;
- adaptation appropriateness;
- questionnaire burden;
- discomfort detection.

Pilot data should not automatically be pooled into the confirmatory dataset.

Whether pilot participants may enter the final analysis should be decided in
advance.

## 39. Main Controlled Phase

After prerequisites and pilot criteria are satisfied, the main proposed study
would compare static and adaptive conditions under the finalized protocol.

A conceptual flow is:

```text
Eligible participant
        |
        v
Baseline / calibration
        |
        v
Practice
        |
        +------------------------------+
        |                              |
        v                              v
Sequence A                         Sequence B
Static first                       Adaptive first
        |                              |
        v                              v
Transition / assessment           Transition / assessment
        |                              |
        v                              v
Adaptive second                    Static second
        |                              |
        +--------------+---------------+
                       |
                       v
              Post-session assessment
                       |
                       v
                 Session close
```

## 40. Interpretation Rules

A result must not be interpreted beyond the evidence collected.

Examples:

### If adaptive accuracy is higher

This does not by itself prove greater engagement.

The difference may also relate to:

- changed difficulty;
- practice;
- order;
- adaptation exposure;
- participant heterogeneity.

### If subjective engagement is higher

This does not prove physiological engagement estimates were valid.

### If fewer predictions are produced

This may reflect conservative abstention rather than system failure.

### If no condition difference is found

This is a valid result and must not be hidden.

### If adaptation is disliked

Technical correctness does not override participant acceptability.

## 41. Safety and Stop Criteria

The final protocol should define explicit stop criteria before participant use.

Potential categories include:

- participant request;
- discomfort;
- simulator sickness where applicable;
- technical instability;
- experimenter safety judgment;
- repeated inappropriate adaptation;
- loss of critical task control.

Exact criteria require institutional and laboratory review.

## 42. Privacy Requirements

The study should follow EngageVR's privacy-preserving defaults.

These include:

- pseudonymous participant identifiers;
- no identity recognition;
- local webcam processing;
- raw-video recording disabled by default;
- minimal data collection;
- no participant data committed to Git.

Any deviation from these defaults must be explicitly justified and reviewed.

## 43. Experimental Records

A complete participant session should, where applicable, provide sufficient
records to reconstruct:

- assigned condition;
- condition order;
- task progression;
- model outputs;
- uncertainty state;
- abstention;
- adaptation decisions;
- adaptation lifecycle;
- experimenter interventions;
- subjective measurements;
- session completion status.

Reconstruction must not require inference from directory names alone.

## 44. Deviations

Protocol deviations should be recorded explicitly.

Examples include:

- wrong condition loaded;
- task restarted;
- configuration changed;
- sensor replaced;
- experimenter disabled adaptation;
- session interrupted;
- participant skipped a questionnaire.

A deviation should not automatically imply that all session data are unusable.

The statistical-analysis plan should define the consequences of relevant
deviation categories.

## 45. Relationship to Other EngageVR Research Questions

The static-versus-adaptive study principally addresses RQ4.

It may generate supporting data relevant to:

- RQ2 — personalization;
- RQ3 — uncertainty gating;
- RQ6 — physiological estimates and subjective feedback;
- RQ7 — missing modalities;
- RQ9 — adaptation strategies;
- RQ10 — sufficient-evidence detection.

However, the study should not claim to answer all of these simply because the
corresponding variables were recorded.

Each question requires an appropriate analysis and valid evidence.

## 46. Decisions Still Required

Before the design becomes an executable protocol, the following must be
finalized:

1. participant population;
2. inclusion criteria;
3. exclusion criteria;
4. task implementation;
5. Unity versus non-Unity participant environment;
6. participant-facing adaptation transport;
7. condition duration;
8. practice duration;
9. baseline/calibration procedure;
10. initial difficulty;
11. condition-order allocation;
12. transition/recovery period;
13. primary outcome;
14. secondary outcomes;
15. subjective instruments;
16. physiological reference-device use, if any;
17. sample-size justification;
18. statistical-analysis plan;
19. stop criteria;
20. consent procedure;
21. risk controls;
22. privacy and retention procedure.

These decisions should be made prospectively.

## 47. Current Status

This document defines the proposed static-versus-adaptive experiment at the
research-design level.

It does not claim that:

- participant recruitment is authorized;
- institutional approval exists;
- the proposed design is final;
- Unity is participant-ready;
- policy-generated live adaptation has been validated;
- the required models are scientifically validated;
- a sample size has been justified;
- a primary outcome has been selected;
- adaptation is beneficial.

Those remain requirements for later Milestone 11 documents and future approved
research.
