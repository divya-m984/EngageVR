# EngageVR Research Proposal

> **Draft for research planning and institutional review**
>
> This document describes a proposed future research programme around the
> EngageVR software prototype. No human-subject study described here has been
> conducted. No institutional or ethical approval is claimed. Any study
> involving participants must undergo appropriate institutional review before
> recruitment, data collection, or experimental use.
>
> Current synthetic demonstrations and software tests are engineering
> self-checks only and are not scientific evaluation.

## 1. Project Title

**EngageVR: An Uncertainty-Aware Multimodal Framework for Personalized
Engagement Estimation and Adaptive Virtual Reality**

## 2. Overview

EngageVR is a software-first research framework for investigating multimodal
estimation of engagement and cognitive load and the use of those estimates in
conservative adaptive virtual environments.

The system combines behavioural, physiological, task-performance, signal-
quality, and model-uncertainty information while explicitly allowing
measurements or predictions to be unavailable when evidence is insufficient.

The framework has been designed around four principles:

1. engagement and cognitive load are estimated rather than directly observed;
2. measurement quality and predictive uncertainty must remain distinct;
3. missing or unreliable evidence must not be silently converted into valid
   measurements;
4. environment adaptation must be conservative and may be refused when the
   available evidence is insufficient.

The current software prototype includes behavioural capture, webcam-based
remote photoplethysmography (rPPG), task telemetry, leakage-aware modelling,
multimodal fusion, personalized calibration, uncertainty-aware inference,
abstention, a conservative adaptation policy, session recording and replay,
research dashboards, and reproducibility infrastructure.

The system has not yet been scientifically validated with human participants.

## 3. Background and Motivation

Adaptive virtual environments have the potential to respond dynamically to
changes in user performance and experience. Such adaptation is difficult,
however, because internal states such as engagement and cognitive load cannot
be directly and perfectly measured.

Individual modalities also have important limitations. Facial and head-motion
features may be affected by pose, occlusion, behaviour, and camera conditions.
Webcam rPPG is sensitive to motion, illumination, skin-region tracking, and
signal quality. Task-performance measurements provide behavioural evidence but
do not uniquely identify the reason for a change in performance. Subjective
feedback can provide valuable self-report information but is intermittent and
may itself be influenced by the experimental context.

EngageVR therefore treats multimodal information as complementary evidence
rather than as direct psychological ground truth.

A second challenge concerns model uncertainty. An adaptive environment that
acts on an uncertain estimate may perform an unnecessary or inappropriate
adaptation. EngageVR consequently separates:

- signal quality: whether a measurement is usable;
- model confidence or predictive uncertainty: how certain the model is about
  an estimate;
- multimodal disagreement: whether modality-specific models disagree;
- abstention: whether the system should produce a prediction at all;
- adaptation gating: whether an already-defined action has sufficient evidence
  to proceed;
- adaptation policy: which action, if any, should be proposed.

This separation is central to the proposed research.

## 4. Problem Statement

Many adaptive systems implicitly assume that a predicted user state is
sufficiently reliable to justify an intervention.

This assumption is problematic when:

- sensor quality is poor;
- modalities are unavailable;
- participant-specific baselines differ substantially;
- probability estimates are poorly calibrated;
- multimodal models disagree;
- training and deployment conditions differ;
- the relationship between estimated state and appropriate adaptation has not
  been experimentally established.

The research problem addressed by EngageVR is therefore not simply whether an
engagement or cognitive-load classifier can be constructed.

The broader problem is:

> Can a multimodal, personalized, uncertainty-aware framework estimate
> engagement and cognitive load while recognizing insufficient evidence, and
> can those estimates be used conservatively enough to support experimentally
> testable adaptation without treating uncertain model outputs as established
> facts about the participant?

## 5. Research Aim

The principal aim of EngageVR is:

> To develop and evaluate a modular, uncertainty-aware multimodal framework for
> estimating engagement and cognitive load and for conservatively controlling
> adaptation in an interactive virtual environment.

### 5.1 Supporting aims

The project is designed to investigate:

1. whether multimodal modelling improves upon individual modalities;
2. whether participant-specific calibration improves upon population-level
   inference;
3. whether calibrated uncertainty and abstention reduce decisions made from
   insufficient evidence;
4. how missing modalities affect prediction reliability;
5. how robust webcam-based rPPG is under realistic motion and illumination
   changes;
6. whether a conservative adaptive environment differs from a static
   environment in task-performance and participant-reported outcomes;
7. whether the system can identify circumstances in which it should not
   generate a prediction or adaptation.

## 6. Research Questions

The canonical research questions are maintained in
`docs/RESEARCH_QUESTIONS.md`.

### 6.1 Primary questions

**RQ1 — Multimodal fusion**

Does multimodal fusion of behavioural, physiological, and task-performance
signals produce more reliable engagement and cognitive-load estimates than
individual modalities?

**RQ2 — Personalization**

Do participant-specific baselines and calibration improve engagement and
cognitive-load estimation relative to population-level models?

**RQ3 — Uncertainty and adaptation**

Can uncertainty estimation and selective prediction reduce inappropriate
adaptation when evidence is insufficient?

**RQ4 — Adaptive versus static environments**

Does an adaptive environment differ from a static environment in task
performance and subjective experience?

**RQ5 — rPPG robustness**

How robust is webcam-based rPPG under motion and illumination changes relevant
to the intended experimental setting?

### 6.2 Secondary questions

The broader research programme additionally considers:

- agreement between physiological estimates, behavioural indicators, and
  subjective reports;
- degradation under missing modalities;
- possible future temporal modelling;
- comparison of adaptation strategies;
- reliable detection of insufficient evidence.

These secondary questions should not all be treated as objectives of a single
participant study.

## 7. Proposed Hypotheses

These hypotheses are prospective and have not been confirmed.

### H1 — Multimodal fusion

Multimodal models will demonstrate different, and potentially improved,
predictive performance and/or calibration relative to the strongest
corresponding single-modality baseline.

### H2 — Personalized calibration

Participant-specific calibration will produce measurable differences in
prediction performance and/or calibration relative to the unmodified
population model.

No assumption is made that personalization must improve every participant or
every metric.

### H3 — Selective prediction

As increasingly uncertain predictions are rejected, error among the remaining
accepted predictions is expected to decrease, subject to the corresponding
reduction in prediction coverage.

### H4 — Adaptive versus static condition

Task-performance and/or participant-reported outcomes will differ between
static and adaptive experimental conditions.

The direction and primary outcome of this hypothesis must be finalized before
data collection in the statistical-analysis plan.

### H5 — rPPG measurement robustness

Increased motion and adverse illumination conditions will be associated with
degradation in webcam-rPPG signal quality, availability, and/or agreement with
a suitable reference measurement.

## 8. Current Technical Framework

The current EngageVR architecture can be represented conceptually as:

```text
Webcam and task telemetry
        |
        v
Behavioural / head-pose / rPPG / task features
        |
        v
Signal-quality assessment
        |
        v
Population and multimodal models
        |
        v
Participant-specific calibration
        |
        v
Uncertainty estimation and abstention
        |
        v
Adaptation evidence gate
        |
        v
Conservative adaptation policy
        |
        v
Task-environment command proposal
```

## 9. Current Adaptation Scope

The present adaptation system is intentionally narrower than the complete
long-term vision of EngageVR.

The implemented conservative controller currently reuses the existing
`set_difficulty` task command.

It includes engineering safeguards for:

- persistence across multiple windows;
- cooldown;
- direction changes;
- adaptation bounds;
- adaptation budgets;
- duplicate input;
- experimenter disablement;
- static versus adaptive experimental modes;
- uncertainty/abstention prerequisites.

Policy evaluation and command construction are currently separated from live
transport.

A policy-generated command has not yet been demonstrated as dispatched,
acknowledged, and applied through a validated participant-facing Unity
experiment.

Therefore, the first proposed experimental study should focus on the
implemented difficulty-adaptation mechanism rather than describing broader
adaptations such as automatic break timing, information density, audiovisual
intensity, or feedback modification as existing functionality.

These broader mechanisms remain possible future extensions.

## 10. Proposed Experimental Study

### 10.1 Study purpose

The proposed controlled study would investigate whether EngageVR's adaptive
condition produces measurable differences from a corresponding static
condition while collecting sufficient information to evaluate the sensing,
modelling, uncertainty, and adaptation pipeline.

### 10.2 Provisional design

A within-participant design is proposed as the initial design candidate.

Each participant would experience both:

1. a **static condition**, in which task behaviour does not change in response
   to EngageVR model estimates; and
2. an **adaptive condition**, in which the approved adaptation policy may
   modify task difficulty when its evidence requirements are satisfied.

Condition order should be counterbalanced where practical.

A final design must be specified before data collection and may be revised
following laboratory, statistical, and institutional review.

## 11. Experimental Variables

### 11.1 Independent variable

Primary experimental condition:

- static;
- adaptive.

### 11.2 Candidate primary dependent variables

Potential performance outcomes include:

- task accuracy;
- reaction time;
- completion time.

The statistical-analysis plan must identify the primary outcome before the
study is conducted.

### 11.3 Candidate subjective outcomes

Potential participant-reported outcomes include:

- subjective engagement;
- perceived cognitive load or mental effort;
- adaptation acceptance;
- comfort;
- perceived disruption.

Questionnaire selection and wording must be reviewed for validity and usage
conditions before deployment.

### 11.4 Supporting system measurements

Supporting measurements may include:

- prediction availability;
- abstention rate;
- calibrated classification confidence;
- regression prediction-interval width where applicable;
- signal-quality measures;
- missing-modality state;
- adaptation proposals;
- adaptation timing;
- task difficulty;
- adaptation HOLD reasons.

These measurements must not automatically be interpreted as participant
benefit.

### 11.5 Candidate control variables

Potential experimental controls include:

- task duration;
- task content;
- webcam placement;
- camera settings;
- ambient illumination;
- baseline/calibration procedure;
- condition order;
- participant-specific calibration period;
- task difficulty at condition start.

## 12. Proposed Data Collection

A future approved study may collect:

- pseudonymous participant/session identifiers;
- task-performance events;
- behavioural feature summaries;
- head-pose and movement features;
- rPPG-derived measurements where signal quality permits;
- signal-quality metadata;
- model estimates;
- model uncertainty;
- abstention decisions;
- adaptation decisions;
- adaptation commands and acknowledgements where implemented;
- subjective responses;
- experimental-condition metadata.

Raw webcam video should remain disabled by default.

Where the research objective can be achieved using derived features instead of
stored images or video, the lower-data-exposure option should be preferred.

No participant names, email addresses, or unnecessary direct identifiers
should form part of the modelling dataset.

## 13. Proposed Analysis Strategy

A detailed statistical-analysis plan will be maintained separately.

At a minimum, analysis should consider:

### 13.1 Predictive modelling

- participant-aware and session-aware splitting;
- model performance;
- calibration;
- confusion matrices where applicable;
- regression error where applicable;
- comparison of single-modality and multimodal models.

### 13.2 Personalization

Population predictions and personalized predictions must both be retained.

Any personalized calibration must use information available before the
evaluation window so that participant-specific leakage is avoided.

### 13.3 Selective prediction

Analysis should include:

- prediction coverage;
- performance among accepted predictions;
- abstention frequency;
- false-confident predictions;
- potentially unnecessary abstentions.

Classification confidence and regression uncertainty must be analysed with
their appropriate, separate semantics.

### 13.4 Experimental comparison

Static-versus-adaptive analysis should report:

- the predefined primary outcome;
- appropriate effect sizes;
- uncertainty or confidence intervals;
- within-participant dependence where a repeated-measures design is used;
- missing and excluded observations;
- condition order where relevant.

Statistical significance alone should not be treated as evidence of practical
benefit.

## 14. Privacy and Data Governance

EngageVR is designed around data minimization.

Current and proposed principles include:

- pseudonymous identifiers;
- no identity recognition;
- local webcam processing;
- raw-video recording disabled by default;
- explicit provenance on generated data;
- no participant data committed to Git;
- no secrets or credentials stored with experiment artifacts;
- separation of generated experimental artifacts from source code;
- explicit representation of missing or unavailable measurements.

Before participant data are collected, the study documentation must define:

- what data are collected;
- why each item is necessary;
- who may access the data;
- where data are stored;
- retention duration;
- withdrawal and deletion procedures;
- backup policy;
- whether any data leave the laboratory or local machine.

## 15. Risks and Participant Considerations

Possible risks that require review before a participant study include:

- visual discomfort or fatigue;
- cognitive fatigue from task participation;
- simulator sickness if an immersive VR headset is later used;
- discomfort caused by difficulty changes;
- frustration or disruption caused by inappropriate adaptation;
- privacy concerns associated with webcam-based measurement;
- sensitivity of physiological and behavioural measurements.

Participants must be able to discontinue participation according to the final
approved protocol.

The adaptive system must never prevent an experimenter from stopping or
disabling adaptation.

## 16. Scientific Boundaries

EngageVR is a research software prototype.

Its outputs must not be interpreted as:

- medical measurements;
- psychological diagnosis;
- validated indicators of a person's true internal mental state;
- clinical monitoring;
- proof that a participant is engaged or disengaged.

In particular:

- rPPG-derived heart rate is a signal-processing estimate;
- poor rPPG quality is not evidence of low engagement;
- model confidence is not certainty;
- expert disagreement is not calibrated uncertainty;
- synthetic accuracy is not human-subject performance;
- adaptation frequency is not adaptation effectiveness;
- successful software tests do not establish scientific validity.

## 17. Current Research Status

### 17.1 Engineering status

The software programme has completed substantial implementation work covering:

- webcam behavioural capture;
- facial landmarks;
- head pose and movement;
- interpretable rPPG methods;
- task telemetry and simulation;
- synchronized session recording and replay;
- leakage-aware baseline models;
- multimodal fusion;
- participant-specific calibration;
- uncertainty estimation;
- selective prediction and abstention;
- conservative adaptation logic;
- read-only research dashboards;
- experiment provenance and artifact integrity;
- reproducibility tooling;
- local experiment tracking;
- DVC;
- Docker;
- continuous-integration infrastructure.

### 17.2 Scientific status

The software's engineering maturity must not be confused with scientific
validation.

At the time of this proposal:

- no approved human-subject EngageVR study has been completed;
- no participant-labelled multimodal dataset has been collected by the project;
- current internal modelling demonstrations are synthetic software checks;
- adaptation effectiveness has not been demonstrated;
- personalized calibration has not been demonstrated to benefit real users;
- Unity runtime validation remains a separate laboratory task;
- the relationship between model estimates and participant experience remains
  an open research question.

## 18. Known Limitations

Current limitations include:

1. absence of human-subject validation;
2. absence of an EngageVR participant-labelled multimodal dataset;
3. incomplete physical webcam validation;
4. incomplete public-dataset rPPG evaluation;
5. absence of comparison with a research-grade physiological reference device;
6. unvalidated engineering thresholds for sensing, uncertainty, and adaptation;
7. no demonstrated human benefit from personalization;
8. no demonstrated human benefit or appropriateness from adaptation;
9. no validated regression-to-adaptation bands;
10. no implemented fatigue estimator or automatic break policy;
11. pending Unity compilation/runtime validation;
12. limited hardware availability during software development.

These limitations are research constraints and should remain visible rather
than being hidden through software defaults.

## 19. Expected Contributions

If evaluated successfully, the project may contribute:

1. a modular framework for multimodal engagement and cognitive-load research;
2. explicit separation of sensing quality and predictive uncertainty;
3. participant-specific calibration while retaining population-level
   predictions;
4. selective prediction and abstention for adaptive systems;
5. conservative control that treats insufficient evidence as a reason not to
   act;
6. systematic evaluation of missing-modality robustness;
7. a reproducible software architecture for future adaptive-VR experiments;
8. an experimental platform for comparing static and adaptive task
   environments.

These are intended research contributions and not claims of demonstrated
scientific findings.

## 20. Future Laboratory Extension

Future laboratory work may extend EngageVR using:

- an immersive VR headset;
- research-grade PPG or ECG;
- electrodermal activity;
- respiration;
- synchronized external physiological references;
- controlled lighting and camera placement;
- validated subjective instruments;
- richer task environments;
- additional adaptation actions;
- more extensive accessibility and usability evaluation;
- longitudinal or repeated-session studies;
- temporal modelling where justified by suitable data.

Reference physiological equipment would allow webcam-derived measurements to
be evaluated against more appropriate external measurements rather than being
treated as self-validating.

Any deep or temporal model should be introduced only where a research question
and suitable dataset justify its additional complexity.

## 21. Expected Research Outcome

This work is intended to determine whether EngageVR provides a scientifically
useful platform for studying multimodal, personalized, uncertainty-aware
adaptation.

The proposal does not assume that:

- multimodal models will always outperform individual modalities;
- personalization will always improve predictions;
- adaptive conditions will outperform static conditions;
- webcam rPPG will remain usable in all conditions;
- every participant will respond positively to adaptation.

Negative, null, heterogeneous, or unavailable results are valid outcomes and
must be reported as such.

## 22. Planned Milestone 11 Supporting Documents

This proposal is the umbrella document for the remaining research
documentation.

Planned supporting documents include:

- hypotheses;
- experimental variables;
- static-versus-adaptive experimental design;
- data-collection protocol;
- consent-template draft;
- risk assessment;
- privacy strategy;
- statistical-analysis plan;
- dataset cards;
- model cards;
- limitations;
- hardware-validation plan;
- future laboratory-extension plan.

Human-subject documents remain drafts until reviewed and approved through the
appropriate institutional process.

## 23. References

The repository's research references are maintained in
`docs/REFERENCES.md`.

Method-specific references should be cited from primary papers, official
dataset publications, or authoritative technical sources wherever possible.
