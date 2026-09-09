# EngageVR Ethics and Privacy Strategy

> **Draft requiring institutional review**
>
> This document defines proposed ethical and privacy principles for future
> EngageVR research. It is not an ethics approval, institutional determination,
> consent authorization, or legal opinion.
>
> No human-subject study described by this repository has been conducted.
> Participant recruitment or data collection must not begin until the relevant
> study materials have undergone appropriate institutional review.

## 1. Purpose

EngageVR processes behavioural, task-performance, and potentially physiological
information in order to study engagement estimation, cognitive-load estimation,
uncertainty-aware inference, and conservative virtual-environment adaptation.

These data may be sensitive even when direct identifiers are absent.

This document therefore defines the privacy and ethical principles that should
govern future participant-facing use of EngageVR.

It should be read alongside:

- `docs/RESEARCH_PROPOSAL.md`;
- `docs/HYPOTHESES.md`;
- `docs/EXPERIMENTAL_VARIABLES.md`;
- `docs/EXPERIMENT_DESIGN.md`;
- `docs/DATA_COLLECTION_PROTOCOL.md`;
- `docs/LIMITATIONS.md`.

## 2. Current Ethical Status

EngageVR is currently a research software prototype.

At the time of this document:

- no EngageVR human-subject study has been completed;
- no institutional or ethical approval is claimed;
- no participant-labelled EngageVR dataset has been collected;
- current internal modelling demonstrations are synthetic software checks;
- adaptation has not been demonstrated to benefit participants;
- personalization has not been demonstrated to benefit participants.

Engineering completeness does not establish ethical or scientific readiness for
human-subject experimentation.

## 3. Core Ethical Principles

Future EngageVR studies should be designed around:

- informed and voluntary participation;
- data minimization;
- participant autonomy;
- the ability to withdraw according to the approved protocol;
- avoidance of unnecessary collection;
- transparent description of model limitations;
- conservative intervention;
- privacy-preserving defaults;
- explicit missingness and uncertainty;
- separation of research evidence from software demonstrations;
- appropriate institutional oversight.

## 4. Research Prototype Boundary

EngageVR must not be presented to participants, researchers, or users as:

- a medical device;
- a diagnostic system;
- a psychological assessment;
- a clinical monitor;
- a validated detector of engagement;
- a validated detector of cognitive load;
- an emotion-reading system;
- a system capable of discovering a participant's true internal mental state.

Its outputs are estimates, proxies, model predictions, and quality/uncertainty
indicators.

## 5. Physiological Interpretation

Webcam rPPG and future physiological sensors require particularly careful
interpretation.

An rPPG-derived heart-rate estimate:

- is a signal-processing estimate;
- may be unavailable;
- may degrade under motion or illumination variation;
- has not been demonstrated to be medically accurate by EngageVR.

HR or HRV values must not be described as direct measurements of engagement.

No physiological signal should be interpreted in isolation as proving:

- disengagement;
- anxiety;
- stress;
- fatigue;
- cognitive overload;
- emotional state.

## 6. Behavioural Interpretation

Facial geometry, blink proxies, mouth geometry, head pose, and movement are
behavioural measurements or proxies.

They must not be described as revealing:

- a participant's true emotion;
- psychological diagnosis;
- intent;
- personality;
- mental-health condition.

Any relationship to engagement or cognitive load remains an empirical research
question.

## 7. Data Minimization

Only information required for a defined research purpose should be collected.

The fact that EngageVR can technically record a variable is not sufficient
justification for collecting it from participants.

Before collection, every participant-data field should have:

- a defined research purpose;
- a defined storage location;
- a defined access policy;
- a defined retention treatment;
- a defined analysis role where applicable.

## 8. Pseudonymous Identification

Research datasets should use pseudonymous participant identifiers.

Direct identifiers should not be used as modelling identifiers.

Examples of information that should remain outside modelling datasets include:

- participant name;
- email address;
- telephone number;
- student number;
- government identifiers;
- social-media identity.

If administrative contact information is required, it should be stored
separately under the approved study procedure.

## 9. Pseudonymization Is Not Anonymization

Pseudonymous behavioural or physiological data may still carry privacy risk.

The project must not claim that replacing a participant's name with an ID makes
the dataset anonymous.

Potential re-identification or linkage risks should be considered in the final
data-management procedure.

## 10. Webcam Privacy

EngageVR's default privacy model is:

> process webcam frames locally and avoid storing raw video.

Raw webcam recording should remain disabled by default.

Where derived features are sufficient for the research question, they should be
preferred over retained images or video.

## 11. Raw Video Exception

A future study may enable raw image or video retention only when there is a
specific research justification.

Before doing so, the approved protocol should define:

- why raw imagery is necessary;
- what imagery is recorded;
- how participants are informed;
- where it is stored;
- who may access it;
- whether it leaves the collection machine;
- how long it is retained;
- how it is deleted;
- whether derived data can remain after deletion.

Enabling raw video must be an explicit study decision rather than an unnoticed
configuration change.

## 12. Local Processing

Where practical, webcam processing should remain local to the collection
machine.

The default research workflow should not require webcam frames to be sent to:

- external cloud APIs;
- third-party recognition systems;
- advertising platforms;
- unrelated remote services.

Any future external processing would require explicit technical, privacy, and
institutional justification.

## 13. Identity Resolution

EngageVR should not perform participant identity recognition.

The research system does not require facial recognition or identity matching.

Face detection or landmark extraction must not be repurposed into identity
resolution without a separate research justification and review.

## 14. Participant Data and Git

Participant data must not be committed to Git.

This includes, where applicable:

- participant recordings;
- webcam video;
- images;
- physiological recordings;
- questionnaire responses;
- identifiable administrative data;
- participant-labelled modelling datasets.

Source control is for software and appropriate non-participant documentation,
not participant-data storage.

## 15. `.gitignore` Is Not A Privacy Control

A file being ignored by Git does not mean that it is safely stored.

Privacy protection also requires decisions concerning:

- filesystem permissions;
- device security;
- access control;
- backups;
- copying;
- synchronization;
- deletion;
- retention;
- transfer.

## 16. Storage

No participant-data storage location is approved by this draft.

Before participant collection, the final study documentation must specify:

- primary storage;
- permitted devices;
- directory/access permissions;
- backup location;
- encryption where required;
- authorized users.

The development repository itself must not be treated as the participant-data
repository.

## 17. Access Control

Participant data should be accessible only to people authorized by the final
approved study procedure.

Access should follow the principle of least privilege.

The study should avoid broad shared-directory permissions or unnecessary copies
of participant data.

## 18. Retention

This draft does not invent a participant-data retention duration.

The final retention policy should be established according to:

- institutional requirements;
- consent language;
- study purpose;
- applicable policy;
- data sensitivity;
- publication/research requirements where appropriate.

## 19. Deletion

The final study procedure should describe how participant data can be deleted
where required.

Deletion planning should account for:

- primary files;
- derived datasets;
- analysis exports;
- backups;
- temporary copies.

The consent process must accurately state any limits on deletion.

## 20. Backups

Participant-data backups, if required, should preserve the same privacy
expectations as primary storage.

Backups should not silently create uncontrolled additional copies.

The final procedure should define:

- backup destination;
- access;
- retention;
- deletion;
- restoration process.

## 21. Data Transfer

EngageVR is designed as a local-first research system.

If participant data must later be shared with another authorized researcher or
institution, the procedure should define:

- recipient;
- permitted subset;
- transfer mechanism;
- protection during transfer;
- destination storage;
- destination retention;
- institutional authorization.

Participant data must not be transferred simply because a convenient consumer
file-sharing service exists.

## 22. Public Release

Participant-level raw data must not automatically be released publicly.

Any future public dataset release would require separate consideration of:

- consent;
- institutional requirements;
- privacy risk;
- de-identification;
- licensing;
- data documentation;
- withdrawal implications.

The existence of a GitHub repository does not imply that participant data will
be open data.

## 23. Synthetic Data

Synthetic data may be committed or distributed when appropriate because it
contains no real participant observations.

However, it must remain clearly labelled.

Synthetic data must not be used to claim:

- human engagement-estimation accuracy;
- human cognitive-load-estimation accuracy;
- participant benefit;
- adaptation effectiveness;
- physiological validity.

## 24. Public Datasets

Public datasets must retain their own:

- provenance;
- license;
- permitted-use conditions;
- participant limitations;
- labels;
- collection protocol.

A dataset being downloadable does not automatically make every use ethically or
legally appropriate.

## 25. Sensitive Inferences

The project should minimize unnecessary inference about participants.

EngageVR should not expand its output vocabulary into unrelated sensitive
inferences simply because multimodal data exist.

Examples outside the current research scope include:

- identity;
- mental-health diagnosis;
- personality;
- political orientation;
- sexual orientation;
- deception detection;
- employability;
- academic ability.

Such inference would require an entirely different justification and review.

## 26. Missingness

Unavailable data must remain unavailable.

The system should explicitly represent:

- signal unavailable;
- feature unavailable;
- poor quality;
- prediction unavailable;
- abstention.

Missing information must not be silently converted into values that could
misrepresent the participant.

## 27. Signal Quality

Signal quality describes measurement usability.

It must not be interpreted as participant state.

For example:

```text
poor rPPG quality
```

does not mean:

```text
low engagement
```

## 28. Model Uncertainty

Predictive uncertainty should remain visible to researchers.

A model should be allowed to abstain.

Uncertain predictions should not be hidden merely because a complete-looking
dashboard is visually preferable.

## 29. Adaptation Ethics

Adaptive systems can affect participant experience.

Therefore, model-driven adaptation should be treated as an intervention rather
than as a neutral software display.

The system should not apply an adaptation solely because a prediction exists.

## 30. Conservative Adaptation

The current EngageVR policy includes safeguards such as:

- evidence gating;
- abstention;
- persistence;
- cooldown;
- difficulty bounds;
- adaptation budget;
- direction-change controls;
- experimenter disablement.

These are engineering safety mechanisms.

They do not prove that an adaptation is ethically appropriate or beneficial to
participants.

## 31. Adaptation Appropriateness Before Benefit

Participant-facing adaptation should first be evaluated for whether it is:

- technically correct;
- understandable;
- tolerable;
- acceptable;
- non-disruptive under predefined criteria.

Only after basic appropriateness is established should stronger claims about
benefit be investigated.

## 32. Experimenter Control

An experimenter should be able to stop or disable participant-facing adaptation
according to the final study procedure.

The software must not prevent an experimenter from responding to:

- participant discomfort;
- participant request;
- technical malfunction;
- unexpected controller behaviour;
- safety concern.

## 33. Participant Autonomy

Participation must be voluntary under the final approved consent procedure.

The design should not imply that participants must tolerate an adaptation or
complete a session against their wishes.

Withdrawal must be addressed explicitly in the consent and study protocol.

## 34. Informed Consent

The final consent process should explain, in language appropriate to the target
participant population:

- the purpose of the study;
- what the participant will do;
- what data will be collected;
- webcam use;
- physiological estimation where applicable;
- adaptation behaviour;
- foreseeable risks/discomforts;
- voluntary participation;
- withdrawal;
- privacy/data handling;
- researcher contact information as required.

This document does not substitute for the future consent form.

## 35. Avoiding Misleading Terminology

Participant-facing materials should avoid unsupported claims such as:

- "we measure your true engagement";
- "the system knows when you are stressed";
- "AI understands your emotions";
- "the model knows when you are overloaded."

More accurate language includes:

- "the system computes experimental estimates";
- "webcam-derived signals may be unavailable";
- "the model may abstain";
- "the research is evaluating whether these estimates are useful."

## 36. Incidental Findings

EngageVR is not designed as a medical screening system.

The protocol should not promise detection of health conditions from webcam or
physiological data.

If future reference sensors could reveal unexpected physiological information,
the institutional protocol should specify how such situations are handled
rather than allowing researchers to improvise medical interpretation.

## 37. Participant Discomfort

Potential participant discomfort may include:

- visual fatigue;
- cognitive fatigue;
- frustration;
- task difficulty;
- adaptation-related disruption;
- simulator sickness if immersive VR is later used;
- discomfort associated with physiological sensors.

Detailed risk controls belong in `docs/RISK_ASSESSMENT.md`.

## 38. Vulnerable Populations

This draft does not define recruitment of vulnerable populations.

Any future inclusion of participants requiring additional protections must be
explicitly justified and reviewed.

The project should not assume that a general adult protocol automatically
covers every population.

## 39. Recruitment

Recruitment procedures have not yet been defined.

Future recruitment must avoid:

- coercion;
- misleading benefit claims;
- overstating scientific validity;
- implying medical or psychological assessment.

The final recruitment language should be reviewed with the other human-subject
documents.

## 40. Compensation

No participant compensation scheme is specified in this draft.

Any future compensation should be defined prospectively and reviewed according
to institutional requirements.

Compensation should not be represented as a guaranteed research benefit.

## 41. Participant Population

The final participant population has not yet been selected.

The eventual protocol should justify:

- age range;
- inclusion criteria;
- exclusion criteria;
- recruitment source;
- sample-size target.

No demographic variable should be collected without a research or procedural
reason.

## 42. Fairness and Representation

Model performance may vary between participant groups.

Future evaluation should consider whether the study population is sufficiently
documented to understand limitations in generalization.

This does not justify indiscriminate collection of sensitive demographics.

Any demographic variable should have:

- a research purpose;
- appropriate consent;
- an analysis plan;
- privacy consideration.

## 43. Webcam and Skin-Related Measurement Limitations

Camera-based physiological and facial measurements may perform differently
under different:

- illumination;
- motion;
- camera hardware;
- skin appearance;
- environmental conditions.

These limitations should be investigated and reported rather than hidden behind
aggregate performance metrics.

Any future fairness evaluation must be designed carefully and must not infer
sensitive attributes from participants merely to create subgroup labels.

## 44. Dataset Bias

Public datasets may contain demographic, procedural, or hardware biases.

A model trained on such data must not be described as universally valid.

Dataset cards should record known population and acquisition limitations.

## 45. Model Bias

Different model families, feature sets, and calibration procedures may behave
differently across participants.

Personalization is not automatically a fairness solution.

The population prediction should remain available for comparison where the
research design requires it.

## 46. Human Oversight

EngageVR's current research design requires human oversight.

The adaptation policy is not intended to operate as an autonomous high-stakes
decision system.

The experimenter should retain authority to:

- disable adaptation;
- stop the session;
- respond to participant concerns;
- classify technical failures;
- document protocol deviations.

## 47. Researcher Interpretation

Researchers must not infer participant states beyond what the measurement and
study design support.

A researcher viewing:

- signal quality;
- engagement estimate;
- cognitive-load estimate;
- model confidence;
- adaptation history;

must maintain the distinctions defined by the project.

## 48. Dashboard Ethics

The dashboard is a research observability tool.

Its visual presentation must not transform uncertain estimates into apparently
certain participant facts.

It should continue to preserve:

- provenance;
- synthetic labels;
- missingness;
- unavailable values;
- uncertainty;
- scientific status.

## 49. No Automated Validation Status

The project should not provide controls that allow a researcher to mark a model
as:

- scientifically validated;
- clinically validated;
- participant approved;

merely through a software action.

Scientific validity comes from evidence, not metadata selection.

## 50. MLOps Boundary

MLflow, DVC, Docker, CI, model manifests, checksums, and reproducibility records
support software engineering.

They do not establish:

- ethical approval;
- informed consent;
- participant safety;
- construct validity;
- medical validity;
- adaptation benefit.

## 51. Research Integrity

Research results should preserve:

- negative findings;
- null findings;
- missing data;
- poor-quality measurements;
- abstentions;
- unsuccessful personalization;
- failed adaptations;
- protocol deviations.

These must not be removed merely to create a cleaner narrative.

## 52. Selective Reporting

The project should not select metrics after seeing results solely because they
support a preferred conclusion.

Primary outcomes and confirmatory analyses should be defined prospectively.

Exploratory findings should be labelled as exploratory.

## 53. Software Changes During a Study

Participant-facing software and configuration should be frozen before the main
study where practical.

Changes during data collection should be:

- documented;
- versioned;
- evaluated for their impact;
- treated as protocol/software deviations where appropriate.

Silent modification creates both scientific and ethical problems.

## 54. Security

The final participant deployment should consider:

- operating-system access;
- filesystem permissions;
- service binding;
- network exposure;
- secrets;
- physical device security;
- backups;
- unauthorized access.

The current localhost research backend is not a hardened external service.

## 55. Network Exposure

The existing backend has been designed as a local research prototype.

It should not be exposed directly to an untrusted external network without
appropriate security architecture.

Participant studies should prefer the minimum network exposure required for the
research task.

## 56. Authentication

The current local prototype does not provide a general production
authentication/authorization architecture.

This limitation must be considered when selecting the participant-data storage
and deployment environment.

## 57. Data Breach or Privacy Incident

The final institutional protocol should define how researchers respond to:

- accidental participant-data exposure;
- unauthorized access;
- accidental upload;
- lost research device;
- unintended raw-video recording;
- unintended data transfer.

Repository documentation cannot replace institutional incident-response
requirements.

## 58. Third-Party Services

A future study should document every third-party service that receives
participant-related data.

The default design should avoid unnecessary third-party processing.

Introducing a service during a study without reviewing its data handling would
change the privacy model.

## 59. External AI Services

Participant webcam frames, physiological recordings, questionnaires, or other
participant research data should not be uploaded to external generative-AI
services merely for convenience.

If a future approved research workflow proposes external automated processing,
that use would require explicit review of:

- necessity;
- consent;
- data-sharing terms;
- privacy;
- retention;
- security;
- institutional requirements.

## 60. Development Tools vs Participant Data

Software-development assistance and participant-data analysis are different
activities.

Code-development tools may be used on source code according to project policy.

Participant data should not be provided to development assistants or external
services unless that processing is explicitly part of an approved research and
data-management procedure.

## 61. Consent and Model Limitations

The participant information process should not require participants to
understand machine-learning mathematics.

It should, however, communicate material facts such as:

- experimental estimates may be wrong;
- measurements may become unavailable;
- adaptation may or may not occur;
- the system is being researched rather than clinically validated.

## 62. Deception

No deception procedure is proposed in the current design.

If a future study requires withholding information about condition assignment
or adaptation behaviour for methodological reasons, that would require explicit
justification and institutional review.

It must not be introduced informally.

## 63. Blinding

The project must not claim participant or experimenter blinding when difficulty
changes make the condition apparent.

Any blinding used in the final study must be described accurately.

## 64. Secondary Use

Participant data collected for one approved purpose should not automatically be
repurposed for unrelated research.

The consent and institutional protocol should define permitted secondary use.

## 65. Data Linkage

Participant data should not be linked to external datasets or online profiles
unless such linkage is explicitly part of the approved research.

Pseudonymous IDs are not permission to reconstruct participant identity.

## 66. Publication

Future publications should report the scientific status accurately.

Publications should distinguish:

- software validation;
- public-dataset component validation;
- pilot data;
- confirmatory participant results;
- exploratory analysis.

Synthetic results must remain labelled as synthetic.

## 67. Reproducibility and Privacy

Research reproducibility does not require public release of sensitive raw
participant data.

Reproducibility may instead be supported through:

- documented protocols;
- schemas;
- analysis code;
- synthetic fixtures;
- configuration;
- derived aggregate results where appropriate;
- controlled-access data where institutionally permitted.

## 68. Dataset Cards

Future dataset cards should document:

- source;
- participant population;
- consent/usage basis where applicable;
- modalities;
- labels;
- limitations;
- privacy considerations;
- permitted use.

Dataset-card completion does not itself authorize data sharing.

## 69. Model Cards

Future model cards should document:

- intended research use;
- training/evaluation data;
- targets;
- metrics;
- calibration;
- limitations;
- known failure cases;
- scientific status.

No current model should be labelled as clinically validated or production
approved.

## 70. Consent Dependency

`docs/CONSENT_TEMPLATE.md` should be drafted only as a template requiring
institutional review.

Its final content depends on unresolved decisions including:

- participant population;
- final task;
- study duration;
- exact sensors;
- raw-video policy;
- reference hardware;
- risks;
- retention;
- withdrawal;
- researcher contacts.

## 71. Risk-Assessment Dependency

`docs/RISK_ASSESSMENT.md` should separately identify:

- participant risks;
- technical risks;
- privacy risks;
- adaptation risks;
- mitigation;
- stop criteria;
- residual risk.

This document establishes principles but does not replace that assessment.

## 72. Decisions Required Before Participant Collection

Before participant recruitment or collection, finalize at least:

1. study population;
2. inclusion/exclusion criteria;
3. participant-facing task;
4. exact sensing modalities;
5. raw-video policy;
6. reference-sensor use;
7. study duration;
8. primary outcome;
9. questionnaires;
10. adaptation exposure;
11. experimenter stop criteria;
12. storage;
13. access control;
14. retention;
15. deletion;
16. backup;
17. transfer;
18. compensation where applicable;
19. recruitment procedure;
20. consent materials;
21. risk assessment;
22. institutional review requirements.

## 73. Current Status

This document is a research-planning draft.

It does not claim:

- ethical approval;
- institutional approval;
- participant consent;
- participant recruitment;
- participant data collection;
- clinical validation;
- psychological validation;
- participant benefit;
- medical accuracy;
- production security.

These remain outside the current validated state of EngageVR.
