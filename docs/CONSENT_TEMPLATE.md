# EngageVR Participant Information and Consent Template

> **DRAFT TEMPLATE — REQUIRES INSTITUTIONAL REVIEW**
>
> This document is a research-planning template only.
>
> It has not been approved for participant recruitment or consent.
> No participant should be enrolled using this draft.
>
> All bracketed fields and unresolved study details must be completed and
> reviewed through the appropriate institutional process before use.

## 1. Study Identification

**Study title:**
EngageVR: An Uncertainty-Aware Multimodal Framework for Personalized
Engagement Estimation and Adaptive Virtual Reality

**Institution:**
[TO BE FINALIZED]

**Principal investigator / faculty supervisor:**
[TO BE FINALIZED]

**Student researcher(s):**
[TO BE FINALIZED]

**Department / laboratory:**
[TO BE FINALIZED]

**Institutional review / protocol identifier:**
[NOT YET ASSIGNED — DO NOT CLAIM APPROVAL]

**Research contact:**
[TO BE FINALIZED]

**Independent participant-rights / ethics contact, if required:**
[TO BE FINALIZED BY INSTITUTION]

## 2. Invitation to Participate

You are being invited to consider taking part in a research study involving an
experimental computer-based task system called EngageVR.

Before deciding whether to participate, please read the information in this
document.

Participation is voluntary.

You may ask questions before deciding whether to participate.

This research system is experimental and is not a medical, psychological, or
diagnostic system.

## 3. Purpose of the Research

EngageVR is a research software prototype designed to investigate whether
multiple sources of information can be used to support experimental estimates
related to engagement and cognitive load during a computer-based task.

Depending on the final approved study, the system may use information such as:

- task performance;
- webcam-derived behavioural features;
- head movement;
- webcam-based remote photoplethysmography (rPPG);
- experimental model estimates;
- model uncertainty;
- participant questionnaires.

The study may also investigate whether a task environment that can adjust its
difficulty differs from a comparable static environment.

The purpose of the study is research evaluation.

The system is not intended to determine a participant's true internal mental
state.

## 4. Why You Are Being Invited

You are being invited because:

[FINAL PARTICIPANT POPULATION AND ELIGIBILITY REASON TO BE INSERTED]

The final approved study must specify:

- participant age range;
- inclusion criteria;
- exclusion criteria;
- recruitment source;
- any additional eligibility requirements.

This draft does not define those criteria.

## 5. Voluntary Participation

Participation is voluntary.

Choosing not to participate should not result in a penalty or loss of benefits
to which you would otherwise be entitled.

You may decide not to participate.

If you begin the study, you may ask to stop according to the final approved
withdrawal procedure.

Any relationship between the researchers and potential participants must be
managed so that participation is not coercive.

## 6. What Participation May Involve

The final approved procedure may include:

1. receiving information about the study;
2. completing eligibility or safety checks;
3. providing consent;
4. being assigned a pseudonymous participant identifier;
5. having the computer/webcam setup checked;
6. completing a baseline or calibration period;
7. completing a practice task;
8. completing a static task condition;
9. completing an adaptive task condition;
10. answering condition-specific questions;
11. completing a final questionnaire or debrief.

The exact study sequence must be finalized before this template is used.

## 7. Proposed Experimental Conditions

The current research design considers two conditions.

### 7.1 Static condition

In the static condition, the task does not change difficulty because of
EngageVR model-driven adaptation.

The research software may still collect approved measurements or run
observational computations if required by the final protocol.

### 7.2 Adaptive condition

In the adaptive condition, EngageVR may use its experimental estimates and
predefined safety/evidence rules to determine whether task difficulty should
change.

An adaptive condition does not mean that the system will necessarily change
difficulty.

The system may decide not to adapt when its evidence is insufficient.

## 8. Current Adaptation Scope

The currently planned initial adaptation mechanism concerns task difficulty.

The first study should not imply that EngageVR automatically controls:

- breaks;
- breathing exercises;
- emotional interventions;
- audio intensity;
- visual complexity;
- medical treatment;
- psychological treatment;

unless such functionality is separately implemented, validated, reviewed, and
included in the approved protocol.

## 9. Study Duration

**Expected total duration:**
[TO BE FINALIZED]

**Approximate duration of each condition:**
[TO BE FINALIZED]

**Break/recovery periods:**
[TO BE FINALIZED]

The final duration must be established before recruitment.

## 10. Computer Task

You may be asked to complete a computer-based cognitive or performance task.

Depending on the final implementation, the task may record information such as:

- responses;
- correct and incorrect answers;
- reaction time;
- task completion;
- task difficulty;
- changes in difficulty.

The exact task and instructions must be documented in the approved protocol.

## 11. Webcam Use

The study may use a webcam during task participation.

The webcam may be used to derive experimental measurements related to:

- whether a face is detected;
- facial landmark geometry;
- eye-closure or blink-related proxies;
- head position;
- head movement;
- signal quality;
- remote photoplethysmography where applicable.

These measurements are research signals and should not be interpreted as
direct measurements of a person's thoughts, emotions, or psychological state.

## 12. Raw Webcam Video

EngageVR is designed so that raw webcam recording is **disabled by default**.

For the currently preferred privacy-preserving study design:

> webcam frames are processed locally and raw video is not retained.

If a future study proposes to store raw webcam video or images, this consent
document must be changed before use to explain:

- exactly what will be recorded;
- why it is necessary;
- storage location;
- access;
- retention;
- deletion;
- whether data leave the collection computer.

Raw video must not be enabled without the corresponding approved participant
information.

## 13. Webcam-Based rPPG

The webcam may be used to estimate a pulse-related signal using a technique
called remote photoplethysmography, or rPPG.

rPPG attempts to estimate small colour changes associated with blood-volume
variation in visible skin regions.

The resulting measurements:

- may be inaccurate;
- may be unavailable;
- may be affected by movement;
- may be affected by illumination;
- may be affected by camera conditions.

EngageVR's rPPG output is not a medical measurement.

It must not be used to diagnose a health condition.

## 14. Heart-Rate and HRV Information

Where the signal is sufficiently usable, the research software may compute an
experimental heart-rate estimate.

Certain heart-rate-variability features may be considered only when the signal
duration and quality are adequate.

These measurements are research estimates.

They are not intended for:

- diagnosis;
- health monitoring;
- clinical decision-making.

The researchers should not interpret them as medical findings.

## 15. Future Reference Sensors

The final study may or may not use additional physiological reference hardware.

Possible future examples include:

- PPG;
- ECG;
- EDA;
- respiration sensors.

**Reference sensors included in this study:**
[TO BE FINALIZED]

If additional participant-connected hardware is used, this template must be
updated with:

- device type;
- participant procedure;
- foreseeable discomfort;
- data collected;
- safety information;
- removal procedure.

## 16. Experimental Model Estimates

EngageVR may produce experimental model outputs related to engagement and
cognitive load.

These outputs are estimates produced by software.

They are not:

- medical conclusions;
- psychological diagnoses;
- measurements of your thoughts;
- proof of your emotional state;
- proof that you are engaged or disengaged.

The research is partly intended to determine whether such estimates are useful
at all.

## 17. Model Uncertainty

The software may report uncertainty or decide not to produce an estimate.

This is expected behaviour.

An unavailable or uncertain model output does not indicate anything negative
about the participant.

## 18. Signal Quality

The software may determine that a webcam or physiological signal is not usable.

Examples may include:

- face not detected;
- excessive movement;
- unsuitable lighting;
- insufficient rPPG quality.

Poor signal quality is a measurement problem.

It must not be interpreted as low engagement or high cognitive load.

## 19. Adaptation

During an adaptive condition, the software may propose changes to task
difficulty.

The system is designed to avoid adaptation when its evidence is insufficient.

Engineering safeguards may include:

- prediction abstention;
- evidence thresholds;
- persistence requirements;
- cooldown;
- bounded difficulty;
- adaptation limits;
- experimenter disable control.

These mechanisms do not guarantee that every adaptation will feel appropriate
or helpful.

## 20. Experimenter Control

The researcher should be able to stop or disable participant-facing adaptation
according to the approved study procedure.

This may occur because of:

- participant request;
- discomfort;
- technical problems;
- unexpected task behaviour;
- safety concern.

## 21. Information That May Be Collected

Subject to the final approved protocol, research data may include:

- pseudonymous participant ID;
- session ID;
- experimental condition;
- task responses;
- accuracy;
- reaction time;
- task difficulty;
- webcam-derived behavioural features;
- head movement;
- rPPG-derived measurements;
- signal-quality indicators;
- model estimates;
- model uncertainty;
- prediction abstention;
- adaptation decisions;
- adaptation events;
- participant questionnaire responses;
- protocol-deviation or technical-failure information.

Only fields required by the approved research should be collected.

## 22. Information That Should Not Be in the Analysis Dataset

The modelling/analysis dataset should not contain unnecessary direct
identifiers such as:

- your name;
- email address;
- telephone number;
- student number;
- government identifier;
- social-media account.

If identity or contact information is required for administration, it should be
stored separately according to the approved study procedure.

## 23. Pseudonymous Data

Research records should use a study participant code rather than your name.

For example:

```text
P0001
```

This reduces direct identification risk.

However, pseudonymous data are not necessarily anonymous.

Behavioural or physiological data may still carry privacy risks.

## 24. Where Data Will Be Stored

**Approved participant-data storage location:**
[TO BE FINALIZED]

**Who will have access:**
[TO BE FINALIZED]

**Backup procedure:**
[TO BE FINALIZED]

**Encryption/security arrangements where applicable:**
[TO BE FINALIZED]

The EngageVR source-code repository itself is not intended to store participant
data.

Participant data must not be committed to Git.

## 25. Data Retention

**Participant-data retention period:**
[TO BE FINALIZED THROUGH INSTITUTIONAL REVIEW]

This draft intentionally does not invent a retention duration.

The final participant information must accurately state how long each relevant
type of data will be kept.

## 26. Data Deletion

The final approved procedure must explain:

- whether participants may request deletion;
- what data can be deleted;
- when deletion is possible;
- how backups are handled;
- whether already aggregated or published results can be removed.

**Final deletion procedure:**
[TO BE FINALIZED]

## 27. Data Sharing

**Will individual-level participant data be shared outside the research team?**
[TO BE FINALIZED]

If yes, the final consent material must explain:

- what data;
- with whom;
- why;
- under what protections;
- whether the data are identifiable, pseudonymous, or otherwise transformed.

Participant data should not be uploaded to external services merely for
convenience.

## 28. Public Release

Participation in this study does not automatically mean that your raw
participant data will be made public.

Any future release of participant-level data would require an appropriate
consent, privacy, and institutional basis.

Software, schemas, synthetic examples, or aggregate research results may be
shared separately according to the final research plan.

## 29. Use of External AI or Development Services

Participant webcam data, physiological data, questionnaire responses, or other
participant research data should not be sent to external generative-AI or
software-development services merely for convenience.

If any approved study requires external automated processing of participant
data, that processing must be explicitly described in the final participant
information and institutional data-management procedure.

## 30. Possible Risks and Discomforts

Depending on the final task and hardware, foreseeable risks may include:

- eye strain;
- visual fatigue;
- cognitive fatigue;
- frustration;
- discomfort from changing task difficulty;
- privacy concerns related to webcam use;
- privacy concerns related to behavioural or physiological data;
- temporary discomfort from future physiological sensors;
- simulator sickness if immersive VR hardware is used.

Some of these risks depend on study features that have not yet been finalized.

The final consent document must include only risks relevant to the actual
approved study while not omitting foreseeable material risks.

## 31. Adaptation-Related Discomfort

The adaptive condition may change task difficulty.

A change could feel:

- poorly timed;
- frustrating;
- disruptive;
- too easy;
- too difficult.

You should be able to tell the researcher if you wish to stop according to the
approved study procedure.

## 32. Immersive VR Risk

**Is an immersive VR headset used in this study?**
[YES / NO — TO BE FINALIZED]

If no headset is used, simulator-sickness language should be adjusted
accordingly.

If immersive VR is used, the final consent material must explain the relevant
risks and stop procedures.

## 33. Reference Sensor Discomfort

**Are participant-connected physiological sensors used?**
[YES / NO — TO BE FINALIZED]

If yes, the final document should describe foreseeable effects such as:

- temporary pressure;
- skin irritation;
- setup discomfort;
- movement restriction;

as relevant to the actual hardware.

## 34. Privacy Risk

Even when a study participant code is used, there is some risk of unauthorized
access to research data.

The research team should reduce this risk using the approved:

- data-minimization procedures;
- storage controls;
- access controls;
- pseudonymous identifiers;
- retention procedures.

No system can honestly be described as having zero privacy risk.

## 35. Potential Benefits

You may receive no direct personal benefit from participating.

The study is designed to contribute research knowledge about experimental
multimodal sensing, uncertainty-aware modelling, and adaptive task systems.

Participation should not be described as providing medical, psychological, or
therapeutic benefit.

## 36. Compensation

**Compensation/reimbursement:**
[TO BE FINALIZED]

If there is no compensation, state that explicitly in the final form.

If compensation is provided, the final document must explain:

- amount or form;
- eligibility;
- whether partial participation is compensated;
- any institutional conditions.

## 37. Costs

**Expected participant costs:**
[TO BE FINALIZED]

The final document should state any reasonably foreseeable participant cost or
state that none is expected.

## 38. Alternatives to Participation

The alternative is not to participate.

If the study is associated with a course, institution, or other relationship,
the final recruitment/consent process should clearly explain any equivalent
alternative where required.

## 39. Withdrawal

You may ask to stop participating according to the approved procedure.

**How to stop participation:**
[TO BE FINALIZED]

**What happens to data already collected:**
[TO BE FINALIZED]

**Deletion options/limitations:**
[TO BE FINALIZED]

These details must be finalized before this form is used.

## 40. Researcher-Initiated Stopping

The researcher may stop your participation if required for reasons such as:

- safety;
- significant discomfort;
- technical failure;
- study-procedure requirements.

The final institutional protocol should define these circumstances.

## 41. Questions During the Study

You may ask questions about the research.

**Research contact:**
[TO BE FINALIZED]

Questions concerning participant rights or complaints should use the
institutionally designated contact, where required.

**Independent/institutional contact:**
[TO BE FINALIZED]

## 42. Medical Advice

EngageVR does not provide medical advice.

Researchers should not use the software's experimental physiological estimates
to diagnose you.

If you have health concerns, you should rely on appropriate qualified health
professionals rather than this research software.

## 43. Unexpected Physiological Information

The current research system is not intended to screen for medical conditions.

If future reference physiological hardware is included, the approved protocol
must define whether and how unexpected measurements are handled.

Researchers should not improvise medical interpretation.

## 44. Confidentiality

The research team should make reasonable efforts to protect participant
information according to the approved protocol.

The final version must describe any circumstances under which confidentiality
cannot be guaranteed or disclosure may be required.

**Final confidentiality wording:**
[TO BE FINALIZED THROUGH INSTITUTIONAL REVIEW]

## 45. Future Use of Data

**Will participant data be retained for future related research?**
[TO BE FINALIZED]

If yes, the final form must state:

- which data;
- what types of future use;
- whether additional consent is required;
- whether sharing is permitted;
- withdrawal limitations.

Data must not automatically be repurposed for unrelated research.

## 46. Research Results

Research findings may eventually be included in:

- academic reports;
- presentations;
- dissertations/theses;
- papers;
- software documentation;
- aggregate research summaries.

The final approved process should describe how participant confidentiality is
protected in dissemination.

## 47. Individual Results

**Will individual research results be returned to participants?**
[TO BE FINALIZED]

Experimental engagement or cognitive-load model outputs should not be presented
as medical or psychological conclusions.

## 48. Photography, Video, and Demonstration Material

Raw webcam video is currently intended to remain disabled by default.

Any separate photography, screenshots, video demonstrations, or identifiable
media intended for:

- presentations;
- publications;
- demonstrations;
- social media;
- repository documentation;

requires an explicit approved process.

Research participation alone must not be treated as permission for public
identifiable media use.

## 49. Audio Recording

**Will audio be recorded?**
[YES / NO — TO BE FINALIZED]

If audio is not required, it should not be collected.

If audio is collected, its purpose and handling must be added to this document.

## 50. Deception

No deception procedure is proposed in the current EngageVR study design.

If the final methodology introduces deception or deliberate withholding of
material study information, this template must be revised and reviewed before
use.

## 51. Condition Disclosure and Blinding

Participants may be able to notice changes in task difficulty.

The final study must not claim participant blinding if the experimental
condition can reasonably become apparent.

**Final condition-information procedure:**
[TO BE FINALIZED]

## 52. Participant Responsibilities

If included in the final approved study, participants may be asked to:

- follow task instructions;
- tell the researcher if they experience discomfort;
- avoid intentionally obstructing the webcam unless they wish to stop;
- answer study questions honestly to the extent they are comfortable;
- inform the researcher if they wish to discontinue.

These responsibilities must not limit the participant's right to stop.

## 53. Researcher Responsibilities

The researcher should:

- follow the approved protocol;
- respect participant withdrawal;
- avoid unsupported claims;
- monitor technical operation;
- protect participant data;
- record relevant protocol deviations;
- stop participant-facing adaptation when required by the safety procedure.

## 54. Unresolved Items Checklist

This template must not be used until the following are resolved:

- [ ] institution;
- [ ] supervisor / principal investigator;
- [ ] researcher contacts;
- [ ] institutional-review status;
- [ ] participant population;
- [ ] inclusion/exclusion criteria;
- [ ] recruitment procedure;
- [ ] study duration;
- [ ] task version;
- [ ] desktop vs immersive environment;
- [ ] exact sensing modalities;
- [ ] raw-video policy;
- [ ] reference physiological hardware;
- [ ] questionnaires;
- [ ] adaptation procedure;
- [ ] stop criteria;
- [ ] foreseeable study-specific risks;
- [ ] compensation;
- [ ] participant costs;
- [ ] storage location;
- [ ] authorized data access;
- [ ] retention;
- [ ] deletion;
- [ ] backups;
- [ ] sharing;
- [ ] future use;
- [ ] withdrawal-data procedure;
- [ ] confidentiality wording;
- [ ] participant-rights contact.

## 55. Consent Statements

The final institutional consent form may include statements such as the
following, subject to required institutional wording:

- I have read or received the participant information for this study.
- I have had the opportunity to ask questions.
- I understand that participation is voluntary.
- I understand that I may stop participation according to the stated procedure.
- I understand what types of data the study proposes to collect.
- I understand that EngageVR is experimental research software and is not a
  medical or psychological diagnostic system.
- I understand the foreseeable risks described in the approved participant
  information.
- I understand how my research data will be handled according to the approved
  procedure.
- I voluntarily agree to participate.

These statements are examples only.

The final consent language must follow institutional requirements.

## 56. Optional Consent Items

If required by the final protocol, separate optional consent decisions should
be considered for activities materially different from basic study
participation.

Examples may include:

- raw-video storage;
- future reuse of data;
- controlled data sharing;
- identifiable images/video for publication or demonstration.

Optional permissions should not be bundled unnecessarily into participation in
the core study.

## 57. Signature Section Template

> **DO NOT USE UNTIL APPROVED**

Participant name: ______________________________________

Participant signature: __________________________________

Date: __________________

Researcher obtaining consent: ___________________________

Researcher signature: __________________________________

Date: __________________

Participant/research identifiers used on research data should remain separate
from this directly identifying consent record.

Electronic consent may require a different approved format.

## 58. Version Control

The final participant information and consent form should carry:

- document version;
- date;
- study/protocol identifier where applicable;
- institutional approval/version information where required.

Participants should receive the version approved for their study session.

## 59. Record Separation

Signed consent records contain direct identifying information.

They should not be stored inside the modelling dataset or ordinary session
recording directory.

The approved study should define how consent records are stored and protected.

## 60. Consistency Requirement

Before approval/use, this template must be reconciled with:

- `docs/EXPERIMENT_DESIGN.md`;
- `docs/DATA_COLLECTION_PROTOCOL.md`;
- `docs/ETHICS_AND_PRIVACY.md`;
- `docs/RISK_ASSESSMENT.md`;
- the statistical-analysis plan;
- final hardware procedures.

Participant-facing language must not promise protections or procedures that the
actual study cannot provide.

## 61. Current Status

This document is an **unapproved consent template**.

It does not establish:

- ethical approval;
- institutional approval;
- valid informed consent;
- authorization to recruit participants;
- authorization to collect participant data.

It must be revised using the final study details and the institution's required
consent format before any participant-facing use.
