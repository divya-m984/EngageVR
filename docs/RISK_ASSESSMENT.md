# EngageVR Risk Assessment

> **Draft requiring institutional review**
>
> This document identifies foreseeable risks associated with proposed future
> EngageVR research and candidate mitigations. It is a research-planning
> document, not an institutional risk determination, safety certification,
> medical assessment, or ethical approval.
>
> No participant study described here has been conducted. Participant-facing
> experimentation must not begin until the final study design, consent
> materials, stop criteria, privacy procedures, and risk controls have undergone
> the appropriate institutional review.

## 1. Purpose

EngageVR is a research software prototype for studying multimodal engagement
and cognitive-load estimation, uncertainty-aware inference, personalization,
and conservative task adaptation.

Future studies may involve:

- desktop or immersive task environments;
- webcam-derived behavioural measurements;
- webcam-based remote photoplethysmography (rPPG);
- task-performance telemetry;
- subjective questionnaires;
- model-driven task-difficulty adaptation;
- optional future reference physiological sensors.

This document identifies risks that should be considered before those
capabilities are used with human participants.

It should be read alongside:

- `docs/RESEARCH_PROPOSAL.md`;
- `docs/EXPERIMENT_DESIGN.md`;
- `docs/EXPERIMENTAL_VARIABLES.md`;
- `docs/DATA_COLLECTION_PROTOCOL.md`;
- `docs/ETHICS_AND_PRIVACY.md`;
- `docs/LIMITATIONS.md`.

## 2. Risk-Assessment Boundary

This document does not determine whether a future study is:

- exempt;
- minimal risk;
- greater than minimal risk;
- institutionally approved;
- medically safe.

Those determinations belong to the relevant institutional process.

The ratings in this document are engineering planning aids only.

## 3. Risk Categories

The current risk assessment considers:

1. participant physical discomfort;
2. cognitive and task burden;
3. adaptation-related risk;
4. sensing and measurement risk;
5. privacy and confidentiality risk;
6. software and technical failure;
7. research-integrity risk;
8. security risk;
9. consent and autonomy risk;
10. future hardware-specific risk.

## 4. Provisional Risk Scale

For internal planning, risks may be described using qualitative likelihood and
severity.

### 4.1 Likelihood

| Rating | Planning meaning |
|---|---|
| Low | Not expected under normal use, but reasonably possible |
| Medium | Could occur during ordinary study operation |
| High | Expected to occur frequently without mitigation |

### 4.2 Severity

| Rating | Planning meaning |
|---|---|
| Low | Temporary inconvenience or minor procedural impact |
| Medium | Meaningful participant discomfort, privacy impact, or study disruption |
| High | Serious harm, major privacy compromise, or unacceptable participant risk |

These labels are not institutional classifications.

## 5. Residual Risk

A mitigation does not make a risk disappear.

Residual risk should be reconsidered after the final study specifies:

- participant population;
- exact task;
- session duration;
- hardware;
- questionnaires;
- adaptation settings;
- storage;
- consent procedure;
- stop criteria.

Where those details remain unresolved, residual risk must remain provisional.

## 6. Participant Physical Discomfort

### Risk

Participants may experience physical discomfort during task participation.

Potential contributors include:

- prolonged sitting;
- sustained screen viewing;
- repetitive responses;
- constrained webcam position;
- physiological sensor placement in future studies.

### Potential effects

- discomfort;
- eye strain;
- headache;
- muscle fatigue;
- desire to stop participation.

### Mitigations

Candidate mitigations include:

- conservative study duration;
- comfortable seating;
- normal posture rather than forced immobility;
- planned breaks where appropriate;
- participant-controlled withdrawal;
- experimenter stop procedure;
- avoiding unnecessary physical constraints.

### Residual risk

Provisional: **Low to Medium**, depending on final session duration and
equipment.

## 7. Visual Fatigue

### Risk

Extended display use or visually demanding tasks may produce visual fatigue.

### Potential effects

- tired eyes;
- headache;
- reduced comfort;
- reduced task performance.

### Mitigations

- avoid unnecessarily long conditions;
- avoid excessive visual stimulation;
- provide appropriate breaks;
- permit participant-initiated stopping;
- document display conditions.

### Residual risk

Provisional: **Low to Medium**.

## 8. Cognitive Fatigue

### Risk

The research task may intentionally require cognitive effort.

Extended participation or excessive difficulty could produce fatigue.

### Potential effects

- frustration;
- reduced concentration;
- slower responses;
- discomfort;
- withdrawal.

### Mitigations

- bounded task duration;
- bounded difficulty;
- practice period;
- conservative initial difficulty;
- experimenter monitoring;
- participant stop option;
- recovery period between conditions where appropriate.

### Residual risk

Provisional: **Low to Medium**.

## 9. Simulator Sickness

### Current status

The current project can operate in desktop mode and has not yet undergone
validated immersive-VR participant testing.

### Future risk

If an immersive headset is introduced, participants may experience symptoms
consistent with simulator or cybersickness.

Potential symptoms may include:

- nausea;
- dizziness;
- headache;
- disorientation;
- visual discomfort.

### Mitigations

Before immersive VR use:

- use an appropriate VR-specific safety protocol;
- define stop criteria;
- provide immediate removal of the headset;
- avoid unnecessarily provocative motion;
- select task duration conservatively;
- consider an appropriate validated discomfort/sickness instrument;
- include relevant participant information in consent materials.

### Residual risk

**Not currently assessable** until the actual VR environment and hardware are
defined.

## 10. Task Frustration

### Risk

Participants may experience frustration because of:

- task errors;
- unfamiliar controls;
- difficulty increases;
- perceived unfairness;
- repeated failure.

### Mitigations

- practice before experimental trials;
- clear instructions;
- bounded difficulty;
- conservative adaptation;
- stop/withdraw option;
- pilot evaluation of difficulty progression.

### Residual risk

Provisional: **Low to Medium**.

## 11. Inappropriate Adaptation

### Risk

The adaptive system may propose a difficulty change that is poorly timed,
unhelpful, uncomfortable, or inappropriate.

Possible causes include:

- incorrect model estimate;
- poor calibration;
- insufficient evidence;
- unexpected task behaviour;
- unvalidated thresholds.

### Important boundary

A technically valid policy decision is not automatically an appropriate
participant intervention.

### Mitigations

Current engineering safeguards include:

- signal/evidence gating;
- prediction abstention;
- persistence requirements;
- cooldown;
- difficulty bounds;
- adaptation budget;
- direction-change controls;
- experimenter disablement.

Future participant mitigation should additionally include:

- pilot evaluation;
- validated command transport;
- explicit stop procedure;
- adaptation-appropriateness assessment before benefit claims.

### Residual risk

Provisional: **Medium** until participant-facing adaptation is validated.

## 12. Excessive Difficulty Escalation

### Risk

Repeated increases in difficulty could exceed a participant's comfortable or
useful task range.

### Mitigations

- hard difficulty bounds;
- limited step size;
- adaptation budget;
- cooldown;
- persistence;
- conflict handling;
- experimenter intervention;
- participant withdrawal.

### Residual risk

Provisional: **Low to Medium**, subject to final configuration.

## 13. Rapid Adaptation Oscillation

### Risk

Frequent alternating difficulty changes could:

- disrupt the task;
- frustrate participants;
- confound experimental interpretation.

### Current engineering controls

The policy includes mechanisms intended to reduce rapid oscillation, including:

- persistence;
- cooldown;
- direction-change control;
- bounded difficulty changes.

### Scientific limitation

Software tests showing deterministic conservative behaviour do not establish
that participants will find the resulting timing comfortable.

### Residual risk

Provisional: **Low to Medium**.

## 14. Adaptation When Evidence Is Insufficient

### Risk

The system could act despite:

- poor sensing;
- unavailable modalities;
- low predictive confidence;
- wide prediction intervals;
- abstention.

### Mitigations

EngageVR is designed so that insufficient evidence can result in:

```text
unavailable
abstain
HOLD
```

rather than mandatory action.

The adaptation gate must not be bypassed for participant-facing operation.

### Residual risk

Provisional: **Low to Medium**, dependent on model validation and integration
correctness.

## 15. Adaptation Transport Failure

### Risk

A policy may produce a valid proposal while:

- the command fails to send;
- the task environment rejects it;
- acknowledgement is not received;
- the actual task state differs from the assumed state.

### Mitigations

The recording must distinguish:

```text
proposed
command built
dispatched
acknowledged
applied
rejected
```

where supported.

Only observed lifecycle states may be treated as participant exposure.

### Residual risk

Provisional: **Medium** until the complete participant-facing runtime is
validated.

## 16. Experimenter Unable to Stop Adaptation

### Risk

A software fault could interfere with the experimenter's ability to stop or
disable adaptive behaviour.

### Required mitigation

Before participant use, the study must verify a reliable experimenter stop or
disable mechanism.

The participant-facing task should have a safe defined fallback state.

### Residual risk

**Not acceptable for participant deployment until tested.**

## 17. Webcam Privacy

### Risk

Webcam use may create participant privacy concerns.

A video frame can contain substantially more information than the derived
features required for the research.

### Default mitigation

Raw webcam recording remains disabled by default.

Frames should be processed locally/in memory where possible.

Derived features should be preferred when sufficient for the research
question.

### Residual risk

Provisional: **Low to Medium**, depending on final data retention.

## 18. Accidental Raw-Video Recording

### Risk

A configuration error could cause raw webcam data to be retained when the
approved protocol does not permit it.

### Mitigations

Before each session:

- verify raw-video configuration;
- include the setting in the operator checklist;
- test the configured output directory;
- inspect study configuration before participant exposure.

If accidental recording occurs, follow the approved privacy/incident procedure.

### Residual risk

Provisional: **Medium** because the privacy impact may be significant even if
likelihood is low.

## 19. Physiological-Data Sensitivity

### Risk

Heart-rate, waveform, or other physiological measurements may be considered
sensitive even without direct identifiers.

### Mitigations

- pseudonymous IDs;
- minimum necessary collection;
- controlled storage;
- limited access;
- no participant data in Git;
- explicit retention/deletion rules;
- no unnecessary external processing.

### Residual risk

Provisional: **Medium**.

## 20. Misinterpretation of Physiological Signals

### Risk

Researchers or participants may interpret rPPG-derived measurements as:

- medical readings;
- diagnostic information;
- direct proof of engagement;
- direct proof of stress or cognitive state.

### Mitigations

All documentation and participant-facing wording should clearly state:

- rPPG values are signal-processing estimates;
- they may be unavailable;
- they are not medical measurements;
- they do not establish psychological state.

### Residual risk

Provisional: **Low** if terminology is controlled.

## 21. Incidental Physiological Findings

### Risk

Future reference physiological sensors may produce unexpected measurements that
a researcher or participant could interpret as medically meaningful.

### Boundary

EngageVR is not designed for medical screening.

Researchers should not improvise diagnosis or clinical interpretation.

### Mitigation

The final institutional protocol should specify how potential incidental
findings are handled if reference hardware creates a realistic possibility of
them.

### Residual risk

**Undetermined** until reference hardware and protocol are chosen.

## 22. Poor Signal Quality Misread as Participant State

### Risk

Poor webcam or rPPG quality could be interpreted incorrectly as:

- low engagement;
- high cognitive load;
- fatigue;
- disengagement.

### Mitigation

Signal quality remains a distinct variable.

Poor-quality measurements should become unavailable where appropriate.

### Residual risk

Provisional: **Low** if existing semantic boundaries are maintained.

## 23. Missing Data Fabrication

### Risk

Unavailable values could be converted into numeric defaults for convenience.

For example:

```text
unavailable heart rate -> 0
```

would introduce scientifically misleading data.

### Mitigation

EngageVR's data model requires explicit missing/unavailable handling.

`None`, unavailable, abstained, and zero must remain distinct.

### Residual risk

Provisional: **Low**, provided existing validation rules remain enforced.

## 24. Incorrect Model Confidence Interpretation

### Risk

A high calibrated probability could be presented as certainty.

### Mitigation

Classification confidence must remain defined as a model probability quantity,
not certainty.

Regression interval width must retain separate semantics.

### Residual risk

Provisional: **Low**.

## 25. Expert Disagreement Mislabelled as Uncertainty

### Risk

Spread between modality-specific predictions could be reported as calibrated
uncertainty without justification.

### Mitigation

Expert disagreement remains a diagnostic unless an explicit calibration method
supports a stronger interpretation.

### Residual risk

Provisional: **Low**.

## 26. Participant Re-Identification

### Risk

Pseudonymous behavioural and physiological data may potentially be linked with
other information.

### Important boundary

Pseudonymization is not anonymization.

### Mitigations

- avoid direct identifiers in analysis datasets;
- separate administrative identity information;
- minimize collected variables;
- restrict access;
- avoid unnecessary external linkage;
- control public release.

### Residual risk

Provisional: **Medium**.

## 27. Administrative-Research Data Linkage

### Risk

A mapping between participant identity and research pseudonym may expose the
identity associated with research data.

### Mitigations

If such a mapping is required:

- store it separately;
- restrict access;
- define its purpose;
- define retention;
- avoid placing it in modelling files or Git.

### Residual risk

Provisional: **Medium**.

## 28. Participant Data Accidentally Committed to Git

### Risk

Research data could accidentally enter source control.

### Potential consequences

- unauthorized disclosure;
- persistent repository history;
- replication to remote systems;
- difficult deletion.

### Mitigations

- participant data stored outside repository source paths where practical;
- explicit Git boundary audits;
- `.gitignore` as an additional safeguard, not the primary control;
- inspect staged files before every commit;
- never use broad unattended Git staging during participant-data work.

### Residual risk

Provisional: **Medium**, because consequence severity can be substantial.

## 29. Participant Data Uploaded to Development or AI Services

### Risk

Participant data could be pasted or uploaded to external development,
generative-AI, debugging, or code-assistance services for convenience.

### Mitigation

Participant research data must remain outside such services unless external
processing is explicitly part of an approved research/data-management
procedure.

Source-code assistance and participant-data processing must remain distinct
activities.

### Residual risk

Provisional: **Low to Medium**, dependent on operational discipline.

## 30. Unauthorized Cloud Synchronization

### Risk

Desktop synchronization or backup software may automatically copy participant
data to an unapproved cloud destination.

### Mitigations

Before collection:

- verify storage path;
- verify synchronization behaviour;
- define approved backup mechanism;
- avoid unreviewed consumer sync directories.

### Residual risk

Provisional: **Medium**.

## 31. Unauthorized Access

### Risk

Another local user, remote user, or service may access participant data.

### Mitigations

- least-privilege permissions;
- protected device account;
- controlled storage;
- minimal network exposure;
- authorized-user list;
- incident procedure.

### Residual risk

Depends on the final laboratory environment.

## 32. Data Loss

### Risk

Research data may be lost because of:

- storage failure;
- accidental deletion;
- corrupted files;
- device loss.

### Mitigations

- approved backup process;
- integrity checks;
- session finalization checks;
- reproducible metadata;
- controlled restoration procedure.

### Residual risk

Provisional: **Low to Medium**.

## 33. Incomplete Deletion

### Risk

A withdrawal/deletion request may remove primary files but leave:

- backups;
- exported copies;
- derived datasets.

### Mitigation

The final protocol should document deletion scope and realistic limitations.

### Residual risk

Undetermined until the retention/backup architecture is finalized.

## 34. Network Exposure

### Risk

The current local backend lacks a production-grade security architecture.

Exposure to an untrusted network could increase unauthorized-access risk.

### Mitigation

Participant deployment should remain local/minimally exposed unless an
appropriate security architecture is introduced and reviewed.

### Residual risk

Provisional: **Low** under loopback/local deployment; higher if externally
exposed.

## 35. Authentication and Authorization Limitations

### Risk

The current prototype should not be assumed to provide production-grade access
control.

### Mitigation

Do not expose participant services externally merely because Docker or a web
interface exists.

Select an appropriate deployment boundary before participant use.

### Residual risk

Dependent on deployment architecture.

## 36. Software Crash

### Risk

The backend, task environment, dashboard, capture process, or another component
may terminate unexpectedly.

### Potential impact

- participant interruption;
- incomplete recording;
- lost data;
- confusion about condition exposure.

### Mitigations

- pre-study smoke testing;
- stable frozen version;
- session recovery semantics;
- explicit interrupted status;
- experimenter procedure for restart/termination.

### Residual risk

Provisional: **Low to Medium**.

## 37. Corrupt Session Data

### Risk

A recording may contain malformed or incomplete data.

### Mitigation

- append-only storage;
- schema validation;
- checksum/integrity mechanisms;
- distinguish partial trailing writes from interior corruption;
- never silently rewrite corrupted history.

### Residual risk

Provisional: **Low**.

## 38. Synchronization Error

### Risk

Measurements from different modalities may be incorrectly aligned.

### Potential impact

This can produce scientifically invalid associations while appearing
technically plausible.

### Mitigation

Use:

- common timestamp semantics;
- session IDs;
- explicit windows;
- protocol-defined event relationships.

Do not synchronize by row position alone.

### Residual risk

Provisional: **Medium**, because synchronization errors may be difficult to
detect retrospectively.

## 39. Wrong Experimental Condition

### Risk

The operator or software may load the wrong condition.

### Mitigation

Pre-session verification should confirm:

- participant pseudonym;
- condition;
- condition order;
- adaptation state;
- configuration.

Condition should also be recorded in session provenance.

### Residual risk

Provisional: **Low** with procedural checking.

## 40. Wrong Software or Configuration Version

### Risk

Different participants could unknowingly receive different:

- model versions;
- thresholds;
- adaptation settings;
- task builds.

### Mitigation

Freeze and record:

- software version;
- model version;
- task build;
- configuration fingerprint;
- adaptation configuration.

### Residual risk

Provisional: **Low** with configuration freeze.

## 41. Threshold Changes During Data Collection

### Risk

Researchers may adjust parameters after observing early participants.

This could invalidate planned comparisons.

### Mitigation

Freeze confirmatory study parameters before data collection.

Any necessary change must be:

- documented;
- justified;
- versioned;
- treated as a deviation/amendment as appropriate.

### Residual risk

Provisional: **Low** if protocol discipline is maintained.

## 42. Model Validity Risk

### Risk

The software may function perfectly while the engagement or cognitive-load
model lacks scientific validity.

### Mitigation

Do not permit software test success to substitute for real-data validation.

Participant-facing adaptation should depend on evidence appropriate to its use.

### Residual risk

Currently **substantial scientific uncertainty**.

## 43. Personalization Harm or Degradation

### Risk

Participant-specific calibration may worsen predictions rather than improve
them.

### Mitigation

- retain population predictions;
- evaluate personalization separately;
- preserve cold-start behaviour;
- do not assume personalization equals benefit;
- use chronologically valid calibration.

### Residual risk

Scientifically **unknown** until real participant data exist.

## 44. Data Leakage

### Risk

Information from evaluation periods may enter model training or participant
calibration.

### Consequence

Performance estimates may appear better than they truly are.

### Mitigation

- participant-aware splitting;
- session-aware splitting;
- chronological calibration boundaries;
- no row-level fallback that violates grouping.

### Residual risk

Provisional: **Low** if established repository safeguards remain intact.

## 45. Synthetic Evidence Misrepresented as Human Evidence

### Risk

Synthetic performance may be presented as evidence that EngageVR works for
people.

### Mitigation

Synthetic data permanently retain:

```text
scientific_evaluation_eligible = false
```

and software-self-check labelling.

### Residual risk

Provisional: **Low** if provenance is preserved.

## 46. Public Dataset Overgeneralization

### Risk

A component may perform well on a public dataset while the result is described
as validation of the entire EngageVR system.

### Mitigation

Report:

- dataset identity;
- modalities;
- labels;
- population;
- collection protocol;
- limitations.

Component-level evidence must remain component-level evidence.

### Residual risk

Provisional: **Low**.

## 47. Selective Reporting

### Risk

Researchers may focus only on outcomes that support a desired conclusion.

### Mitigation

Predefine:

- primary outcome;
- hypotheses;
- analysis plan;
- exclusions;
- multiplicity procedure where relevant.

Report:

- null findings;
- negative findings;
- poor-quality data;
- failed personalization;
- abstention;
- unavailable results.

### Residual risk

Provisional: **Low** with prospective analysis planning.

## 48. Multiple Testing

### Risk

Testing many models, outcomes, subgroups, or adaptation metrics may increase the
chance of apparently positive findings.

### Mitigation

The statistical-analysis plan should distinguish:

- confirmatory outcomes;
- secondary outcomes;
- exploratory analyses;

and define multiplicity handling where appropriate.

### Residual risk

Depends on the final analysis plan.

## 49. Participant Coercion or Undue Influence

### Risk

Participants may feel pressured to participate because of:

- researcher/student relationships;
- perceived academic obligation;
- compensation;
- misunderstanding of voluntariness.

### Mitigation

Recruitment and consent language should clearly state voluntary participation.

The final procedure should avoid coercive recruitment structures.

### Residual risk

Depends on recruitment context.

## 50. Consent Misunderstanding

### Risk

Participants may misunderstand the system as:

- clinically validated;
- psychologically diagnostic;
- capable of accurately reading emotion or engagement.

### Mitigation

Consent materials must use restrained terminology and explain that:

- estimates may be wrong;
- measurements may be unavailable;
- adaptation may not occur;
- the system is experimental.

### Residual risk

Provisional: **Low to Medium**, depending on communication quality.

## 51. Withdrawal Handling

### Risk

Participants may wish to stop but the operational procedure may be unclear.

### Mitigation

The final protocol and consent materials should define:

- how to stop;
- who to tell;
- what happens to already collected data;
- deletion rights/limitations where applicable.

### Residual risk

Undetermined until the final institutional procedure is defined.

## 52. Vulnerable Participants

### Risk

Some participant populations may require additional protections.

### Current boundary

No vulnerable population is specifically proposed.

### Mitigation

Do not assume a general adult procedure applies automatically to every
population.

Any such inclusion requires explicit justification and review.

## 53. Accessibility

### Risk

The task or research interface may exclude or burden participants with certain
access needs.

### Current limitation

The software has not undergone participant-based accessibility validation.

### Mitigation

Future study design should identify accessibility requirements relevant to the
target population.

### Residual risk

Undetermined until the participant population is selected.

## 54. Fairness and Differential Measurement Performance

### Risk

Webcam-derived behavioural or rPPG measurements may perform differently across
people or environmental conditions.

### Mitigation

- report acquisition limitations;
- examine measurement failure rather than hiding it;
- document study population;
- avoid claiming universal validity;
- design any subgroup analysis prospectively and ethically.

### Important restriction

Do not infer sensitive participant attributes merely to create subgroup labels.

## 55. Reference Sensor Discomfort

### Future risk

PPG, ECG, EDA, respiration, or other reference sensors may introduce:

- skin irritation;
- pressure/discomfort;
- setup burden;
- movement restriction.

### Mitigation

Only use hardware according to appropriate instructions and approved study
procedures.

### Residual risk

Undetermined until hardware is selected.

## 56. Hardware Electrical or Physical Safety

### Current status

No research-grade participant sensor configuration has been finalized.

### Future requirement

Any participant-connected hardware must be appropriate for the intended
research setting and reviewed according to relevant laboratory/institutional
requirements.

Software documentation must not be treated as hardware safety certification.

## 57. Device Failure

### Risk

A webcam, reference sensor, display, or VR device may fail during collection.

### Mitigation

- pre-session checks;
- safe task fallback;
- experimenter intervention;
- explicit technical-failure recording;
- do not fabricate missing data.

## 58. Privacy Incident

### Examples

- participant file copied to wrong location;
- unauthorized user gains access;
- raw video recorded unexpectedly;
- participant data uploaded externally;
- research device lost.

### Required response

The final approved protocol must define incident handling.

This document does not replace institutional incident-response procedures.

## 59. Researcher Overreliance on Dashboard Output

### Risk

A visually polished dashboard may make uncertain model outputs appear more
authoritative than they are.

### Mitigation

Continue to display:

- provenance;
- missingness;
- uncertainty;
- abstention;
- scientific eligibility;
- limitations.

No dashboard control should confer scientific validation status.

## 60. MLOps Overinterpretation

### Risk

Reproducibility infrastructure may be mistaken for scientific validity.

### Boundary

The following can demonstrate engineering integrity:

- CI;
- Docker;
- MLflow;
- DVC;
- checksums;
- reproducible pipelines.

They cannot establish:

- ethical approval;
- construct validity;
- physiological validity;
- participant benefit;
- adaptation effectiveness.

## 61. Operational Stop Conditions

The final participant protocol should define explicit stop conditions.

Candidate categories include:

- participant requests to stop;
- participant discomfort;
- simulator sickness where applicable;
- repeated inappropriate adaptation;
- task-control failure;
- critical sensor/equipment failure;
- privacy concern;
- experimenter safety judgment.

Exact criteria require institutional review.

## 62. Safe System State

When adaptation cannot continue safely or validly, the task should move to a
predefined safe state.

Depending on the final study, this may mean:

- HOLD current difficulty;
- disable adaptation;
- pause task;
- end participant exposure.

The safe state must be technically validated before participant use.

## 63. Risk Register

| ID | Risk | Initial planning concern | Main mitigation | Residual status |
|---|---|---|---|---|
| R01 | Physical/visual fatigue | Low-Medium | Duration limits, breaks, stop option | Provisional |
| R02 | Cognitive fatigue/frustration | Medium | Practice, bounded difficulty, stop option | Provisional |
| R03 | Simulator sickness | Unknown | VR-specific protocol and stop criteria | Not yet assessable |
| R04 | Inappropriate adaptation | Medium | Gating, abstention, pilot, experimenter control | Provisional |
| R05 | Excessive/oscillating adaptation | Medium | Bounds, cooldown, persistence, budget | Provisional |
| R06 | Adaptation transport failure | Medium | Lifecycle recording and runtime validation | Pending validation |
| R07 | Webcam privacy | Medium | Local processing, raw video off | Provisional |
| R08 | Accidental raw-video retention | Medium | Pre-session configuration audit | Provisional |
| R09 | Physiological-data sensitivity | Medium | Pseudonymization, access control, minimization | Provisional |
| R10 | Participant re-identification | Medium | Separate identity mapping, least access | Provisional |
| R11 | Participant data committed to Git | Medium-High consequence | Storage separation and staged-file audits | Provisional |
| R12 | External-service data disclosure | Medium | No unapproved participant-data upload | Provisional |
| R13 | Unauthorized access | Environment dependent | Permissions and minimal network exposure | Pending deployment design |
| R14 | Data loss/corruption | Medium | Backups, integrity checks, append-only recording | Provisional |
| R15 | Synchronization error | Medium | Common timestamps and explicit window linkage | Provisional |
| R16 | Wrong condition/configuration | Medium | Pre-session verification and provenance | Provisional |
| R17 | Model invalidity | High scientific concern | Real-data evaluation before strong claims | Unresolved |
| R18 | Personalization degradation | Unknown | Preserve population baseline and evaluate separately | Unresolved |
| R19 | Synthetic/public-data overclaim | Medium | Provenance and scientific-eligibility labels | Provisional |
| R20 | Consent misunderstanding | Medium | Restrained participant-facing language | Pending consent draft |
| R21 | Withdrawal ambiguity | Medium | Approved withdrawal/data procedure | Pending protocol |
| R22 | Reference-device discomfort | Unknown | Hardware-specific protocol | Not yet assessable |
| R23 | Accessibility burden | Unknown | Population-specific accessibility review | Not yet assessable |
| R24 | Selective reporting | Medium | Prospective hypotheses and analysis plan | Provisional |

## 64. Risks That Block Participant Deployment

The following should be treated as unresolved deployment blockers until the
relevant control exists:

1. no approved institutional/ethical pathway;
2. no finalized consent process;
3. no finalized stop criteria;
4. participant-facing task runtime not validated;
5. participant-facing adaptation transport not validated if adaptation is used;
6. no finalized participant-data storage/access/retention procedure;
7. models driving adaptation lack evidence appropriate to their intended use;
8. no validated safe fallback for critical participant-facing failures.

These are not minor documentation omissions.

## 65. Pilot Exit Criteria

Before moving from a technical/pilot phase to a main participant comparison,
candidate exit criteria may include:

- task completes reliably;
- session recording is complete and interpretable;
- condition assignment is correct;
- adaptation commands reach the intended task state;
- acknowledgement/state recording works;
- experimenter stop control works;
- no uncontrolled raw-video recording occurs;
- participant burden is acceptable under the reviewed protocol;
- data-quality failure is represented explicitly;
- no safety issue requires redesign.

The final criteria must be defined before the pilot.

## 66. Main-Study Risk Review

Immediately before the main controlled study, the risk assessment should be
reviewed against the final:

- participant population;
- sample procedure;
- task version;
- hardware;
- sensing modalities;
- model version;
- adaptation configuration;
- questionnaire set;
- session duration;
- storage architecture;
- incident procedure.

Any material design change may require the risk assessment to be revised.

## 67. Documentation Dependencies

This risk assessment informs:

- `docs/CONSENT_TEMPLATE.md`;
- the final participant information materials;
- the data-collection protocol;
- the statistical-analysis plan where exclusions or stopping affect analysis;
- future hardware-validation procedures.

The consent template should not introduce materially different risks without
updating this assessment.

## 68. Current Status

This document is a prospective risk assessment.

It does not claim:

- that all risks have been identified;
- that institutional review has occurred;
- that residual risks have been accepted;
- that participant-facing hardware is safe;
- that adaptation is safe or beneficial;
- that model outputs are scientifically validated;
- that participant recruitment may begin.

Risk assessment must be revisited once the final study and laboratory
environment are defined.
