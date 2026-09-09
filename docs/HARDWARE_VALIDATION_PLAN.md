# EngageVR Hardware Validation Plan

> **Draft for research and laboratory planning**
>
> This document defines prospective hardware-validation work for EngageVR.
>
> It does not claim that the listed hardware has been validated, that a
> participant-facing laboratory setup is approved, or that any physiological
> measurement is medically accurate.
>
> Human-subject use requires the appropriate institutional review and
> participant-safety procedures in addition to the engineering validation
> described here.

## 1. Purpose

EngageVR currently contains software interfaces and pipelines for:

- webcam capture;
- behavioural feature extraction;
- webcam remote photoplethysmography (rPPG);
- task telemetry;
- Unity/task-environment integration;
- uncertainty-aware model inference;
- conservative adaptation;
- future physiological reference sensors.

Software tests alone cannot establish that these components behave correctly on
real physical hardware.

This document defines a staged plan for validating the physical acquisition and
participant-facing hardware paths.

## 2. Current Hardware Validation Status

At the time of this document:

- no physical webcam has been validated end to end with EngageVR;
- webcam rPPG has not been validated against a physical reference sensor;
- UBFC-rPPG real-data evaluation remains pending;
- Unity has not been validated as a participant-facing runtime;
- no VR headset configuration has been validated;
- no participant-connected physiological reference device has been finalized;
- no human-subject hardware study has been conducted.

These are open research and engineering requirements.

## 3. Validation Tracks

Hardware validation should be separated into the following tracks:

1. physical webcam acquisition;
2. webcam behavioural-feature acquisition;
3. webcam-rPPG component validation;
4. physiological reference-device validation;
5. timing and synchronization;
6. Unity/task-environment validation;
7. immersive-VR hardware validation where applicable;
8. integrated participant-facing laboratory validation.

Passing one track does not imply that another track has passed.

## 4. Validation Principles

Hardware validation should preserve:

- explicit device identity;
- reproducible configuration;
- timestamp provenance;
- quality diagnostics;
- explicit unavailable states;
- no fabricated sensor values;
- separation of engineering validity from scientific validity;
- documented failure modes;
- participant safety where human testing is later approved.

## 5. Validation Levels

The project should distinguish several validation levels.

### Level 0 — Software simulation

Examples:

- synthetic RGB traces;
- synthetic camera frames;
- mock sensors;
- deterministic task fixtures;
- synthetic session recordings.

Purpose:

- software correctness;
- schema verification;
- failure-path testing.

This level provides no physical-hardware evidence.

### Level 1 — Bench hardware validation

Physical hardware is operated without a research participant where feasible.

Examples:

- webcam enumeration;
- frame acquisition;
- frame-rate measurement;
- timestamp checks;
- device reconnection;
- task-environment command transport.

Purpose:

- verify physical device integration.

### Level 2 — Controlled operator validation

A researcher/developer may exercise the physical pipeline for engineering
purposes where permitted.

Purpose:

- verify complete acquisition paths;
- discover operational failures;
- validate setup instructions.

This is not automatically participant research evidence.

### Level 3 — Approved participant component validation

A reviewed study may evaluate a specific component such as:

- webcam rPPG against a reference sensor;
- motion sensitivity;
- illumination sensitivity;
- behavioural-feature availability.

### Level 4 — Approved integrated participant study

The complete participant-facing configuration is evaluated under the final
study protocol.

No lower level should be described as Level 4 evidence.

## 6. Physical Webcam Validation

### 6.1 Objective

Verify that EngageVR can acquire physical webcam frames reliably using the
intended participant hardware.

### 6.2 Device information to record

For every validated webcam configuration, record:

- manufacturer;
- model;
- device identifier where appropriate;
- interface/backend;
- resolution;
- nominal frame rate;
- observed frame rate;
- pixel format;
- exposure behaviour where available;
- autofocus state where relevant.

### 6.3 Minimum bench checks

Validate:

- device discovery;
- successful open;
- frame acquisition;
- expected dimensions;
- timestamp generation;
- sustained capture;
- clean shutdown;
- reopen after shutdown;
- behaviour after disconnection where practical.

### 6.4 Frame-rate validation

Nominal frame rate and observed frame rate should be distinguished.

Record:

```text
requested_fps
observed_fps
frame_interval_distribution
dropped_frame_count
```

where available.

A device advertising 30 fps does not establish that 30 fps was delivered
throughout the recording.

## 7. Webcam Timing Validation

Frame timestamps are required for:

- rPPG windows;
- behavioural windows;
- task synchronization;
- multimodal alignment.

Validate:

- monotonic timestamps;
- no unexplained timestamp reversal;
- expected frame spacing;
- behaviour under temporary capture stalls;
- relationship between capture timestamps and the project time base.

Timing errors must not be repaired by arbitrary row indexing.

## 8. Webcam Resolution and Cropping

The validation should record whether:

- the requested resolution is actually delivered;
- aspect ratio is preserved;
- face-region extraction behaves correctly;
- resizing changes the rPPG or behavioural pipeline.

Any preprocessing performed before feature extraction should be versioned.

## 9. Illumination Validation

Webcam-derived measurements depend strongly on lighting.

A controlled bench or approved component study should characterize behaviour
under defined illumination conditions.

Possible factors include:

- illuminance;
- light-source position;
- spectral characteristics where measurable;
- temporal flicker;
- backlighting;
- daylight variation.

Qualitative labels such as:

```text
good lighting
bad lighting
```

should be supplemented by quantitative measurements where practical.

## 10. Motion Validation

Motion can degrade:

- face tracking;
- landmarks;
- rPPG;
- head-pose estimation.

Candidate controlled conditions may include:

- low-motion reference condition;
- small natural head movement;
- deliberate head rotation;
- task-natural movement.

The exact motion protocol must be operationally defined before scientific data
collection.

## 11. Face-Detection Validation

Validate that the physical webcam pipeline correctly records states such as:

- face detected;
- face temporarily lost;
- no face;
- multiple detections if the implementation permits them;
- invalid geometry.

Face-detection failure must not become a fabricated behavioural measurement.

## 12. Behavioural-Feature Validation

Candidate physical checks include:

- blink/eye-closure proxy availability;
- mouth-geometry proxy availability;
- head yaw;
- head pitch;
- head roll;
- head-motion features.

These tests initially establish only that the software produces stable,
interpretable measurements.

They do not establish psychological validity.

## 13. Behavioural Reference Validation

Where a behavioural feature requires quantitative validation, the reference
method must be defined.

Possible methods may include:

- manually annotated controlled events;
- known geometric movement;
- an appropriate validated tracking reference.

The reference process should be independent of the measurement being tested.

## 14. Webcam-rPPG Validation Objective

The rPPG validation track should determine whether EngageVR can estimate a
physiological pulse-related signal from a physical webcam under defined
conditions.

It should evaluate measurement behaviour, not engagement.

## 15. Candidate rPPG Methods

Current candidate algorithms include:

- GREEN;
- CHROM;
- POS.

Each method should be identified explicitly in every validation result.

## 16. rPPG Reference Requirement

Webcam rPPG must not validate itself.

Heart-rate accuracy or agreement testing requires an appropriate independent
reference measurement.

Candidate reference technologies may include:

- contact PPG;
- ECG.

The final device must be selected before the scientific validation protocol is
frozen.

## 17. Reference Device Selection Criteria

A reference device should be evaluated for:

- measurement principle;
- suitability for the intended use;
- sampling rate;
- data-access capability;
- timestamp accessibility;
- synchronization capability;
- participant burden;
- placement;
- manufacturer documentation;
- institutional/laboratory suitability.

Consumer-device convenience alone is not sufficient justification.

## 18. Reference Device Metadata

The validation record should identify:

- manufacturer;
- model;
- hardware revision where relevant;
- firmware version where available;
- sensor type;
- sampling rate;
- acquisition software;
- placement;
- configuration.

## 19. rPPG Synchronization

The webcam and reference physiological stream require a documented
synchronization strategy.

Possible strategies include:

- shared acquisition clock;
- synchronized event marker;
- recorded alignment signal;
- measured clock offset.

The final method must define:

- initial offset;
- drift;
- timestamp resolution;
- alignment uncertainty.

## 20. No Row-Position Synchronization

The following is not a valid synchronization method:

```text
webcam row 1000 = reference row 1000
```

unless an acquisition contract explicitly establishes that relationship.

Synchronization should use actual temporal information.

## 21. rPPG Candidate Outcomes

Potential validation outcomes include:

- valid-window proportion;
- HR availability;
- signal-quality index;
- HR absolute error;
- HR signed error/bias;
- RMSE where appropriate;
- agreement against the reference;
- failure rate.

The final primary rPPG outcome must be predefined in the statistical analysis
plan before confirmatory evaluation.

## 22. rPPG Agreement

Correlation alone is not sufficient evidence of agreement.

A validation plan should evaluate actual differences between the webcam estimate
and reference measurement.

The exact agreement procedure should be finalized prospectively.

## 23. rPPG Window Eligibility

A window may be unavailable because of:

- insufficient duration;
- missing face;
- excessive motion;
- unsuitable illumination;
- insufficient signal quality;
- invalid physiological estimate.

Unavailable windows should remain unavailable.

They must not be converted to:

```text
heart_rate = 0
```

## 24. rPPG Signal Quality

Signal quality is an engineering/measurement quantity.

It is not:

- engagement;
- cognitive load;
- stress;
- participant compliance;
- model confidence.

Physical validation should characterize what quality scores correspond to in
terms of measurement failure or reliability.

## 25. rPPG Illumination Study

A future controlled study may compare rPPG performance across illumination
conditions.

For every condition, record:

- measured or controlled light level;
- source;
- geometry;
- camera configuration;
- algorithm;
- reference measurement;
- motion condition.

## 26. rPPG Motion Study

A future controlled study may compare rPPG performance across motion
conditions.

Potential factors include:

- low-motion reference;
- controlled head movement;
- natural task motion.

Motion conditions should be predefined rather than categorized after observing
performance.

## 27. rPPG Skin-Related Generalization

Camera-based physiological measurement may vary across participants and imaging
conditions.

Future validation should report population limitations honestly.

The project must not claim uniform performance across skin appearances or
participant groups without evidence.

Sensitive attributes should not be inferred merely to create evaluation
categories.

## 28. Public-Dataset Component Validation

UBFC-rPPG may provide component-level validation before a local reference-device
study.

That analysis can evaluate only the quantities supported by its:

- video;
- physiological reference;
- acquisition protocol.

It cannot validate:

- engagement prediction;
- cognitive-load prediction;
- personalization benefit;
- adaptation effectiveness.

## 29. Task Hardware Validation

The participant-facing task environment should be validated independently of
the sensing stack.

Validate:

- input devices;
- display;
- task timing;
- response capture;
- session start/stop;
- difficulty state;
- event emission;
- recovery after task failure.

## 30. Unity Build Validation

If Unity is used, the exact participant-facing build must be compiled and
tested.

Minimum checks include:

- successful project import;
- successful compilation;
- scene loading;
- task interaction;
- telemetry emission;
- command receipt;
- difficulty change;
- session completion;
- clean shutdown.

Source-code inspection alone does not establish runtime validity.

## 31. Python/Unity Protocol Validation

Where Python and Unity communicate, validate:

- connection establishment;
- version/protocol compatibility;
- heartbeat behaviour;
- message parsing;
- malformed-message handling;
- task events;
- adaptation commands;
- acknowledgement;
- disconnection;
- reconnection where supported.

## 32. Adaptation Command Validation

For `set_difficulty`, verify the full participant-facing lifecycle.

Conceptually:

```text
proposal
    ->
command built
    ->
dispatched
    ->
received
    ->
acknowledged
    ->
applied
```

Only lifecycle states actually observable in the implementation should be
claimed.

## 33. Command/State Consistency

After an acknowledged difficulty change, validate that:

- task state actually changed;
- recorded task state matches the participant-facing state;
- subsequent events use the updated state.

An acknowledgement alone is not proof that the intended participant-visible
state was applied unless the protocol establishes that contract.

## 34. Failed Command Handling

Test:

- invalid difficulty;
- malformed command;
- lost connection;
- missing acknowledgement;
- duplicate command;
- delayed acknowledgement;
- rejected command.

A failure should leave the system in a defined safe state.

## 35. Experimenter Disable Control

Before participant-facing adaptive use, validate that the experimenter can
disable adaptation reliably.

Test disablement:

- before a session;
- during an adaptive session;
- after a proposal but before a command where possible;
- after communications failure.

The resulting system state must be predictable.

## 36. Safe Fallback State

A critical participant-facing failure should lead to a predefined safe state.

Depending on final protocol, this may mean:

```text
HOLD difficulty
disable adaptation
pause task
end task
```

The fallback must be tested rather than documented only.

## 37. Display Validation

The participant-facing display setup should record:

- display model where relevant;
- resolution;
- refresh rate;
- scaling;
- task window/fullscreen configuration;
- viewing arrangement where scientifically relevant.

These variables can affect task timing, comfort, and visual presentation.

## 38. Input-Device Validation

Where reaction time or response timing is an outcome, input devices matter.

Record:

- keyboard/controller type;
- event-capture path;
- operating-system/input configuration where relevant.

If millisecond-level conclusions are required, end-to-end input latency may
need independent characterization.

## 39. Immersive VR Validation Boundary

Immersive VR requires a separate validation layer.

A desktop Unity build does not establish that:

- an HMD works;
- tracking works;
- frame timing is acceptable;
- the task is comfortable in VR;
- simulator sickness risk is acceptable.

## 40. VR Hardware Metadata

If an HMD is used, record:

- manufacturer;
- model;
- firmware/runtime version;
- refresh rate;
- rendered resolution;
- tracking mode;
- controllers;
- connection method;
- host GPU/CPU;
- VR runtime.

## 41. VR Runtime Validation

Before participant use, validate:

- headset detection;
- tracking;
- controller input;
- frame rendering;
- application start/stop;
- loss-of-tracking behaviour;
- boundary/guardian behaviour where applicable;
- task recovery.

## 42. VR Performance

Participant-facing VR should be evaluated for:

- sustained frame rate;
- dropped/reprojected frames where measurable;
- latency;
- thermal/performance degradation;
- task stability over the expected session duration.

A build launching successfully is not sufficient.

## 43. Simulator-Sickness Validation

If immersive VR is introduced, participant risk assessment must be updated.

The final approved study should define:

- symptoms monitored;
- stop criteria;
- recovery procedure;
- participant instructions;
- appropriate assessment instrument where justified.

No current simulator-sickness safety claim is made.

## 44. Physical Sensor Safety

Any participant-connected physiological hardware requires device-specific
review.

Consider:

- electrical safety;
- skin contact;
- pressure;
- adhesives;
- cleaning;
- infection-control procedure where relevant;
- cable entanglement;
- movement restriction.

Software testing does not establish hardware safety.

## 45. Sensor Cleaning and Reuse

Reusable participant-contact sensors require an appropriate cleaning procedure.

The final laboratory protocol should document:

- cleaning agent/procedure;
- compatibility with device instructions;
- responsibility;
- timing between participants.

## 46. Equipment Placement

The final setup should document placement of:

- webcam;
- display;
- HMD where applicable;
- reference sensors;
- cables;
- computer.

Placement should reduce:

- accidental movement;
- obstruction;
- discomfort;
- unsafe cable routing.

## 47. Laboratory Environment

Record relevant environmental conditions such as:

- illumination;
- major external light changes;
- room setup;
- display geometry;
- network conditions where relevant;
- background activity where relevant.

## 48. Device Configuration Freeze

The main study should freeze relevant hardware configuration.

Examples include:

- webcam;
- resolution;
- frame rate;
- reference sensor;
- HMD;
- firmware/runtime;
- display;
- input devices.

Replacing hardware during the study creates a potentially meaningful protocol
change.

## 49. Hardware Replacement

If hardware fails and must be replaced:

- record the replacement;
- identify affected sessions;
- document configuration differences;
- assess whether analysis is affected.

A replacement must not be hidden because the interface remains compatible.

## 50. Long-Run Stability

Participant-facing hardware should be tested for at least the expected session
duration.

Candidate checks include:

- capture continuity;
- memory/resource growth;
- dropped frames;
- sensor disconnects;
- thermal effects;
- task stability;
- synchronization drift.

## 51. Failure Injection

Where safe and practical, bench testing should intentionally exercise failures
such as:

- webcam disconnect;
- task termination;
- network/socket loss;
- delayed acknowledgement;
- missing sensor input;
- unavailable model output.

The goal is to verify safe and observable recovery behaviour.

## 52. Logging During Hardware Validation

Validation records should preserve:

- timestamp;
- device/configuration;
- software version;
- test case;
- expected behaviour;
- observed behaviour;
- pass/fail;
- failure details.

Free-text notes may supplement but should not replace structured results where a
structured representation is practical.

## 53. Validation Evidence

Acceptable engineering evidence may include:

- test logs;
- device metadata;
- captured timing summaries;
- screenshots;
- recorded protocol events;
- checksums;
- structured validation reports.

A screenshot alone should not be the only evidence for a quantitative claim.

## 54. Hardware Validation Matrix

| Component | Current status | Next evidence required |
|---|---|---|
| Physical webcam capture | Pending | Sustained physical acquisition test |
| Behavioural webcam features | Pending physical validation | Controlled physical feature checks |
| Webcam rPPG | Pending physical/scientific validation | Physical webcam + independent reference |
| UBFC-rPPG adapter | Implemented | Real dataset evaluation |
| Reference physiological sensor | Not selected | Device selection and protocol |
| Desktop task environment | Software implementation exists | Participant-facing runtime validation |
| Unity build | Pending runtime validation | Compile and execute exact build |
| Python/Unity transport | Pending participant-facing validation | End-to-end command/acknowledgement test |
| `set_difficulty` application | Pending live validation | Demonstrate applied task-state change |
| Experimenter adaptation disable | Requires participant-runtime validation | Physical end-to-end stop test |
| VR headset | Not selected/validated | Device-specific validation if used |
| Integrated laboratory setup | Not validated | Full dry run before approved study |

## 55. Evidence Classes

Every validation result should state its evidence class.

Suggested categories:

```text
synthetic_test
bench_hardware
operator_engineering_test
public_dataset_component_evaluation
approved_participant_component_study
approved_integrated_participant_study
```

These categories must not be silently promoted.

## 56. Hardware Validation Does Not Equal Scientific Validation

A webcam delivering frames correctly does not establish that:

- rPPG is accurate;
- engagement estimates are valid;
- cognitive-load estimates are valid.

Likewise, a reference sensor operating correctly does not establish that a model
uses its outputs appropriately.

## 57. Hardware Validation Does Not Equal Ethical Approval

A safe and technically functioning hardware setup does not authorize participant
recruitment.

Institutional requirements remain separate.

## 58. Hardware Validation Does Not Equal Adaptation Benefit

A successful end-to-end `set_difficulty` command proves only that the task can
change state.

It does not establish that the change:

- improves performance;
- improves engagement;
- reduces cognitive load;
- is comfortable;
- is beneficial.

Those are participant research questions.

## 59. Pre-Participant Hardware Checklist

Before an approved participant session:

- [ ] exact webcam configuration validated;
- [ ] physical frame acquisition stable;
- [ ] timing checked;
- [ ] required behavioural features available;
- [ ] rPPG quality/unavailable handling tested;
- [ ] reference sensor validated if required;
- [ ] synchronization procedure validated;
- [ ] task runtime validated;
- [ ] Unity build validated if used;
- [ ] adaptation transport validated if used;
- [ ] acknowledgement/state handling validated;
- [ ] experimenter disable control tested;
- [ ] safe fallback tested;
- [ ] display/input configuration recorded;
- [ ] VR runtime validated if used;
- [ ] participant-contact sensor procedure reviewed if used;
- [ ] storage destination verified.

## 60. rPPG Scientific Validation Checklist

Before making a scientific rPPG performance claim:

- [ ] real physiological recordings available;
- [ ] independent reference measurement available;
- [ ] reference sampling rate known;
- [ ] synchronization validated;
- [ ] rPPG algorithm frozen;
- [ ] windowing procedure frozen;
- [ ] quality rules frozen;
- [ ] primary outcome frozen;
- [ ] statistical-analysis procedure frozen;
- [ ] motion conditions documented;
- [ ] illumination conditions documented;
- [ ] participant grouping handled correctly;
- [ ] missing/unavailable windows reported.

## 61. Unity/Adaptation Validation Checklist

Before participant-facing adaptive Unity use:

- [ ] exact Unity build compiles;
- [ ] exact build runs;
- [ ] task interaction works;
- [ ] telemetry is emitted correctly;
- [ ] Python connection works;
- [ ] protocol versions agree;
- [ ] `set_difficulty` is received;
- [ ] valid command changes the task state;
- [ ] acknowledgement is recorded;
- [ ] invalid command is rejected safely;
- [ ] disconnect behaviour is safe;
- [ ] experimenter disablement works;
- [ ] final safe fallback works.

## 62. Decisions Still Required

Before completing the hardware-validation programme, resolve:

1. participant-facing webcam model;
2. webcam resolution/frame rate;
3. physical capture backend;
4. rPPG reference device;
5. reference sampling/access method;
6. synchronization mechanism;
7. controlled illumination protocol;
8. controlled motion protocol;
9. final task environment;
10. whether Unity is required;
11. whether immersive VR is required;
12. HMD model if required;
13. input devices;
14. participant-facing computer hardware;
15. safe fallback behaviour;
16. hardware cleaning procedure where applicable;
17. long-run validation duration;
18. acceptance thresholds for engineering tests;
19. rPPG primary validation outcome;
20. approved participant component-validation procedure.

## 63. Future Validation Artifacts

Future hardware validation may produce:

```text
artifacts/hardware_validation/
```

with run-specific outputs such as:

```text
device_manifest.json
validation_config.json
timing_summary.json
test_results.json
failure_log.jsonl
```

These names are prospective.

They must not be documented as implemented repository contracts until the
corresponding software exists.

## 64. Current Status

This document defines the planned hardware-validation path.

It does not claim:

- physical webcam validation;
- real webcam-rPPG validation;
- reference-device validation;
- Unity runtime validation;
- VR-headset validation;
- participant-facing adaptation validation;
- hardware safety certification;
- scientific validation;
- institutional approval.

Those remain future engineering, laboratory, and research activities.
