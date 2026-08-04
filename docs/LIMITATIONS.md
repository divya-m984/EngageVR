# EngageVR -- Limitations

## Scientific Limitations

1. **No experimental validation.** EngageVR has not been tested with human
   participants. All engagement and cognitive-load outputs are model estimates
   derived from software-only development. They must not be treated as medical,
   psychological, or diagnostic conclusions.

2. **Engagement is not directly measurable.** There is no objective ground-truth
   sensor for engagement or cognitive load. The system uses behavioural,
   physiological, and task-performance proxies that are associated with these
   constructs but do not define them.

3. **Physiological signals are proxies.** Heart rate, HRV, and facial behaviour
   correlate with autonomic states but do not uniquely identify engagement,
   cognitive load, stress, or fatigue. The same physiological pattern can arise
   from different causes.

4. **rPPG is not equivalent to contact PPG.** Webcam-based remote
   photoplethysmography provides estimates of pulse rate under constrained
   conditions. It is sensitive to motion, illumination changes, skin tone, and
   distance. Its accuracy degrades substantially outside controlled settings.

5. **HRV from short windows is unreliable.** Time-domain HRV features (SDNN,
   RMSSD) require sufficient recording duration and signal quality. The system
   will return `unavailable` rather than compute scientifically meaningless
   short-window HRV values.

6. **Facial expressions do not reveal internal states.** Facial action units and
   landmark movements are observable behaviours, not windows into a person's
   mental state. Cultural, individual, and contextual variation is substantial.

7. **No causal claims.** The system identifies statistical associations, not
   causal relationships. Adaptation experiments require controlled designs to
   support causal inference.

8. **Single-modality insufficiency.** Cognitive load cannot be reliably inferred
   from one signal alone. The system is designed for multimodal fusion precisely
   because individual signals are insufficient.

## Hardware Limitations

1. **No VR headset available.** The Unity environment runs in desktop mode
   (monitor, keyboard, mouse). VR-specific features (tracked controllers,
   6-DOF head tracking, stereoscopic rendering) are not implemented.

2. **No research-grade sensors.** Contact PPG, ECG, EDA, and respiration are
   represented as adapter interfaces with no hardware implementation.

3. **Consumer webcam only.** Frame rate, resolution, and sensor quality are
   limited by the laptop's built-in or USB webcam. No infrared or
   depth-sensing camera is available.

4. **No controlled testing environment.** Lighting, noise, distance, and
   background are not controlled. This affects rPPG quality and face-tracking
   stability.

## Software Limitations

1. **No participant data.** All current data is synthetic (permanently labelled)
   or derived from public datasets. No participant data has been collected.

2. **No institutional ethics approval.** Human-subject experimentation requires
   institutional review board approval, informed consent, and appropriate
   supervision. None of these are in place.

3. **Public datasets have their own biases.** Demographic composition, recording
   conditions, and labelling schemes vary across public datasets. Results on
   one dataset do not generalize to all populations or settings.

4. **Synthetic data is not evidence.** Synthetic data is used exclusively for
   software integration testing and must never be presented as experimental
   evidence or validation.

5. **Model confidence is not certainty.** A high confidence score indicates
   model agreement, not objective truth. Calibration and uncertainty estimation
   reduce but do not eliminate this limitation.

## Distinction: Signal Quality vs. Engagement

The system explicitly distinguishes between:

| Condition | Meaning | System Response |
|-----------|---------|-----------------|
| Low signal quality | Sensor data is unreliable (motion, occlusion, noise) | Report quality issue; may abstain from prediction |
| Low model confidence | Model is uncertain about its estimate | Report low confidence; may abstain; suppress adaptation |
| Missing data | A modality is unavailable | Mask the modality; predict with remaining signals if sufficient |
| Low engagement estimate | Model estimates the user is disengaged | Report estimate with confidence; adaptation policy decides |
| High cognitive load estimate | Model estimates cognitive load is high | Report estimate with confidence; adaptation policy decides |

**A poor-quality signal must never automatically be interpreted as low
engagement or high cognitive load.**

## rPPG Limitations (Milestone 3)

### Validation status

1. **No physical-webcam validation.** No physical V4L2 webcam is
   available in the development environment. The rPPG pipeline has never
   been run on live camera frames. Every automated test uses synthetic
   RGB traces, synthetic frames, or local fixtures.

2. **No public-dataset validation.** UBFC-rPPG is not present locally.
   The adapter's discovery and parsing are tested against temporary
   deterministic fixtures containing no real dataset content. **No MAE,
   RMSE, bias, coverage, or accuracy figure against any public dataset
   exists in this repository.**

3. **Synthetic recovery is not validation.** The `rppg-demo` command
   reports how closely the pipeline recovers a pulse frequency that the
   program itself inserted. That is a software self-check. It is not
   evidence about real physiological signals, and it is never to be
   presented as accuracy.

4. **No medical or diagnostic use.** Estimated pulse rate from a camera
   is not a medical measurement. It has not been compared against any
   reference device here. It must never be used for diagnosis,
   screening, monitoring, or any clinical purpose.

### Signal-condition limitations

5. **Motion.** Head motion changes the ROI's pixel content and its
   illumination geometry simultaneously. CHROM and POS suppress
   intensity-only variation, but neither removes motion that changes
   the *chrominance* of the sampled skin. Sustained motion degrades or
   invalidates the estimate.

6. **Illumination change.** Ambient light change, screen glow from the
   task itself, and shadows crossing the face all inject energy that can
   fall inside the pulse band. GREEN is essentially defenceless against
   this by construction.

7. **Camera auto-exposure and auto-white-balance.** Consumer webcams
   continuously adjust gain and colour balance. Those adjustments are
   *correlated with the scene*, so they are not removed by detrending
   and can imitate or cancel a pulse. Disabling auto-exposure is
   strongly advisable and is not currently enforced by the capture layer.

8. **Video compression.** Compression discards exactly the low-amplitude
   chrominance detail that carries the plethysmographic signal. The
   modulation is well under 1 % of the pixel value. Any lossy-compressed
   source may have no recoverable signal at all.

9. **Low frame rate.** The band's upper edge must be strictly below
   Nyquist. At 30 fps a 4 Hz (240 BPM) edge is fine; below roughly 10
   fps the configuration is rejected outright rather than silently
   aliasing.

10. **Skin-region tracking errors.** ROIs are derived from landmark
    bounding boxes with a fixed inward inset. Under head rotation,
    occlusion (hair, glasses, hands), or landmark jitter, the box can
    drift onto hair, background, or eyes. The valid-pixel and clipping
    checks catch gross failures, not subtle drift.

11. **Skin tone and lighting interact.** rPPG signal strength depends on
    the light reaching and returning from the dermis, which varies with
    skin tone and with illumination spectrum. This project performs no
    skin-tone inference and makes **no claim** about equitable
    performance across skin tones. UBFC-rPPG's demographic composition
    is not documented and could not establish such a claim in any case.

12. **Spectral resolution bounds precision.** BPM precision is limited by
    the Welch bin width — at the default 8 s segment and 30 fps, 0.125 Hz
    = 7.5 BPM. Every estimate reports its own `frequency_resolution_hz`.
    A BPM stated more precisely than that resolution is not meaningful.

13. **Spectral estimation cannot separate a harmonic from a fundamental.**
    If the second harmonic is stronger than the fundamental within the
    band, the peak search will select it, and the reported rate will be
    double the true one. No harmonic disambiguation is implemented.

14. **No method is superior.** GREEN, CHROM, and POS are implemented from
    their primary references. Relative performance depends on
    illumination, motion, camera, and subject. Nothing in this repository
    ranks them, and agreement between two methods is weak corroboration
    only — both read the same corrupted pixels.

15. **HRV is deliberately absent.** See DEC-022. Inter-beat intervals
    derived from an unvalidated camera waveform would look precise and
    mean nothing.

### rPPG quality is not engagement

16. **A low rPPG quality score describes the camera signal, never the
    person.** It means the measurement cannot be trusted. It does not
    mean the person is disengaged, stressed, fatigued, or cognitively
    loaded. A low-quality window returns `unavailable`, not a low value.

## Behavioural Proxy Limitations

1. **Eye Aspect Ratio is a geometric proxy.** EAR measures eye openness from
   landmark geometry (Soukupova & Cech, 2016). It does not measure fatigue,
   attention, or drowsiness directly.

2. **Blink detection has configurable thresholds.** Detection accuracy depends
   on threshold tuning, lighting, glasses, and individual anatomy. False
   positives and negatives are expected.

3. **Head-pose estimation uses a generic 3D model.** The PnP solver uses a
   canonical face geometry, not a personalized model. Accuracy varies with
   face shape, distance, and camera parameters.

4. **Mouth Aspect Ratio is not speech detection.** MAR measures vertical lip
   separation. It does not detect speech, yawning, or specific expressions.

5. **Capture quality is not engagement.** Brightness, blur, and motion scores
   describe frame quality, not user state. Poor quality must never produce a
   low-engagement label.

## Data Handling Limitations

1. **No personally identifiable information.** Participants are identified by
   pseudonymous IDs only. No names, emails, or biometric templates are stored.

2. **Raw video disabled by default.** Webcam frames are processed in memory.
   Only extracted features are stored unless explicitly enabled by the
   experimenter.

3. **No data leaves the local machine** in the default configuration.

## Task Environment, Protocol, and Replay Limitations (Milestone 4)

### Scientific status of task telemetry

1. **The task is a software telemetry source, not an instrument.** Accuracy,
   reaction time, and timeout counts describe what the task program
   observed. They are **not** engagement, attention, cognitive-load, or
   fatigue measurements.

2. **The task has not been experimentally designed.** No pilot has been run,
   no psychometric properties are known, no norms exist, and no
   institutional approval has been obtained. Until it is designed and
   approved by qualified supervision, nothing derived from it may be
   described as measuring a psychological construct.

3. **Every simulated response is fabricated.** The Python simulator draws
   responses, reaction times, and timeouts from a seeded random number
   generator. No person performs the simulated task. The lognormal shape used
   for reaction times was chosen only because it is positive and
   right-skewed; **it is not a model of human reaction times** and no
   parameter was fitted to any data.

4. **A replay is not new data.** Replaying a recording produces no new
   observation. Replayed output is permanently labelled `REPLAY`, and
   replayed synthetic output carries `SYNTHETIC` as well.

### Validation status

5. **Unity has not been compiled or executed.** No Unity Editor or Unity Hub
   is installed in the development environment, and Unity was not downloaded
   automatically. The C# client exists as source and is written against the
   checked-in protocol fixtures, but it has never been compiled, its EditMode
   and PlayMode tests have never run, and no player has been built. The
   Milestone 4 criterion "the Python simulator and Unity use the same
   versioned protocol" is therefore only **half verified**.

6. **No end-to-end test with a human operator.** The desktop task's keyboard
   input path has not been exercised by a person, only by unit-level logic
   tests that supply key presses directly.

### Engineering limitations

7. **Single process only.** The connection registry lives in one process's
   memory. Under multiple uvicorn workers, a command routed by one worker
   would never reach a client connected to another, and an observer would see
   only its own worker's traffic. Multi-worker and distributed operation are
   **not supported**.

8. **No production authentication.** The local backend has no
   authentication, no authorization, no transport encryption, and no rate
   limiting. It binds to loopback by default; binding elsewhere requires an
   explicit flag. Anyone who can reach the port can read and inject session
   data.

9. **Clocks are not synchronized.** Nothing in this system synchronizes the
   clocks of independent machines. Clock offset is reported only as an
   *estimate* from heartbeat round trips, with `rtt/2` uncertainty valid only
   under an unverified symmetric-delay assumption. Cross-process transport
   delay is reported as unavailable rather than as a number that would
   actually be measuring clock offset.

10. **Duplicate detection has a finite horizon.** The ordering tracker
    retains the most recent 4096 message ids per source to keep memory
    bounded. A duplicate arriving after more than 4096 intervening messages
    from the same source will not be detected.

11. **Durability is "flushed", not "power-cut safe".** Events are flushed to
    the operating system on the configured cadence but are not fsynced per
    record. The guarantee is that every line the OS accepted is readable, not
    that every line survives a power failure. `summary.json` is written
    atomically and is fsynced.

12. **Backpressure can lose non-critical telemetry.** When a bounded queue is
    full, non-critical messages are dropped rather than blocking the process.
    Drops are counted, logged, and written to `dropped.jsonl`, so they are
    never silent — but the data is gone. Critical messages are never dropped;
    the connection is failed instead.

13. **No adaptation policy exists.** Milestone 4 implements command
    *transport* only. Every command is issued manually or by a test script.
    No claim is made anywhere that applying a command improves engagement or
    any other outcome.
