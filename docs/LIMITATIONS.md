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
