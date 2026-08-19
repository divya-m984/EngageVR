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

## Feature Dataset and Baseline Modelling Limitations (Milestone 5)

### Scientific status

**Milestone 5 baseline-model pipeline implementation complete; scientific
evaluation on real participant-labelled data pending.**

Every number this pipeline has produced came from deterministic SYNTHETIC
data that this repository generated from a data-generating process it
chose. Those numbers measure whether the code is wired together correctly.
They are:

- **not** model accuracy;
- **not** evidence that engagement can be estimated;
- **not** evidence that cognitive load can be estimated;
- **not** evidence about generalisation to a new person;
- **not** comparable with any published result on real data;
- **not** psychological, clinical, diagnostic, or experimental evidence.

Which feature group helps a model recover a latent variable that this
software itself inserted is a fact about the generator, not about people.

### No validated labels exist

There is no approved participant engagement label and no approved
cognitive-load label in this project. No questionnaire has been selected,
no instrument has been validated, no annotation protocol has been written,
and no ethics approval has been sought or obtained.

The target schema declares five source categories
(`subjective_self_report`, `experiment_condition`, `expert_annotation`,
`public_dataset_annotation`, `synthetic_generator`). **Only the last is
populated.** The others exist so real labels can be ingested later without
redefining the schema — their emptiness is the point.

### Measurements are not labels

Task accuracy, error rate, timeout rate, and reaction time are software
measurements of what the task program observed. Difficulty level is an
experimental manipulation. Camera-based heart rate is an unvalidated
signal-processing estimate. Behavioural proxies and head pose are
geometry. Capture quality describes the measurement.

None of these is an engagement or cognitive-load label, and
`reject_automatic_derivation` refuses to convert any of them into one.

### The feature layer has never seen real data

The behavioural, head-pose, rPPG, and capture-quality aggregators have
never been run on a live webcam session. They are exercised against typed
fixtures and against the synthetic generator. Consequently:

- the minimum-evidence thresholds are engineering defaults, not validated
  cut-offs, and meeting one does not make a window scientifically
  adequate;
- the 10-second default window duration and step are defaults, not
  findings — nothing has been optimised against anything;
- the feature catalog's coverage reflects what the earlier milestones
  implemented, not what a validated engagement model would need.

### Evaluation-design limitations

- Grouped cross-validation prevents leakage **between the groups a dataset
  actually declares**. With one participant per session and no repeated
  participants, session grouping is the strongest available guarantee and
  is weaker than participant grouping.
- Passing the scientific-mode gate establishes **eligibility, not
  validity**. It checks data source and target provenance. It cannot check
  whether a label means what its instrument claims.
- Fold aggregation weights folds equally. With very unequal group sizes,
  the aggregate is not the pooled score and should not be read as one.
- Hyperparameter grids are deliberately tiny. Nothing here is a claim
  about achievable performance.
- The rule-based estimators are **software checks**. They threshold one
  arbitrarily chosen feature, their probabilities are not calibrated, and
  they are not validated indicators of anything.
- No model is selected as a champion and none is production-ready.

### Calibration limitations

- Calibration is fitted on groups disjoint from those used to fit the base
  estimator and never on the outer test fold. It is still a *sample*
  estimate, and a small calibration set produces an unstable one — which
  is why isotonic calibration is refused below 50 rows or 10 rows per
  class rather than silently produced.
- When a fold's calibration set is empty or missing a class, calibration
  is skipped and the reason is recorded. Aggregate calibration metrics
  therefore have a smaller valid-fold count than the accuracy metrics
  beside them, and the count is reported alongside every aggregate.
- A calibrated probability is **not certainty** and is **not signal
  quality**. Model probability and signal quality are separate fields
  throughout.
- Abstention, coverage-versus-performance analysis, and any online
  confidence policy are **not implemented**. They belong to Milestone 7.

### Interpretation limitations

- Association is not causation. A feature a model leans on is not a
  measurement of the construct being modelled.
- Correlated features share credit arbitrarily. The dataset contains
  genuine collinearity — three response proportions summing to one, order
  statistics ordered by construction — so a linear model may split a
  coefficient between near-duplicates and a permutation test may score
  both as unimportant because either substitutes for the other.
- Interpretation data is recorded even when a model does not beat chance,
  with a warning attached. Reading importances from such a model describes
  noise.
- SHAP is not used in this milestone.

### Not implemented in Milestone 5

Multimodal-fusion architectures, modality masks, quality-aware weighting,
early-versus-late fusion comparison, temporal neural networks,
personalisation, online inference, confidence-based abstention, adaptation
policy, dashboard pages, MLflow, DVC, Docker, and deployment. `all_available`
in the ablation set means "no feature group was removed"; it is not a
fusion architecture. Fusion arrived in Milestone 5's successor and is
described below; the ablation set was **not** renamed.

### No medical, psychological, or adaptive-effectiveness claim

Nothing in the modelling layer supports a medical, diagnostic, screening,
monitoring, psychological, or clinical conclusion, and nothing here shows
that adapting an environment on the basis of these estimates would help
anyone.

## Multimodal-Fusion Limitations (Milestone 6)

### Scientific status

**Milestone 6 multimodal-fusion implementation complete; scientific
evaluation on real participant-labelled multimodal data pending.**

Every fusion number this repository has produced came from SYNTHETIC data.
Which fusion architecture best recovers a latent variable that this
repository itself inserted is a fact about the generator, not a fact about
fusion, about engagement, or about any person. No strategy is a champion,
none is validated, and none is production-ready.

### Fusion does not create validity

Combining four unvalidated measurement channels produces one unvalidated
estimate. Fusion can improve robustness to a missing channel, and it can
make the contribution of each channel inspectable. It cannot make an
engagement or cognitive-load estimate valid, and nothing here shows that it
does.

### Signal quality is still not a state

Quality-aware fusion weights an expert by how usable its signal was. A low
weight means the camera or task signal was poor. It is never low
engagement, never high cognitive load, and never a statement about the
person. Quality and model probability are kept in separate fields on every
record precisely so the two cannot be confused.

Modality quality itself has never been validated against anything external.
The rPPG quality index in particular is an interpretable engineering
construction with equal component weights and hard gates (DEC-021), not a
calibrated measure of signal fidelity.

### Missing-modality results are software results

The ten scenarios and the seeded synthetic dropout describe how this code
behaves when told a modality is absent. No real camera has failed, no real
rPPG window has been rejected in the field, and no real task telemetry has
gone missing here. Coverage numbers from those scenarios are not real-world
robustness results.

Scenarios are applied at evaluation time: the models were trained once on
the recorded availability. A system trained on a genuinely rPPG-free
dataset would be a different system, and nothing here measures it.

### Expert disagreement is not uncertainty

Disagreement diagnostics describe ensemble spread. They are not calibrated
uncertainty, not confidence, and not signal quality, and nothing in this
milestone abstains on them. Formal uncertainty-aware inference and
abstention are Milestone 7.

### Thresholds are engineering defaults

The expert minimum-evidence gates (10 fit rows, 2 independent groups), the
minimum meta-training row count (20), the neutral missing-quality fallback
(0.5), and the minimum effective weight (1e-9) are engineering defaults.
None is an empirically validated cut-off, and meeting one does not make a
window, a fold, or a fusion scientifically adequate. `minimum_quality`
defaults to 0.0 for exactly this reason: no validated quality cut-off
exists to put there.

### Personalization has been verified as software only

Milestone 6 implements per-participant calibration: a personal-baseline
z-score, a few-shot correction, an explicit cold-start path, and separate
population-versus-personalized reporting over identical evaluation windows.
Every limitation of the fusion layer above applies to it unchanged, plus:

- **No personalized model here has been fitted to a real participant
  label.** No subject in any run is a person; every one is a synthetic
  identifier this repository generated.
- **There is no evidence of a personalization benefit, in either
  direction.** On this repository's generator the personalized variants
  score *worse* than the population baseline on all four targets. That is a
  property of a generator whose targets track absolute feature levels —
  which within-subject z-scoring removes — not a property of
  personalization, and not a property of people. It is reported as observed
  rather than tuned away.
- `RQ2` in `docs/RESEARCH_QUESTIONS.md` — do personalized baselines
  outperform population models — is **unanswered**. A synthetic self-check
  cannot answer it.
- The minimum-evidence gates (3 calibration windows, 2 calibration classes,
  3 finite values per feature) and the correction constants
  (`kappa = 5.0`, `alpha = 1.0`) are engineering defaults. None is
  validated, and none was tuned: tuning on synthetic data would fit the
  generator.
- The calibration/evaluation boundary has never been exercised against a
  real session, a real pause, or a real device dropout. It has only met
  windows this repository laid out on a regular grid.
- A subject's personal baseline is estimated from as few as three finite
  values per feature. Whether that is enough for any real signal is
  unknown.
- Personalized calibration here is **subject adaptation**, not uncertainty
  calibration. It produces no confidence estimate and withholds no
  prediction.

### Not implemented in Milestone 6

Temporal neural networks, online inference, confidence-based abstention,
selective prediction, personalized confidence thresholds, adaptation
policy, dashboard pages, MLflow, DVC, Docker, and deployment. Those
thresholds and abstention are Milestone 7; the only thresholds in the
personalization layer decide whether a *correction is fitted*, never
whether a *prediction is issued*.

No per-subject model is trained from scratch: a handful of calibration
windows cannot support one, and a per-subject estimator fitted on five
windows would describe those five windows.

No deep or neural fusion exists. Feature concatenation is called
concatenation.

### No medical, psychological, or adaptive-effectiveness claim

Nothing in the fusion layer supports a medical, diagnostic, screening,
monitoring, psychological, or clinical conclusion, and nothing here shows
that adapting an environment on the basis of a fused estimate would help
anyone.


## Milestone 7: uncertainty-aware inference and abstention

### Every threshold is an engineering default

`population_confidence_threshold: 0.70`, `alpha: 0.10`,
`minimum_personal_calibration_windows: 5`, `personal_target_coverage: 0.80`,
`personal_shrinkage_constant: 10.0`, and every minimum-evidence count are
**engineering defaults**. None was selected by looking at a result, none is
empirically optimal, none is validated, and none is a production threshold.

### No confidence value here has been checked against an outcome

Calibration was fitted on SYNTHETIC labels. A calibrated probability states
how often outcomes of a kind occurred at a predicted probability *in the
evaluated folds*, and every one of those folds came from data this
repository generated. Nothing here says how often a real person is engaged
when the model reports 0.8.

### Conformal coverage assumes exchangeability, which grouping violates

Split conformal prediction guarantees marginal coverage of at least
`1 - alpha` **when the calibration and test points are exchangeable**.
Under grouped cross-validation those rows come from **different people**,
and a physiological or behavioural signal is not exchangeable between one
person and another.

On the 30-subject synthetic dataset, empirical interval coverage varies
between **0.846 and 0.963 per fold** against a nominal 0.90. The cross-fold
mean lands near nominal, but the per-fold spread is wide — which is what a
violated exchangeability assumption produces. A mean that happens to land
near nominal is not the guarantee holding, and on a different set of
subjects it need not.

**No conformal coverage guarantee is claimed for real EngageVR data**, and
per-fold coverage must be measured on real subjects before any coverage
claim is made.

### A synthetic coverage curve is not evidence

A curve computed on synthetic data describes the generator this repository
wrote. It is not evidence of real-world calibration, reliability, safety,
or usefulness, and a lower area under the risk-coverage curve establishes
none of those things.

### The regression coverage curve is trivial by construction

Split conformal produces one quantile per fold, so every window in a fold
shares one interval width and the width sweep is all-or-none. That is a
property of the method, stated rather than smoothed over. No more
interesting curve was manufactured.

### Personalized thresholds are label-free by design, and limited by it

The per-subject threshold reads only the subject's own earlier confidence
scores. That makes leakage structurally impossible, but it also means the
rule cannot target accepted accuracy for that subject — it can only
preserve a target *acceptance rate*. Whether that is the right per-subject
objective is unknown and untested.

Subject-conditional conformal intervals are **not implemented**: a
per-subject residual distribution from a handful of calibration windows
would overfit, and doing it with labels would put a subject's own outcomes
into their own interval.

### An abstention rate is not a safety property

A high abstention rate means the declared rule was often unsatisfied. It
does not mean the accepted predictions are correct, trustworthy, or safe to
act on, and no result here shows that abstaining improves anything.

### Not implemented in Milestone 7

Adaptation policy, action selection, cooldown, hysteresis, manual override,
static-versus-adaptive experimental modes, online inference, any new API
route, any transport-protocol change, dashboard pages, MLflow, DVC, Docker,
and deployment. The Milestone 7 adaptation **gate** can block an
already-chosen action; it cannot choose one.

### No safety, psychological, or clinical claim

Nothing in the uncertainty layer supports a medical, diagnostic, screening,
monitoring, psychological, or clinical conclusion. A confidence score is
not certainty, not psychological confidence, and not safety; an abstention
is not a guarantee that anything harmful was avoided.
