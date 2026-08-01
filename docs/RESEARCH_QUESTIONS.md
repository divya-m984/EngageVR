# EngageVR -- Research Questions

## Overview

The research questions below define what EngageVR is designed to investigate.
None of these questions have been experimentally tested yet. The software
prototype is being built so that these questions can eventually be investigated
under appropriate laboratory conditions with institutional approval.

All engagement and cognitive-load outputs are model estimates. They are not
medical, psychological, or diagnostic conclusions.

## Primary Research Questions

### RQ1: Multimodal Fusion vs. Single Modality

**Does multimodal fusion of behavioural, physiological, and task-performance
signals produce more reliable engagement and cognitive-load estimates than any
single modality alone?**

- Compare individual-modality models against early and late fusion.
- Measure prediction performance and calibration for each configuration.
- Report results separately for each data source (public dataset, synthetic).

### RQ2: Personalized vs. Population-Level Models

**Do personalized baselines (per-user calibration, z-score normalization)
outperform population-level models for engagement and cognitive-load estimation?**

- Compare population model, population + user correction, and few-shot
  calibrated models.
- Evaluate cold-start behaviour when no user history is available.
- Use participant-aware cross-validation to prevent data leakage.

### RQ3: Uncertainty Estimation and Inappropriate Adaptations

**Can uncertainty estimation (calibrated confidence, ensemble disagreement, or
conformal prediction) prevent inappropriate environment adaptations when
evidence is insufficient?**

- Measure selective prediction performance (coverage vs. accuracy).
- Count adaptations made under low confidence and assess their outcomes.
- Compare uncertainty-gated adaptation against always-adapt baseline.

### RQ4: Adaptive vs. Static VR Environments

**Does an adaptive virtual environment that adjusts task difficulty, feedback,
and pacing based on estimated engagement and cognitive load improve task
performance and subjective experience compared with a static environment?**

- Requires a controlled study (static vs. adaptive conditions).
- Dependent variables: task accuracy, reaction time, subjective engagement,
  perceived cognitive load, adaptation acceptance.
- This question cannot be answered until participant studies are approved.

### RQ5: Webcam-Based rPPG Robustness

**How robust is webcam-based remote photoplethysmography under participant
movement and illumination changes typical of desktop VR use?**

- Evaluate rPPG signal quality across motion and lighting conditions.
- Compare green-channel, CHROM, and POS methods.
- Validate against public rPPG datasets with reference PPG/ECG where available.
- Report signal-quality distributions, not just aggregate accuracy.

## Secondary Research Questions

### RQ6: Physiological Estimates vs. Subjective Feedback

**How closely do physiological estimates (HR, HRV features) and behavioural
indicators agree with subjective engagement and cognitive-load self-reports?**

- Correlation analysis between modalities and subjective responses.
- Requires validated subjective instruments and participant data.

### RQ7: Missing Modalities and Prediction Reliability

**How do missing modalities (e.g., poor rPPG, face not detected, no task data)
affect prediction reliability, and can the system detect when it does not have
enough evidence to make a prediction?**

- Systematically ablate modalities and measure performance degradation.
- Evaluate abstention rate and its correlation with actual prediction error.

### RQ8: Temporal Modelling

**Does temporal modelling (sequence models over feature windows) outperform
independent-window classification for engagement estimation?**

- Compare window-level classifiers against LSTM, GRU, or TCN on the same
  feature set.
- Only pursue after interpretable baselines are evaluated (Milestone 5+).

### RQ9: Adaptation Strategies

**Which adaptation strategies (difficulty adjustment, feedback frequency,
pacing, break timing) produce measurable improvements in engagement and task
performance without disrupting the participant?**

- Requires within-subject or between-subject experimental design.
- Requires participant studies with appropriate approval.

### RQ10: Sufficient Evidence Detection

**Can the system reliably detect when it does not have enough evidence to
produce a trustworthy engagement or cognitive-load estimate?**

- Evaluate abstention decisions against ground truth (where available).
- Measure false-confident predictions vs. unnecessary abstentions.

## Methodological Constraints

- No research question can be answered with synthetic data alone.
- Public-dataset experiments validate individual components, not the full
  system.
- System-level validation requires participant studies under controlled
  conditions with institutional ethics approval.
- All results must clearly state the data source, sample size, and limitations.
- Physiological signals are proxies and associations, not direct measurements
  of engagement or cognitive load.
