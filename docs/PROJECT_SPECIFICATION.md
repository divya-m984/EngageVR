You are my lead research engineer, scientific software architect, and implementation partner. I want you to help me build the complete software prototype described below, not merely write a proposal or generate disconnected code samples.

PROJECT IDENTITY

Project name: EngageVR

Full research title:EngageVR: An Uncertainty-Aware Multimodal Framework for Personalized Engagement Estimation and Adaptive Virtual Reality

Repository name: engagevr

Primary objective a scientifically credible, modular software prototype that estimates engagement and cognitive load from multimodal behavioural, physiological, and task-performance signals. The estimated state, signal quality, and model confidence will be used to adapt a Unity-based virtual environment.

This project is being prepared for potential research work in an Applied Perception Engineering laboratory studying virtual reality, affective computing, physiological measurement, remote photoplethysmography, HRV, subjective evaluation, and human perception.

PROJECT BACKGROUND

I am currently developing EngageVR as a software-first system.

The intended modalities are:

Facial behaviour

Head movement

Camera-based remote photoplethysmography, or rPPG

Heart rate and heart-rate variability

Task-performance telemetry

Subjective engagement and cognitive-load feedback

Optional future wearable signals such as PPG, ECG, EDA, and respiration

The system should estimate:

Engagement level

Cognitive-load level

Possible disengagement or fatigue trends

Prediction confidence

Signal quality

Whether sufficient evidence exists to make an adaptation

The Unity environment should be capable of adapting:

Task difficulty

Instruction pace

Information density

Feedback frequency

Visual complexity

Audio intensity

Break timing

Recovery or breathing prompts

CURRENT REAL-WORLD CONSTRAINTS

I do not currently have access to:

A VR headset

Research-grade PPG, ECG, EDA, or respiration sensors

A controlled participant-testing room

A validated multimodal participant dataset collected by me

Institutional approval for human-subject experimentation

I currently have access to:

A normal laptop

A webcam

Python

Unity or the ability to install Unity

Public datasets

Synthetic streams for software testing

Standard development tools

Arch Linux

Git and GitHub

Docker

VS Code

Claude Code or Codex

The project must therefore work in a software-only desktop mode without a VR headset. The Unity environment must be usable through a normal monitor, keyboard, and mouse. Hardware integrations must be represented through clean adapter interfaces so real sensors can be added later.

Do not pretend that I have completed participant studies, collected research-grade signals, or clinically validated engagement or cognitive load.

SCIENTIFIC POSITIONING

EngageVR is a research prototype, not a medical or diagnostic system.

Do not claim that:

HRV directly proves engagement

rPPG diagnoses anxiety

Facial expressions reveal a person’s true mental state

Physiological signals provide perfect ground-truth labels

Cognitive load can be inferred reliably from one signal alone

The system should describe its outputs as estimates, proxies, associations, or model predictions.

Physiological and behavioural signals must complement:

Task-performance measurements

Subjective feedback

Experimental conditions

Signal-quality indicators

The software must distinguish between:

Low engagement

High cognitive load

Poor sensor quality

Missing data

Low model confidence

A poor-quality signal must never automatically be interpreted as disengagement.

CORE RESEARCH QUESTIONS

Design the system so it can eventually investigate:

Does multimodal fusion outperform individual modalities?

Do personalized baselines outperform population-level models?

Can uncertainty estimation prevent inappropriate environment adaptations?

Does adaptive VR improve engagement or task performance compared with a static environment?

How robust is webcam-based rPPG under movement and illumination changes?

How closely do physiological estimates agree with subjective feedback and task performance?

How do missing modalities affect prediction reliability?

Can the system detect when it does not have enough evidence to make a prediction?

Does temporal modelling outperform independent-window classification?

Which adaptation strategies produce measurable improvements without disrupting the participant?

REQUIRED SOFTWARE ARCHITECTURE

Build the system as independent but integrated components.

Capture layer

Create modules for:

Webcam video capture

Face detection

Facial landmarks

Head pose

Head-motion features

Facial activity features

Region-of-interest extraction for rPPG

Task-performance event capture

Subjective questionnaire capture

Simulated sensor streams

Future wearable-device adapters

Every modality must use a common timestamp system.

Signal-processing layer

Implement configurable pipelines for:

Frame-quality assessment

Face-detection confidence

Motion estimation

Illumination-change detection

Skin-region extraction

RGB trace extraction

Detrending

Band-pass filtering

Normalization

rPPG waveform extraction

Heart-rate estimation

Peak detection

Inter-beat interval estimation

HRV feature extraction where the signal duration and quality permit it

Window aggregation

Missing-data handling

Artifact detection

Signal-quality scoring

Possible rPPG methods may include:

Green-channel baseline

CHROM

POS

A maintained learning-based method as a later extension

Start with interpretable signal-processing baselines before deep learning.

All algorithms must have references documented in the repository. Before implementing a published method, verify the original paper or an authoritative implementation.

Feature layer

Organize features into categories.

Facial and behavioural:

Blink rate

Eye-closure duration

Facial landmark movement

Mouth movement

Head yaw

Head pitch

Head roll

Head-motion velocity

Head-motion variability

Face-detection stability

Gaze-related proxies only where technically defensible

Physiological:

Estimated heart rate

Mean inter-beat interval

SDNN

RMSSD

pNN50 where valid

Frequency-domain HRV only when the recording duration and quality are sufficient

rPPG signal-to-noise ratio

Peak-detection confidence

Missing-signal percentage

Task performance:

Accuracy

Reaction time

Error rate

Completion time

Hint usage

Inactivity duration

Response consistency

Difficulty level

Number of retries

Subjective feedback:

Self-reported engagement

Perceived task difficulty

Mental effort

Fatigue

Comfort

Presence or immersion

Optional NASA-TLX-compatible fields

Optional short engagement-rating fields

Do not use copyrighted questionnaire wording without checking usage conditions. Build configurable generic questionnaire components where necessary.

Data and synchronization layer

Create a clear schema for:

Session

Participant pseudonym

Experiment condition

Timestamp

Modality

Raw value

Processed value

Feature window

Signal quality

Prediction

Confidence

Adaptation event

Subjective response

Task result

Use pseudonymous IDs. Do not store names, email addresses, or unnecessary identifying information.

Preferred local storage:

Parquet for time-series and feature data

JSON for configuration and metadata

SQLite or PostgreSQL for session summaries

Optional object storage adapter later

Create explicit schema validation.

Machine-learning layer

Implement the modelling process in stages.

Stage A: deterministic demo mode

Use a clearly labelled synthetic stream to test:

Data flow

Predictions

Confidence propagation

Adaptation commands

Dashboard updates

Synthetic data must never be presented as experimental evidence.

Stage B: interpretable baselines

Implement:

Logistic regression

Random forest

XGBoost or another maintained gradient-boosting implementation

Simple rule-based baseline

Stage C: multimodal fusion

Support:

Early fusion

Late fusion

Modality-specific models

Missing-modality masks

Signal-quality-aware weighting

Stage D: temporal modelling

Only after baselines work, consider:

LSTM

GRU

Temporal convolutional network

Transformer-based temporal model

Do not use a deep model merely because it is more complex.

Stage E: personalization

Support:

Per-user baseline calibration

Z-score normalization relative to a personal baseline

Few-shot calibration

Population model plus user-specific correction

Cold-start mode

Personalized thresholds

Stage F: uncertainty and calibration

Implement at least one defensible method such as:

Calibrated probabilities

Ensemble disagreement

Bootstrap uncertainty

Conformal prediction

Monte Carlo dropout for a suitable neural model

Measure:

Expected calibration error

Brier score

Reliability diagrams

Selective prediction performance

Coverage versus accuracy

The model must be allowed to abstain.

Possible output:

{"engagement_estimate": 0.68,"cognitive_load_estimate": 0.74,"engagement_class": "moderate","cognitive_load_class": "high","confidence": 0.61,"signal_quality": 0.82,"available_modalities": ["face","head_pose","rppg","task_performance"],"missing_modalities": ["eda","wearable_ecg"],"abstain": false,"reason": null}

If evidence is insufficient:

{"engagement_estimate": null,"cognitive_load_estimate": null,"confidence": 0.21,"signal_quality": 0.29,"abstain": true,"reason": "Insufficient rPPG quality and unstable face tracking"}

DATASET STRATEGY

Search for suitable public datasets only through official project pages, original papers, institutional repositories, or trusted dataset repositories.

Possible categories to investigate:

Webcam rPPG datasets

PPG and ECG datasets

Stress and affect datasets

Engagement datasets

Cognitive-load datasets

Facial-behaviour datasets

Multimodal affect datasets

Before using any dataset:

Verify its license.

Document its source.

Record permitted use.

Record modalities.

Record participant count.

Record sampling rates.

Record labels.

Record demographic limitations.

Record known biases.

Do not commit restricted raw data to Git.

Do not merge unrelated datasets and pretend they form synchronized multimodal samples.

Keep dataset-specific experiments separate where labels or modalities are incompatible.

Create dataset adapters rather than hard-coding one dataset format.

If no suitable joint multimodal dataset exists, use public datasets to validate individual components separately and use a clearly labelled simulated synchronized stream for integration testing.

UNITY COMPONENT

Build a simple Unity desktop environment first.

The first environment should be a controlled cognitive task rather than a visually excessive game.

Suggested initial task:

A sequence-classification, working-memory, visual-search, or response-inhibition task

Configurable difficulty

Timed trials

Accuracy and reaction-time collection

Break periods

Adjustable visual complexity

Adjustable feedback frequency

Adjustable instruction pace

Unity must send events such as:

Session started

Trial started

Stimulus presented

Response received

Correct or incorrect

Reaction time

Difficulty changed

Break triggered

Session ended

Unity must receive adaptation commands such as:

{"command": "set_difficulty","value": 3,"reason": "High estimated engagement and moderate cognitive load","confidence": 0.79}

Use a local communication mechanism such as:

WebSocket

HTTP

Local message broker only if justified

Prefer a simple FastAPI WebSocket bridge for the first implementation.

Provide a non-Unity simulator that can send and receive the same messages so the backend can be developed even when Unity is not running.

ADAPTATION POLICY

Start with an interpretable bounded policy before reinforcement learning.

Required safety controls:

Confidence threshold

Signal-quality threshold

Minimum observation window

Adaptation cooldown

Hysteresis

Maximum difficulty change per adaptation

No adaptation during unreliable sensing

Manual override

Experimenter lock

Complete adaptation log

Example policy:

High engagement + low load difficulty slightly

Moderate engagement + moderate load current state

Declining engagement + low or moderate load feedback or introduce variation

High load + declining performance difficulty or reduce information density

Low confidence or low signal quality not adapt

Sustained fatigue indicators a break

Every adaptation must store:

Time

Previous state

New state

Model estimates

Confidence

Signal quality

Triggering evidence

Rule used

Whether manual or automatic

DEFAULT WINDOWING

Make all values configurable.

Reasonable initial software defaults may include:

Webcam target: approximately 30 frames per second

Facial/head features: frame-level capture with 5–10 second aggregates

rPPG estimation: approximately 20–30 second rolling windows

HRV: only calculate features when sufficient reliable inter-beat data exists

Model inference: every 1–5 seconds

Adaptation cooldown: at least 20–30 seconds

Baseline calibration period: approximately 1–3 minutes

Do not calculate short-window HRV features when they are not scientifically meaningful. Return unavailable instead.

DASHBOARD

Build a Streamlit dashboard for development and experiment monitoring.

Required pages:

Project overview

Live session

Webcam and face-tracking diagnostics

rPPG waveform and signal quality

Heart-rate and HRV features

Facial and head-motion features

Task performance

Engagement and cognitive-load estimates

Confidence and uncertainty

Adaptation timeline

Session comparison

Dataset documentation

Model performance

Calibration analysis

System status

Privacy and limitations

The dashboard must visibly distinguish:

Live data

Public dataset data

Synthetic demo data

Missing data

Unreliable data

Model prediction

Subjective response

Never present synthetic data without a permanent visible label.

MLOPS AND REPRODUCIBILITY

Use the rigor of a real research software project.

Include:

Python virtual environment

pyproject.toml

Locked or pinned dependencies

Ruff

Type checking

Pytest

Pre-commit hooks

Structured logging

Configuration management

DVC for reproducible dataset and pipeline stages where appropriate

MLflow for experiment tracking

Docker for Python services

GitHub Actions

Model registry structure

Data validation

Feature validation

Drift checks

Reproducible random seeds

Environment diagnostics

Makefile or task runner

Clear local setup instructions

Do not introduce MLflow, DVC, Docker, or distributed infrastructure before the basic local pipeline works. Add them incrementally.

PREFERRED TECHNOLOGY STACK

Use practical maintained tools.

Python side:

Python 3.12 unless the local environment requires another compatible version

FastAPI

Pydantic

OpenCV

NumPy

SciPy

pandas

scikit-learn

XGBoost if compatible

PyTorch only when temporal modelling is introduced

MediaPipe or a maintained equivalent for facial landmarks and head pose

Streamlit

Plotly or Matplotlib

SQLAlchemy

PyArrow

MLflow

DVC

Pandera or Pydantic-based dataframe validation

Unity side:

C#

A currently supported Unity LTS version

Unity UI Toolkit or a simple maintained UI approach

WebSocket or HTTP communication

Infrastructure:

Docker

GitHub Actions

Git

Linux-compatible scripts

Before choosing exact package versions, check official documentation and compatibility. Pin only versions that have been tested together.

PROPOSED REPOSITORY STRUCTURE

Create or evolve toward this structure:

engagevr/├── README.md├── LICENSE├── pyproject.toml├── Makefile├── .env.example├── .gitignore├── docker-compose.yml├── configs/│   ├── capture.yaml│   ├── signal_processing.yaml│   ├── model.yaml│   ├── adaptation.yaml│   └── experiment.yaml├── docs/│   ├── PROJECT_PLAN.md│   ├── ARCHITECTURE.md│   ├── RESEARCH_QUESTIONS.md│   ├── DATASETS.md│   ├── EXPERIMENT_PROTOCOL.md│   ├── ETHICS_AND_PRIVACY.md│   ├── LIMITATIONS.md│   ├── HARDWARE_INTEGRATION.md│   ├── PROGRESS.md│   └── DECISIONS.md├── data/│   ├── README.md│   ├── raw/│   ├── interim/│   ├── processed/│   └── synthetic/├── models/├── artifacts/├── notebooks/├── unity/│   └── EngageVR/├── src/│   └── engagevr/│       ├── capture/│       ├── face/│       ├── head_pose/│       ├── rppg/│       ├── physiology/│       ├── task/│       ├── questionnaires/│       ├── synchronization/│       ├── schemas/│       ├── features/│       ├── datasets/│       ├── training/│       ├── inference/│       ├── uncertainty/│       ├── personalization/│       ├── adaptation/│       ├── api/│       ├── dashboard/│       ├── simulator/│       ├── storage/│       └── utils/├── scripts/├── tests/│   ├── unit/│   ├── integration/│   ├── system/│   └── fixtures/└── .github/└── workflows/

Adjust the structure only when there is a clear architectural reason.

IMPLEMENTATION MILESTONES

Milestone 0: repository audit and plan

Inspect the current repository if it exists.

Do not overwrite working files.

Identify installed tools and versions.

Write PROJECT_PLAN.md.

Write ARCHITECTURE.md.

Write RESEARCH_QUESTIONS.md.

Write LIMITATIONS.md.

Produce a milestone checklist.

Record architectural decisions.

Milestone 1: foundation

Create Python package.

Configure virtual environment.

Add pyproject.toml.

Add linting and tests.

Add configuration loading.

Add logging.

Add common timestamped schemas.

Add session metadata model.

Add synthetic event generator.

Add initial CI.

Acceptance criteria:

Clean installation

Tests pass

Synthetic session can be generated

Schemas reject invalid data

Milestone 2: webcam behavioural capture

Webcam capture

Face detection

Facial landmarks

Head-pose estimation

Blink and eye-closure proxies

Head-motion features

Capture-quality metrics

Privacy-preserving option that stores features without storing video

Acceptance criteria:

Live capture runs on a normal laptop

Frame rate is reported

Missing-face conditions are handled

Features are timestamped

Tests exist for feature calculations

Milestone 3: rPPG pipeline

Skin-region extraction

RGB trace creation

Motion and illumination diagnostics

Baseline green-channel method

POS or CHROM method

Filtering

Heart-rate estimation

Signal-quality index

Visualization

Public-dataset evaluation adapter

Acceptance criteria:

Pipeline works on a known public sample

Signal quality is reported

Unreliable windows return unavailable

No fake accuracy claims

Milestone 4: task environment and simulator

Implement task-event schema

Build Python task simulator

Build basic Unity desktop task

Send task telemetry to backend

Receive adaptation commands

Store synchronized event history

Acceptance criteria:

Backend works without Unity

Unity and simulator use the same protocol

Full session replay is possible

Milestone 5: baseline models

Create windowed feature dataset

Train interpretable models

Add grouped or participant-aware splitting when applicable

Add cross-validation

Add calibration

Add confusion matrices and regression metrics as relevant

Add ablation experiments

Track experiments

Acceptance criteria:

No leakage between participant sessions

Metrics are reproducible

Data origin is documented

Synthetic data is excluded from scientific evaluation

Milestone 6: multimodal fusion

Add modality masks

Add quality-aware fusion

Compare early and late fusion

Add missing-modality tests

Add personalized calibration

Acceptance criteria:

System remains functional with missing signals

Quality-aware fusion is compared with naive fusion

Personalized and population baselines are separately reported

Milestone 7: uncertainty-aware inference

Calibrate model probabilities

Add abstention logic

Add coverage-versus-performance analysis

Expose confidence through API

Prevent adaptation when confidence is insufficient

Acceptance criteria:

Predictions can abstain

Confidence is not confused with signal quality

Adaptation policy respects both thresholds

Milestone 8: adaptive environment

Implement rule-based adaptation engine

Add cooldown and hysteresis

Add manual control

Log every adaptation

Add static and adaptive experimental modes

Acceptance criteria:

No rapid adaptation oscillation

Same input produces reproducible policy output

Experimenter can disable adaptation

Static and adaptive modes are clearly separated

Milestone 9: dashboard

Implement all core monitoring pages

Show real-time and replay modes

Show signal-quality warnings

Show uncertainty

Show adaptation history

Add exportable session report

Acceptance criteria:

Dashboard runs locally

Synthetic, public, and live data are labelled

Missing or unreliable measurements are visible

Session report can be reproduced

Milestone 10: MLOps and packaging

Add MLflow

Add DVC stages

Add Docker

Add GitHub Actions

Add model versioning

Add drift checks

Add system smoke tests

Add release instructions

Acceptance criteria:

Clean clone can reproduce the demo

CI passes

Model artifact and configuration are versioned

Dockerized backend and dashboard work

Milestone 11: research documentation

Create:

Research proposal

Research questions

Hypotheses

Experimental variables

Static versus adaptive experiment design

Data-collection protocol

Consent-template draft

Risk assessment

Privacy strategy

Statistical-analysis plan

Dataset cards

Model cards

Limitations

Hardware-validation plan

Future laboratory extension plan

Do not claim ethical approval. Mark all human-subject documents as drafts requiring institutional review.

EXPERIMENTAL DESIGN TARGET

Prepare the software for an eventual controlled study with:

Independent variable:

Static environment

Adaptive environment

Possible dependent variables:

Task accuracy

Reaction time

Completion time

Subjective engagement

Perceived cognitive load

HR and HRV changes where valid

Behavioural indicators

Adaptation acceptance

Simulator sickness or discomfort where applicable

Possible controls:

Baseline rest period

Fixed task order or counterbalancing

Consistent lighting

Consistent webcam placement

Consistent task duration

Participant-specific baseline

Static difficulty condition

Adaptation-disabled condition

Prepare the protocol, but do not run or claim a participant study without approval and suitable supervision.

TESTING REQUIREMENTS

Tests must cover:

Schema validation

Timestamp synchronization

Missing modalities

Poor signal quality

Face absent

Webcam unavailable

No detected peaks

Too-short HRV windows

Model abstention

Adaptation cooldown

Adaptation hysteresis

WebSocket messages

Unity simulator communication

Data export

Reproducibility

Public dataset adapter

Synthetic-data labels

Privacy settings

Include integration and smoke tests, not only unit tests.

ENGINEERING RULES

Do not generate placeholder code that is never connected.

Do not fabricate metrics, participants, datasets, sensor readings, or validation results.

Do not claim clinical validity.

Do not silently replace missing values with misleading defaults.

Do not calculate features when minimum quality requirements are not met.

Keep modules small and typed.

Write tests with each feature.

Update documentation after every milestone.

Store configuration outside source code.

Preserve compatibility with Linux.

Avoid unnecessary cloud dependencies.

Keep the first complete demo locally runnable.

Prefer interpretable baselines before deep learning.

Explain major design decisions.

Use official documentation and primary research sources for technical methods.

Record every external dataset and algorithm reference.

Never expose webcam frames outside the local machine.

Make raw-video storage disabled by default.

Do not commit datasets, model artifacts, secrets, or personal data.

Add clear disclaimers wherever synthetic or nonvalidated outputs are displayed.

README REQUIREMENTS

The README must include:

Project title

Research motivation

Problem statement

Current status

Software-only constraint

Scientific disclaimer

Architecture diagram

Supported modalities

Installation

Quick-start demo

Webcam mode

Simulator mode

Unity mode

Dataset setup

Training

Evaluation

Dashboard

Tests

Docker

Limitations

Privacy

Research roadmap

Hardware-extension roadmap

Citation section

License

It must clearly state:

“EngageVR is a research software prototype. Its engagement and cognitive-load outputs are model estimates and must not be treated as medical, psychological, or diagnostic conclusions.”

REQUIRED FIRST DEMO

The first end-to-end demo must work without external hardware.

Flow:

Start backend.

Start dashboard.

Start synthetic task simulator or Unity desktop task.

Optionally enable webcam capture.

Capture facial/head features.

Estimate rPPG only when signal quality permits.

Generate task events.

Produce an engagement and cognitive-load estimate.

Produce confidence and signal-quality values.

Abstain when evidence is insufficient.

Send an adaptation command.

Display the event in the dashboard.

Save the complete session.

Replay the session later.

WORKING STYLE

Work incrementally and behave like a senior engineer maintaining a real repository.

At the start:

Inspect the repository and current environment.

Summarize what already exists.

Identify missing dependencies.

Propose the final architecture.

Create a milestone plan.

Begin with Milestone 0.

Do not attempt the entire project in one uncontrolled code dump.

After each milestone, report:

What was implemented

Files created

Files modified

Commands to run

Tests executed

Test results

Remaining limitations

Next milestone

Any decisions I need to make

Do not repeatedly ask me to approve obvious implementation details. Make sensible engineering decisions and document them. Ask questions only when a choice would materially change the research direction, architecture, privacy model, or required hardware.

When a command fails:

Diagnose the actual cause

Fix it

Do not hide the failure

Do not report success until the relevant tests pass

Keep docs/PROGRESS.md updated throughout development.

MY RELEVANT BACKGROUND

I am a third-year B.Tech Artificial Intelligence and Data Science student.

My technical background includes:

Python

TypeScript

JavaScript

Bash

FastAPI

React

Vite

Node.js

Express

NestJS

scikit-learn

XGBoost

MLflow

DVC

Streamlit

PostgreSQL

MongoDB

Docker

GitHub Actions

Git

Linux and Arch Linux

I previously developed Oceanographic, an end-to-end machine-learning and MLOps prototype for multivariate marine environmental sensor data. I am comfortable with reproducible pipelines, validation, model evaluation, APIs, dashboards, Docker, testing, and monitoring.

Do not simplify the project into a beginner tutorial. Build it as a serious undergraduate research prototype while keeping the first version feasible on my current hardware.

START NOW

Begin by:

Inspecting the current directory and repository.

Checking Python, Git, Docker, Unity-related files, and available system dependencies.

Creating or updating docs/PROJECT_PLAN.md.

Creating docs/ARCHITECTURE.md.

Creating docs/RESEARCH_QUESTIONS.md.

Creating docs/LIMITATIONS.md.

Proposing the exact Milestone 1 implementation.

Then implementing Milestone 1 with tests.

Do not claim that EngageVR has been experimentally validated. The immediate goal is to produce a credible, modular, reproducible, demonstrable software prototype that can later be evaluated using laboratory VR equipment, reference physiological sensors, controlled experiments, and approved participant studies.
