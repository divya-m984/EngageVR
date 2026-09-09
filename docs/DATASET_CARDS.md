# EngageVR Dataset Cards

> **Research documentation**
>
> This document records research-facing dataset cards for datasets currently
> represented by EngageVR and for the proposed future participant dataset.
>
> A dataset card documents provenance, intended use, limitations, and scientific
> status. It does not grant permission to use, redistribute, or publish a
> dataset.
>
> No participant-labelled EngageVR dataset currently exists.

## 1. Purpose

The technical acquisition and adapter documentation for public datasets remains
in `docs/DATASETS.md`.

This document has a different purpose.

It summarizes each dataset from a research-governance perspective, including:

- origin;
- modalities;
- intended use;
- prohibited or unsupported use;
- labels;
- population information where known;
- scientific eligibility;
- licensing/access status;
- known limitations;
- relationship to EngageVR research questions.

The cards must preserve uncertainty.

Unknown dataset properties must remain unknown rather than being filled from
assumption.

## 2. Dataset Inventory

| Dataset | Source type | Present locally | Current EngageVR use | Scientific status |
|---|---|---:|---|---|
| UBFC-rPPG | Public research dataset | No | Adapter and fixture-tested integration | Real dataset, evaluation pending |
| EngageVR synthetic feature dataset | Synthetic/generated | Generated locally when requested | Software and pipeline verification | Permanently ineligible as human evidence |
| Future EngageVR participant dataset | Future human-subject dataset | No | Not yet collected | Does not currently exist |

## 3. Card 1 — UBFC-rPPG

### 3.1 Dataset name

**UBFC-rPPG**

### 3.2 Dataset type

Public research dataset for remote photoplethysmography evaluation.

### 3.3 Current EngageVR status

EngageVR contains an adapter for UBFC-rPPG.

The real dataset is not present in the current project environment.

Adapter discovery, validation, and ground-truth parsing have been tested using
temporary deterministic fixtures.

Those fixtures contain no real UBFC-rPPG participant data and do not produce
scientific metrics.

### 3.4 Official source

The authoritative source currently recorded by EngageVR is:

```text
https://sites.google.com/view/ybenezeth/ubfcrppg
```

The source is hosted by Université Bourgogne Franche-Comté.

Users must obtain the dataset through the official source.

EngageVR does not automatically download the dataset.

### 3.5 Associated citation

The repository records the associated publication as:

> S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois (2017).
> "Unsupervised skin tissue segmentation for remote photoplethysmography."
> Pattern Recognition Letters.

Users of the dataset should verify and cite the authoritative publication
appropriately.

### 3.6 Access

Access is external to EngageVR.

The software:

- does not download UBFC-rPPG;
- does not accept dataset terms on behalf of the user;
- does not redistribute dataset contents.

Dataset acquisition and permitted use remain the user's responsibility.

### 3.7 Licence and permitted use

**Status: REQUIRES MANUAL VERIFICATION**

The repository currently records that no explicit licence or permitted-use
statement was found on the official dataset page when the adapter documentation
was prepared.

Therefore, EngageVR makes no claim that the dataset may be:

- redistributed;
- published;
- used commercially;
- used for any particular research purpose.

Absence of an explicit licence must not be interpreted as permission.

Before scientific use, the user should establish that the intended use is
permitted.

### 3.8 Modalities

The repository records the following as verified:

| Modality | Status |
|---|---|
| Facial video | Verified |
| Reference PPG waveform | Verified |
| Reference heart rate | Verified |

### 3.9 Video characteristics

The repository records:

| Property | Value |
|---|---|
| Camera | Logitech C920 HD Pro |
| Resolution | 640 × 480 |
| Frame rate | 30 fps |
| Format | Uncompressed 8-bit RGB |

These values should remain linked to the authoritative dataset documentation.

### 3.10 Reference measurement

The repository records the reference signal as originating from a:

```text
CMS50E transmissive pulse oximeter
```

and including:

- reference PPG waveform;
- PPG-derived heart rate.

### 3.11 Reference sampling rate

**Status: REQUIRES MANUAL VERIFICATION**

The repository does not currently claim a verified reference sampling rate.

The adapter therefore does not invent one.

A correct sampling rate must be established before any time-aligned comparison
is scientifically interpreted.

### 3.12 Population

The dataset contains recordings from a relatively small participant sample.

The authoritative page, as currently documented by EngageVR, does not provide
enough demographic information for the project to characterize the dataset's
population comprehensively.

Therefore, demographic composition must remain:

```text
not sufficiently documented by the current authoritative source
```

rather than inferred.

### 3.13 Recording conditions

The repository describes UBFC-rPPG as containing:

- a smaller simple set;
- a larger realistic set;
- recordings involving seated indoor acquisition;
- recordings made using one documented camera setup.

The dataset must not be assumed to represent uncontrolled real-world webcam
conditions.

### 3.14 Labels and targets

UBFC-rPPG provides physiological reference information.

It does **not** provide EngageVR engagement labels.

It does **not** provide EngageVR cognitive-load labels.

Therefore, UBFC-rPPG cannot directly evaluate:

- engagement classification;
- cognitive-load estimation;
- personalization benefit for those targets;
- adaptation effectiveness.

### 3.15 Intended EngageVR use

The appropriate intended use is component-level evaluation of the rPPG
pipeline.

Potential future analyses may include, where scientifically justified:

- heart-rate estimation error;
- measurement availability;
- signal-quality behaviour;
- comparison among GREEN, CHROM, and POS;
- agreement with the provided physiological reference.

### 3.16 Unsupported use

UBFC-rPPG must not be presented as evidence that:

- EngageVR accurately detects engagement;
- EngageVR accurately detects cognitive load;
- EngageVR personalization works;
- EngageVR adaptation benefits participants;
- EngageVR works across all demographic groups;
- webcam rPPG is medically validated.

### 3.17 Known limitations

Current documented limitations include:

- relatively small participant sample;
- limited demographic documentation;
- constrained acquisition conditions;
- one documented camera configuration;
- physiological reference based on peripheral PPG rather than ECG;
- no engagement labels;
- no cognitive-load labels;
- reference sampling rate requiring manual verification.

### 3.18 Dataset merging

UBFC-rPPG should not be pooled casually with unrelated datasets.

Differences may include:

- population;
- acquisition hardware;
- reference devices;
- recording conditions;
- labels;
- sampling procedures.

A pooled metric across incompatible datasets may have no clear scientific
interpretation.

### 3.19 Current scientific evaluation

**Status: PENDING**

No UBFC-rPPG accuracy, MAE, RMSE, bias, coverage, or agreement result currently
exists in EngageVR from the real dataset.

No scientific metric should be added until the real data and real reference
signals have actually been evaluated.

---

## 4. Card 2 — EngageVR Synthetic Feature Dataset

### 4.1 Dataset name

**EngageVR Synthetic Feature Dataset**

### 4.2 Dataset type

Deterministically generated synthetic modelling dataset.

### 4.3 Origin

Generated by EngageVR software from a documented seeded synthetic process.

It contains no observations from real people.

### 4.4 Purpose

The dataset exists to verify software behaviour, including:

- feature-dataset generation;
- schema validation;
- model-training pipelines;
- participant/session grouping logic;
- multimodal fusion;
- personalization plumbing;
- uncertainty estimation;
- abstention;
- adaptation-controller integration;
- dashboard behaviour;
- MLOps/reproducibility infrastructure.

### 4.5 Participant identifiers

Synthetic subject identifiers use explicit synthetic naming.

Conceptually:

```text
synthetic-subject-0001
synthetic-subject-0002
...
```

They must not be described as participants.

### 4.6 Provenance labels

Synthetic records must remain explicitly labelled as synthetic.

The current repository uses fields and labels such as:

```text
data_source: synthetic
synthetic_label: SYNTHETIC
```

and forbids scientific evaluation through dataset/target provenance.

### 4.7 Scientific eligibility

**Scientific human-subject evaluation: PROHIBITED**

The dataset may verify that software behaves correctly.

It must not be used to establish:

- engagement-estimation validity;
- cognitive-load-estimation validity;
- rPPG validity in humans;
- personalization benefit;
- adaptation benefit;
- participant comfort;
- participant safety.

### 4.8 Labels

Synthetic targets are generated by the synthetic data process.

They are not participant annotations.

They are not empirical psychological ground truth.

A high predictive score against a synthetic target means only that the
software/model recovered structure present in the synthetic generator under the
tested setup.

### 4.9 Storage

Generated synthetic feature datasets are stored under project artifact
locations that are excluded from ordinary Git tracking.

The repository source does not commit generated modelling datasets.

### 4.10 Reproducibility

The synthetic generator supports deterministic software testing under defined
configuration and seed conditions.

Reproducibility of a synthetic dataset does not make the dataset scientifically
valid for humans.

### 4.11 Intended use

Appropriate uses include:

- unit tests;
- integration tests;
- deterministic demonstrations;
- pipeline smoke tests;
- model-interface tests;
- reproducibility tests;
- dashboard fixtures;
- schema and provenance verification.

### 4.12 Unsupported use

The synthetic feature dataset must not support claims such as:

- "EngageVR achieves X% accuracy on humans";
- "personalization improves real users";
- "adaptive VR improves engagement";
- "the model is scientifically validated";
- "the model is clinically validated."

### 4.13 Current scientific status

```text
SOFTWARE SELF-CHECK ONLY
```

All scientific interpretations remain prohibited.

---

## 5. Card 3 — Future EngageVR Participant Dataset Template

> **THIS DATASET DOES NOT CURRENTLY EXIST**
>
> This section is a template for the dataset card that must be completed if an
> institutionally reviewed EngageVR participant study is conducted in the
> future.
>
> It does not describe an existing dataset.

### 5.1 Proposed dataset name

[TO BE FINALIZED]

A future name should identify the study without exposing participant identity.

### 5.2 Dataset type

Proposed future pseudonymous human-subject multimodal research dataset.

### 5.3 Current status

```text
NOT COLLECTED
```

No participant-labelled EngageVR dataset currently exists.

### 5.4 Institutional status

[TO BE FINALIZED]

The future card must record:

- institution;
- protocol/study identifier where applicable;
- approval/review status using institutionally accurate wording;
- approved study version.

This template must never be pre-filled with an approval claim.

### 5.5 Participant population

[TO BE FINALIZED]

Future documentation should include, where scientifically and ethically
appropriate:

- recruitment population;
- age eligibility;
- inclusion criteria;
- exclusion criteria;
- number enrolled;
- number completing each required component.

No participant count is asserted by this template.

### 5.6 Consent basis

[TO BE FINALIZED]

The final card should identify the approved consent/data-use basis relevant to
the dataset.

### 5.7 Participant identification

Research records should use pseudonymous identifiers.

Direct identifying administrative information should remain outside the
modelling dataset.

### 5.8 Proposed modalities

Depending on the approved study, candidate modalities may include:

- task telemetry;
- webcam-derived behavioural features;
- head-pose/movement features;
- webcam-rPPG;
- subjective participant responses;
- model predictions;
- uncertainty/abstention;
- adaptation records;
- optional approved reference physiological measurements.

Only modalities actually collected should appear in the final dataset card.

### 5.9 Raw webcam data

Default proposed state:

```text
raw video retained: no
```

If the approved study changes this, the final dataset card must document:

- what raw media were retained;
- why;
- consent basis;
- access;
- retention;
- deletion;
- sharing limitations.

### 5.10 Targets and labels

The final card must specify every target's source.

Potential sources may include:

- task-derived outcomes;
- participant self-report;
- researcher-defined experimental condition;
- externally validated measurement where appropriate.

Model predictions must not be relabelled as ground truth.

### 5.11 Experimental conditions

If the static-versus-adaptive study is conducted, the final card should record:

- static condition definition;
- adaptive condition definition;
- condition-order procedure;
- starting difficulty;
- adaptation configuration;
- relevant protocol version.

### 5.12 Dataset structure

[TO BE FINALIZED]

The final card should document:

- participant grouping;
- session grouping;
- trial/window structure;
- timestamps;
- modality relationships;
- missingness representation.

### 5.13 Missing data

The dataset must distinguish, where known:

- not collected;
- technically unavailable;
- quality-gated unavailable;
- model abstention;
- participant non-response;
- participant withdrawal;
- protocol exclusion.

Missing values must not silently become zero.

### 5.14 Signal quality

The final card should document:

- quality fields;
- quality thresholds;
- unavailable rules;
- sensor-specific limitations.

Poor signal quality must not be interpreted as participant state.

### 5.15 Privacy

The final card should document:

- pseudonymization;
- storage location;
- authorized access;
- retention;
- deletion;
- backup;
- transfer;
- public-release status.

Pseudonymized must not be equated with anonymous.

### 5.16 Sharing

[TO BE FINALIZED]

The future card must state clearly whether data are:

- private to the research team;
- controlled access;
- shareable under defined conditions;
- publicly released.

Repository publication of software does not imply publication of participant
data.

### 5.17 Licence / data-use terms

[TO BE FINALIZED]

Participant datasets require data-use conditions appropriate to the consent and
institutional procedure.

### 5.18 Known limitations

The final card should report limitations such as:

- sample size;
- recruitment population;
- demographic representation where appropriately documented;
- hardware;
- environment;
- missing modalities;
- quality failure;
- protocol deviations;
- subjective-label limitations;
- model-target limitations.

Limitations should remain visible even when results are favourable.

### 5.19 Intended use

The final card should state the exact research questions the dataset is suitable
for.

Dataset existence alone does not make it suitable for every EngageVR research
question.

### 5.20 Prohibited / unsupported use

Unless separately justified and approved, the future EngageVR participant
dataset should not be used for:

- medical diagnosis;
- psychological diagnosis;
- identity recognition;
- unrelated sensitive inference;
- high-stakes automated decisions;
- claims beyond the participant population and protocol studied.

### 5.21 Dataset-card completion checklist

Before the future dataset is used for scientific reporting:

- [ ] source/protocol identified;
- [ ] participant population documented;
- [ ] participant count documented;
- [ ] consent/data-use basis documented;
- [ ] modalities documented;
- [ ] targets and their provenance documented;
- [ ] experimental conditions documented;
- [ ] missingness semantics documented;
- [ ] signal-quality rules documented;
- [ ] storage/access documented;
- [ ] retention/deletion documented;
- [ ] sharing status documented;
- [ ] limitations documented;
- [ ] scientific-eligibility status documented.

## 6. Cross-Dataset Rules

### 6.1 Preserve provenance

Every result must remain traceable to the dataset that produced it.

### 6.2 Do not pool incompatible datasets casually

Datasets should not be combined solely to increase row count.

Population, acquisition, label, modality, and protocol compatibility must be
considered first.

### 6.3 Do not invent missing labels

A dataset without engagement labels does not become an engagement dataset
because EngageVR can compute an engagement prediction.

### 6.4 Synthetic and real data remain distinct

Synthetic data must never be relabelled or promoted into participant evidence.

### 6.5 Public and participant data remain distinct

Public-dataset validation and future EngageVR participant evaluation answer
different questions and must be reported separately.

## 7. Dataset-to-Research-Question Mapping

| Dataset | RQ1 | RQ2 | RQ3 | RQ4 | RQ5 |
|---|---:|---:|---:|---:|---:|
| UBFC-rPPG | No engagement target | No | Limited component-level only | No | Yes, rPPG component |
| EngageVR synthetic features | Software check only | Software check only | Software check only | Software check only | Software check only |
| Future EngageVR participant dataset | Potentially | Potentially | Potentially | Yes, if designed accordingly | Potentially, if reference protocol supports it |

`Potentially` means only that the future dataset could support the question if
the approved study actually collects the necessary valid measurements and
targets.

It is not a claim that the future dataset will answer every research question.

## 8. Current Dataset Status

The current repository state is:

- UBFC-rPPG adapter exists;
- real UBFC-rPPG evaluation remains pending;
- synthetic feature datasets support software verification;
- no public dataset currently supplies engagement/cognitive-load labels to the
  modelling pipeline;
- no participant-labelled EngageVR dataset exists;
- no EngageVR model has been scientifically evaluated against real
  participant-labelled engagement or cognitive-load data.

These boundaries must remain visible in future research reporting.
