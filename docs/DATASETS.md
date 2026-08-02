# EngageVR -- Public Datasets

This document records every public dataset the project has an adapter
for.  **No dataset is downloaded by this software.**  Obtaining the data
through the official channel, and accepting whatever terms the provider
states, is the user's responsibility.

Dataset files are never committed to this repository.  `data/` is
gitignored except for its `.gitkeep` placeholders.

## Verification status

| Field | Meaning |
|-------|---------|
| **Verified** | Read directly from the official source listed below. |
| **Community convention** | Consistent across widely-used community implementations, but not stated on the official page. Confirm against your own copy. |
| **REQUIRES MANUAL VERIFICATION** | Not stated by the source. Do not assume it. |

---

## UBFC-rPPG

### Official source

**Verified.** https://sites.google.com/view/ybenezeth/ubfcrppg

Hosted by Université Bourgogne Franche-Comté (UBFC).  Access is via a
download link published on that page.  Contacts listed on the page:
Yannick Benezeth, Richard Macwan, Serge Bobbia.

### Citation

**Verified.**

> S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois (2017).
> "Unsupervised skin tissue segmentation for remote photoplethysmography."
> *Pattern Recognition Letters*.

Cite this paper in any work that uses the dataset.

### Access procedure

1. Visit the official page above.
2. Follow the download link published there.
3. Read whatever terms the page or the download states, and satisfy
   yourself that your intended use is permitted (see **Licence** below).
4. Extract the archive to a local path.
5. Point EngageVR at it, either with `--root` or by setting
   `rppg.datasets.ubfc_rppg_root` in `configs/defaults.yaml`.

EngageVR will not perform steps 1-4 for you.  The adapter contains no
network code, and a test asserts that it never will.

### Licence

**REQUIRES MANUAL VERIFICATION.**

At the time this adapter was written, no explicit licence or
permitted-use statement was found on the official page.  This project
therefore makes **no claim** about what the dataset may be used for.

Before using UBFC-rPPG for anything -- including internal evaluation,
publication, or redistribution -- contact the dataset authors and obtain
an explicit statement of permitted use.  Absence of a stated licence is
not permission.

### Modalities

| Modality | Status | Detail |
|----------|--------|--------|
| Facial video | **Verified** | Uncompressed 8-bit RGB |
| Reference PPG waveform | **Verified** | From a contact pulse oximeter |
| Reference heart rate | **Verified** | Provided alongside the waveform |

### Video format

**Verified** from the official page:

| Property | Value |
|----------|-------|
| Camera | Logitech C920 HD Pro |
| Resolution | 640 x 480 |
| Frame rate | 30 fps |
| Format | Uncompressed 8-bit RGB |

### Reference physiological signal

**Verified:** ground truth was recorded with a **CMS50E transmissive
pulse oximeter**, and comprises both the PPG waveform and PPG-derived
heart rate.

**REQUIRES MANUAL VERIFICATION:** the reference sampling rate.  The
official page does not state it unambiguously, and community sources
disagree.  The adapter therefore leaves `sampling_rate_hz` as `None`
rather than guessing.  Establish the true rate for your copy of the data
before performing any time-aligned comparison — a wrong assumed rate
silently corrupts every error metric.

### Organisation

**Verified** (official page): the dataset is published in two parts —
a smaller "simple" set and a larger "realistic" set, the latter recorded
with subjects performing a task.  Not all recordings are distributed.

**Community convention** for the on-disk layout, consistent with the
loader in the widely-used `rPPG-Toolbox` reference implementation:

```
<root>/
  subject1/
    vid.avi
    ground_truth.txt
  subject2/
    vid.avi
    ground_truth.txt
  ...
```

`ground_truth.txt` is whitespace-separated numeric text.

| Row | Content | Status |
|-----|---------|--------|
| 0 | Reference PPG waveform | **Verified** against the reference loader |
| 1 | Heart rate | **Community convention** |
| 2 | Timestamps | **Community convention** |

The adapter requires row 0 and reads rows 1 and 2 *opportunistically*:
their absence is normal, not an error, and a row whose length does not
match the waveform is discarded rather than silently truncated to fit.
Confirm the row semantics against your own copy before relying on rows
1 or 2.

### Known limitations

- **Small.** Tens of subjects, not hundreds. Results do not generalise.
- **Demographically narrow.** The dataset's demographic composition is
  not documented on the official page. rPPG performance is known to vary
  with skin tone, and this dataset cannot be used to characterise that
  variation.
- **Constrained recording conditions.** Indoor, seated, one camera, one
  lighting setup. Performance here says nothing about performance under
  the uncontrolled conditions this project actually targets.
- **Reference device is itself a proxy.** A transmissive pulse oximeter
  measures peripheral pulse, not cardiac electrical activity. Agreement
  with it is not agreement with ECG.
- **No engagement or cognitive-load labels.** UBFC-rPPG can validate the
  *pulse-rate* component only. It says nothing about the engagement
  estimation this project is ultimately about.

### Do not merge

UBFC-rPPG must not be pooled with any other dataset.  Recording
conditions, reference devices, and populations differ; a metric computed
over a pooled set describes nothing in particular.  The adapter
interface enforces one adapter per dataset.

---

## Current evaluation status

**Public-dataset evaluation is PENDING.**

UBFC-rPPG is **not present** in this environment.  The adapter's
discovery, validation, and ground-truth parsing are tested against
temporary deterministic fixtures that contain no real dataset content.
Those fixtures are structural stand-ins only and are never used to
produce a metric.

No accuracy, MAE, RMSE, bias, or coverage figure against UBFC-rPPG
exists in this repository, and none will be reported until the pipeline
has actually been run against the real data with its real reference
signals.
</content>
