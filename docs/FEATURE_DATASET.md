# Windowed Feature Dataset (Milestone 5)

## Status

**No validated EngageVR participant dataset exists.** Everything described
here is the *software* that would build such a dataset. The only datasets
this repository can currently produce are deterministic SYNTHETIC ones,
generated for pipeline verification and permanently excluded from
scientific evaluation.

## What a window is

A feature window is one row of the modelling dataset: a fixed-duration
slice of one session, summarised into named scalar features.

| Parameter | Config key | Default |
|-----------|-----------|---------|
| Duration | `features.window_duration_seconds` | 10.0 s |
| Step | `features.window_step_seconds` | 10.0 s |
| Partial-window policy | `features.partial_window_policy` | `drop` |
| Minimum partial duration | `features.minimum_partial_duration_seconds` | 0.0 s |

Boundaries are **half-open**, `[start, end)`. A sample exactly on a
boundary belongs to the later window and to that window only, so nothing
is counted twice and nothing falls through the gap.

Window `k` starts at `session_start + k * step`, computed from the session
start every time rather than accumulated. A long session therefore does
not drift, and two builds of the same session produce byte-identical
boundaries.

`step < duration` produces **overlapping** windows, which is recorded on
every row as `windows_overlap`. Overlap has a consequence for splitting:
adjacent windows share evidence, so they must never land on opposite sides
of a fold boundary. That is guaranteed by grouping (see
`docs/MODEL_EVALUATION.md`).

### Partial windows

The trailing fragment of a session is **dropped** by default. A shorter
window is summarised from less evidence than every other row, and mixing
the two silently changes what a feature means from row to row. The
`keep_if_minimum` policy retains it when it is at least
`minimum_partial_duration_seconds` long; the row is then flagged
`is_partial` during construction.

### Containment

Every window is verified to lie inside its source session. A window
extending past the recorded end of a session would summarise a period for
which no evidence exists, and `assert_windows_within_session` rejects it.

## Row schema

Schema version: **1.0** (`FEATURE_SCHEMA_VERSION`).
Feature catalog version: **1.0** (`FEATURE_CATALOG_VERSION`).

Both appear on every row, so a dataset never has to be interpreted against
a guess about which contract produced it.

### Column conventions

| Prefix | Contents | Nullable |
|--------|----------|----------|
| *(none)* | identity, grouping, timing, schema version | mostly no |
| `feat__` | measured or aggregated feature value | yes |
| `avail__` | per-feature availability flag | no |
| `modality_available__` | whether a modality contributed at all | no |
| `modality_quality__` | modality signal quality in [0, 1] | yes |
| `target__` | target value | yes |
| `target_meta__<target>__<field>` | target provenance | yes |

Feature values, availability, signal quality, targets, and provenance live
in **separate columns**. They are never merged, because `0.0` and "we
could not measure it" are different observations and collapsing them
destroys the difference.

### Identity and grouping columns

`window_id`, `session_id`, `subject_id`, `subject_kind`,
`experiment_condition`, `data_source`, `synthetic_label`.

`subject_id` is pseudonymous. Synthetic subjects are named
`synthetic-subject-0001`, never "participant 1".

### Timing columns

`window_index`, `window_start_utc`, `window_end_utc`,
`window_start_monotonic_seconds`, `window_end_monotonic_seconds`,
`window_duration_seconds`, `window_step_seconds`, `windows_overlap`.

The wall clock and the monotonic clock are kept distinct, following the
Milestone 4 invariant (DEC-029). Monotonic bounds are supplied together or
not at all.

### Target columns

Per target: the value, plus `task_type`, `source_type`,
`source_instrument`, `observed_at_utc`, `interval_start_utc`,
`interval_end_utc`, `synthetic_label`, `provenance_notes`, and
`scientific_evaluation_permitted`.

Provenance travels **on the row**, so it cannot be lost by moving values
into a table.

## Missing data

- A missing measurement is `null`. It is never zero-filled at this layer.
- Imputation is a modelling decision made **inside a training fold**; see
  `docs/BASELINE_MODELS.md`.
- Poor signal quality is recorded as quality, never as a feature value and
  never as a low target value.
- Non-finite values are rejected outright: an undefined result is reported
  as unavailable, not as `inf` or `NaN`.

Three `missing_behaviour` categories are declared per feature:

| Category | Meaning |
|----------|---------|
| `null_when_unavailable` | The measurement could not be taken. |
| `null_when_evidence_insufficient` | The minimum-evidence gate was unmet. |
| `zero_when_no_events` | Zero is a genuine observation. |

The last one matters: a blink count of 0 over a window in which a face was
tracked throughout is a real observation; a blink count over a window with
no face at all is not.

## Feature catalog

Every feature has a catalog entry declaring its canonical name,
description, source modality, unit, aggregation formula, minimum evidence,
missing-value behaviour, quality dependency, and whether it is permitted
as a predictor. A feature with no entry cannot enter a dataset, and a
column with no entry cannot reach a model.

The catalog is versioned and **ordered**. Order fixes the dataset column
order, which is part of the fingerprint, so a reordering is a detectable
change rather than a silent one.

### Groups

| Group | Count | Examples |
|-------|-------|----------|
| `behavioural` | 14 | face presence, eye-openness proxy, blink proxy, eye-closure, mouth-opening proxy, tracking stability, missingness |
| `head_pose` | 12 | yaw / pitch / roll mean, SD, range; angular velocity mean and max; motion variability; availability |
| `rppg` | 11 | heart-rate estimate, quality score, ROI availability, valid-frame %, timestamp jitter, spectral peak ratio, peak prominence, illumination stability, motion score, unavailable-window %, method |
| `task` | 16 | attempted / completed trials, correct / incorrect / timeout counts and proportions, RT mean / median / SD / min / max, difficulty, pauses, inactivity |
| `quality` | 8 | brightness, blur, motion, under/over-exposure %, blurry %, dropped-frame %, window missingness |

61 entries in total; 60 are permitted predictors.

### Non-predictor features

`rppg_method` records which extraction algorithm produced a window. It is
kept for provenance and is **not** a permitted predictor: the method
identifier is a property of the pipeline configuration, and a model that
used it would be learning a processing artefact rather than a signal
property. It is stored as a categorical column; categorical features are
never permitted predictors in this milestone.

### Feature formulas and units

Every entry carries its formula and its unit in the catalog itself, which
is snapshotted beside each dataset as `<stem>.feature_catalog.json`. A
representative selection:

| Feature | Unit | Formula | Minimum evidence |
|---------|------|---------|------------------|
| `face_presence_pct` | percent | `100 * count(face_present) / count(frames)` | ≥1 frame |
| `eye_openness_proxy_mean` | dimensionless | `mean(mean_ear)` over frames with a usable EAR | ≥`min_face_frames` frames |
| `blink_proxy_rate_per_min` | events/min | `60 * blink_count / face_present_duration` | face present ≥`min_face_seconds` |
| `head_yaw_sd_deg` | degrees | population SD of `yaw_deg` | ≥2 poses |
| `rppg_heart_rate_bpm` | beats/min | carried through from an **accepted** rPPG window | one accepted window |
| `rppg_unavailable_window_pct` | percent | `100 * rejected / attempted` | ≥1 rPPG window |
| `task_correct_proportion` | proportion | `correct / (correct + incorrect + timeout)` | ≥`min_resolved_trials` |
| `task_inactivity_seconds` | seconds | max gap between consecutive in-window events | ≥2 events |
| `window_missing_feature_pct` | percent | `100 * null / total` catalog features | always |

### Aggregation rules

- **Minimum evidence gates the computation, not the interpretation.** A
  mean over one frame is arithmetically valid and scientifically useless,
  so when the gate is unmet the feature is `null`.
- **Rejected rPPG windows contribute nothing.** A window whose quality
  gate failed does not enter the heart-rate summary, the spectral
  summaries, or any average. It is counted in
  `rppg_unavailable_window_pct` and nowhere else. Diagnostics (quality
  score, jitter, valid-frame %) still describe every attempted window,
  because they are how the rejection is explained.
- **No construct is inferred.** Nothing in the aggregation layer produces
  an engagement, cognition, emotion, fatigue, stress, or attention value.
- **A timeout contributes no reaction time.** No response is a different
  observation from a wrong response and is never folded into an error rate.

Minimum-evidence thresholds live in `features.aggregation` in
`configs/defaults.yaml`. They are software thresholds, not empirically
validated cut-offs, and meeting them does not make a window scientifically
adequate.

## Leakage risks and the checks that catch them

| Risk | Check |
|------|-------|
| Target or a re-encoding of it in the predictor matrix | `assert_no_leakage` refuses any `target__` / `target_meta__` column |
| Identifier or timestamp lets a model recover group membership | `NON_PREDICTOR_COLUMNS` are structurally excluded |
| Evidence from after the predicted window | `POST_WINDOW_TOKENS` name-pattern check (`session_total`, `final_`, `next_`, `adaptation_outcome`, `end_of_session`, …) |
| Session-completion information in an earlier window | `select_in_window` returns nothing at or after `end`; the same token check catches a named summary |
| Adaptation outcome after the prediction window | `adaptation_applied_after`, `adaptation_outcome` tokens |
| Undeclared column reaching a model | Any `feat__` column absent from the catalog is refused |
| Statistics fitted outside the training fold | Structural: preprocessing lives inside the `Pipeline` |

## Storage layout

For a dataset written to `<stem>.parquet`:

```
<stem>.parquet               the table (Snappy-compressed Parquet)
<stem>.metadata.json         provenance, fingerprint, disclaimers
<stem>.feature_catalog.json  the catalog the dataset was built against
```

Metadata and the catalog are plain JSON on purpose: the origin of a
dataset must be inspectable without opening the table, and certainly
without loading a model file.

All three are written **atomically** — to a temporary file in the same
directory, flushed, fsynced, then `os.replace`d — so a reader sees either
the previous document or the complete new one.

The default location is `features.dataset_directory`
(`artifacts/datasets`), which is gitignored. **Generated Parquet datasets
are never committed.**

## Dataset fingerprint

`dataset_fingerprint` is a SHA-256 over a canonical UTF-8 rendering of:

1. the dataset schema version;
2. the feature catalog version;
3. the exact column order;
4. the window duration, step, and overlap flag;
5. every row's values in that column order, with rows sorted by
   `(session_id, window_index, window_id)`.

Floats are rendered with `repr`, which round-trips exactly for IEEE-754
doubles, so a value differing in the last bit produces a different
fingerprint.

**Wall-clock values are excluded.** `created_at_utc` is not part of the
canonical content: two equivalent builds must fingerprint identically, and
a creation timestamp would guarantee they never do.

The fingerprint changes when row content, the schema, the column order,
the feature order, the target set, or the window geometry changes. It does
not change when rows are supplied in a different order, because rows are
sorted canonically first.

## Dataset metadata

`<stem>.metadata.json` records the schema version, catalog version, row
count, feature count, full column order, subject and session counts,
data-source counts, subject-kind counts, experiment-condition counts,
window geometry, per-feature missingness, per-target class distributions
or numeric summaries, the creation configuration, the random seed (for
generated datasets), input-session fingerprints where applicable, the
dataset fingerprint and the algorithm that produced it, the creation
timestamp, scientific-evaluation eligibility, and the disclaimers.

A dataset with no metadata document cannot be read by the training
pipeline: a dataset with no recorded provenance is not usable, because
nothing could establish where its rows came from.

## Target model and provenance

Four targets are supported:

| Target | Task | Domain |
|--------|------|--------|
| `engagement_class` | classification | `low` / `medium` / `high` |
| `engagement_score` | regression | [0, 1] |
| `cognitive_load_class` | classification | `low` / `medium` / `high` |
| `cognitive_load_score` | regression | [0, 1] |

Class vocabularies are **ordered**, and that order is part of the
contract: it fixes the column order of every confusion matrix and
probability array, so a stored result can be re-read without guessing.

Every target observation must state: `target_name`, `task_type`, value,
class vocabulary or numeric range, `source_type`, `source_instrument`,
`observed_at_utc`, the interval it describes, `subject_id`, `session_id`,
`data_source`, a synthetic label where applicable, `provenance_notes`, and
`scientific_evaluation_permitted`.

Permitted source categories: `subjective_self_report`,
`experiment_condition`, `expert_annotation`,
`public_dataset_annotation`, `synthetic_generator`.

**Only `synthetic_generator` is currently populated.** The others are
declared so real labels can be ingested later without redefining the
schema, and so a reader can see at a glance which categories remain empty.

### Label alignment

A target's `interval_start_utc` and `interval_end_utc` name the period the
label describes; `observed_at_utc` names when it was recorded. For
synthetic targets, the interval is exactly the feature window and the
observation instant is the window end.

For a real label this alignment is a research decision, not a data
transformation: a self-report given after a block does not describe the
instant it was given. Nothing here assumes otherwise, which is why the
interval is a required field rather than derived.

### Measurements are not labels

`reject_automatic_derivation` refuses to synthesise a target from any
measurement group, with a stated reason:

- **Task accuracy, error rate, timeout rate, reaction time** — software
  measurements of what the task program observed. A person can respond
  accurately while disengaged and slowly while highly engaged.
- **Difficulty level** — an experimental manipulation, not a measurement
  of the load a particular person experienced. Treating it as cognitive
  load assumes the manipulation worked, which is the hypothesis.
- **Camera-based heart rate** — an unvalidated signal-processing estimate
  that responds to posture, temperature, movement, and illumination.
- **Behavioural proxies and head pose** — geometry. Facing a screen is not
  attending to it.
- **Signal quality** — describes the measurement, not the person.

There is no `allow` flag. Producing a real label requires an external
instrument and a documented protocol, not a keyword argument.

### Synthetic prohibition

A synthetic target must carry `synthetic_label: "SYNTHETIC"` and must set
`scientific_evaluation_permitted: false`. Both are schema-enforced and
cannot be overridden. A scientific-mode run rejects any row whose
predictors or targets are synthetic.

## The synthetic generator

`engagevr.features.synthetic` builds deterministic datasets for software
verification. Its hidden data-generating process is documented in full in
the module docstring and summarised here.

Two latent variables drive each window and are **never emitted as
columns**:

```
e_raw = e_base + subject_effect_e + session_effect_e + drift_e(t) + ar_e(t)
l_raw = l_base + subject_effect_l + session_effect_l + drift_l(t) + ar_l(t)
        - coupling * (e_raw - e_base)
E = clip(e_raw, 0, 1)
L = clip(l_raw, 0, 1)
```

- `subject_effect_*` — one draw per synthetic subject (**group effect**)
- `session_effect_*` — one draw per session (**session effect**)
- `drift_*` — a linear trend across the session
- `ar_*` — a first-order autoregressive process across windows
- `coupling` — makes load mildly anti-correlated with engagement

Each observable feature is drawn as
`base + a*(E - 0.5) + b*(L - 0.5) + Normal(0, sd)`, clipped to its catalog
range. A feature with `a = b = 0` is **irrelevant by construction**;
several are included deliberately (mouth-opening proxies, head roll,
capture brightness, blur, exposure percentages) so that feature selection
and interpretation have something to get wrong.

Structural relationships are then imposed so the table is one an
aggregation step could actually have emitted: the three response
proportions sum to one, counts are consistent with their proportions,
minimum eye-openness never exceeds the mean, and the reaction-time order
statistics are ordered. These produce genuine collinearity.

Targets are noisy observations of the latents:

```
engagement_score = clip(E + Normal(0, label_noise_sd), 0, 1)
engagement_class = threshold(engagement_score, class_thresholds)
```

so a perfect score is not attainable even in principle.

The generator also produces modality dropout, quality-gated rPPG windows
(estimates cleared, diagnostics retained), configurable class imbalance,
and correlated and irrelevant features.

**The generator is not tuned toward any model family.** The relationships
are additive and monotone, which suits a linear model, while the
missingness, the clipping, and the class thresholds introduce
non-linearity, which suits a tree. No choice was made after looking at a
result.

## Privacy

Nothing that could identify a person is representable. There is no column
for a name, an email, a frame, an image, a landmark array, a payload blob,
or a secret, and `assert_no_identity_columns` /
`assert_no_identifier_values` assert that none appears. Participants are
identified by a pseudonymous `subject_id` only.

## Commands

```bash
uv run python -m engagevr features-demo \
  --seed 42 \
  --subjects 30 \
  --sessions-per-subject 2 \
  --windows-per-session 20 \
  --output artifacts/datasets/m5-synthetic.parquet
```

Add `--imbalanced` to shift the class thresholds. Run it twice with the
same seed and the printed `Dataset fingerprint` is identical.

## Limitations

- No validated participant dataset exists, so no dataset in this
  repository supports a scientific conclusion.
- The aggregation layer has never been run on a real webcam session: the
  behavioural, head-pose, rPPG, and quality aggregators are exercised
  against typed fixtures and against the synthetic generator only.
- Minimum-evidence thresholds are engineering defaults, not validated
  cut-offs.
- Window duration and step have not been optimised against anything;
  10 s is a default, not a finding.
