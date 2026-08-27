# Research Dashboard (Milestone 9)

**Status:** Implementation complete (2026-08-25 live and replay modes;
2026-08-27 automatic live refresh and provenance labelling).
Scientific evaluation and human-subject validation remain pending.

> **SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION.** Every run this
> dashboard can currently display was computed from SYNTHETIC data generated
> by this repository. EngageVR produces software estimates and controller
> diagnostics. Dashboard visualizations do not establish engagement,
> cognitive load, psychological state, health status, safety, or adaptation
> benefit.

---

## 1. What this is

**READ-ONLY RESEARCH OBSERVABILITY.**

The dashboard makes the evidence, provenance, limitations, quality,
uncertainty, selective-prediction behaviour, and adaptation-controller
behaviour that previous milestones already recorded *inspectable*, without
changing any of their semantics.

It displays what a run recorded. It computes no new scientific quantity.

## 2. What this is not

It is not a training pipeline, an online inference server, a live webcam
application, a participant-monitoring product, a clinical or psychological
dashboard, an autonomous adaptation console, a model-selection or champion
system, or an MLOps platform.

### Real-time mode means observability, not inference

`docs/PROJECT_PLAN.md` §"Milestone 9" requires "real-time and replay modes".
Both are delivered, and both are **read-only presentation modes** over
recordings the Milestone 4 session recorder already persisted (DEC-090,
which supersedes the part of DEC-083 that placed them outside the
milestone).

*Real-time* here means: re-read an existing session recording **on a
conservative automatic interval** and present records that were persisted
before this view saw them (§5a, DEC-094). It does **not** mean
webcam → feature extraction → model → engagement estimate. No model is
loaded, no camera is opened, no inference is run, and no estimate is
produced anywhere in the dashboard package. A session recording structurally
cannot carry an engagement or cognitive-load value, so there is none to
show; the pages say so rather than leaving a blank.

*Replay* here means: navigate the records already on disk, by hand. It does
**not** re-run the simulator, re-emit messages over a transport, regenerate
anything, repair a damaged recording, or advance on its own.

The distinction that matters: **real-time observation is not real-time
inference.** The live page re-reads a file on a timer. What it shows is
what another subsystem already wrote.

## 2a. The three evidence modes

The sidebar's first control chooses the mode. They are never merged into one
ambiguous state, and each states its own evidence source.

| Mode | Evidence | Catalogue | Pages |
|---|---|---|---|
| **Experiment artifacts** (default) | Milestone 5-8 run directories under `dashboard.artifact_root` | `catalogue.py` | the ten pages of §10 |
| **Live session** | a session recording under `dashboard.session_root`, re-read every `live_refresh_seconds` | `session_catalogue.py` | Live session observation |
| **Session replay** | a recording already complete or interrupted, navigated by hand | `session_catalogue.py` | Session replay |

Only the live mode refreshes on a timer. Replay never advances on its own,
and the artifact observatory never polls.

The artifact observatory remains the primary research view and is unchanged
by the other two. A page checks the mode it was given rather than trusting
its own heading: `live_session_page` refuses a replay context and
`replay_page` refuses a live one, so neither can be rendered under the
other's name by a routing mistake.

A recorded session is **not** an experiment run. The two catalogues are
separate types, scanned from separate roots, and a recording never appears
in the run selector.

## 3. Framework and dependencies

| | |
|---|---|
| Framework | **Streamlit 1.62** |
| Plotting | Streamlit-native charts (`st.line_chart`, `st.bar_chart`, `st.scatter_chart`, `st.dataframe`) |
| Dependency added | `streamlit>=1.62,<2` — one, and only one |

Streamlit was already specified by `docs/PROJECT_PLAN.md:194` and
`docs/PROJECT_SPECIFICATION.md:703`, so the choice was made before this
milestone.

**No Plotly, Altair, matplotlib, or Bokeh is imported by this repository.**
Streamlit ships Altair and pydeck as its own transitive dependencies; no
EngageVR module imports either, and there is no second visualization
library to keep consistent with the first. No React, Next.js, Vue, Angular,
Node.js, or separate REST API was added.

## 4. Launch

```bash
# Through the project CLI (the documented path)
uv run python -m engagevr dashboard

# Against a different artifact root
uv run python -m engagevr dashboard --artifact-root artifacts/experiments

# Print the command without starting a server
uv run python -m engagevr dashboard --print-command

# Directly, for development
uv run streamlit run src/engagevr/dashboard/app.py
```

The dashboard binds to `127.0.0.1:8501` and does **not** open a browser
automatically. It has no authentication, no authorisation, and no audit log,
because it is a local research tool; binding it to a routable interface
would publish a browser for the artifact root to anyone who can reach the
port.

The artifact root reaches the Streamlit process through the environment
variable `ENGAGEVR_DASHBOARD_ARTIFACT_ROOT`, because Streamlit owns
`sys.argv` of the script it runs.

### Server-free check

```bash
uv run python -m engagevr dashboard-check
uv run python -m engagevr dashboard-check --artifact-root artifacts/experiments --json
```

`dashboard-check` discovers every run, classifies it, verifies checksums,
and prints a table (or JSON). It starts no server, so it is the command the
test suite exercises and the one to use in a terminal.

```bash
# List every recorded session
uv run python -m engagevr dashboard-sessions

# Build one session's report and print it (JSON or Markdown)
uv run python -m engagevr dashboard-sessions --session demo-session
uv run python -m engagevr dashboard-sessions --session demo-session --format markdown
```

`dashboard-sessions` is the server-free half of the live and replay modes.
The report is **printed, not saved**: writing it would be the one file
operation this milestone does not have.

## 5. Architecture

```
artifacts/experiments/
        |
        v
  catalogue.py        discovery, family detection, status, integrity
        |
        v
  loaders.py          typed read-only access (JSON + Parquet only)
        |
        v
  views_*.py          per-family view models
        |
        v
  schemas/dashboard.py   typed presentation models (extra="forbid")
        |
        v
  components.py -> pages.py -> app.py    (Streamlit)
```

The live and replay modes follow the same shape from a different root:

```
artifacts/sessions/<session-id>/{manifest,events.jsonl,summary,dropped}
        |
        v
  session_reader.py      tail-safe read-only JSONL parsing
        |
        v
  session_catalogue.py   session discovery, status, provenance
        |
        v
  views_session.py  /  session_report.py     view models, pure report
        |
        v
  components.py -> session_pages.py -> app.py   (Streamlit)
```

| Module | Streamlit? | Responsibility |
|---|---|---|
| `schemas/dashboard.py` | no | Typed view models with structural invariants |
| `dashboard/catalogue.py` | no | Run discovery, family detection, status, checksums |
| `dashboard/loaders.py` | no | Read-only JSON and Parquet access, column selection |
| `dashboard/formatting.py` | no | All display formatting; `None` never becomes `0` |
| `dashboard/aggregation.py` | no | Display-only aggregates (bins, counts, residuals) |
| `dashboard/presentation.py` | no | Terminology and limitations as typed data |
| `dashboard/views_dataset.py` | no | Dataset provenance and measurement quality |
| `dashboard/views_models.py` | no | Classification and regression results |
| `dashboard/views_fusion.py` | no | Fusion and personalization |
| `dashboard/views_uncertainty.py` | no | Selective prediction |
| `dashboard/views_adaptation.py` | no | Controller behaviour |
| `schemas/dashboard_session.py` | no | Typed session view models, replay cursor, report |
| `dashboard/session_reader.py` | no | Tail-safe read-only recording parsing |
| `dashboard/session_catalogue.py` | no | Session discovery, status, provenance |
| `dashboard/views_session.py` | no | Live and replay view models |
| `dashboard/session_report.py` | no | The pure, deterministic session report |
| `dashboard/components.py` | yes | Provenance banners, tables, charts, metric cards |
| `dashboard/pages.py` | yes | The ten artifact pages |
| `dashboard/session_pages.py` | yes | The live and replay pages |
| `dashboard/app.py` | yes | Entry point, mode selector, selectors, caching |
| `dashboard/launch.py` | no | `streamlit run` argv construction |
| `cli_milestone9.py` | no | `dashboard`, `dashboard-check`, `dashboard-sessions` |

The layering is enforced by a test: nothing below `components.py` may import
Streamlit, which is why the unit tests need no browser, no socket, and no
server.

**The dashboard does not call training or evaluation functions as part of
rendering.** There is no path from a click to a retrained model, a re-run
fusion, a recalculated uncertainty, or a new adaptation decision. If a run
artifact is absent, the view says *unavailable*; no pipeline is silently
re-run to fill the gap.

## 5a. Session catalogue, live observation, and replay

### Why not `SessionStore`

`engagevr.storage.session_store.SessionStore` has exactly the right layout
knowledge and exactly the wrong failure behaviour for this job.
`iter_messages` raises on the first bad line, which would blank a live view
because a writer was mid-flush; `recover` is tolerant but rebuilds a whole
summary and is not incremental. Neither distinguishes *a torn final line
while the recorder is running* from *a malformed line in the middle of a
finished recording*, and on a live page those need opposite reactions.

So `session_reader.py` reuses the store's **layout constants and its
identifier validation** — the parts that are pure — and does its own
parsing. It imports no writer, constructs no recorder, and opens no file for
anything but reading (DEC-092).

### Tail safety

The recording is snapshotted whole on each pass. A line is *complete* only
when a newline terminated it. Five states are distinguished, and they are
**not** the artifact catalogue's `corrupt`:

| State | Meaning | Consequence |
|---|---|---|
| complete, decodes | a sound record | presented |
| complete, will not decode | malformed interior record | **stays visible** with its 1-based line number, the problem code, and the reason |
| incomplete trailing line | no terminating newline | reported as **transient**; not counted, not parsed, not called corruption |
| interrupted session | no `summary.json` | *active or incomplete*, fully inspectable |
| stream unavailable / source removed | no `events.jsonl`, or the directory is gone | its own status, with a stated reason |

A final complete line that happens to lack its newline is indistinguishable
from a torn one, so it is reported as partial. That is the honest reading,
and it resolves itself on the next pass.

Decoding uses `engagevr.protocol.validation.decode_stored_message` — the
same validator the session store uses — so a record this dashboard presents
as sound is sound by the project's own definition, not a looser local one.

### Ordering

Records are presented in file order, which is recorded arrival order. They
are **never** re-sorted by sequence number. Two independent sources of
ordering information are shown, side by side and labelled apart:

- **recorded by the receiver** — `ingestion.anomalies`, what the receiver
  observed at the time;
- **derived from recorded sequence numbers** — duplicates, reversals, and
  gaps visible when the stored numbers are read back.

Neither is repaired. No message is invented for a gap, nothing is
renumbered, and a reversal that actually happened stays a reversal.

### Session status

| Status | When |
|---|---|
| `completed` | `summary.json` records `completed=true` |
| `interrupted` | `summary.json` records `completed=false` |
| `active_or_incomplete` | no `summary.json` — still running, or stopped; both legitimate, neither a failure |
| `failed` | **only** when the summary's `disconnect_reason` is `internal_failure` or `invalid_protocol` |
| `unreadable` | the manifest is absent or unparseable |
| `stream_unavailable` | no event stream |

### Refresh policy

The live page refreshes **automatically**, and it is the only page that
does (DEC-094, revised). The page body is a Streamlit fragment declared
`st.fragment(run_every=dashboard.live_refresh_seconds)` — a native
mechanism of the installed Streamlit 1.62, so no dependency was added for
it. Each firing re-reads the recording with the read-only session reader
and redraws what it finds. The header states:

```
Mode: LIVE OBSERVATION
Automatic refresh: every 5 seconds
```

The manual *Read new records* button remains, for a reader who wants the
next pass now; both go through the same body. Every pass reports how many
complete records arrived since the previous one.

**Replay does not auto-advance** and the **experiment-artifact mode does
not poll.** The fragment is constructed inside `live_session_page` and
nowhere else, and a test asserts `run_every` appears exactly once in the
whole package.

The interval is validated by `views_session.live_refresh_interval` before
it reaches Streamlit: it must be a finite number, greater than zero, and
at least `MINIMUM_LIVE_REFRESH_SECONDS` (2s). A refused value switches the
timer off with the reason stated; it is **not clamped** to the minimum,
because a page refreshing at a cadence nobody chose is exactly the
unattributed behaviour this dashboard must not have. The page still
renders, and its manual control still works.

Nothing is interpolated or fabricated between two passes, and a recording
that shrank between passes is reported as an error, because an append-only
file cannot shrink on its own.

#### Real-time observation is not real-time inference

What refreshes is a **view of records another subsystem already wrote**.
No model is loaded, no camera is opened, no estimate is produced, nothing
is sent, and nothing is written, on any cadence. A session recording
structurally cannot carry an engagement or cognitive-load value, so a
faster refresh could not produce one.

The session catalogue and every session read are deliberately **not**
cached. The experiment catalogue is cached on directory modification times
because a run is written once; an append to a recording leaves its
directory's modification time untouched, so the same key would go stale in
exactly the mode that must not — and a cached read would give a live view
that cannot show an appended record.

### Replay navigation

`DashboardReplayState` is a frozen cursor. `first`, `last`, `step_forward`,
and `step_backward` return new instances clamped to the recorded range, so
stepping past either end is not an error a page must handle — it is not
representable. The buttons move the position through a callback, so the
controls' enabled state never lags one interaction behind what the page
shows.

Replay preserves synthetic status, data source, session id, sequence
information, timestamps, anomalies, and completion state exactly as
recorded. A record that already carried replay metadata when it was written
says so, and names the session it originally came from.

### Provenance of a session

The Milestone 4 session format declares **no** scientific eligibility, so
every session is presented as ineligible with that stated as the reason.
`data_source = live` does not change it: *live* says where the bytes came
from, not that a study was designed, labelled, approved, or validated.

The synthetic banner follows the record composition — how many records carry
the permanent `SYNTHETIC` label — not the mode. With no record read yet, the
page says provenance is not established, which is different from saying the
records are not synthetic.

### Session report

`session_report.build_report(read, mode=...)` is a **pure function** of an
already-completed read. It opens nothing, writes nothing, and derives
nothing that is not already in the recording. A *partial* read is refused
rather than reported: counts taken from half a file read exactly like counts
taken from all of it.

The report's identity is its content. `report_fingerprint` is a SHA-256 over
the canonical content with two fields excluded — the fingerprint itself and
`exported_at_utc`. Wall-clock time therefore takes no part in identity, and
the same recording reported twice a week apart yields byte-identical JSON.

Contents: schema version, session id, source mode, recorded data sources,
synthetic status, eligibility (always false) with its reason, the standing
disclaimer, the software-self-check banner where it applies, session start
and end, completion state, record counts, malformed line numbers, quality
and estimate *unavailable* statements, adaptation message counts, anomaly
counts and reasons, source paths for audit, and a SHA-256 of every source
file.

There is **no clean variant**. `is_synthetic`,
`scientific_evaluation_eligible`, the eligibility reason, the disclaimer,
and the banner are required fields, and the model refuses a synthetic report
that omits the banner, a report that claims eligibility, or a report that
rewords the disclaimer. Downloading sends a copy to the browser; the
recording on disk is untouched, and the report's own checksums let a reader
confirm that afterwards rather than take it on trust.

The report contains no raw frame, image, video, face crop, landmark array,
personal name, email address, secret, or estimator binary — none of which is
representable in the source format either. Participant identifiers are not
lifted into the report.

## 6. Artifact catalogue

`build_catalogue(root, validate_checksums=...)` scans a configured root
(default `artifacts/experiments`) and returns one summary per candidate
directory. Ordering is by directory name — deterministic, and independent of
filesystem iteration order and of modification times.

### Run-family detection

Families are detected from **artifact signatures**, never from directory
names. A folder called `m7-trial` that holds no Milestone 7 document is not
a Milestone 7 run.

| Family | Distinguishing artifacts |
|---|---|
| `adaptation` | `adaptation_summary.json` + `adaptation_policy_config.json` |
| `uncertainty` | `uncertainty.json` + `uncertainty_config.json` |
| `personalization` | `personalization.json` + `personalization_config.json` |
| `fusion` | `fusion_metrics.json` + `fusion_config.json` |
| `baseline` | `manifest.json` + `metrics.json` + `splits.json`, **and none of the above** |
| `unknown` | nothing matched |

Detection runs in two passes. The first requires *every* distinguishing
artifact and yields a confident classification. The second accepts *any* of
them, so a fusion run that died before writing `fusion_metrics.json` is
reported as an **incomplete fusion run** rather than as an unclassifiable
directory — the reader needs to know which run failed. Baseline carries a
*disqualifying* set, because every training family writes `manifest.json`,
`metrics.json`, and `splits.json`; "baseline" means "a training run carrying
none of the later milestones' documents".

Directory names appear in the run selector as display metadata. They take no
part in classification, and the detection note on each run says which
artifacts were used.

Where a manifest records `configuration.milestone` or `configuration.kind`,
it is cross-checked against the signature. A disagreement produces a visible
error-level warning; the signature is used and the conflict is not resolved
silently.

### Run status

| Status | Meaning |
|---|---|
| `completed` | A conclusive manifest or summary says the run finished, and every required artifact is present |
| `failed` | A conclusive manifest records a failure, with its reason |
| `incomplete` | No conclusive document, or a required artifact is absent |
| `corrupt` | A document exists but could not be parsed |
| `unsupported` | The artifact declares a schema version this dashboard cannot interpret |
| `unknown` | Nothing recognisable was found |

**A directory existing is not a successful run.** Results from an incomplete
or failed run are never displayed as though the run had completed.

### Supported artifact versions

`dataset_schema_version` and `feature_catalog_version` are validated against
`1.0`. An unknown version makes the run `unsupported` with a stated reason.
The format is never guessed at.

## 7. Integrity and checksums

`validate_checksums` (default `true`) compares each run's recorded SHA-256
against the bytes on disk.

| Status | Meaning |
|---|---|
| `valid` | Every recorded checksum matched |
| `mismatched` | At least one file's bytes differ from what the run recorded |
| `checksum_file_unavailable` | The run records no `checksums.json` |
| `referenced_file_missing` | A checksum names a file that is no longer present |
| `checksum_file_corrupt` | `checksums.json` exists but could not be parsed |
| `not_checked` | Verification was switched off |

A mismatch produces a visible **error**, both in the run's provenance banner
and on the Run-integrity page. Nothing is deleted, regenerated, or repaired.
Switching verification off does not make a mismatched run valid; it makes
the status `not_checked`, which is a different statement.

## 8. Provenance banner

Every result-bearing page renders `provenance_banner()` **at the top** —
not in an expander, not in a footer, not in a tooltip. It shows:

- data source, synthetic status, evaluation mode
- `scientific_evaluation_eligible`, stated in words
- target, task type, run id, run directory, family
- dataset fingerprint and split fingerprint where recorded
- model / fusion strategy / policy mode where recorded
- run status and checksum status, each with text as well as colour
- any warnings, at their declared level

For a synthetic run it additionally renders, prominently:

```
SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION
```

There is **no green "validated" UI**. Software checks passing is not
validation, and the dashboard has no visual vocabulary that could say
otherwise.

### 8a. Synthetic, public, and live are labelled in words

`data_source` values are terse — `synthetic`, `public_dataset`, `live` —
and `public_dataset` in a monospaced cell is easy to skim past.
`presentation.data_source_label` renders the label **beside** the recorded
value, never instead of it, and `data_source_statement` renders what that
source does and does not establish:

| Recorded | Label | Statement |
|---|---|---|
| `synthetic` | `SYNTHETIC (recorded as 'synthetic')` | generated by this repository; a software self-check, never evidence about a person |
| `public_dataset` | `PUBLIC (recorded as 'public_dataset')` | being public does not make anything here scientifically eligible |
| `live` | `LIVE (recorded as 'live')` | *live* says where the bytes came from, not that a study was designed, labelled, approved, or validated |
| `mixed` | `MIXED (recorded as 'mixed')` | the strictest reading present applies |
| anything else | `UNRECOGNISED (recorded as '…')` | shown verbatim, never interpreted |
| absent | `Unavailable — no data source is recorded` | stated, never filled in |

The vocabulary is the project's own `engagevr.schemas.session.DataSource`
enum; no provenance string is invented here, and a test asserts every enum
member has both a label and a statement.

These appear on the artifact provenance banner, the session provenance
banner, the *Recorded data sources* table on both session pages, the
session catalogue table, and the *Data source of every window* table on the
dataset page.

**Neither public nor live implies scientific eligibility.** They are
independent fields: `data_source` says where content came from, and
`scientific_evaluation_eligible` is a separate declaration the artifact
makes. A public run and a live recording are both rendered as
`scientific_evaluation_eligible = false` with the reason stated, and the
standing disclaimer is rendered in every mode for every provenance.

No public dataset and no live participant recording exists in this
repository, so `tests/unit/test_dashboard_provenance_labels.py` builds all
three cases as **temporary fixtures** and renders them through the real
pages. "There is nothing to label" would not have shown that the labelling
works — only that it had never been exercised.

## 9. Provenance propagation

`DashboardProvenance` refuses to be constructed with
`scientific_evaluation_eligible=True` when the artifact says the data is
synthetic. `derive()` refuses to change either flag, so a chart, a filtered
table, or an exported view cannot lose the synthetic flag on the way to the
page.

**No dashboard control can override artifact provenance.** There is no
"Mark as scientifically validated" button, no "Treat as real" switch, and no
configuration key that could reach those fields. Where two sources disagree
(a manifest that claims eligibility while recording a self-check evaluation
mode), the run is shown as `corrupt` with the contradiction stated.

## 10. Pages

1. **Overview** — run counts by family and status, then the selected run's
   identity, group/fold counts, artifact completeness, and checksum state.
   Deliberately **no headline engagement score**: the page orients a
   researcher to an experiment, it does not monitor a person.
2. **Dataset and provenance** — dataset fingerprint, row/window count,
   subject and session counts, window geometry, targets and their
   distributions, data-source counts, split strategy, folds, and the
   recorded leakage-audit result. A count the artifact does not record is
   *Unavailable*; nothing is derived from a filename.
3. **Signal and feature quality** — per-feature missingness, modality
   coverage, and the missingness distribution.
4. **Baseline models** — Milestone 5 results, task-aware (see §11).
5. **Multimodal fusion** — strategies, experts, support weights,
   disagreement diagnostics, modality availability, robustness scenarios.
6. **Personalization** — population and personalized arms side by side, the
   metric difference, coverage and cold-start counts, per-fold and
   per-subject diagnostics.
7. **Uncertainty and abstention** — selective accounting, abstention
   reasons, thresholds, and the task-appropriate distributions and curves.
8. **Adaptive environment** — controller behaviour only (see §15).
9. **Run integrity** — every discovered run's status, integrity, synthetic
   flag, eligibility, fingerprint, missing artifacts, and failure reason;
   then the selected run's artifact inventory.
10. **Limitations and scientific status** — the standing limitations as a
    real page, the project vocabulary, and the privacy statement.

### The two session pages

Reached through the mode selector rather than the page list, because they
answer a different question about different evidence.

11. **Live session observation** — `Mode: LIVE OBSERVATION`, the stated
    automatic-refresh cadence, the session provenance banner, record and
    malformed counts, what arrived since the previous pass, the recorded
    task state and difficulty, message-type composition, recorded data
    sources with their labels, ordering anomalies, adaptation messages, an
    explicit list of what a recording cannot carry, the most recent records,
    and the session-report export. Everything below the heading and the
    cadence statement is inside the auto-refreshing fragment; the scope
    warning above them is not, because a statement about what this view is
    may not be something a timer can remove.
12. **Session replay** — the same common blocks, plus a clamped cursor with
    *jump to beginning*, *step backward*, *step forward*, *jump to end*, and
    a position slider; one record's full detail at the cursor; the complete
    recorded sequence; and the same export.

Neither page shows an engagement value, a cognitive-load value, a model
confidence, an interval, or an abstention, because a session recording
contains none. Each says so explicitly rather than leaving a gap, and
neither carries a red/green person-status indicator or a "user is
disengaged" alert — there is no field from which such a thing could be
constructed.

## 11. Classification versus regression

The UI is task-aware and **hides** the irrelevant controls rather than
showing disabled ones.

| | Classification | Regression |
|---|---|---|
| Result metrics | accuracy, balanced accuracy, macro P/R/F1, weighted F1 | MAE, RMSE, median AE, R² |
| Distribution | calibrated confidence, predictive entropy, probability margin | prediction-interval width |
| Coverage axis | `confidence_threshold` | `maximum_interval_width` |
| Axis units | probability in [0, 1] | the regression target's own units |
| Direction | raising it is **stricter**: coverage non-increasing | raising it is **more permissive**: coverage non-decreasing |
| Extra views | confusion matrix, calibration, reliability diagram | observed-vs-predicted, residual distribution, residual-vs-predicted |

`UncertaintyDashboardData` **refuses** to hold a calibrated-confidence field
on a regression run and **refuses** to hold an interval field on a
classification run. A page cannot show the wrong control, because the view
model will not carry it.

Neither axis is relabelled "uncertainty threshold". `1 - interval_width` is
never computed: an interval width is in the target's own units, is not
confined to [0, 1], and is not convertible into a confidence score.

A coverage curve whose recorded axis disagrees with the task type is **not
displayed**, with the disagreement stated. A curve written before DEC-072
(which carries no `axis` field, because the two axes shared one grid) is
likewise refused with a stated reason, because which axis it was swept over
cannot be established and must not be guessed.

## 12. Terminology

These are separate quantities. None is a synonym for another, and there is
**no single combined "uncertainty score"** anywhere in the dashboard.

| Quantity | What it is not |
|---|---|
| Signal quality | not engagement, not cognitive load, not model confidence |
| Calibrated classification confidence | not certainty, not signal quality, undefined for a regression target |
| Predictive entropy | not a confidence score, not a probability |
| Probability margin | not the confidence score, not an interval width |
| Ensemble disagreement | not calibrated uncertainty, not confidence, not signal quality |
| Fusion support weight | not a probability of correctness, not model confidence |
| Regression prediction-interval width | not a probability, not confined to [0, 1], never shown as `1 - width` |
| Selective coverage | not accuracy, not interval coverage |
| Abstention rate | not an error rate, not the unavailable count |
| Empirical interval coverage | not selective coverage, not a guarantee, not established across subjects |

There is no card named simply "Confidence".

## 13. Signal quality

The quality page carries this statement beside every view:

> Signal quality describes measurement reliability and availability. It is
> not engagement, not cognitive load, and not model confidence. A window
> with poor signal quality is a window that was hard to measure, not a
> person who was disengaged.

Low quality is **never** labelled disengaged, inattentive, low cognitive
load, or "poor participant". If the selected run recorded no quality
information, the page says so; it is never fabricated from a confidence
score that happens to live in the same numeric range.

## 14. Confusion matrices and regression plots

Confusion matrices are rendered from recorded counts only. For a synthetic
run the row axis is **`observed synthetic label`** and the column axis is
**`predicted class`**. The word *ground truth* does not appear: this
repository has no participant ground truth, and the axis label is exactly
where that would silently be claimed.

Regression scatter plots use stored observed and predicted values. Residuals
are differences between two values the run already recorded; nothing is
re-predicted. For a synthetic run the observed axis reads *synthetic
target*, never *true engagement*. No statistical significance is implied by
any pattern.

## 15. Adaptation views

The adaptation page reports **controller behaviour**. It has no
effectiveness card, no benefit metric, and **no field in which one could be
written** — the view model has no such attribute and `extra="forbid"`
prevents adding one.

Displayed: evaluated windows, Milestone 7 eligible/blocked, HOLD count and
reason distribution, proposals, increases, decreases, direction reversals,
proposal spacing, longest same-direction streak, blocked oscillation
attempts, the configured guards, per-session behaviour, the difficulty
trace, the scenario table, and the lifecycle.

### Lifecycle states stay separate

| Stage | Meaning | Milestone 8 |
|---|---|---|
| Proposal | The policy decided a change would be appropriate | > 0 |
| Command built | A `set_difficulty` payload was constructed in memory | > 0 |
| Dispatched | The command was transmitted to a running environment | **0** |
| Acknowledged | The environment confirmed receipt | **0** |
| Applied | The environment confirmed the change took effect | **0** |

These five counts are **never added together** and never collapsed into one
"adaptations" number. `AdaptationLifecycleCounts` refuses an ordering that
could not have happened: a command without a proposal, a dispatch exceeding
the commands built, an acknowledgement without a dispatch.

### Difficulty trace

`window_order` against recorded difficulty, one series per session, with the
subtitle **"Synthetic controller scenario — software diagnostic only"**. A
flat line means the controller held, which is the ordinary outcome. It is
never labelled a participant adaptation response.

### Static versus adaptive

A run in `experiment_mode=static` is shown as a **legitimate experimental
control condition**. Every policy decision holding is the expected behaviour
of that condition — not a malfunction, not a disabled participant, and not
low engagement.

### Controller comparison

The Milestone 8 guard-free comparison is displayed only as a
**software-controller action-frequency comparison**. It shows that the
temporal guards mechanically reduce how often the controller acts. It is not
a claim that either controller is better, safer, or more effective for any
person.

## 16. Personalization views

Population and personalized results are shown side by side over identical
evaluation windows, with the difference in a column headed
`personalized - population` and titled **"Δ metric on synthetic
software-check data"**.

On the current synthetic runs the personalized arm scores **lower**, and
that is displayed exactly as recorded. Negative differences are never
hidden, and the words *improvement*, *benefit*, and *gain* do not appear in
any data row. Cold-start subjects and population fallbacks are reported as
designed behaviour, not as failures.

Per-subject views are labelled *subject-wise software evaluation* — a
group-level diagnostic. There is no "best subject" or "worst subject" table
and no ranking of any kind.

## 17. Selective-prediction accounting

Any summary showing accepted / abstained / unavailable reconciles:

```
accepted + abstained + unavailable = evaluated
```

When it does not, the page shows an **artifact validation error** with the
recorded counts unchanged. The mismatch is never normalised away.

An **abstention is not an error** and is not counted as one. An
**unavailable** window is a separate state again: nothing was withheld,
because nothing was produced.

## 18. Missing, corrupt, and incomplete artifacts

| Situation | Behaviour |
|---|---|
| Artifact root absent | Dashboard starts; states the root does not exist |
| No runs found | Stated as a fresh root, not an error |
| Required artifact missing | Run is `incomplete`; no result is displayed |
| Optional artifact missing | One view says *unavailable* with the filename; the rest of the page renders |
| Malformed JSON | Run is `corrupt` with the parser's message; listed, not skipped |
| Missing Parquet column | The chart that needed it is unavailable, naming the column |
| Checksum mismatch | Visible error; nothing regenerated |
| Unsupported version | Run is `unsupported`; the format is not guessed |

**One bad run never takes the dashboard down.** A bad run is visible as a
bad run.

Error messages name the artifact: *"coverage_curve.json is unavailable for
this run"*, not *"something went wrong"*. When a chart cannot render, the
reason is shown; zeros are never substituted.

## 19. Numeric formatting

All formatting is centralised in `dashboard/formatting.py`.

- `None` renders as **"Unavailable"**, never as `0`
- `NaN` and `±inf` are **refused** at construction and render as
  "Unavailable" with a stated reason
- `0.0` remains a legitimate zero
- counts render as integers; a fractional count is refused
- percentages carry `%`; probabilities do not, so the two stay distinguishable
- interval widths carry the regression target's units and never a `%` sign
- precision is consistent across every metric of the same kind

There is no `float(value or 0)` anywhere in the dashboard, and a test
enforces it.

## 20. What the dashboard may and may not derive

**May** (display aggregation over persisted records): histogram bins, group
counts, means and medians for a summary row, confusion-matrix totals,
residuals from stored observed and predicted values, and the arrays a
recorded curve is plotted from.

**May not** (modelling): retrained predictions, new probability
calibrations, new conformal quantiles, new personalization corrections, new
adaptation decisions. A histogram of stored confidences is a picture of
stored confidences; a recalibration of them would be a claim.

The reliability diagram is descriptive: it pools the calibration bins the
run already recorded, weighted by window count. **No calibration model is
fitted or refitted.**

## 21. Read-only guarantees

The dashboard **may**: discover runs, load artifacts, validate checksums,
filter tables, select runs and sessions, and render.

It **may not**, and structurally cannot: modify an artifact, change
configuration, promote or champion a model, retrain, recalibrate, dispatch
an adaptation command, acknowledge a command, perform a Git operation, or
delete anything.

There is no "Apply adaptation" button and no "Run model" button.

Enforced by AST tests over the dashboard's own source:

- no module calls `write_text`, `write_bytes`, `mkdir`, `unlink`, `rmtree`,
  `rename`, `replace`, `write_json_atomic`, or `write_parquet_atomic`
- no module opens a file in a write mode
- no module calls `fit`, `predict`, `calibrate`, `evaluate_policy`,
  `evaluate_adaptation_gate`, `build_adaptation_command`, `dispatch`, or
  `send`
- no module imports a training or adaptation runner, the transport layer,
  the API layer, `websockets`, `fastapi`, `uvicorn`, `joblib`, `pickle`, or
  `sklearn`
- only `launch.py` imports `subprocess`, and only to start Streamlit
- no module mentions a Git-writing command

For the live and replay modes, additionally:

- no module constructs `SessionRecorder`, `JsonlWriter`, or `ReplayPlayer`,
  and none calls `open_recorder`, `record_drop`, `publish`, `broadcast`, or
  `connect`
- no module imports `engagevr.replay.player`, `engagevr.replay.clock`,
  `engagevr.task.simulator`, `engagevr.task.generator`,
  `engagevr.storage.jsonl`, `asyncio`, or a socket module
- `session_reader.py` opens files in `"rb"` only and mentions no write mode
- the browser download control is the only export path; it is not a write to
  a source artifact, and `session_pages.py` calls no `write_text` or
  `write_bytes`

Reading is also checked behaviourally, not only structurally: tests digest
every file of a recording before and after a full read, a catalogue scan, a
replay traversal, and a report build, and assert the digests are unchanged.

## 22. Model files are never opened

`models/*.joblib` are Python pickles, and loading a pickle executes code in
it. The dashboard reads **JSON and Parquet only**. Everything needed to
judge a run — provenance, metrics, splits, disclaimers — is in the JSON
documents, so auditing a run never requires unpickling anything.

## 23. Privacy

Subject and session identifiers in this repository are **pseudonymous
labels generated by software**. The dashboard shows them for research audit
when present, and `dashboard.show_subject_ids` controls that as an audit
convenience rather than a privacy control — there is nothing personal in the
artifacts to conceal.

The dashboard does not resolve identifiers to people, display names or email
addresses, show profile pictures or avatars, infer identity, rank
participants, or produce "best"/"worst" subject tables.

It reads **no video, no image, and no webcam frame**. It imports no capture,
face, or rPPG module, and mentions no `cv2`, `imread`, `st.image`,
`st.video`, or `st.camera_input`. Test fixtures contain no personal name,
email address, or contact detail — subject identifiers are of the form
`synthetic-subject-0001`.

## 24. Visual design and accessibility

A restrained research-software interface: tables, line charts, scatter
charts, histograms, confusion matrices as labelled tables, and compact
metric cards. No neon styling, no medical-monitor aesthetic, no gauges, no
gamified scores, no avatars, no rankings, no decorative animation.

**Colour is never the only carrier of meaning.** Every run status, integrity
state, class label, and eligibility flag is written out in words. Colour on
a class label describes a category, not desirability: nothing here says
green means psychologically good.

Every chart carries a title, both axis names, and — where a misreading is
possible — a note on the axis's units and semantics. Every chart has a table
of its plotted values beneath it. Disclaimers are rendered at normal body
size, never as fine print.

## 25. Caching

`st.cache_data` caches the catalogue scan only. The cache key carries the
name and modification time of every candidate run directory, so it
invalidates when a run is written, replaced, or removed. Only read-only
parsing is cached: no mutable policy or model state crosses that boundary,
and no long-lived file handle is created — every read opens and closes.

## 26. Performance

The catalogue reads small JSON documents only; listing a hundred runs opens
no Parquet file. Detailed artifacts load after the reader selects a run and
a page. Parquet reads select only the columns a view displays. Table
rendering is limited by `dashboard.max_table_rows`, and truncation is
**stated**, never silent — the artifact on disk stays complete.

No database is introduced.

## 27. Configuration

```yaml
dashboard:
  artifact_root: "artifacts/experiments"
  default_run_family: null
  validate_checksums: true
  max_table_rows: 1000
  show_subject_ids: true
  histogram_bins: 20
  session_root: "artifacts/sessions"
  live_refresh_seconds: 5.0
  enable_session_report_export: true
```

Every setting controls **presentation and discovery**. Validation: both
roots must be repository-relative and must not escape with `..`; the default
family must be a known family or `null`; `max_table_rows` is a positive
integer; `histogram_bins` is between 2 and 200; `live_refresh_seconds` must
be finite, greater than zero, and between 2 and 3600 — checked twice, once
by `DashboardConfig` and again by `views_session.live_refresh_interval`
before the value reaches Streamlit, because a page context is a plain
dataclass and the day one is built from something other than
`configs/defaults.yaml` the page validator is the only thing left. It drives
the live page's automatic refresh and nothing else.

`enable_session_report_export` removes the download control. It cannot alter
what a report *contains*: provenance, the disclaimer, and the
software-self-check banner are required fields of the report model.

`DashboardConfig` has `extra="forbid"` and carries **no** field for
`scientific_evaluation_eligible`, `is_synthetic`, a confidence threshold,
the policy mapping, or model outputs. Dashboard configuration cannot alter a
scientific calculation, and it cannot override artifact provenance.

## 28. Testing

889 tests, none of which requires a webcam, MediaPipe asset, Unity, network
access, participant data, external dataset, browser, long-running Streamlit
server, MLflow, DVC, or Docker. None of them sleeps, and none waits on a
timer: the automatic refresh is verified where it is configured — at the
fragment boundary — rather than by waiting for a firing.

| Module | Covers |
|---|---|
| `test_dashboard_catalogue.py` | discovery, family detection, status, integrity, ordering |
| `test_dashboard_schemas.py` | structural invariants of every view model |
| `test_dashboard_formatting.py` | formatting rules and display-only aggregation |
| `test_dashboard_views.py` | dataset, quality, baseline, fusion, personalization |
| `test_dashboard_uncertainty.py` | selective accounting, both coverage axes |
| `test_dashboard_adaptation.py` | controller counts, lifecycle, wording |
| `test_dashboard_readonly.py` | AST boundary tests, layering, privacy, configuration |
| `test_cli_dashboard.py` | all three commands, launch-argv construction, report printing |
| `test_dashboard_app.py` | artifact page smoke tests via Streamlit's `AppTest` |
| `test_dashboard_session_reader.py` | tail safety, ordering, session discovery, read purity |
| `test_dashboard_replay.py` | replay cursor, mode distinction, session view models |
| `test_dashboard_session_report.py` | determinism, reconciliation, provenance, privacy |
| `test_dashboard_session_app.py` | live and replay page smoke tests via `AppTest` |
| `test_dashboard_live_refresh.py` | interval validation, the fragment boundary, and which modes may schedule one |
| `test_dashboard_provenance_labels.py` | synthetic, public, and live labelling through the real pages |
| `dashboard_fixtures.py` | builders for minimal run directories, synthetic or public |
| `session_fixtures.py` | builders for minimal recordings — synthetic, live, public, and damaged |

`session_fixtures.py` writes recordings line by line rather than through
`SessionRecorder`, for two reasons: a test must be able to produce states a
well-behaved recorder never produces — a torn final line, a malformed
interior line, a duplicated sequence number — and the dashboard tests must
not depend on the writer they assert the dashboard never uses.

Page smoke tests use `streamlit.testing.v1.AppTest`, which runs the script
in-process with no browser and no socket. **No Selenium, Playwright, or
browser dependency was added.**

The refresh tests observe `run_every` as it is handed to `st.fragment` and
let the real decorator run, so the fragment renders exactly as it would in
a browser while the cadence is checked without waiting for one. The
provenance tests build a public run and a live recording as temporary
fixtures, because neither exists in this repository and an untested
labelling path is not a delivered one.

## 29. Limitations

1. Nothing this dashboard displays is scientific evidence. Every run it can
   currently show is synthetic.
2. No validated participant-labelled engagement or cognitive-load dataset
   exists, so no view here is about engagement or cognitive load.
3. No human-subject evaluation of adaptation has been performed. The
   adaptation page reports what a software controller did.
4. The dashboard has never been used by anyone but its author, and its
   usability has not been evaluated.
5. Accessibility follows stated rules (axis labels, text beside colour,
   readable disclaimers) but has not been tested with assistive technology.
6. It reads the artifact contracts of Milestones 5–8 as they stand at
   Milestone 9. A contract change requires a corresponding dashboard change;
   the reader is protected by the *unsupported* status, not by adaptability.
7. Pre-DEC-072 uncertainty artifacts cannot have their coverage curve
   displayed, because the axis they were swept over is not recorded.
8. Physical-webcam, UBFC-rPPG, and Unity validation all remain pending, as
   they were before this milestone.
9. The live mode observes a *recording*, not a running process. It cannot
   report that a session has stalled, only that no new record has appeared
   since the previous read, and it cannot distinguish a session still
   running from one that stopped without writing a summary.
10. The live mode has never been exercised against a session being written
    by a real Unity client, because the Unity task has never been compiled
    or run. It has been exercised against recordings produced by the Python
    simulator and against synthetic fixtures.
11. The live refresh has a floor of two seconds and a browser tab must stay
    open for it to fire. It is a research view, not a wall display or a
    monitoring service, and it reports nothing when nobody is looking.
12. A session report is a presentation artifact. It creates no evidence, and
    two identical reports of a synthetic recording are two identical
    statements about synthetic data.

## 30. Pending validation

1. A validated participant-labelled dataset, without which no page here can
   become evidence.
2. Human-subject evaluation of engagement, cognitive load, and adaptation.
3. Usability evaluation of the dashboard with a researcher who did not write
   it.
4. Accessibility testing with assistive technology.
5. Live observation of a session written by a compiled Unity client.

---

> Milestone 9 research dashboard implementation complete; scientific
> evaluation and human-subject validation remain pending.
