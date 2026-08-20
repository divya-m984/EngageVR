# EngageVR -- Adaptive Environment (Milestone 8)

> **No adaptation rule in this repository is psychologically validated,
> pedagogically optimal, therapeutic, safe, or demonstrated to benefit any
> person.** The mapping from an estimated state to an adaptation direction is
> an **engineering demonstration rule**. Every threshold, dwell time,
> cooldown, bound, and budget below is an engineering default. No
> human-subject evaluation of adaptation appropriateness, usability, or
> benefit has taken place.

---

## 1. Responsibility boundary

Milestone 8 is the first layer in this project that *chooses* anything. It is
deliberately small, and it sits between two layers that already existed.

| Milestone | Question it answers |
|---|---|
| M5 / M6 | What is the estimated engagement or cognitive-load state? |
| M7 | Is there enough evidence and confidence to act on that estimate at all? |
| **M8** | **Given an ELIGIBLE estimate and the current task state, should a conservative change be PROPOSED?** |
| M4 | How does an explicit adaptation command reach a task client? |

M8 **does not** retrain a model, recalibrate a probability, redefine
uncertainty, change the fusion layer, or send anything.

The boundary is enforced by imports, not by convention.
`src/engagevr/adaptation/policy.py` imports two schema modules and its own
mapping table. It imports nothing from `engagevr.api`, `engagevr.transport`,
`engagevr.task`, or `engagevr.training`, so a reader can establish from its
import list that it cannot open a socket and cannot recompute a Milestone 7
quantity. A test asserts this by parsing the module's AST.

---

## 2. The Milestone 7 gate is a hard prerequisite

An `AdaptationGateRecord` with `decision = BLOCKED` makes the policy return
`HOLD`. **There is no override flag anywhere in Milestone 8.**

The dependency is enforced in three independent places:

1. **Control flow.** The gate check runs before the mapping is consulted.
2. **The proposal schema.** `AdaptationProposal` embeds the
   `AdaptationGateRecord` of *both* targets and refuses to validate unless
   both are `ELIGIBLE`. A proposal's eligibility is therefore provable from
   the object rather than asserted by it.
3. **The decision schema.** `AdaptationPolicyDecision` refuses to validate a
   `propose_adaptation` decision whose recorded gate decisions are anything
   other than `eligible`.

Milestone 7's own reasons are preserved verbatim, in Milestone 7's canonical
order, on `AdaptationTargetSuggestion.gate_reasons`. Milestone 8 may *add* its
own reason (`gate_blocked`, `prediction_abstained`, `prediction_unavailable`)
but never erases the underlying ones.

Milestone 8 never recomputes `max(probabilities)`, never lowers a threshold,
never overrides a personalized or population threshold decision, and never
uses entropy or margin to make a blocked window eligible.

---

## 3. Policy inputs

`AdaptationInput` is everything one evaluation may see:

| Field | Meaning |
|---|---|
| `session_id`, `subject_id`, `window_id` | pseudonymous references already present in the M5-M7 artifacts |
| `window_order` | monotonic index within the session; the policy's only notion of time |
| `current_difficulty` | what the environment **reports** it is at, or `None` for unknown |
| `engagement` | `AdaptationTargetEvidence`: the M7 `AbstentionDecision` **and** `AdaptationGateRecord`, carried whole |
| `cognitive_load` | the same, for the load target |
| `is_synthetic`, `scientific_evaluation_eligible` | provenance; synthetic can never be eligible |

There is no frame, no landmark, no name, and no email here, and the model
forbids extra fields, so none can be smuggled in.

**Both ordinal targets are required.** This is derived from the mapping rather
than imposed on it: an increase requires high engagement *and* low cognitive
load, and a decrease requires high cognitive load. A single target cannot
select any direction, so partial evidence holds with
`insufficient_evidence` rather than inventing the missing signal's state.

---

## 4. Action vocabulary

**The protocol did not change.** `git diff --exit-code -- protocol/` is clean.

The Milestone 4 protocol already defines `AdaptationCommandName`:

| Action | Value | Used by the policy? |
|---|---|---|
| `set_difficulty` | integer >= 0 | **yes** |
| `set_stimulus_interval` | positive number | no |
| `pause_task` | none | no |
| `resume_task` | none | no |

The policy reasons *internally* about a direction:

```
increase | decrease | hold
```

and an approved `increase`/`decrease` is resolved into a target level before
the existing `set_difficulty` command is built:

```
desired = current +/- step        (step from configuration, clamped to bounds)
```

`pause_task` and `resume_task` are deliberately unused. The only
specification rule that would call for a break is *"sustained fatigue
indicators -> a break"*, and no fatigue estimator exists in this repository.
Inferring fatigue from blink proxies or heart-rate estimates is exactly the
measurement-to-construct leap `reject_automatic_derivation` refuses. A test
asserts that no Milestone 8 module mentions either command.

---

## 5. HOLD is a first-class decision

A hold is normal and common. It is not an error, not a failure, and not a
missing value. A hold states at least one reason and carries **no proposal**,
and therefore no command payload.

| Reason | When |
|---|---|
| `adaptation_disabled` | the experimenter lock is on |
| `static_experiment_mode` | the session runs in the static condition |
| `duplicate_window` | this window was already evaluated; the repeat was absorbed |
| `prediction_unavailable` | M7 recorded no prediction for a required target |
| `prediction_abstained` | M7's selective layer abstained on a required target |
| `gate_blocked` | M7's adaptation gate blocked a required target |
| `insufficient_evidence` | a required target contributed no record at all |
| `no_policy_for_target` | no mapping is configured for that task type |
| `no_expressible_action` | the state pair indicates a response the protocol cannot express |
| `target_in_deadband` | every contributing state fell in the neutral region |
| `adaptation_not_needed` | the mapping resolved to hold for this state pair |
| `direction_conflict` | the two targets suggested opposite directions |
| `insufficient_persistence` | fewer consecutive supporting windows than the dwell requirement |
| `direction_change_blocked` | this window's direction differs from the one being counted |
| `cooldown_active` | a previous proposal's cooldown has not elapsed |
| `session_adaptation_budget_exhausted` | the session's budget is spent |
| `current_state_unavailable` | the environment reported no current difficulty |
| `already_at_minimum` / `already_at_maximum` | the bound was already reached |

`proposal_eligible` is the single reason a proposing decision carries; the
schema refuses a proposal that also records a blocking reason, and refuses a
hold that records `proposal_eligible`.

Declaration order in `AdaptationPolicyReason` is the canonical reporting
order, so two runs of one configuration produce identical documents.

---

## 6. The mapping table

### Where it comes from

`docs/PROJECT_SPECIFICATION.md` states an example policy. Three of its rows
are expressible with `set_difficulty` and are implemented verbatim:

- "High engagement + low load -> increase difficulty slightly"
- "Moderate engagement + moderate load -> maintain current state"
- "High load + declining performance -> reduce difficulty"

Two are **not** implemented, and their absence is deliberate:

- *"Declining engagement + low or moderate load -> feedback or introduce
  variation."* Neither feedback nor stimulus variation is an action this
  protocol can express, and "declining" is a temporal derivative of a quantity
  this project has never validated. The rule holds and records
  `no_expressible_action` rather than substituting a difficulty change for the
  response the specification actually named. In particular, **"low engagement
  therefore make it harder" is not implemented**: it is a psychological
  assumption, not a reading of the evidence.
- *"Sustained fatigue indicators -> a break."* No fatigue estimator exists.

### Two principles and a default

- **P1 -- overload protection.** Cognitive load `HIGH` suggests `DECREASE`.
  This is the only protective direction and the only rule that fires on one
  signal's state.
- **P2 -- engagement headroom.** Engagement `HIGH` suggests `INCREASE`, and an
  increase is proposed only when cognitive load is affirmatively `LOW`. A
  `MEDIUM` load is the absence of a reading that supports an increase, not a
  reading that supports one.
- **P3 -- default hold.**

### The table

| engagement \ cognitive load | low | medium | high |
|---|---|---|---|
| **low** | hold (`no_expressible_action`) | hold (`no_expressible_action`) | **decrease** |
| **medium** | hold (`adaptation_not_needed`) | hold (`target_in_deadband`) | **decrease** |
| **high** | **increase** | hold (`adaptation_not_needed`) | hold (`direction_conflict`) |

The table lives as data in `MAPPING_TABLE`, so it can be checked against this
document without following control flow. It is total: all nine pairs have an
entry and there is no fall-through case.

### Per-target suggestions are recorded separately

Every decision records what each signal alone suggested
(`engagement.suggested_direction`, `cognitive_load.suggested_direction`),
whether they conflicted, the resolved direction, and why. Conflicts are never
hidden.

---

## 7. Ordinal semantics

`low < medium < high` is represented explicitly, twice, and the two
declarations must agree:

1. `TargetSpec.class_order_is_ordinal` on the target schema declares that the
   vocabulary is an *ordered* scale. A vocabulary without this flag carries no
   ordering, and the policy refuses to invent one.
2. `ORDINAL_CLASSIFICATION_TARGETS` in the policy schema names the exact
   vocabulary each target must declare.

Neither array position nor alphabetical order is consulted. A vocabulary that
was reordered, extended, or renamed stops the policy with
`AdaptationPolicyError` rather than quietly changing what "high" means.
An unknown class label is refused, never mapped to the neutral state.

---

## 8. Conflict resolution

| Setting | Behaviour |
|---|---|
| `hold` (default) | opposite suggestions produce `HOLD` with `direction_conflict` |
| `prefer_decrease` | the protective direction is taken; the conflict is still recorded |

There is deliberately **no `prefer_increase`**: conflicting evidence never
resolves toward the more demanding state.

---

## 9. Deadband

For the classification targets, `medium` is the neutral class and acts as the
deadband: a state pair of `medium`/`medium` holds with `target_in_deadband`.
This is what stops the controller acting because an estimate landed
infinitesimally on one side of a boundary.

For regression targets (disabled by default -- see section 10) the deadband is
an explicit `low_below` / `high_above` pair with a neutral region between
them, inclusive of both boundaries.

---

## 10. Regression mapping (disabled by default)

`adaptation.policy.regression_mapping.enabled` defaults to `false`. The
project's ordinal classification targets already carry a neutral class, so
nothing is forced into a continuous form for symmetry, and a regression target
supplied while the mapping is off holds with `no_policy_for_target`.

When it is enabled:

- Both `low_below` and `high_above` must be stated for **both** score targets.
  There is no default; a default boundary would be an invented threshold on a
  scale this repository has never measured, so enabling the mapping without
  stating them is a configuration error.
- Boundaries are validated finite, ordered (`low_below < high_above`), and
  inside the target's declared range. Units come from the target spec
  (`dimensionless`, both targets normalised to `[0, 1]`).
- `value < low_below` is `LOW`; `value > high_above` is `HIGH`; everything
  between, **inclusive of both boundaries**, is `MEDIUM`.
- With `require_interval_inside_band` (default `true`), the whole Milestone 7
  prediction interval must lie in one region before that region's state is
  used. An interval straddling a boundary reads as neutral. This is a use of
  the interval's width as a deadband, not a re-derivation of Milestone 7's
  acceptance rule.

No threshold here was tuned against synthetic output.

---

## 11. Persistence (dwell)

One eligible window is not enough.

- **Default: 3 consecutive policy-evaluation windows** supporting the same
  acting direction. At the default `windowing.model_inference_seconds` of
  5 s that is 15 s of persistent evidence, which implements the
  specification's "minimum observation window" safety control. **It is an
  engineering default and is unvalidated.**
- *Consecutive* means consecutive policy-evaluation windows.
- A window resolving to an acting direction either extends the count for that
  direction or **restarts it at 1** for the new one.
- A window resolving to `HOLD` **resets** the count to zero and clears the
  pending direction. It does **not** decay it: a decay would let evidence
  separated by contradicting windows accumulate into a dwell requirement that
  was never actually met.
- A window blocked or abstained by Milestone 7 resolves to `HOLD`, so **a
  blocked window never counts as supporting evidence**. No evidence is
  fabricated for missing windows.
- A direction change records `direction_change_blocked` rather than
  `insufficient_persistence`, so the trace distinguishes "not enough yet" from
  "this window changed its mind".
- A **session change** resets everything (section 14).
- A **duplicate** window -- same `window_id` at the same `window_order` as the
  previous evaluation -- is absorbed idempotently: no count advances, no guard
  expires, and the state is returned unchanged.
- An **out-of-order** window (order at or below the last evaluated order, with
  a different id) raises `AdaptationPolicyError`. It is refused, never
  reordered.
- The count is auditable: `persistence_count_before` and
  `persistence_count_after` appear on every decision and every trace row.

---

## 12. Cooldown

**Window-based, not time-based**, so an offline replay is exactly
reproducible. One primitive, `cooldown_windows`; there is no
`cooldown_seconds` alongside it.

- **Default: 6 windows.** At 5 s per window that is the 30 s the
  specification asks for ("adaptation cooldown: at least 20-30 seconds").
- Semantics: `cooldown_remaining` counts windows that must still pass. A
  proposal sets it to `cooldown_windows`, so the next proposal can occur at
  the earliest **`cooldown_windows + 1` evaluation windows later**. With the
  default that is a minimum spacing of 7 windows.
- It decreases by one on **every evaluated window**, whether or not that
  window carried usable evidence. **Window passage is defined independently of
  evidence**: time passing is a property of the stream, not of what was in it.
  A blocked window is not evidence, but it is still a window.
- A proposal during cooldown is impossible: the guard produces `HOLD` with
  `cooldown_active`, and the decision schema additionally refuses to validate
  a proposal whose `cooldown_remaining_before` is non-zero.
- The remaining cooldown is recorded on every decision and every trace row.
- A new session starts at zero (section 14).
- A duplicate window does **not** tick the cooldown down.

---

## 13. Hysteresis

**Hysteresis is implemented through the deadband, the persistence
requirement, the direction-change reset, and the cooldown, rather than
through a dedicated knob.** No `hysteresis_*` setting exists, and a test
asserts that none appears in the configuration.

An immediate reversal is prevented by three mechanisms acting together:

1. A proposal **restarts** the dwell count at zero, so the next proposal --
   in either direction -- needs fresh evidence rather than inheriting the
   previous window's.
2. A direction change restarts the count at 1, so an isolated opposing window
   can never reach the requirement.
3. The cooldown blocks the next `cooldown_windows` windows regardless.

With the defaults a reversal therefore requires at least 7 windows, of which
at least 3 consecutive must support the new direction. The scenario suite
demonstrates this (`direction-reversal`).

---

## 14. State machine

`AdaptationPolicyState` is a frozen, serialisable Pydantic model. There is no
module-level singleton anywhere in `engagevr.adaptation`, and a test asserts
that `policy.py` contains no mutable module-level assignment.

| Field | Meaning |
|---|---|
| `session_id` | the session this state belongs to |
| `last_window_id`, `last_window_order` | stream position, for duplicate and ordering checks |
| `current_difficulty` | the last difficulty the environment **reported** -- never the last one proposed |
| `pending_direction`, `persistence_count` | the dwell count and what it counts toward |
| `cooldown_remaining` | windows that must still pass |
| `last_applied_direction`, `last_adaptation_window_order` | the most recent **proposal** |
| `adaptation_count` | proposals made this session |
| `evaluated_window_count` | windows seen |

`evaluate_policy(input, state, configuration)` is pure: it mutates nothing,
reads no clock, draws no random number, and returns a decision carrying both
`state_before` and `state_after`. The same triple always yields the same
decision, so test snapshots are possible and a run is reproducible.

### Transitions

| Situation | persistence | cooldown | budget |
|---|---|---|---|
| duplicate window | unchanged | unchanged | unchanged |
| any hold | reset to 0 | -1 (floor 0) | unchanged |
| acting direction, same as pending | +1 | -1 (floor 0) | unchanged |
| acting direction, different | reset to 1 | -1 (floor 0) | unchanged |
| proposal | reset to 0 | set to `cooldown_windows` | +1 |

### Session reset

Session boundaries matter. On a new session id the runner starts a **cold**
`AdaptationPolicyState.for_session(...)`:

- pending persistence resets;
- cooldown resets (the protocol carries none across a session);
- the previous applied direction does not leak;
- the previous subject or session identifier does not leak;
- the adaptation budget resets.

Cold start is an explicit condition, not an absence: cooldown is zero because
no proposal has been made in this session, and that fact is recorded rather
than inferred from missing history. Applying one session's state to another
raises `AdaptationPolicyError`.

---

## 15. Bounds

Difficulty has explicit inclusive bounds and a configured step. The policy is
given the current level; it never assumes one.

- **Defaults: `[1, 5]`, step 1.** `minimum` is 1 because that is where the
  task client and simulator start (`task.default_difficulty`). `maximum` is 5
  as an engineering default: the reaction task has no measured ceiling.
- The root configuration validator refuses a configuration whose bounds do not
  contain `task.default_difficulty`, so the disagreement is caught at load
  time rather than as a run-time error on every window.
- A reported difficulty **outside** the bounds raises
  `AdaptationPolicyError`. It is a data or configuration error, refused rather
  than clamped.
- An unknown (`None`) current difficulty holds with
  `current_state_unavailable`. **An unknown level is never read as level
  zero.**
- At the requested boundary the policy **holds** with `already_at_maximum` /
  `already_at_minimum` rather than emitting a command that changes nothing.
- If a configured step larger than one would overshoot, the proposal is
  clamped **and says so**: `requested_difficulty` and `proposed_difficulty`
  are both preserved and `clamping_applied` is `true`. A clamped proposal is
  never reported as the move that was asked for. `requested_difficulty` is
  deliberately unbounded so that what was asked for survives; only
  `proposed_difficulty` is constrained, because only it can become a command.

### Step size is not a control gain

`step` is a configured constant. It is **never** scaled by confidence.
Confidence decided, in Milestone 7, whether this window may be acted on at
all; reusing it as a gain would let a barely admissible estimate move the
environment further than a clear one. `AdaptationProposal` carries no
confidence field at all, and a test asserts this.

---

## 16. Session adaptation budget

`max_adaptations_per_session` caps how many proposals one session may make.
**Default: 10**; `null` means unlimited. Reaching it holds with
`session_adaptation_budget_exhausted`.

The budget counts **proposals**, not applied changes. That is the
conservative direction: a proposal that was never applied still consumes
budget, because the policy has no way to know whether the environment acted.

---

## 17. Proposal vs command vs dispatch vs acknowledgement

These are five different facts and they are never collapsed.

```
prediction -> M7 gate -> policy -> proposal -> command object
                                                  |
                                     (Milestone 8 stops here)
                                                  |
                                      dispatch -> acknowledgement -> applied
```

| Status | Means | Who may set it |
|---|---|---|
| `proposed` | the rule recommended a change | the policy |
| `command_built` | an `AdaptationCommandPayload` exists as a value | `record_command_built` |
| `dispatched` | the command left this process | a caller that actually sent it |
| `acknowledged` | a task client answered | only from a real acknowledgement payload |
| `applied` | the client said it applied the change | only from an accepted acknowledgement carrying `applied_at_utc` |
| `rejected` | the client refused, with its own stated reason | only from a rejected acknowledgement |

`AdaptationHistoryEntry` enforces this: `applied` without an acknowledgement
carrying an instant fails validation, and an acknowledgement naming a
different `command_id` is refused.

**The policy never treats a proposal as an applied adaptation.** The policy
state records the difficulty the *environment reported*, never the one the
policy proposed.

### The command builder

`build_adaptation_command(proposal, issued_at_utc=...)` is pure. It:

- reuses the existing `set_difficulty` action;
- preserves the session id (on the proposal) and the proposal and Milestone 7
  source-prediction ids (compactly, inside the bounded `reason` field -- the
  protocol has no free provenance field and this milestone did not add one);
- sets issue and expiry times only through the existing protocol semantics,
  with the issue instant supplied by the caller because the policy reads no
  clock;
- sets `is_manual=False`, which is what distinguishes a policy-derived command
  from Milestone 4's manual and scripted ones;
- uses a deterministic `command_id` derived from the deterministic
  `proposal_id`, so a retransmission is absorbed by the client's existing
  idempotency rule rather than double-stepping the difficulty;
- **refuses** a hold, a blocked gate, an out-of-bounds target level, and a
  non-task-client target role;
- **sends nothing.**

### Dispatch

Milestone 8 does **not** dispatch. `adaptation-demo --dispatch` exists only so
that asking for it produces a stated refusal rather than silently doing
nothing. No project requirement asks for live transport of a policy-derived
command in this milestone, and turning one on by default would put an
unvalidated rule in control of a running environment.

Tests assert structurally that no module in `engagevr.adaptation` imports a
transport module or calls `send`, `broadcast`, `publish`, or `dispatch`.

---

## 18. Experimenter controls and experimental conditions

Two separate things, kept separate on purpose:

| Setting | Meaning | Hold reason |
|---|---|---|
| `adaptation.enabled` | the **experimenter lock**: hold every window regardless of evidence | `adaptation_disabled` |
| `adaptation.experiment_mode` | the **static vs adaptive experimental condition** | `static_experiment_mode` |

A static condition that silently depended on the policy happening to propose
nothing would not be a static condition. The mode is recorded on every
decision and every trace row, and it participates in the configuration
fingerprint, so a static run and an adaptive run get different run ids.

---

## 19. Configuration

```yaml
adaptation:
  enabled: true              # experimenter lock
  experiment_mode: "adaptive"  # static | adaptive
  policy:
    mode: "conservative_rule_based"
    minimum_persistence_windows: 3
    cooldown_windows: 6
    max_adaptations_per_session: 10   # null = unlimited
    conflict_resolution: "hold"       # hold | prefer_decrease
    difficulty:
      minimum: 1
      maximum: 5
      step: 1
    regression_mapping:
      enabled: false
      require_interval_inside_band: true
      engagement_score:     {low_below: null, high_above: null}
      cognitive_load_score: {low_below: null, high_above: null}
```

Validated at load time: `minimum <= maximum`; `step > 0` and no larger than
the whole range; `persistence >= 1`; `cooldown >= 0`; budget `>= 0` or
`null`; band boundaries finite, ordered, and inside the target range; the
bounds contain `task.default_difficulty`; unknown policy mode, conflict
resolution, or experiment mode refused by name.

**There is no confidence or signal-quality threshold in this section.**
Milestone 7 owns both (`uncertainty.*`). A second copy here would be a second
gate that could disagree with the first. A test asserts that no field in the
adaptation configuration contains `confidence` or `quality`.

---

## 20. Signal quality

**Signal quality can never choose a direction.** It reaches the policy only
in two forms, both provenance:

- inside a Milestone 7 gate record, as a reason such as
  `signal_quality_below_gate`;
- as `AdaptationTargetSuggestion.minimum_recorded_quality`, a diagnostic field
  carried beside the decision.

There is no rule anywhere of the form "low quality -> decrease difficulty".
Poor signal quality holds; it does not become an adaptation. The
`AdaptationTargetSuggestion` schema enforces the general form of this: a
direction cannot be recorded without an ordinal state, and an ordinal state
comes only from a declared-ordinal class label or an explicitly configured
regression band.

---

## 21. Failure behaviour: invalid vs unavailable

| Condition | Behaviour |
|---|---|
| unknown class label | **raise** `AdaptationPolicyError` |
| out-of-order window | **raise** |
| current difficulty outside bounds | **raise** |
| one session's state applied to another | **raise** |
| non-finite regression estimate or interval | **raise** |
| unknown policy mode / conflict rule / experiment mode | **raise** at configuration load |
| missing target evidence | **hold**, `insufficient_evidence` |
| unknown current difficulty | **hold**, `current_state_unavailable` |
| blocked or abstained gate | **hold**, with M7's reasons preserved |
| regression target with mapping off | **hold**, `no_policy_for_target` |
| duplicate window | **hold**, `duplicate_window`, state unchanged |

Nothing turns a missing difficulty into zero, a missing target into "medium",
a missing gate into eligible, or a missing history into "no cooldown" without
recording the cold-start condition.

---

## 22. The scenario suite

`uv run python -m engagevr adaptation-demo` runs 15 deterministic scenarios.

> **These are controller tests, not participant simulations.** Each window
> state was chosen by the author to make one branch of the policy run. Nothing
> in the suite models a person, a task, a physiological process, or anyone's
> response to an adaptation.

Every evidence record is built by putting a real `AbstentionDecision` through
the real `evaluate_adaptation_gate`, so the Milestone 7 gate path is genuinely
exercised rather than stubbed. A scenario **cannot** manufacture an eligible
gate for an abstained window: the gate refuses, exactly as it does in a real
run.

| # | Scenario | Exercises |
|---|---|---|
| 1 | `stable-neutral` | deadband; every window holds |
| 2 | `persistent-increase` | one increase after the dwell requirement |
| 3 | `persistent-decrease` | one decrease after the dwell requirement |
| 4 | `single-window-spike` | one window never adapts |
| 5 | `conflicting-evidence` | opposite suggestions hold |
| 6 | `gate-blocked` | M7 blocked; M7's quality reason preserved |
| 7 | `prediction-abstained` | M7 abstained on confidence |
| 8 | `cooldown-suppression` | repeated action suppressed; spacing 7 |
| 9 | `direction-reversal` | reversal needs fresh evidence and cooldown expiry |
| 10 | `minimum-bound` | `already_at_minimum` |
| 11 | `maximum-bound` | `already_at_maximum` |
| 12 | `budget-exhausted` | proposals stop at the session budget |
| 13 | `session-change` | a new session starts cold |
| 14 | `duplicate-window` | the repeat advances no count |
| 15 | `no-usable-target` | partial evidence holds |

`--list-scenarios` prints each scenario's description and expectation.

---

## 23. Artifacts

```
artifacts/experiments/<run>/
    adaptation_policy_config.json   the resolved configuration and its fingerprint
    scenarios.json                  what each scenario exercises
    adaptation_trace.parquet        ONE ROW PER POLICY EVALUATION
    adaptation_summary.json         controller metrics and provenance
    checksums.json                  SHA-256 of every artifact above
```

The trace carries one row per evaluation, with the M7 gate decision and
reasons for both targets, both states, both suggestions, the conflict flag,
the resolved direction, persistence and cooldown before and after, current /
requested / proposed difficulty, the decision kind, the policy reasons, the
budget, the proposal id, whether a command was built, the lifecycle status,
the experiment mode, the configuration fingerprint, and the synthetic and
scientific-eligibility flags.

**The trace deliberately carries no wall-clock column.** Every value in it is
a function of the inputs, the configuration, and the initial state, so two
runs of one configuration produce byte-identical Parquet and a determinism
check is a checksum comparison rather than a tolerance. Timestamps live in the
summary, where they are provenance rather than data.

No raw media, face crop, landmark, name, email, or secret is stored. The
subject, session, and window references are the pseudonymous ones already
present in the Milestone 5-7 artifacts. `artifacts/` is gitignored.

---

## 24. Controller metrics

> **These are controller-behaviour metrics, not human-benefit metrics.** They
> count what the software did on a fixed input sequence.

Reported: evaluated windows; M7-eligible and M7-blocked windows; holds;
proposals; increases; decreases; hold-reason counts; direction reversals;
minimum proposal spacing (windows); longest same-direction streak; the
fraction of eligible windows that adapted; blocked oscillation attempts; and,
per session, the final reported difficulty and the proposal count.

`AdaptationControllerMetrics` validates that the counts reconcile: holds plus
proposals equal evaluated windows, increases plus decreases equal proposals,
eligible plus blocked equal evaluated windows, and proposals never exceed
eligible windows.

**Never reported:** improved engagement, reduced cognitive load, learning
improvement, comfort improvement, therapeutic effect, or adaptation
effectiveness. No approved study supports any of these, and none exists.

### Counterfactual software-controller comparison

The same input sequence is optionally run through a **naive controller**:
dwell 1, no cooldown, no budget. It shows only that the temporal guards
mechanically reduce how often the controller acts.

It is **not** a claim that either controller is better, safer, or more useful
for any person, and the conservative policy was not tuned to win it. Turn it
off with `--no-naive-comparison`.

---

## 25. Determinism

Given the same ordered inputs, the same configuration, and the same initial
state, the decisions and the final state are logically reproducible.

- `evaluate_policy` reads no clock and draws no random number.
- `proposal_id` is `sha256(session|window|order|direction|current|proposed|
  configuration_fingerprint)[:16]`; `command_id` is derived from it.
- `run_id` is a hash of the configuration fingerprint, the evaluation mode,
  the input sequence's identity, and the package version.
- The trace Parquet is byte-identical across runs (verified by checksum).

---

## 26. Scientific eligibility

A synthetic adaptation trace is **never** scientifically eligible, and
`AdaptationRunSummary` refuses to validate one that claims to be.
`adaptation-demo` prints `scientific_evaluation_eligible=false`.

A scientific-mode run **refuses synthetic policy inputs** outright. And even
with real input data, **scientific input eligibility is not policy
validation**: real data would make an evaluation possible, not make the rule
correct.

---

## 27. Unity

The checked-in Unity source already handles `set_difficulty` in
`AdaptationReceiver.cs`, and Milestone 8 adds no wire action, so no Unity
change was needed and none was made. Automated tests do not require Unity.

**Unity compilation and runtime validation remain pending** -- no Unity Editor
is installed in this environment, and no claim of Unity runtime behaviour is
made here.

---

## 28. Limitations

1. **No human-subject evaluation of any kind has taken place.** No adaptation
   proposed by this policy has ever been shown to a person.
2. The mapping is an engineering demonstration rule derived from an example in
   the project specification. It is not a validated interpretation of human
   state.
3. Persistence, cooldown, bounds, step, and budget are engineering defaults.
   None was selected by looking at a result and none is optimal.
4. The policy consumes estimates whose own validity is unestablished: no
   validated participant-labelled engagement or cognitive-load study exists
   (see `docs/LIMITATIONS.md`).
5. The specification's "feedback or introduce variation" and "trigger a break"
   responses are **not implemented**. The policy can only change difficulty.
6. `set_stimulus_interval` is not used, so the policy cannot vary pacing.
7. The controller has been exercised only on hand-written scenarios. It has
   never run against a live stream, a real session, or a real M7 run.
8. Time is counted in evaluation windows, not seconds. A stream whose window
   cadence differs from `windowing.model_inference_seconds` changes what the
   cooldown means in wall-clock terms.
9. The state remembers only the last window, so a repeat of an *older* window
   is detected by the ordering rule rather than by identity.
10. No adaptation has been dispatched, acknowledged, or applied by this
    milestone.

---

## 29. Pending validation

1. Design and gain institutional approval for a human-subject study of
   adaptation appropriateness, usability, and acceptability.
2. Obtain validated participant engagement and cognitive-load labels before
   any estimate feeding this policy can be evaluated at all.
3. Evaluate whether participants find the proposed adaptations appropriate,
   before any question of benefit is raised.
4. Only then consider a static-versus-adaptive comparison, and only with a
   pre-registered analysis plan.
5. Unity compilation and runtime validation of the existing `set_difficulty`
   path.
6. Live-stream integration, if and when a project requirement calls for it.

---

**Milestone 8 adaptive-environment implementation complete; human-subject
evaluation of adaptation appropriateness, usability, and benefit pending.**
