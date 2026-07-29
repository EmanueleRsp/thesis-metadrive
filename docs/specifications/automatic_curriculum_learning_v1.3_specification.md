# ScenarioNet ACL with Semantic-Arm EMA Selection and Prioritized Replay

## Metadata

- Feature: ScenarioNet automatic curriculum learning with EMA arm selection
- Specification ID: `ACL-SN-EMA-001`
- Version: `v1.3`
- Status: `APPROVED`
- Date: `2026-07-29`
- Supersedes: `docs/specifications/automatic_curriculum_learning_v1.2_specification.md`
- Related specifications: `docs/specifications/scenarionet_integration_v1.1_specification.md`, `docs/specifications/rl_baselines_v1_specification.md`
- Related ADRs: `docs/decisions/ADR-014-scenarionet-acl-learning-potential-only.md`, `docs/decisions/ADR-016-scenario-acl-vectorized-execution.md`, `docs/decisions/ADR-024-runtime-scenario-data-abort.md`, `docs/decisions/ADR-028-scenario-acl-generate-eligibility-renormalization.md`, `docs/decisions/ADR-029-acl-reward-scale-normalization.md`, `docs/decisions/ADR-030-acl-recorded-rationale-record.md`, `docs/decisions/ADR-032-acl-generate-catalog-decoupling.md`
- Authoritative: `YES`
- Source ExecPlan: `docs/implementation/automatic_curriculum_learning_v1.3_exec_plan.md` (`ACL-SN-CAT-003`)

## 1. Purpose and context

This version corrects the definition of a *fresh* catalog record, which `v1.1` and `v1.2` inherited from the
procedural-generation era of the project and which is wrong for a frozen finite catalog. It also records the
first explicit selectivity rationale for the scenario buffer.

**Changed behavior (`DEC-008`/`DEC-009`/`DEC-010`, approved 2026-07-29, `ADR-032`):**

1. **`DEC-008` — Generate eligibility no longer depends on scenario-buffer membership.** `v1.1`/`v1.2` defined a
   fresh record as one the buffer does not hold. That rule caused two defects: `FIND-001`, in which an arm whose
   whole pool was absorbed by the buffer left the curriculum permanently, and `FIND-005`, in which the bandit's
   per-arm feedback was drawn systematically from the low-LP residual of each arm, because the buffer retains the
   high-LP records by construction. Generate now excludes only quarantined (`ADR-024`) and in-flight (`ADR-016`)
   records.
2. **`DEC-009` — Per-arm coverage cycles.** Within the drawn arm, Generate samples uniformly among records not yet
   visited in that arm's current coverage cycle; an exhausted cycle increments a per-arm counter and clears the
   visited set. This is sampling without replacement within a cycle, and it exists because the catalog is finite
   and curated.
3. **`DEC-010` — `buffer_capacity` 1000 → 250.** The buffer becomes a genuinely selective active set (12.5% of
   the 2,000-record training catalog) rather than an indexed half of it.

**Relationship to the reference methods.** `DEC-008` is a *restoration*, not a deviation: Abouelazm et al. (2025)
§III-E.1 and the PLR/ACCEL family draw exploration levels i.i.d. from the level space and never exclude buffer
members, because in an unbounded parameter space redrawing a buffered level has negligible probability. Their
phrase "randomly sampling unseen scenarios" describes a property of that space, not an enforced rule; this project
turned it into one when it finitized the space into a frozen catalog. `DEC-009` and `DEC-010`, by contrast, are
declared project additions specific to a finite curated catalog, with the justification recorded in `ADR-032` and
summarized in `RAT-010`/`RAT-011`.

**Unchanged from `v1.2`:** reward-scale normalization (`REQ-007`), positive-part off-policy learning potential
(`REQ-008`), the EMA update (`REQ-001`), temperature and exploration floor (`REQ-002`), the 40/60 schedule
(`REQ-003`), the 70/30 replay mixture (`REQ-004`), LP-only usefulness with Rulebook excluded (`ADR-014`),
prohibition of mutation, and `RAT-001..RAT-009` (`ADR-030`).

## 2. Scope

### In scope

- six frozen ScenarioNet semantic arms A0–A5, as immutable partitions of the frozen training catalog;
- deterministic Generate/Replay selection;
- Generate candidate eligibility and per-arm coverage cycles (new in v1.3);
- buffer commit semantics when a Generate draw lands on a record already in the buffer (new in v1.3);
- EMA score update for the selected generated arm;
- temperature-scaled softmax with explicit exploration floor;
- 40% Generate and 60% Replay after warm-up;
- existing 70/30 usefulness/staleness replay sampling;
- per-arm reward-scale normalization of the LP fed to the rank window;
- positive-part (`max(., 0)`) TD3/SAC learning-potential formulas;
- reward-scale-normalized LP as the value stored for replay-buffer retention/eviction priority;
- checkpoint/resume persistence of scores, reward-scale estimates, buffer, coverage state, counters, and RNG;
- default use by `make run` when no curriculum override is supplied.

### Out of scope

- mutation, generated children, or writes to ScenarioDescription/source data;
- Rulebook criticality, safety rank, or margins in ACL usefulness;
- any change to the frozen catalog, its splits, or the arm assignment of a record;
- intra-arm prioritization by learning potential (the MAB prioritizes at arm level; `RAT-010`);
- ablation of `buffer_capacity` (`LIM-005`);
- variance inflation of prediction-error-based LP at critic convergence (`LIM-002`);
- changes to transition replay, reward scalarization, observation schema, or dataset splits.

## 3. Terminology and invariants

`K=6` is the number of arms. `C_i` is the immutable set of frozen training-catalog records assigned to arm `i`;
`C_i` is never modified, consumed, or reduced by curriculum activity. `Λ` is the scenario buffer, a set of
references to catalog records with `|Λ| <= buffer_capacity`. `q_i` is the EMA score of arm `i`, `p_i` its Generate
selection probability, `\widetilde{LP}` normalized learning potential, `LP_i` the raw per-episode learning
potential attributed to arm `i`, `s_i` the per-arm reward-scale EMA, and `alpha` the EMA coefficient (shared by
`q_i` and `s_i`). Scores and normalized LP are finite values in `[0,1]`. `s_i` is a finite non-negative
reward-magnitude estimate, clamped away from zero at read time by a fixed floor `s_min = 1e-3`. `eta` is the
exploration mixture coefficient and `tau` the positive softmax temperature. `Q` is the run-local quarantine set
(`ADR-024`) and `I_t` the set of scenario UIDs currently in flight in other vectorized worker slots (`ADR-016`).
`V_i` is the set of records of arm `i` already visited in that arm's current coverage cycle, and `c_i` the arm's
coverage-cycle counter.

**Generate** means selecting a catalog record through the bandit and the coverage policy; it does **not** mean a
record the buffer has never held, and it does not create or modify a scenario. **Replay** means selecting a record
already in `Λ` through the usefulness/staleness mixture. During warm-up, all selections are Generate. Eviction
from `Λ` removes only the buffer reference; the record remains in the catalog and in its arm and remains
Generate-eligible.

## 4. Inputs and prohibited information

| Input | Meaning | Range/source | Missing behavior | Policy-visible |
|---|---|---|---|---|
| arm index | selected A0–A5 | integer `[0,5]`, frozen catalog | fatal configuration error | No |
| raw LP | algorithm-specific episode learning potential | finite `>=0`, ACL §12 | fatal invalid metric | No |
| normalized LP | rank of the reward-scale-normalized LP within the shared window | finite `[0,1]` | fatal invalid metric | No |
| coverage state | per-record visited flag and per-arm cycle counter | run-local, persisted | fatal checkpoint error | No |
| RNG state | deterministic parent selector state | NumPy generator state | fatal checkpoint error | No |
| buffer record | frozen ScenarioNet identity and metrics | immutable dataset identity | replay selection error | No |

Future tracks, privileged Rulebook criticality, safety rank, evaluator metrics, mutation output, and
transition-replay priorities are prohibited from arm scores, replay ranking, coverage state, or mode selection.

## 5. Outputs

The selector emits a mode, arm (for Generate), scenario identity, and selection probability. Diagnostics expose
current scores, reward-scale estimates, probabilities, update count, mode counts, per-arm coverage-cycle counters
and within-cycle coverage fractions, and both the raw LP (`U`) and the reward-scale-normalized LP (`U_scaled`) per
committed episode. These are curriculum diagnostics/reproducibility metadata, not policy observations or training
rewards.

## 6. Functional requirements

### REQ-001: EMA arm score

For a generated episode assigned to arm `i`, update only that arm:

`q_i <- (1-alpha) q_i + alpha * normalized_LP`.

No inverse-probability correction is applied (`RAT-004`). Scores remain finite and in `[0,1]`. Replay episodes
never update `q_i`.

### REQ-002: Temperature and exploration

Before Generate sampling, compute `p_i=(1-eta) softmax(q_i/tau)+eta/K` over the arms currently eligible for
Generate per REQ-009 (`ADR-028`: an ineligible arm receives exactly probability zero, and both the softmax and the
`eta/K` floor are renormalized over the eligible subset). Require `0<eta<=1`, `tau>0`, `0<alpha<=1`; probabilities
must be finite, sum to one within numerical tolerance, and every eligible arm must satisfy
`p_i >= eta/K_eligible` (`RAT-006`). Under REQ-009 all arms are eligible except when an arm's entire pool is
simultaneously quarantined or in flight, so in normal operation the floor covers all `K=6` arms as in the
reference method.

### REQ-003: Generate/Replay schedule

While `len(Λ)<warmup_buffer_size`, select Generate. Afterwards select Generate with probability `0.40` and Replay
with probability `0.60`. Replay never creates or mutates a ScenarioDescription. If no arm is Generate-eligible per
REQ-009, the selection degrades to Replay and the event is recorded as such; if the buffer is also empty, the run
fails explicitly.

### REQ-004: Replay ranking preservation

Retain usefulness/staleness mixture `0.70/0.30`, rank exponent `1.0`, and staleness offset `1`. No Rulebook-derived
value may affect rank or replay probability. A record's staleness reference `last_seen_step` is set when the record
is inserted into `Λ` and is refreshed by **every** committed episode on that record, whether the episode was
selected through Generate or through Replay: a Generate visit is an equally valid observation of the current policy
on that scenario (`DEC-008`).

### REQ-005: Persistence and compatibility

Persist EMA scores, reward-scale estimates, update count, buffer, per-record coverage state, per-arm coverage-cycle
counters, quarantine, counters, and RNG under checkpoint schema `acl_ema_v3`. Checkpoints created under
`acl_ema_v2` (`v1.2`), `acl_ema_v1` (`v1.1`), or the cumulative-weight schema are incompatible and require explicit
restart; silent interpretation under a different schema is forbidden.

### REQ-006: Default curriculum

The root `make run` configuration selects this approved ACL profile by default. Explicit `curriculum=disabled` or
another approved override remains authoritative.

### REQ-007: Per-arm reward-scale normalization

Maintain a per-arm reward-scale EMA `s_i`, initialized to `1.0` identically for every arm and updated only on
committed Generate episodes using the same `alpha` as `q_i`:

`s_i <- (1-alpha) s_i + alpha * |episode_reward|`.

For every committed episode (Generate or Replay) assigned to arm `i`, before it enters the shared rank window,
compute:

`LP_scaled = LP_i / max(s_i, s_min)`,

using the value of `s_i` as it stands *before* that same episode's `update_reward_scale` call, so an episode cannot
normalize itself. `_normalize_learning_potential` and the shared rank window operate on `LP_scaled`, not on raw
`LP_i`. `LP_scaled` is also the value stored in `ScenarioRecord.learning_potential`/`usefulness` and therefore
governs replay-buffer retention/eviction priority (`DEC-007`). The raw `LP_i` remains unchanged only in diagnostic
logs (`RAT-001`; `live_event_context["U"]`, `buffer_events`).

### REQ-008: Positive-part off-policy learning potential

The production TD3 and SAC learning-potential computation (`agent/planners/core/lifecycle.py:
acl_learning_potential`, `acl_ready_learning_potentials`) and its documented fallback
(`curriculum/scenario_acl/usefulness.py: compute_td3_learning_potential`, `compute_sac_learning_potential`) use

`LP = mean(max(delta_t, 0))`

over the episode's collected TD residuals `delta_t`, matching the existing PPO formula `mean(max(A_t, 0))`. This
does not resolve the irreducible variance-inflation limitation shared by all three algorithms (`LIM-002`).

### REQ-009: Generate candidate eligibility and per-arm coverage cycles (new in v1.3, `DEC-008`/`DEC-009`)

A record `s` of arm `i` is **Generate-admissible** iff `s ∉ Q` and `s ∉ I_t`. Membership in `Λ` must not affect
admissibility, directly or indirectly.

Arm `i` is **Generate-eligible** iff it has at least one Generate-admissible record. REQ-002 samples the arm over
the eligible set.

Within the drawn arm `i`, the candidate set is

`F_i = { s ∈ C_i : s ∉ Q, s ∉ I_t, s ∉ V_i }`.

If `F_i` is empty, the arm's coverage cycle closes before selection: `c_i <- c_i + 1`, `V_i <- {}`, and `F_i` is
recomputed as `{ s ∈ C_i : s ∉ Q, s ∉ I_t }`. The record is then drawn **uniformly at random** from `F_i` using the
seeded selector RNG (`RAT-010`), and `V_i <- V_i ∪ {s}` is applied at selection time, not at commit time, so a
record in flight cannot be drawn twice within a cycle. A record whose episode ends in a typed data-abort
(`ADR-024`) remains marked as visited and is added to `Q`.

Coverage cycles are maintained per arm. No global cycle exists.

### REQ-010: Buffer commit semantics for Generate episodes (new in v1.3, `DEC-008`)

On commit of a Generate episode on record `s` with a valid learning potential:

- if `s ∈ Λ`, update the existing entry in place with the new `LP_scaled`, normalized usefulness, refreshed
  `last_seen_step`, incremented visit count, and refreshed diagnostics; the buffer must hold exactly one entry per
  `scenario_uid`;
- otherwise, attempt insertion under the unchanged admission rule: insert if `|Λ| < buffer_capacity`, else replace
  the lowest-`LP_scaled` entry iff the candidate's `LP_scaled` is strictly greater, else reject.

In every case the MAB is updated per REQ-001 and REQ-007, because the episode was selected through Generate. A
Generate episode landing on a buffered record must be recorded as a distinct buffer action, not as a rejected
insert. Eviction removes only the buffer reference; the evicted record stays in `C_i` and stays Generate-admissible.

## 7. Algorithmic contract

Initialization: `q_i=0.50`, `s_i=1.0`, `c_i=0`, `V_i={}` for every arm; `Λ={}`; `Q={}`. Use numerically stable
log-sum-exp softmax. The following is the normative sequential abstraction; the implementation executes it under
deterministic vectorized scheduling with `(collection_tick, worker_id, episode_id)` commit ordering (`ADR-016`),
which is where `I_t` becomes non-empty.

```
for each curriculum episode t:
    # --- mode selection ---
    if |Λ| < warmup_buffer_size: mode <- GENERATE
    else: mode <- GENERATE with probability 0.40, else REPLAY

    if mode = GENERATE:
        E_t <- { i : {s in C_i : s not in Q and s not in I_t} != {} }        # REQ-009
        if E_t = {}:
            if Λ = {}: fail explicitly
            mode <- REPLAY, record "generate_arm_exhausted"                  # REQ-003
        else:
            p <- softmax/eta-floor over E_t, zero elsewhere                  # REQ-002
            i ~ Categorical(p)
            F_i <- { s in C_i : s not in Q, s not in I_t, s not in V_i }     # REQ-009
            if F_i = {}:
                c_i <- c_i + 1 ; V_i <- {}
                F_i <- { s in C_i : s not in Q, s not in I_t }
            s ~ Uniform(F_i) ; V_i <- V_i + {s}                             # RAT-010
            roll out policy in ScenarioEnv(s) ; store valid transitions
            if typed data-abort:                                             # ADR-024
                Q <- Q + {s} ; remove s from Λ if present
                no LP, no MAB update, no buffer update
            else:
                LP_raw   <- algorithm-specific positive-part LP              # REQ-008
                LP_scaled<- LP_raw / max(s_i, s_min)                         # REQ-007, pre-update s_i
                LP_norm  <- RankNorm(LP_scaled ; H + {LP_scaled})            # RAT-001
                append LP_scaled to H, keep last W
                q_i <- (1-alpha) q_i + alpha * LP_norm                       # REQ-001
                s_i <- (1-alpha) s_i + alpha * |episode_reward|              # REQ-007
                if s in Λ: update entry in place                             # REQ-010
                else: insert, or replace argmin_{x in Λ} LP_scaled(x) if better, else reject

    if mode = REPLAY:
        s ~ P_replay over Λ, P_replay = 0.70 P_U + 0.30 P_C                  # REQ-004
        roll out policy in ScenarioEnv(s) ; store valid transitions
        if typed data-abort: Q <- Q + {s} ; remove s from Λ ; no LP, no updates
        else:
            LP_raw, LP_scaled, LP_norm as above (arm i = arm(s))
            append LP_scaled to H, keep last W
            if s not in Λ (concurrently evicted before commit):
                record "skipped_evicted_before_commit", no buffer update
            else: update entry in place with LP_scaled, LP_norm, rank, last_seen_step
            no update to q_i or s_i

    persist buffer, coverage state, quarantine, MAB state, counters, RNG
```

`P_U(x) ∝ rank(LP_scaled(x))^{-beta}` with `beta = 1.0`, ranks recomputed on every buffer mutation;
`P_C(x) ∝ max(offset, offset + t - last_seen_step(x))` with `offset = 1`. `RankNorm` averages tied ranks.

## 8. State, timing, reset, serialization

State is parent-owned and updated after an episode's LP is available. Coverage state (`V_i`, `c_i`) is updated at
selection time; buffer, MAB, and reward-scale state at commit time. Vectorized completions are committed in
deterministic `(collection_tick, worker_id, episode_id)` order. Reset clears no EMA state; a new run initializes
scores to `0.50`, reward-scale estimates to `1.0`, coverage counters to `0`, and visited sets to empty. Resume
restores all selector state including coverage state, and rejects incompatible schema/version.

## 9. Configuration

| Field | Type | Default | Range | Frozen |
|---|---|---:|---|---|
| `buffer_capacity` | int | 250 | `>0`; smaller than the smallest arm pool is recommended | YES |
| `warmup_buffer_size` | int | 100 | `>=0`, `<= buffer_capacity` | YES |
| `generate_probability` | float | 0.40 | `[0,1]` | YES |
| `exploit_probability` | float | 0.60 | `[0,1]`, complementary | YES |
| `mab.num_arms` | int | 6 | exactly 6 | YES |
| `mab.update_method` | string | `ema` | exactly `ema` | YES |
| `mab.alpha` | float | 0.10 | `(0,1]` | YES |
| `mab.initial_score` | float | 0.50 | `[0,1]` | YES |
| `mab.eta` | float | 0.20 | `(0,1]` | YES |
| `mab.temperature` | float | 0.50 | `>0` | YES |
| `mab.use_importance_correction` | bool | false | false only | YES |
| `mab.use_target_mab` | bool | false | bool; optional delayed target scores | YES |
| `mab.target_sync_interval` | int | 1 | `>0`; used when target MAB is enabled | YES |
| replay weights | floats | 0.70/0.30 | nonnegative, sum 1 | YES |

`buffer_capacity = 250` is 12.5% of the 2,000-record training catalog and yields a warm-up fill ratio
`ρ = warmup_buffer_size / buffer_capacity = 0.40` (`DEC-010`, `RAT-011`). Coverage cycles have no configurable
parameter: they are structural, not tunable. The per-arm reward-scale estimator (`s_i`, initial `1.0`, floor
`1e-3`) is not independently configurable. Invalid values fail before learner construction. The target-MAB path is
retained but disabled by default. Legacy cumulative fields (`weight_clip_*`, `initial_weight_decay`) are rejected
or migrated explicitly, never silently ignored.

## 10. Errors and diagnostics

Non-finite LP, invalid probability, wrong arm count, mutation configuration, incompatible checkpoint, missing
frozen record, or a Generate draw on a record reported eligible but absent from the catalog is fatal. An empty
eligible-arm set with a non-empty buffer is a recorded degradation, not an error; with an empty buffer it is fatal.
Each committed episode records arm, pre/post score, raw LP (`U`), reward-scale-normalized LP (`U_scaled`),
rank-normalized LP (`U_norm`), probability, buffer action, and update count. Per-arm coverage-cycle transitions are
logged once per transition, as are Generate-eligibility transitions. Rulebook diagnostics remain diagnostic-only.

## 11. Reproducibility and compatibility

The specification ID/version, resolved YAML, dataset/catalog hashes, seed, and selector state are persisted.
Checkpoints from `v1.1`/`v1.2` cannot resume under `v1.3`; the default migration policy is restart. The observable
RNG call order changes relative to `v1.2` (`ADR-032`), so `v1.1`/`v1.2` seeds do not reproduce their runs under
this version. Experiments completed under `v1.1`/`v1.2` remain reproducible under their own recorded configuration,
but curriculum-arm-selection claims derived from them are not reliable (`FIND-001`, `FIND-005`); their
policy-performance results are unaffected.

## 12. Acceptance criteria

### AC-001: EMA bounded update

Given `q=[0.5]*6`, arm A0 receives normalized LP `1` repeatedly and A1–A5 receive `0`; scores move toward those
values, remain in `[0,1]`, and only the selected arm changes per update. (REQ-001)

### AC-002: Probability contract

The resulting probabilities are finite, sum to one, satisfy the exploration floor over the eligible set, and with
A0 higher than the others have `p(A0)>1/6`. (REQ-002)

### AC-003: Warm-up and 40/60 schedule

Before 100 records every selection is Generate; after warm-up a seeded long sequence uses both modes with
deterministic replay and empirical proportions consistent with 0.40/0.60 within the test tolerance. (REQ-003)

### AC-004: Replay and Rulebook separation

Changing Rulebook diagnostics while holding LP fixed leaves score, rank, replacement, replay probability, and MAB
feedback unchanged. (REQ-004)

### AC-005: Checkpoint compatibility

`acl_ema_v3` state round-trips exactly (scores, target scores, reward-scale estimates, update count, coverage
state); `acl_ema_v2`, `acl_ema_v1`, and cumulative-schema checkpoints are rejected with an explicit incompatibility
error. (REQ-005)

### AC-006: Default composition

Composing the root config without a curriculum override resolves to the `v1.3` ACL profile; explicit
disabled/alternative profiles still resolve as requested. (REQ-006)

### AC-007: Reward-scale normalization removes the cross-arm scale confound

Given two arms with identical raw LP distributions but different reward-magnitude distributions, after each arm's
reward-scale estimate has converged, the reward-scale-normalized LP distributions are approximately equal, and the
shared rank window assigns the two arms statistically indistinguishable ranks. Before any reward-scale update
(`t=0`), normalization is a no-op for every arm. (REQ-007)

### AC-008: Positive-part TD3/SAC learning potential

Given a residual sequence with mixed signs, `compute_td3_learning_potential`, `compute_sac_learning_potential`, and
the production path `acl_learning_potential` equal `mean(max(delta,0))`, not `mean(|delta|)`; an all-negative
residual sequence yields `LP=0`. (REQ-008)

### AC-009: Generate candidates are invariant to buffer contents

For a fixed arm, quarantine set, in-flight set, and coverage state, the Generate candidate set and the
Generate-eligible arm set are identical whether the buffer is empty, holds an arbitrary subset of the arm, or holds
the arm's entire pool. (REQ-009, `FIND-005`)

### AC-010: An absorbed arm keeps receiving Generate draws and EMA updates

With the buffer holding every record of one arm and that arm having the highest score, a seeded selection sequence
still produces Generate selections on that arm, and its EMA score changes. (REQ-009, regression for `FIND-001` at
its root)

### AC-011: Coverage cycle completeness and restart

Repeated Generate draws on one arm visit every admissible record exactly once before any record repeats; the draw
immediately after exhaustion increments the arm's cycle counter, clears its visited set, and the counters of the
other arms are unchanged. Quarantined and in-flight records are excluded without blocking cycle closure.
(REQ-009)

### AC-012: Generate on a buffered record updates in place

A committed Generate episode on a record already in the buffer produces exactly one buffer entry for that
`scenario_uid`, with refreshed `LP_scaled`, refreshed `last_seen_step`, and an incremented visit count; the
recorded buffer action is the dedicated update action, not `rejected`; and the MAB score and reward-scale estimate
of the arm are updated. (REQ-010, REQ-004)

## 13. Required validation categories

Required: nominal/boundary, invalid configuration, numerical stability, update order, reset, deterministic seed,
checkpoint/resume, no privileged information, replay separation, coverage-cycle completeness and restart,
buffer-independence of Generate candidates, data-abort interaction with coverage state, regression, and end-to-end
smoke. Mutation validation remains required as a prohibited feature. The reward-scale estimator retains its
`v1.2` validation set. Distributional and lexicographic LP variants are covered by existing algorithm-specific
tests.

## 14. Traceability

| Requirement | Acceptance | Source/decision |
|---|---|---|
| REQ-001/002 | AC-001/002 | Proposal §2–6; project adaptation; `ADR-028` |
| REQ-003 | AC-003 | Proposal §7/9; ADR-014 |
| REQ-004 | AC-004/012 | ACL v1.1 §28.3; ADR-014; `DEC-008` |
| REQ-005 | AC-005 | ACL v1.1 §28.4; ADR-016; `DEC-009` |
| REQ-006 | AC-006 | User request 2026-07-23 |
| REQ-007 | AC-007 | `FIND-004`; `DEC-006`; `DEC-007`; `ADR-029` |
| REQ-008 | AC-008 | `FIND-002`; `DEC-006`; `ADR-029` |
| REQ-009 | AC-009/010/011 | `FIND-001`; `FIND-005`; `DEC-008`; `DEC-009`; `ADR-032` |
| REQ-010 | AC-012 | `DEC-008`; `ADR-032` |

## 15. Recorded rationales, open decisions, and known limitations

### 15.1 Recorded rationales

`RAT-001..RAT-009` are retained unchanged from `v1.2` §15.1 and `ADR-030` (rank normalization; shared rank window;
bounded EMA instead of cumulative accumulation; no inverse-probability correction; disabled target MAB; the
temperature term; rejection of full paper fidelity; semantic-tier initialization; benign Replay contamination of
the rank window). Two rationales are added in `v1.3`:

- **`RAT-010`** — The draw within an arm is uniform over the current-cycle candidate set, not ordered by visit
  count or staleness. Within a cycle every record is visited at most once, so visit counts differ by at most one
  and any such ordering degenerates to deterministic round-robin, adding a second prioritization mechanism that
  competes with the MAB for attribution of curriculum effects while adding no behavior. Uniform draws also keep the
  per-arm LP samples exchangeable, which is what `REQ-001`'s EMA and `RAT-001`'s rank normalization assume.
- **`RAT-011`** — `buffer_capacity` must satisfy `|Λ| << |C_train|` for the buffer's admission rule ("beats the
  current buffer minimum") to be a meaningful selectivity filter. At the `v1.2` value of 1000 out of 2000 the rule
  degraded to "above the median" and the Explore and Exploit distributions largely overlapped. `250` restores
  selectivity (12.5% of the catalog) and brings the warm-up fill ratio to `ρ=0.40`, close to Abouelazm's `ρ=0.5`.
  The value `1000` had no independent provenance in this project: it was inherited from a paper where it denoted a
  vanishing fraction of an unbounded level space.

### 15.2 Recorded findings

- **`FIND-001`** (2026-07-27) — Frozen-pool exhaustion under the buffer-membership exclusion: an arm whose entire
  pool the buffer had absorbed could no longer be drawn for Generate, so its EMA froze while it held the highest
  selection probability. Observed live in `td3_sb3/seed_0/20260726_055617` (`A4_vru`, bit-identical score from MAB
  update 1420 to 3118), and present at 332/224/207 of 333 records in three other completed runs. Mitigated by
  `ADR-028` (eligibility renormalization); root cause removed by `DEC-008`.
- **`FIND-005`** (2026-07-29) — Systematic downward bias of the bandit's per-arm feedback. Because the buffer
  retains an arm's high-LP records by construction, excluding buffer members from the Generate draw made each arm's
  eligible residual converge toward records already rejected as low-LP, so `q_i` estimated
  `E[LP | arm i, LP below the eviction threshold]` instead of `E[LP | arm i]`. The bias is proportional to how
  productive the arm is, applies to all arms rather than only exhausted ones, and is active from early training. It
  invalidates the premise of `REQ-001`. Removed by `DEC-008`. This finding is analytical, derived from the code and
  the buffer admission rule; it is not quantified against a run, and no measurement is available that isolates its
  magnitude in the completed experiments.

### 15.3 Closed decisions

`DEC-001`, `DEC-004`, `DEC-005`, `DEC-006`, and `DEC-007` are retained from `v1.2` §15.2. New in `v1.3`:

| ID | Question | Resolution | Status |
|---|---|---|---|
| `DEC-008` | Should Generate eligibility depend on scenario-buffer membership? | No. Buffer membership is removed from the candidate filter; only quarantine and in-flight exclusions remain. A Generate draw may land on a buffered record and updates it in place (`REQ-010`). Restores the exploration semantics of the reference methods (`ADR-032` "Where the rule comes from") | APPROVED 2026-07-29 |
| `DEC-009` | How is catalog coverage maintained once `DEC-008` allows repeats? | Per-arm coverage cycles (sampling without replacement within a cycle), justified by the finite curated catalog: at `n=m` draws, i.i.d. sampling reaches 63% distinct records against 100% with cycles. Declared as a project addition to the reference methods, not part of the correction | APPROVED 2026-07-29 |
| `DEC-010` | Does `buffer_capacity` remain 1000 after `DEC-008`? | No: 250. After `DEC-008` capacity no longer affects MAB liveness or feedback bias, so it is chosen purely for replay selectivity (`RAT-011`) | APPROVED 2026-07-29 |
| `DEC-011` | How is the unreachable sequential ACL path treated? | Aligned with the new semantics rather than deleted, so the two paths cannot diverge silently; removal tracked separately | APPROVED 2026-07-29 |

### 15.4 Known limitations (not resolved by this version)

- **`LIM-002`** — Variance inflation of prediction-error LP at critic convergence, unchanged from `v1.2`: at
  `E[delta]=0`, `mean(max(delta,0))` is proportional to `sigma(delta)`, so a noisy-but-unlearnable scenario still
  yields nonzero LP. The principled fix is a learning-progress signal; out of scope (no ablation budget,
  architectural change).
- **`LIM-003`** — `MEAS-003` showed a nearly static arm-probability ordering under `v1.1`. Whether `v1.2`'s
  reward-scale normalization and `v1.3`'s `DEC-008` change this materially is an empirical question for the re-run
  experiments, not established here. `DEC-008` removes one identified cause of static ordering (`FIND-005`), which
  makes this the primary hypothesis to check in the new runs.
- **`LIM-004`** — The `commit_event` buffer-record construction path is a private closure inside the vectorized
  training loop with no isolated-testing harness; `v1.2`'s `LP_scaled` wiring was verified by code reading only.
  `REQ-010` adds behavior to the same closure and inherits the same limitation for the branch selection, although
  `AC-012` covers the buffer-level contract.
- **`LIM-005`** — `buffer_capacity = 250` is a reasoned project choice (`RAT-011`), not an ablated one. No
  experiment in this thesis isolates the effect of buffer capacity on final policy performance, and none is
  planned; the claim supported by the evidence is that the buffer is selective, not that 250 is optimal.
- **`LIM-006`** — `FIND-005` is established analytically, not measured. The completed `v1.1`/`v1.2` runs are not
  re-analyzable for it, because the quantity that would demonstrate it — the LP distribution of the records
  excluded from the Generate draw — was never logged for records outside the buffer.

## 16. References

`automatic_curriculum_learning_v1.2_specification.md`; `automatic_curriculum_learning_v1.1_specification.md`;
`automatic_curriculum_learning_v1_specification.md` §§12–13/28; ADR-014; ADR-016; ADR-024; ADR-028; ADR-029;
ADR-030; ADR-032; `docs/implementation/automatic_curriculum_learning_v1.3_exec_plan.md`;
`docs/implementation/automatic_curriculum_learning_v1.2_exec_plan.md`;
`docs/implementation/scenario_acl_exhausted_arm_mab_starvation_exec_plan.md`; Peng et al., 2024, IROS §III-A/III-B;
Abouelazm et al., 2025, IV §III-E (exploration/exploitation phases, Table I parameters `N=1000`, `ρ=0.5`,
`D=0.8`, `ω=0.7`); both in `docs/papers/curriculum learning/`; stable-baselines3 and repository deterministic
vector execution contracts.

## 17. Implementation handoff checklist

- [x] Scope, exclusions, formulas, state, configuration, diagnostics, compatibility, and acceptance criteria
      defined.
- [x] Material decisions (`DEC-008`, `DEC-009`, `DEC-010`, `DEC-011`) approved by the user on 2026-07-29 and
      recorded in `ADR-032`.
- [x] Canonical filename and `Authoritative: YES` set after approval.
- [ ] Implementation complete: `selection.py`, `catalog_state.py`, `driver.py`, `mab.py`, `buffer.py`, config;
      mandatory tests, lint, and format-check pass (ExecPlan §14).

## 18. Approval record

- Approved by: user
- Approval date: 2026-07-29
- Approval evidence: session of 2026-07-29 — the user stated that specifications are a decision record to be
  corrected when wrong, accepted the cost of redoing the in-progress runs ("Non mi interessa se poi devo rifare le
  run"), reviewed the paper-fidelity analysis separating `DEC-008` (restoration of the reference exploration
  semantics) from `DEC-009` (finite-catalog addition) and the revised `DEC-010`, and instructed "procedi"
- Repository path: `docs/specifications/automatic_curriculum_learning_v1.3_specification.md`
- Project index updated: YES
