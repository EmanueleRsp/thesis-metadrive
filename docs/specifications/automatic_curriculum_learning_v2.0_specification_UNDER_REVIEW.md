# ScenarioNet ACL with Outcome-Based Windowed Learning Progress

## Metadata

- Feature: ScenarioNet automatic curriculum learning, arm-level teacher
- Specification ID: `ACL-SN-EMA-001`
- Version: `v2.0`
- Status: `UNDER_REVIEW`
- Date: `2026-09-01` (revision 3; revision 2 dated `2026-07-31`, revision 1 `2026-07-30`)
- Supersedes: `docs/specifications/automatic_curriculum_learning_v1.3_specification.md`
- Related specifications: `docs/specifications/scenarionet_integration_v1.1_specification.md`,
  `docs/specifications/rulebook_v5.1_specification.md`,
  `docs/specifications/rl_baselines_v1_specification.md`,
  `docs/specifications/evaluation_protocol_v1.0_specification.md`
- Related ADRs: `docs/decisions/ADR-014-scenarionet-acl-learning-potential-only.md`,
  `docs/decisions/ADR-016-scenario-acl-vectorized-execution.md`,
  `docs/decisions/ADR-024-runtime-scenario-data-abort.md`,
  `docs/decisions/ADR-028-scenario-acl-generate-eligibility-renormalization.md`,
  `docs/decisions/ADR-029-acl-reward-scale-normalization.md`,
  `docs/decisions/ADR-030-acl-recorded-rationale-record.md`,
  `docs/decisions/ADR-032-acl-generate-catalog-decoupling.md`,
  `docs/decisions/ADR-063-rss-longitudinal-demoted-to-diagnostic.md`,
  `docs/decisions/ADR-066-wrongway-deleted-dashed-line-retained.md`,
  `docs/decisions/ADR-068-speed-limit-normative-sub-rule.md`,
  `docs/decisions/ADR-070-at-fault-gating-for-interaction-sub-rules.md`,
  `docs/decisions/ADR-071-r1-at-fault-classification-and-truncation.md`,
  `docs/decisions/ADR-072-relaxable-lane-rules-below-mission-progress.md`,
  `docs/decisions/ADR-076-l6-progress-rate-below-relaxable-lane-compliance.md`
- Supporting evidence: `docs/audits/acl_v2_teacher_power_analysis_2026-07-31/` (synthetic power and
  closed-loop dynamics analysis, source `FIND-010`; window-size power sweep, source `FIND-012`)
- Authoritative: `NO` (review candidate; `v1.3` remains authoritative until this document is approved)

## 1. Purpose And Context

### 1.1 The defect this version corrects

Every ACL version from `v1` to `v1.3` derives the teacher's per-arm feedback from a prediction-error
*learning potential*: `mean(max(GAE, 0))` for PPO and `mean(max(delta, 0))` for TD3/SAC. `ADR-029`
established by measurement that this signal is proportional to the reward magnitude of the arm that
produced it (Spearman correlation between median LP and median `|reward|`, per arm: `+0.771`, `+0.886`,
`+0.829`, `+0.943` across four runs), and `v1.2` added per-arm reward-scale normalization (`REQ-007`) to
remove the *scale* component of that confound.

The residual defect was recorded but not fixed, as `LIM-002` in `v1.2`/`v1.3` §15.4: at critic convergence
`E[delta] = 0`, so `mean(max(delta, 0))` is proportional to `sigma(delta)`. An arm that is structurally
noisier — more stochastic near-miss interactions, more discrete outcome switching — therefore keeps a
systematically higher feedback value with no learning in progress. `ADR-029` named the principled fix
explicitly and placed it out of scope: "The principled fix is a learning-progress signal (slope of return
per arm over repeated visits)".

Two observations recorded during the review of this version confirm that the residual defect is the
dominant one and that scale normalization did not remove it (`FIND-006`, `FIND-007`, §15.2).

### 1.2 What this version does

`v2.0` replaces the prediction-error family entirely with a **windowed, ordinal, outcome-based learning
progress** signal. For each arm the teacher compares the `H` most recent Generate episodes against the `H`
preceding ones, on quantities produced by the environment and the Rulebook rather than by the learner, and
asks a single question: *are the recent outcomes better than the older ones?* The comparison is the
Vargha–Delaney common-language effect size, whose expectation is exactly `0.5` when the two windows come
from the same distribution regardless of that distribution's variance. That property is what removes
`LIM-002` at its root, and it has been verified by measurement over a `250x` range of noise scale
(`FIND-010`, `AC-201`) rather than asserted analytically.

The resulting feedback is then attenuated by an explicit **learnability gate** so that an arm that is
already solved and an arm that is entirely out of reach both converge to the neutral value, and only arms
of intermediate current difficulty receive a bonus.

### 1.3 Second, independent motivation: algorithm independence

Under `v1.3` PPO's teacher feedback is computed from GAE advantages and TD3/SAC's from TD residuals. These
are different quantities with different units, different variance, and different sensitivity to critic
quality, so the curriculum is *not the same curriculum* across the algorithm arm of the experimental
design. Under `v2.0` the teacher consumes only `success`, `route_completion`, and Rulebook level
costs. The identical curriculum contract therefore applies to the scalar, lexicographic, distributional,
and lexicographic+distributional reward settings and to every planner. This makes the reward-setting ×
curriculum comparison of `EVAL-PROTOCOL` v1.0 interpretable in a way it is not under `v1.3`.

### 1.4 Scientific sources and project adaptations, kept distinct

**From the literature.** Using the learner's measurable progress on a task as the reward of a teaching
multi-armed bandit is the learning-progress curriculum of Graves et al. (2017) and the Teacher-Student
Curriculum Learning of Matiisen et al. (2017). Estimating "did the distribution improve" by a
non-parametric two-sample ordinal comparison, with `0.5` as the no-effect value, is the Vargha–Delaney
`A` measure (Vargha and Delaney, 2000), equivalent to the normalized Mann–Whitney statistic. Steering a
curriculum toward tasks of intermediate current competence rather than toward the best or worst tasks is
the competence-based curriculum principle, in the sense of PORTAL (2024) and SITP (Nesterova et al., 2023).
Combining a curricular priority with an explicit staleness term in a replay buffer over levels is the PLR
construction (Jiang et al., 2021).

**Project adaptations, declared as such.** None of these sources defines its progress measure over
Rulebook level costs, none uses a lexicographic priority ordering over safety dimensions to select
*which* dimension the progress is measured on, and none operates over a frozen finite curated scenario
catalog partitioned into six semantic arms. The specific combination — priority-ordered dimension
selection with a calibrated neutrality band, a Goldilocks gate on per-dimension violation statistics, and an
arm-balanced recency buffer over a finite catalog — is an original construction of this thesis. It is not
presented as a result taken from any of the cited papers. In particular, `v2.0`'s scenario buffer is **no
longer a prioritized level replay buffer** in the PLR sense: its non-staleness term is the arm's current
curricular probability, not a per-level score. Only the staleness term retains PLR parentage.

### 1.5 What this version does not claim

This specification does not claim, and no evidence in this repository establishes, that `v2.0` accelerates
convergence or improves final policy performance relative to `v1.3` or to `curriculum=disabled`. The
claims it does support are stated in §15.5 and are properties of the signal, not of the outcome. Whether
the curriculum helps is an experimental question, to be answered under `EVAL-PROTOCOL` v1.0's seed and
statistics protocol and not by any single run.

**Stated in advance, on measured evidence (`FIND-013`, and its status under `FIND-014`).** On every
completed run available when revision 2 was written, the policy's driving competence did not improve while
the scalar reward did. A learning-progress curriculum of any family — this one, TSCL, Graves — measures
improvement, and with no improvement to measure it correctly returns neutral and samples uniformly. That
prediction was made against a reward configuration that **no longer exists**: `RULEBOOK-V5.1` was written
in response to the same defect, was approved 2026-08-14 and amended 2026-08-20, and its implementation was
verified 2026-09-01. Under the specification measured then, the human expert scored a mean episode return
of `-203.35` with `46.55 %` of expert episodes below standstill; under `RULEBOOK-V5.1` the same 1100-record
Waymo panel scores the expert at `+70.70` with `3.36 %`. **`FIND-013`'s prediction of uniform sampling is
therefore obsolete rather than falsified**: it was correct for the reward it measured, and no measurement
exists for the reward now in production. Whether a policy improves under `RULEBOOK-V5.1` is an open
empirical question, and it is the question the `SNR` measurement of `LIM-201` now depends on.

### 1.6 What revision 3 changes

Revision 3 changes **no mechanism**. The ordinal comparison, the permutation band, the gate, the feedback,
the bandit, the buffer, the calibration and the persistence contract are those of revision 2, and every
measured result behind them (`FIND-010`, `FIND-012`) is untouched, because none of them depends on which
Rulebook produced the episodic key.

What changes is the **rulebook this specification consumes**. Revision 2 was written against the three
macro rules `R1`/`R2`/`R3` of Rulebook v4.7–v4.9. `RULEBOOK-V5.1` replaced that grouping with six ordered
levels, and four of its decisions reach this contract directly: `ADR-071` (at-fault classification and
truncation of not-at-fault contacts), `ADR-070` (interaction sub-rules inapplicable at standstill),
`ADR-063` (`rss` longitudinal removed from the channel), and `ADR-072` (the relaxable lane rules moved
*below* mission progress as a level of their own). The teacher therefore gains a fifth dimension and loses
the assumption that the level composition of revision 2 still exists. `REQ-001`, `REQ-002`, `REQ-005`,
`REQ-006`, §3.2, §4, §7, §10 and the corresponding acceptance criteria are re-based accordingly, and
`DEC-206`/`DEC-207` record the two choices the re-base forces. `FIND-014` records the re-base itself and
`FIND-015` a source asymmetry it exposes.

## 2. Scope

### In Scope

- six frozen ScenarioNet semantic arms `A0_simple_low_traffic` … `A5_critical_mixed`, as immutable
  partitions of the frozen training catalog;
- per-arm, per-dimension sliding observation windows over committed Generate episodes;
- the episodic outcome statistics derived from `RULEBOOK-V5.1` levels L1, L2, L3 and L5 and from the
  task outcome;
- the ordinal progress measure, its exact conditional permutation neutrality band, and the
  priority-ordered dimension selection;
- the learnability gate and the resulting bounded arm feedback;
- the EMA arm score update, temperature-scaled softmax, and exploration floor (unchanged mechanics);
- the balanced-coverage calibration phase and the Replay activation condition;
- Generate candidate eligibility and per-arm coverage cycles (unchanged from `v1.3` `REQ-009`);
- arm-balanced scenario-buffer admission and eviction, and the replay sampling mixture;
- checkpoint/resume persistence of windows, counters, scores, buffer, coverage state, and RNG;
- retention of the `v1.3` prediction-error learning potential as a **diagnostic-only** channel;
- default use by `make run` when no curriculum override is supplied.

### Out Of Scope

- mutation, generated children, or any write to `ScenarioDescription` or source data;
- any change to the frozen catalog, its splits, or the arm assignment of a record;
- intra-arm prioritization: the teacher prioritizes at arm level only;
- per-record learnability or per-record progress estimation (`RAT-206`);
- `RULEBOOK-V5.1` levels L4 (`mission_progress`) and L6 (`progress_rate`) as separate teacher
  dimensions (`RAT-203`);
- changes to reward scalarization, the observation schema, transition replay, or dataset splits;
- ablation of `H`, `buffer_capacity`, `band_level`, or `tau` (`LIM-204`);
- removing the visit-frequency coupling of window-based progress measures (`LIM-203`);
- any claim about final policy performance (§1.5).

### Optional Or Deferred

- reinstating a per-record curricular score once a per-record progress estimator with acceptable variance
  exists; deferred, no design in this version;
- replacing the exploration floor with a formal non-stationary bandit guarantee; deferred.

## 3. Terminology, Assumptions, And Preconditions

### 3.1 Symbols

| Symbol | Meaning | Domain |
|---|---|---|
| `K` | number of arms | exactly `6` |
| `i` | arm index | `[0, 5]` |
| `C_i` | immutable set of frozen training-catalog records of arm `i` | never modified by curriculum activity |
| `Λ` | scenario buffer, references into the catalog | `|Λ| <= buffer_capacity` |
| `n_i` | number of buffer records belonging to arm `i` | `>= 0` |
| `n_i*` | balanced target occupancy of arm `i` in `Λ` | `REQ-010` |
| `H` | window half-size | `20`, frozen (§9, `RAT-202`) |
| `d` | teacher dimension | `L1`, `L2`, `L3`, `T`, `L5` |
| `x_{d,e}` | episodic key of dimension `d` for episode `e`, **lower is better** | `REQ-002` |
| `N^{app}_{k,e}` | number of steps of episode `e` at which level `k` is applicable | `>= 0` |
| `N^{viol}_{k,e}` | number of applicable steps of episode `e` with `c_k(t) > 0` | `<= N^{app}_{k,e}` |
| `v_{1,e}` | episode-level **at-fault** collision incidence of L1 (`ADR-071`) | `{0, 1}` |
| `F_{k,e}` | violated-applicable-step fraction of level `k` | `[0, 1]` |
| `W_i^d` | FIFO of the last `2H` valid observations of dimension `d` on arm `i` | ordered oldest→newest |
| `O_i^d`, `R_i^d` | older half and recent half of `W_i^d` | `H` entries each |
| `G_{i,d}` | ordinal progress measure | `[0, 1]`, neutral `0.5` |
| `p^{perm}_{i,d}` | exact conditional two-sided permutation p-value of `G_{i,d}` | `(0, 1]` |
| `D_{i,d}` | learnability gate | `[0, 1]` |
| `d*` | selected dimension for this update | `L1|L2|L3|T|L5` or none |
| `A_i` | arm feedback | `[0, 1]`, neutral `0.5` |
| `q_i` | EMA arm score | `[0, 1]` |
| `p_i` | Generate selection probability of arm `i` | `[0, 1]`, `sum = 1` |
| `alpha`, `tau`, `eta` | EMA coefficient, softmax temperature, exploration mixture | `0.10`, `0.50`, `0.20` |
| `Q` | run-local quarantine set (`ADR-024`) | run-local |
| `I_t` | scenario UIDs in flight in other worker slots (`ADR-016`) | run-local |
| `V_i`, `c_i` | per-arm coverage visited set and cycle counter (`v1.3` `REQ-009`) | run-local |

`Generate` means selecting a catalog record through the bandit and the coverage policy. `Replay` means
selecting a record already in `Λ`. Eviction from `Λ` removes only the buffer reference; the record remains
in `C_i` and remains Generate-admissible.

A **valid Generate episode** for arm `i` is a committed Generate episode on a record of arm `i` that did
not end in a typed data-abort (`ADR-024`) and that executed at least one environment step. Only valid
Generate episodes update windows, counters, arm scores, and the buffer.

### 3.2 Assumptions

| Assumption | Source | Classification | Failure behavior |
|---|---|---|---|
| Rulebook v2 exposes, per step, the six level margins in `info["rule_reward_vector"]` with names in `info["rule_metadata"]["rule_names"]`, in the `MACRO_RULE_ORDER` of `RULEBOOK-V5.1` §3 | `rulebook/v2/wrapper.py:285`, `types.py:99` | verified repository property, runtime-validated | fatal if absent while ACL is enabled |
| Rulebook v2 exposes, per step, per-level applicability in `info["rule_components"][name]["applicable"]`, where `name` is the level's channel value (`collision_safety`, `interaction_risk`, `non_relaxable_compliance`, `relaxable_lane_compliance`) | `rulebook/v2/wrapper.py:295`, `aggregation.py:219-231` | verified repository property, runtime-validated | fatal if absent while ACL is enabled |
| For the cost levels the margin equals `-cost` with `cost` in `[0, 1]` | `aggregation.py` `RulebookResult.margins` | guaranteed upstream | fatal if a margin falls outside `[-1, 0]` |
| L1 (`collision_safety`) is `applicable=False` on every step without a new **at-fault** contact, and its cost is then `0`. Under `ADR-071` this covers two distinct cases: no contact at all, and a contact classified not-at-fault, which is charged nothing and **truncates** the episode | `components/collision.py:112`, `:224`, `components/collision_fault.py` | verified repository property | drives `REQ-002` L1 handling and `DEC-207` |
| L3 (`non_relaxable_compliance`) is applicable on every step, because `offroad` carries `applicable=True` unconditionally | `components/road.py:138` | verified repository property | L3 falls back to the unobserved branch of `REQ-002` if violated |
| L2 (`interaction_risk`) sub-rules `ttc`, `clearance` and `rss_lateral` are applicable only when a relevant actor exists **and** `v_ego > 5e-02 m/s` (`ADR-070`); `rss` longitudinal is evaluated but carries `contributes_to_channel=False` (`ADR-063`) and is therefore not part of this level | `components/rss_lateral.py`, `clearance.py`, `ttc.py`, `registry.py` | verified repository property | drives `REQ-002` L2 handling |
| L5 (`relaxable_lane_compliance`) is applicable when at least one of `solid_line`, `wrong_carriageway`, `dashed_line` is applicable, and aggregates by normalized sum over a **fixed** denominator of 3 | `components/road.py:322`, `:200`, `:402`, `aggregation.py:63` `aggregate_sum_component` | verified repository property | drives `REQ-002` L5 handling |
| `success` and `route_completion` are available per episode | `agent/agent.py` episode accumulation | guaranteed upstream | fatal if missing |
| Vectorized completions are committed in deterministic `(collection_tick, worker_id, episode_id)` order | `ADR-016` | guaranteed upstream | non-deterministic resume; fatal on ordering violation |

**Explicit research limitation, not an assumption:** the six arms do not form a total difficulty order, so
no particular arm ordering over time is required, expected, or validated.

## 4. Inputs And Prohibited Information

| Input | Meaning/type | Shape/unit | Range | Source/validity | Missing-data behavior | Policy-visible |
|---|---|---|---|---|---|---|
| arm index | selected arm | scalar int | `[0, 5]` | frozen catalog | fatal configuration error | NO |
| per-step level margins | `m_k(t)` for L1, L2, L3, L5 | 4 of the 6 emitted floats per step | `[-1, 0]` | Rulebook v2 monitor | fatal | NO |
| per-step level applicability | `applicable` flag per level | 4 of the 6 emitted bools per step | `{0,1}` | Rulebook v2 monitor | fatal | NO |
| not-at-fault truncation flag | episode ended by a not-at-fault contact (`ADR-071`) | bool | `{0,1}` | Rulebook v2 / env | fatal if absent | NO |
| episode success | terminal task outcome | bool | `{0,1}` | environment | fatal | NO |
| episode route completion | max route completion in episode | float | `[0, 1]` | environment | fatal | NO |
| data-abort flag | typed non-evaluability | bool | `{0,1}` | `ADR-024` | episode discarded from all ACL state | NO |
| coverage state | per-record visited flag, per-arm cycle counter | run-local | — | persisted artifact | fatal checkpoint error | NO |
| window state | per-arm per-dimension FIFO of episodic keys | `<= 2H` entries | — | persisted artifact | fatal checkpoint error | NO |
| RNG state | deterministic parent selector state | NumPy generator state | — | persisted artifact | fatal checkpoint error | NO |

**Prohibited from every teacher quantity** (`q_i`, `A_i`, `G`, `D`, dimension selection, buffer admission,
buffer eviction, replay probability, mode selection): future information of any kind; privileged simulator
state; the evaluation panel, its metrics, or any validation/test split quantity; mutation output;
transition-replay priorities; **the prediction-error learning potential and any quantity derived from it**
(`REQ-013` makes it diagnostic-only); the episode's own reward magnitude; `RULEBOOK-V5.1` L4 or L6 as an
independent dimension; the diagnostic, non-normative sub-rules of `RULEBOOK-V5.1` §3 (`rss` longitudinal,
not-at-fault collision cost), which do not reach a level channel and must not reach a teacher quantity
either.

## 5. Outputs

| Output | Meaning | Range | Consumer | Classification |
|---|---|---|---|---|
| mode | `GENERATE` or `REPLAY` | enum | driver | control |
| arm, scenario identity, selection probability | selection provenance | — | driver, logs | control + reproducibility metadata |
| `q_i`, `p_i` | arm scores and probabilities | `[0,1]` | teacher, monitor | diagnostic-only |
| `G_{i,d}`, `p^{perm}_{i,d}`, `D_{i,d}`, `d*`, `A_i` | teacher internals of each update | `[0,1]` / enum | logs | diagnostic-only |
| `C_{k,e}`, `v_{1,e}`, `N^{app}_{k,e}`, `N^{viol}_{k,e}`, `F_{k,e}` | per-episode Rulebook statistics, for `k` in `{L1, L2, L3, L5}` | `[0,1]` / counts | logs, windows, gate | training signal (teacher) + diagnostic |
| `U`, `U_scaled`, `U_norm` | prediction-error learning potential and its `v1.3` transforms | `>= 0` / `[0,1]` | logs only | **diagnostic-only** (`REQ-013`) |
| per-arm coverage cycle id, within-cycle coverage fraction | catalog coverage | counts | logs | diagnostic-only |

No output of this specification is a policy observation or a training reward.

## 6. Functional Requirements

### REQ-001: Teacher dimensions and their priority order

- Required observable behavior: the teacher operates on exactly five dimensions, in this fixed priority
  order:

  | position | dimension | Rulebook channel |
  |---|---|---|
  | 1 | `L1` | `collision_safety` |
  | 2 | `L2` | `interaction_risk` |
  | 3 | `L3` | `non_relaxable_compliance` |
  | 4 | `T` | task outcome, standing in the position of L4 `mission_progress` |
  | 5 | `L5` | `relaxable_lane_compliance` |

  The order is the `RULEBOOK-V5.1` §3 level order, with the task outcome occupying L4's position.
- Applicability: always.
- Invariants: the order is frozen and is not configurable.
- Interactions: `REQ-005` selects one dimension per update from this order.
- Rationale: `RAT-203` and `DEC-206`. `T` stands in L4's position because `route_completion` is the
  episodic counterpart of L4's per-step signed advance, so admitting both would place two measurements of
  one quantity adjacent in the priority order. `L5` is last because `ADR-072` placed the relaxable lane
  rules **below** mission progress: a curriculum that consulted lane relaxation before progress would
  invert the ordering the rulebook exists to assert. `L6` (`progress_rate`) is excluded for the same
  reason as L4 — it is a rate over the quantity `T` already measures — and additionally because under
  `ADR-076` it carries a time preference, not a competence.

### REQ-002: Episodic outcome statistics

For every valid Generate episode `e` on arm `i`, compute the following, using only steps of that episode.

**L1 (`collision_safety`).**

`C_{1,e} = max over all steps t of c_1(t)`, where `c_1(t) = max(0, -m_1(t))`.

L1 is **always observed**: every valid Generate episode appends an L1 observation. The maximum is taken
over all steps, not only applicable ones.

- Justification, verified against this repository: L1's applicability predicate is "a new **at-fault**
  contact occurred at this step" (`components/collision.py:112`, `:224`), and its cost is `0` whenever it
  is not applicable. For L1, "not applicable" therefore means *satisfied*, not *unobserved*. Averaging
  over applicable steps would make the L1 window contain only collision episodes, which would measure
  injury severity conditional on a crash instead of collision behavior, would prevent a nearly-solved arm
  from ever refilling its L1 window, and would prevent its gate from ever closing. Recorded as `FIND-008`,
  and re-verified against `RULEBOOK-V5.1` in `FIND-014`.
- `ADR-071` consequence, normative: a contact classified **not at fault** is charged nothing and is
  therefore invisible in `C_{1,e}`, which is the intended semantics — the teacher measures the arm's
  at-fault collision behavior, not its exposure to other agents' errors. The same contact **truncates**
  the episode, which is not invisible: see `DEC-207` and the `T` dimension below.

**L2, L3 and L5 (`interaction_risk`, `non_relaxable_compliance`, `relaxable_lane_compliance`).**

Let `N^{app}_{k,e}` be the number of steps at which level `k` is applicable. If `N^{app}_{k,e} = 0`
the dimension is **unobserved for this episode** and its FIFO is not updated — it is not silently recorded
as zero cost. Otherwise:

`C_{k,e} = ( sum over applicable t of c_k(t) ) / N^{app}_{k,e}`,  `c_k(t) = max(0, -m_k(t))`.

**Gate statistics.** Let `N^{viol}_{k,e}` be the number of applicable steps of episode `e` with
`c_k(t) > 0`.

- **L1 — episode-level at-fault collision incidence.** `v_{1,e} = 1[C_{1,e} > 0]`, i.e. `1` iff the
  episode contains at least one **at-fault** contact. This is the quantity consumed by the L1 gate.
- **L2, L3 and L5 — violated-applicable-step fraction.**
  `F_{k,e} = N^{viol}_{k,e} / N^{app}_{k,e}` when `N^{app}_{k,e} > 0`, else undefined and not recorded.
  This is the quantity consumed by the L2, L3 and L5 gates.
- `v_{k,e}` for `k` in `{L2, L3, L5}` is **not defined by this version** and must not appear in any teacher
  quantity. `F_{1,e}` may be logged as a diagnostic but is not consumed.

- Justification for the L1 / L2–L3–L5 asymmetry, verified against this repository (`FIND-011`): L1's
  applicability predicate fires only on a new at-fault contact, so a collision is a rare, discrete,
  episode-defining event and its incidence spans `[0, 1]` meaningfully across a run. L2's sub-rules
  (`ttc`, `clearance`, `rss_lateral`) are applicable whenever a relevant actor exists and the ego is
  moving, and are violated at *some* step in essentially every episode with traffic, so an episode-level
  indicator would sit at `v_{2,e} = 1` almost surely, driving `vbar -> 1` and closing the L2 gate
  permanently. L3 and L5 have the same problem to a lesser degree. The step fraction `F` is graded, spans
  `[0, 1]`, and measures *how much of the episode* was spent in violation, which is the quantity whose
  intermediate value actually identifies an arm of intermediate difficulty.
- `ADR-070` interaction, recorded because it improves the statistic: under `RULEBOOK-V5.1` the three L2
  sub-rules are inapplicable while `v_ego <= 5e-02 m/s`. Those steps therefore leave `N^{app}_{2,e}`
  entirely, instead of entering it as violated steps in which the ego was standing still. `F_{2,e}` under
  this rulebook is thus a cleaner measure of interaction behavior than the same formula would have been
  under v4.9, where §4.10.3 of `RULEBOOK-V5.0` measured `68.2 %` of the stopped-ego penalty mass as a
  controlled-invariance failure.

**Task dimension `T`.** `T_e = (success_e, route_completion_e)`. Observed on every valid Generate episode
**except** one truncated by a not-at-fault contact (`ADR-071`), which is excluded from the `T` window only
and recorded as such (`DEC-207`).

- Invariants: `C_{k,e}` in `[0, 1]`; `F_{k,e}` in `[0, 1]`; `route_completion_e` in `[0, 1]`; all values
  finite.
- Edge cases: an episode with zero environment steps is not a valid Generate episode and updates nothing.
  An episode truncated by a not-at-fault contact remains a valid Generate episode: it counts toward
  `N^{gen}_i`, updates the `L1`, `L2`, `L3` and `L5` windows, and enters the buffer normally.
- Failure behavior: a non-finite value, a margin outside `[-1, 0]` for a consumed level, or a missing
  applicability flag is fatal.

### REQ-003: Comparable keys and the sliding windows

Each dimension defines an episodic key with **lower is better**:

- `x_{k,e} = C_{k,e}` for `k` in `{L1, L2, L3, L5}`;
- `x_{T,e} = (-success_e, -route_completion_e)`, compared lexicographically.

For every arm `i` and dimension `d` the teacher maintains a FIFO `W_i^d` of at most `2H` keys, ordered
oldest to newest, appended only by valid Generate episodes and only for dimensions observed in that
episode. `O_i^d` is the older half and `R_i^d` the newer half once `|W_i^d| = 2H`.

- Replay episodes never update any window. They are selected by the teacher itself and are therefore not
  unbiased observations of the arm's distribution (`RAT-205`).
- Dimension `d` is **available** for arm `i` iff `|W_i^d| = 2H`. `L1` becomes available for every arm at
  the end of calibration (`REQ-008`), and `T` at the same time unless not-at-fault truncations have
  removed observations from its window (`DEC-207`); `L2`, `L3` and `L5` become available when enough
  episodes have observed them. `L3` is expected to track `L1` closely, because `offroad` is applicable at
  every step (§3.2).
- Windows are updated at commit time, in the deterministic `(collection_tick, worker_id, episode_id)`
  order of `ADR-016`.

### REQ-004: Ordinal progress measure

For each available dimension `d` of arm `i`:

`G_{i,d} = (1 / H^2) * sum over x in R_i^d, y in O_i^d of psi(x, y)`,
`psi(x, y) = 1 if x < y, 0.5 if x = y, 0 if x > y`, under the dimension's key order.

- Invariants: `G` in `[0, 1]`; `G > 0.5` means the recent window is better; `E[G] = 0.5` whenever the two
  windows are exchangeable, independently of the distribution's variance.
- Equivalent form used by the implementation: with pooled midranks ascending in the key over the `2H`
  entries and `W_R` the rank sum of `R_i^d`, `G_{i,d} = 1 - (W_R - H(H+1)/2) / H^2`.
- This is the Vargha–Delaney `A` measure of the recent window against the older window.

### REQ-005: Neutrality band and priority-ordered dimension selection

For each available dimension the teacher computes the **exact conditional two-sided permutation p-value**
of `G_{i,d}` under exchangeability of the `2H` pooled observations:

`p^{perm}_{i,d} = P( |W'_R - E| >= |W_R - E| )` over all `C(2H, H)` equally likely assignments of the
pooled midranks to the two windows, with `E = H(2H+1)/2`.

- The distribution is computed exactly, by dynamic programming over the observed tie groups, in doubled
  integer rank units so that no floating-point tolerance is required. It consumes no random numbers.
- A dimension **fires** iff `p^{perm}_{i,d} <= band_level`, with `band_level = 0.05` applied **per
  dimension, uncorrected**.
- `d*` is the **first dimension in the `REQ-001` priority order** `(L1, L2, L3, T, L5)` that fires.
- If no dimension fires, or if fewer than two dimensions are available, `d*` is undefined.

**No multiplicity correction is applied.** Revision 1 of this document specified Holm's step-down
correction at family-wise level `0.05`. It is removed, for a measured reason recorded as `FIND-010`:

- Holm is **not** the factor that determines whether the curriculum functions. In the closed-loop
  simulation the peak arm probability during an arm's learning phase was `0.24–0.27` without Holm against
  `0.23–0.26` with it at the operating point, and at low signal **neither** configuration departed
  materially from uniform. Any claim that multiplicity correction blocked the curriculum is false and is
  falsifiable with the recorded script.
- What Holm suppresses is a doubling of the false-fire rate on a stationary arm (`0.032–0.035` uncorrected
  against `0.006–0.007` corrected). Those false fires are **symmetric in sign** under exchangeability, so
  they cost variance and not bias: the mean feedback of the stationary noisy arm was `0.498–0.500` under
  both configurations, with `sd(p_i) ~ 0.006`.
- The correction is therefore removed because it buys roughly a factor two of sensitivity at a bias cost
  measured to be negligible, not because it obstructed the teacher. Its removal also eliminates a
  dependence of the effective per-dimension level on *how many dimensions happen to be available*, which
  under Holm made an arm with only `T` and `L1` available systematically more responsive than an arm with
  all five dimensions available — an artifact with no scientific justification.

**Required framing, normative for all derived documentation.** This procedure is re-executed after every
valid Generate episode on the arm, over windows that overlap by `2H - 1` observations. It is therefore a
**calibrated neutrality band derived from the data's own permutation null**, not a hypothesis test with a
run-level type-I error guarantee. No document produced from this specification may state or imply a
significance claim at level `0.05` over a run, nor use the word "significant" for a fired dimension. Its
statistical role is that false fires are symmetric in sign under exchangeability, so they inflate the
variance of `A_i` without biasing its expectation away from `0.5` — a property measured, not assumed
(`FIND-010`). Recorded as `RAT-204` and `LIM-202`.

### REQ-006: Learnability gate

For the selected dimension `d*` of arm `i`, computed over the **recent** window `R_i^{d*}` only:

In every case the gate has the form `D = 4 * g * (1 - g)` and differs only in the statistic `g`:

- if `d* = L1`: `g = vbar = (1/H) * sum over e in R_i^{L1} of v_{1,e}` — the fraction of recent episodes
  containing at least one at-fault contact;
- if `d*` in `{L2, L3, L5}`: `g = Fbar = (1/H) * sum over e in R_i^{d*} of F_{d*,e}` — the mean
  per-episode fraction of applicable steps spent in violation;
- if `d* = T`: `g = rbar = (1/H) * sum over e in R_i^T of route_completion_e`.

- Invariants: `D` in `[0, 1]`; `D = 0` at `g in {0, 1}`; `D = 1` at `g = 0.5`. The factor `4` is analytic
  normalization of the maximum to `1` and is not a tuned parameter.
- Justification for not using the mean episodic cost `C`, verified against this repository: L1's cost is a
  MAIS3+F injury probability and L3's is dominated by a graded off-road cost, so mean episodic costs occupy
  roughly `[0, 0.1]` in practice and `4C(1-C)` would never approach `1`. This would systematically
  attenuate the Rulebook dimensions relative to `T`, whose gate spans `[0, 1]` naturally. Recorded as
  `FIND-009`.
- Justification for the L1 / L2–L3–L5 asymmetry: see `REQ-002` and `FIND-011`. An episode-level indicator
  on L2 would saturate at `1` in essentially every episode with traffic and close the L2 gate permanently;
  the step fraction is graded and does not.
- The comparison `G` deliberately keeps the fine-grained cost `C` on every Rulebook dimension, which is
  sensitive to partial improvement; only the gate uses the coarser incidence or fraction. `G` and `D`
  therefore consume different statistics of the same episode by design.
- `route_completion` rather than success rate is used for the `T` gate because a success rate of `0` early
  in training would close the gate on the task dimension for every arm simultaneously, whereas
  `route_completion` is informative from the first episode.

### REQ-007: Arm feedback

`A_i = 0.5 + D_{i,d*} * (G_{i,d*} - 0.5)` when `d*` is defined; `A_i = 0.5` otherwise.

- Invariants: `A_i` in `[0, 1]`; `A_i = 0.5` exactly when no dimension fires, when the gate is closed, or
  when `G = 0.5`.
- Regression (`G < 0.5`) yields `A_i < 0.5`. The absolute value is **not** taken: doing so would convert
  symmetric fluctuation back into positive priority, which is the failure mode this version removes.
  Protection against catastrophic forgetting is delegated to the exploration floor (`REQ-009`) and to the
  replay staleness term (`REQ-011`).

### REQ-008: Balanced calibration and Replay activation

The run begins in a **calibration phase**. Let `N^{gen}_i` be the number of valid Generate episodes on arm
`i`. While `min_i N^{gen}_i < 2H`:

- every selection is Generate; Replay is disabled regardless of buffer size;
- the arm is chosen deterministically as the one with the smallest `N^{gen}_i`, ties broken by a single
  draw from the seeded selector RNG over the tied arms in ascending arm index;
- the intra-arm draw, coverage cycles, and quarantine/in-flight exclusion are exactly as in `REQ-012`;
- `q_i` remains at `initial_score` and no arm feedback is computed;
- windows and the scenario buffer are populated normally.

Calibration therefore costs `K * 2H = 240` valid Generate episodes. Replay becomes eligible when **both**
`|Λ| >= warmup_buffer_size` and `min_i N^{gen}_i >= 2H` hold; the second condition normally implies the
first. A record that enters quarantine during calibration does not count toward `N^{gen}_i`, and the
selector continues drawing other admissible records of the same arm.

- Failure behavior: if an arm cannot reach `2H` valid Generate episodes because its whole pool is
  quarantined, the run fails explicitly rather than activating the curriculum on incomplete windows.

### REQ-009: EMA arm score, temperature, and exploration

After calibration, on every valid Generate episode of arm `i`:

`q_i <- (1 - alpha) * q_i + alpha * A_i`, with `alpha = 0.10` and `q_i` initialized to `0.50`.

Only the arm that produced the episode is updated. Replay episodes never update `q_i`. No
inverse-probability correction is applied (`RAT-201`).

Before each Generate arm draw: `p_i = (1 - eta) * softmax(q_i / tau) + eta / K_eligible`, computed over
the arms currently Generate-eligible per `REQ-012`, with an ineligible arm receiving exactly probability
zero and both terms renormalized over the eligible subset (`ADR-028`). Require `tau > 0`,
`0 < eta <= 1`, `0 < alpha <= 1`; probabilities must be finite, sum to one within numerical tolerance, and
satisfy `p_i >= eta / K_eligible` for every eligible arm.

- Invariant of the neutral state: when every `A_i = 0.5`, every `q_i` converges to `0.5` and `p` converges
  to the uniform distribution `1/6`. **The curriculum's default state is uniform sampling**, and it departs
  from uniform only on measured, gated evidence.

### REQ-010: Scenario buffer as an arm-balanced recency memory

`buffer_capacity = 250`. The balanced target occupancy is
`n_i* = floor(250 / 6) + (1 if i < 250 mod 6 else 0)`, i.e. `[42, 42, 42, 42, 41, 41]` for `A0…A5`.

Each record carries two independent timestamps: `last_generate_step`, refreshed **only** by a committed
valid Generate episode on that record, and `last_seen_step`, refreshed by **every** committed episode on
that record, Generate or Replay.

On commit of a valid Generate episode on record `s` of arm `i`:

1. if `s` is in `Λ`: update the entry in place — refresh both timestamps, increment the visit count,
   refresh the episodic statistics — and perform no eviction. The recorded buffer action is
   `generate_on_buffered_record`.
2. else if `|Λ| < buffer_capacity`: insert `s`.
3. else if `n_i < n_i*`: choose the donor arm `j` maximizing `(n_j - n_j*)`, ties by ascending arm index;
   evict that arm's record with the smallest `last_generate_step`, ties by ascending `scenario_uid`; then
   insert `s`.
4. else: evict arm `i`'s own record with the smallest `last_generate_step`, ties by ascending
   `scenario_uid`; then insert `s`.

Replay commits never insert and never evict; they refresh `last_seen_step`, the visit count, and
diagnostics only. Eviction removes only the buffer reference. A record removed by `ADR-024` quarantine is
removed from `Λ` without replacement.

- Invariants: at most one entry per `scenario_uid`; `|n_i - n_i*| <= 1` once `|Λ| = buffer_capacity`;
  `|Λ| <= buffer_capacity` always.
- **Removed behavior:** admission and eviction no longer depend on any learning-potential or usefulness
  value, and the buffer no longer maintains a usefulness rank. `v1.3` `REQ-004`'s
  `P_U ∝ rank(LP_scaled)^{-beta}` is removed together with `beta`.
- Justification: retaining the highest-`LP` records reproduced the diagnosed bias through a second channel
  covering 60% of post-calibration training time (`ADR-029` `DEC-007` established this channel's
  importance while fixing it only for scale). With no per-record progress estimator available
  (`RAT-206`), a balanced recency memory is the construction that adds no per-record bias.

### REQ-011: Replay sampling

`P_replay(r) = 0.70 * P_progress(r) + 0.30 * P_stale(r)`.

- `P_progress(r) = p^{full}_{arm(r)} / n_{arm(r)}`, where `p^{full}` is the Generate distribution of
  `REQ-009` computed **without** the eligibility mask, i.e. over all six arms. Arms with `n_i = 0`
  contribute nothing and the vector is renormalized to sum to one. Dividing by `n_i` guarantees
  `sum over r in arm i of P_progress(r) = p^{full}_i`, so buffer occupancy cannot distort the arm-level
  exposure the curriculum intends.
- `P_stale(r) = (1 + age(r)) / sum_j (1 + age(j))` with `age(r) = current_episode_id - last_seen_step(r)`,
  and staleness offset `1` unchanged from `v1.3`.
- Both terms are finite and the mixture is renormalized to sum to one.

Consequently, after calibration the fraction of exposure governed by the arm-level curriculum is
`0.40 + 0.60 * 0.70 = 0.82` of `p^{full}_i`, and `0.18` is governed by staleness.

### REQ-012: Generate candidate eligibility and per-arm coverage cycles

Unchanged from `v1.3` `REQ-009` (`DEC-008`/`DEC-009`, `ADR-032`), restated here because it is part of this
contract:

A record `s` of arm `i` is Generate-admissible iff `s not in Q` and `s not in I_t`. Membership in `Λ` must
not affect admissibility. Arm `i` is Generate-eligible iff it has at least one admissible record. Within
the drawn arm the candidate set is `F_i = { s in C_i : s not in Q, s not in I_t, s not in V_i }`; if `F_i`
is empty the cycle closes (`c_i <- c_i + 1`, `V_i <- {}`) and `F_i` is recomputed without the `V_i`
exclusion. The record is drawn uniformly from `F_i` and `V_i <- V_i + {s}` is applied at selection time.
Coverage cycles are per arm; no global cycle exists.

### REQ-013: Prediction-error learning potential retained as diagnostic-only

The `v1.3` computation chain — `acl_learning_potential` / `acl_ready_learning_potentials`, the documented
fallbacks `compute_ppo_learning_potential`, `compute_td3_learning_potential`,
`compute_sac_learning_potential`, the per-arm reward-scale EMA `s_i`, and the shared rank window — is
**retained and continues to run**, and its outputs `U`, `U_scaled`, `U_norm` continue to be logged per
committed episode.

- It must not influence `q_i`, `A_i`, `G`, `D`, dimension selection, buffer admission, buffer eviction,
  replay probability, or mode selection. Any such influence is a contract violation.
- Purpose: it makes the `v1.3` and `v2.0` signals measurable **within the same run**, so the claim that the
  new signal is not proportional to reward magnitude can be established by measurement rather than
  analytically. This directly addresses `v1.3` `LIM-006`, where the corresponding evidence for `FIND-005`
  could not be reconstructed retrospectively.
- Cost: the computation already exists and is already executed; no additional simulator or learner work is
  introduced.

### REQ-014: Persistence, schema, and resume

Persist, under checkpoint schema `acl_progress_v1`: arm scores `q_i`, per-arm valid Generate counts
`N^{gen}_i`, the calibration-complete flag, every window `W_i^d` with its keys, the per-arm reward-scale
estimates and the shared rank window (diagnostic channel, `REQ-013`), the update count, the scenario
buffer with both timestamps per record, the per-arm coverage state (`acl_coverage_v1`, unchanged), the
quarantine set, mode counters, and the selector RNG state.

Checkpoints written under `acl_ema_v3`, `acl_ema_v2`, `acl_ema_v1`, or the cumulative-weight schema are
incompatible and are rejected with an explicit error naming this specification. Silent reinterpretation
under a different schema is forbidden. The default migration policy is restart.

## 7. Mathematical And Algorithmic Contract

Initialization: `q_i = 0.50`, `N^{gen}_i = 0`, `W_i^d = []`, `c_i = 0`, `V_i = {}`, `Λ = {}`, `Q = {}`.

```
for each curriculum episode t:

    # --- mode selection ---
    calibrated <- ( min_i N_gen[i] >= 2H )
    if not calibrated:
        mode <- GENERATE
    elif |Λ| < warmup_buffer_size or not use_replay:
        mode <- GENERATE
    else:
        mode <- GENERATE with probability 0.40, else REPLAY

    if mode = GENERATE:
        E_t <- { i : { s in C_i : s not in Q and s not in I_t } != {} }        # REQ-012
        if E_t = {}:
            if Λ = {}: fail explicitly
            mode <- REPLAY, record "generate_arm_exhausted"
        elif not calibrated:
            i <- argmin over E_t of N_gen[i], ties broken by one seeded draw   # REQ-008
        else:
            p <- softmax/eta-floor over E_t, zero elsewhere                    # REQ-009
            i ~ Categorical(p)

        if mode = GENERATE:
            F_i <- { s in C_i : s not in Q, s not in I_t, s not in V_i }       # REQ-012
            if F_i = {}: c_i <- c_i + 1 ; V_i <- {} ; recompute F_i
            s ~ Uniform(F_i) ; V_i <- V_i + {s}
            roll out policy in ScenarioEnv(s) ; store valid transitions

            if typed data-abort:                                              # ADR-024
                Q <- Q + {s} ; remove s from Λ if present
                no statistics, no window, no counter, no score, no buffer change
            else:
                compute C_L1e, C_L2e, C_L3e, C_L5e, v_1e,                     # REQ-002
                        F_L2e, F_L3e, F_L5e, T_e (unless not-at-fault truncation)
                append observed keys to W_i^d                                 # REQ-003
                N_gen[i] <- N_gen[i] + 1
                compute U, U_scaled, U_norm and log them                      # REQ-013

                if calibrated_before_this_episode:
                    Avail <- { d : |W_i^d| = 2H }
                    for d in Avail:
                        G[d]    <- VarghaDelaney(R_i^d, O_i^d)                # REQ-004
                        pperm[d]<- ExactPermutationTwoSided(R_i^d, O_i^d)     # REQ-005
                    Fire <- { d in Avail : pperm[d] <= band_level }            # REQ-005
                    d*   <- first d in (L1,L2,L3,T,L5) with d in Fire
                    A_i  <- 0.5 + D(i,d*) * (G[d*] - 0.5)  if d* defined       # REQ-006/007
                            else 0.5
                    q_i  <- 0.9 q_i + 0.1 A_i                                 # REQ-009
                else:
                    A_i, d* undefined ; q_i unchanged                          # REQ-008

                apply buffer admission/eviction for s                          # REQ-010

    if mode = REPLAY:
        r ~ P_replay over Λ                                                    # REQ-011
        roll out policy in ScenarioEnv(record(r)) ; store valid transitions
        if typed data-abort: Q <- Q + {r} ; remove r from Λ ; no updates
        else:
            compute U, U_scaled, U_norm and log them                           # REQ-013
            refresh last_seen_step, visit count, diagnostics on r              # REQ-010
            no window update, no q_i update, no insertion, no eviction

    persist windows, counters, buffer, coverage state, quarantine, scores, RNG # REQ-014
```

**Exact permutation null (normative construction).** Assign midranks ascending in the dimension key over
the `2H` pooled entries, doubled to integers so that midranks of tied groups remain exact. Let `S` be the
doubled rank sum of the recent window and `E2 = H(2H+1)` its exact null expectation in doubled units
(`820` at `H = 20`). Enumerate, by dynamic programming over the tie groups, the counts `f[S]` of `H`-subsets
of the pooled multiset achieving each doubled rank sum; `sum_S f[S] = C(2H, H) = 137846528820` at `H = 20`.
Then

`p^{perm} = ( sum over S with |S - E2| >= |S_obs - E2| of f[S] ) / C(2H, H)`.

All comparisons are integer, so the result is exact and reproducible across platforms, and the procedure
consumes no random numbers. When both windows are constant and equal — the common case of an arm with no
violations of a rule — `f` is concentrated on a single `S` and `p^{perm} = 1`, which correctly reports "no
detectable change".

**Cost and representation, normative.** The recursion is over the achievable rank-sum range, of size
`O(H^2)`, and **never** over the `C(2H, H)` assignments, which are only the normalizing constant. The
counts must be accumulated in exact 64-bit integers; `C(40, 20) = 137846528820 < 2^63`, and the
implementation must reject any `H` for which `C(2H, H)` would not fit, rather than silently switching to
floating point or to arbitrary-precision arithmetic. The reference implementation recorded in
`docs/audits/acl_v2_teacher_power_analysis_2026-07-31/` was validated at `H = 10` against exhaustive
enumeration of all `184756` splits (six trials, agreement to `1e-12`, including tie-heavy inputs) and
measured at `0.059 ms` per call with a cold cache and `0.019 ms` warm. Caching keyed on the tie structure
and the observed rank sum is permitted and does not alter the result.

**Numerical contract.** Every quantity is finite. `C in [0,1]`, `v_1 in {0,1}`, `F in [0,1]`,
`route_completion in [0,1]`,
`G in [0,1]`, `D in [0,1]`, `A in [0,1]`, `q in [0,1]`. Softmax uses a numerically stable log-sum-exp.
Probabilities are renormalized once and validated to sum to one within `1e-9`. Any NaN or infinity in an
input or an intermediate is fatal.

## 8. Applicability, State, And Timing

State is parent-owned. Coverage state (`V_i`, `c_i`) is updated at **selection** time so that a record in
flight cannot be drawn twice in a cycle. Windows, counters, arm scores, and the buffer are updated at
**commit** time, in the deterministic `(collection_tick, worker_id, episode_id)` order of `ADR-016`.

Termination and truncation are not distinguished by the teacher: `success` is a task outcome and a
time-limit truncation simply yields `success = 0` with whatever `route_completion` was reached. A typed
data-abort (`ADR-024`) is neither: it produces no teacher observation at all.

Reset semantics: a new run initializes as in §7. Resume restores every persisted field of `REQ-014` and
rejects an incompatible schema. Calibration state is persisted, so a resume during calibration continues
calibration rather than restarting it.

## 9. Configuration

| Field | Type | Default | Valid range | Meaning | Required | Frozen |
|---|---|---:|---|---|---|---|
| `buffer_capacity` | int | 250 | `>= K`, `>0` | scenario buffer size | YES | YES |
| `warmup_buffer_size` | int | 100 | `>=0`, `<= buffer_capacity` | secondary Replay gate | YES | YES |
| `generate_probability` | float | 0.40 | `[0,1]` | post-calibration Generate share | YES | YES |
| `exploit_probability` | float | 0.60 | `[0,1]`, complementary | post-calibration Replay share | YES | YES |
| `progress.window_half_size` | int | 20 | `>= 5`, `C(2H,H) < 2^63` | `H` | YES | YES |
| `progress.band_level` | float | 0.05 | `(0, 0.5)` | per-dimension neutrality band, uncorrected | YES | YES |
| `mab.num_arms` | int | 6 | exactly 6 | arms | YES | YES |
| `mab.update_method` | string | `ema` | exactly `ema` | score update | YES | YES |
| `mab.alpha` | float | 0.10 | `(0,1]` | EMA coefficient | YES | YES |
| `mab.initial_score` | float | 0.50 | `[0,1]` | neutral initialization | YES | YES |
| `mab.eta` | float | 0.20 | `(0,1]` | exploration mixture | YES | YES |
| `mab.temperature` | float | 0.50 | `>0` | softmax temperature | YES | YES |
| `mab.feedback` | string | `windowed_outcome_progress` | exactly that value | feedback family | YES | YES |
| `mab.use_importance_correction` | bool | false | false only | — | YES | YES |
| `mab.use_target_mab` | bool | false | bool | optional delayed target scores | NO | YES |
| `replay_sampling.omega` | float | 0.70 | `[0,1]` | progress/staleness mixture | YES | YES |
| `replay_sampling.staleness_offset` | int | 1 | `>=1` | staleness offset | YES | YES |
| `recent_window_size` | int | 100 | `>0` | rank window of the **diagnostic** LP channel (`REQ-013`) | YES | YES |

**Rejected legacy fields.** `replay_sampling.beta`, `mab.feedback: rank_normalized_learning_potential`, and
`progress.multiplicity_correction` are removed from the contract. A configuration containing any of them
fails validation before learner construction with an explicit message naming this specification; they are
never silently ignored. The legacy cumulative fields `weight_clip_*` and `initial_weight_decay` remain
rejected as in `v1.3`.

### 9.1 Choice of `H`

`H = 20`. Revision 1 of this document specified `H = 10` on a parsimony argument (reuse of the EMA horizon
`1/alpha`). That argument is retained as an observation but is **not** the basis for the value, because
the EMA horizon governs score smoothing and not the resolution of the progress measurement. The basis is
the following, recorded as `RAT-202` and sourced from `FIND-010`.

Define the arm's operating signal-to-noise ratio

`SNR_i = ( mean key over O_i^d - mean key over R_i^d ) / sd( key within arm i )`,

i.e. the policy improvement on arm `i` across the `H`-episode gap between the two window centres, in units
of the between-episode (equivalently between-scenario) standard deviation inside the arm. The true effect
size is `A = Phi(SNR / sqrt(2))`.

`H` enters the design **twice**: it sets the sample size of each window, so power grows as `sqrt(H)`, and
it sets the separation between window centres, so for a locally constant improvement rate per arm-episode
`SNR` itself grows approximately linearly in `H`. Both effects were measured directly rather than
extrapolated (`FIND-012`, `window_size_sweep.py`, `4000` repetitions per cell, band level `0.05`).

**Operating case** — an arm improving at a fixed rate per arm-episode, so a wider window spans
proportionally more training. `SNR` is quoted at `H = 10` and scales with `H`:

| `SNR` at `H=10` | `H=10` | `H=15` | `H=20` | `H=30` |
|---:|---:|---:|---:|---:|
| `0.00` | `0.040` | `0.048` | `0.046` | `0.050` |
| `0.50` | `0.157` | `0.481` | `0.854` | `1.000` |
| `0.75` | `0.306` | `0.815` | `0.993` | `1.000` |
| `1.00` | `0.515` | `0.970` | `1.000` | `1.000` |
| `1.25` | `0.713` | `0.997` | `1.000` | `1.000` |

**Control** — the same sweep with `SNR` held fixed as `H` grows, isolating the sample-size contribution
alone. This is not the operating situation and is reported to show how much of the gain comes from each
mechanism: at `SNR = 0.75` the detection rate moves `0.323 -> 0.464 -> 0.613 -> 0.797` for
`H = 10, 15, 20, 30`, i.e. sample size alone accounts for well under half of the gain in the table above.

The two error directions are strongly asymmetric:

- `H` too small — the band under-fires, no dimension is selected, `A_i` stays at `0.5`, and the curriculum
  degenerates to **uniform sampling**, which is exactly the `curriculum=disabled` baseline. The failure is
  benign and self-announcing in the logs.
- `H` too large — the signal remains correct but becomes **stale**: calibration lengthens and the window
  keeps reporting improvement after the arm has already plateaued, delaying handover to the next arm.

`H = 20` is chosen because it reaches `>= 0.85` detection across the entire plausible signal range,
including the pessimistic `SNR = 0.5` case at which `H = 10` achieves only `0.157`, at a calibration cost
of `K * 2H = 240` valid Generate episodes and no meaningful computational cost (§7). `H = 30` is not
chosen: it buys a further improvement only in the single lowest-signal row, at `50%` more calibration and
proportionally more handover latency. If the deferred measurement of `LIM-201` returns
`SNR at H=10 < 0.5`, `H = 30` is the indicated revision.

**Stated limitation of this table.** The operating case assumes the improvement rate is locally constant
across the window span. An arm that plateaus *inside* the window does not have its gap widened by
increasing `H`, so the operating rows are an upper bound and the control rows a lower bound on the real
gain. The truth lies between them, and the control rows alone already justify `H = 20` over `H = 10`. The
false-fire rate under exchangeability stays at or below the nominal `0.05` for every `H` tested
(`0.038`, `0.042`, `0.049`, `0.049`), so none of this gain is bought with inflated false positives.

### 9.2 Identified tuning lever, deliberately not exercised

`mab.temperature` (`tau = 0.50`) is the single parameter that controls how sharply a given feedback spread
translates into selection probability. Because `A_i` lives in a narrow band around `0.5`, `tau` is the
correct lever if a run shows the curriculum to be under-concentrated, and it changes no statistical
property of the estimator. It is recorded here so that adjusting it later does not require reopening this
document, and it is **left at the `v1.3` value in this revision** — no run has yet demonstrated a need,
and `FIND-007` is if anything evidence of prior over-concentration. Any change must be recorded and must
precede the runs it affects (§11).

## 10. Errors, Logging, And Diagnostics

**Fatal:** non-finite statistic, margin outside `[-1, 0]` for a consumed level, missing Rulebook
applicability or margin data while ACL is enabled, invalid probability vector, wrong arm count, mutation
configuration, incompatible checkpoint schema, rejected legacy configuration field, a Generate draw on a
record reported eligible but absent from the catalog, an arm unable to complete calibration, and a Replay
selection with an empty buffer and no Generate-eligible arm.

**Recorded degradation, not an error:** an empty Generate-eligible arm set with a non-empty buffer; a
concurrent eviction of a record before its in-flight Replay episode commits
(`skipped_evicted_before_commit`, unchanged from `v1.3`).

**Required per-committed-episode log fields:** mode, arm, scenario UID, coverage cycle id,
`C_{k,e}`/`N^{app}_{k,e}`/`N^{viol}_{k,e}`/`F_{k,e}` for `k` in `{L1, L2, L3, L5}`, `v_{1,e}`, `success`,
`route_completion`, the not-at-fault truncation flag and whether the episode entered the `T` window
(`DEC-207`), buffer action, and the diagnostic `U`, `U_scaled`, `U_norm`. The **global step index** of the
commit must be logged alongside the arm and episode counters, so that the `SNR` and `LIM-203` diagnostics
can be computed offline without re-instrumenting a run.

**Required per-teacher-update log fields:** available dimensions, `G_{i,d}` and `p^{perm}_{i,d}` for each,
the set of fired dimensions, `d*`, the gate statistic `g` and `D_{i,d*}`, `A_i`, `q_i` before and after,
`p_i`, and the update count.

**Required transition logs, once per transition rather than once per episode:** dimension availability
changes, calibration completion, coverage-cycle closures, and Generate-eligibility changes.

Every value in §5 marked diagnostic-only must be classified as such wherever it is surfaced. Rulebook
diagnostics remain diagnostic-only with respect to the reward and the policy observation; their use as
teacher inputs here is a curriculum input and does not make them policy-visible.

## 11. Reproducibility And Compatibility

The specification ID and version, the resolved YAML, dataset and catalog hashes, the seed, and full
selector state are persisted. `H`, `band_level`, and `tau` are part of the recorded configuration.

**Breaking changes relative to `v1.3`:**

- checkpoint schema `acl_ema_v3` → `acl_progress_v1`; no migration path, restart required;
- `ScenarioRecord` gains `last_generate_step` and loses the meaning of `usefulness`/`usefulness_norm`/
  `rank` as ranking inputs, so persisted `v1.3` buffers cannot be reused;
- the RNG call order changes: the calibration phase consumes at most one draw per selection instead of a
  coin flip plus an arm draw, so `v1.3` seeds do not reproduce their runs;
- `replay_sampling.beta` and the previous `mab.feedback` value are rejected;
- the teacher requires the six-level rulebook of `RULEBOOK-V5.1`. A run configured with a pre-v5.1
  rulebook composition does not satisfy §3.2 and must fail validation rather than mapping the old macro
  names onto the new dimensions.

Experiments completed under `v1.1`/`v1.2`/`v1.3` remain reproducible under their own recorded
configuration. Their policy-performance results are unaffected by this version; their curriculum-selection
claims were already qualified by `FIND-001`/`FIND-005` and are further qualified by `FIND-006`/`FIND-007`.

The evaluation panel, the validation/test splits, and `EVAL-PROTOCOL` v1.0 are untouched.

**Pre-registration requirement.** This specification must be approved and frozen **before** any comparison
run under it begins. The claim that no parameter was chosen by observing run performance is only
defensible if the approval precedes the runs, and the approval record in §18 is the evidence for it.

## 12. Acceptance Criteria

### AC-201: Neutral expectation under a stationary noisy arm

- Given: two windows of `2H` episodic keys drawn i.i.d. from the same distribution, for a distribution
  whose scale is varied over at least two orders of magnitude.
- When: `G` is computed for each draw over many seeded repetitions.
- Then: the sample mean of `G` is `0.5` within Monte-Carlo tolerance for every scale level, and the
  ordering of mean `G` across scale levels shows no monotone trend. The fire rate of `REQ-005` is likewise
  flat across scale levels.
- Reference values, already measured on the specified construction at `H = 10` (`FIND-010`): over a `250x`
  scale range the sample mean `G` was `0.4989`, `0.4987`, `0.5021`, `0.5008` and the fire rate `0.045`,
  `0.043`, `0.050`, `0.043`. The implementation test must reproduce this qualitative flatness; the same
  quantities computed from the `v1.3` prediction-error signal must show the monotone trend it replaces.
- Related requirements: `REQ-004`. This is the direct regression for `LIM-002`.

### AC-202: Monotone response to genuine improvement

- Given: an older window from distribution `P_old` and a recent window from `P_new` stochastically
  dominating it in the "better" direction, at three separation levels.
- When: `G` is computed.
- Then: `G > 0.5` in all three cases and increases monotonically with separation; with the two windows
  exchanged, `G < 0.5` symmetrically.
- Related requirements: `REQ-004`.

### AC-202b: Noisy stationary arm is not preferred, in closed loop

- Given: a seeded closed-loop teacher over `K = 6` arms in which one arm is stationary and carries several
  times the between-episode noise of the others, driven through `REQ-004`…`REQ-009` for a run-length
  number of episodes.
- When: the mean feedback `A` received by each arm and the total Generate count per arm are accumulated.
- Then: the stationary noisy arm's mean `A` is `0.5` within Monte-Carlo tolerance, it does not receive more
  Generate draws than the improving arms, and its terminal `p_i` is at the exploration floor.
- Reference values already measured at `H = 10` (`FIND-010`): mean `A` of the noisy stationary arm was
  `0.498–0.499` at every signal level tested, against `0.506–0.547` for the improving arms, and it received
  the fewest Generate draws of all six arms (`1436–1491` against `1500–1600`).
- Related requirements: `REQ-004`, `REQ-007`, `REQ-009`. This is the closed-loop regression for `FIND-006`.

### AC-203: L1 statistic is defined for collision-free episodes, and counts only at-fault contacts

- Given: an episode in which no step has a new at-fault contact, so every L1 step is `applicable=False`.
- When: the episodic statistics are computed.
- Then: `C_{1,e} = 0`, `v_{1,e} = 0`, the L1 window receives the observation, and the episode counts toward
  L1 availability. Given instead an episode with one at-fault contact of cost `0.4` and one of cost `0.7`,
  `C_{1,e} = 0.7` and `v_{1,e} = 1`. Given an episode whose only contact is classified **not at fault**
  (`ADR-071`), `C_{1,e} = 0` and `v_{1,e} = 0`, the L1 window still receives the observation, and the
  episode does **not** enter the `T` window (`DEC-207`).
- Related requirements: `REQ-002`, `FIND-008`, `FIND-014`, `DEC-207`.

### AC-203b: L2/L3/L5 gate statistic is the step fraction and does not saturate

- Given: an episode of `100` steps in which L2 is applicable at `80` steps and violated at `12` of them,
  and separately an episode in which L2 is applicable at `80` steps and violated at exactly one.
- When: the episodic statistics are computed.
- Then: `F_{2,e} = 0.15` and `F_{2,e} = 0.0125` respectively, `v_{2,e}` is not computed at all, and a
  recent window composed of such episodes yields a strictly positive gate `D`. Given a window of episodes
  each containing at least one L2 violation, `D` must **not** be `0`. The same must hold for L3 and L5.
- Additionally, given an episode in which the ego is stationary for `40` of `100` steps with a relevant
  actor present, those `40` steps must not appear in `N^{app}_{2,e}` (`ADR-070`).
- Related requirements: `REQ-002`, `REQ-006`, `FIND-011`. This is the regression for the saturation defect
  that an episode-level indicator would have introduced on L2.

### AC-204: L2/L3/L5 unobserved episodes do not become zero-cost observations

- Given: an episode in which level L2 is applicable at zero steps, and separately one in which L5 is
  applicable at zero steps because the map carries no lane marking and no opposing surface.
- When: the episodic statistics are computed.
- Then: no observation is appended for that level, `|W_i^{L2}|` (respectively `|W_i^{L5}|`) is unchanged,
  and its availability is unaffected; `C_{k,e}` is reported as undefined, never as `0`.
- Related requirements: `REQ-002`.

### AC-205: Exact permutation band

- Given: two windows of `H` identical constant values.
- When: the permutation p-value is computed.
- Then: `p^{perm} = 1.0` exactly, `G = 0.5` exactly, and no dimension is selected. Given instead two
  windows with no ties and complete separation, `p^{perm} = 2 / C(2H,H)` exactly and `G` is `0` or `1`.
  In both cases the computation consumes zero random numbers and is bit-identical across repeated calls.
- Additionally, at `H = 10` the implementation must agree, to `1e-12`, with exhaustive enumeration of all
  `C(20,10) = 184756` splits on at least six inputs, of which at least two are tie-heavy. All intermediate
  counts must be exact integers, and an `H` for which `C(2H,H)` exceeds `2^63` must be rejected at
  configuration validation rather than computed in floating point.
- Related requirements: `REQ-005`, §7.

### AC-206: Priority-ordered selection without multiplicity correction

- Given: constructed windows in which L1 is neutral, L2 fires, and `T` fires with a smaller p-value than
  L2, with all five dimensions available.
- When: the dimension is selected.
- Then: `d* = L2`, not `T`. Given instead windows in which only `T` and `L5` fire, `d* = T`, never `L5`.
  The band `0.05` is applied to each dimension's raw `p^{perm}` independently,
  and the number of available dimensions does not change the threshold applied to any of them. Given
  instead windows in which no dimension fires, `d*` is undefined and `A_i = 0.5` exactly. A configuration
  supplying `progress.multiplicity_correction` fails validation.
- Related requirements: `REQ-001`, `REQ-005`, `FIND-010`.

### AC-207: Gate closes at both extremes, on the right statistic per dimension

- Given: a recent window on `L1` with `vbar = 0`, then `1`, then `0.5`, each combined with `G = 0.8`.
- When: `A_i` is computed.
- Then: `A_i = 0.5`, `0.5`, and `0.8` respectively.
- Given the same three values of `Fbar` on `L2`, `L3` and `L5`, and of `rbar` on `T`, the same three
  results must hold. The implementation must consume `v_{1,e}` on `L1`, `F_{k,e}` on `L2`, `L3` and `L5`,
  and `route_completion` on `T`; consuming `v` on `L2`, `L3` or `L5`, or `F` on `L1`, is a contract
  violation.
- Related requirements: `REQ-002`, `REQ-006`, `REQ-007`, `FIND-011`.

### AC-208: Calibration completeness and Replay gating

- Given: a seeded run from a fresh state.
- When: selections proceed until calibration completes.
- Then: every selection before completion is Generate, no `q_i` changes from `0.50`, each arm reaches
  exactly `2H` valid Generate episodes, the per-arm counts differ by at most one at every intermediate
  point, `L1` is available for every arm at completion and `T` is available for every arm that recorded no
  not-at-fault truncation, and the first Replay cannot occur before
  both `REQ-008` conditions hold. Episodes ending in a typed data-abort do not count toward `2H`.
- Related requirements: `REQ-008`.

### AC-209: Neutral state is uniform sampling

- Given: `A_i = 0.5` fed to every arm repeatedly from the initial state.
- When: `p` is computed after convergence.
- Then: every `q_i` equals `0.50` and `p` equals the uniform distribution `1/6` within `1e-9`. With one arm
  driven to `A = 0.8` and the rest neutral, that arm's `p` exceeds `1/6` and every other arm still
  satisfies the exploration floor.
- Related requirements: `REQ-007`, `REQ-009`.

### AC-210: Buffer balance, eviction key, and replay mass

- Given: a full buffer at `capacity = 250`.
- When: Generate commits arrive on arms in an unbalanced pattern.
- Then: `|n_i - n_i*| <= 1` holds after every commit, `|Λ| = 250` is preserved, the evicted record is
  always the one with the smallest `last_generate_step` in the chosen donor arm, and a record whose
  `last_seen_step` was refreshed by a Replay but whose `last_generate_step` is oldest is still the eviction
  target. Additionally, `sum over r in arm i of P_progress(r)` equals `p^{full}_i` within `1e-9`.
- Related requirements: `REQ-010`, `REQ-011`.

### AC-211: Replay episodes do not move the teacher

- Given: an arm with full windows and a fixed `q_i`.
- When: an arbitrary number of Replay episodes on that arm's records commit, with arbitrary outcomes.
- Then: `q_i`, every `W_i^d`, and `N^{gen}_i` are unchanged; only `last_seen_step`, visit counts, and
  diagnostics change; no insertion or eviction occurs.
- Related requirements: `REQ-003`, `REQ-009`, `REQ-010`.

### AC-212: Prediction-error LP is inert

- Given: two runs identical in seed and configuration, in which the diagnostic learning-potential values
  are forced to arbitrarily different finite values.
- When: the seeded selection sequence is generated.
- Then: the mode sequence, arm sequence, scenario sequence, `q_i` trajectory, buffer contents, and every
  eviction are bit-identical; only the logged `U`/`U_scaled`/`U_norm` differ.
- Related requirements: `REQ-013`. This is the mechanical guarantee that the removed signal is truly
  removed.

### AC-213: Data-abort isolation

- Given: a Generate episode ending in a typed data-abort (`ADR-024`).
- When: the completion commits.
- Then: no window is appended, `N^{gen}_i` is unchanged, `q_i` is unchanged, the record is quarantined and
  removed from `Λ` without replacement, and the record remains marked visited in its coverage cycle.
- Related requirements: `REQ-002`, `REQ-008`, `REQ-010`, `REQ-012`.

### AC-214: Checkpoint round-trip and rejection

- Given: a mid-run state with partially filled windows, mixed dimension availability, an incomplete
  calibration, and a full buffer.
- When: the state is persisted and reloaded.
- Then: every field of `REQ-014` round-trips exactly and the subsequent seeded selection sequence is
  identical to the uninterrupted one. `acl_ema_v3`, `acl_ema_v2`, `acl_ema_v1`, and cumulative-schema
  payloads are rejected with an explicit error.
- Related requirements: `REQ-014`.

### AC-215: Algorithm independence

- Given: identical seeds, identical scenario sequences, and identical episodic outcomes, under PPO, TD3,
  and SAC.
- When: the teacher updates are computed.
- Then: the `G`, `D`, `A_i`, and `q_i` trajectories are identical across the three planners.
- Related requirements: `REQ-002`, `REQ-013`, §1.3.

### AC-216: Default composition and explicit override

- Given: the root configuration with no curriculum override, and separately `curriculum=disabled`.
- When: the configuration is composed.
- Then: the first resolves to this `v2.0` ACL profile and the second to the disabled profile. A
  configuration carrying `replay_sampling.beta` or the legacy `mab.feedback` value fails validation with an
  explicit message.
- Related requirements: `REQ-009`, `REQ-010`, §9.

### AC-217: Emergence, plateau, and handover in closed loop

- Given: a seeded closed-loop teacher over `K = 6` synthetic arms in which three arms improve in sequence
  at staggered onsets, at a stated `SNR` per window gap of at least `1.25`.
- When: the selection probabilities are snapshotted along the run.
- Then: each improving arm's `p_i` rises above `1/K` during its improvement phase, peaks after that phase
  begins, and returns to `1/K` within the exploration floor once it plateaus; the peaks occur in the order
  of the onsets; and every `p_i` returns to the neutral value by the end of the run.
- Reference values already measured at `H = 10` (`FIND-010`): peaks of `0.242`, `0.245`, `0.246` in onset
  order, all returning to `0.167` before the end of the run.
- **Falsification clause, normative.** At `SNR = 0.75` the same construction did **not** depart materially
  from uniform, with or without multiplicity correction. This acceptance criterion is therefore a property
  of the teacher **at a stated signal level** and is not a claim that any real arm attains that level. Any
  document derived from this specification that presents `AC-217` must state the `SNR` at which it holds.
- Related requirements: `REQ-004`…`REQ-009`, `LIM-201`.

## 13. Required Validation Categories

| Category | Status |
|---|---|
| nominal and boundary behavior | required |
| invalid and incomplete inputs | required |
| masks, padding, state, reset, and update order | required |
| termination and truncation | required (teacher treats truncation as a task outcome; data-abort is separate) |
| deterministic seeds and reproducibility | required |
| numerical stability, NaN, and infinity | required |
| compatibility and migration | required (schema rejection; no migration path) |
| absence of future and privileged information | required, including `AC-212` inertness of the removed signal |
| upstream, downstream, and end-to-end integration | required (vectorized ACL smoke; sequential path aligned) |
| regressions for known bugs | required (`FIND-006`…`FIND-015`) |
| measured properties of the signal | required (`AC-201`, `AC-202b`, `AC-217` reproduce `FIND-010` on the implementation, not only on the audit script) |
| mutation as a prohibited feature | required, unchanged |

## 14. Traceability

| Requirement | Acceptance criteria | Scientific source or approved decision |
|---|---|---|
| `REQ-001` | `AC-206` | `RULEBOOK-V5.1` §3 level order; `ADR-072`, `ADR-076`; project adaptations `RAT-203`, `DEC-206` |
| `REQ-002` | `AC-203`, `AC-203b`, `AC-204` | `RULEBOOK-V5.1` §3.2/§3.3 and `aggregation.py` contract; `ADR-070`, `ADR-071`; `FIND-008`, `FIND-011`, `FIND-014`, `DEC-207` |
| `REQ-003` | `AC-211` | Matiisen et al. 2017 windowed progress; `RAT-205` |
| `REQ-004` | `AC-201`, `AC-202`, `AC-202b`, `AC-217` | Vargha and Delaney 2000; Graves et al. 2017; resolves `LIM-002`; `FIND-010` |
| `REQ-005` | `AC-205`, `AC-206` | exact permutation null; `RAT-204`; `FIND-010` (removal of Holm) |
| `REQ-006` | `AC-207`, `AC-203b` | competence-based curriculum (PORTAL 2024; SITP 2023); `FIND-009`, `FIND-011`, `FIND-014` |
| `REQ-007` | `AC-207`, `AC-209`, `AC-202b` | project construction; `RAT-207` |
| `REQ-008` | `AC-208` | project construction, derived from `K` and `H` |
| `REQ-009` | `AC-209` | ACL `v1.1` `REQ-001`/`REQ-002`; `ADR-028`; `RAT-201` |
| `REQ-010` | `AC-210`, `AC-213` | `ADR-029` `DEC-007` channel analysis; `RAT-206` |
| `REQ-011` | `AC-210` | Jiang et al. 2021 staleness; project construction for the progress term |
| `REQ-012` | `AC-213` | `ADR-032` `DEC-008`/`DEC-009`, carried forward unchanged |
| `REQ-013` | `AC-212`, `AC-215` | `v1.3` `LIM-006`; user decision 2026-07-30 |
| `REQ-014` | `AC-214` | `v1.3` `REQ-005`; `ADR-016` |

## 15. Open Decisions And Limitations

### 15.1 Decisions

`DEC-201`…`DEC-204` were **proposed by the assistant and selected by the user on 2026-07-30**, then
**reopened by the user on 2026-07-31** on the grounds that the selection had not been a considered
approval. They were re-examined in revision 2, two of them were changed on evidence. `DEC-205` was added
in revision 2. `DEC-206` and `DEC-207` are new in revision 3 and are forced by the re-base onto
`RULEBOOK-V5.1`: `DEC-206` was **selected by the user on 2026-09-01** from three stated alternatives,
`DEC-207` is proposed by the assistant and is not yet selected. None of these should be cited as approved
before the approval recorded in §18.

| ID | Question | Resolution in revision 2 | Status |
|---|---|---|---|
| `DEC-201` | How is L1's episodic cost defined, given that L1 is `applicable=False` without an at-fault contact? | Unchanged in substance: maximum over all steps, L1 always observed; "not applicable" means satisfied. Re-based in revision 3 onto `ADR-071`, under which "not applicable" additionally covers a not-at-fault contact charged nothing (`REQ-002`) | reopened 2026-07-31, **confirmed**; re-based 2026-09-01, pending approval |
| `DEC-202` | What does the Goldilocks gate consume? | **Changed** in revision 2: `v_{1,e}` incidence on L1, the violated-applicable-step fraction `Fbar` on the step-fraction levels, `route_completion` on `T`; `G` keeps the fine-grained cost `C`. Revision 3 extends the `Fbar` branch to L5 (`REQ-002`, `REQ-006`) | reopened 2026-07-31, **revised** on `FIND-011`; extended 2026-09-01, pending approval |
| `DEC-203` | How is the neutrality band framed and corrected? | **Changed**: exact conditional permutation band at `0.05` **per dimension, uncorrected**; Holm removed on `FIND-010`; framing as a calibrated deadband retained and strengthened (`REQ-005`) | reopened 2026-07-31, **revised** on `FIND-010`, pending approval |
| `DEC-204` | Is the prediction-error learning potential removed or retained? | Unchanged: retained as a strictly inert diagnostic channel, so the two signals are measurable within one run (`REQ-013`) | reopened 2026-07-31, **confirmed**, pending approval |
| `DEC-205` | What is `H`? | `H = 20`, on the **measured** power sweep of `FIND-012` and the asymmetry of the two error directions (§9.1); `H = 30` is the indicated revision if the deferred `SNR` measurement returns below `0.5`; the measurement is a post-hoc validation, not a gate (`LIM-201`) | new in revision 2, pending approval |
| `DEC-206` | Which `RULEBOOK-V5.1` levels are teacher dimensions, and in what order? | Five dimensions in the order `L1 → L2 → L3 → T → L5`. `T` occupies L4's position because `route_completion` is L4's episodic counterpart; `L5` is last because `ADR-072` placed the relaxable lane rules below mission progress; `L6` is excluded with L4 under `RAT-203`. Alternatives considered and rejected: four dimensions with L5 dropped (loses a measurable dimension for no gain, since priority ordering already consults it last), and `L1 → L2 → L3 → L5 → T` (would place lane relaxation above progress, inverting `ADR-072`) | new in revision 3, **selected by the user 2026-09-01**, pending approval as part of the set |
| `DEC-207` | What happens to an episode truncated by a not-at-fault contact (`ADR-071`)? | It remains a valid Generate episode and updates `L1`, `L2`, `L3`, `L5`, the counters and the buffer, but is **excluded from the `T` window only**. Rationale: the truncation caps `route_completion` for a reason the policy is not charged for, so admitting it into `T` would inject a downward bias correlated with traffic density and therefore with the arm — precisely the kind of arm-correlated confound this version exists to remove. The per-step fractions of the other levels are normalized by applicable steps and are unbiased, merely noisier, on a shorter episode. Alternative rejected: admitting it everywhere, which is simpler but reintroduces an exogenous, arm-correlated term into the highest-traffic arms' `T` statistic | new in revision 3, **proposed, not yet selected** |

**`DEC-202`, why it changed.** Revision 1 used an episode-level violation indicator `v_{k,e}` for all of
R1–R3. The user identified, and repository inspection confirmed, that this saturates on R2: its
subcomponents are applicable whenever a relevant actor exists and are violated at some step in essentially
every episode with traffic, so `vbar -> 1` and the R2 gate would close permanently. Recorded as
`FIND-011`.

**`DEC-203`, why it changed.** See `REQ-005` and `FIND-010`. The correction was removed because its bias
cost was measured to be negligible while it halved sensitivity — **not** because it was blocking the
curriculum, a claim the same measurement falsifies.

### 15.2 Recorded findings

- **`FIND-006`** (2026-07-30, user diagnostic run, 120k steps) — Under `v1.2`/`v1.3` reward-scale
  normalization, `A4_vru`'s reward-scale EMA grew from `48.9` to `104.7` over the run while its bandit
  score remained the highest at the end (`0.597` against `~0.40–0.46` for the other arms). Normalization
  therefore did not remove the preference for the structurally noisiest arm, which is the behavior
  `LIM-202`'s predecessor `LIM-002` predicts.
- **`FIND-007`** (2026-07-30, user diagnostic run) — Same configuration and budget with ACL enabled versus
  disabled, fixed evaluation panel, scalar reward, 40 final test episodes: out-of-road rate `0.60` with ACL
  against `0.525` without, collision rate `0.275` against `0.10`, route completion slightly lower with ACL.
  **Evidentiary weight, stated explicitly:** one seed and 40 episodes. The confidence intervals of these
  proportions are wide and partially overlapping. This finding motivates the redesign; it does **not**
  establish that the `v1.3` curriculum harms policy performance, and no document derived from this
  specification may claim that it does. The load-bearing evidence for the redesign is analytical
  (`LIM-002` plus `ADR-029`'s measured `+0.771…+0.943` correlations), corroborated by `FIND-006`.
- **`FIND-008`** (2026-07-30, code verification) — R1 (`collision_impact`) is `applicable=False` on every
  step without a new contact onset (`components/collision.py`), and the macro aggregation propagates it
  (`aggregation.py`). Averaging R1's cost over applicable steps would restrict its window to collision
  episodes, invert its semantics from "collision behavior" to "severity given a collision", and prevent a
  nearly-solved arm from refilling the window or closing the gate. Corrected by `DEC-201`.
  **Status under `RULEBOOK-V5.1`:** re-verified and still true, with the applicability predicate now
  reading "a new **at-fault** contact" (`FIND-014`).
- **`FIND-009`** (2026-07-30, code verification) — R1's cost is a MAIS3+F injury probability and R3's is a
  graded off-road cost, so mean episodic costs occupy roughly `[0, 0.1]`. A gate `4C(1-C)` on the mean cost
  would attenuate the Rulebook dimensions to at most `~0.4` while the task gate spans `[0, 1]`, an
  unintended scale artifact. Corrected by `DEC-202`. **Status under `RULEBOOK-V5.1`:** unchanged in kind.
  The cost scales were redefined by `ADR-065`/`ADR-067` but remain small-valued per step, and L5's
  normalized sum over a fixed denominator of 3 is if anything smaller still.
- **`FIND-010`** (2026-07-31, synthetic power and closed-loop dynamics analysis;
  `docs/audits/acl_v2_teacher_power_analysis_2026-07-31/`, reproducible script and captured output) — Four
  measured results on the exact construction of `REQ-002`…`REQ-009`, at `H = 10`:
  1. **Scale invariance.** Over a `250x` range of noise scale with both windows exchangeable, mean `G` was
     `0.4989 / 0.4987 / 0.5021 / 0.5008` and fire rate `0.045 / 0.043 / 0.050 / 0.043`, with no monotone
     trend. This is the measured replacement for the analytical claim that `LIM-002` is removed.
  2. **Closed-loop neutrality of a noisy stationary arm.** Mean feedback `0.498–0.499` at every signal
     level, against `0.506–0.547` for improving arms, and the **fewest** Generate draws of the six.
     Directly answers `FIND-006`.
  3. **Holm is not the deciding factor.** Peak arm probability during the learning phase was `0.24–0.27`
     uncorrected against `0.23–0.26` corrected at `SNR = 1.25`, and at `SNR = 0.75` neither configuration
     departed materially from uniform. False fires on a stationary arm were `0.032–0.035` uncorrected
     against `0.006–0.007` corrected, with **identical mean feedback** under both. Basis for `DEC-203`.
  4. **Cost.** The exact permutation p-value costs `0.059 ms` cold and `0.019 ms` warm per call, `0.24 ms`
     for a four-dimension commit — `0.30 ms` for the five dimensions of revision 3, by the same per-call
     figure — after replacing arbitrary-precision integers with `int64` arrays — a
     `70x` improvement over the first implementation, which cost `4.2 ms`. Bears on `LIM-208`.
  **Evidentiary weight, stated explicitly:** this is a synthetic simulation of the teacher, not of the
  learner or the environment. It establishes properties of the signal and of the bandit loop given a
  stated `SNR`. It establishes nothing about the `SNR` any real arm attains, which is the subject of
  `LIM-201`, and nothing about policy performance (§1.5).
- **`FIND-011`** (2026-07-31, user observation confirmed by code verification) — R2's subcomponents
  (`rss`, `rss_lateral`, `ttc`, `clearance`) are applicable whenever a relevant actor exists, and at least
  one of them is violated at some step in essentially every episode containing traffic. An episode-level
  violation indicator would therefore sit at `v_{2,e} = 1` almost surely, driving `vbar -> 1` and closing
  the R2 gate permanently; R3 has the same defect to a lesser degree. R1 does not, because its
  applicability predicate fires only on a new contact onset. Corrected by the revised `DEC-202`.
  **Status under `RULEBOOK-V5.1`:** the conclusion carries to L2, whose sub-rules are now `ttc`,
  `clearance` and `rss_lateral` only — `rss` longitudinal left the channel under `ADR-063` — and to L3 and
  L5. The saturation argument does not depend on which sub-rules compose the level, only on their being
  applicable on most steps and violated at some step of most episodes (`FIND-014`).
- **`FIND-012`** (2026-07-31, window-size sweep; `window_size_sweep.py` and its captured output in the same
  audit directory) — Detection power of the neutrality band measured across `H in {10, 15, 20, 30}` and
  `SNR in [0, 2]`, `4000` repetitions per cell. Three results:
  1. In the operating case — improvement rate fixed per arm-episode, so the window gap scales with `H` —
     detection at `SNR = 0.5` (quoted at `H = 10`) rises `0.157 -> 0.481 -> 0.854 -> 1.000` across the
     four `H` values, and at `SNR = 0.75` it rises `0.306 -> 0.815 -> 0.993 -> 1.000`.
  2. Holding `SNR` fixed, so only sample size grows, the same `SNR = 0.75` row rises
     `0.323 -> 0.464 -> 0.613 -> 0.797`. Sample size therefore accounts for **less than half** of the gain;
     the widened window gap accounts for the rest. This is the direct measurement of the `H^{1.5}` claim
     that §9.1 of revision 2 had only extrapolated.
  3. The false-fire rate under exchangeability is `0.038 / 0.042 / 0.049 / 0.049` for the four `H` values,
     at or below the nominal `0.05`, and mean `G` is `0.4998 / 0.5023 / 0.5015 / 0.4990`. The added power
     is not bought with inflated false positives, and the `AC-201` invariance holds at every `H`.
  The DP mass was asserted equal to `C(2H, H)` on every call, and `C(60,30) = 1.18e17 < 2^63` confirms the
  int64 contract of §7 holds through `H = 30`.
  **Evidentiary weight:** synthetic, as `FIND-010`. It measures the estimator, not the arms. The operating
  rows assume a locally constant improvement rate and are therefore an upper bound; the fixed-`SNR` rows
  are a lower bound.
- **`FIND-013`** (2026-07-31, measurement on completed runs; `replay_teacher_on_real_runs.py`,
  `learning_curve_check.py` and their captured outputs in the same audit directory) — The proposed `v2.0`
  teacher was replayed over the real committed Generate sequences of
  `EXP_thesis_RP_thesis_CUR_scenario_acl_scenarionet_REW_scalar_reward`, seed 0 of each planner (PPO
  475209 steps, TD3 350154, SAC 325143; none reached the 1.5M `thesis` budget). Two results:
  1. **Measured `SNR ~ 0.00–0.05` on every arm and every reconstructible dimension**, with mean `G`
     `0.491–0.514`, mean `A` `0.496–0.514`, and fire rates equal to the deadband's nominal false-positive
     rate. Against §9.1 this is the **stationary row**. The proposed teacher, driven by real data, would
     have sampled uniformly for the entire run. R2 could not be reconstructed because per-macro-rule
     episodic costs are absent from current logs — a gap `REQ-002` closes.
  2. **The cause is not the teacher.** Binning each run into eight parts, the scalar reward improves
     substantially in all three algorithms (TD3 `-80.4 -> -54.8`; PPO `-233.1 -> -112.5`; SAC `+99.7`)
     while driving competence does not: success rate stays between `0%` and `8%` throughout and does not
     trend up in TD3 or SAC, and out-of-road rate **rises** in every case (PPO `~0.15 -> ~0.42`).
  **Consequences for this specification.** (a) `H` cannot be confirmed against this data (`LIM-201`).
  (b) `FIND-007` is further weakened: it compares two configurations neither of which learns to drive, so
  its differences are dispersion around a non-learning baseline, not a curriculum effect. (c) `FIND-006`
  **keeps its full force** — that the `v1.3` teacher preferred the noisiest arm while nothing was being
  learned is precisely the defect, and the replay shows `v2.0` correctly returns neutral on the same data.
  (d) This version is validated as far as this evidence permits — correctly silent when there is nothing
  to detect (`FIND-013`), demonstrably responsive when there is (`FIND-010`, `FIND-012`) — but **cannot be
  validated end to end** until a configuration exists in which the policy improves. That is a
  reward-specification problem, outside this specification's scope.

- **`FIND-014`** (2026-09-01, code and specification verification against `RULEBOOK-V5.1`) — This
  specification was written against Rulebook v4.7–v4.9's three macro rules. `RULEBOOK-V5.1` (approved
  2026-08-14, amended 2026-08-20, implementation verified 2026-09-01) replaced them with six ordered
  levels. Verified point by point against the implementation:
  1. **Level composition.** L1 `collision_safety`; L2 `interaction_risk` = `ttc`, `clearance`,
     `rss_lateral`; L3 `non_relaxable_compliance` = `offroad`, `signal`, `stop`, `crosswalk`,
     `vehicle_yield`, `speed_limit`; L4 `mission_progress`; L5 `relaxable_lane_compliance` = `solid_line`,
     `wrong_carriageway`, `dashed_line`; L6 `progress_rate` (`types.py:99`, `registry.py`). The six names
     are emitted in `info["rule_metadata"]["rule_names"]` (`wrapper.py:285`) and the macro results are
     present in `info["rule_components"]` keyed by channel value (`aggregation.py:219-231`), so the two
     interface assumptions of §3.2 survive the rename.
  2. **`rss` longitudinal is no longer in a channel** (`ADR-063`, `contributes_to_channel=False`), so the
     sub-rule revision 2 listed first among R2's components no longer contributes to any teacher input.
  3. **`wrongway` is deleted** (`ADR-066`), so revision 2's assumption that L3 is always applicable
     because `offroad` *or* `wrongway` is applicable now rests on `offroad` alone — which is
     `applicable=True` unconditionally (`components/road.py:138`), so the assumption holds.
  4. **`speed_limit` is new** at L3 (`ADR-068`) and **`vehicle_yield` moved into L3**, while
     `solid_line`, `wrong_carriageway` and `dashed_line` moved out to L5 (`ADR-072`). L3 is therefore a
     different quantity from R3, not a renaming.
  5. **At-fault gating** (`ADR-070`) makes the three L2 sub-rules inapplicable at `v_ego <= 5e-02 m/s`,
     and **at-fault classification** (`ADR-071`) makes L1 applicable only on an at-fault contact while a
     not-at-fault contact truncates the episode. Both reach `REQ-002` directly.
  6. **L5's aggregation is a normalized sum over a fixed denominator of 3**, not a `max`
     (`aggregation.py:63`), so an L5 cost of `1/3` can mean one saturated sub-rule rather than a uniformly
     bad step. This does not affect the ordinal comparison, which never interprets the cost's magnitude.
  **Consequences.** `REQ-001`, `REQ-002`, `REQ-005`, `REQ-006`, §3.2, §4, §7, §10, `AC-203`, `AC-203b`,
  `AC-204`, `AC-206`, `AC-207`, `AC-208` and `RAT-203` are re-based in revision 3; `DEC-206` and
  `DEC-207` are opened. No mechanism, weight, or measured result changes.
- **`FIND-015`** (2026-09-01, measurement recorded in `RULEBOOK-V5.1` `M8a`, restated here for its
  curriculum consequence) — On the PG half of the frozen training mixture, `signal`, `stop`, `crosswalk`
  and `speed_limit` are applicable on **zero** of 1089 replayed PG records, `vehicle_yield` on 196 against
  725 of 1100 on Waymo, and `clearance` (L2) on zero against 674. L3 on a PG record is therefore
  `offroad` and, occasionally, `vehicle_yield`; on a Waymo record it is a six-sub-rule level. Because the
  six semantic arms are correlated with the source — accepted as design in open item `D5` — **the `L3`
  teacher dimension does not measure the same quantity across arms**, and the same holds to a lesser
  degree for `L2`. This does not break any property of the estimator: `G` is computed *within* one arm's
  own window, so no cross-arm comparison of `L3` costs is ever formed. What it means is that a fired `L3`
  dimension is a statement about a different sub-rule mixture depending on which arm fired it, and it
  must be reported as such. Recorded as `LIM-209`.

### 15.3 Recorded rationales

- **`RAT-201`** — No inverse-probability correction on the EMA update, carried forward from `v1.1`
  `RAT-004`: the exploration floor bounds the propensity away from zero and the correction's variance cost
  is not justified at `K = 6`.
- **`RAT-202`** — `H = 20` is chosen from the measured growth of detection power in `H` (`FIND-012`) and
  the asymmetry of the two error directions — under-powered degenerates to the uniform baseline,
  over-powered costs handover latency — not from the EMA horizon. See §9.1 and `DEC-205`. Revision 1's
  parsimony argument (`H = 1/alpha`) is retained only as an observation.
- **`RAT-203`** — `RULEBOOK-V5.1` L4 (`mission_progress`) and L6 (`progress_rate`) are not independent
  teacher dimensions. L4 measures signed advance along the route and would be largely redundant with
  `route_completion` inside `T`, adding a correlated dimension to the priority order for no additional
  information and one more opportunity for a false fire; `T` therefore occupies L4's position rather than
  sitting beside it. L6 is a *rate* over the same quantity and, under `ADR-076`, expresses a time
  preference rather than a competence — an arm on which the agent arrives sooner is not thereby an arm on
  which it is learning. Revision 2 stated this rationale for v4.7's single R4 channel; it is restated here
  for the two channels that replaced it.
- **`RAT-204`** — The permutation band is a calibrated deadband, not an inference procedure. Its purpose is
  to obtain a neutrality threshold from the data's own null rather than by choosing a number while looking
  at run performance. Because it is a deadband and not a test, family-wise correction has no scientific
  meaning here: there is no run-level error rate to control (`LIM-202`), so correcting for multiplicity
  would trade measured sensitivity for a guarantee this design never claimed. See `REQ-005` and `FIND-010`.
- **`RAT-205`** — Only Generate episodes update the windows. Replay episodes are drawn by the teacher
  itself, so including them would let the teacher's own preference feed back into its measurement of the
  arm's distribution.
- **`RAT-206`** — Learnability and progress are estimated at arm level, never per record. A single episode
  cannot separate an adequately difficult scenario from a lucky success, an unlucky failure, or transient
  policy variation; using it for admission or replay priority would recreate the single-episode noise
  sensitivity this version removes.
- **`RAT-207`** — `A_i` uses the signed deviation `G - 0.5`, never `|G - 0.5|`. Taking the absolute value
  would map symmetric fluctuation back onto positive priority and reintroduce the failure mode of the
  prediction-error family.

### 15.4 Known limitations, intentional and not hiding missing behavior

- **`LIM-201` — The operating `SNR` of the real arms has been measured only under a reward specification
  that has since been replaced, so `H` remains unvalidated.** The measurement is `FIND-013`: `SNR ~ 0.01`
  on every arm and every reconstructible dimension, i.e. the stationary row of §9.1. Its cause was
  outside this specification's scope — the policy did not improve on any driving outcome, so there was no
  learning progress for any window-based teacher to detect — and that cause was addressed by
  `RULEBOOK-V5.1`, approved after `FIND-013` was recorded and verified in production on 2026-09-01
  (§1.5, `FIND-014`). **The measurement is therefore stale rather than wrong**: it is the correct value
  for the reward it measured, and it says nothing about the reward now in production, under which no run
  has yet been completed. `H = 20` rests on the power argument of §9.1 alone and is **provisional**: it is
  the correct choice *conditional on* a run in which learning occurs, and it cannot be confirmed until one
  exists. The rest of this limitation states what the choice would rest on once such a run is available.
  `FIND-010` and `FIND-012` establish what the teacher does *given* a signal level, and `FIND-012`
  materially improved the picture that `FIND-010` alone suggested: at `H = 20` the band reaches `>= 0.85`
  detection down to `SNR = 0.5` quoted at `H = 10`, a regime at which `H = 10` achieves only `0.157`. The
  earlier reading — that an `SNR` of `0.5` would defeat any window-based ordinal teacher at these budgets
  — was drawn from the `H = 10` sweep alone and is **superseded**; it holds for `H = 10`, not for the
  chosen `H`. Nothing in this repository measures which regime the six real arms occupy. `H = 20` is
  chosen from §9.1, whose asymmetry argument makes the *unfavourable* direction benign — an under-powered
  teacher degenerates to uniform sampling, i.e. to the `curriculum=disabled` baseline. That property is
  what makes shipping `H = 20` against `FIND-013` safe rather than reckless: on data with no signal this
  teacher provably samples uniformly, which is the `curriculum=disabled` behaviour. The following is the
  validation to run once a learning configuration exists:

  > Group the committed per-episode records of a run by arm, ordered by global step. Within each arm,
  > estimate the local trend of the episodic key over a span of `H` arm-episodes and the residual
  > between-episode standard deviation; their ratio is `SNR_i`. Use a **`curriculum=disabled` run**: under
  > uniform sampling the arms have comparable visit rates and comparable window spans, which is the regime
  > the `v2.0` teacher occupies at initialization, whereas a `v1.3` run's visit rates are driven by the
  > defective signal and would contaminate the estimate. Do not use a short profile: early training has
  > the steepest learning and would **over**-estimate `SNR`, biasing `H` downward.

  Under the instrumentation of runs completed to date only `T` is measurable this way, since
  `route_completion` is already logged per episode and is exactly the `T` key, while `L1` and `L3` have
  only coarse binary proxies (`collision`, `out_of_road`) that give a lower bound on power, and `L2` and
  `L5` have none until `REQ-002` is implemented. **The expected operating behavior of this design remains
  a curriculum that stays close to uniform for much of a run and departs from it only on measured
  evidence**, and it must be reported as such rather than presented as an active curriculum.
- **`LIM-209` — The `L3` dimension is not the same quantity across arms, because the arms are correlated
  with the data source.** `FIND-015` measures it: on PG records four of L3's six sub-rules are never
  applicable and a fifth is rare, so L3 there is essentially `offroad`, while on Waymo records it is a
  six-sub-rule level; `L2` differs less sharply but in the same direction, since `clearance` never applies
  on PG. The estimator is unaffected — `G` compares an arm's recent window against that same arm's older
  window, so no cross-arm comparison of costs is ever formed, and the ordinal statistic never interprets a
  cost's magnitude. What is affected is interpretation: "arm `i` improved on L3" means a different
  statement depending on the arm's source composition, and any curriculum analysis derived from this
  specification must report the arm's source composition alongside a fired `L3`. Removing the asymmetry is
  not in scope here: it is a property of the frozen catalog and of `RULEBOOK-V5.1`'s applicability
  predicates, accepted as design in open item `D5`.
- **`LIM-202` — No run-level type-I guarantee.** The band is re-evaluated after every valid Generate
  episode on overlapping windows, so no significance claim holds over a run. False fires are symmetric
  in sign under exchangeability and therefore inflate the variance of `A_i` without biasing its
  expectation — measured, not assumed: mean feedback `0.498–0.500` on a stationary arm with a false-fire
  rate of `0.032–0.035`, `sd(p_i) ~ 0.006` (`FIND-010`). See `REQ-005` and `RAT-204`.
- **`LIM-203` — Visit-frequency coupling, and the readiness/transfer confound.** The windows are measured
  in per-arm episodes, not in global training steps. A frequently sampled arm measures progress over a
  shorter stretch of training and tends to look stationary; a rarely sampled arm measures over a longer
  stretch and tends to look improving. Two consequences must be reported separately.

  1. **Anti-concentration.** The coupling is negative feedback on `p_i` and therefore acts in the
     corrective direction relative to `FIND-007`, and it is also the mechanism that produces automatic
     handover between arms. It must **not** be described as merely favourable: it is a confound that
     happens to point the right way here, and it caps how sharply the curriculum can concentrate even
     when concentration is warranted.
  2. **Readiness and transfer are not separated from arm-local learning.** Because the network is shared,
     an arm's key can improve while the agent is training on *other* arms. The measured `G` therefore
     conflates "this arm is learnable now" with "the agent became ready for this arm elsewhere". This
     version does not separate the two and does not claim to.

  Required diagnostic, to make the confound measurable rather than merely acknowledged: log, per teacher
  update, the mean global-step index of each window and their difference
  `Δs_{i,d} = meanStep(R_i^d) - meanStep(O_i^d)`. A systematic association between `Δs` and `G` across
  arms is the signature of the coupling; reporting `Δs` alongside `G` allows a reader to judge it. `Δs` is
  diagnostic-only and must never enter `A_i` — dividing `G` by `Δs`, or by any visit-rate factor, would
  reintroduce a magnitude into an ordinal statistic, destroy the scale invariance measured in `FIND-010`,
  and create positive feedback between selection frequency and score.

  The coupling is a structural property of window-based progress measures — Matiisen et al. (2017) share
  it — and it is not removed by this version. The principled alternatives (requiring a minimum global-step
  separation between the two windows, or defining the windows over step intervals rather than episode
  counts) are deferred, with no design in this version.
- **`LIM-204` — No ablation of `H`, `buffer_capacity`, `band_level`, or `tau`.** These are reasoned
  choices, not ablated ones. No experiment in this thesis isolates their effect and none is planned. `tau`
  in particular is identified in §9.2 as the correct lever for under-concentration and is deliberately
  left unchanged.
- **`LIM-205` — Calibration cost dominates short profiles, for efficacy claims only.** Calibration
  requires `K * 2H = 240` valid Generate episodes. Expressed as a fraction of a run this is small for
  `run_profile=thesis` (1.5M steps) and large for `run_profile=fast` (120k steps), but the exact fraction
  **is not yet known**: it depends on mean episode length in consumed steps, which varies by arm, and the
  `25–50%` figure carried in revision 1 was an estimate and not a measurement. The ExecPlan must report
  calibration cost in **consumed environment steps** measured on a real run, not in episodes.

  The consequence is narrower than revision 1 stated. `fast`-profile runs remain **valid** for correctness
  tests, for the `AC-212` inertness check, for overhead measurement, and for falsification of the
  mechanics. What they cannot support is any **efficacy** claim about the curriculum, because a run in
  which calibration occupies a large share of the budget spends most of its time in uniform sampling by
  construction and therefore cannot distinguish the curriculum from its own baseline.
- **`LIM-206` — The scenario buffer is no longer prioritized level replay.** With arm-balanced admission
  and `P_progress = p_i / n_i`, the buffer is a recency-balanced revisit memory whose only per-record term
  is staleness. Only that term retains PLR parentage; the buffer must not be described as PLR in derived
  documentation.
- **`LIM-207` — The commit-path closure still has no isolated harness.** Carried forward from `v1.3`
  `LIM-004`: `commit_event` is a private closure inside the vectorized training loop. `AC-210`/`AC-211`
  cover the buffer-level and teacher-level contracts, but the branch selection inside the closure remains
  verified by code reading plus the passing suite.
- **`LIM-208` — Cost of the new instrumentation, in full.** Three distinct costs are introduced and all
  three must be measured in the ExecPlan's smoke run rather than assumed.
  1. **Per-step accumulation.** `REQ-002` requires per-slot accumulation of level costs, applicability,
     and violated-step counts at every environment step of the vectorized loop, for the four consumed
     levels. The data is already present in the step info, but the accumulation is new work in the hot
     loop, bounded by roughly four float comparisons and eight counter increments per slot per step. It
     scales with `num_envs x steps`, which makes it the **dominant** of the three despite being the
     cheapest per unit.
  2. **Per-commit permutation.** Up to five exact p-values per valid Generate episode. Measured at
     `0.059 ms` cold and `0.019 ms` warm per call, `0.24 ms` per four-dimension commit and therefore
     `0.30 ms` for five (`FIND-010`), and
     `O(H^2)` in the recursion so `H = 20` does not change the order. This is negligible against episode
     wall-clock, **provided** the `int64` construction of §7 is implemented; the arbitrary-precision
     variant measured `70x` slower and must not be used.
  3. **Logging volume.** §10 requires per-episode Rulebook statistics plus per-update teacher internals
     plus the `Δs` diagnostic of `LIM-203`, which is materially more per-episode data than `v1.3` emitted.
     Its effect on artifact size and on I/O in the vectorized loop is unmeasured.

  The `AC-212` inertness requirement means the `v1.3` diagnostic chain also continues to run (`REQ-013`),
  so `v2.0` pays for both signals for the duration of the comparison.

### 15.5 Claims this specification supports

1. The teacher's feedback is no longer structurally proportional to the variance of a prediction-error
   signal (`AC-201`), a property **measured** over a `250x` scale range rather than argued analytically
   (`FIND-010`).
2. A noisy but stationary arm receives a neutral expected feedback of `0.5`, both in isolation (`AC-201`)
   and inside the closed bandit loop, where it also receives no selection advantage (`AC-202b`).
3. An arm that is already solved on the selected dimension, and an arm entirely out of reach on it, are
   both attenuated to neutral (`AC-207`).
4. The teacher is algorithm-independent: identical outcomes produce identical curricula under PPO, TD3, and
   SAC, and therefore across the four reward settings (`AC-215`).
5. The neutral state of the curriculum is uniform sampling (`AC-209`).
6. Given an arm signal-to-noise ratio of at least `1.25` per window gap, the teacher reproduces
   emergence, plateau, and handover in the order of the arms' onsets (`AC-217`). **This is conditional on
   a signal level no measurement in this repository establishes** (`LIM-201`), and must always be stated
   with that condition.
7. Whether this improves sample efficiency or final performance is an open experimental hypothesis, to be
   tested against `curriculum=disabled` under `EVAL-PROTOCOL` v1.0 (§1.5).

**Claims this specification explicitly does not support**, listed because they are the plausible
overstatements of the results above: that multiplicity correction was preventing the `v1.3` curriculum
from working (`FIND-010` falsifies it); that `H = 20` is validated against real data (it is validated
against a synthetic power sweep, `FIND-012`, and the real arms measure `SNR ~ 0.01`, `FIND-013`); that the
arms of this thesis attain a signal level at which the teacher becomes active; that `v2.0` will produce a
non-uniform curriculum under `RULEBOOK-V5.1` (`FIND-013` predicted uniform sampling under the superseded
reward, and no measurement exists under the current one — see `LIM-201`); and that the `v1.3` curriculum
harms policy performance (`FIND-007`, one seed, 40 episodes, and further weakened by `FIND-013`).

## 16. References

| Source | Exact concept supported |
|---|---|
| Graves et al., 2017, ICML, "Automated Curriculum Learning for Neural Networks" | learning progress as the reward of a bandit teacher over a fixed set of tasks |
| Matiisen et al., 2017, arXiv:1707.00183, "Teacher-Student Curriculum Learning" | windowed progress estimation per task; non-stationary bandit teacher; the visit-frequency coupling of `LIM-203` |
| Vargha and Delaney, 2000, J. Educ. Behav. Stat. 25(2), 101–132 | the `A` measure `P(better) + 0.5 P(tie)` and its no-effect value `0.5` (`REQ-004`) |
| Holm, 1979, Scand. J. Statist. 6(2), 65–70 | step-down family-wise correction, specified in revision 1 and **removed** in revision 2 (`REQ-005`, `FIND-010`); retained here for provenance |
| PORTAL, AAAI 2024, doi:10.1609/aaai.v38i14.29524 | selecting tasks matched to the agent's current competence |
| Nesterova et al., 2023, arXiv:2301.00691 (SITP) | algorithm-independent outcome signals driving a curriculum at low overhead |
| Jiang et al., 2021, arXiv:2010.03934 (PLR) | the staleness term of `REQ-011` only; the value-loss score is explicitly not adopted |
| Peng et al., 2024, IROS §III-A/III-B; Abouelazm et al., 2025 §III-E | the `v1`–`v1.3` lineage this version supersedes, retained for provenance |
| `docs/decisions/ADR-029` | the measured `+0.771…+0.943` LP/`|reward|` correlations and the naming of the principled fix |
| `docs/decisions/ADR-032`, `ADR-028`, `ADR-024`, `ADR-016` | carried-forward eligibility, coverage, data-abort, and vectorized-ordering contracts |
| `docs/specifications/rulebook_v5.1_specification.md`, `src/thesis_rl/rulebook/v2/aggregation.py` | the level margin/cost/applicability contract of `REQ-002` |
| `docs/audits/acl_v2_teacher_power_analysis_2026-07-31/` | `FIND-010`: measured scale invariance, closed-loop neutrality of a noisy arm, the removal of Holm, and the permutation cost. `FIND-012`: the window-size power sweep behind §9.1 and `DEC-205` |

## 17. Implementation Handoff Checklist

- [x] Scope, exclusions, and optional behavior are explicit.
- [x] Inputs and outputs define types, units, ranges, and prohibited information.
- [x] Formulas, algorithms, applicability, and fallbacks are unambiguous.
- [x] State, timing, reset, termination, truncation, and data-abort behavior are defined.
- [x] Configuration fields, frozen defaults, and rejected legacy fields are identified.
- [x] Errors, diagnostics, reproducibility, compatibility, and migration are covered.
- [x] Every core requirement maps to objective acceptance criteria.
- [x] Required validation categories are selected.
- [x] Scientific sources, project adaptations, and original constructions are kept distinct (§1.4).
- [x] Known limitations are intentional and do not hide missing requirements (§15.4).
- [ ] **One material decision remains open** — `DEC-201`…`DEC-206` are resolved and await approval as a
      set; `DEC-207` (not-at-fault truncated episodes excluded from the `T` window) is proposed and needs
      selection. The `SNR` measurement of `LIM-201` is a declared post-hoc validation and is explicitly
      **not** a gate on approval; under `RULEBOOK-V5.1` it additionally cannot be taken until a run under
      the new reward exists.
- [ ] Approval recorded, `_UNDER_REVIEW` removed from the filename, `docs/project_index.md` updated, ADR
      written, ExecPlan created.

## 18. Approval Record

- Approved by: `pending`
- Approval date: `pending`
- Approval evidence and provenance, stated precisely because revision 1 overstated it:
  - **2026-07-30** — the user presented the redesign and stated that the existing specifications are not
    binding if a better solution exists. `DEC-201`…`DEC-204` were **proposed by the assistant and selected
    by the user** in response to `FIND-006`…`FIND-009`. Revision 1 recorded these as "APPROVED", which was
    an overstatement of a selection made without full deliberation.
  - **2026-07-31** — the user **reopened** `DEC-201`…`DEC-204`, corrected the gate statistic for R2/R3
    (`FIND-011`), and requested a synthetic power analysis before any approval. The analysis was
    authorized, executed, and recorded as `FIND-010`.
  - **2026-07-31, revision 2** — `DEC-201` and `DEC-204` confirmed; `DEC-202` and `DEC-203` revised on
    evidence; `DEC-205` added. The user directed that `tau` remain unchanged pending evidence of
    under-concentration (§9.2).
  - **2026-09-01, revision 3** — re-based onto `RULEBOOK-V5.1` after its implementation was verified
    (`FIND-014`). The user selected the five-dimension order `L1 → L2 → L3 → T → L5` (`DEC-206`) from
    three stated alternatives, and directed that the specification be rewritten before the learnability
    runs rather than after them, so that §11's pre-registration requirement is satisfied. `DEC-207` is
    proposed and not yet selected. No mechanism, weight, or measured result was changed.
  - Approval of this document as the authoritative contract is **not yet given**.
- Approval notes: approval must precede any comparison run under this version (§11, pre-registration).
- Repository path: `docs/specifications/automatic_curriculum_learning_v2.0_specification_UNDER_REVIEW.md`
  (canonical path on approval: `docs/specifications/automatic_curriculum_learning_v2.0_specification.md`)
- Project index updated: `NO` (pending approval)
