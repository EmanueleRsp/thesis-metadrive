# ScenarioNet ACL with Outcome-Based Windowed Learning Progress

## Metadata

- Feature: ScenarioNet automatic curriculum learning, arm-level teacher
- Specification ID: `ACL-SN-EMA-001`
- Version: `v2.0`
- Status: `UNDER_REVIEW`
- Date: `2026-07-30`
- Supersedes: `docs/specifications/automatic_curriculum_learning_v1.3_specification.md`
- Related specifications: `docs/specifications/scenarionet_integration_v1.1_specification.md`,
  `docs/specifications/rulebook_v4.9_specification.md`,
  `docs/specifications/rl_baselines_v1_specification.md`,
  `docs/specifications/evaluation_protocol_v1.0_specification.md`
- Related ADRs: `docs/decisions/ADR-014-scenarionet-acl-learning-potential-only.md`,
  `docs/decisions/ADR-016-scenario-acl-vectorized-execution.md`,
  `docs/decisions/ADR-024-runtime-scenario-data-abort.md`,
  `docs/decisions/ADR-028-scenario-acl-generate-eligibility-renormalization.md`,
  `docs/decisions/ADR-029-acl-reward-scale-normalization.md`,
  `docs/decisions/ADR-030-acl-recorded-rationale-record.md`,
  `docs/decisions/ADR-032-acl-generate-catalog-decoupling.md`
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
`LIM-002` at its root.

The resulting feedback is then attenuated by an explicit **learnability gate** so that an arm that is
already solved and an arm that is entirely out of reach both converge to the neutral value, and only arms
of intermediate current difficulty receive a bonus.

### 1.3 Second, independent motivation: algorithm independence

Under `v1.3` PPO's teacher feedback is computed from GAE advantages and TD3/SAC's from TD residuals. These
are different quantities with different units, different variance, and different sensitivity to critic
quality, so the curriculum is *not the same curriculum* across the algorithm arm of the experimental
design. Under `v2.0` the teacher consumes only `success`, `route_completion`, and Rulebook macro-rule
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
Rulebook macro-rule costs, none uses a lexicographic priority ordering over safety dimensions to select
*which* dimension the progress is measured on, and none operates over a frozen finite curated scenario
catalog partitioned into six semantic arms. The specific combination — priority-ordered dimension
selection with a calibrated neutrality band, a Goldilocks gate on episode-level violation rate, and an
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

## 2. Scope

### In Scope

- six frozen ScenarioNet semantic arms `A0_simple_low_traffic` … `A5_critical_mixed`, as immutable
  partitions of the frozen training catalog;
- per-arm, per-dimension sliding observation windows over committed Generate episodes;
- the episodic outcome statistics derived from Rulebook v2 macro rules R1–R3 and from the task outcome;
- the ordinal progress measure, its exact conditional permutation neutrality band, and the
  multiplicity-corrected, priority-ordered dimension selection;
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
- Rulebook R4 (`route_progress`) as a separate teacher dimension (`RAT-203`);
- changes to reward scalarization, the observation schema, transition replay, or dataset splits;
- ablation of `H`, of `buffer_capacity`, or of the significance level (`LIM-203`, `LIM-204`);
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
| `H` | window half-size | `10`, frozen |
| `d` | teacher dimension | `R1`, `R2`, `R3`, `T` |
| `x_{d,e}` | episodic key of dimension `d` for episode `e`, **lower is better** | `REQ-002` |
| `v_{k,e}` | episode-level violation indicator of macro rule `k` | `{0, 1}` |
| `W_i^d` | FIFO of the last `2H` valid observations of dimension `d` on arm `i` | ordered oldest→newest |
| `O_i^d`, `R_i^d` | older half and recent half of `W_i^d` | `H` entries each |
| `G_{i,d}` | ordinal progress measure | `[0, 1]`, neutral `0.5` |
| `p^{perm}_{i,d}` | exact conditional two-sided permutation p-value of `G_{i,d}` | `(0, 1]` |
| `D_{i,d}` | learnability gate | `[0, 1]` |
| `d*` | selected dimension for this update | `R1|R2|R3|T` or none |
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
| Rulebook v2 exposes, per step, the macro-rule margins in `info["rule_reward_vector"]` with names in `info["rule_metadata"]["rule_names"]` | `src/thesis_rl/rulebook/v2/wrapper.py` | guaranteed upstream, runtime-validated | fatal if absent while ACL is enabled |
| Rulebook v2 exposes, per step, per-macro-rule applicability in `info["rule_components"][name]["applicable"]` | `src/thesis_rl/rulebook/v2/wrapper.py`, `aggregation.py` | guaranteed upstream, runtime-validated | fatal if absent while ACL is enabled |
| For R1–R3 the macro margin equals `-cost` with `cost` in `[0, 1]` | `aggregation.py` `RulebookResult.margins` | guaranteed upstream | fatal if a margin falls outside `[-1, 0]` |
| R1 (`collision_impact`) is `applicable=False` on every step without a new contact onset, and its cost is then `0` | `components/collision.py` | verified repository property | drives `REQ-002` R1 handling |
| At least one R3 subcomponent (`offroad`, `wrongway`) is applicable on every step | `components/road.py` | verified repository property | R3 falls back to the unobserved branch of `REQ-002` if violated |
| R2 subcomponents are applicable only when a relevant actor exists | `components/rss.py`, `rss_lateral.py`, `clearance.py`, `ttc.py` | verified repository property | drives `REQ-002` R2 handling |
| `success` and `route_completion` are available per episode | `agent/agent.py` episode accumulation | guaranteed upstream | fatal if missing |
| Vectorized completions are committed in deterministic `(collection_tick, worker_id, episode_id)` order | `ADR-016` | guaranteed upstream | non-deterministic resume; fatal on ordering violation |

**Explicit research limitation, not an assumption:** the six arms do not form a total difficulty order, so
no particular arm ordering over time is required, expected, or validated.

## 4. Inputs And Prohibited Information

| Input | Meaning/type | Shape/unit | Range | Source/validity | Missing-data behavior | Policy-visible |
|---|---|---|---|---|---|---|
| arm index | selected arm | scalar int | `[0, 5]` | frozen catalog | fatal configuration error | NO |
| per-step macro margins | `m_k(t)` for R1–R3 | 3 floats per step | `[-1, 0]` | Rulebook v2 monitor | fatal | NO |
| per-step macro applicability | `applicable` flag per macro rule | 3 bools per step | `{0,1}` | Rulebook v2 monitor | fatal | NO |
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
(`REQ-013` makes it diagnostic-only); the episode's own reward magnitude; Rulebook R4 as an independent
dimension.

## 5. Outputs

| Output | Meaning | Range | Consumer | Classification |
|---|---|---|---|---|
| mode | `GENERATE` or `REPLAY` | enum | driver | control |
| arm, scenario identity, selection probability | selection provenance | — | driver, logs | control + reproducibility metadata |
| `q_i`, `p_i` | arm scores and probabilities | `[0,1]` | teacher, monitor | diagnostic-only |
| `G_{i,d}`, `p^{perm}_{i,d}`, `D_{i,d}`, `d*`, `A_i` | teacher internals of each update | `[0,1]` / enum | logs | diagnostic-only |
| `C_{k,e}`, `v_{k,e}`, `N^{app}_{k,e}`, `F_{k,e}` | per-episode Rulebook statistics | `[0,1]` / counts | logs, windows | training signal (teacher) + diagnostic |
| `U`, `U_scaled`, `U_norm` | prediction-error learning potential and its `v1.3` transforms | `>= 0` / `[0,1]` | logs only | **diagnostic-only** (`REQ-013`) |
| per-arm coverage cycle id, within-cycle coverage fraction | catalog coverage | counts | logs | diagnostic-only |

No output of this specification is a policy observation or a training reward.

## 6. Functional Requirements

### REQ-001: Teacher dimensions and their priority order

- Required observable behavior: the teacher operates on exactly four dimensions, in this fixed priority
  order: `R1 = collision_impact`, `R2 = dynamic_interaction_safety`, `R3 = road_traffic_compliance`,
  `T = task`. The order is the Rulebook v2 lexicographic macro-rule priority followed by the task outcome.
- Applicability: always.
- Invariants: the order is frozen and is not configurable.
- Interactions: `REQ-005` selects one dimension per update from this order.
- Rationale: `RAT-203`. Rulebook R4 (`route_progress`) is deliberately excluded as an independent
  dimension because it measures progress along the route and is therefore largely redundant with the
  `route_completion` component of `T`.

### REQ-002: Episodic outcome statistics

For every valid Generate episode `e` on arm `i`, compute the following, using only steps of that episode.

**R1 (`collision_impact`).**

`C_{1,e} = max over all steps t of c_1(t)`, where `c_1(t) = max(0, -m_1(t))`.

R1 is **always observed**: every valid Generate episode appends an R1 observation. The maximum is taken
over all steps, not only applicable ones.

- Justification, verified against this repository: R1's applicability predicate is "a new contact onset
  occurred at this step" (`components/collision.py`), and its cost is `0` whenever it is not applicable.
  For R1, "not applicable" therefore means *satisfied*, not *unobserved*. Averaging over applicable steps
  would make the R1 window contain only collision episodes, which would measure injury severity
  conditional on a crash instead of collision behavior, would prevent a nearly-solved arm from ever
  refilling its R1 window, and would prevent its gate from ever closing. Recorded as `FIND-008`.

**R2 and R3 (`dynamic_interaction_safety`, `road_traffic_compliance`).**

Let `N^{app}_{k,e}` be the number of steps at which macro rule `k` is applicable. If `N^{app}_{k,e} = 0`
the dimension is **unobserved for this episode** and its FIFO is not updated — it is not silently recorded
as zero cost. Otherwise:

`C_{k,e} = ( sum over applicable t of c_k(t) ) / N^{app}_{k,e}`,  `c_k(t) = max(0, -m_k(t))`.

**Episode-level violation indicator, for all of R1–R3.**

`v_{k,e} = 1` if at least one step at which rule `k` is applicable has `c_k(t) > 0`, else `0`. For R1 this
is equivalent to `1[C_{1,e} > 0]`.

**Diagnostic-only violation frequency.**

`F_{k,e} = N^{viol}_{k,e} / N^{app}_{k,e}` when `N^{app}_{k,e} > 0`, else undefined. `F` is logged and
never consumed by the teacher.

**Task dimension `T`.** `T_e = (success_e, route_completion_e)`. Always observed.

- Invariants: `C_{k,e}` in `[0, 1]`; `route_completion_e` in `[0, 1]`; all values finite.
- Edge cases: an episode with zero environment steps is not a valid Generate episode and updates nothing.
- Failure behavior: a non-finite value, a margin outside `[-1, 0]` for R1–R3, or a missing applicability
  flag is fatal.

### REQ-003: Comparable keys and the sliding windows

Each dimension defines an episodic key with **lower is better**:

- `x_{k,e} = C_{k,e}` for `k` in `{R1, R2, R3}`;
- `x_{T,e} = (-success_e, -route_completion_e)`, compared lexicographically.

For every arm `i` and dimension `d` the teacher maintains a FIFO `W_i^d` of at most `2H` keys, ordered
oldest to newest, appended only by valid Generate episodes and only for dimensions observed in that
episode. `O_i^d` is the older half and `R_i^d` the newer half once `|W_i^d| = 2H`.

- Replay episodes never update any window. They are selected by the teacher itself and are therefore not
  unbiased observations of the arm's distribution (`RAT-205`).
- Dimension `d` is **available** for arm `i` iff `|W_i^d| = 2H`. `T` becomes available for every arm at the
  end of calibration (`REQ-008`); R1 becomes available at the same time; R2 and R3 become available when
  enough episodes have observed them.
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
- Holm's step-down correction is applied at family-wise level `significance_level = 0.05` over the
  dimensions **available for that arm at that update**, which is between 2 and 4 dimensions.
- `d*` is the **first dimension in the `REQ-001` priority order** whose Holm-corrected p-value rejects.
- If no dimension rejects, or if fewer than two dimensions are available, `d*` is undefined.

**Required framing, normative for all derived documentation.** This procedure is re-executed after every
valid Generate episode on the arm, over windows that overlap by `2H - 1` observations. It is therefore a
**calibrated neutrality band derived from the data's own permutation null**, not a hypothesis test with a
run-level type-I error guarantee. No document produced from this specification may state or imply a
significance claim at level `0.05` over a run. Its statistical role is that false rejections are symmetric
in sign under exchangeability, so they inflate the variance of `A_i` without biasing its expectation away
from `0.5`. Recorded as `RAT-204` and `LIM-202`.

### REQ-006: Learnability gate

For the selected dimension `d*` of arm `i`, computed over the **recent** window `R_i^{d*}` only:

- if `d*` in `{R1, R2, R3}`: `vbar = (1/H) * sum over e in R_i^{d*} of v_{d*,e}` and
  `D_{i,d*} = 4 * vbar * (1 - vbar)`;
- if `d* = T`: `rbar = (1/H) * sum over e in R_i^T of route_completion_e` and
  `D_{i,T} = 4 * rbar * (1 - rbar)`.

- Invariants: `D` in `[0, 1]`; `D = 0` at `vbar in {0, 1}` and at `rbar in {0, 1}`; `D = 1` at `0.5`. The
  factor `4` is analytic normalization of the maximum to `1` and is not a tuned parameter.
- Justification for using the episode-level violation rate rather than the mean cost, verified against this
  repository: R1's cost is a MAIS3+F injury probability and R3's cost is a graded off-road cost, so mean
  episodic costs occupy roughly `[0, 0.1]` in practice and `4C(1-C)` would never approach `1`. This would
  systematically attenuate the Rulebook dimensions relative to `T`, whose gate spans `[0, 1]` naturally.
  `vbar` spans `[0, 1]` by construction, equals `0` exactly when the arm no longer violates the rule and
  `1` exactly when it always does. Recorded as `FIND-009`.
- The comparison `G` deliberately keeps the fine-grained cost `C`, which is sensitive to partial
  improvement; only the gate uses the coarser `v`.
- `route_completion` rather than success rate is used for the `T` gate because a success rate of `0` early
  in training would close the gate on the task dimension for every arm simultaneously, whereas
  `route_completion` is informative from the first episode.

### REQ-007: Arm feedback

`A_i = 0.5 + D_{i,d*} * (G_{i,d*} - 0.5)` when `d*` is defined; `A_i = 0.5` otherwise.

- Invariants: `A_i` in `[0, 1]`; `A_i = 0.5` exactly when no dimension rejects, when the gate is closed, or
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

Calibration therefore costs `K * 2H = 120` valid Generate episodes. Replay becomes eligible when **both**
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
                compute C_1e, C_2e, C_3e, v_1e, v_2e, v_3e, T_e               # REQ-002
                append observed keys to W_i^d                                 # REQ-003
                N_gen[i] <- N_gen[i] + 1
                compute U, U_scaled, U_norm and log them                      # REQ-013

                if calibrated_before_this_episode:
                    Avail <- { d : |W_i^d| = 2H }
                    for d in Avail:
                        G[d]    <- VarghaDelaney(R_i^d, O_i^d)                # REQ-004
                        pperm[d]<- ExactPermutationTwoSided(R_i^d, O_i^d)     # REQ-005
                    Rej  <- Holm(pperm, level=0.05)
                    d*   <- first d in (R1,R2,R3,T) with d in Rej
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
(`210` at `H = 10`). Enumerate, by dynamic programming over the tie groups, the counts `f[S]` of `H`-subsets
of the pooled multiset achieving each doubled rank sum; `sum_S f[S] = C(2H, H) = 184756`. Then

`p^{perm} = ( sum over S with |S - E2| >= |S_obs - E2| of f[S] ) / C(2H, H)`.

All comparisons are integer, so the result is exact and reproducible across platforms, and the procedure
consumes no random numbers. When both windows are constant and equal — the common case of an arm with no
violations of a rule — `f` is concentrated on a single `S` and `p^{perm} = 1`, which correctly reports "no
detectable change".

**Numerical contract.** Every quantity is finite. `C in [0,1]`, `v in {0,1}`, `route_completion in [0,1]`,
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
| `progress.window_half_size` | int | 10 | `>= 5` | `H` | YES | YES |
| `progress.significance_level` | float | 0.05 | `(0, 0.5)` | Holm family-wise level | YES | YES |
| `progress.multiplicity_correction` | string | `holm` | exactly `holm` | correction method | YES | YES |
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

**Rejected legacy fields.** `replay_sampling.beta` and `mab.feedback: rank_normalized_learning_potential`
are removed from the contract. A configuration containing either fails validation before learner
construction with an explicit message naming this specification; they are never silently ignored. The
legacy cumulative fields `weight_clip_*` and `initial_weight_decay` remain rejected as in `v1.3`.

`H = 10` is chosen to reuse the existing EMA horizon `1/alpha` rather than introduce an independently
tuned parameter. This is a **parsimony argument, not a derivation**: the EMA horizon governs score
smoothing, not the resolution of the progress measurement. It is recorded as such in `RAT-202` and its
consequence for statistical power is recorded in `LIM-201`.

## 10. Errors, Logging, And Diagnostics

**Fatal:** non-finite statistic, margin outside `[-1, 0]` for R1–R3, missing Rulebook applicability or
margin data while ACL is enabled, invalid probability vector, wrong arm count, mutation configuration,
incompatible checkpoint schema, rejected legacy configuration field, a Generate draw on a record reported
eligible but absent from the catalog, an arm unable to complete calibration, and a Replay selection with an
empty buffer and no Generate-eligible arm.

**Recorded degradation, not an error:** an empty Generate-eligible arm set with a non-empty buffer; a
concurrent eviction of a record before its in-flight Replay episode commits
(`skipped_evicted_before_commit`, unchanged from `v1.3`).

**Required per-committed-episode log fields:** mode, arm, scenario UID, coverage cycle id,
`C_{k,e}`/`v_{k,e}`/`N^{app}_{k,e}`/`F_{k,e}` for R1–R3, `success`, `route_completion`, buffer action, and
the diagnostic `U`, `U_scaled`, `U_norm`.

**Required per-teacher-update log fields:** available dimensions, `G_{i,d}` and `p^{perm}_{i,d}` for each,
the Holm rejection set, `d*`, `D_{i,d*}`, `A_i`, `q_i` before and after, `p_i`, and the update count.

**Required transition logs, once per transition rather than once per episode:** dimension availability
changes, calibration completion, coverage-cycle closures, and Generate-eligibility changes.

Every value in §5 marked diagnostic-only must be classified as such wherever it is surfaced. Rulebook
diagnostics remain diagnostic-only with respect to the reward and the policy observation; their use as
teacher inputs here is a curriculum input and does not make them policy-visible.

## 11. Reproducibility And Compatibility

The specification ID and version, the resolved YAML, dataset and catalog hashes, the seed, and full
selector state are persisted. `H`, the significance level, and the correction method are part of the
recorded configuration.

**Breaking changes relative to `v1.3`:**

- checkpoint schema `acl_ema_v3` → `acl_progress_v1`; no migration path, restart required;
- `ScenarioRecord` gains `last_generate_step` and loses the meaning of `usefulness`/`usefulness_norm`/
  `rank` as ranking inputs, so persisted `v1.3` buffers cannot be reused;
- the RNG call order changes: the calibration phase consumes at most one draw per selection instead of a
  coin flip plus an arm draw, so `v1.3` seeds do not reproduce their runs;
- `replay_sampling.beta` and the previous `mab.feedback` value are rejected.

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
  whose variance is varied over at least two orders of magnitude.
- When: `G` is computed for each draw over many seeded repetitions.
- Then: the sample mean of `G` is `0.5` within Monte-Carlo tolerance for every variance level, and the
  ordering of mean `G` across variance levels shows no monotone trend.
- Related requirements: `REQ-004`. This is the direct regression for `LIM-002`.

### AC-202: Monotone response to genuine improvement

- Given: an older window from distribution `P_old` and a recent window from `P_new` stochastically
  dominating it in the "better" direction, at three separation levels.
- When: `G` is computed.
- Then: `G > 0.5` in all three cases and increases monotonically with separation; with the two windows
  exchanged, `G < 0.5` symmetrically.
- Related requirements: `REQ-004`.

### AC-203: R1 statistic is defined for collision-free episodes

- Given: an episode in which no step has a new contact onset, so every R1 step is `applicable=False`.
- When: the episodic statistics are computed.
- Then: `C_{1,e} = 0`, `v_{1,e} = 0`, the R1 window receives the observation, and the episode counts toward
  R1 availability. Given instead an episode with one contact of cost `0.4` and one of cost `0.7`,
  `C_{1,e} = 0.7` and `v_{1,e} = 1`.
- Related requirements: `REQ-002`, `FIND-008`.

### AC-204: R2/R3 unobserved episodes do not become zero-cost observations

- Given: an episode in which macro rule R2 is applicable at zero steps.
- When: the episodic statistics are computed.
- Then: no R2 observation is appended, `|W_i^{R2}|` is unchanged, and R2 availability is unaffected;
  `C_{2,e}` is reported as undefined, never as `0`.
- Related requirements: `REQ-002`.

### AC-205: Exact permutation band

- Given: two windows of `H = 10` identical constant values.
- When: the permutation p-value is computed.
- Then: `p^{perm} = 1.0` exactly, `G = 0.5` exactly, and no dimension is selected. Given instead two
  windows with no ties and complete separation, `p^{perm} = 2 / C(20,10)` exactly and `G` is `0` or `1`.
  In both cases the computation consumes zero random numbers and is bit-identical across repeated calls.
- Related requirements: `REQ-005`.

### AC-206: Priority-ordered selection under Holm

- Given: constructed windows in which R1 is neutral, R2 rejects, and `T` rejects with a smaller p-value
  than R2, with all four dimensions available.
- When: the dimension is selected.
- Then: `d* = R2`, not `T`, and the Holm correction is applied over exactly the available dimensions.
  Given instead windows in which no dimension rejects, `d*` is undefined and `A_i = 0.5` exactly.
- Related requirements: `REQ-001`, `REQ-005`.

### AC-207: Gate closes at both extremes

- Given: a selected Rulebook dimension whose recent window has `vbar = 0`, then `vbar = 1`, then
  `vbar = 0.5`, each combined with `G = 0.8`.
- When: `A_i` is computed.
- Then: `A_i = 0.5`, `0.5`, and `0.8` respectively. The same holds for `T` with `rbar` in `{0, 1, 0.5}`.
- Related requirements: `REQ-006`, `REQ-007`.

### AC-208: Calibration completeness and Replay gating

- Given: a seeded run from a fresh state.
- When: selections proceed until calibration completes.
- Then: every selection before completion is Generate, no `q_i` changes from `0.50`, each arm reaches
  exactly `2H` valid Generate episodes, the per-arm counts differ by at most one at every intermediate
  point, `T` and `R1` are available for every arm at completion, and the first Replay cannot occur before
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
| regressions for known bugs | required (`FIND-006`…`FIND-009`) |
| mutation as a prohibited feature | required, unchanged |

## 14. Traceability

| Requirement | Acceptance criteria | Scientific source or approved decision |
|---|---|---|
| `REQ-001` | `AC-206` | Rulebook v2 macro priority; project adaptation `RAT-203` |
| `REQ-002` | `AC-203`, `AC-204` | Rulebook v2 `aggregation.py` contract; `FIND-008` |
| `REQ-003` | `AC-211` | Matiisen et al. 2017 windowed progress; `RAT-205` |
| `REQ-004` | `AC-201`, `AC-202` | Vargha and Delaney 2000; Graves et al. 2017; resolves `LIM-002` |
| `REQ-005` | `AC-205`, `AC-206` | exact permutation null; Holm 1979; `RAT-204` |
| `REQ-006` | `AC-207` | competence-based curriculum (PORTAL 2024; SITP 2023); `FIND-009` |
| `REQ-007` | `AC-207`, `AC-209` | project construction; `RAT-207` |
| `REQ-008` | `AC-208` | project construction, derived from `K` and `H` |
| `REQ-009` | `AC-209` | ACL `v1.1` `REQ-001`/`REQ-002`; `ADR-028`; `RAT-201` |
| `REQ-010` | `AC-210`, `AC-213` | `ADR-029` `DEC-007` channel analysis; `RAT-206` |
| `REQ-011` | `AC-210` | Jiang et al. 2021 staleness; project construction for the progress term |
| `REQ-012` | `AC-213` | `ADR-032` `DEC-008`/`DEC-009`, carried forward unchanged |
| `REQ-013` | `AC-212`, `AC-215` | `v1.3` `LIM-006`; user decision 2026-07-30 |
| `REQ-014` | `AC-214` | `v1.3` `REQ-005`; `ADR-016` |

## 15. Open Decisions And Limitations

### 15.1 Decisions resolved before review

| ID | Question | Resolution | Status |
|---|---|---|---|
| `DEC-201` | How is R1's episodic cost defined, given that R1 is `applicable=False` without a contact onset? | Maximum over all steps, R1 always observed; "not applicable" means satisfied (`REQ-002`) | APPROVED 2026-07-30 |
| `DEC-202` | What does the Goldilocks gate consume? | The episode-level violation rate `vbar` over the recent window; `G` keeps the fine-grained cost `C` (`REQ-006`) | APPROVED 2026-07-30 |
| `DEC-203` | How is the neutrality band framed and corrected? | Exact conditional permutation band with Holm at `0.05`, documented as a calibrated deadband and never as a run-level significance claim (`REQ-005`) | APPROVED 2026-07-30 |
| `DEC-204` | Is the prediction-error learning potential removed or retained? | Retained as a strictly inert diagnostic channel, so the two signals are measurable within one run (`REQ-013`) | APPROVED 2026-07-30 |

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
- **`FIND-009`** (2026-07-30, code verification) — R1's cost is a MAIS3+F injury probability and R3's is a
  graded off-road cost, so mean episodic costs occupy roughly `[0, 0.1]`. A gate `4C(1-C)` on the mean cost
  would attenuate the Rulebook dimensions to at most `~0.4` while the task gate spans `[0, 1]`, an
  unintended scale artifact. Corrected by `DEC-202`.

### 15.3 Recorded rationales

- **`RAT-201`** — No inverse-probability correction on the EMA update, carried forward from `v1.1`
  `RAT-004`: the exploration floor bounds the propensity away from zero and the correction's variance cost
  is not justified at `K = 6`.
- **`RAT-202`** — `H = 10` reuses `1/alpha` as a parsimony choice, not as a derivation. See §9.
- **`RAT-203`** — Rulebook R4 (`route_progress`) is not an independent teacher dimension because it
  measures progress along the route and would be largely redundant with `route_completion` inside `T`,
  adding a fourth correlated test to the multiplicity correction for no additional information.
- **`RAT-204`** — The permutation band is a calibrated deadband, not an inference procedure. Its purpose is
  to obtain a neutrality threshold from the data's own null rather than by choosing a number while looking
  at run performance. See `REQ-005` and `LIM-202`.
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

- **`LIM-201` — Statistical power at `H = 10`.** With `H = 10` per window and Holm over up to four
  dimensions, the effective level at the R1 position is `0.05 / 4` and the power to detect a moderate
  effect (`G ≈ 0.8`) is well below `50%`. **The expected operating behavior of this design is therefore a
  curriculum that stays close to uniform for much of a run and departs from it only on clear evidence.**
  This is stated in advance as predicted behavior, not discovered after the fact. Given `FIND-006` and the
  prior over-concentration this version corrects, a conservative default is the intended risk posture, but
  it must be reported as such.
- **`LIM-202` — No run-level type-I guarantee.** The band is re-evaluated after every valid Generate
  episode on overlapping windows, so no significance claim holds over a run. False rejections are symmetric
  in sign under exchangeability and therefore inflate the variance of `A_i` without biasing its
  expectation. See `REQ-005`.
- **`LIM-203` — Visit-frequency coupling.** The windows are measured in per-arm episodes, not in global
  training steps. A frequently sampled arm measures progress over a shorter stretch of training and tends
  to look stationary; a rarely sampled arm measures over a longer stretch and tends to look improving. The
  coupling is anti-concentrating and therefore acts in the corrective direction here, and the exploration
  floor bounds its effect, but it is a structural property of window-based progress measures — Matiisen
  et al. (2017) share it — and it is not removed by this version.
- **`LIM-204` — No ablation of `H`, `buffer_capacity`, or the significance level.** These are reasoned
  choices, not ablated ones. No experiment in this thesis isolates their effect and none is planned.
- **`LIM-205` — Calibration cost dominates short profiles.** Calibration requires `120` valid Generate
  episodes, roughly `2–4%` of `run_profile=thesis` (1.5M steps) but `25–50%` of `run_profile=fast`
  (120k steps). **Curriculum-selection claims must not be derived from `fast`-profile runs under `v2.0`.**
- **`LIM-206` — The scenario buffer is no longer prioritized level replay.** With arm-balanced admission
  and `P_progress = p_i / n_i`, the buffer is a recency-balanced revisit memory whose only per-record term
  is staleness. Only that term retains PLR parentage; the buffer must not be described as PLR in derived
  documentation.
- **`LIM-207` — The commit-path closure still has no isolated harness.** Carried forward from `v1.3`
  `LIM-004`: `commit_event` is a private closure inside the vectorized training loop. `AC-210`/`AC-211`
  cover the buffer-level and teacher-level contracts, but the branch selection inside the closure remains
  verified by code reading plus the passing suite.
- **`LIM-208` — Per-step teacher instrumentation cost.** `REQ-002` requires per-slot accumulation of
  macro-rule costs and applicability at every environment step of the vectorized loop. The data is already
  present in the step info, but the accumulation is new work in the hot loop; its cost is bounded by three
  float comparisons and three counter increments per slot per step and must be measured in the ExecPlan's
  smoke run, not assumed negligible.

### 15.5 Claims this specification supports

1. The teacher's feedback is no longer structurally proportional to the variance of a prediction-error
   signal (`AC-201`).
2. A noisy but stationary arm receives a neutral expected feedback of `0.5` (`AC-201`).
3. An arm that is already solved on the selected dimension, and an arm entirely out of reach on it, are
   both attenuated to neutral (`AC-207`).
4. The teacher is algorithm-independent: identical outcomes produce identical curricula under PPO, TD3, and
   SAC, and therefore across the four reward settings (`AC-215`).
5. The neutral state of the curriculum is uniform sampling (`AC-209`).
6. Whether this improves sample efficiency or final performance is an open experimental hypothesis, to be
   tested against `curriculum=disabled` under `EVAL-PROTOCOL` v1.0 (§1.5).

## 16. References

| Source | Exact concept supported |
|---|---|
| Graves et al., 2017, ICML, "Automated Curriculum Learning for Neural Networks" | learning progress as the reward of a bandit teacher over a fixed set of tasks |
| Matiisen et al., 2017, arXiv:1707.00183, "Teacher-Student Curriculum Learning" | windowed progress estimation per task; non-stationary bandit teacher; the visit-frequency coupling of `LIM-203` |
| Vargha and Delaney, 2000, J. Educ. Behav. Stat. 25(2), 101–132 | the `A` measure `P(better) + 0.5 P(tie)` and its no-effect value `0.5` (`REQ-004`) |
| Holm, 1979, Scand. J. Statist. 6(2), 65–70 | step-down family-wise correction (`REQ-005`) |
| PORTAL, AAAI 2024, doi:10.1609/aaai.v38i14.29524 | selecting tasks matched to the agent's current competence |
| Nesterova et al., 2023, arXiv:2301.00691 (SITP) | algorithm-independent outcome signals driving a curriculum at low overhead |
| Jiang et al., 2021, arXiv:2010.03934 (PLR) | the staleness term of `REQ-011` only; the value-loss score is explicitly not adopted |
| Peng et al., 2024, IROS §III-A/III-B; Abouelazm et al., 2025 §III-E | the `v1`–`v1.3` lineage this version supersedes, retained for provenance |
| `docs/decisions/ADR-029` | the measured `+0.771…+0.943` LP/`|reward|` correlations and the naming of the principled fix |
| `docs/decisions/ADR-032`, `ADR-028`, `ADR-024`, `ADR-016` | carried-forward eligibility, coverage, data-abort, and vectorized-ordering contracts |
| `docs/specifications/rulebook_v4.9_specification.md`, `src/thesis_rl/rulebook/v2/aggregation.py` | the macro-rule margin/cost/applicability contract of `REQ-002` |

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
- [ ] **No material decision remains open** — `DEC-201`…`DEC-204` are resolved; user approval of this
      document as a whole is still pending.
- [ ] Approval recorded, `_UNDER_REVIEW` removed from the filename, `docs/project_index.md` updated, ADR
      written, ExecPlan created.

## 18. Approval Record

- Approved by: `pending`
- Approval date: `pending`
- Approval evidence: session of 2026-07-30 — the user presented the redesign, stated that the existing
  specifications are not binding if a better solution exists, and approved `DEC-201`…`DEC-204` in response
  to the review findings `FIND-006`…`FIND-009`. Approval of this document as the authoritative contract is
  **not yet given**.
- Approval notes: approval must precede any comparison run under this version (§11, pre-registration).
- Repository path: `docs/specifications/automatic_curriculum_learning_v2.0_specification_UNDER_REVIEW.md`
  (canonical path on approval: `docs/specifications/automatic_curriculum_learning_v2.0_specification.md`)
- Project index updated: `NO` (pending approval)
