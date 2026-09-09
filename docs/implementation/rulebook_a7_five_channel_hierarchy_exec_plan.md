# ExecPlan — A7 five-channel rulebook architecture and `γ = 0.9982` (`A7`)

## 1. Metadata

| Field | Value |
|---|---|
| Feature | Replace the six-level `RULEBOOK-V5.1` hierarchy with the A7 five-channel architecture — progress last and unthresholded, `L6` deleted, the negotiable-lane channel charged by a bounded satisfaction indicator — and move the shared discount to `γ = 0.9982` |
| Plan ID | `A7` |
| Authoritative specification for the architecture being replaced | `docs/specifications/rulebook_v5.1_specification.md` (`RULEBOOK-V5.1`, `APPROVED` 2026-08-14, amended 2026-08-20 and 2026-09-07), `SCAL-V1.4` (§5) |
| Authoritative specification for A7 | **None. It does not exist yet, and writing it is `DEC-A7-001` and milestone `M2` of this plan.** No production implementation may begin before it is approved |
| Evidence of record | `docs/audits/rulebook_architecture_2026-09-09/` (the candidate bench, the impossibility results, the recommendation and its derivations) and `docs/audits/progress_channel_integrity_2026-09-09/` (authoritative wherever the two disagree) |
| Status | `AWAITING_DECISIONS` |
| Created | 2026-09-09 |
| Last updated | 2026-09-09 |
| Branch | `worktree-a7-execplan`, from `main` at `3e58ce0` |
| Related ADRs | ADR-072 (partly reverted: the negotiable lane rules return above progress), ADR-076 (`L6`, deleted), ADR-075 and ADR-081 (the discount), ADR-063…ADR-071 (the sub-rules, untouched), ADR-035 and ADR-053 (context for `C50`/`D14`). **A new ADR is required** for the architecture and the discount; the next free number is `ADR-083` |
| Owner | Single maintainer; there is no reviewer to assign (`AGENTS.md`, Branching And Pull Requests) |

**What is approved, and by whom.** The user approved **the A7 architecture** and
**`γ = 0.9982`** on 2026-09-09. Nothing else in this document is approved.
`w₅ = 0.15`, `φ = 0`, the removal of `η` and `λ₆`, every threshold value, and
every change to a mandatory test are this plan's content and are recorded in §6
as gates.

**`λ₄ = 2.0` stands** and is not proposed for change (§6, `DEC-A7-010`).

---

## 2. Objective And Scope

### Observable capability

After this plan the reward a learner optimizes has **five channels and four free
weights** instead of six and six:

```
K1  collision safety           at-fault impact                        threshold 0
K2  interaction risk           ttc, clearance, rss_lateral            threshold from the panel
K3  non-negotiable compliance  offroad, signal, stop, crosswalk,       threshold from the panel
                               vehicle_yield, speed_limit             — and it cannot be 0
K4  negotiable lane compliance solid_line, wrong_carriageway,          threshold from the panel
                               dashed_line
K5  mission progress           signed route advance                   UNTHRESHOLDED, last
```

and every algorithm configuration shares `γ = 0.9982`, which is the first
four-decimal value satisfying the discount criterion at the **measured** training
horizon of 500 control steps rather than at the Waymo-only 199 the criterion was
evaluated against.

### Why it is needed

Three independent reasons, each measured rather than asserted:

1. **The shipped discount does not satisfy its own criterion.** `ln(a)/−ln(γ) > L`
   at `a = 2.5, γ = 0.996` gives a break-even of **228.6** control steps, and
   **591 of 2200 training records (26.86 %)** run longer, so inside those episodes
   a future higher-priority violation is damped below a present lower-priority one
   — the inversion the discount was chosen to prevent. Verified chain from the
   frozen index to the truncation rule in `docs/open_items.md` `C49`.
2. **Ordering O3 is currently lost.** A trajectory taking the legal route loses to
   an illegal shortcut on the scalar arm at the shipped discount, margin
   **−3.5789** where the undiscounted value is **+0.2000** (`D15`,
   `TEST-RB5.1-16c`). Under A7 it holds again at both discounts (+0.581 at
   `γ = 0.996`, +2.458 at `γ = 0.9982`).
3. **Two of the six weights cannot be calibrated.** `η` is the one weight the
   expert panel cannot discriminate — the logged human almost never relaxes — and
   `λ₆`'s value is an open decision (`D15`). A7 deletes both channels that carry
   them, so the reward stops carrying two different per-step scalings.

The object is a **reduction in complexity**: fewer channels, fewer weights, one
fewer per-step normalization, and one fewer constrained channel in a set the
thresholded literature does not know how to satisfy jointly (§7.4).

### How success is recognized

| # | Criterion | Instrument |
|---|---|---|
| 1 | The expert panel's **per-episode** exposure distributions exist, so every A7 budget is a measurement rather than a choice | `M1` |
| 2 | `fraction_below_standstill` under the A7 reward is **measured** at or below `AC-RB5.1-04`'s 7.45 % ceiling, not projected | `M1` |
| 3 | Production's five-channel vector equals the offline instrument on every step of the frozen Waymo `train` panel to `1e-9` | `M7`, the `T-RB51-12` pattern |
| 4 | The ordering battery holds under A7, and every strict-lexicographic failure is reported with the channel that caused it | `M3`, `M7` |
| 5 | The discount criterion's verdict is asserted next to the value that makes it true | `M6` |

### In scope

The five-channel hierarchy and its aggregations; the deletion of the `L6`
`progress_rate` level and its `advance_shortfall` sub-rule; the scalar adapter
(`SCAL-V1.5`) and its §5.4 rank-preservation predicate; the discount on every
algorithm configuration and its guard; the budgets `τ₁`–`τ₄` as *values*; the
A7 specification document; the amendment of the `AB-LEARN` pre-registration; the
`RB51` disposition; the measurement additions to
`scripts/measure_expert_rulebook_transition.py` that `M1` needs.

### Out of scope

- **The thresholded-lexicographic algorithm and the mechanism that enforces a
  budget.** `D1` records that the object a threshold applies to differs by
  mechanism, and `I1b` proves an episodic budget has no per-state equivalent. This
  plan fixes the budget *values* from the panel; the arm that consumes them is
  separate, later work.
- **The distributional arm.**
- **`C50`** — the negative-clip ratchet. Decided 2026-09-09: change nothing in the
  reward. Carried here as a declared limitation (§15).
- **`D14`** — the parallel-corridor exposure. Decided 2026-09-09: no remedy,
  recorded as an observation. Carried here as a declared limitation (§15).
- **`C52`** — ADR-035 asserts a continuity bound production has not had since
  2026-08-03. The ADR is approved, so amending it is the user's decision.
- **`RB51`'s `M9`** (multi-hop route reachability). It amends
  `rulebook_v4.11` §2.9.5 and touches no A7 channel; it detaches rather than
  riding this plan (§6, `DEC-A7-009`).
- Retraining, algorithm selection, observation or encoder changes, and any
  decision about GPU time.

### Compatibility

- **Checkpoint compatibility breaks**, by three independent routes: the margin
  vector changes shape from 6 to 5, the vector schema id changes, and the weight
  set changes. All three reach the checkpoint reward-semantics identity.
- **The recorded artifact schema breaks.** `MacroRule`'s *values* are recorded in
  CSVs, evaluation artifacts and analysis tables, so deleting `progress_rate` and
  re-ordering the levels is an output-contract change. This is the same class of
  break `DEC-RB51-005` approved on 2026-08-20 for the same enum, with the same
  reasoning: a level list that no longer matches the hierarchy causes later
  misreadings.
- **No new observation field**, and the §3.4 atomic vector is unchanged: fourteen
  entries, thirteen sub-rule costs plus `Δq`. A7 changes the comparison order and
  the scalar adapter, not what is measured.
- **`AB-LEARN`'s pre-registered arm B moves**, because its reward is the object A7
  replaces (§6, `DEC-A7-008`).

---

## 3. Authoritative Requirements

A7 has no approved specification, so this table cannot cite one, and it must not
pretend to. Each row therefore names the **ground** it actually rests on:
`mechanism` (algebra, an invariant, dimensional analysis), `measurement` (with its
source), `literature` (with its citation), or `approved decision` (the user's, of
2026-09-09). `AGENTS.md`'s Scientific Argument Standards forbid the fifth
possibility — citing this repository's own specification as evidence that a choice
is right — and §6 records the two places where that is nevertheless the only
argument available, which is a finding rather than a justification.

Rows marked **→ spec** become text in the A7 specification document under
`DEC-A7-001`.

| ID | Requirement | Ground |
|---|---|---|
| `REQ-A7-01` | Five channels in the order `K1 ≻ K2 ≻ K3 ≻ K4 ≻ K5`, with the sub-rule membership of the §2 block. **→ spec** | Approved decision, 2026-09-09. Why progress is *last*: Vamplew, Dazeley, Berry, Issabekov & Dekker (Machine Learning 84, 2011) §3.2.3 p. 58 — "objective n will be unconstrained, hence `C_n = +∞`"; Censi et al. Fig. 10 place own progress at the bottom of their rulebook and Definition 17 adds new rules there |
| `REQ-A7-02` | The §3.4 atomic cost vector is unchanged — fourteen entries, thirteen sub-rule costs plus `Δq` — and no observation field is added. **→ spec** | Mechanism: A7 changes the comparison order and the scalar adapter, both of which §3.4 already declares to be adapters rather than the contract |
| `REQ-A7-03` | `K4` aggregates its three sub-rules by normalized sum with a fixed denominator of 3, exactly as `L5` does today. **→ spec** | Mechanism: the intra-level aggregation is unchanged by a change of position. Measured corollary: `dashed_line` 1145 violated steps and `solid_line` 758 total 1903 against the channel's 1897, so six steps carry two sub-rules at once — the sum is almost unexercised but not never |
| `REQ-A7-04` | `K5` is **unthresholded** and carries the bare signed advance `Δq = clip(Δs/D_REF, −1, +1)`, `D_REF = v_ref·Δt = 2.2222 m`, `ΔQ_MAX = 1`. **→ spec** | Mechanism: `I3` proves a threshold on the progress channel has an *empty* requirement — `P3` needs `τ ≤ 27.97` and `P9` needs `τ > 27.97` on the §4.6 reference pair, at the same critical value, because at `γ < 1` the two comparisons are the same comparison at that channel |
| `REQ-A7-05` | The `L6` `progress_rate` level and its `advance_shortfall` sub-rule are deleted, and the `Δt/T_REF` per-step scaling disappears with `η` and `λ₆`. **→ spec** | Measurement: `c_L6 = 1 − clip(Δq,0,1)` fires on **99.4024 %** of expert steps at `p50 = 0.884` and is a pointwise function of the progress channel, so its return distribution carries nothing beyond its mean once progress is known. Mechanism: with progress last, the degenerate "relax as little as possible" objective `L6` was introduced to prevent is unreachable |
| `REQ-A7-06` | `K4` is charged by a bounded satisfaction indicator with a severity slope, at weight `w₅`, inside the priority block as a sub-unit level. **→ spec** | Mechanism: this is what makes the negotiable channel decide *before* progress in the ordered arms and gives it a finite exchange rate in the scalar arm — `I5` proves both are needed together and only a scalarization supplies the second |
| `REQ-A7-07` | Four free weights `(a, σ, λ₄, w₅)` at `a = 2.5`, `σ = 0.30`, `λ₄ = 2.0`; `η`, `λ₆` and `φ` are removed. **→ spec** | `a`, `σ`, `λ₄`: approved and measured (ADR-081). `η`, `λ₆`, `φ`: §6 gates `DEC-A7-003` and `DEC-A7-004` |
| `REQ-A7-08` | The rank-preservation predicate is restated for the A7 tail and enforced at construction, refusing inadmissible weights rather than pricing them. **→ spec** | Mechanism: below the condition, one step of progress or of relaxation overturns a higher-level violation, so the weights stop being an ordering. Re-derived independently in §5 |
| `REQ-A7-09` | One shared discount `γ = 0.9982` on every algorithm configuration, with `learning_potential_gamma` equal to it, and the criterion's **verdict** asserted beside it. **→ spec** | Approved decision, 2026-09-09. `γ ≥ exp(−ln a / 500) = 0.998169` is mechanism given the measured `L = 500`. The shaping coupling is Ng, Harada & Russell: potential-based shaping is policy-invariant only when its discount is the MDP's |
| `REQ-A7-10` | The budgets `τ₁`–`τ₄` are fixed by the logged expert's **per-episode** exposure distribution, denominated per unit of mission span, and a budget that fails to admit the human is falsified. **→ spec** | Mechanism plus measurement: `Q` is a scenario constant the agent cannot influence, so `X/Q` is duration-invariant and comparable across a panel whose missions span 13–247 m. A zero budget on `K3` is falsified by the panel — the expert violates it on **0.6418 %** of steps, 92.4 % of that `offroad` |
| `REQ-A7-11` | `fraction_below_standstill` under A7 is measured on the frozen Waymo `train` panel and does not exceed 7.45 % | Measurement: it is the one acceptance column no algebra derives, because the standstill baseline itself moves when `λ₆` is deleted (§5) |
| `REQ-A7-12` | Production's five channels equal the offline instrument on every step of the frozen panel to `1e-9` | Mechanism: the instrument is what every published figure was derived from, so agreement is what converts those figures from claims about a script into claims about production. This is `RB51`'s own `T-RB51-12` discipline, which passed at `0.0` divergence on four sub-rules and `5.97e-06` on `offroad` |
| `REQ-A7-13` | The ordering battery holds under A7 in the scalar arm, and every strict-lexicographic failure is reported with the deciding channel rather than repaired | Measurement: `g3_battery.py` scores ten orderings for A7 as passing under both the scalar rule and strict lexicographic comparison. Mechanism for why failures are reported: `I2` — if any channel is non-zero with positive probability on every moving trajectory, the do-nothing trajectory is optimal under strict lexicographic comparison, and the interaction channel fires on 0.3978 % of expert steps |
| `REQ-A7-14` | The `AB-LEARN` pre-registration is amended explicitly, before any screening run under A7 | Mechanism: `AB-LEARN` §7.5 point 3 — the freeze gate "is legitimate only because it was pre-registered", so a reward that moves must move *before* the runs that decide the hypotheses, and be recorded as having done so |
| `REQ-A7-15` | The limitations A7 does not remove are declared, not implied: `P11`, collide-to-escape on the scalar arm, `C50`, `D14`, and the thresholded arm's two open mechanism routes | Measurement, each with its figure in §15 |

---

## 4. Current Repository Analysis

Every statement is labelled. `VERIFIED` means read in the tree at `3e58ce0`
today; `INFERRED` marks a consequence not directly executed, and an inference
that affects behaviour is an approval gate.

### 4.1 What production actually is

| Fact | Evidence | Label |
|---|---|---|
| Production runs the six-level scalarization, not `SCAL-V1.1` | `conf/scalarization/default.yaml`: `mode: six_level_priority_weighted_rank`, `vector_schema_id: rulebook_v5_1_six_level_v1`, `priority_base: 2.5`, `severity: 0.30`, `flat_tie_breaker: 0.25`, `progress_weight: 2.0`, `relaxable_weight: 1.0`, `progress_rate_weight: 0.2`, `step_dt_s: 0.1`, `reference_time_s: 1.0` | `VERIFIED` |
| Six levels are implemented and named as the specification names them | `src/thesis_rl/rulebook/v2/types.py:99-134` — `MacroRule` with `COLLISION_SAFETY`, `INTERACTION_RISK`, `NON_RELAXABLE_COMPLIANCE`, `MISSION_PROGRESS`, `RELAXABLE_LANE_COMPLIANCE`, `PROGRESS_RATE`; `MACRO_RULE_ORDER` is the single source of the ordering and `COST_MACRO_RULES` separates the five costs from the one utility | `VERIFIED` |
| The scalar adapter implements §5.1 term by term | `src/thesis_rl/reward/scalarization.py:393-425` | `VERIFIED` |
| Every algorithm configuration shares `γ = 0.996`, and the two shaping discounts track it | `conf/agent/planner/algorithm/{ppo,ppo_sb3,sac,sac_sb3,td3,td3_sb3}.yaml`; seven `gamma:` lines across six files, two of them `learning_potential_gamma` | `VERIFIED` |

**Two statements the handoff carried are wrong, and both come from the same
stale source.** They are corrected here because sequencing depends on them.

1. **`RB51` is not `AWAITING_DECISIONS`, and the six-level hierarchy *was*
   implemented.** `docs/project_index.md:232` still says
   "**`AWAITING_DECISIONS`**; `DEC-RB51-001` … and `DEC-RB51-002` … gate M4 and
   M7" and "Production is at `SCAL-V1.1` with four margins and fourteen
   components". That row is dated **2026-08-20** and was never updated. The plan
   itself records `DEC-RB51-001`, `-002` and `-005` **approved on 2026-08-20**,
   status `IN_PROGRESS`, and milestones `M0`–`M7` plus `M-DIAG` **done by
   2026-09-01**; `docs/open_items.md` closed `B2` on 2026-09-01 with "`RB51`
   `M0`-`M8` complete"; and the configuration above is the six-level mode.
   `VERIFIED`. The "`SCAL-V1.1` with four margins" sentence is the plan's own
   2026-08-20 findings-log entry describing the state *before* implementation,
   promoted into the index as if it were a present-tense fact.
2. **`RB51`'s open work is not `M4`/`M7` but the tail of `M8`.** `T-RB51-12` and
   `T-RB51-13` passed on 2026-09-01 (1100 records, mean 70.70, below standstill
   3.36 %, `offroad` divergence `5.97e-06` explained by production's area
   epsilon), and `M8a`/`M8b`/`M8c` ran. What remains unticked is the
   reconciliation: `make test`/`make lint`/`make smoke` as a set, the
   `AC-RB5.1-*` walk-through, the index update, and `M9`'s conditional
   extension. `VERIFIED`.

The general lesson is the one the handoff states and this is an instance of: a
grep hit proves someone wrote the name, not that the code has the thing — and a
register row proves what was true on its date.

### 4.2 The code A7 changes

| Concern | Current state | Path |
|---|---|---|
| Margin vector arity | Declared once, as data, keyed by mode: `{bounded_*: 4, six_level_priority_weighted_rank: 6}` | `reward/scalarization.py:52-58` `_REQUIRED_MARGIN_COUNT_BY_MODE` |
| Required base per mode | `six_level_priority_weighted_rank` requires exactly `SIX_LEVEL_PRIORITY_BASE = 2.5`; a mismatch raises at construction | `reward/scalarization.py:38-44, 105-110` |
| **The progress index is a literal `3`** | `_canonicalize_bounded` range-checks index 3 as `[-1, +1]` and every other index as `[-1, 0]`, with a docstring stating that the index "is 3 in both the four-level and the six-level vector, because ADR-072 added L5 and L6 *below* progress rather than around it" | `reward/scalarization.py:331-359` |
| §5.4 predicate | Constructor-time, over the three priority weights, with `tail = λ₄ + η·(Δt/T_REF) + λ₆·(Δt/T_REF)` and `bound = (1+σ)·Σ(lower) + φ·|lower| + tail` | `reward/scalarization.py:147-196` `_validate_six_level_weights` |
| `φ` applies to the priority block only | `flat * margin` is added inside the three-level loop, and `flat * len(lower)` counts only the priority levels below `k` | `reward/scalarization.py:413, 191` |
| Level membership and aggregation | `L5` normalized sum over three with a declared denominator; `L6` explicit denominator 1 | `rulebook/v2/aggregation.py`, `rulebook/v2/registry.py` |
| `L6`'s sub-rule | `advance_shortfall`, deliberately not named `progress_rate` so the aggregated result cannot overwrite the atomic one | `rulebook/v2/components/progress_rate.py` |
| Below-standstill baseline | `−λ₆·(Δt/T_REF)·T`, because a stopped ego pays `c_L6 = 1` on every step | `scripts/measure_expert_rulebook_transition.py:444-486` `v51_standstill_return` |

**The literal `3` is the trap in this change, and it fails loudly rather than
silently.** Under A7 progress is index **4** of five. Feeding an A7 vector to the
current function would range-check `K4` — a negated cost — as `[-1, +1]` and `K5`
as `[-1, 0]`, so **every step with positive progress would raise
`ScalarizationEvaluationError` and abort the step**. `INFERRED` from the code
above, and `TEST-A7-05` in §9 exists to pin it. The fix is to make the progress
index a declared property of the schema, in the same table that already declares
the arity, for the reason that table's own comment gives: a required value written
down twice is a required value that can drift.

### 4.3 The measurement instrument

| Concern | Current state | Path |
|---|---|---|
| The per-step channel site | `v51_l1`, `v51_l2`, `v51_l3`, `v51_l5`, `v51_l6` and `v51_l4 = Δq` are computed per step; only **violated-step counts** are accumulated per channel, plus `Δq` totals and maxima | `scripts/measure_expert_rulebook_transition.py:2543-2562` |
| The reward under test | `v51_reward(...)`, `SCAL-V1.4` written out independently of production | `:729-768` |
| The §5.4 predicate, offline | `v51_is_rank_preserving(...)` | `:771-798` |
| Two weight grids | `v51_weight_grid()` sweeps `(λ₄, η, λ₆)` **pinned at `a = 2.2, σ = 0`**; `v51_calibration_grid()` sweeps `(a, σ, λ₄)` at `η = 1.0, λ₆ = 0.2` | `:367-380, 417-441` |
| The pinning is deliberate and correct | `FINAL_PRIORITY_BASE = 2.2`, `FINAL_SEVERITY = 0.0`, documented at `:249-258` so the grid's baseline row keeps reproducing the published §5.5 figures | `VERIFIED` |
| The calibration is **undiscounted** | `episode_return += scalarized.reward` | `:2129` |
| The telescoping residual is now signed and per-episode | `v51_telescoping_max_surplus`, `..._max_deficit`, `v51_telescoping_residuals` | `:1303-1312` (`C51`, closed 2026-09-09) |
| Argmax blame exists but does not reach the v5.1 channels | `worst_named` returns `(cost, blame)`; `final_blame_r2`/`final_blame_r3` are consumed only by the four-level `final` family's reward-mass attribution at `:2631-2632`. The v5.1 channels discard it: `v51_l2 = final_r2` takes the value without the label, and `v51_l3 = max(final_control_r3, variant_offroad, final_speed_limit_cost)` builds its `max` inline with no blame at all | `VERIFIED` at `:2478-2500, 2545-2547, 2631-2632` |

**Measured, not estimated: how many grid members exist at the production weight
pair.** Re-deriving both grids' rank-preservation filters from the transcribed
predicate gives `v51_weight_grid` **77 admissible members of 100** and
`v51_calibration_grid` **16 of 36**, and at `a = 2.5, σ = 0.30` the calibration
grid contains **exactly one** member: `λ₄ = 2.0`. `λ₄ = 2.5` and `2.8` are
inadmissible there. So the statement "the grids sweep `λ₄` upward, or downward at
the pinned baseline pair" is exact: **no reduction of `λ₄` has ever been priced at
the weights production runs**, and adding two members there triples that cell.
`VERIFIED` by execution today (§14).

### 4.4 The mandatory tests A7 moves

| Test | What it pins today | What A7 does to it |
|---|---|---|
| `tests/test_scal_v14.py` | Writes §5.1 out independently with `SIGMA = 0.30`, `PHI = 0.25`, `LAMBDA4 = 2.0`, `ETA = 1.0`, `LAMBDA6 = 0.2`, `DT_RATIO = 0.1` as module constants; parametrises six six-entry margin vectors; parametrises the inadmissible-weight cases on `relaxable_weight` and `progress_rate_weight`; `test_l6_reaches_only_the_last_term` asserts `λ₆` moves nothing else | The formula, the arity, the weight names and one whole test change. `DEC-A7-006` |
| `tests/test_rulebook_v51_orderings.py` | The `O1`–`O6` fixtures, written as per-step `(l1, l2, l3, Δq, l5)` tuples with `l6` derived from the advance; `_shipped_gamma()` reads `γ` from the configuration rather than hardcoding it | Channel order, the `l6` derivation and `η`/`λ₆` all change. It also inherits `FINAL_PRIORITY_BASE = 2.2, FINAL_SEVERITY = 0.0`, which is `V4`. `DEC-A7-007` |
| `tests/test_hydra_agent_presets.py` | `test_every_algorithm_shares_one_hierarchy_preserving_discount` pins one shared discount without pinning its value; `test_the_hierarchy_preserving_horizon_is_measured_from_the_frozen_index` reads `L = 500` from the committed index and pins the **requirement** `γ ≥ 0.998169` | Its own docstring hands the verdict assertion to this change: "the criterion's verdict lands with the approved `gamma = 0.9982` rather than here, next to the value that makes it hold". `M6` |
| `tests/test_rulebook_v51_levels.py` and 7 further test files | Six-level membership, aggregation and diagnostics | Mechanical migration of the same class as `RB51`'s `M1` ripple |

**The ripple is measured, not guessed.** Eleven source, configuration and script
files and eight test files reference the six-level shape explicitly
(`six_level`, `MACRO_RULE_ORDER`, `RELAXABLE_LANE_COMPLIANCE`); nine source and
eight test files reference `progress_rate`, `PROGRESS_RATE` or
`advance_shortfall`, about thirty references in total; thirteen test files touch
`margins`. `RB51`'s comparable 4→6 change rippled to 50 tests, all encoding the
old contract. `VERIFIED` by enumeration today.

### 4.5 The two plans A7 collides with

Both were read in full, not through their index rows.

**`AB-LEARN` (`implementation/reward_learnability_ab_screening_exec_plan.md`).**

| Fact | Detail | Label |
|---|---|---|
| Arm B's reward *is* what A7 replaces | `REQ-AB-009`: "The reward under test is `RULEBOOK-V5.1`'s six-level vector scalarized by `SCAL-V1.4` as configured in `conf/scalarization/default.yaml`; no scalarization parameter is tuned during the screening" | `VERIFIED` |
| A passing screening freezes both jointly | `DEC-AB-005`, approved 2026-09-01, elaborated in §7.5 | `VERIFIED` |
| **No live screening evidence exists under any reward** | The seed-0 pair launched 2026-09-05 and both arms died on 2026-09-06 (arm A on `C9` at 175k, arm B on a CUDA OOM at 50k); `DEC-AB-007` relabelled arm A as a descriptive baseline; the relaunch is unmade and `DEC-AB-004`, GPU authorization, is the user's | `VERIFIED` |
| Those runs were never evaluable through the official panel path | `src/thesis_rl/runtime/final_panels.py:167-173` refuses any checkpoint but `final.zip`, and neither screening run has one | `VERIFIED` |
| Single-factor invariance survives an A7 change | Arm A builds no scalarizer (`reward=monitor_only`), so its `scalarization` block is present but inert; both arms inherit `conf/scalarization/default.yaml`, so changing it moves both identically and `AC-AB-002` still holds — but it must be **re-verified** by re-running the resolved-config diff, not assumed | `INFERRED`; `M8` runs the diff |
| **`AB-LEARN` §5 is stale on the discount** | It records "`γ = 1` (ADR-075)" under Assumptions And Invariants; production has been at `γ = 0.996` since ADR-081, and this plan moves it again | `VERIFIED` |

**`RB51` (`implementation/rulebook_v5.1_six_level_hierarchy_exec_plan.md`).**
Status, gates and milestone state as corrected in §4.1. Its `M8` measurements
found three things that bear on A7 and are **not** reasons to reopen anything:
`STOP` controls are dropped by route reachability on 400 of 536 (74.6 %) against
`SIGNAL`'s 313 of 744; three `L3` sub-rules are nearly inert on Waymo
(`crosswalk` 15/1100, `stop` 115/1100, `signal` 409/1100); and **five of the six
`L3` sub-rules never apply on PG at all**, so `K3` there is `offroad` alone and
the two sources are graded by substantially different rulebooks. A7 changes the
order, not the membership, so all three carry across unchanged and constrain what
may be *claimed* about `K3`, not what it is.

### 4.6 Directly relevant debt

`C49` (the discount; this plan closes it), `C50` and `D14` (decided; declared),
`C52` (the user's), `D1` (`τ₄`'s mechanism; out of scope), `D15` (O3 at the
shipped discount; A7 restores O3 on the scalar arm and `I1` explains why the
undiscounted dominance form is unobtainable), `V4` (the ordering fixtures'
weight pair; folded into `DEC-A7-007`), `V1` (`make smoke` not run since the
discount change), `C2`/`C3` (`signal` under-firing, independent of the order),
`C8` (a run's provenance artifacts mislabel the rulebook family — worth knowing
before reading any A7 run's banner, and not A7's to fix).

---

## 5. Assumptions And Invariants

| Item | Value | How established | Violation handling |
|---|---|---|---|
| Units | metres, seconds, m/s; `Δt = 0.1 s` | `VERIFIED`, unchanged | — |
| `D_REF` | `v_ref·Δt = 2.2222 m`; `Δq = clip(Δs/D_REF, −1, +1)`, `ΔQ_MAX = 1` | `VERIFIED`, unchanged by A7 | The clip is normative, not fitted |
| `T_REF` | **Disappears.** With `η` and `λ₆` gone, no channel is scaled by `Δt/T_REF` | Consequence of `REQ-A7-05` | The configuration keys go with it |
| Channel ranges | every cost channel in `[0, 1]`; `K5` in `[−1, +1]` | `SPECIFIED`; enforced by `_canonicalize_bounded` | Out of range raises and aborts the step |
| Margin sign convention | costs are carried **negated**, so larger is better on every entry and a lexicographic consumer needs no per-entry knowledge | `VERIFIED` at `types.py:453-457` | Preserved exactly |
| **Progress index** | becomes **4** of five, and must be declared data rather than a literal | §4.2 | `TEST-A7-05` |
| Discount | `γ = 0.9982`, one value for every arm, `learning_potential_gamma` equal to it | Approved 2026-09-09 | `test_every_algorithm_shares_one_hierarchy_preserving_discount` fails on disagreement |
| Measured training horizon | `L = 500` control steps: the frozen index's `length` is the scenario's `SD.LENGTH`, the runtime reads the same field, and the episode truncates at `scenario_length − 1` with `horizon: null` and `extra_steps_after_scenario: 0` | `VERIFIED`, chain in `C49` | `horizon_steps == 500` in the guard fails on a regenerated index |
| The calibration is discount-free | `episode_return` is an undiscounted sum and the rank-preservation inequality is per-step and `γ`-free | `VERIFIED` at `:2129` | So `γ` requires **no** re-run of the 217,189-transition grid |
| Termination/truncation | unchanged: at-fault contacts terminate and are charged (ADR-071), not-at-fault truncate through `MAX_STEP` and cost nothing | `VERIFIED`, out of scope | — |
| Seeds and splits | unchanged; validation and test panels are consulted by no calibration in this plan | `VERIFIED` | — |
| Determinism | the evaluator stays a pure function of the transition snapshot plus declared memory fields | `VERIFIED` | Fails closed |

### 5.1 The A7 scalar adapter, and its rank-preservation condition

The form, transcribed from the recommendation:

```
r_t =  Σ_{k=1..3} a^(4−k) · [ (step(m_k) − 1) + σ·m_k ]     priority block
     +  w₅ · [ (step(m₅) − 1) + σ·m₅ ]                      K4, a sub-unit level
     +  λ₄ · Δq_t                                           K5, open-ended
```

with `m_k = −c_k ∈ [−1, 0]`, and `η`, `λ₆`, `φ` and the `Δt/T_REF` factor
removed. Rank preservation becomes

```
a^(4−k)  >  (1+σ)·Σ_{j>k, j≤3} a^(4−j)  +  w₅·(1+σ)  +  φ·|{j : k<j≤3}|  +  λ₄·ΔQ_MAX
```

**Re-derived independently today** (§14) rather than relayed, because one of these
figures does not survive a careless transcription. At `a = 2.5, σ = 0.30,
λ₄ = 2.0, w₅ = 0.15`:

| configuration | `k=1` | `k=2` | `k=3` | thinnest |
|---|---:|---:|---:|---:|
| A0 today (six levels, `φ = 0.25`, `η = 1.0`, `λ₆ = 0.2`) | 1.1165 | 1.1121 | 1.1792 | **1.1121** |
| A7 with `φ = 0.25` retained | 1.1105 | 1.0975 | 1.1390 | **1.0975** |
| A7 with `φ = 0` | 1.1514 | 1.1478 | 1.1390 | **1.1390** |

Two things follow, and the second is a trap worth recording. First, **`φ = 0`
returns A7's only structural cost on this axis**: the thinnest margin goes
1.0975 → 1.1390, above today's 1.1121. Second, `φ` multiplies **only the priority
levels below `k`** — which is what the implementation does, `flat * margin` inside
the three-level loop — so a transcription that lets `φ` or `w₅` count as an
ordinary lower level gives 1.0911 / 1.0513 / 1.0225 and a thinnest margin at
`k=3` instead of `k=2`. That is wrong, and it is the shape of error `AGENTS.md`
warns about: the arithmetic must be reproduced against the code's own term
placement, not against a plausible reading of the formula. At `k=3` there are no
priority levels below, so `φ` contributes nothing there and the two A7 rows agree
at 1.1390.

### 5.2 The discount, priced

| quantity | `γ = 0.996` | `γ = 0.9982` |
|---|---:|---:|
| break-even `ln(a)/−ln(γ)` against `L = 500` | 228.6 — **fails** | 508.6 — holds, by **8.6 steps** |
| whole-episode damping `γ^L` against `1/a = 0.4` | 0.1348 | 0.4062 |
| effective horizon `1/(1−γ)` | 250 steps (25.0 s) | 556 steps (55.6 s) |
| O3 margin on the scalar arm, six-level | −3.5789 | −1.9934 |

**The cost, in one sentence:** `γ = 0.9982` doubles the effective horizon from 250
to 556 steps and leaves **40.6 % rather than 13.5 %** of a spuriously bootstrapped
constant alive at the end of the longest episode, which weakens ADR-081's own
contraction argument by the same factor 2.2 — and it costs nothing in
calibration, weights or measurement, because the grid is undiscounted. The
criterion is an identity worth stating: `Δ > L` **is** `γ^L ≥ 1/a`, so it is the
most damping the contraction argument can have without the hierarchy argument
failing. The 8.6-step margin is a property of the current frozen index, not of the
code: one 510-step scenario in a regenerated index would put the criterion back in
deficit, which is what `horizon_steps == 500` in the guard exists to say.

`VERIFIED` by execution today (§14).

---

## 6. Decisions And Approval Gates

No dependent work starts while a gate is unresolved. `M1` depends on no gate.

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-A7-001` | Specification clarification | A7 has no specification document. What form does it take? | (a) a new `RULEBOOK-V5.2` + `SCAL-V1.5` superseding v5.1 §3/§4/§5/§9/§10; (b) an in-place amendment of v5.1, the form ADR-081 used; (c) a separate amendment file, the form `OBS-V1.3.1` used | **(a)** | A7 makes §3 (level count and membership), §4.6 (`L6`), §5.1 (the formula), §5.4 (the predicate), §5.5 (the weights), §9 (nine of seventeen acceptance criteria) and §10 (the fixtures) wrong **simultaneously**. ADR-081's in-place form worked for two constants; a reader of v5.1 plus five amendments would have to reconstruct the architecture from a changelog. The index's authority model already keys on version | Awaiting approval |
| `DEC-A7-002` | Specification deviation | `w₅`, the weight on `K4` | 0.15 / 0.25 / any value in the derived window | **`w₅ = 0.15`** | See §6.1. Cost stated there | Awaiting approval |
| `DEC-A7-003` | Specification deviation | `φ`, the shared absolute tie-breaker | remove (`φ = 0`) / keep 0.25 | **remove** | See §6.2. Cost stated there, with its falsifier in `M1` | Awaiting approval |
| `DEC-A7-004` | Specification deviation | `η` and `λ₆`, and the `L6` level that carries `λ₆` | remove both / keep `L6` below `K5` / keep `η` only | **remove both** | See §6.3 | Awaiting approval |
| `DEC-A7-005` | Specification clarification | The budgets `τ₁`–`τ₄` have no values | (a) fix each from the expert per-episode distribution at a **pre-registered** quantile; (b) choose values after seeing the distribution; (c) defer until the thresholded mechanism is chosen (`D1`) | **(a)**, criterion in §6.4, values filled in by `M1` | (b) is the post-hoc choice `EVAL-PROTOCOL` `REQ-018` prohibits in its own domain and the same objection applies here; (c) leaves the architecture without the budgets that define three of its five channels | Criterion awaiting approval; values awaiting `M1` |
| `DEC-A7-006` | Mandatory test change | `tests/test_scal_v14.py` changes substantially | amend / delete and rewrite / leave and add a second file | **amend in place**, itemised in §6.5 | It is a mandatory test and `AGENTS.md` requires recorded approval. One test (`test_l6_reaches_only_the_last_term`) has no A7 counterpart and is **deleted**, not weakened | Awaiting approval |
| `DEC-A7-007` | Mandatory test change | The `O1`–`O6` fixtures change, and they carry `V4` | amend for A7 only / amend and parametrise over both weight pairs / amend and read the shipped pair from the configuration | **amend and parametrise over both pairs**, closing `V4` in the same change | The fixtures assert at `a = 2.2, σ = 0`, which production has not used since ADR-081. Every ordering was checked to hold at the shipped pair (O3 excepted, the known `D15` failure), so this is a verification gap, not a behavioural one — but A7 rewrites these fixtures anyway, so folding it in costs one change instead of two | Awaiting approval |
| `DEC-A7-008` | Specification clarification | `AB-LEARN`'s arm B is pre-registered on the reward A7 replaces | (a) amend the pre-registration explicitly and re-register arm B against A7 before any run; (b) run the screening on the old reward first, then A7; (c) leave `AB-LEARN` untouched and let it drift | **(a)** | See §6.6 | Awaiting approval |
| `DEC-A7-009` | Scope | Two ExecPlans for one reward | (a) A7 supersedes `RB51` downstream of the hierarchy, `RB51` closes as `IMPLEMENTED` with its residue named; (b) A7 becomes a milestone of `RB51`; (c) A7 sequences behind `RB51`'s `M8`/`M9` | **(a)** | See §6.7 | Awaiting approval |
| `DEC-A7-010` | Implementation detail | `λ₄` | keep 2.0 / reduce to 1.9 (buys back the §5.4 margin) / reduce to 1.25 (`λ₄ ≤ a/2`) | **keep 2.0.** Not reopened here | See §6.8. `M1` prices 1.9 and 1.25 at the production pair as *measurements*, which turns "we do not know" into "we know and chose"; the proposal stays 2.0 | Not a gate — recorded so the measurement is not mistaken for a proposal |
| `DEC-A7-011` | Specification clarification | The recorded artifact schema and the checkpoint identity both break | declare both breaks / preserve the old channel names / emit both schemas | **declare both**, as `DEC-RB51-005` did for the same enum | `MacroRule` values reach CSVs, evaluation artifacts and analysis tables; the vector arity and schema id reach the checkpoint reward-semantics identity. Keeping a deleted level in the recorded schema is exactly the stale naming `DEC-RB51-005` refused | Awaiting approval |
| `DEC-A7-012` | Specification clarification | Nine of the seventeen `AC-RB5.1-*` criteria stop applying or change meaning | restate them in the A7 specification / carry them unchanged / drop them silently | **restate**, with the disposition table in §9.3 | `AC-RB5.1-10` (`η` reaches only `L5`), `-13` (`λ₆` below its O3 bound), `-15` (the standstill baseline) and `-17` (`L6` refines only ties) become `NOT_APPLICABLE`; `-01`, `-02`, `-04`, `-06`, `-12`, `-14` and `-16` change their content | Awaiting approval |

### 6.1 `w₅ = 0.15` — criterion, derivation, value, cost

**Criterion, stated before the value.** `w₅` is an exchange rate between
negotiable lane exposure and arrival time, so both of its bounds must be stated in
the same physical currency — seconds of arrival time — and neither may be a
dimensionless preference.

- **Lower bound: the reference shortcut must not pay.** Direct measurement of the
  scalar margin against the §4.6 reference pair gives `w₅ > 0.1313` at
  `γ = 0.996` and `w₅ > 0.0736` at `γ = 0.9982`.
- **Upper bound: crossing a marking must stay cheaper than the manoeuvre it
  replaces.** Stopping from urban speed `v = 10 m/s` and returning to it at a
  comfortable `a_c = 2 m/s²` costs `v/a_c = 5.0 s` of delay. One second of marking
  contact must cost less than that: `w₅ < 0.4477` at `γ = 0.996` and
  `w₅ < 0.2641` at `γ = 0.9982`.
- **Independent cap:** rank preservation caps `w₅` at `(a − λ₄·ΔQ_MAX)/(1+σ) =
  0.384615`. Re-derived today (§14).

**The binding bound is the lower one at the shipped discount**, deliberately, so
that the architecture decision stands whether or not the discount decision is
taken: `w₅ > 0.1313`, rounded up to **0.15** for margin, because the reference
shortcut is recorded as a stipulation rather than a measurement.

**Value and cost in one sentence.** `w₅ = 0.15` — the first round value above the
binding lower bound at the discount A7 does not depend on — at the cost of the
§5.4 thinnest-margin movement priced in §5.1 and of an exchange rate that a
learner will read as: cross a marking only if doing so buys **1.83 m/s** of extra
route advance, and one second of full lane relaxation is worth **2.8 seconds** of
arrival time.

Both windows contain 0.15 with room: as a fraction of its own effective upper
bound the window is **65.9 %** at `γ = 0.996` and **72.1 %** at `γ = 0.9982`,
against **0.193 %** of the admissible `η` range within which the shipped
architecture can buy O3 back at all — about 350× wider. Re-derived today, and
stated against its own upper bound in both cases, because the two figures use
different effective upper bounds (the §5.4 cap at `γ = 0.996`, the physical bound
at `γ = 0.9982`) and mixing the denominators is how a ratio becomes an artefact.

**What `w₅` does not buy, and it must be said in the same place.** No admissible
`w₅` opposes a *fast* off-corridor drive. The crossover is
`w₅ = λ₄·Δq/(1+σ)`: **0.313983** at the expert's mean pace `Δq = 0.204089`, which
is admissible, and **1.538462** at the clip, which is not — against a cap of
0.384615. The ratio of the clip-pace requirement to the cap is
`λ₄/(a − λ₄·ΔQ_MAX)`, in which `σ` cancels, so it is **exactly 4** for every
admissible severity and no re-tuning of `σ` reaches it. That denominator is the
§5.4 tail, so "no admissible `w₅` opposes a fast off-route drive, by a factor of
four" is the same statement as "progress consumes four fifths of the per-step
budget". At `w₅ = 0.15` a fully violated `K4` step costs **0.195** against
**2.000** of progress at the clip, so it opposes nothing at any pace. Verified
independently today (§14). This is `D14`'s exposure, declared in §15, and the one
quantity that would move it is `λ₄` relative to `a` — which is why `M1` prices it.

### 6.2 `φ = 0` — four grounds that compound, and the falsifier

1. **Its stated job is already done.** `φ` breaks the tie the satisfaction
   indicator creates between two margin vectors with the same discrete
   satisfaction pattern. At `σ = 0`, where `φ` was set, the priority term was
   constant inside the violated set and the tie was real. At `σ = 0.30` it is not:
   the slope `σ·a^(4−k)` is non-zero at every level. Mechanism, not preference.
2. **Its shape is the one ADR-081 itself criticises.** Being absolute, its grading
   is inversely proportional to importance: `φ` is **5.1 %** of the severity slope
   at `k=1`, **11.8 %** at `k=2` and **25.0 %** at `k=3`.
3. **No document derives the value.** 0.25 traces to Veer et al. (ICRA 2023,
   Theorem 1) averaged-robustness tie-breaker `1/N` with `N = 4`, for *their*
   four-level schema, here summed over three margins. **The only argument for
   `φ = 0.25` in this repository is the specification, and by `AGENTS.md`'s
   Scientific Argument Standards that is a finding rather than a justification** —
   and it points at the constant, not at the code.
4. **It returns A7's only structural cost on the §5.4 axis**, 1.0975 → 1.1390
   (§5.1).

**Value and cost in one sentence.** `φ = 0` — because at `σ = 0.30` the tie it
breaks no longer exists — at the cost of `a_req^max` falling from **12.43** to
**10.96 m/s²**, i.e. from 1.38× to 1.22× the ~9 m/s² a real vehicle can produce,
which is the margin by which the reward's local gradient still points toward
braking in the hardest conflict a vehicle could resolve.

**Falsified before recommending, and falsifiable after.** Over the ten-ordering
battery at both discounts no ordering changes sign and no reward-hacking probe
changes verdict; the largest movement is O5 (waiting at a red) from +9.075 to
+8.083. The residual risk is that `φ = 0`'s effect on the expert's mean return
under the *six-level* reward is unmeasured — it was priced only on the four-level
family, whose rows cannot be read across. **The falsifier is one grid member at
`φ = 0` in `M1`'s run**: if `fraction_below_standstill` breaches 7.45 %, revert to
0.25 and accept the 1.0975 margin.

### 6.3 Removing `η` and `λ₆`

- **`η`** is, by the shipped specification's own admission, the one weight the
  expert panel cannot discriminate, because the logged human almost never relaxes.
  A weight that no measurement can fix is a weight chosen by taste.
- **`λ₆`** carries a level whose cost fires on **99.4024 %** of expert steps at
  `p50 = 0.884` and is a pointwise function of the progress channel — the only
  channel whose return distribution carries nothing beyond its mean once progress
  is known, which is precisely what the distributional component cannot use. Its
  value is an open decision (`D15`).
- **Deleting `L6` deletes the `Δt/T_REF` normalization entirely**, so the reward
  stops carrying two different per-step scalings.

**The cost, stated in the same breath, and there are two parts.** First,
below-standstill: the criterion moves from `R₀ < −λ₆·(Δt/T_REF)·Σ Δq⁺ = −0.81` to
`R₀ < 0`, so the count can only **rise**, by the mass in a 0.81-wide band, which
the local density between p5 and p10 bounds at **0.53 pp** — `4.55 % → ≈5.1 %`,
and `≈5.2 %` once the `w₅` indicator's own ≈0.27 reward units of expert cost are
added, against the 7.45 % ceiling. This is the one acceptance column no algebra
derives, and `M1` measures it instead of projecting it. Second, the completion
incentive: closing the mission rather than covering 99 % of it and idling is worth
**0.66–2.99** reward units under the six-level reward and **0.57–0.77** under A7,
so `L6` supplies 13–74 % of it. Both figures are an order of magnitude below the
cost of one fully-violated interaction step — **8.125** under A7, where `φ = 0`,
and **8.375** with `φ = 0.25` retained, which are the two figures `REVIEW.md`
quotes in different sections without naming the configuration each belongs to
(re-derived in §14) — so neither architecture makes the last stretch worth a
risky manoeuvre; the real gap is that mission success is a **zero-value
terminal**, and it is shared. A7's incentive is
smaller in magnitude and better in kind: the six-level one is proportional to
`199 − T_c`, the remaining length of the Waymo log, while A7's is proportional to
the remaining mission distance, which is a property of the task.

A terminal completion bonus is **not** recommended as compensation: the minimum
value that makes the last stretch worth one fully-violated interaction step is
`B ≥ 9.7`, and in the ordered arms `B` lives entirely inside `K5`, below every
safety channel, so it can cause no regression there and fixes nothing either. The
diagnostic that would reopen it is the fraction of evaluation episodes reaching
≥95 % route completion without a gate crossing, which nothing currently reports.

### 6.4 The budget criterion — stated now, valued by `M1`

Three requirements on any budget `τ_i`, in this order:

1. **It must admit the logged competent driver.** A budget quantified over
   zero-exposure completions describes no trajectory the agent can produce: the
   expert accrues interaction cost on 0.3978 % of steps and non-negotiable
   compliance cost on 0.6418 %. `τ₃ = 0` is therefore falsified **by the panel**,
   not by preference.
2. **It must be denominated per unit of mission span**, `X_i/Q`, because `Q` is a
   scenario constant the agent cannot influence, so the budget is
   duration-invariant, immune to dilution by dawdling, and comparable across
   missions spanning 13–247 m. This is not a route-length normalization of the
   *reward*, which was measured and rejected; the per-step vector is untouched and
   only the budget's units change.
3. **The quantile must be chosen before the numbers are seen**, or the budget is
   fitted to the distribution it is supposed to be tested against.

**Pre-registered rule.** `τ_i = p99` of the logged expert's per-episode `X_i/Q`
distribution on the frozen Waymo `train` panel, with the maximum reported beside
it and **the excluded episodes listed per record**. Rationale, and the cost in the
same sentence: p99 admits all but the expert's worst 1 % — about 11 of 1100
records — at the cost of rejecting eleven human episodes, which is only defensible
if they can be inspected, so the rule is void unless the per-record rows are
emitted. If those episodes are not attributable to a declared panel defect, the
value moves to the maximum; that is a decision on `M1`'s evidence, pre-registered
here as a rule rather than taken later as a preference. The maximum alone is
rejected as the primary rule for the reason the reward's own history gives: a
single outlier record then dictates a parameter for every other record.

**`τ₁ = 0`, conditionally and measurably.** At-fault impact is non-zero on at most
one step (ADR-071), and a non-colliding trajectory has `K1 = 0` exactly, so a zero
budget is what makes the ordered arms prefer enduring 200 steps of interaction
violation to colliding at fault — measured at `K1 = 0.0000` against `0.5788`,
where the scalar arm prefers colliding by 1092.8. `τ₁ = 0` is admissible **iff the
logged expert records no at-fault impact on the panel**, which is not yet
measured. `M1` emits `X_imp`'s per-episode distribution for exactly that reason;
if it is non-zero anywhere, `τ₁` follows rule (1) like the others and the finding
is reported.

### 6.5 What changes in `tests/test_scal_v14.py`

Itemised, so the approval is informed rather than blanket:

| Change | Why it is not a weakening |
|---|---|
| `ETA`, `LAMBDA6`, `DT_RATIO` module constants and `PHI` removed | The weights they name cease to exist |
| `expected_reward` rewritten to the A7 form, still written out **independently of the implementation** | The property that makes this test worth having is preserved exactly: it asserts the formula term by term rather than a remembered number |
| Six six-entry margin vectors become five-entry vectors with `K4` and `K5` in their new positions | Arity change |
| `test_l6_reaches_only_the_last_term` **deleted** | It asserts that varying `λ₆` moves no other contribution; `λ₆` is gone and there is nothing left to assert. Its *pattern* is preserved by a new `test_w5_reaches_only_the_negotiable_channel` |
| `test_inadmissible_weights_cannot_be_constructed` re-parametrised off `relaxable_weight` and `progress_rate_weight` and onto `progress_weight`, `severity` and `negotiable_weight` | Same predicate, new fields; the case count does not fall |
| `test_progress_and_the_two_levels_below_it_form_a_finite_exchange` restated | Under A7 its fixture still holds (a creep with `K4` violated scores `−0.195 + 2e-6` against a standstill's `0`), but the docstring's claim changes: with `L6` gone, a creep with `K4` **satisfied** earns `+2e-6` against standing still's `0`. That is a real behavioural change, it is `P11`'s violation in miniature, and the test must assert it explicitly rather than leave it implicit |
| New: `TEST-A7-05`, the progress index | §4.2's literal `3` |

### 6.6 How `AB-LEARN`'s pre-registration is amended

The principle is `AB-LEARN`'s own, in §7.5 point 3: the freeze gate "is legitimate
only because it was pre-registered", and "a gate declared after seeing the
numbers would be the post-hoc promotion `REQ-018` prohibits". The same standard
decides this case, and it decides it cleanly: **no live screening evidence exists
under any reward.** Both seed-0 runs died, arm A's seven evaluations are retained
as a descriptive baseline that "licenses nothing" by `DEC-AB-007`, arm B reached
one evaluation, neither run has the `final.zip` the official panel path requires,
and the relaunch has not been made. So moving arm B's reward now costs nothing
scientifically — there are no results it could have been chosen to flatter — and
moving it later would cost everything.

Recommended amendment, as a new `DEC-AB-008` in that plan:

- `REQ-AB-009` is restated against the A7 specification: the reward under test
  becomes A7's five-channel vector scalarized by `SCAL-V1.5` as configured, with
  no scalarization parameter tuned during the screening. The clause that matters —
  no tuning *during* the screening — is unchanged.
- `H1`–`H4` and the §7.3 decision rule are **unchanged**. They are properties of a
  reward, not of that particular reward: a learning slope, an inverted incentive,
  a compliance dividend.
- `AC-AB-002`'s single-factor invariance is **re-verified by execution**, because
  arm A's `scalarization` block is inert but present and the diff must be shown to
  be confined to the reward group and the run identity.
- Every artifact from the 2026-09-05 and 2026-09-06 launches is void as A/B
  evidence and retained as provenance only, which is already `REQ-AB-003`'s
  relaunch discipline.
- `AB-LEARN` §5's stale "`γ = 1` (ADR-075)" is corrected to `γ = 0.9982`.
- **A7 is frozen before the screening runs**, by the same pre-registration logic
  `AB-LEARN` §7.5 applies to the ACL: the specification is approved and the
  implementation verified first, and the screening then *measures* the reward
  rather than gating it.

`DEC-AB-004`, GPU authorization, is untouched and remains the user's.

### 6.7 `RB51`'s disposition — one reward, one owner

A7 amends what `RB51` implemented, so the question is which plan owns the reward
afterwards. Recommendation: **A7 owns it, and `RB51` closes.**

- `RB51`'s milestones `M0`–`M7` and `M-DIAG` are done and in production (§4.1), so
  it closes as `IMPLEMENTED`, not as abandoned.
- Its unfinished `M8` residue is **reconciliation against acceptance criteria A7
  rewrites**. Forcing `RB51` through a full walk-through of `AC-RB5.1-01`…`-17`
  and then superseding nine of them the same week is waste; the disposition table
  in §9.3 does that walk-through once, under A7, and names what carries.
- `RB51`'s `M9` (multi-hop route reachability, gated on `M8b`, which ran) touches
  `rulebook_v4.11` §2.9.5 and no A7 channel. It **detaches** into its own item
  rather than riding A7 — it is a `signal`/`stop` selectability question, and
  `C2`/`C3` already register it.
- `docs/project_index.md:232` is corrected as part of A7's `M8`: it is stale in
  three ways (status, gates, and "Production is at `SCAL-V1.1`"), and leaving it is
  what produced the second-hand error §4.1 corrects.

This is what "two plans for one reward is the failure to avoid" resolves to:
`AB-LEARN` does not own the reward, it *consumes* it; `RB51` owned it and is done;
A7 owns it now.

### 6.8 `λ₄` stays at 2.0, and `M1` prices the alternative anyway

A derivation shows that for *some* admissible `w₅` to oppose an off-corridor drive
at clip pace the crossover must fall inside the §5.4 cap, which requires
`2·λ₄·ΔQ_MAX ≤ a`, i.e. `λ₄ ≤ a/2 = 1.25`, with `σ` cancelling (§6.1). The
shipped `λ₄ = 2.0` is 1.6× that.

It is **not proposed**, for a reason that is measured rather than deferential. On
the only grid that prices a reduction, `λ₄ ∈ {0.5, 1.0, 1.5, 2.0, 2.15}` gives
below-standstill `8.8 / 6.1 / 4.5 / 3.4 / 3.2 %`, so interpolating to `λ₄ = 1.25`
costs about **+1.9 pp**, which would put A7 near **7.10 %** against the 7.45 %
ceiling. But that grid runs at `a = 2.2, σ = 0`, so the figure is an
extrapolation off the wrong weight pair — and §4.3 measured that the production
pair has **exactly one** admissible calibration member, `λ₄ = 2.0`. Raising `a`
instead of lowering `λ₄` needs `a ≥ 4.0` (`a³ = 64` against 15.6), and ADR-081
already measured and rejected `a = 3.0`.

So `M1` adds **two** grid members at `a = 2.5, σ = 0.30` under the A7 reward,
`λ₄ = 1.9` (the value that buys back A7's §5.4 margin) and `λ₄ = 1.25` (the value
that admits a `w₅` opposing an off-corridor drive at clip pace). Both are
admissible under the A7 tail — margins `1.1600 / 1.1693 / 1.1933` and
`1.2188 / 1.3312 / 1.7301`, re-derived today — and both are free: the run already
prices 77 + 16 members, so three more at the production pair is noise against a
45-minute budget. **They are measurements, not proposals.** The recommendation
stays `λ₄ = 2.0`, and the point of measuring is that the next person to ask reads
a number instead of an extrapolation.

---

## 7. Proposed Design

### 7.1 The channel structure

`MacroRule` loses `PROGRESS_RATE` and `MACRO_RULE_ORDER` becomes the five-channel
order, with `MISSION_PROGRESS` **last**. `MACRO_RULE_ORDER` stays the single
source of the ordering and nothing else may restate it. `COST_MACRO_RULES`
continues to derive from it by excluding the one utility level, so the invariant
that stops a sign error from turning progress into a penalty is preserved by
construction rather than by a second list.

`RulebookResult.margins` becomes a five-tuple and `costs` a four-tuple, validated
against the order rather than against a fixed count — which is already how the
type validates itself, so the change is to the order, not to the validator.

Intra-level aggregation is untouched: `K2` and `K3` by `max` across objects and
sub-rules, `K4` by normalized sum with the declared denominator 3. The
`advance_shortfall` sub-rule and its component file are deleted; the
`progress_rate` **level** name disappears with them, which is what makes the
enum's own name-collision guard moot rather than violated.

### 7.2 The scalar adapter

A new mode `five_channel_priority_weighted_rank` in `reward/scalarization.py`,
implementing §5.1's A7 form, with:

- a new `SIX_LEVEL_VECTOR_SCHEMA_ID` sibling — `rulebook_a7_five_channel_v1` — for
  the same reason the six-level id is distinct from the four-level one: a consumer
  reading the old schema and ignoring the tail would be reading a **different
  preference order**, not a truncated one;
- the arity **5** declared in `_REQUIRED_MARGIN_COUNT_BY_MODE`, the table that is
  already the single source of arity;
- **the progress index declared in the same table**, so `_canonicalize_bounded`
  reads it instead of testing `index == 3` (§4.2). This is the one change in the
  file that would fail silently if made carelessly and loudly if made
  incorrectly, and `TEST-A7-05` pins both directions;
- `severity` and `progress_weight` retained; `negotiable_weight` (`w₅`) added;
  `flat_tie_breaker`, `relaxable_weight`, `progress_rate_weight`, `step_dt_s` and
  `reference_time_s` removed from the A7 path. The five legacy modes keep them and
  keep their own arities, unchanged, so earlier runs stay reproducible —
  `DEC-RB51-003`'s reason applies unchanged;
- the §5.4 predicate restated for the A7 tail as `_validate_five_channel_weights`,
  refusing inadmissible weights at construction rather than pricing them.

`ScalarizationConfig.__post_init__` gains the mode's required base and weight set;
`from_mapping`'s allow-list gains `negotiable_weight`. An unknown field still
raises, so a configuration left carrying `relaxable_weight` under the A7 mode
fails the run instead of being silently ignored — which is the behaviour to want,
because a stale weight in a configuration file is exactly how a run ends up
optimizing something nobody intended.

### 7.3 Configuration, artifacts and logging

`conf/scalarization/default.yaml` moves to the A7 mode, schema id and weight set,
and its comment block records the derivations rather than the values alone. The
six algorithm configurations move to `γ = 0.9982`, with `learning_potential_gamma`
following. The recorded channel names in CSVs and analysis tables lose
`progress_rate` and gain the new order (`DEC-A7-011`). The `contracts/`
reward-semantics identity gains the new schema id and weight set, which is the
intentional checkpoint break.

### 7.4 Realistic alternatives, and why not

| Candidate | Result | Why not |
|---|---|---|
| **A0**, the status quo | fails O3 on the scalar arm at the shipped discount; fails O1, O2 (with traffic) and O3 under strict lex | It is the baseline, and its `γ` does not satisfy its own criterion |
| **A1** progress last, negotiable as a priority level at weight 1 | **falsified**: fails O2 (−5.07) and O4 (−13.08) | Accumulated relaxation outranks a collision — 20 steps of full-severity relaxation cost 26 against `a³ = 15.6`. Any candidate that buys "negotiable compliance above progress" as *scalar dominance* dies here |
| **A2b** progress last, compliance merged | **falsified**: fails O2, O4 (−23.08) and O5 (−0.15) | Same mechanism |
| **A2a** progress last, collision merged with interaction — 4 channels, 3 constrained | survives | Halves the impact gradient (margin 2.82 → 1.22) and merges away two of the four sparse tail-shaped channels. There is also a **floor on simplification**: with fewer than three constrained channels of genuinely different priority the lexicographic arm has nothing to compare against the scalarization and the experiment loses its object. A7 has four; A2a has three and pays for the third by merging an outcome with an anticipatory indicator |
| **A6** six levels, negotiable by indicator | passes all ten orderings on the scalar arm, fails O3 under strict lex | **No simplification** — six channels, six weights. It is the candidate that shows the indicator is what fixes the scalar arm, and the position is what fixes the ordered arms |
| **A1c** progress last, negotiable in the continuous tail | survives, weak scalar arm | Fails O3 on the scalar arm, which is the ordering the whole restructure was about |
| A pure constrained MDP, no ordering | — | It is what any thresholded architecture *becomes* in the feasible regime; adopting it discards graceful degradation when budgets cannot all be met, for no simplification of the reward |
| A two-tier architecture | — | With one constrained channel there is no *order* among constraints, so the comparison the thesis exists to make has no object |

**One cost of the thresholded arm is A7's to state, because A7 reduces it.** The
thresholded arm has two mechanism routes and both carry a declared gap. The
policy-gradient route (Tercan & Prabhu's Lexicographic REINFORCE) compares an
accumulated episodic return against the threshold but accumulates it
**undiscounted from a single sampled episode** while its objective is
`J(θ) = V^{πθ}(s_init)`. The state-augmentation route is proved for **one**
constrained channel and is open for several — their Appendix D.3.3: "extending
the approach above to this setting is **not straightforward** … we need to know
which constraints can be satisfied together" — and it needs the accumulated
exposure in the policy's input, which would be an `OBS-V1.3.x` amendment. Also on
the ledger: Pineda, Wray & Zilberstein's Lemma 1, that finding an optimal
*deterministic* policy for a lexicographic MDP is NP-hard. **A0 has five
constrained channels against A7's four, and A0's fifth is progress, the one `I3`
shows cannot be thresholded at all.** Every constrained channel removed is one
fewer constraint in a set the literature does not know how to satisfy jointly,
which raises the value of removing channels above what the candidate scoring
priced it at.

Two further scoping notes, so the literature is not over-claimed. "The last
channel is unthresholded" is the documented requirement of the **absolute
thresholding** family (Vamplew et al. 2011 §3.2.3; Tercan & Prabhu §3); under
**slacking** the loop runs to the last objective and slacks it too, so it is a
statement about the mechanism, not about the architecture. And per-state
Q-value thresholds are the wrong mechanism here for a *measured* reason rather
than an argued one: Vamplew et al. §7.2 pp. 75–76 report thresholded lexicographic
Q-learning performing "extremely poorly when the time objective is thresholded",
because its action selection "considers only the expected future reward … ignoring
any rewards received earlier in the current episode", failing "regardless of the
value of the threshold"; and Pineda et al. measured that lexicographic value
iteration with the per-state slack its own bound prescribes "failed to make any
significant change in costs, with respect to using no slack".

### 7.5 What A7 does **not** change, deliberately

The sub-rule definitions and their geometry (ADR-063…ADR-071); the at-fault gate;
the termination and truncation contract; the observation; the atomic vector; the
frozen panels and the split discipline; `a`, `σ` and `λ₄`. A7 is a change of
**order and adapter**, and the smaller the surface, the more of `RB51`'s
falsification evidence survives it intact.

---

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-A7-01` | `AC-A7-01` | `rulebook/v2/types.py` (`MacroRule`, `MACRO_RULE_ORDER`) | `TEST-A7-01` | Planned |
| `REQ-A7-02` | `AC-A7-02` | `rulebook/v2/registry.py`, `subrule_diagnostics.py` | `TEST-A7-02` | Planned |
| `REQ-A7-03` | `AC-A7-03` | `rulebook/v2/aggregation.py` | `TEST-A7-03` | Planned |
| `REQ-A7-04` | `AC-A7-04` | `rulebook/v2/components/progress.py` (unchanged), `reward/scalarization.py` | `TEST-A7-04`, `TEST-A7-05` | Planned |
| `REQ-A7-05` | `AC-A7-05` | delete `rulebook/v2/components/progress_rate.py`; `registry.py`; `reward/scalarization.py` | `TEST-A7-06` | Planned |
| `REQ-A7-06` | `AC-A7-06` | `reward/scalarization.py` | `TEST-A7-07`, `TEST-A7-08` | Planned |
| `REQ-A7-07` | `AC-A7-07` | `conf/scalarization/default.yaml`, `reward/scalarization.py` | `TEST-A7-09` | Planned |
| `REQ-A7-08` | `AC-A7-08` | `reward/scalarization.py` (`_validate_five_channel_weights`) | `TEST-A7-10` | Planned |
| `REQ-A7-09` | `AC-A7-09` | `conf/agent/planner/algorithm/*.yaml` | `TEST-A7-11` | Planned |
| `REQ-A7-10` | `AC-A7-10` | the A7 specification; `scripts/measure_expert_rulebook_transition.py` | `TEST-A7-12` | Planned (`M1`) |
| `REQ-A7-11` | `AC-A7-11` | measurement only | `TEST-A7-13` | Planned (`M1`) |
| `REQ-A7-12` | `AC-A7-12` | measurement only | `TEST-A7-14` | Planned (`M7`) |
| `REQ-A7-13` | `AC-A7-13` | `tests/test_rulebook_v51_orderings.py`, retargeted | `TEST-A7-15`, `TEST-A7-16` | Planned |
| `REQ-A7-14` | `AC-A7-14` | `docs/implementation/reward_learnability_ab_screening_exec_plan.md` | `TEST-A7-17` | Planned (`M8`) |
| `REQ-A7-15` | `AC-A7-15` | this document §15 | — | Planned |

---

## 9. Test Strategy Defined Before Implementation

### 9.1 Acceptance criteria

- `AC-A7-01` — `MACRO_RULE_ORDER` is exactly `(collision_safety, interaction_risk,
  non_relaxable_compliance, negotiable_lane_compliance, mission_progress)`, and no
  other module restates an ordering.
- `AC-A7-02` — the atomic vector still exposes fourteen entries, thirteen sub-rule
  costs plus `Δq`, and the observation dimension `D` is unchanged.
- `AC-A7-03` — `K4`'s denominator stays 3 when a sub-rule is inapplicable.
- `AC-A7-04` — `K5` carries the bare signed advance, and `Σ_t Δq_t =
  (s_T − s_0)/D_REF` to numerical tolerance below the clip.
- `AC-A7-05` — no channel, sub-rule, configuration key or observation field named
  `progress_rate`, `advance_shortfall`, `relaxable_weight` or
  `progress_rate_weight` survives on the A7 path.
- `AC-A7-06` — a satisfied `K4` contributes exactly zero and a violated one costs
  `w₅·(1 + σ·c)`, with `w₅` reaching no other contribution.
- `AC-A7-07` — the resolved scalarization block is the A7 mode at
  `(a, σ, λ₄, w₅) = (2.5, 0.30, 2.0, 0.15)` with the A7 schema id.
- `AC-A7-08` — the §5.4 predicate admits that set and refuses each of a declared
  list of inadmissible ones, at construction.
- `AC-A7-09` — every algorithm configuration declares `γ = 0.9982`, every
  `learning_potential_gamma` equals it, and `ln(a)/−ln(γ) > L` holds at the `L`
  read from the frozen index.
- `AC-A7-10` — every budget `τ₁`–`τ₄` is a value read off the expert per-episode
  distribution by §6.4's rule, with the excluded records listed.
- `AC-A7-11` — `fraction_below_standstill` under A7, **measured**, ≤ 7.45 %.
- `AC-A7-12` — production equals the offline instrument to `1e-9` on every step of
  the frozen Waymo `train` panel.
- `AC-A7-13` — the ordering battery holds on the scalar arm, and each
  strict-lexicographic failure is reported with the deciding channel.
- `AC-A7-14` — `AB-LEARN` records the amendment, and its resolved-config diff
  still shows single-factor invariance.
- `AC-A7-15` — §15 declares `P11`, collide-to-escape, `C50` and `D14` with their
  figures.

### 9.2 Mandatory matrix, frozen before any production change

| ID | Level | Behaviour | Fixture/input | Expected | Requirement |
|---|---|---|---|---|---|
| `TEST-A7-01` | Unit | Five channels in the declared order; nothing hard-codes it twice | `MACRO_RULE_ORDER`, `COST_MACRO_RULES` | order matches §2; `COST_MACRO_RULES` derives from it | `REQ-A7-01` |
| `TEST-A7-02` | Unit | The atomic vector is unchanged | synthetic component results | fourteen entries, thirteen sub-rule costs | `REQ-A7-02` |
| `TEST-A7-03` | Unit | `K4` fixed denominator | one applicable sub-rule, two not | `c/3`, not `c/1` | `REQ-A7-03` |
| `TEST-A7-04` | Unit | `K5` signed and clipped; `Σ Δq` telescopes below the clip | station sequence | `(s_T−s_0)/D_REF` | `REQ-A7-04` |
| `TEST-A7-05` | Unit | **The progress index is 4, and the range check follows it** | `(0,0,0,−1,+1)` accepted; `(0,0,0,+1,0)` rejected at index 3 | first passes, second raises `ScalarizationEvaluationError` | `REQ-A7-04` |
| `TEST-A7-06` | Unit | Nothing named `progress_rate`/`advance_shortfall` survives on the A7 path | registry, config allow-list | `KeyError` / configuration error | `REQ-A7-05` |
| `TEST-A7-07` | Unit | The A7 formula equals §5.1 term by term, written out independently | five-entry channel vectors, six cases | exact | `REQ-A7-06` |
| `TEST-A7-08` | Unit | `w₅` reaches only the `K4` term | one vector, two `w₅` values | priority contributions identical; reward differs by `Δw₅·[(step−1)+σ·m₅]` | `REQ-A7-06` |
| `TEST-A7-09` | Config | The resolved scalarization block is the A7 set | `--cfg job --resolve` | exact values and schema id | `REQ-A7-07` |
| `TEST-A7-10` | Unit | The §5.4 predicate is a constructor gate | the selected set, plus five inadmissible ones | admits / raises with "rank-preservation" | `REQ-A7-08` |
| `TEST-A7-11` | Integration | One shared discount, its shaping twin, **and the criterion's verdict** | the six algorithm configs and the frozen index | `γ = 0.9982` everywhere; `508.6 > 500` | `REQ-A7-09` |
| `TEST-A7-12` | Measurement | The expert per-episode exposure distributions exist and fix the budgets | `M1`'s run | four distributions plus per-record tail rows | `REQ-A7-10` |
| `TEST-A7-13` | Measurement | Below-standstill under A7 | `M1`'s grid member | ≤ 7.45 % | `REQ-A7-11` |
| `TEST-A7-14` | Integration | Production vs the instrument, per step | 20 frozen records, then the full 1100 | equal to `1e-9` | `REQ-A7-12` |
| `TEST-A7-15` | Regression | The ordering battery under A7, scalar arm | the retargeted `O1`–`O6` fixtures at **both** weight pairs | pass, O3 included | `REQ-A7-13`, `V4` |
| `TEST-A7-16` | Regression | Strict-lexicographic failures are reported with the deciding channel | same fixtures | each failure names its channel | `REQ-A7-13` |
| `TEST-A7-17` | Config | `AB-LEARN` single-factor invariance survives | resolved-config diff of the two arms | differences confined to the reward group and run identity | `REQ-A7-14` |
| `TEST-A7-18` | Property | Every cost channel in `[0,1]`, `K5` in `[−1,1]`, on a random grid | random channel vectors | invariant holds | `REQ-A7-01` |
| `TEST-A7-19` | Numerical | Sub-tolerance margins clamp to exactly zero and the indicator does not fire | `−1e-9` and `−1e-6` at `K3` | clamped case earns `λ₄`; `−1e-6` does not | `REQ-A7-06` |
| `TEST-A7-20` | Smoke | End-to-end training under the A7 reward | `make smoke` | exit 0, no NaN | all |
| `TEST-A7-21` | Regression | The full suite is green and no test is skipped, weakened or xfailed | `make gate` | `PASS · FULL` | all |

`TEST-A7-14` is the load-bearing one, for the reason `RB51`'s `T-RB51-12` was: it
converts A7's published numbers from claims about a script into claims about
production. `TEST-A7-05` is the one that would otherwise be discovered at
runtime.

**Written first, red for the right reason.** `M3` writes `TEST-A7-01`…`-10` and
`-18`…`-19` against the unmodified tree and records **which error each fails
with**, because a test that fails on an import is not yet testing anything. That
is `RB51`'s `M0` discipline, which caught exactly this.

### 9.3 Disposition of `RULEBOOK-V5.1`'s acceptance criteria

Recorded here so `DEC-A7-012` is a decision on a table rather than on a
principle.

| Criterion | Under A7 |
|---|---|
| `AC-RB5.1-01` O1–O6 on the §10 fixtures | **Changes.** O3 holds again on the scalar arm at both discounts (+0.581 / +2.458). Restated as `AC-A7-13` |
| `AC-RB5.1-02` O1–O6 under strict lex, or each failure reported | **Changes.** A7's strict-lex arm loses O2 where A0's passes it — but A0's pass is a zero-traffic artefact: add one `K2` step in forty at the 0.05 residual the repository's own test uses and A0 fails O2 too. Once the residual any real trajectory accrues is present, A7 is no worse on every ordering and strictly better on O3 |
| `AC-RB5.1-03` expert mean return positive | **Carries.** Re-measured in `M1` |
| `AC-RB5.1-04` below standstill ≤ 7.45 % | **Carries as the binding cost.** `AC-A7-11`, measured not projected |
| `AC-RB5.1-05` the §4.1 clip binds on no step the agent can produce | **Carries as `NOT ESTABLISHED`.** The engine-force cap bounds travel, not projection. `l4_clip_binding_steps` makes it measurable |
| `AC-RB5.1-06` the selected weights satisfy §5.4 for every `k` | **Changes** to the A7 tail. `AC-A7-08`; margins in §5.1 |
| `AC-RB5.1-07` `Σ Δq` telescopes | **Carries.** `AC-A7-04`. Inexact where the clip binds, which is `C50`/`C51`'s territory and unchanged by A7 |
| `AC-RB5.1-08` every atomic cost exposed | **Carries unchanged.** `AC-A7-02` |
| `AC-RB5.1-09` validation and test splits consulted by no calibration | **Carries unchanged** |
| `AC-RB5.1-10` `η` reaches only `L5` | **`NOT_APPLICABLE`** — `η` is deleted |
| `AC-RB5.1-11` v5.1 not worse than v5.0 on five columns | **Carries, re-measured** in `M1` |
| `AC-RB5.1-12` O3 against the §4.6 reference shortcut in both arms | **Changes** — this is A7's headline gain |
| `AC-RB5.1-13` `λ₆` strictly below its O3 bound | **`NOT_APPLICABLE`** — `λ₆` is deleted |
| `AC-RB5.1-14` the predicate admits `(λ₄, η, λ₆)` and rejects an inadmissible `λ₆` | **Changes** to `(λ₄, w₅)` |
| `AC-RB5.1-15` the below-standstill baseline is `−λ₆·(Δt/T_REF)·T` | **`NOT_APPLICABLE`** — with `L6` gone the baseline returns to exactly 0, which is what makes the count rise (§6.3) |
| `AC-RB5.1-16` one shared discount above the 199-step horizon | **Changes**: `γ = 0.9982` above the **measured 500-step** horizon. `AC-A7-09` |
| `AC-RB5.1-17` `L6` refines only ties | **`NOT_APPLICABLE`** — `L6` is deleted. The premise was itself undiscounted |

### 9.4 Commands

Every one exists in this repository today.

| Purpose | Command |
|---|---|
| Merge gate with recorded evidence | `make gate` |
| Working-loop check (excludes the nine `integration` tests) | `make check` |
| Narrowed iteration (marks the run `PARTIAL`) | `make gate GATE_ARGS="tests/test_scal_v14.py -k a7"` |
| Full tests through the primary environment | `make test` |
| Full tests in a provisioned container | `uv run --no-sync python -m pytest -q` |
| Rulebook v2 suite, scoped Ruff, whitespace | `make rulebook-v2-check` |
| Lint | `make lint` |
| Focused format check | `make format-check PYTHON_QUALITY_PATHS="<paths>"` |
| End-to-end training smoke | `make smoke` |
| Whitespace | `git diff --check` |
| Config composition | `docker compose -f compose.yaml run --rm dev uv run --no-sync python -m thesis_rl.cli.train --config-name <preset> --cfg job --resolve` |

Type checking is **unavailable**: no global mypy target or configuration exists
and this plan does not invent one. New and materially modified public interfaces
carry annotations.

**Two operational facts, so a red result is classified rather than chased.** From
a fresh worktree the first `make gate` can fail about ten bundled-fixture tests
that pass on a relaunch — a project-machine condition, not a repository defect.
And `/scratch` is a shared 1.8 TB volume carrying docker's root; it hit 100 % on
2026-09-09 and produced `OSError: Errno 28` on the same class of tests, which is
also the machine and not the code.

---

## 10. Milestones

Each is sized to be independently verifiable and to fit one agent context window.

### `M1` — The measurement — **not started, no gate**

Instrument-only. It changes no approved behaviour, adds no production code path
and needs no specification, so it can begin immediately and its outputs are what
several §6 gates are decided on.

- [ ] Objective: emit what A7's budgets and A7's declared cost are read from.
- Files: `scripts/measure_expert_rulebook_transition.py`,
  `tests/test_expert_rulebook_transition_instrument.py` (or the existing
  instrument tests), and an audit directory for the outputs.
- Tasks:
  1. **Per-episode exposure distributions** for `X_int = Σ max(c_ttc, c_clearance,
     c_rss_lateral)`, `X_hard = Σ max(c_offroad, c_signal, c_stop, c_crosswalk,
     c_vehicle_yield, c_speed_limit)`, `X_soft = Σ (c_solid_line +
     c_wrong_carriageway + c_dashed_line)/3` and `X_imp`, **both raw and divided
     by the mission span `Q`**, with quantiles and the per-record rows of the top
     1 %. The per-step values already exist at `:2543-2550`; the addition is four
     episode accumulators reset at episode start and appended at episode end, in
     the same shape as `v51_telescoping_residuals` already uses.
  2. **`a7_reward(...)` and `a7_is_rank_preserving(...)`**, transcribed from §5.1
     independently of production, beside the existing `v51_*` pair — the oracle
     discipline that made `T-RB51-12` possible.
  3. **An A7 grid**: `w₅ ∈ {0.15, 0.25}` × `φ ∈ {0, 0.25}` at
     `a = 2.5, σ = 0.30, λ₄ = 2.0`, plus `λ₄ ∈ {1.25, 1.9}` at
     `w₅ = 0.15, φ = 0`. Six members. `fraction_below_standstill` per member,
     against a standstill baseline that is exactly 0 under A7 — so
     `v51_standstill_return`'s A7 counterpart must return 0 rather than inherit
     the `λ₆` form.
  4. **Argmax-within-level frequency** on the v5.1/A7 channels: how many
     applicable steps each sub-rule wins its level's `max`. `worst_named` returns
     the label already, but the v5.1 channels discard it — `v51_l2` takes
     `final_r2`'s value without its blame and `v51_l3` builds its `max` inline
     with no blame at all (§4.3). The existing `final_episode_blame` is reward
     *mass* on the four-level family, which cannot answer this: mass and frequency
     differ, and a sub-rule with zero mass is indistinguishable from one that
     never applied. `F12`'s specific question is whether `clearance`, whose
     maximum cost is 0.7094 against 1.0 for the other two `K2` sub-rules, is ever
     the argmax at all.
  5. **Two `λ₄` members at the production pair** (task 3), which is the free
     output §6.8 exists to justify.
- Tests: instrument unit tests for the four accumulators against a hand-built
  episode; `a7_reward` against §5.1 term by term; the argmax counter against a
  fixture where the argmax is known.
- Commands: `make check`, then the 45-minute panel run, then
  `make gate GATE_ARGS="tests/test_expert_rulebook_transition_instrument.py"`.
- Completion evidence: the run's JSON report committed to a dated audit directory
  (summaries and the tail rows, not the megabyte-wide per-record files, which are
  regenerable — the convention the two 2026-09-09 audits already follow), plus the
  six-member table.
- Decision dependencies: none. Its outputs feed `DEC-A7-002`, `-003`, `-005`.

### `M2` — The specification document and the ADR — **blocked on `DEC-A7-001`**

- [ ] Objective: the approved contract A7 is implemented against.
- Files: `docs/specifications/rulebook_v5.2_specification.md` (name subject to
  `DEC-A7-001`), `docs/decisions/ADR-083-*.md`, `docs/project_index.md`.
- Tasks: write §2–§5 (channels, aggregation, `K5`, the adapter, the predicate),
  §9 (the acceptance criteria of §9.3), §10 (the fixtures), and the budgets from
  `M1`; record the approval evidence and date; set `APPROVED` /
  `Authoritative: YES` only after the user's approval, and move it into
  `docs/specifications/` at that point. **No production code before this is
  approved** — `AGENTS.md` is explicit that an `UNDER_REVIEW` specification is not
  an implementation contract.
- Decision dependencies: `DEC-A7-001` through `-005`, `-011`, `-012`.

### `M3` — The frozen test matrix — **blocked on `M2`**

- [ ] Objective: §9.2's matrix written and failing for the right reason.
- Files: `tests/test_a7_channels.py`, `tests/test_scal_v15.py`,
  `tests/test_a7_orderings.py`.
- Evidence: for each test, the exact error it fails with on the unmodified tree.
- Decision dependencies: `DEC-A7-006`, `-007`.

### `M4` — The scalar adapter (`SCAL-V1.5`) — **blocked on `M3`**

- [ ] `reward/scalarization.py`: the new mode, the arity and **the progress index
  as declared data**, the weight set, `_validate_five_channel_weights`, the schema
  id. The five legacy modes untouched.
- Tests: `TEST-A7-05`, `-07`, `-08`, `-10`, `-18`, `-19`.
- Commands: `make gate GATE_ARGS="tests/test_scal_v15.py"`, then `make check`.

### `M5` — The five channels — **blocked on `M4`**

- [ ] `rulebook/v2/{types,registry,aggregation}.py`, the deletion of
  `components/progress_rate.py`, and the consumers `RB51`'s `M1` had to update for
  the mirror-image change: `subrule_diagnostics`, the ACL `usefulness` weights,
  `video_diagnostics`, `analysis/tables/make_rulebook_tables.py`,
  `contracts/reward_semantics.py`.
- Tests: `TEST-A7-01`, `-02`, `-03`, `-04`, `-06`.
- Expect a ripple of the same class as `RB51`'s 50 tests (§4.4). A test that
  encoded the old contract is migrated; a test that encoded a *behaviour* is a
  finding and stops the milestone.

### `M6` — The discount — **blocked on `M2`**

- [ ] Six algorithm configurations to `γ = 0.9982`, both
  `learning_potential_gamma` with them, and the **verdict assertion**
  (`break_even_steps > horizon_steps`) added beside the value, which
  `tests/test_hydra_agent_presets.py`'s own docstring reserves for this change.
  Closes `C49`.
- Tests: `TEST-A7-11`.
- Independent of `M4`/`M5` and cheap; sequenced after `M2` only because the
  specification records the value.

### `M7` — Oracle agreement and the panel — **blocked on `M5`, `M6`**

- [ ] `TEST-A7-14` on 20 records, then the full 1100; re-measure
  `AC-RB5.1-03`/`-04`/`-11`'s columns under A7 against `M1`'s projection and
  record the difference; `TEST-A7-15`, `-16`, `-20`.
- Evidence: `oracle_max_divergence` per sub-rule, the panel table, `make smoke`
  exit 0. `V1` is discharged here or explicitly carried.

### `M8` — Reconciliation of the surrounding documents — **blocked on `M7`**

- [ ] `AB-LEARN` amended per `DEC-A7-008`, including its stale `γ` line, and
  `TEST-A7-17` executed. `RB51` closed per `DEC-A7-009`, and its `M9` detached.
  `docs/project_index.md:232` corrected and A7's own row added.
  `docs/open_items.md`: `C49` closed, `V4` closed, `D15` addressed, `C50`/`D14`
  restated as A7's declared limitations, `V1` updated.

### `M9` — Adversarial review and the gate — **blocked on `M8`**

- [ ] `make gate` `PASS · FULL` with its log cited; an adversarial review from a
  session that did not write the change, given this plan and the `M2`
  specification as the yardstick; §15 completed; the final diff read.

**Note on where the gate evidence lives.** `make gate` writes to
`outputs/gate/<timestamp>-<commit>.log` **inside the worktree**, so a log produced
here dies with the worktree. The gate that counts is relaunched from `main` after
the merge.

---

## 11. Progress And Findings Log

**2026-09-09 — plan created.** Read in full: `.agent/PLANS.md`; both 2026-09-09
audit directories; `BEHAVIOURAL-SPEC-DRAFT.md` (`P1`–`P14`, `I1`–`I5`);
`docs/open_items.md` rows `C49`, `C50`, `C52`, `C53`, `D1`, `D14`, `D15`, `V4`;
**both** colliding ExecPlans end to end; `tests/test_scal_v14.py`,
`tests/test_rulebook_v51_orderings.py`, `tests/test_hydra_agent_presets.py`'s
discount guards; `src/thesis_rl/reward/scalarization.py`;
`src/thesis_rl/rulebook/v2/types.py`; and the instrument's channel site, grids and
blame path.

**Finding, material, and it changes the sequencing.** The handoff's statements
about `RB51` are second-hand from `docs/project_index.md:232`, and that row is
stale: `RB51`'s gates were approved 2026-08-20, `M0`–`M7` and `M-DIAG` are done,
the six-level hierarchy **is** in production, and the row's "Production is at
`SCAL-V1.1` with four margins" is the plan's own pre-implementation findings-log
sentence promoted into the register. What is open in `RB51` is the tail of `M8`,
not `M4`/`M7`. Recorded in §4.1; disposition in `DEC-A7-009`; the row is corrected
in `M8`.

**Finding, material.** `_canonicalize_bounded` hard-codes the progress index as
`3`. Under A7 progress is index 4, so the unmodified function would range-check
`K4` as `[-1,+1]` and `K5` as `[-1,0]` and **raise on every step with positive
progress**. It fails loudly rather than silently, which is the good case, and
`TEST-A7-05` pins both directions. §4.2.

**Finding, minor, and a correction to the audit's own citation.** `REVIEW.md` §6
says `worst_named` "already exists and is discarded at
`measure_expert_rulebook_transition.py:2455-2456`". At the commit that document
was written against, those two lines are `v51_l2 = final_r2` and `v51_l3 =
max(...)`, so the claim is right in substance and easy to misread as broader than
it is: the blame labels **are** consumed, but only by the four-level `final`
family's reward-mass attribution, and the v5.1 channels — the ones A7 replaces —
throw them away. What `F12` needs is argmax *frequency*, which the mass
attribution cannot supply. §4.3, `M1` task 4.

**Arithmetic reproduced rather than relayed** (§14 records the commands). The
`w₅` cap 0.384615, the two crossovers 0.313983 and 1.538462, the factor of
exactly 4, `λ₄ ≤ a/2 = 1.25`, the discount table, both grids' admissible member
counts, and A7's §5.4 margins all reproduce.

**One transcription trap, found by reproducing a figure that did not match.** A
first transcription of A7's §5.4 predicate put `w₅` into the lower-level set for
both the `(1+σ)` factor **and** the `φ` count, giving 1.0911 / 1.0513 / 1.0225 and
a thinnest margin at `k=3`. `REVIEW.md`'s 1.0975 at `k=2` is correct: `φ`
multiplies only the *priority* levels below `k`, which is what the implementation
does. Recorded in §5.1 because the same reading would produce a wrong §5.4
predicate in `M4`, and because it is the class of error the audit's own §7
confesses to — a figure computed under one term placement and quoted under
another.

**Finding, minor, same class.** `REVIEW.md` quotes the cost of one fully-violated
interaction step as **8.375** in §4 and **8.125** in §8.5. Both are right: the
first retains `φ = 0.25`, the second is A7's `φ = 0`. Re-derived
(`a²·[(0−1) + σ·(−1)] + φ·(−1)` at `a = 2.5, σ = 0.30`). Since the completion
incentive is a comparison *between* A0 and A7, quoting one of the two for both
sides would be a mixed-frame figure, so §6.3 and §15 now carry both with their
configuration named.

**Next step:** the user's approval of §6's gates. `M1` needs none of them and can
start now.

---

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-A7-001` | `RULEBOOK-V5.1` §3: six ordered levels with relaxable lane compliance and progress rate below progress | Five channels with negotiable lane compliance **above** progress and no progress-rate level | The architecture the user approved 2026-09-09; grounds per requirement in §3 | Architecture approved 2026-09-09; the specification text is `DEC-A7-001` | `AC-RB5.1-01`, `-02`, `-10`, `-13`, `-15`, `-17`; `tests/test_rulebook_v51_*`, `tests/test_scal_v14.py` |
| `DEV-A7-002` | `RULEBOOK-V5.1` §5.1 / `SCAL-V1.4`: six free weights `(a, σ, φ, λ₄, η, λ₆)` | Four free weights `(a, σ, λ₄, w₅)` | §6.1–§6.3 | `DEC-A7-002`, `-003`, `-004` | `conf/scalarization/default.yaml`, `reward/scalarization.py`, `tests/test_scal_v14.py` |
| `DEV-A7-003` | ADR-081 / `AC-RB5.1-16`: `γ = 0.996` with the criterion evaluated at `L = 199` | `γ = 0.9982` with the criterion evaluated at the measured `L = 500` | `C49`; §5.2 | Approved 2026-09-09 | the six algorithm configs, `tests/test_hydra_agent_presets.py` |
| `DEV-A7-004` | ADR-072: the relaxable lane rules sit below progress | Reverted for those three sub-rules | v5.0's pathology was the **price**, not the placement: v5.0 charged them at `a = 2.2` per violated step, so standing still won past **37** relaxed steps against a mean Waymo mission. At `w₅ = 0.15`, 14.7× less, standing still wins only past **416** relaxed steps at full severity — twice the Waymo episode, and 848 against a mean PG mission over a 500-step episode. Re-derived today | `DEC-A7-002` | ADR-083 records the reversal |
| `DEV-A7-005` | `AB-LEARN` `REQ-AB-009`: the reward under test is v5.1 + `SCAL-V1.4` | The reward under test becomes A7 + `SCAL-V1.5` | §6.6 | `DEC-A7-008` | `AB-LEARN` §3, §5, §7.1, §14 |

---

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `docs/implementation/rulebook_a7_five_channel_hierarchy_exec_plan.md` | Added | This plan |
| `docs/specifications/rulebook_v5.2_specification.md` | Planned addition | `DEC-A7-001`; the contract `M4`–`M7` implement against |
| `docs/decisions/ADR-083-*.md` | Planned addition | The architecture and the discount |
| `scripts/measure_expert_rulebook_transition.py` | Planned modification | `M1`: four episode accumulators, `a7_reward`, the A7 grid, the argmax counter, two `λ₄` members |
| `src/thesis_rl/reward/scalarization.py` | Planned modification | The A7 mode, arity **and progress index** as declared data, the weight set, the predicate, the schema id |
| `src/thesis_rl/rulebook/v2/types.py` | Planned modification | Five channels, five-margin result |
| `src/thesis_rl/rulebook/v2/registry.py` | Planned modification | Level re-mapping, `advance_shortfall` deregistered |
| `src/thesis_rl/rulebook/v2/aggregation.py` | Planned modification | Four cost channels; `K4`'s declared denominator unchanged |
| `src/thesis_rl/rulebook/v2/components/progress_rate.py` | Planned deletion | `L6` is deleted |
| `src/thesis_rl/rulebook/v2/subrule_diagnostics.py` | Planned modification | Level membership |
| `src/thesis_rl/curriculum/scenario_acl/usefulness.py` | Planned modification | Per-level weights |
| `src/thesis_rl/runtime/io/video_diagnostics.py` | Planned modification | Level labels |
| `src/thesis_rl/analysis/tables/make_rulebook_tables.py` | Planned modification | Recorded channel names |
| `src/thesis_rl/contracts/reward_semantics.py` | Planned modification | The checkpoint identity |
| `conf/scalarization/default.yaml` | Planned modification | The A7 mode and weight set |
| `conf/agent/planner/algorithm/{ppo,ppo_sb3,sac,sac_sb3,td3,td3_sb3}.yaml` | Planned modification | `γ = 0.9982` and its shaping twin |
| `tests/test_scal_v14.py` | Planned modification | `DEC-A7-006` |
| `tests/test_rulebook_v51_orderings.py` | Planned modification | `DEC-A7-007`, and `V4` |
| `tests/test_hydra_agent_presets.py` | Planned modification | The verdict assertion |
| `tests/test_a7_channels.py`, `tests/test_scal_v15.py`, `tests/test_a7_orderings.py` | Planned addition | §9.2's matrix |
| `tests/test_rulebook_v51_levels.py`, `test_rulebook_v2_{monitor,wrapper,transition}.py`, `test_rulebook_v51_diagnostics.py`, `test_audit_block_b_reward_identity.py`, `test_analysis_optional_ci_and_r4_split.py`, `test_hydra_preset_run_configs.py`, `test_rulebook_provenance_identity.py` | Planned modification | The measured ripple (§4.4) |
| `docs/implementation/reward_learnability_ab_screening_exec_plan.md` | Planned modification | `DEC-A7-008` |
| `docs/implementation/rulebook_v5.1_six_level_hierarchy_exec_plan.md` | Planned modification | `DEC-A7-009` |
| `docs/project_index.md` | Planned modification | A7's row, and the correction of `RB51`'s |
| `docs/open_items.md` | Planned modification | `C49`, `V4`, `D15`, `C50`, `D14`, `V1` |
| `docs/audits/a7_calibration_2026-09-XX/` | Planned addition | `M1`'s committed evidence |

---

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| Independent re-derivation of A7's §5.4 margins, the `w₅` cap and crossovers, both grids' admissible member counts, and the discount table (standalone transcription of the predicate, no repository import) | `PASS` | 2026-09-09 | A0 1.1165 / 1.1121 / 1.1792; A7 with `φ=0.25` 1.1105 / **1.0975** / 1.1390; A7 with `φ=0` 1.1514 / 1.1478 / **1.1390**. Cap 0.384615; crossovers 0.313983 and 1.538462; ratio exactly 4.0; `a/2 = 1.25`. `v51_weight_grid` **77 of 100** admissible, `v51_calibration_grid` **16 of 36**, and **exactly one** member at `a = 2.5, σ = 0.30`. `γ = 0.9982`: break-even **508.6** against `L = 500`, `γ^L = 0.4062` against `1/a = 0.4`, effective horizon **556**; `γ = 0.996`: break-even **228.6**, `γ^L = 0.1348`, horizon 250; required `γ = 0.998169`. `λ₄ ∈ {1.25, 1.9}` admissible under the A7 tail at the production pair. One fully-violated interaction step costs **8.125** at `φ = 0` and **8.375** at `φ = 0.25`. Standing still wins past **416** relaxed steps at `w₅ = 0.15` against a mean Waymo mission and **848** against a mean PG one, versus **37** at v5.0's `a = 2.2`. Every figure agrees with the audit except the two recorded in §11 |
| `git fetch origin` and branch check | `PASS` | 2026-09-09 | `main` at `3e58ce0`, level with `origin/main`, tree clean |
| `make check` | `NOT_RUN` | — | This change adds one document and no code path, so there is nothing for the suite to exercise. It is run at the head of `M1`, which is the first milestone that touches a file the suite reads |
| `make gate` | `NOT_RUN` | — | `M9`. The user's recorded state for `3e58ce0` is `PASS · FULL · 1927 passed`; this plan does not re-cite it as its own evidence |
| `TEST-A7-01` … `TEST-A7-21` | `NOT_RUN` | — | Defined in §9.2 before any production change, as `AGENTS.md` requires. Each is scheduled on the milestone that implements its requirement |
| `make smoke` | `NOT_RUN` | — | `M7`. It has not been run since the discount changed (`V1`), and this plan changes the discount again |

**Nothing above is recorded as passing that was not executed.** The remaining
risk of the one row that did run is that it re-derives the predicate rather than
importing it; that is deliberate — an oracle that imports the thing it checks
cannot disagree with it — and `M4` closes the gap by making
`_validate_five_channel_weights` and `a7_is_rank_preserving` two independent
implementations of the same inequality, which is what `T-RB51-12` did for the
channels.

---

## 15. Final Reconciliation

Not reachable: the plan is `AWAITING_DECISIONS`, no gate in §6 is resolved, and
no milestone has started.

### Known limitations, stated in advance

1. **`P11` — no credit without motion — is violated, and A7 makes it visible
   rather than causing it.** One clipped route-projection jump with no motion pays
   **+1.345** under A7 against a standing-still baseline of exactly **0.000**;
   under the six-level reward the same probe's *gap* is +1.358 but the baseline is
   **−2.757**, so the exploit is a relative gain there and an absolute one here.
   Deleting `L6` removes the negative baseline that was masking it. This is the
   declared cost of `DEC-A7-004` and it is bounded by `C50`, below.
2. **`C50` — the negative clip is an unbounded ratchet.** A closed loop over a
   hairpin whose legs are one lane apart pays **+36 channel units = +72 reward
   units per lap at zero net displacement**, linear in laps, executed against the
   real `RoutePolyline`. Decided 2026-09-09: **change nothing in the reward.** The
   position rests on two narrow facts, and the narrower one is worth carrying: the
   frozen population does not admit the geometry **within 5 m of lateral reach**,
   and nothing in the runtime bounds the reach — `out_of_route_done: false`,
   `relax_out_of_road_done: false`, and `is_physically_out_of_road` degrades for
   route drift by ADR-053's design. The same audit run records an on-route
   under-charge of **max +128.751 with 1601 of 3500 routes positive** and a
   branch-switch worth **180.484 channel units** at about 58 m of lateral
   excursion. **And every algebraic closure spends a specification amendment**:
   bounding the station cursor at one clip width is the one candidate that
   *removes* a mechanism — the clip becomes the identity and the telescoping
   acceptance criterion becomes exact by construction — and
   `driving_mission_v1.1_specification.md:104` withdrew it as a
   "continuity/clamp" protocol. Under A7 this matters more than under A0, because
   in the thresholded regime `K5` is the only gradient inside budget.
3. **`D14` — a legal parallel corridor is not excluded, and A7 does not close the
   remedy space.** Measured over all 3,500 frozen records at 1 m station spacing:
   **318 (9.1 %)** have same-direction drivable surface outside the route's own
   carriageway, at least one ego width wide, reachable across at most one ego
   width of non-drivable surface, from which the ego misses the final gate; median
   length 15.0 m at a median offset of 10.70 m; exactly 1 of the 318 is part of
   the assigned route. **Read it in metres, not as a share of route** — the share
   is flattered by short routes, median 29.1 m against the index's 113.6 m — so
   the honest headline is **6 records (0.17 %)** carrying a corridor worth a whole
   mean mission and **7 (0.20 %)** whose route is even of median length. The
   mechanism is junction geometry: 199 of the 318 are `topology=intersection`,
   over-represented at 12.2 % against 3.4 % for `simple`. **Decided 2026-09-09:
   no remedy, recorded as an observation.**

   **And the finding is about the specification, not about A7.** An earlier
   reading held that A7 leaves nothing below `K4` and therefore closes the last
   remedy shape; that is wrong and was disputed. The remedy wants to sit *above*
   progress, and A7 is the architecture that puts it there — under the shipped
   order an off-corridor rule's natural home is `L5`, *below* progress, so the
   off-route trajectory banks the larger progress total and `L5` is never
   consulted; under A7 the same channel is `K4`, above `K5`, where an in-corridor
   trajectory has `K4 = 0` exactly and wins before progress is compared. What
   forecloses the remedy is that
   `driving_mission_v1.1_specification.md:104` forecloses **three different
   remedy shapes in one sentence** — "continuity/clamp/freeze/recovery/
   accumulated-travel/HMM protocols", "off-route `R4` zeroing", and "runtime
   authority of any final lateral envelope". So it is not that no remedy has been
   found: **every shape that would work was withdrawn by one approved document.**
   By `AGENTS.md`'s Scientific Argument Standards that is a finding, and usually
   the specification is the thing to fix. Unforeclosing any of the three is the
   user's decision and is not this plan's.
4. **Collide-to-escape is a scalar-arm result, not a defect.** Enduring 200 steps
   of interaction violation at `c = 0.9` against colliding at fault on step 20,
   under A7: the scalar arm **prefers colliding by 1092.8** (−1265.96 against
   −173.14) while the ordered arms compare `K1` first, where the non-colliding
   trajectory is **0.0000** against **0.5788** and wins at any `τ₁ < 0.5788`.
   Architecture-independent, and `I4` says no bounded scalar sum can do better.
   This is the mirror image of `I5`, where a scalarization is structurally better
   than any ordered arm, and both magnitudes are measured. It belongs in the
   write-up as a result.
5. **`P14`'s second clause holds under no architecture.** Mission success is a
   **zero-value terminal**, so completing rather than stopping short is worth
   0.57–0.77 reward units under A7 against **8.125** for one fully-violated
   interaction step at `φ = 0` (8.375 if `φ = 0.25` is retained). A7's incentive
   is smaller in magnitude and better in kind than A0's (§6.3), and a terminal
   bonus is not recommended.
6. **`O3` as a dominance is unobtainable, and `I1` proves it.** No Markov,
   bounded, per-step progress channel has a duration-invariant discounted return,
   so the ordering must be restated as a **finite exchange rate**. A7 satisfies it
   as an exchange rate on one `w₅`; it does not restore the undiscounted identity,
   and no hierarchy in any order at any level count can.
7. **The thresholded arm's mechanism is open** (§7.4), and `D1`'s `τ₄` decision
   is downstream of it. A7 reduces the number of constrained channels from five to
   four and removes the one that could not be thresholded at all, which is a
   reduction of the problem, not a solution to it.
8. **`K3`'s evidence is Waymo-only.** Five of the six `K3` sub-rules never apply
   on PG, so `K3` there is `offroad` alone and the two sources are graded by
   substantially different rulebooks. Pooled reporting stays prohibited.
9. **A7's projected below-standstill is a projection until `M1` runs.** ≈5.2 %
   against 7.45 %, built from a measured 4.55 %, a bounded +0.53 pp from deleting
   `λ₆`, and ≈0.27 reward units of `w₅` indicator cost. `M1`'s grid member is what
   replaces the word "projected" with a number.
10. **`φ = 0`'s effect on the expert's mean return under the six-level reward is
    unmeasured**, and its falsifier is one grid member in the same run (§6.2).

### Deferred required work

`D1`'s threshold mechanism and `τ₄`; the distributional arm; `RB51`'s detached
`M9`; `C52`'s ADR-035 amendment; the `AB-LEARN` relaunch, which needs
`DEC-AB-004`.

### Optional improvements

One cheap measurement would close `C50` either way: whether the medial axis of
each route's largest fold is itself drivable. And
`route_fully_outside_max_run`, silently empty on every official panel evaluation
until `C53` was fixed on 2026-09-09, is the statistic that would turn `C50` and
`D14` from geometry into behaviour — it has not been run over a panel since the
fix.
