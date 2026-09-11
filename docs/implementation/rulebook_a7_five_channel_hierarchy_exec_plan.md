# ExecPlan — A7 five-channel rulebook architecture and `γ = 0.9982` (`A7`)

## 1. Metadata

| Field | Value |
|---|---|
| Feature | Replace the six-level `RULEBOOK-V5.1` hierarchy with the A7 five-channel architecture — progress last and unthresholded, `L6` deleted, the negotiable-lane channel charged by a bounded satisfaction indicator — and move the shared discount to `γ = 0.9982` |
| Plan ID | `A7` |
| Authoritative specification for the architecture being replaced | `docs/specifications/rulebook_v5.1_specification.md` (`RULEBOOK-V5.1`, `APPROVED` 2026-08-14, amended 2026-08-20 and 2026-09-07), `SCAL-V1.4` (§5) |
| Authoritative specification for A7 | **None. It does not exist yet, and writing it is `DEC-A7-001` and milestone `M2` of this plan.** No production implementation may begin before it is approved |
| Evidence of record | `docs/audits/rulebook_architecture_2026-09-09/` (the candidate bench, the impossibility results, the recommendation and its derivations) and `docs/audits/progress_channel_integrity_2026-09-09/` (authoritative wherever the two disagree) |
| Status | `APPROVED` — every gate in §6 resolved 2026-09-10. `M1` executed 2026-09-10 (§10, §11, §14); **`M2` written 2026-09-11 and awaiting the user's approval of `RULEBOOK-V5.2` and `ADR-083`** (§10, §11, §14); `M3` onward wait on that approval |
| Created | 2026-09-09 |
| Last updated | 2026-09-11 |
| Branch | `worktree-a7-execplan`, from `main` at `3e58ce0` |
| Related ADRs | ADR-072 (partly reverted: the negotiable lane rules return above progress), ADR-076 (`L6`, deleted), ADR-075 and ADR-081 (the discount), ADR-063…ADR-071 (the sub-rules, untouched), ADR-035 and ADR-053 (context for `C50`/`D14`). **A new ADR is required** for the architecture and the discount; the next free number is `ADR-083` |
| Owner | Single maintainer; there is no reviewer to assign (`AGENTS.md`, Branching And Pull Requests) |

**What is approved, and by whom.** The user approved **the A7 architecture** and
**`γ = 0.9982`** on 2026-09-09, and on **2026-09-10** ratified every remaining gate
in §6: `w₅ = 0.15`, `φ = 0`, the budget *rule*, both mandatory-test changes, the
specification form, the `AB-LEARN` amendment, the `RB51` disposition and the
vocabulary break with its pooling guard — plus the two decisions an adversarial
review of §6 added, `DEC-A7-013` and `DEC-A7-014`.

**`M1` measured what was still open, on 2026-09-10, and nothing it found reopens
a decision.** `τ₁ = 0.000`, `τ₂ = 110.3798`, `τ₃ = 21.2874`, `τ₄ = 23.3865`
(undiscounted realized maxima, no panel-defect exclusion declared; §6.4). `φ = 0`'s
falsifier did not trip: `fraction_below_standstill` is **4.64 %** at `φ = 0` against
**4.73 %** at `φ = 0.25`, both under the 7.45 % ceiling (§6.2). `DEC-A7-010`'s two
alternatives measured **6.45 %** at `λ₄ = 1.25` and **4.91 %** at `λ₄ = 1.9` against
`λ₄ = 2.0`'s **4.64 %** — confirming the pre-registered reading order (§6.8).
`DEC-A7-013`'s co-occurrence measured **15 of 217,189 steps (0.0069 %)** across
**5 of 1100 episodes**, near-zero correlation (§6.9). Full evidence in
`docs/audits/a7_m1_measurement_2026-09-10/`.

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
  plan fixes the budget *values* from the panel and declares the requirement on
  the undiscounted realized form; **which object a threshold is enforced on stays
  `D1`'s**, which is why `M1` emits all three rather than one (§6.4).
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
- **Writing the ACL re-base.** `DEC-A7-014` records that `ACL-SN-EMA-001` v2.0
  cannot run on an A7 rulebook and blocks the dependent ACL work behind `M5`, but
  no ACL revision is drafted here: the A7 specification does not exist yet, so a
  revision written now would be drafted against a contract that is not yet
  authoritative (§6.10).
- **The `make_rulebook_tables.py` name defect** (§4.6). It predates A7, it is
  fixed separately and ahead of `M5`, and folding it in would put two causes in
  one diff and destroy the pre-fix evidence.
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
| §5.4 predicate | Constructor-time, over the three priority weights, with `tail = λ₄ + η·(Δt/T_REF) + λ₆·(Δt/T_REF)` and `bound = (1+σ)·Σ(lower) + φ·(number of lower priority levels) + tail` | `reward/scalarization.py:147-196` `_validate_six_level_weights` |
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

### 4.5 The three plans A7 collides with

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

**`ACL-PROG-004` / `ACL-SN-EMA-001` v2.0 — the third collision, and this section
missed it.** Found by an adversarial review of §6 rather than here.

| Fact | Detail | Label |
|---|---|---|
| The ACL contract cannot run on an A7 rulebook | `ACL-SN-EMA-001` v2.0 §3.2 treats **six level margins as fatal if absent**: true today (`MACRO_RULE_ORDER` has six entries), false after `M5` | `VERIFIED` |
| This plan already edits a file the ACL plan owns | §13 lists `curriculum/scenario_acl/usefulness.py` as a planned modification; its weight map is keyed on `progress_rate` at `:37`, the level A7 deletes | `VERIFIED` |
| `DEC-206`'s ordering rests on a circular argument | It rejected `…→L5→T` because that "would invert ADR-072" — a citation of this repository's own ADR, which `AGENTS.md` does not admit as evidence. Applying its own `REQ-001` principle to the A7 order yields the rejected form | `VERIFIED` |
| `REQ-002` is unaffected | A7 preserves sub-rule names, membership, `K4`'s denominator of 3, the at-fault gate and the atomic vector | `VERIFIED` |

Disposition in `DEC-A7-014` and §6.10: record the re-base as a gate here, and do
not write an ACL revision before the A7 specification exists.

### 4.6 Directly relevant debt

`C49` (the discount; this plan closes it), `C50` and `D14` (decided; declared),
`C52` (the user's), `D1` (`τ₄`'s mechanism; out of scope), `D15` (O3 at the
shipped discount; A7 restores O3 on the scalar arm and `I1` explains why the
undiscounted dominance form is unobtainable), `V4` (the ordering fixtures'
weight pair; folded into `DEC-A7-007`), `V1` (`make smoke` not run since the
discount change), `C2`/`C3` (`signal` under-firing, independent of the order),
`C8` (a run's provenance artifacts mislabel the rulebook family — worth knowing
before reading any A7 run's banner, and not A7's to fix), and `ACL-PROG-004`'s
`DEC-206` (reopened without prejudice by `DEC-A7-014`).

**One defect found while verifying this section, and it is not A7's to fix.**
`analysis/tables/make_rulebook_tables.py:18` declares
`R4_PROGRESS_MARGIN_RULE_NAME = "route_progress"` and compares it against the
recorded rule name at `:200` and `:203`. But `route_progress` is a **v1** rulebook
name; the v2 value is `MacroRule.MISSION_PROGRESS.value == "mission_progress"`,
built at `rulebook/v2/wrapper.py:429` and carried to the `rule_name` column. The
comparison can never match, so `rulebook_r4_progress_margin.csv` is written with a
header and no rows, and the signed `[-1,+1]` progress margin instead falls into
`rule_violation_by_rule.csv` under violation-rate columns — the terminology mixing
that file's own comment declares must never happen. History dates the constant to
`e4be417` ("rulebook v4.8"), so it was born correct and was **orphaned by
`DEC-RB51-005`'s rename of 2026-08-20**, which updated fifty tests but not this
consumer; nothing went red because it is not a test. Classification per
`AGENTS.md`: a **defect of the repository**. It is fixed separately and **before**
`M5` touches that file, so the two causes do not share a diff and the pre-fix
evidence can still be captured; recorded in `docs/open_items.md` with a regression
test that asserts the filtered name against `MACRO_RULE_ORDER` rather than against
a literal, so the next rename cannot orphan it again.

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

**Every gate was resolved by the user on 2026-09-10.** Three of the twelve originally listed here were not decisions at all and are marked as such rather than carrying a signature they did not need: `DEC-A7-004` is entailed by the approved architecture, `DEC-A7-010` was decided when `λ₄` was approved, and `DEC-A7-012` is a mechanical consequence of the ratified weights. Two decisions were **added** after an adversarial review of this section found them missing: `DEC-A7-013` and `DEC-A7-014`. `M1` depends on no gate and may begin.

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-A7-001` | Specification clarification | A7 has no specification document. What form does it take? | (a) a new `RULEBOOK-V5.2` + `SCAL-V1.5` superseding v5.1 §3/§4/§5/§9/§10; (b) an in-place amendment of v5.1, the form ADR-081 used; (c) a separate amendment file, the form `OBS-V1.3.1` used | **(a)** | **Measured, and it reverses the cost this row first assumed.** `RULEBOOK-V5.1` is cited by 85 files with **112 section citations** across 52 of them, and **63 of those 112 (56 %) point at sections A7 rewrites** — §3 (19), §4.6 (14, a section that ceases to exist), §5.5 (10), §5 (7), §4.4 (7), §5.1 and §5.4 (3 each). An in-place amendment leaves all 63 silently meaning something else. A new version invalidates none, because this repository **never re-points citations**: `SUPERSEDED` is a first-class status in `docs/project_index.md`'s own vocabulary, v5.0→v5.1 and v4.12→v4.13 are both full-version precedents, and v4.7 is still cited by name in 25 live `src/`/`tests/` files | **Approved 2026-09-10** |
| `DEC-A7-002` | Specification deviation | `w₅`, the weight on `K4` | 0.15 / 0.25 / any value in the derived window | **`w₅ = 0.15`** | See §6.1. Costs the expert **0.265** reward units per episode | **Approved 2026-09-10** |
| `DEC-A7-003` | Specification deviation | `φ`, the shared absolute tie-breaker | remove (`φ = 0`) / keep 0.25 | **remove** | See §6.2, with its falsifier in `M1`. The braking criterion is pinned in its strict form, `a_req^max > 9 m/s²`, which 10.96 clears by 22 % | **Approved 2026-09-10** |
| `DEC-A7-004` | Consequence, not a gate | `η` and `λ₆`, and the `L6` level that carries `λ₆` | — | **removed** | **This is entailed by the approved architecture and was wrongly listed as open.** A7 *is* five channels, so `L6` is gone by the block the user approved, and `w₅` replaces `η` as `K4`'s weight — that substitution is the mechanism that distinguishes A7 from A6. Recorded as a consequence in §6.3, with the argument restated on measurement rather than on the shipped specification's own admission | **Not a gate** |
| `DEC-A7-005` | Specification clarification | The budgets `τ₁`–`τ₄` have no values, **and no agreed object** | rule: (a) the panel maximum with declared exclusions; (b) a quantile; (c) defer to `D1`. Object: undiscounted realized `X_i` / per-span `X_i/Q` / discounted expected `E[Σ γ^t c_k]` | **(a) on all three objects**, criterion in §6.4, values filled in by `M1` | **Revised: the first draft of this row pre-registered p99 of `X_i/Q`, which contradicted its own criterion and silently decided a question this plan puts out of scope.** A budget that fails to admit the human is falsified, and that implies the maximum, not a quantile. And `I1b` proves the three objects do not coincide, while `D1` — which object a threshold applies to — is out of scope, so `M1` emits all three and A7 declares the requirement on the undiscounted realized form | **Rule approved 2026-09-10; measured by `M1` 2026-09-10**: `τ₁ = 0.000`, `τ₂ = 110.3798`, `τ₃ = 21.2874`, `τ₄ = 23.3865`, no exclusion declared (§6.4) |
| `DEC-A7-006` | Mandatory test change | `tests/test_scal_v14.py` changes substantially | amend / delete and rewrite / leave and add a second file | **amend in place**, itemised in §6.5 | It is a mandatory test and `AGENTS.md` requires recorded approval. One test (`test_l6_reaches_only_the_last_term`) has no A7 counterpart and is **deleted**, not weakened | **Approved 2026-09-10** |
| `DEC-A7-007` | Mandatory test change | The `O1`–`O6` fixtures change, and they carry `V4` | amend for A7 only / amend and parametrise over both weight pairs / amend and read the shipped pair from the configuration | **amend and parametrise over both pairs**, closing `V4` in the same change | The fixtures assert at `a = 2.2, σ = 0`, which production has not used since ADR-081. Every ordering was checked to hold at the shipped pair (O3 excepted, the known `D15` failure), so this is a verification gap, not a behavioural one — but A7 rewrites these fixtures anyway, so folding it in costs one change instead of two | **Approved 2026-09-10** |
| `DEC-A7-008` | Specification clarification | `AB-LEARN`'s arm B is pre-registered on the reward A7 replaces | (a) amend the pre-registration explicitly and re-register arm B against A7 before any run; (b) run the screening on the old reward first, then A7; (c) leave `AB-LEARN` untouched and let it drift | **(a)**, and `DEC-AB-005` **survives** | See §6.6. The screening still freezes: approving the A7 specification freezes the *contract*, the screening freezes the *question of optimizability*. Without that second freeze nothing stops an A8 | **Approved 2026-09-10** |
| `DEC-A7-009` | Scope | Two ExecPlans for one reward | (a) A7 supersedes `RB51` downstream of the hierarchy, `RB51` closes as `IMPLEMENTED` with its residue named; (b) A7 becomes a milestone of `RB51`; (c) A7 sequences behind `RB51`'s `M8`/`M9` | **(a)**, with §9.3 carrying two columns | See §6.7. §9.3 states each criterion's **arrival state under v5.1** beside its disposition under A7, so the table is the reconciliation `RB51` never got rather than only a transition | **Approved 2026-09-10** |
| `DEC-A7-010` | Consequence, not a gate | `λ₄` | keep 2.0 / reduce to 1.9 / reduce to 1.25 | **keep 2.0** | Decided when `λ₄` was approved and not reopened here. §6.8 pre-registers what would reopen it, **before** `M1` produces the two numbers, so their reading cannot be post-hoc | **Not a gate** |
| `DEC-A7-011` | Specification clarification | The recorded vocabulary and the checkpoint identity both break | declare both / preserve the old channel names / emit both schemas | **declare both, and add one guard** | **Narrowed and strengthened.** It is a break of *vocabulary*, not of schema: `CSVRecorder.SCHEMAS` has no column built from `MacroRule`, so no header moves and no reader raises. And it is **not** the same class as `DEC-RB51-005`: that rename changed every channel name, so a stale consumer failed visibly, while A7 keeps four of five names, so a stale consumer produces a populated and wrong table. `_build_condition_id` (`analysis/aggregate/aggregate_runs.py:184-204`) buckets by `rulebook_config` and never by `vector_schema_id`, so pre- and post-A7 runs can pool silently. The guard: `vector_schema_id` joins `condition_id` | **Approved 2026-09-10** |
| `DEC-A7-012` | Consequence, not a gate | Nine of the seventeen `AC-RB5.1-*` criteria stop applying or change meaning | — | **restate**, table in §9.3 | A mechanical consequence of the ratified weights: once `η`, `λ₆` and `φ` are gone, `AC-RB5.1-10`, `-13`, `-15` and `-17` have no referent. Recorded rather than signed | **Not a gate** |
| `DEC-A7-013` | Specification clarification | **`K2 ≻ K3` is the only step in the hierarchy whose sole surviving argument is this repository's own specification** | keep the order and declare it unsupported / reorder / defer | **keep, declare, and measure it in `M1`** | See §6.9. `AGENTS.md`'s Scientific Argument Standards require this to be said out loud, and A7 is the moment the hierarchy is rewritten — freezing an unpriced order a second time makes the third time harder. The measurement that would price it, `K2`/`K3` co-occurrence, does not exist and is free in the run `M1` already needs | **Approved 2026-09-10** |
| `DEC-A7-014` | Blocking technical issue | **A7 re-bases the `ACL-SN-EMA-001` v2.0 contract, and no document records the coupling** | write ACL revision 4 now / record a re-base gate here / ignore it | **record the gate here; do not write revision 4 now** | See §6.10. ACL v2.0 §3.2 treats six level margins as *fatal if absent* — true today, false after `M5`, so **the ACL as written cannot run on an A7 rulebook**. This plan's own Files table already modifies `curriculum/scenario_acl/usefulness.py`, whose weight map is keyed on `progress_rate` (`:37`), while §4.5 named only `AB-LEARN` and `RB51`. Writing revision 4 now is blocked anyway: the A7 specification does not exist yet | **Approved 2026-09-10** |

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

**The criterion is pinned in its strict form, `a_req^max > 9 m/s²`, and that is a
decision rather than a convenience.** `φ = 0` spends about a third of the margin
ADR-081 bought over the physical limit, so the strict form is what makes the
choice admissible and it should not be adopted silently. The reason it is the
right form: 9 m/s² is already peak braking on dry asphalt, the worst case rather
than the typical one, so margin *above* that number protects against nothing
physical — only against uncertainty in the number itself. And if margin is later
wanted back, the lever is `σ`, not `φ`: `σ` scales with each level's own weight
while `φ` is absolute, which is why `φ` is 5.1 % of the severity slope at `k=1`
and 25.0 % at `k=3` — the most important level is the flattest, the very shape
ADR-081 criticised.

**Falsified before recommending, and falsifiable after.** Over the ten-ordering
battery at both discounts no ordering changes sign and no reward-hacking probe
changes verdict; the largest movement is O5 (waiting at a red) from +9.075 to
+8.083. The residual risk is that `φ = 0`'s effect on the expert's mean return
under the *six-level* reward is unmeasured — it was priced only on the four-level
family, whose rows cannot be read across. **The falsifier is one grid member at
`φ = 0` in `M1`'s run**: if `fraction_below_standstill` breaches 7.45 %, revert to
0.25 and accept the 1.0975 margin.

**Measured, 2026-09-10: not falsified.** At the production pair
(`a = 2.5, σ = 0.30, λ₄ = 2.0, w₅ = 0.15`), `fraction_below_standstill` under the
A7 reward on the full 1100-episode `train` panel is **4.64 %** at `φ = 0` against
**4.73 %** at `φ = 0.25` retained — both far under the 7.45 % ceiling, and the
movement (**+0.09 pp**) is a fraction of the ceiling's remaining headroom
(2.81 pp at `φ = 0`). `φ = 0` stands. The residual risk this falsifier existed to
close is closed: the six-level-reward figure this section could not price before
is now measured directly under the five-channel one it actually governs.

### 6.3 Removing `η` and `λ₆` — a consequence, argued on measurement

Not a gate (`DEC-A7-004`): A7 *is* five channels, so `L6` goes with the block the
user approved, and `w₅` replaces `η` as `K4`'s weight. What this section owes is
therefore not a justification for a choice but an argument that survives without
citing the document being replaced — and the first draft of it did not, so it is
restated here on measurement.

- **`η`.** The earlier wording — "by the shipped specification's own admission,
  the one weight the expert panel cannot discriminate" — cites the specification,
  which `AGENTS.md` forbids as evidence. The measured statement is both available
  and stronger: **it is not that the panel fails to measure `η`, it is that
  nothing pins it.** Across `η ∈ [0, 5]` the expert's mean episode return moves by
  **less than 0.1 on ≈70**, because the logged human almost never relaxes a lane
  rule; and the window in which `η` could buy ordering O3 back is **0.193 %** of
  the admissible range, so fixing it there would be fitting a weight to a single
  constructed fixture. A parameter that neither a measurement nor an ordering
  determines is a parameter that should not exist. Its job passes to `w₅`, which
  an ordering *does* determine — a window of 65.9 % to 72.1 % (§6.1).
- **`λ₆` and the `L6` level.** Four independent statements, none of them the
  specification. *Its justification has expired*: ADR-076 rested on `γ = 1`, which
  ADR-081 removed **eighteen days later**. *It duplicates a preference already
  expressed above it*: at `γ < 1` the discount supplies **90.5 %** of the time
  preference (79.7 % at `γ = 0.9982`) and supplies it at a higher priority than
  `L6`, so `L6` is a minority duplicate of something the hierarchy already says.
  *It carries no independent information*: `c_L6 = 1 − clip(Δq, 0, 1)` is a
  pointwise function of the progress channel, so the six-channel vector has **rank
  five**, and it is the only channel whose return distribution carries nothing
  beyond its mean once progress is known — precisely what the distributional
  component cannot use. *Its zero point is unreachable*: that zero sits at the
  engine cap, **4.9×** the expert's mean speed, so the channel fires on
  **99.4024 %** of expert steps at `p50 = 0.884`.
- **Deleting `L6` deletes the `Δt/T_REF` normalization entirely**, so the reward
  stops carrying two different per-step scalings.

**The cost, stated in the same breath, and there are two parts.** First,
below-standstill: the criterion moves from `R₀ < −λ₆·(Δt/T_REF)·Σ Δq⁺ = −0.81` to
`R₀ < 0`, so the count can only **rise**, by the mass in a 0.81-wide band, which
the local density between p5 and p10 bounds at **0.53 pp** — `4.55 % → ≈5.1 %`,
and `≈5.2 %` once the `w₅` indicator's own **0.265** reward units of expert cost per
episode are
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

### 6.4 The budget criterion — rule approved, values from `M1`

**This section was rewritten after an adversarial review found the first draft
wrong in two independent ways**: it pre-registered a quantile that contradicted
its own stated criterion, and it silently fixed the *object* a threshold applies
to — a question this plan declares out of scope.

#### The object, which has to be named before a number means anything

Three candidate objects exist for `τ_i`, and `I1b` proves they do not coincide:

| object | where it comes from | who would consume it |
|---|---|---|
| `X_i` — undiscounted realized per-episode exposure | the behavioural specification's own quantities | Lexicographic REINFORCE, which compares an accumulated episodic return |
| `X_i/Q` — the same, per unit of mission span | the behavioural specification's "budget units" note | the same, made comparable across a panel spanning 13–247 m |
| `E[Σ γ^t c_k]` — discounted expected channel return | the only form a per-state absolute threshold can hold | an absolute-thresholding mechanism on Q-values |

The third is not interchangeable with the first two, and the reason is a
mechanism rather than a preference: **a realized budget would require the
accumulated exposure in the policy's input**, which is an `OBS-V1.3.x` amendment
and is forbidden here. Tercan & Prabhu's Appendix D.3 states it directly for this
exact case — "the corresponding discounted threshold **actually depends on the
trajectory**". Meanwhile the policy-gradient route accumulates its episodic return
**undiscounted, from a single sampled episode**, so it consumes the first object,
not the third.

**Which object `τ` lives on is `D1`'s decision, and `D1` is out of this plan's
scope** (§2). So A7 does two things and not a third: it **declares the requirement
on the undiscounted realized form**, which is the altitude a behavioural budget
belongs at, and it has `M1` **emit all three**, because they are the same
accumulator under three weightings and cost nothing extra in the same run. What
A7 does not do is choose the mechanism by choosing a number's units.

#### The rule, and why it is the maximum rather than a quantile

Two requirements, in this order:

1. **The budget must admit the logged competent driver.** A budget quantified over
   zero-exposure completions describes no trajectory the agent can produce: the
   expert accrues interaction cost on 0.3978 % of steps and non-negotiable
   compliance cost on 0.6418 %. `τ₃ = 0` is therefore falsified **by the panel**,
   not by preference.
2. **The rule must be fixed before the numbers are seen**, or the budget is fitted
   to the distribution it is supposed to be tested against.

**Rule, approved 2026-09-10.** `τ_i` = the **maximum** of the logged expert's
per-episode value on the frozen Waymo `train` panel, minus any episode
attributable to a **declared** panel defect, with the excluded episodes **listed
per record**; p99 and p95 reported beside it as sensitivity.

The first draft pre-registered p99 instead, and that was wrong on its own terms:
requirement (1) says a budget that fails to admit the human is falsified, and p99
excludes about eleven human episodes while asserting that excluding a human
episode falsifies the budget. The two do not hold together. The objection that
sent the draft to a quantile — that one outlier record would dictate a parameter
for every other — was imported from a different situation, the route-length
normalization in which the shortest route in the panel set the bound for all of
them; here the budget is per-episode and the outlier is a trajectory a competent
human actually produced, so a budget that excludes it is a budget asserting the
human was incompetent there.

**The cost, in the same sentence:** the maximum is the loosest budget that stays
falsifiable, so in the feasible regime the thresholded arm does not constrain that
channel at all for a policy at or below human exposure, and its differentiation
from the scalar control on that channel comes from the *ordering* alone. The rule
is void unless the per-record rows are emitted — an aggregate that cannot be
audited per record is an assertion — which is why `M1` carries them.

#### `τ₁ = 0`, conditionally and measurably

At-fault impact is non-zero on at most one step (ADR-071), and a non-colliding
trajectory has `K1 = 0` exactly, so a zero budget is what makes the ordered arms
prefer enduring 200 steps of interaction violation to colliding at fault —
measured at `K1 = 0.0000` against `0.5788`, where the scalar arm prefers colliding
by 1092.8. `τ₁ = 0` is admissible **iff the logged expert records no at-fault
impact on the panel**, which is not yet measured. `M1` emits `X_imp`'s per-episode
distribution for exactly that reason; if it is non-zero anywhere, `τ₁` follows the
rule above like the others and the finding is reported.

#### One implementation note that changed a bench's conclusions

In the thresholded comparison the clip for a **cost** channel is `max(c, τ)`, not
`min(c, τ)`; for the progress channel it is `min(v, τ)`, because there higher is
better. Getting it backwards makes a collision tie with a non-collision and
inverts the reading of the progress channel — it was found and fixed in the
review's own bench, where it had changed conclusions. It follows that **`τ₁ = 0`
means zero tolerance, i.e. strict lexicographic comparison on that channel** — not
"everything ties". This belongs in the A7 specification's threshold section so the
eventual thresholded arm cannot re-invert it.

#### Measured, 2026-09-10: the four values, on all three objects

`M1` ran the rule above against the full frozen Waymo `train` panel — **1100
records, 217,189 transitions, 0 skipped** — and applied it as approved: the
per-episode maximum, no episode excluded (no panel defect is declared for this
run), with the excluded-record list therefore empty rather than omitted. Full
per-record top-1 % rows and the per-episode distributions are committed at
`docs/audits/a7_m1_measurement_2026-09-10/`.

| `τ_i` | channel | undiscounted realized max (`τ_i`) | per-span max | discounted (`γ=0.9982`) max | `p99` (realized) |
|---|---|---:|---:|---:|---:|
| `τ₁` | `X_imp` — K1 at-fault impact | **0.000000** | 0.000000 | 0.000000 | 0.000000 |
| `τ₂` | `X_int` — K2 interaction risk | **110.379799** | 0.720226 | 98.733955 | 8.067951 |
| `τ₃` | `X_hard` — K3 non-negotiable compliance | **21.287443** | 0.264242 | 19.613411 | 3.323886 |
| `τ₄` | `X_soft` — K4 negotiable lane compliance | **23.386514** | 0.514168 | 20.197176 | 2.737476 |

**`τ₁ = 0` is admissible, measured rather than assumed.** `X_imp` is exactly
`0.000000` for every one of the 1100 episodes, on all three objects: the logged
expert records no at-fault impact anywhere on the panel. The conditional §6.4
above stated is satisfied, not asserted.

**`τ₂`'s and `τ₃`'s maxima sit far above their own `p99`** (110.38 against 8.07
for `X_int`; 21.29 against 3.32 for `X_hard`), which is the shape the rule
anticipated — the panel's per-episode exposure is heavy-tailed, and the rule is
the maximum precisely because a competent human produced that outlier episode.
The single record setting each maximum (`waymo:training_20s:71344f609367eace`
for `X_int`, `waymo:training_20s:5f8217a5da0b24ff` for `X_hard`,
`waymo:training_20s:1ef1a62b6bebe06` for `X_soft`) is named in the committed
top-1 % rows for exactly this reason: **no panel defect is declared against any
of them here**, so the rule's maximum stands unmodified, and a future session
with grounds to declare one has the per-record evidence to act on rather than an
aggregate to take on faith.

**`p95`/`p99` reported as the required sensitivity**, not as the rule: had the
rule instead been "p99", `τ₂` would read 8.07 against the panel maximum's
110.38 — the exact eleven-human-episode exclusion §6.4 already rejected on its
own criterion, now with a number attached rather than only an argument.

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
- **`DEC-AB-005` survives, and saying so is load-bearing.** Approving the A7
  specification freezes the *contract* — what the reward is. The screening freezes
  a different thing, the *question of optimizability*, and it is the only
  instrument that has ever attacked this reward with an optimizer. Without that
  second freeze restated explicitly, nothing stops an A8: `RULEBOOK-V5.1` was also
  approved before `AB-LEARN`, and A7 exists precisely because the reward was
  revised after that approval. What would license the freeze is unchanged —
  `H1` and `H3` holding at the screening budget, escalated per `DEC-AB-005`.

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

**Pre-registered, before `M1` produces those two numbers.** `λ₄` reopens **only**
if `M1` measures A7's below-standstill at `λ₄ = 2.0` above the 7.45 % ceiling, and
in that case the first lever is **1.9**, not 1.25. Two reasons, and the second is
the one that has not been written down before. 1.9 also buys back A7's §5.4 margin
and costs roughly 0.2 pp against 1.25's ~1.9 pp. And **1.25 buys an option that
cannot be exercised**: its purpose is to admit a `w₅` opposing an off-corridor
drive at clip pace, but under A7 the channel that would have to fire is `K4`, and
an ego on a legal parallel carriageway violates none of its three sub-rules —
`offroad` is zero by construction over the union of vertically compatible lanes,
and `wrong_carriageway` charges only opposing surface. Firing would need a new
"off the assigned corridor" sub-rule, which needs a runtime lateral envelope,
which is the third of the three remedy shapes
`driving_mission_v1.1_specification.md:104` withdrew (§15). So the 1.9 pp would
purchase a capability the approved specification forecloses.

**Measured, 2026-09-10: the pre-registered reading holds, at the actual figures
rather than the extrapolation.** On the full 1100-episode panel, at
`w₅ = 0.15, φ = 0`: `λ₄ = 2.0` (the standing recommendation) gives
`fraction_below_standstill` **4.64 %**; `λ₄ = 1.9` gives **4.91 %**
(**+0.27 pp**); `λ₄ = 1.25` gives **6.45 %** (**+1.81 pp**). Both stay under the
7.45 % ceiling, so neither reopens `λ₄` — the pre-registered trigger condition
is not met — but the ordering the pre-registration bet on is confirmed at the
actual weights rather than the `a = 2.2, σ = 0` grid's extrapolation: `1.9`
costs about a sixth of what `1.25` costs (0.27 against 1.81 pp, against the
extrapolation's 0.2 against ~1.9), and `2.0` stands.

### 6.9 `K2 ≻ K3` is the hierarchy's least-supported step, and it is a finding

Every other adjacency in A7 rests on a mechanism or a measurement. `K1 ≻ K2` rests
on outcome-versus-indicator, and merging them is measured to halve the mitigation
gradient (`P8` margin 2.82 → 1.22). `K3 ≻ K4` is the negotiable/non-negotiable
distinction, i.e. the semantic core of minimum-violation (Castro, Tumova, Karaman,
Frazzoli & Rus), and merging them is measured to fail three orderings on the
scalar arm (O2, O4 at −23.08, O5 at −0.15). `K4 ≻ K5` has three independent
grounds (§6.1, and the literature in §3).

**`K2 ≻ K3` has none of that.** The only derivation that exists is `rulebook_v4.7`'s
— without an interaction level, a near-miss legally in lane could be preferred to a
brief illegal deviation with wide margin — v5.0 asserted the order without
restating it, and v5.1 inherited it. **The measurement that would price it, the
co-occurrence of the two channels, does not exist**; the review that scored the
candidate architectures says so in its own "what I did not verify" section, and
scored `A2a`'s merge on constructed fixtures alone for the same reason.

By `AGENTS.md`'s Scientific Argument Standards this has to be said out loud rather
than carried: **it is the one place in the hierarchy where the only argument is
this repository's own specification.** The recommendation is nevertheless to keep
the order — reordering on no evidence would replace an unsupported claim with a
different unsupported claim — but to record it in the A7 specification as the
least-supported step and to **measure it in `M1`**, which needs the run anyway. The
cost of not measuring it: A7 freezes an unpriced ordering for the second time, and
a third freeze is harder to reopen than a second.

**Measured, 2026-09-10: rare, and uncorrelated when it happens.** On the full
1100-episode, 217,189-step panel, `K2` and `K3` are simultaneously non-zero on
**15 steps (0.0069 %)**, across **5 of 1100 episodes (0.45 %)**. On exactly those
15 steps the joint distribution shows `K2` at `p50 = 0.2197` and `K3` at
`p50 = 0.0253` (both reach 1.0 at `p99`, i.e. the tail of this already-rare set
still touches full severity), with a Pearson correlation of **−0.0161** —
indistinguishable from zero at this sample size. This is a finding, not a
resolution: it does not derive `K2 ≻ K3` from a mechanism, but it bounds *how
often the order could matter at all* on the logged expert, and the answer is
almost never jointly, which is itself evidence about how much is actually
staked on this adjacency in practice. The specification (`M2`) records this
number beside the order, as declared in `M2`'s own scope note above.

### 6.10 A7 re-bases the ACL contract, and no document recorded it

`ACL-SN-EMA-001` v2.0 §3.2 treats **six level margins as a property that is fatal
if absent**. That is true today — `MACRO_RULE_ORDER` has six entries — and false
after `M5`. **The ACL as written therefore cannot run on an A7 rulebook**, by its
own design rather than by an oversight.

The gap was in *this* plan: §4.5 listed only `AB-LEARN` and `RB51` among the
collisions, while §13's Files table already modifies
`curriculum/scenario_acl/usefulness.py`, whose weight map is keyed on
`progress_rate` (`:37`) — the level A7 deletes. A plan that edits a file without
naming the plan that owns it is the same failure `DEC-A7-009` exists to prevent,
committed one plan further along.

Two further consequences, both recorded rather than resolved here:

- **`DEC-206`'s ordering is reopened without prejudice.** It chose `L1→L2→L3→T→L5`
  and rejected `…→L5→T` for one stated reason: that the latter "would invert
  ADR-072". Applying its own `REQ-001` principle to the A7 order yields the
  rejected form — and the reason it was rejected was a citation of an ADR of this
  repository, which `AGENTS.md` does not admit as evidence. So reopening it leaves
  it **open with no guarantee in either direction**, which is the honest state.
- **`REQ-002` is *not* touched.** A7 preserves the sub-rule names, the membership,
  `K4`'s fixed denominator of 3, the at-fault gate and the atomic vector.

**What this plan does, and deliberately does not do.** It records the re-base as a
gate and blocks ACL work that depends on the six-margin property behind A7's `M5`.
It does **not** write an ACL revision now: the A7 specification does not exist yet
(`DEC-A7-001`, `M2`), and this plan forbids implementation before that approval, so
a revision written now would be drafted against a contract that is not yet
authoritative. The ACL plan's own `M1` is invariant to A7 — it reads channels by
name — and may proceed; the ordering boundary is at its `M0`/`M2`.

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
| `TEST-A7-22` | Regression | Two runs whose scalarization vector schemas differ never share a `condition_id` | two synthetic run metadata records differing only in `vector_schema_id` | distinct `condition_id`s | `REQ-A7-11`, `DEC-A7-011` |

`TEST-A7-14` is the load-bearing one, for the reason `RB51`'s `T-RB51-12` was: it
converts A7's published numbers from claims about a script into claims about
production. `TEST-A7-05` is the one that would otherwise be discovered at
runtime.

**Written first, red for the right reason.** `M3` writes `TEST-A7-01`…`-10` and
`-18`…`-19` against the unmodified tree and records **which error each fails
with**, because a test that fails on an import is not yet testing anything. That
is `RB51`'s `M0` discipline, which caught exactly this.

### 9.3 Disposition of `RULEBOOK-V5.1`'s acceptance criteria

Recorded here so `DEC-A7-012` is a table rather than a principle. **Two columns,
not one** (`DEC-A7-009`): the first states each criterion's *arrival state under
`RULEBOOK-V5.1`* — the reconciliation `RB51` closes without ever having written —
and the second its disposition under A7. With only the second column this would
be a transition table, and v5.1's own criteria would be settled nowhere.

| Criterion | Arrival state under v5.1 | Under A7 |
|---|---|---|
| `AC-RB5.1-01` O1–O6 on the §10 fixtures | `PASS` undiscounted; **O3 fails at the shipped `γ = 0.996`** (`D15`) | **Changes.** O3 holds again on the scalar arm at both discounts (+0.581 / +2.458). Restated as `AC-A7-13` |
| `AC-RB5.1-02` O1–O6 under strict lex, or each failure reported | `PASS` undiscounted; O1 fails at L2 and the failure is reported, as the criterion allows | **Changes.** A7's strict-lex arm loses O2 where A0's passes it — but A0's pass is a zero-traffic artefact: add one `K2` step in forty at the 0.05 residual the repository's own test uses and A0 fails O2 too. Once that residual is present, A7 is no worse on every ordering and strictly better on O3 |
| `AC-RB5.1-03` expert mean episode return positive | `PASS` — **+70.70** at `a = 2.2, σ = 0` | **Carries.** Re-measured under A7 in `M1` |
| `AC-RB5.1-04` below standstill ≤ 7.45 % | `PASS` — 3.36 % at the calibration pair, 4.55 % at the shipped one | **Carries as the binding cost.** `AC-A7-11`, measured not projected |
| `AC-RB5.1-05` the §4.1 clip binds on no step the agent can produce | **`NOT ESTABLISHED`**, and it stays so: the engine-force cap bounds the ego's travel, not its projection | **Carries as `NOT ESTABLISHED`.** `l4_clip_binding_steps` is what makes it measurable rather than deduced |
| `AC-RB5.1-06` the selected weights satisfy §5.4 for every `k` | `PASS` — thinnest margin 1.1121 at `k=2` | **Changes** to the A7 tail: thinnest 1.1390 at `k=3`. `AC-A7-08`; margins in §5.1 |
| `AC-RB5.1-07` `Σ Δq` telescopes | `PASS` for agent trajectories; inexact on the expert panel where the clip binds | **Carries.** `AC-A7-04`. The residual is `C50`/`C51`'s territory and A7 does not touch it |
| `AC-RB5.1-08` every atomic cost exposed | `PASS` | **Carries unchanged.** `AC-A7-02` |
| `AC-RB5.1-09` validation and test splits consulted by no calibration | `PASS` | **Carries unchanged** |
| `AC-RB5.1-10` `η` reaches only `L5` | `PASS` — p1 and p5 identical across `η ∈ [0, 5]`, which is also why nothing pins `η` | **`NOT_APPLICABLE`** — `η` is deleted |
| `AC-RB5.1-11` v5.1 not worse than v5.0 on five columns | `PASS` — dominates on all five | **Carries, re-measured** in `M1` |
| `AC-RB5.1-12` O3 against the §4.6 reference shortcut in both arms | `PASS` undiscounted, margin +0.2000 decided at L5; **fails at the shipped discount** | **Changes** — this is A7's headline gain, and the ordering is decided at `K4` rather than never reached |
| `AC-RB5.1-13` `λ₆` strictly below its O3 bound | `PASS` — 0.2 < 0.25 | **`NOT_APPLICABLE`** — `λ₆` is deleted |
| `AC-RB5.1-14` the predicate admits `(λ₄, η, λ₆)` and rejects an inadmissible `λ₆` | `PASS` — 2.12 < 2.5 after ADR-081 | **Changes** to `(λ₄, w₅)`, same predicate shape |
| `AC-RB5.1-15` the below-standstill baseline is `−λ₆·(Δt/T_REF)·T` | `PASS` — `v51_standstill_return` | **`NOT_APPLICABLE`** — with `L6` gone the baseline returns to exactly 0, which is what makes the count rise (§6.3) |
| `AC-RB5.1-16` one shared discount above the horizon | `PASS` **as written**, and the criterion was wrong: it names the 199-step Waymo-only horizon, and `C49` shows the real one is 500 | **Changes**: `γ = 0.9982` above the **measured 500-step** horizon, with the verdict asserted. `AC-A7-09` |
| `AC-RB5.1-17` `L6` refines only ties | `PASS` undiscounted; the premise is itself undiscounted | **`NOT_APPLICABLE`** — `L6` is deleted |

**One arrival state is worth reading twice.** `AC-RB5.1-16` passed while being
false about the quantity it names — the guard asserted a horizon nobody had
measured, under a docstring claiming it was measured. That is the shape `C49`
records, and it is why `AC-A7-09` asserts the criterion's **verdict** and not only
its inputs.


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

### `M1` — The measurement — **done, 2026-09-10**

Instrument-only. It changes no approved behaviour, adds no production code path
and needs no specification, so it can begin immediately and its outputs are what
several §6 gates are decided on.

- [x] Objective: emit what A7's budgets and A7's declared cost are read from.
- Files: `scripts/measure_expert_rulebook_transition.py`,
  `tests/test_expert_rulebook_transition_instrument.py` (or the existing
  instrument tests), and an audit directory for the outputs.
- Tasks:
  1. **Per-episode exposure distributions** for `X_int = Σ max(c_ttc, c_clearance,
     c_rss_lateral)`, `X_hard = Σ max(c_offroad, c_signal, c_stop, c_crosswalk,
     c_vehicle_yield, c_speed_limit)`, `X_soft = Σ (c_solid_line +
     c_wrong_carriageway + c_dashed_line)/3` and `X_imp`, on **all three objects
     of §6.4**: raw `X_i`, `X_i/Q`, and the discounted `Σ γ^t c_k` at
     `γ = 0.9982`. Quantiles plus **the full per-record rows of the top 1 %**,
     because §6.4's rule is void without them. The per-step values already exist
     at `:2543-2550`; the addition is four episode accumulators reset at episode
     start and appended at episode end, in the same shape as
     `v51_telescoping_residuals` already uses, carrying three running sums each
     instead of one. Emitting all three is what stops `D1`'s eventual choice of
     mechanism from costing a second 45-minute run.
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
  6. **`K2`/`K3` co-occurrence** (`DEC-A7-013`): on how many steps, and on how
     many episodes, both channels are non-zero at once, with the joint
     distribution of `(c_K2, c_K3)` on those steps. It is the measurement that
     would price the hierarchy's least-supported adjacency, it does not exist, and
     the accumulators of task 1 already compute both channels per step — so it is
     two counters and a small joint histogram.
- Tests: instrument unit tests for the four accumulators against a hand-built
  episode, asserting all three objects separately so a weighting error cannot hide
  behind a matching total; `a7_reward` against §5.1 term by term; the argmax
  counter and the co-occurrence counter against fixtures where the answer is known
  by construction.
- Commands: `make check`, then the 45-minute panel run, then
  `make gate GATE_ARGS="tests/test_expert_rulebook_transition_instrument.py"`.
- Completion evidence: the run's JSON report committed to a dated audit directory
  (summaries and the tail rows, not the megabyte-wide per-record files, which are
  regenerable — the convention the two 2026-09-09 audits already follow), plus the
  six-member table.
- Decision dependencies: none — every gate is resolved. Its outputs supply the
  **values** `DEC-A7-005` leaves open, the falsifier `DEC-A7-003` is conditional
  on, the co-occurrence `DEC-A7-013` requires, and the two `λ₄` figures `DEC-A7-010`
  pre-registers a reading for.

**Executed 2026-09-10.** All six outputs measured on the full frozen Waymo
`train` panel (1100 records, 217,189 transitions, 0 skipped) in one run;
committed evidence at `docs/audits/a7_m1_measurement_2026-09-10/`; values
reconciled into §6.2 (`φ`), §6.4 (`τ₁`–`τ₄`), §6.8 (`λ₄`) and §6.9
(co-occurrence) above. **One blocking defect found and fixed on the way**,
unrelated to A7's design (`C54`, `docs/open_items.md`): the script's
`production_scalarization` had hard-coded a `priority_base` the six-level mode
stopped accepting on 2026-09-07 (ADR-081), so every record failed at
construction and the report silently read `"scenarios_measured": 0` — the run
could not have produced any of the six outputs before this was fixed. Details,
regression test and pre-fix evidence in `docs/open_items.md` `C54`. `make check`
and the focused instrument tests (33 tests, `tests/test_measure_expert_rulebook_transition.py`)
pass; `make gate` is `M9`'s.

### `M2` — The specification document and the ADR — **written 2026-09-11, awaiting approval**

- [x] Objective: the contract A7 is implemented against — **written**; it becomes
  the *approved* contract only on the user's approval, which is what `M3` waits
  for.
- Files: `docs/specifications/rulebook_v5.2_UNDER_REVIEW_specification.md`
  (`DEC-A7-001`; the `_UNDER_REVIEW` suffix follows `AGENTS.md` and the
  `rulebook_v5.0_UNDER_REVIEW_specification.md` precedent, and is dropped on
  approval), `docs/decisions/ADR-083-progress-last-and-the-discount-at-the-measured-horizon.md`,
  `docs/project_index.md`.
- Tasks: write §2–§5 (channels, aggregation, `K5`, the adapter, the predicate),
  §9 (the acceptance criteria of §9.3), §10 (the fixtures), and the budgets from
  `M1`; record the approval evidence and date; set `APPROVED` /
  `Authoritative: YES` only after the user's approval, and move it into
  `docs/specifications/` at that point. **No production code before this is
  approved** — `AGENTS.md` is explicit that an `UNDER_REVIEW` specification is not
  an implementation contract.
- The specification also carries three things the review of §6 added: the
  threshold *object* and the `max(c, τ)` / `min(v, τ)` clip convention (§6.4),
  `K2 ≻ K3` declared as the least-supported adjacency with `M1`'s co-occurrence
  beside it (§6.9), and the three identifier conventions this repository already
  uses recorded in one place, so a future search knows what to look for rather
  than discovering it by missing something.
- Decision dependencies: all gates resolved 2026-09-10; blocked only on `M1`'s
  values for the budgets and on the user's approval of the document itself.

**Written 2026-09-11**, in an isolated worktree, with **no production file
touched** — the whole milestone is three documents. What it produced, against the
five bullets above:

- **`RULEBOOK-V5.2`** (`UNDER_REVIEW`, `Authoritative: NO`): §1 with the O1–O6
  orderings and the two rows that change; §3 the five channels, their membership,
  their aggregation and the three normative placement decisions, plus §3.6
  (`K2 ≻ K3` declared as the least-supported adjacency with `M1`'s co-occurrence
  beside it) and §3.7 (the three identifier conventions already in use, with
  file-and-line citations, and the fourth this document adds); §4 `K5`, the
  deletion of `L6`, the discount with its verdict, and §4.5 the budgets with the
  three objects, the rule, the measured values and the `max(c, τ)` / `min(v, τ)`
  clip convention; §5 `SCAL-V1.5`, its form, its §5.4 predicate and the selected
  weights; §9 the acceptance criteria plus the two-column disposition of
  `AC-RB5.1-01`…`-17`; §10 the frozen test matrix and the O1–O6 fixtures; §11 the
  fourteen declared limitations; §14 the open decisions and the derivation record.
- **`ADR-083`** (`Proposed`): the architecture and the discount, the reason the
  ADR-072 reversal is not a return to v5.0 (priced, not argued), `w₅`, `φ = 0`,
  the budgets, the falsified alternatives, and the consequences including both
  re-based contracts.
- **`docs/project_index.md`**: one row in Scientific And Functional Documents
  registering `RULEBOOK-V5.2` as **`CANDIDATE`** — which is what the index exists
  for, since its stated purpose is to stop an apparently newer document from being
  mistaken for an approved contract — and one row in Decisions for `ADR-083` as
  `PROPOSED`. The v5.1 row is **not** moved to `SUPERSEDED` and A7's ExecPlan
  Registry row is **not** rewritten: the first is only true on approval and the
  second is `M8`'s.

Three deviations from the letter of this section, each decided rather than
assumed, and all three recorded in §12:
`DEV-A7-007` (the `_UNDER_REVIEW` filename), `DEV-A7-008` (`ADR-083` is
`Proposed`, not `Approved`) and `DEV-A7-009` (one added acceptance-criterion id,
`AC-A7-16`, and the `TEST-A7-15a`…`-15g` fixture decomposition).

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
- [ ] **The pooling guard of `DEC-A7-011`**: `vector_schema_id` joins
  `_build_condition_id` (`analysis/aggregate/aggregate_runs.py:184-204`). One
  longer tuple, and it is what makes a pre-A7 and a post-A7 run impossible to
  average together rather than merely unlikely to be — necessary here and not
  under `RB51` because A7 keeps four of five channel names, so a stale consumer
  produces a populated wrong table instead of an empty one.
- Tests: `TEST-A7-01`, `-02`, `-03`, `-04`, `-06`, `-22`.
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

- [ ] `AB-LEARN` amended per `DEC-A7-008`, including its stale `γ` line and the
  explicit survival of `DEC-AB-005`, and `TEST-A7-17` executed. `RB51` closed per
  `DEC-A7-009`, and its `M9` detached. A7's own row added to
  `docs/project_index.md`, whose `RB51` row was already corrected when this plan
  was registered. `docs/open_items.md`: `C49` closed, `V4` closed, `D15`
  addressed, `C50`/`D14` restated as A7's declared limitations, `V1` updated.
- [ ] **`ACL-PROG-004` reconciled per `DEC-A7-014`** (§6.10): the re-base recorded
  in that plan and in the index, `DEC-206` marked reopened without prejudice, and
  the six-margin *fatal if absent* property of `ACL-SN-EMA-001` v2.0 §3.2 named as
  what `M5` invalidates. No ACL revision is written here.

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

**2026-09-10 — every gate ratified, and an adversarial review of §6 changed four
things.** The user ratified `DEC-A7-002`, `-003` and the budget rule, signed the
two mandatory-test changes, and recorded no dissent on the specification form, the
`AB-LEARN` amendment, the `RB51` disposition or the vocabulary break. Status
`AWAITING_DECISIONS` → `APPROVED`.

*Three rows were withdrawn as gates, because they were never decisions.*
`DEC-A7-004` is entailed by the approved architecture — A7 *is* five channels, so
`L6` goes with the block that was approved and `w₅` replaces `η` by the same
mechanism that distinguishes A7 from A6. `DEC-A7-010` was decided when `λ₄` was
approved. `DEC-A7-012` is a mechanical consequence of the ratified weights.
Presenting all three as open inflated a plan that was already motivated, and the
inflation was the method's: a design-tree interrogation applied *after* a plan is
written manufactures questions where choices are already made.

*One recommendation of this plan was wrong and is reversed.* §6.4 pre-registered
`τ_i = p99` of `X_i/Q`. It contradicted its own criterion — a budget that fails to
admit the human is falsified, and p99 excludes about eleven human episodes — and
the objection that produced it (one outlier dictating a parameter for all) was
imported from the route-length normalization, where the shortest route set the
bound for every record. Corrected to the panel maximum with declared exclusions
and per-record rows. This is the only reversal in the plan; nothing else changed
direction.

*Two decisions were missing and are added.* `DEC-A7-013`: `K2 ≻ K3` is the one
adjacency in the hierarchy whose sole surviving argument is this repository's own
specification, which `AGENTS.md` requires be said out loud, and the co-occurrence
that would price it does not exist — so it becomes `M1`'s sixth output.
`DEC-A7-014`: A7 re-bases the `ACL-SN-EMA-001` v2.0 contract, whose §3.2 treats six
level margins as *fatal if absent*, and §4.5 had named only two colliding plans
while §13 was already editing `curriculum/scenario_acl/usefulness.py`, keyed on
`progress_rate` at `:37`. The same failure `DEC-A7-009` exists to prevent,
committed one plan further along.

*One question the plan had answered too narrowly.* The budgets have three
candidate **objects**, not one, and `I1b` proves they do not coincide; which one a
threshold lives on is `D1`'s decision, which this plan puts out of scope. `M1` now
emits all three, so that decision costs no second run.

*Two figures re-derived, one measurement commissioned, one defect found.* The
§5.4 margins and the discount table reproduce (§14). `DEC-A7-001`'s cost was
measured rather than assumed: 112 section citations across 52 files, **63 of them
pointing at sections A7 rewrites**, against a repository that never re-points
citations — which reverses the argument from "a changelog is ugly" to "an in-place
amendment silently invalidates 56 % of them". And
`analysis/tables/make_rulebook_tables.py:18` was found comparing a **v1** rule name
against a v2 recorded value since `DEC-RB51-005`'s rename, producing an empty table
and a wrong average in silence; classified and routed to its own fix ahead of `M5`
(§4.6).

**Next step:** `M1`. Every gate is resolved and it depends on none of them.

---

**2026-09-10 — `M1` executed.** In an isolated worktree
(`worktree-a7-m1-measurement`, from `main` at `48c25ca`). Read in full before
touching anything: this plan's §4.3, §6.4, §6.8, §6.9 and §10/`M1` — the audits
under `docs/audits/` were deliberately not reread, because this plan already
carries their derivations with sources, and rereading the audits directly had
cost three prior sessions on since-retracted claims.

**One blocking defect found before any of the six outputs could be produced,
and it is not A7's.** `production_scalarization` in
`scripts/measure_expert_rulebook_transition.py` constructed
`ScalarizationConfig(mode="six_level_priority_weighted_rank", priority_base=2.2, ...)`.
`priority_base=2.2` was correct when that line was written (2026-09-01); ADR-081
moved the six-level mode's *required* base to 2.5 on 2026-09-07
(`SIX_LEVEL_PRIORITY_BASE`, `d983c87`) without this call site following it, so
`ScalarizationConfig.__post_init__` rejected construction on every one of the
1100 records, the replay's own exception handler counted each as a skipped
scenario, and the report read `"scenarios_measured": 0` — silently, exactly the
shape a comment already on that line documents for a different, earlier cause.
Classified per `AGENTS.md` as a repository defect (would fail identically for
any contributor, any machine) rather than a local gap, recorded as `C54`
(`docs/open_items.md`) rather than carried in this plan, per the Proportionality
rule: a bounded fix that changes no approved behaviour. Fixed by importing
`SIX_LEVEL_PRIORITY_BASE` instead of restating it as a second literal, so the
two cannot drift apart again; regression test
`test_production_scalarization_config_matches_the_shipped_configuration` reads
`conf/scalarization/default.yaml` directly rather than hardcoding a comparison
value, the same discipline `_shipped_gamma()` already uses in
`tests/test_rulebook_v51_orderings.py`. Pre-fix evidence is an executed run, not
an inference: the unmodified tree, run end to end, reports
`"scenarios_measured": 0` and `"scenarios_skipped":
{"error:ScalarizationConfigurationError": 1100}`
(`outputs/a7_m1/a7-m1-run.log`).

**The six outputs, measured on the fixed tree**, full frozen Waymo `train`
panel, 1100 records, 217,189 transitions, 0 skipped (`docs/audits/a7_m1_measurement_2026-09-10/`):

1. Per-episode exposure distributions for `X_imp`, `X_int`, `X_hard`, `X_soft`
   on all three §6.4 objects — reconciled into §6.4 above as `τ₁`–`τ₄`.
2. `a7_reward(...)` and `a7_is_rank_preserving(...)`, transcribed independently
   from §5.1 and checked against this plan's own worked figures (1.1514 /
   1.1478 / 1.1390 at `φ=0`; the fully-violated-K2 costs 8.125 / 8.375) before
   being trusted for the run — the transcription trap the plan records at §5.1
   (`φ` reaching K4 as well as K1-K3) was checked for and does not reproduce.
3. The six-member A7 grid's `fraction_below_standstill` — reconciled into §6.2
   (`φ`) and §6.8 (`λ₄`) above; every member under the 7.45 % ceiling.
4. Argmax-within-level frequency: on `K2`, `clearance` wins 96 of 217,189 steps
   (0.044 %) — rare, but `F12`'s "ever at all" is answered yes; `rss_lateral`
   615 (0.283 %), `ttc` 153 (0.070 %). On `K3`, `crosswalk` and `speed_limit`
   never win the argmax on this panel (absent from the count entirely);
   `offroad` 1285 (0.592 %), `signal` 38 (0.018 %), `vehicle_yield` 55
   (0.025 %), `stop` 16 (0.007 %). Both levels' non-"none" complements
   (0.3978 % for `K2`, 0.6418 % for `K3`) reproduce the per-step violation
   rates §6.4 already cited from an independent source, which cross-checks the
   new counters against a figure they were not built from.
5. The two `λ₄` alternatives — reconciled into §6.8 above.
6. `K2`/`K3` co-occurrence — reconciled into §6.9 above.

**Committed**: `docs/audits/a7_m1_measurement_2026-09-10/` (summaries and the
per-record top-1 % tail rows the budget rule requires; not the full 217,189-row
per-record file, which is regenerable from the committed command). `make check`
green; the focused instrument suite (33 tests,
`tests/test_measure_expert_rulebook_transition.py`, including 10 new for this
milestone) green. `M9`'s own gate re-run from `main` after the merge is still
what closes the plan (its log dies with this worktree), but the full `make
gate` was also run here before merging this milestone's branch, per
`AGENTS.md`'s Branching And Pull Requests — `PASS | FULL`, 1937 passed, 5
skipped, in 4m04s (§14).

**Next step:** `M2` — the A7 specification document, now unblocked with the
budget values it needed.

---

**2026-09-11 — `M2` written.** In an isolated worktree
(`worktree-a7-m2-specification`, from `main` at `7f8860e`, level with
`origin/main`). Read in full before writing: `AGENTS.md`, this plan end to end
(because `M2` has to reconcile §3, §5, §6.4, §6.9 and §9 against each other),
`docs/audits/a7_m1_measurement_2026-09-10/` (README and `a7_m1_summary.json`),
`RULEBOOK-V5.1` §1–§5, §9, §10 and §12, `rulebook_v5.0_UNDER_REVIEW_specification.md`
§12, `ADR-081` for the ADR form, and `docs/project_index.md`'s vocabulary and
maintenance rules. The two 2026-09-09 audit directories were deliberately **not**
reread, for the reason `M1` gives: this plan already carries their derivations
with sources, and those documents contain retracted claims.

**No production file is touched.** The milestone is three documents, and
`AGENTS.md` forbids production implementation while the specification is
`UNDER_REVIEW`.

**Every formula and value was reproduced before being written down, not
transcribed.** A standalone script importing nothing from this repository
recomputed the §5.4 predicate from its derivation, the margins in all four
configurations, the `w₅` cap and both crossovers, the factor of exactly 4,
`λ₄ ≤ a/2`, the per-step costs, the discount table and the required `γ`, and the
standing-still crossover of `DEV-A7-004`. **The transcription trap of §5.1 was
reconstructed and reproduces this plan's own wrong figures** (1.0911 / 1.0513 /
1.0225 with the thinnest margin at `k=3`), so the right ones are checked against
a known-wrong alternative rather than only against themselves. The shipped
predicate and scalar form were also read in
`src/thesis_rl/reward/scalarization.py:147-196, 393-425` to confirm the term
placement the arithmetic assumes — `φ` multiplies only the priority levels below
`k`, and `w₅` enters once with `(1+σ)` because `K4` now carries an indicator and
a severity slope where `η` carried neither. Every budget, grid and co-occurrence
figure was read back out of `a7_m1_summary.json` rather than copied from §6.

**One figure was reported as not reproducing, and that report was wrong.** This
log first recorded that §6.1's "**1.83 m/s**" did not reproduce and that
`RULEBOOK-V5.2` carried 2.17 m/s instead. **§6.1 is right.** The exchange rate is
`Δq = w₅·(1 + σ·c_K4)/λ₄`, and 1.83 m/s is its value at `c_K4 = 1/3` — one
marking, which is the severity of the reference construction §6.1's own sentence
is about — while 2.17 m/s is the value at `c_K4 = 1`, the fully violated channel.
The error was in the specification, which attached the `c = 1` figure to the
one-marking sentence; `RULEBOOK-V5.2` §5.5 now states the formula and all three
severities, and §14.2 records the correction. Nothing in the contract moved
either way.

**Two figures were reported as carried rather than reproduced, and they do
reproduce.** O3's scalar margins and `w₅`'s lower bounds are recomputable from
v5.1 §4.6's stipulated pair plus one clause this plan does not state — the 30
marking steps are the shortcut's **first** 30. Reconstructed and validated
against a figure it was not built from: under the v5.1 form it reproduces **all
six entries** of v5.1 §4.4's published margin table, and only at that placement.
A7's margins then come out **+0.5813** at `γ = 0.996` and **+2.4579** at
`γ = 0.9982`, and the bounds **0.131342** / **0.073555**, against §6.1's printed
0.1313 / 0.0736. The placement matters and should be stated in §6.1: at
`γ = 0.996` the margin falls to **−0.0141** once the marking starts past step 33.
What does **not** reproduce from any live source is `w₅`'s *upper* bounds
0.4477 / 0.2641 — six candidate models tried, none produces the pair.

**One citation is incomplete and is flagged in the document rather than dropped.**
Pineda, Wray & Zilberstein supply two load-bearing results for the thresholded
arm's limitations (Lemma 1's NP-hardness, and the measured null result of
per-state slack), and no full bibliographic entry for them exists anywhere in
this repository — only this plan's §7.4. `RULEBOOK-V5.2` §12 carries the citation
marked as incomplete and to be completed before approval.

**Three deviations decided in this milestone**, all reported rather than silent:
`DEV-A7-007` (the `_UNDER_REVIEW` filename), `DEV-A7-008` (`ADR-083` is
`Proposed`, since the decisions are approved but this wording is not) and
`DEV-A7-009` (`AC-A7-16` added, because §9.2 maps `TEST-A7-22` to a requirement
that does not cover it; plus the `TEST-A7-15a`…`-15g` fixture decomposition).

**2026-09-11 — an independent audit of `M2`, and one blocking finding.** Thirteen
agent sessions re-checked six zones against the primary sources (the algebra of
§5.1/§5.4 against the shipped code; every number against
`a7_m1_summary.json`; this plan against itself; `RULEBOOK-V5.2` against
`RULEBOOK-V5.1` and ADR-069/072/073/074/075/076/081; the code against the
contract; and O1–O6 re-derived from scratch), each zone with an adversarial
reviewer instructed to refute. Every claim reported below was then re-verified
by hand before being written down.

**The blocking finding: the §5.4 predicate counts the progress swing once, and
`rulebook_v5.0` §6.3 counted it twice.** v5.0 states the same predicate with
`2λ`, derives it symmetrically, and names the reason — the cost channels are
bounded in `[−1, 0]` while the progress channel is bounded in `[−1, +1]`. The
two-sided form reproduces **both** anchors v5.0 checks itself against, `a > 2`
(Veer et al.) and `a ≥ 2.92` at `σ = 1` (reproduced: 2.9196); the one-sided form
reproduces **neither** (`a > 1`, `a ≥ 2.83`). A two-state counterexample refutes
the "iff" at the selected weights: `K3` violated as `m₃ → 0⁻` with `Δq = +1`
scores −0.5000 against −2.0000 for nothing violated with `Δq = −1`. Under the
two-sided form `λ₄ = 2.0` is inadmissible at `k = 2, 3` — **and the six-level
weight set in production today fails at `k = 1` as well** — so this is a
pre-existing defect of `RULEBOOK-V5.1` §5.4 and of ADR-081's calibration, not one
A7 introduced. But A7 is the change that rewrites the predicate and `AC-A7-08`
makes it a constructor gate, so freezing it again is a decision. Note the
corroboration: `λ₄ ≤ a/2 = 1.25`, which §6.8 attributes to an unrelated
desideratum, **is** the two-sided bound at `w₅ = 0`. Recorded as question 1 of
`RULEBOOK-V5.2` §14.1 with three candidate answers and their costs; the
inequality is left exactly as production enforces it until the user decides.

**One further material finding, and it is not an error.** At the budgets §6.4
fixes, the **thresholded** comparison does not reproduce A7's gain on O3: the
reference pair accrues `Σ c_K4 = 10.0` against `τ₄ = 23.386514`, so it ties on
every cost channel and falls through to progress, where the shortcut wins. It
would need 70 of its 160 steps on a marking to leave the budget. This is §6.4's
own declared cost of the maximum rule applied to the one ordering the restructure
was for; `RULEBOOK-V5.2` §4.5 and §11.15 now declare it, because §11.1 invokes
the thresholded comparison where it favours A7.

**Five further questions for the sessions that agreed these versions**, all in
`RULEBOOK-V5.2` §14.1: the model behind `w₅`'s upper bounds 0.4477 / 0.2641
(unreproducible, and 0.2641 is the binding one at the adopted discount); the
90.5 % / 79.7 % time-preference split of §6.3 (unreproducible; the live version
in v5.1 §4.6 gives 71.9 %); the marking placement of the reference pair; whether
`l4_clip_binding_steps` / `l5_reached_steps` keep their literal names; and
`REQ-A7-10`'s "per unit of mission span", which the approved budget rule and the
specification both contradict.

**Corrections applied to `M2`'s own documents**, all of them the specification's
errors rather than this plan's: the exchange-rate severity (above); the two
"not reproduced" rows of §14.2; `K3` on the PG panel (§3.5 said "five sub-rules
rather than six" where §8 and §11.12 say `offroad` alone); the expert-mean
comparison (71.34 against **68.31** at the matched weight pair, not 70.70 at
`(2.2, 0)`); φ's share of the slope (denominator named); the ADR-081 weakening
factor (2.2× in effective horizon, **3.01×** in `γ^L`, the currency ADR-081
argues in); "exactly two rows" → three, plus the thresholded case; §11's
carry-over rule, which was keyed on item numbers this document renumbers; one
cross-reference and one reference count.

**Defects of this plan the audit found and `M2` did not fix**, because they are
the plan's to fix and none changes the contract: §15's opening still says no
milestone has started; §9.2 maps `TEST-A7-18`, `-19` and `-22` to requirements
that do not cover them and §8's traceability table stops at `TEST-A7-17`; §14
records `TEST-A7-01 … -21` as `NOT_RUN` beside the evidence that `-12`, `-13` and
`-21` ran; §6.4's third budget object is labelled an expectation and measured as a
realized maximum; the `relaxable_lane_compliance → negotiable_lane_compliance`
rename appears only inside `AC-A7-01`, with no requirement or decision row
carrying it; `AC-A7-15` has no test anywhere; and `REQ-A7-10`'s "13–247 m" span
figure does not reproduce against the frozen index (Waymo `train` measures
10.0–578.4 m).

**2026-09-11 — question 1 priced, and a recommendation.** Nine further agent
sessions investigated the blast radius, how negative `Δq` can actually go, the
solution space, and whether an expert-measured bound is admissible evidence for a
contract that must hold for a learned policy. Every load-bearing figure below was
re-verified by hand.

**The defect is real and its practical reach is narrow.** The binding break-even
is at `K3`: the compliant trajectory must lose **0.3389 m of station in one
0.1 s step** (3.39 m/s of backward projection) before the ordering inverts;
`K2` needs 0.89 m and `K1` needs 2.28 m, i.e. **the collision channel is not
purchasable even at the clip** (break-even `−1.0275` against a clip of `−1`).
The logged expert never comes close — `behind_peak.max_m = 0.053 m` over 217,189
steps, committed at `docs/audits/reward_calibration_2026-09-07/`, a factor 6.4
— and no policy can *choose* to lose that station, because two actions available
in one state differ by 0.025–0.045 of `Δq` in a 0.1 s step. **But the predicate
claims far more than that**, and the counterexample needs no knife-edge: `K3`
violated at **full** severity with `Δq = +1` scores −1.2500 against −2.0000 for a
clean trajectory with `Δq = −1`.

**The halving was deliberate and is recorded in code, though never justified.**
`scripts/measure_expert_rulebook_transition.py` carries both forms ninety lines
apart: `is_rank_preserving` (`:816`) is two-sided with the symmetric derivation
and both anchors in its docstring, and `v51_is_rank_preserving` (`:903`) is
one-sided with a docstring saying "v5.0's progress channel swung over `[-1, 1]`
and contributed `2 * lambda`, while here the tail is … roughly twenty times
smaller, which is exactly why eta comes out unconstrained". That records a
consequence, not a justification.

**Recommendation, priced in `RULEBOOK-V5.2` §14.3**: generalise the swing to
`λ₄·(ΔQ_MAX − ΔQ_MIN)`, declare `ΔQ_MIN = −1` because one clip expression
produces both bounds, and restate §5.4 as a **conditional guarantee plus a
runtime falsifier** rather than as an unconditional theorem — the condition being
`Δq ≥ −0.1525` on the compliant trajectory, with a per-episode counter on the
diagnostic path that already exists. No weight moves, no measured figure decays,
no run is needed, and the reading of that counter is pre-registered in §14.3 with
its fallback (`λ₄ ≤ 1.1525`, measured at 6.45 % below-standstill and a mean of
41.12 at the nearest measured point).

**Two corrections to this plan's own record of the finding**, both verified: the
one-sided form's anchor is `a > 1.8393` (binding at `k = 1`; `a > 1` is the
`k = 3` slice alone), and under the two-sided form at `σ = φ = 0` all three
levels bind at exactly 2.0. **Three further consequences for other documents**,
recorded rather than acted on: ADR-081's five-row decision table (`:245-251`) is
computed against the one-sided predicate and no row survives the generalised
form; `docs/open_items.md` `C50`'s alternative remedy (removing the negative
clip) would make the predicate unsatisfiable rather than merely unchanged; and
`tests/test_scal_v14.py:179-201`, the test that should have caught this, pins
`Δq = 0` on the compliant side — it assumes exactly what is in question.

**Next step:** the user's decision on question 1 of `RULEBOOK-V5.2` §14.1 — the
recommendation is §14.3's first row — then approval of `RULEBOOK-V5.2` and
`ADR-083`. `M3` is blocked on that approval, and so is every milestone after it.

---

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-A7-001` | `RULEBOOK-V5.1` §3: six ordered levels with relaxable lane compliance and progress rate below progress | Five channels with negotiable lane compliance **above** progress and no progress-rate level | The architecture the user approved 2026-09-09; grounds per requirement in §3 | Architecture approved 2026-09-09; the specification text is `DEC-A7-001` | `AC-RB5.1-01`, `-02`, `-10`, `-13`, `-15`, `-17`; `tests/test_rulebook_v51_*`, `tests/test_scal_v14.py` |
| `DEV-A7-002` | `RULEBOOK-V5.1` §5.1 / `SCAL-V1.4`: six free weights `(a, σ, φ, λ₄, η, λ₆)` | Four free weights `(a, σ, λ₄, w₅)` | §6.1–§6.3 | `DEC-A7-002`, `-003`, `-004` | `conf/scalarization/default.yaml`, `reward/scalarization.py`, `tests/test_scal_v14.py` |
| `DEV-A7-003` | ADR-081 / `AC-RB5.1-16`: `γ = 0.996` with the criterion evaluated at `L = 199` | `γ = 0.9982` with the criterion evaluated at the measured `L = 500` | `C49`; §5.2 | Approved 2026-09-09 | the six algorithm configs, `tests/test_hydra_agent_presets.py` |
| `DEV-A7-004` | ADR-072: the relaxable lane rules sit below progress | Reverted for those three sub-rules | v5.0's pathology was the **price**, not the placement: v5.0 charged them at `a = 2.2` per violated step, so standing still won past **37** relaxed steps against a mean Waymo mission. At `w₅ = 0.15`, 14.7× less, standing still wins only past **416** relaxed steps at full severity — twice the Waymo episode, and 848 against a mean PG mission over a 500-step episode. Re-derived today | `DEC-A7-002` | ADR-083 records the reversal |
| `DEV-A7-005` | `AB-LEARN` `REQ-AB-009`: the reward under test is v5.1 + `SCAL-V1.4` | The reward under test becomes A7 + `SCAL-V1.5` | §6.6 | `DEC-A7-008` | `AB-LEARN` §3, §5, §7.1, §14 |
| `DEV-A7-007` | This plan's §10/`M2` and §13 name the specification file `docs/specifications/rulebook_v5.2_specification.md` | It is written as `docs/specifications/rulebook_v5.2_UNDER_REVIEW_specification.md`, and the suffix is dropped on approval | `AGENTS.md` requires the canonical filename to carry `_UNDER_REVIEW` until approval, and `rulebook_v5.0_UNDER_REVIEW_specification.md` is the precedent for keeping such a file *in* `docs/specifications/` meanwhile. This plan's own `M2` text agrees in substance — it says to "move it into `docs/specifications/` at that point" — so the two readings differ only in the filename | Decided in `M2`, 2026-09-11; reported for approval | `docs/project_index.md`; every citation of the specification path |
| `DEV-A7-008` | §13 lists `ADR-083` as the record of "the architecture and the discount", both approved 2026-09-09/-10, which would make it `Approved` on arrival | `ADR-083` is written with status **`Proposed`** | The decisions it records are approved; **its text is not** — the user has not seen it, and marking a new document `Approved` would attribute an approval that was never given to this wording. The status line states exactly which parts carry a signature and which do not, so no evidence is lost | Decided in `M2`, 2026-09-11; reported for approval | `docs/project_index.md` Decisions row |
| `DEV-A7-009` | §9.1 defines `AC-A7-01`…`-15` and §9.2 defines `TEST-A7-01`…`-22`, with `TEST-A7-22` mapped to `REQ-A7-11` | The specification adds one criterion, `AC-A7-16`, and decomposes the ordering fixtures as `TEST-A7-15a`…`-15g` | `REQ-A7-11` is the below-standstill ceiling and does not cover the pooling guard, so `TEST-A7-22` had no criterion to be reconciled against — a test that no acceptance criterion claims cannot close a requirement. The fixture decomposition follows the sub-case convention `RULEBOOK-V5.1` §10 already uses (`TEST-RB5.1-15b`, `-16b`, `-16c`) | Decided in `M2`, 2026-09-11; reported for approval | `RULEBOOK-V5.2` §9, §10; this plan's §8 traceability on the next pass |
| `DEV-A7-006` | `ACL-SN-EMA-001` v2.0 §3.2: six level margins are a property that is fatal if absent, and `DEC-206`'s ordering | The property becomes false after `M5`, so the ACL cannot run on an A7 rulebook until it is re-based; `DEC-206` is reopened without prejudice, its stated reason having been a citation of ADR-072 | §6.10 | `DEC-A7-014` | `ACL-PROG-004`; `curriculum/scenario_acl/usefulness.py`; the ACL plan's `M0`/`M2` boundary |

---

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `docs/implementation/rulebook_a7_five_channel_hierarchy_exec_plan.md` | Added | This plan |
| `docs/specifications/rulebook_v5.2_UNDER_REVIEW_specification.md` | **Done, `M2`, 2026-09-11** | `DEC-A7-001`; the contract `M4`–`M7` implement against, `UNDER_REVIEW` and not authoritative until approved (`DEV-A7-007`) |
| `docs/decisions/ADR-083-progress-last-and-the-discount-at-the-measured-horizon.md` | **Done, `M2`, 2026-09-11** | The architecture and the discount; status `Proposed` (`DEV-A7-008`) |
| `scripts/measure_expert_rulebook_transition.py` | **Done, `M1`, 2026-09-10** | Four episode exposure accumulators, `a7_reward`/`a7_is_rank_preserving`, the A7 grid, the K2/K3 argmax and co-occurrence counters, and `production_scalarization_config()` (`C54` fix) |
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
| `src/thesis_rl/analysis/aggregate/aggregate_runs.py` | Planned modification | `DEC-A7-011`'s pooling guard: `vector_schema_id` joins `_build_condition_id` |
| `docs/implementation/automatic_curriculum_learning_v2.0_exec_plan.md` | Planned modification | `DEC-A7-014`: the re-base recorded, `DEC-206` reopened |
| `conf/scalarization/default.yaml` | Planned modification | The A7 mode and weight set |
| `conf/agent/planner/algorithm/{ppo,ppo_sb3,sac,sac_sb3,td3,td3_sb3}.yaml` | Planned modification | `γ = 0.9982` and its shaping twin |
| `tests/test_scal_v14.py` | Planned modification | `DEC-A7-006` |
| `tests/test_rulebook_v51_orderings.py` | Planned modification | `DEC-A7-007`, and `V4` |
| `tests/test_hydra_agent_presets.py` | Planned modification | The verdict assertion |
| `tests/test_a7_channels.py`, `tests/test_scal_v15.py`, `tests/test_a7_orderings.py` | Planned addition | §9.2's matrix |
| `tests/test_rulebook_v51_levels.py`, `test_rulebook_v2_{monitor,wrapper,transition}.py`, `test_rulebook_v51_diagnostics.py`, `test_audit_block_b_reward_identity.py`, `test_analysis_optional_ci_and_r4_split.py`, `test_hydra_preset_run_configs.py`, `test_rulebook_provenance_identity.py` | Planned modification | The measured ripple (§4.4) |
| `docs/implementation/reward_learnability_ab_screening_exec_plan.md` | Planned modification | `DEC-A7-008` |
| `docs/implementation/rulebook_v5.1_six_level_hierarchy_exec_plan.md` | Planned modification | `DEC-A7-009` |
| `docs/project_index.md` | **Partly done, `M2`, 2026-09-11** | `M2` added the `RULEBOOK-V5.2` row (`CANDIDATE`) and the `ADR-083` row (`PROPOSED`). Still `M8`'s: moving the v5.1 row to `SUPERSEDED` for §3/§4/§5/§9/§10 **on approval**, and rewriting A7's own ExecPlan Registry row, which still reads `AWAITING_DECISIONS` from 2026-09-09 |
| `docs/open_items.md` | Partly done | `C54` added **`M1`, 2026-09-10** (the `production_scalarization` defect found while launching `M1`); `C49`, `V4`, `D15`, `C50`, `D14`, `V1` remain `M8`'s |
| `docs/audits/a7_m1_measurement_2026-09-10/` | **Done, `M1`, 2026-09-10** | `M1`'s committed evidence: summaries and top-1 % tail rows |
| `tests/test_measure_expert_rulebook_transition.py` | **Done, `M1`, 2026-09-10** | 10 new tests: `C54`'s regression, `a7_reward`/`a7_is_rank_preserving` term by term and against the transcription trap, the grid, the standstill baseline, the standstill baseline's report-level integration, exposure, merge order-independence, argmax and co-occurrence |

---

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| Independent re-derivation of A7's §5.4 margins, the `w₅` cap and crossovers, both grids' admissible member counts, and the discount table (standalone transcription of the predicate, no repository import) | `PASS` | 2026-09-09 | A0 1.1165 / 1.1121 / 1.1792; A7 with `φ=0.25` 1.1105 / **1.0975** / 1.1390; A7 with `φ=0` 1.1514 / 1.1478 / **1.1390**. Cap 0.384615; crossovers 0.313983 and 1.538462; ratio exactly 4.0; `a/2 = 1.25`. `v51_weight_grid` **77 of 100** admissible, `v51_calibration_grid` **16 of 36**, and **exactly one** member at `a = 2.5, σ = 0.30`. `γ = 0.9982`: break-even **508.6** against `L = 500`, `γ^L = 0.4062` against `1/a = 0.4`, effective horizon **556**; `γ = 0.996`: break-even **228.6**, `γ^L = 0.1348`, horizon 250; required `γ = 0.998169`. `λ₄ ∈ {1.25, 1.9}` admissible under the A7 tail at the production pair. One fully-violated interaction step costs **8.125** at `φ = 0` and **8.375** at `φ = 0.25`. Standing still wins past **416** relaxed steps at `w₅ = 0.15` against a mean Waymo mission and **848** against a mean PG one, versus **37** at v5.0's `a = 2.2`. Every figure agrees with the audit except the two recorded in §11 |
| `git fetch origin` and branch check | `PASS` | 2026-09-09 | `main` at `3e58ce0`, level with `origin/main`, tree clean |
| `make gate` | `PASS` | 2026-09-09 | `gate: PASS (2 not applicable: whitespace/pending, whitespace/untracked) \| FULL \| 450492d (worktree-a7-execplan, tree clean) \| 20260909T210257Z` — **1927 passed, 5 skipped** in 4m05s, the same count `main` reports at `3e58ce0`, so this change moves nothing in the suite. Run on `450492d`, which is this plan and the register row; the `§14` row itself was added afterwards. **The first run of the same gate on the same tree failed 16 tests** — `test_pg_validation`, `test_scenario_waymo`, three in `test_scenario_catalog_build`, two in `test_scenarionet_smoke`, `test_forced_rule_scenarios`, `test_rulebook_v2_live_integration` and seven in `test_rulebook_synthetic_scenarios` — every one of them reading the bundled Waymo assets or building a catalog from them. That is the first-run race between the sixteen `pytest-xdist` workers over state the assets generate on first read, documented at `docs/workflows/agent_operations.md` §"Running checks from a git worktree" item 4 after `git submodule update` in a fresh worktree. Classified per `AGENTS.md` Completion as **missing on the project machine**, not a repository defect: the relaunch was unchanged and green. The log lives inside the worktree and dies with it, so `M9` relaunches the gate from `main` after the merge |
| `make gate` | `PASS` | 2026-09-10 | The ratification pass. `gate: PASS (2 not applicable: whitespace/pending, whitespace/untracked) \| FULL \| 9374b24 (worktree-a7-ratified, tree clean) \| 20260909T220910Z` — **1927 passed, 5 skipped** in 4m14s, the same count as the previous run and as `main`, so ratifying the gates and adding `DEC-A7-013`/`-014` moves nothing in the suite. Green on the first attempt this time: the worktree's submodules were already initialised, which is the condition whose absence produced the 16 first-run failures recorded in the row above |
| `TEST-A7-01` … `TEST-A7-21` | `NOT_RUN` | — | Defined in §9.2 before any production change, as `AGENTS.md` requires. Each is scheduled on the milestone that implements its requirement |
| `make smoke` | `NOT_RUN` | — | `M7`. It has not been run since the discount changed (`V1`), and this plan changes the discount again |
| Focused instrument tests, `tests/test_measure_expert_rulebook_transition.py` | `PASS` | 2026-09-10 | **33 passed** (10 new for `M1`): the four exposure accumulators against a hand-built episode on all three objects separately; `a7_reward`/`a7_is_rank_preserving` against §5.1 term by term and against this plan's own worked figures, including a reconstruction of the `φ`-reaches-K4 transcription trap that reproduces the plan's own wrong numbers (1.0911/1.0513/1.0225) so the right ones are checked against a known-wrong alternative, not only against themselves; the grid's six admissible members; the standstill baseline explicitly at 0 for every `a7_*` name and its report-level integration; the argmax and co-occurrence counters against fixtures with a known answer; merge order-independence; and `C54`'s regression (`production_scalarization_config()` against `conf/scalarization/default.yaml`, read directly) |
| `production_scalarization_config()` pre-fix reproduction | `FAIL` (expected) | 2026-09-10 | The unmodified tree, run end to end against the full panel: `"scenarios_measured": 0`, `"scenarios_skipped": {"error:ScalarizationConfigurationError": 1100}`, every sampled message `"Mode 'six_level_priority_weighted_rank' requires priority_base=2.5, got 2.2."`. `C54`'s pre-fix evidence; log at `outputs/a7_m1/a7-m1-run.log` (not committed — regenerable, and the shared `/scratch` output path is per-session) |
| `make check` | `PASS` | 2026-09-10 | `gate: PASS (2 not applicable: whitespace/untracked, whitespace/range) \| PARTIAL (pytest args: -m not integration) \| 48c25ca (worktree-a7-m1-measurement, tree dirty) \| 20260909T224528Z` — **1925 passed, 6 skipped** in 2m14s. First attempt after `git submodule update` in the fresh worktree failed the same two bundled-Waymo-fixture tests `docs/workflows/agent_operations.md` documents (`test_pg_validation.py::test_bundled_export_validation_can_be_read`, `test_scenario_features.py::test_bundled_waymo_feature_extraction_is_route_aware`); unchanged relaunch was green, so not chased, per this plan's own constraint |
| Focused `ruff check`/`ruff format --check`, `scripts/measure_expert_rulebook_transition.py tests/test_measure_expert_rulebook_transition.py` | `PASS` | 2026-09-10 | Both clean after one `ruff format` pass on the two files (whitespace only; no logic changed, confirmed by rerunning the focused test suite unchanged after formatting) |
| The `M1` panel run: `python scripts/measure_expert_rulebook_transition.py --data-root /workspace/data/scenarionet --frozen-index /workspace/thesis-metadrive/data/scenarionet/frozen/scenario_selection_index.json --split train --source waymo --workers 24 --output /workspace/outputs/a7_m1_measurement_waymo_train_full.json` | `PASS` | 2026-09-10 | **1100/1100 records measured, 0 skipped, 217,189 transitions** — the identical scope `RULEBOOK-V5.1` §5.5 and the 2026-09-01 comfort run were measured on. Run in a named tmux session (`a7-m1-run-v2`) teeing to `outputs/a7_m1/a7-m1-run-v2.log`. The six outputs are reconciled into §6.2, §6.4, §6.8, §6.9 and §11 above; full evidence at `docs/audits/a7_m1_measurement_2026-09-10/` |
| `make gate` | `PASS` | 2026-09-10 | Before merging this milestone's branch. `gate: PASS (1 not applicable: whitespace/range) \| FULL \| 48c25ca (worktree-a7-m1-measurement, tree dirty (5 files)) \| 20260909T234058Z` — **1937 passed, 5 skipped** in 4m04s, exactly the 2026-09-10 ratification gate's 1927 plus this milestone's 10 new instrument tests. `whitespace/range` is `NOT APPLICABLE` here because `HEAD` is not ahead of `origin/main` on this uncommitted run, unrelated to the pytest count. The log dies with this worktree; `M9` relaunches it from `main` after the merge |

| Independent re-derivation for `M2`: the §5.4 predicate from its own derivation, the margins in four configurations including the transcription trap, the `w₅` cap and crossovers, the factor 4, `λ₄ ≤ a/2`, the per-step costs, the discount table, the required `γ`, `λ₄ ∈ {1.9, 1.25}`'s admissibility and `DEV-A7-004`'s standing-still crossover (standalone script, no repository import) | `PASS` with **one disagreement** | 2026-09-11 | Every figure reproduces: A7 `φ=0` **1.1514 / 1.1478 / 1.1390**; `φ=0.25` **1.1105 / 1.0975 / 1.1390**; A0 **1.1165 / 1.1121 / 1.1792**; the trap **1.0911 / 1.0513 / 1.0225**; cap **0.384615**; crossovers **0.313983** and **1.538462**; ratio **exactly 4.0**; `a/2 = 1.25`; `K4` full **0.195**, `K2` full **8.125**/**8.375**; break-even **228.6** and **508.6**, `γ^500` **0.1348** and **0.4062**, horizons **250** and **555.6**, required `γ` **0.998169**; `λ₄=1.9` **1.1600 / 1.1693 / 1.1933** and `λ₄=1.25` **1.2188 / 1.3312 / 1.7301**; `N*` **416** at `Q ≈ 90 m`, **848** at `Q ≈ 184 m`, **37** at v5.0's price. **The disagreement**: §6.1's "1.83 m/s" does not reproduce — the same criterion gives **2.17 m/s** (§11, 2026-09-11). Term placement confirmed against `reward/scalarization.py:147-196, 393-425` |
| `M1`'s published values read back from the committed evidence rather than from §6 (`docs/audits/a7_m1_measurement_2026-09-10/a7_m1_summary.json`) | `PASS` | 2026-09-11 | `τ₁`–`τ₄` on all three objects, `p95`/`p99`, the max-setting record uids, the six-member grid's `fraction_below_standstill` and mean returns, and the `K2`/`K3` co-occurrence all agree with §6.4/§6.8/§6.9 to the published precision. `scenarios_measured: 1100`, `measured_steps: 217189`, `scenarios_skipped: {}`, `declared_panel_defect_exclusions: []` on every channel — so the "0 exclusions declared" claim is read off the artifact, not restated |
| `git diff --check` | `PASS` | 2026-09-11 | `M2`'s three documents plus this plan; no whitespace error |
| Independent adversarial audit of `M2` — six zones, thirteen sessions, each zone refuted by a session that did not find it; every reported claim re-verified by hand afterwards | `PASS with one blocking finding` | 2026-09-11 | **Blocking**: the §5.4 predicate counts the progress swing once where `rulebook_v5.0` §6.3 counts it twice; the two-sided form reproduces both of v5.0's published anchors (`a > 2`; `a ≥ 2.92`, reproduced 2.9196) and the one-sided form reproduces neither (`a > 1`; 2.8312), and a two-state counterexample at the selected weights scores −0.5000 against −2.0000. Under the two-sided form `λ₄ = 2.0` fails at `k = 2, 3` and today's six-level weights fail at `k = 1` too — a pre-existing v5.1/ADR-081 defect, now question 1 of `RULEBOOK-V5.2` §14.1. **Material, not an error**: at `τ₄ = 23.386514` the thresholded comparison ties the O3 pair on every cost channel (exposure 10.0 against the budget) and falls through to progress — declared as §4.5 and §11.15. **Reproduced against a figure they were not built from**: the reference-pair reconstruction returns all six entries of v5.1 §4.4's published table, hence +0.5813 / +2.4579 and 0.131342 / 0.073555. **Unreproducible from any live source**: `w₅`'s upper bounds 0.4477 / 0.2641 and §6.3's 90.5 % / 79.7 %. Five further questions and ten documentation corrections in §11 above |
| `make gate` / `make check` for `M2` | `NOT APPLICABLE` | 2026-09-11 | `M2` touches no Python, no configuration and no test: three Markdown documents and this plan. There is nothing in the diff for Ruff or pytest to inspect, and recording a green suite here would be recording that `M1`'s tree is still green, which `M1` already established. `M3` is the next milestone with a test surface, and `M9` re-runs the gate from `main` after the merge |

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
   `λ₆`, and 0.265 reward units of `w₅` indicator cost per episode. `M1`'s grid
   member is what
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
