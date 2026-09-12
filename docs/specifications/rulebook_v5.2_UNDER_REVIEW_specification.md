# Specification: Five-channel rulebook with progress last, and the discount that reaches the measured horizon

## Metadata

- Feature: `rulebook_a7_five_channel_hierarchy`
- Specification ID: `RULEBOOK-V5.2`
- Version: `5.2`
- Status: **`UNDER_REVIEW`**
- Authoritative: **`NO`**. This document is not an implementation contract and
  no production code may be written against it while this line reads `NO`
  (`AGENTS.md`, Sources Of Truth). `RULEBOOK-V5.1` §3, §4, §5, §9 and §10 remain
  authoritative until the day this document is approved.
- Date: `2026-09-11`
- Approved: *pending*
- Approval evidence: *pending*. What **is** already approved, and is recorded
  here rather than re-decided: the **five-channel architecture** and
  **`γ = 0.9982`**, explicitly approved by the user on **2026-09-09**; and every
  remaining gate of the `A7` ExecPlan §6 — `w₅ = 0.15`, `φ = 0`, the budget
  *rule*, the two mandatory-test changes, the form of this document, the
  `AB-LEARN` amendment, the `RB51` disposition and the vocabulary break — on
  **2026-09-10**. Approving *this text* is a separate act, and it is what
  changes the two lines above.
- Supersedes, **effective on approval only**:
  - `docs/specifications/rulebook_v5.1_specification.md` §3 (the hierarchy and
    its aggregation), §4 (`L4`, `L6` and the discount), §5 (`SCAL-V1.4` and its
    weights), §9 (acceptance criteria) and §10 (test matrix). A new version
    rather than an in-place amendment, because `RULEBOOK-V5.1` is cited by 85
    files with 112 section citations across 52 of them and **63 of those 112
    (56 %) point at sections this document rewrites**; an in-place amendment
    leaves all 63 silently meaning something else, while a new version
    invalidates none of them (`A7` `DEC-A7-001`, measured).
  - `SCAL-V1.4` in full, replaced by `SCAL-V1.5` (§5).
- Retains from `RULEBOOK-V5.1`, unchanged and not restated: §1.1's *method* of
  stating what the hierarchy buys as falsifiable orderings (the orderings
  themselves are restated in §1.1 below, because two of them change), §2 (Test A,
  Test B, controlled invariance), §3.2's sub-rule definitions, §6 (observation
  consequences), §7 (diagnostics), §11 (known limitations, extended in §11
  below), §12 (references, extended in §12 below) and §13 (out of scope).
  `rulebook_v5.0_UNDER_REVIEW_specification.md` remains the evidence of record
  for every per-sub-rule Test A / Test B table.
- Related ADRs: **`ADR-083`** (this architecture and this discount) is the
  decision record for this document. `ADR-063`…`ADR-071` carry over **unchanged**
  — A7 changes no sub-rule definition, no geometry and no at-fault gate.
  `ADR-072` is **partly reverted** (the negotiable lane rules return above
  progress; ADR-083 records why the reversal is not a return to v5.0).
  `ADR-076` (`L6 progress_rate`) is **superseded** — the level is deleted.
  `ADR-075` and `ADR-081` (the discount) are amended by `ADR-083`. `ADR-069` and
  `ADR-074` are superseded for the scalar form only. `ADR-073` (signed `L4`)
  carries over as `K5`.
- Related specifications: `evaluation_protocol_v1.3_specification.md`,
  `observation_v1.3_specification.md`,
  `observation_lidar_v2.0.2_amendment.md`,
  `driving_mission_v1.1_specification.md`.
- Implementation plan: `docs/implementation/rulebook_a7_five_channel_hierarchy_exec_plan.md`
  (`A7`), milestone `M2`. Milestones `M3`–`M9` are blocked on this document being
  approved.
- Measurement evidence of record: `docs/audits/a7_m1_measurement_2026-09-10/`
  — the full frozen Waymo `train` panel, **1100 records, 217,189 transitions, 0
  skipped**, replayed through production's own `evaluate_transition`. Every
  budget value in §4.5, every `fraction_below_standstill` figure in §5.5 and the
  co-occurrence figure in §3.6 come from that run and from nowhere else. The
  O1–O6 orderings of §1.1 are established by the **constructed** fixtures of
  §10, which replay cannot supply because it holds no counterfactual.
- Open decision carried, not resolved: **`D1`** — which of three candidate
  *objects* a threshold is enforced on (§4.5). This document declares the
  requirement on one of them and deliberately fixes no mechanism.

---

## 1. Purpose

`RULEBOOK-V5.1` placed the lane rules a competent driver may relax **below**
mission progress, and that placement bought the one ordering the redesign existed
for: completing a mission with a brief lane violation beats standing still for
ever. It also created three problems that are measured rather than argued.

1. **The shipped discount does not satisfy its own criterion.** The criterion
   that keeps a future higher-priority violation from being damped below a
   present lower-priority one is `ln(a)/−ln(γ) > L`. At `a = 2.5, γ = 0.996` the
   break-even is **228.6** control steps while **591 of 2200 training records
   (26.86 %)** run longer, so inside those episodes the hierarchy inverts. The
   criterion had been evaluated against a Waymo-only horizon of 199 steps; the
   measured training horizon is **500** (`docs/open_items.md` `C49`).
2. **Ordering O3 is lost at the shipped discount.** A trajectory taking the legal
   route loses to an illegal shortcut on the scalar arm, margin **−3.5789** where
   the undiscounted value is **+0.2000** (`D15`): the shortcut arrives sooner, so
   its discounted progress total is strictly larger and the comparison is decided
   at progress before the lane channel is ever consulted.
3. **Two of the six weights cannot be pinned.** Across `η ∈ [0, 5]` the logged
   expert's mean episode return moves by **less than 0.1 on ≈70**, because the
   logged human almost never relaxes a lane rule — so no measurement discriminates
   `η` — and `λ₆`'s value is an open decision. A parameter that neither a
   measurement nor an ordering determines is a parameter that should not exist.

This document changes the **order** and the **adapter**, and nothing else.

> Progress is the **last** channel and it is **unthresholded**. Above it sit the
> four channels that carry costs, in the order collision safety, interaction
> risk, non-negotiable compliance, negotiable lane compliance. The negotiable
> lane channel is charged by a bounded satisfaction indicator with a severity
> slope, which is what lets it decide *before* progress in the ordered arms and
> still carry a finite exchange rate in the scalar arm.

Two literatures agree on the position, and neither is this repository. Vamplew,
Dazeley, Berry, Issabekov and Dekker state the requirement of the absolute
thresholding family directly — "objective n will be unconstrained, hence
`C_n = +∞`" (§3.2.3, p. 58) — and Censi et al. place own progress at the bottom
of their rulebook (Fig. 10) and add new rules there (Definition 17). The
negotiable/non-negotiable split above it is the minimum-violation semantics of
Castro, Tumova, Karaman, Frazzoli and Rus, expressed as a rule hierarchy rather
than as a planner.

**Six free weights become four.** `(a, σ, φ, λ₄, η, λ₆)` becomes
`(a, σ, λ₄, w₅)`, one per-step normalization (`Δt/T_REF`) disappears, and the
number of channels a thresholded arm would have to constrain jointly falls from
five to four — including the one channel a threshold provably cannot be placed on
(§4.2).

### 1.1 What this buys, stated as falsifiable orderings

The design target is not "positive expert return". It is that the reward ranks
controlled trajectory pairs the way a competent driver would. These orderings are
the acceptance criteria of §9 and the fixtures of §10.

| # | ordering that must hold | which channel decides it under A7 |
|---|---|---|
| O1 | legal completion ≻ standing still | scalar arm: `K5`. **Strict lexicographic: fails, and the failure is reported** (§11.1) |
| O2 | completion needing a brief lane relaxation ≻ standing still | scalar arm: the finite `K4`/`K5` exchange. **Strict lexicographic: fails at `K4`** (§11.2) |
| O3 | legal route ≻ illegal shortcut, both completing | `K4`, **above** progress, so the shortcut loses before duration is compared |
| O4 | lane relaxation ≻ collision | `K1` |
| O5 | waiting at a red ≻ running it to finish | `K3` |
| O6 | necessary relaxation ≻ gratuitous relaxation | `K4`; the gratuitous violation buys no progress, so it loses at that channel rather than one channel lower |

**What changes relative to `RULEBOOK-V5.1`, and it is three rows.**

| | under v5.1 | under A7 |
|---|---|---|
| O3, scalar arm | **fails** at the shipped `γ = 0.996` (−3.5789), passes undiscounted (+0.2000) | **passes at both discounts** — +0.5813 at `γ = 0.996`, +2.4579 at `γ = 0.9982` (§14.2) |
| O3, strict lexicographic | **fails** at the shipped `γ`: decided at `L4`, where the shortcut arrives sooner | **passes**: decided at `K4`, above progress |
| O2, strict lexicographic | passes | **fails** at `K4` (§11.2) |

**O3 under the thresholded comparison is a fourth case and it does *not* change
in A7's favour at the measured budgets** — the pair ties on every cost channel
and falls through to progress (§4.5). The gain is on the scalar and strict
lexicographic arms.

The O3 row is what this restructure is for: it stops being a property of an exact
undiscounted tie — which §11.9 records as unobtainable for *any* bounded per-step
progress channel — and becomes a property of the **order**, which discounting
cannot move.

The O2 row is a real loss and is not presented as anything else. It is stated
with its measured qualification: v5.1's strict-lex pass on O2 is a **zero-traffic
artefact**. Add one interaction step in forty at the 0.05 residual this
repository's own ordering test already uses and v5.1 fails O2 as well; once that
residual is present A7 is no worse on any ordering and strictly better on O3.

---

## 2. Method

Unchanged from `RULEBOOK-V5.1` §2, which is unchanged from v5.0 §2, and not
restated: **Test A** (admissibility — a rule bearing a satisfaction indicator
must be satisfiable by a competent driver, verified by expert replay, and can
only reject), **Test B** (observability — memory is admissible only when the
observation carries the same state and that state is a plausible perception
output), and **controlled invariance**.

Those three tests constrain the sub-rules and say nothing about their
**placement**; the orderings O1–O6 are the placement test, checked on constructed
fixtures. A7 changes only placement and the adapter, so **every Test A and
Test B result in v5.0 and v5.1 survives it unexamined**, and that is the reason
the surface is kept this small: the smaller the change, the more of the existing
falsification evidence remains valid.

One addition, and it is a standard this document is held to rather than a test of
the rulebook. By `AGENTS.md`'s Scientific Argument Standards a choice may not be
justified by citing a specification of this repository. Two places in this
document have no other argument available, and both are declared as findings
rather than dressed as derivations: `K2 ≻ K3` (§3.6) and `φ`'s original value of
0.25, which is why `φ` is removed (§5.5).

---

## 3. The rulebook

Five channels, strictly ordered:

```
K1  collision safety            ≻
K2  interaction risk            ≻
K3  non-negotiable compliance   ≻
K4  negotiable lane compliance  ≻
K5  mission progress               — last, and unthresholded
```

| channel | recorded value | sub-rules |
|---|---|---|
| `K1` | `collision_safety` | at-fault collision impact |
| `K2` | `interaction_risk` | `ttc`, `clearance`, `rss_lateral` |
| `K3` | `non_relaxable_compliance` | `offroad`, `signal`, `stop`, `crosswalk`, `vehicle_yield`, `speed_limit` |
| `K4` | `negotiable_lane_compliance` | `solid_line`, `wrong_carriageway`, `dashed_line` |
| `K5` | `mission_progress` | signed route advance `Δq` (§4) |
| — | diagnostic, non-normative | `rss` longitudinal, not-at-fault collisions, every atomic sub-rule cost |

`MACRO_RULE_ORDER` is the single source of the ordering and nothing else may
restate it. The cost channels are derived from it by excluding the one utility
channel, so the invariant that stops a sign error from turning progress into a
penalty holds by construction rather than by a second list.

**`K3`'s recorded value stays `non_relaxable_compliance` while its prose label
becomes "non-negotiable compliance".** This is deliberate and it is the one place
in this document where the two disagree. The recorded values are an output
contract — they appear in CSVs, evaluation artifacts and analysis tables — so a
rename costs every downstream consumer, and `K3`'s membership and position are
unchanged by A7. `K4`'s value *does* change, from `relaxable_lane_compliance` to
`negotiable_lane_compliance`, because that channel's charging rule and its
position both change and a level named for what it no longer is is how a later
reading goes wrong. §3.7 records this and the three prior conventions in one
place.

`wrongway` remains deleted (v5.0 §5.6, ADR-066). The `L6 progress_rate` level and
its `advance_shortfall` sub-rule are **deleted** (§4.4), and the `Δt/T_REF`
per-step scaling is deleted with them, so the reward no longer carries two
different per-step normalizations.

### 3.1 Why `K1` and `K2` are separate

Unchanged from v5.1 §3.1: a collision is an **outcome** and TTC/clearance/
RSS-lateral are **anticipatory indicators**. An indicator that fires is not a
failure; a collision is. Merging them is not merely inelegant — it is measured to
halve the mitigation gradient, a margin of 2.82 falling to 1.22 on the audit's
impact probe — so the merge that would have produced a four-channel hierarchy
pays for the fourth channel with the distinction that makes the first one mean
something.

### 3.2 Sub-rule definitions

Unchanged from v5.0 §5, with the two at-fault decisions incorporated exactly as
v5.1 §3.2 incorporates them:

- **`K1`** applies ADR-071: contacts are classified with nuPlan's taxonomy;
  at-fault contacts are charged and **terminate** the episode, not-at-fault
  contacts are charged **nothing** and **truncate** it with a `V(s)` bootstrap.
  The not-at-fault rate is reported as a diagnostic.
- **`K2`** applies ADR-070: `ttc`, `clearance` and `rss_lateral` are
  **inapplicable when `v_ego ≤ 5e-02 m/s`**.

No sub-rule definition, geometry, tolerance or applicability gate is changed by
this document.

### 3.3 Intra-channel aggregation

Unchanged in every respect from v5.1 §3.3 except that one level ceases to exist.
Three different aggregations, because three different questions are being asked.

**Across objects within one sub-rule — `max`.** `c_ttc = max over actors`: the
most critical actor.

**Within `K2` — `max`.** `c_K2 = max(c_ttc, c_clearance, c_rss_lateral)`.
Defensible here specifically because the three are complementary **detectors of
one property** — being outside the safe interaction envelope — so that
`c_K2 = 0 ⟺ all three are 0`. It is not a claim that they are cardinally
commensurable.

**Within `K3` — `max`.** `c_K3 = max(c_offroad, c_signal, c_stop, c_crosswalk,
c_vehicle_yield, c_speed_limit)`. Minimax semantics: the channel reports the
worst non-negotiable violation present.

**Within `K4` — normalized sum with a fixed denominator of 3.**

```
c_K4 = ( c_solid_line + c_wrong_carriageway + c_dashed_line ) / 3
```

The quantity of interest is the *total amount of relaxation*, so a detour that
crosses a solid line **and** enters the opposing carriageway must cost more than
one that only straddles; `max` would make the second violation free, which would
break O6. **The denominator is fixed at 3 and is never recomputed over the
applicable sub-rules only** — recomputing it would change the reward scale
mid-episode as sub-rules become applicable. An inapplicable sub-rule contributes
0 and is reported through its own applicability mask.

A change of position does not change an intra-level aggregation, and the panel
says the sum is not decorative: `dashed_line` is violated on 1145 steps and
`solid_line` on 758, totalling 1903 against the channel's own 1897 violated
steps, so **six steps carry two sub-rules at once**. The sum is almost
unexercised on the logged expert, but not never.

**`K5` has no aggregation.** It is one signed quantity (§4.1).

### 3.4 The atomic vector is the contract, not the aggregation

Unchanged. The rulebook exposes to every algorithm the same per-step vector of
**atomic** costs, before any aggregation — fourteen entries, thirteen sub-rule
costs plus `Δq`:

```
( c_collision_at_fault, c_ttc, c_clearance, c_rss_lateral,
  c_offroad, c_signal, c_stop, c_crosswalk, c_vehicle_yield, c_speed_limit,
  Δq,
  c_solid_line, c_wrong_carriageway, c_dashed_line )
```

The five-channel aggregation of §3.3 and the scalarization of §5 are **adapters**
over that vector, not part of it. This is what makes A7 a small change: the thing
that is *measured* is untouched, and only the comparison order and the scalar
adapter move. **No observation field is added and the observation dimension `D`
is unchanged.**

### 3.5 Normative placement decisions, stated rather than inherited

Three placements are choices this document makes and a reader may contest. Two
are v5.1's, restated because they still apply; the third is new.

1. **`offroad` is non-negotiable (`K3`), above progress.** Consequence: pulling
   onto a shoulder to pass an obstruction is ranked below not completing the
   mission. Defensible on precedent — nuPlan treats `drivable_area_compliance` as
   a multiplicative penalty, i.e. as hard — and the 0.3 m tolerance of v5.0 §5.4
   already absorbs bounding-box over-approximation. It is nevertheless the
   placement most likely to be wrong. **Recorded as a decision, not as a fact.**
2. **`speed_limit` is non-negotiable (`K3`).** Inapplicable throughout the PG
   panel by ADR-068's provenance gate, so on that half of the training mixture
   `K3` carries fewer sub-rules there. Measured 2026-09-01: **five of the six
   never apply on any PG record**, so on that half of the mixture `K3` is
   `offroad` alone (§8, §11.12) — the provenance gate is the reason
   `speed_limit` is one of them, not the whole story.
3. **The three negotiable lane rules sit above progress, which reverses ADR-072
   for those sub-rules — and the reversal is not a return to v5.0.** v5.0's
   pathology was the **price**, not the placement: v5.0 charged a relaxable
   violation at the priority weight `a = 2.2` per violated step, so standing
   still won past **37** relaxed steps against a mean Waymo mission. Under A7 the
   same channel is charged `w₅·(1+σ) = 0.195` per fully violated step, and
   standing still wins only past
   `N* = λ₄·Q/(D_REF·w₅·(1+σ))` relaxed steps at full severity, where `Q` is the
   mission span in metres — **416** steps at a mean Waymo span of ≈90 m and
   **848** at a mean procedural-generation span of ≈184 m, i.e. between two and
   four times a whole 500-step episode. Re-derived for this document, with the
   two spans stated because the figures are meaningless without them (§14).
   ADR-083 records the reversal.

### 3.6 `K2 ≻ K3` is the hierarchy's least-supported step, and that is a finding

Every other adjacency in this hierarchy rests on a mechanism or on a measurement.
`K1 ≻ K2` is outcome-versus-indicator, and merging the two is measured to halve
the mitigation gradient (§3.1). `K3 ≻ K4` is the negotiable/non-negotiable
distinction, the semantic core of minimum-violation planning, and merging the two
is measured to fail three orderings on the scalar arm. `K4 ≻ K5` has three
independent grounds (§5.5) plus the literature of §1.

**`K2 ≻ K3` has none of that.** The only derivation that exists anywhere is
`rulebook_v4.7`'s — that without an interaction level a near-miss legally in lane
could be preferred to a brief illegal deviation with a wide margin — v5.0
asserted the order without restating it, and v5.1 inherited it. By `AGENTS.md`'s
Scientific Argument Standards this must be said out loud: **it is the one place
in the hierarchy where the only surviving argument is this repository's own
specification.** The order is kept, because reordering on no evidence replaces
one unsupported claim with another; it is recorded here as the step to attack
first if the hierarchy is ever reopened.

**What is measured, and what it does and does not settle.** The measurement that
would price the adjacency is the co-occurrence of the two channels, and it did not
exist before 2026-09-10. On the full frozen Waymo `train` panel — 1100 records,
217,189 steps:

| quantity | value |
|---|---:|
| steps with both `K2` and `K3` non-zero | **15 of 217,189 (0.0069 %)** |
| episodes carrying at least one such step | **5 of 1100 (0.45 %)** |
| `c_K2` on those 15 steps, `p50` / `p99` | 0.2197 / 1.0 |
| `c_K3` on those 15 steps, `p50` / `p99` | 0.0253 / 1.0 |
| Pearson correlation on those 15 steps | **−0.0161** |

Source: `docs/audits/a7_m1_measurement_2026-09-10/` (`a7_measurement.k2_k3_cooccurrence`).

This bounds *how often the order could matter at all* on the logged expert — and
the answer is almost never jointly, with severities that do not move together —
which is evidence about how much is staked on the adjacency in practice. It is
**not** a derivation of the order, and this document does not present it as one.
A future session reopening `K2 ≻ K3` needs a mechanism, not a larger version of
this table.

### 3.7 Identifier conventions: three already in use, and the fourth this document adds

This is recorded in one place because a search that knows only one of these
conventions finds a subset of the truth and reports it as the whole, which has
already produced one defect in this repository. All four name the same objects.

| # | convention | where it lives now | example |
|---|---|---|---|
| 1 | **`R`-numbering** — v4.7's four macro rules `R1`–`R4`, and the rule *names* of the v1 rulebook | recorded metric names, analysis-table filenames and blame symbols; it is the oldest and the least visible | `r2_violated_fraction_of_all_steps` and `final_blame_r2`/`_r3` (`scripts/measure_expert_rulebook_transition.py`); `rulebook_r4_progress_margin.csv` and `R4_PROGRESS_MARGIN_RULE_NAME = "route_progress"` (`src/thesis_rl/analysis/tables/make_rulebook_tables.py:18`) |
| 2 | **`L`-numbering** — v5.x's six levels `L1`–`L6` | the prose and formulae of `RULEBOOK-V5.1`, the offline instrument's per-step symbols, and test names | `v51_l1`…`v51_l6` (`scripts/measure_expert_rulebook_transition.py`); `test_l6_reaches_only_the_last_term` |
| 3 | **`MacroRule` values** — snake_case, and an **output contract** because they are recorded in CSVs, evaluation artifacts and analysis tables | `src/thesis_rl/rulebook/v2/types.py` | `collision_safety`, `interaction_risk`, `non_relaxable_compliance`, `mission_progress`, `relaxable_lane_compliance`, `progress_rate` |
| 4 | **`K`-numbering** — this document's five channels `K1`–`K5` | this document, `ADR-083`, and the `A7` ExecPlan | `K4` is convention 2's `L5`; its recorded value is convention 3's `relaxable_lane_compliance` today and becomes `negotiable_lane_compliance` on implementation |

Three consequences, each of them a real failure mode rather than an inconvenience.

- **A search for one convention misses the others.** `L5` does not appear where
  `r3` does; `progress_rate` does not appear where `advance_shortfall` or
  `route_progress` do. Anyone verifying that a channel is gone must search all
  four.
- **The conventions are not in step, and `K4` is where they part.** `K4` is the
  fourth channel, its weight is called `w₅` because it weighted v5.1's fifth
  level, and its recorded value changes while `K3`'s does not (§3). §5.1 uses
  `m₄` for its margin, not `m₅`, so that the margin index follows the channel it
  belongs to.
- **This collision has already caused a defect.** `make_rulebook_tables.py:18`
  compares the **v1** rule name `route_progress` against the **v2** recorded
  value `mission_progress`, so the comparison can never match: the progress
  margin table is written with a header and no rows, and the signed `[−1, +1]`
  margin falls into a violation-rate table instead. It was born correct and was
  orphaned by the 2026-08-20 rename, which updated fifty tests but not this
  consumer, and nothing went red because it is not a test. It is registered and
  fixed as its own defect, ahead of any A7 implementation, and is named here
  because it is the evidence that this section is load-bearing rather than
  tidy-minded.

---

## 4. `K5` — mission progress, the discount, and the budgets

### 4.1 Definition

Unchanged from `RULEBOOK-V5.1` §4.1 (ADR-073), including the clip and its
calibration. Let `s_t` be the arc-length projection onto the assigned route
polyline and

```
Δq_t  = clip( (s_{t+1} − s_t) / D_REF , −1, +1 ),   D_REF = v_ref · Δt = 2.2222 m
K5_t  = Δq_t                                        ΔQ_MAX = +1,  ΔQ_MIN_CLIP = −1
```

**Both bounds are declared, and they come from one expression.** Earlier
revisions of this channel named only `ΔQ_MAX`, which invited the reading that the
negative side is a detail rather than a bound; §5.4's predicate depends on both,
and a specification that names one of them is a specification that will be
quantified over the other by accident.

with `v_ref = MISSION_PROGRESS_REFERENCE_SPEED_MPS = 22.2222 m/s` and
`Δt = 0.1 s`. **The channel is the bare signed advance.** Nothing new is
perceived, mapped or observed.

Three properties carry over unchanged and this document depends on all three.
`D_REF` **is a unit, not a calibration** — changing it only rescales `λ₄`
inversely, and it is fixed at `v_ref·Δt` so that `λ₄` is directly comparable with
the priority weights `a³, a², a`. Undiscounted,
`Σ_t Δq_t = (s_T − s_0)/D_REF`, so the total depends on **distance covered**
rather than on speed. And the clip is what **enforces** `ΔQ_MAX = 1`, which the
§5.4 predicate rests on: it is not merely witnessed by vehicle dynamics, because
`s` is a projection and outruns the ego's own travel on the inside of a bend
(ADR-035 sizes that gap at a factor of about 2) and can jump at a branch
selection. `AC-A7-04` re-asserts the telescoping identity; `AC-RB5.1-05`'s
question — whether the clip binds on a step the agent can produce — carries
over **`NOT ESTABLISHED`** (§9.2), because the engine-force cap bounds travel and
not projection.

**Normalizing by the route's own length is rejected, and stays rejected**
(v5.1 §4.1.1, measured): every mission would then be worth at most `λ₄`
regardless of length while the cost channels stay per-step, and the §5.4 bound
would be set by the **shortest** route in the panel (19.96 m against a 171.98 m
mean), so one degenerate record would dictate `λ₄` for all 1100. Completion
fraction is reported as an episodic metric instead.

### 4.2 Why `K5` is last

Two grounds, and the second is the one that removes a whole class of alternative.

**Mechanism, from the literature.** Under absolute thresholding the last
objective is the unconstrained one (Vamplew et al. §3.2.3, p. 58; Tercan &
Prabhu §3). Progress is the channel a competent driver trades against everything
else, and it is the only channel of the five whose zero is *not* the desired
behaviour, so it is the one that must be left free.

**Mechanism, from this hierarchy's own algebra: a threshold on progress has an
empty requirement.** On the reference pair of §5.5 the two properties a
progress threshold would have to deliver require `τ ≤ 27.97` and `τ > 27.97`
respectively — the same critical value from both sides — because at `γ < 1` the
two comparisons *are* the same comparison at that channel. There is therefore no
`τ₅`: not "none chosen", none admissible. This is what makes A7's channel count
a genuine reduction rather than a relabelling, since v5.1's hierarchy asked a
thresholded arm to constrain five channels, one of which cannot be constrained
at all.

### 4.3 No threshold on `K5`, and no time-preference channel below it

`RULEBOOK-V5.1` placed a sixth level `L6 progress_rate` below the negotiable lane
rules to carry a time preference that an undiscounted return removed. That level
is **deleted**, together with its `advance_shortfall` sub-rule, its weight `λ₆`
and the `Δt/T_REF` scaling. Four statements, none of which is a citation of the
document being replaced — **three of them measured and reproducible, the first
one not**: the 90.5 % / 79.7 % split below does not reproduce from §4's
definitions, and the only live version of that comparison (v5.1 §4.6, +2.05 of
discounted progress advantage against 0.8 of `L6` gain) gives **71.9 %** at
`γ = 0.996`. See §14.1, question 3. The deletion does not rest on it: the other
three grounds are independent of each other and of the discount.

- **Its justification expired.** It rested on `γ = 1`, which was removed
  eighteen days later. At `γ < 1` the discount supplies **90.5 %** of the time
  preference in the comparison `L6` was introduced for (79.7 % at `γ = 0.9982`),
  and supplies it at a **higher** priority than `L6` had, so `L6` was a minority
  duplicate of something the hierarchy already said.
- **It carries no independent information.** `c_L6 = 1 − clip(Δq, 0, 1)` is a
  pointwise function of the progress channel, so the six-channel vector had
  **rank five**; it is the only channel whose return distribution carries nothing
  beyond its mean once progress is known, which is precisely what a
  distributional arm cannot use.
- **Its zero point is unreachable.** That zero sits at the engine cap, **4.9×**
  the logged expert's mean speed, so the channel fires on **99.4024 %** of expert
  steps at `p50 = 0.884`.
- **With progress last, the objective it was introduced to prevent is
  unreachable.** `L6` existed so that a time preference could not pay for a lane
  violation; under A7 the lane channel is *above* progress, so the payment is
  impossible by order rather than by weight.

**The cost is declared in the same place, in two parts.** First, the
below-standstill count can only rise: the diagnostic's baseline moves from
`−λ₆·(Δt/T_REF)·Σ Δq⁺` to exactly **0**, because a stopped ego no longer pays a
per-step charge. §5.5 reports the **measured** consequence rather than the
projection. Second, the incentive to close a mission rather than cover 99 % of it
and idle falls from 0.66–2.99 reward units to **0.57–0.77**, so `L6` was
supplying 13–74 % of it. Both figures are an order of magnitude below the
**8.125** cost of one fully violated interaction step, so neither architecture
makes the last stretch worth a risky manoeuvre; the real gap is that mission
success is a **zero-value terminal**, which is shared and is declared in §11.8. A
terminal completion bonus is **not** specified as compensation: the minimum value
that would make the last stretch worth one fully violated interaction step is
`B ≥ 9.7`, and in the ordered arms `B` lives entirely inside `K5`, below every
safety channel, so it can cause no regression there and fixes nothing either.

### 4.4 The discount: `γ = 0.9982`, with the criterion's verdict asserted beside it

**One discount, shared by every algorithm configuration**, with
`learning_potential_gamma` equal to it. The coupling of the shaping discount to
the MDP's is not a convention: potential-based shaping is policy-invariant only
when the two agree (Ng, Harada & Russell).

**The criterion.** The priority weights are geometric in `a`, and exponential
discounting erodes them at different rates, so a future higher-priority violation
can be damped below a present lower-priority one. The step at which that happens
is the break-even `Δ = ln(a)/−ln(γ)`, and the hierarchy survives the discount
over an episode of length `L` iff

```
ln(a) / −ln(γ)  >  L        equivalently        γ^L  ≥  1/a
```

The two forms are the same statement, which is worth writing down: `γ = 0.9982`
is therefore **the most damping the contraction argument can have** without the
hierarchy argument failing.

**The horizon is measured, not assumed.** `L = 500` control steps: the frozen
index's `length` is the scenario's `SD.LENGTH`, the runtime reads the same field,
and the episode truncates at `scenario_length − 1` with `horizon: null` and
`extra_steps_after_scenario: 0`. The 199 that the previous criterion was evaluated
against was a Waymo-only figure.

**Value and cost, in one place.** `γ ≥ exp(−ln a / 500) = 0.998169` at `a = 2.5`,
and `0.9982` is the first four-decimal value satisfying it.

| quantity | `γ = 0.996` (superseded) | **`γ = 0.9982`** |
|---|---:|---:|
| break-even `ln(a)/−ln(γ)` against `L = 500` | 228.6 — **fails** | **508.6** — holds, by 8.6 steps |
| whole-episode damping `γ^L` against `1/a = 0.4` | 0.1348 | **0.4062** |
| effective horizon `1/(1−γ)` | 250 steps (25.0 s) | **556 steps (55.6 s)** |

The cost, stated plainly: the effective horizon doubles and **40.6 % rather than
13.5 %** of a spuriously bootstrapped constant survives to the end of the longest
episode, which weakens ADR-081's contraction argument. **The factor depends on
the currency and both are stated here, because quoting one of them alone
understates or overstates the cost**: in effective horizon `1/(1−γ)` the
weakening is **2.2×** (250 → 556 steps), while in `γ^L` — the currency ADR-081
itself argues in — it is **3.01×** at `L = 500` (0.1348 → 0.4062) and 1.55× at
the `L = 199` ADR-081 actually used. It costs nothing in calibration, weights or
measurement, because the
weight calibration is an undiscounted per-step condition (§5.4) and the panel
measurement is an undiscounted sum.

**The 8.6-step margin is a property of the current frozen index, not of the
code.** A single 510-step scenario in a regenerated index would put the criterion
back in deficit. `AC-A7-09` therefore requires the **verdict** to be asserted
next to the value — `break_even_steps > horizon_steps` with the horizon read from
the committed index — and not merely the value itself. That distinction is not
pedantic: the superseded criterion *passed* while being false about the quantity
it named.

### 4.5 The budgets `τ₁`–`τ₄`

This section fixes four **values** and one **clip convention**. It deliberately
does not choose a thresholding mechanism, and it deliberately does not choose
which object a threshold is enforced on.

#### The object has to be named before a number means anything

Three candidate objects exist for `τ_i`, and they do not coincide:

| object | what it is | who would consume it |
|---|---|---|
| `X_i` | undiscounted realized per-episode exposure | a policy-gradient lexicographic method, which compares an accumulated episodic return |
| `X_i/Q` | the same, per metre of mission span | the same, made comparable across a panel whose missions span 13–247 m |
| `E[Σ γ^t c_k]` | discounted expected channel return | an absolute-thresholding mechanism on Q-values |

The third is not interchangeable with the first two, for a mechanism rather than
a preference: a **realized** budget requires the accumulated exposure in the
policy's input, which is an `OBS-V1.3.x` amendment and is out of scope here.
Tercan & Prabhu state it directly for this case — the corresponding discounted
threshold "actually depends on the trajectory" (Appendix D.3) — while the
policy-gradient route accumulates its episodic return **undiscounted, from a
single sampled episode**, and therefore consumes the first object.

**This document declares the requirement on the undiscounted realized form
`X_i`**, which is the altitude a behavioural budget belongs at, and records all
three measured values so that the eventual choice costs no second measurement
run. **Which object a threshold is enforced on remains open as `D1`**
(`docs/open_items.md`), and choosing a number's units here would decide it by
stealth.

#### The rule, fixed before the numbers were seen

Two requirements, in this order.

1. **The budget must admit the logged competent driver.** A budget quantified
   over zero-exposure completions describes no trajectory an agent can produce:
   the expert accrues interaction cost on **0.3978 %** of steps and
   non-negotiable compliance cost on **0.6418 %**. A zero budget on `K3` is
   therefore falsified **by the panel**, not by preference.
2. **The rule must be fixed before the numbers are seen**, or the budget is
   fitted to the distribution it is supposed to be tested against.

**Rule.** `τ_i` = the **maximum** of the logged expert's per-episode value on the
frozen Waymo `train` panel, minus any episode attributable to a **declared**
panel defect, with the excluded episodes **listed per record**; `p95` and `p99`
reported beside it as sensitivity.

The maximum rather than a quantile follows from requirement (1) and is not a
matter of taste: a quantile excludes human episodes while the rule's own first
requirement says that excluding a human episode falsifies the budget. Had `p99`
been adopted, `τ₂` would read **8.07** against the panel's **110.38** — an
exclusion of about eleven episodes a competent human actually produced.

**The cost, in the same sentence:** the maximum is the loosest budget that stays
falsifiable, so for a policy at or below human exposure the thresholded arm does
not constrain that channel at all, and its differentiation from the scalar
control on that channel comes from the *ordering* alone. **The rule is void
unless the per-record rows are emitted** — an aggregate that cannot be audited
per record is an assertion — which is why the evidence directory carries them.

#### The measured values

Measured on **2026-09-10**, full frozen Waymo `train` panel, **1100 records,
217,189 transitions, 0 skipped**, logged-expert replay through production's own
`evaluate_transition`. **No panel defect is declared for this run**, so the
excluded-record list is empty rather than omitted and each `τ_i` is the
unmodified maximum. Source of record:
`docs/audits/a7_m1_measurement_2026-09-10/` (`a7_m1_summary.json`,
`a7_measurement.exposure_by_channel`).

| `τ_i` | channel | exposure quantity | **`τ_i`** (undiscounted realized max) | per-span max | discounted max (`γ = 0.9982`) | `p99` | `p95` |
|---|---|---|---:|---:|---:|---:|---:|
| `τ₁` | `K1` | `X_imp` — at-fault impact | **0.000000** | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `τ₂` | `K2` | `X_int = Σ_t max(c_ttc, c_clearance, c_rss_lateral)` | **110.379799** | 0.720226 | 98.733955 | 8.067951 | 1.132411 |
| `τ₃` | `K3` | `X_hard = Σ_t max(c_offroad, c_signal, c_stop, c_crosswalk, c_vehicle_yield, c_speed_limit)` | **21.287443** | 0.264242 | 19.613411 | 3.323886 | 0.640988 |
| `τ₄` | `K4` | `X_soft = Σ_t (c_solid_line + c_wrong_carriageway + c_dashed_line)/3` | **23.386514** | 0.514168 | 20.197176 | 2.737476 | 0.000425 |

The record setting each undiscounted maximum is named, so that a future session
with grounds to declare a panel defect has a record to point at rather than an
aggregate to take on faith: `waymo:training_20s:71344f609367eace` for `X_int`,
`waymo:training_20s:5f8217a5da0b24ff` for `X_hard`,
`waymo:training_20s:1ef1a62b6bebe06` for `X_soft`. Excluding any of them from a
future `τ_i` requires a declared, specific reason for that record.

**`τ₁ = 0` is admissible, measured rather than assumed.** `X_imp` is exactly
`0.000000` for every one of the 1100 episodes on all three objects: the logged
expert records no at-fault impact anywhere on the panel. A zero budget there is
what makes the ordered arms prefer enduring 200 steps of interaction violation to
colliding at fault — `K1 = 0.0000` against `0.5788` on that probe — where the
scalar arm prefers colliding by 1092.8 (§11.7).

**`τ₂` and `τ₃` sit far above their own `p99`** (110.38 against 8.07; 21.29
against 3.32), which is the shape the rule anticipated: the per-episode exposure
is heavy-tailed and the rule is the maximum precisely because a competent human
produced the outlier episode. The tail is consistent with a small number of
genuinely difficult records rather than with measurement noise, but that has not
been independently confirmed, and no panel defect is declared on that basis.

#### The clip convention, and why `τ₁ = 0` does not mean "everything ties"

For a **cost** channel the thresholded comparison clips with

```
max(c, τ)         cost channels K1..K4       — everything at or below the budget is equivalent
min(v, τ)         the progress channel K5    — everything at or above it is equivalent
```

The direction follows from what the clip is for: a budget says "do not
distinguish inside the budget", which for a cost is achieved by raising every
value below `τ` up to `τ`, and for a quantity where higher is better by lowering
every value above `τ` down to `τ`. Getting it backwards is not a cosmetic error —
`min(c, τ₁) = min(c, 0) = 0` for every non-negative cost, so a collision would
tie with a non-collision, and on `K5` the reading of the channel inverts. It is
recorded here because it changed the conclusions of a bench in which it occurred.

It follows that **`τ₁ = 0` means zero tolerance, i.e. strict lexicographic
comparison on that channel** — `max(c, 0) = c` — and not that all values collapse.

**One consequence of the budget rule has to be stated here rather than left for a
reader to discover, because it touches this document's headline claim.** At the
measured budgets the thresholded comparison does **not** reproduce A7's gain on
O3. The reference pair of §5.5 accrues `Σ c_K4 = 30 × 1/3 = 10.0` units of
negotiable exposure against `τ₄ = 23.386514`, so `max(10.0, τ₄) = max(0, τ₄)`:
the shortcut and the legal route **tie on every cost channel**, the comparison
falls through to `K5`, and the shortcut wins there by arriving sooner. The
shortcut would need **70 of its 160 steps** on a marking — 35 % of the episode —
to leave the budget at all. This is not a defect of the rule; it is the rule's own
declared cost ("for a policy at or below human exposure the thresholded arm does
not constrain that channel at all, and its differentiation from the scalar
control on that channel comes from the *ordering* alone") applied to this
particular pair. But it means the O3 gain of §1.1 is a property of the **scalar**
arm and of the **strict lexicographic** arm, and *not* of the thresholded arm at
these budgets — and §11.1 leans on the thresholded comparison where it helps
(O1), so the asymmetry must be declared rather than left implicit. Recorded as
§11.16.

**There is no `τ₅`** (§4.2): not "unspecified", but inadmissible, and the last
channel is unthresholded by the requirement of the thresholding family this arm
belongs to. Under **slacking** rather than thresholding the loop runs to the last
objective and slacks it too, so "the last channel is unthresholded" is a statement
about the mechanism and not about this architecture; the mechanism itself is
`D1`'s and is out of scope (§13).

---

## 5. Scalarization — `SCAL-V1.5`

### 5.1 Form

```
r_t  =  Σ_{k=1..3} a^(4−k) · [ (1_sat(m_k) − 1) + σ · m_k ]      priority block, K1..K3
        +  w₅ · [ (1_sat(m₄) − 1) + σ · m₄ ]                     K4, a sub-unit level
        +  λ₄ · Δq_t                                             K5, open-ended
```

with `m_k = −c_Kk ∈ [−1, 0]` for `k ∈ {1, 2, 3}`, `m₄ = −c_K4 ∈ [−1, 0]`,
`Δq_t ∈ [−1, +1]` as defined in §4.1, `1_sat(m) = 1` iff `m = 0`, and margins
canonicalized to exactly 0 before `1_sat` is applied. Selected values are in
§5.5.

**Three things changed relative to `SCAL-V1.4`, and each is a term rather than a
number.**

1. `K4` moves from the continuous tail into the priority block *as a sub-unit
   level*: it gains a satisfaction indicator and the severity slope, and loses
   the `Δt/T_REF` factor. A satisfied `K4` contributes **exactly 0**; a fully
   violated one costs `w₅·(1+σ)`.
2. The `L6` term is **deleted** with its level (§4.2).
3. The shared absolute tie-breaker `φ` is **removed**, i.e. `φ = 0`, and the term
   `φ·Σ_k m_k` disappears (§5.5).

`K1`–`K3` are `SCAL-V1.2` verbatim, so their per-step dominance is inherited
rather than re-argued. The version is bumped because the *form* changed, not
because a weight moved.

**Why `K4` needs the indicator and the position together.** The indicator is what
gives the channel a discrete satisfaction event, so that a satisfied `K4`
contributes nothing and a violated one is charged a fixed crossing price with a
severity slope on top; the position is what makes it decide before progress in
the ordered arms. Neither alone is sufficient: with the indicator but below
progress, an ordered arm never reaches it on any pair that differs in progress;
above progress but charged continuously, the scalar arm's exchange rate against
progress collapses as the violation becomes small. The scalar arm additionally
needs a **finite** exchange rate, which no ordering supplies, and that is the one
respect in which the scalarization expresses something the ordered arms cannot.

### 5.2 Why `K5` cannot be an additional priority level

Unchanged in substance from v5.1 §5.2, and it is why `λ₄·Δq` sits outside the
priority block. `SCAL`'s per-step dominance works because each level carries a
**satisfaction indicator**, which makes the level discrete. Progress is
continuous, and strict dominance of a lane channel over progress would require
`w₅·(1+σ) > λ₄·Δq` for **every** `Δq > 0`, including `Δq → 0⁺` — which no finite
`λ₄` violates and no finite `w₅` needs. The two ways to force it are both
rejected: an indicator on progress (`1[Δq > 0]`) rewards infinitesimal creep,
converting "stop for ever" into "creep for ever along the marking"; an indicator
on a progress threshold (`1[Δq > δ]`) introduces an unmotivated constant and a
cliff a policy will sit on.

Under A7 the exchange runs the other way round from v5.1 — `K4` is above `K5`, so
the finite exchange is what lets progress ever outweigh a lane violation at all —
but the algebra is the same and so is the conclusion: **the finite exchange is
not a concession, it is the only construction that claims exactly what it can
prove.**

### 5.3 What is and is not lost relative to `SCAL-V1.4`

Nothing that was proved is lost. The proved property is strict per-step dominance
over the priority block, and it is retained exactly where it was demonstrated;
`SCAL-V1.4` never had episodic lexicographic order either.

What is **gained** is O3 on the scalar arm at a discount (§1.1), and what is
**lost** is one per-step normalization and two weights. The scalarization remains
the adapter for **one of four arms**: the lexicographic and distributional arms
consume §3.4's vector directly and never evaluate this expression.

### 5.4 Rank-preservation condition

> **RESOLVED 2026-09-12 — the progress term is counted on both sides, and the
> guarantee is stated as conditional.** What follows records why, because the
> correction is not cosmetic and it changes what §5.4 claims.
>
> `RULEBOOK-V5.1` §5.4 carried `λ₄·ΔQ_MAX` on the violated side only.
> `rulebook_v5.0_UNDER_REVIEW_specification.md` §6.3 states the same predicate
> with `2λ`, derives it symmetrically ("a violation at level `k` scores at best
> `−w_k + λ`; the same level satisfied scores at worst … `− λ`"), and names the
> reason: "that is where the asymmetry between the three cost channels (bounded
> in `[−1, 0]`) and the progress channel (bounded in `[−1, 1]`) is resolved".
> The two-sided form reproduces **both** published anchors v5.0 checks it
> against — `a > 2` at `σ = φ = 0, λ = 1`, which is Veer et al.'s own condition,
> and `a ≥ 2.92` at `σ = 1, λ = 1` (reproduced: 2.9196) — and the one-sided form
> reproduces **neither** (reproduced: **1.8393**, the tribonacci constant, and
> 2.8312; the binding level is `k = 1` in both, and the `a > 1` that an earlier
> revision of this note quoted is the `k = 3` slice alone). Under the two-sided
> form at `σ = φ = 0` all three levels bind at exactly 2.0, which is the
> coincidence a correct condition produces and the one-sided form does not.
> At the weights of §5.5 a two-state counterexample refutes the "iff" as written,
> and it does **not** need a knife-edge violation: `K3` violated at **full**
> severity with `Δq = +1` scores **−1.2500**, while nothing violated at all with
> `Δq = −1` scores **−2.0000**, so the scalarization prefers a completely
> violated non-negotiable channel, by 0.75 reward units.
>
> **The halving is recorded in the instrument's own code as deliberate, and its
> consequence with it.** `scripts/measure_expert_rulebook_transition.py` carries
> *both* forms ninety lines apart: `is_rank_preserving` (`:816`) is two-sided
> (`+ 2.0 * progress_weight`) and its docstring states the symmetric derivation
> and both anchors, while `v51_is_rank_preserving` (`:903`) is one-sided and its
> docstring says "v5.0's progress channel swung over `[-1, 1]` and contributed
> `2 * lambda`, while here the tail is … roughly twenty times smaller, **which is
> exactly why eta comes out unconstrained**". So the change was seen and its
> effect was written down — but what is written is a *consequence*, not a
> justification, and "only the tail differs" conceals that what differs is how
> many times the progress channel enters.
> Under the two-sided form `λ₄ = 2.0` is **inadmissible** at `k = 2` and `k = 3`
> (and the six-level weight set in production today fails at `k = 1` as well),
> and the corollary `λ₄ ≤ a/2 = 1.25` that an earlier revision attributed to a
> different desideratum is *exactly* the two-sided bound at `w₅ = 0` — the same
> number arriving twice by two routes. **This is not a defect introduced by A7**:
> the one-sided form is `RULEBOOK-V5.1` §5.4's, it is what
> `reward/scalarization.py` enforces today, and ADR-081 calibrated `a` and `σ`
> against it. But A7 is the change that rewrites the predicate, so carrying it
> forward unexamined would have frozen it a third time.
>
> **The correction adopted here** is neither of the two forms: it is the
> generalised one, `λ₄·(ΔQ_MAX − ΔQ_MIN)`, with `ΔQ_MIN` a declared symbol.
> Taking `ΔQ_MIN` from the expert panel (`−0.0239`, which would leave a thinnest
> ratio of 1.1147 and cost nothing) was considered and **rejected on principle**:
> one clip expression produces both bounds (§4.1), so reading `+1` as normative
> construction and the negative side as an empirical convenience — inside the
> same inequality — is not available. The price is stated below and in §14.3:
> this document loses an unconditional theorem and gains a **conditional
> guarantee with a runtime falsifier**. No weight moves.

Level `k ∈ {1, 2, 3}` dominates everything below it iff

```
a^(4−k)  >  (1 + σ) · Σ_{j>k, j≤3} a^(4−j)  +  w₅ · (1 + σ)  +  φ · (3 − k)  +  λ₄ · (ΔQ_MAX − ΔQ_MIN)
```

**`ΔQ_MIN` is a declared quantity of this contract and it takes two values that
must never be conflated.**

| symbol | value | what it is |
|---|---:|---|
| `ΔQ_MIN_CLIP` | **−1** | what §4.1's clip permits. The channel is signed and one `clip(·, −1, +1)` produces both bounds |
| `ΔQ_MIN_GUARANTEED` | **−0.1525** | the condition under which the selected weights preserve the ordering. Not a free constant: it is what `(a, σ, λ₄, w₅) = (2.5, 0.30, 2.0, 0.15)` *buys*, and it tightens if any of them moves |

**At `ΔQ_MIN_CLIP` the selected weights do not satisfy the condition**, and this
document states it rather than hiding it: the ratios are **1.0035 / 0.8395 /
0.5959**, so only `k = 1` holds — the collision channel, which cannot be bought
even at the clip. The six-level set running under `SCAL-V1.4` fails all three
(0.9769 / 0.8202 / 0.6068). **§5.4 is therefore not an unconditional theorem,
and no revision of this specification was ever entitled to state it as one.**

**What the selected weights do buy is this, and it is the operative statement:**

> The per-step ordering is lexicographic in `K1 ≻ K2 ≻ K3 ≻ K4` **for as long as
> the compliant trajectory satisfies `Δq ≥ ΔQ_MIN_GUARANTEED = −0.1525`** — that
> is, for as long as it loses no more than **0.339 m of route station in one
> 0.1 s step**.

Three facts about that condition, each with its source, because a condition
without them is an excuse:

- It is **6.4× outside** anything the logged expert does: the station never falls
  more than **0.053 m** below its running maximum over 217,189 steps
  (`behind_peak.max_m` = 0.053, `steps_beyond_1m` = 0, committed in
  `docs/audits/reward_calibration_2026-09-07/rb51_calibration_a_sigma_l4.json`).
- It is **not reachable by choice**: two actions available in the same state
  differ by 0.025–0.045 of `Δq` in one 0.1 s step, so no policy can *select* a
  0.339 m station loss. Only route geometry can impose one.
- It is **not vacuous**: §11.5's closed loop is exactly the geometry that
  produces large negative `Δq`, and the *positive* side of the same clip binds on
  0.60 % of expert steps at up to 3.609 m — so the projection demonstrably
  outruns the vehicle, and there is no reason to assume it does so in one
  direction only.

**The guarantee is falsified at runtime rather than assumed.** `AC-A7-17`
requires a per-episode count of steps with `Δq ≤ ΔQ_MIN_GUARANTEED`, published on
the diagnostic path that already carries `l4_clip_binding_steps`. A non-zero count
is a measured violation of a declared condition, read by the rule pre-registered
in §14.3.

**The constructor gate evaluates the inequality at the `ΔQ_MIN` it is given**,
which is therefore configured rather than a literal. Given `ΔQ_MIN_GUARANTEED` it
admits the selected weights; given `ΔQ_MIN_CLIP` it refuses them, which is the
correct behaviour for that input. `TEST-A7-10` pins both directions.

**Derivation**, in the same structure as v5.1 §5.4 and re-derived independently
for this document rather than transcribed. With level `k` violated, the best
attainable score is `−a^(4−k) + λ₄·ΔQ_MAX`, approached as `m_k → 0⁻`. With level
`k` satisfied and every channel below it maximally violated, the worst attainable
score is `−(1+σ)·Σ_{j>k, j≤3} a^(4−j) − w₅·(1+σ) − φ·(3−k)`. Requiring the second
to exceed the first gives the condition.

**The symmetry in that derivation is the whole correction.** "Every channel below
it maximally violated" now applies to `K5` as well: the violated side takes
`Δq = +ΔQ_MAX` and the satisfied side `Δq = ΔQ_MIN`, so the progress channel
contributes its whole swing `λ₄·(ΔQ_MAX − ΔQ_MIN)` instead of half of it. The
superseded form pinned the satisfied side at `Δq = 0`, which is an assumption
about the route projection — that the compliant trajectory never loses station —
stated nowhere and false in the geometry §11.5 measures.

**Three placements in that expression are load-bearing, and getting any of them
wrong produces a plausible-looking wrong predicate.**

- `w₅` enters **once**, with the factor `(1+σ)`, because `K4` is now an indicator
  level whose maximal per-step magnitude is `w₅·(1+σ)`. Under `SCAL-V1.4` the
  corresponding term was `η·(Δt/T_REF)` **without** `(1+σ)`, because that channel
  was charged continuously and carried no severity slope. The `(1+σ)` is a
  consequence of change 1 in §5.1, not a decoration.
- `φ` multiplies **only the priority levels below `k`**, which is `(3 − k)`. `K4`
  is not one of them. At `k = 3` there are none, so `φ` contributes nothing there.
- `λ₄·(ΔQ_MAX − ΔQ_MIN)` is the whole progress tail, and **both** bounds hold
  because one clip enforces them (§4.1). Writing only `λ₄·ΔQ_MAX` here — as every
  revision before this one did — pins the satisfied side at `Δq = 0` and is the
  defect this section corrects.

**The trap, recorded because it was hit.** A transcription that lets `w₅` join
the ordinary lower-level set — inside the `(1+σ)` sum **and** in the `φ` count —
gives margins of 1.0911 / 1.0513 / 1.0225 with the thinnest at `k = 3`. Those
numbers are wrong, and they are wrong in a way that looks right: they are
monotone and they are near 1. The predicate must be reproduced against the term
placement of §5.1, not against a plausible reading of it.

**Margins at the selected weights** `a = 2.5, σ = 0.30, λ₄ = 2.0, w₅ = 0.15`,
as the ratio of the left side to the right side at each `k` (a value above 1 is
admissible), re-derived for this document (§14):

| configuration | `k = 1` | `k = 2` | `k = 3` | thinnest |
|---|---:|---:|---:|---:|
| `SCAL-V1.4` today (six levels, `φ = 0.25`, `η = 1.0`, `λ₆ = 0.2`) | 1.1165 | 1.1121 | 1.1792 | **1.1121** at `k = 2` |
| **`SCAL-V1.5`, `φ = 0`** | 1.1514 | 1.1478 | 1.1390 | **1.1390** at `k = 3` |
| `SCAL-V1.5` with `φ = 0.25` retained | 1.1105 | 1.0975 | 1.1390 | 1.0975 at `k = 2` |

`φ = 0` therefore **returns** the only structural cost A7 has on this axis: the
thinnest margin goes 1.0975 → 1.1390, above `SCAL-V1.4`'s own 1.1121 (§5.5).

**The condition is enforced at construction.** Inadmissible weights are refused
when the scalarizer is built, not priced at use, because below the condition one
step of progress or of relaxation can overturn a higher-level violation and the
weights stop being an ordering. `AC-A7-08` and `TEST-A7-10` pin that it is a
constructor gate.

**The cap on `w₅` follows from the same inequality**, at its binding level
`k = 3` where the lower-level sum is empty, and it is now a **function of
`ΔQ_MIN`** rather than a constant:

```
w₅  <  [ a − λ₄·(ΔQ_MAX − ΔQ_MIN) ] / (1 + σ)
```

| `ΔQ_MIN` | cap on `w₅` | note |
|---:|---:|---|
| 0 (the superseded form) | 0.384615 | the figure earlier revisions printed |
| −0.0239 (expert panel) | 0.3478 | |
| **−0.1525** (`ΔQ_MIN_GUARANTEED`) | **0.1500** | the selected `w₅ = 0.15` sits exactly on it |
| −1 (`ΔQ_MIN_CLIP`) | negative | no `w₅` is admissible |

**That the cap at `ΔQ_MIN_GUARANTEED` equals the selected `w₅` exactly is not a
coincidence and must not be read as one.** `ΔQ_MIN_GUARANTEED` is *defined* as
the largest station loss at which the selected weights still satisfy the
inequality, so the two are two readings of one fact: **at `w₅ = 0.15` the weights
buy a guarantee down to −0.1525, and raising `w₅` tightens the guarantee.** At
`w₅ = 0.25`, the alternative §5.5 prices, the guarantee shrinks to `Δq ≥ −0.0875`
(0.194 m per step), which is still 3.7× outside the expert's measured 0.053 m.

The `λ₄` headroom likewise depends on `ΔQ_MIN`: `λ₄ < a/(ΔQ_MAX − ΔQ_MIN)`, i.e.
2.5 at `ΔQ_MIN = 0`, **2.1692** at `ΔQ_MIN_GUARANTEED` — which the shipped
`λ₄ = 2.0` satisfies — and 1.25 at `ΔQ_MIN_CLIP`. The last of those is the
`λ₄ ≤ a/2` that earlier revisions attributed to an unrelated desideratum: it is
this inequality at `w₅ = 0` and the full clip, which is why the same number
arrived twice by two routes.

### 5.5 Selected weights

**`a = 2.5`, `σ = 0.30`, `λ₄ = 2.0`, `w₅ = 0.15`.** `a`, `σ` and `λ₄` carry over
from ADR-081 unchanged and are not reopened. `η`, `λ₆` and `φ` cease to exist.

#### `w₅ = 0.15`

**Criterion, stated before the value.** `w₅` is an exchange rate between
negotiable lane exposure and arrival time, so both of its bounds must be stated
in the same physical currency and neither may be a dimensionless preference.

- **Lower bound — the reference shortcut must not pay.** The reference pair is
  the one `RULEBOOK-V5.1` §4.6 stipulates and this document keeps, because a
  bound is only comparable across the two architectures if the construction is
  identical: the *cheapest* illegal shortcut saving the *most* time, which over a
  200-step episode saves **40 steps** while riding **one** marking for **30
  steps** at `c_K4 = 1/3`. **The 30 marking steps are the shortcut's first 30**,
  and that clause is normative rather than cosmetic: the placement is what the
  executed v5.1 fixture uses, it is the only placement at which v5.1 §4.4's
  published margin table reproduces, and the margin is sensitive to it at the
  superseded discount (below). Measuring the scalar margin of that pair gives
  `w₅ > 0.1313` at `γ = 0.996` and `w₅ > 0.0736` at `γ = 0.9982` — the bound
  loosens at the higher discount because arriving sooner is worth less there.
  **Both figures are reproduced in §14.2** (0.131342 and 0.073555), by a
  reconstruction validated against all six entries of v5.1 §4.4's published
  table, so they are checkable rather than carried; `TEST-A7-15c` is what turns
  them into executed evidence under this document.
- **Upper bound — crossing a marking must stay cheaper than the manoeuvre it
  replaces.** Stopping from an urban `v = 10 m/s` and returning to it at a
  comfortable `a_c = 2 m/s²` costs `v/a_c = 5.0 s` of delay, and one second of
  marking contact must cost less than that: `w₅ < 0.4477` at `γ = 0.996` and
  `w₅ < 0.2641` at `γ = 0.9982`. **These two figures do not reproduce from §4's
  and §5's definitions**, and the model that converts 5.0 s of arrival delay into
  reward units is stated in no live document — see §14.1, open question 2. They
  are carried here because the selected `w₅` sits inside them either way, but
  `0.2641` is the bound that actually constrains `w₅` at the adopted discount and
  it is the denominator of the 72.1 % window below, so this is not a decorative
  gap.
- **Independent cap** from §5.4: `w₅ < 0.384615`.

**The binding bound is the lower one at the superseded discount**, deliberately,
so that this architecture stands whether or not the discount decision of §4.4 is
taken: `w₅ > 0.1313`, rounded up to **0.15** for margin, because the reference
shortcut is a stipulated construction rather than a measurement. **That
robustness is qualified at `γ = 0.996` and unqualified at `γ = 0.9982`**: the O3
margin at the superseded discount falls from **+0.5813** with the marking on the
shortcut's first 30 steps to **−0.0141** once it starts at step 34, so at that
discount the ordering survives only for placements inside the first ~33 steps,
while at the adopted `γ = 0.9982` every placement holds (**+2.4579** to
**+1.4509**). Reproduced in §14.2. Both windows contain 0.15 with room —
**65.9 %** of its own effective upper bound at `γ = 0.996` and **72.1 %** at
`γ = 0.9982` (a ratio that mixes two different effective upper bounds; see
§14.1), against the **0.193 %** of the admissible `η` range within which
`SCAL-V1.4` could buy O3 back at all.

**What the value means, in units — at two severities, because the sentence one
writes depends on which.** The exchange rate is
`Δq_break-even = w₅·(1 + σ·c_K4)/λ₄`, i.e. the extra route advance per step that
exactly pays for carrying `c_K4`:

| `c_K4` | what it is | per-step cost | break-even `Δq` | sustained advance |
|---|---|---:|---:|---:|
| `1/3` | **one marking** — the severity of the reference pair | 0.165 | 0.0825 | **1.83 m/s** |
| `2/3` | two of the three sub-rules | 0.180 | 0.0900 | 2.00 m/s |
| `1` | the channel **fully** violated | 0.195 | 0.0975 | **2.17 m/s** |

So a learner reads `w₅ = 0.15` as: **cross one marking only if doing so buys
about 1.8 m/s of extra route advance while you are on it**, rising to 2.2 m/s if
all three negotiable sub-rules are violated at once. An earlier revision of this
section quoted only the `c_K4 = 1` figure and attached it to the
one-marking sentence, which understates what a single marking costs by 18 %; the
two severities are distinguished here because the reference construction the
bounds below are measured on uses `c_K4 = 1/3`.

**What `w₅` does not buy, and it belongs in the same paragraph.** No admissible
`w₅` opposes a **fast** off-corridor drive. The crossover is
`w₅ = λ₄·Δq/(1+σ)`: **0.313983** at the expert's mean pace `Δq = 0.204089`, which
is admissible, and **1.538462** at the clip, which is not — against the cap of
**0.1500** at `ΔQ_MIN_GUARANTEED` (0.384615 under the superseded one-sided form).
The ratio of the clip-pace requirement to the cap is
`λ₄/[a − λ₄·(ΔQ_MAX − ΔQ_MIN)]`, in which `σ` cancels, so it is a property of the
weights and no re-tuning of `σ` reaches it: **10.26** at `ΔQ_MIN_GUARANTEED`,
against the 4.00 the one-sided form reported. That denominator *is* the §5.4
tail, so "no admissible `w₅` opposes a fast off-route drive" is the same statement
as "progress consumes **92.2 %** of the `k = 3` per-step budget" (80 % under the
superseded form). Correcting the predicate therefore does not create this
limitation — it reveals that it was understated by a factor of 2.6. Declared in
§11.3.

#### `φ = 0`

Four grounds that compound.

1. **Its stated job is already done.** `φ` breaks the tie the satisfaction
   indicator creates between two margin vectors with the same discrete
   satisfaction pattern. At `σ = 0`, where `φ` was set, the priority term was
   constant inside the violated set and the tie was real; at `σ = 0.30` the slope
   `σ·a^(4−k)` is non-zero at every level and there is no tie left to break.
   Mechanism, not preference.
2. **Its shape is the one ADR-081 criticised.** Being absolute, its grading is
   inversely proportional to importance: `φ` supplies **5.1 %** of the level's
   total in-violation slope at `k = 1`, **11.8 %** at `k = 2` and **25.0 %** at
   `k = 3` — i.e. `φ/(σ·a^(4−k) + φ)`; as a fraction of the severity slope alone
   it is 5.3 / 13.3 / 33.3 %. The most important level is the flattest either
   way.
3. **No document derives the value.** 0.25 traces to Veer et al.'s averaged-
   robustness tie-breaker `1/N` with `N = 4`, for *their* four-level schema, here
   summed over three margins. The only argument for `φ = 0.25` in this repository
   is a specification of this repository, which by `AGENTS.md`'s Scientific
   Argument Standards is a **finding** rather than a justification — and it points
   at the constant, not at the code.
4. **It returns A7's only structural cost on the §5.4 axis**, 1.0975 → 1.1390.

**Value and cost in one sentence.** `φ = 0` — because at `σ = 0.30` the tie it
breaks no longer exists — at the cost of `a_req^max` falling from **12.43** to
**10.96 m/s²**, i.e. from 1.38× to 1.22× the ≈9 m/s² a real vehicle can produce,
which is the margin by which the reward's local gradient still points toward
braking in the hardest conflict a vehicle could resolve. **The criterion is
pinned in its strict form, `a_req^max > 9 m/s²`**: 9 m/s² is already peak braking
on dry asphalt, so margin above it protects against nothing physical, only
against uncertainty in the number itself. If margin is wanted back the lever is
`σ`, which scales with each level's own weight, not `φ`, which does not.

#### Measured under the A7 reward, on the full panel

Measured 2026-09-10 on the frozen Waymo `train` panel — 1100 records, 217,189
transitions, 0 skipped — with the standstill baseline at exactly **0** under A7
(`L6` is gone). Source: `docs/audits/a7_m1_measurement_2026-09-10/`.

| grid member | `w₅` | `φ` | `λ₄` | `fraction_below_standstill` | expert mean episode return |
|---|---:|---:|---:|---:|---:|
| **selected** | **0.15** | **0** | **2.0** | **4.64 %** | **71.34** |
| `φ` retained | 0.15 | 0.25 | 2.0 | 4.73 % | 71.20 |
| `w₅` alternative | 0.25 | 0 | 2.0 | 4.82 % | 71.17 |
| `w₅` + `φ` alternative | 0.25 | 0.25 | 2.0 | 4.91 % | 71.02 |
| `λ₄` alternative | 0.15 | 0 | 1.9 | 4.91 % | 67.32 |
| `λ₄` alternative | 0.15 | 0 | 1.25 | 6.45 % | 41.12 |

Every member is admissible under §5.4 and every member is **under the 7.45 %
ceiling** that `AC-A7-11` inherits from the superseded architecture's own
acceptance criterion. Three readings, all of them pre-registered before the run:

- **`φ = 0` is not falsified.** The movement from `φ = 0.25` to `φ = 0` is
  **+0.09 pp**, against **2.81 pp** of remaining headroom to the ceiling. The
  falsifier — revert to 0.25 if the ceiling is breached — does not trip.
- **`λ₄ = 2.0` stands.** It was to be reopened only if its own member breached
  the ceiling, which it does not (4.64 %). Had it been reopened, the first lever
  is 1.9 (**+0.27 pp**) and not 1.25 (**+1.81 pp**), and 1.25 buys an option that
  cannot be exercised: the channel that would have to fire is `K4`, and an ego on
  a legal parallel carriageway violates none of its three sub-rules.
- **Deleting `L6` costs less than the projection.** The projection was ≈5.2 %
  from a measured 4.55 % plus a bounded 0.53 pp; the measurement is 4.64 %.

**The expert mean is positive and higher than the superseded architecture's, and
the comparison has to name its baseline.** A7 measures **71.34** at the shipped
`(a, σ) = (2.5, 0.30)`. The v5.1 figure published in that document's §5.5 is
**70.70**, but it was measured at the *pre-*ADR-081 pair `(2.2, 0)`; at the
shipped pair the six-level reward measures **68.31**
(`docs/audits/reward_calibration_2026-09-07/`). So the single-factor statement —
identical panel, identical atomic costs, varying only the adapter — is
**71.34 against 68.31**, and the claim holds under either baseline. The 70.70
comparison is retained only because it is the number the previous specification
publishes. A consistency check rather than a design target (§9.2).

---

## 6. Observation consequences

**None.** `RULEBOOK-V5.1` §6 carries over unchanged. A7 adds no observation
field, changes no observation dimension and needs no `OBS-V1.3.x` or
`OBS-LIDAR-V2.0.x` amendment, because the atomic vector of §3.4 is untouched and
this document changes only the comparison order and the scalar adapter.

One consequence is worth stating in the negative, because it bounds a whole class
of future work: a **realized** episodic budget would need the accumulated
exposure in the policy's input, which *is* an observation amendment. That is one
of the reasons §4.5 declares the requirement rather than the mechanism.

## 7. Diagnostics

`RULEBOOK-V5.1` §7 carries over, with the deltas the channel change forces and
nothing else:

- the recorded channel vocabulary loses `progress_rate` and takes the order of
  §3; `advance_shortfall` disappears from every diagnostic surface;
- `l4_clip_binding_steps` is retained under the `K5` name and is what makes
  `AC-RB5.1-05`'s open question measurable on agent trajectories rather than
  deducible (§9.2). Note that it counts `|Δq| ≥ 1 − ε`, i.e. it is **blind to the
  sign**, so it does not answer §5.4's question;
- **one counter is added, and it is the falsifier of §5.4's guarantee**:
  `k5_below_guaranteed_min_steps`, the per-episode count of steps with
  `Δq ≤ ΔQ_MIN_GUARANTEED = −0.1525`, published on the same per-episode path as
  the counter above. It is signed by construction. `AC-A7-17`, `TEST-A7-23`, and
  the reading rule is pre-registered in §14.3. A signed minimum
  (`k5_min_delta_q`) is published beside it so that a violation can be sized and
  not merely detected;
- **one guard is added**, because a stale consumer of A7 output fails
  *invisibly*. Four of five channel names survive A7, so a consumer written for
  the six-level vocabulary produces a **populated and wrong** table rather than
  an empty one — unlike the 2026-08-20 rename, where every name moved and a stale
  consumer failed visibly. The scalarization vector schema id therefore joins the
  analysis condition identity, so that two runs whose vector schemas differ can
  never be pooled into one condition. `AC-A7-16` / `TEST-A7-22`.

The **checkpoint reward-semantics identity breaks**, by three independent routes:
the margin vector changes arity from 6 to 5, the vector schema id changes, and the
weight set changes. This is intentional and is not a compatibility bug.

## 8. Measurement status

Three instruments establish three different classes of statement, and they are
not interchangeable. Reading a figure from the wrong one is how this repository
has previously mis-stated its own results, so each figure in this document names
its instrument.

| class of statement | instrument | status |
|---|---|---|
| Sub-rule admissibility (Test A / Test B) | expert replay, per sub-rule | Established by v5.0, **inherited unchanged**: A7 changes no sub-rule |
| Channel exposure, expert return, below-standstill, argmax frequency, `K2`/`K3` co-occurrence | expert replay of 1100 frozen Waymo `train` records, 217,189 transitions, through production's own `evaluate_transition` | **Measured 2026-09-10** (`docs/audits/a7_m1_measurement_2026-09-10/`) |
| Orderings O1–O6 and the reward-hacking probes | **constructed** fixtures — replay holds one trajectory per scenario and therefore no counterfactual | **`NOT_RUN` under A7.** Scheduled as `TEST-A7-15`/`-16`; the figures quoted in §1.1 are the `A7` ExecPlan's, from its own bench |
| Production equals the offline instrument per step | oracle agreement on the frozen panel to `1e-9` | **`NOT_RUN`.** `AC-A7-12` / `TEST-A7-14`, and it is the criterion that converts every figure above from a claim about a script into a claim about production |

One figure in this document is carried from the `A7` ExecPlan and reproduces from
no live source: `w₅`'s **upper** bounds 0.4477 / 0.2641, because the model that
prices 5.0 s of arrival delay in reward units is stated nowhere (§14.1,
question 2). The O3 scalar margins and `w₅`'s **lower** bounds, which an earlier
revision of this document also listed as carried, are reproduced in §14.2 from
the reference pair — the reconstruction is validated against all six entries of
v5.1 §4.4's published table. `TEST-A7-15c` is what turns all of them into
executed evidence under this document.

**One caveat inherited and not reduced.** Every calibration here is **Waymo-only**
and `train`-only. Five of the six `K3` sub-rules never apply on the procedural
panel, so `K3` there is `offroad` alone and the two sources are graded by
substantially different rulebooks; pooled reporting across sources stays
prohibited (§11.12).

---

## 9. Acceptance criteria

### 9.1 A7's criteria

| ID | criterion | status |
|---|---|---|
| `AC-A7-01` | The channel order is exactly `(collision_safety, interaction_risk, non_relaxable_compliance, negotiable_lane_compliance, mission_progress)`, and no module restates an ordering | `NOT_RUN` — `TEST-A7-01` |
| `AC-A7-02` | The atomic vector still exposes fourteen entries, thirteen sub-rule costs plus `Δq`, and the observation dimension `D` is unchanged | `NOT_RUN` — `TEST-A7-02` |
| `AC-A7-03` | `K4`'s denominator stays 3 when a sub-rule is inapplicable | `NOT_RUN` — `TEST-A7-03` |
| `AC-A7-04` | `K5` carries the bare signed advance, and `Σ_t Δq_t = (s_T − s_0)/D_REF` to numerical tolerance below the clip | `NOT_RUN` — `TEST-A7-04` |
| `AC-A7-05` | No channel, sub-rule, configuration key or observation field named `progress_rate`, `advance_shortfall`, `relaxable_weight` or `progress_rate_weight` survives on the A7 path | `NOT_RUN` — `TEST-A7-06` |
| `AC-A7-06` | A satisfied `K4` contributes exactly zero and a violated one costs `w₅·(1 + σ·c)`, with `w₅` reaching no other contribution | `NOT_RUN` — `TEST-A7-07`, `-08` |
| `AC-A7-07` | The resolved scalarization block is the A7 mode at `(a, σ, λ₄, w₅) = (2.5, 0.30, 2.0, 0.15)` with the A7 schema id | `NOT_RUN` — `TEST-A7-09` |
| `AC-A7-08` | The §5.4 predicate admits that weight set and refuses each of a declared list of inadmissible ones, **at construction** | `NOT_RUN` — `TEST-A7-10` |
| `AC-A7-09` | Every algorithm configuration declares `γ = 0.9982`, every `learning_potential_gamma` equals it, and the criterion's **verdict** `ln(a)/−ln(γ) > L` holds at the `L` read from the frozen index | `NOT_RUN` — `TEST-A7-11` |
| `AC-A7-10` | Every budget `τ₁`–`τ₄` is a value read off the expert per-episode distribution by §4.5's rule, with the excluded records listed | **PASS** — measured 2026-09-10, exclusion list empty and declared (§4.5); `TEST-A7-12` |
| `AC-A7-11` | `fraction_below_standstill` under A7 is **measured** at or below 7.45 % | **PASS** — 4.64 % measured, not projected (§5.5); `TEST-A7-13` |
| `AC-A7-12` | Production equals the offline instrument to `1e-9` on every step of the frozen Waymo `train` panel | `NOT_RUN` — `TEST-A7-14` |
| `AC-A7-13` | The ordering battery holds on the scalar arm, and **each strict-lexicographic failure is reported with the deciding channel** rather than repaired | `NOT_RUN` — `TEST-A7-15`, `-16` |
| `AC-A7-14` | The `AB-LEARN` pre-registration records the reward change explicitly **before** any screening run under A7, and its resolved-config diff still shows single-factor invariance | `NOT_RUN` — `TEST-A7-17` |
| `AC-A7-15` | §11 declares the limitations A7 does not remove, each with its figure | **PASS** — §11 |
| `AC-A7-16` | Two runs whose scalarization vector schemas differ never share one analysis condition identity | `NOT_RUN` — `TEST-A7-22` |
| `AC-A7-17` | The §5.4 predicate is evaluated at a **configured** `ΔQ_MIN`, not a literal; it admits the selected weights at `ΔQ_MIN_GUARANTEED = −0.1525` and refuses them at `ΔQ_MIN_CLIP = −1`; and every episode publishes `k5_below_guaranteed_min_steps` and `k5_min_delta_q`, so the guarantee of §5.4 is falsifiable by measurement rather than assumed | `NOT_RUN` — `TEST-A7-10`, `TEST-A7-23` |

`AC-A7-13` is deliberately permissive in the same way its predecessor was: a
strict-lexicographic failure on O1 and O2 is a **result to report**, not a defect
to repair (§11.1, §11.2). What the criterion forbids is a failure that is not
attributed to the channel that caused it.

### 9.2 Disposition of `RULEBOOK-V5.1`'s acceptance criteria

Two columns, not one. The first states each criterion's **arrival state under
v5.1** — the reconciliation the v5.1 ExecPlan closes without ever having written
— and the second its disposition here. With only the second column this would be
a transition table and v5.1's own criteria would be settled nowhere.

| criterion | arrival state under v5.1 | under A7 |
|---|---|---|
| `AC-RB5.1-01` O1–O6 on the fixtures | `PASS` undiscounted; **O3 fails at `γ = 0.996`** | **Changes.** O3 holds on the scalar arm at both discounts (+0.581 / +2.458). Restated as `AC-A7-13` |
| `AC-RB5.1-02` O1–O6 under strict lex, or each failure reported | `PASS` undiscounted; O1 fails at `L2` and the failure is reported | **Changes.** A7 additionally loses **O2** at `K4`; v5.1's pass there is a zero-traffic artefact (§1.1, §11.2). Restated as `AC-A7-13` |
| `AC-RB5.1-03` expert mean episode return positive | `PASS` — **+70.70** | **Carries.** Re-measured under A7: **+71.34** (§5.5) |
| `AC-RB5.1-04` below standstill ≤ 7.45 % | `PASS` — 3.36 % at the calibration pair, 4.55 % at the shipped one | **Carries as the binding cost** — `AC-A7-11`, measured 4.64 % |
| `AC-RB5.1-05` the clip binds on no step the agent can produce | **`NOT ESTABLISHED`** — the engine-force cap bounds travel, not projection | **Carries as `NOT ESTABLISHED`.** `l4_clip_binding_steps` is what makes it measurable rather than deduced (§4.1, §7) |
| `AC-RB5.1-06` the selected weights satisfy the predicate for every `k` | `PASS` — thinnest margin 1.1121 at `k = 2` | **Changes** to the A7 tail: thinnest **1.1390** at `k = 3`. `AC-A7-08`; margins in §5.4 |
| `AC-RB5.1-07` `Σ Δq` telescopes | `PASS` for agent trajectories; inexact on the expert panel where the clip binds | **Carries.** `AC-A7-04`. The residual is the negative-clip item's territory and A7 does not touch it (§11.5) |
| `AC-RB5.1-08` every atomic cost exposed | `PASS` | **Carries unchanged.** `AC-A7-02` |
| `AC-RB5.1-09` validation and test splits consulted by no calibration | `PASS` | **Carries unchanged.** No calibration in this document consults either |
| `AC-RB5.1-10` `η` reaches only `L5` | `PASS` — p1 and p5 identical across `η ∈ [0, 5]`, which is also why nothing pins `η` | **`NOT_APPLICABLE`** — `η` is deleted. Its pattern survives as `AC-A7-06`'s "`w₅` reaches no other contribution" |
| `AC-RB5.1-11` v5.1 not worse than v5.0 on five columns | `PASS` — dominates on all five | **Carries, re-measured**: mean 71.34 and below-standstill 4.64 % against v5.0's +31.14 and 7.45 % |
| `AC-RB5.1-12` O3 against the reference shortcut in both arms | `PASS` undiscounted, margin +0.2000 decided at `L5`; **fails at the shipped discount** | **Changes** — this is A7's headline gain, and the ordering is decided at `K4` rather than never reached |
| `AC-RB5.1-13` `λ₆` strictly below its O3 bound | `PASS` — 0.2 < 0.25 | **`NOT_APPLICABLE`** — `λ₆` is deleted |
| `AC-RB5.1-14` the predicate admits `(λ₄, η, λ₆)` and rejects an inadmissible `λ₆` | `PASS` — 2.12 < 2.5 after ADR-081 | **Changes** to `(λ₄, w₅)`, same predicate shape. `AC-A7-08` |
| `AC-RB5.1-15` the below-standstill baseline is `−λ₆·(Δt/T_REF)·T` | `PASS` | **`NOT_APPLICABLE`** — with `L6` gone the baseline is exactly 0, which is what makes the count rise (§4.3, §5.5) |
| `AC-RB5.1-16` one shared discount above the horizon | `PASS` **as written**, and the criterion was wrong: it names a 199-step Waymo-only horizon and the measured one is 500 | **Changes**: `γ = 0.9982` above the **measured 500-step** horizon, with the verdict asserted. `AC-A7-09` |
| `AC-RB5.1-17` `L6` refines only ties | `PASS` undiscounted; the premise is itself undiscounted | **`NOT_APPLICABLE`** — `L6` is deleted |

**One arrival state is worth reading twice.** `AC-RB5.1-16` passed while being
false about the quantity it names: the guard asserted a horizon nobody had
measured, under a docstring claiming it was measured. That is why `AC-A7-09`
asserts the criterion's **verdict** and not only its inputs.

---

## 10. Test matrix

Frozen before any production change, as the required workflow demands. `M3` of
the `A7` ExecPlan writes `TEST-A7-01`…`-10` and `-18`…`-19` against the
unmodified tree and records **which error each fails with**, because a test that
fails on an import is not yet testing anything.

| ID | level | behaviour | fixture / input | expected | criterion |
|---|---|---|---|---|---|
| `TEST-A7-01` | Unit | Five channels in the declared order; nothing hard-codes it twice | the order constant and the derived cost-channel tuple | order matches §3; the cost tuple derives from it | `AC-A7-01` |
| `TEST-A7-02` | Unit | The atomic vector is unchanged | synthetic component results | fourteen entries, thirteen sub-rule costs plus `Δq` | `AC-A7-02` |
| `TEST-A7-03` | Unit | `K4` fixed denominator | one applicable sub-rule, two not | `c/3`, not `c/1` | `AC-A7-03` |
| `TEST-A7-04` | Unit | `K5` signed and clipped; `Σ Δq` telescopes below the clip | station sequence | `(s_T − s_0)/D_REF`; a stretch covered forward and back nets exactly zero | `AC-A7-04` |
| `TEST-A7-05` | Unit | **The progress index is 4 of five, and the range check follows it** | `(0,0,0,−1,+1)` accepted; `(0,0,0,+1,0)` rejected at index 3 | first passes; second raises the scalarization evaluation error | `AC-A7-04` |
| `TEST-A7-06` | Unit | Nothing named `progress_rate` / `advance_shortfall` survives on the A7 path | registry and configuration allow-list | lookup error / configuration error | `AC-A7-05` |
| `TEST-A7-07` | Unit | The §5.1 form is reproduced term by term, **written out independently of the implementation** | five-entry channel vectors, six cases | exact | `AC-A7-06` |
| `TEST-A7-08` | Unit | `w₅` reaches only the `K4` term | one vector, two `w₅` values | priority contributions identical; reward differs by `Δw₅·[(1_sat − 1) + σ·m₄]` | `AC-A7-06` |
| `TEST-A7-09` | Config | The resolved scalarization block is the A7 set | resolved job configuration | exact values and schema id | `AC-A7-07` |
| `TEST-A7-10` | Unit | The §5.4 predicate is a **constructor** gate, evaluated at a configured `ΔQ_MIN` | the selected set at `ΔQ_MIN_GUARANTEED` and at `ΔQ_MIN_CLIP`, plus five inadmissible sets | admits at −0.1525; **raises at −1**, naming rank preservation; ratios 1.1261 / 1.0870 / 1.0000 and 1.0035 / 0.8395 / 0.5959 | `AC-A7-08`, `AC-A7-17` |
| `TEST-A7-23` | Integration | The guarantee's falsifier reaches the per-episode record, signed | an episode with one step at `Δq = −0.2` and one at `Δq = −0.05` | `k5_below_guaranteed_min_steps == 1`, `k5_min_delta_q == −0.2`; the sign-blind clip counter is unchanged | `AC-A7-17` |
| `TEST-A7-11` | Integration | One shared discount, its shaping twin, **and the criterion's verdict** | the six algorithm configurations and the frozen index | `γ = 0.9982` everywhere; `508.6 > 500` | `AC-A7-09` |
| `TEST-A7-12` | Measurement | The expert per-episode exposure distributions exist and fix the budgets | the 2026-09-10 panel run | four distributions on three objects, plus per-record tail rows | `AC-A7-10` |
| `TEST-A7-13` | Measurement | Below-standstill under A7 | the A7 grid member at the selected weights | ≤ 7.45 % | `AC-A7-11` |
| `TEST-A7-14` | Integration | Production versus the offline instrument, per step | 20 frozen records, then the full 1100 | equal to `1e-9` | `AC-A7-12` |
| `TEST-A7-15` | Regression | The ordering battery under A7, scalar arm, **at both weight pairs** | the retargeted O1–O6 fixtures | pass, O3 included | `AC-A7-13` |
| `TEST-A7-16` | Regression | Strict-lexicographic failures are reported with the deciding channel | the same fixtures | O1 names `K2`; O2 names `K4` | `AC-A7-13` |
| `TEST-A7-17` | Config | `AB-LEARN` single-factor invariance survives | resolved-config diff of the two arms | differences confined to the reward group and run identity | `AC-A7-14` |
| `TEST-A7-18` | Property | Every cost channel in `[0,1]`, `K5` in `[−1,+1]`, on a random grid | random channel vectors | invariant holds | `AC-A7-01` |
| `TEST-A7-19` | Numerical | Sub-tolerance margins clamp to exactly zero and the indicator does not fire | `−1e-9` and `−1e-6` at `K3` | clamped case earns `λ₄`; `−1e-6` does not | `AC-A7-06` |
| `TEST-A7-20` | Smoke | End-to-end training under the A7 reward | the repository smoke target | exit 0, no NaN | all |
| `TEST-A7-21` | Regression | The full suite is green and no test is skipped, weakened or xfailed | the merge gate | `PASS · FULL` | all |
| `TEST-A7-22` | Regression | Two runs whose scalarization vector schemas differ never share a condition identity | two synthetic run metadata records differing only in the schema id | distinct identities | `AC-A7-16` |

### 10.1 The O1–O6 fixtures

**Constructed, not replayed**, and written as per-step channel vectors rather
than as scenes: whether a given geometry yields `c_solid_line = 0.4` is the
sub-rule evidence's business, while these assert the *hierarchy* — which channel
a cost lands in and what that placement implies for the ordering. Both comparison
rules are exercised wherever they differ.

| ID | fixture | what it must show under A7 |
|---|---|---|
| `TEST-A7-15a` | O1: legal completion vs standstill | scalar arm prefers completion; **strict lex prefers the standstill and the failure names `K2`** |
| `TEST-A7-15b` | O2: completion needing a brief relaxation vs standstill | scalar arm prefers completion through the finite `K4`/`K5` exchange; **strict lex prefers the standstill and the failure names `K4`** |
| `TEST-A7-15c` | O3: legal route vs the §5.5 reference shortcut, both completing | both arms prefer the legal route, **decided at `K4`**, at `γ = 0.996` **and** `γ = 0.9982` — the ordering this restructure exists for |
| `TEST-A7-15d` | O4: lane relaxation vs at-fault collision | decided at `K1`, in both arms |
| `TEST-A7-15e` | O5: waiting at a red vs running it to finish | decided at `K3`, in both arms |
| `TEST-A7-15f` | O6: necessary vs gratuitous relaxation, equal completion | decided at `K4`; the gratuitous relaxation buys no progress |
| `TEST-A7-15g` | the interaction indicator makes continuous violation lose to standing still | a `K2` violation on every step costs `a²·(1+σ)` per step whatever the severity, so 40 such steps lose to standing still — **correctly**, and it is why the measured **0.3978 %** firing rate matters more than the severity when it fires |

**Two properties of these fixtures are themselves requirements.**

1. **They are parametrised over both weight pairs** — the shipped
   `(a, σ) = (2.5, 0.30)` and the historical `(2.2, 0)` the previous fixtures
   asserted at — because a fixture asserting at a weight pair production has not
   used since 2026-09-07 verifies nothing about production. This closes a
   verification gap that was open, not a behavioural one: every ordering was
   checked to hold at the shipped pair, O3 excepted.
2. **The discount is read from the configuration, not hardcoded.** A fixture that
   restates `γ` is a second source of a value this document makes single.

**One fixture of `RULEBOOK-V5.1` §10 has no A7 counterpart and is deleted rather
than weakened**: the test asserting that `λ₆` moves no contribution other than the
last. Its *pattern* — a weight reaches exactly one channel — survives as
`TEST-A7-08` on `w₅`. Deleting it is an approved mandatory-test change
(`A7` `DEC-A7-006`), not a silent removal.

---

## 11. Known limitations

**The list below is renumbered 1–15 and its numbers do not correspond to
`RULEBOOK-V5.1` §11's.** Carry-over is therefore by *subject*, never by number:
every v5.1 limitation stays live unless an item below names it and replaces it,
and v5.1 §11 has two items numbered 12, which makes a by-number mapping
undefined in any case. What is **not** restated below and stays live includes the
achievable-success ceiling of the frozen panels and the caveat that the reference
shortcut is a stipulated construction rather than a measured trajectory — that
second one is load-bearing for §5.5 and is named here so it cannot lapse. Each
item states its figure, because a limitation without a magnitude is an apology
rather than a declaration.

1. **Strict lexicographic ordering still prefers standing still on O1, and no
   hierarchy can avoid it.** Standing still is exactly `(0,0,0,0,0)`, so under a
   strict comparison it wins against any trajectory that accrues even ε of
   interaction cost — which every real trajectory in traffic does, at a measured
   **0.3978 %** of expert steps. A **thresholded** comparison passes O1, which is
   what makes `τ₂ > 0` load-bearing rather than a tolerance. Architecture-
   independent.
2. **A7 additionally loses O2 under strict lexicographic comparison**, at `K4`:
   the standstill has `c_K4 = 0` exactly and wins before progress is compared.
   The qualification is measured and is stated with the loss: v5.1's pass on O2
   is a zero-traffic artefact, and one interaction step in forty at the 0.05
   residual this repository's own ordering test uses is enough to make v5.1 fail
   it too. Once that residual is present, A7 is no worse on any ordering and
   strictly better on O3. Reported by `TEST-A7-16`, not repaired.
3. **No admissible `w₅` opposes a fast off-corridor drive, by a factor of
   10.26** (§5.5). At `w₅ = 0.15` a fully violated `K4` step costs **0.195**
   against **2.000** of progress at the clip, so it opposes nothing at any pace,
   and the ratio `λ₄/[a − λ₄·(ΔQ_MAX − ΔQ_MIN)]` is independent of `σ`. Earlier
   revisions reported this as "exactly 4" because they evaluated it at
   `ΔQ_MIN = 0`; correcting §5.4 does not create the limitation, it reveals that
   it was understated 2.6-fold. Equivalently, progress consumes **92.2 %** of the
   `k = 3` per-step budget. The only lever
   that would move it is `λ₄` relative to `a`, and reducing `λ₄` to the value
   that admits such a `w₅` costs **+1.81 pp** of below-standstill while buying an
   option that **cannot be exercised**: the channel that would have to fire is
   `K4`, and an ego on a legal parallel carriageway violates none of its three
   sub-rules.
4. **A legal parallel corridor is not excluded, and this document does not close
   the remedy space.** Measured over all 3,500 frozen records at 1 m station
   spacing: **318 (9.1 %)** have same-direction drivable surface outside the
   route's own carriageway, at least one ego width wide, reachable across at most
   one ego width of non-drivable surface, from which the ego misses the final
   gate; median length 15.0 m at a median offset of 10.70 m. Read in metres
   rather than as a share of route, the honest headline is **6 records (0.17 %)**
   carrying a corridor worth a whole mean mission. The mechanism is junction
   geometry: 199 of the 318 are intersections.

   **And the finding is about a specification, not about A7.** The remedy wants
   to sit *above* progress, and A7 is the architecture that puts it there: under
   the superseded order an off-corridor rule's natural home was `L5`, *below*
   progress, so the off-route trajectory banked the larger progress total and
   `L5` was never consulted; under A7 the same channel is `K4`, above `K5`, where
   an in-corridor trajectory has `c_K4 = 0` exactly and wins before progress is
   compared. What forecloses the remedy is that
   `driving_mission_v1.1_specification.md:104` withdraws **three** remedy shapes
   in one sentence — continuity/clamp/freeze/recovery protocols, off-route
   progress zeroing, and the runtime authority of any final lateral envelope. By
   `AGENTS.md`'s Scientific Argument Standards that is a finding, and usually the
   specification is the thing to fix. Unforeclosing any of the three is the
   user's decision and is not this document's.
5. **The negative clip is an unbounded ratchet, and A7 does not change it.** A
   closed loop over a hairpin whose legs are one lane apart pays **+36 channel
   units = +72 reward units per lap at zero net displacement**, linear in laps,
   executed against the real route polyline. Decided: change nothing in the
   reward. The position rests on the frozen population not admitting that
   geometry within 5 m of lateral reach, and nothing in the runtime bounds the
   reach. The same evidence records an on-route under-charge of **max +128.751
   with 1601 of 3500 routes positive**. **Under A7 this matters more than
   before**, because in the thresholded regime `K5` is the only gradient left
   inside budget.
6. **No credit without motion is violated, and A7 makes it visible rather than
   causing it.** One clipped route-projection jump with no motion pays **+1.345**
   under A7 against a standing-still baseline of exactly **0.000**; under the
   six-level reward the same probe's *gap* is +1.358 but the baseline is
   **−2.757**, so the exploit was a relative gain there and is an absolute one
   here. Deleting `L6` removes the negative baseline that was masking it. It is
   bounded by item 5.
7. **Collide-to-escape is a scalar-arm result, not a defect.** Enduring 200 steps
   of interaction violation at `c = 0.9` against colliding at fault on step 20:
   the scalar arm **prefers colliding by 1092.8** (−1265.96 against −173.14),
   while the ordered arms compare `K1` first, where the non-colliding trajectory
   is **0.0000** against **0.5788** and wins at any `τ₁ < 0.5788`. No bounded
   scalar sum can do better, and this is the mirror image of the case where a
   scalarization is structurally better than any ordered arm (§5.1). It belongs in
   the write-up as a result.
8. **Mission success is a zero-value terminal**, so completing rather than
   stopping short is worth **0.57–0.77** reward units under A7 against **8.125**
   for one fully violated interaction step. A7's incentive is smaller in magnitude
   and better in kind than its predecessor's — proportional to remaining mission
   *distance*, a property of the task, rather than to the remaining length of the
   logged episode — and a terminal bonus is **not** recommended (§4.2).
9. **O3 as a dominance is unobtainable.** No Markov, bounded, per-step progress
   channel has a duration-invariant discounted return, so the ordering must be
   stated as a **finite exchange rate**. A7 satisfies it as an exchange rate on
   one weight; it does not restore the undiscounted identity, and no hierarchy in
   any order at any channel count can.
10. **The thresholded arm's mechanism is open**, and `D1`'s choice of object is
    downstream of it (§4.5). Both routes carry a declared gap: the policy-gradient
    route compares an accumulated episodic return against the threshold but
    accumulates it **undiscounted from a single sampled episode** while its
    objective is the initial-state value; the state-augmentation route is proved
    for **one** constrained channel and its authors state that extending it to
    several is "not straightforward … we need to know which constraints can be
    satisfied together", and it needs the accumulated exposure in the policy's
    input, i.e. an observation amendment. Also on the ledger: finding an optimal
    *deterministic* policy for a lexicographic MDP is NP-hard (Pineda, Wray &
    Zilberstein, Lemma 1). **A7 reduces the number of constrained channels from
    five to four and removes the one that could not be thresholded at all**
    (§4.2), which is a reduction of the problem and not a solution to it.
11. **Per-state Q-value thresholds are the wrong mechanism here, measured rather
    than argued.** Vamplew et al. §7.2, pp. 75–76 report thresholded lexicographic
    Q-learning performing "extremely poorly when the time objective is
    thresholded", because its action selection "considers only the expected future
    reward … ignoring any rewards received earlier in the current episode",
    failing "regardless of the value of the threshold"; and Pineda et al. measured
    that lexicographic value iteration with the per-state slack its own bound
    prescribes "failed to make any significant change in costs, with respect to
    using no slack".
12. **`K3`'s evidence is Waymo-only.** Five of the six `K3` sub-rules never apply
    on the procedural panel, so `K3` there is `offroad` alone. Pooled reporting
    across sources stays prohibited. Two sub-rules — `crosswalk` and
    `speed_limit` — **never win `K3`'s `max`** on the Waymo panel at any
    frequency, which bounds which sub-rules the `K3 ≻ K4` choice can ever be
    decided by on this evidence; `offroad` wins it on 1285 steps (0.5917 %),
    `vehicle_yield` 55, `signal` 38, `stop` 16.
13. **`K2 ≻ K3` is unsupported by any mechanism**, and its co-occurrence is
    measured rather than derived (§3.6). This is the limitation to attack first.
14. **The budgets are the loosest falsifiable values.** Because `τ_i` is the
    panel maximum, a policy at or below human exposure is unconstrained on that
    channel, and the thresholded arm's differentiation there comes from the
    ordering alone (§4.5). `τ₂`'s maximum is 13.7× its own `p99`.
15. **§5.4's guarantee is conditional, and the condition is not proved — it is
    monitored.** The per-step ordering is lexicographic while the compliant
    trajectory loses no more than 0.339 m of station per step; at the full clip
    the selected weights hold only at `K1`. Nothing in the runtime bounds the
    backward projection — the production mission tracker projects without a jump
    envelope — so the condition rests on measurement (0.053 m on the expert
    panel, 6.4× of margin) and on reachability (two actions in one state differ
    by 0.025–0.045 of `Δq`), not on a proof. `AC-A7-17` is what turns it from an
    assumption into a checked invariant, and §14.3 pre-registers what a non-zero
    count means.
16. **A7's gain on O3 does not reach the thresholded arm at these budgets.** The
    reference pair accrues 10.0 units of `K4` exposure against `τ₄ = 23.386514`,
    so it ties on every cost channel and the comparison falls through to progress,
    where the shortcut wins by arriving sooner; the shortcut would need 70 of its
    160 steps on a marking to leave the budget. The gain is a property of the
    scalar and strict-lexicographic arms. This is the same fact as item 14 read
    against the one ordering the restructure was for, and it is declared because
    §11.1 invokes the thresholded comparison where it favours A7 (O1); the two
    statements have to be made in the same voice.

---

## 12. References

`RULEBOOK-V5.0` §12 refs. 1–16 carry over unchanged and are the references for
every inherited sub-rule and for the thresholded-lexicographic literature. Five
are load-bearing for *this* document and are restated with what they are used
for; two are added.

- **Censi, Slutsky, Wongpiromsarn, Yershov, Pendleton, Fu, Frazzoli (2019).**
  *Liability, Ethics, and Culture-Aware Behavior Specification using Rulebooks.*
  ICRA. — Fig. 10 places own progress at the bottom of the rulebook and
  Definition 17 adds new rules there; the basis for §1's position of `K5`.
- **Tumova, Reyes Castro, Karaman, Frazzoli, Rus (2013).** *Minimum-violation LTL
  Planning with Conflicting Specifications.* ACC. arXiv:1303.3679. — the
  negotiable/non-negotiable semantics of `K3 ≻ K4` (§3.6).
- **Veer, Leung, Cosner, Chen, Pavone (2023).** *Receding Horizon Planning with
  Rule Hierarchies for Autonomous Vehicles.* ICRA. arXiv:2212.03323. — Theorem
  1's rank-preserving reward, the structure §5.4 restates, and the `1/N`
  tie-breaker whose transplant into this schema is what §5.5 withdraws.
- **Ng, Harada, Russell (1999).** *Policy Invariance Under Reward
  Transformations.* ICML. — potential-based shaping is policy-invariant only when
  its discount is the MDP's; the basis for coupling
  `learning_potential_gamma` to `γ` (§4.4).
- **Tercan, Prabhu (2024).** *Thresholded Lexicographic Ordered Multiobjective
  Reinforcement Learning.* ECAI. arXiv:2408.13493. — §3 for the unthresholded
  last objective; Appendix D.3 for the discounted threshold depending on the
  trajectory (§4.5); Appendix D.3.3 for state augmentation with several
  constrained channels being "not straightforward" (§11.10).
- **Added: Vamplew, Dazeley, Berry, Issabekov, Dekker (2011).** *Empirical
  evaluation methods for multiobjective reinforcement learning algorithms.*
  Machine Learning 84. — §3.2.3, p. 58: "objective n will be unconstrained,
  hence `C_n = +∞`", the requirement §1 and §4.2 rest on; §7.2, pp. 75–76: the
  measured failure of thresholding a time objective (§11.11).
- **Added: Pineda, Wray, Zilberstein.** — Lemma 1: finding an optimal
  deterministic policy for a lexicographic MDP is NP-hard; and the measured null
  result of per-state slack (§11.10, §11.11). *This citation is incomplete: the
  full bibliographic entry is not recorded anywhere in this repository and must
  be completed before approval.* It is carried here rather than dropped because
  the two results it supplies are load-bearing for §11.

---

## 13. Out of scope

`RULEBOOK-V5.1` §13 carries over — comfort and jerk remain excluded from the
rulebook and from the reward, and may be logged as diagnostics only — plus:

- **The thresholded-lexicographic algorithm and the mechanism that enforces a
  budget.** This document fixes budget **values** and declares the requirement on
  the undiscounted realized form; **which object a threshold is enforced on is
  `D1`'s** (§4.5), and an episodic budget has no per-state equivalent.
- **The distributional arm.**
- **The negative-clip ratchet** and the **parallel-corridor exposure**: decided,
  and carried as §11.5 and §11.4.
- **Retraining, algorithm selection, observation or encoder changes**, and any
  decision about GPU time.
- **Any change to a sub-rule**: definition, geometry, tolerance, applicability
  gate or at-fault classification.

---

## 14. Open decisions, and the derivation record

### 14.1 Open

Three of these were open by design before this document was written. **Six more
were found by an independent audit of it on 2026-09-11** and are listed beside
them, because an `UNDER_REVIEW` document that hides them is worse than one that
does not. **Question 1 was the blocking one and was resolved on 2026-09-12**;
the remaining five do not block approval, but two of them (questions 2 and 4)
support the value of a weight and should close before `M4`.

| # | question | status, and what each answer changes |
|---|---|---|
| ~~1~~ | ~~Does the §5.4 predicate count the progress swing once or twice?~~ | **RESOLVED 2026-09-12**, user-approved. Neither: the swing is generalised to `λ₄·(ΔQ_MAX − ΔQ_MIN)` with `ΔQ_MIN` a declared symbol taking two values (§5.4), §5.4 is restated as a **conditional guarantee with a runtime falsifier** (`AC-A7-17`, `TEST-A7-23`), and no weight moves. The alternatives and their prices are kept in §14.3 as the record of why |
| 2 | What model converts 5.0 s of arrival delay into reward units — i.e. where do `w₅ < 0.4477` and `w₅ < 0.2641` come from (§5.5)? | Not reproducible from §4 or §5; six candidate models were tried and none produces the pair, and the required `γ`-dependence lies outside the natural families. `0.2641` is the bound that constrains `w₅` at the adopted discount and is the denominator of the 72.1 % window, so this is not decorative. If it cannot be restated, it joins §14.2's not-reproduced list and the window figures are withdrawn or re-derived |
| 3 | Which two quantities give the discount's **90.5 % / 79.7 %** share of the time preference (§4.3)? | Not reproducible; the only live version of that comparison (v5.1 §4.6, +2.05 against 0.8) gives **71.9 %** at `γ = 0.996`, and no function with a fixed denominator spans 90.5 → 79.7 across the two discounts. §4.3 now carries three reproducible grounds plus this one flagged, and the deletion of `L6` does not rest on it |
| 4 | Is the marking placed on the shortcut's **first** 30 steps in the §5.5 reference pair? | Answered affirmatively here, on evidence: it is the only placement at which v5.1 §4.4's whole published table reproduces (§14.2). Recorded because the `γ = 0.996` margin goes **+0.5813 → −0.0141** once the placement starts past step 33, so the "stands whether or not the discount decision is taken" claim is placement-dependent at that discount and unconditional at `γ = 0.9982`. If the placement is meant to be free, the binding lower bound becomes `w₅ > 0.2212`, which `w₅ = 0.15` does not satisfy |
| 5 | Do the recorded per-episode columns `l4_clip_binding_steps` and `l5_reached_steps` keep those literal names under A7 (§7)? | "Retained under the `K5` name" admits both readings. Keeping them costs nothing; renaming them adds three source files and four test files to the implementation and needs an acceptance criterion that none currently supplies |
| 6 | Does `REQ-A7-10`'s "denominated per unit of mission span" survive, given that the approved rule and §4.5 declare the requirement on the **undiscounted realized** form? | The two readings give different numbers (`τ₂` = 110.379799 realized against 0.720226 per-span). §4.5 follows the approved budget rule; the ExecPlan requirement row still says per-span and should be reconciled to it |
| `D1` | Which of the three objects of §4.5 a threshold is enforced on, and by which mechanism | Open by design. The three objects do not coincide, and choosing one here would decide the mechanism by choosing a number's units. All three values are measured so the eventual choice costs no second run. **Index collision worth recording**: v5.1 §4.6's `τ₄` is the *progress* budget, while this document's `τ₄` is the negotiable-lane budget and there is no `τ₅` (§4.2) — `D1`'s subject moved with the renumbering |
| `K2 ≻ K3` | Whether the adjacency is right | No mechanism exists in either direction; reordering on no evidence would replace one unsupported claim with another. Declared as §3.6 and §11.13, with the co-occurrence measured |
| `AC-RB5.1-05` | Whether the `K5` clip binds on a step an agent can produce | The engine-force cap bounds travel, not projection. Now measurable per episode rather than deducible (§4.1, §7) |

### 14.2 What was reproduced for this document, and what was not

Every quantity below was recomputed from the definitions in §4 and §5 by a
standalone script that imports nothing from this repository, so that an oracle
cannot agree with the thing it checks. The two figures that are **not**
reproduced are named, not omitted.

| quantity | reproduced value | verdict |
|---|---|---|
| §5.4 margins, `SCAL-V1.5` at `φ = 0` | 1.1514 / 1.1478 / 1.1390, thinnest at `k = 3` | agrees |
| §5.4 margins, `φ = 0.25` retained | 1.1105 / 1.0975 / 1.1390, thinnest at `k = 2` | agrees |
| §5.4 margins, `SCAL-V1.4` today | 1.1165 / 1.1121 / 1.1792 | agrees |
| The transcription trap (`w₅` as an ordinary lower level) | 1.0911 / 1.0513 / 1.0225, thinnest at `k = 3` | reproduces the **wrong** figures exactly, which is what makes the right ones checked against a known-wrong alternative |
| `w₅` cap, crossovers, the factor 4, `λ₄ ≤ a/2` | 0.384615 / 0.313983 / 1.538462 / exactly 4.0 / 1.25 | agrees |
| Per-step costs: fully violated `K4`, fully violated `K2` | 0.195; 8.125 at `φ = 0` and 8.375 at `φ = 0.25` | agrees |
| The discount table of §4.4 and the required `γ` | 228.6 / 0.1348 / 250 and 508.6 / 0.4062 / 555.6; `γ ≥ 0.998169` | agrees |
| `λ₄ ∈ {1.9, 1.25}` admissible under the A7 tail | 1.1600 / 1.1693 / 1.1933 and 1.2188 / 1.3312 / 1.7301 | agrees |
| Standing still versus relaxing (§3.5) | `N* = λ₄Q/(D_REF·w₅(1+σ))`: 416 steps at `Q ≈ 90 m`, 848 at `Q ≈ 184 m`, against 37 at v5.0's per-step price | agrees, **with the two spans made explicit**: the `A7` ExecPlan quotes 416 and 848 for "a mean Waymo mission" and "a mean PG mission" without stating either span, and the spans above are what its own formula requires them to be |
| The exchange rate of `w₅ = 0.15` in physical units | **1.83 m/s** at `c_K4 = 1/3` and **2.17 m/s** at `c_K4 = 1` | agrees. **An earlier revision of this row was wrong and is corrected here**: it recorded that the `A7` ExecPlan §6.1's "1.83 m/s" did not reproduce. It reproduces exactly, at `c_K4 = 1/3` — one marking, the severity of the reference construction — while 2.17 is the fully-violated-channel figure. The error was attaching the `c = 1` figure to the one-marking sentence; §5.5 now states both severities with the formula |
| `τ₁`–`τ₄` and every exposure figure of §4.5 | read back from `a7_m1_summary.json` and equal to the values published in the `A7` ExecPlan §6.4 to six decimals | agrees |
| The `K2`/`K3` co-occurrence of §3.6 | read back from `a7_m1_summary.json`: 15 / 217,189, 5 / 1100, p50 0.2197 and 0.025313, correlation −0.0161 | agrees |
| The `fraction_below_standstill` grid of §5.5 | read back from `a7_m1_summary.json` for all six members | agrees |
| O3's scalar margins at both discounts, and the marking-placement sensitivity | **+0.5813** at `γ = 0.996` and **+2.4579** at `γ = 0.9982`, marking on the shortcut's first 30 steps; +0.0023 at offset 33 and **−0.0141** at offset 34 for `γ = 0.996`, +1.4509 at offset 130 for `γ = 0.9982` | agrees with §1.1's +0.581 / +2.458. **An earlier revision of this row recorded them as "not reproduced"; they do reproduce.** The reconstruction is validated against a figure it was not built from: under the v5.1 form it reproduces **all six entries** of v5.1 §4.4's published margin table (+0.2000 / −1.1396 / −2.9784 / −3.5789 / −4.0198 / −4.7416) to four decimals, and only with the marking on the first 30 steps |
| `w₅`'s lower bounds from the reference pair | **0.131342** at `γ = 0.996` and **0.073555** at `γ = 0.9982` | agrees with §5.5's 0.1313 / 0.0736. **Also previously recorded as "not reproduced", also wrong**: the pair is fully specified by v5.1 §4.6 plus the first-30-steps placement, so the bound is a bisection on the same reconstruction. At a free placement the binding bound would instead be `w₅ > 0.2212` (§14.1, question 4). The **upper** bounds 0.4477 / 0.2641 remain unreproduced — §14.1, question 2 |

### 14.3 The predicate: options priced, and the reading pre-registered

The first row is the **adopted** option (2026-09-12, user-approved); the rest are
kept as the record of what it was chosen against. Every option was worked out
against the code and the committed artifacts, and the prices are measured
wherever a measurement exists.

| option | keeps the shipped weights? | price |
|---|---|---|
| **ADOPTED — generalised swing, `ΔQ_MIN` declared with two values, §5.4 restated as a conditional guarantee plus a runtime falsifier** | **yes** | The specification loses an unconditional theorem and gains a declared condition with a physical reading (0.339 m of station per step) plus a per-episode counter. No weight moves, no measured figure decays, no run is needed. Editorial cost, all of it paid in this revision: §4.1 (both bounds declared), §5.4, §5.5, §11.3 and `ADR-083`, since the `w₅` cap and the clip-pace ratio become functions of `ΔQ_MIN` — the ratio was reported as "exactly 4" and is **10.26** at the guarantee, so the §11.3 limitation was understated 2.6-fold. Implementation cost: `ΔQ_MIN` becomes a configured field of the scalarization block, and the diagnostic path gains two per-episode fields |
| Generalised swing with `ΔQ_MIN` taken from the expert panel (−0.0239) | yes, with the thinnest ratio at 1.1147 — better than today's 1.1121 | **Rejected**: it reads one side of one clip expression as construction and the other as measurement, inside the same inequality. The same artifact that yields 0.053 m backwards also records the *positive* side of that clip binding on 1298 of 217,189 steps at up to 3.609 m — so the projection demonstrably outruns the vehicle, and 0.053 m is a property of a human who took every fold the right way round, not of the mechanism |
| Adopt the two-sided form and recalibrate now | **no** — needs `λ₄ < 1.1525` at `a = 2.5` | Buys the unconditional theorem for roughly half the expert return: `λ₄ = 1.25` is measured at 6.45 % below-standstill and a mean of 41.12 against 71.34, and `λ₄ ≈ 1.15` extrapolates to ≈6.8 % against the 7.45 % ceiling. Available at any time as the fallback; not something to pay before knowing it is needed |
| Raise `a` and keep `λ₄ = 2.0` | no | Needs `a ≥ 4.195`, a 68 % move in the quantity ADR-081 calibrated, which already measured and rejected `a = 3.0` |
| Asymmetric clip `Δq ∈ [−δ, +1]` | yes | Makes `ΔQ_MIN` true by construction and opens a worse exploit: it breaks ADR-073's forward/backward compensation, so an oscillating policy banks `+1` forward and pays only `−δ` back, and it deepens the §11.5 loop |
| Restrict the predicate to equal-progress pairs (`λ₄` drops out) | yes, with comfortable ratios | Declares out of scope exactly the trade — safety against advance — that the reward exists to price |
| Quantify over two actions available in one state | yes | The right intuition about *reachability* (two actions differ by 0.025–0.045 of `Δq` in one 0.1 s step, so no policy can *choose* to lose 0.34 m) but the wrong predicate: restricting to one state also shrinks the cost side, so the derivation's supremum is no longer attained and no closed form results. It belongs in the *argument* for the assumption, not in the condition |

**The reading, pre-registered before the number exists.** The runtime counter
reports, per episode, the number of steps with `Δq ≤ −0.1525` and the fraction of
episodes carrying at least one. On the first evaluation panel run under A7:

- **zero steps** → the declared condition is corroborated on a trained policy;
  §5.4 keeps the conditional form, `λ₄ = 2.0` stands, and the counter becomes a
  permanent regression;
- **non-zero but under 0.01 % of steps and confined to under 1 % of episodes** →
  tail geometry rather than behaviour: record it as a limitation naming the
  records, keep `λ₄`, and §11.5 becomes the primary defect ahead of §5.4;
- **at or above 0.01 % of steps, or present on over 1 % of episodes** → the
  assumption is false in practice, and the fallback is recalibration to
  `λ₄ ≤ 1.1525`, whose price is the third row above.

The 0.01 % threshold is chosen before the number and has a referent: the
*positive* side of the same clip binds on **0.60 %** of expert steps, so a
backward rate sixty times rarer is distinguishable from it rather than
confusable with it.

### 14.4 Two identifiers this document adds

`AC-A7-16` and the fixture decomposition `TEST-A7-15a`…`-15g` are introduced
here. The `A7` ExecPlan §9.2 assigns `TEST-A7-22` to a requirement that does not
cover it, so the pooling guard had no acceptance criterion to be reconciled
against; `AC-A7-16` is that criterion. The fixture decomposition follows the
convention `RULEBOOK-V5.1` §10 already uses for sub-cases.
