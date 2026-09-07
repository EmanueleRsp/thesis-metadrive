# Specification: Liveness-aware six-level rulebook and hybrid scalarization

## Metadata

- Feature: `rulebook_v2_falsified_redesign`
- Specification ID: `RULEBOOK-V5.1`
- Version: `5.1`
- Status: `APPROVED`
- Date: `2026-08-12`
- Approved: `2026-08-14`
- Supersedes, effective on approval 2026-08-14:
  - `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md` **in full**.
    v5.0 was never approved and never became an implementation contract; this
    document replaces it rather than amending it, because the change is to the
    macro hierarchy itself and a reader must not have to hold two documents open
    to know which channel a sub-rule belongs to. v5.0 keeps its `_UNDER_REVIEW`
    filename, which is historically accurate — it never was approved — and is
    retained as the **evidence of record** for the per-sub-rule Test A / Test B
    tables this document inherits unchanged.
  - everything v5.0 listed as superseded: `rulebook_v4.7_specification.md` §6.4
    and §7.1–§7.5, `rulebook_v4.13_specification.md` for the `wrongway` subject,
    and `rulebook_scalarization_v1.1_specification.md` §7.6.
  - additionally, and unlike v5.0: v4.7's **macro-rule grouping**, its **`max`
    aggregation** and its **R4 progress definition**, all three of which v5.0
    left authoritative and this document redefines.
- Inherits unchanged from v5.0, which remains the evidence of record for them:
  every sub-rule definition (§5), both falsification tests (§2), the measurement
  instrument (§3), and the Test A / Test B evidence per sub-rule (§4.2–§4.9).
  Where this document cites a v5.0 figure it says so.
- Related ADRs: `ADR-063` .. `ADR-069` carry over unchanged; `ADR-070` (at-fault
  gate) and `ADR-071` (R1 at-fault classification, **approved** 2026-08-12) are
  incorporated into §3. New with this document: `ADR-072` (five-level
  hierarchy), `ADR-073` (signed `L4`), `ADR-074` (`SCAL-V1.3`), `ADR-075`
  (`γ = 1`) and `ADR-076` (`L6 progress_rate`, and `SCAL-V1.4`).
- Related specifications: `evaluation_protocol_v1.3_specification.md`,
  `observation_v1.3_specification.md`, `observation_lidar_v2.0.1_amendment.md`.
- Measurement evidence: re-measured on the same 1100 Waymo `train` records,
  217,189 transitions, **0 skipped, 0 errors**
  (`scripts/measure_expert_rulebook_transition.py`). Sub-rule figures are
  inherited from v5.0; every macro-channel and return figure is re-derived here
  (§5.5, §8). The O1–O6 orderings are established separately by the constructed
  fixtures of §10 (`tests/test_rulebook_v51_orderings.py`), which replay cannot
  supply because it holds no counterfactual.
- Approval evidence: explicit user approval 2026-08-14 ("approvo la v5.1"),
  after §5.5's full-panel measurement and the O1–O6 fixtures of §10 were both
  reported, and after the three normative placement decisions of §3.5 and the
  two open requirements (`REQ-RB5.1-GAMMA`, `REQ-RB5.1-OBS-01`) were stated as
  unresolved. **Amended 2026-08-14**, under explicit approval the same day: L4
  becomes the *signed* advance (ADR-073), which withdraws `REQ-RB5.1-OBS-01` by
  removing its subject and leaves the measured returns unchanged (+73.85 ->
  +73.84).
- **Amended 2026-08-20**, under explicit approval the same day: `γ = 1` on every
  channel and every arm (ADR-075), and a **sixth level `L6 progress_rate` below
  relaxable lane compliance** (ADR-076), which makes the scalarization
  `SCAL-V1.4`. The two are not severable — `γ = 1` removes all time pressure and
  L6 restores it at a level where it cannot pay for a lane violation. This
  amendment **resolves `REQ-RB5.1-GAMMA`** and withdraws two claims §4.4
  previously made, both falsified by measurement. An intermediate proposal
  placing the same quantity *inside* L4 was implemented, measured and rejected;
  ADR-076 records why.
- Authoritative: `YES` for §3 (the hierarchy and its aggregation), §4 (L4, its
  L6 and the discount), §5 (`SCAL-V1.4` and its weights), §9 and §10.
  **No open requirement remains**: `REQ-RB5.1-OBS-01` was withdrawn on
  2026-08-14 for want of a subject, and `REQ-RB5.1-GAMMA` was resolved on
  2026-08-20 by ADR-075.

---

## 1. Purpose

v5.0 falsified the rulebook against the logged human expert and rebuilt every
sub-rule that failed. It succeeded at that, and it recorded its own failure
honestly in its limitation 8: **the resulting reward still makes standing still
preferable to completing the mission.**

v5.0 attributed that failure to strict priority as such, and deferred the remedy
to the learning algorithm. That attribution was wrong, and its own §6.5 already
contained the correction. The pathology does not require strict priority. It
requires *every* road rule to outrank progress — which is a **placement choice**
that v5.0 inherited from v4.7 without re-examining it.

This document changes the placement.

> Rules a competent driver may relax in order to complete a mission are placed
> **below** progress. Above progress remain only the rules that are never
> relaxable. Two trajectories that both complete then tie on the upper channels
> and are separated by how much they had to relax.

That is the minimum-violation semantics of Castro, Tumova, Karaman, Frazzoli and
Rus (ref. 15) expressed as a rule hierarchy rather than as a planner, and it is
the structural remedy v5.0 §11.8 identified but deferred.

### 1.1 What this buys, stated as falsifiable orderings

The design target is not "positive expert return". It is that the reward ranks
controlled trajectory pairs the way a competent driver would. These orderings
are the acceptance criteria of §9 and the fixtures of §10:

| # | ordering that must hold | mechanism |
|---|---|---|
| O1 | legal completion ≻ standing still | both tie on L1–L3; L4 separates |
| O2 | completion needing a brief lane relaxation ≻ standing still | both tie on L1–L3; L4 separates before L5 is reached |
| O3 | legal route ≻ illegal shortcut, both completing | tie on L1–L4; L5 separates |
| O4 | lane relaxation ≻ collision | L1 separates |
| O5 | waiting at red ≻ running it to finish | L3 separates, above L4 |
| O6 | necessary relaxation ≻ gratuitous relaxation | tie on L1–L4; L5 separates |

**What v5.0 actually fails, and it is narrower than an earlier revision of this
section claimed.** That revision asserted v5.0 fails O1, O2, O3 and O6. Working
the orderings out from the two constructions gives:

| | v5.0 | v5.1 |
|---|---|---|
| O1 | fails | fails |
| **O2** | **fails** | **passes** |
| O3 | passes | passes |
| O4 | passes | passes |
| O5 | passes | passes |
| O6 | passes | passes |

v5.0 passes O3 and O6 because its lane rules sit in R3, *above* progress, so the
illegal shortcut loses at R3 and the gratuitous violation loses on accumulated R3
cost. **The single ordering this restructure buys is O2** — and O2 is the
motivating pathology of the entire redesign: "standing still for ever beats
completing with a brief violation".

O1 is not a rulebook property in either version. Standing still is exactly
`(0,0,0,0,0)`, so under a *strict* lexicographic comparison it wins against any
trajectory that accrues even ε of L2 cost, which every real trajectory in traffic
does. No hierarchy in which stopping is safe can avoid this, and every admissible
hierarchy has that property. A **thresholded** comparison passes O1; see §11.1.

---

## 2. Method

Unchanged from v5.0 §2 and not restated: **Test A** (admissibility — a rule
bearing a satisfaction indicator must be satisfiable by a competent driver,
verified by expert replay, and can only reject), **Test B** (observability —
memory is admissible only when the observation carries the same state and that
state is a plausible perception output), and **controlled invariance** (a region
is admissible as a per-step penalty only if from every state inside it some
action keeps the ego inside).

One addition, forced by §1.1: those three tests constrain the sub-rules but say
nothing about their **placement**. A rule can be admissible, observable and
controlled-invariant and still be in the wrong channel. The orderings O1–O6 are
the placement test, and they are checked on constructed fixtures rather than on
expert replay, because replay contains no counterfactual.

---

## 3. The rulebook

Six levels, strictly ordered:

```
L1  collision safety         ≻
L2  interaction safety       ≻
L3  non-relaxable compliance ≻
L4  mission progress         ≻
L5  relaxable lane compliance ≻
L6  progress rate
```

| level | channel | sub-rules |
|---|---|---|
| L1 | `collision_safety` | at-fault collision impact |
| L2 | `interaction_risk` | `ttc`, `clearance`, `rss_lateral` |
| L3 | `non_relaxable_compliance` | `offroad`, `signal`, `stop`, `crosswalk`, `vehicle_yield`, `speed_limit` |
| L4 | `mission_progress` | signed route advance (§4) |
| L5 | `relaxable_lane_compliance` | `solid_line`, `wrong_carriageway`, `dashed_line` |
| L6 | `progress_rate` | `advance_shortfall` (§4.6) |
| — | diagnostic, non-normative | `rss` longitudinal, not-at-fault collisions, every atomic sub-rule cost |

**L6 is the last level and it is deliberately last** (ADR-076). It expresses a
preference for arriving sooner, which under `γ = 1` (ADR-075) nothing else in
the rulebook expresses. Placing it *below* L5 is what keeps that preference from
paying for a lane violation: an illegal shortcut carries `c_L5 > 0` and loses
before the level is reached, so O3 stays a theorem rather than becoming a
calibration.

`wrongway` remains deleted (v5.0 §5.6, ADR-066).

### 3.1 Why L1 and L2 are separate

They could be merged into one safety channel. They are not, because a collision
is an **outcome** and TTC/clearance/RSS-lateral are **anticipatory indicators**.
An indicator that fires is not a failure; a collision is. Merging them would let
an indicator reading dominate an actual impact of lesser normalized magnitude.

### 3.2 Sub-rule definitions

Unchanged from v5.0 §5, which remains the evidence of record, with the two
at-fault decisions incorporated:

- **L1** applies ADR-071: contacts are classified with nuPlan's taxonomy;
  at-fault contacts are charged and terminate the episode, not-at-fault contacts
  are charged **nothing** and **truncate** it with a `V(s)` bootstrap. The
  not-at-fault rate is reported as a diagnostic.
- **L2** applies ADR-070: `ttc`, `clearance` and `rss_lateral` are
  **inapplicable when `v_ego ≤ 5e-02 m/s`**.

### 3.3 Intra-level aggregation

Three different aggregations, because three different questions are being asked.
This is a change from v4.7's uniform `max` and is deliberate.

**Across objects within one sub-rule — `max`.** `c_ttc = max over actors`. The
semantics is "the most critical actor", and it is unchanged.

**Within L2 — `max`.** `c_L2 = max(c_ttc, c_clearance, c_rss_lateral)`. Defensible
here specifically because the three are complementary **detectors of one
property** — being outside the safe interaction envelope — so that
`c_L2 = 0 ⟺ all three are 0`. It is not a claim that they are cardinally
commensurable.

**Within L3 — `max`.** `c_L3 = max(c_offroad, c_signal, c_stop, c_crosswalk,
c_vehicle_yield, c_speed_limit)`. Minimax semantics: the level reports the worst
non-relaxable violation present. It does **not** assert that half a red light
equals half an off-road excursion; it uses the shared normalized scale only to
identify the worst case.

**Within L5 — normalized sum.**

```
c_L5 = ( c_solid_line + c_wrong_carriageway + c_dashed_line ) / 3
```

Here the quantity of interest is the *total amount of relaxation*, so a detour
that crosses a solid line **and** enters the opposing carriageway must cost more
than one that only straddles. `max` would make the second violation free, which
would break O6.

**Within L6 — a single sub-rule**, so no aggregation arises. The denominator is
kept explicit at 1 so the atomic vector of §3.4 stays uniform across levels.

**The denominator is fixed at 3 and is never recomputed over the applicable
sub-rules only.** Recomputing it would change the reward scale mid-episode as
sub-rules become applicable. An inapplicable sub-rule contributes 0 and is
reported through its own applicability mask.

### 3.4 The atomic vector is the contract, not the aggregation

The rulebook exposes to every algorithm the same per-step vector of **atomic**
costs, before any aggregation:

```
( c_collision_at_fault, c_ttc, c_clearance, c_rss_lateral,
  c_offroad, c_signal, c_stop, c_crosswalk, c_vehicle_yield, c_speed_limit,
  Δq,
  c_solid_line, c_wrong_carriageway, c_dashed_line )
```

The five-channel aggregation of §3.3 and the scalarization of §5 are **adapters**
over that vector, not part of it. This matters because the four planned
algorithms consume it differently — the scalar baseline through §5, the
lexicographic arms through the five channels directly, the distributional arms
through the return distributions of the same channels — and a comparison in which
the arms see different measurements would not be a comparison of algorithms.

`max` therefore never destroys information: it is computed *from* the atomic
vector, which remains logged in full.

### 3.5 Normative placement decisions, stated rather than inherited

Three placements are choices this document makes and a reader may contest. They
are listed here so that contesting them does not require reverse-engineering the
table.

1. **`offroad` is non-relaxable (L3), above progress.** Consequence: pulling onto
   a shoulder to pass an obstruction is ranked below not completing the mission.
   That is defensible on precedent — nuPlan treats `drivable_area_compliance` as
   a multiplicative penalty, i.e. as hard — and the 0.3 m tolerance of v5.0 §5.4
   already absorbs bounding-box over-approximation. It is nevertheless the
   placement most likely to be wrong, and moving it to L5 is a one-line change
   that would require re-deriving §9. **Recorded as a decision, not as a fact.**
2. **`speed_limit` is non-relaxable (L3).** Exceeding a posted limit to complete
   a mission is never admissible. Inapplicable throughout the PG panel by
   ADR-068's provenance gate, so on that half of the training mixture L3 carries
   five sub-rules rather than six.
3. **`dashed_line` is relaxable (L5).** Its cost is
   `penetration × time_factor` with the time factor 0 below `DASHED_T0_S = 1.0 s`,
   so a legal lane change costs **exactly zero by construction** and only
   sustained straddling is charged. That is the profile of a tidiness rule, not
   of a legality rule. The supervisor's explicit requirement — that sustained
   straddling be penalized — is preserved; only its priority changes.

---

## 4. L4 — mission progress, and L6 — progress rate

### 4.1 Definition

Let `s_t` be the arc-length projection onto the assigned route polyline and

```
Δq_t  = clip( (s_{t+1} − s_t) / D_REF , −1, +1 ),   D_REF = v_ref · Δt = 2.2222 m
L4_t  = Δq_t
```

with `v_ref = MISSION_PROGRESS_REFERENCE_SPEED_MPS = 22.2222 m/s`. **The channel
is the bare advance.** A per-step time cost subtracted here was proposed,
implemented, measured and rejected (ADR-076): inside L4 it would price duration
*above* lane compliance, so O3 would stop being an exact tie decided by L5 and
become a calibration. The time preference lives at L6 instead (§4.6). `Δq_t = 1`
therefore reads as "advanced one reference-speed step". `s_t` is what R4 reads
today (`components/progress.py`); nothing new is perceived, mapped or observed.

**`D_REF` is a unit, not a calibration.** The episode ceiling of §5.4 works out
to `a · (distance covered) / (longest single step)`, in which `D_REF` cancels
exactly: changing it only rescales `λ₄` inversely. It is fixed at `v_ref · Δt`
so that `λ₄` is directly comparable with the priority weights `a³, a², a`.

**The clip is the vehicle's physical bound, not a design choice.** MetaDrive
fixes `max_speed_km_h = 80` for every vehicle type
(`third_party/metadrive/metadrive/component/pg_space.py:233`) and enforces it by
cutting engine force above that speed (`base_vehicle.py:499`), with no override
in this repository's configuration. 80 km/h is 22.22 m/s, i.e. exactly `v_ref`.
**The agent therefore cannot produce a step above `D_REF`, and the clip never
binds on any policy this reward will train.** It exists so §5.4 has a declared
bound, not to shape behaviour.

### 4.1.1 Why not normalize by the route's own length

Normalizing by `L_route`, so that an episode's progress return is the completion
fraction in `[0, 1]`, was specified first and then measured. It fails for a
reason that is not obvious in advance, and the negative result is kept here
because the alternative is intuitively attractive:

- Every mission is then worth at most `λ₄` **regardless of length**, while the
  penalty channels stay per-step and grow with episode length. A long mission
  faces more exposure against the same fixed budget.
- Worse, the §5.4 bound is set by the **shortest route in the panel**: measured
  **19.96 m**, against a 171.98 m mean. On a route that short one step buys a
  large fraction of the mission, so that single degenerate record dictates `λ₄`
  for all 1100. Measured ceiling **32.7**, below v5.0's own 40.3.

With a fixed `D_REF` the binding case becomes the **longest single step**, which
is far less extreme, and a longer mission earns proportionally more — which is
right, since a longer mission plausibly needs more relaxation.

Completion fraction is not lost. It leaves the reward and is reported as an
**episodic metric** (§7), which is where the evaluation protocol wants it and
where no scale invariance obscures it.

Two further corrections of the earlier revision, both measured:

- *Renormalizing the route so that `q_0 = 0`* (the panel's ego starts **31.6 %**
  into its own assigned route, a real defect worth fixing for the reported
  metric) changes the reward ceiling by **nothing**: `L_route` cancels.
- *Capping the step advance at `v_ref` while still dividing by `L_route`* also
  changed the binding quantity by nothing — max `Δq` was `0.033371` with and
  without it, because that maximum came from a short route rather than a fast
  step — while breaking the telescoping identity on 1298 steps.

### 4.2 Why signed, and why not monotone

Undiscounted, `Σ_t Δq_t = (s_T − s_0) / D_REF`: the total depends on **distance
covered**, not on speed. Two policies that both complete the same mission cover
the same route arc length and therefore tie at L4, which is what lets L5 decide
O3. Correcting an earlier revision: this holds under v5.0's formulation too, so
O3 is *not* something this definition buys — the claim that v5.0's margin is "a
speed ratio, so the faster policy always scores more" is wrong, because summed at
fixed `Δt` that margin is distance.

What the signed form does buy:

- **oscillation is not a progress strategy** — covering a stretch forward and
  back contributes `+x − x = 0`, so the return is invariant to how many times a
  stretch is traversed;
- **standing still scores exactly 0**, with forward motion scoring more and
  reverse motion scoring less;
- **no hidden state**, which is why `REQ-RB5.1-OBS-01` no longer exists.

An earlier revision used the **monotone** advance — `σ` the running maximum of
`s` — for the first of those properties. It is withdrawn (ADR-073), because the
signed form obtains the same invariance by compensation rather than by memory,
while the monotone form carries state no observation field contains. Three
measured facts settled it:

- `enable_reverse=False` is MetaDrive's default and is not overridden here, so
  **the agent cannot reverse under power at all** — the exploit the memory
  guarded against is unreachable;
- `σ − s` is non-zero on **20.99 %** of expert steps but never exceeds
  **0.053 m**: that is projection jitter, not reversing, and the running maximum
  rectifies it directionally while the signed form compensates it;
- the expert mean moves from **+73.85 to +73.84** — the two are empirically
  indistinguishable.

The signed form additionally credits the **final position** rather than the peak,
and **penalizes reverse motion**, restoring the property v5.0 §5.6 relied on when
it deleted `wrongway`.

### 4.3 What is deliberately given up

The episode return is no longer the completion fraction, so it is not directly
comparable with the evaluation protocol's `route_completion`. That comparability
moves to the reported metric of §7 rather than being carried by the reward, which
is the standard separation between a training signal and a score.

### 4.4 The discount, resolved: `γ = 1` (ADR-075)

> **Amended by ADR-079 (2026-09-07, DRAFT pending user confirmation): `γ = 0.996`.**
> Two corrections to what follows, neither of which changes its *reasoning*.
> First, the break-even table below compares a future collision (`a³`) against a
> present **L3** violation (`a¹`), a ratio of `a²`. The binding comparison is one
> level apart — L1 against L2, and identically L2 against L3 — with ratio `a`, so
> every figure in the table is **2× too generous**. At `γ = 0.99` the real
> break-even is 78 steps, not 157; the conclusion that 0.99 breaks inside the
> episode therefore holds for a stronger reason than stated, and the "safe" mark
> on 0.995 does not survive at `a = 2.2` over a 199-step episode. Second, the
> undiscounted case has a cost this section does not weigh: with roughly two
> thirds of episodes ending in a **bootstrapped truncation** rather than a true
> terminal, `γ = 1` leaves the value level pinned only by the terminating
> minority, so approximation bias walks it freely. The requirement is
> `Δ = ln(a) / −ln(γ) > L`, which at `a = 2.5` (ADR-079) and `L = 199` gives
> `γ > 0.99541`.

An earlier revision left this open and listed three options. Measurement closed
it, and **withdrew two claims that revision made**:

1. that the choice is "immaterial" for the **scalar** arm — it is not;
2. that at `γ = 0.99` O3 "holds only approximately" — it fails by −1.5 to −9.5.

#### The reason that is not about O3

The hierarchy prices a collision at `a³ = 10.648` and a non-relaxable violation
at `a¹ = 2.2`. Exponential discounting erodes the two at different rates, so
past some horizon **a future collision costs less than one L3 violation now**:

| `γ` | break-even step | seconds | vs a 200-step episode |
|---|---:|---:|---|
| **0.99** | **157** | **15.7 s** | **hierarchy breaks inside the episode** |
| 0.995 | 315 | 31.5 s | safe |
| 0.999 | 1576 | 157.6 s | safe |
| 1.0 | never | — | safe |

At `γ = 0.99` on ~20 s episodes a policy choosing between running a red light
now and colliding in 16 seconds prefers the collision. This is independent of
L4, of the scalarization and of O3.

#### O3 under discounting, measured

Scalar arm, margin `A − B` at `λ₄ = 2.0`, `η = 1.0` on the panel's mean
**mission span** (90.17 m, `Q = 40.58`); `B` completes in 160 steps riding one
marking for 30. Positive means O3 holds:

| `γ` | scalar O3 margin |
|---|---:|
| **1.0** | **+1.000** |
| 0.999 | −0.442 |
| 0.995 | −3.579 |
| 0.99 | −4.423 |

Only `γ = 1` gives O3 exactly, and by construction: it is the unique value for
which `Σ_t Δq_t` telescopes, hence for which two trajectories reaching the same
place tie at L4 whatever their duration.

Three remedies were falsified before selecting — a threshold on L4 (at
`γ = 0.99` the `τ₄` that ties the legal route with the shortcut is 1.975, which
also ties it with a run abandoning **20 %** of the route), potential-based
shaping (`γ^T Φ_T` still favours the shorter episode, since termination is on
arrival), and raising `η` (bounded by `λ₄ + 0.1·η < a`). ADR-075 records all
three with figures.

**An earlier revision of these tables used the assigned route (171.98 m) where
the mission spans only 52 % of it (90.17 m), overstating every L4 quantity by
1.9×.** The tables above are restated at the correct span. No conclusion
changes: the break-even step is independent of the span, and O3 still fails at
every `γ < 1`.

#### Decision

**`γ = 1` on every channel and every arm**, one discount for all four so that a
difference in results stays attributable to the preference structure under test.
`learning_potential_gamma` must track it, or Ng et al.'s policy-invariance
theorem for potential-based shaping no longer applies. Declared fallback if the
value-based arms fail to stabilize: **`γ = 0.999` for every arm together**, with
the O3 degradation reported.

**`REQ-RB5.1-GAMMA` is resolved.**

### 4.6 `L6 progress_rate` (ADR-076)

`γ = 1` fixes the hierarchy and removes something at the same time: **all time
pressure**. The channel telescopes, so two trajectories reaching the same place
score identically however long they take, and nothing else in the rulebook
prefers the faster one — `speed_limit` is an upper bound only.

**Arrival is not a sufficient bound, and this was measured rather than assumed.**

| | p5 | p50 | p95 |
|---|---:|---:|---:|
| mission span `s_goal − s_start` | 13.0 m | **67.5 m** | 247.2 m |
| horizon | 197 steps | 199 steps | 200 steps |
| **mean speed required to arrive** | 0.65 m/s | **3.40 m/s** | 12.42 m/s |

The agent is capped at 22.22 m/s, so at the median it may travel **6.5× faster
than the minimum that still arrives**. Inside that band L4 is constant while
slower driving reduces exposure to the L2 sub-rules, all of which scale with
speed. Without L6 the optimum is to crawl at 12 km/h for twenty seconds, with a
full mission score.

The failure mode is documented: CaRL (Jaeger et al., CoRL 2025) report that a
reward with collision and off-road penalties but no progress term makes *«a
policy that always stays stationary ... optimal in reactive traffic»*. Their own
remedy is the branch §4.4 closes — *«our reward does not encode that getting to
the goal faster is better ... most RL algorithms naturally encode a notion of
urgency via a discount factor»* — which is available to them because their
reward is a soft-constrained sum with no priority ordering to erode.

#### Definition

```
c_L6,t  =  1 − clip( Δq_t , 0 , 1 )   ∈ [0, 1]
```

Nothing new is perceived: `Δq_t` is what §4.1 already computes. Standing still
and reversing both clip to zero advance and cost the maximum, consistent with
L4's own signed treatment of reverse motion.

#### Why below L5, which is the entire point

An illegal shortcut carries `c_L5 > 0` and therefore **loses at L5 before L6 is
consulted**. The weight on L6 is consequently *unconstrained by O3 in the
lexicographic and distributional arms*: how strongly the reward prefers arriving
sooner is decoupled from whether arriving sooner can buy a lane violation.

That decoupling is what a time cost inside L4 cannot provide, and it is why the
first proposal was rejected.

#### Why the advance shortfall rather than a flat time counter

Summed over any completing trajectory the two coincide:

```
Σ_t c_L6,t  =  T − Σ_t Δq_t  =  T − (s_goal − s_start)/D_REF
```

so L6 **ranks by duration exactly**, and the constant cancels from every
comparison — which is why the `λ₆` bound below does not depend on mission
length. What the advance-based form adds is a **per-step gradient**: advancing
more lowers the cost now. A flat counter gives the same ranking with no local
signal.

#### Two route lengths, which are different quantities

This document quotes **171.98 m** and **90.17 m** and both are correct, so the
distinction is stated once here rather than left to be re-derived.

- **171.98 m** is the *assigned route polyline*: the concatenated centrelines of
  the assigned lanes (`build_assigned_route_polyline`), i.e. the geometry `s_t`
  is projected onto. It is the denominator of the reported completion fraction.
- **90.17 m** is the *mission*, `s_goal − s_start`. It is **arc length along that
  polyline**, not a straight line — the chord between the same endpoints
  measures 87.29 m, 2.9 % shorter — and it is the distance the ego actually
  travels, because the ego enters the first lane partway along and stops partway
  into the last (`q_start 0.316 → q_end 0.837`, i.e. 52 % of the polyline).

**Every L4 quantity uses the mission**, since `Σ_t Δq_t = (s_T − s_0)/D_REF` is a
difference of stations. `Q = 90.17 / 2.2222 = 40.58`. The measurement confirms it
independently: the mean advance is `0.2041` per step, and `40.58 / 199 = 0.2039`.

An earlier revision of §4.4 and of ADR-075 used the polyline length in place of
the mission, overstating every L4 figure by 1.9×; both are corrected.

#### `λ₆`, and the one arm where it is bounded

Reference shortcut, chosen as the *cheapest* illegal shortcut saving the *most*
time: over a 200-step episode it saves **40 steps** while riding **one** marking
for **30 steps** at `c_L5 = 1/3`.

```
gain = λ₆ · 40 · (Δt/T_REF)     cost = η · (1/3) · (Δt/T_REF) · 30 = 1.0
=>  λ₆ < 0.25      (η = 1.0)
```

**`λ₆ = 0.2`**, a 20 % margin. §5.4's tail gains `λ₆·(Δt/T_REF)`:
`2.0 + 0.1 + 0.02 = 2.12 < a = 2.2`.

This bound binds on the **scalar arm only**. Summing every channel re-couples
what an ordering separates, and the consequence is measurable: `λ₆ = 0.2` buys
**3.18** reward units between a crawl and a full-speed completion, against
`a² = 4.84` for one violated L2 step. The scalar arm therefore prefers speed only
while crawling would save **fewer than 0.66 L2 steps**. That is a scalarization
failing to express what an ordering expresses — a **result to report**, not a
defect to engineer around.

#### Thresholds for the thresholded-lexicographic arm

- **`τ₄ > 0` is required**, and what ADR-076 buys is that the width needed
  collapses: two completing trajectories tie at L4 to exact arithmetic, where a
  time cost inside L4 would have needed 0.400 and `γ = 0.99` needed 1.975 — a
  width that also conflated completing the route with abandoning a fifth of it.

  **`τ₄ = 0` is nevertheless the fixture value, not a training value, and an
  earlier revision of this section wrongly presented it as sufficient.** The §10
  fixtures tie exactly because both trajectories are constructed over the same
  mission; two *learned* trajectories have equal L4 with probability zero, so at
  `τ₄ = 0` the comparison would stop at L4 on essentially every pair and **L5 and
  L6 would decide nothing.**

  `τ₄` is therefore load-bearing rather than a tolerance: it is the minimum
  progress a lane relaxation must buy to be worth taking. Too small and the ego
  crosses a solid line to gain centimetres; too large and abandoning the route
  counts as completing it.

  **The value and the comparison rule belong to the algorithm specification**,
  where `d₂` already belongs (§11.1), and this document deliberately fixes
  neither. **They cannot be fixed here**, because the object a threshold applies
  to is not the same across the candidate algorithms:

  - *Absolute Thresholding* (Gábor et al. 1998) admits actions whose **Q-value**
    exceeds a real number — a per-state-action threshold on expected *future*
    return, which is not a completion fraction;
  - *Absolute Slacking* (Li & Czarnecki 2019) admits actions within a slack of
    the **optimal value in that state** — already relative, so it sidesteps the
    mission-length problem for free;
  - a *policy-gradient* construction (Tercan & Prabhu 2024) has no value function
    and compares **expected returns**, which is episodic.

  What this document therefore records is only the constraint, which holds under
  all three: `τ₄ > 0`; it must express "the progress difference is too small to
  justify a lane relaxation"; and a single absolute constant cannot serve every
  mission when spans run from 13 m to 247 m. **Fixing it is sequenced after the
  rulebook is frozen**, not before. Tracked as `D1` in `docs/open_items.md`,
  where the analysis is kept.

  One thing is settled here because it is a property of the reward rather than of
  the comparison: **the channel is not renormalized.** `Σ Δq` stays in absolute
  units, so §4.1.1's measured rejection stands and the scalar arm's weight
  calibration is untouched. Any per-mission normalization a threshold rule may
  need belongs to that rule, where within one scenario it is a positive constant
  and reorders nothing.
- **`τ₆`: none.** L6 is the last objective, and thresholded lexicographic
  ordering leaves the last objective unthresholded (Gábor et al. 1998; Vamplew
  et al. 2011).

With a positive `τ₄` the arm is structurally identical to the setting in which
TLO was originally validated: a thresholded prefix, an unthresholded final
objective minimizing time to the terminal state, and `γ = 1`.

---

## 5. Scalarization — `SCAL-V1.4`

### 5.1 Form

```
r_t  =  Σ_{k=1..3} a^(4−k) · [ (step(m_k) − 1) + σ · m_k ]
        + φ · Σ_{k=1..3} m_k
        + λ₄ · Δq_t
        − λ₅ · c_L5,t · (Δt / T_REF)
        − λ₆ · c_L6,t · (Δt / T_REF)
```

with `m_k = −c_Lk ∈ [−1, 0]` for `k = 1..3`, margins canonicalized to exactly 0
before `step` is applied, `Δq_t ∈ [−1, 1]` as defined in §4.1,
`c_L5,t, c_L6,t ∈ [0, 1]`, and `T_REF = 1 s`. Selected values are in §5.5.

**`SCAL-V1.4` is `SCAL-V1.3` plus the L6 term.** L1–L3 remain `SCAL-V1.2`
verbatim and L4/L5 are unchanged; only the sixth level is new. The version is
bumped because the *form* gained a term, not because anything above it moved.

**L1–L3 keep v5.0's per-step dominance. L4 and L5 form a finite exchange.** The
first three levels are `SCAL-V1.2` unchanged; only the tail differs.

### 5.2 Why L4 and L5 cannot be additional `SCAL` levels

`SCAL`'s per-step dominance works because each level carries a **satisfaction
indicator** `step(m_k)`, which makes the level discrete. Progress is continuous.
Strict dominance of L4 over L5 would require `λ₄·Δq > λ₅·c_L5` for **every**
`Δq > 0`, including `Δq → 0⁺`. **No finite `λ₄` satisfies that.**

Two ways to force it, both rejected:

1. **Indicator on L4** (`1[Δq > 0]`) — makes `SCAL` uniform across all five
   levels, and rewards infinitesimal creep: an ego inching forward at 0.01 m/s
   satisfies the indicator, gains dominance over L5, and rides a solid line for
   free. This converts "stop forever" into "creep forever along the marking",
   which is a worse degeneracy than the one being removed.
2. **Indicator on a progress threshold** (`1[Δq > δ]`) — removes the creep, but
   introduces an unmotivated calibrated constant and a cliff the policy will sit
   on, which is the class of construction this project has rejected throughout.

The finite exchange is therefore not a concession. It is the only construction
that claims exactly what it can prove.

### 5.3 What is and is not lost relative to `SCAL-V1.2`

Nothing is lost, because `SCAL-V1.2` never had episodic lexicographic order
either — v5.0 §6.5 states this correctly and v5.0 §11.8 wrongly denied it, a
contradiction corrected in that document. What changes is that this
specification stops claiming it. The proved property is retained exactly where it
was demonstrated: **strict per-step dominance over L1–L3**.

The scalarization is in any case the adapter for **one of four arms**. The
lexicographic and distributional arms consume §3.4's vector directly and never
evaluate this expression, so an imperfect `η` degrades a baseline rather than the
experiment.

### 5.4 Rank-preservation condition

Level `k ∈ {1,2,3}` dominates iff

```
a^(4−k)  >  (1 + σ) · Σ_{j>k, j≤3} a^(4−j)  +  φ · (3 − k)
            +  λ₄ · ΔQ_MAX  +  (λ₅ + λ₆) · (Δt / T_REF)
```

Derivation, unchanged in structure from v5.0 §6.3: with level `k` violated the
score is at best `−a^(4−k) + λ₄·ΔQ_MAX`, approached as `m_k → 0⁻`; with level `k`
satisfied and every lower level maximally violated it is at worst
`−(1+σ)·Σ_{j>k} a^(4−j) − φ·(3−k) − λ₅·(Δt/T_REF)`. Requiring the second to
exceed the first gives the condition.

It reduces to v5.0's whenever the utility swing `λ₄·ΔQ_MAX + λ₅·(Δt/T_REF)`
equals that document's `2λ`, so the two are the same statement about different
tails.

**The binding constraint is `k = 3`**, where the lower-level sum is empty:

```
a  >  λ₄ · ΔQ_MAX  +  λ₅ · (Δt / T_REF)
```

With `ΔQ_MAX = 1` (§4.1), `a = 2.2`, `Δt = 0.1 s` and `T_REF = 1 s`:

```
λ₄  +  0.1 · (λ₅ + λ₆)  <  2.2
```

At the selected weights: `2.0 + 0.1·(1.0 + 0.2) = 2.12 < 2.2`. `λ₆` is bounded
twice over — by this per-step condition and, more tightly, by the episodic O3
condition of §4.6 — and the second is what fixes its value.

**`λ₄` is a calibrated parameter of the same standing as `a`, not a convention,
and assuming otherwise is what the first measurement falsified.** Under the
rejected `L_route` normalization of §4.1.1, `λ₄ = 1` gave the expert a mean of
**−8.49** against v5.0's +22.05 on the same pilot: the mission was worth less
than the violations incurred completing it, so standing still won *more* often,
the exact opposite of this document's purpose. `λ₄` and `η` are fixed together in
§5.5, by measurement.

### 5.5 Selected weights

> **Amended by ADR-079 (2026-09-07, DRAFT pending user confirmation):
> `a = 2.5`, `σ = 0.30`.** `λ₄ = 2.0`, `η = 1.0`, `λ₆ = 0.2` and `φ = 0.25` are
> unchanged, so the measured calibration below stands. Two things this section
> states that turned out to be load-bearing in the other direction. `σ = 0` is
> described here as "inherited from `SCAL-V1.2`", and it is: the instrument
> sweeps `σ` only in the four-level counterfactual family, so it had **never**
> been priced in this hierarchy. And `a = 2.2` is the first round value above
> §5.4's lower bound of 2.12 — a lower bound, with no upper bound anywhere in
> this document. The reason to move it is that §5.4 caps `σ` at 0.1227 when
> `a = 2.2`, and no `σ` in that window makes the reward's local gradient point
> the right way in a conflict a real vehicle could still brake out of; the
> criterion is `a_req^max = (w₂σ + φ)·v_ref / (2·λ₄·τ)`, which the shipped
> weights put at **1.46 m/s²** against a braking limit of about 9. Measured on
> this same 1100-record panel by the `(a, σ, λ₄)` grid ADR-079 added:
> `fraction_below_standstill` **3.36 % → 4.55 %**, `w₃`/tail **1.04 → 1.18**,
> `a_req^max` **1.46 → 12.43 m/s²**, expert p1 **−61.8 → −99.4**.

**`λ₄ = 2.0`, `η = 1.0`, `λ₆ = 0.2`**, with `a = 2.2`, `σ = 0`, `φ = 0.25`
inherited from `SCAL-V1.2`. Admissible: `2.0 + 0.1·1.2 = 2.12 < 2.2`.

Measured on the full 1100-record Waymo `train` panel, 217,189 transitions, 0
skipped, 0 errors, varying only the weights on identical atomic costs:

| rulebook | mean | p1 | p5 | p50 | below standstill |
|---|---:|---:|---:|---:|---:|
| v5.0 specified (`SCAL-V1.2`) | +31.14 | −150.85 | −15.86 | +25.72 | 7.45 % |
| v5.1 `λ₄ = 1.5`, `λ₆ = 0.2` | +50.55 | −84.37 | −1.41 | +37.19 | 4.55 % |
| v5.1 `λ₄ = 2.0`, `λ₆ = 0` | +73.84 | −59.67 | +9.16 | +55.23 | 3.36 % |
| v5.1 `λ₄ = 2.0`, `λ₆ = 0.1` | +72.27 | −60.74 | +7.23 | +53.52 | 3.36 % |
| **v5.1 `λ₄ = 2.0`, `η = 1`, `λ₆ = 0.2`** | **+70.70** | **−61.81** | **+5.29** | **+51.85** | **3.36 %** |
| v5.1 `λ₄ = 2.0`, `λ₆ = 0.25` | +69.92 | −62.87 | +4.33 | +50.18 | 3.36 % |

v5.1 dominates v5.0 on every column: mean 2.3×, the p1 tail less than half as
deep, **p5 positive** where v5.0 was at −15.86, and below-standstill episodes
under half.

**Two consistency checks worth stating.** At `λ₆ = 0` the panel reproduces
**+73.84**, bit-identical to the measurement taken before the level existed, so
L1–L5 are untouched and L6 is cleanly separable. And below-standstill is
**identical at 3.36 % across every `λ₆`**, because a stopped ego pays
`c_L6 = 1` on every step and the diagnostic compares against
`−λ₆·(Δt/T_REF)·T` rather than against 0 — the baseline moves with the channel,
as it must.

**L6's "violation rate" is not a meaningful statistic** and is reported only for
uniformity: `c_L6 > 0` on **99.40 %** of expert steps, because it is positive
whenever the ego is below 80 km/h. The meaningful figure is the mean cost,
**0.796**, i.e. a mean advance of `0.2041` per step — which equals the mean
mission span over the horizon (`40.58 / 199 = 0.2039`) to four figures, an
independent check on the whole chain from route projection to channel.

`λ₄ = 2.0` is preferred to the marginally better `λ₄ = 2.15` because the
constraint is `λ₄ + 0.1·η < 2.2`: at 2.15 only `η ≤ 0.5` remains admissible,
while at 2.0 the budget reaches `η = 2`. Six points of mean return buy room for
L5, which is the channel this entire restructure exists to create.

**`η` and the two bounds on it.** With the `Δt/T_REF` scaling,
`C_L5 = Σ_t c_L5,t · (Δt/T_REF)` reads as **equivalent seconds of
maximum-severity relaxable violation**, and `η` as the fraction of a mission
traded for one such second. Above: a *gratuitous* relaxation unlocks no progress,
so it loses for **any** `η > 0` and O6 is satisfied structurally. Below: a
detour holding `c_L5 = 0.5` for 2 s to unlock half a mission requires `η < 0.5`
in mission units, which at `λ₄ = 2.0` is comfortably satisfied.

**Measured caveat: `η` is nearly inert on the expert panel** — across `η ∈ [0, 5]`
the mean moves by under 0.1. The expert almost never relaxes, so the panel cannot
discriminate. `η = 1.0` is therefore selected on the argument above and on
admissibility headroom, **not** on measured expert return, and it must be
validated by `TEST-RB5.1-06` rather than by replay. This is stated rather than
hidden: it is the one weight in this document not pinned by measurement.

### 5.5.1 Why the Test A figures are conservative

MetaDrive caps every vehicle at 80 km/h = `v_ref`, so the agent cannot produce a
step above `D_REF` and the §4.1 clip never binds on it. The logged Waymo expert
is a real car under no such cap: its step advance reaches **3.609 m** (130 km/h),
with p50 0.489, p90 1.394, p99 2.021 and p99.9 2.920 m, and the clip binds on
**1298 of 217,189 steps (0.6 %)**.

Those steps are the expert being charged for speed the policy could never
attain. **The +70.70 above is therefore a lower bound on what this reward gives a
competent driver**, and the telescoping identity — inexact on the expert panel
for exactly those 1298 steps — is exact for every trajectory the agent can
generate.

---

## 6. Observation consequences

Unchanged from v5.0 §7 and not restated. `speed_limit` still requires amending
`OBS-V1.3` and `OBS-LIDAR-V2.0` and still intentionally breaks checkpoint
compatibility.

L4 adds nothing. It reads `s_t`, already available to the rulebook, and depends
only on the **current step's** advance — no running maximum, no history. An
earlier revision carried `REQ-RB5.1-OBS-01`, requiring the monotone maximum to be
reconstructible from the observation window; ADR-073's amendment removed the
maximum, so the requirement has no subject and is **withdrawn**.

---

## 7. Diagnostics

Reported, never in the reward: `rss` longitudinal (v5.0 §4.7, ADR-063),
not-at-fault collision rate (ADR-071), every atomic cost of §3.4, every
applicability mask, and the per-level channel values.

Two new counters this document requires:

- `l4_clip_binding_steps` — how often the §4.1 clip binds. Zero for any agent
  trajectory by vehicle dynamics; non-zero only on expert replay (§5.5.1);
- `l5_reached_steps` — how often two compared trajectories tie through L4 so that
  L5 decides. If this is zero in practice, the restructure has not achieved what
  §1.1 claims and the result must be reported as such;
- **`mean_ego_speed_by_source`** — mean and p95 ego speed, reported **separately
  for PG and Waymo**. Required by limitation 13: L6 rewards speed and `speed_limit`
  is admitted on Waymo only, so the two sources are under different normative
  regimes and the difference must be visible rather than inferred.

---

## 8. Measurement status

The sub-rule evidence of v5.0 §4.2–§4.9 **carries over untouched**: those tables
are per-sub-rule, and no sub-rule definition changed. That is the dividend of
having measured atomically.

Re-measured for this document on the same 1100 Waymo `train` records — 217,189
transitions, **0 skipped, 0 errors**:

| quantity | measured |
|---|---|
| channel violation rates | L2 **0.3978 %**, L3 **0.6418 %**, L5 **0.8734 %** |
| episode returns | §5.5 table |
| below standstill | **3.36 %** at the selected weights |
| step-advance distribution | p50 0.489, p90 1.394, p99 2.021, p99.9 2.920, max 3.609 m |
| route lengths | min 19.96 m, mean 171.98 m |

L2's 0.3978 % is bit-identical to v5.0's macro R2, which confirms L2 is that
channel unchanged. L3 + L5 (1.5152 %) exceeds v5.0's macro R3 (1.1796 %) because
`max` aggregation collapsed concurrent violations that two separate channels now
expose; this is expected, not a regression.

The measurements above establish **admissibility** (Test A). The **orderings**,
which are what this document exists for, are established separately by the
constructed fixtures of §10, because expert replay holds one trajectory per
scenario and therefore no counterfactual. Both halves are now in place.

---

## 9. Acceptance criteria

| ID | criterion | status |
|---|---|---|
| `AC-RB5.1-01` | O1–O6 all hold on the §10 fixtures under `SCAL-V1.4` | **PASS** — `tests/test_rulebook_v51_orderings.py` |
| `AC-RB5.1-02` | O1–O6 all hold under strict lexicographic ordering on the five channels, **or** each failure is reported with the channel that caused it | **PASS** — O1 fails at L2 and the failure is asserted; O2–O6 hold |
| `AC-RB5.1-03` | Expert mean episode return is positive | **PASS** — +70.70 |
| `AC-RB5.1-04` | Expert episodes below standstill do not exceed the v5.0 figure of 7.45 % | **PASS** — 3.36 % |
| `AC-RB5.1-05` | The §4.1 clip binds on no step the agent can produce | **PASS** by vehicle dynamics (§5.5.1); binds on 0.6 % of *expert* steps, which are above the agent's physical maximum |
| `AC-RB5.1-06` | The selected `(a, σ, φ, λ₄, λ₅)` satisfies §5.4 for every `k` | **PASS** — `2.0 + 0.1 = 2.1 < 2.2`; inadmissible pairs are never priced |
| `AC-RB5.1-07` | `Σ_t Δq_t = (s_T − s_0)/D_REF` to numerical tolerance | **PASS** for agent trajectories; inexact on the expert panel only where the clip binds (§5.5.1) |
| `AC-RB5.1-08` | Every atomic cost of §3.4 is exposed alongside the aggregated channels | **PASS** — inherited from instrument |
| `AC-RB5.1-09` | Validation and test splits are consulted by no calibration in this document | **PASS** — inherited from v5.0 `AC-RB5-12` |
| `AC-RB5.1-10` | L1–L4 are bit-identical across `η` settings — `η` reaches only L5 | **PASS** — p1 and p5 identical across `η ∈ [0, 5]` |
| `AC-RB5.1-11` | v5.1 is not worse than v5.0 on mean, p1, p5, p50 and below-standstill | **PASS** — dominates on all five |
| `AC-RB5.1-12` | O3 holds against the §4.6 reference shortcut, which arrives sooner, in **both** the scalar and the strict-lex comparison | **PASS** — `TEST-RB5.1-16` |
| `AC-RB5.1-13` | `λ₆` is strictly below its O3 bound | **PASS** — 0.2 < 0.25 |
| `AC-RB5.1-14` | The §5.4 predicate admits `(λ₄, η, λ₆) = (2.0, 1.0, 0.2)` and rejects an inadmissible `λ₆` | **PASS** — 2.12 < 2.2 |
| `AC-RB5.1-15` | The below-standstill diagnostic compares against `−λ₆·(Δt/T_REF)·T`, not against 0 | **PASS** — `v51_standstill_return` |
| `AC-RB5.1-16` | `γ` is 1 in every algorithm configuration, and `learning_potential_gamma` equals it | **PASS** — six configs updated, guarded by `test_every_algorithm_is_undiscounted` |
| `AC-RB5.1-17` | L6 refines only ties: O1–O6 hold unchanged with the sixth level present | **PASS** — `TEST-RB5.1-20` |

`AC-RB5.1-02` is deliberately permissive: strict lex is expected to fail O1 for
the reason given in §11.1, and that failure is a **result to report**, not a
defect to repair.

## 10. Test matrix

| ID | test |
|---|---|
| `TEST-RB5.1-01` | O1 fixture: legal completion vs standstill |
| `TEST-RB5.1-02` | O2 fixture: obstacle requiring brief relaxation vs standstill |
| `TEST-RB5.1-03` | O3 fixture: legal route vs faster illegal shortcut, both completing |
| `TEST-RB5.1-04` | O4 fixture: lane relaxation vs collision |
| `TEST-RB5.1-05` | O5 fixture: waiting at red vs running it to complete |
| `TEST-RB5.1-06` | O6 fixture: necessary vs gratuitous relaxation, equal completion |
| `TEST-RB5.1-07` | `Σ Δq = (s_T − s_0)/D_REF` below the clip; a stretch covered forward and back nets exactly zero |
| `TEST-RB5.1-08` | The §4.1 clip binds on no step reachable at `max_speed_km_h`, and `D_REF` equals `v_ref · Δt` |
| `TEST-RB5.1-09` | §5.4 predicate agrees with exhaustive search over the weight grid |
| `TEST-RB5.1-10` | Per-step dominance of L1–L3 holds on a fixture covering all three |
| `TEST-RB5.1-11` | `c_L5` denominator stays 3 when a sub-rule is inapplicable |
| `TEST-RB5.1-12` | `c_L2 = 0` iff all three L2 sub-rules are 0 |
| `TEST-RB5.1-13` | Varying `η` leaves L1–L4 channel values bit-identical |
| `TEST-RB5.1-14` | ADR-071: at-fault terminates and charges; not-at-fault truncates and charges nothing |
| `TEST-RB5.1-15` | L4 ties **exactly** between two completing runs of different duration |
| `TEST-RB5.1-16` | O3 against the §4.6 reference shortcut, in both comparisons |
| `TEST-RB5.1-17` | `λ₆` stays strictly below its O3 bound, and so does every grid member |
| `TEST-RB5.1-18` | `Σ c_L6 = T − Q` for every completion time, with L4 unchanged |
| `TEST-RB5.1-19` | Standing still and reversing both cost the maximum at L6, and the standstill baseline follows |
| `TEST-RB5.1-20` | L6 does not disturb O1–O6 above it |
| `TEST-RB5.1-21` | §5.4 admits the selected weights with `λ₆` and rejects an inadmissible one |
| `TEST-RB5.1-22` | Every algorithm config is undiscounted, and the shaping discount tracks `γ` (`tests/test_hydra_agent_presets.py`) |

Fixtures `TEST-RB5.1-01..06` are **constructed**, not replayed: expert replay
contains no counterfactual and cannot establish an ordering between a taken and
an untaken trajectory. This is the test class v5.0 lacked entirely.

**Implemented** in `tests/test_rulebook_v51_orderings.py`, all passing.

`TEST-RB5.1-03`'s speed-independence fixture and `TEST-RB5.1-07`'s
repeated-traversal fixture were briefly restated while a time cost sat inside L4,
because that cost broke the exact L4 tie they assert. ADR-076 moved the cost to
L6 and **both are restored to their original form**: L4 ties exactly, and L6
separates the two afterwards. The episode is a small worked example of why the
level matters — a quantity placed one level too high silently invalidates the
fixtures that guard the level above it. They are written as per-step channel vectors rather than as scenes:
whether a given geometry yields `c_solid_line = 0.4` is the sub-rule evidence's
business (v5.0 §4), while these assert the *hierarchy* — which channel a cost
lands in and what that placement implies for the ordering. Both comparison rules
are exercised wherever they differ, so `AC-RB5.1-02`'s reported strict-lex
failure is an asserted property rather than an omission.

Writing them produced one finding worth recording. A first draft modelled
"driving through traffic" as L2 > 0 on *every* step and the fixture failed: the
satisfaction indicator charges `a² = 4.84` per violated step whatever the
severity, so 40 such steps cost 193.6 against 20 of progress and lose to standing
still — correctly. That is v5.0 §6.6's affordability constraint reappearing from
a different direction, and it is now asserted directly by
`test_l2_indicator_makes_continuous_violation_lose_to_standing_still`. It also
sharpens why the measured **0.3978 %** matters: the constraint is on how *often*
an L2 sub-rule fires, not on how mild it is when it does.

---

## 11. Known limitations

1. **Strict lexicographic ordering still prefers standing still, and no rulebook
   can prevent it.** Such an agent compares channels in order and stops at the
   first that differs. Standing still is exactly 0 on L1–L3 — ADR-070's gate
   makes it *provably* so — while any trajectory moving through traffic accrues
   some L2 cost, measured at **0.3978 %** of expert steps in v5.0 §4.5. L4 is
   therefore never reached. This holds for every hierarchy in which stopping is
   safe, i.e. every admissible hierarchy, and the restructure of §3 neither
   causes nor cures it.

   A **thresholded** ordering does cure it, by converting "both within budget"
   into a tie at the safety channels so the lower channels are reached. The
   threshold `d₂` is not invented: the instrument already produces the expert's
   **per-episode** L2 cost across all 1100 records, and a `d₂` that fails to
   admit the human expert is thereby falsified. Fixing `d₂` belongs to the
   algorithm specification, not here; what belongs here is the observation that
   the data to fix it already exists.

2. **`η` is the one weight in this document not pinned by measurement.** §5.4
   leaves it nearly free (`λ₄ + 0.1·η < 2.2`), and the expert panel cannot
   discriminate: across `η ∈ [0, 5]` the mean return moves by under 0.1, because
   the expert almost never relaxes a lane rule. `η = 1.0` rests on the
   two-sided argument of §5.5 and on admissibility headroom, and must be
   validated by `TEST-RB5.1-06`. The exposure is bounded: `η` enters only the
   scalarization, which is the adapter for one of the four planned arms — the
   lexicographic and distributional arms consume §3.4's vector directly and never
   evaluate it.

3. **`D_REF` fixes the reward's scale but not its fairness across mission
   lengths.** A longer mission now earns proportionally more, which §4.1.1 argues
   is right. The converse is that a short mission has a small progress budget
   against per-step penalties of unchanged size, so short scenarios are
   intrinsically harsher. The panel's shortest route is 19.96 m against a
   171.98 m mean; those records were **not** excluded, and no record was dropped
   to improve any figure in this document.

4. **The `max` within L3 is a placement of convenience.** It identifies the worst
   non-relaxable violation on a shared normalized scale without any claim that
   the scales are cardinally comparable across a red light, an off-road excursion
   and an overspeed. A finer treatment would need per-sub-rule priorities within
   L3, which the atomic vector of §3.4 leaves available.


5. **The calibration remains Waymo-only.** Test A excludes PG because a PG
   record's logged ego is `IDMPolicy`, and replaying a headway-maintaining
   controller to validate a headway rule is circular; that argument is unchanged
   and no adapter affects it.

   What *has* changed since v5.0 wrote this limitation: that document claimed a
   PG static adapter "requires ... to be written". It does not.
   `build_pg_static_adapter_result`
   (`src/thesis_rl/rulebook/v2/context/pg_static_adapter.py:144`) already exists
   and already reads `polygon`. The instrument simply calls the Waymo adapter
   unconditionally. **A PG coverage measurement — applicability rates and
   geometric sanity, not Test A — is therefore a dispatch change, not new
   capability, and is no longer credibly deferred.**

6. **Test A can only reject.** A rule the expert never violates may still be
   vacuous.

7. **L1 is `NOT_MEASURED` offline.** The replay has no physical contacts, so
   ADR-071's classification is validated by constructed fixtures
   (`TEST-RB5.1-14`) rather than by replay.

8. **The orderings O1–O6 are necessary, not sufficient.** Satisfying them shows
   the reward ranks six constructed pairs correctly. It does not show that the
   optimal policy under that reward is a good driver, which no offline analysis
   can show.

9. **The scalar arm expresses a weaker time preference than the ordered arms,
   measurably** (§4.6). Summing every channel re-couples what an ordering
   separates, so `λ₆` stays bounded by O3 at `< 0.25` and buys 3.18 reward units
   between a crawl and a full-speed completion, against `a² = 4.84` for one
   violated L2 step: the scalar arm prefers speed only while crawling would save
   fewer than **0.66** L2 steps. The lexicographic and distributional arms carry
   no such bound, because an illegal shortcut loses at L5 before L6 is reached.
   This is a property of scalarization, not a defect of the rulebook, and it is
   **a result to report**.

10. **The reference shortcut of §4.6 is a stipulation, not a measurement.** The
    expert panel cannot supply one: the logged human relaxes a lane rule on
    0.8734 % of steps and never takes an illegal shortcut, so there is no
    empirical distribution of shortcut profiles to fit. It is declared there and
    enforced by `TEST-RB5.1-16`. A shortcut more aggressive than the reference is
    *easier* for O3, not harder, which is why the cheapest one was chosen.

11. **L6 charges a legitimately stopped ego.** An ego correctly waiting at a red
    light pays `c_L6 = 1` per step. Gating L6 on whether an L3 rule is active was
    considered and rejected: it couples two levels and adds a gate to calibrate,
    while the hierarchy already supplies the justification — a trajectory that
    runs the red differs at L3, above, and two trajectories both waiting tie at
    L6 for as long as they both wait.

12. **`γ = 1` carries an off-policy stability risk** (§4.4). The Bellman operator
    is not a sup-norm contraction at `γ = 1`; the task is proper, so the
    formulation stays well-posed, but bootstrapped Q-learning is practically less
    stable than at 0.99. The declared fallback is `γ = 0.999` **for every arm
    together**, with the O3 degradation reported. Separately, Tercan & Prabhu's
    own remedy for thresholded ordering is policy gradient rather than a `γ`
    choice, which makes PPO the safe carrier of the thresholded arm and SAC/TD3
    the flagged combination.

13. **L6 rewards speed, and PG admits no speed limit — accepted, not fixed.**
    ADR-068's provenance gate rejects every PG lane speed limit, because the PG
    exporter writes whichever default the lane constructor held, under a `_kmh`
    key, without conversion. Until ADR-076 this cost nothing: L4 is
    speed-independent, so no part of the reward rewarded going faster. **L6 does
    reward it**, up to `v_ref`, and on PG nothing normative opposes that:
    `ttc`/`clearance`/`rss_lateral` require other agents, and `offroad` fires only
    after the ego has left the road. Half the training panel is PG.

    **This is accepted rather than repaired**, on the user's decision of
    2026-08-20 and for three reasons. PG blocks are synthetic, so there is no
    traffic law to encode — MetaDrive's block design speeds are simulator
    parameters, not norms — and inventing a PG limit would be exactly the
    unmotivated calibrated constant this project rejects elsewhere (§5.2). The
    geometry constrains reactively: a PG curve taken at `v_ref` leaves the road,
    and `offroad` sits at L3, above L6. And it is a difference **between sources**,
    not between arms: every arm sees the same data, so it does not confound the
    comparison this thesis makes.

    What it costs is that PG and Waymo are under different normative regimes for
    speed. §7's `mean_ego_speed_by_source` exists so that this is **measured and
    reported** rather than assumed away.

14. **Seven of the 1100 Waymo `train` missions are not completable by the
    agent.** They require a mean speed above the vehicle's 22.22 m/s cap because
    the logged human exceeded 80 km/h. The same holds for 1 of 150 validation and
    1 of 555 test Waymo records; no PG record is affected. **Maximum achievable
    success is therefore 99.36 % (train), 99.33 % (validation), 99.82 % (test)**,
    and reported success rates must be read against that ceiling rather than
    against 100 %. No record was excluded: the frozen panel is not edited to
    improve a figure.

    **Completing them would be legal**, which was checked rather than assumed:
    their posted limits are 29.06 m/s (65 mph) or 31.29 m/s (70 mph) and the
    required mean speed, 22.85–29.80 m/s, is below limit + tolerance in every
    case. The obstruction is the vehicle, not the rulebook.

    **Raising `max_speed_km_h` was considered and rejected on 2026-08-20**, for a
    reason that is quantitative rather than procedural. §4.1's episode ceiling is
    `a · (distance covered) / (longest single step)`, and the longest step the
    agent can produce *is* the cap times `Δt`, so **the cap sits in the
    denominator of what a mission can be worth**:

    | cap | `Q` (mean mission) | ceiling | change |
    |---:|---:|---:|---:|
    | 80 km/h | 40.58 | 84.4 | — |
    | 100 | 32.46 | 67.5 | −20 % |
    | **110** (covers all nine) | 29.51 | **61.4** | **−27 %** |
    | 120 | 27.05 | 56.3 | −33 % |

    `λ₄` cannot compensate, being bounded by `λ₄ + 0.1·(λ₅+λ₆) < a`. Every mission
    would therefore be worth 27 % less while every per-step penalty stayed
    unchanged — the direction of the v5.0 pathology this document exists to
    correct. Raising `v_ref` with the cap keeps the telescoping identity and does
    not change that arithmetic.

---

## 12. References

Unchanged from v5.0 §12. Ref. 15 (Castro, Tumova, Karaman, Frazzoli, Rus,
*Incremental sampling-based algorithm for minimum-violation motion planning*) is
promoted from "recorded alternative" to the semantic basis of §1.

## 13. Out of scope

Unchanged from v5.0 §13, plus: comfort and jerk are **excluded from the rulebook
and from the reward**. If action smoothing is wanted it belongs downstream of the
policy, identical and frozen across all arms, with the rulebook evaluating the
executed action; raw action, filtered action and jerk may be logged as
diagnostics only.
