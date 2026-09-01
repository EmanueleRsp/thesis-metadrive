# Specification: Falsified rulebook and rank-preserving per-step scalarization

## Metadata

- Feature: `rulebook_v2_falsified_redesign`
- Specification ID: `RULEBOOK-V5.0`
- Version: `5.0`
- Status: `UNDER_REVIEW`
- Date: `2026-08-10`
- Supersedes, on approval:
  - `docs/specifications/rulebook_v4.7_specification.md` §6.4, §7.1, §7.2, §7.3,
    §7.4 and §7.5 (the sub-rule definitions this document redefines or removes);
  - `docs/specifications/rulebook_v4.13_specification.md` in full for the
    `wrongway` subject, which is deleted rather than reformulated;
  - `docs/specifications/rulebook_scalarization_v1.1_specification.md` §7.6
    (the priority-weighted severity term).
  Every other section of v4.7 and of the v4.8–v4.13 amendments, including the
  macro-rule grouping, the `max` aggregation, the applicability mechanism and
  the R4 progress definition, remains authoritative and unchanged.
- Related ADRs (new, drafted with this document): `ADR-063` .. `ADR-069`.
  Supersedes `ADR-060` (the unified wrong-direction sub-rule is deleted, not
  reformulated) and withdraws `DEC-RSEC-001` (the R2 cost deadband).
- Related specifications: `docs/specifications/evaluation_protocol_v1.3_specification.md`
  (§2.3 excludes a per-step stagnation penalty from the reward; this document
  depends on that exclusion), `docs/specifications/observation_v1.3_specification.md`
  and `docs/specifications/observation_lidar_v2.0_specification.md` (§7 below
  amends both).
- ExecPlan: `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`
- Measurement evidence: `scripts/measure_expert_rulebook_transition.py`,
  1100 Waymo `train` records, 217,189 transitions, 0 skipped, 0 errors.
- Approval evidence: **pending**. This document is not an implementation
  contract until the user approves it and it is promoted to `APPROVED` /
  `Authoritative: YES`.
- Authoritative: `NO`.

---

## 1. Purpose

A policy trained under the current reward learns to stand still and only moves
once the logged scenario ends. Every previous attempt to fix this adjusted a
parameter. This document instead **falsifies the rulebook against the logged
human expert** and rebuilds only what fails.

The finding that motivates the redesign, measured rather than argued:

> Replaying the logged Waymo self-driving car through the production rulebook,
> the **human expert's mean episode return is −203.35**, and **46.55 % of expert
> episodes score below a policy that never moves**.

A reward under which competent human driving is worse than doing nothing cannot
rank policies by driving quality. No amount of training fixes that, and no
training curve is evidence about it: the measurement below uses logged poses
only and is therefore a property of the metric definitions, independent of any
policy.

### 1.1 Relationship to the literature

Knox, Allievi, Banzhaf, Schmitt and Stone, *Reward (Mis)design for Autonomous
Driving* (Artificial Intelligence, 2023), give eight sanity checks for
autonomous-driving reward functions. Their check 7, *trial-and-error reward
design*, is the one this document is built to avoid:

> "the specification of the reinforcement learning problem should not in
> principle be affected by the preferred solution to the problem"

Their objection is to tuning a reward by observing the trained agent. This
document tunes nothing against agent behaviour: every decision below is taken
against **logged expert trajectories**, on the `train` split only, and every
threshold that has a published anchor takes the published value rather than the
value that would make the numbers work. Where no threshold survives, the rule is
removed and the gap is declared, not papered over with a fitted parameter.

Calibrating metric thresholds on expert data is standard in this field rather
than a compromise: nuPlan's comfort bounds are, in the devkit's own words,
*"thresholds with default values determined empirically from examination of a
dataset of expert trajectories"*. The novelty here is only that the same
instrument is applied to a **reward** rather than to an evaluation metric.

---

## 2. Method: two falsification tests

Both tests can only **reject**. Passing one is not evidence that a rule is good;
failing one is evidence that it is unusable.

### 2.1 Test A — admissibility

> A rule that carries a satisfaction indicator must be satisfiable by a
> competent driver.

Operationally: replay the logged expert and measure the fraction of steps on
which the rule is violated. A rule the expert violates on a large fraction of
steps is charging the agent for driving, not for driving badly.

Test A cannot validate a threshold, because a rule the expert never violates may
still be vacuous. It can only reject.

### 2.2 Test B — observability

> A rule may carry memory, but then the observation must carry the same state,
> and that state must be plausible as a real perception output.

Three branches, in order of preference:

1. **express the rule memorylessly on the transition** — preferred;
2. **bounded memory whose state is a plausible mid-perception feature** —
   admissible, provided the observation is extended to carry it;
3. **otherwise** — the rule becomes a diagnostic and leaves the reward.

Branch 2 is what the non-Markovian-reward literature does: in NMRDPs (Bacchus,
Boutilier & Grove, 1996) and in reward machines (Toro Icarte et al., ICML 2018 /
JAIR 2022) the automaton state is always **exposed to the agent**, never hidden
inside the reward. A latch that the agent cannot see makes the reward
non-Markovian in the agent's own state space.

The plausibility requirement is a scientific constraint, not an engineering one:
an observation feature invented purely to make a reward rule computable does not
correspond to anything a perception stack produces, and a policy trained on it
would not transfer.

### 2.3 Controlled invariance — the decisive structural criterion

A region of state space is admissible as a **per-step penalty** only if from
every state inside it there exists an action that keeps the ego inside the
admissible set.

- Road surface, lane direction, solid markings, signals and priority are
  controlled-invariant: nothing another agent does forces the ego across them.
- **The longitudinal safe-distance envelope of other vehicles is not.** A
  cut-in places the ego inside the envelope instantaneously, and no action
  available this step restores the gap.

A planner that meets an infeasible constraint reports infeasibility and executes
a fallback. A per-step reward has no such mechanism: it simply keeps charging.
Hard constraints in this repository are expressed by **termination**, which is
the mechanism that already exists for the unavoidable case.

---

## 3. The measurement instrument

`scripts/measure_expert_rulebook_transition.py` drives the logged Waymo SDC
track through production's own `evaluate_transition` — the same entry point the
live wrapper calls — with the real `RulebookMemory` threaded across steps. All
twelve normative sub-rules are therefore evaluated exactly as in training,
including the four traffic-control rules and their persistence latches.

Two things are deliberately not simulated, and are reported as such:

- **R1 `collision`** reads physics contact records, which do not exist offline.
  The replay passes an empty contact set, so R1 is `NOT_MEASURED` rather than
  measured as zero.
- **Ego pose and velocity come from the recorded track**, never from a policy.

Guards, each with a regression test (§10):

- the offline signal-state colour map is pinned to the live MetaDrive reader
  (`TEST-RSEC-014`), and reads only the current step (`TEST-RSEC-013`);
- the measured component list is pinned to the registry (`TEST-RSEC-015`) —
  this guard caught a real defect: the registry registers `wrong_way` while
  `evaluate_wrongway` names its result `wrongway`, so the sub-rule had been
  reading as permanently satisfied;
- every counterfactual and geometric variant is checked, per step, against the
  cost production actually produced, and the run aborts on divergence. This
  guard caught two further defects during development: both RSS rules are scoped
  on the **pre**-transition state while the geometric rules use the post state,
  and the variant family did not canonicalize near-zero margins the way
  `scalarize_rulebook_margins` does.

Sample: the frozen selection index, `split=train`, `source=waymo`, 1100 records,
217,189 transitions, 0 skipped, 0 errors. Validation and test splits are
untouched, so the evaluation protocol remains uncontaminated by this
calibration.

---

## 4. Evidence

### 4.1 The production rulebook on the logged expert

Episode return under `SCAL-V1.1` (`bounded_priority_weighted_rank`, base 3):

| p1 | p5 | p10 | p25 | p50 | p90 | mean | below standstill |
|---:|---:|---:|---:|---:|---:|---:|---:|
| −1909.63 | −1165.87 | −768.99 | −246.62 | **+5.59** | +56.06 | **−203.35** | **46.55 %** |

By channel — mean and median:

| channel | mean | p50 |
|---|---:|---:|
| R1 collision impact | 0.00 | 0.00 |
| R2 dynamic interaction safety | **−216.87** | 0.00 |
| R3 road/traffic compliance | −26.78 | 0.00 |
| R4 progress | +40.30 | +30.40 |

The R2 median is zero while its mean is −216.87: the damage is **bimodal, not a
uniform offset**. Roughly half the episodes never touch R2 and half are
destroyed. This is worse than a bias, because it is variance in the reward
signal itself. The run is Waymo-only, so the bimodality is not a Waymo/PG
artifact; it tracks traffic density.

### 4.2 Test A, per sub-rule

Fraction of all steps violated / fraction of applicable steps violated / mean
cost on violated steps:

| sub-rule | % all steps | % applicable | mean cost | verdict |
|---|---:|---:|---:|---|
| `rss` | **8.400** | **18.290** | 0.337 | **rejected** |
| `offroad` | 1.190 | 1.190 | 0.097 | tolerance artifact |
| `solid_line` | 1.120 | 1.140 | **1.000** | binary cost artifact |
| `wrong_carriageway` | 0.843 | 0.843 | 0.066 | geometric artifact |
| `dashed_line` | 0.530 | 0.580 | 0.219 | passes |
| `clearance` | 0.327 | 0.560 | 0.441 | scoping artifact |
| `rss_lateral` | 0.312 | 0.720 | 0.610 | passes |
| `vehicle_yield` | 0.190 | 0.440 | 0.934 | latch (§4.3) |
| `ttc` | 0.122 | 0.122 | 0.545 | passes |
| `signal` | 0.020 | 0.060 | 0.540 | latch-free, passes |
| `stop` | 0.010 | 0.120 | 0.900 | passes |
| `crosswalk` | 0.010 | 8.530 | 1.000 | latch; 129 applicable steps only |
| `wrongway` | 0.0000046 | 0.0000046 | 0.003 | 1 step in 217,189 |
| macro R2 | 9.038 | — | 0.350 | |
| macro R3 | 3.018 | — | 0.498 | |

`rss` is the only sub-rule rejected outright by Test A. Its rate is an order of
magnitude above every other rule and two orders above the rules that clearly
pass.

### 4.3 Traffic-control latch attribution

`crosswalk` and `vehicle_yield` both compute `cost = 1.0 if active_latch_for_zone
else approach_cost`. A cost of exactly 1.0 whose `before_gate` diagnostic is
false is therefore the latch branch: the ego is inside the zone being charged
for an entry decision already made, and nothing it does this step changes the
cost.

| branch | steps |
|---|---:|
| `vehicle_yield:latch` | 353 |
| `vehicle_yield:approach` | 55 |
| `crosswalk:latch` | 11 |
| `crosswalk:approach` | 0 |
| **latch share** | **364 / 419 = 86.9 %** |

### 4.4 Counterfactual rulebooks

Each is recomputed exactly from the same per-step component costs in the same
pass — no re-simulation, no extrapolation.

| rulebook | p1 | p5 | p10 | p50 | mean | below standstill |
|---|---:|---:|---:|---:|---:|---:|
| production | −1909.63 | −1165.87 | −768.99 | +5.59 | −203.35 | 46.55 % |
| latches removed only | −1909.63 | −1165.87 | −761.92 | +5.83 | −201.42 | 45.73 % |
| `rss` removed only | −699.58 | −210.84 | −75.13 | +18.85 | −5.29 | 19.64 % |
| + `wrongway`, `dashed_line`, latches | −699.58 | −172.23 | −63.05 | +19.80 | −1.04 | 18.09 % |
| + geometric tolerances | −462.53 | −96.47 | −5.25 | +24.63 | +15.72 | 10.55 % |
| + `clearance` scoping | −434.44 | −58.34 | +3.87 | +26.00 | +19.90 | 9.27 % |

Three conclusions follow directly:

1. **Removing `rss` alone accounts for ~97 % of the deficit** (−203.35 →
   −5.29). The controlled-invariance diagnosis is confirmed quantitatively.
2. **The latch is worth ≈ 2 points of 203.** It is 86.9 % of the traffic-control
   cost, but the traffic-control cost is small. It is removed for observability
   and credit assignment (§5.4), **not** for scale.
3. **The R3 geometric tolerances are load-bearing, not polish.** Without them
   18.09 % of expert episodes still lose to standing still; with them, 10.55 %.

### 4.5 The specified rulebook, measured exactly

The rows in §4.4 are intermediate counterfactuals. The rulebook **exactly as
specified in §5 and §6** — `clearance` scoped, `rss_lateral` unchanged, `ttc` at
0.95 s, the three R3 geometric redefinitions, `dashed_line` kept, `wrongway`
dropped, latches dropped, `speed_limit` added, scalarized at `a = 2.2, σ = 0,
φ = 0.25, λ = 1` — measures:

| rulebook | p1 | p5 | p10 | p50 | mean | below standstill | R2 % | R3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| production | −1909.63 | −1165.87 | −768.99 | +5.59 | **−203.35** | **46.55 %** | 9.038 | 3.018 |
| specified, at-fault gate removed | −190.25 | −23.13 | +4.57 | +25.01 | +29.09 | 8.82 % | 0.6068 | 1.1796 |
| specified, gate at 5e-03 m/s | −150.85 | −16.45 | +4.84 | +25.48 | +30.74 | 7.64 % | 0.4383 | 1.1796 |
| **specified (§5, §6), gate at 5e-02 m/s** | **−150.85** | **−15.86** | **+4.90** | **+25.72** | **+31.14** | **7.45 %** | **0.3978** | **1.1796** |

The last three rows differ **only** in the ADR-070 at-fault gate. R3 is identical
to six figures across all of them, which is the structural confirmation that the
gate reaches the interaction sub-rules and nothing else.

**The threshold choice is robust.** Between nuPlan's two published values the p1
is identical and the mean differs by 0.4 in 31: most of the effect is already
present at the smaller threshold, so the result does not rest on a fitted
constant. 5e-02 is adopted because it is the *blame* threshold
(`_get_collision_type`, "Threshold for 0 speed due to noise") while 5e-03 guards
a division in the TTC metric, and blame is the principle being imported.

This is the number the acceptance criteria in §9 are evaluated against. It is
lower than the §6.4 grid row for the same scalarization member (+31.50) because
that row was computed on the intermediate `clearance`-scoped-plus-`rss_lateral`-
scoped rulebook with the original TTC thresholds and without `dashed_line` or
`speed_limit`; the specified rulebook is stricter on four counts and is
correspondingly more expensive. The stricter number is the one that stands.

### 4.6 The geometric redefinitions

| sub-rule | production | redefined | change |
|---|---:|---:|---|
| `offroad` | 1.190 % | **0.593 %** | 0.3 m tolerance band |
| `solid_line` | 1.120 %, cost **1.000** | **0.349 %**, cost 0.328 | graded penetration, tolerance 0.3 |
| `wrong_carriageway` | 0.843 % | **0.000 %** (0 of 217,189) | ego **centroid** inside the exclusive opposing surface |
| `clearance` | 0.327 % | **0.199 %** | only VRU whose centroid is on the drivable surface |
| `rss_lateral` | 0.312 % | 0.267 % | neighbour's inward motion may not raise the requirement |

Solid-line tolerance sweep (the detection buffer is the marking's real half
width, 0.075 m, not production's 1 cm numerical epsilon):

| tolerance | 0.1 | 0.2 | **0.3** | 0.4 |
|---|---:|---:|---:|---:|
| % steps | 0.599 | 0.430 | **0.349** | 0.249 |

`wrong_carriageway` at **exactly zero violations over 217,189 steps** is the
sharpest single result in the campaign: the expert never places its centre on
the opposing carriageway. Production's any-overlap criterion was measuring
bounding-box corners clipping the opposing surface in curves — geometric noise,
not a normative event.

`rss_lateral`'s 14 % improvement is **not** adopted (§5.3): the measured benefit
does not justify a further deviation, and the hypothesis behind it — that the
cost came from the neighbour's motion — was falsified by this very measurement.

### 4.7 The longitudinal channel: what was tried and rejected

**Time headway.** `THW = gap / v_ego`, swept over six thresholds and four
minimum-ego-speed gates, as a percentage of applicable steps:

| threshold | gate 0.5 m/s | gate 2 m/s | gate 5 m/s | gate 8 m/s |
|---:|---:|---:|---:|---:|
| 0.3 s | 1.70 | 1.95 | 2.27 | 1.55 |
| 0.5 s | 2.80 | 3.24 | 3.81 | 2.67 |
| 1.0 s | 5.92 | 6.83 | **8.19** | 6.51 |
| 1.5 s | 11.89 | 13.78 | 17.37 | 18.49 |
| 2.0 s | 27.84 | 33.06 | 43.99 | 50.67 |

The rate **rises** when low speeds are excluded. The hypothesis that the expert
fails THW because it queues at crawl speed is therefore falsified: the expert
holds short headways precisely at urban flow speeds. No literature-anchored
threshold (1 s, 2 s) approaches the 0.1–0.7 % band of the rules that pass.

At urban speeds RSS with ρ = 1 s is algebraically a headway rule of ≈ 1.6–1.9 s
plus a constant, so the RSS response-time sweep and the THW threshold sweep vary
the same quantity under two parameterizations. Choosing ρ to make the expert
pass would be a claim about the ego's physical reaction time selected because it
makes the numbers work — exactly Knox et al.'s check 7.

**Responsive RSS.** RSS's own answer to the controlled-invariance objection is
the *proper response*: the ego must brake, not maintain a distance. The literal
proper response carries an unbounded per-actor latch and is not Markovian. Its
memoryless projection is: charge only when the gap is unsafe **and the ego is
not decelerating**. From every state inside the envelope the action "brake" is
available and zeroes the cost, so the region becomes controlled-invariant and
the rule prices a decision rather than a state.

Percentage of applicable steps:

| ρ | no response condition | **decelerating** | > 0.5 m/s² | > 1 m/s² |
|---|---:|---:|---:|---:|
| **1.0 s (published)** | 18.29 | **7.17** | 13.31 | 16.78 |
| 0.5 s | — | 2.27 | 4.25 | 5.23 |
| 0.3 s | — | 1.58 | 3.07 | 3.78 |

The response condition removes 61 % of violations at published parameters — the
construction works — but 7.17 % is still ~25× the band of the rules that pass.
The reason is visible in the data: the expert sits **stably** inside the
envelope at constant speed, where no response is due because nothing is
happening. The RSS envelope is calibrated for the leader braking at 8 m/s²; real
following relies on that braking not occurring.

**Conclusion: no longitudinal safe-distance rule is admissible as a per-step
penalty at published parameters, not even in its controlled-invariant
projection.** This is a negative result and is reported as one (§8, §11).

### 4.8 TTC threshold sweep

Uniform across actor classes, so the values are comparable with nuPlan's single
0.95 s bound (production uses 0.8 s for vehicles, 1.0 s for VRU):

| threshold | 0.4 s | 0.6 s | 0.8 s | **0.95 s** | 1.2 s |
|---|---:|---:|---:|---:|---:|
| % steps | 0.056 | 0.077 | 0.118 | **0.156** | 0.258 |

TTC needs no tolerance: at every swept threshold it stays an order of magnitude
below the other rules. Moving to 0.95 s costs 0.04 percentage points and buys a
published anchor in place of a repository-chosen value.

### 4.9 Speed limit

Waymo records carry genuine posted limits converted from mph (24.14 km/h =
15 mph, 40.23 = 25, 72.42 = 45, 80.47 = 50), present on essentially every lane
and accompanied by the source datum `speed_limit_mph`. All 1100 Waymo train
records carry a limit on the assigned route.

| tolerance | 0 | 1.0 m/s | **2.23 m/s (nuPlan)** | 4.47 m/s |
|---|---:|---:|---:|---:|
| % steps | 2.200 | 0.002 | **0.000** | 0.000 |

At nuPlan's published tolerance the expert **never** violates it, over 217,187
applicable steps.

#### 4.9.1 The PG panel carries a value, and it is not a speed limit

An earlier draft of this specification stated that PG carries only the 1000 km/h
sentinel. That is wrong, and the correction is why §5.7 gates on provenance
rather than on value. Measured over the 1100 PG train records, **697 carry a
route-lane `speed_limit_kmh`, and every one of them is exactly 20.**

The value is a lane-constructor default, and which one depends on the code path
that built the lane:

| producer | value | PG blocks |
|---|---:|---|
| `metadrive/component/lane/abs_lane.py:22` — `self.speed_limit = 1000  # should be set manually` | 1000 | straights, first block |
| `metadrive/component/pgblock/create_pg_block_utils.py:26,45` — `speed_limit: float = 20` | 20 | curves, intersections |

This reproduces the observed per-lane histogram exactly, and the 697 are the
records whose assigned route passes through a curve or an intersection.

The unit is also not the one the exported key claims. MetaDrive's PG blocks
document their limits **in m/s** — `ramp.py:33`
(`SPEED_LIMIT = 12  # 12 m/s ~= 40 km/h`) and `tollgate.py:19`
(`SPEED_LIMIT = 3  # m/s ~= 5 miles per hour`), both self-consistent and both
physically absurd under the km/h reading — while the export writes
`lane.speed_limit` out verbatim under a `_kmh` key with no conversion
(`node_road_network.py:321`, `edge_road_network.py:127`). The only unit
conversion in MetaDrive is `mph_to_kmh`, on the real-map read path. The one
contradicting site, `base_vehicle.py:963` (`lane.speed_limit < self.speed_km_h`),
is a property with no caller.

**This is an upstream defect, not one this project introduced**, and its
consequence here was measured rather than assumed. Reading the field at its
label turns 20 m/s into 5.56 m/s; the PG panel's own reference driver —
MetaDrive's `IDMPolicy` at `NORMAL_SPEED = 30` km/h, whose logged speed is
p50 = p90 = 8.33 m/s, max 8.91 — then exceeds `limit + τ` on **51.21 %** of PG
steps against **0.23 %** on Waymo under the same route-lane approximation. Under
the m/s reading the same rule fires on 0 %. Neither reading is usable: one is off
by 3.6x, the other prices a constructor default as a traffic norm. §5.7 therefore
admits a limit only where the real-map provenance datum is present, and ADR-068
records the three alternatives rejected.

---

### 4.10 What the residual below-standstill tail is made of

The specified rulebook leaves **82 of 1100** expert episodes below standstill
(7.45 %). This section measures what that tail consists of. Both accountings
below are computed on the specified rulebook itself, not carried over from the
§4.4 counterfactuals, which differ from it in four sub-rules.

§4.10.1 and §4.10.2 report the rulebook **as specified**, i.e. with the ADR-070
at-fault gate. §4.10.3 reports the same measurement **without** the gate, because
that is the evidence that motivated it, and closes with the before/after.

Attribution is by differencing: for each step, the reward the same step would
have earned with one channel satisfied, minus the reward it actually earned.
Under `max` aggregation exactly one sub-rule sets each channel's cost, so that
difference is wholly attributable to it, and with `σ = 0` the two channel terms
are separable, so the two differences sum to the step's total penalty. The
scalarization weights are therefore never restated in the accounting.

#### 4.10.1 Which sub-rule owns the tail

| sub-rule | episodes it dominates (of 82) | share | penalty mass inside them (reward units) |
|---|---:|---:|---:|
| `rss_lateral` | 31 | 37.8 % | 2653.4 |
| `dashed_line` | 14 | 17.1 % | 2253.5 |
| `solid_line` | 10 | 12.2 % | 1460.1 |
| `clearance` | 9 | 11.0 % | 442.9 |
| `offroad` | 9 | 11.0 % | 780.6 |
| `ttc` | 9 | 11.0 % | 575.2 |
| `signal` | 0 | — | 21.1 |
| `stop` | 0 | — | 4.9 |
| `vehicle_yield` | 0 | — | 2.2 |
| `speed_limit` | 0 | — | **0.0** |

**The tail is not dominated by the proximity rules.** `rss_lateral` is the single
largest owner at 37.8 %, but the lane-marking and road-geometry cluster
`dashed_line` + `solid_line` + `offroad` owns 33 of 82 (40.2 %) — more than it.
That cluster is invisible in the §4.4 `proposed` counterfactual because
`dashed_line` is not part of that rulebook at all, which is precisely why the
attribution had to be measured on the specified rulebook rather than inferred
from a neighbouring one. By penalty mass, `dashed_line` (2253.5) is the most
expensive R3 sub-rule retained.

`speed_limit` contributes **zero** reward units, consistent with its zero
violations at τ = 2.23 m/s (§4.9) and with the provenance gate (§5.7). This is
Test A passing, not evidence of vacuity: the sub-rule exists to close R4's
80 km/h degeneracy, not to fire on the expert (§11.7).

#### 4.10.2 How much of the tail was avoidable by slowing down

Of the **8194.0** reward units of penalty charged inside those 82 episodes, the
share charged while the ego was already at or below a crawl:

| ego speed at the charged step | share, as specified | share, gate removed |
|---|---:|---:|
| ≤ 0.1 m/s (essentially stopped) | **13.9 %** | 31.4 % |
| ≤ 0.5 m/s | 21.0 % | 37.0 % |
| ≤ 1.0 m/s (walking pace) | 27.1 % | 41.9 % |
| ≤ 2.0 m/s | 41.1 % | 53.0 % |

The threshold is swept rather than fixed, so the conclusion does not rest on one
chosen value; the split is monotone by construction, since raising the threshold
can only move mass into the low-speed class.

**As specified, 13.9 % of the residual penalty is charged to an already-stopped
ego, down from 31.4 % before the gate**, and §4.10.3 shows what remains is the
position rules, which are not a defect. Slowing further could not have avoided
it, so it cannot be what makes standing still preferable in those episodes.

This still has a consequence for the baseline. §11.1 notes that standing still
scores exactly 0 here only because R1 is `NOT_MEASURED` offline. The residual
13.9 % shows the 0 is optimistic for a second, independent reason: a genuinely
stopped vehicle still accrues position cost — a footprint resting across a lane
marking — and the 0 baseline credits it with none of that. The gate removed the
*interaction* component of this bias, not all of it.

The complementary 86.1 % was charged while the ego was moving, and was in
principle avoidable by driving differently. What this measurement **cannot**
establish is that it was avoidable *while retaining positive R4* — that requires
evaluating a policy, not replaying a log (§11.3).

#### 4.10.3 Which sub-rule charges the stopped ego, and whether that is a defect

The aggregate split of §4.10.2 says how much of the tail was charged to an
already-stopped ego. It cannot say whether that is **the expert stopping in a bad
place** or **the rulebook charging a stationary ego for something no action of
its own can avoid**. Those have opposite remedies, and only the sub-rule identity
separates them. Penalty mass inside the 97 failing episodes, crossed with ego
speed at the charged step:

| sub-rule | mass in those episodes | at ≤ 0.1 m/s | at ≤ 0.5 m/s | at ≤ 1.0 m/s | at ≤ 2.0 m/s | share at ≤ 0.1 |
|---|---:|---:|---:|---:|---:|---:|
| `clearance` | 2031.4 | **1623.1** | 1756.3 | 1780.6 | 1819.4 | **79.9 %** |
| `ttc` | 872.0 | 326.7 | 351.7 | 376.1 | 395.6 | **37.5 %** |
| `solid_line` | 1460.1 | 420.5 | 572.1 | 677.3 | 815.3 | 28.8 % |
| `dashed_line` | 2253.5 | 610.7 | 788.8 | 915.4 | 1162.1 | 27.1 % |
| `rss_lateral` | 2880.4 | 261.7 | 355.7 | 577.0 | 1154.8 | **9.1 %** |
| `offroad` | 796.1 | **0.0** | 0.0 | 2.2 | 119.0 | **0.0 %** |
| `signal` | 21.1 | 0.0 | 0.0 | 2.4 | 7.1 | 0.0 % |
| `stop` | 4.9 | 0.0 | 0.0 | 0.0 | 2.5 | 0.0 % |
| `vehicle_yield` | 6.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 % |

**`clearance` fails controlled invariance on a stopped ego.** Almost 80 % of its
cost in the failing tail is charged while the ego is at or below 0.1 m/s, and it
alone accounts for 1623.1 of the 3242.8 stopped-ego units — half of them.
`evaluate_clearance` is `max(0, 1 − distance / 1.0 m)` with **no test of relative
motion**: from a state where a pedestrian is 1.2 m away and approaching, with the
ego boxed in traffic, no ego action keeps the distance above 1 m. That is the
same argument that rejected `rss` in §4.7, applied to a rule that survived it.

**`ttc` shows the same failure at 37.5 %**, and it is the sub-rule for which the
reference benchmark ships an explicit remedy this specification did not adopt:
nuPlan's `time_to_collision_within_bound` carries
`stopped_speed_threshold = 5e-03` m/s and returns no value at all when
`ego_speed <= stopped_speed_threshold`. §5.6 adopted nuPlan's 0.95 s threshold
without the speed gate that accompanies it in the source.

**`rss_lateral` does not show this failure** — 9.1 % only, despite being the
single largest contributor overall (2880.4). Its cost is overwhelmingly charged
to a moving ego. This falsifies the natural expectation that the lateral rule is
a stationary-ego problem, and means an ego-speed gate would barely touch it.

**`offroad` is charged at 0.0 %**, the clean confirmation that a position rule
behaves as predicted: the expert never stops off the drivable surface.

**`solid_line` and `dashed_line` are not defects, despite ~28 %.** From a state
stopped astride a lane marking an action that leaves it exists — moving — so the
region stays controlled-invariant. This is the expert stopping across a marking,
which is a normative disagreement, not an unavoidable charge.

**The split.** Of the 3242.8 units charged to a stopped ego:

- **2211.5 (68.2 %) are a rulebook defect** by the controlled-invariance test —
  interaction rules charging a stationary ego: `clearance` 1623.1 + `ttc` 326.7 +
  `rss_lateral` 261.7;
- **1031.2 (31.8 %) are the expert stopping badly** — `solid_line` 420.5 +
  `dashed_line` 610.7.

Against the whole ungated tail: the defect is **21.4 %** of its penalty mass, the
expert stopping badly **10.0 %**, and the remaining **68.6 %** is charged to a
moving ego and is genuine normative disagreement with how the logged human drove.

#### 4.10.4 What the gate actually removed

ADR-070 gates `clearance`, `ttc` and `rss_lateral` on an ego at or below
5e-02 m/s. Measured on the same 1100 records, the same accounting after the gate:

| sub-rule | mass before | mass after | share at ≤ 0.1 m/s, before → after |
|---|---:|---:|---|
| `clearance` | 2031.4 | **442.9** | 79.9 % → **7.8 %** |
| `ttc` | 872.0 | 575.2 | 37.5 % → **5.2 %** |
| `rss_lateral` | 2880.4 | 2653.4 | 9.1 % → 1.5 % |
| `dashed_line` | 2253.5 | **2253.5** | 27.1 % → 27.1 % |
| `solid_line` | 1460.1 | **1460.1** | 28.8 % → 28.8 % |
| `offroad` | 796.1 | 780.6 | 0.0 % → 0.0 % |

**The position sub-rules are unchanged to the last decimal**, which is the
structural confirmation that the gate reaches what it was scoped to reach and
nothing else. `clearance` loses 78 % of its mass, which is the predicted result
of removing charges no ego action could avoid.

Episode-level effect: **97 → 82** episodes below standstill, mean **+29.09 →
+31.14**, macro R2 **0.6068 % → 0.3978 %**, macro R3 **unchanged at 1.1796 %**
(§4.5).

What remains charged to a stopped ego is now dominated by `solid_line` and
`dashed_line` — the expert stopping astride a lane marking — which §4.10.3
classifies as a normative disagreement rather than a defect. **The defect
portion of the residual is closed; the disagreement portion is not, and is not
meant to be.**

---

## 5. The rulebook

Thirteen normative sub-rules, one fewer than the current twelve-plus-`progress`
registry, with strictly better coverage. Macro grouping, `max` aggregation and
the applicability mechanism are unchanged from v4.7.

### 5.1 R1 — collision impact

Unchanged.

### 5.2 R2 — dynamic interaction safety

| sub-rule | change |
|---|---|
| `rss` | **removed from the reward**; retained as a reported diagnostic (§8) |
| `rss_lateral` | **unchanged** |
| `ttc` | threshold becomes **0.95 s uniformly** across actor classes (nuPlan `least_min_ttc`), replacing 0.8 s / 1.0 s |
| `clearance` | scoped: only pedestrians and cyclists whose **footprint centroid lies on the drivable surface** are candidates; the 1.0 m threshold and the graded cost are unchanged |
| all three | **inapplicable when `v_ego ≤ 5e-02 m/s`** (ADR-070) |

`clearance`'s scoping uses the same centre-entry criterion as
`wrong_carriageway`. A person waiting on the kerb of a narrow street is inside
1 m of every passing vehicle and is in conflict with none of them; the unscoped
rule priced normal urban driving.

**`REQ-RB5-R2-GATE`.** `clearance`, `ttc` and `rss_lateral` are **inapplicable**
whenever the ego speed is at or below `5e-02 m/s`, using the applicability
mechanism that already exists — not scored as satisfied at zero cost, which
would be a different and wrong statement.

These three sub-rules have their cost set by another agent's state. From a
stationary ego no action avoids what another agent brings to it, so the region
they define is **not controlled-invariant** — the same test that rejected `rss`
in §4.7, which §4.10.3 measured them failing (`clearance` charges 79.9 % of its
tail cost to a stopped ego). The gate restores controlled invariance by removing
exactly the states in which it fails.

The threshold is nuPlan's published at-fault value
(`no_ego_at_fault_collisions._get_collision_type`, `stopped_speed_threshold =
5e-02`, "Threshold for 0 speed due to noise"), under which
`STOPPED_EGO_COLLISION` is excluded from at-fault. It is deliberately tiny:
0.05 m/s is 0.18 km/h, at which R4's margin is under 0.5 % of its maximum, so
crawling under the gate buys immunity at zero progress. **A gate at any of
§4.10.2's larger sweep thresholds would be exploitable and must not be used.**

The position sub-rules (§5.4) and the traffic-control sub-rules are **not**
gated: from a state stopped astride a lane marking an action that leaves it
exists, so the charge is a normative disagreement, not an unavoidable cost.

### 5.3 R2 — what is deliberately not changed

`rss_lateral` keeps production's `d_safe^lat`, including the neighbour's inward
displacement term. The scoped variant that removes the neighbour's ability to
raise the ego's requirement is implementable in two lines and is a strict
relaxation (`TEST-RSEC-017`), but its measured benefit is 0.312 % → 0.267 %.
The hypothesis that motivated it — that the cost came from the neighbour's
motion — is falsified by that same number: the cost comes almost entirely from
the ego's own lateral motion, which it is legitimate to charge. One fewer
deviation to justify, for a benefit that does not appear in the data.

### 5.4 R3 — road and traffic compliance

| sub-rule | change |
|---|---|
| `offroad` | area fraction outside the drivable surface **widened by 0.3 m** (nuPlan `drivable_area_compliance`) |
| `solid_line` | cost becomes **graded lateral penetration** with tolerance 0.3, replacing the binary 1.0 on any contact; detection buffers the marking by its real half width 0.075 m |
| `wrong_carriageway` | applies only when the ego **centroid** lies inside the exclusive opposing surface; the invaded-area fraction is otherwise unchanged |
| `dashed_line` | **retained unchanged** (§5.5) |
| `wrongway` | **removed** (§5.6) |
| `signal`, `stop`, `crosswalk`, `vehicle_yield` | **persistence latches removed**; the memoryless approach term is retained |
| `speed_limit` | **new** (§5.7) |

The 0.3 m off-road band exists because the bounding box over-approximates the
vehicle — a measurement artifact, not permissiveness about leaving the road.
The 0.3 solid-line penetration tolerance corresponds to roughly the same 0.3 m
of lateral slack for a 1.85 m wide vehicle, so the two are one decision rather
than two independent calibrations.

**Latch removal.** A latch is a state variable set by an event that stays set
until another event clears it. `crosswalk` and `vehicle_yield` latch to cost 1.0
once the ego is inside the zone, so the agent is charged every step for an entry
decision taken earlier, with no action available that reduces the cost. That is
both a credit-assignment defect and a Test B branch-3 failure: the latch state
is not in the observation. The memoryless approach term — which prices the
decision to enter while it is still being made — is retained.

### 5.5 `dashed_line` is retained, and costs nothing to observe

`cost = penetration × time_factor`, where the time factor ramps from 0 at
`DASHED_T0_S = 1.0 s` to 1 at `DASHED_TCAP_S = 2.0 s` of continuous contact with
the same marking. The memory exists for a reason: a lane change legitimately
crosses a dashed marking; sustained straddling does not.

This is Test B branch 2, and the branch is already satisfied: **both observation
paths already carry 21 steps of history, and the 21 was derived from
`DASHED_TCAP_S` itself.** `stacked_lidar_v2.py` states it in its own module
docstring, and `conf/obs/semantic_v3.yaml` sets `context_history_length: 21`. At
the 10 Hz control period 21 samples span 2.1 s, so the timer saturates strictly
inside the window and is reconstructible from the observation. No observation
change, no schema change, no ADR on the observation contract.

Sustained straddling of a lane marking was an explicit supervisor requirement.

### 5.6 `wrongway` is removed, not reformulated

The rule fires on **1 step in 217,189**. Reverse motion on the assigned route is
already covered by `wrong_carriageway` (direction relative to the carriageway)
and by R4, whose margin is negative for negative route advance. A sub-rule that
is empirically inert and semantically duplicated adds surface without adding
specification. `ADR-060`, which reformulated it to be memoryless, is superseded:
the correct action is deletion, not reformulation.

### 5.7 `speed_limit` is added

**Definition.** With `v_limit` the posted limit of the ego's associated route
lane and `τ = 2.23 m/s` (5 mph, nuPlan `speed_limit_compliance`):

```
cost = clip( (v_ego − (v_limit + τ)) / v_limit , 0, 1 )
```

The excess is normalized by the limit so a given cost means the same relative
overspeed on a 15 mph street and a 45 mph road.

**Applicability is decided by provenance, not by value.** A lane speed limit is
normative only where the record also carries the real-map `speed_limit_mph`
datum it was converted from. Every ScenarioNet producer writes
`speed_limit_kmh`, but only a real-map converter fills it from a posted limit;
MetaDrive's PG exporter writes out whichever default the lane constructor
happened to hold, under the same key, without converting the unit its own blocks
document in m/s (§4.9). The gate subsumes the rejection of the unrecorded `0.0`
and of the `≥ 999` sentinel, and additionally rejects every PG default. See
ADR-068 for the upstream defect and for the three alternatives considered.

Consequence, stated rather than hidden: the rule is inapplicable throughout the
PG panel, so the degeneracy it closes is closed on the Waymo half of the
training mixture only (§11.5).

**Why it is needed rather than merely nice.** R4's margin is
`clip(Δs / (22.22 · Δt), −1, 1)` — the ratio of route speed to a reference speed
of 80 km/h. The reward therefore **directly incentivizes driving at up to
80 km/h, including on a 25 mph street**, and nothing in the current rulebook
opposes it except the consequences of leaving the road or colliding.
`speed_limit` is not an addition for completeness; it closes a degeneracy that
R4's own definition creates.

**Why it is admissible.** It passes every test that rejected `rss`:
controlled-invariant (the ego can always decelerate, and no other agent can push
it above the limit); memoryless; observable and physically plausible (the posted
limit is an HD-map attribute every production stack carries); anchored in
published values rather than fitted; and violated on **0.000 %** of expert
steps.

**Fallback prohibition.** The v1 extractor at
`src/thesis_rl/envs/wrappers.py:462` falls back to `ego_vehicle.max_speed_km_h`
when the lane carries no limit. That constant is the vehicle's own cap (80 km/h),
not a legal limit; reusing it would make the rule a vehicle cap disguised as a
norm. The fallback must not be carried over: absent a recorded limit, the rule
is inapplicable.

### 5.8 R4 — progress

Unchanged.

---

## 6. Scalarization

### 6.1 The construction and what the per-step setting forces

Veer, Leung, Cosner, Chen and Pavone, *Receding Horizon Planning with Rule
Hierarchies for Autonomous Vehicles* (ICRA 2023), Theorem 1, define the
**rank-preserving reward**

```
R(ρ) = Σ_{i=1..N} ( a^(N−i+1) · step(ρ_i) + (1/N) · ρ_i ),   a > 2,  ρ_i ∈ [−a/2, a/2]
```

with `step(x) = 0` for `x < 0` and `1` for `x ≥ 0`, robustness values normalized
by `ρ = tanh(ρ_STLCG / s)`. Remark 1 replaces `step` with `sigmoid(c·ρ)` for
differentiability — a **smoothed** indicator, not its removal. The satisfaction
indicator is the heart of the theorem, and rank preservation fails without it.

Using this per **step** rather than per **trajectory** forces exactly two
adaptations, both of which this specification states explicitly:

1. **The step term must be shifted** so a satisfied step scores 0 rather than
   `+a^e`. Unshifted, a stopped ego collects the entire priority stack every step
   forever — the standing-still degeneracy in its purest form. For trajectory
   ranking the shift is a harmless constant; for a per-step sum it is not,
   because episodes differ in length.
2. **Progress cannot remain a `1/N` tie-breaker.** For Veer et al. progress
   breaks ties among equally rule-compliant trajectories. Here progress is the
   task.

### 6.2 `SCAL-V1.2`: the parametrized family

```
r = Σ_{k=1..3} a^(4−k) · [ (step(m_k) − 1) + σ · m_k ]  +  φ · Σ_{k=1..3} m_k  +  λ · m_4
```

with `m_k ∈ [−1, 0]` for `k ≤ 3`, `m_4 ∈ [−1, 1]`, and margins canonicalized to
exactly 0 within the scalarization's numerical tolerance before `step` is
applied. `σ` scales severity **inside** the priority weight (`SCAL-V1.1`'s
form); `φ` is Veer et al.'s own flat `1/N` tie-breaker, **outside** it.

### 6.3 Rank-preservation condition

One step's ordering is lexicographic in the three cost channels if and only if,
for every level `k`:

```
a^(4−k)  >  (1 + σ) · Σ_{j>k} a^(4−j)  +  φ · (3 − k)  +  2λ
```

A violation at level `k` scores at best `−w_k + λ`; the same level satisfied
scores at worst `−(1+σ)·Σ_{j>k} w_j − φ·(3−k) − λ`, because a violated lower
level ranges over `[−w_j(1+σ) − φ, −w_j)`.

The condition reproduces both published anchors without being told them:

- at `σ = φ = 0, λ = 1` it reduces to **`a > 2`**, which is Veer et al.'s own
  condition;
- at `σ = 1, λ = 1` it forces **`a ≥ 2.92`**, which is why `SCAL-V1.1` §7.6 had
  to re-derive the base as 3 when it moved severity inside the priority weight.

`λ` enters the condition as `2λ`. That is where the asymmetry between the three
cost channels (bounded in `[−1, 0]`) and the progress channel (bounded in
`[−1, 1]`) is resolved: **the hierarchy itself states how much progress weight it
can tolerate.**

The predicate is implemented as a decision procedure and verified against an
exhaustive counterexample search (`TEST-RSEC-019`). It is deliberately
**conservative**: it compares against the supremum of the violating case, which
is approached as the margin tends to zero but never attained, so it rejects
knife-edge members that enumeration cannot break. Refusing an admissible member
costs a little return; admitting an inadmissible one would silently destroy the
hierarchy.

### 6.4 Selected member

**`a = 2.2`, `σ = 0`, `φ = 0.25`, `λ = 1`.**

Measured on the same rulebook and the same per-step costs, varying only the
scalarization:

| member | p1 | p5 | p10 | p50 | mean | below standstill |
|---|---:|---:|---:|---:|---:|---:|
| a=2.01, σ=0, φ=0 | −128.26 | −7.07 | +5.28 | +26.87 | **+32.87** | 6.27 % |
| **a=2.2, σ=0, φ=0.25** | **−155.38** | **−13.38** | **+5.08** | **+26.36** | **+31.50** | **7.18 %** |
| a=2.5, σ=0, φ=0 | −195.08 | −20.94 | +4.90 | +26.15 | +29.77 | 8.00 % |
| a=2.5, σ=0.5, φ=0 | −243.99 | −28.47 | +4.83 | +26.12 | +27.55 | 8.45 % |
| a=3, σ=0, φ=0 | −276.79 | −39.04 | +4.62 | +26.09 | +26.08 | 8.73 % |
| a=3, σ=0.5, φ=0 | −335.29 | −50.27 | +4.38 | +26.01 | +22.99 | 9.18 % |
| **a=3, σ=1, φ=0 — `SCAL-V1.1`** | −434.44 | −58.34 | +3.87 | +26.00 | **+19.90** | 9.27 % |
| *(no indicator — not rank-preserving)* | *−118.27* | *+2.14* | *+5.84* | *+28.15* | *+34.11* | *4.91 %* |

Rejected by the condition and therefore never evaluated: `a = 2.01` and
`a = 2.2` with any `σ > 0`, and `a = 2.5` with `σ = 1`.

**Rationale for `a = 2.2, σ = 0, φ = 0.25` over the marginally better `a = 2.01`:**

1. `a = 2.01` is on the edge. The level-3 constraint is `a > 2λ`; at `λ = 1` the
   margin is **0.01**. Any later change — an added priority level, a different
   progress weight — breaks it. At `a = 2.2` the margin is 0.2.
2. `a = 2.01` **cannot afford Veer et al.'s own tie-breaker**: `φ = 0.25` at base
   2.01 with `λ = 1` is rejected by the condition. So "a = 2.01, φ = 0" is not
   the published formula — it is the published formula stripped of its severity
   term, which is the term that supplies a gradient toward compliance inside a
   violation.
3. `a = 2.2, σ = 0, φ = 0.25` is Veer et al.'s structure intact — priority
   weight on the step term, severity as a flat `1/N` tie-breaker — with the
   **single** adaptation `λ: 1/N → 1` that §6.1 shows the per-step setting
   forces. The flat tie-breaker costs 0.23 points of mean return (31.73 → 31.50).

Removing the indicator entirely yields +34.11, but converts the reward into a
weighted sum with no hierarchy guarantee. Staying inside rank preservation
captures **+31.50 of the +34.11 available against a +19.90 baseline, i.e. 82 %
of the achievable gain**, and `a = 2.01` captures 91 %. The theoretical property
costs roughly one point out of fourteen — there was no trade-off to make, only a
badly chosen base.

### 6.5 What this scalarization does and does not guarantee

It must be described precisely, because the natural shorthand is wrong:

- **Per step**, the ordering is strictly lexicographic in R1 ≻ R2 ≻ R3 ≻ R4.
- **Per episode**, the return is a sum of per-step scores, so it is **not**
  lexicographic. Two R3 violations and one R2 violation of equal magnitude
  exchange at the ratio `a = 2.2`.

The correct phrasing is *"geometric priority weights, strict per-step dominance,
exchange rate `a` at episode level"* — not *"lexicographic ordering"*. The
rulebook literature (Censi et al., 2019; Veer et al., 2023) ranks whole
trajectories; summing a per-step lexicographic score does not preserve
lexicographic order over trajectories.

### 6.6 The design constraint this imposes on the rulebook

With the indicator, a violation at level `k` costs `a^(4−k)` **regardless of
severity**. At the expert's mean progress margin of 0.205 per step, one R2
violation costs ≈ 24 steps of typical progress. A sub-rule that fires on a
fraction `f` of steps therefore costs `a² · f · T` per episode of `T` steps,
independent of how mild the violations are.

**Consequently every R2 sub-rule must fire on well under 0.1 % of steps for the
rulebook to be affordable to a competent driver.** This is the principal design
constraint of the rulebook, it is not visible from the sub-rule definitions, and
it is why frequency (Test A) and economic weight are different quantities.

---

## 7. Observation consequences

Only one, and it is required by `speed_limit`:

**`REQ-RB5-OBS-01`.** The posted speed limit of the ego's associated route lane
must be exposed in the observation, in both the semantic (`OBS-V1.3`) and LiDAR
(`OBS-LIDAR-V2.0`) paths, normalized consistently with the existing speed
features, with an explicit "unavailable" encoding. `D` changes in both paths;
**checkpoint compatibility is intentionally broken**, which is acceptable because
the production runs have not started.

The "unavailable" encoding must be emitted under exactly the condition that makes
the sub-rule inapplicable in §5.7 — the absence of real-map provenance — not
merely for an unrecorded or sentinel value. Otherwise the agent would read a
numeric limit on PG that the reward does not enforce, which is a worse defect
than reading nothing: the observation would assert a norm that no cost backs.
On the PG panel the feature is therefore "unavailable" on every step.

This is Test B branch 2: the posted limit is an HD-map attribute that every
production autonomous-driving stack carries, so it is a plausible mid-perception
feature and not a quantity invented to make a reward rule computable.

`dashed_line` requires **no** observation change (§5.5).

---

## 8. Diagnostics

The following are computed every step, logged, and reported in the evaluation —
and enter no reward channel:

- **`rss` longitudinal** at published parameters (ρ = 1.0 s, `a_max` = 3.5 m/s²,
  `b` = 8.0 m/s²). This is RSS's correct use in the literature, keeps the
  published parameters untouched, and gives the thesis a safety measure
  comparable with other work. The expert's rate, **18.29 % of applicable steps**,
  is the reference against which a trained policy's rate is read.
- **Time headway** at 1.0 s and 2.0 s, with the minimum-ego-speed gate reported.
- **Responsive RSS** (§4.7) at ρ = 1.0 s.
- **Static-obstacle clearance**, already diagnostic-only per `REQ-R2-02`.

Reporting them is the scientifically stronger position: the limitation is
quantified rather than removed. If the trained policy's RSS violation rate is
far above the expert's, that is a result to report.

---

## 9. Acceptance criteria

| ID | Criterion |
|---|---|
| `AC-RB5-01` | The logged expert's mean episode return under the specified rulebook and scalarization is **positive**, on 1100 Waymo `train` records. *Measured: +31.14.* |
| `AC-RB5-02` | The fraction of expert episodes scoring below standstill is **below 10 %**. *Measured: 7.45 % (82 of 1100).* |
| `AC-RB5-03` | The expert's p10 episode return is **positive**. *Measured: +4.90.* |
| `AC-RB5-04` | The macro R2 channel is violated on **< 0.5 %** of all expert steps. *Measured: 0.3978 %.* The criterion is stated on the macro because `max` aggregation makes it the quantity the reward actually charges; the ungated per-sub-rule rates are `rss_lateral` 0.312 %, `clearance` 0.199 % scoped, `ttc` 0.156 % at 0.95 s. |
| `AC-RB5-13` | The ADR-070 gate leaves macro R3 **bit-identical** to the ungated rulebook, proving it reaches only the interaction sub-rules. *Measured: 1.1796 % in both.* |
| `AC-RB5-05` | `wrong_carriageway` is violated on **0** expert steps. |
| `AC-RB5-06` | `speed_limit` is violated on **0** expert steps at τ = 2.23 m/s, and is inapplicable on every lane lacking the real-map `speed_limit_mph` provenance datum — which is every PG lane (§4.9.1). |
| `AC-RB5-07` | The selected scalarization member satisfies the §6.3 condition, verified by the decision procedure and by exhaustive counterexample search. |
| `AC-RB5-08` | The family reproduces `SCAL-V1.1` exactly at `(a=3, σ=1, φ=0, λ=1)`. |
| `AC-RB5-09` | No traffic-control sub-rule reads a latch; the latch attribution counter reports 0 latch-branch steps. |
| `AC-RB5-10` | `rss` produces a reported diagnostic on every step and contributes to no reward channel. |
| `AC-RB5-11` | The `dashed_line` timer is reconstructible from the observation window in both observation paths. |
| `AC-RB5-12` | Validation and test splits are not consulted by any calibration in this document. |

---

## 10. Test matrix

Existing, already passing:

| ID | Subject |
|---|---|
| `TEST-RSEC-013` | Signal states read only the requested step (no lookahead). |
| `TEST-RSEC-014` | Offline signal-state map pinned to the live MetaDrive reader. |
| `TEST-RSEC-015` | Measured components cover every normative registry component. |
| `TEST-RSEC-016` | THW gate excludes slow steps as inapplicable, not as satisfied. |
| `TEST-RSEC-017` | Scoped lateral RSS is a strict relaxation of production. |
| `TEST-RSEC-018` | The scalarization family reproduces `SCAL-V1.1` exactly. |
| `TEST-RSEC-019` | Rank-preservation predicate is sound against exhaustive search. |
| `TEST-RSEC-020` | The family grid admits only rank-preserving members. |
| `TEST-RSEC-021` | Swept RSS safe distance equals production's at ρ = 1 s. |
| `TEST-RSEC-022` | Responsive RSS is zero whenever the ego brakes hard enough. |

Required before implementation is complete:

| ID | Subject |
|---|---|
| `TEST-RB5-01` | `clearance` ignores a VRU whose centroid is off the drivable surface and charges one whose centroid is on it, at identical distance. |
| `TEST-RB5-02` | `solid_line` cost is strictly increasing in lateral penetration and is 0 below the tolerance. |
| `TEST-RB5-03` | `offroad` is 0 for a footprint fully inside the 0.3 m band and positive beyond it. |
| `TEST-RB5-04` | `wrong_carriageway` is 0 when only a corner overlaps and positive when the centroid enters. |
| `TEST-RB5-05` | `crosswalk` and `vehicle_yield` return the approach cost inside the zone, never a latched 1.0. |
| `TEST-RB5-06` | `wrongway` is absent from the registry and from aggregation. |
| `TEST-RB5-07` | `speed_limit` is 0 at `v_limit + τ` and positive above; it is inapplicable on a sentinel, on an unrecorded limit, and on a lane carrying `speed_limit_kmh` without the `speed_limit_mph` provenance datum (the PG defaults 20 and 1000); and it never falls back to the vehicle cap or to `ScenarioLane.MAX_SPEED_LIMIT`. |
| `TEST-RB5-08` | `ttc` uses 0.95 s for every actor class. |
| `TEST-RB5-13` | At `v_ego ≤ 5e-02 m/s` each of `clearance`, `ttc`, `rss_lateral` reports **inapplicable**, not satisfied-at-zero; just above the threshold each reports its normal cost; and `offroad`, `solid_line`, `dashed_line`, `wrong_carriageway` and every traffic-control sub-rule are unaffected at any ego speed. |
| `TEST-RB5-09` | The new scalarization mode rejects a `(a, σ, φ, λ)` that fails §6.3. |
| `TEST-RB5-10` | Per-step lexicographic dominance holds on a fixture covering all three levels. |
| `TEST-RB5-11` | End-to-end: expert replay reproduces `AC-RB5-01`..`AC-RB5-06`. |
| `TEST-RB5-12` | Observation carries the posted speed limit with the unavailable encoding, in both paths. |

Mandatory validation commands: `make test`, `make lint`,
`make format-check PYTHON_QUALITY_PATHS=...` on touched files,
`make rulebook-v2-check`, `make smoke`, `make config`, `make config-gpu`.

---

## 11. Known limitations

1. **R1 is `NOT_MEASURED` offline.** The replay has no physics contacts, so the
   expert's collision channel is zero by construction and standing still scores
   exactly 0. In the live environment standing still is not free: it is
   rear-ended (R1), fails the evaluation protocol's making-progress gate at 0.2,
   and exhausts the horizon. **The expert-versus-standstill comparison is
   therefore conservative**, understating the expert's advantage by an amount
   this instrument cannot measure.
2. **No longitudinal safe-distance rule survives** (§4.7). The reward's
   anticipatory longitudinal coverage is `ttc` (relative velocity) plus R1.
   An agent tailgating at matched speed is not penalized until the leader
   brakes. nuPlan's closed-loop score has no following-distance metric either
   and relies on TTC and collisions, so this matches the reference benchmark —
   but it is a limitation, and `rss` is reported as a diagnostic because of it.
3. **The residual 7.45 % of expert episodes below standstill is characterized,
   but the characterization has a known limit.** Blame attribution and the
   crawl split are reported in §4.10. Two cautions apply to any reading of them.

   First, *the logged human is one sample, not the maximum achievable*. "The
   expert scores below standstill in 7.45 % of scenarios" is not "no policy
   beats standstill in those scenarios". A slower trajectory keeping a wider
   berth could score positive in the same scenario, in which case the reward is
   doing its job — saying the human passed too close — and offers a gradient
   away from standing still. Only if *no* trajectory scores positive there does
   the degeneracy actually survive. §4.10's crawl split bounds this: penalty the
   expert incurred while already at a crawl was not avoidable by slowing, and
   therefore cannot be what makes standing still preferable. What it cannot do
   is prove that the avoidable remainder *is* avoidable while retaining positive
   R4; that would require evaluating a policy, not a replay.

   Second, an imperfect expert score is the norm in this field: the reference
   benchmark's own log-replay expert scores 80/100 on nuPlan's reactive
   closed-loop score, and a learned planner beats it.
4. **`crosswalk` is measured on 129 applicable steps only.** Its Test A result
   is not statistically meaningful; it is retained on its definition, not on its
   measurement.
5. **The calibration is Waymo-only, and the PG half of the training mixture is
   uncalibrated.** Two distinct facts, both consequential.

   *Why PG is excluded from Test A, and why that is correct.* A PG scenario's
   logged ego is produced by MetaDrive's `IDMPolicy`
   (`src/thesis_rl/scenarios/pg/generator.py:42,50`), not by a human. Test A
   asks whether a competent human satisfies a rule; replaying IDM answers a
   different question, and for the longitudinal rule it is circular — IDM is by
   construction a headway-maintaining controller, so RSS and THW on IDM would
   return near-zero and would have "confirmed" exactly the rule the Waymo humans
   rejected at 18.29 % applicability (§4.7). Test A can only reject, and a
   rejection established on human data is not weakened by PG's absence.

   *What the exclusion costs.* The acceptance-side numbers — the 7.45 %, the
   channel violation rates, the residual's composition — are Waymo-train
   statements, while training draws 50 % PG. Sub-rule applicability rates on PG
   geometry are unmeasured. §4.9.1 is a concrete instance of what that gap can
   hide: it was found by inspecting PG data directly, not by the instrument.

   *Why it was not simply measured, and how small the remaining gap is.* The
   offline instrument calls `build_waymo_static_adapter_result`
   (`adapter_version="waymo-v2"`) unconditionally. Run against PG records that
   adapter would not fail: PG lanes carry no `width` field, so it would fall back
   to its 3.5 m default while ignoring the true `polygon` PG does provide,
   silently producing a wrong drivable surface under `random_lane_width` and
   corrupting `offroad`, `solid_line` and `wrong_carriageway`. **That is a
   property of calling the wrong adapter, not of a missing one.**
   `build_pg_static_adapter_result`
   (`src/thesis_rl/rulebook/v2/context/pg_static_adapter.py:144`,
   `adapter_version="pg-v2"`) already exists and already reads `polygon`, falling
   back to a buffered centerline only when the record omits it. The gap is
   therefore the instrument's failure to dispatch on record provenance, not an
   absent capability. An earlier revision of this limitation asserted that a PG
   static adapter had still to be written; that was **wrong** and is corrected
   here. The Test A exclusion above is unaffected — it rests on the IDM
   circularity, which no adapter changes.
6. **Macro-level granularity.** R2 aggregates four sub-rules by `max`, so no
   priority can be expressed *within* R2. A Censi-style rulebook is a pre-order
   over rules rather than four macro levels. The per-sub-rule costs are already
   computed and exposed, so a finer hierarchy is available later without
   re-instrumenting anything — relevant for the planned lexicographic and
   distributional algorithms, which consume the cost vector directly and bypass
   the scalarization entirely.
7. **Test A cannot validate, only reject.** A rule the expert never violates may
   still be vacuous; `speed_limit`'s 0.000 % is evidence of admissibility, not of
   usefulness.
8. **Strict priority makes standing still preferable to the mission. The
   responsibility lies with this document's *hierarchy*, not with its
   scalarization.** This is the most consequential limitation in this document.

   *The exchange rate.* With `a = 2.2` the priority weights are R1 = 10.648,
   R2 = 4.84, R3 = 2.2, while the expert's mean R4 margin is 0.204 per step and a
   whole episode's R4 return averages **+40.30**:

   | channel | cost of one violated step | in steps of typical expert progress | violated steps that cancel an entire episode of R4 |
   |---|---:|---:|---:|
   | R3 | 2.20 | 10.8 | **18.3** (1.8 s at 10 Hz) |
   | R2 | 4.84 | 23.7 | **8.3** (0.8 s) |
   | R1 | 10.65 | 52.2 | **3.8** |

   A legally stopped ego scores **exactly 0**: the §5.2 gate makes the
   interaction sub-rules inapplicable, the position and control sub-rules do not
   fire in-lane, and R4 is 0.

   *Under §6's scalarization the preference is priced, not absolute.* The
   exchange rate is finite and the third column above is its magnitude. Riding a
   solid line for two seconds to pass a parked obstacle costs 44 against a mean
   episode R4 return of +40.30 and therefore loses; the same manoeuvre completed
   in five steps costs 11 and wins. The bias toward inaction is real and is
   quantified by the table, but any claim that this reward *always* prefers to
   stop overstates a bounded quantity.

   *Under a strict lexicographic ordering it is unconditional, and no rulebook
   can fix it.* Such an agent compares channels in order and stops at the first
   that differs. Standing still is exactly 0 on R1–R3, while any trajectory that
   moves through traffic accrues some R2 cost — 0.3978 % of expert steps by the
   §4.5 measurement. R4 is therefore never reached, whatever it contains. This is
   a property of the ordering rather than of the cost definitions: it holds for
   every hierarchy in which stopping is safe, which is every admissible
   hierarchy. A **thresholded** ordering removes it by converting "both within
   budget" into a tie at the safety channels so the lower channels are reached;
   that mechanism is the subject of the remedy discussion below.

   *What §6 does and does not claim.* §6.5 is the governing text and states it
   correctly: the scalarization is strictly lexicographic **per step** and is
   **not** lexicographic over episodic returns, because summing a per-step score
   does not preserve the order over trajectories. Veer et al.'s Theorem 1 ranks
   whole trajectories and does not establish the episodic property. An earlier
   revision of this subsection asserted that the scalarization "is the exact
   lexicographic order written as a scalar"; that contradicted §6.5 and is
   **withdrawn**.

   *It is also a divergence from the framework cited.* Censi et al.'s rulebook is
   a pre-order over **realizations** — complete trajectories. Comparing "stops
   forever" against "completes the mission with a brief violation" as whole
   objects can rank the second higher. §6 imposes the order **per step**, which
   is strictly stronger and is what makes standing still unbeatable. Standing
   still is a stationary zero-return policy, not an absorbing state of the
   dynamics: other agents keep interacting and the episode still ends at its
   horizon. §11.6 notes the related granularity gap.

   *It contradicts this project's own evaluation protocol.* The protocol carries
   nuPlan's `making progress` gate at 0.2. The reward's optimum in such a
   scenario — stopping — fails the criterion the resulting policy is then scored
   by. Reward and evaluation disagree about what a good policy is.

   *There are two remedies, and only the second belongs to the algorithm.*

   **In the hierarchy.** The pathology requires *every* R3 sub-rule to outrank
   R4. That placement is a choice made by this document, not a consequence of
   strict priority. Rules a competent driver may relax in order to complete a
   mission — solid line, opposing carriageway, sustained straddling — can be
   placed **below** progress, leaving above it only the rules that are never
   relaxable. Two trajectories that both complete then tie on the upper channels
   and are separated by the relaxable channel, which is the minimum-violation
   semantics of ref. 15 expressed as a hierarchy rather than as a planner. This
   is a restructure of §3 and would invalidate every macro figure in §4, so it is
   recorded here as the identified structural remedy and deferred to a successor
   version, not adopted.

   **In the algorithm.** Slack must not be introduced into the reward: it would
   destroy the rank preservation of §6.3, the only proved property this
   construction has. The literature's answer is thresholded
   lexicographic ordering, where the thresholds are slack variables stating how
   much worse than optimal is still sufficient (refs. 12, 13, 14): with
   `τ_k > 0` the stopping policy is no longer the unique survivor at levels 2–3,
   and the policy that briefly violates R3 reaches level 4 and wins on progress.

   Four constraints on that remedy, all inherent to TLO rather than to this
   rulebook, and all of which must be stated when it is used:

   1. **The ordering must apply to returns or Q-values, not to immediate costs.**
      A greedy per-step lexicographic action selection reproduces the pathology
      exactly: stopping wins at every individual step regardless of slack.
      Published TLO does apply the threshold to values — objectives are compared
      through clipped values `min(value_i, τ_i)` — so this constraint is
      satisfied by the standard formulation, not something to be engineered.
   2. **The published threshold is per-state per-objective, not an episode
      budget** (ref. 14). The slack is therefore re-spent at every state: it
      admits the necessary manoeuvre, but it provides **no guarantee on total
      violation over an episode**. An episode-level budget is a different object,
      whose optimal policy depends on how much budget has already been spent and
      is therefore not Markovian in the base state.
   3. **A scalar per-objective slack cannot distinguish a necessary violation
      from a gratuitous one.** A `τ₃` large enough to cover the 40 R3-units of
      passing the obstacle also authorises 40 units of unmotivated violation.
   4. **The slack is an unfixed hyperparameter, and uniform values are the
      documented failure mode.** No published value applies, so it must be
      tuned — which reopens the calibration question §1.1 subjects the rest of
      this document to — and ref. 14 reports that the thresholding methods "fail
      to guarantee reaching the goal ... when used threshold/slack values are
      uniform throughout state space", which is what a naive implementation does.

   **The remedy is itself an open problem, and this must not be overstated.**
   Ref. 14 states that existing lexicographic RL approaches "were all noted to be
   heuristics without theoretical guarantees as the Bellman equation is not
   applicable to them", and that TLQ "does not enjoy the convergence guarantees
   of its origin algorithm". Its Proposition 4.1 identifies a failure case where
   TLQ cannot reach the goal at all — though under a precondition (**constrained**
   objective terminating and reachability-based, **unconstrained** objective
   non-terminating) that is *not* this rulebook's configuration, where the
   constrained objectives R1–R3 are dense per-step costs and the unconstrained R4
   is dense per-step progress. The general soundness gap applies here; that
   specific proposition does not, and claiming otherwise would be a coincidence
   this document has not established.

   Ref. 14 also supplies a candidate answer: **Lexicographic Projection
   Optimization**, a policy-gradient method that projects gradients onto
   hypercones instead of learning a value function per objective, with a
   guarantee (their Prop. 5.1) that already-satisfied objectives do not decrease,
   and a principled stopping condition that TLQ lacks.

   The dual formulation does not share limitation 2: **minimum-violation
   planning** (ref. 15) makes task completion a requirement and *minimises*
   violation subject to it, rather than bounding violation and maximising
   progress. A third structural answer is the constrained-MDP formulation
   (ref. 16), where the constraint holds in expectation over the episode rather
   than at every step — a state-wise hard constraint is precisely what makes
   standing still unbeatable. Both are recorded as principled alternatives,
   neither is adopted.

   This limitation is what makes §6 a **deliberate strict baseline** rather than
   the intended final reward: it exhibits the pathology in quantified form
   (the table above), and the thresholded algorithms are the treatment measured
   against it.

---

## 12. References

Literature actually used for a decision or a value in this document:

1. **Censi, Slutsky, Wongpiromsarn, Yershov, Pendleton, Fu, Frazzoli (2019).**
   *Liability, Ethics, and Culture-Aware Behavior Specification using
   Rulebooks.* ICRA. — the priority-structure formalism the macro grouping
   implements; the basis for §6.5's statement that rulebooks rank trajectories.
2. **Veer, Leung, Cosner, Chen, Pavone (2023).** *Receding Horizon Planning with
   Rule Hierarchies for Autonomous Vehicles.* ICRA. arXiv:2212.03323. — Theorem
   1's rank-preserving reward, its `a > 2` condition, the `1/N` tie-breaker and
   the `tanh` normalization; the basis for §6.1–§6.4.
3. **Knox, Allievi, Banzhaf, Schmitt, Stone (2023).** *Reward (Mis)design for
   Autonomous Driving.* Artificial Intelligence 316. arXiv:2104.13906. — the
   eight sanity checks; check 7 (trial-and-error reward design) is the standard
   §1.1 holds this work to, and check 2 (human–reward preference mismatch) is
   what Test A operationalizes against logged data.
4. **nuPlan devkit, `docs/metrics_description.md` (Motional).** — the 0.3 m
   drivable-area tolerance (§5.4), the 0.95 s `least_min_ttc` bound (§5.2), the
   2.23 m/s `speed_limit_compliance` tolerance (§5.7), the 0.2 making-progress
   gate, and the statement that comfort thresholds were determined empirically
   from expert trajectories (§1.1).
5. **Dauner, Hallgarten, Geiger, Chitta (2023).** *Parting with Misconceptions
   about Learning-based Vehicle Motion Planning.* CoRL. — the Val14 log-replay
   expert scores (CLS-NR 94, CLS-R 80) and PDM-Closed's CLS-R 92, used in §11.3
   to establish that an imperfect expert score is normal.
6. **Shalev-Shwartz, Shammah, Shashua (2017).** *On a Formal Model of Safe and
   Scalable Self-driving Cars.* arXiv:1708.06374. — the RSS longitudinal safe
   distance and the proper response, used in §4.6 and §8.
7. **Vogel (2003).** *A comparison of headway and time to collision as safety
   indicators.* Accident Analysis & Prevention 35(3). — THW and TTC are
   empirically near-independent and both are needed; the basis for §4.6's
   treatment of THW and TTC as complementary rather than interchangeable.
8. **Loulizi, Bichiou, Rakha (2019).** *Steady-State Car-Following Time Gaps: An
   Empirical Study Using Naturalistic Driving Data.* Journal of Advanced
   Transportation. — observed steady-state following gaps, used in §4.6 to
   explain why a threshold set at the mode of human behaviour necessarily fires
   on a substantial fraction of steps.
9. **Bacchus, Boutilier, Grove (1996).** *Rewarding Behaviors.* AAAI. and
   **Toro Icarte, Klassen, Valenzano, McIlraith (2018/2022).** *Using Reward
   Machines for High-Level Task Specification and Decomposition in Reinforcement
   Learning* (ICML) / *Reward Machines* (JAIR). — non-Markovian reward
   specification always exposes the automaton state to the agent; the basis for
   Test B (§2.2) and for latch removal (§5.4).
10. **Ziebart, Maas, Bagnell, Dey (2008).** *Maximum Entropy Inverse
    Reinforcement Learning.* AAAI. — the noisily-rational demonstrator
    assumption, used in §11.3 to justify not targeting a zero expert-violation
    rate.
11. **Ng, Harada, Russell (1999).** *Policy Invariance Under Reward
    Transformations.* ICML, and **Berducci et al. (2024).** *HPRS: hierarchical
    potential-based reward shaping from task specifications.* Frontiers in
    Robotics and AI. — the multiplicative-hierarchy alternative considered and
    not adopted in §6, and its policy-invariance guarantee.
12. **Gábor, Kalmár, Szepesvári (1998).** *Multi-criteria Reinforcement
    Learning.* ICML. — the original thresholded lexicographic ordering: an
    objective is optimised only up to a threshold before control passes to the
    next. The origin of the slack idea §11.8 relies on.
13. **Skalse, Hammond, Griffin, Abate (2022).** *Lexicographic Multi-Objective
    Reinforcement Learning.* IJCAI. — the unified value-based and policy-based
    framework, in which the thresholds are **slack variables** stating how much
    worse than optimal is still sufficient; the remedy §11.8 points to for the
    strict-priority inaction attractor, and the formulation the planned
    lexicographic algorithms consume the rulebook through rather than through
    §6's scalarization.
14. **Tercan, Prabhu (2024).** *Thresholded Lexicographic Ordered Multiobjective
    Reinforcement Learning.* ECAI. arXiv:2408.13493. — the basis for every
    caveat in §11.8 on the remedy: that existing lexicographic RL approaches are
    heuristics without theoretical guarantees because the Bellman equation does
    not apply to them; that TLQ loses Q-learning's convergence guarantee; that
    the threshold is applied per state per objective through clipped values
    `min(value_i, τ_i)` rather than as an episode budget; that uniform slack
    across the state space is the documented failure mode; Proposition 4.1's
    goal-unreachability case and its precondition, which §11.8 records as **not**
    matching this rulebook's configuration; and the LPO policy-gradient
    alternative with its Proposition 5.1 non-decrease guarantee.
15. **Tumova, Reyes Castro, Karaman, Frazzoli, Rus (2013).** *Minimum-violation
    LTL Planning with Conflicting Specifications.* ACC. arXiv:1303.3679. — the
    dual formulation recorded in §11.8: task completion is required and the
    violation is *minimised* subject to it, which a scalar per-objective slack
    cannot express. Considered and not adopted.
16. **Altman (1999).** *Constrained Markov Decision Processes.* Chapman &
    Hall. — expected-cost constraints over an episode rather than per-step hard
    constraints; the third structural answer surveyed in §11.8 to why a
    state-wise hard constraint produces an inaction attractor.

---

## 13. Out of scope

Comfort as a rulebook channel is deliberately excluded. It is defensible on the
same criteria as `speed_limit` (controlled-invariant, anchored on nuPlan's
expert-derived acceleration and jerk percentiles), but the supervisor's guidance
is that ride comfort belongs downstream, in a filter or damper acting on the
produced control commands, rather than in the rulebook. It is recorded here as a
future extension, to be revisited once the rulebook is frozen and the first runs
have completed.

Termination conditions, the mission contract, the evaluation protocol, the
curriculum, and the encoder are unchanged by this document.
