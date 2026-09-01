# ADR-072: relaxable lane rules move below mission progress

- Status: **Approved** — carried by `RULEBOOK-V5.1`, approved 2026-08-14
- Date: 2026-08-14
- Approval evidence: explicit user approval of `RULEBOOK-V5.1` on 2026-08-14
  ("approvo la v5.1"), which carries this decision.
- Affected specifications: `docs/specifications/rulebook_v5.1_specification.md`
  §1.1, §3, §3.3, §3.5. Supersedes v4.7's macro-rule grouping and its uniform
  `max` aggregation, which `RULEBOOK-V5.0` left authoritative.
- Related: ADR-070 (at-fault gate, which makes standing still provably 0 on L2),
  ADR-065 and ADR-066 (the geometric definitions of the sub-rules being moved).

## Context

`RULEBOOK-V5.0` §11.8 recorded, as the most consequential limitation in that
document, that its reward **makes standing still preferable to completing the
mission**. It attributed the pathology to strict priority as such and deferred
the remedy to the learning algorithm.

That attribution was wrong, and v5.0's own §6.5 already contained the
correction. The pathology does not require strict priority. It requires *every*
road rule to outrank progress — which is a **placement choice** v5.0 inherited
from v4.7 without re-examining it. With `solid_line` in R3 and R3 above R4, a
two-second lane-marking contact to pass a parked obstacle costs about what the
whole mission is worth, while waiting behind it forever costs nothing.

## Decision

Five levels, strictly ordered, with the relaxable lane rules **below** progress:

| level | channel | sub-rules |
|---|---|---|
| L1 | `collision_safety` | at-fault collision impact |
| L2 | `interaction_risk` | `ttc`, `clearance`, `rss_lateral` |
| L3 | `non_relaxable_compliance` | `offroad`, `signal`, `stop`, `crosswalk`, `vehicle_yield`, `speed_limit` |
| L4 | `mission_progress` | monotone route advance (ADR-073) |
| L5 | `relaxable_lane_compliance` | `solid_line`, `wrong_carriageway`, `dashed_line` |

The dividing question is **whether a competent driver may relax the rule in order
to complete a mission**. Crossing a solid line to pass an obstruction: yes.
Running a red light: no. That is the whole criterion.

This is the minimum-violation semantics of Castro, Tumova, Karaman, Frazzoli and
Rus expressed as a rule hierarchy rather than as a planner.

**L1 and L2 stay separate** because a collision is an *outcome* and TTC /
clearance / RSS-lateral are *anticipatory indicators*. An indicator that fires is
not a failure; a collision is.

**Three aggregations, not one.** v4.7 used `max` everywhere. This decision keeps
`max` across objects within a sub-rule, and within L2 and L3, but uses a
**normalized sum with a fixed denominator of 3** within L5. The reason is
specific: at L5 the quantity of interest is the *total amount of relaxation*, so a
detour that crosses a solid line **and** enters the opposing carriageway must cost
more than one that only straddles. Under `max` the second concurrent violation is
free, which breaks ordering O6 in exactly the case it exists to catch.

## What this actually buys, measured against what was claimed

An earlier draft of the specification claimed v5.0 fails four of the six
orderings. Working them out from the two constructions gives one:

| | v5.0 | v5.1 |
|---|---|---|
| O1 legal completion ≻ standing still | fails | fails |
| **O2 completion needing brief relaxation ≻ standing still** | **fails** | **passes** |
| O3 legal route ≻ illegal shortcut | passes | passes |
| O4 relaxation ≻ collision | passes | passes |
| O5 waiting at red ≻ running it | passes | passes |
| O6 necessary ≻ gratuitous relaxation | passes | passes |

v5.0 passes O3 and O6 because its lane rules sit above progress, so the illegal
shortcut loses at R3 and the gratuitous violation loses on accumulated R3 cost.

**The single ordering this restructure buys is O2** — and O2 is the motivating
pathology of the entire redesign. The narrower claim is the honest one.

Verified by 17 constructed fixtures in `tests/test_rulebook_v51_orderings.py`,
all passing. Expert replay cannot establish any of this: it holds one trajectory
per scenario and therefore no counterfactual.

## Measured consequence on the expert panel

1100 Waymo `train` records, 217,189 transitions, 0 skipped, 0 errors:

| rulebook | mean | p1 | p5 | p50 | below standstill |
|---|---:|---:|---:|---:|---:|
| v5.0 specified | +31.14 | −150.85 | −15.86 | +25.72 | 7.45 % |
| **v5.1** | **+73.85** | **−59.67** | **+9.16** | **+55.23** | **3.36 %** |

Channel rates: L2 **0.3978 %**, L3 **0.6418 %**, L5 **0.8734 %**. L2 is
bit-identical to v5.0's macro R2, confirming that channel is carried over
unchanged. L3 + L5 exceeds v5.0's macro R3 (1.1796 %) because `max` aggregation
collapsed concurrent violations that two separate channels now expose; this is
expected, not a regression.

## Risks and limitations

1. **O1 is not fixed and no hierarchy can fix it.** Standing still is exactly
   `(0,0,0,0,0)` — ADR-070's gate makes L2 provably 0 below 0.05 m/s — while any
   trajectory moving through traffic accrues some L2 cost (measured 0.3978 % of
   expert steps). A **strict** lexicographic comparison therefore stops at L2 and
   never reaches progress. This holds for every hierarchy in which stopping is
   safe, which is every admissible hierarchy. Only a **thresholded** comparison
   fixes it, by making "both within budget" a tie at the safety channels. The
   failure is asserted as an executable property, not merely documented.
2. **`offroad` at L3 is the placement most likely to be wrong.** Pulling onto a
   shoulder to pass an obstruction is ranked below not completing the mission.
   Defensible on precedent — nuPlan treats `drivable_area_compliance` as a
   multiplicative penalty — and the 0.3 m tolerance already absorbs bounding-box
   over-approximation. Moving it to L5 is a one-line change that would require
   re-deriving every acceptance figure. **Recorded as a decision, not a fact.**
3. **The `max` within L3 is a convenience.** It identifies the worst
   non-relaxable violation on a shared normalized scale, without any claim that a
   red light and an off-road excursion are cardinally comparable. A finer
   treatment needs per-sub-rule priorities within L3, which the atomic cost
   vector leaves available.
4. **`dashed_line` at L5 depends on its time factor.** Its cost is
   `penetration × time_factor` with the factor 0 below 1.0 s, so a legal lane
   change costs exactly zero by construction and only sustained straddling is
   charged. If that definition ever changed, the placement would have to be
   revisited: the supervisor's explicit requirement is preserved here, only its
   priority changes.
5. **The fixtures test the hierarchy, not the geometry.** Whether a given scene
   yields `c_solid_line = 0.4` is the sub-rule evidence's business. A sub-rule
   that mis-measures its own cost would pass every ordering fixture.
