# ADR-070: at-fault gating for the interaction sub-rules on a stopped ego

- Status: **Approved** — carried by `RULEBOOK-V5.1`, approved 2026-08-14
  specification's approval**
- Date: 2026-08-11
- Approval evidence: user instruction to apply and re-measure, 2026-08-11, after
  the §4.10.3 evidence and the risk list below were presented.
- Affected specifications: `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
  §4.10.3 (evidence), §5.2, §5.6; would change `AC-RB5-04` and every §4 figure.
- Supersedes nothing. Related: ADR-063 (`rss` rejected on controlled invariance),
  ADR-067 (`clearance` scoping, `ttc` threshold).

## Context

RULEBOOK-V5.0 §4.10.3 measures, inside the 97 expert episodes the specified
rulebook leaves below standstill, which sub-rule charges penalty while the ego is
already stopped:

| sub-rule | mass in those episodes | at ≤ 0.1 m/s | share |
|---|---:|---:|---:|
| `clearance` | 2031.4 | 1623.1 | **79.9 %** |
| `ttc` | 872.0 | 326.7 | **37.5 %** |
| `rss_lateral` | 2880.4 | 261.7 | 9.1 % |
| `solid_line` | 1460.1 | 420.5 | 28.8 % |
| `dashed_line` | 2253.5 | 610.7 | 27.1 % |
| `offroad` | 796.1 | 0.0 | 0.0 % |

The interaction sub-rules account for **2211.5 of the 3242.8 stopped-ego units
(68.2 %)**, i.e. **21.4 %** of the tail's total penalty mass.

This is a controlled-invariance failure of the same kind that rejected `rss` in
§4.7. `evaluate_clearance` is `max(0, 1 − distance / 1.0 m)` with no test of
relative motion: from a state where a pedestrian is 1.2 m away and closing, with
the ego boxed in, no ego action keeps the distance above 1 m. A rule bearing a
satisfaction indicator must be satisfiable; here it is not.

## What the literature says

The concept is **blame / responsibility**, not a deadband.

**RSS** (Shalev-Shwartz, Shammah, Shashua) formalizes *dangerous situation*,
*proper response* and *notion of blame*, and holds that a vehicle following the
proper response is never responsible. The canonical illustration in that
literature is exactly this case — being struck from behind while stopped at a red
light, where prevention is impossible. (RSS's `μ` lateral fluctuation margin is a
genuine deadband and is already implemented here as
`LATERAL_MARGIN_MU_M = 0.10`; it addresses position noise, not blame.)

**nuPlan implements it, with published thresholds.** Two places:

1. `no_ego_at_fault_collisions.py`:

   ```python
   def _get_collision_type(
       ego_state, tracked_object, stopped_speed_threshold: float = 5e-02
   ) -> CollisionType:
       """:param stopped_speed_threshold: Threshold for 0 speed due to noise."""
       is_ego_stopped = ego_state.dynamic_car_state.speed <= stopped_speed_threshold
       if is_ego_stopped:
           collision_type = CollisionType.STOPPED_EGO_COLLISION
       ...
   ```

   At-fault is `{ACTIVE_FRONT_COLLISION, STOPPED_TRACK_COLLISION,
   ACTIVE_LATERAL_COLLISION (conditional)}`. `STOPPED_EGO_COLLISION` and
   `ACTIVE_REAR_COLLISION` are **excluded**.

2. `time_to_collision_within_bound.py`: `stopped_speed_threshold = 5e-03` m/s,
   and TTC is not computed at all when `ego_speed <= stopped_speed_threshold`
   ("Remain default if we don't have any agents or ego is stopped").

**The asymmetry this specification proposed is also nuPlan's.**
`drivable_area_compliance` and `driving_direction_compliance` — the position
metrics — carry no speed gate. Only the interaction metrics do. §4.10.3 measures
the same asymmetry independently: `offroad` charges 0.0 % at a stopped ego.

## Proposed decision

`clearance`, `ttc` and `rss_lateral` become **inapplicable** when the ego speed is
at or below a published stopped threshold, using the applicability mechanism that
already exists. Position sub-rules (`offroad`, `solid_line`, `dashed_line`,
`wrong_carriageway`) and the traffic-control sub-rules are **unchanged**.

The threshold must be at nuPlan's published magnitude (`5e-02` m/s = 0.18 km/h),
not at the larger sweep thresholds of §4.10.2. See the risks.

## Risks, none of which the literature retires

1. **nuPlan's gate is an evaluation metric, not a training reward.** A planner
   being scored does not optimize the scorer; an RL agent optimizes the reward
   and will find any hole in it. Transplanting an evaluation gate into a reward
   is therefore not automatically safe, and the source offers no evidence on this
   because these metrics were never designed as rewards. The mitigation is
   magnitude: at 0.05 m/s the R4 margin is ~0, so crawling under the gate to
   become immune earns nothing. This is an argument, not a proof, and a gate at
   the larger thresholds of §4.10.2 (0.5, 1.0, 2.0 m/s) would be exploitable.

2. **The gate improves the standstill baseline too, and the net direction is not
   deducible.** The below-standstill statistic assumes standing still scores
   exactly 0. §4.10.2 shows a genuinely stopped vehicle accrues R2/R3 cost, so the
   true baseline is worse than 0. Gating removes that cost, making standstill
   *genuinely* free rather than free by assumption. The measured percentage would
   fall — the expert gains while the baseline is 0 by definition — but the live
   comparison moves in both directions at once. **This must be measured, not
   argued.**

3. **The gate is a subset of nuPlan's logic.** nuPlan also checks
   `is_track_stopped` (the ego is at fault when it drives into a stationary
   object) and `is_agent_behind` (a rear impact is not the ego's fault). An
   ego-speed-only gate is the coarse version. It is defensible, but must be
   described as such and never as "nuPlan's at-fault logic".

4. **It touches only 21.4 % of the tail.** The 68.6 % charged to a moving ego is
   untouched, as is the 10.0 % from position rules. The gate is a correctness fix,
   not a way to drive the below-standstill figure down.

5. **`rss_lateral` gains almost nothing** (9.1 %). Including it in the gate is
   justified by consistency of principle, not by measured effect.

6. **Procedural cost.** This changes `AC-RB5-04`, the R2 violation rates, and
   every figure in §4. It requires re-measuring the full 1100-record panel and
   re-deriving §4.5. It is not a local edit.

## Measured outcome

Applied and re-measured on the same 1100 Waymo `train` records, 217 189
transitions, 0 skipped, 0 errors.

| rulebook | p1 | p5 | p10 | p50 | mean | below standstill | R2 % | R3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gate removed | −190.25 | −23.13 | +4.57 | +25.01 | +29.09 | 8.82 % | 0.6068 | 1.1796 |
| gate at 5e-03 | −150.85 | −16.45 | +4.84 | +25.48 | +30.74 | 7.64 % | 0.4383 | 1.1796 |
| **gate at 5e-02** | **−150.85** | **−15.86** | **+4.90** | **+25.72** | **+31.14** | **7.45 %** | **0.3978** | **1.1796** |

**The threshold choice is robust.** Between the two published values p1 is
identical and the mean differs by 0.4 in 31: most of the effect is present at the
smaller one, so the result does not rest on a fitted constant. 5e-02 is adopted
because it is the blame threshold while 5e-03 guards a division.

**The gate reached what it was scoped to reach and nothing else.** Macro R3 is
identical to six figures across all three rows, and the position sub-rules'
penalty mass is unchanged to the last decimal (`dashed_line` 2253.5, `solid_line`
1460.1). `clearance` loses 78 % of its mass (2031.4 → 442.9) and its stopped-ego
share falls 79.9 % → 7.8 %; `ttc` 37.5 % → 5.2 %.

Residual: the tail's not-avoidable-by-slowing share falls from 31.4 % to 13.9 %,
and what remains is `solid_line` + `dashed_line`, which §4.10.3 classifies as
normative disagreement rather than defect. **The defect portion is closed.**

Risk 2 above resolved in the opposite direction to the one feared: because the
gate makes a legally stopped vehicle score genuinely ≈ 0, it makes the
below-standstill baseline *accurate* rather than optimistic, where before it was
wrong by an unquantified amount. It does not hand the standstill policy anything
it did not already have by assumption.

Risks 1, 3, 4 and 5 stand as written and are unretired by the measurement.

## Follow-up this evidence opens

R1 has **no** at-fault classification: `evaluate_collision_impact` charges every
contact onset as a function of closing speed and actor class alone. A stopped ego
struck from behind therefore pays full cost at the highest priority level — the
canonical RSS not-to-blame case, and the level at which nuPlan actually applies
its at-fault logic. This ADR deliberately does not address it; it is R2 only.
