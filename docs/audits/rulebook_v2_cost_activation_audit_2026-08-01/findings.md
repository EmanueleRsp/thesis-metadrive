# Rulebook v2 Cost-Activation Audit (2026-08-01)

Status: `ANALYSIS ONLY — NO PRODUCTION CODE CHANGED — AWAITING USER APPROVAL`

## 1. Scope And Trigger

The user reviewed evaluation GIFs of a `RUN_PROFILE=medium` run and reported five
suspected defects in Rulebook v2 sub-rule cost/margin computation:

1. driving into the opposing carriageway is not captured by `wrongway`;
2. the yellow solid line is not captured by `solid_line`;
3. `rss` activates far too often;
4. the traffic-signal cost appears not to work;
5. a stationary ego keeps receiving a constant negative reward.

This document reports what was verified against the code, the authoritative
specifications, and — where possible — against live MetaDrive geometry and the
frozen ScenarioNet selection index. Every claim below is labelled with the
evidence that supports it. No production code, configuration, or specification
was modified by this audit.

Authoritative sources consulted: `docs/specifications/rulebook_v4.7_specification.md`
(§7.1–§7.6, §6.2), `rulebook_v4.8_specification.md`, `rulebook_v4.9_specification.md`,
`rulebook_scalarization_v1.0_specification.md`, and `docs/project_index.md`.

## 2. Executive Summary

| ID | User point | Verdict | Root cause | Severity |
| --- | --- | --- | --- | --- |
| F0 | (cross-cutting) | Confirmed | `bounded_satisfaction_rank` binarises every macro margin at a `1e-8` tolerance, so an epsilon-level sub-rule cost carries the same penalty as a maximal one | Critical |
| F1 | 1 | Confirmed, spec-level gap | No sub-rule measures occupancy of an opposing-direction lane; `wrongway` is a signed-velocity rule by design, `offroad` unions *all* lanes regardless of legal direction | High |
| F2 | 2 | Confirmed, implementation bug | The PG static adapter drops `ROAD_LINE_SOLID_SINGLE_YELLOW`; the Waymo adapter drops four further marking classes | High |
| F3 | 3 | Confirmed | `b_e` is capped at `4.0 m/s²` while the measured ego value is `>= 10.7 m/s²`, and RSS stays applicable at standstill | High |
| F4 | 4 | Partially confirmed | The signal cost is by construction a one-step event with no persistence, plus two silent control-drop paths and no applicability diagnostic | Medium-High |
| F5 | 5 | Confirmed | `offroad` fires on sub-cm² slivers between adjacent lane polygons; combined with F0 this yields a permanent R3 violation for a correctly positioned vehicle | Critical |

F0 and F5 together are the single largest defect: they can make a legally and
safely driving (or stopped) ego receive `-2.01` per step indefinitely.

## 3. Findings

### F0 — Graded sub-rule costs are binarised by the active scalarizer

Evidence:

- `conf/scalarization/default.yaml`: `mode: bounded_satisfaction_rank`,
  `numerical_tolerance: 1.0e-8`, `priority_base: 2.01`.
- `src/thesis_rl/reward/scalarization.py:239-259`: the satisfaction pattern is
  `value == 0.0` after snapping only `|value| <= 1e-8`; the rank contribution is
  `base**k * (is_satisfied - 1.0)`, i.e. a full `-2.01` (R3), `-4.04` (R2),
  `-8.12` (R1) as soon as a macro cost exceeds `1e-8`.
- The only graded term is `continuous = sum(canonical)/4`, whose magnitude is at
  most `0.25` per macro rule — an order of magnitude below the rank term.

Consequence: every graded cost in the Rulebook (the MAIS3+F injury curve, the
dashed-line penetration factor, the RSS relative deficit, the off-road area
fraction) is, for the learner, effectively a step function at `cost > 0`. A
`1e-4 m²` geometric sliver and a full off-road excursion are indistinguishable.

This does not by itself make any sub-rule wrong, but it converts every
false-positive activation described below into a maximal penalty. It must be
resolved together with F5, otherwise the F5 fix only reduces the frequency of an
already saturating penalty.

### F1 — Occupying the opposing carriageway carries no cost

Evidence:

- `src/thesis_rl/rulebook/v2/components/road.py:116-146` (`evaluate_wrongway`):
  the cost is `clip([-v_parallel]_+ / v_max, 0, 1)`, i.e. strictly reverse motion
  along the canonical route tangent. Rulebook v4.7 §7.3.2 mandates exactly this
  and explicitly states that geometric orientation alone must not be penalised.
- `src/thesis_rl/rulebook/v2/geometry/drivable.py:65-96` and v4.7 §7.2.1: the
  drivable surface is the union of **every** vertically compatible drivable lane
  and, verbatim, "Non dipende … dalla direzione legale". `cache.route_lanes`
  is populated with all map lanes (`pg_static_adapter.py:325`,
  `waymo_static_adapter.py`), so the opposing carriageway is part of the
  reference surface and `offroad` returns `0`.
- No other component measures lane direction: `ttc`, `rss`, `rss_lateral`,
  `clearance` are actor-relative, and `progress` credits longitudinal advance
  with no lateral gate (`route_outside_fraction` is diagnostic-only, see
  `components/progress.py:30-50`).

Consequence: an ego travelling forward in the oncoming lane has
`wrongway = 0`, `offroad = 0`, and — on an empty oncoming lane — zero cost from
every other sub-rule. The only mechanism that could charge it is `solid_line`,
which F2 shows is broken for exactly the marking that separates the two
carriageways on PG maps.

The user's proposal (an area-fraction rule analogous to off-road) is the correct
shape and is compatible with the existing geometry stack.

### F2 — Lane-marking classes are silently dropped by the static adapters

Evidence (PG, reproduced live):

```
MetaDriveEnv({'map': 'SCS', ...}).current_map.get_map_features() ->
    36  LANE_SURFACE_STREET
    24  ROAD_LINE_BROKEN_SINGLE_WHITE
    12  ROAD_LINE_SOLID_SINGLE_WHITE
    12  ROAD_LINE_SOLID_SINGLE_YELLOW
```

`third_party/metadrive/metadrive/component/map/pg_map.py:156-168`
(`PGMap.get_line_type`) emits `LINE_SOLID_SINGLE_YELLOW` for every continuous
yellow line — which on a PG map is precisely the centreline separating the two
carriageways — and `LINE_BROKEN_SINGLE_YELLOW` for broken yellow lines.

`src/thesis_rl/rulebook/v2/context/pg_static_adapter.py:38-44` maps only:

```python
"ROAD_LINE_SOLID_SINGLE_WHITE"  -> LANE_MARKING_SOLID
"ROAD_LINE_SOLID_DOUBLE_YELLOW" -> LANE_MARKING_SOLID   # never emitted by PG
"ROAD_LINE_BROKEN_SINGLE_WHITE" -> LANE_MARKING_DASHED
```

`ROAD_LINE_SOLID_SINGLE_YELLOW` has no entry, so `build_pg_static_adapter_result`
hits the `if feature_class is None: continue` branch and the feature never enters
`map_feature_catalog`. Downstream, `transition.py:1329-1344` filters that catalog
by `MapFeatureClass.LANE_MARKING_SOLID`, so `evaluate_solid_line` never sees the
yellow centreline. The only solid markings surviving on a PG map are the outer
white road edges (`PGLineType.SIDE`).

The same catalog feeds the semantic observation
(`envs/observations/causal_semantic.py:1027-1042`), so the agent cannot perceive
the yellow centreline either.

Waymo path: `waymo_static_adapter.py:38-46` covers
`SOLID_SINGLE_WHITE`, `SOLID_DOUBLE_YELLOW`, `SOLID_SINGLE_YELLOW`,
`BROKEN_SINGLE_WHITE`. The ScenarioNet converter
(`third_party/scenarionet/scenarionet/converter/waymo/type.py:45-67`) can also emit
`ROAD_LINE_SOLID_DOUBLE_WHITE`, `ROAD_LINE_BROKEN_SINGLE_YELLOW`,
`ROAD_LINE_BROKEN_DOUBLE_YELLOW`, `ROAD_LINE_PASSING_DOUBLE_YELLOW`, and
`ROAD_EDGE_MEDIAN`, none of which is mapped. The four missing line classes are
real markings that are silently invisible to both the Rulebook and the
observation.

### F3 — RSS-longitudinal over-triggers

Two independent causes.

**F3a — `b_e` is capped below the measured value.**
v4.7 §6.2.4 step 10 fixes `b_e = min(4.0, floor(10 b_meas)/10)`, implemented at
`src/thesis_rl/rulebook/v2/calibration.py:17,96`
(`MAX_REFERENCE_BRAKE_MPS2 = 4.0`). The production artifact recorded in
`docs/implementation/rulebook_v2_implementation_plan.md:896` is the capped
`ego_min_brake_mps2 = 4.0`.

A 12-trial re-run of the normative protocol
(`python -m thesis_rl.cli.rulebook_v2_braking_trials --config conf/rulebook_v2/ego_calibration.json --trials-per-target 3`,
raw results in `braking_trials_sample_12.json`) measured mean decelerations of
`10.73 … 16.92 m/s²`, minimum `10.73`. The 5th-percentile lower quantile over the
full 40-trial protocol is therefore approximately `10.7 m/s²`; the cap discards a
factor of `2.7`.

The cap is also internally inconsistent: the same specification fixes
`b_i = 8.0 m/s²` as the braking assumed for the *front* vehicle, which in these
scenarios is a vehicle of the same class. The model therefore assumes the ego
brakes half as well as identical surrounding traffic.

Resulting safe distances `d_safe` (ρ = 1.0 s, `a_max_acc` = 3.5 m/s², `b_i` = 8.0):

| `v_e` | `v_i` | `b_e = 4.0` | `b_e = 8.0` | `b_e = 9.5` |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 3.28 m | 2.52 m | 2.39 m |
| 5 | 5 | 14.22 m | 9.70 m | 8.99 m |
| 10 | 10 | 28.28 m | 16.89 m | 15.09 m |
| 15 | 15 | 45.47 m | 24.08 m | 20.70 m |
| 20 | 20 | 65.78 m | 31.27 m | 25.82 m |

With `b_e = 4.0` a car-following state at a 2 s headway yields
`q_rss = 1 - d/d_safe ≈ 0.29 … 0.39` at every speed — i.e. a permanent R2
violation in any traffic. Under F0 that is `-4.04` per step.

**F3b — RSS remains applicable at standstill.**
`safe_distance_m` (`components/rss.py:40-51`) always adds the response terms
`v_e ρ + ½ a_max_acc ρ²` and `(v_e + ρ a_max_acc)² / (2 b_e)`. At `v_e = v_i = 0`
this is `3.28 m` with `b_e = 4.0`. An ego stopped `2 m` behind a stopped leader —
a normal queue — scores `q_rss = 0.39` for as long as it waits, and the only
actions that reduce it are reversing (which then triggers `wrongway`) or driving
into the leader. This is a non-escapable cost and a direct contributor to F5.

The candidate-scoping logic itself (`transition.py:253-335`) was reviewed and is
sound: same-traffic-stream gating, concordant-heading gating, bumper-to-bumper
gap, and front-vehicle predicate all behave as v4.7 §6.2.1 requires. The
over-triggering is in the parameters and the applicability domain, not in the
"which vehicles to attend to" logic.

### F4 — Traffic-signal cost

Three separate issues; only the third is unambiguously a defect.

**F4a — the violation is a single-step event (by construction).**
`evaluate_signal_transition` (`components/controls.py:190-299`) charges
`cost = 1.0` only on the step whose swept front bumper crosses the control line
while the pre-state colour is RED (or a committed YELLOW), then adds the group to
`resolved_signal_group_ids` so it is never charged again. The approach branch
charges `1 - post_delta / d_req` only inside the ego's own stopping distance.
With `b_e = 4.0` and `Δt = 0.1 s`, `d_req` is `3.6 m` at 5 m/s and `13.5 m` at
10 m/s; below those distances a red light is already unavoidable. So a full
red-light run costs one step of R3 in an episode of hundreds — this is why it
"looks like it does not work" even when it is working exactly as specified.
Every other persistent R3 sub-rule (off-road, wrong-way, illegal-entry latches)
represents duration; the signal does not.

**F4b — two silent control-drop paths.**
- `derive_control_line` raises `ControlLineOffRouteError` and both adapters
  `continue` without recording anything, so a light whose orthogonal lane section
  does not intersect the assigned route polyline disappears with no diagnostic.
- `_selected_control` (`transition.py:499-522`) additionally requires
  `control.movement_key.approach_lane_id in cache.task_route.lane_ids` — strict
  lane-ID membership. The catalog flag `has_route_traffic_light` is computed by a
  much looser geometric-proximity test (`scenarios/waymo_topology.py:177-192`), so
  a scenario can be tagged as signalised while the Rulebook selects no control at
  all.

Catalog statistics from the frozen index (`data/scenarionet/frozen/scenario_selection_index.json`):
3500 records, 1805 Waymo / 1695 PG, all `rulebook_eligible = true`,
`has_route_traffic_light = true` for 828 (23.7%). Signalised scenarios are
therefore present; whether the Rulebook actually selects their controls at
runtime is *not currently observable*.

**F4c — no applicability diagnostic anywhere.**
`runtime/io/video_diagnostics.py:100-124` prints `signal c=0.00` identically for
"no control selected" (`NOT_APPLICABLE`) and "control selected, green, satisfied".
The evaluation GIFs therefore cannot distinguish a working signal rule from a
never-selected one. The same holds for `stop`, `crosswalk`, and `vehicle_yield`.

**Note.** `derive_lane_movement_key` returning `None` for any traffic light in a
Waymo scenario appends `movement_key_ambiguous:*` to `validation_errors`, and
`catalog_eligibility.py:111` sets `rulebook_eligible = not errors`. Since a
signalised approach lane usually has several exit lanes, this plausibly excluded
a large share of signalised Waymo scenarios during catalog construction. The
frozen index only retains the surviving records, so the exclusion rate cannot be
measured from it; it should be re-measured from the catalog filter audit output.

### F5 — Stationary ego with a constant negative reward

Three concurrent mechanisms were identified; the first is quantified and is by
far the largest.

**F5a — `offroad` fires on inter-lane polygon seams (measured).**
`OFFROAD_AREA_EPSILON_M2 = 1e-4` (`components/road.py:20`) is an absolute `1 cm²`
tolerance on a footprint of roughly `8.3 m²`, i.e. a relative threshold of
`1.2e-5`. The union of adjacent lane polygons is not watertight.

Measured on a live PG map (`map='SCS'`, 36 lanes, default 3.5 m lane width),
sweeping a `4.5 m × 1.85 m` footprint along every lane centreline at several
lateral offsets and computing `footprint.difference(union_of_lane_polygons).area`:

| lateral offset from lane centre | steps with `offroad > 0` | median area ratio | after a 10 cm morphological closing |
| ---: | ---: | ---: | ---: |
| 0.0 m | 2.8% | 0.0556 | 2.8% |
| 0.4 m | 2.8% | 0.0556 | 2.8% |
| 0.8 m | 14.9% | 0.0000 | 2.8% |
| 1.2 m | 37.1% | 0.0002 | 2.8% |

`shapely.union_all` of the 36 lane polygons of this map yields a single polygon
with **681 interior holes**, individual areas `1e-4 … 5e-4 m²`.

Interpretation: the residual 2.8% baseline is legitimate (the footprint sticking
out past the first/last lane of the map, ratio 5.6%, all located at
`i in {1, len-2}` of the first and last lanes). Everything above it is pure
numerical sliver: at a lateral offset of 1.2 m — a completely normal in-lane
position, since half a lane is 1.75 m and half the car is 0.925 m — **37% of
positions report a non-zero off-road cost of about 2 cm²**. Under F0 each of
those steps is a full `-2.01`.

A stationary vehicle parked over such a seam receives that penalty forever, with
no action that removes it. Waymo lane polygons are built independently per lane
from per-point left/right widths (`waymo_static_adapter.py:_lane_polygon_from_widths`),
so the same or a larger seam effect is expected there.

**F5b — RSS standstill trap.** See F3b: `q_rss ≈ 0.39` for a stopped ego 2 m
behind a stopped leader, permanently.

**F5c — `dashed_line` timer never resets while occupying.**
`evaluate_dashed_line` (`components/road.py:188-269`) accumulates the timer while
the marking intersects the footprint, saturating at `DASHED_TCAP_S = 2.0 s`, and
multiplies by the lateral penetration. Stopping astride a dashed marking gives a
permanent `cost = penetration`. The penetration factor already grades a light
touch to ≈0, so a bumper corner over the line is *not* charged — but any
penetration above `1e-8` is a full R3 violation under F0. This is the mechanism
the user hypothesised; it is real but secondary to F5a.

The remaining candidates were checked and cleared: `progress` returns margin `0`
(not negative) for a stationary ego; `evaluate_signal_transition` returns `0`
when `speed = 0` because `d_req = 0`; `rss_lateral` gives `d_safe_lat ≈ 0.16 m`
for two stationary vehicles, below any real adjacent-lane gap; `ttc` never fires
for two stationary bodies.

## 4. Proposed Corrections

All of these are behaviour-changing and therefore require explicit approval per
`AGENTS.md` §"Decision And Change Control"; two require amending an approved
specification. Ordered by expected impact per unit of risk.

### P1 — Close the drivable-surface seams (fixes F5a)

- In `drivable_surface_for_ego`, apply a morphological closing to the union:
  `union.buffer(eps).buffer(-eps)` with `eps = GEOMETRY_EPSILON_M`-class value.
  The measurement above shows `eps = 0.10 m` removes every sliver-driven
  activation while leaving the genuine map-edge gaps intact. A smaller `0.05 m`
  was insufficient at the seams present in PG maps.
- `OFFROAD_AREA_EPSILON_M2` stays at `1e-4`. An earlier revision of this document
  additionally proposed a relative tolerance `max(1e-4, 0.01 * A(P_e))`; it is
  **withdrawn**. Its only justification was robustness against the F0
  binarisation, which is a defect to be fixed where it lives. On an `8.3 m²`
  footprint a 1% relative tolerance would silently exempt `0.08 m²` of genuine
  off-road, i.e. a semantic weakening disguised as a numerical tolerance.

Specification impact: v4.7 §7.2.2 already defines `ε_A` as a numerical tolerance
"fissata dai test di geometria e non trattata come parametro semantico", so
changing its value is in scope. The closing operator is a new numerical
tolerance on `C_drive` and should be recorded in §7.2.1 plus an ADR.

Risk: a 10 cm closing also bridges genuine 20 cm-wide gaps between two lanes; on
real road geometry such a gap is not a place a vehicle can be "off-road" in any
meaningful sense.

### P2 — Introduce a satisfaction tolerance in the scalarizer (fixes F0)

Three options, in order of preference:

1. **Add `satisfaction_tolerance` to `ScalarizationConfig`** (default e.g.
   `0.02`), used only for the `bounded_satisfaction_rank` pattern test:
   `pattern = tuple(abs(v) <= satisfaction_tolerance for v in canonical[:3])`.
   Small, auditable, keeps the approved lexicographic semantics, and makes the
   rulebook robust to any residual epsilon-level cost. Requires a SCAL-V1.0
   amendment plus ADR.
2. Switch the profile to `bounded_centered_sigmoid` with a lower
   `sigmoid_sharpness` (30 is effectively a step; 3–5 grades over the whole
   `[-1, 0]` range). No code change, only configuration — but it changes the
   reward scale of every existing run and breaks comparability with the runs
   already completed.
3. Do nothing in the scalarizer and rely exclusively on per-component deadbands
   (P1, P4). Cheapest, but leaves the system one geometry regression away from
   the same failure mode.

Recommendation: option 1, in the same change as P1.

### P3 — Recalibrate `b_e` (fixes F3a, tightens F4a)

Replace the normative cap by a *physical-plausibility* bound instead of the RSS
*reference* value:

```python
PHYSICAL_MAX_BRAKE_MPS2 = 8.0   # dry-asphalt tyre-road limit, and equal to b_i
b_e = min(PHYSICAL_MAX_BRAKE_MPS2, floor(10 * b_meas) / 10)
```

With the measured `b_meas ≈ 10.7` this yields `b_e = 8.0`, making the ego's
assumed braking equal to the `b_i = 8.0` already assumed for identical
surrounding vehicles, and halving `d_safe` at every speed (table in F3a).

Side effects, all in the intended direction: `d_req` for the signal rule and
`d_stop` for crosswalk/vehicle-yield shrink by the same factor, so the "can no
longer stop" region becomes physically accurate instead of twice too large.

Specification impact: amends v4.7 §6.2.4 step 10 and the §2782 parameter table.
Requires a new artifact regeneration (`make rulebook-v2-collect-trials` +
`rulebook-v2-calibrate`); the ego `config_hash` does not change, so the frozen
catalog eligibility index remains valid.

### P4 — Scope RSS out of the standstill regime (fixes F3b, F5b)

Make `rss` `NOT_APPLICABLE` for a candidate when both longitudinal speeds are
below a standstill threshold (`v_e <= 0.1 m/s and v_i <= 0.1 m/s`). Rationale:
the RSS response term models what the ego *could* do during the reaction time;
charging it while the ego is executing the proper response (remaining stopped)
penalises compliance and creates a state with no cost-reducing action. Queueing
behind a stopped leader is the normal case, not a violation.

Alternative considered and not recommended: gating on closing dynamics
(`v_e > v_i`), which would also silence genuine same-speed tailgating — the core
case the safe-distance rule exists for.

Specification impact: new applicability clause in v4.7 §6.2.1. ADR required.

### P5 — Map the missing lane-marking classes (fixes F2)

- `pg_static_adapter._FEATURE_CLASSES`: add
  `ROAD_LINE_SOLID_SINGLE_YELLOW -> LANE_MARKING_SOLID` and
  `ROAD_LINE_BROKEN_SINGLE_YELLOW -> LANE_MARKING_DASHED`.
- `waymo_static_adapter._FEATURE_CLASSES`: add
  `ROAD_LINE_SOLID_DOUBLE_WHITE -> LANE_MARKING_SOLID`,
  `ROAD_LINE_PASSING_DOUBLE_YELLOW -> LANE_MARKING_DASHED` (Waymo semantics: a
  double yellow that *permits* passing, i.e. functionally a broken line; an
  earlier revision of this document said `LANE_MARKING_SOLID` and was wrong),
  `ROAD_LINE_BROKEN_SINGLE_YELLOW -> LANE_MARKING_DASHED`,
  `ROAD_LINE_BROKEN_DOUBLE_YELLOW -> LANE_MARKING_DASHED`, and decide explicitly
  whether `ROAD_EDGE_MEDIAN -> ROAD_BOUNDARY`.
- Replace the silent `continue` on an unmapped feature type with an explicit
  recorded diagnostic so a future schema extension cannot disappear again.

Specification impact: none — this is a pure adapter-coverage bug relative to
v4.7 §7.4/§7.5, which speak of "boundary continue"/"boundary tratteggiate"
without restricting the colour.

Interaction to decide: `evaluate_solid_line` charges a hard `1.0` on any contact
within `1 cm`. Once the PG yellow centreline exists, the rule will fire on every
centreline touch. Under F0 this is a full R3 violation on a 1 cm grazing contact.
Recommend adding a minimal-penetration condition to the *occupancy* branch (the
completed-crossing branch stays binary), analogous to the dashed-line
penetration factor. This one *does* change §7.4.1 semantics and needs approval.

Note also that this fix changes the semantic observation content
(`causal_semantic.py`), i.e. the observation vector's information content — not
its dimension. Existing checkpoints remain shape-compatible but are no longer
behaviourally comparable.

### P6 — New sub-rule: opposing-carriageway occupancy (fixes F1)

Add an R3 component `wrong_carriageway`, cost:

```
q_wrong_carriageway = A(P_e ∩ (C_opp \ C_aligned)) / A(P_e)
```

where, over the vertically compatible drivable lanes of the step, `C_aligned` is
the union of lanes whose local centreline tangent at the ego projection satisfies
`t_lane · t_route >= cos(60°)` and `C_opp` the union of those with
`t_lane · t_route <= -cos(60°)`. Lanes that are neither (crossing branches inside
a junction) contribute to neither set.

Subtracting `C_aligned` is what makes the rule safe inside intersections: an ego
turning left is inside its own route-aligned junction lane, so the overlap with
opposing through-lane polygons is cancelled and the cost stays `0`. On a
two-way road the oncoming lane is not covered by any aligned lane, so the
invaded area fraction is charged.

The rule is memoryless and carries **no time ramp**. An earlier revision proposed
one as robustness against transient junction overlap; it is withdrawn because
that overlap was hypothesised rather than measured, and because `offroad` — the
direct structural analogue — has none. If transient junction activations appear,
they are a defect of the direction-cone logic to be fixed there.

Implementation is cheap: `drivable_surface_for_ego` already iterates the lanes
and projects the ego onto each centreline; it can return the three surfaces from
the same pass. `evaluate_wrongway` is left exactly as specified, preserving the
v4.7 §7.1 separation of concerns (position vs direction of motion vs markings).

Open design question for the user: on a road whose centreline is a *broken*
yellow line, overtaking into the oncoming lane is legal. v1 of the rule would
charge it anyway. Options: (a) charge unconditionally (simplest, treats it as a
risk rather than an infraction); (b) suppress the cost while the nearest
separating marking is a passing-permitted class. Recommendation: (a) for the
first implementation, with the marking class recorded as a diagnostic so (b) can
be decided from data.

Specification impact: new subsection v4.7 §7.3-bis plus registry/aggregation
entry. ADR required.

### P7 — Make the signal rule observable and persistent (fixes F4)

Split into three independent steps:

1. **Diagnostics first (no behaviour change).** Emit, per episode, the number of
   selected signal/stop/crosswalk/vehicle-yield controls, the number of steps
   with each component `applicable`, and the count of controls dropped by
   `ControlLineOffRouteError` and by the `approach_lane_id in route` filter. Add
   `applicable`/`status` and the signal colour + signed distance to the GIF
   overlay. This turns F4 from a hypothesis into a measurement and costs nothing
   scientifically.
2. **Re-measure the catalog exclusion rate** attributable to
   `movement_key_ambiguous` on traffic lights, from the catalog-filter audit
   output rather than from the frozen index.
3. **Only then**, if the diagnostics confirm that controls are selected and the
   signal is simply too sparse, consider a red-light persistence latch mirroring
   the existing crosswalk/vehicle-yield illegal-entry latches: after an illegal
   crossing, keep `cost = 1.0` while the ego remains inside the controlled
   junction. This makes the violation represent duration, consistent with v4.7
   §346 for the other persistent R3 conditions. Requires approval and an ADR.

## 5. Suggested Sequencing

| Milestone | Content | Approval needed |
| --- | --- | --- |
| M0 | P7.1 + P7.2 diagnostics only | No (additive diagnostics) |
| M1 | P1 (off-road seam closing) | Yes — §7.2.1 tolerance |
| M2 | P3 + P4 (RSS calibration and standstill scoping) | Yes — §6.2.1/§6.2.4 amendment |
| M3 | P5a/P5b (marking coverage and unmapped-type diagnostic) | No — adapter coverage bug |
| M4 | P6 (`wrong_carriageway`) | Yes — new §7.3-bis |
| M5 | P5c (solid-line deadband) and P7.3 (signal persistence), each gated on its own measurement | Yes |

P2 (scalarizer satisfaction tolerance) is **deferred to a separate ExecPlan** by
explicit user decision: one problem at a time. Nothing in M0–M5 may be sized or
justified by the F0 binarisation; every tolerance introduced must stand on its
own geometric or physical justification.

These milestones are carried by
`docs/implementation/rulebook_v2_cost_activation_corrections_v1_exec_plan.md`,
with acceptance-test-first coverage and a regression test per fixed defect.

## 6. Reproduction Commands

```bash
# F2: live PG marking-type census
uv run --no-sync python -c "import collections; from metadrive.envs.metadrive_env import MetaDriveEnv; env=MetaDriveEnv({'map':'SCS','start_seed':0,'num_scenarios':1,'use_render':False,'log_level':50,'traffic_density':0.0,'store_map':True}); env.reset(); print(collections.Counter(str(v.get('type')) for v in env.current_map.get_map_features().values())); env.close()"

# F3a: measured ego braking (12-trial sample)
PYTHONPATH=src uv run --no-sync python -m thesis_rl.cli.rulebook_v2_braking_trials \
    --config conf/rulebook_v2/ego_calibration.json --out <out.json> --trials-per-target 3

# F4b: frozen-index signal statistics
python3 -c "import json,collections; d=json.load(open('data/scenarionet/frozen/scenario_selection_index.json')); print(collections.Counter(r.get('has_route_traffic_light') for r in d['records']))"
```

The F5a sweep script is not checked in; it is a ~30-line Shapely loop over
`current_map.get_map_features()` reproduced in full in the F5a table above.

## 7. Limitations

- No live evaluation run was executed; F4 rests on code analysis plus the frozen
  index, not on runtime counters. P7.1 exists precisely to close that gap.
- The F5a measurement used PG geometry only; the Waymo lane-polygon seam
  behaviour is inferred from the construction method, not measured, because the
  scenario database is not mounted in this session.
- The `b_meas` estimate uses 12 trials, not the normative 40; the protocol's
  lower 5th percentile over 40 trials may differ slightly from `10.73 m/s²`, but
  not enough to approach the `4.0` cap.
