# ADR-035: Lateral-RSS Applicability Restricted To Longitudinally Overlapping Pairs

- Status: APPROVED
- Date: 2026-07-29
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-29
- Supersedes: NONE (amends the applicability contract introduced by ADR-025)
- Affected specification: `docs/specifications/rulebook_v4.8_specification.md`,
  version 4.8, status `APPROVED` — §7 (steps 3-5 of the longitudinal gate) and
  §8 (applicability) are amended by this decision and require a specification
  amendment to be brought back into agreement with the implementation
- Affected ExecPlan:
  `docs/implementation/rulebook_v2_evaluation_fidelity_v4.9_exec_plan.md`
  (`DEC-EF-01`, `DEV-EF-01`)

## Context

Rulebook v4.8 §8 makes a lateral-RSS pair applicable whenever the actor is a
`VEHICLE`, both ego and actor have an unambiguous lane association, the lanes
belong to the same road structure or are compatible adjacent lanes, and the
local legal directions are concordant. §7 steps 3-5 then require identifying
which of the pair is the rear vehicle and which the front, and reusing the
v4.7 §6.2.2 RSS-longitudinal formula for that ordered pair.

The implementation of `transition.py::_rss_lateral_candidates` checked only
that each actor was concordant with **its own** lane, and
`_longitudinal_unsafe_gate` placed the ego unconditionally in the rear role,
documenting the choice as the only one consistent with the available
calibration (`RSSCalibrationArtifact.ego_min_brake_mps2` calibrates the ego
only).

An audit on 2026-07-29 reproduced three consequences of this scoping, all on
synthetic straight/perpendicular geometry with the frozen §9 parameters:

1. A vehicle 20 m **behind** the ego, receding (ego 20 m/s, actor 5 m/s),
   yields `longitudinal_unsafe = True` and `q_RSS,lat = 1.0`. A vehicle 20 m
   behind and **closing** (ego 5 m/s, actor 20 m/s) yields
   `longitudinal_unsafe = False` and `q_RSS,lat = 0.0`. The roles are
   inverted: a false positive at maximum cost, and a false negative.
2. A vehicle on a perpendicular intersection branch, 30 m ahead and 6 m to the
   side, yields `q_RSS,lat = 0.937`, although §8 requires `NOT_APPLICABLE` for
   different intersection branches.
3. Two vehicles in the **same** lane have a lateral edge-to-edge gap of exactly
   `0.0` by construction, while `d_safe^lat ≈ 0.1625 m` at zero inward speed.
   Any same-lane leader within the RSS-longitudinal distance therefore costs
   `1.0`. Measured at 20 m/s: `q_RSS,long` = 0.840 / 0.460 / 0.156 at 15 / 40 /
   60 m, while `q_RSS,lat` = 1.000 in all three cases.

Consequence (3) is not an implementation error — it follows from a literal
reading of §7-§8 — but it makes `R2 = max{...}` a binary indicator in ordinary
car-following, discarding the graded longitudinal signal the specification
deliberately defines. Consequences (1) and (2) additionally make `R2` report a
violation during correct driving. Under the configured
`bounded_satisfaction_rank` scalarization the satisfied/violated pattern
carries a rank term of `priority_base**2 = 4.04` for `R2`, so a spurious
violation dominates every `R3`/`R4` contribution.

The lateral-RSS model assumes a shared tangent axis along which both vehicles
travel, with the safety-relevant dynamics normal to it. That assumption holds
for vehicles that are **abreast**. It does not hold for a vehicle behind, a
vehicle ahead in the same lane, or a vehicle on a crossing branch — which is
precisely the set of cases that produced the three defects.

## Decision

A lateral-RSS pair is applicable only when, in addition to the conditions of
v4.8 §8 already implemented:

1. the tangent-axis intervals of the two footprints, measured in the ego's
   anchored route frame, **overlap** (using the existing
   `SIGNED_DISTANCE_EPSILON_M` deadband); and
2. the misalignment between the two lanes' local tangents is within
   `LATERAL_RSS_MAX_TANGENT_MISALIGNMENT_RAD = pi/4` (45°).

For an admitted pair the tangent intervals overlap by construction, so v4.8 §7
step 2 already sets `I_long,unsafe = 1`. Steps 3-5 of §7 — the identification
of the rear and front vehicle and the reuse of the RSS-longitudinal formula —
are therefore unreachable and are removed rather than patched. No braking
calibration for a non-ego vehicle is introduced.

The cost formula of §7, the frozen parameters of §9, the worst-of aggregation,
and the `R2` aggregation of §7 are unchanged.

## Rationale

- **One rule, three defects.** The overlap requirement removes the rear-vehicle
  role inversion, the same-lane saturation, and the crossing-branch false
  positive simultaneously, because all three are the same modelling error:
  applying an abreast-pair metric to pairs that are not abreast.
- **It dissolves the calibration problem instead of solving it.** The
  alternative of introducing a conservative minimum braking constant for
  non-ego vehicles would add a frozen parameter with no cited source, which
  v4.8 §9 forbids without a new specification version. With the overlap
  requirement there is no rear non-ego vehicle to calibrate.
- **The metric remains meaningful and graded in its proper domain.** Verified
  on two abreast vehicles in adjacent lanes with a 1.65 m gap: an inward drift
  of 1.0 m/s gives `d_safe^lat = 1.41 m` and cost `0.0`; 1.5 m/s gives
  `d_safe^lat = 2.5 m` and cost `0.34`.
- **The angular bound is a derived implementation constant, not a scientific
  parameter.** 45° excludes perpendicular (90°) and opposing (180°) lanes while
  tolerating the 20-30° divergence between the tangents of an ego and an
  abreast vehicle on a tight curve. It is recorded here so it can be carried
  into the specification amendment rather than living silently in code.
- **Precedence.** The repository's guiding sources place explicit user approval
  above specification text. The user directed on 2026-07-29 that a
  specification is a documentation and reproducibility instrument, not a reason
  to preserve a falsified measurement, and approved this decision.

## Consequences

- `q_RSS,lat` becomes `NOT_APPLICABLE` for rear vehicles, same-lane leaders,
  and crossing branches. `R2` recovers the graded `q_RSS,long` signal in
  car-following, which is the most common traffic situation.
- **Accepted coverage loss:** there is no preventive lateral signal for a
  vehicle ahead in an adjacent lane that begins to converge *before* the two
  are abreast. This is consistent with v4.8 §8's own position that excluding a
  pair from the lateral sub-metric loses no safety coverage, because TTC,
  collision, crosswalk/vehicle-conflict-zone and vehicle-yield remain active
  and unchanged.
- **Numeric compatibility break:** `R2` values, and therefore the scalarized
  reward, change on every run containing traffic. Runs produced before this
  decision are not comparable with runs produced after it. The boundary must be
  stated when reporting results.
- v4.8 §7 steps 3-5 and §8 no longer describe the implementation. A
  specification amendment is required to restore agreement; until it is
  approved, this ADR plus `DEV-EF-01` in the ExecPlan are the authoritative
  record of the divergence.
- No public interface, configuration key, observation shape, or checkpoint
  format changes.

## Addendum 2026-07-29: three further decisions recorded under this ADR

The same session resolved three related items. They are recorded here rather
than in separate ADRs because each is either plain conformance or a derived
implementation constant, not a new scientific choice.

### §2.9.6 control-line anchor (conformance, no deviation)

`derive_control_line` built the orthogonal section through the **raw control
point** and selected the step-5 component by distance to it, whereas §2.9.6
steps 3 and 5 are both anchored on `q_c`, the projection of the control point
onto the lane centerline. Waymo's `STOP_SIGN.position` is the physical sign at
the roadside: measured over 991 sign/lane pairs from a 400-scenario sample,
**985 (99.4%) fall outside the lane polygon**, with a median lateral offset from
the centerline of **4.32 m** (p90 6.46 m, max 19.03 m).

Consequences of the defect were both a wrongly placed stop line when derivation
succeeded, and an abort ("multiple unresolved components") when two components
tied within `eps_geom` of a point outside the polygon. Measured over the same
sample: stop-control derivation succeeded for **239/991 (24%)** with the raw
anchor and **956/991 (96.5%)** with `q_c`. Roughly three quarters of the
dataset's stop signs were silently dropped.

Fixed to follow §2.9.6. No deviation; no decision required.

Latent fragility recorded but not addressed: the vertical-compatibility check
compares the sign's *mounted* height against the lane elevation. The measured
median `|dz|` is 2.57 m against a 3.0 m tolerance intended for grade separation,
and 2.3% of signs already exceed it. The check happens to work only because a
sign pole is shorter than an overpass clearance; tightening
`VERTICAL_COMPATIBILITY_TOLERANCE_M` would start dropping controls. The
principled fix is to compare the sign's ground projection, which is a separate
decision.

### Route-projection continuity bound (`ROUTE_CONTINUITY_JUMP_FACTOR = 2.0`)

`RoutePolyline.project` used `previous_s_m` only to break ties among candidates
already within `eps_geom` of the minimum planar distance, so on a
self-intersecting or closely parallel route a strictly-closer far branch won
outright and the coordinate jumped, producing spurious `R4` progress. The
post-state projection is now given a plausibility bound of
`2 * v_max * delta_t`.

The factor is a derived implementation constant, not a scientific parameter:
the physical bound on one step's advance is `v_max * delta_t` — the same
normalizer v4.7 §8.2 already uses for `m_4` — and the factor of 2 covers the
fact that cutting the inside of a curve advances the centerline coordinate
faster than the ego's own displacement.

The bound is a **preference, not a gate**: when no candidate is plausible the
unbounded selection is kept. An earlier attempt to fail closed instead broke
three pre-existing transition tests that legitimately move the ego further than
the bound in one synthetic step, and it would have converted a rare geometry
into an episode abort with no compensating benefit.

### R1 attribution for a contact without a pre-state record (user decision)

A new contact whose actor has no pre-state record has no computable closing
speed (v4.9 §4.1 needs the pre-state normal). The user's reasoning, adopted
here: if the actor did not exist at decision time, no alternative action was
available to the ego, so `R1 = 0` is the correct **causal attribution** rather
than a fallback — consistent with the Rulebook charging the ego for the risk its
own action creates.

The implementation adds the discriminator that reasoning implies. The post-state
separates two situations that were previously indistinguishable:

- actor **present in the post-state** — it genuinely appeared during the step;
  `R1 = 0`, reported as `appeared_onset_actor_ids`;
- actor **in neither snapshot** — the snapshot pipeline never observed an object
  the physics engine did (e.g. a class excluded from the live actor registry
  that can still produce a Bullet contact); reported separately as
  `unobserved_onset_actor_ids`, because this is an instrumentation gap and not
  an exculpation.

The second branch is kept non-fatal for now and left measurable rather than
escalated, for the same reason as the bound above.

## Addendum 2026-07-30: control vertical frame and dashed-line cost shape

Two further decisions approved by the user on 2026-07-30, recorded here rather
than in separate ADRs because both are corrections to the same evaluation-fidelity
programme. `R4` was explicitly excluded from this round of approval.

### `DEC-EF-07` — control vertical compatibility bounds a mounting height

`derive_control_line` rejected a control when
`abs(control_point_z - controlled_lane_z) > VERTICAL_COMPATIBILITY_TOLERANCE_M`.
That constant is 3.0 m because it is the order of magnitude of an overpass
clearance: below it, two geometries are "the same road"; above it, two levels.

Measured over 250 frozen Waymo scenarios, with lane centerlines and control
points expressed in the same `z_origin` frame:

| Control kind | n | min | p01 | median | p95 | max | `dz < −0.5 m` | `\|dz\| > 3.0 m` |
|---|---|---|---|---|---|---|---|---|
| `TRAFFIC_LIGHT` (`stop_point`) | 1319 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.0% | 0.0% |
| `STOP_SIGN` (`position`) | 1901 | −3.315 | −0.251 | +2.567 | +2.971 | +3.668 | 0.4% | 3.8% |

The distributions are qualitatively different because the two reference points
are: `TRAFFIC_LIGHT.stop_point` lies on the carriageway, and `STOP_SIGN.position`
is the physical sign on its post. The test was therefore comparing a pole height
against an overpass clearance, and it happened to work only because a stop-sign
post is shorter than an overpass clearance — by about 0.4 m of margin at p95.

It erred in both directions. It discarded 3.8% of stop signs for a reason
unrelated to level separation, and it *accepted* a control belonging to a road
2.5 m below this one, which is precisely the confusion the constant exists to
prevent.

**Decision.** Replace the symmetric test with an asymmetric, mounting-aware
bound. The control point's ground projection is unknown but bounded: a control is
mounted at or above the surface it governs and never below it, so the ground lies
in `[control_point_z - CONTROL_MOUNTING_HEIGHT_MAX_M, control_point_z]`. The
control is level-compatible when that interval reaches the surface:

    -CONTROL_BELOW_SURFACE_TOLERANCE_M <= dz <= CONTROL_MOUNTING_HEIGHT_MAX_M

with `CONTROL_MOUNTING_HEIGHT_MAX_M = 7.0` (MUTCD requires 5.2 m clearance to the
bottom of an overhead sign, so a gantry-mounted sign centroid reaches ~6-7 m) and
`CONTROL_BELOW_SURFACE_TOLERANCE_M = 1.0` (a downward *noise* allowance only,
deliberately far below a grade separation; p01 of the measured offset is
−0.251 m and only 0.4% of samples fall below −0.5 m).

Additionally, `_route_curvilinear_crossing_s_m` filtered candidate crossings by
vertical compatibility against the raw control-point elevation. The crossing lies
on the control line, which lies on the carriageway by construction, so the
comparison now uses the control line's own elevation. Previously a legitimately
mounted sign leaked its mounting height into that test and could discard the only
crossing on the ego's own level, turning the control into a spurious off-route
skip.

**Why the loosened positive bound is acceptable.** The strict level test was
never the discriminator it appeared to be. The controlled lane comes from the
**authoritative** association in the data — `STOP_SIGN.lane` and
`TRAFFIC_LIGHT.lane` — which is trusted everywhere else in the adapter, and a
control on an overpass lane is associated with overpass lanes, not with the ego's.
The route-crossing test then independently confirms the level. This check is
therefore a plausibility bound, and it is documented as such.

**Consequence.** Changes which controls are derivable, hence scenario
eligibility, hence dataset composition. Measured on a 150-scenario Waymo sample:
`stop` controls 69 → 72 (+4.3%), `signal` controls 93 → 93 (unchanged, as the
identically-zero offsets predict), scenarios with at least one control 94 → 96.
The frozen selection index must be rebuilt before this takes effect at runtime.

### `DEC-EF-08` — dashed-line cost graded in space as well as in time

Supersedes the deferred `DEC-EF-05`. v4.7 §7.5.3 makes the dashed-line cost a
function of dwell time alone, saturating at 1.0 after 2 s. Activation, however,
is a spatial **step**: the marking is active when it touches the ego footprint
anywhere, down to a bumper corner. The two together mean an ego drifting with one
wheel over the marking and an ego straddling the marking with its centre on it
receive the *same* cost of 1.0. `R3` cannot distinguish them, and since `R3`
outranks `R4` in the hierarchy, both behaviours are taught the same penalty.

**Decision.** Multiply the existing time ramp by a lateral penetration factor:

    cost = p * f(timer),    p = clip(1 - d / d_max, 0, 1)

`f` is unchanged (cost-free until `DASHED_T0_S`, quadratic ramp, saturated at
`DASHED_TCAP_S`), so the "initial cost-free period then progressive saturation"
shape is preserved and only the spatial axis is added. `d` is the
centroid-to-marking distance and `d_max` is the distance from the centroid to the
footprint boundary **in the direction of the marking**, so `p = 1` when the
marking passes through the centroid and `p = 0` when it is tangent to the
footprint edge.

`d_max` is deliberately the directional half-extent rather than half the vehicle
width (0.926 m). During a lane change the ego is yawed relative to the marking,
so its extent towards the marking exceeds half its width; a fixed half-width
normalizer would leave a dead band in which the marking is still inside the
footprint but `p` has already reached 0.

The two factors retain distinct meanings and are reported separately as
`time_factor` and `lateral_penetration`: the timer is *how long* the ego has been
engaged with this marking, `p` is *how badly* it is engaged right now. An ego that
lingers at the edge (cost 0, timer accumulating) and then cuts inward is
therefore penalized immediately, without a fresh grace period.

**Consequence.** Changes `R3` on every run and invalidates reward comparison with
prior runs. v4.7 §7.5 requires an amendment for documentation; the implementation
is a deliberate, approved deviation until then.

## Alternatives Considered

- **Keep literal v4.8 conformance.** Rejected: it preserves a measurement that
  reports a maximum-cost safety violation during correct driving.
- **Three separate patches** (geometric rear/front identification, a lane
  adjacency predicate, a same-lane exclusion). Rejected: strictly more code and
  more constants for the same behaviour, and it would still require a braking
  value for non-ego rear vehicles.
- **A conservative minimum braking constant for non-ego vehicles.** Rejected:
  introduces an uncited frozen parameter, forbidden by v4.8 §9 without a new
  specification version, for a case the overlap requirement removes entirely.
