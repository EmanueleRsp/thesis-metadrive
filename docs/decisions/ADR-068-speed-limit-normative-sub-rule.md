# ADR-068: `speed_limit` becomes a normative R3 sub-rule, with an observation amendment

- Status: **Approved** — carried by `RULEBOOK-V5.1`, approved 2026-08-14
- Date: 2026-08-10
- Approval evidence: pending; carried by `rulebook_v5.0_UNDER_REVIEW`. The user
  accepted the checkpoint-compatibility break explicitly, on the grounds that
  the production runs have not started.
- Affected specifications: `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
  §5.7, §7; amends `docs/specifications/observation_v1.3_specification.md` and
  `docs/specifications/observation_lidar_v2.0_specification.md`.

## Context

`speed_limit_compliance` carries weight 4 of 16 in nuPlan's weighted-average
block and is the only metric in that benchmark's closed-loop score with no
counterpart in this rulebook. Rulebook v1 had `check_speed_limit`; the v2
registry does not.

The data supports it on the Waymo panel. Waymo records carry genuine posted
limits converted from mph - 16.09 / 24.14 / 40.23 / 48.28 / 64.37 / 72.42 /
80.47 km/h, i.e. 10 / 15 / 25 / 30 / 40 / 45 / 50 mph - on essentially every
lane, with 0.0 marking an unrecorded limit, and each such lane also carries the
source datum `speed_limit_mph` it was converted from. All 1100 Waymo train
records carry a limit on the assigned route.

### The PG panel carries a value, and it is not a speed limit

An earlier draft of this ADR stated that PG carries only the 1000 km/h sentinel.
That is wrong, and the correction is the reason this decision needs a provenance
gate rather than a value filter. Measured over the 1100 PG train records, 697
carry a route-lane `speed_limit_kmh` and **every one of them is exactly 20**.

The value is a lane-constructor default, and which default depends on the code
path that built the lane:

| producer | value | PG blocks |
|---|---:|---|
| `metadrive/component/lane/abs_lane.py:22` — `self.speed_limit = 1000  # should be set manually` | 1000 | straights, first block |
| `metadrive/component/pgblock/create_pg_block_utils.py:26,45` — `speed_limit: float = 20` | 20 | curves, intersections |

Neither is a posted limit. Worse, the unit is not the one the exported key
claims. MetaDrive's PG blocks document their speed limits **in m/s**:

- `metadrive/component/pgblock/ramp.py:33` — `SPEED_LIMIT = 12  # 12 m/s ~= 40 km/h`
- `metadrive/component/pgblock/tollgate.py:19` — `SPEED_LIMIT = 3  # m/s ~= 5 miles per hour`

Both comments are self-consistent, and the alternative reading is physically
absurd (a 12 km/h motorway ramp, a 3 km/h toll booth). The export path then
writes `lane.speed_limit` out verbatim under a `_kmh` key with no conversion
(`metadrive/component/road_network/node_road_network.py:321` and
`edge_road_network.py:127`); the only unit conversion anywhere in MetaDrive is
`mph_to_kmh`, on the real-map read path. MetaDrive contradicts itself once, at
`component/vehicle/base_vehicle.py:963` (`lane.speed_limit < self.speed_km_h`),
but that `overspeed` property has no caller in the codebase and carries no
weight as evidence of intent.

This is an upstream defect, not one this project introduced. Its consequence
here is concrete and was measured: reading the field at its label turns 20 m/s
into 5.56 m/s, and the PG panel's own reference driver — MetaDrive's `IDMPolicy`
at `NORMAL_SPEED = 30` km/h = 8.33 m/s — then exceeds `limit + 2.23` on
**51.21 %** of PG steps. Under the m/s reading the same rule fires on 0 %. Either
way the field cannot be priced: one reading is off by 3.6x, and the other reports
a constructor default as a traffic norm.

Measured on the Waymo expert, cost `clip((v_ego - (v_limit + tau)) / v_limit, 0, 1)`:

| tolerance tau | 0 | 1.0 m/s | **2.23 m/s** | 4.47 m/s |
|---|---:|---:|---:|---:|
| % of steps violated | 2.200 | 0.002 | **0.000** | 0.000 |

Over 217,187 applicable steps the expert never violates it at nuPlan's published
tolerance.

## Decision

`speed_limit` is added as a normative R3 sub-rule with tau = 2.23 m/s (5 mph).

**A lane speed limit is admitted as normative only where the record also carries
the real-map `speed_limit_mph` datum it was converted from.** Where it does not,
the sub-rule is inapplicable on that lane. This subsumes the existing rejections
of the unrecorded `0.0` and of the `>= 999` sentinel, and additionally rejects
every PG constructor default.

The posted limit is added to both observation paths; `D` changes and checkpoint
compatibility is intentionally broken.

### Why provenance and not the value

Three alternatives were considered against the PG defect above.

1. **Filter by value** — reject implausibly low limits. Rejected: Waymo
   genuinely posts 8.05 and 16.09 km/h (5 and 10 mph) on driveways and parking
   aisles, so any floor high enough to exclude PG's 20 also discards real data,
   and the threshold would be fitted to one simulator's constant.
2. **Correct the unit for PG** — read the PG field as m/s. Rejected: it makes
   the rulebook depend on reverse-engineering an upstream bug that upstream may
   fix, and even when correct it prices a constructor default as a traffic norm.
   It also silently reintroduces the ramp (12) and tollgate (3) block constants.
3. **Regenerate the PG corpus with explicit limits.** Rejected for this change:
   it invalidates the frozen selection index and every split derived from it,
   which is a dataset-versioning decision far larger than this sub-rule, and it
   would still be inventing limits MetaDrive's maps do not model.

Provenance is the only criterion that is a property of the data rather than of a
threshold, is verifiable per record, and stays correct if upstream changes the
default or fixes the unit.

## Rationale

**It closes a degeneracy that R4 creates.** R4's margin is
`clip(delta_s / (22.22 * delta_t), -1, 1)` - the ratio of route speed to a
reference speed of 80 km/h. The reward therefore directly rewards driving at up
to 80 km/h, on a 25 mph street as much as on a motorway, and nothing in the
current rulebook opposes it except the consequences of leaving the road or
colliding. This is not a rule added for completeness.

**It passes every test that rejected `rss`.** Controlled-invariant: the ego can
always decelerate, and no other agent can push it above the limit - the exact
property the RSS envelope lacks. Memoryless. Observable and physically
plausible: the posted limit is an HD-map attribute every production stack
carries, so it is a real mid-perception feature and not a quantity invented to
make a reward rule computable. Anchored: 2.23 m/s is published, not fitted.

**Inapplicability is honest, not a gap.** Under the provenance gate the rule is
inapplicable throughout the PG panel and is reported as such, using the
applicability mechanism that already exists. The consequence is stated rather
than hidden: on PG nothing opposes R4's 80 km/h reference except the vehicle's
own 80 km/h cap, so the degeneracy this rule closes is closed on the Waymo half
of the training mixture only. That is a property of what PG models, not a
weakening of the rule — a procedurally generated road carries no posted limit to
comply with. It is recorded as a limitation in RULEBOOK-V5.0 §11.

## Prohibition

Two fallbacks are prohibited, both of which would turn a vehicle or generator
constant into a legal norm:

1. The v1 extractor at `src/thesis_rl/envs/wrappers.py:462` falls back to
   `ego_vehicle.max_speed_km_h` when the lane carries no limit. That constant is
   the vehicle's own cap (80 km/h), not a legal limit. It must not be carried
   over.
2. `ScenarioLane.MAX_SPEED_LIMIT = 100` km/h
   (`metadrive/component/lane/scenario_lane.py:24,40`) is MetaDrive's own
   fallback when a lane record carries neither speed-limit key. It is a
   simulator default and must not reach the rulebook: a lane with no recorded
   limit is inapplicable, never 100 km/h.

## Consequences

Recorded in `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
and in `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`.
This ADR is not an implementation authorisation on its own: the specification
must be promoted to `APPROVED` first.
