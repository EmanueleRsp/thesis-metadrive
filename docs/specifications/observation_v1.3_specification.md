# Perception-Bounded Semantic Observation Specification

**Document ID:** OBS-V1.3
**Version:** 1.3-perception-bounded
**Status:** APPROVED
**Authoritative:** YES
**Date:** 2026-07-29
**Approval evidence:** explicit user approval 2026-07-29 (`DEC-001`, `DEC-002`, `DEC-003`, `DEC-004`, `DEC-005`, `DEC-006`); `DEC-007`, `DEC-008`, `DEC-009`, `DEC-010`, `DEC-011` decided under explicit delegation
**Supersedes:** OBS-V1.2 for the selected semantic-observation implementation path
**Related decisions:** ADR-004, ADR-022, ADR-026, ADR-033
**Related documents:** ENC-V1.3, RULEBOOK-V4.7, `docs/implementation/semantic_observation_causal_correctness_v1.3_exec_plan.md`

## 1. Purpose

OBS-V1.3 keeps the perception model of OBS-V1.2 unchanged and corrects the
observation built on top of it. A static review of the builder found three
classes of defect: features whose numeric value was not the physical quantity
the contract named, channels that exposed Rulebook state or non-local map
knowledge to the policy, and dimensions that were constant or exact duplicates
of another field.

The perception contract of OBS-V1.2 §6 is carried over verbatim and is not
restated here. Everything below is either a correction or a field-list change.

## 2. Observation schema

The flat observation is `float32`, finite, and has exact dimension

```text
D = 3009
```

| Group | Shape | Flat values | Change from OBS-V1.2 |
|---|---:|---:|---|
| `ego_history` | `(5, 10)` | 50 | unchanged |
| `ego_history_mask` | `(5,)` | 5 | unchanged |
| `ego_current` | `(3,)` | 3 | unchanged |
| `route` | `(10, 7)` | 70 | unchanged |
| `route_mask` | `(10,)` | 10 | unchanged |
| `dynamic` | `(16, 5, 22)` | 1,760 | field 18 redefined (§5.4) |
| `dynamic_mask` | `(16, 5)` | 80 | unchanged |
| `static` | `(8, 13)` | 104 | fields 2-3, 4-5, 6-10, 12 redefined (§5.5) |
| `static_mask` | `(8,)` | 8 | unchanged |
| `lane_road` | `(12,)` | 12 | **-2** |
| `control` | `(8, 15)` | 120 | **-16** |
| `control_mask` | `(8,)` | 8 | unchanged |
| `interaction` | `(8, 33)` | 264 | **-16** |
| `interaction_mask` | `(8,)` | 8 | unchanged |
| `context_history` | `(21, 23)` | 483 | **-21**, renamed (§5.8) |
| `context_history_mask` | `(21,)` | 21 | renamed (§5.8) |
| `signal_onset_state` | `(3,)` | 3 | renamed (§5.8) |

`3064 - 55 = 3009`. The latent-query raw token count stays `143`: only
per-group feature widths change, never the group cardinalities.

## 3. Removed fields

| Field | Group | Reason |
|---|---|---|
| Left/right adjacent lane available | `lane_road` | Permanently `0.0`. Neither `EpisodeCache` nor `RouteLaneRecord` carries lateral adjacency, so the value was never derivable. `DEC-002` |
| Yellow-onset distance | `control` | The identical scalar already appears as `signal_onset_state[1]`. `DEC-001` |
| Yellow-onset required stopping distance | `control` | `v·dt + v²/(2a)` with constant `a` is a bijection of `signal_onset_state[2]`: information zero, and not the *current* stopping distance since it uses the onset speed. `DEC-001` |
| Incompatible entry latched | `interaction` | A Rulebook violation latch. Forbidden by OBS-V1.2 §1/§12 and by this document. See ADR-033 |
| Roundabout relation | `interaction` | Identical to bit 3 of the zone-type one-hot in the same token. `DEC-001` |
| Control governs the ego movement | `context_history` | Identical by construction to "a movement-relevant control is present": the active control is selected only when it governs the ego lane. `DEC-001` |

## 4. Causality

OBS-V1.2 §9 is carried over. Three additional rules are normative.

1. **No Rulebook state.** The observation MUST NOT read `RulebookMemory`. This
   includes `crosswalk_illegal_entries`, `vehicle_yield_illegal_entries`,
   `preexisting_ego_occupancy_zone_ids`, `resolved_signal_group_ids` and
   `resolved_stop_group_ids`. `Preexisting occupancy active` is retained but
   MUST be reconstructed by the builder: the first time a zone enters the
   candidate set, the builder records whether the ego footprint already
   intersected it, and holds that flag while the zone remains a candidate.
2. **No world frame.** Every policy-visible vector and angle MUST be expressed
   in the ego frame at step `k`. A map feature has no heading of its own; it
   MUST carry the local tangent of its geometry at the point nearest the ego,
   never a constant. A constant makes the ego-relative sin/cos pair a direct
   encoding of the ego world heading.
3. **Local control knowledge only.** Associating a traffic control with an
   actor's approach lane MUST apply the same local control horizon and vertical
   guard as the control tokens. Outside that horizon the association is `none`,
   never the type of a control the ego could not know about.

## 5. Corrected field definitions

### 5.1 Lane width

Lane width is the transverse width of the lane polygon, measured along the
normal to the lane centerline, and MUST be invariant under rotation of the map
frame. The deterministic fallback when the normal chord degenerates is
`polygon.area / centerline.length`. A bounding-box extent is not a width.

Applies to `lane_road[0]` and `route[:, 5]`. `route[:, 5]` continues to use the
lane containing the route sample, per OBS-V1.2 §6.3.

### 5.2 Boundary clearance and type

`lane_road[1:3]` are signed footprint clearances to the **nearest** qualifying
boundary on each side: positive before contact, zero at contact, negative
during overlap.

The negative branch is defined as follows (`DEC-007`): split the ego footprint
by the boundary geometry, take the piece that does not contain the footprint
centroid, and report the negated maximum distance of that piece's vertices from
the boundary. The footprint is convex, so this maximum is always attained at a
vertex and the measure is exact.

The side of a boundary MUST be determined at the same point used for its
distance, that is the point of the geometry nearest to the ego footprint. A
representative point of the whole geometry is not admissible for long features.

`lane_road[11]` is the curvature of the **assigned route**, not of the ego lane.
The OBS-V1.1 name "current lane curvature" was wrong about the implementation
and is corrected here rather than in the code.

### 5.3 Ego lateral offset

`ego_history[:, 7]` is the signed lateral offset from the **assigned-route**
centerline, not from the associated lane. The OBS-V1.1 name "lane offset" was
wrong about the implementation and is corrected here.

### 5.4 Route lateral offset of other objects

`dynamic[:, :, 18]` and `static[:, 12]` are **signed** lateral offsets from the
assigned route, positive to the left of the route tangent (`DEC-006`). The
OBS-V1.1 wording "route lateral distance" implied an unsigned magnitude, while
the implementation clamped a signed value to `[0, 1]` and collapsed the whole
right half-plane onto zero.

### 5.5 Static objects

- Position is the point of the geometry nearest to the ego, per OBS-V1.2 §6.1.
- Heading is the local tangent at that point (§4 rule 2).
- Dimensions describe a **local window** of radius 10 m around the ego, not the
  extent of the whole geometry, and MUST be measured with a rotation-invariant
  construction (minimum rotated rectangle, falling back to the clipped length
  for line-like geometries).
- The type one-hot encodes the taxonomy the source can actually discriminate
  (`DEC-003`, `DEC-010`): `0` traffic cone, `1` traffic barrier, `2` other
  obstacle, `3` road boundary, `4` other non-drivable. Slots `0-2` come from
  actors, slots `3-4` from the map feature catalogue.
- **Every slot MUST be reachable.** The OBS-V1.2 taxonomy of
  cone/barrier/wall/stationary-vehicle/generic was never emitted (every static
  object was reported as `generic`), and its first replacement kept two dead
  columns: `stationary vehicle` is unreachable by construction, because a
  parked car carries `ActorClass.VEHICLE` and is emitted in the `dynamic`
  group, and `unknown` was never assigned because the live class mapping fails
  closed on unrecognised actors.
- Slots `0-2` are populated from `ActorSnapshot.static_subclass`, a refinement
  of `STATIC_COLLIDABLE` carried for the observation only. The Rulebook does not
  branch on it. The distinction is admissible under §10.1: after physical
  admission this contract assumes ideal semantic classification, which a fused
  camera/LiDAR stack supports, and cones, barriers and warning triangles differ
  both in appearance and in the response they require. A warning triangle is a
  hazard marker rather than a rigid obstacle and shares slot `2` with statics
  whose source type is not recognised.
- A static object the assigned route cannot accept MUST degrade its own token
  and increment the route-incompatibility diagnostic, never fail the whole
  observation.

### 5.6 Traffic controls

The control token has 15 values in this order: midpoint `x, y`; signed route
distance; direction `sin, cos`; control type one-hot (2); signal state one-hot
(5); controls ego movement; active control; state valid.

- All route distances to a control are measured from the **ego front bumper**,
  because the Rulebook decides control-line crossing with the swept front
  bumper. A centre-based distance would offset every threshold the policy has
  to learn by half a vehicle length. This applies to the control token, the
  context row and the signal-onset state.
- The control type MUST be read from the control record. It MUST NOT be
  inferred from the signal-state string: an unobservable signal keeps the
  sentinel state and would otherwise be encoded as a stop control.
- A stop control sets `not-signal` at index 4 of the state one-hot with
  `state_valid = 1`. An unobservable signal emits an all-zero state one-hot with
  `state_valid = 0`. The two situations are distinct and MUST stay
  distinguishable.
- Ranking is OBS-V1.1 §8.3 criteria 1-2 only: controls governing the ego
  approach lane first, then non-negative route distance, then absolute route
  distance, then control group id. Criterion 3 ("not yet resolved") is
  permanently dropped: it was implementable only by reading a Rulebook latch
  (ADR-033, `DEC-009`).

### 5.7 Interaction tokens

The token has 33 values: zone centroid `x, y`; route distance to entry and
exit; zone type one-hot (4); ego inside; other inside; other actor type one-hot
(3); ego and other occupancy intervals (4); interval validity and open-end (4);
ego and other approach control one-hot (4 + 4); map/control-derived
right-of-way (3); pre-existing occupancy active.

- Entry and exit distances are the true curvilinear limits of the zone along
  the assigned route, obtained by intersecting the route polyline with the zone
  polygon. Reporting the centroid abscissa for both makes the zone appear to
  have zero depth and renders the occupancy-interval comparison unusable.
- Ranking MUST be by criticality, using only causally available criteria: ego
  or other inside the zone; overlapping occupancy intervals; pre-existing
  occupancy; smaller route distance to entry; smaller valid `t_in`; smaller
  actor distance; then `(zone_id, actor_id)` as tie-break. Ordering by
  identifier alone is not admissible.

### 5.8 Context history

Renamed from `compliance_history` by `DEC-011`. The previous name described the
Rulebook rule the group was originally introduced to support, not its content,
and invited the reading that the group carries Rulebook verdicts. It does not:
every value is an observation of the ego's own state or of the local road
context, recorded by the observation builder. No field of this group is read
from `RulebookMemory` (§4, ADR-033).

The row has 23 values: ego speed; left/right local clearance; left and right
boundary type one-hot (4 + 4); dashed-boundary intersection; dashed-boundary
continuity; movement-relevant control present; active-control continuity;
active control type one-hot (2); observed signal state one-hot (5); signed
front-bumper distance to the active control.

The two continuity indicators MUST be `0.0` unless the previous recorded row
belongs to step `k-1`. They distinguish the same currently relevant boundary or
control from a newly relevant one; across a step gap that distinction is not
established.

The dashed-boundary lookup MUST select the nearest intersecting dashed marking
at the ego's own level, applying the vertical guard.

#### 5.8.1 Why the length is 21

The control step is `physics_world_step_size × decision_repeat = 0.02 × 5 =
0.1 s`, so 21 rows span 2.1 s: the current step plus the 2.0 s of
`DASHED_TCAP_S` (Rulebook v2 `components/road.py`).

This length is deliberately equal to the decision window of the longest
road-discipline rule, and the reason must be stated rather than left implicit.
A behaviour evaluated over a 2 s window is not merely harder to learn from
0.5 s of memory — it is not identifiable from it, because two trajectories that
differ only outside the visible window are indistinguishable to the policy while
being scored differently. Sizing observed memory to the decision horizon is the
standard remedy.

Matching the *window* of a rule is not the same as observing its *output*. The
group carries no timer, latch or verdict; it carries the raw quantities over the
interval in which the agent must decide. The prohibition in §4 is on Rulebook
state reaching the policy, and it is not weakened here.

Two consequences follow and MUST be documented rather than silently relied on:

- `ego_history` (5 × 10) and `dynamic` (16 × 5 × 22) cover 0.5 s of ego
  dynamics and of other actors. `context_history` is the only source of memory
  beyond 0.5 s for any quantity, and the only history of the road context at
  all. It is not redundant with either.
- The most recent row duplicates information already present at the current
  step in `lane_road`, `controls` and `ego_history[-1]` — 23 of 3009
  dimensions. The uniform ring buffer is kept in preference to a special case
  for the last row.

## 6. Instrumentation

Overflow diagnostics MUST be produced for `dynamic`, `static`, `controls` and
`interactions`: total candidates, selected, dropped for capacity, and the
number of conflict-critical candidates dropped. They remain debug-only and
never enter the observation, the reward or the curriculum.

## 7. Legacy modes

`semantic_v2` and every other historical observation mode retain their
contracts bit-for-bit (`DEC-008`). The corrections in this document are
implemented as overrides on the OBS-V1.3 builder; the shared base methods are
untouched. A regression test asserts that the legacy builder still emits a
14-value `lane_road` group, a 35-wide interaction token, and the historical
bounding-box lane width.

## 8. Acceptance criteria

- The schema has exactly 3,009 finite `float32` values and exactly 143 LQ raw
  tokens.
- An observation built from a context with a populated `RulebookMemory` is
  bit-identical to one built from the same context with an empty memory.
- A rigid rotation of the whole scene leaves the static tokens unchanged.
- Lane width is invariant under a 90-degree rotation of the map.
- The nearest boundary wins for any catalog iteration order; an overlapping
  boundary reports a negative clearance.
- An unobservable signal is typed as a signal with an all-zero, invalid state;
  a stop control sets `not-signal`.
- A right-side actor reports a negative route lateral offset of the same
  magnitude as its left-side mirror.
- A critical interaction is not dropped in favour of an alphabetically earlier
  non-critical one.
- Every correction has a deterministic regression test.

## 9. Compatibility

OBS-V1.3 is incompatible with OBS-V1.2 flat dimensions and encoder weights.
There is no automatic migration. Runs, checkpoints, normalization statistics
and manifests MUST carry the schema version and dimensions. Historical
experiments remain reproducible only by selecting their original observation
and encoder contracts.

## 10. Known limitations

Carried forward and unchanged by this revision:

1. Semantic tracking and classification remain ideal after physical admission.
2. Other actors' lane association (`live_lane_id`) is exact. This is the
   heaviest remaining idealisation: it drives the same-lane relation, the
   conflict-zone pairing and the approach-control association, and a real stack
   is least certain exactly where it matters most.
3. Ego localisation on the HD map is exact.
4. Static map features are admitted without an occlusion test. This is
   deliberate: it models map knowledge, not a detector.
5. Right-of-way availability depends on scenario metadata carrying
   `rulebook_vehicle_yield`; its dataset coverage is a separate question from
   its causal validity.
