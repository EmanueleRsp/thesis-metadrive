# Specification: Driving Mission v1.1 (Route Coordinate)

## Metadata

- Specification ID: `DRIVING-MISSION-V1.1`
- Version: `1.1`
- Status: `APPROVED`
- Date: `2026-08-04`
- Approved: `2026-08-04` by explicit user approval in the task conversation
- Authoritative: `YES`
- Supersedes after approval: `driving_mission_v1.0_specification.md`
- Related ADR: `ADR-054`
- Related plan: `docs/implementation/route_coordinate_driving_mission_v1.1_exec_plan.md`
- Audit: `docs/audits/driving_mission_feasibility_2026-08-03/findings.md`

This is the approved implementation authority for the unified driving mission
contract. It supersedes DRIVING-MISSION-V1.0 after registration in
`docs/project_index.md`.

## 1. Mission and normalized record

The mission follows one frozen assigned route to its terminal. It is not future-SDC trajectory imitation, mandatory centerline following, intermediate checkpoint compliance, online shortest-path planning, or a way to put legality, heading, or lateral offset into R4. R1--R3 own safety and legality.

PG and Waymo may use source-specific offline adapters, but both emit the same normalized record. Runtime is source-independent and consumes ordered route lane occurrences, frozen start/final occurrences, frozen final goal station, provenance, geometry identities, and one frozen `final_gate_segment`. Future SDC trajectory and native navigation are never runtime authorities.

The record preserves UID, source path, split, source identity, occurrence identity/order, oriented 3D centerlines, cumulative XY arc length, goal station, adapter/provenance versions, source geometry hash, and the final gate builder identity. Correction is offline and correction-first: only uniquely source-supported repairs are allowed; no runtime fallback, invented link, goal, or route is allowed. All 3,500 identities remain in scope.

## 2. Canonical route and initialization

Concatenate ordered oriented occurrence centerlines into one immutable route

```text
r : [0,S] -> R^3
```

parameterized by XY arc length. Only consecutive duplicate points and numerically null segments may be removed; no connecting geometry may be invented. The first occurrence contains the causal reset pose or ends at the shared boundary with the second. Initial association searches only the first occurrence, or the second only at that boundary, with mandatory vertical compatibility and deterministic tie-breaking toward the preceding occurrence. No global nearest, heading, or minimum-`s` heuristic is permitted. After association, `s_start = 0`. Offline correction is required when the invariant is absent.

## 3. Stateful cursor and snapshot

The tracker owns current occurrence, current segment, and previous exact `s`. Each step searches local contiguous route segments, extending only through consecutive frozen-route segments, and stores the exact nearest vertically compatible projection without saturation. Heading or motion direction may break geometric ties only; it is never a validity requirement.

The contract contains no continuity factor, bounded envelope, `CLAMPED`, bound-based freeze, accumulated travel, recovery/catch-up, HMM, or probabilistic matcher. Non-finite coordinates, empty geometry, or no vertically compatible projection are audit/data-abort errors, not recoverable mission states.

The simulator commits ego state; the tracker updates once; an immutable `MissionSnapshot` is frozen; R4, completion, semantic/LiDAR observation, Rulebook route relevance, metrics, and video consume that snapshot. Native navigation is diagnostic only.

## 4. R4 and completion

```text
delta_s_t = s_(t+1) - s_t
R4(t) = clip(delta_s_t / (22.2222222222 * delta_t), -1, 1)
```

R4 is positive forward, zero without progress, negative in reverse, and is not altered by off-route, heading, wrong-way, lateral offset, legality, R1--R3, scenario speed cap, or terminal bonuses. Clipping applies only to R4.

```text
c_inst_t = clip(s_t / s_goal, 0, 1)
c_max_t  = max(c_max_(t-1), c_inst_t)
```

Instantaneous completion may decrease; maximum completion is monotone and used for reporting/evaluation. Completion 1 does not imply success.

## 5. Observation

Semantic and LiDAR observations consume the same immutable snapshot and never reproject ego independently. Preserve ten local route samples at 5 m spacing and 50 m look-ahead, transformed to the ego frame, with existing goal masking:

```text
q_i = r(min(s_t + 5*i, s_goal)), i = 1..10
```

They are local route samples/waypoints, not checkpoints or milestones; they have no persistent state and do not affect R4, completion, or success. Route-relative heading error may be visible to policy but cannot alter mission `s`.

## 6. Offline final gate and success

There is one directed swept front-bumper gate at `s_goal`. The offline builder is source-independent and constructs exactly one frozen `final_gate_segment` as follows:

1. compute route point, tangent, and normal at `s_goal`;
2. construct the cross-section along the normal;
3. intersect it with all available static lane polygons;
4. discard vertically incompatible intersections;
5. discard lanes whose static tangent does not satisfy `t_lane^T t_route > 0`;
6. represent remaining intersections as cross-section intervals;
7. merge only connected intervals or gaps no greater than the canonical geometry tolerance (`0.01 m`);
8. parameterize the section as `x(u) = g + u*n`, with `g = r(s_goal)` and anchor `u=0`;
9. using a closed, tolerance-equivalent `covers` relation, select the unique merged component whose interval `[a,b]` satisfies `a - epsilon <= 0 <= b + epsilon`;
10. retain the final occurrence polygon only as a consistency check.
9. freeze its endpoints as the sole gate segment.

The segment stores world-space geometry, static tangent, elevation reference, final occurrence identity, provenance/evidence, source geometry hash, and builder identity. The construction directly defines the local terminal drivable surface. It does not infer a global legal relationship from proximity or parallelism. Neighbor IDs, shared boundaries, road identities, and any lateral envelope may be retained only as diagnostic/reproducibility evidence; none is a runtime authority or prerequisite.

Zero components covering the anchor are a record/geometry error. More than one component covering the anchor after tolerance merging is a builder error; no tie-break based on width, centroid, lane ID, candidate order, or source metadata is permitted. A single covering component is frozen. The builder must report opposite/perpendicular candidates, vertical exclusions, non-guidable gaps, and final-occurrence consistency. Runtime consumes the frozen segment passively and performs no lane, neighbor, boundary, component, proximity, or fallback search.

`mission_success` is true only when the front bumper crosses the frozen segment in the positive direction. It is independent of completion, speed, and stopping. Terminations are collision, physical out-of-road, and mission success. Time limits and other approved temporal limits truncate. Rulebook violations do not terminate unless an authoritative specification says so.

## 7. Migration and validation

Preserve all 3,500 UIDs, source paths, split membership, and PG/Waymo identity. Regenerate only dependent mission artifacts; do not alter source data or exclude/replace records under this document. Audit route finiteness/non-emptiness, ordered connectivity, reset invariant, goal containment, `s_goal > 0`, cursor behavior at self-intersections/roundabouts/parallel segments, vertical separation, identical route samples across observation arms, final-gate component selection, opposite/perpendicular/vertical exclusions, and absence of runtime future-SDC access.

The read-only 2026-08-04 anchor-based geometric prototype built 3,500 unique gates (PG 1,695; Waymo 1,805), found zero anchor-missing cases and zero anchor ambiguities. The four former component ambiguities all resolve uniquely because the anchor, rather than the final-occurrence polygon, selects the component. The prior 210-metadata result is historical diagnostic evidence only and is not a validity criterion.

Implementation remains unauthorized until this text is explicitly approved. Required tests cover record/geometry validation, reset and exact cursor behavior, R4/completion, shared snapshots, gate positive crossing and exclusions, termination/truncation, causality, the full 3,500-record audit, and PG/Waymo smoke tests.

## 8. Obsolete elements

Remove intermediate gates/checkpoints, future-trajectory imitation, global nearest/minimum-`s` initialization, continuity/clamp/freeze/recovery/accumulated-travel/HMM protocols, off-route R4 zeroing, speed-cap normalization, terminal bonuses, completion-as-success, independent observation reprojection, source-specific compatible-span completeness as a prerequisite, the 210-record exclusion/blocker, and runtime authority of any final lateral envelope.
