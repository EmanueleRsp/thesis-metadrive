# DRIVING-MISSION-V1.1.1 — Mission-local Route Occurrence Orientation Amendment

- Specification ID: `DRIVING-MISSION-V1.1.1`
- Parent specification: `DRIVING-MISSION-V1.1`
- Status: `APPROVED`
- Authoritative: `YES`
- Date: `2026-08-04`
- Approved: `2026-08-04` by explicit user approval in the task conversation

## 1. Purpose and scope

This document is a material amendment to the approved `DRIVING-MISSION-V1.1`
contract. It corrects the representation of route occurrence orientation and
mission-local trimming. It does not change the frozen topological route: the
ordered `assigned_route_lane_ids` already present in the frozen record remains
the sole route sequence.

This amendment does not authorize planning, shortest-path search, native
navigation, map-only route replacement, runtime fallback, source-pickle edits,
record exclusion, or scenario substitution.

## 2. Offline-only use of the SDC trajectory

The complete SDC trajectory may be consumed by a source-specific offline
adapter solely to determine and validate the direction in which each frozen
route occurrence is traversed. The trajectory is not persisted in the mission
record and is never read by runtime code.

The causal reset pose and frozen terminal goal anchor remain the boundary
conditions of the mission. The trajectory is not an alternative route source
and must not be used to choose lane IDs outside `assigned_route_lane_ids`.

## 3. Oriented route occurrences

Every element of the frozen ordered lane-ID sequence is represented as a
distinct immutable occurrence, including repeated lane IDs:

```text
RouteOccurrence
├── occurrence_index
├── lane_id
├── orientation: FORWARD | REVERSED
├── oriented_centerline_points_xyz
├── source_start_s_m
├── source_end_s_m
├── orientation_provenance
└── source_geometry_hash
```

For each occurrence, the offline adapter associates temporally ordered valid
SDC poses with that occurrence, projects them onto the source centerline with
mandatory vertical compatibility, and evaluates the ordered source stations.
The association is sequence-constrained by the frozen occurrence order: once
the adapter advances to a later occurrence, a pose cannot be assigned back to
an earlier occurrence. This is required because lane polygons can overlap at
junctions and a global polygon scan would contaminate an earlier occurrence
with later poses.

- `FORWARD` is selected only when verified temporal progress is predominantly
  toward increasing source station.
- `REVERSED` is selected only when verified temporal progress is predominantly
  toward decreasing source station.
- Ego heading is not the primary authority.
- If the orientation is not uniquely determined, the adapter must fail the
  audit and report the record; it must not apply an orientation fallback.

For a source lane length `L`, a reversed occurrence uses

```text
s_oriented = L - s_source
```

and stores the source centerline points in reverse order, with mission-local
tangents inverted consistently. The original source lane record is not
mutated.

## 4. Frozen route continuity

After orientation, occurrences remain in exactly the frozen order. The
oriented end of each occurrence must coincide with the oriented start of the
next occurrence within the canonical horizontal and vertical tolerances. No
connector, new lane ID, or planning result may be introduced.

If no orientation assignment is both supported by the temporally associated
poses and geometrically connected, the record is reported as ambiguous or
invalid for offline correction. It is not silently repaired at runtime.

## 5. Mission-local canonical route

The adapter constructs one immutable mission-local canonical route

```text
r : [0, s_goal] -> R^3
```

from the oriented frozen occurrences. It must:

1. begin exactly at the projection of the causal reset pose;
2. preserve the ordered oriented centerline geometry of the frozen route;
3. end exactly at the frozen terminal goal anchor;
4. contain the explicit projected reset and goal points as endpoints;
5. remove the prefix before reset and suffix after goal;
6. have `s_start = 0` by construction; and
7. have `s_goal` equal to the positive XY arc length of the trimmed polyline.

Only consecutive duplicate points and numerically null segments may be
removed. No connector or interpolation that changes the frozen route geometry
may be invented.

The normalized mission record stores the mission-local polyline directly. A
source-station offset must not be an authority-bearing runtime field. Any
implementation retaining an internal offset for compatibility must demonstrate
that it is mathematically equivalent to consuming the already trimmed
mission-local polyline and must not expose it as mission progress semantics.

## 6. Runtime and legal-semantics separation

Runtime consumes the frozen mission-local polyline directly for route `s`,
route tangents, R4, completion, route samples, final-gate direction, and all
route-relative mission consumers. The runtime does not orient occurrences,
trim routes, read future SDC poses, or construct an alternative route.

The amendment does not mutate `RouteLaneRecord`, lane polygons, source
topology, traffic-control geometry, source lane direction used by Rulebook
legality, wrong-way semantics, or wrong-carriageway semantics. If mission-local
orientation disagrees with source-declared legal lane direction, the adapter
reports an explicit scientific conflict; it does not mutate the Rulebook
interpretation.

## 7. Identity, provenance, and migration

All 3,500 UIDs, source paths, split membership, and PG/Waymo identities remain
unchanged. Only dependent mission artifacts and the normalized mission schema
may change after approval.

Each occurrence records orientation provenance and source geometry hash. The
mission hash covers the ordered occurrence identities, orientation values,
oriented 3D geometry, trimmed mission-local canonical route, reset and goal
anchors, final gate, and builder identity. The SDC trajectory itself is not
part of the serialized record.

## 8. Required audit and acceptance criteria

The offline audit must report, separately for PG and Waymo:

- `FORWARD` and `REVERSED` occurrence counts;
- missions containing at least one reversed occurrence;
- routes that become connected and have positive mission length;
- ambiguous orientation or association cases;
- conflicts between mission progress and source-declared legal direction;
- candidate lane alternatives for diagnostic purposes only;
- exact reset and goal endpoint containment;
- `s_goal > 0` for every resolved record; and
- identity/hash preservation without source-pickle mutation.

For `waymo:training_20s:5e7bbc00b872c2ba`, the audit must verify the complete
valid-pose station sequence, net source-station displacement, positive and
negative difference counts, temporal progression, vertical compatibility, and
candidate-lane alternatives. The expected orientation is `REVERSED` and the
expected trimmed mission length is approximately `59.8 - 5.4 = 54.4 m`, subject
to the exact projected values.

The anchor-based final gate remains unchanged except that its route tangent and
anchor are taken from the trimmed mission-local route.

## 9. Approval record

The amendment was explicitly approved on `2026-08-04`. It is authoritative for
the occurrence-oriented route representation, mission-local trimming, and all
affected v1.1 consumers. No other scientific behavior is changed.
