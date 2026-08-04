# ADR-055: Mission-local orientation and trimming of frozen route occurrences

- Status: `APPROVED`
- Date: `2026-08-04`
- Applies to: proposed `DRIVING-MISSION-V1.1.1` amendment
- Approval: explicit user approval recorded `2026-08-04`

## Context

`DRIVING-MISSION-V1.1` freezes the ordered `assigned_route_lane_ids`, but its
normalized record currently retains source lane IDs and source centerline order
without preserving the traversal orientation of each occurrence. The audit
found a Waymo record whose causal reset projects at approximately `59.823 m`
and whose terminal projects at approximately `5.377 m` on the source-oriented
centerline. The temporal station sequence is strictly decreasing, so the
source geometry order is opposite to the mission traversal.

The existing runtime `route_offset_m` approach does not represent a canonical
mission-local route and cannot by itself distinguish occurrence orientation or
trim the first and final occurrences.

## Decision

Amend the mission contract so that source-specific offline adapters:

1. retain exactly the frozen ordered lane-ID route;
2. associate the complete SDC trajectory with each route occurrence only
   offline;
3. determine `FORWARD` or `REVERSED` from verified temporal source-station
   progression, without making ego heading authoritative;
4. preserve each occurrence's orientation, trimmed source stations,
   provenance, and geometry hash;
5. validate oriented occurrence connectivity in the frozen order; and
6. construct and serialize one trimmed mission-local canonical route beginning
   at reset and ending at the frozen goal anchor, with `s_start=0` and positive
   `s_goal`.

Runtime consumes that polyline directly and never reads the SDC trajectory,
reorients lanes, plans a route, or applies a fallback. Source lane records and
Rulebook legal-direction semantics remain unchanged.

## Alternatives rejected

- Treating the negative goal station as an isolated data-abort.
- Replacing the assigned route with shortest-path or native-navigation output.
- Reversing all source lanes globally.
- Using ego heading as the route-orientation authority.
- Taking an absolute station difference without proving occurrence orientation.
- Keeping the full source route and applying a runtime source-station offset.

## Evidence

The initial read-only audit used a global polygon scan and therefore allowed
late poses at overlapping junction polygons to contaminate earlier
occurrences. The corrected sequence-constrained audit was run over all 3,500
frozen records and built all 3,500 records: 1,695 PG and 1,805 Waymo. It
classified 12,405 occurrences as `FORWARD` and one as `REVERSED`; no route was
ambiguous or globally invalid, and no record was excluded or replaced.

For `waymo:training_20s:5e7bbc00b872c2ba`, the complete valid-pose audit still
finds 199 valid poses, 187 negative station deltas, 0 positive deltas, 11 zero
deltas, no alternative lane candidate, and a reversed trimmed length of
`54.368483 m`.

For `waymo:training_20s:6042e6c648fca15d`, the former apparent conflict was an
audit defect: the lane-184 polygon overlaps the later junction geometry. The
correct sequential association assigns poses to the frozen occurrences in
order with source stations approximately `37.846 -> 41.196` on lane 184,
`0.077 -> 26.788` on lane 180, and `0.267 -> 74.179` on lane 182. The unique
global orientation is `FORWARD, FORWARD, FORWARD`, and the mission-local route
has positive length `104.572425 m`.

## Consequences

The normalized mission schema, mission hash, offline builder, artifact
materialization, runtime route construction, tracker initialization, route
samples, final-gate tangent, and tests use the sequence-constrained offline
association. Runtime behavior remains unchanged: it consumes only the frozen
mission-local route and never performs this association.

## Approval record

Approved by explicit user approval on `2026-08-04`. This ADR authorizes the
occurrence-oriented schema, global two-state orientation resolution,
mission-local route trimming, implementation, dependent artifact regeneration,
and project-index registration.
