# ADR-054: Route-Coordinate Driving Mission v1.1

## Status

`APPROVED` — explicit user approval recorded on 2026-08-04.

## Context

The candidate protocol mixed route progress with intermediate gates, bounded projection, recovery/freeze behavior, and source-specific final-lane metadata. The required mission is instead a frozen assigned route with exact stateful longitudinal coordinate, pure R4, shared snapshots, and one terminal gate. The frozen population contains 3,500 records (1,695 PG and 1,805 Waymo); its selection index and source files are aligned and available.

## Decision proposed

Adopt the route-coordinate contract in `DRIVING-MISSION-V1.1`: concatenate immutable oriented 3D occurrences; initialize from the first occurrence or its shared boundary only; use a sequential exact cursor without clamp/freeze/recovery/probabilistic matching; compute R4 only from `delta_s`; separate instantaneous and maximum completion; share one snapshot; and preserve the ten fixed local route samples.

Construct the final gate offline and source-independently from the local geometry. At `s_goal`, let `g=r(s_goal)` and parameterize the normal cross-section as `x(u)=g+u*n`, with anchor `u=0`. Intersect it with all static lane polygons, retain only vertically compatible intersections whose static tangent has positive dot product with the route tangent, convert them to intervals, merge overlaps and gaps `<= 0.01 m`, and select the unique merged component whose closed interval covers zero within the same tolerance. The final-occurrence polygon is only a consistency check. Zero or multiple anchor-covering components are builder errors; no width, centroid, lane ID, order, or metadata tie-break is allowed. Freeze exactly one `final_gate_segment` with endpoints, tangent, elevation, final occurrence identity, provenance/evidence, source geometry hash, and builder identity. Runtime only performs directed swept-front-bumper crossing. Neighbor/boundary/road metadata and a lateral envelope are diagnostic evidence only.

## Evidence and consequences

The read-only anchor-based geometric prototype audited all 3,500 records: 3,500 gates built uniquely (PG 1,695; Waymo 1,805), zero components failed to cover the anchor, and zero records had multiple anchor-covering components after tolerance merging. The four former ambiguities all became unique under the anchor criterion. The former strict metadata result of 3,290 determined and 210 unresolved is superseded as a gate-validity criterion.

Candidate-level rejection totals were: 139,146 with no cross-section intersection, 152,379 opposite/perpendicular, 4,257 vertically incompatible, and 1 null intersection. These are diagnostics, not record counts. Representative successful classes include multi-lane contiguous components, singleton terminal lanes, and components separated by non-guidable gaps. The full audit is in `findings.md`.

Migration preserves UID, source path, split, and PG/Waymo identity. No source artifact, record, or project index is changed by this ADR. No runtime lane classification, neighbor search, proximity fallback, or component expansion is allowed.

## Alternatives rejected

- intermediate gates/checkpoints and trajectory imitation: inconsistent with route-terminal semantics;
- global nearest/HMM and bounded projection: permit nonlocal jumps or unapproved state;
- R4 coupling to heading, legality, or off-route state: violates R1--R3 ownership;
- requiring source-declared neighbor/boundary semantics for every compatible lane: unnecessary for the local geometric terminal-surface definition;
- silently selecting the final occurrence when components are ambiguous: invents a deterministic answer not established by the builder.

## Approval record

Approved 2026-08-04 by explicit user approval in the task conversation. The
approval includes the anchor-based final gate, `0.01 m` offline merge/coverage
tolerance, 3,500/3,500 static audit evidence, and authorization to implement
according to the ExecPlan. No production implementation detail may change the
approved scientific contract without a new approval and ADR amendment.
