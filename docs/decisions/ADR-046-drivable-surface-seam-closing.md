# ADR-046: Morphological closing of the drivable-surface union

- Status: Approved
- Date: 2026-08-01
- Approval evidence: explicit user approval of
  `docs/implementation/rulebook_v2_cost_activation_corrections_v1_exec_plan.md`
  (`DEC-RBCOST-001`) in this conversation, following
  `docs/audits/rulebook_v2_cost_activation_audit_2026-08-01/findings.md` (F5a).
- Affected specification: `rulebook_v4.7_specification.md` §7.2.1 (amended by
  `rulebook_v4.10_specification.md`).

## Context

`offroad` (`components/road.py:evaluate_offroad`) fires on the sub-cm² slivers
left behind when independently built adjacent lane polygons are unioned:
measured on a live PG map (`map='SCS'`, 36 lanes), the union carries 681
interior holes of `1e-4..5e-4 m²`, and a footprint sweep at a lateral offset of
1.2 m from the lane centre — a completely normal in-lane position — reports a
non-zero off-road cost on 37.1% of steps. Combined with the scalarizer's
binarisation (out of scope of this decision, tracked separately), this can make
a correctly positioned, stationary vehicle receive a full `-2.01` R3 penalty
indefinitely.

## Decision

`drivable_surface_for_ego` (`geometry/drivable.py`) applies a morphological
closing, `union.buffer(+0.10).buffer(-0.10)`, to the lane-polygon union before
returning it. Measured: `0.05 m` left seam activations at some lateral offsets,
`0.10 m` removed every seam-driven activation while leaving the genuine
map-edge gap (footprint overhanging the first/last lane of the map, ~2.8%
baseline) untouched. `OFFROAD_AREA_EPSILON_M2` (the absolute area tolerance)
stays at `1e-4 m²`, unchanged.

The closing operator is monotone non-decreasing in area, so it can only add
surface — it can only lower an `offroad` cost, never raise one.

An earlier draft of this decision also proposed replacing the absolute
tolerance with a `1%`-relative one. It was withdrawn: its only justification
was robustness against the scalarizer's binarisation (out of scope, see
`AGENTS.md` "one problem at a time"), and on an `8.3 m²` footprint it would
silently exempt `0.08 m²` of genuine off-road.

## Consequences

A vehicle correctly positioned on a lane no longer receives a spurious
off-road cost from inter-lane polygon seams, on any source (PG measured; Waymo
lane polygons are built by the same independent-per-lane method and are
expected, not yet measured, to exhibit the same class of defect). A genuine
gap between two lane polygons narrower than 20 cm is now bridged and no longer
reported as off-road; no such gap is known to exist in the current map
sources. `_union_selected_surfaces`'s cache semantics and hashing behaviour are
unchanged.
