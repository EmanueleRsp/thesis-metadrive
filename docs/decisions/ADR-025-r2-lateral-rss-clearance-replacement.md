# ADR-025: R2 Lateral-RSS Clearance Replacement

- Status: APPROVED
- Date: 2026-07-24
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-24
- Supersedes: NONE
- Affected specification: `docs/specifications/rulebook_v4.8_specification.md`,
  version 4.8, status `APPROVED`, authoritative (for the §6.4 subset it
  amends; v4.7 remains authoritative for everything else)
- Affected ExecPlan: `docs/implementation/r2_lateral_rss_clearance_v4.8_exec_plan.md`

## Context

Rulebook v4.7 §6.4 defines the geometric clearance cost for `R2` as a pure
polygon-distance threshold, identical in spirit for `VEHICLE` and
`STATIC_COLLIDABLE` actors (with different `D_min` constants). Per Censi et
al. — already cited in v4.7 §4.2 to justify `R2 ≻ R3` — a pure geometric
proximity metric is not equivalent to a collision-safety constraint: it
penalizes stable-but-close configurations (parallel adjacent lanes, an
overtaken parked vehicle, two stationary vehicles) identically to a genuine
lateral-approach conflict, and can in principle reward a policy for
violating a lane boundary just to increase distance from a non-threatening
obstacle.

`docs/specifications/rulebook_v4.8_specification.md` replaces the vehicle
clearance sub-metric with a scoped lateral-RSS metric (Intel `ad-rss-lib`
lateral-distance formulation, adapted to the project's tangent/normal route
frame) and demotes static clearance to diagnostic-only, while leaving VRU
clearance unchanged.

## Decision

1. `q_clear,vehicle` → `q_RSS,lat,scoped` (new scoped lateral-RSS metric,
   specification §7-8).
2. `q_clear,static` → diagnostic-only (removed from cost, distance still
   logged as `static_polygon_distance_m` inside the `clearance` component's
   `diagnostics`, never in `raw`/cost — `DEC-R2-04`).
3. `q_clear,VRU` → unchanged.
4. New R2 aggregation:
   `c_2(t) = max{q_RSS,long, q_RSS,lat, q_TTC, q_clear,VRU}`.
5. Snapshot: `pre_state` for the new lateral metric, matching RSS-longitudinal
   and TTC (`DEC-R2-02`, resolved by the user on 2026-07-24).

## Consequences

`R2`'s cost distribution shifts for near-but-stable lateral configurations
(decreases) and for genuine lateral-approach conflicts not caught today by
RSS-longitudinal or TTC (may increase). Any already-run experiment baseline
calibrated on v4.7's `R2` is not directly comparable after this change (see
specification §11). No policy input, reward-vector shape, or training-data
selection rule changes. Implementation proceeds under
`docs/implementation/r2_lateral_rss_clearance_v4.8_exec_plan.md`.

## Approval Record

- Approved by: user
- Approval evidence: explicit instruction "Approvo la specifica v4.8,
  procedi con l'implementazione" in this conversation, after the
  specification, its scope, and consequences were presented for review.
