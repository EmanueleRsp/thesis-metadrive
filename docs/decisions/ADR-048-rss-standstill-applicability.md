# ADR-048: Longitudinal RSS is not applicable at standstill

- Status: Approved
- Date: 2026-08-01
- Approval evidence: explicit user approval of
  `docs/implementation/rulebook_v2_cost_activation_corrections_v1_exec_plan.md`
  (`DEC-RBCOST-003`) in this conversation, following
  `docs/audits/rulebook_v2_cost_activation_audit_2026-08-01/findings.md`
  (F3b/F5b).
- Affected specification: `rulebook_v4.7_specification.md` §6.2.1 (amended by
  `rulebook_v4.10_specification.md`, new applicability clause).

## Context

`safe_distance_m` (`components/rss.py`) always adds the response-time terms
`v_e * rho + 0.5 * a_max_acc * rho²` and `(v_e + rho * a_max_acc)² / (2 b_e)`,
so `d_safe(0, 0) > 0` even when both the ego and the front vehicle are fully
stopped (`3.28 m` at the previous `b_e = 4.0`, `2.52 m` at the new `b_e = 8.0`,
see `ADR-047`). An ego stopped 2 m behind a stopped leader — an ordinary
queue — therefore scored a non-zero `q_rss` for as long as it waited, with no
available action that reduced it: reversing triggers `wrong_way`, advancing
into the leader triggers a collision.

## Decision

`evaluate_rss` drops a candidate from evaluation when both
`ego_speed_mps <= 0.1` and `front_speed_mps <= 0.1`. If every candidate in a
step is dropped this way, the result is `NOT_APPLICABLE` (matching the
existing empty-candidate-tuple path) with a `standstill_dropped` diagnostic
recording how many were excluded; if some candidates remain (e.g. one stopped
leader and one moving actor), they are evaluated normally and only the
standstill one is excluded.

`0.1 m/s` is above the residual velocity a stopped rigid body retains in the
physics solver and an order of magnitude below any speed at which a following
manoeuvre is under way. An alternative considered — gating on closing dynamics
(`v_e > v_i`) instead of an absolute threshold — was rejected because it would
also silence genuine same-speed tailgating, the core case the rule exists for.

## Consequences

A vehicle correctly executing the appropriate response to a queue (remaining
stopped) no longer accumulates R2 cost. RSS applicability, and therefore R2's
worst-component selection, becomes state-dependent on the standstill
threshold in a way it was not before; this is additive to the aggregation
contract (`aggregate_max_component` already treats a non-applicable
sub-component as contributing nothing) and requires no change there. A
following manoeuvre that never reaches standstill, including a slow
same-speed tailgate, is unaffected.
