# ADR-061: Bounded control-reference route polyline for traffic-control coordinates

- Status: Approved
- Date: 2026-08-09
- Approval evidence: explicit user approval in this conversation. The user
  reported the symptom (*"il semaforo che sta subito dopo la fine della route
  non venga individuato"*), proposed a perception-style range criterion, and
  approved the resulting design after the separation between the perception
  layer and the arbiter layer was established (*"Ok"* to the traffic-light
  items).
- Affected specification: `docs/specifications/rulebook_v4.13_specification.md`
  §4 (amends `rulebook_v4.7_specification.md` §2.9.5/§2.9.6 as already amended
  by `rulebook_v4.11_specification.md`).
- Extends: `ADR-051`.
- ExecPlan: `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`
  (`DEC-RSEC-005`, `DEV-RSEC-004`).

## Context

Traffic-control selection currently fails in two stages.

**Stage 1 — construction.** `derive_control_line`
(`rulebook/v2/geometry/controls.py:111`) computes `route_s_m` as the minimum
curvilinear coordinate among the intersections between the control line and the
canonical route polyline, buffered by `eps_geom`. If the control line does not
intersect that polyline at a vertically compatible level, the control is
discarded with `ControlLineOffRouteError` and **no record is ever created**.
The polyline is the concatenation of the `assigned_route_lane_ids` only.

**Stage 2 — selection.** `_selected_control` (`transition.py:530`) additionally
requires the control's `approach_lane_id` to lie in the route or in its
unambiguous single successor (`route_reachable_control_lane_ids`,
`transition.py:502`, `ADR-051`).

`ADR-051` fixed stage 2 for a single hop and raised the working-signal rate from
42.9% to 74.8% on the 828 `has_route_traffic_light` Waymo records. The residue
is recorded in the same dry-run: **52 records (6.3%) retain zero `SIGNAL`
control after construction** — stage 1 — and **157 (19%) are still excluded by
the approach filter**, either because the controlled lane lies more than one hop
past the route end or because the terminal lane branches. In total **209/828
(25.2%) of signalised scenarios never have a signal selected at runtime.**

The root cause is the same one `ADR-051` identified: `assigned_route_lane_ids`
for a fixed-window recording reflects the human driver's actually-recorded path,
which legitimately stops short of a junction the driver had not yet crossed. The
RL policy is not bound to that path.

Two candidate criteria were rejected before this one.

*Perception-bounded selection* ("only controls the ego can see") was rejected
because the Rulebook is the ground-truth arbiter of cost. If the cost depended
on what the agent perceives, the optimal policy would include not perceiving
traffic lights. Perception limits belong in the observation layer, where they
already are (`OBS-V1.3`'s signal camera, `ADR-045`), and the separation is what
makes the comparison between observation arms readable.

*Euclidean proximity fallback* was rejected because v4.7 §2.9.5 explicitly
forbids selecting a control by distance alone; it would readmit controls
governing other approaches of the same junction.

## Decision

Introduce a second polyline, the **control reference polyline**, used
**exclusively** as the `route` argument of `derive_control_line`:

> Start from the canonical assigned-route polyline. While the terminal lane has
> exactly one successor and the cumulative extension beyond the route end stays
> below a bounded distance `D`, append that successor's centerline.

Its correctness rests on one invariant, which must be asserted in code and
covered by a test:

> **Prefix identity.** The control reference polyline is point-identical to the
> canonical route polyline on `[0, L]`, where `L` is the canonical route length.

Prefix identity is what keeps `control.route_s_m` and the ego's `front_s` in the
same curvilinear frame. Without it, the coordinate mismatch would reintroduce
exactly the defect `REQ-EF-05` removed, in which a control carried a coordinate
from a different frame and appeared already passed.

Everything else keeps the canonical route: R4 progress, mission completion, the
final gate, wrong-direction reference tangents, and every other route consumer.

`D` is a physical distance rather than a hop count, which is the defensible form
of the user's original range intuition inside the Rulebook's own topological
vocabulary. Its value is set from the measurement recorded in the ExecPlan
(how many of the 157 residual records are recovered at each budget), not chosen
a priori.

## Consequences

- A control whose line falls beyond the canonical route end now receives a
  well-defined `route_s_m` and is constructed instead of being discarded,
  closing the 6.3% lost at stage 1.
- A control at `route_s_m > L` remains **selectable** — `front_s` saturates at
  `L`, so the control is always "ahead" — but can never register a crossing.
  Only the graded approach cost applies. This is the correct behavior for the
  case that motivated the change: when the mission goal is a stop line, the ego
  must be charged for approaching a red light too fast and must not be charged
  for stopping at it (`d_req = 0` at standstill yields cost `0`).
- `route_s_m` is unchanged for every control that is already built today, by
  prefix identity. The change is purely additive in coverage.
- Ambiguity is still never resolved by guessing: a branching terminal lane stops
  the extension, exactly as in `ADR-051`.
- Records whose controlled lane lies beyond `D`, or past a branch, remain
  uncovered. The residual is to be re-measured with the same 828-record dry-run
  and reported, not assumed to be zero.

Regression tests: `TEST-RSEC-008` (prefix identity of the two polylines on
`[0, L]`), `TEST-RSEC-009` (a signal two hops past the route end becomes
selectable; an ambiguous branch stays `NOT_APPLICABLE`). Acceptance is
`AC-RSEC-009`: the zero-selectable-signal fraction on the frozen catalog must
fall strictly below the 25.2% baseline while no already-built control changes
its `route_s_m`.
