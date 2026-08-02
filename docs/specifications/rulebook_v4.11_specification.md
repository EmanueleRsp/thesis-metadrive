# Specification: Traffic-control route-successor extension

## Metadata

- Feature: `rulebook_v2_traffic_control_route_successor_extension`
- Specification ID: `rulebook-v2-control-route-successor`
- Version: `4.11`
- Status: `APPROVED`
- Date: `2026-08-02`
- Supersedes: `docs/specifications/rulebook_v4.7_specification.md`, version
  `4.7-final-implementation-complete` (only for the operational predicate of
  §2.9.5's "movimento ego pertinente" filter; the rest of v4.7 remains
  authoritative and unchanged)
- Related specifications: `docs/specifications/rulebook_v4.10_specification.md`
  (independent R2/R3 cost-activation corrections; no interaction)
- Related ADR: `docs/decisions/ADR-051-traffic-control-route-successor-extension.md`
- Related ExecPlan:
  `docs/implementation/rulebook_v2_cost_activation_corrections_v1_exec_plan.md`
- Authoritative: YES

## 1. Purpose And Context

v4.7 §2.9.5 requires a traffic-control candidate's `MovementKey` to coincide
with "il movimento ego pertinente" without pinning down the exact predicate;
the implementation (`transition.py:_selected_control`) reads it as
`approach_lane_id in assigned_route_lane_ids`. An empirical measurement (see
`ADR-051`) over the frozen catalog's 828 `has_route_traffic_light=true`
Waymo scenarios found this predicate excludes 421 (50.8%) of constructed
`SIGNAL` controls, and that 255 of those (60.6%) are excluded purely because
`assigned_route_lane_ids` -- built from a 20-second recorded human path -- ends
one lane short of a junction the recorded driver had not yet crossed, while
the RL policy driving the ego during training is not bound to that recorded
path and can reach the junction within the same episode.

## 2. Amendment To §2.9.5 (Traffic-Control Candidate Selection)

The "movimento ego pertinente" predicate is clarified to be:

```
lane_id in assigned_route_lane_ids
or (
    lane_id in successors(assigned_route_lane_ids[-1])
    and len(successors(assigned_route_lane_ids[-1])) == 1
)
```

where `successors(lane_id)` is the set of topologically connected successor
lanes recorded on the route-lane catalog for the episode. This mirrors the
disambiguation already normative for `MovementKey.exit_lane_id` derivation
(v4.7, `derive_lane_movement_key`): an unambiguous single continuation of the
route is treated as part of the ego's path; an ambiguous branch (more than
one successor) is not guessed and is not included.

This is a single-hop extension. A control whose lane is two or more
successors beyond the route's terminal lane is not covered by this version.

`route_s`, the control-line derivation (§2.9.6), the selection tie-break, and
every cost formula that consumes a selected control are unchanged -- this
amendment affects only which controls are *candidates*.

## 3. Compatibility

- No change to `TrafficControlRecord`, `RuleComponentResult`, or the
  observation vector.
- Episode returns on scenarios where a previously-unselected `SIGNAL`/`STOP`
  control now becomes selectable are not comparable with runs completed
  before this version.
- The `control_line_diagnostics` counters introduced in the same
  session's post-approval follow-up
  (`rulebook_v2_cost_activation_corrections_v1_exec_plan.md`) reflect
  whichever route-membership predicate is active; their numeric baseline
  (57.1% zero-signal rate) is superseded by this version and should be
  re-measured, not assumed frozen.
