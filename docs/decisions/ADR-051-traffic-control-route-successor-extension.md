# ADR-051: Extend traffic-control route-membership to the unambiguous route successor

- Status: Approved
- Date: 2026-08-02
- Approval evidence: explicit user approval in this conversation ("procedi ad
  annotarlo e poi procedi all'implementazione"), following an empirical
  root-cause investigation launched at the user's request ("credo che il
  prossimo step sia indagare e scoprirne la causa") after the F4b control-drop
  diagnostics (`ADR`-less post-approval follow-up to
  `rulebook_v2_cost_activation_corrections_v1_exec_plan.md`) measured that
  57.1% (473/828) of `has_route_traffic_light=true` frozen-catalog Waymo
  scenarios never have a `SIGNAL` control selected by the Rulebook at
  runtime.
- Affected specification: `rulebook_v4.7_specification.md` §2.9.5's
  operational predicate for "movimento ego pertinente" (added by
  `rulebook_v4.11_specification.md`).

## Context

`_selected_control` (`rulebook/v2/transition.py`) filters traffic-control
candidates to `control.movement_key.approach_lane_id in
cache.task_route.lane_ids`. `approach_lane_id` is the raw lane the Waymo/PG
adapter attaches the physical control to (`dynamic_map_states[...].lane` for
signals, `feature.lane` for stop signs) -- the lane whose entry the control
governs.

A read-only dry-run over all 828 `has_route_traffic_light=true` records in
the frozen catalog found 421 (50.8%) retain a constructed `SIGNAL` control
that this filter excludes entirely. Root-cause sampling (12 cases, all
matching; a full-catalog re-check confirmed 255/421 = 60.6% match
completely, 264/421 = 62.7% at least partially) found a single, consistent
pattern: the excluded control's lane is always the topologically unambiguous
single successor of the route's terminal lane, never an unrelated lane.

Interpretation: `assigned_route_lane_ids` for these `training_20s` Waymo
records is built from the recorded human driver's actual 20-second path
(`waymo_sdc_offline_task_annotation`). When the recorded driver had not yet
crossed a junction within that window, the recorded route legitimately ends
one lane short of it -- but the RL policy driving the ego during training is
not bound to that recorded path and can reach and cross that lane within the
same episode. The traffic light governing it is then physically relevant to
the ego but is not selectable, because its lane never appears in the static
`assigned_route_lane_ids` array. This is not a data or geometry defect: the
control-line derivation and `route_s_m` placement are both correct (the line
sits exactly at the boundary between the route's last lane and the light's
lane). The remaining 166/421 (39.4%) are unrelated approaches of the same
physical junction, correctly excluded.

## Decision

Extend the route-membership predicate used for control selection (the
operational reading of v4.7 §2.9.5's "movimento ego pertinente") to also
accept a lane that is the **single, topologically unambiguous successor** of
the route's terminal lane -- i.e. `lane in route_lane_ids or (lane in
successors(route_lane_ids[-1]) and len(successors(route_lane_ids[-1])) ==
1)`. This mirrors the disambiguation `derive_lane_movement_key`
(`geometry/lanes.py`) already applies when resolving the ego's own
`exit_lane_id`: an unambiguous next lane is treated as part of the ego's
path, an ambiguous branch is not guessed.

Implemented as a new pure helper,
`route_reachable_control_lane_ids(cache) -> frozenset[str]`, computed once
per episode (static, does not change over the episode) and passed to every
`_selected_control` call site in place of the raw `cache.task_route.lane_ids`.

Scope: single hop only in this version. A route ending two or more lanes
short of a light (chained unambiguous successors) is not covered here --
`ADR-051`'s empirical basis is the measured one-hop pattern; extending
further needs its own measurement, tracked as a follow-up (see the ExecPlan's
"further extension" analysis).

## Consequences

More `SIGNAL`/`STOP` controls become selectable at runtime; R3 signal/stop
costs can now activate in scenarios where they previously never applied.
Verified by re-running the same 828-record read-only dry-run after
implementation: the zero-signal-after-filter count dropped from 421 to 157
(264 scenarios recovered, matching the "at least one dropped control matches
the pattern" count measured before implementing). Of the 828 signalised
frozen-catalog Waymo records, controls are now selectable in 619 (74.8%, up
from 355/828 = 42.9% before this change); the remaining 209 (52 lost at
adapter construction for other reasons, 157 genuinely unrelated approaches or
route ends more than one lane short) are unaffected by this single-hop
version. No change to control-line geometry, `route_s_m`, or any cost
formula -- only which controls are candidates for selection. Episode returns
on scenarios where a previously-unselected control now activates are not
comparable with runs completed before this change.
