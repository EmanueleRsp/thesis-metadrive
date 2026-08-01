# ADR-043: Evaluation Video Ego-Trail And Planned-Checkpoint Overlay

- Status: `APPROVED`
- Date: 2026-07-31
- Decision owners: thesis repository maintainer
- Approval evidence: explicit user request "come preferisci ... Procedi pure
  poi ll'implementazione" on 2026-07-31, following presentation of `DEC-001`
  (amend ADR-020 vs. drop the ego-trail requirement) in
  `docs/implementation/evaluation_video_route_and_ego_trail_overlay_v1_exec_plan.md`
- Related plan: `docs/implementation/evaluation_video_route_and_ego_trail_overlay_v1_exec_plan.md`
- Amends: `decisions/ADR-020-evaluation-video-diagnostics.md`

## Context

ADR-020 established the shared evaluation-GIF diagnostic annotator and
explicitly excluded "ego-history lines" from the overlay. The user requested,
in a later session, two additions for qualitative episode inspection: (1)
discrete markers for the ego's planned navigation checkpoints, and (2) the
ego's actually-traveled position history up to the current frame. The second
item conflicts by name with ADR-020's exclusion, making it a specification
deviation rather than an implementation detail.

Investigation found that the repository's route geometry is not MetaDrive's
live `navigation.checkpoints` (unused anywhere in `src/thesis_rl/`), but a
frozen `assigned_route_lane_ids` lane sequence resolved against precomputed
canonical lane geometry (`RoutePolyline.from_lane_centerlines`,
`build_assigned_route_polyline`); the existing `route_past`/`route_future`
overlay is the planned-route polyline itself, sliced at the ego's current
projection, not the ego's real trajectory.

## Decision

Amend ADR-020: permit an ego-trail layer distinct in style from the existing
`route_past` (planned route already covered, green) so the two remain
visually separable, since a route-projection slice and the ego's real path
are not always identical (lateral deviation, lane changes). Both additions
are raster-only, degrade silently when geometry is unavailable, and do not
change policy observations, rewards, termination/truncation, or metrics.

Added to the overlay:

- discrete planned-checkpoint markers (small hollow diamonds, amber), one per
  lane in the frozen assigned-route lane sequence
  (`RoutePolyline.lane_start_points_xyz`), distinct from the single
  next-target marker;
- the ego's actually-traveled position history for the current episode (thin,
  low-alpha violet polyline), toggleable via
  `conf/video/default.yaml`'s `video.topdown.draw_ego_trail` (default `true`).

Every other exclusion in ADR-020 (road boundaries, explicit violation zones,
terminal-status panels, duplicated reward fields, geometric RSS/TTC bands)
remains unchanged.

## Consequences

Positive:

- both layers reuse the existing world-to-screen projection pipeline
  (`diagnostic_geometry`/`_draw_geometry`), no new coordinate system;
- checkpoint markers require no new runtime state (derived once per
  `RoutePolyline` construction);
- the ego trail is naturally episode-scoped because
  `LiveEvalEpisodeRecorder` (parent-process live recording) is instantiated
  once per episode, and the worker-process render path clears its trail
  buffer explicitly on every `reset`/`reset_slots` command.

Negative:

- `RoutePolyline` gains one additive field
  (`lane_start_points_xyz`, default `()`) purely for diagnostic use, read by
  no rulebook geometry or projection logic;
- the worker process (`deterministic_subproc_vec_env.py`) now carries one
  more piece of long-lived per-slot state that must be correctly cleared at
  every episode boundary — a class of bug already present for other
  per-slot state in this file, mitigated by a focused regression test.

## Alternatives rejected

- Keeping ADR-020's exclusion and dropping the ego-trail request (Plan
  `DEC-001` option B) was rejected: the user had full context of what the
  overlay already shows and asked for the trail specifically, with no
  indication ADR-020's original, general-purpose exclusion should still
  apply to this concrete use case.
- Reusing MetaDrive's native `TopDownRenderer.draw_target_vehicle_trajectory`
  was rejected in favor of extending the existing custom annotator, to avoid
  a second, inconsistent overlay/coordinate pipeline (per ADR-020's original
  "one visual contract for live and replay GIFs" rationale).
