# ADR-020: Compact Evaluation Video Diagnostics

- Status: `APPROVED; amended by ADR-043`
- Date: 2026-07-21
- Decision owners: thesis repository maintainer
- Approval evidence: explicit user approval in the current task on 2026-07-21
- Related plan: `docs/implementation/evaluation_video_diagnostics_v1_exec_plan.md`

## Context

Evaluation GIFs are useful for qualitative analysis but currently provide little
information about the scalar reward, Rulebook subrules, route progress, and
runtime state that produced each frame. The repository has both live official
recording and non-authoritative offline replay paths, so duplicated formatting
would risk visual inconsistency.

## Decision

Add a shared diagnostic frame annotator used by both live final-evaluation
recording and offline replay. The annotation is a raster-only diagnostic layer
and must not alter policy observations, rewards, termination, truncation, or
evaluation metrics.

The panel uses black text on a semi-transparent white background and contains:

- algorithm name;
- current step;
- ego speed and heading;
- selected reward for the current transition;
- cumulative selected reward;
- route completion;
- compact R1--R4 Rulebook values and available subrule values.

The scenario overlay contains, when current runtime geometry is available:

- traversed route in green;
- target/waypoint marker;
- optional route remainder only when directly available at the current frame;
- thin orange neighbor outlines;
- thin red outline for the RSS/TTC-relevant actor when its identity resolves.
- **(amended by ADR-043, 2026-07-31)** discrete planned-checkpoint markers
  (amber hollow diamonds), one per lane in the frozen assigned-route lane
  sequence;
- **(amended by ADR-043, 2026-07-31)** the ego's actually-traveled position
  history for the current episode (thin, low-alpha violet polyline),
  toggleable via `video.topdown.draw_ego_trail` (default on).

Road boundaries, explicit violation zones, terminal-status panels, and
duplicated native/scalarized reward fields are excluded. Ego-history lines
were excluded here originally but are permitted, in a style distinct from the
green traversed-route line, per ADR-043. RSS/TTC
numeric diagnostics remain in the panel; geometric RSS safety bands and TTC
cones are deferred until exact world-to-frame geometry is available without an
approximation.

## Consequences

Positive:

- one visual contract for live and replay GIFs;
- richer diagnosis without scientific pipeline changes;
- compact display remains readable and avoids duplicating what the top-down
  scene already shows;
- missing optional geometry degrades to the base frame rather than failing an
  evaluation.

Negative:

- frame rendering and annotation add CPU work;
- environment-specific geometry adapters may support some backends more fully
  than others;
- deferred RSS/TTC geometry may be added later.

## Alternatives rejected

- A verbose panel listing every status string was rejected as visually
  excessive; status is encoded through compact formatting/color where useful.
- Separate live and replay formatters were rejected to avoid inconsistent GIFs.
- Approximate RSS/TTC polygons and cones were rejected because misleading
  geometry is worse than omitting it.
- A separate debug view showing all privileged actors was rejected because the
  native top-down animation already shows the scene and the diagnostic layer
  should not imply policy visibility.

