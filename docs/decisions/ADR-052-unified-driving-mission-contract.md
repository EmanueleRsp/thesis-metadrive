# ADR-052: Unified Driving Mission Contract

- Status: `Proposed`
- Date: `2026-08-02`
- Decision owner: thesis repository maintainer
- Approval date: `NONE`
- Supersedes: ADR-004 only for the representation of the frozen route; preserves
  its offline-annotation and no-future-runtime constraints. If approved and
  verified, subsumes ADR-051's one-hop route-successor workaround and amends
  ADR-043's diagnostic checkpoint source.
- Affected specifications:
  - `docs/specifications/driving_mission_v1.0_specification_UNDER_REVIEW.md`
  - ScenarioNet Integration v1.1-v1.3 route/success subsets
  - Rulebook v4.7-v4.11 route-dependent subsets
  - OBS-V1.3 route-dependent fields
  - OBS-LIDAR-V2.0 navigation source
- Affected ExecPlans:
  `docs/implementation/driving_mission_v1.0_exec_plan.md`

## Context

The repository currently freezes an ordered `assigned_route_lane_ids` list,
projects R4 and local observations onto a concatenated preferred polyline, uses
an independent MetaDrive reference trajectory for success/completion, and adds
targeted route extensions for individual Rulebook consumers. The same physical
ego state can consequently be treated as on-task, complete, relevant to a
control, or off-corridor by different subsystems.

The code and frozen JSON index establish that the data needed for a richer
annotation can be derived offline: all 3,500 selected records have a non-empty
route and canonical source path, and current static adapters already construct
lane geometry and successors. The unverified issue is complete topology and
goal coverage across source pickle files; it is an approval-gated preflight,
not a reason to retain divergent runtime semantics.

## Proposed Decision

If approved, adopt one versioned `DrivingMissionRecord` and one
environment-owned `MissionTracker` as the sole task-navigation authority.

The record contains ordered route sections, preferred lane/span, legal allowed
lane spans, directed exit gates, a directed final goal, provenance, builder
version, and source-geometry hash. Current PG and Waymo records retain identical
offline SDC-derived semantics. Runtime never reads future SDC state.

The tracker updates once per committed transition and publishes one immutable
snapshot used by:

- R4 and Rulebook route-dependent relevance/geometry;
- semantic and LiDAR local route observations;
- completion, success, termination, metrics, and diagnostic overlays.

Legal alternate lanes are accepted when the next ordered gate remains
reachable. The ordered gates and destination do not change. Remaining distance
is the deterministic shortest legal graph distance through all pending gates.
R4 is:

```text
clip((D_t - D_(t+1)) / (22.2222222222 m/s * delta_t), -1, 1)
```

The reference speed is global, not an ego or scenario configured cap. Reported
completion is the maximum-so-far `clip(1 - D_t / D_0, 0, 1)`, while the signed
per-transition R4 remains able to penalize reversal or route-length increase.

Intermediate and final milestones use forward swept-front-bumper crossings of
directed compatible gates. The final goal has no stop condition. A graph-
unreachable pending gate is proposed as a task-failure termination named
`mission_unreachable`; the time limit remains truncation.

This ADR remains proposed until every open `DEC-MSN-*` item in the candidate
specification is explicitly approved. Approval also authorizes new dataset,
observation, checkpoint, and replay identities; it does not authorize silent
exclusion of a scenario that fails the full-catalog preflight.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Keep the preferred polyline and only change R4 normalization | Small patch | Leaves success, observations, controls, and legal alternate lanes inconsistent | Does not resolve the verified contract split |
| Use MetaDrive native checkpoints and trajectory completion everywhere | Existing simulator behavior | Source trajectory is not the approved frozen task contract and can couple runtime to recorded future/native semantics | Conflicts with ADR-004 and cross-source task identity |
| Use point-radius checkpoints or a weighted sum of Euclidean checkpoint distances | Simple implementation | Ignores direction/lane legality, double counts downstream geometry, discontinuously removes terms, and rewards geometric shortcuts | Not a valid road-network mission coordinate |
| Require exact expert/preferred lane occupancy | Easy route coordinate | Penalizes legal parallel-lane use and confounds navigation preference with traffic-law compliance | Too restrictive for the research objective |
| Replan freely to only the final destination | Robust recovery | Can change intended turns and bypass ordered task content | Alters the benchmark task |
| Exact discounted potential shaping | Formal policy-invariance result when applied correctly | Changes the approved R4 objective and its relation to the lexicographic Rulebook; requires separate experimental approval | Deferred as a distinct reward variant |

## Consequences

Positive consequences:

- one auditable definition of task progress and completion;
- legal alternate-lane behavior without an arbitrary route-adherence penalty;
- removal of actor-cap dependence from R4 normalization;
- route-relevant traffic controls and interaction zones follow actual pending
  mission movements;
- checkpoint markers become true mission gates rather than lane-start labels.

Costs and compatibility consequences:

- a cross-cutting migration across dataset annotations, Rulebook, environment,
  both observations, metrics, video diagnostics, tests, and documentation;
- new frozen dataset selection identity and mission hash;
- new observation schema IDs even if tensor widths remain unchanged;
- no compatibility with existing checkpoints, replay buffers, normalizers,
  golden traces, or experiment aggregation;
- required read-only preflight over all 3,500 trusted source artifacts before
  materialization;
- potential per-step routing cost, mitigated by reset-time graph construction
  and cached downstream distances, to be measured by existing timing metrics.

## Validation And Traceability

Affected candidate requirements: `REQ-MSN-001` through `REQ-MSN-015`.
Affected acceptance criteria: `AC-MSN-001` through `AC-MSN-010`.

Mandatory evidence includes the full-catalog topology/goal audit, deterministic
gate and routing unit tests, actor-cap invariance, gate-distance continuity,
cross-consumer snapshot identity, causality, reset/vectorization isolation,
compatibility rejection, Rulebook regressions including ADR-051, and PG/Waymo
end-to-end smoke tests. Exact tests and commands are frozen in the companion
ExecPlan before production implementation.

## Approval Record

- Approved by: `NONE`
- Approval evidence: `NONE`; proposed for review in a future session
- Notes: approval must explicitly resolve `DEC-MSN-001` through
  `DEC-MSN-009` and does not imply approval of source-record exclusions
