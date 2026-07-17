# ADR-004: Assigned Route Metadata For PG And Waymo

- Status: `Approved`
- Date: `2026-07-17`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-17`
- Supersedes: route-source clauses of ADR-002
- Affected specifications: `docs/specifications/observation_v1.1_specification.md`,
  ID `OBS-V1.1`
- Affected ExecPlans: `docs/implementation/semantic_observation_encoder_v1_exec_plan.md`

## Context

The route is a task/navigation input known to the ego before control begins.
Waymo scenarios do not expose an equivalent route independently of their
recorded SDC motion. Current PG exports also do not persist the generator's
assigned route. Requiring a map-only destination inferred without a task
annotation would create a different task for the two sources.

## Decision

For every scenario, publish one immutable `assigned_route_lane_ids` task
annotation before reset and construct the runtime `RoutePolyline` solely from
those lane IDs and canonical map centerlines.

- Existing PG annotations use the same offline SDC map-matching pipeline as
  Waymo; a future persisted generator route may replace that source only
  through a separately approved data change.
- Waymo annotations may be map-matched from the complete SDC trajectory only
  during offline dataset preparation, then are frozen as task metadata.
- Runtime code must never index, replay, compare against, or otherwise read
  future SDC samples. It may consume only the frozen assigned route.
- Route provenance is logging/manifest metadata and is not a policy feature.
- A missing, invalid, or non-contiguous assigned route fails closed before
  control; no native navigation or dynamic future-data fallback is allowed.

## Consequences

The policy receives a known navigation mission in both sources, including its
planned turns, while dynamic actor state, traffic controls, and all non-route
features remain causal. Existing data and runtime adapters must persist and
validate the assigned-route annotation. Historical runs without it are not
implicitly migrated. Current PG and Waymo annotations therefore have identical
route semantics and differ only in source-format adapters.

## Approval Record

- Approved by: user
- Approval evidence: explicit user message in this Codex conversation on
  `2026-07-17`: “dai va bene allora”.
