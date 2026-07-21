# ADR-021: Source-Bounded Reactive Scenario Traffic

- Status: `APPROVED`
- Date: 2026-07-21
- Decision owners: thesis repository maintainer
- Approval evidence: explicit user approval in the current task on 2026-07-21
- Related plan: `docs/implementation/source_bounded_reactive_traffic_v1_exec_plan.md`

## Context

ScenarioNet can assign a recorded vehicle to MetaDrive's reactive
`TrajectoryIDMPolicy`. The upstream traffic manager removes replay-policy
vehicles when their source track becomes invalid, but lets reactive vehicles
continue until they reach their generated IDM destination. A live failure showed
that this continuation can place an actor outside its recorded Waymo support and
produce geometry inconsistent with the assigned route.

## Decision

Keep `reactive_traffic=true`. Replace the traffic manager only in
`ThesisScenarioEnv` with a thesis-owned subclass that removes a reactive vehicle
at the first simulation step for which its source track's `state.valid` flag is
false (including an out-of-range step).

No vehicle is replaced, respawned, or given synthesized features after removal.
Replay-policy behaviour, source selection, observations, rewards, Rulebook
logic, termination, truncation, datasets, and configuration defaults remain
unchanged.

## Consequences

Positive:

- reactive interactions remain available during each actor's recorded support;
- unsupported IDM extrapolation cannot create a stale traffic actor;
- removed actors are absent from later observations and Rulebook inputs.

Negative:

- reactive traffic density can decrease before the scenario horizon;
- the post-scenario tail can contain fewer or no traffic actors once their
  source tracks end.

## Alternatives rejected

- Keeping the upstream IDM lifecycle was rejected because it permits
  out-of-support traffic dynamics.
- Switching all traffic to replay was rejected because it removes the approved
  reactive-traffic configuration.
- Synthesizing or clamping features after source expiry was rejected because it
  would introduce unapproved traffic data outside the source record.
