# ADR-053: Geometric Gate-Only Mission Boundaries

## Status

`APPROVED` — explicit user approval on 2026-08-03.

## Context

The first Unified Driving Mission runtime treated loss of a legal lane-graph
path as `mission_unreachable` termination. In live training this made lateral
deviation, including travel on a parallel or opposing carriageway, terminate
episodes before the Rulebook costs could assess the behavior. This conflicts
with the intended separation: gates define ordered task advancement, while
Rulebook costs assess legality and risk.

## Decision

Driving mission v1.0 is amended to use ordered, directed geometric gates as
the sole task-progress and task-success boundary. A runtime gate spans the
source-declared lateral carriageway component at its local cross-section, with
a 0.5 m road-envelope margin. Mere geometric proximity never expands a gate
to a crossing road or nearby service road. Crossing requires forward
swept-front-bumper intersection and vertical compatibility, but not
association to a preferred, allowed, or same-direction lane.

Runtime mission tracking shall not terminate for loss of lane association,
loss of a legal graph path, a lane change, travel on an opposing carriageway,
or partial off-road occupancy. The only episode terminations remain approved
collision/physical-out-of-road conditions and ordered final-gate success; the
time limit remains truncation. Rulebook costs retain responsibility for
wrong-way, wrong-carriageway, off-road fraction, collision, RSS, and other
unsafe or illegal behavior.

Remaining distance is the directed geometric distance from the ego front
bumper to the pending gate plus fixed Euclidean distances between subsequent
ordered gate anchors. It is finite for every valid materialized mission and is
independent of current lane association.

## Consequences

- `mission_unreachable` is retained only as a backward-compatible field in the
  snapshot schema and is always false for the amended runtime.
- Existing legal-route graph data remains useful for static diagnostics and
  Rulebook relevance migration, but is no longer a task termination authority.
- Gate crossings on an opposing carriageway can advance the ordered task only
  when the ego physically crosses the gate in the mission-forward direction;
  Rulebook costs record the illegality.

## Validation

Required regression coverage includes a non-lane-associated gate crossing,
an initially disconnected static lane graph that remains an active mission,
no `mission_unreachable` termination in environment boundaries, and a gate
envelope fixture that excludes a perpendicular crossing road.
