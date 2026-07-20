# ADR-015: R1 Pre-State Canonical-Footprint Centerline Normal

- Status: `Approved`
- Date: `2026-07-20`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-20`
- Amends: `docs/specifications/rulebook_v4.7_specification.md` §§5.3, 5.7, 5.9, 11.2, 14, and 15.1
- Affected specification: Rulebook v4.7
- Affected ExecPlan: `docs/implementation/rulebook_r1_collision_floor_bugfix_exec_plan.md`

## Context

Native Bullet manifold normals are not a stable 2.5D R1 input. The audited run
contained four vehicle crashes with zero raw normal closing speed, and a
controlled live contact exposed a normal with no horizontal component. Contact
point ordering also changes under penetration. These source-specific details
made the `1e-6` numerical floor dominate otherwise non-trivial crashes.

## Decision

For every R1 collision onset, derive the unit normal from the pre-state
canonical-footprint centers, oriented ego to the other actor. The callback and
current contact query provide only stable actor identity and onset/active state.
The pre-state velocity, actor-class speed caps, squared closing-speed formula,
floor, and onset-only behavior remain unchanged.

If the two pre-state footprint centers coincide within `1e-6 m`, R1 fails fast
with `RulebookEvaluationError`; it does not manufacture an arbitrary axis.

## Consequences

Front, rear, and side collisions use one deterministic causal construction
independent of Bullet manifold normals or manifold availability. Glancing
collisions are evaluated against the centerline axis rather than a local
physical contact-surface normal. Existing external producers of
`ContactOnsetRecord` need only actor ID and actor class; manifold point and
normal fields are no longer part of that contract.

## Approval Record

- Approved by: user
- Approval evidence: user message “va bene” on 2026-07-20, after the proposed
  always-on pre-state canonical-footprint normal was explained.
