# ADR-003: Causal CTRV Conflict-Zone Occupancy Prediction

- Status: APPROVED
- Date: 2026-07-17
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-17
- Supersedes: NONE
- Affected specification: docs/specifications/rulebook_v4.7_specification.md,
  version 4.7-final-implementation-complete, authoritative
- Base specification: docs/specifications/rulebook_v4.6_specification.md,
  version 4.6-final-implementation-complete
- Affected ExecPlan: docs/implementation/rulebook_v2_implementation_plan.md

## Context

Version 4.6 uses exact constant-velocity, constant-heading, non-rotating
continuous SAT for TTC and conflict-zone occupancy. This is exact for its model
but cannot represent a vehicle already turning through a curved merge, turn, or
roundabout. The v4.7 review candidate proposes a deliberately limited causal
CTRV model only for vehicle conflict-zone occupancy.

## Proposed Decision

1. Preserve v4.6 exact CV continuous SAT for TTC without behavioural change.
2. For ego and other live vehicles in crosswalk/vehicle-yield conflict zones,
   estimate yaw rate causally by OLS over bounded history and propagate the full
   footprint with CTRV.
3. Reuse exact CV continuous SAT for insufficient-history, stationary, and
   straight-limit vehicle cases; preserve v4.6 CV/non-rotating prediction for
   pedestrians and cyclists.
4. Keep the update transactional: one central history writer builds a preview,
   and history commits only with all successful memory/cache deltas.
5. Prohibit future trajectories, validity masks, route/maneuver labels, and
   offline intent data from every online prediction path.

## Decision status

The user approved the CTRV model, its engineering defaults, the history
lifecycle rules, strict timestamp monotonicity, and the separation of v4.6
and v4.7 experimental conditions on 2026-07-17. The defaults are frozen
a-priori and must not be calibrated or selected using training performance,
dataset-wide search, sensitivity analysis, ablation, or training runs.

The complete v4.7 specification and this ADR were explicitly approved on
2026-07-17. Production implementation remains subject to the approved ExecPlan
and its conformance gates.

## Frozen Defaults

| ID | Frozen value | Evidence |
|---|---:|---|
| DEC-CTRV-001 | history window 0.5 s | user approval 2026-07-17 |
| DEC-CTRV-002 | minimum OLS samples 3 | user approval 2026-07-17 |
| DEC-CTRV-003 | stationary 0.1 m/s; straight 1e-3 rad/s | user approval 2026-07-17 |
| DEC-CTRV-004 | sweep maximum step 0.02 s | user approval 2026-07-17; conformance oracle required |
| DEC-CTRV-005 | exact CV straight-case reuse | user approval 2026-07-17 |

Additional frozen behavior: history resets after any actor absence; timestamps
must be strictly increasing sim_time_s; v4.6 and v4.7 outputs are separate
experimental conditions.

## Consequences

Vehicle conflict-zone timing may change in curved motion, so v4.6 and v4.7
results are not reward-equivalent and must not be pooled. In-episode v4.6
monitor-memory serialization is incompatible; runs restart at reset. Learner
interfaces and policy observations remain unchanged, but a scientific run must
not continue across rulebook versions.

The numerical rotating solver is deterministic but resolution dependent. It is
not an intent predictor, map-conditioned predictor, safety shield, or guarantee
of safe learning. Required validation is limited to implementation conformance:
unit/boundary tests, lifecycle/timestamp tests, future-data guards, one
synthetic fine-oracle comparison, a fixed PG/Waymo smoke suite, deterministic
replay, finite bounded outputs, and exact straight-branch v4.6 equality.
Dataset-wide calibration, policy-performance validation, path-conditioned
prediction comparison, CTRV-vs-CV ablation, sensitivity analysis, and training
runs for parameter selection are excluded.

## Required Evidence For Conformance

- verified live PG/Waymo actor identity lifecycle, timestamp monotonicity,
  heading/velocity/footprint conventions, and causality guards;
- protected CTRV acceptance matrix, including straight equivalence, analytic
  turns, wrap, lifecycle/transactionality, no-future access, and one synthetic
  fine-oracle sweep validation;
- a small fixed PG/Waymo integration-smoke suite covering straight and curved
  conflicts;
- updated ExecPlan mapping requirements to code and conformance tests.

## Approval Record

- Approved by: user
- Approval evidence: explicit user approval in the Codex conversation on
  2026-07-17.
- Notes: approval covers the CTRV model, frozen engineering defaults, causal
  history lifecycle, conformance-only validation scope, and v4.6/v4.7
  experimental separation.
