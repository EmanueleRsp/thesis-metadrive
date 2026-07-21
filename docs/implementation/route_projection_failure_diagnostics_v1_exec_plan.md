# Route Projection Failure Diagnostics v1 ExecPlan

## 1. Metadata

- Feature: fail-closed diagnostics for unavailable vertical route projections.
- Plan ID: `ROUTE-PROJECTION-DIAGNOSTICS-V1`.
- Authoritative specifications: `docs/specifications/observation_v1.1_specification.md` §§7.4, 10 and 13.5; `docs/specifications/scenarionet_integration_v1.1_specification.md` §§17.2 and 24.
- Status: `IMPLEMENTED`.
- Created and last updated: 2026-07-21.

## 2. Objective And Scope

Expose the concrete geometry behind a fatal semantic-observation route-projection
failure: scenario/step, actor and ego elevations, nearest planar route segment,
its interpolated elevation, minimum vertical mismatch and compatible-segment
count. Preserve the 3 m vertical threshold, all token values, selection, reward,
termination and dataset policy.

## 3. Requirements, Design, And Traceability

| ID | Acceptance criterion | Implementation | Test |
|---|---|---|---|
| `REQ-RPD-001` | A rejected actor projection reports all diagnostic fields. | `RoutePolyline.projection_diagnostics`; semantic builder error context. | route/semantic unit regression. |
| `REQ-RPD-002` | Valid projections and current finite-token contract remain unchanged. | no success-path change. | existing causal semantic tests. |
| `REQ-RPD-003` | No scientific fallback is added. | bounded diff review. | `git diff --check`. |

The diagnostic helper computes its nearest planar segment deterministically using
the existing segment order. It does not select a segment or alter `project()`.
The semantic builder raises the existing error category only after preserving the
original failure as its cause.

Commands: focused pytest for route geometry and causal semantic batches; focused
Ruff format/check; `git diff --check`.

## 4. Milestones And Findings

- [x] M1: verify that Waymo map geometry is 3D, while the selected project
  dataset invalidates Waymo routes with z-range above 4 m.
- [x] M2: add deterministic diagnostic context and regression tests.
- [x] M3: run focused validation and publish the reproduction result.

Finding: current failures do not establish an overpass. A selected dynamic actor
can be vertically compatible with the ego but lack a compatible assigned-route
segment; the next run must reveal whether this is a true plane separation or a
live/source elevation-alignment defect.

Implementation: `RoutePolyline.projection_diagnostics()` reports the nearest
planar segment independently of vertical acceptance. Dynamic-token construction
preserves the original fatal condition but adds scenario, step, actor/ego and
route-elevation facts. Focused Docker pytest passed 34 tests in 1.98s; focused
Ruff format/check and `git diff --check` passed.

## 5. Decisions, Deviations, And Files

No approval gate: diagnostics do not alter observable scientific behavior. No
fallback, threshold or feature schema change is authorized.

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/rulebook/v2/geometry/route.py` | Modify | Compute read-only projection diagnostics. |
| `src/thesis_rl/envs/observations/causal_semantic.py` | Modify | Attach scenario and actor context to the existing failure. |
| `tests/test_rulebook_v2_geometry.py` | Modify | Diagnostic-helper regression. |
| `tests/test_causal_semantic_batch.py` | Modify | Semantic error-context regression. |
| `docs/project_index.md` | Modify | Register the implementation record. |

## 6. Validation Results And Reconciliation

`REQ-RPD-001`--`REQ-RPD-003` are implemented and verified by focused unit
tests. The next user-owned ScenarioNet run is the required real-fixture
diagnostic; no experimental result is valid until the elevation relation is
understood.
