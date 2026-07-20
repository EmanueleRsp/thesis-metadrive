# Rulebook R1 collision-floor bugfix ExecPlan

## 1. Metadata

- Feature: Rulebook v2 R1 collision severity live-contact correctness
- Plan ID: `R1-COLLISION-FLOOR-BUGFIX`
- Authoritative specification: `docs/specifications/rulebook_v4.7_specification.md`, Rulebook v4.7, approved
- Status: `VERIFIED`
- Created: 2026-07-20
- Last update: 2026-07-20
- Related ADRs: `docs/decisions/ADR-015-r1-pre-state-centerline-normal.md`
- Owner: Codex

## 2. Objective And Scope

Ensure that a real collision does not systematically collapse to margin
`-1e-6` because the live Bullet contact normal is invalid or unavailable.
Use one deterministic pre-state canonical-footprint centerline for every R1
normal, while preserving the collision floor, onset-only semantics, actor-class
caps, and fail-fast behavior.

In scope: Bullet normal fallback orientation, current-contact reconciliation,
stale persistent-manifold filtering, deterministic regression tests, and
traceability documentation. Out of scope: changing the R1 scientific formula,
floor value, normalization caps, or native MetaDrive collision flags.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| REQ-R1-001 | Compute normal closing speed from pre-state velocities and canonical-footprint centerline unit normals | §5.3 |
| REQ-R1-002 | Emit collision only for new contact onsets and preserve active-contact memory | §5.2 |
| REQ-R1-003 | Apply the bounded cost and `1e-6` numerical floor exactly | §§5.4–5.5 |
| REQ-R1-004 | Do not use stale/persistent contacts as current active contacts | §5.2 |

## 4. Current Repository Analysis

- `SPECIFIED`: amended R1 uses `m1=-c1`, squared normal closing speed,
  pre-state velocity, pre-state canonical-footprint centerline normals, and
  `epsilon_col=1e-6` in `rulebook_v4.7_specification.md` §§5.2–5.5.
- `VERIFIED`: the pure evaluator computes the formula correctly from the
  canonical pre-state positions.
- `VERIFIED`: `context/metadrive_live.py` derives coincident-point fallback
  normals from `getNormalWorldOnB` with the sign opposite to Bullet's B-to-A
  convention.
- `VERIFIED`: callback-only actor IDs are unioned into current active IDs even
  when no longer present in the live manifold/contact query.
- `VERIFIED`: persistent manifold records are accepted without a distance
  validity check when the binding exposes `getDistance`.
- `VERIFIED`: these issues can produce zero normal closing speed and therefore
  the observed floor margin, especially for coincident Bullet contact points.

## 5. Assumptions And Invariants

- R1 derives exactly one unit normal per onset actor from canonical pre-state
  footprint centers, oriented ego to the other actor.
- Coincident pre-state centers within `1e-6 m` fail fast.
- A current active contact must be present in the current persistent manifold
  or contact-test result when those live queries are available.
- A callback-only contact remains an onset candidate, but must not remain active
  after the current live query confirms separation.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| DEC-R1-001 | specification clarification | Live Bullet vehicle contacts can expose zero-XY normals or no manifold. | A: retain manifold normal; B: derive every R1 normal from pre-state canonical footprint centers. | B | Changes the R1 normal source for every collision; preserves velocity, cap, floor, and onset semantics. | Approved 2026-07-20; ADR-015 |

## 7. Proposed Design

1. Store only stable actor identity/class in a contact onset record; no Bullet
   manifold point or normal is required.
2. Add a small adapter helper to identify whether a current live contact query
   is available. When available, derive active IDs only from current query
   results; retain callback IDs only as a compatibility fallback when no query
   exists.
3. Ignore persistent manifold points whose finite distance is strictly
   positive, while retaining compatibility with bindings/test doubles that do
   not expose distance.
4. Derive the R1 normal from the centroids of the ego and onset actor canonical
   pre-state footprints, and fail fast for coincident centers.
5. Add tests for front, rear, side, node-only callback, stale callback contact
   removal, stale manifold removal, and a non-floor R1 result.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| REQ-R1-001 | Front, rear, and side impacts derive the expected positive closing speed from pre-state centers | `components/collision.py`, `transition.py` | `test_rulebook_v2_collision.py` | IMPLEMENTED/VERIFIED |
| REQ-R1-002 | Separation removes active ID while onset is retained | `context/metadrive_live.py` | `test_rulebook_v2_metadrive_live.py` | IMPLEMENTED/VERIFIED |
| REQ-R1-003 | Existing formula and floor remain unchanged | `components/collision.py` | `test_rulebook_v2_collision.py` | IMPLEMENTED/VERIFIED |
| REQ-R1-004 | Positive-distance manifold is excluded | `context/metadrive_live.py` | `test_rulebook_v2_metadrive_live.py` | IMPLEMENTED/VERIFIED |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Expected result |
|---|---|---|---|
| TEST-R1-001 | Unit | Front, rear, and side pre-state centerlines | Each impact has the expected non-floor cost |
| TEST-R1-002 | Unit | Coincident pre-state centers | `RulebookEvaluationError` |
| TEST-R1-003 | Unit | Positive pre-state closing speed through R1 | Cost is greater than floor |
| TEST-R1-004 | Unit | Callback contact disappears before snapshot | Active IDs are empty, onset is preserved |
| TEST-R1-005 | Unit | Persistent manifold has positive distance | It is excluded from active contacts |
| TEST-R1-006 | Unit | Node-only Bullet callback with resolvable actor | Onset is retained without a manifold |
| TEST-R1-007 | Regression | Existing collision/live adapter suite | All focused tests pass |

Commands: `python -m pytest -q tests/test_rulebook_v2_collision.py
tests/test_rulebook_v2_metadrive_live.py tests/test_rulebook_v2_snapshot.py
tests/test_rulebook_v2_live_adapter.py tests/test_rulebook_v2_live_integration.py`;
`git diff --check`; focused Ruff on modified Python files where available.

## 10. Milestones

- [x] Replace Bullet normal extraction with deterministic pre-state centerline normal and direct regressions.
- [x] Correct live active-contact reconciliation and stale manifold filtering.
- [x] Run focused tests and diff checks.
- [x] Reconcile this plan and run the containerized Rulebook v2 regression and
  live integration smoke.

## 11. Progress And Findings Log

| Date | Finding/evidence | Action/result |
|---|---|---|
| 2026-07-20 | Pure evaluator returns `0.0625` for a 5 m/s canonical impact; floor is not intrinsic to the formula. | Fix is scoped to live contact adapter. |
| 2026-07-20 | Coincident fallback uses Bullet B-to-A normal without the required sign inversion. | Regression and correction planned. |
| 2026-07-20 | Corrected coincident-point normal for both Bullet node orderings. | Two regressions pass. |
| 2026-07-20 | Callback-only IDs could remain active after live separation; stale positive-distance manifolds were accepted. | Current-query reconciliation and distance filtering implemented; two regressions pass. |
| 2026-07-20 | Run `20260720_152057` contains four `crash_vehicle` records with `new_collision=true` and raw closing speed squared exactly zero, plus one `crash_vehicle` record with no R1 onset. | Confirmed that the floor is generated before bounded cost; continued live audit. |
| 2026-07-20 | Controlled live Bullet contact exposed a normal with no XY component. | DEC-R1-001 opened; no unapproved fallback implemented. |
| 2026-07-20 | User approved always-on deterministic pre-state canonical-footprint centerline normal. | ADR-015 and Rulebook v4.7 clarification added; implementation in progress. |
| 2026-07-20 | Containerized Rulebook v2 suite and scoped Ruff completed after the approved change. | 183 passed, 1 expected skip; Ruff and `git diff --check` passed. |

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/rulebook/v2/components/collision.py` | Modify | Derive deterministic R1 normal and emit diagnostic provenance |
| `src/thesis_rl/rulebook/v2/context/metadrive_live.py` | Modify | Record onset identity without requiring a manifold |
| `src/thesis_rl/rulebook/v2/types.py` | Modify | Remove manifold geometry from contact-onset contract |
| `tests/test_rulebook_v2_collision.py` | Modify | Add front/rear/side and coincident-center regressions |
| `tests/test_rulebook_v2_metadrive_live.py` | Modify | Add node-only callback regression |
| `docs/decisions/ADR-015-r1-pre-state-centerline-normal.md` | Add | Record approved material decision |
| `docs/implementation/rulebook_r1_collision_floor_bugfix_exec_plan.md` | Add/update | Record implementation and validation |

## 14. Validation Results

| Command | Result | Date | Notes |
|---|---|---|---|
| `python -m pytest -q tests/test_rulebook_v2_collision.py tests/test_rulebook_v2_metadrive_live.py tests/test_rulebook_v2_snapshot.py tests/test_rulebook_v2_live_adapter.py tests/test_rulebook_v2_f6_contracts.py tests/test_rulebook_v2_transition.py tests/test_rulebook_v2_live_integration.py` | PASS | 2026-07-20 | 48 passed after deterministic centerline implementation |
| `make rulebook-v2-check` | PASS | 2026-07-20 | Containerized suite: 183 passed, 1 expected skip; scoped Ruff and whitespace check passed |
| `git diff --check` | PASS | 2026-07-20 | No whitespace errors |

## 15. Final Reconciliation

REQ-R1-001 through REQ-R1-004 are implemented and verified. R1 now derives
every normal from causal pre-state canonical-footprint centers, exposes the
normal and provenance in diagnostics, and accepts node-only callbacks as onset
evidence. The contact floor remains meaningful for genuine zero normal closing
speed, but no longer reflects Bullet normal projection or manifold availability.

Known limitation: exactly coincident pre-state centers fail fast by approved
contract. The recorded validation includes the containerized live integration
smoke; it does not rerun a long training experiment, which is not a substitute
for deterministic correctness tests.
