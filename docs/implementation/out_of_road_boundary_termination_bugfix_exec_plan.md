# ExecPlan — Boundary-only out-of-road termination regression

## 1. Metadata

- Feature: ScenarioNet out-of-road termination
- Plan ID: `BUG-SN-BOUNDARY-001`
- Authoritative specification: `docs/specifications/scenarionet_integration_v1.1_specification.md`, `SCENARIONET-INTEGRATION` v1.1, `APPROVED`
- Status: `IMPLEMENTED`
- Date: 2026-07-20
- Related ADRs: `docs/decisions/ADR-001-scenarionet-v1-1-dataset-policy.md`

## 2. Objective and scope

Prevent a boundary-only MetaDrive probe from being reported as a physical
sidewalk exit. Preserve termination for actual sidewalk/guardrail contact and
for lateral displacement beyond `max_lateral_dist`.

## 3. Authoritative requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-BOUNDARY-001` | Boundary-only contact/continuous-line crossing is non-terminal; physical road exit remains terminal. | §20.1–§20.2 |

## 4. Current repository analysis

- `VERIFIED`: `BaseVehicle._state_check` maps `ROAD_EDGE_BOUNDARY`,
  `ROAD_EDGE_SIDEWALK`, and `GUARDRAIL` to `crash_sidewalk`.
- `VERIFIED`: `SceneContextAdapter.is_physically_out_of_road` previously
  treated every `crash_sidewalk` flag as a physical exit.
- `VERIFIED`: `ThesisScenarioEnv.done_function` also retained
  `CRASH_SIDEWALK` in the aggregate termination predicate after clearing
  `OUT_OF_ROAD`.
- `VERIFIED`: the native `ScenarioEnv.done_function` also derives the
  aggregate `CRASH` key from `CRASH_SIDEWALK`; clearing only
  `CRASH_SIDEWALK` leaves a boundary-only episode terminal.
- `VERIFIED`: `BaseVehicle.contact_results` is the native persistent set updated
  by `_state_check`; it records the exact sidewalk probe result and is the
  available local discriminator between `ROAD_EDGE_BOUNDARY` and
  `ROAD_EDGE_SIDEWALK`/`GUARDRAIL`.
- `VERIFIED`: the native reward/cost paths call `_is_out_of_road`, so the same
  physical predicate must be exposed through the subclass override to avoid
  penalising a boundary-only probe as out of road before `done_function` runs.
- `VERIFIED`: the reported GIF is not present in the current workspace, so
  visual re-render validation is unavailable in this checkout.

## 5. Invariants

- `ROAD_EDGE_BOUNDARY` alone does not imply `physical_out_of_road`.
- `ROAD_EDGE_SIDEWALK` or `GUARDRAIL` implies physical exit.
- `abs(current_lateral) > max_lateral_dist` remains a physical exit.
- A boundary-only probe must not leave `CRASH` or `CRASH_SIDEWALK` active in
  the returned termination info.
- A physical sidewalk/guardrail contact and an independent native collision
  remain terminal even when a boundary probe was also observed.
- Existing collision, destination, and timeout semantics remain unchanged.

## 6. Decisions and approval gates

No unresolved approval gate. The fix implements the already-approved §20.1
contract and introduces no new public configuration or dependency.

## 7. Proposed design

Use the native persistent `contact_results` classification to refine the
generic `crash_sidewalk` flag. Expose that physical predicate through
`ThesisScenarioEnv._is_out_of_road` so native reward/cost and termination paths
share the same semantics. When the adapter classifies boundary-only contact as
non-physical, clear `CRASH_SIDEWALK`, recompute the derived `CRASH` key, and
then recompute aggregate termination. Preserve a pre-existing generic crash
when it is not derived from the sidewalk flag.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-BOUNDARY-001` | `ROAD_EDGE_BOUNDARY` does not terminate; sidewalk/guardrail and lateral exit do | `src/thesis_rl/envs/scene_context.py`, `src/thesis_rl/envs/thesis_scenario_env.py` | Focused boundary, shared-predicate, physical-contact, lateral-exit, and collision regressions in `tests/test_thesis_scenario_env.py` | Implemented and verified with focused suite |

## 9. Test strategy

```text
uv run --no-sync python -m pytest -q tests/test_thesis_scenario_env.py
uv run --no-sync ruff check src/thesis_rl/envs/scene_context.py src/thesis_rl/envs/thesis_scenario_env.py tests/test_thesis_scenario_env.py
uv run --no-sync ruff format --check src/thesis_rl/envs/scene_context.py src/thesis_rl/envs/thesis_scenario_env.py tests/test_thesis_scenario_env.py
git diff --check
```

Mandatory matrix frozen before the production edit:

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-BOUNDARY-001` | Unit | Boundary probe classification | `crash_sidewalk=True`, `contact_results={ROAD_EDGE_BOUNDARY}` | physical exit is false | `REQ-BOUNDARY-001` |
| `TEST-BOUNDARY-002` | Regression | Native aggregate termination | `out_of_road=True`, `crash_sidewalk=True`, `crash=True`, boundary contact | `done=False`, all boundary crash flags false | `REQ-BOUNDARY-001` |
| `TEST-BOUNDARY-003` | Unit | Physical sidewalk/guardrail | `ROAD_EDGE_SIDEWALK` or `GUARDRAIL` | physical exit and termination remain true | `REQ-BOUNDARY-001` |
| `TEST-BOUNDARY-004` | Unit | Lateral fallback | `abs(current_lateral) > max_lateral_dist` | physical exit remains true | `REQ-BOUNDARY-001` |
| `TEST-BOUNDARY-005` | Regression | Native collision passthrough | boundary probe plus `crash_vehicle=True` | termination remains true | `REQ-BOUNDARY-001` |
| `TEST-BOUNDARY-006` | Integration-oriented unit | Shared native predicate | boundary probe through `_is_out_of_road` | native reward/cost predicate is false | `REQ-BOUNDARY-001` |

## 10. Milestones

- [x] M1 — Confirm native flag conflation and freeze deterministic regression matrix.
- [x] M2 — Refine physical-boundary classification, shared native predicate, and aggregate termination.
- [x] M3 — Run focused validation and reconcile final diff. The repository
  `.venv` still lacks pytest/Ruff, so the supported `uv run --no-sync` commands
  remain unavailable; the system pytest ran the exact focused test file with a
  test-process-only `omegaconf` stub.

## 11. Progress and findings log

- `2026-07-20`: Confirmed the native rectangular probe sets `crash_sidewalk`
  for `ROAD_EDGE_BOUNDARY`; implemented classification and tests.
- `2026-07-20`: Native inspection found a missed aggregate: MetaDrive derives
  `CRASH` from `CRASH_SIDEWALK`, so the first draft still terminated on a
  boundary-only probe. The reward/cost call path also uses `_is_out_of_road`;
  the correction must therefore cover both shared predicate and done info.
- `2026-07-20`: Python syntax compilation and `git diff --check` passed. The
  focused pytest/Ruff commands could not run: `uv` cannot write its global
  cache, `.venv` has no pytest/Ruff, and Docker access is denied.
- `2026-07-20`: Added the shared `_is_out_of_road` override, recomputation of
  the derived crash aggregate with generic-crash preservation, and physical
  sidewalk/collision regressions. Focused test file passed: `30 passed`.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/envs/scene_context.py` | Modified | Distinguish boundary-only contact from physical exit |
| `src/thesis_rl/envs/thesis_scenario_env.py` | Modified | Apply the physical predicate to native call paths and prevent boundary-only aggregate termination |
| `tests/test_thesis_scenario_env.py` | Modified | Boundary, aggregate, physical-contact, collision, and shared-predicate regression coverage |

## 14. Validation results

| Command | Result | Date | Notes |
|---|---|---|---|
| Python syntax compilation | PASS | 2026-07-20 | Modified Python files compile successfully |
| Focused pytest in provisioned project environment | NOT_RUN | 2026-07-20 | `uv run --no-sync` could not run because `.venv` lacks pytest; Docker socket is inaccessible |
| Focused pytest equivalent | PASS | 2026-07-20 | System pytest 9.0.2, exact `tests/test_thesis_scenario_env.py`, with a process-only `omegaconf` import stub: `30 passed` |
| Focused Ruff | NOT_RUN | 2026-07-20 | `uv run --no-sync` could not spawn Ruff because `.venv` lacks it; Docker socket is inaccessible |
| `git diff --check` | PASS | 2026-07-20 | No whitespace errors |

## 15. Final reconciliation

`REQ-BOUNDARY-001` is implemented and verified by the focused equivalent suite.
The repository's provisioned `uv` environment still lacks pytest and Ruff, so
the exact primary-environment commands remain pending. The reported GIF and
its source run artifacts were not present in this checkout; visual re-render
confirmation and a live ScenarioNet replay remain pending.
