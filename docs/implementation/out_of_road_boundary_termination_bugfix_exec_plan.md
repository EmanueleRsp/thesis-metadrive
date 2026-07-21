# ExecPlan — Boundary-only out-of-road termination regression

## 1. Metadata

- Feature: ScenarioNet out-of-road termination and Rulebook lane geometry
- Plan ID: `BUG-SN-BOUNDARY-001`
- Authoritative specification: `docs/specifications/scenarionet_integration_v1.1_specification.md`, `SCENARIONET-INTEGRATION` v1.1, `APPROVED`; `docs/specifications/rulebook_v4.7_specification.md`, Rulebook v4.7, `APPROVED`
- Status: `IN_PROGRESS`
- Date: 2026-07-21
- Related ADRs: `docs/decisions/ADR-001-scenarionet-v1-1-dataset-policy.md`

## 2. Objective and scope

Prevent a boundary-only MetaDrive probe and reference-route deviation from
being reported as a physical sidewalk exit. Correct the ScenarioNet/Waymo
lane-width geometry used by the Rulebook off-road component. Preserve
termination for actual sidewalk/guardrail contact and the Rulebook's
normative union of drivable lane surfaces.

## 3. Authoritative requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-BOUNDARY-001` | Boundary-only contact/continuous-line crossing is non-terminal; physical road exit remains terminal. | §20.1–§20.2 |
| `REQ-BOUNDARY-002` | A fully off-surface ego footprint is terminal even when ScenarioNet emits no road-edge contact. | §20.1–§20.2; `rulebook_v4.7_specification.md` §2.9.4 |
| `REQ-RULEBOOK-OFFROAD-001` | Off-road area is computed against correctly reconstructed vertically compatible drivable lane polygons. | `rulebook_v4.7_specification.md` §2.9.4 |

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
- Route-relative lateral deviation alone is not a physical exit.
- An ego footprint entirely outside the canonical, vertically compatible
  Rulebook drivable surface is a physical exit even if ScenarioNet has emitted
  no road-edge contact for that control step.
- Route-navigation values (`current_lateral`, `dist_to_left_side`,
  `dist_to_right_side`, `on_lane`) are diagnostics only and never establish a
  physical exit in ScenarioNet.
- ScenarioNet/Waymo `width[:, 0]` and `width[:, 1]` are respectively the
  per-point left and right centerline-to-boundary distances; their sum is the
  lane width and must not be halved again.
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
generic `crash_sidewalk` flag. Do not use route-navigation diagnostics as a
boundary test because ScenarioNet's `TrajectoryNavigation` measures all of
them against the assigned route. Reconstruct Waymo lane polygons from their
per-sample left/right widths rather than flattening and halving those values.
Expose the physical predicate through `ThesisScenarioEnv._is_out_of_road` so native
reward/cost and termination paths share the same semantics. When the adapter
classifies boundary-only contact as non-physical, clear `CRASH_SIDEWALK`,
recompute the derived `CRASH` key, and then recompute aggregate termination.
Preserve a pre-existing generic crash when it is not derived from the sidewalk
flag.

For Rulebook-v2 ScenarioNet episodes, the shared predicate additionally uses
the current canonical live ego footprint and the cache's vertically compatible
drivable-lane union. Only an outside area equal to the whole footprint within
the Rulebook area epsilon is terminal; partial off-road occupancy remains a
Rulebook violation without changing termination semantics.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-BOUNDARY-001` | `ROAD_EDGE_BOUNDARY` and reference-lane deviation do not terminate; sidewalk/guardrail do | `src/thesis_rl/envs/scene_context.py`, `src/thesis_rl/envs/thesis_scenario_env.py` | Focused boundary, shared-predicate, physical-contact, reference-lane, and collision regressions in `tests/test_thesis_scenario_env.py` | In progress |
| `REQ-RULEBOOK-OFFROAD-001` | A Waymo asymmetric left/right width generates the full asymmetric lane polygon | `src/thesis_rl/rulebook/v2/context/waymo_static_adapter.py` | `tests/test_rulebook_v2_waymo_adapter.py::test_waymo_adapter_uses_per_side_widths_without_halving_them_again` | In progress |
| `REQ-BOUNDARY-002` | A fully off-surface live footprint remains terminal without native contact | `src/thesis_rl/envs/scene_context.py` | Full-footprint geometric-exit regression in `tests/test_thesis_scenario_env.py` | Implemented and deterministically verified |

## 9. Test strategy

```text
uv run --no-sync python -m pytest -q tests/test_thesis_scenario_env.py tests/test_rulebook_v2_waymo_adapter.py
uv run --no-sync ruff check src/thesis_rl/envs/scene_context.py src/thesis_rl/envs/thesis_scenario_env.py src/thesis_rl/rulebook/v2/context/waymo_static_adapter.py tests/test_thesis_scenario_env.py tests/test_rulebook_v2_waymo_adapter.py
uv run --no-sync ruff format --check src/thesis_rl/envs/scene_context.py src/thesis_rl/envs/thesis_scenario_env.py src/thesis_rl/rulebook/v2/context/waymo_static_adapter.py tests/test_thesis_scenario_env.py tests/test_rulebook_v2_waymo_adapter.py
git diff --check
```

Mandatory matrix frozen before the production edit:

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-BOUNDARY-001` | Unit | Boundary probe classification | `crash_sidewalk=True`, `contact_results={ROAD_EDGE_BOUNDARY}` | physical exit is false | `REQ-BOUNDARY-001` |
| `TEST-BOUNDARY-002` | Regression | Native aggregate termination | `out_of_road=True`, `crash_sidewalk=True`, `crash=True`, boundary contact | `done=False`, all boundary crash flags false | `REQ-BOUNDARY-001` |
| `TEST-BOUNDARY-003` | Unit | Physical sidewalk/guardrail | `ROAD_EDGE_SIDEWALK` or `GUARDRAIL` | physical exit and termination remain true | `REQ-BOUNDARY-001` |
| `TEST-BOUNDARY-004` | Regression | Reference-route deviation | Negative reference side distance, no physical contact | physical exit remains false | `REQ-BOUNDARY-001` |
| `TEST-BOUNDARY-005` | Regression | Native collision passthrough | boundary probe plus `crash_vehicle=True` | termination remains true | `REQ-BOUNDARY-001` |
| `TEST-BOUNDARY-006` | Integration-oriented unit | Shared native predicate | boundary probe through `_is_out_of_road` | native reward/cost predicate is false | `REQ-BOUNDARY-001` |
| `TEST-BOUNDARY-007` | Regression | Route deviation on road | `current_lateral=5.0`, positive native side distances | `done=False` and `out_of_road=False` | `REQ-BOUNDARY-001` |
| `TEST-RULEBOOK-OFFROAD-001` | Unit | ScenarioNet/Waymo asymmetric lane width | left/right widths `(2.0 m, 1.0 m)` | polygon spans `[-1.0 m, +2.0 m]`, not a halved median buffer | `REQ-RULEBOOK-OFFROAD-001` |
| `TEST-BOUNDARY-008` | Regression | Missing native road-edge contact | valid live ego footprint wholly outside canonical Rulebook surface, no contacts | physical exit and `out_of_road` are true | `REQ-BOUNDARY-002` |

## 10. Milestones

- [x] M1 — Confirm native flag conflation and freeze deterministic regression matrix.
- [x] M2 — Refine physical-boundary classification, shared native predicate, and aggregate termination.
- [x] M3 — Run focused validation and reconcile final diff. The repository
  `.venv` still lacks pytest/Ruff, so the supported `uv run --no-sync` commands
  remain unavailable; the system pytest ran the environment, Rulebook wrapper,
  evaluation artifacts, and Waymo adapter suite with test-process-only import
  stubs: `52 passed`.
- [ ] M4 — Run the affected ScenarioNet evaluation in the provisioned runtime
  and confirm that a boundary-only trajectory remains non-terminal, while a
  fully off-surface trajectory terminates even if `contact_results` is empty.

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
- `2026-07-20`: The new evaluation manifest exposed a second false-positive
  path: `TrajectoryNavigation.current_lateral` is route-relative, while the
  `max_lateral_dist=4.0` fallback was treated as a road-boundary test. Replaced
  that fallback with native `dist_to_left_side`/`dist_to_right_side` and
  `on_lane=False`, and added a regression for large route deviation on-road.
- `2026-07-20`: The later live trajectory for
  `waymo:training_20s:3976d7f407ac1ca2` disproved that second remedy. At the
  terminal step `crash_sidewalk=False`, the only contact was
  `ROAD_LINE_BROKEN_SINGLE_WHITE`, and `dist_to_right_side=-0.3987 m`; source
  inspection proved that distance is also route-relative. Physical termination
  must therefore use contact primitives only.
- `2026-07-20`: ScenarioNet's Waymo converter was verified to store per-point
  left/right distances in the two columns of `lane["width"]`. The Rulebook
  adapter flattened those values and halved them again, shrinking the drivable
  polygons and producing false off-road area. Added full asymmetric-polygon
  reconstruction and a deterministic regression.
- `2026-07-20`: The focused environment, Rulebook wrapper, evaluation
  artifact, and Waymo adapter suite passed with `52 passed`. The corrected
  adapter also built the exact failing scenario with `77` valid lane records,
  assigned route `("128",)`, and no validation errors. Its lane `128` area
  changed from `125.257 m²` under the old narrowed construction to `636.538 m²`
  under the per-side reconstruction.
- `2026-07-21`: Live run `20260721_070110` exposed the complementary false
  negative. Its timeout trajectories have empty native contacts and
  `physical_out_of_road=false`, while the Rulebook evaluates the same live
  footprint with `outside_area_m2 == ego_area_m2`. Added the canonical
  full-footprint geometric proof, and persisted its boolean and areas through
  Rulebook diagnostics and final-evaluation trajectories. Focused equivalent
  tests passed (`53 passed`); a new live evaluation remains required for M4.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/envs/scene_context.py` | Modified | Distinguish boundary-only contact from physical exit |
| `src/thesis_rl/envs/thesis_scenario_env.py` | Modified | Apply the physical predicate to native call paths and prevent boundary-only aggregate termination |
| `src/thesis_rl/rulebook/v2/context/waymo_static_adapter.py` | Modified | Reconstruct ScenarioNet/Waymo per-side lane polygons without narrowing them |
| `src/thesis_rl/scenarios/bootstrap.py` | Modified | Keep the runtime native-termination inventory aligned with the physical-boundary primitives |
| `tests/test_thesis_scenario_env.py` | Modified | Boundary, aggregate, physical-contact, route-deviation, collision, and shared-predicate regression coverage |
| `tests/test_rulebook_v2_waymo_adapter.py` | Modified | Verify asymmetric ScenarioNet/Waymo width geometry |
| `src/thesis_rl/runtime/io/eval_artifacts.py` | Modified | Preserve termination and native road-boundary diagnostics in trajectory JSONL |
| `src/thesis_rl/rulebook/v2/wrapper.py` | Modified | Preserve native road-boundary diagnostics in Rulebook transition logs |
| `tests/test_rulebook_v2_wrapper.py` | Modified | Verify termination flags and road-boundary diagnostics cross the Rulebook wrapper |
| `tests/test_eval_artifacts.py` | Modified | Verify termination and road-boundary diagnostics are persisted |
| `conf/video/default.yaml` | Modified | Enable trajectory logs for final-evaluation auditability |
| `src/thesis_rl/envs/scene_context.py` | Modified | Add the missing-contact full-footprint geometric proof |
| `src/thesis_rl/envs/thesis_scenario_env.py` | Modified | Pass the environment to road-geometry diagnostics |
| `src/thesis_rl/rulebook/v2/wrapper.py` | Modified | Persist geometric-exit diagnostics in Rulebook transition logs |
| `src/thesis_rl/runtime/io/eval_artifacts.py` | Modified | Persist geometric-exit diagnostics in trajectory JSONL |
| `tests/test_thesis_scenario_env.py` | Modified | Regress full-footprint exit without native contact |
| `tests/test_rulebook_v2_wrapper.py` | Modified | Regress Rulebook diagnostic propagation |
| `tests/test_eval_artifacts.py` | Modified | Regress trajectory diagnostic persistence |

## 14. Validation results

| Command | Result | Date | Notes |
|---|---|---|---|
| Python syntax compilation | PASS | 2026-07-20 | Modified Python files compile successfully |
| Focused pytest in provisioned project environment | NOT_RUN | 2026-07-20 | `uv run --no-sync` could not run because `.venv` lacks pytest; Docker socket is inaccessible |
| Focused pytest equivalent | PASS | 2026-07-20 | System pytest 9.0.2, environment, Waymo adapter, Rulebook wrapper, and evaluation artifact tests with process-only `omegaconf`/`rich` stubs: `52 passed` |
| Exact failing-scenario static adapter | PASS | 2026-07-20 | `waymo:training_20s:3976d7f407ac1ca2` built with 77 lane records, assigned lane `128`, and no validation errors |
| Focused Ruff | NOT_RUN | 2026-07-20 | `uv run --no-sync` could not spawn Ruff because `.venv` lacks it; Docker socket is inaccessible |
| `git diff --check` | PASS | 2026-07-20 | No whitespace errors |
| Focused equivalent suite | PASS | 2026-07-21 | `53 passed`: environment, Rulebook wrapper, evaluation artifacts, and Waymo adapter with process-only `omegaconf`/`rich` compatibility stubs |

## 15. Final reconciliation

`REQ-BOUNDARY-001` and `REQ-RULEBOOK-OFFROAD-001` are implemented and verified
by focused deterministic tests plus direct static-adapter construction on the
exact failing scenario. The repository's provisioned `uv` environment still
lacks pytest and Ruff, so the exact primary-environment commands remain
pending. A live ScenarioNet re-evaluation is still required to close M4; its
expected result is no out-of-road termination for the recorded terminal state.

## 16. Full termination and Rulebook audit

- `2026-07-20`: Audited every production boundary in the ScenarioNet path:
  MetaDrive vehicle state detection, native reward, native cost, native
  `done_function`, `ThesisScenarioEnv.step`, Rulebook v2 monitor wrapper,
  legacy reward wrapper, vector transition normalization, agent episode
  accounting, and evaluation trajectory artifacts.
- `2026-07-20`: Confirmed that `ThesisScenarioEnv` is the constructor used by
  the ScenarioNet factory and Scenario ACL replay path. Direct `ScenarioEnv`
  construction remains only in the separate native MetaDrive smoke/replay
  path, outside the ScenarioNet v1.1 termination contract.
- `2026-07-20`: Added explicit `terminated`/`truncated` info fields at the
  environment and wrapper boundaries, and persisted `crash_sidewalk`,
  `physical_out_of_road`, `crossed_continuous_line`, `termination_reason`,
  `terminated`, and `truncated` in Rulebook diagnostics and trajectory logs.
- `2026-07-20`: Focused equivalent suite covering the environment regression,
  Rulebook wrapper propagation, evaluation artifacts, and the Waymo geometry
  adapter passed with `52 passed`. The normal system pytest command remains
  unavailable because the checkout environment lacks `omegaconf` and `rich`;
  the provisioned `.venv` also lacks pytest/Ruff.
- `2026-07-20`: Running the broader equivalent set exposed three pre-existing
  `RuleRewardWrapper` test failures caused by the installed Gymnasium version
  rejecting its non-`gymnasium.Env` dummy fixture. Those failures are outside
  this bugfix and do not involve termination behavior.
- `2026-07-20`: Enabled trajectory logging for final evaluation artifacts and
  added native road-boundary diagnostics to Rulebook and JSONL records. The
  next live evaluation will therefore identify the exact termination source
  instead of requiring visual inference from the GIF alone.
- `2026-07-21`: Live run `20260721_070110` exposed the complementary false
  negative. Its final-evaluation timeout episodes have empty native contacts
  and `physical_out_of_road=false`, but the Rulebook evaluates the same live
  footprint with `outside_area_m2 == ego_area_m2`. ScenarioNet therefore does
  not reliably produce a `ROAD_EDGE_*` contact after the vehicle has left the
  mapped road. The shared physical predicate must add the canonical Rulebook
  full-footprint geometric proof while retaining contact classification and
  excluding route-relative navigation diagnostics.
