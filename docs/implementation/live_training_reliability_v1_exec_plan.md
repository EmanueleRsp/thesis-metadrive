# Live Training Reliability v1 ExecPlan

## 1. Metadata

- Feature and plan ID: `LIVE-TRAINING-RELIABILITY-V1`.
- Authoritative contracts: `docs/specifications/observation_v1.2_specification.md` (APPROVED), `docs/specifications/rulebook_v4.7_specification.md` §7.6 (APPROVED), `docs/specifications/automatic_curriculum_learning_v1_specification.md` §§12--13 and §28 (AUTHORITATIVE), and `docs/specifications/rl_baselines_v1_specification.md` (APPROVED).
- Status: `IN_PROGRESS` (static-map projection fix implemented; validation pending).
- Created/updated: 2026-07-23.
- Related ADRs: ADR-014, ADR-016, ADR-022.

## 2. Objective and scope

Make the requested ScenarioNet ACL runs robust to non-policy failures while preserving frozen data, the vertical perception guard, Rulebook fail-fast behavior for a pertinent invalid signal, and ACL learning-potential semantics. Scope: static-map observation projection, traffic-control state propagation, PPO-SB3 device selection, live event parity, and ACL MAB diagnostics. No data mutation, reward/LP formula change, schema change, or fallback from invalid pertinent signal data is in scope.

## 3. Requirements and current analysis

| ID | Requirement | Source | Verified current evidence |
|---|---|---|---|
| REQ-001 | Static HD-map features use nearest geometry plus range/vertical guards and preserve the route-ordering contract; optional features without a vertically compatible route segment are omitted and masked. | OBS-V1.2 §§3, 6.3, 7 | `_build_static_v12` applies an ego pre-filter then route-aware projection. The supplied Waymo trace shows an optional `road_boundary` passing the former and failing the latter. |
| REQ-002 | A pertinent signal with unknown/incomplete state is `NOT_EVALUABLE`, fail-fast; non-pertinent signals do not affect eligibility. | Rulebook v4.7 §7.6.1 | TD3 traceback reaches `evaluate_signal_transition` with unknown pre/post state. |
| REQ-003 | ACL value and MAB feedback depend only on algorithm-specific LP and retain deterministic state. | ACL §12--13, §28 | Existing runtime uses six arms; update/logging evidence remains to be verified. |
| REQ-004 | PPO-SB3 scientific backend remains supported and observable runtime diagnostics are consistent with other algorithms. | RL Baselines v1 | `device=auto` reaches SB3 PPO; its MLP CUDA warning is reproduced by configuration inspection. |

Invariant: feature elevation is tested with `is not None`; zero is a valid elevation. Map features retain their source 2.5D profile and are projected using the interpolated elevation at the nearest observed point; an incompatible route projection remains fail-fast with diagnostics. Pertinent unknown controls remain fatal. PPO alone resolves `auto` to CPU, with an explicit configured device still honored.

## 4. Decisions and design

| ID | Category | Decision | Status |
|---|---|---|---|
| DEC-001 | implementation detail | Preserve source 2.5D elevation profiles and use local interpolation; never project onto another vertical plane. Omit optional static features with no compatible route segment and retain their mask. | Implemented |
| DEC-002 | implementation detail | Resolve PPO-SB3 `auto` to `cpu`; preserve an explicit user device override. | In progress |
| DEC-003 | blocking technical issue | Inspect ACL update/monitor flow before modifying weights; formulas and hyperparameters are protected. | In progress |

Design: preserve source 2.5D profiles in `MapFeatureRecord`, interpolate elevation at the nearest XY point, and enrich any incompatible static projection failure with scenario/feature/route diagnostics. Preserve `RoutePolyline.project` fail-closed and do not alter observation masks. Validate signal snapshots at catalog/preflight boundary if the trace reveals a non-pertinent control being selected; otherwise retain the present failure and improve context only. Route PPO device resolution in builder/factory before model construction. Supply the vectorized ACL loop with the same episode context callback used by the single-environment live monitor and expose MAB selection/update state without changing it.

## 5. Traceability and mandatory test matrix

| Requirement | Acceptance criteria | Tests | Status |
|---|---|---|---|
| REQ-001 | AC-001: incompatible optional feature is masked/omitted; compatible and zero-elevation features retain coordinates. | `tests/test_causal_semantic_batch.py` static projection regressions | Implemented; validation pending |
| REQ-002 | AC-002: pertinent UNKNOWN still fails; irrelevant UNKNOWN remains non-applicable. | controls/transition regressions | Planned |
| REQ-003 | AC-003: a deterministic MAB update changes weights/probabilities; live rows show current state. | ACL unit and vector-monitor regression | Planned |
| REQ-004 | AC-004: PPO auto device is CPU; explicit device passes through; vector events include ACL context. | SB3/direct backend and agent monitor tests | Planned |

Commands: focused `pytest`, focused Ruff format/check, `git diff --check`; representative smoke through the existing `make smoke` only if the provisioned environment permits it.

## 6. Milestones and findings

- [x] M1: classify supplied failures and map them to code.
- [x] M2: establish exact signal and static-feature paths with tests.
- [x] M3: implement minimal fixes and live-monitor parity.
- [ ] M4: validate and reconcile documentation/index.

Finding 2026-07-23: SAC, TD3, and PPO supplied traces fail in `_build_static_v12` for the same Waymo `road_boundary`. The feature passes the ego vertical pre-filter but has no route-compatible segment. The implementation now omits only this optional case, while structural projection errors remain fail-fast.

## 7. Deviations

No deviations identified.

## 8. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/envs/observations/causal_semantic.py` | Modify | Local-elevation static-map projection, explicit route-incompatibility omission, and debug diagnostics. |
| `src/thesis_rl/rulebook/v2/types.py` | Modify | Canonical 2.5D feature profile and interpolation. |
| `src/thesis_rl/rulebook/v2/context/pg_static_adapter.py` and `waymo_static_adapter.py` | Modify | Preserve source elevation profiles. |
| `src/thesis_rl/runtime/wiring/builders.py` | Modify | PPO runtime device selection. |
| `src/thesis_rl/curriculum/scenario_acl/driver.py` | Modify | Vector monitor parity/MAB diagnostics if verified. |
| `tests/test_causal_semantic_batch.py` | Modify | Compatible, incompatible, mixed, empty, structural-error, and Waymo-equivalent projection regressions. |
| `tests/test_sb3_direct_backends.py` and ACL tests | Modify | Device and monitor/MAB regressions. |

## 9. Validation results and final reconciliation

| Command | Result | Date | Notes |
|---|---|---|---|
| `pytest -q tests/test_causal_semantic_batch.py tests/test_scenario_acl_mab.py tests/test_sb3_direct_backends.py` | NOT_RUN | 2026-07-23 | Host Python lacks `omegaconf`; use the primary container command. |
| `pytest -q tests/test_causal_semantic_batch.py` | FAIL (environment) | 2026-07-23 | System Python cannot collect tests: `omegaconf` is missing. |
| `UV_CACHE_DIR=/tmp/thesis-uv-cache uv run --no-sync python -m pytest -q tests/test_causal_semantic_batch.py` | FAIL (environment) | 2026-07-23 | Repository `.venv` has no `pytest`; no test execution occurred. |
| `python -c 'import ast; ...'` | PASS | 2026-07-23 | AST parsing passed for the modified source and test files. |
| `git diff --check` | PASS | 2026-07-23 | No whitespace errors. |
| Scenario smoke for `waymo:training_20s:14139de6bce54874` | NOT_RUN | 2026-07-23 | Primary runtime/container unavailable in this workspace; exact follow-up remains the requested `make run` command with each SB3 algorithm. |
| `.venv/bin/python -m ruff format --check ...` and `ruff check ...` | NOT_RUN | 2026-07-23 | The checked project environment has no Ruff module. |
| `git diff --check` | PASS | 2026-07-23 | No whitespace errors. |

REQ-001, PPO device/event parity, and ACL monitor visibility are implemented pending primary-environment validation. REQ-002 remains partial: the Rulebook correctly fails closed, but the selected catalog contains a scenario whose live pertinent signal reaches `UNKNOWN`; rebuild/filter the frozen catalog before resuming experiments.
