# PG Static-Obstacle Amendment v1.1 ExecPlan

## 1. Metadata

- Plan ID: `PG-STATIC-OBSTACLE-001`
- Authoritative specification: `docs/specifications/scenarionet_integration_v1.1_specification.md`, ScenarioNet Integration v1.1, `APPROVED`
- Status: `IMPLEMENTED`
- Created / last updated: 2026-07-29
- Related ADR: `ADR-034-pg-static-obstacle-generation-and-arm-classification.md`
- Approval: explicit user instruction on 2026-07-29 to enable static obstacles in PG profiles and integrate them in arm classification.

## 2. Objective and scope

Enable bounded MetaDrive accident-scene generation for PG profiles and preserve a clean `A0_simple_low_traffic` reference set. A static obstacle remains an orthogonal tag; it must not by itself reclassify a scenario as `A5_critical_mixed`.

In scope: five PG-profile probabilities, generated-scenario static-obstacle detection/tagging, `A0` exclusion, manifests, specification/ADR, and deterministic tests. Out of scope: a new arm, route-blockage inference, changes to `A2`--`A5`, dataset regeneration, and checkpoint migration.

## 3. Authoritative requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-PSO-001` | PG profiles use approved per-block accident probabilities P0=0.00, P1=0.03, P2=0.08, P3=0.08, P5=0.15. | §9, amended 2026-07-29 |
| `REQ-PSO-002` | A realized generated static obstacle is tagged and affects arm assignment only by making `A0` ineligible. | §8/§16, amended 2026-07-29 |
| `REQ-PSO-003` | A bare static-obstacle tag never promotes a scenario to `A5`. | §16, amended 2026-07-29 |

## 4. Current repository analysis

- `src/thesis_rl/scenarios/pg/profiles.py`: verified profile values are currently hard-coded to zero and guarded against any non-zero value.
- `src/thesis_rl/scenarios/pg/generator.py`: verified `GenerationSpec.accident_prob` is passed unchanged to MetaDrive.
- `src/thesis_rl/scenarios/pg/exporter.py`: verified records currently call arm/tag functions without a static-obstacle argument.
- `src/thesis_rl/scenarios/arms.py`: verified `has_static_obstacle` is already supported by tag derivation but ignored by primary-arm assignment.
- `semantic_v3` already represents visible `STATIC_COLLIDABLE` actors; no observation-schema change is needed.

## 5. Assumptions and invariants

- MetaDrive interprets `accident_prob` per eligible road block; the configured value is not a per-scenario target.
- `has_static_obstacle` means a realized static accident-scene object, not merely a non-zero configured probability.
- Existing topology, traffic, VRU, and criticality precedence is unchanged. `A0` requires both existing simple/traffic predicates and no static obstacle.
- Existing frozen data are immutable; the change applies only to future generation/rebuilds.

## 6. Decisions and approval gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-PSO-001` | specification clarification | Static-obstacle schedule and arm semantics | retain zero / new arm / bounded schedule plus A0 exclusion | bounded schedule plus A0 exclusion | dataset composition and labels | Approved by user 2026-07-29; ADR-034 |

## 7. Proposed design

`PGProfile` validates probabilities in `[0, 1]` and supplies the approved constants. The generator records realized static objects in generation metadata before closing MetaDrive. The exporter derives the boolean from that metadata, adds `has_static_obstacle`, and passes it into the classifier. `assign_primary_arm` accepts this explicit boolean and only gates the terminal A0 branch.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-PSO-001` | `AC-PSO-001` exact five values | `profiles.py` | `test_pg_profiles.py` | Planned |
| `REQ-PSO-002` | `AC-PSO-002` static simple entry is A1 and tagged | `exporter.py`, `arms.py` | `test_scenario_arms.py` | Planned |
| `REQ-PSO-003` | `AC-PSO-003` static tag alone does not alter A2--A5 precedence | `arms.py` | `test_scenario_arms.py` | Planned |

## 9. Test strategy

| ID | Level | Behavior | Expected result | Requirement |
|---|---|---|---|---|
| `TEST-PSO-001` | Unit | sample every profile | exact configured probability | `REQ-PSO-001` |
| `TEST-PSO-002` | Unit | simple low-traffic features plus static obstacle | `A1_traffic` | `REQ-PSO-002` |
| `TEST-PSO-003` | Unit | topology/VRU/critical cases plus static obstacle | existing arm is preserved | `REQ-PSO-003` |

Commands: `uv run --no-sync python -m pytest -q tests/test_pg_profiles.py tests/test_scenario_arms.py`; `make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/scenarios/pg/profiles.py src/thesis_rl/scenarios/pg/exporter.py src/thesis_rl/scenarios/arms.py tests/test_pg_profiles.py tests/test_scenario_arms.py"`; `git diff --check`. A live PG-generation smoke is required before use of regenerated data.

## 10. Milestones

- [x] M1: Amend specification and ADR; add tests.
- [x] M2: Implement profile, realized metadata, and classifier wiring.
- [x] M3: Run safe available validation and reconcile the plan.

## 11. Progress and findings log

- 2026-07-29: User approved `P0=0.00`, `P1=0.03`, `P2=0.08`, `P3=0.08`, `P5=0.15`; static obstacles exclude only A0. Existing workspace changes are unrelated and preserved.
- 2026-07-29: Sandbox test execution is blocked because the local virtual environment lacks `pytest` and `pyarrow`; no dependencies were installed. Source/test syntax compilation and `git diff --check` remain the safe available checks.
- 2026-07-29: Compiled all five production modules and both focused test files from source with no bytecode writes; `git diff --check` passed. The focused pytest and Ruff commands remain unrun because their tools/dependencies are absent locally.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/scenarios/pg/profiles.py` | Modify | Approved probabilities and validation |
| `src/thesis_rl/scenarios/pg/generator.py` | Modify | Persist realized static-object metadata |
| `src/thesis_rl/scenarios/pg/exporter.py` | Modify | Tag/classify realized obstacles |
| `src/thesis_rl/scenarios/arms.py` | Modify | Exclude static obstacles from A0 |
| `tests/test_pg_profiles.py` | Modify | Profile-value regression |
| `tests/test_scenario_arms.py` | Modify | Arm-classification regressions |
| `docs/specifications/scenarionet_integration_v1.1_specification.md` | Modify | Approved contract amendment |
| `docs/decisions/ADR-034-pg-static-obstacle-generation-and-arm-classification.md` | Add | Approval record |

## 14. Validation results

| Command | Result | Date | Notes |
|---|---|---|---|
| Focused pytest | NOT_RUN | 2026-07-29 | Local virtual environment lacks `pytest`; run the listed command in the provisioned container |
| Focused Ruff format check | NOT_RUN | 2026-07-29 | `ruff` is not installed locally; run the listed command in the provisioned container |
| Source/test syntax compilation | PASS | 2026-07-29 | Seven changed Python files compiled from source without bytecode writes |
| `git diff --check` | PASS | 2026-07-29 | No whitespace errors |

## 15. Final reconciliation

`REQ-PSO-001` and `REQ-PSO-002` are implemented; their deterministic pytest
coverage is present but not executed locally because the environment is not
provisioned. `REQ-PSO-003` is implemented and regression-covered by the same
unrun focused suite. No dataset has been regenerated. Before experimental use,
run the focused suite and a live PG generation smoke to verify that the local
MetaDrive exporter retains generated static objects in replayed scenarios.
