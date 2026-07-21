# ExecPlan: Rulebook synthetic scenario descriptors v1

## 1. Metadata

- Feature: persistent deterministic ScenarioDescription fixtures for Rulebook v4.7 conformance.
- Plan ID: `RB-SYNTH-SCENARIOS-V1`.
- Authoritative specification: `docs/specifications/rulebook_v4.7_specification.md`, version `4.7-final-implementation-complete`, approved and authoritative.
- Status: `IN_PROGRESS`.
- Created / last updated: 2026-07-21.
- Related ADR: `ADR-003-causal-ctrv-conflict-zone-prediction.md` applies to vehicle conflict scenarios.
- Approval evidence: user explicitly approved the initial three-scenario feasibility spike in the Codex conversation on 2026-07-21.

## 2. Objective and scope

Create a repository-owned deterministic conformance-fixture foundation using real MetaDrive `ScenarioEnv` state. The initial milestone is limited to generated and persisted `ScenarioDescription` fixtures for red light, crosswalk/pedestrian, and vehicle/pedestrian collision.

In scope: a generator, persisted descriptors and manifest, strict descriptor validation, headless reset/step smoke tests, and a Rulebook-v4.7 adapter/evaluator integration check. These fixtures are test-only and do not enter ScenarioNet training, curriculum pools, dataset splits, or experimental reporting.

Out of scope: Waymo acquisition or redistribution; changing Rulebook formulas, thresholds, controls, ScenarioNet eligibility policy, public runtime configuration, or the golden suite. Complete subrule coverage follows only after the spike is verified.

## 3. Authoritative requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-001` | Rulebook evaluation uses current live `ScenarioEnv` state and map/control geometry, never future track data. | §§1.1–1.2, 11 |
| `REQ-002` | Scenarios must be validated before use; missing required inputs fail fast and are never converted to zero cost. | §2.3, §15.11 |
| `REQ-003` | R1 collision evaluation distinguishes participant class, uses transition semantics, and retains finite bounded outputs. | §§5.2–5.4, §15.1 |
| `REQ-004` | R3 red-light and crosswalk controls use canonical geometry, transition state, and causal memory. | §§7.6, 7.8, §§15.6, 15.8 |
| `REQ-005` | Conformance tests are deterministic and preserve frozen Rulebook/CTRV behavior. | §§2.5, 3.1–3.2, 15; ADR-003 |

## 4. Current repository analysis

- `VERIFIED`: `ScenarioDescription.sanity_check()` requires metadata, tracks, dynamic states, map features, fixed-length arrays, position, and heading (`third_party/metadrive/metadrive/scenario/scenario_description.py`).
- `VERIFIED`: the local `ScenarioEnv` replays `PEDESTRIAN` tracks and materializes crosswalk polygons and dynamic traffic lights (`scenario_traffic_manager.py`, `scenario_block.py`, `scenario_light_manager.py`).
- `VERIFIED`: Rulebook static adapters are source-specific PG/Waymo; `ThesisScenarioEnv` selects them by record source (`src/thesis_rl/envs/thesis_scenario_env.py`). The Waymo static adapter accepts the MetaDrive descriptor schema and uses persisted assigned-route metadata, so it can be used as an isolated test-only normalization boundary without claiming Waymo provenance.
- `VERIFIED`: `ScenarioOnlineEnv.set_scenario()` accepts an in-memory descriptor after `ScenarioDescription.sanity_check()`. This avoids creating a test dataset catalog solely to prove live descriptor loading (`third_party/metadrive/metadrive/manager/scenario_data_manager.py`).
- `VERIFIED`: R1/R3 unit coverage and live-adapter tests already exist under `tests/test_rulebook_v2_*.py`; `test_forced_rule_scenarios.py` is historical Rulebook v1 debug coverage, not v4.7 conformance coverage.
- `VERIFIED`: `tests/fixtures/` exists but has no descriptor-fixture hierarchy.

## 5. Assumptions and invariants

- Coordinates are MetaDrive planar metres, timestamps are fixed, and no evaluation reads frames after its declared transition.
- Descriptors contain complete fixed-length NumPy arrays and pass `ScenarioDescription.sanity_check()` before persistence.
- A manifest records fixture ID, generator version, seed, target transition, expected statuses, and numeric tolerances.
- Tests do not teleport actors after reset; the evaluated transition comes from the live environment.
- Fixtures remain outside all training and ScenarioNet data paths.

## 6. Decisions and approval gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-001` | implementation detail | Persistence format | generator only / generator plus small generated pickles | Commit generator, small generated `.pkl` fixtures, and manifest | Reproducibility | Approved by user scope approval 2026-07-21 |
| `DEC-002` | implementation detail | Source adaptation in test-only path | change runtime policy / isolated adapter or harness | Use `ScenarioOnlineEnv` for live loading and call the Waymo static adapter directly only as schema-compatible test normalization; do not broaden production dataset policy or assert Waymo provenance | Compatibility | Decided 2026-07-21 |
| `DEC-003` | specification clarification | Exact continuous R3 targets | status only / analytic numeric oracle | Freeze formula-derived expectation with tolerance; request approval only if ambiguity is found | Test oracle | Pending feasibility evidence |
| `DEC-004` | source contract | The production transition evaluator passed fixed empty inputs to `vehicle_yield`; priority and roundabout facts cannot be inferred from geometry. | implement only occupancy / extend source-bound records for all predicates / leave rule `NOT_APPLICABLE` | Derive movements, zones, occupancy, and STOP-vs-NONE live; require explicit validated metadata for pairwise and roundabout priority, with no fallback. | New source-data contract and ScenarioNet eligibility implications | Approved by user 2026-07-21; ADR-023 |

## 7. Proposed design

Add `tests/fixtures/rulebook_scenarios/` with binary descriptors, JSON manifest, and a short English README. A typed test-support generator builds all arrays and map features from declarative definitions. A harness loads each descriptor through `ScenarioOnlineEnv`, runs schema validation, and evaluates only the declared transition through the existing Rulebook-v4.7 boundary. The harness calls the existing Waymo static adapter directly as a schema-compatible normalization test, without assigning the descriptor a Waymo source. It compares current/pre-state outputs with the manifest oracle.

No Waymo asset is copied. Future real-data integration fixtures may refer to permitted local assets by UID/hash, separately from this suite.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-001` | `AC-001`, `AC-002` | planned generator/harness | planned fixture smoke/integration tests | Planned |
| `REQ-002` | `AC-001`, `AC-003` | planned descriptor validator | planned validation tests | Planned |
| `REQ-003` | `AC-004` | planned collision fixture | planned R1 live fixture test | Planned |
| `REQ-004` | `AC-005`, `AC-006` | planned control fixtures | planned R3 live fixture tests | Planned |
| `REQ-005` | `AC-002`, `AC-007` | planned manifest/harness | planned determinism test | Planned |

## 9. Test strategy defined before implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `AC-001` / `TEST-001` | descriptor | Schema validity | all three descriptors | `ScenarioDescription.sanity_check()` succeeds | `REQ-002` |
| `AC-002` / `TEST-002` | integration | Headless loading | all descriptors | reset and declared steps succeed with finite outputs | `REQ-001`, `REQ-005` |
| `AC-003` / `TEST-003` | integration | Rulebook inputs | all descriptors | complete inputs or named fail-fast error; never silent zero | `REQ-002` |
| `AC-004` / `TEST-004` | integration | R1 VRU collision | collision descriptor | applicable/evaluable component matches manifest outcome | `REQ-003` |
| `AC-005` / `TEST-005` | integration | R3 red control | red-light descriptor | status and bounded cost match manifest | `REQ-004` |
| `AC-006` / `TEST-006` | integration | R3 crosswalk yield | crosswalk descriptor | component/memory result matches manifest | `REQ-004` |
| `AC-007` / `TEST-007` | regression | Reproducibility | each fixture twice | identical declared Rulebook output/status | `REQ-005` |

Planned commands: focused pytest for the fixture test file; focused Ruff lint and format-check via `PYTHON_QUALITY_PATHS`; `git diff --check`. No global type-check command is configured. A representative headless `ScenarioEnv` smoke is mandatory.

## 10. Milestones

- [x] M1 — inspect source-adapter boundary and select test-only fixture API. `ScenarioOnlineEnv` loads descriptors in memory; the existing Waymo static adapter provides schema-compatible static normalization without a runtime source-policy change.
- [ ] M2 — implement and persist descriptors plus manifest; run schema and headless smoke checks. The initial three descriptors, generated manifest, schema validation, static normalization, and one real `ScenarioOnlineEnv` reset/step are implemented; live Rulebook-v4.7 transition assertions remain.
- [ ] M3 — connect Rulebook v4.7 and establish formula-backed expected outcomes; resolve `DEC-003` if needed. All initial descriptors now capture live pre/post snapshots, normalize static geometry, and evaluate a complete v4.7 transition with the mandated calibrated-braking input. The crosswalk/pedestrian case is applicable and has a positive deterministic cost; formula-specific oracle values and remaining families are pending.
- [ ] M4 — run regressions/quality checks, review diff, and reconcile this plan.

### Planned full-suite coverage after the feasibility spike

The three initial descriptors are a loading and integration gate, not complete
Rulebook coverage. The full suite will be organized as reusable scenario
families. Every applicable normative subrule receives satisfied, violated,
boundary, and `NOT_APPLICABLE` coverage; invalid descriptor cases verify the
specified fail-fast behavior.

| Family | Live descriptor cases | Rulebook behavior validated |
|---|---|---|
| Foundation | straight route; multilayer crossing; invalid route/lane/control variants | route projection, 2.5D compatibility, eligibility, reset state, transactional fail-fast |
| R1 collision | vehicle front/rear/lateral; tangent; persistent/recontact; static object; pedestrian; cyclist; invalid initial overlap | contact-onset lifecycle, pre-state normal/velocity, class-aware cap/floor, maximum aggregation |
| R2 RSS | front vehicle with safe/boundary/unsafe gap; adjacent/rear/opposite/ambiguous lane | canonical association, bumper gap, RSS applicability and cost |
| R2 TTC and clearance | parallel/crossing/overlap; vehicle/VRU/static; in/out prediction horizon | CV TTC, candidate radius, clearance, class coverage |
| R3 road | partial/full off-road; stationary/reverse wrong-way; solid crossing; dashed crossing and timer durations | drivable area, signed route speed, geometric crossing, temporal memory |
| R3 signal | green; red safe/eroded/crossing; yellow must-stop/cannot-stop; phase-change crossings; unrelated/unknown signal | active control selection, pre/post signal semantics, frozen yellow obligation, resolved-state lifecycle |
| R3 stop | no/rolling/partial/full stop; separated stops; late/far stop; control-line deadband/crossing | canonical line derivation, dwell timer/best dwell, deficit cost, resolved lifecycle |
| R3 crosswalk | pedestrian/cyclist; safe/overlap/open-end gap; stationary/out-of-zone/overpass VRU; legal/illegal/preexisting entry and exit | canonical zone, causal intervals, candidate filters, commitment, illegal-entry memory |
| R3 vehicle yield | occupied zone; stop-vs-uncontrolled; roundabout; explicit merge priority; ambiguous and dual-stop cases | the four scoped priority predicates, `NOT_APPLICABLE` boundaries, zone-memory lifecycle |
| R4 progress | forward/stop/reverse/clipped progress; segment boundary; self-intersecting route | raw/normalized progress and continuity tie-break |
| Cross-cutting | identical repeat; PG/Waymo-compatible canonical geometry; missing runtime core input | determinism, source normalization, no-future guard, finite bounded outputs |

Unit-level conformance tests remain mandatory for fine numerical and fault paths
that a simulator descriptor cannot isolate reliably: canonicalization at 1 mm,
concave decomposition, exact continuous-SAT interval variants, duplicate delta
writes, cache conflict/rollback, callback without manifold, and all explicit
`RulebookEvaluationError` paths. These are complementary to—not substitutes
for—the live descriptor cases.

## 11. Progress and findings log

- 2026-07-21: user approved the three-fixture feasibility spike. Verified local MetaDrive support for pedestrian tracks, crosswalk polygons, stop-sign feature typing, and dynamic traffic lights. No production behavior changed.
- 2026-07-21: selected `ScenarioOnlineEnv.set_scenario()` as the live descriptor path. The current Waymo static adapter can normalize these descriptors when they carry frozen assigned-route metadata; it is used only for test normalization and does not label fixtures as Waymo.
- 2026-07-21: mapped the complete post-spike scenario families to the mandatory Rulebook v4.7 matrix (§§15.0–15.11). Unit-level conformance remains necessary for numerical and transactional fault paths that cannot be causally isolated in a live simulator rollout.
- 2026-07-21: implemented the initial three persistent descriptors and their generator. `PYTHONPATH=src pytest -q tests/test_rulebook_synthetic_scenarios.py` passed (5 tests): schema, static normalization, generation/reload, checked-in consistency, and headless reset/step. MetaDrive requires lane polygons in XY and accesses `metadata.dataset`; both are generator invariants.
- 2026-07-21: completed the first live Rulebook transition integration: a fixed-red synthetic descriptor is loaded by `ScenarioOnlineEnv`, captured by the production live snapshot adapters, evaluated with the production transition factory, and returns a complete evaluable signal component. The test uses a local explicit `RSSCalibrationArtifact` because the approved signal formula requires calibrated braking; it does not create or alter a calibration artifact.
- 2026-07-21: extended live loading and Rulebook-transition coverage to all initial descriptors. The crosswalk/pedestrian transition is applicable and yields a strictly positive crosswalk cost; the collision descriptor currently verifies the no-onset baseline before physical contact-specific scenarios are added.
- 2026-07-21: installed the production MetaDrive collision callback recorder in the synthetic vehicle/pedestrian fixture. Over a live rollout, its onset records produce a strictly positive R1 collision cost through the production transition evaluator. The test advances cache and memory transactionally between steps and does not teleport either actor.
- 2026-07-21: added and persisted `rss_front_vehicle`, a same-lane slower front-vehicle descriptor. Its live Rulebook transition makes RSS, TTC, and clearance applicable, providing the first integrated R2 family fixture.
- 2026-07-21: added and persisted `stop_sign` and `wrong_way`. The stop feature is converted by the static adapter into the canonical Rulebook control line and is applicable in a live transition; the reversed-reference fixture makes the `wrongway` component applicable. The existing red-light route also validates the live R4 progress component.
- 2026-07-21: added and persisted `offroad`, whose ego reference is outside the only canonical route-lane polygon. Its live R3 `offroad` component is applicable with a positive geometric cost, independently of native MetaDrive flags.
- 2026-07-21: added and persisted `vehicle_cyclist_collision`, derived from the initially separated pedestrian collision geometry but with the live actor typed as `CYCLIST`. The production contact recorder and R1 evaluator report a positive live onset for both VRU classes.
- 2026-07-21: added and persisted `solid_line` and `dashed_line`. The solid marking intersects the ego canonical footprint and produces positive live R3 cost; the dashed marking is live-evaluable for the timer lifecycle without treating a sub-threshold dwell as a violation.
- 2026-07-21: added and persisted `rss_rear_vehicle` and `unknown_signal`. The rear vehicle is explicitly excluded from the RSS front-vehicle domain; the relevant UNKNOWN signal is schema-loadable but rejected by the static Rulebook validation with the specified named error instead of being assigned zero cost.
- 2026-07-21: audit found that `evaluate_transition()` builds `vehicle_input` with a fixed empty interval, no prioritized actors, and no entered actors (`src/thesis_rl/rulebook/v2/transition.py`). Consequently live vehicle-yield can never be applicable. Existing static adapters emit an empty `movement_priority_records` tuple and do not retain roundabout/priority metadata. This is a production conformance gap, not a fixture gap.
- 2026-07-21: the first full Docker Rulebook regression exposed a NumPy pickle compatibility failure: host-generated fixture pickles referenced `numpy._core.numeric`, unavailable in the pinned container. Fixtures were regenerated with the primary container interpreter; documentation now requires that command for committed artifacts. Full regression then passed.
- 2026-07-21: implemented live vehicle-yield integration. The transition now derives unambiguous lane movements, 2.5D conflict zones, occupancy intervals, STOP-vs-NONE priority, and lazy cache records from causal snapshots. Explicit pairwise and roundabout priority remain source metadata only. Unit/integration tests exercise all four §7.9 predicates; `vehicle_yield_pairwise` is a persisted MetaDrive descriptor with a live pairwise-priority transition.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `docs/implementation/rulebook_synthetic_scenario_descriptors_v1_exec_plan.md` | Added | Living implementation record |
| `tests/fixtures/rulebook_scenarios/` | Added | Generated descriptors, manifest, and usage instructions |
| `tests/rulebook_scenario_fixtures.py` | Added | Declarative fixture generator |
| `tests/generate_rulebook_scenarios.py` | Added | Reproducible artifact generator |
| `tests/test_rulebook_synthetic_scenarios.py` | Added | Schema, persistence, normalization, and live-load tests |

## 14. Validation results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `git status --short` | PASS | 2026-07-21 | Existing unrelated changes found and will be preserved. |
| `PYTHONPATH=src pytest -q tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-21 | 5 passed. |
| `PYTHONPATH=src pytest -q tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-21 | 6 passed after the live Rulebook transition test was added. |
| `PYTHONPATH=src pytest -q tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-21 | 10 passed after all initial descriptors received live loading and transition coverage. |
| `PYTHONPATH=src pytest -q tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-21 | 11 passed after live R1 pedestrian-collision onset coverage. |
| `PYTHONPATH=src pytest -q tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-21 | 15 passed after the R2 front-vehicle descriptor and component checks. |
| `PYTHONPATH=src pytest -q tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-21 | 20 passed after R3 stop/wrong-way and R4 progress live-component coverage. |
| `PYTHONPATH=src pytest -q tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-21 | 22 passed after R3 geometric off-road coverage. |
| `PYTHONPATH=src pytest -q tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-21 | 26 passed after yellow-signal and cyclist-collision descriptors. |
| `PYTHONPATH=src pytest -q tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-21 | 30 passed after solid/dashed road-marking descriptors. |
| `PYTHONPATH=src pytest -q tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-21 | 33 passed after rear-RSS and UNKNOWN-signal fail-fast coverage. |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_contracts.py tests/test_rulebook_v2_transition.py tests/test_rulebook_v2_vehicle_yield.py tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-21 | 74 passed after live vehicle-yield wiring, source-contract validation, and four persisted scoped-predicate fixtures. |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_*.py tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-21 | 224 passed after container-regenerated fixtures. |
| `make format-check PYTHON_QUALITY_PATHS="tests/rulebook_scenario_fixtures.py tests/generate_rulebook_scenarios.py tests/test_rulebook_synthetic_scenarios.py"` | PASS | 2026-07-21 | 3 files already formatted. |
| `make lint PYTHON_QUALITY_PATHS="tests/rulebook_scenario_fixtures.py tests/generate_rulebook_scenarios.py tests/test_rulebook_synthetic_scenarios.py"` | PASS | 2026-07-21 | Ruff reported all checks passed. |
| `git diff --check` | PASS | 2026-07-21 | No whitespace errors. |

## 15. Final reconciliation

Vehicle-yield transition wiring is reconciled: each §7.9 predicate has a deterministic live-transition test, and each predicate has a persistent descriptor fixture. Explicit pairwise and roundabout facts remain source-bound metadata under ADR-023; unannotated ambiguous contexts remain `NOT_APPLICABLE`. The broader fixture plan remains a living record for its already-listed optional matrix expansions.
