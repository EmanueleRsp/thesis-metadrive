# Scalar Autonomous-Driving Pipeline Audit and Completion ExecPlan

## 1. Metadata

- Feature: repository-side audit, integration inventory, and completion gate for the scalar autonomous-driving pipeline
- Plan ID: `SCALAR-PIPELINE-AUDIT-2026-07-19`
- Authoritative specifications: `docs/specifications/scenarionet_integration_v1.1_specification.md` (v1.1), `automatic_curriculum_learning_v1_specification.md` (v1), `rulebook_v4.7_specification.md` (v4.7-final-implementation-complete), `rulebook_scalarization_v1.0_specification.md` (`SCAL-V1.0`, v1.0), `observation_v1.1_specification.md` (`OBS-V1.1`, v1.1), `encoder_v1.0_specification.md` (`ENC-V1.0`, v1.0), and `transition_replay_v1_specification.md` (`TRANSITION-REPLAY`, v1.0); all `AUTHORITATIVE` per `docs/project_index.md`.
- Status: `IN_PROGRESS`
- Audit result classification: `METADATA AUDIT COMPLETE — LIVE DATASET VALIDATION PENDING`
- Created / last updated: `2026-07-19`
- Related ADRs: ADR-001, ADR-002, ADR-003, ADR-004, ADR-005, ADR-006, ADR-007, ADR-008, ADR-009, ADR-010, ADR-011, ADR-012, ADR-014.
- Branch / owner: current worktree / thesis repository maintainer.

## 2. Objective And Scope

Produce a read-only audit of the frozen ScenarioNet selection, a proposed train-only 48-record golden-suite manifest, and a source/config/test conformance inventory for the complete scalar pipeline. Implement only defects already required by the authoritative specifications. The protected Waymo/PG source files and every `ScenarioDescription` are immutable inputs.

In scope: frozen-index reconstruction, immutable reference manifests, live read-only source checks, focused integration validation, and approval gates. Out of scope: dataset materialization or repair, dataset mutation, scientific selection of `n_steps`, an ACL semantic change, and full thesis experiments.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-AUDIT-001` | Audit the selected ScenarioNet population without changing source data; preserve source/split/arm, validation, route, signal, horizon, reproducibility, and leakage evidence. | ScenarioNet v1.1 §§3–17, 24, 27 |
| `REQ-AUDIT-002` | Create a reference-only, train-only golden-suite proposal with eight records per A0–A5 and proportional source allocation. | User milestone A2; ScenarioNet v1.1 §§6–8, 22 |
| `REQ-AUDIT-003` | Preserve causal Rulebook, scalarization, semantic observation, encoder, termination/truncation, and checkpoint contracts. | Rulebook v4.7 §§2–3, 11–15; SCAL-V1.0; OBS-V1.1; ENC-V1.0 |
| `REQ-AUDIT-004` | Keep transition replay v1 at exactly `n_steps in {1,3}` and the conformance default `3`; do not plan five-step behavior. | TRANSITION-REPLAY REQ-001–033, §§9, 11–13 |
| `REQ-AUDIT-005` | Reconcile ACL implementation with the approved ScenarioNet no-mutation, learning-potential-only core. | ACL v1 §28; ADR-014 |
| `REQ-AUDIT-006` | Exercise only supported validation and record unavailable required validation honestly. | All selected specifications; `AGENTS.md` |

## 4. Current Repository Analysis

- `VERIFIED`: `data/scenarionet/frozen/scenario_selection_index.json` contains schema `scenarionet_frozen_selection_v1`, 3,500 records, frozen artifact digests, split manifest, and source references. The protected runtime root is mounted in Docker, but its current train view matches the non-frozen catalog rather than the final frozen catalog.
- `VERIFIED`: `src/thesis_rl/scenarios/frozen.py` reconstructs a catalog from the frozen index and checks exact source/split counts, indices, eligible statuses, and, when mounted, source files.
- `VERIFIED`: `conf/agent/planner/algorithm/{td3_sb3,sac_sb3}.yaml` set the approved `n_steps: 3`; `src/thesis_rl/sb3_extensions/replay/config.py` rejects values outside `{1,3}`.
- `VERIFIED`: Scenario ACL production code is ScenarioNet-specific with six semantic arms, MAB, buffer/replay, staleness, persisted buffer/bandit/RNG state, and mutation rejection. `usefulness.py` records Rulebook criticality diagnostically while the selected core ranks by algorithm-specific learning potential.
- `VERIFIED`: ACL v1 §28 and ADR-014 govern the six-arm ScenarioNet core: mutation is prohibited and `ScenarioUsefulness.value` is learning-potential-only. §12 calculators are implemented for PPO, TD3, and SAC and wired through custom and SB3 planner backends; runtime learner execution remains blocked by live ScenarioNet/Rulebook wiring, not Torch availability.
- `VERIFIED`: the Docker GPU environment mounts `/workspace/data/scenarionet` read-only. It verified all 3,500 frozen references exist, read-only content checks on all 48 golden references, and raw zero-policy ScenarioEnv smoke for one PG and one Waymo reference. The mounted runtime train view is coherent with `catalog/scenario_catalog.parquet` but not with the frozen catalog (360 PG entries differ); this prevents a frozen-catalog learner claim. This is not full Rulebook v4.7, semantic-observation, or learner validation.

## 5. Assumptions And Invariants

- The frozen index is an immutable selection description, not a replacement for source-file validation.
- A Waymo `source_log_id` is provenance; per ADR-001 it is not a split group unless it proves a shared group. Original source scenario identity is the safe static leakage key available in the index.
- Golden-suite rows are references only; no file copy, data rewrite, or `ScenarioDescription` change is permitted.
- `terminated` and `truncated` remain separate. A bootstrappable truncation requires the pre-reset final observation.
- `n_steps=3` remains the approved core value. Supported values remain exactly `{1,3}`; five-step behavior is not planned.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-AUDIT-001` | Validation boundary | The protected root is mounted read-only in the Docker GPU container, but only reference existence/content and two raw ScenarioEnv paths have been exercised. | Run progressively broader read-only checks / claim full validation. | Continue focused checks, beginning with the scalar vertical path. | Blocks `VERIFIED` status; does not block safe focused checks. | Open |
| `DEC-AUDIT-002` | Specification clarification | ACL v1 §11 made Rulebook criticality the first component of final usefulness. | Retain ACL v1 formula / approve a narrow ScenarioNet-scope ACL v1 amendment. | Approved narrow amendment in ACL v1 §28 and ADR-014. | Usefulness, ranking, replacement, replay probability, MAB feedback, records, tests, and state migration. | Approved 2026-07-19 |

## 7. Proposed Design

Add a standard-library, read-only audit command that consumes only the frozen selection index and writes versioned output outside the protected dataset. It must emit (1) an audit report, (2) a catalog fingerprint, (3) a candidate coverage matrix, and (4) a reference-only golden-suite manifest. Selection is deterministic: train records are grouped by semantic arm, source quotas use largest-remainder rounding, then a documented metadata coverage score breaks ties. The report labels fields unavailable without a mounted source root and never treats metadata-only validation as a live simulation result.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests / evidence | Status |
|---|---|---|---|---|
| `REQ-AUDIT-001` | `AC-AUDIT-001` | audit command and generated report | frozen-index reconstruction command | In progress |
| `REQ-AUDIT-002` | `AC-AUDIT-002` | golden manifest and coverage matrix | deterministic re-run digest | In progress |
| `REQ-AUDIT-003` | `AC-AUDIT-003` | conformance inventory | source/config/test inspection | In progress |
| `REQ-AUDIT-004` | `AC-AUDIT-004` | reconciliation section | replay config and planner inspection | In progress |
| `REQ-AUDIT-005` | `AC-AUDIT-005` | ACL §28, usefulness calculators, planner backends, and config | `tests/test_scenario_acl_usefulness.py`, ACL config matrix, planner import smoke | IMPLEMENTED; runtime learner verification pending GPU |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-AUDIT-001` | command | Frozen audit reconstruction | repository frozen index | exact 3,500 records and reproducibility fingerprint | `REQ-AUDIT-001` |
| `TEST-AUDIT-002` | command | Golden candidate allocation | repository frozen index | exactly eight unique train references per A0–A5 | `REQ-AUDIT-002` |
| `TEST-AUDIT-003` | focused unit | Existing replay/ACL configuration contracts | existing test modules | no regression in currently executable environment | `REQ-AUDIT-004`, `REQ-AUDIT-005` |
| `TEST-AUDIT-004` | smoke | S0 raw ScenarioEnv | mounted immutable source root, final golden manifest, zero policy | one PG and one Waymo reset/step path | `REQ-AUDIT-003` |
| `TEST-AUDIT-005` | integration | vectorized source provider reset | mounted immutable source root | worker spawn, sampling, reset, route publishing | `REQ-AUDIT-003` |
| `TEST-AUDIT-006` | smoke | S0–S6 full scalar stages | mounted immutable source root and golden manifest | PG/Waymo path and stage diagnostics | `REQ-AUDIT-003` |

Commands use `docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev uv run --no-sync ...` for all GPU/container validation. `make rulebook-v2-check` and learner smoke commands remain follow-up validation, not substitutes for the documented S0–S6 stages.

## 10. Milestones

- [x] A1: Locate frozen selection and perform static immutable-index audit.
- [ ] A1: Complete live Rulebook validation and exclusion-input verification; dataset-wide source loadability, route, horizon, and trajectory structural checks have passed.
- [x] A2: Generate deterministic candidate 48-reference suite and coverage matrix from actual frozen catalog metadata.
- [x] A2: Inspect all 48 candidate `ScenarioDescription` files and finalize the reference-only suite.
- [x] A3: Produce source/config/test conformance inventory and amendment surface.
- [x] A3: Implement ACL §12 algorithm-specific learning-potential formulas and propagate the result through custom/SB3 planner backends.
- [x] A4: Fix the verified vectorized reset route-metadata defect and add a regression.
- [ ] A4/A5: Complete the scalar S0 vertical path and staged PG/Waymo learner smoke. Current blockers are the catalog/runtime mismatch for the frozen catalog, missing live `rulebook_v2_adapter`, and MetaDrive scenario-cache reset invariant.

## 11. Progress And Findings Log

- 2026-07-19: Confirmed all selected specifications are authoritative from the index and read the required plans/ADRs. The frozen index contains exact split targets but the runtime root is absent.
- 2026-07-19: Static audit found exact source totals (Waymo/PG 1,000/1,000 train, 250/250 validation, 500/500 test), no duplicate UIDs or scenario IDs, all selected metadata statuses valid, and no empty assigned-route lists.
- 2026-07-19: Authoritative-text reconciliation established that ScenarioNet mutation is already disabled and ACL v1 §23.3 permits no-mutation replay. The user approved ACL v1 §28 and ADR-014: mutation remains off and usefulness is learning-potential-only; ACL v1 §12 learning-potential formulas remain an existing implementation task.
- 2026-07-19: The user confirmed that transition replay v1 is final: `n_steps=3` is the project core, supported values remain `{1,3}`, and no five-step work, comparison, or v1.1 amendment is planned.
- 2026-07-19: Added `scripts/audit_frozen_scenarionet.py` and generated only external compact audit/reference artifacts under `docs/audits/scalar_pipeline_audit_2026-07-19/`. The command does not write to source roots and refuses to overwrite output.
- 2026-07-19: The candidate selection is exactly 48 unique train records (8 per arm). Actual train source composition is one-sided in every arm, so proportional largest-remainder allocation is PG-only for A0–A2 and Waymo-only for A3–A5.
- 2026-07-19: Added a static component inventory and authoritative ACL reconciliation table. They identify incomplete PER verification, the narrow ACL usefulness amendment, and the final transition-replay v1 boundary.
- 2026-07-19: The corrected ARM64 Docker GPU environment reports `aarch64`, `torch 2.9.1+cu128`, CUDA available, and a GH200 GPU. A later container probe reports 10.84 GiB free; this permits short diagnostics but is not a scientific capacity claim.
- 2026-07-19: `verify_frozen_sources` found all 3,500 mounted immutable references. `scripts/finalize_golden_suite.py` content-validated all 48 final references (24 PG, 24 Waymo) without writing to the source root.
- 2026-07-19: `scripts/validate_frozen_scenarionet_content.py` opened all 3,500 sources read-only and passed loadability, horizon, SDC trajectory, and assigned-route map-membership checks. Its first attempt treated NumPy-backed arrays as non-sequences; the checker was corrected and regression-tested before the final pass. No dataset defect was found.
- 2026-07-19: Raw zero-policy ScenarioEnv S0 smoke passed for PG `PGMap-1920034` (runtime index 169) and Waymo `632ee424a8a6f73b` (runtime index 1358), ten steps each, with observations `(161,)`, actions `(2,)`, and neither termination nor truncation.
- 2026-07-19: Vectorized ScenarioEnv tests exposed `Config.pop()` incompatibility in `_publish_assigned_route_metadata`. Replaced the dict-only default-argument call with membership-guarded single-argument removal and added a regression for MetaDrive-style `Config.pop(key)`.
- 2026-07-19: The user approved the narrow ScenarioNet ACL amendment. ACL v1 §28 and ADR-014 now make mutation prohibited and usefulness learning-potential-only; configuration rejects `use_rule_criticality=true` and its focused ACL regression matrix passes.
- 2026-07-19: Implemented ACL §12 learning-potential calculators: PPO positive GAE-style residuals, TD3 absolute TD residuals, and SAC entropy-aware absolute TD residuals. Removed critic-loss-only fallback; missing algorithm-specific inputs fail closed. Custom and SB3 planner backends propagate `learning_potential` through lifecycle and Agent chunk summaries.
- 2026-07-19: ACL/replay matrix passed (`50` tests on first run, then `47` after the focused rerun); planner import and py_compile smoke passed. The five stale Hydra encoder alias assertions were aligned to canonical `latent_query_v2`; the focused Hydra matrix now passes `32/32`.
- 2026-07-19: Standard GPU smoke (`presets/test/smoke_train`) completed 2,000 TD3-SB3 steps, evaluations, and final checkpoint; `learning_starts=10000` yielded zero learner updates, so this is an environment/checkpoint smoke only.
- 2026-07-19: ScenarioNet learner attempts were not promoted to S1: the frozen catalog differs from mounted runtime train by 360 PG records, Rulebook v4.7 fails closed because `ThesisScenarioEnv` lacks `rulebook_v2_adapter`, and a legacy v1 attempt reaches MetaDrive's scenario-cache invariant. No protected file was changed.
- 2026-07-19: A candidate vector-worker cardinality fix was tested but reverted: setting one loaded scenario per worker collapses MetaDrive's global seed modulo and produces catalog/runtime identity mismatches. The original vectorized path remains blocked by the ScenarioDataManager multi-scenario invariant; no unverified workaround was retained.

## 12. Deviations

No approved deviations. The inaccessible protected source root prevents live verification only; it does not authorize a fallback dataset or a change in effective dataset.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `docs/implementation/scalar_autonomous_driving_pipeline_audit_exec_plan.md` | Add | Living audit and approval record |
| `docs/project_index.md` | Modified | Register the cross-cutting audit ExecPlan and its evidence status |
| `docs/specifications/automatic_curriculum_learning_v1_specification.md` | Modified | Add approved ScenarioNet scalar ACL amendment (§28) |
| `docs/decisions/ADR-014-scenarionet-acl-learning-potential-only.md` | Added | Approval record for ScenarioNet ACL usefulness and mutation boundary |
| `src/thesis_rl/curriculum/config.py` | Modified | Reject Rulebook-criticality usefulness configuration |
| `tests/test_scenario_acl_config.py` | Modified | Regression for rejected Rulebook-criticality usefulness configuration |
| `src/thesis_rl/curriculum/scenario_acl/usefulness.py` | Modified | ACL §12 PPO/TD3/SAC formulas and fail-closed dispatch |
| `src/thesis_rl/curriculum/scenario_acl/__init__.py` | Modified | Export algorithm-specific learning-potential APIs |
| `src/thesis_rl/agent/planners/core/lifecycle.py` | Modified | Propagate learning potential from backend updates |
| `src/thesis_rl/agent/agent.py` | Modified | Include learning potential in episode/chunk summaries |
| `src/thesis_rl/agent/planners/algorithms/{ppo,td3,sac}.py` | Modified | Compute ACL potential in custom backends |
| `src/thesis_rl/agent/planners/algorithms/{ppo_sb3,td3_sb3,sac_sb3}.py` | Modified | Compute ACL potential in SB3 backends |
| `tests/test_scenario_acl_usefulness.py` | Modified | Formula, entropy, terminal, and proxy-rejection tests |
| `tests/test_planner_lifecycle.py` | Modified | Regression that lifecycle exposes finite algorithm-specific learning potential |
| `conf/agent/planner/algorithm/{ppo,ppo_sb3}.yaml` | Modified | Freeze ACL learning-potential gamma/lambda defaults |
| `scripts/audit_frozen_scenarionet.py` | Added | Read-only frozen-index audit and candidate-manifest generator |
| `scripts/finalize_golden_suite.py` | Added | Read-only content validation and final reference manifest writer |
| `scripts/validate_frozen_scenarionet_content.py` | Added | Read-only all-record loadability, trajectory, and route validator |
| `src/thesis_rl/envs/thesis_scenario_env.py` | Modified | Config-compatible route metadata publication during reset |
| `tests/test_thesis_scenario_env.py` | Modified | Regression for MetaDrive-style single-argument `Config.pop` |
| `tests/test_hydra_preset_test_configs.py` | Modified | Canonical encoder alias formatting and `latent_query_v2` assertion |
| `tests/test_hydra_agent_presets.py` | Modified | Canonical `latent_query_v2` assertions for LQ agent presets |
| `tests/test_hydra_preset_run_configs.py` | Modified | Canonical `latent_query_v2` assertion for ScenarioNet ACL composition |
| `tests/test_validate_frozen_scenarionet_content.py` | Added | Regression for NumPy-like state arrays in the dataset validator |
| `docs/audits/scalar_pipeline_audit_2026-07-19/audit_report.md` | Added | Versioned static audit report outside protected data |
| `docs/audits/scalar_pipeline_audit_2026-07-19/golden_suite_candidate_manifest.json` | Added | 48 reference-only train candidate manifest |
| `docs/audits/scalar_pipeline_audit_2026-07-19/golden_suite_candidate_coverage.csv` | Added | Candidate coverage matrix |
| `docs/audits/scalar_pipeline_audit_2026-07-19/golden_suite_content_validated/golden_suite_manifest.json` | Added | Final 48-reference manifest after read-only content checks |
| `docs/audits/scalar_pipeline_audit_2026-07-19/golden_suite_content_validated/golden_suite_content_evidence.csv` | Added | Per-reference content-validation evidence |
| `docs/audits/scalar_pipeline_audit_2026-07-19/live_content_validation_final/live_content_validation_report.md` | Added | Dataset-wide read-only structural validation report |
| `docs/audits/scalar_pipeline_audit_2026-07-19/live_content_validation_final/live_content_validation_issues.csv` | Added | Empty issue record for the passing live structural validation |
| `docs/audits/scalar_pipeline_audit_2026-07-19/component_conformance_inventory.md` | Added | Component matrix and required amendments |
| `docs/audits/scalar_pipeline_audit_2026-07-19/acl_authoritative_reconciliation.md` | Added | Exact ACL v1 clause reconciliation |
| `docs/audits/scalar_pipeline_audit_2026-07-19/execution_environment_requirements.md` | Added | Observed ARM64/GPU capacity and container-validation requirements |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| static frozen-index reconstruction | PASS | 2026-07-19 | 3,500 records; fingerprint recorded in audit output |
| `.venv/bin/python -m py_compile scripts/audit_frozen_scenarionet.py` | PASS | 2026-07-19 | Script syntax compiled successfully. |
| `.venv/bin/python scripts/audit_frozen_scenarionet.py --index ... --output-dir docs/audits/scalar_pipeline_audit_2026-07-19` | PASS | 2026-07-19 | Wrote report, 48-reference manifest, and coverage CSV without touching protected data. |
| deterministic re-run in `/tmp` and `cmp` of all three artifacts | PASS | 2026-07-19 | Report, candidate manifest, and coverage CSV were byte-identical. |
| `docker compose ... pytest -q tests/test_transition_replay_config.py ... tests/test_scenario_acl_usefulness.py` | PASS | 2026-07-19 | 46 passed; approved `{1,3}` transition and ACL focused contracts. |
| `docker compose ... pytest -q tests/test_scenario_frozen.py ... tests/test_checkpointing.py` | PASS | 2026-07-19 | 93 passed; static ScenarioNet, scalarization, observation, encoder, and checkpoint components. |
| `docker compose ... verify_frozen_sources(...)` | PASS | 2026-07-19 | All 3,500 immutable frozen references exist under `/workspace/data/scenarionet`. |
| `docker compose ... python scripts/finalize_golden_suite.py ...` | PASS | 2026-07-19 | Finalized 48 references after read-only content checks; 24 PG and 24 Waymo. |
| `docker compose ... python scripts/validate_frozen_scenarionet_content.py ...` | PASS | 2026-07-19 | 3,500/3,500 passed read-only source loadability, horizon, SDC trajectory, and assigned-route membership checks. |
| `docker compose ... python -m thesis_rl.cli.scenarios.smoke ... --scenario-index 169/1358 --steps 10 --policy zero` | PASS | 2026-07-19 | Raw ScenarioEnv S0 partial smoke passed once per source. |
| `docker compose ... pytest -q tests/test_thesis_scenario_env.py tests/test_scenarionet_vectorized_integration.py` | PASS | 2026-07-19 | 25 tests exit 0 after the route-metadata regression fix. |
| `docker compose ... pytest -q tests/test_scenario_acl_config.py tests/test_scenario_acl_usefulness.py tests/test_scenario_acl_buffer.py tests/test_scenario_acl_mab.py` | PASS | 2026-07-19 | 26 passed after ACL v1 §28 / ADR-014 configuration enforcement. |
| `docker compose ... pytest -q tests/test_scenario_acl_usefulness.py ... tests/test_transition_boundary.py` | PASS | 2026-07-19 | 50 passed; ACL §12 formulas, ACL state/replay, and approved transition replay boundaries. |
| `docker compose ... ruff check ... planner/usefulness files` | PASS | 2026-07-19 | All modified ACL/planner files pass focused Ruff. |
| `docker compose ... pytest -q tests/test_planner_lifecycle.py tests/test_hydra_agent_presets.py tests/test_hydra_preset_run_configs.py tests/test_hydra_preset_test_configs.py tests/test_checkpointing.py` | PASS | 2026-07-19 | Initial run: 48 passed, 5 stale Hydra alias failures; after canonical `latent_query_v2` alignment, the complete focused lifecycle/Hydra/checkpoint matrix passes 44/44. |
| `docker compose ... pytest -q tests/test_scenario_frozen.py ... tests/test_agent_pipeline.py` | PASS | 2026-07-19 | 102 passed across ScenarioNet, scalarization, observation, encoder, checkpoint, and Agent regressions. |
| `docker compose ... py_compile ... && python -c 'import ... PPO/TD3/SAC SB3 backends'` | PASS | 2026-07-19 | Syntax and planner import smoke passed without learner/GPU initialization. |
| `docker compose ... ruff check ... && ruff format --check ...` | PASS | 2026-07-19 | Modified source, test, and new scripts pass focused Ruff checks. |
| `docker compose ... python -m thesis_rl.cli.train --config-name presets/test/smoke_train` | PASS | 2026-07-19 | GPU standard smoke completed 2,000 steps, two evaluations, and final checkpoint; no updates because `learning_starts=10000`. |
| `docker compose ... pytest -q tests/test_rulebook_v2_contracts.py tests/test_rulebook_v2_wrapper.py tests/test_rulebook_v2_monitor.py tests/test_rulebook_v2_causal_context.py tests/test_rulebook_v2_live_adapter.py tests/test_rulebook_v2_causality.py` | PASS | 2026-07-19 | 41 focused Rulebook v4.7 contract/live-adapter tests pass; this does not prove `ThesisScenarioEnv` live adapter wiring. |
| ScenarioNet S1 attempt with frozen catalog | BLOCKED | 2026-07-19 | Read-only catalog/runtime mismatch: 360 PG train references differ; no repair executed. |
| ScenarioNet S1 attempt with Rulebook v4.7 | BLOCKED | 2026-07-19 | Wiring fails closed: `ThesisScenarioEnv` does not expose the required live `rulebook_v2_adapter`. |
| legacy Rulebook-v1 learner attempt | BLOCKED | 2026-07-19 | MetaDrive `ScenarioDataManager` raises `It seems you access multiple scenarios in one episode`; no scientific result claimed. |

## 15. Final Reconciliation

Pending. `IMPLEMENTED` and `VERIFIED` must remain distinct until the audit artifacts, focused checks, mounted-data validation, and final source/config/test reconciliation are complete.
