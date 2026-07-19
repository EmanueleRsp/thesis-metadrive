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

In scope: frozen-index reconstruction, immutable reference manifests, live read-only source checks, focused integration validation, and approval gates. Out of scope: dataset materialization or repair, dataset mutation, scientific selection of `n_steps`, any ACL semantic change beyond approved §28/ADR-014, and full thesis experiments.

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

- `VERIFIED`: `data/scenarionet/frozen/scenario_selection_index.json` contains schema `scenarionet_frozen_selection_v1`, 3,500 records, frozen artifact digests, split manifest, and source references. It is the sole versioned selection authority for rebuilding the canonical catalog and runtime artifacts.
- `VERIFIED`: `src/thesis_rl/scenarios/frozen.py` reconstructs a catalog from the frozen index and checks exact source/split counts, indices, eligible statuses, and, when mounted, source files.
- `VERIFIED`: `conf/agent/planner/algorithm/{td3_sb3,sac_sb3}.yaml` set the approved `n_steps: 3`; `src/thesis_rl/sb3_extensions/replay/config.py` rejects values outside `{1,3}`.
- `VERIFIED`: Scenario ACL production code is ScenarioNet-specific with six semantic arms, MAB, buffer/replay, staleness, persisted buffer/bandit/RNG state, and mutation rejection. `usefulness.py` records Rulebook criticality diagnostically while the selected core ranks by algorithm-specific learning potential.
- `VERIFIED`: ACL v1 §28 and ADR-014 govern the six-arm ScenarioNet core: mutation is prohibited and `ScenarioUsefulness.value` is learning-potential-only. §12 calculators are implemented for PPO, TD3, and SAC and wired through custom and SB3 planner backends; short S3 and S5 source-backed ACL smokes now pass, while long-run resume and thesis experiments remain pending.
- `VERIFIED`: the Docker GPU environment mounts `/workspace/data/scenarionet` read-only. It verified all 3,500 frozen references exist, read-only content checks on all 48 golden references, and raw zero-policy ScenarioEnv smoke for one PG and one Waymo reference. The canonical catalog hash and all runtime mappings now exactly match the frozen selection for all primary splits. This is not full Rulebook v4.7, semantic-observation, or learner validation.
- `IMPLEMENTED`: `ThesisScenarioEnv` now exposes a deferred Rulebook v2 adapter factory. After reset it installs the vendor-preserving Bullet callback hook, builds the immutable static cache, captures the causal live snapshot, and supplies the complete registry transition evaluator. Representative S0–S5 and diagnostic S6 source-backed smokes pass; dataset-wide live validation, long-run resume, and thesis experiments remain unverified.

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
| `REQ-AUDIT-005` | `AC-AUDIT-005` | ACL §28, usefulness calculators, planner backends, and config | `tests/test_scenario_acl_usefulness.py`, ACL config matrix, planner import smoke, S3/S5 ACL smokes | IMPLEMENTED; long-run resume and representative multi-episode evidence pending |

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

## 9.1 Canonical Frozen-Replay Alignment Fix

The versioned frozen index is a reconstruction manifest, not a second runtime
dataset. The defect was that replay wrote a second catalog/runtime tree under
`*_frozen` paths while a later catalog regeneration left the canonical tree on
a different selection.

Approved implementation decision (2026-07-19): the frozen index remains the
sole versioned selection authority. Frozen replay rebuilds the canonical
catalog, split manifest, and `runtime/<split>` mappings. No dataset-view
selector or frozen runtime tree is exposed to training code. Explicit Hydra
catalog/runtime overrides remain diagnostic-only and fail closed when
incoherent.

Acceptance criteria and traceability:

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| `REQ-AUDIT-007` | Frozen replay reconstructs the canonical catalog, split manifest, and runtime mappings for every split. | Frozen replay CLI and Make target. | Replay verification, live index/catalog/runtime equality matrix. |
| `REQ-AUDIT-008` | Runtime configuration has one canonical root and does not select a second dataset view. | `conf/env/scenarionet.yaml`, `.env.example`, environment/ACL wiring. | Hydra composition and canonical runtime validation. |
| `REQ-AUDIT-009` | Existing explicit incoherent overrides fail closed. | Existing factory validator, retained unchanged. | Factory validation regression. |

Validation commands: focused Ruff/format checks for modified Python files;
ScenarioNet/ACL configuration tests; replay verification; and a Docker
index/catalog/runtime equality check. Only derived catalog, manifest, and
runtime mapping artifacts may be rewritten; source scenarios remain immutable.

## 10. Milestones

- [x] A1: Locate frozen selection and perform static immutable-index audit.
- [ ] A1: Complete live Rulebook validation and exclusion-input verification; dataset-wide source loadability, route, horizon, and trajectory structural checks have passed.
- [x] A2: Generate deterministic candidate 48-reference suite and coverage matrix from actual frozen catalog metadata.
- [x] A2: Inspect all 48 candidate `ScenarioDescription` files and finalize the reference-only suite.
- [x] A3: Produce source/config/test conformance inventory and amendment surface.
- [x] A3: Implement ACL §12 algorithm-specific learning-potential formulas and propagate the result through custom/SB3 planner backends.
- [x] A4: Fix the verified vectorized reset route-metadata defect and add a regression.
- [x] A4: Fix the provider/DataManager reset-order defect that cached the next scenario before `before_reset`; add a regression and real repeated-reset evidence.
- [x] A4: Rebuild canonical catalog/runtime mappings from the frozen index and remove redundant frozen-derived runtime artifacts.
- [x] A4/A5: Implement and execute representative scalar S0–S6 diagnostic paths. S0–S5 source-backed smokes and diagnostic S6 pass with finite updates/actions, evaluation, and checkpoint publication.
- [ ] A4/A5: Complete multi-episode PG/Waymo semantic-policy parity, default-length PPO validation, checkpoint/resume (including PER and ACL state), and final reconciliation; these remain before `VERIFIED`.

## 11. Progress And Findings Log

- 2026-07-19: Confirmed all selected specifications are authoritative from the index and read the required plans/ADRs. The frozen index contains exact split targets; the runtime root is mounted, but its train view is from a different catalog generation.
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
- 2026-07-19: An earlier ScenarioNet learner attempt was blocked by the then-mixed runtime view; canonical replay has since removed that blocker. The deferred Rulebook v2 adapter now reaches the live PG reset/step path. The mounted data root contains the approved read-only `rulebook_v2/calibration_b_e.json` plus its hash-bound ego config; one real PG and one real Waymo reset/step now pass. A Waymo elevation-datum mismatch was fixed by translating only the common static-cache `z` datum to MetaDrive's live ego `z=0`; relative elevation differences remain intact and a regression test covers the transformation.
- 2026-07-19: Docker GPU staged diagnostics now pass: S1 TD3/MLP with PER and ACL off (20 steps), S2 TD3/LQ with PER and ACL off (20 steps), S3 TD3/LQ with approved ACL and PER off (20 steps), S4 TD3/LQ with PER on and ACL off (20 steps), S5 TD3/LQ with ACL and PER on (20 steps), and diagnostic S6 PPO/LQ with replay off (32 steps, rollout override 16). Each completed evaluation and checkpoint; S3/S5 also persisted ACL/MAB state. These are runtime smokes, not scientific performance results.
- 2026-07-19: Root-caused the legacy reset failure: `_reset_global_seed()` accessed `current_scenario` before `ScenarioDataManager.before_reset`, leaving two cached scenarios. Moving route injection to `_get_reset_return()` preserves reset ordering. The focused ScenarioEnv suite passes `24`, three real consecutive resets pass, and vectorized PG/Waymo integration passes `2`.
- 2026-07-19: An explicit legacy Rulebook-v1 TD3/MLP learner diagnostic on the coherent runtime completed 300 steps with finite actor/critic losses, 150 gradient steps in the final chunk, evaluations, and final checkpoint. It validates learner/reset plumbing only and is not an S1 v4.7 result.
- 2026-07-19: Root-caused the canonical mismatch: frozen-index creation was correct, as its recorded catalog hash exactly matched the former frozen replay catalog. The canonical catalog was regenerated after freeze. Reverted the erroneous two-view configuration, changed frozen replay to rebuild canonical artifacts, regenerated the canonical catalog, split manifest, and runtime mappings from the index, and removed redundant frozen-derived artifacts. The canonical catalog hash now equals the index hash and all primary runtime mappings validate.
- 2026-07-19: Added isolated live-adapter increments in `rulebook/v2/context/metadrive_live.py`: stable ScenarioNet actor IDs, explicit MetaDrive actor taxonomy, finite pose/velocity/footprint normalization, lane identity, speed-cap validation, deterministic vehicle collection, public-registry actor collection for VRUs, explicit ego/other-actor partitioning, fail-closed rejection of unknown public-registry objects, a Bullet contact-onset normalizer/step buffer with current-manifold persistence, and current signal-state extraction through MetaDrive's source/object mapping. The increment remains source-neutral and is not wired into v4.7 training yet.
- 2026-07-19: Runtime smokes exposed and fixed three implementation defects: non-ego Bullet callbacks are ignored while node-only callbacks defer to the current manifold; single-point Bullet manifold bindings are called with index `0` (with a compatibility fallback); and the drivable surface now unions all vertically compatible lanes so an off-lane ego yields an off-road cost instead of an invalid empty surface. Regression tests cover each case.
- 2026-07-19: Semantic observation route construction now reuses the reset Rulebook cache after live elevation-datum alignment, preventing false 2.5D route mismatches on Waymo evaluation. ACL buffer persistence now creates its artifact parent directory before writing.
- 2026-07-19: Preserved `exit_lanes` in both static adapters and added `derive_lane_movement_key`: the assigned ego route may disambiguate a branch, a unique successor may resolve an actor, and ambiguous topology returns no key with a validation error. No priority is inferred from geometry; `vehicle_yield` remains NOT_APPLICABLE without explicit movement-priority records. Added `build_episode_cache`, reset memory initialization, and a complete source-neutral `evaluate_transition` composition through the fixed registry. The focused Rulebook suite passes 166 tests with one environment-dependent skip; source-backed learner smoke and checkpoint/resume remain pending.
- 2026-07-19: Read-only static-adapter probes on actually loaded canonical records produced zero validation errors for one PG record, one Waymo record, and one Waymo record with route traffic lights (the latter yielded two route-relevant signal controls). The mounted read-only calibration artifact was then loaded and the complete deferred live adapter passed ten control steps on one real PG and one real Waymo record; this remains representative smoke evidence, not dataset-wide live validation.
- 2026-07-19: Corrected the stale Rulebook configuration version from `4.6-final-implementation-complete` to the authoritative `4.7-final-implementation-complete`; added a focused regression test. This changes configuration identity only and does not claim live adapter completion.

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
| `src/thesis_rl/cli/scenarios/replay_frozen_dataset.py` | Modified | Rebuild canonical catalog, split manifest, and runtime mappings from the frozen index |
| `src/thesis_rl/rulebook/v2/context/metadrive_live.py` | Added | Explicit MetaDrive live actor normalization for the Rulebook snapshot boundary |
| `tests/test_rulebook_v2_metadrive_live.py` | Added | Regression coverage for stable IDs, taxonomy, dimensions, speed caps, public-registry VRU collection, ego/actor partitioning, fail-closed unknown objects, contact normals/buffering, and current signal states |
| `src/thesis_rl/rulebook/v2/geometry/lanes.py` | Modified | Preserve successor topology and derive movement keys only when unambiguous |
| `src/thesis_rl/rulebook/v2/context/{pg_static_adapter,waymo_static_adapter}.py` | Modified | Preserve source `exit_lanes` and reject ambiguous control movement keys |
| `src/thesis_rl/rulebook/v2/transition.py` | Added | Immutable episode-cache builder, reset memory initialization, and complete source-neutral registry transition composition |
| `src/thesis_rl/rulebook/v2/__init__.py` | Modified | Export transition composition APIs |
| `tests/test_rulebook_v2_{geometry,pg_adapter,waymo_adapter,transition}.py` | Added/modified | Topology, movement-key, adapter-preservation, and complete-registry transition regressions |
| `tests/test_rulebook_v2_scenarionet_integration.py` | Added | Real canonical PG/Waymo ten-step Rulebook v4.7 source smoke |
| `conf/rulebook/v2.yaml` | Modified | Aligns runtime Rulebook identity with authoritative v4.7 |
| `tests/test_rulebook_v2_config.py` | Added | Regression for authoritative Rulebook v4.7 configuration identity |
| `src/thesis_rl/rulebook/v2/geometry/drivable.py` | Modified | Preserve the full vertically compatible drivable surface for off-road evaluation |
| `tests/test_rulebook_v2_geometry.py` | Modified | Regression for off-lane off-road surface availability |
| `tests/test_scenario_acl_buffer.py` | Modified | Regression for ACL artifact-parent creation |
| `conf/env/scenarionet.yaml` | Restored | One canonical ScenarioNet root and runtime view |
| `.env.example` | Modified | Documents that canonical paths derive from the root and frozen index |
| `tests/test_scenarionet_vectorized_integration.py` | Modified | Live canonical catalog/index/runtime equality validation for all primary splits |
| `tests/test_thesis_scenario_env.py` | Modified | Regressions for MetaDrive-style `Config.pop` and reset-order provider access |
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
| `docker compose ... pytest -q tests/test_thesis_scenario_env.py` | PASS | 2026-07-19 | 24 tests pass, including route-metadata and provider reset-order regressions. |
| `docker compose ... replay_frozen_dataset --verify-only` | PASS | 2026-07-19 | Verified all 3,500 frozen source references before canonical reconstruction. |
| `docker compose ... replay_frozen_dataset --overwrite` | PASS | 2026-07-19 | Rebuilt canonical catalog, split manifest, and runtime mappings from the frozen index without modifying source scenarios. |
| `docker compose ... pytest -q tests/test_scenarionet_vectorized_integration.py tests/test_scenario_frozen.py tests/test_scenarionet_config.py tests/test_scenario_acl_*.py` | PASS | 2026-07-19 | 33 passed; canonical catalog/index/runtime equality, vectorized integration, frozen contracts, and ACL wiring. |
| `docker compose ... pytest -q tests/test_scenarionet_vectorized_integration.py` | PASS | 2026-07-19 | 2 real PG/Waymo vectorized integration tests pass after reset-order fix. |
| `docker compose ... pytest -q tests/test_scenario_acl_config.py tests/test_scenario_acl_usefulness.py tests/test_scenario_acl_buffer.py tests/test_scenario_acl_mab.py` | PASS (previous run) | 2026-07-19 | 26 passed after ACL v1 §28 / ADR-014 configuration enforcement. The final all-ACL glob run later passed 34. |
| `docker compose ... pytest -q tests/test_scenario_acl_usefulness.py ... tests/test_transition_boundary.py` | PASS | 2026-07-19 | 50 passed; ACL §12 formulas, ACL state/replay, and approved transition replay boundaries. |
| `docker compose ... ruff check ... planner/usefulness files` | PASS | 2026-07-19 | All modified ACL/planner files pass focused Ruff. |
| `docker compose run --rm dev python -m pytest -q tests/test_rulebook_v2_transition.py` | PASS | 2026-07-19 | Complete registry transition smoke: 2 passed. |
| `docker compose run --rm dev ruff check ...` | PASS | 2026-07-19 | Focused Ruff for transition, topology, and static-adapter changes. |
| `docker compose ... pytest -q tests/test_planner_lifecycle.py tests/test_hydra_agent_presets.py tests/test_hydra_preset_run_configs.py tests/test_hydra_preset_test_configs.py tests/test_checkpointing.py` | PASS | 2026-07-19 | Initial run: 48 passed, 5 stale Hydra alias failures; after canonical `latent_query_v2` alignment, the complete focused lifecycle/Hydra/checkpoint matrix passes 44/44. |
| `docker compose ... pytest -q tests/test_scenario_frozen.py ... tests/test_agent_pipeline.py` | PASS | 2026-07-19 | 102 passed across ScenarioNet, scalarization, observation, encoder, checkpoint, and Agent regressions. |
| `docker compose ... py_compile ... && python -c 'import ... PPO/TD3/SAC SB3 backends'` | PASS | 2026-07-19 | Syntax and planner import smoke passed without learner/GPU initialization. |
| `docker compose ... ruff check ... && ruff format --check ...` | PASS | 2026-07-19 | Modified source, test, and new scripts pass focused Ruff checks. |
| `docker compose ... python -m thesis_rl.cli.train --config-name presets/test/smoke_train` | PASS | 2026-07-19 | GPU standard smoke completed 2,000 steps, two evaluations, and final checkpoint; no updates because `learning_starts=10000`. |
| `docker compose ... pytest -q tests/test_rulebook_v2_contracts.py tests/test_rulebook_v2_wrapper.py tests/test_rulebook_v2_monitor.py tests/test_rulebook_v2_causal_context.py tests/test_rulebook_v2_live_adapter.py tests/test_rulebook_v2_causality.py` | PASS | 2026-07-19 | 41 focused Rulebook v4.7 contract/live-adapter tests pass; this does not prove `ThesisScenarioEnv` live adapter wiring. |
| ScenarioNet canonical catalog/runtime alignment | PASS | 2026-07-19 | Canonical artifacts were regenerated from the frozen index; catalog hash and runtime mappings match exactly. |
| `docker compose ... pytest -q tests/test_scenario_frozen.py tests/test_scenario_manifests.py tests/test_scenario_validation.py tests/test_scenarionet_config.py tests/test_scenarionet_pipeline.py tests/test_scenarionet_vectorized_integration.py tests/test_thesis_scenario_env.py` | PASS | 2026-07-19 | 84 focused tests pass against the canonical mounted root, including all three split equality checks and PG/Waymo vectorized reset/step coverage. |
| `docker compose ... replay_frozen_dataset --verify-only` | PASS | 2026-07-19 | 3,500 frozen references verified; canonical catalog SHA-256 equals the index declaration `8d8a72b81db34fe41595114883adaddc1be3f3c1fc4e55754ec427f07b942801`. |
| canonical index/catalog/runtime equality script | PASS | 2026-07-19 | UID sets and runtime basenames match for train=2,000, validation=500, and test=1,000. |
| `docker compose ... pytest -q tests/test_scenarionet_vectorized_integration.py` | PASS | 2026-07-19 | 5 canonical PG/Waymo split and vectorized S0 integration tests pass against the repaired runtime chain. |
| `make rulebook-v2-check` | PASS | 2026-07-19 | 169 Rulebook v4.7 tests pass, 1 environment-dependent test skipped; focused Ruff and `git diff --check` pass after the latest regressions. |
| `docker compose run --rm dev pytest -q tests/test_scenario_acl_*.py` | PASS | 2026-07-19 | 34 ACL tests pass, including buffer-parent persistence and approved learning-potential/mutation boundaries. |
| `docker compose run --rm dev pytest -q tests/test_thesis_scenario_env.py tests/test_scenario_acl_*.py` | PASS | 2026-07-19 | 58 regression tests pass after reset-order, aligned-route, deferred-adapter, and ACL persistence changes. |
| `docker compose ... pytest -q tests/test_rulebook_v2_config.py` | PASS | 2026-07-19 | Rulebook configuration identity regression passes; focused Ruff check passes. |
| `docker compose ... pytest -q tests/test_rulebook_v2_scenarionet_integration.py -m integration` | PASS | 2026-07-19 | One real canonical PG and one real Waymo reset/step complete Rulebook v4.7 with the mounted hash-bound calibration artifact. |
| `docker compose ... pytest -q tests/test_rulebook_v2_metadrive_live.py tests/test_rulebook_v2_live_adapter.py` | PASS | 2026-07-19 | 19 focused tests pass for explicit MetaDrive actor normalization, public-registry VRU collection, ego/actor partitioning, fail-closed unknown objects, contact onset normalization/buffering, current signal extraction, and existing live snapshot contracts. |
| `docker compose ... ruff check/format-check` on live adapter files | PASS | 2026-07-19 | New adapter module and regression test pass focused Ruff lint and formatting checks. |
| canonical fixed-sequence PG/Waymo actor/contact hook probe | PASS | 2026-07-19 | One real PG and one real Waymo scenario each produced finite live vehicle snapshots; non-ego and node-only Bullet callbacks are filtered/deferred, and complete transition evaluation is covered by the focused registry suite and the ten-step source smoke. |
| canonical fixed-sequence Waymo live-signal probe | PASS | 2026-07-19 | One real train Waymo scenario with 14 traffic lights produced 14 current states through the ScenarioNet physical-ID mapping; no future `dynamic_map_states` sequence was read. Unknown states remain explicit; complete evaluation passed in the ten-step source smoke. |
| canonical fixed-sequence Waymo ego/actor snapshot probe | PASS | 2026-07-19 | One real train Waymo VRU scenario produced 12 unique finite other-actor snapshots (11 vehicles, 1 cyclist) after one ego exclusion; pedestrian/static-object parity remains unverified. |
| `docker compose ... pytest -q tests/test_hydra_preset_test_configs.py tests/test_hydra_preset_run_configs.py tests/test_hydra_agent_presets.py` | PASS | 2026-07-19 | 32 Hydra preset/config tests pass; the previous LQ-vs-`latent_query_v2` test-alignment concern is not present in the current tree. |
| ScenarioNet S0–S5 Rulebook/learner paths | PASS; final reconciliation pending | 2026-07-19 | S0 ten-step PG/Waymo path, S1 TD3/MLP, S2 TD3/LQ, S3 ACL+LQ, S4 PER+LQ, and S5 ACL+PER+LQ completed short source-backed smokes with finite updates/actions, evaluation, and checkpoints. Long-run multi-episode parity, resume, and scientific runs remain pending. |
| ScenarioNet S6 PPO path | PASS (diagnostic) | 2026-07-19 | PPO/LQ with replay disabled completed 32 steps, two PPO updates, evaluation, and checkpoint using a diagnostic rollout override (`n_steps=16`, batch 8); the approved default rollout remains 2048. |
| legacy Rulebook-v1 learner diagnostic | PASS (diagnostic only) | 2026-07-19 | 300 steps, finite TD3 actor/critic losses, 150 final-chunk gradient steps, evaluations and final checkpoint; not an S1 v4.7 result. |

## 15. Final Reconciliation

Pending. `IMPLEMENTED` and `VERIFIED` must remain distinct until the audit artifacts, focused checks, mounted-data validation, and final source/config/test reconciliation are complete.
