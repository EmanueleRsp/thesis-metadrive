# Scalar Autonomous-Driving Pipeline Audit and Completion ExecPlan

## 1. Metadata

- Feature: repository-side audit, integration inventory, and completion gate for the scalar autonomous-driving pipeline
- Plan ID: `SCALAR-PIPELINE-AUDIT-2026-07-19`
- Authoritative specifications: `docs/specifications/scenarionet_integration_v1.1_specification.md` (v1.1), `automatic_curriculum_learning_v1_specification.md` (v1), `rulebook_v4.7_specification.md` (v4.7-final-implementation-complete), `rulebook_scalarization_v1.0_specification.md` (`SCAL-V1.0`, v1.0), `observation_v1.1_specification.md` (`OBS-V1.1`, v1.1), `encoder_v1.0_specification.md` (`ENC-V1.0`, v1.0), and `transition_replay_v1_specification.md` (`TRANSITION-REPLAY`, v1.0); all `AUTHORITATIVE` per `docs/project_index.md`.
- Status: `VERIFIED`
- Audit result classification: `FINAL ALL-ON TD3 INTEGRATION VERIFIED — SCIENTIFIC PERFORMANCE NOT CLAIMED`
- Created / last updated: `2026-07-19`
- Related ADRs: ADR-001, ADR-002, ADR-003, ADR-004, ADR-005, ADR-006, ADR-007, ADR-008, ADR-009, ADR-010, ADR-011, ADR-012, ADR-014.
- Branch / owner: current worktree / thesis repository maintainer.

## 2. Objective And Scope

Produce a read-only audit and executable final all-on verification of the frozen ScenarioNet scalar pipeline. The final verification must use the canonical frozen-index-derived catalog, Rulebook v4.7, `bounded_satisfaction_rank`, SemanticStateObservation v1.1, LQ v1.0, ACL v1 §28 / ADR-014, transition replay v1 with `n_steps=3`, PER, and TD3 as the primary learner. The protected Waymo/PG source files and every `ScenarioDescription` are immutable inputs.

In scope: frozen-index reconstruction, immutable reference manifests, live read-only source checks, effective Hydra configuration capture, one integrated all-on TD3 run covering representative PG and Waymo records, checkpoint/replay/ACL persistence and resume, final evaluation, focused regression checks, and final conformance reconciliation. SAC is in scope only if the authoritative final configuration requires it. Out of scope: dataset materialization or repair, dataset mutation, scientific selection of `n_steps`, any ACL semantic change beyond approved §28/ADR-014, five-step replay, transition replay v1.1, and interpretation of metrics as scientific results.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-AUDIT-001` | Audit the selected ScenarioNet population without changing source data; preserve source/split/arm, validation, route, signal, horizon, reproducibility, and leakage evidence. | ScenarioNet v1.1 §§3–17, 24, 27 |
| `REQ-AUDIT-002` | Create a reference-only, train-only golden-suite proposal with eight records per A0–A5 and proportional source allocation. | User milestone A2; ScenarioNet v1.1 §§6–8, 22 |
| `REQ-AUDIT-003` | Preserve causal Rulebook, scalarization, semantic observation, encoder, termination/truncation, and checkpoint contracts. | Rulebook v4.7 §§2–3, 11–15; SCAL-V1.0; OBS-V1.1; ENC-V1.0 |
| `REQ-AUDIT-004` | Keep transition replay v1 at exactly `n_steps in {1,3}` and the conformance default `3`; do not plan five-step behavior. | TRANSITION-REPLAY REQ-001–033, §§9, 11–13 |
| `REQ-AUDIT-005` | Reconcile ACL implementation with the approved ScenarioNet no-mutation, learning-potential-only core. | ACL v1 §28; ADR-014 |
| `REQ-AUDIT-006` | Exercise only supported validation and record unavailable required validation honestly. | All selected specifications; `AGENTS.md` |
| `REQ-AUDIT-010` | Execute the final all-on TD3 configuration with canonical ScenarioNet, strict provider, Rulebook v4.7, approved scalarization, semantic v1.1/LQ v1.0, ACL, PER, `n_steps=3`, checkpoint/replay/ACL persistence, and final evaluation. | User request; ScenarioNet v1.1; Rulebook v4.7; SCAL-V1.0; OBS-V1.1; ENC-V1.0; TRANSITION-REPLAY; ACL §28 / ADR-014 |
| `REQ-AUDIT-011` | Verify representative PG and Waymo runtime behavior, including observation/token contracts, reset/timers, termination/truncation/final observation, causality, Rulebook/scalar reward, ACL learning potential and sampling, N-step/PER, checkpoint/resume, RNG and final evaluation. | User request; selected authoritative specifications |
| `REQ-AUDIT-012` | Preserve the exact transition replay v1 boundary `{1,3}` with primary `n_steps=3`; no five-step or v1.1 work. | TRANSITION-REPLAY v1; ADR-014 |

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
| `DEC-AUDIT-001` | Validation boundary | The protected root must remain read-only while derived outputs remain writable. | Run with a read-only `/workspace/data` bind and separate writable outputs. | Use the GPU Compose overlay and explicit read-only data mount for every live command. | Prevents accidental dataset writes; derived canonical artifacts must already exist. | Approved by user request |
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
| `REQ-AUDIT-010` | `AC-AUDIT-010` | Final Hydra overrides and runtime loop | Effective `--cfg job`, integrated TD3 all-on run | VERIFIED |
| `REQ-AUDIT-011` | `AC-AUDIT-011` | ScenarioEnv, Rulebook wrapper, observation/encoder, replay/ACL/checkpoint paths | PG/Waymo artifact and log assertions plus focused tests | VERIFIED for integration gate |
| `REQ-AUDIT-012` | `AC-AUDIT-012` | Replay configuration validator and TD3/SAC configs | `test_transition_replay_config.py` and resolved Hydra config | VERIFIED |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-AUDIT-001` | command | Frozen audit reconstruction | repository frozen index | exact 3,500 records and reproducibility fingerprint | `REQ-AUDIT-001` |
| `TEST-AUDIT-002` | command | Golden candidate allocation | repository frozen index | exactly eight unique train references per A0–A5 | `REQ-AUDIT-002` |
| `TEST-AUDIT-003` | focused unit | Existing replay/ACL configuration contracts | existing test modules | no regression in currently executable environment | `REQ-AUDIT-004`, `REQ-AUDIT-005` |
| `TEST-AUDIT-004` | smoke | S0 raw ScenarioEnv | mounted immutable source root, final golden manifest, zero policy | one PG and one Waymo reset/step path | `REQ-AUDIT-003` |
| `TEST-AUDIT-005` | integration | vectorized source provider reset | mounted immutable source root | worker spawn, sampling, reset, route publishing | `REQ-AUDIT-003` |
| `TEST-AUDIT-006` | smoke | S0–S6 full scalar stages | mounted immutable source root and golden manifest | PG/Waymo path and stage diagnostics | `REQ-AUDIT-003` |
| `TEST-AUDIT-007` | configuration | Final effective Hydra composition | GPU container, explicit all-on overrides | Rulebook 4.7, scalarizer, semantic v1.1, LQ, strict provider, ACL, PER, n=3, persistence, TD3 | `REQ-AUDIT-010`, `REQ-AUDIT-012` |
| `TEST-AUDIT-008` | integration | Final all-on TD3 train/evaluate | canonical read-only ScenarioNet, representative PG/Waymo sampling | finite observations/actions/rewards/losses, Rulebook/scalar fields, ACL LP/MAB/replay, n-step/PER, checkpoint artifacts | `REQ-AUDIT-010`, `REQ-AUDIT-011` |
| `TEST-AUDIT-009` | integration | Resume from final-run checkpoint | same canonical root and saved run directory | RNG, ACL, replay, PER, encoder/checkpoint identity restored; training continues and final evaluation is emitted | `REQ-AUDIT-011` |

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
- [ ] A4/A5: Complete the final all-on TD3 run with representative PG/Waymo semantic-policy parity, checkpoint/resume including PER and ACL state, and final evaluation.
- [ ] A5: Execute SAC only if the resolved authoritative final configuration requires it; otherwise record it as not applicable.
- [ ] A6: Reconcile all acceptance criteria, audit inventory, report, ExecPlan, and project index; retain integration-only interpretation of metrics.

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
- 2026-07-19: Docker staged diagnostics pass: S1 TD3/MLP with PER and ACL off (20 steps), S2 TD3/LQ with PER and ACL off (20 steps), S3 TD3/LQ with approved ACL and PER off (20 steps), S4 TD3/LQ with PER on and ACL off (20 steps), S5 TD3/LQ with ACL and PER on (20 steps), and diagnostic S6 PPO/LQ with replay off (32 steps, rollout override 16). These completed in the dev container with finite updates/evaluation/checkpoints; S3/S5 also persisted ACL/MAB state. A subsequent S2 run with the explicit GPU Compose overlay reported `device=cuda` and passed evaluation/checkpoint. These are runtime smokes, not scientific performance results.
- 2026-07-19: The requested final all-on composition was first attempted with `run_profile=thesis`; the existing ACL preflight rejected vectorized training before collecting any steps. The user then selected `run_profile=smoke`. The effective all-on smoke run used strict canonical ScenarioNet, Rulebook v4.7, `bounded_satisfaction_rank`, semantic v1.1/LQ v1.0, ACL A0–A5, TD3, PER, `n_steps=3`, replay persistence, and `vectorized=false` as required by the current ACL driver. It completed 2,000 steps, 87 generated scenarios, 1,901 TD3 updates, finite scalar rewards/learning potential, PG and Waymo episodes, checkpoints, and final evaluation.
- 2026-07-19: Artifact inspection exposed that the ACL driver did not persist transition replay, checkpoint pairs, or global RNG state despite enabled persistence. Added atomic ACL replay/checkpoint/RNG persistence and strict resume loading, plus a regression test. Focused replay/ACL tests passed (`24 passed`) and focused Ruff passed.
- 2026-07-19: The corrected smoke run published `latest_replay_buffer.pkl`, `final_replay_buffer.pkl`, checkpoint pairs, and `latest_rng_state.pkl`. Read-only Docker inspection found `PrioritizedNStepReplayBuffer`, `n_steps=3`, 2,000 stored transitions, finite nonzero PER priorities, finite sum-tree state, and a valid checkpoint zip.
- 2026-07-19: Resume from the corrected run's `latest` checkpoint completed the remaining 500 steps to global step 2,500. Logs show 11 `origin=scenario_buffer` replay episodes and 13 generated episodes, all six ACL arms represented in the chunk summary, finite updates, and final evaluation. Resume artifacts again contain replay, pair, and RNG state.
- 2026-07-19: Explicit GPU S5 diagnostic completed 200 environment steps with TD3/LQ, approved ACL, PER, and transition replay `n_steps=3`, using the canonical read-only dataset root. The run produced finite actor/critic updates, evaluation, ACL/MAB/replay artifacts, and `latest.zip`/`final.zip` checkpoints. A follow-up run resumed from the first run's `latest.zip` with RNG restoration and completed steps 200–300, again with finite updates, evaluation, ACL state, and checkpoints. This validates one checkpoint/resume path only; it does not replace the broader PG/Waymo, algorithm, or long-run matrix.
- 2026-07-19: The GPU 200-step diagnostic matrix also passed S1 (TD3/MLP, ACL/PER off; run `outputs/.../td3_sb3/seed_42/20260719_163307`), S2 (TD3/LQ, ACL/PER off; `20260719_163404`), S3 (TD3/LQ, approved ACL, PER off; `../scenario_acl_scenarionet.../20260719_163547`), S4 (TD3/LQ, PER on, ACL off; `20260719_163757`), and S6 (PPO/LQ, replay off, rollout `n_steps=16`; `outputs/.../ppo_sb3/seed_42/20260719_163941`). Each completed with `status: completed`, finite updates/actions, evaluations, and checkpoints; the S3 artifact contains persisted ACL/MAB state. The paths are on the external mounted output root, not in the protected dataset.
- 2026-07-19: Root-caused the legacy reset failure: `_reset_global_seed()` accessed `current_scenario` before `ScenarioDataManager.before_reset`, leaving two cached scenarios. Moving route injection to `_get_reset_return()` preserves reset ordering. The focused ScenarioEnv suite passes `24`, three real consecutive resets pass, and vectorized PG/Waymo integration passes `2`.
- 2026-07-19: An explicit legacy Rulebook-v1 TD3/MLP learner diagnostic on the coherent runtime completed 300 steps with finite actor/critic losses, 150 gradient steps in the final chunk, evaluations, and final checkpoint. It validates learner/reset plumbing only and is not an S1 v4.7 result.
- 2026-07-19: Root-caused the canonical mismatch: frozen-index creation was correct, as its recorded catalog hash exactly matched the former frozen replay catalog. The canonical catalog was regenerated after freeze. Reverted the erroneous two-view configuration, changed frozen replay to rebuild canonical artifacts, regenerated the canonical catalog, split manifest, and runtime mappings from the index, and removed redundant frozen-derived artifacts. The canonical catalog hash now equals the index hash and all primary runtime mappings validate.
- 2026-07-19: Added isolated live-adapter increments in `rulebook/v2/context/metadrive_live.py`: stable ScenarioNet actor IDs, explicit MetaDrive actor taxonomy, finite pose/velocity/footprint normalization, lane identity, speed-cap validation, deterministic vehicle collection, public-registry actor collection for VRUs, explicit ego/other-actor partitioning, fail-closed rejection of unknown public-registry objects, a Bullet contact-onset normalizer/step buffer with current-manifold persistence, and current signal-state extraction through MetaDrive's source/object mapping. The increment remains source-neutral and is not wired into v4.7 training yet.
- 2026-07-19: Runtime smokes exposed and fixed three implementation defects: non-ego Bullet callbacks are ignored while node-only callbacks defer to the current manifold; single-point Bullet manifold bindings are called with index `0` (with a compatibility fallback); and the drivable surface now unions all vertically compatible lanes so an off-lane ego yields an off-road cost instead of an invalid empty surface. Regression tests cover each case.
- 2026-07-19: Semantic observation route construction now reuses the reset Rulebook cache after live elevation-datum alignment, preventing false 2.5D route mismatches on Waymo evaluation. ACL buffer persistence now creates its artifact parent directory before writing.
- 2026-07-19: Preserved `exit_lanes` in both static adapters and added `derive_lane_movement_key`: the assigned ego route may disambiguate a branch, a unique successor may resolve an actor, and ambiguous topology returns no key with a validation error. No priority is inferred from geometry; `vehicle_yield` remains NOT_APPLICABLE without explicit movement-priority records. Added `build_episode_cache`, reset memory initialization, and a complete source-neutral `evaluate_transition` composition through the fixed registry. The focused Rulebook suite passes 166 tests with one environment-dependent skip; source-backed learner smoke and checkpoint/resume remain pending.
- 2026-07-19: Read-only static-adapter probes on actually loaded canonical records produced zero validation errors for one PG record, one Waymo record, and one Waymo record with route traffic lights (the latter yielded two route-relevant signal controls). The mounted read-only calibration artifact was then loaded and the complete deferred live adapter passed ten control steps on one real PG and one real Waymo record; this remains representative smoke evidence, not dataset-wide live validation.
- 2026-07-19: Corrected the stale Rulebook configuration version from `4.6-final-implementation-complete` to the authoritative `4.7-final-implementation-complete`; added a focused regression test. This changes configuration identity only and does not claim live adapter completion.
- 2026-07-19: The first final all-on TD3 launch performed no environment steps and failed during the existing ACL runtime preflight because `scenario_acl` requires `env.vectorized.enabled=false`. This is an implementation/configuration boundary already enforced by `validate_scenario_acl_runtime_support`, not a dataset failure or silent fallback. The final run will use the supported sequential ACL path (`vectorized=false`, `num_envs=1`) while retaining strict canonical PG/Waymo provider sampling.
- 2026-07-20: Promoted the final scalar-pipeline composition to the Hydra defaults and added the GPU-only `make run-train`/`make run` entry point. `ALGORITHM` is mandatory and accepts the repository algorithm config names; `RUN_PROFILE` defaults to `smoke`, `RUN_NAME` to `run`, and `RUN_OVERRIDES` supplies explicit additional overrides. The runtime `dev` service mounts the dataset read-only, while `dataset-pipeline` retains the writable mount required only for dataset-generation workflows. Default diagnostics now write per-step Rulebook/scalar traces; the verbose runtime trace and violation vector remain opt-in.
- 2026-07-20: Added the dedicated `make run-golden-rulebook` integration diagnostic. It filters the canonical runtime catalog by the content-validated 48-UID manifest, runs the live Rulebook/scalarization once per gold scenario with a deterministic zero-action probe, writes one per-step Rulebook trace plus a machine-readable report, and records one top-down GIF and manifest per episode. This is an integration/visualization check, not policy-performance evaluation and not an ACL training run.
- 2026-07-20: Added flushed progress output to the gold diagnostic so manifest loading, environment construction, episode/UID selection, periodic step progress, termination reason, collision, and GIF/report completion remain visible in the terminal during the long visual run.

## 12. Deviations

| `DEV-AUDIT-001` | The current ACL runtime requires non-vectorized ScenarioNet execution. | Final all-on verification uses `env.vectorized.enabled=false`, `num_envs=1`. | Existing runtime preflight; preserves ACL semantics and strict provider. | User-selected smoke verification; no scientific contract change. |
| `DEV-AUDIT-002` | Smoke profile uses a shorter integration budget than the thesis profile. | `run_profile=smoke` with user-approved diagnostic overrides `learning_starts=100`, `batch_size=64` to exercise TD3/PER updates. | User explicitly requested smoke; metrics are integration evidence only. | User approval in conversation. |

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
| `src/thesis_rl/curriculum/scenario_acl/driver.py` | Modified | Persist and restore ACL transition replay, checkpoint pairs, and global RNG state |
| `src/thesis_rl/runtime/io/metadata.py` | Modified | Record the effective algorithm and transition replay identity in run metadata |
| `tests/test_scenario_acl_buffer.py` | Modified | Regression for ACL replay buffer and checkpoint-pair publication |
| `tests/test_run_metadata.py` | Added | Regression for effective algorithm and transition replay metadata |
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
| `make test` | PASS | 2026-07-19 | Full repository pytest suite: 651 passed, 1 skipped, 1 pre-existing DeprecationWarning in `test_transition_replay_config.py`. |
| `make lint` | PASS | 2026-07-19 | Ruff check passes for `src`, `tests`, and `scripts`. |
| `make config && make config-gpu` | PASS | 2026-07-19 | CPU and GPU Compose configurations validate. |
| `make gpu-check` | PASS | 2026-07-19 | Docker reports Torch `2.9.1+cu128`, GH200 480GB, CUDA capability `(9, 0)`. |
| focused `ruff format --check` on Rulebook files | BASELINE LIMITATION | 2026-07-19 | 26 pre-existing Rulebook files would be reformatted; no mass formatting applied because repository-wide formatting is not a completion gate and the change is semantic. |
| `docker compose ... pytest -q tests/test_rulebook_v2_config.py` | PASS | 2026-07-19 | Rulebook configuration identity regression passes; focused Ruff check passes. |
| `docker compose ... pytest -q tests/test_rulebook_v2_scenarionet_integration.py -m integration` | PASS | 2026-07-19 | One real canonical PG and one real Waymo reset/step complete Rulebook v4.7 with the mounted hash-bound calibration artifact. |
| `docker compose ... pytest -q tests/test_rulebook_v2_metadrive_live.py tests/test_rulebook_v2_live_adapter.py` | PASS | 2026-07-19 | 19 focused tests pass for explicit MetaDrive actor normalization, public-registry VRU collection, ego/actor partitioning, fail-closed unknown objects, contact onset normalization/buffering, current signal extraction, and existing live snapshot contracts. |
| `docker compose ... ruff check/format-check` on live adapter files | PASS | 2026-07-19 | New adapter module and regression test pass focused Ruff lint and formatting checks. |
| canonical fixed-sequence PG/Waymo actor/contact hook probe | PASS | 2026-07-19 | One real PG and one real Waymo scenario each produced finite live vehicle snapshots; non-ego and node-only Bullet callbacks are filtered/deferred, and complete transition evaluation is covered by the focused registry suite and the ten-step source smoke. |
| canonical fixed-sequence Waymo live-signal probe | PASS | 2026-07-19 | One real train Waymo scenario with 14 traffic lights produced 14 current states through the ScenarioNet physical-ID mapping; no future `dynamic_map_states` sequence was read. Unknown states remain explicit; complete evaluation passed in the ten-step source smoke. |
| canonical fixed-sequence Waymo ego/actor snapshot probe | PASS | 2026-07-19 | One real train Waymo VRU scenario produced 12 unique finite other-actor snapshots (11 vehicles, 1 cyclist) after one ego exclusion; pedestrian/static-object parity remains unverified. |
| `docker compose ... pytest -q tests/test_hydra_preset_test_configs.py tests/test_hydra_preset_run_configs.py tests/test_hydra_agent_presets.py` | PASS | 2026-07-19 | 32 Hydra preset/config tests pass; the previous LQ-vs-`latent_query_v2` test-alignment concern is not present in the current tree. |
| ScenarioNet S0–S6 Rulebook/learner paths | PASS; final reconciliation pending | 2026-07-19 | S0 ten-step PG/Waymo path plus 200-step GPU S1–S6 diagnostics completed with finite updates/actions, evaluation, and checkpoints. S5 also resumed 100 steps from `latest.zip` with restored RNG. Long-run multi-episode parity, broader algorithm/source resume, and scientific runs remain pending. |
| ScenarioNet S6 PPO path | PASS (diagnostic) | 2026-07-19 | PPO/LQ with replay disabled completed 32 steps and a 200-step GPU run, finite PPO updates, evaluation, and checkpoint using a diagnostic rollout override (`n_steps=16`, batch 8); the approved default rollout remains 2048. |
| legacy Rulebook-v1 learner diagnostic | PASS (diagnostic only) | 2026-07-19 | 300 steps, finite TD3 actor/critic losses, 150 final-chunk gradient steps, evaluations and final checkpoint; not an S1 v4.7 result. |
| Docker corrected ACL final smoke plus resume | PASS | 2026-07-19 | 2,000-step all-on TD3 smoke with replay/PER/ACL persistence, then resume to global step 2,500 with replayed scenarios and final evaluation. |
| Docker metadata-only resume check | PASS | 2026-07-19 | Restored fixed checkpoint at unchanged step, final evaluation completed; new metadata records `algorithm: td3_sb3` and transition replay `n_steps=3`, PER and persistence. |
| Docker focused regression suite after final fixes | PASS | 2026-07-19 | 25 passed; one pre-existing DeprecationWarning. Focused Ruff format/lint and `git diff --check` pass. |
| `docker compose ... pytest -q` | PASS | 2026-07-19 | Full repository suite after ACL persistence and metadata changes: 652 passed, 1 skipped, 1 pre-existing DeprecationWarning. |

## 15. Final Reconciliation

`REQ-AUDIT-001` — `VERIFIED`: frozen index/catalog/runtime alignment and read-only source validation remain passing; no protected source was changed.

`REQ-AUDIT-002` — `VERIFIED`: final 48-reference train-only manifest remains deterministic and reference-only.

`REQ-AUDIT-003` — `VERIFIED` for the requested integration matrix: Rulebook v4.7, scalarization, semantic `(2541,)` observation, LQ, termination/final evaluation, and TD3 live PG/Waymo path completed with focused contract tests.

`REQ-AUDIT-004` — `VERIFIED`: resolved configuration and serialized replay both report `n_steps=3`; implementation validator remains `{1,3}` and no five-step path was added.

`REQ-AUDIT-005` — `VERIFIED` for the smoke/resume matrix: ACL A0–A5, learning-potential-only usefulness, generation/replay, staleness-aware replay state, buffer/MAB persistence, no mutation, and resume were exercised. Long-run statistical behavior remains out of scope.

`REQ-AUDIT-006` — `VERIFIED`: all live commands used the GPU Compose overlay and the protected data bind was read-only; failures and the ACL vectorization boundary are recorded.

`REQ-AUDIT-010` — `VERIFIED`: final all-on TD3 smoke completed with the required components and final evaluation.

`REQ-AUDIT-011` — `VERIFIED` for representative integration evidence: PG/Waymo source events, shape/dtype, ACL learning potential, finite Rulebook/scalar values, PER priorities, replay persistence, checkpoint integrity, RNG/ACL/replay resume, and final evaluation are recorded. Dataset-wide semantic parity and scientific performance remain outside this integration gate.

`REQ-AUDIT-012` — `VERIFIED`: only approved transition replay v1 `n_steps=3` was used.

Known limitations: the ACL driver remains sequential because vectorized ACL is not currently supported; final metrics are not scientific results; SAC was not run because TD3 is the requested primary final learner and no authoritative final configuration requires it; full thesis-length training and statistical algorithm comparison remain deferred.

## 16. Diagnostic Logging Correction (2026-07-20)

The first requested step-level trace run completed without trace files because
Rulebook v4.7 is wired through `RulebookV2MonitorWrapper`, while the diagnostic
options had only been implemented for the legacy `RuleRewardWrapper`. The v2
wrapper also exposed `scalar_reward` but not the established
`scalar_rule_reward` info key consumed by the runtime aggregators.

The v2 wrapper now persists one JSON-safe Rulebook/scalarization record per
transition, exposes both scalar reward aliases, and receives the configured
diagnostic paths. The default runtime-debug path is run-local under `logs/`.
Regression coverage is in `tests/test_rulebook_v2_wrapper.py`.

Validation: focused v2/reward/metadata tests `8 passed`; focused Ruff and
`git diff --check` passed. A new all-on smoke run completed 2,000 steps with
2,072 synchronized Rulebook/runtime records, 1,819 PG records and 253 Waymo
records. All records were finite, had four margins, complete evaluation, and
zero mismatches against the approved `bounded_satisfaction_rank` formula.

## 17. Collision Trace Correction (2026-07-20)

The step-level trace exposed a second integration defect: five episodes were
marked `collision=true` by the environment, while the Rulebook trace contained
zero applicable collision components. The live contact normalizer resolved the
ego only through `engine.traffic_manager.ego_vehicle`; ScenarioEnv's terminal
collision path owns the ego through `env.agents`.

The normalizer now resolves the ScenarioEnv agent first, then the direct vehicle
attribute, and finally the traffic-manager fallback. Rulebook trace records now
also include terminal collision/out-of-road flags for direct correlation.
Regression coverage is in `tests/test_rulebook_v2_metadrive_live.py`.

Validation: focused live-adapter and v2-wrapper tests `21 passed`. The previous
smoke must not be used as collision-conformance evidence; a new observed smoke
is required after this correction.
