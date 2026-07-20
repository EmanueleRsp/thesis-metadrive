# Project Document Authority Index

## Purpose And Current Status

This index prevents an apparently newer, exploratory, or implementation-tracking
document from being mistaken for an approved scientific contract.

- Last repository inspection: 2026-07-20
- Index status: `CURRENT_WITH_DOCUMENTED_GAPS`
- Approval evidence: explicit user confirmations recorded on 2026-07-16 and 2026-07-17
- Rule: repository evidence establishes paths, versions, links, and reported
  implementation status; explicit user approval establishes scientific authority.

## Status Vocabulary

- `AUTHORITATIVE`: explicitly approved by the user and selected for use.
- `CANDIDATE`: relevant document exists but authority is not established.
- `MISSING`: no matching document was found.
- `IMPLEMENTED`: implementation is reported complete, without implying full
  reconciliation.
- `VERIFIED`: mandatory validation and specification reconciliation are recorded.
- `SUPERSEDED`: an approved authority record explicitly identifies a replacement.

## Scientific And Functional Documents

| Area | Exact authoritative or candidate document | Authority | Implementation record | Remaining gap |
|---|---|---|---|---|
| Rulebook v2 | `specifications/rulebook_v4.7_specification.md`; version `4.7-final-implementation-complete` | `AUTHORITATIVE`; explicit user approval 2026-07-17 | `implementation/rulebook_v2_implementation_plan.md`; F11 `COMPLETATA`; causal CTRV wiring, conformance suite and live Waymo smoke implemented | Final document/diff reconciliation required before `VERIFIED` |
| Rulebook scalarization v1.0 | `specifications/rulebook_scalarization_v1.0_specification.md`; ID `SCAL-V1.0`, version `1.0` | `AUTHORITATIVE`; explicit user approval 2026-07-17 | `implementation/rulebook_scalarization_v1.0_exec_plan.md`; implementation in progress | Configurable scalarizer for Rulebook margins and scalar RL baselines |
| Semantic observation v1.1 | `specifications/observation_v1.1_specification.md`; ID `OBS-V1.1`, version `1.1-final-implementation-complete`, amended 2026-07-17 by ADR-004 | `AUTHORITATIVE`; explicit user approvals 2026-07-16 and 2026-07-17 | `implementation/semantic_observation_encoder_v1_exec_plan.md`; `IN_PROGRESS`; v1.1 schema, strict `(2541,)` adapter, causal batch builder, reset installation, post-commit observation refresh, causal-context boundary, assigned-route persistence, metadata-only adapters, causal LiDAR wiring, and checkpoint manifest primitives exist | End-to-end checkpoint caller migration, full semantic sensor smoke matrix, visual PG/Waymo validation, and final reconciliation remain before `VERIFIED` |
| Automatic curriculum learning | `specifications/automatic_curriculum_learning_v1_specification.md`; version v1 amended by §28 / ADR-014 | `AUTHORITATIVE`; original approval 2026-07-16, ScenarioNet scalar amendment explicitly approved 2026-07-19 | `implementation/scenario_acl_implementation_plan.md`; its internal implementation stages v1/v2 are complete, v3 not started, and v4 deferred | Runtime learner smoke, live ScenarioNet/Rulebook wiring, resume validation, and final reconciliation before `VERIFIED` |
| ScenarioNet integration | `specifications/scenarionet_integration_v1.1_specification.md`; version `1.1` | `AUTHORITATIVE`; explicit user approval 2026-07-16 | `implementation/scenarionet_integration_spec_v1.1_exec_plan.md`; `IN_PROGRESS` | v1 artifacts and implementation require reconciliation against v1.1 before `VERIFIED` |
| RL baselines | `specifications/rl_baselines_v1_specification.md`; ID `RL-BASELINES`, version `1.0` | `AUTHORITATIVE`; explicit user approval 2026-07-20 | `implementation/run_profile_td3_diagnostics_exec_plan.md`; `IN_PROGRESS`; fork-backed PPO/TD3/SAC baseline wiring and duration-profile matrix are partially reconciled; SB3 deviation ledger recorded in `implementation/sb3_fork_deviation_ledger.md` | Resolved end-to-end presets, manifest save/load path, ACL formula reconciliation, GPU learner smoke, and full acceptance matrix remain |
| Encoder architecture v1.0 | `specifications/encoder_v1.0_specification.md`; ID `ENC-V1.0`, version `1.0-final-implementation-complete` | `AUTHORITATIVE`; explicit user approval 2026-07-16 | `implementation/semantic_observation_encoder_v1_exec_plan.md`; `IN_PROGRESS`; schema-driven MLP/LQ core, strict SB3 bridge validation, and explicit checkpoint generation validation APIs are implemented | Complete runtime checkpoint publication/load migration and end-to-end smoke matrix before `VERIFIED` |
| Transition-level replay v1 | `specifications/transition_replay_v1_specification.md`; ID `TRANSITION-REPLAY`, version `1.0` | `AUTHORITATIVE`; explicit user approval 2026-07-17 | `implementation/transition_replay_v1.0_exec_plan.md`; `IN_PROGRESS`; configuration, N-step, canonical collection, PER, persistence, and reward compatibility implemented | Focused replay matrix and current Hydra preset matrix pass; source-backed learner smoke and checkpoint/resume remain pending |
| Lexicographic/distributional RL | No dedicated approved specification found | `MISSING` | No authority can be inferred from literature or exploratory documents | Approved algorithms, interfaces, and acceptance criteria |
| Experimental and reporting protocols | `protocols/algorithm_comparison_protocol.md`, `protocols/csv_evaluation_objectives.md`, and `protocols/live_eval_video_protocol.md`; no versions declared | `CANDIDATE` | Operational commands exist in `setup/validation_commands.md` | Exact approved versions and authority confirmation |

Document paths in the document and implementation columns are relative to
`docs/`. Source, test, and configuration paths are relative to the repository
root.

## Historical Material

- Rulebook version `4.7-final-implementation-complete` is the canonical
  approved identifier; it supersedes Rulebook v4.6 for the selected rulebook
  scope. Earlier archived Rulebook labels used a different
  increment convention, so their numeric relationship must not be interpreted as
  semantic-version precedence.
- `specifications/rulebook_v4.6_specification.md` remains retained historical
  material for reproducibility of v4.6 experiments and is not the current
  implementation authority.
- `archive/plans/rulebook_v1_specification.md` is archived historical material.
  No inspected authority record formally establishes its supersession chain.
- The temporary names `rulebook_v4.4_final_corrected(1).md`,
  `rulebook_v4.4_final.md`, `rulebook_v4.1_final_updated.md`, and
  `observation_spec_v1.0_final_implementation_complete.md` were not found.
- Do not create a supersession relationship from these names alone.
- ScenarioNet integration v1 is superseded by
  `specifications/scenarionet_integration_v1.1_specification.md` following the explicit
  user approval recorded on 2026-07-16. Its specification and implementation
  plan remain historical traceability records.

## Decisions

| ADR | Status | Approval evidence | Affected scope |
|---|---|---|---|
| `decisions/ADR-001-scenarionet-v1-1-dataset-policy.md` | `APPROVED` | Explicit user approval of ScenarioNet Integration v1.1 on 2026-07-16 | ScenarioNet v1.1 dataset, ACL arm, horizon, and eligibility policy |
| `decisions/ADR-002-semantic-observation-and-encoder-contract.md` | `APPROVED` | Explicit user approval of observation v1.1 and encoder v1.0 on 2026-07-16 | Semantic observation, encoder architecture, SB3 integration, and checkpoint compatibility |
| `decisions/ADR-004-assigned-route-metadata-for-pg-and-waymo.md` | `APPROVED` | Explicit user approval on 2026-07-17 | Offline assigned-route metadata and runtime anti-leakage boundary for PG and Waymo |
| `decisions/ADR-003-causal-ctrv-conflict-zone-prediction.md` | `APPROVED` | Explicit user approval on 2026-07-17 | Causal filtered CTRV for vehicle conflict-zone occupancy, frozen defaults, history lifecycle, and experimental separation |
| `decisions/ADR-007-waymo-batch-throughput.md` | `APPROVED; amended by ADR-010` | Explicit user approval on 2026-07-17 | Waymo acquisition throughput: 64 unseen shards per cycle, with cap amended to 256 |
| `decisions/ADR-008-pg-compositional-replenishment.md` | `APPROVED; amended by ADR-009` | Explicit user approval on 2026-07-17 | Bounded PG composition replenishment trigger and two-cycle limit |
| `decisions/ADR-009-pg-targeted-replenishment.md` | `APPROVED` | Explicit user approval on 2026-07-17 | Targeted 1,750-candidate PG budget allocated by observed arm deficits; no parameter calibration in this cycle |
| `decisions/ADR-010-waymo-cap-expansion.md` | `APPROVED` | Explicit user approval on 2026-07-17 | Waymo cumulative cap increased to 256 unseen shards; batch size remains 64 |
| `decisions/ADR-011-rulebook-scalarization-v1.md` | `APPROVED` | Explicit user approval of SCAL-V1.0 on 2026-07-17 | Scalarization modes, rulebook adapters, default, reward interface, future replay semantics, and compatibility |
| `decisions/ADR-012-stratified-source-arm-split-allocation.md` | `APPROVED` | Explicit user approval on 2026-07-18 | ScenarioNet primary-split source-arm stratification while retaining exact global source totals |
| `decisions/ADR-013-pytorch-cross-architecture-validation.md` | `APPROVED` | Explicit user approval on 2026-07-19 | PyTorch 2.9.1 cross-architecture pin and narrowly validated ARM64 cuSPARSELt checker exception |
| `decisions/ADR-014-scenarionet-acl-learning-potential-only.md` | `APPROVED` | Explicit user approval on 2026-07-19 | ScenarioNet ACL usefulness equals learning potential; mutation is prohibited; Rulebook curriculum inputs are diagnostic-only |
| `decisions/ADR-016-scenario-acl-vectorized-execution.md` | `APPROVED` | Explicit user approval of DEC-VEC-001--DEC-VEC-005 and deterministic resume restart policy on 2026-07-20 | Parent-controlled selective reset, fresh-batch uniqueness, per-episode LP provenance, deterministic commit ordering, PPO partial-rollout persistence, and explicit active-slot restart on resume |
| `decisions/ADR-015-r1-pre-state-centerline-normal.md` | `APPROVED` | Explicit user approval on 2026-07-20 | R1 derives every ego-to-other normal from pre-state canonical-footprint centers, independent of Bullet manifold geometry |

## ExecPlan Registry

| Feature | Specification | ExecPlan | Reported status | Last document update |
|---|---|---|---|---|
| Rulebook v2 | `specifications/rulebook_v4.7_specification.md` | `implementation/rulebook_v2_implementation_plan.md` | `COMPLETED; final reconciliation before VERIFIED` | 2026-07-17 |
| Rulebook scalarization v1.0 | `specifications/rulebook_scalarization_v1.0_specification.md` | `implementation/rulebook_scalarization_v1.0_exec_plan.md` | `IMPLEMENTATION IN PROGRESS` | 2026-07-17 |
| Transition-level replay v1 | `specifications/transition_replay_v1_specification.md` | `implementation/transition_replay_v1.0_exec_plan.md` | `IN_PROGRESS` | 2026-07-17 |
| Rulebook v2 catalog filter parallelization | Rulebook v2 §15.11; ScenarioNet v1 §17/§24 | `implementation/rulebook_v2_catalog_filter_parallelization_exec_plan.md` | `IN_PROGRESS` | 2026-07-16 |
| ScenarioNet catalog build parallelization | Historical ScenarioNet v1 | `implementation/scenarionet_catalog_build_parallelization_exec_plan.md` | `IMPLEMENTED`; reconciliation under v1.1 pending | 2026-07-16 |
| ScenarioNet pipeline integrity and restructure v2 | Historical ScenarioNet v1 | `implementation/scenarionet_pipeline_restructure_v2_exec_plan.md` | `SUPERSEDED` by v1.1 planning | 2026-07-16 |
| Semantic observation and encoder v1 | `specifications/observation_v1.1_specification.md`; `specifications/encoder_v1.0_specification.md` | `implementation/semantic_observation_encoder_v1_exec_plan.md` | `IN_PROGRESS`; causal batch builder and committed observation lifecycle now implemented and focused-tested; checkpoint, smoke matrix, and visual reconciliation remain | 2026-07-17 |
| Scenario ACL | `specifications/automatic_curriculum_learning_v1_specification.md` | `implementation/scenario_acl_implementation_plan.md` | Internal stages v1/v2 reported complete; later stages incomplete/deferred | Date not declared in metadata |
| Scenario ACL deterministic vectorized execution | `specifications/automatic_curriculum_learning_v1_specification.md` §12/§28; `specifications/scenarionet_integration_v1.1_specification.md` §23 | `implementation/scenario_acl_vectorized_execution_v1_exec_plan.md` | `COMPLETED`; DEC-VEC-001--006 and resume restart policy approved by ADR-016; focused tests, real TD3/PPO fixture smokes, and checkpoint resume passed | 2026-07-20 |
| ScenarioNet integration v1 | Historical `specifications/scenarionet_integration_v1_specification.md` | `implementation/scenarionet_integration_implementation_plan.md` | `SUPERSEDED`; retain for traceability | 2026-07-15 |
| ScenarioNet integration v1.1 | `specifications/scenarionet_integration_v1.1_specification.md` | `implementation/scenarionet_integration_spec_v1.1_exec_plan.md` | `IN_PROGRESS` | 2026-07-16 |
| ScenarioNet existing-source rebuild | `specifications/scenarionet_integration_v1.1_specification.md` | `implementation/scenarionet_existing_source_rebuild_exec_plan.md` | `IMPLEMENTED`; full no-cache eligibility revalidation, split/runtime rebuild, and frozen-index recreation target added | 2026-07-20 |
| ScenarioNet boundary-only termination regression | `specifications/scenarionet_integration_v1.1_specification.md` | `implementation/out_of_road_boundary_termination_bugfix_exec_plan.md` | `IMPLEMENTED`; focused equivalent suite passed; provisioned-environment Ruff/pytest and live visual replay remain pending | 2026-07-20 |
| Scalar autonomous-driving pipeline audit and completion | Authoritative ScenarioNet v1.1, ACL v1 §28, Rulebook v4.7, scalarization v1.0, observation v1.1, encoder v1.0, and transition replay v1.0 | `implementation/scalar_autonomous_driving_pipeline_audit_exec_plan.md` | `VERIFIED` for the final smoke integration gate; canonical read-only PG/Waymo, Rulebook/scalarization, semantic `(2541,)`/LQ, TD3/PER/n=3, ACL generation/replay, checkpoint/replay/RNG persistence and resume, and final evaluation passed. Full thesis-length/statistical validation remains out of scope; SAC was not required by the final TD3 configuration. | 2026-07-19 |
| Documentation structure | User instructions dated 2026-07-16 | `implementation/repository_documentation_restructure_exec_plan.md` | `VERIFIED` | 2026-07-16 |
| Repository quality commands | User process decision dated 2026-07-16 | `implementation/repository_quality_commands_exec_plan.md` | `VERIFIED` | 2026-07-16 |
| PyTorch cross-architecture compatibility | User request dated 2026-07-19; ADR-013 | `implementation/pytorch_cross_architecture_compatibility_exec_plan.md` | `IN_PROGRESS`; approved 2.9.1 pin and ARM64 CUDA validator exception pending full validation | 2026-07-19 |

## Maintenance Rules

1. Register one authoritative specification for each selected feature version.
2. Record path, stable ID when available, version, approval evidence, status,
   related ADRs, and ExecPlan.
3. Never infer authority or supersession from a filename, date, or larger version.
4. Preserve historical documents needed for traceability.
5. Keep specification status distinct from implementation status.
6. `VERIFIED` requires mandatory validation and final reconciliation.
7. Templates guide future documents; they do not invalidate an approved legacy
   specification solely because its structure differs.
8. Update this index when authority, version, implementation status, or an
   applicable ADR changes.
