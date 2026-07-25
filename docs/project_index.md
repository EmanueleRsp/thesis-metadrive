# Project Document Authority Index

## Purpose And Current Status

This index prevents an apparently newer, exploratory, or implementation-tracking
document from being mistaken for an approved scientific contract.

- Last repository inspection: 2026-07-24
- Index status: `CURRENT_WITH_DOCUMENTED_GAPS`
- Approval evidence: explicit user confirmations recorded on 2026-07-16,
  2026-07-17, 2026-07-21, and 2026-07-24
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
| Rulebook v2, R2 clearance amendment | `specifications/rulebook_v4.8_specification.md`; version `4.8`; amends only v4.7 §6.4 (vehicle/static clearance → scoped lateral-RSS / diagnostic-only) | `AUTHORITATIVE` for the amended §6.4 subset; explicit user approval 2026-07-24, recorded by ADR-025 | `implementation/r2_lateral_rss_clearance_v4.8_exec_plan.md`; `VERIFIED`; `rss_lateral` component, VRU-only clearance, aggregation/registry wiring, and REQ-R2-06 regression test implemented and tested (306-test rulebook suite and `make rulebook-v2-check` pass) | Known limitation: no R2 sub-metric other than TTC catches a static obstacle whose geometry converges beyond the TTC horizon (AC-R2-06, accepted by design) |
| Rulebook scalarization v1.0 | `specifications/rulebook_scalarization_v1.0_specification.md`; ID `SCAL-V1.0`, version `1.0` | `AUTHORITATIVE`; explicit user approval 2026-07-17 | `implementation/rulebook_scalarization_v1.0_exec_plan.md`; implementation in progress | Configurable scalarizer for Rulebook margins and scalar RL baselines |
| Perception-bounded semantic observation v1.2 | `specifications/observation_v1.2_specification.md`; ID `OBS-V1.2`, version `1.2-perception-bounded` | `AUTHORITATIVE`; explicit user approval 2026-07-21, recorded by ADR-022 | `implementation/perception_bounded_semantic_observation_v1.2_exec_plan.md`; M0–M6 complete; physical first-hit/signal gates, causal history, source-limited unknowns, encoder/checkpoint path, and provider-backed smoke verified | Ideal semantic tracking remains an explicit baseline limitation; calibrated stochastic tracking is deferred |
| Automatic curriculum learning | `specifications/automatic_curriculum_learning_v1_specification.md`; version v1 amended by §28 / ADR-014 | `AUTHORITATIVE`; original approval 2026-07-16, ScenarioNet scalar amendment explicitly approved 2026-07-19 | `implementation/scenario_acl_implementation_plan.md`; its internal implementation stages v1/v2 are complete, v3 not started, and v4 deferred | Runtime learner smoke, live ScenarioNet/Rulebook wiring, resume validation, and final reconciliation before `VERIFIED` |
| ScenarioNet integration | `specifications/scenarionet_integration_v1.1_specification.md`; version `1.1` | `AUTHORITATIVE`; explicit user approval 2026-07-16 | `implementation/scenarionet_integration_spec_v1.1_exec_plan.md`; `IN_PROGRESS` | v1 artifacts and implementation require reconciliation against v1.1 before `VERIFIED` |
| RL baselines | `specifications/rl_baselines_v1_specification.md`; ID `RL-BASELINES`, version `1.0` | `AUTHORITATIVE`; explicit user approval 2026-07-20 | `implementation/run_profile_td3_diagnostics_exec_plan.md`; `IN_PROGRESS`; fork-backed PPO/TD3/SAC baseline wiring and duration-profile matrix are partially reconciled; SB3 deviation ledger recorded in `implementation/sb3_fork_deviation_ledger.md` | Resolved end-to-end presets, manifest save/load path, ACL formula reconciliation, GPU learner smoke, and full acceptance matrix remain |
| Perception-bounded encoder v1.1 | `specifications/encoder_v1.1_specification.md`; ID `ENC-V1.1`, version `1.1-perception-bounded` | `AUTHORITATIVE`; explicit user approval 2026-07-21, recorded by ADR-022 | `implementation/perception_bounded_semantic_observation_v1.2_exec_plan.md`; encoder, SB3 bridge, checkpoint schema identity, and provider-backed smoke complete | No migration path to legacy OBS/ENC checkpoints by design |
| Transition-level replay v1 | `specifications/transition_replay_v1_specification.md`; ID `TRANSITION-REPLAY`, version `1.0` | `AUTHORITATIVE`; explicit user approval 2026-07-17 | `implementation/transition_replay_v1.0_exec_plan.md`; `IN_PROGRESS`; configuration, N-step, canonical collection, PER, persistence, and reward compatibility implemented | Focused replay matrix and current Hydra preset matrix pass; source-backed learner smoke and checkpoint/resume remain pending |
| Lexicographic/distributional RL | No dedicated approved specification found | `MISSING` | No authority can be inferred from literature or exploratory documents | Approved algorithms, interfaces, and acceptance criteria |
| Evaluation and algorithm comparison protocol | `specifications/evaluation_protocol_v1.0_specification.md`; ID `EVAL-PROTOCOL`, version `1.0` | `AUTHORITATIVE`; explicit user approval 2026-07-24 across four review passes (seeds, checkpoint policy, uncertainty convention, data-abort validity, primary metrics incl. R1--R3/R4 split, qualitative selection, ablation scope, candidate-protocol supersession, clean-tree policy, canonical analysis entry point, and the PPO end-of-budget atomic-rollout-boundary decision `DEC-015`); amended 2026-07-25 (optional opt-in confidence interval alongside mandatory mean/SD, `DEC-003`; no-backfill data-abort policy reconsidered and reconfirmed, `DEC-004`; tracked-subset GIF rendering mechanism added to qualitative selection, `DEC-014`) | `implementation/evaluation_protocol_v1.0_exec_plan.md`; `IN_PROGRESS` (see registry entry below) | One residual gap remains: `REQ-016` full end-to-end analysis regeneration not run against a real multi-condition comparison block. `REQ-008` (applicability-aware R1--R3 aggregation) implemented 2026-07-25 (Milestone 10). All other requirements implemented and live- or unit-verified (§15 of the ExecPlan) |
| Experimental and reporting protocols | `protocols/algorithm_comparison_protocol.md`, `protocols/csv_evaluation_objectives.md`, and `protocols/live_eval_video_protocol.md`; no versions declared | `CANDIDATE`; normative statistical/seed content of `algorithm_comparison_protocol.md` (10-seed protocol, 95% CI formula, reward×curriculum ablation framing) superseded by `EVAL-PROTOCOL` v1.0; `csv_evaluation_objectives.md` retained as the subordinate implementation-level CSV schema `EVAL-PROTOCOL` references; `live_eval_video_protocol.md` retained as implementation guidance, amended by `EVAL-PROTOCOL` REQ-014's post-hoc qualitative-selection scheme. See `EVAL-PROTOCOL` `DEC-008`. | Operational commands exist in `setup/validation_commands.md` | Implementation reconciliation against `EVAL-PROTOCOL`; none of the three candidate documents is deleted, only superseded/consolidated/amended in normative scope |

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
- OBS-V1.1 and ENC-V1.0 remain historical, reproducible contracts for their
  existing experiments. ADR-022 supersedes them only for the selected semantic
  observation/encoder implementation path; it does not change legacy runtime
  modes implicitly.
- Rulebook v4.8 (`specifications/rulebook_v4.8_specification.md`) amends only
  v4.7 §6.4 (R2 vehicle/static clearance), replacing the vehicle sub-metric
  with a scoped lateral-RSS metric and demoting static clearance to
  diagnostic-only per ADR-025, explicit user approval 2026-07-24. v4.7
  remains authoritative and unchanged for every other section (§2-§4, §6.1-
  §6.3, §6.5-§6.6, §7-§17), including RSS longitudinal, TTC, VRU clearance,
  and vehicle-yield.
- `EVAL-PROTOCOL` v1.0 (`specifications/evaluation_protocol_v1.0_specification.md`)
  is approved and authoritative, following explicit user approval on
  2026-07-24. It replaces the normative statistical/seed content of
  `protocols/algorithm_comparison_protocol.md` (10-seed protocol, the
  `1.96*s/sqrt(n)` 95% CI formula, and the reward-setting x curriculum
  ablation framing), consolidates `protocols/csv_evaluation_objectives.md`
  as a subordinate implementation-level CSV schema, and amends
  `protocols/live_eval_video_protocol.md` only where its REQ-014 post-hoc
  qualitative-selection scheme changes it. All three candidate documents
  are retained for traceability, not deleted. Approval of `EVAL-PROTOCOL`
  is a scientific-contract decision only; the repository implementation
  does not yet conform (no ExecPlan exists yet; see the ExecPlan Registry
  entry below and `EVAL-PROTOCOL` §11 for the specific verified gaps).

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
| `decisions/ADR-018-parallel-evaluation-and-test.md` | `APPROVED` | Explicit user instruction to implement synchronous evaluation/test parallelism on 2026-07-21 | Spawned evaluation/test workers, deterministic reset/reduction, ScenarioNet/ACL scheduling, and live video rendering |
| `decisions/ADR-019-asynchronous-evaluation-queue.md` | `APPROVED` | Explicit user confirmation of FIFO asynchronous ordinary/ACL diagnostics, staged barrier, fatal errors, and parent Rich UI on 2026-07-21 | Immutable evaluator snapshots, FIFO queue, fatal propagation, and training/final-test integration |
| `decisions/ADR-020-evaluation-video-diagnostics.md` | `APPROVED` | Explicit user approval of compact live/replay diagnostic overlays on 2026-07-21 | Shared GIF annotator, selected/cumulative reward, Rulebook subrules, route/target and optional actor outlines |
| `decisions/ADR-021-source-bounded-reactive-traffic.md` | `APPROVED` | Explicit user approval to retain reactive traffic only through source-track validity on 2026-07-21 | Source-bounded ScenarioNet IDM traffic lifecycle; no fallback traffic after an actor record ends |
| `decisions/ADR-022-perception-bounded-semantic-observation.md` | `APPROVED` | Explicit user approval to proceed on 2026-07-21 | OBS-V1.2/ENC-V1.1, planar first-hit LiDAR, symbolic signal visibility, causal memory, and strict preflight policy |
| `decisions/ADR-023-scoped-vehicle-yield-source-contract.md` | `APPROVED` | Explicit user instruction to implement the proposed source-bound vehicle-yield contract on 2026-07-21 | Rulebook v4.7 §7.9 causal live wiring; explicit pairwise and roundabout metadata; no geometric-priority fallback |
| `decisions/ADR-024-runtime-scenario-data-abort.md` | `APPROVED` | Explicit user approval of the typed data-abort/quarantine contract on 2026-07-23 | Typed runtime scenario non-evaluability, valid-prefix truncation, vector selective reset, ACL exclusion, run-local quarantine, evaluation coverage and comparison exclusions |
| `decisions/ADR-025-r2-lateral-rss-clearance-replacement.md` | `APPROVED` | Explicit user instruction "Approvo la specifica v4.8, procedi con l'implementazione" on 2026-07-24 | Rulebook v4.7 §6.4 R2 clearance replacement: vehicle clearance → scoped lateral-RSS metric, static clearance → diagnostic-only, VRU clearance unchanged, new R2 aggregation, `pre_state` snapshot |

## ExecPlan Registry

| Feature | Specification | ExecPlan | Reported status | Last document update |
|---|---|---|---|---|
| Rulebook v2 | `specifications/rulebook_v4.7_specification.md` | `implementation/rulebook_v2_implementation_plan.md` | `COMPLETED; final reconciliation before VERIFIED` | 2026-07-17 |
| Rulebook v2, R2 lateral-RSS clearance replacement | `specifications/rulebook_v4.8_specification.md` | `implementation/r2_lateral_rss_clearance_v4.8_exec_plan.md` | `VERIFIED`; ADR-025 approved; scoped lateral-RSS component, VRU-only clearance, aggregation/registry wiring, and REQ-R2-06 regression test implemented; full rulebook suite (306 tests) and `make rulebook-v2-check` pass | 2026-07-24 |
| Vehicle-yield pre/post-state conformance | `specifications/rulebook_v4.7_specification.md` §7.9 (DEC-005) | `implementation/vehicle_yield_pre_post_state_conformance_v4.7_exec_plan.md` | `VERIFIED`; conformance restored, `make rulebook-v2-check` and full rulebook suite passed | 2026-07-24 |
| Rulebook scalarization v1.0 | `specifications/rulebook_scalarization_v1.0_specification.md` | `implementation/rulebook_scalarization_v1.0_exec_plan.md` | `IMPLEMENTATION IN PROGRESS` | 2026-07-17 |
| Transition-level replay v1 | `specifications/transition_replay_v1_specification.md` | `implementation/transition_replay_v1.0_exec_plan.md` | `IN_PROGRESS` | 2026-07-17 |
| Rulebook v2 catalog filter parallelization | Rulebook v2 §15.11; ScenarioNet v1 §17/§24 | `implementation/rulebook_v2_catalog_filter_parallelization_exec_plan.md` | `IN_PROGRESS` | 2026-07-16 |
| ScenarioNet catalog build parallelization | Historical ScenarioNet v1 | `implementation/scenarionet_catalog_build_parallelization_exec_plan.md` | `IMPLEMENTED`; reconciliation under v1.1 pending | 2026-07-16 |
| ScenarioNet pipeline integrity and restructure v2 | Historical ScenarioNet v1 | `implementation/scenarionet_pipeline_restructure_v2_exec_plan.md` | `SUPERSEDED` by v1.1 planning | 2026-07-16 |
| Semantic observation and encoder v1 | Historical `specifications/observation_v1.1_specification.md`; `specifications/encoder_v1.0_specification.md` | `implementation/semantic_observation_encoder_v1_exec_plan.md` | `SUPERSEDED` for selected new semantic development by ADR-022; retain the reported v1.1 implementation record for reproducibility | 2026-07-21 |
| Perception-bounded semantic observation and encoder v1.2/v1.1 | `specifications/observation_v1.2_specification.md`; `specifications/encoder_v1.1_specification.md` | `implementation/perception_bounded_semantic_observation_v1.2_exec_plan.md` | `IMPLEMENTED`; M0–M6 complete, including canonical provider-backed `semantic_v3`/`lq_v3` training smoke | 2026-07-22 |
| Scenario ACL | `specifications/automatic_curriculum_learning_v1_specification.md` | `implementation/scenario_acl_implementation_plan.md` | Internal stages v1/v2 reported complete; later stages incomplete/deferred | Date not declared in metadata |
| Scenario ACL deterministic vectorized execution | `specifications/automatic_curriculum_learning_v1_specification.md` §12/§28; `specifications/scenarionet_integration_v1.1_specification.md` §23 | `implementation/scenario_acl_vectorized_execution_v1_exec_plan.md` | `COMPLETED`; DEC-VEC-001--006 and resume restart policy approved by ADR-016; focused tests, real TD3/PPO fixture smokes, and checkpoint resume passed | 2026-07-20 |
| Subprocess worker failure diagnostics v1 | `specifications/scenarionet_integration_v1.1_specification.md` §23--§24/§27.2 | `implementation/subproc_worker_failure_diagnostics_v1_exec_plan.md` | `IMPLEMENTED`; worker failures now report command, slot, PID, exit code and remote Python traceback context without changing sampling or execution semantics | 2026-07-21 |
| Subprocess worker fail-fast v2 | `specifications/scenarionet_integration_v1.1_specification.md` §23--§24; `specifications/rl_baselines_v1_specification.md` §8; ADR-019 | `implementation/subproc_worker_fail_fast_v2_exec_plan.md` | `IMPLEMENTED`; readiness-based response collection surfaces any worker failure without waiting behind a slow slot, then reaps owned workers without changing sampling or execution semantics | 2026-07-21 |
| Route projection failure diagnostics v1 | `specifications/observation_v1.1_specification.md` §7.4/§10/§13.5; `specifications/scenarionet_integration_v1.1_specification.md` §17.2/§24 | `implementation/route_projection_failure_diagnostics_v1_exec_plan.md` | `IMPLEMENTED`; failure reports identify live actor/ego/route elevation relationships without changing projection or feature semantics | 2026-07-21 |
| Live training reliability v1 | `specifications/observation_v1.2_specification.md`; Rulebook v4.7 §7.6; ACL v1 §§12--13/§28; RL Baselines v1 | `implementation/live_training_reliability_v1_exec_plan.md` | `IN_PROGRESS`; static optional-map projection, PPO device/event parity, ACL diagnostics, and stale signal-eligibility investigation | 2026-07-23 |
| ScenarioNet ACL EMA selection v1.1 | `specifications/automatic_curriculum_learning_v1.1_specification.md` (`ACL-SN-EMA-001`) | `implementation/automatic_curriculum_learning_v1.1_exec_plan.md` | `APPROVED; implementation verified`; EMA arm scores, temperature sampling, 40/60 Generate/Replay, and LP-only replay | 2026-07-23 |
| Source-bounded reactive traffic v1 | `specifications/scenarionet_integration_v1.1_specification.md` §17/§23--24; ADR-021 | `implementation/source_bounded_reactive_traffic_v1_exec_plan.md` | `IMPLEMENTED`; approved reactive traffic remains active only while the source track state is valid; focused lifecycle regression passed | 2026-07-21 |
| Deterministic parallel evaluation and final test | `specifications/rl_baselines_v1_specification.md`; `protocols/live_eval_video_protocol.md` | `implementation/parallel_evaluation_test_v1_exec_plan.md` | `IMPLEMENTED`; spawned validation/test workers, ordered metric reduction, ScenarioNet/ACL sequence preservation, live video integration, focused tests, full suite, and end-to-end smoke passed | 2026-07-21 |
| Asynchronous evaluation queue v1 | `specifications/rl_baselines_v1_specification.md` | `implementation/asynchronous_evaluation_v1_exec_plan.md` | `IMPLEMENTED`; ordinary and ACL diagnostic evaluation queued against immutable snapshots; staged evaluation and final test remain synchronous/separate; focused tests and smoke pass; unrelated full-suite fixture failure remains | 2026-07-21 |
| Evaluation video diagnostics v1 | `specifications/rl_baselines_v1_specification.md`; Rulebook v4.7; candidate live video protocol | `implementation/evaluation_video_diagnostics_v1_exec_plan.md` | `IN_PROGRESS`; approved compact shared live/replay annotation scope; implementation and validation pending | 2026-07-21 |
| ScenarioNet integration v1 | Historical `specifications/scenarionet_integration_v1_specification.md` | `implementation/scenarionet_integration_implementation_plan.md` | `SUPERSEDED`; retain for traceability | 2026-07-15 |
| ScenarioNet integration v1.1 | `specifications/scenarionet_integration_v1.1_specification.md` | `implementation/scenarionet_integration_spec_v1.1_exec_plan.md` | `IN_PROGRESS` | 2026-07-16 |
| ScenarioNet existing-source rebuild | `specifications/scenarionet_integration_v1.1_specification.md` | `implementation/scenarionet_existing_source_rebuild_exec_plan.md` | `IMPLEMENTED`; full no-cache eligibility revalidation, split/runtime rebuild, and frozen-index recreation target added | 2026-07-20 |
| ScenarioNet physical-road termination regression | `specifications/scenarionet_integration_v1.1_specification.md` | `implementation/out_of_road_boundary_termination_bugfix_exec_plan.md` | `IN_PROGRESS`; the 2026-07-21 live run exposed a missing-contact false negative after the prior boundary-only false-positive fix | 2026-07-21 |
| Scalar autonomous-driving pipeline audit and completion | Authoritative ScenarioNet v1.1, ACL v1 §28, Rulebook v4.7, scalarization v1.0, observation v1.1, encoder v1.0, and transition replay v1.0 | `implementation/scalar_autonomous_driving_pipeline_audit_exec_plan.md` | `VERIFIED` for the final smoke integration gate; canonical read-only PG/Waymo, Rulebook/scalarization, semantic `(2541,)`/LQ, TD3/PER/n=3, ACL generation/replay, checkpoint/replay/RNG persistence and resume, and final evaluation passed. Full thesis-length/statistical validation remains out of scope; SAC was not required by the final TD3 configuration. | 2026-07-19 |
| Documentation structure | User instructions dated 2026-07-16 | `implementation/repository_documentation_restructure_exec_plan.md` | `VERIFIED` | 2026-07-16 |
| Repository quality commands | User process decision dated 2026-07-16 | `implementation/repository_quality_commands_exec_plan.md` | `VERIFIED` | 2026-07-16 |
| PyTorch cross-architecture compatibility | User request dated 2026-07-19; ADR-013 | `implementation/pytorch_cross_architecture_compatibility_exec_plan.md` | `IN_PROGRESS`; approved 2.9.1 pin and ARM64 CUDA validator exception pending full validation | 2026-07-19 |
| Runtime scenario data-abort v1 | Rulebook v4.7 RSA-1; Transition Replay v1 RSA-1; ACL v1 RSA-1; RL Baselines v1 RSA-1; ScenarioNet v1.1 RSA-1; ADR-024 | `implementation/runtime_scenario_data_abort_v1_exec_plan.md` | `IMPLEMENTED`; typed data-abort, PER boundary closure, a prefix-preserving PPO/GAE boundary (`MaskedRolloutBuffer`, no more full-rollout discard), evaluation-path exclusion, ACL/non-ACL quarantine persistence across resume, and forensic JSONL wiring implemented and tested (buffer-, backend-, and loop-level, including a real SB3 PPO backend test); uniform replay remains deliberately fatal pending a larger safe N-step mutation; representative smoke not yet run | 2026-07-23 |
| Evaluation and algorithm comparison protocol v1.0 | `specifications/evaluation_protocol_v1.0_specification.md`, `EVAL-PROTOCOL` v1.0 | `implementation/evaluation_protocol_v1.0_exec_plan.md` | `IN_PROGRESS`; Milestones 1--6, 8, 9, 10 complete, live- or unit-verified (incl. Milestone 5's PPO atomic-boundary/`DEC-015` overshoot path live-verified against a real SB3 PPO run, Milestone 2's panel manifest generated against the real ScenarioNet catalog and hash-recorded in run metadata, Milestone 9's tracked-subset feature-diversity GIF selection/rendering and category-taxonomy reconciliation, Milestone 10's applicability-aware R1--R3 seed-level aggregation for `REQ-008`); `REQ-016` full end-to-end analysis regeneration against real aggregated artifacts not run; full suite run 970/974 passing, the same 4 pre-existing unrelated failures (Rulebook v2 route geometry, a concurrent unrelated session's in-progress work) | 2026-07-25 |

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
