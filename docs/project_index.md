# Project Document Authority Index

## Purpose And Current Status

This index prevents an apparently newer, exploratory, or implementation-tracking
document from being mistaken for an approved scientific contract.

- Last repository inspection: 2026-07-16
- Index status: `CURRENT_WITH_DOCUMENTED_GAPS`
- Approval evidence: explicit user confirmations recorded on 2026-07-16
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
| Rulebook v2 | `specifications/rulebook_v4.6_specification.md`; version `4.6-final-implementation-complete` | `AUTHORITATIVE`; user confirmation 2026-07-16 | `implementation/rulebook_v2_implementation_plan.md`; reports implementation in progress | Final reconciliation required before `VERIFIED` |
| Semantic observation v1.1 | No dedicated specification supplied | `MISSING` | Implementation exists in `src/thesis_rl/envs/observations/semantic_state.py`, `conf/obs/semantic_state.yaml`, and `tests/test_semantic_state_observation.py`; no dedicated ExecPlan found | User will provide the approved v1.1 specification through `../incoming/` |
| Automatic curriculum learning | `specifications/automatic_curriculum_learning_v1_specification.md`; version v1 | `AUTHORITATIVE`; version and authority confirmed by the user on 2026-07-16 | `implementation/scenario_acl_implementation_plan.md`; its internal implementation stages v1/v2 are complete, v3 not started, and v4 deferred | Final reconciliation required before `VERIFIED` |
| ScenarioNet integration | `specifications/scenarionet_integration_spec_v1.1.md`; version `1.1` | `AUTHORITATIVE`; explicit user approval 2026-07-16 | `implementation/scenarionet_integration_spec_v1.1_exec_plan.md`; `IN_PROGRESS` | v1 artifacts and implementation require reconciliation against v1.1 before `VERIFIED` |
| RL baselines | No dedicated approved specification found | `MISSING` | PPO, SAC, and TD3 configurations and tests exist; the local SB3 submodule is pinned at commit `6a196a60c7df3550ac5832caad54ef8dce9a6f31` | Approved behavioral specification and accepted deviations from upstream |
| Encoder architecture | No dedicated approved specification found | `MISSING` | Encoder code and configuration exist | Approved encoder contract and compatibility requirements |
| Replay-buffer extensions | No dedicated approved specification found | `MISSING` | No authority can be inferred from implementation notes | Approved N-step/PER scope and algorithm-specific semantics |
| Lexicographic/distributional RL | No dedicated approved specification found | `MISSING` | No authority can be inferred from literature or exploratory documents | Approved algorithms, interfaces, and acceptance criteria |
| Experimental and reporting protocols | `protocols/algorithm_comparison_protocol.md`, `protocols/csv_evaluation_objectives.md`, and `protocols/live_eval_video_protocol.md`; no versions declared | `CANDIDATE` | Operational commands exist in `setup/validation_commands.md` | Exact approved versions and authority confirmation |

Document paths in the document and implementation columns are relative to
`docs/`. Source, test, and configuration paths are relative to the repository
root.

## Historical Material

- Rulebook version `4.6-final-implementation-complete` remains the canonical
  approved identifier. Earlier archived Rulebook labels used a different
  increment convention, so their numeric relationship must not be interpreted as
  semantic-version precedence.
- `archive/plans/rulebook_v1_specification.md` is archived historical material.
  No inspected authority record formally establishes its supersession chain.
- The temporary names `rulebook_v4.4_final_corrected(1).md`,
  `rulebook_v4.4_final.md`, `rulebook_v4.1_final_updated.md`, and
  `observation_spec_v1.0_final_implementation_complete.md` were not found.
- Do not create a supersession relationship from these names alone.
- ScenarioNet integration v1 is superseded by
  `specifications/scenarionet_integration_spec_v1.1.md` following the explicit
  user approval recorded on 2026-07-16. Its specification and implementation
  plan remain historical traceability records.

## Decisions

| ADR | Status | Approval evidence | Affected scope |
|---|---|---|---|
| `decisions/ADR-001-scenarionet-v1-1-dataset-policy.md` | `APPROVED` | Explicit user approval of ScenarioNet Integration v1.1 on 2026-07-16 | ScenarioNet v1.1 dataset, ACL arm, horizon, and eligibility policy |

## ExecPlan Registry

| Feature | Specification | ExecPlan | Reported status | Last document update |
|---|---|---|---|---|
| Rulebook v2 | `specifications/rulebook_v4.6_specification.md` | `implementation/rulebook_v2_implementation_plan.md` | `IN_PROGRESS` | 2026-07-16 |
| Rulebook v2 catalog filter parallelization | Rulebook v2 §15.11; ScenarioNet v1 §17/§24 | `implementation/rulebook_v2_catalog_filter_parallelization_exec_plan.md` | `IN_PROGRESS` | 2026-07-16 |
| ScenarioNet catalog build parallelization | Historical ScenarioNet v1 | `implementation/scenarionet_catalog_build_parallelization_exec_plan.md` | `IMPLEMENTED`; reconciliation under v1.1 pending | 2026-07-16 |
| ScenarioNet pipeline integrity and restructure v2 | Historical ScenarioNet v1 | `implementation/scenarionet_pipeline_restructure_v2_exec_plan.md` | `SUPERSEDED` by v1.1 planning | 2026-07-16 |
| Semantic observation | `MISSING` | `MISSING` | Implementation exists; specification reconciliation unavailable | Not established |
| Scenario ACL | `specifications/automatic_curriculum_learning_v1_specification.md` | `implementation/scenario_acl_implementation_plan.md` | Internal stages v1/v2 reported complete; later stages incomplete/deferred | Date not declared in metadata |
| ScenarioNet integration v1 | Historical `specifications/scenarionet_integration_v1_specification.md` | `implementation/scenarionet_integration_implementation_plan.md` | `SUPERSEDED`; retain for traceability | 2026-07-15 |
| ScenarioNet integration v1.1 | `specifications/scenarionet_integration_spec_v1.1.md` | `implementation/scenarionet_integration_spec_v1.1_exec_plan.md` | `IN_PROGRESS` | 2026-07-16 |
| Documentation structure | User instructions dated 2026-07-16 | `implementation/repository_documentation_restructure_exec_plan.md` | `VERIFIED` | 2026-07-16 |
| Repository quality commands | User process decision dated 2026-07-16 | `implementation/repository_quality_commands_exec_plan.md` | `VERIFIED` | 2026-07-16 |

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
