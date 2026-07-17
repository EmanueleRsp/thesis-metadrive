# ExecPlan: Rulebook Scalarization v1.0 Review And Implementation Handoff

## 1. Metadata

- Feature: Configurable rulebook scalarization for scalar RL baselines
- Plan ID: `SCAL-V1.0-PLAN`
- Specification path: `docs/specifications/rulebook_scalarization_v1.0_specification.md`
- Specification ID/version: `SCAL-V1.0`, `1.0`
- Specification authority: `APPROVED`; `Authoritative: YES`
- Plan status: `IMPLEMENTATION IN PROGRESS`
- Created: 2026-07-17
- Last updated: 2026-07-17
- Branch: verified from the current worktree; branch name not required for this review
- Related ADRs: `docs/decisions/ADR-011-rulebook-scalarization-v1.md`
- Owner: thesis repository maintainer
- Approval evidence: explicit user approval in this Codex conversation on 2026-07-17 (`"se non resta altro da dire sì"`).

## 2. Objective And Scope

The candidate proposes a deterministic scalar reward adapter for PPO, TD3, and
SAC over the ordered four-margin Rulebook v4.6/v4.7 output. It defines three
named modes, bounded-input validation, diagnostic preservation, downstream
scalar-return semantics, and run/checkpoint compatibility rules.

This ExecPlan maps the approved specification to the current repository and
freezes the implementation and validation strategy.

In scope:

- complete-document Definition-of-Ready review;
- verification of current v1 and v2 rulebook/reward paths, configurations, tests,
  checkpoint metadata, and learner buffers;
- traceability of `REQ-SCAL-*` to `AC-SCAL-*`;
- implementation of the pure scalarizer and wiring at the existing rulebook
  vector boundary;
- configuration-selected scalar reward delivery and diagnostic preservation;
- checkpoint/run metadata compatibility and mandatory tests.

Out of scope:

- N-step replay and PER implementations;
- changes to rulebook formulas, thresholds, internal aggregation, termination,
  or the true lexicographic learner;
- full three-mode training ablations.

## 3. Authoritative Requirements

The following requirements are transcribed from the candidate without redefining
its scientific behavior.

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-SCAL-001` | Support exactly three named modes and default to `bounded_satisfaction_rank`. | §6, REQ-SCAL-001 |
| `REQ-SCAL-002` | Preserve the historical scaled, tanh-normalized, uncentered sigmoid formula and require four valid legacy scales. | §6, REQ-SCAL-002; §7.2 |
| `REQ-SCAL-003` | Provide the bounded centered sigmoid mode with no rule scales/tanh and no categorical progress gate. | §6, REQ-SCAL-003; §7.4 |
| `REQ-SCAL-004` | Provide the default bounded satisfaction-rank mode and its per-transition satisfaction-pattern dominance behavior. | §6, REQ-SCAL-004; §7.5 |
| `REQ-SCAL-005` | Disable native environment reward mixing for conformant core runs. | §6, REQ-SCAL-005 |
| `REQ-SCAL-006` | Preserve algorithm-independent scalar reward semantics for PPO, TD3, and SAC. | §6, REQ-SCAL-006 |
| `REQ-SCAL-007` | Scalarize each transition before PPO/GAE, replay, N-step accumulation, and critic/PER use; retain the vector for ACL/evaluation. | §6, REQ-SCAL-007 |
| `REQ-SCAL-008` | Apply the formula identically to valid non-terminal, terminal, and truncated transitions. | §6, REQ-SCAL-008 |
| `REQ-SCAL-009` | Emit decomposed diagnostics sufficient to reconstruct the scalar reward. | §6, REQ-SCAL-009 |
| `REQ-SCAL-010` | Reject same-run resume when scalarization mode or relevant parameters differ, with explicit replay/rollout compatibility rules. | §6, REQ-SCAL-010 |

## 4. Current Repository Analysis

The labels below distinguish evidence from inference and pending decisions.

| Status | Verified fact and path | Consequence |
|---|---|---|
| `VERIFIED` | `src/thesis_rl/reward/managers/hybrid_rulebook_manager.py:HybridRulebookRewardManager.compute` implements the historical `tanh` + uncentered sigmoid + exponential-priority formula, but for a dynamic number of v1 rules. | This is a candidate legacy implementation path, not yet proven to be the complete historical experiment. |
| `VERIFIED` | `conf/reward/rulebook_defaults.yaml` defines `a=2.01`, `c=30.0`, `lambda_env=0.0`, `lambda_rule=1.0`, and named v1 scales; `conf/reward/scalar_reward.yaml` selects `behavior: scalar_reward`. | Current scalar mode already disables native mixing numerically, but through the old lambda contract rather than `SCAL-V1.0`. |
| `VERIFIED` | `src/thesis_rl/reward/managers/hybrid_rulebook_manager.py` uses `np.exp` directly and reports raw margins, tanh-bounded values, and generic metadata; it does not expose the candidate's mode, contribution tuple, continuous tie-breaker, or specification identity. | Stable sigmoid and decomposed diagnostics require new implementation work after approval. |
| `VERIFIED` | `src/thesis_rl/envs/wrappers.py:RuleRewardWrapper` selects `monitor_only`, `scalar_reward`, `hybrid`, or `lexicographic`; it passes `env_reward` into the manager and publishes generic `info` fields. | Existing v1 wrapper semantics are broader than the candidate's conformant core contract. |
| `VERIFIED` | `src/thesis_rl/runtime/wiring/builders.py:maybe_wrap_env_with_reward_manager` sends Rulebook v2 through `RulebookV2MonitorWrapper` before considering reward behavior. | v2 currently bypasses scalar reward selection. |
| `VERIFIED` | `src/thesis_rl/rulebook/v2/aggregation.py:aggregate_rulebook_result` emits the exact four margins in `MACRO_RULE_ORDER`, with `m_{1:3}` as negatives of bounded costs and `m_4` as progress. `RulebookResult.complete_evaluation` is enforced by `src/thesis_rl/rulebook/v2/monitor.py`. | The candidate's four-margin precondition matches v2, not the current v1 evaluator. |
| `VERIFIED` | `src/thesis_rl/rulebook/v2/wrapper.py:RulebookV2MonitorWrapper` adds the vector and diagnostics to `info` but returns the native environment reward. | Scalarization integration must define whether and where this wrapper changes. |
| `VERIFIED` | `conf/rulebook/selection.yaml` is a four-rule v1 configuration, but its rules are not the v2 four macro-rules; v1 rule functions can emit physical-unit values such as collision energy and speed margin. | Applying the candidate's bounded v2 validation to v1 `selection` would be incompatible. |
| `VERIFIED` | `tests/test_reward_manager.py`, `tests/test_reward_wrapper.py`, `tests/test_reward_runtime_wiring.py`, and `tests/test_hydra_preset_run_configs.py` protect current v1 formula, lambda mixing, wrapper selection, and config behavior. | Existing tests must be preserved or deliberately migrated only after approval. |
| `VERIFIED` | PPO stores `Transition.scalar_reward` in its rollout buffer; TD3/SAC store it in replay through `src/thesis_rl/agent/planners/algorithms/{ppo,td3,sac,ppo_sb3,td3_sb3,sac_sb3}.py`. | Per-transition scalar delivery exists as an internal learner contract. |
| `VERIFIED` | The repository has PPO rollout `n_steps` and replay buffers, but no configured N-step return implementation or PER implementation was found for the baseline paths. | `REQ-SCAL-007` is partly a semantic contract for absent extensions; exact scope must be stated. |
| `VERIFIED` | `src/thesis_rl/contracts/checkpoint_manifest.py:CheckpointManifest` contains observation/encoder/algorithm/seed compatibility fields but no scalarization identity or parameters. `src/thesis_rl/runtime/io/metadata.py` records only generic reward type/behavior/rulebook config. | `REQ-SCAL-010` and the manifest logging requirements need an explicit integration design. |
| `SPECIFIED` | The approved default target is Rulebook implementation family v2, specification 4.7, with bounded satisfaction-rank; scalarization remains downstream of every selected rulebook. | Implementation must preserve the vector boundary and exact rulebook identity. |
| `VERIFIED` | Exact historical scale provenance is not established by repository evidence. | Legacy mode records repository source/digest; exact historical reproduction is not claimed without an external artifact. |

## 5. Assumptions And Invariants

These are candidate-contract invariants, pending approval and target-path
confirmation:

- bounded input is exactly four dimensionless margins in collision, interaction,
  compliance, progress order; legacy input is an explicitly ordered vector of
  length `N` with `N` positive scales;
- bounded `m1:m3 ∈ [-1, 0]`, `m4 ∈ [-1, 1]`, with `1e-8` numerical
  canonicalization;
- scalarization is pure, per-transition, deterministic, seed/algorithm/source
  independent, and has no reset state;
- `RulebookResult.complete_evaluation` must be true before scalarization;
- terminal and truncation flags do not alter the formula;
- the raw four-margin vector and rule diagnostics remain separate from the
  policy-visible scalar reward;
- core parameters are `a=2.01`, `c=30.0` where applicable, and native reward
  weight is exactly zero;
- changing scalarization semantics invalidates same-run continuation unless an
  approved migration recomputes preserved margin vectors.

The current repository does not yet establish all of these invariants on one
runtime path. The user clarified that scalarization is downstream of whichever
rulebook is selected and every mode receives a vector: the default experiment
is Rulebook implementation family v2, specification 4.7, with
`bounded_satisfaction_rank`; the legacy formula may remain usable with other
compatible ordered vectors when explicit scales are supplied, while the two
bounded modes require the four-macro-margin contract shared by v4.6/v4.7.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-001` | specification clarification | Which rulebook path is normative: v2/v4.7 four macro-margins, v1 four configured rules, or both? | v2/v4.7 only; v1 only; separate mode-specific adapters | Scalarization is downstream of every selected rulebook and every mode receives a vector. The bounded modes require the four-macro-margin contract shared by v4.6/v4.7; the legacy mode may consume compatible ordered vectors with explicit per-component scales. Default: Rulebook implementation family v2, specification 4.7, plus `bounded_satisfaction_rank`. | Changes adapter interfaces, validation domain, and experiment labels | Clarified by user; incorporated in candidate |
| `DEC-002` | specification clarification | Does `SCAL-V1.0` replace `scalar_reward` or coexist with it? | Replace; coexist under the existing scalar-reward entry point; leave current mode untouched | Keep `behavior: scalar_reward` as the runtime entry point and select the formula through `scalarization.mode`; make the Rulebook 4.7/rank combination the default preset. Exactly one mode supplies the learner-facing reward. | Changes configs while preserving intentional compatibility | Clarified by user; incorporated in candidate |
| `DEC-003` | implementation detail | What does “PPO, TD3, and SAC” mean for repository backends? | Internal backends only; SB3 fork-backed baselines only; both | Cover both registered backends where active, with parity tests at the common wrapper boundary | Determines implementation and smoke matrix | Approved in plan |
| `DEC-004` | specification clarification | N-step and PER are specified but absent from current baseline paths. | Implement replay extensions now; define forward-compatible semantics; remove them | Treat them as conditional constraints: if a future N-step extension is enabled, sum already scalarized per-step rewards; if PER is enabled, use scalar critic TD error. Do not implement either in this feature. `PPO.n_steps` remains rollout length, not an N-step return. | Avoids unapproved replay architecture/dependencies | Clarified by user; incorporated in candidate |
| `DEC-005` | specification clarification | What checkpoint/run artifact is authoritative for scalarization compatibility? | Manifest only; run metadata only; both | Record and compare scalarization identity in both checkpoint manifest and run metadata; reject same-run resume before learner/replay load on mismatch. Allow model-only transfer initialization only as a new run identity. | Changes checkpoint schema and migration policy | Approved by user; incorporated in candidate |
| `DEC-006` | specification clarification | Are unused legacy fields rejected in bounded modes? | Reject non-null scales; ignore but log; preserve current permissive config | Reject non-null legacy scales in conformant bounded-mode configs to prevent accidental scale dependence. | Changes config validation and user-facing errors | Approved by user; incorporated in specification |
| `DEC-007` | blocking technical issue | Repository verification and implementation remain incomplete. | Implement and validate; defer | Proceed with the approved implementation plan; do not claim VERIFIED until mandatory checks and reconciliation pass. | Blocks final verification only | Resolved by approval; implementation work remains |

## 7. Proposed Design

The approved implementation design is:

1. introduce a pure scalarization module with typed configuration/result objects;
2. consume the ordered vector emitted by the selected rulebook at the existing
   environment/wrapper boundary; rulebook extraction and adapters are
   preconditions, not implementation scope for this feature;
3. connect the available rulebook result to the configured scalarizer before transition storage;
4. keep the complete margin vector, schema, rulebook diagnostics, and scalarizer
   decomposition in `info` and evaluation artifacts;
5. extend run metadata and checkpoint manifests with exact scalarization and
   rulebook compatibility identity;
6. preserve the existing v1 manager as a regression-controlled legacy path while
   the new `scalarization.mode` selects the formula for conformant scalar runs.

N-step and PER are not added. Existing learner backends continue to consume the
single per-transition `scalar_reward` field.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-SCAL-001` | `AC-SCAL-001` | `src/thesis_rl/reward/scalarization.py` and `conf/scalarization/default.yaml` | `tests/test_scalarization.py` | Implemented; validation pending |
| `REQ-SCAL-002` | `AC-SCAL-003`, `004`, `009` | Generic legacy branch in `src/thesis_rl/reward/scalarization.py` | `tests/test_scalarization.py` | Implemented; validation pending |
| `REQ-SCAL-003` | `AC-SCAL-005`, `006`, `009`, `010` | Bounded centered sigmoid branch | `tests/test_scalarization.py` | Implemented; validation pending |
| `REQ-SCAL-004` | `AC-SCAL-007`, `008`, `009`, `010` | Bounded satisfaction-rank branch | `tests/test_scalarization.py` | Implemented; validation pending |
| `REQ-SCAL-005` | `AC-SCAL-014` | scalar reward wiring/config validation | `tests/test_scalarization_wiring.py` | Implemented; validation pending |
| `REQ-SCAL-006` | `AC-SCAL-011` | common `scalar_reward` transition boundary | `tests/test_scalarization_wiring.py` | Implemented; validation pending |
| `REQ-SCAL-007` | `AC-SCAL-012` | current per-transition storage; future extensions excluded | conditional contract in spec/plan | Implemented as boundary contract |
| `REQ-SCAL-008` | `AC-SCAL-013` | scalarizer call before wrapper transition storage | `tests/test_rulebook_v2_wrapper.py` | Implemented; validation pending |
| `REQ-SCAL-009` | `AC-SCAL-016` | `ScalarizationResult` and runtime/evaluation diagnostics | `tests/test_scalarization.py`, `tests/test_rulebook_v2_wrapper.py` | Implemented; validation pending |
| `REQ-SCAL-010` | `AC-SCAL-015` | `CheckpointManifest` v2, run metadata, and runtime reward-semantics sidecars | `tests/test_checkpointing.py`, `tests/test_reward_semantics.py` | Implemented; validation pending |

## 9. Test Strategy Defined Before Implementation

The candidate's minimum matrix is retained here as the initial protected
strategy. Exact fixtures and paths must be frozen after the approval gates.

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-SCAL-001` | Unit | mode registry/default/unknown mode | Hydra and typed config | exact default; unknown mode fails | `REQ-SCAL-001` |
| `TEST-SCAL-002` | Unit | boundary validation/canonicalization | four-margin boundary and epsilon cases | canonical values or fail-fast | `REQ-SCAL-004`, `REQ-SCAL-008` |
| `TEST-SCAL-003` | Unit | legacy neutral and scale errors | zero vector; missing/invalid scales | `15.246554505`; invalid config fails | `REQ-SCAL-002` |
| `TEST-SCAL-004` | Unit | smooth neutral/progress/counterexample | reference vectors from AC-005/006 | exact references and documented non-rank behavior | `REQ-SCAL-003` |
| `TEST-SCAL-005` | Unit/property | rank references and exhaustive dominance | all 8 satisfaction patterns and extreme tie-breakers | strict first-differing-rule dominance | `REQ-SCAL-004` |
| `TEST-SCAL-006` | Unit | same-pattern monotonicity | one-coordinate valid perturbations | increasing margin increases reward | `REQ-SCAL-002`–`004` |
| `TEST-SCAL-007` | Unit | stable numerical behavior | large sigmoid arguments, NaN, infinity | finite result or contract error | `REQ-SCAL-002`, `009` |
| `TEST-SCAL-008` | Integration | algorithm parity | same v2 result through PPO/TD3/SAC active adapters | equal scalar rewards | `REQ-SCAL-006` |
| `TEST-SCAL-009` | Integration | order of operations | sequence of margins, terminal/truncated flags | scalarize each step; flags do not alter scalar result | `REQ-SCAL-007`, `008` |
| `TEST-SCAL-010` | Integration | vector preservation/diagnostics | v2 wrapper transition | raw vector retained; logged fields reconstruct reward | `REQ-SCAL-009` |
| `TEST-SCAL-011` | Compatibility | resume mismatch | manifest/config pairs with mode/parameter changes | fail before learning/replay continuation | `REQ-SCAL-010` |
| `TEST-SCAL-012` | Regression | old v1 behavior | current manager and current scalar preset | no unapproved change to historical path | compatibility decision |
| `TEST-SCAL-013` | Smoke | representative runtime | approved v2 PG/Waymo or available deterministic fixture | finite scalar reward, vector, diagnostics | all applicable |

Known repository commands for later validation:

- focused pytest through the primary environment: `uv run --no-sync python -m pytest -q <paths>`;
- full tests: `make test`;
- lint/format focused on modified files: `make lint` and `make format-check PYTHON_QUALITY_PATHS=<paths>`;
- smoke: `make smoke`;
- whitespace: `git diff --check`.

The first two pytest commands attempted during review could not run because
`uv` is not installed in the current shell. No test result is claimed.

## 10. Milestones

- [x] M0 — Read the complete candidate, template, workflow, index, and applicable approved ADRs.
- [x] M1 — Inspect current v1/v2 reward paths, configurations, learners, checkpoint metadata, and tests.
- [x] M2 — Record repository facts, incompatibilities, and approval gates.
- [x] M3 — Resolve DEC-001/DEC-002 with the user and revise the mode/rulebook boundary and default wording in the candidate.
- [x] M3b — Incorporate conditional N-step/PER semantics, checkpoint policy, and legacy scale provenance boundary in the candidate.
- [x] M3c — Obtain explicit approval of the complete updated candidate.
- [x] M4 — Migrate the approved specification, create ADR-011, register the authority, and freeze the initial protected test matrix.
- [x] M5 — Implement the pure scalarizer and config-driven vector-boundary wiring.
- [x] M6 — Implement checkpoint/run metadata compatibility fields, runtime
  sidecars, and diagnostics.
- [ ] M7 — Run mandatory tests, quality checks, smoke validation, and final reconciliation.

## 11. Progress And Findings Log

### 2026-07-17 — Review started and completed

- Read `incoming/rulebook_scalarization_v1.0_specification_UNDER_REVIEW.md` in full.
- Read `docs/engineering_workflow.md`, `.agent/PLANS.md`,
  `docs/project_index.md`, and approved ADR-002/ADR-003.
- Verified current reward manager, wrapper/wiring, v1/v2 rulebook contracts,
  configurations, learner storage paths, checkpoint manifest, and tests.
- `git diff --check`: PASS after creating this review plan.
- Focused pytest commands: NOT_RUN; `uv` unavailable (`/bin/bash: uv: command not found`).
- Findings: user clarified downstream placement, default Rulebook 4.7 plus rank
  mode, and configuration-selected coexistence. N-step/PER are future
  replay-extension semantics rather than current implementation scope.
- Next step: obtain explicit approval of the complete updated candidate;
  no production implementation is permitted before that point.

### 2026-07-17 — User clarification integrated

- The user confirmed that every mode receives a margin vector; bounded modes
  require the v4.6/v4.7 four-macro contract, while legacy mode may use an
  explicitly adapted ordered vector of length `N` with `N` scales.
- The user confirmed the default: Rulebook implementation family v2,
  specification 4.7, and `bounded_satisfaction_rank`.
- The user confirmed one configured producer for the existing `scalar_reward`
  interface, conditional future semantics for N-step/PER, and the checkpoint
  compatibility policy.
- The candidate specification was updated but remains `UNDER_REVIEW`.
- Next step: request explicit approval of the complete updated candidate; do
  not migrate or implement before approval.

### 2026-07-17 — Specification approved and handoff completed

- The user explicitly approved the complete updated specification.
- The specification was marked `APPROVED`, `Authoritative: YES`, version `1.0`,
  and moved to `docs/specifications/`.
- ADR-011 was created and approved; `docs/project_index.md` now registers the
  specification, ADR, and implementation plan.
- Production implementation is authorized under this ExecPlan.

### 2026-07-17 — Runtime checkpoint compatibility completed

- Added `contracts/reward_semantics.py` with canonical identity construction,
  sidecar writing, and fail-closed comparison.
- `Agent.save` now writes the reward-semantics sidecar when configured;
  `runtime.wiring.builders.load_planner` validates it before backend loading.
- Training and Scenario ACL drivers install the identity from the run config;
  missing sidecars are treated as transfer-only artifacts.
- Added 3 runtime compatibility tests; the focused runtime set passes 44 tests,
  and the final scalarization/v1-v2 compatibility set passes 36 tests.

### 2026-07-17 — Initial implementation

- Added `ScalarizationConfig`, `ScalarizationResult`, and the three pure formula
  branches in `src/thesis_rl/reward/scalarization.py`.
- Added nested Hydra configuration under `conf/scalarization/default.yaml` and
  connected configured v2 `scalar_reward` runs at the wrapper boundary.
- Preserved `env_reward` and the ordered Rulebook vector in `info`; scalarizer
  diagnostics include parameters and decomposed contributions.
- Extended run metadata and bumped checkpoint manifest version to reject
  reward-semantics-incompatible continuation artifacts.
- Added scalarization, wrapper, wiring, and manifest tests; test execution is
  still pending because the current shell has neither `uv` nor `pytest`.

## 12. Deviations

- The repository Rulebook default remains unchanged. Rulebook v2/4.7 selection
  and environment adapters are owned by the Rulebook implementation; this
  feature consumes their emitted vector and does not implement them.
- N-step replay and PER remain conditional future semantics and are not
  implemented, as approved in DEC-004.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `docs/specifications/rulebook_scalarization_v1.0_specification.md` | Moved and approved | Authoritative scientific contract |
| `docs/decisions/ADR-011-rulebook-scalarization-v1.md` | Created and approved | Durable material decisions |
| `docs/implementation/rulebook_scalarization_v1.0_exec_plan.md` | Updated to authoritative implementation plan | Requirement-to-code-to-test traceability |
| `docs/project_index.md` | Updated | Authority and implementation registry |
| `src/thesis_rl/reward/scalarization.py` | Added | Pure scalarizer and diagnostics |
| `src/thesis_rl/rulebook/v2/wrapper.py`, `runtime/wiring/builders.py` | Modified | Scalar reward boundary for an available Rulebook result |
| `src/thesis_rl/runtime/io/metadata.py`, `contracts/checkpoint_manifest.py` | Modified | Run/checkpoint reward identity |
| `src/thesis_rl/contracts/reward_semantics.py` | Added | Runtime checkpoint sidecar identity and fail-closed comparison |
| `src/thesis_rl/agent/agent.py`, `runtime/loops/train_loop.py`, `curriculum/scenario_acl/driver.py` | Modified | Sidecar write and run identity installation |
| `conf/scalarization/default.yaml` | Added | Approved default formula configuration |
| `tests/test_scalarization.py`, `tests/test_scalarization_wiring.py` | Added | Scalarization acceptance tests |
| `tests/test_rulebook_v2_wrapper.py`, `tests/test_checkpointing.py` | Modified | Wiring and compatibility regressions |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `git diff --check` | `PASS` | 2026-07-17 | No whitespace errors after implementation edits; pre-existing local changes remain untouched. |
| `python3 -m compileall -q src/thesis_rl tests` | `PASS` | 2026-07-17 | Syntax compilation completed in the current shell. |
| Direct scalarizer smoke script | `PASS` | 2026-07-17 | All three modes, generic legacy `N`, and large sigmoid argument returned finite values. |
| `make lint` | `PASS` | 2026-07-17 | Ruff check passed in the provisioned Docker environment. |
| Focused scalarization/wiring/checkpoint pytest set | `PARTIAL` | 2026-07-17 | 41 passed, 1 unrelated pre-existing encoder preset failure. |
| `make test` | `PARTIAL` | 2026-07-17 | 568 passed, 2 skipped, 5 unrelated pre-existing encoder preset failures. |
| Focused `make format-check` | `PASS` | 2026-07-17 | 11 modified feature files already formatted after the targeted format run. |
| `make config` | `PASS` | 2026-07-17 | Docker Compose configuration validation passed. |
| Focused runtime compatibility pytest set | `PASS` | 2026-07-17 | 44 passed, including sidecar write/mismatch/missing semantics tests. |
| Final scalarization/v1-v2 compatibility pytest set | `PASS` | 2026-07-17 | 36 passed, including the historical v1 fixture. |
| Final `make test` | `PARTIAL` | 2026-07-17 | 575 passed, 2 skipped, 5 pre-existing encoder preset failures. |
| Final `make lint` | `PASS` | 2026-07-17 | Ruff check passed after runtime sidecar integration. |
| `make smoke` | `NOT_RUN` | 2026-07-17 | Run after implementation. |

## 15. Final Reconciliation

| Requirement/criterion | Status | Explanation |
|---|---|---|
| `REQ-SCAL-001`–`REQ-SCAL-009` | `IMPLEMENTED/PENDING VALIDATION` | Pure formulas, vector-boundary wiring, diagnostics, metadata, and current learner boundary are implemented; focused tests pass, while full validation has five unrelated preset failures. |
| `REQ-SCAL-010` | `IMPLEMENTED/PENDING VALIDATION` | Manifest/API fields, run metadata, and runtime sidecar validation before learner loading are implemented. |
| `AC-SCAL-001`–`AC-SCAL-016` | `PARTIAL` | Acceptance tests execute in Docker; live end-to-end Rulebook extraction remains an external precondition. |

Known limitations:

- the host shell lacks `uv` and `pytest`; Docker Compose was used for the
  available focused/full validation;
- exact historical scale provenance is not established by repository evidence;
- N-step/PER behavior is specified conditionally for future or absent
  extensions and is not implemented by this feature;
- The established `train_loop` continues to save planner archives through
  `Agent.save`; it now writes and validates the separate reward-semantics
  sidecar. The immutable generation API remains available for stricter future
  checkpoint publication, but is not required for current runtime resume.

Deferred required work:

- run mandatory tests and reconcile every requirement before claiming VERIFIED;
- validate live environment integration separately when the Rulebook
  implementation supplies its approved vector adapter.

Optional work deferred by the candidate:

- full three-mode training ablations.

Readiness: `IMPLEMENTED BUT NOT VERIFIED`; not ready for experimental use until
mandatory validation and the live v2 integration pass.
