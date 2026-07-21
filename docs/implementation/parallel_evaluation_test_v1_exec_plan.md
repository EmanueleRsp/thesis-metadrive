# Parallel Evaluation And Test v1 ExecPlan

## 1. Metadata

- Feature: deterministic process-based parallel validation and final test
- Plan ID: `PAR-EVAL-TEST-V1`
- Authoritative specification: `docs/specifications/rl_baselines_v1_specification.md`
  (`RL-BASELINES`, version `1.0`, `APPROVED`, `Authoritative: YES`)
- Related protocol: `docs/protocols/live_eval_video_protocol.md` (candidate
  protocol; implementation scope explicitly approved by the user)
- Related ADRs: `ADR-018-parallel-evaluation-and-test.md`, `ADR-016`
- Status: `IMPLEMENTED`; focused validation and end-to-end smoke passed; full
  repository suite retains two unrelated baseline failures
- Created: 2026-07-21
- Last updated: 2026-07-21
- Owner: thesis repository maintainer

## 2. Objective And Scope

Add process-based parallel execution for validation/evaluation and final test
episodes, including live rendering and official per-episode video artifacts,
while preserving the sequential episode assignment, per-episode metrics, CSV
ordering, termination/truncation semantics, and deterministic primary policy
behavior.

In scope:

- configurable evaluation and final-test worker counts;
- `spawn` process workers with explicit manual episode reset;
- deterministic MetaDrive seed assignment;
- deterministic ScenarioNet provider-sequence reconstruction and forced runtime
  index assignment;
- vectorized episode stepping with parent-side policy inference and metric
  reduction in `episode_id` order;
- live final-evaluation rendering, manifests, trajectory logs, and GIF output;
- standalone evaluation and training-time intermediate/final evaluation paths;
- sequential fallback for one worker and backward-compatible public behavior.

Out of scope, explicitly deferred by the user:

- evaluation running asynchronously while training continues;
- changing scientific training profiles or learner `n_envs`;
- changing reward, observation, Rulebook, ACL, dataset, checkpoint, or CSV
  metric definitions;
- throughput auto-tuning or claiming a universally optimal worker count.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-PET-001` | Primary evaluation remains deterministic, uses no gradients, and mutates no learner/replay/ACL state. | RL-BASELINES §7.1, `REQ-RLB-017` |
| `REQ-PET-002` | Evaluation worker count is independent from the frozen training profile and preserves policy observation/action contracts. | RL-BASELINES §11.4; `REQ-RLB-015` |
| `REQ-PET-003` | Episode assignment and aggregate reduction are deterministic and preserve the sequential episode order. | RL-BASELINES §3.3, §10.8, `AC-RLB-021` |
| `REQ-PET-004` | True termination, truncation, terminal observation, and reset boundaries remain distinct. | RL-BASELINES `REQ-RLB-014`, `AC-RLB-018` |
| `REQ-PET-005` | ScenarioNet evaluation uses the same deterministic provider sequence as the sequential path; workers do not independently resample evaluation records. | ScenarioNet v1.1 §23 and repository provider contract |
| `REQ-PET-006` | Official live video is recorded during the same evaluation episode and retains manifest, trajectory, and CSV links. | Live Evaluation Video Protocol §§2, 3, 5 |
| `REQ-PET-007` | Existing one-worker evaluation and offline replay remain compatible. | Live Evaluation Video Protocol §§6, 8 |
| `REQ-PET-008` | The async train/evaluation overlap remains deferred and is not enabled by this change. | Explicit user approval, 2026-07-21 |

## 4. Current Repository Analysis

| Classification | Verified fact |
|---|---|
| `VERIFIED` | `Agent.evaluate` in `src/thesis_rl/agent/agent.py` executes one environment and one episode at a time. |
| `VERIFIED` | `build_env` creates a single environment; `build_train_env` already owns the process-vector construction used for training. |
| `VERIFIED` | `DeterministicSubprocVecEnv` currently auto-resets generic workers; evaluation requires an explicit no-auto-reset mode to assign exact episode seeds/UIDs. |
| `VERIFIED` | Live artifact recording is already integrated in `Agent.evaluate` and writes GIF, manifest, and optional trajectory JSONL. |
| `VERIFIED` | Training and standalone evaluation already write `eval_episodes.csv` from `metrics["per_episode"]`; preserving list ordering preserves CSV ordering. |
| `VERIFIED` | The current preprocessor factory exposes only the stateless identity preprocessor. |
| `INFERRED` | Parent-side feed-forward policy inference plus process-isolated environment stepping preserves the observable sequential policy trajectory for the supported non-recurrent baselines. |
| `AWAITING_CONFIRMATION` | None. The user explicitly authorized implementation of the agreed scope and deferred only asynchronous overlap. |

## 5. Assumptions And Invariants

- `episode_id` is the canonical ordering key and is one-based in artifacts.
- Native MetaDrive evaluation uses `base_seed + episode_idx` exactly as the
  sequential path.
- ScenarioNet evaluation reconstructs the provider's worker-0 sequence in the
  parent, then resets workers with the selected runtime index; provider RNG is
  not allowed to race across workers.
- `terminated` and `truncated` remain separate, and the terminal observation
  remains attached to the completed episode before reset.
- `spawn` is mandatory for process workers because MetaDrive/Panda3D state must
  not be inherited through `fork`.
- A worker count is capped at the number of episodes and must be positive.
- The parent reduces all per-episode arrays only after sorting by `episode_id`.
- Current supported policy/preprocessor contracts are feed-forward/stateless;
  unsupported stateful preprocessors fail fast for parallel evaluation.
- Live video files are episode-local; no worker writes a shared CSV or manifest.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-PET-001` | specification clarification | Parallelism boundary | overlap training/evaluation; parallelize episodes while training pauses | episode-level parallelism only | preserves curriculum/evaluation synchronization | Approved by user 2026-07-21 |
| `DEC-PET-002` | implementation detail | Worker protocol | generic auto-reset; parent-controlled reset | `spawn` workers with explicit reset after each completed episode | exact seed/UID assignment and boundary preservation | Approved by user 2026-07-21 |
| `DEC-PET-003` | implementation detail | Policy placement | copy policy into workers; parent inference | parent-side deterministic feed-forward inference | avoids checkpoint/policy copies and preserves action semantics | Approved by user 2026-07-21 |
| `DEC-PET-004` | compatibility decision | Default worker counts | one; machine-specific recommendations | validation `12`, final test `8`, capped by episode count | operational throughput default; override remains available | Approved by user 2026-07-21 |
| `DEC-PET-005` | specification clarification | ScenarioNet ordering | independent provider sampling; parent sequence | reconstruct sequential provider-0 sequence and force runtime indices | prevents scenario/statistical drift | Approved by user 2026-07-21 |
| `DEC-PET-006` | deferred feature | Async evaluation while training | concurrent process; synchronous evaluation | defer to a later change | no change to stage gates or training timing in this cycle | Approved deferral by user 2026-07-21 |

## 7. Proposed Design

1. Extend `DeterministicSubprocVecEnv` with an evaluation mode that does not
   auto-reset completed workers and with batched render commands.
2. Add evaluation environment wiring that builds isolated workers with
   `spawn`, preserves the merged evaluation config, and uses the same observation
   and action spaces as the sequential environment.
3. Add a deterministic ScenarioNet sequence helper that uses the same catalog,
   split, provider kind, source probabilities, arm filter, eligibility manifest,
   and global seed as the sequential environment.
4. Extend `Agent.evaluate` with a parallel branch. It runs a fixed number of
   active slots, calls the existing policy pipeline in stable slot order,
   collects slot-local episode state, and feeds completed episodes to the same
   aggregate/output path after canonical ordering.
5. Make artifact recording slot-aware. The existing official path layout and
   CSV fields remain unchanged; rendering is performed in worker processes and
   frames are associated with the canonical episode ID.
6. Add `experiment.eval_workers`, `experiment.test_workers`, and
   `experiment.evaluation_start_method`. The existing sequential behavior is
   selected when the resolved worker count is `1`.
7. Keep `render_selected_videos.py` offline replay behavior unchanged and
   authoritative-live preference intact.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-PET-001` | `AC-PET-001` | `Agent.evaluate`, parallel evaluator | `tests/test_parallel_evaluation.py` | PASS |
| `REQ-PET-002` | `AC-PET-002` | evaluation config/builders | `tests/test_parallel_evaluation_config.py` | PASS |
| `REQ-PET-003` | `AC-PET-003`, `AC-PET-004` | evaluator ordering/reducer | `tests/test_parallel_evaluation.py` | PASS |
| `REQ-PET-004` | `AC-PET-005` | vector boundary protocol | `tests/test_deterministic_subproc_vec_env.py`, parallel tests | PASS |
| `REQ-PET-005` | `AC-PET-006` | ScenarioNet sequence helper and ACL schedules | focused ScenarioNet/provider regression plus code review | PASS; smoke not dataset-backed |
| `REQ-PET-006` | `AC-PET-007` | existing recorder plus vector render | `tests/test_eval_artifacts.py`, parallel tests, smoke | PASS |
| `REQ-PET-007` | `AC-PET-008` | one-worker fallback/offline path | existing artifact/video tests | PASS |
| `REQ-PET-008` | `AC-PET-009` | synchronous training/evaluation loop | smoke and runtime review | PASS |

## 9. Test Strategy Defined Before Implementation

### Acceptance criteria

- `AC-PET-001`: repeated deterministic evaluation produces identical ordered
  episode metrics and does not change model/checkpoint state.
- `AC-PET-002`: invalid worker counts/start methods fail; worker count is capped
  by episode count; defaults resolve to 12 validation and 8 final test workers.
- `AC-PET-003`: parallel native MetaDrive evaluation assigns exactly the same
  seed sequence as sequential evaluation.
- `AC-PET-004`: aggregate metrics and per-rule rows equal the sequential result
  for a controlled deterministic environment, including episode ordering.
- `AC-PET-005`: terminal observation, terminated/truncated flags, and reset
  observation remain correctly separated in manual-reset vector evaluation.
- `AC-PET-006`: parallel ScenarioNet sequence uses the same ordered runtime
  indices as the provider-0 sequential sequence, with no worker resampling.
- `AC-PET-007`: every live-recorded episode produces the existing GIF/manifest/
  trajectory bundle and the returned per-episode paths remain canonical.
- `AC-PET-008`: one worker retains current `Agent.evaluate` and offline replay
  behavior.
- `AC-PET-009`: no asynchronous evaluation process is started by training; the
  training loop still waits for evaluation results before curriculum decisions.

### Mandatory test matrix

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-PET-001` | Unit | worker config validation | zero, negative, excessive workers, invalid start method | explicit errors/cap/defaults | `REQ-PET-002` |
| `TEST-PET-002` | Integration | manual-reset vector env | two deterministic terminating workers | no implicit reset; exact reset seeds | `REQ-PET-003`, `REQ-PET-004` |
| `TEST-PET-003` | Unit | native seed schedule | base seed and episode count | `base_seed + episode_id - 1` | `REQ-PET-003` |
| `TEST-PET-004` | Unit | metric equivalence | deterministic two-slot fixture | sequential and parallel dictionaries/per-episode arrays equal | `REQ-PET-001`, `REQ-PET-003` |
| `TEST-PET-005` | Unit | boundary flags | terminal and timeout fixtures | flags/final observations preserved | `REQ-PET-004` |
| `TEST-PET-006` | Unit | ScenarioNet provider sequence | uniform/arm-uniform/fixed-sequence provider fixtures | forced runtime-index sequence equals sequential provider-0 sequence | `REQ-PET-005` |
| `TEST-PET-007` | Integration | live artifacts | deterministic render fixture, two episodes | canonical GIF/manifest/JSONL paths and metadata | `REQ-PET-006` |
| `TEST-PET-008` | Regression | one-worker compatibility | existing agent/artifact fixtures | existing tests and output schema remain valid | `REQ-PET-007` |
| `TEST-PET-009` | Regression | async scope | training/evaluation call path inspection fixture | no background eval process/thread | `REQ-PET-008` |

### Commands

```bash
uv run --no-sync python -m pytest -q tests/test_parallel_evaluation_config.py tests/test_parallel_evaluation.py tests/test_deterministic_subproc_vec_env.py tests/test_eval_artifacts.py tests/test_agent_pipeline.py
make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/agent/agent.py src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py src/thesis_rl/runtime/wiring/builders.py src/thesis_rl/runtime/io/eval_artifacts.py src/thesis_rl/runtime/io/video_utils.py tests/test_parallel_evaluation.py tests/test_parallel_evaluation_config.py"
make lint PYTHON_QUALITY_PATHS="src/thesis_rl/agent/agent.py src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py src/thesis_rl/runtime/wiring/builders.py src/thesis_rl/runtime/io/eval_artifacts.py src/thesis_rl/runtime/io/video_utils.py tests/test_parallel_evaluation.py tests/test_parallel_evaluation_config.py"
git diff --check
make smoke
```

The full Docker/GPU smoke remains applicable when the provisioned environment
is available; host-level `python` is not treated as a substitute for it.

## 10. Milestones

- [x] M1: ExecPlan/ADR and acceptance matrix recorded.
- [x] M2: manual-reset vector worker and deterministic render protocol.
- [x] M3: parallel evaluator and canonical metric reduction.
- [x] M4: ScenarioNet deterministic sequence and all runtime wiring.
- [x] M5: focused tests and quality checks.
- [x] M6: smoke, reconciliation, and documentation/index update.

## 11. Progress And Findings Log

| Date | Finding/action | Evidence/result | Next step |
|---|---|---|---|
| 2026-07-21 | Current evaluation is single-env and sequential; training vector path already exists. | `Agent.evaluate`, `build_env`, `DeterministicSubprocVecEnv` inspection | implement M2 |
| 2026-07-21 | User deferred asynchronous train/evaluation overlap. | Current request annotation | exclude from implementation and tests beyond no-background-eval guard |
| 2026-07-21 | Implemented spawned evaluation/test workers with ordered reduction, explicit reset, ScenarioNet runtime-index scheduling, ACL Waymo arm schedules, and live rendering. | Focused tests `22 passed`; config tests `5 passed`; smoke completed successfully. | update index and final diff |
| 2026-07-21 | Full repository suite completed with `708 passed, 1 skipped, 2 failed`; failures are unrelated baseline fixtures (`test_forced_rule_scenarios`, `test_golden_suite`). | Failure traces do not enter the new parallel evaluation path. | retain as known repository-level limitations |

## 12. Deviations

1. `experiment.eval_workers=12` and `experiment.test_workers=8` are machine
   informed defaults, not a measured optimum; worker count remains configurable.
2. The current parallel branch supports the repository's stateless identity
   preprocessor/feed-forward policy contract and fails fast for other
   preprocessor types.
3. Official GIF encoding remains parent-side after worker rendering, so video
   encoding and frame transfer can limit throughput.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py` | Modified | manual reset and batched render |
| `src/thesis_rl/runtime/wiring/builders.py` | Modified | evaluation worker construction and ScenarioNet sequence |
| `src/thesis_rl/agent/agent.py` | Modified | parallel evaluation branch and reduction |
| `src/thesis_rl/runtime/io/eval_artifacts.py` | Reused unchanged | existing live artifact contract remains authoritative |
| `src/thesis_rl/runtime/io/video_utils.py` | Modified | cached worker-frame consumption |
| `conf/experiment/default.yaml` | Modified | evaluation worker defaults |
| `src/thesis_rl/runtime/loops/train_loop.py` | Modified | pass worker settings to validation/final test |
| `src/thesis_rl/runtime/loops/eval_loop.py` | Modified | pass worker settings to standalone test |
| `src/thesis_rl/curriculum/scenario_acl/driver.py` | Modified | pass worker settings to ACL evaluation/test |
| `tests/test_parallel_evaluation.py` | Added | evaluator/metric equivalence and live-artifact tests |
| `tests/test_parallel_evaluation_config.py` | Added | config validation tests |
| `tests/test_deterministic_subproc_vec_env.py` | Modified | manual reset regression tests |
| `docs/decisions/ADR-018-parallel-evaluation-and-test.md` | Added | approved material decisions |
| `docs/project_index.md` | Modified | register plan and ADR |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| Repository inspection and source/spec review | PASS | 2026-07-21 | Completed before production changes |
| Focused pytest matrix | PASS | 2026-07-21 | `22 passed`; configuration/ScenarioNet tests separately `6 passed` |
| Focused Ruff | PASS | 2026-07-21 | All modified source/tests passed |
| Compose validation | PASS | 2026-07-21 | standard and GPU compose configurations valid |
| Full pytest suite | PARTIAL | 2026-07-21 | `708 passed, 1 skipped, 2 failed`; unrelated existing failures recorded above |
| End-to-end smoke | PASS | 2026-07-21 | `presets/test/smoke_train`; intermediate and final parallel evaluations completed |
| `git diff --check` | PASS | 2026-07-21 | no whitespace errors |

## 15. Final Reconciliation

Implementation is complete for the approved synchronous scope. The plan is
not marked `VERIFIED` because the repository-wide suite still has two unrelated
baseline failures; the focused feature matrix, quality checks, compose checks,
smoke, and documentation reconciliation pass.

Known limitation: subprocess/GPU execution is logically deterministic under the
same software/data/configuration contract; cross-platform bitwise equality is
not claimed.

Deferred required work: asynchronous evaluation concurrent with training.

Optional follow-up: machine-local throughput benchmark and automatic worker
recommendation based on measured scenarios/second and video encode throughput.
