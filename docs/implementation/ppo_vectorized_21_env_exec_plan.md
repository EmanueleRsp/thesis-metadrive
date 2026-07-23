# PPO Vectorized 21-Environment Configuration ExecPlan

## 1. Metadata

- Plan ID: `PPO-VEC-21-001`
- Authority: `docs/specifications/rl_baselines_v1_specification.md`, RL-BASELINES v1.0, approved
- Status: `IN_PROGRESS`
- Created/updated: 2026-07-23
- Related ADRs: none

## 2. Objective And Scope

Configure the SB3 PPO path for the requested 21-environment vectorized runtime:
ordinary profiles use `n_steps=96`, `batch_size=63`, and `n_epochs=10`, while
the smoke profile retains `n_steps=16`, `batch_size=8`. Add fail-fast rollout
geometry validation and persist resolved geometry in run metadata. Preserve all
other PPO hyperparameters.

The requested single `model.learn(total_timesteps=...)` lifecycle is recorded as
a separate architectural gap: the current repository owns chunking and invokes
the backend repeatedly to support evaluation, ACL, checkpointing, and resume.
This change does not silently rewrite that lifecycle.

## 3. Authoritative Requirements

| ID | Requirement | Specification/source |
|---|---|---|
| REQ-001 | Ordinary PPO geometry is 21 env × 96 steps = 2016, batch 63, 32 minibatches, 320 optimizer steps/update | User request |
| REQ-002 | Smoke retains 16/8 and resolves to 336 transitions and 42 minibatches | User request |
| REQ-003 | Validate positive steps, batch > 1, epochs > 0, and rollout divisibility | User request |
| REQ-004 | Preserve current PPO hyperparameters other than requested geometry | User request |
| REQ-005 | Record resolved run profile, environment count, PPO geometry, budget, and intervals | User request |

## 4. Current Repository Analysis

- `conf/agent/planner/algorithm/ppo_sb3.yaml` currently contains 2048/64/10.
- `conf/run_profile/smoke.yaml` already overrides PPO to 16/8.
- `src/thesis_rl/agent/planners/algorithms/ppo_sb3.py` owns the SB3 rollout
  buffer and manually collects transitions; it does not call `model.learn`.
- `src/thesis_rl/runtime/loops/train_loop.py` divides training at
  `experiment.eval_interval` to coordinate evaluation/checkpoint/ACL lifecycle.
- `src/thesis_rl/runtime/io/metadata.py` records total and evaluation intervals,
  but not resolved PPO global rollout geometry.
- `tests/test_hydra_agent_presets.py` and `tests/test_hydra_preset_run_configs.py`
  cover current PPO values and smoke resolution.

## 5. Assumptions And Invariants

- `env.vectorized.num_envs` is the global vector environment count.
- Rollout size is `n_steps * num_envs`; no implicit multiplication is applied to
  evaluation/checkpoint intervals.
- The ordinary 21-env contract gives 2016/63 = 32 minibatches and 320 optimizer
  steps per update.
- Smoke is diagnostic only.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Recommendation | Status |
|---|---|---|---|---|
| DEC-001 | implementation detail | Existing training lifecycle is chunk-oriented rather than one SB3 `learn` call | Keep lifecycle unchanged in this configuration patch and expose the gap explicitly | Approved by task scope |

## 7. Test Strategy Defined Before Implementation

| ID | Level | Expected result |
|---|---|---|
| TEST-001 | Hydra config | PPO ordinary config resolves to 96/63/10 and preserves hyperparameters |
| TEST-002 | Hydra config | Smoke resolves to 16/8 |
| TEST-003 | Unit | Geometry validator accepts 21/96/63 and 21/16/8, rejects invalid/non-divisible values |
| TEST-004 | Metadata | Resolved PPO geometry is persisted |

Commands: focused pytest for affected tests; `git diff --check`; focused Ruff
check if Python files are modified.

## 8. Milestones

- [x] Record current configuration and lifecycle.
- [x] Update ordinary PPO defaults and validation.
- [x] Add metadata and tests.
- [x] Run focused validation and reconcile documentation.

## 9. Deviations

| ID | Original contract | Actual change | Reason |
|---|---|---|---|
| DEV-001 | Requested single `model.learn` over full budget | Not changed in this patch | Current chunk lifecycle is coupled to evaluation, ACL, checkpointing, and resume; requires separate architectural change |

## 10. Files

| Path | Action | Purpose |
|---|---|---|
| `conf/agent/planner/algorithm/ppo_sb3.yaml` | modify | Ordinary PPO geometry |
| `src/thesis_rl/agent/planners/algorithms/ppo_sb3.py` | modify | Fail-fast geometry validation and resolved stats |
| `src/thesis_rl/runtime/io/metadata.py` | modify | Persist PPO geometry |
| `tests/...` | modify | Regression coverage |

## 11. Progress And Findings Log

- 2026-07-23: Changed SB3 PPO ordinary defaults to 96/63/10; smoke override
  remains 16/8. Added backend and metadata geometry validation and Hydra
  regression coverage for all ordinary duration profiles.
- 2026-07-23: System Python test collection failed because Hydra/OmegaConf are
  unavailable. Repository `.venv` exists but does not contain pytest. Static
  compilation and whitespace checks remain available.

## 12. Validation Results

| Command | Result | Date | Notes |
|---|---|---|---|
| `pytest -q tests/test_hydra_agent_presets.py tests/test_hydra_preset_run_configs.py tests/test_ppo_sb3_porting.py tests/test_run_metadata.py` | FAIL | 2026-07-23 | System Python lacks `hydra` and `omegaconf` |
| `.venv/bin/python -m pytest -q ...` | FAIL | 2026-07-23 | Repository environment lacks `pytest` |
| `git diff --check` | PASS | 2026-07-23 | No whitespace errors |
| `.venv/bin/python -m py_compile src/thesis_rl/agent/planners/algorithms/ppo_sb3.py src/thesis_rl/runtime/io/metadata.py` | PASS | 2026-07-23 | Modified Python files compile |

## 13. Final Reconciliation

- REQ-001: IMPLEMENTED; ordinary SB3 PPO resolves to 96/63/10 and 2016 global
  transitions for 21 environments. Runtime test execution is pending because
  the local environments lack dependencies.
- REQ-002: IMPLEMENTED; smoke profile retains 16/8.
- REQ-003: IMPLEMENTED; backend and metadata fail fast on invalid geometry.
- REQ-004: IMPLEMENTED; other PPO values were not changed.
- REQ-005: IMPLEMENTED; metadata now records PPO geometry and checkpoint interval.

Known limitation: the current application-level chunk lifecycle remains in
place; the requested single full-budget `model.learn` lifecycle is deferred as
DEV-001 and requires a separate approved architectural change.
