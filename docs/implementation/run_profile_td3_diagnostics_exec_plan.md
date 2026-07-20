# Run-profile TD3 budgets and diagnostic logging

## 1. Metadata

- Feature/plan ID: `RUN-PROFILE-TD3-DIAGNOSTICS`
- Authoritative specification: `docs/specifications/rl_baselines_v1_specification.md`, ID `RL-BASELINES`, version `1.0`, `APPROVED` and `AUTHORITATIVE: YES`.
- Status: `VERIFIED`
- Created: 2026-07-20
- Last update: 2026-07-20
- Branch: current working branch
- Related ADRs: `docs/decisions/ADR-014-scenarionet-acl-learning-potential-only.md`, `docs/specifications/transition_replay_v1_specification.md`
- Owner: repository maintainer

## 2. Objective and scope

Make `make run ALGORITHM=td3_sb3 RUN_PROFILE=fast` execute the 120,000-step
fast protocol with TD3 warm-up and batch settings sized for that budget. Keep
the runtime profile (ACL ON, one environment by default) orthogonal to the
duration profile, add the approved SAC/PPO budget-sensitive overrides, disable
replay persistence by default, and keep the per-step Rulebook margin JSONL
trace disabled by default except for `smoke`.

In scope: Hydra run-profile configuration, shared reward diagnostic defaults,
algorithm-specific budget overrides, replay-persistence default, configuration
regression tests, and this traceability record. Out of scope: changing TD3/SAC/
PPO mathematical formulas, dataset policy, ACL vectorization, and explicit
diagnostic presets such as scale tuning.

## 3. Authoritative requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-RP-001` | `fast` uses 120,000 total timesteps. | `conf/run_profile/fast.yaml` and existing repository contract |
| `REQ-RP-002` | TD3 profile budgets use the approved warm-up/batch matrix. | `RL-BASELINES` §3.4 |
| `REQ-RP-003` | Heavy rule-margin logging is enabled by default only for `smoke`. | User request, 2026-07-20 |
| `REQ-RP-004` | Explicit diagnostic overrides remain available. | Existing tuning/debug presets |
| `REQ-RP-005` | SAC profile warm-up/batch overrides follow the approved duration matrix. | `RL-BASELINES` §3.4 |
| `REQ-RP-006` | Replay persistence is disabled by default for TD3/SAC. | `RL-BASELINES` §9.6; user approval, 2026-07-20 |
| `REQ-RP-007` | PPO smoke uses a clearly labeled diagnostic rollout override. | `RL-BASELINES` §3.4 |

## 4. Current repository analysis

- **VERIFIED:** `conf/run_profile/fast.yaml` already specifies `120000` steps.
- **VERIFIED:** `_resolve_planner_cfg()` in
  `src/thesis_rl/runtime/wiring/builders.py` merges a profile-level
  `planner` mapping over algorithm defaults.
- **VERIFIED:** `td3_sb3` defaults are `learning_starts=10000` and
  `batch_size=256`; the smoke profile overrides them to `100` and `64`.
- **VERIFIED:** the approved TD3 duration matrix is smoke `100/64`, fast
  `5000/256`, and medium/long/tune/thesis `10000/256`.
- **VERIFIED:** the approved SAC duration matrix is smoke `100/64`, fast
  `1000/256`, medium/tune `5000/256`, and long/thesis `10000/256`.
- **VERIFIED:** `rule_margin_log_path` was changed from `null` to a run-local
  path in commit `f58ca1a`, causing it to be active for every inherited reward
  profile.
- **VERIFIED:** both `RuleRewardWrapper` and `RulebookV2MonitorWrapper` skip
  margin file creation when the path is `None`.
- **VERIFIED:** the 48-record golden suite is reference-only and is not selected
  by ordinary `make run`; it remains available to the dedicated diagnostic.

## 5. Assumptions and invariants

- `experiment.total_timesteps` remains the profile budget and is not changed by
  this task.
- `learning_starts` and `batch_size` are integer TD3/SB3 planner overrides;
  the selected algorithm still controls all other hyperparameters.
- SAC warm-up and batch overrides are applied only to `sac_sb3`; PPO smoke
  overrides are applied only to `ppo_sb3`.
- `transition_replay.persistence.enabled=false` is the default for TD3/SAC;
  explicit stateful continuation remains a separate opt-in workflow.
- Rulebook evaluation and attached structured info remain active; only the
  heavy JSONL margin trace is gated.
- Explicit CLI/preset assignment of `reward.rule_margin_log_path` continues to
  override the default, including tuning/debug workflows.

## 6. Decisions and approval gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-RP-001` | specification clarification | What fast/long-run TD3 values should be used? | Keep algorithm defaults; use budget-aware matrix | Smoke `100/64`, fast `5000/256`, medium/long/tune/thesis `10000/256` | Learner warm-up and update batch only | Approved by user, 2026-07-20 |
| `DEC-RP-002` | implementation detail | Where should default margin logging be gated? | Runtime conditional; profile config overrides | Set shared default to `null`, add path only to smoke | File output volume and default diagnostics | Approved by explicit user request |
| `DEC-RP-003` | specification clarification | How should SAC warm-up vary with budget? | Keep `100` for every profile; use budget-aware matrix | Smoke `100`, fast `1000`, medium/tune `5000`, long/thesis `10000`; batch `256` | SAC replay diversity before updates | Approved by user, 2026-07-20 |
| `DEC-RP-004` | specification clarification | Should replay persistence be enabled? | Persist replay; model-only restart | Keep persistence OFF by default due to storage/I/O cost | Restart semantics and disk use | Approved by user, 2026-07-20 |
| `DEC-RP-005` | specification clarification | What is the role of the 48-record golden suite? | Use for policy data; retain as diagnostic fixture | Retain only for deterministic Rulebook/adapter integration and regression checks | Prevents biased source×arm fixture from entering scientific claims | Approved by user, 2026-07-20 |

## 7. Proposed design

Add algorithm-specific `planner` blocks to every duration profile. The runtime
resolves only the block matching the selected fork-backed algorithm, preventing
TD3/SAC/PPO overrides from leaking into one another. Set the shared reward
default path to `null`, set `reward.rule_margin_log_path` in `smoke.yaml` to the
existing run-local JSONL path, and set TD3/SAC replay persistence to `false`.
Keep explicit diagnostic preset overrides unchanged. The golden suite is not
part of the normal training path.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-RP-001` | `fast` resolves to 120,000 steps | `conf/run_profile/fast.yaml` | `tests/test_hydra_preset_run_configs.py` | Planned |
| `REQ-RP-002` | All profiles resolve expected TD3 warm-up/batch values | `conf/run_profile/*.yaml` | `tests/test_hydra_preset_run_configs.py` | Planned |
| `REQ-RP-003` | Smoke has a margin path; all other profiles resolve `None` | `conf/reward/rulebook_defaults.yaml`, `conf/run_profile/smoke.yaml` | `tests/test_hydra_preset_run_configs.py` | Verified |
| `REQ-RP-004` | Explicit tuning path remains present | `conf/presets/td3/td3_scalar_reward_scale_tuning_no_curr.yaml` | config composition test / inspection | Verified |
| `REQ-RP-005` | All profiles resolve expected SAC warm-up/batch values | `conf/run_profile/*.yaml`, `src/thesis_rl/runtime/wiring/builders.py` | `tests/test_hydra_preset_run_configs.py` | Planned |
| `REQ-RP-006` | TD3/SAC persistence resolves to `false` by default | `conf/agent/planner/algorithm/{td3_sb3,sac_sb3}.yaml` | `tests/test_hydra_preset_run_configs.py` | Planned |
| `REQ-RP-007` | PPO smoke resolves to `n_steps=16`, `batch_size=8` | `conf/run_profile/smoke.yaml`, `src/thesis_rl/runtime/wiring/builders.py` | `tests/test_hydra_preset_run_configs.py` | Planned |

## 9. Test strategy defined before implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-RP-001` | Hydra config | fast composition | `run_profile=fast`, `agent/planner/algorithm=td3_sb3` | 120,000 steps and `5000/512` | `REQ-RP-001`, `REQ-RP-002` |
| `TEST-RP-002` | Hydra config | profile matrix | default, smoke, fast, medium, long, tune, thesis | exact expected planner values | `REQ-RP-002` |
| `TEST-RP-003` | Hydra config | diagnostic gating | all named profiles | only smoke has margin path | `REQ-RP-003` |
| `TEST-RP-004` | Hydra config | explicit diagnostic preset | TD3 scale tuning preset | explicit margin path remains configured | `REQ-RP-004` |
| `TEST-RP-005` | Hydra config | SAC duration matrix | all named duration profiles | expected warm-up/batch values resolve only for `sac_sb3` | `REQ-RP-005` |
| `TEST-RP-006` | Hydra config | replay persistence default | TD3/SAC scientific configs | persistence is `false` | `REQ-RP-006` |
| `TEST-RP-007` | Hydra config | PPO diagnostic smoke | `run_profile=smoke`, `ppo_sb3` | `n_steps=16`, `batch_size=8` | `REQ-RP-007` |

Validation commands: `pytest -q tests/test_hydra_preset_run_configs.py`,
`pytest -q tests/test_hydra_agent_presets.py`, `ruff check` on modified Python
tests, `git diff --check`, and `make config` if the container is available.

## 10. Milestones

- [x] M1 — Update profile, replay, and diagnostic configurations.
- [x] M2 — Add configuration regression tests.
- [x] M3 — Run focused validation and reconcile documentation.

## 11. Progress and findings log

- 2026-07-20: Confirmed fast budget is 120,000 and approved the duration matrix
  for TD3, SAC, and diagnostic PPO smoke settings. Confirmed global margin
  logging can be disabled with `null` without runtime changes.
- 2026-07-20: Added algorithm-specific profile overrides, smoke-only default
  margin logging, replay persistence OFF, and regression coverage. The focused
  Hydra matrix passed with 33 tests; Ruff, format, Docker composition, and
  whitespace checks also passed.

## 12. Deviations

| `DEV-RP-001` | Earlier local run-profile values used TD3 batches larger than the approved v1 core batch. | Profiles and TD3 default now use `batch_size=256`, with smoke `64`. | Aligns the implementation with approved RL-BASELINES v1.0. | User approval 2026-07-20 | Config tests and spec matrix |

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `conf/config.yaml` | Modified | Apply run-profile reward overrides after the reward preset |
| `conf/run_profile/*.yaml` | Modified | Explicit TD3/SAC/PPO duration-profile budgets |
| `conf/agent/planner/algorithm/{td3_sb3,sac_sb3}.yaml` | Modified | TD3 core batch and replay persistence default |
| `conf/agent/planner/algorithm/ppo_sb3.yaml` | Modified | Approved PPO head width |
| `conf/reward/rulebook_defaults.yaml` | Modified | Disable heavy margin trace by default |
| `conf/run_profile/smoke.yaml` | Modified | Enable margin trace for smoke only |
| `src/thesis_rl/runtime/wiring/builders.py` | Modified | Apply profile overrides only to matching fork-backed SB3 backends |
| `tests/test_hydra_preset_run_configs.py` | Modified | TD3/SAC/PPO profile and logging regression coverage |
| `tests/test_hydra_agent_presets.py` | Modified | PPO head-width regression coverage |

## 14. Validation results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync pytest -q tests/test_hydra_preset_run_configs.py tests/test_hydra_agent_presets.py` | PASS | 2026-07-20 | 33 passed |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/runtime/wiring/builders.py tests/test_hydra_preset_run_configs.py tests/test_hydra_agent_presets.py` | PASS | 2026-07-20 | All checks passed |
| `docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/runtime/wiring/builders.py tests/test_hydra_preset_run_configs.py tests/test_hydra_agent_presets.py` | PASS | 2026-07-20 | 3 files already formatted |
| `git diff --check` | PASS | 2026-07-20 | No whitespace errors |
| `make config` | PASS | 2026-07-20 | `docker compose config --quiet` succeeded |

## 15. Final reconciliation

All requirements and acceptance criteria in this focused run-profile plan are
implemented and verified by the Hydra/configuration matrix above. The host-only
direct pytest/Ruff attempts were unavailable because the host lacks those
dependencies; the repository container is the primary environment and its
checks passed.

Known limitation: no 120,000-step training run was launched in this task; the
request was to prepare the command and configuration. The exact command is
`make run ALGORITHM=td3_sb3 RUN_PROFILE=fast`.
