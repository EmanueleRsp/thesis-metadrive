# Algorithm-invariant run configuration and event logging

## 1. Metadata

- Feature/plan ID: `ALGORITHM-INVARIANT-RUN-CONFIG`
- Authoritative specification: `docs/specifications/rl_baselines_v1_specification.md`, ID `RL-BASELINES`, version `1.0`, `APPROVED`, `AUTHORITATIVE: YES`.
- Status: `VERIFIED`
- Created: 2026-07-21
- Last update: 2026-07-21
- Branch: current working branch
- Related ADRs: `ADR-016-scenario-acl-vectorized-execution.md`
- Owner: repository maintainer

## 2. Objective and scope

Ensure that changing `make run ALGORITHM=...` changes only the selected learner
and explicitly algorithm-dependent configuration. Shared environment,
observation, encoder, decoder bridge, reward/scalarization, curriculum,
vectorization, logging, evaluation, video, seed, and run-profile behavior must
remain unchanged. Standardize the visible completed-episode event line across
scalar and vector training paths so it cannot regress when a different learner
selects a path with equivalent semantics.

Out of scope: changes to RL formulas, reward semantics, dataset policy, ACL
selection, or the approved PPO/TD3/SAC hyperparameters.

## 3. Authoritative requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-AIR-001` | The three fork-backed algorithms share the same scientific pipeline and differ only in approved algorithm-native mechanisms. | RL-BASELINES §§1.2, 2.1, 8 |
| `REQ-AIR-002` | PPO has no transition replay or PER; TD3/SAC retain their approved replay/PER behavior. | RL-BASELINES §§1.4, 3.4, 9.6, 12 |
| `REQ-AIR-003` | Run-profile overrides apply only to the selected algorithm and preserve shared values. | Existing approved run-profile plan and RL-BASELINES §3.4 |
| `REQ-AIR-004` | Completed-episode monitor events have one stable schema independent of training path. | RL-BASELINES §§8, 10 and runtime logging contract |

## 4. Current repository analysis

- **VERIFIED:** `Makefile:run-train` supplies the common environment,
  observation, encoder, decoder, reward, scalarization, curriculum, provider,
  vectorization, seed, profile, name, and video arguments; only
  `agent/planner/algorithm=$(ALGORITHM)` selects the learner.
- **VERIFIED:** `_resolve_planner_cfg()` in
  `src/thesis_rl/runtime/wiring/builders.py` applies only the matching
  `planner.td3`, `planner.sac`, or `planner.ppo` profile block to the selected
  SB3 fork.
- **VERIFIED:** replay validation in
  `src/thesis_rl/sb3_extensions/replay/config.py` rejects transition replay
  and PER for `ppo_sb3` while retaining the configured TD3/SAC replay policy.
- **VERIFIED:** `Agent.train()` writes the legacy completed-episode string at
  `src/thesis_rl/agent/agent.py:618`, while `Agent.train_vectorized()` writes a
  newer string at `src/thesis_rl/agent/agent.py:1255`.
- **INFERRED:** the observed PPO discrepancy is caused by reaching the scalar
  formatter or by a stale runtime artifact, not by an algorithm-specific event
  contract. The observable fix is to share one formatter and test both paths.

## 5. Assumptions and invariants

- `env=ScenarioNet`, semantic observation v1.1, LQ encoder, scalarized reward,
  ACL, provider strictness, seed, run profile, and video settings are shared.
- PPO remains on-policy with GAE and no PER/replay; this is not changed.
- TD3/SAC retain 3-step replay and proportional PER; this is not changed.
- A scalar training path has one logical environment slot, represented as
  `env=0` in the stable monitor line. Vector training retains its worker slot.
- Event ordering and all numeric metrics remain unchanged; only presentation
  formatting is centralized.

## 6. Decisions and approval gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-AIR-001` | implementation detail | How to prevent formatter drift? | Duplicate compatible strings; shared formatter helper | Shared private formatter used by scalar and vector paths | Stable monitor output, no scientific behavior change | Approved by user request, 2026-07-21 |
| `DEC-AIR-002` | implementation detail | Scalar path has no worker index | Omit slot; add canonical slot `0` | Add `env=0` to match vector schema | Presentation-only compatibility change | Approved by user request, 2026-07-21 |

## 7. Proposed design

Add a private formatter in `Agent` that accepts episode id, slot, length,
rewards, termination reason, route completion, and optional ACL context. Replace
both existing inline f-strings with this helper. Add deterministic tests for
scalar and vector formatting, plus Hydra composition tests comparing shared
configuration fingerprints across `ppo_sb3`, `td3_sb3`, and `sac_sb3` while
allowing only documented algorithm-native fields to differ.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-AIR-001` | Shared Hydra fields are equal for all three algorithms | `Makefile`, Hydra composition | `tests/test_hydra_preset_run_configs.py` | Planned |
| `REQ-AIR-002` | PPO replay/PER is disabled; TD3/SAC matrix remains unchanged | replay resolver and algorithm configs | existing replay/config tests plus Hydra regression | Planned |
| `REQ-AIR-003` | Only matching profile block changes learner-specific fields | `_resolve_planner_cfg()` | existing profile matrix tests | Planned |
| `REQ-AIR-004` | Scalar and vector paths produce identical event schema | `src/thesis_rl/agent/agent.py` | `tests/test_agent_pipeline.py` | Planned |

## 9. Test strategy defined before implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-AIR-001` | Unit | scalar event formatting | deterministic episode payload | stable `Episode ... env=0 ...` line | `REQ-AIR-004` |
| `TEST-AIR-002` | Unit | vector event formatting | deterministic slot and payload | same schema with selected slot | `REQ-AIR-004` |
| `TEST-AIR-003` | Hydra | shared run composition | each SB3 algorithm, same profile | shared fields equal; only algorithm-native fields differ | `REQ-AIR-001`, `REQ-AIR-003` |
| `TEST-AIR-004` | Hydra | replay compatibility | PPO, TD3, SAC | PPO replay/PER off; TD3/SAC approved values retained | `REQ-AIR-002` |

Commands: `pytest -q tests/test_agent_pipeline.py tests/test_hydra_preset_run_configs.py tests/test_transition_replay_config.py`, focused Ruff on modified Python files, `git diff --check`, and `make config`.

## 10. Milestones

- [x] M1 — Trace configuration and identify duplicated event formatters.
- [x] M2 — Add regression tests.
- [x] M3 — Centralize event formatting.
- [x] M4 — Run focused validation and reconcile the plan.

## 11. Progress and findings log

- 2026-07-21: Confirmed the Makefile passes the same runtime arguments for all
  algorithms and selects only the algorithm group. Confirmed two separate
  event-line formatters in `Agent`; the previous claim that PPO inherently
  selected the old path was not proven.
- 2026-07-21: Added `_format_episode_event()` and routed both scalar and vector
  monitor paths through it. The scalar path now emits the same schema as the
  vector path with canonical `env=0`; callback context is evaluated once.
- 2026-07-21: Added Hydra regression coverage for the exact common overrides
  used by `make run`, across `ppo_sb3`, `td3_sb3`, and `sac_sb3`. Verified PPO
  replay/PER is disabled while TD3/SAC retain replay/PER.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/agent/agent.py` | Planned modification | Shared event formatter |
| `tests/test_agent_pipeline.py` | Planned modification | Scalar/vector event regression tests |
| `tests/test_hydra_preset_run_configs.py` | Planned modification | Algorithm-invariant composition matrix |
| `docs/implementation/algorithm_invariant_run_configuration_exec_plan.md` | Added | Traceability and validation record |

## 14. Validation results

| Command | Result | Date | Notes |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_agent_pipeline.py tests/test_hydra_preset_run_configs.py tests/test_transition_replay_config.py` | PASS | 2026-07-21 | 42 passed |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/agent/agent.py tests/test_agent_pipeline.py tests/test_hydra_preset_run_configs.py` | PASS | 2026-07-21 | All checks passed |
| `docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/agent/agent.py tests/test_agent_pipeline.py tests/test_hydra_preset_run_configs.py` | PASS | 2026-07-21 | 3 files already formatted |
| `git diff --check` | PASS | 2026-07-21 | No whitespace errors |
| `docker compose config --quiet` | PASS | 2026-07-21 | Compose configuration valid |

## 15. Final reconciliation

`REQ-AIR-001` is VERIFIED by the Hydra shared-pipeline matrix. `REQ-AIR-002`
is VERIFIED by the existing replay tests and the new PPO/TD3/SAC composition
assertions. `REQ-AIR-003` is VERIFIED by the existing run-profile matrix and
the new exact Makefile override matrix. `REQ-AIR-004` is VERIFIED by the
formatter regression test and the two runtime paths now sharing the helper.

Known limitation: no full GPU training run was launched; the change is
presentation-only and the focused deterministic tests plus Compose validation
passed. No deviations or unresolved issues remain.
