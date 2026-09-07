# ExecPlan: Observation / Signal / Chunk-Boundary Audit Fixes (OBS-AUDIT-FIX-001)

## 1. Metadata

- Feature: fixes for four defects found by the 2026-09-07 audit of the per-step
  observation, the observation-to-agent pipeline and the traffic-light reward
  path.
- Plan ID: `OBS-AUDIT-FIX-001`
- Authoritative specifications (unchanged by this plan):
  `docs/specifications/observation_v1.3_specification.md` (`OBS-V1.3`,
  APPROVED) with `observation_v1.3.1_amendment.md` §2/§3;
  `docs/specifications/observation_v1.2_specification.md` §7 (LiDAR admission,
  actor identity, history gaps);
  `docs/specifications/rulebook_v4.7_specification.md` §7.6.7 (signal state
  mapping, flashing states) as amended through `rulebook_v5.1_specification.md`;
  `docs/specifications/transition_replay_v1_specification.md` (episode
  boundary semantics of the replay buffer).
- Status: `IMPLEMENTED`
- Created: 2026-09-07. Last update: 2026-09-07.
- Branch: `worktree-fix-observation-signal-audit`.
- Approval: explicit user approval 2026-09-07 for A1 (recommended variant),
  A2, A3 and A5 of the audit report, recorded in §6.
- Owner: thesis author.

## 2. Objective And Scope

Four defects were confirmed by code reading and, for A1 and A3, reproduced
empirically in the dev container with synthetic ScenarioNet descriptors:

- **A1.** The OBS-V1.2 LiDAR admission gate compares MetaDrive object ids
  (random UUID names) against ScenarioNet actor ids, so in `obs=semantic_v3`
  the `dynamic`, `interactions` and live-actor `static` blocks are always
  empty. The agent is blind to every other road user.
- **A2.** `observe()` runs twice per environment step (once inside MetaDrive's
  `step()` with the previous causal context, once after the Rulebook commits
  the new context). The `context_history` row for step `k` is appended twice,
  the 21-row window holds about 11 distinct steps and the continuity flags are
  zero in every row but the newest.
- **A3.** MetaDrive simplifies `LANE_STATE_FLASHING_STOP` to `LIGHT_UNKNOWN`
  and `LANE_STATE_FLASHING_CAUTION` to `LIGHT_YELLOW` before the repository
  reads the light object. A flashing red light therefore aborts the episode as
  an invalid signal transition instead of being priced as `RED`, and a flashing
  yellow loses its `NOT_APPLICABLE` treatment.
- **A5.** In the non-ACL training loop every chunk boundary calls
  `env.reset()` on all vector slots without marking the in-flight episodes as
  truncated, so n-step windows and PPO advantages cross episode boundaries.

In scope: the four fixes, the station anchor of the `dynamic` block (required
once A1 makes the block non-empty), regression tests, documentation.

Out of scope (explicitly left to the user): the reward-contract questions of
the audit (running a red light being positive expected value, ADR-061 route
extension, goal placement before a stop line), normalization scales, and every
item listed as non-blocking in the audit report.

Compatibility: no observation schema, dimension, configuration key or
checkpoint identity changes. The numerical content of `dynamic`,
`interactions`, `static` and `context_history` changes for every episode with
traffic, so policies trained before this change are not comparable with
policies trained after it.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-AF-01` | An actor whose LiDAR first hit resolves to it is admitted to the semantic tracker; admission uses one stable identity shared with the Rulebook's `ActorSnapshot.actor_id`. | OBS-V1.2 §7 (admission), OBS-V1.3 §4 |
| `REQ-AF-02` | The dynamic block's route station difference is measured against the committed mission station (single station authority). | DRIVING-MISSION-V1.1 §3/§5; OBS-V1.3 §5.6 |
| `REQ-AF-03` | `context_history` holds one row per actual environment step, right-aligned, with continuity flags set only against the immediately preceding step. | OBS-V1.3 §5.8 |
| `REQ-AF-04` | Live signal states are mapped from the source-recorded state: `FLASHING_STOP → RED`, `FLASHING_CAUTION → FLASHING_YELLOW`, unknown remains `UNKNOWN`. | Rulebook v4.7 §7.6.7 (mapping table already encoded in `_LIVE_SIGNAL_STATE_MAP`) |
| `REQ-AF-05` | Consecutive training chunks on the same live vector environment continue the in-flight episodes; a chunk boundary is not an episode boundary. | TRANSITION-REPLAY v1 §episode boundaries; ADR-075 (`γ`, proper episodes) |

## 4. Current Repository Analysis

- `VERIFIED` `src/thesis_rl/envs/observations/perception.py:30-32,90-107`:
  `_object_id` returns `obj.id`; `sweep()` collects those ids.
- `VERIFIED` `src/thesis_rl/rulebook/v2/context/metadrive_live.py:56-66`:
  `_stable_actor_id` maps `obj.id` through
  `engine.traffic_manager.obj_id_to_scenario_id`.
- `VERIFIED` `third_party/metadrive/.../scenario_traffic_manager.py:208,268,296,332`
  and `base_env.py:272`: object names are random unless
  `force_reuse_object_name=True`; the key is set nowhere in `src/` or `conf/`.
- `VERIFIED` (empirical, dev container, synthetic scenario with one traffic
  vehicle 15 m ahead): sweep ids `{UUID}`, snapshot ids `{"77"}`,
  intersection empty.
- `VERIFIED` `causal_semantic.py:1553-1607` (`build`) has no per-step
  idempotence; `_append_timestamped_context_row` (`:2500-2563`) appends
  unconditionally. `third_party/metadrive/.../base_env.py:620` calls
  `observe()` inside `step()`; `rulebook/v2/wrapper.py:321-324` calls
  `_refresh_causal_observation` which calls `observe()` again
  (`thesis_scenario_env.py:726-747`).
- `VERIFIED` `causal_semantic.py:913` projects the ego without a station anchor
  for the dynamic block; every other OBS-V1.3 block anchors at
  `mission_s_m`.
- `VERIFIED` `third_party/metadrive/metadrive/type.py:221-235` and
  `component/traffic_light/scenario_traffic_light.py:6`: `set_status` receives
  the simplified state; `get_state()["object_state"]` can only be one of
  `TRAFFIC_LIGHT_{RED,YELLOW,GREEN,UNKNOWN}`. The light manager keeps the raw
  per-frame sequence in `_episode_light_data[lane_id]["object_state"]` and
  applies index `episode_step` in `after_step`
  (`scenario_light_manager.py:68-75`), freezing past the scenario length.
- `VERIFIED` `runtime/loops/train_loop.py:1766-1788` passes no
  `initial_observations`; `agent/agent.py:1028` resets when it is `None`;
  `agent.py:1808` already returns `last_observations`;
  `curriculum/scenario_acl/driver.py:1151-1157` forwards it on the ACL path.
  The staged-curriculum path closes and rebuilds the environment
  (`train_loop.py:1894,1989,2372,2596`), so a reset is unavoidable there.
- `VERIFIED` tests mocking `first_hit_lidar_sweep` supply ids identical to the
  snapshot ids (`tests/test_perception_bounded_semantic.py`), which is why A1
  was never caught.

## 5. Assumptions And Invariants

- Actor identity: the ScenarioNet scenario id when the traffic manager exposes
  `obj_id_to_scenario_id`, otherwise the MetaDrive object id (PG maps). Both
  the sweep and the snapshot use the same rule.
- Station anchor: `mission_s_m` is the committed ego station (metres along the
  route); the dynamic block's `Δs` is `actor_s − mission_s_m`, clipped at 50 m.
- `context_history`: one row per `(scenario_id, step_index)`; a repeated
  `build()` for the same key returns the batch already built for that step.
- Signal state: read from the source sequence at index
  `min(episode_step, len − 1)`; the simplified object state is only a fallback
  when the raw sequence is unavailable (unit-test fakes, non-ScenarioNet
  lights). `UNKNOWN` still fails closed.
- Chunk boundary: `train_vectorized` receives the previous chunk's
  `last_observations` and `last_episode_lengths` when the environment object is
  unchanged; any environment rebuild clears the carry.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-AF-001` | specification clarification | Which identity the LiDAR sweep resolves to. | (a) map sweep ids through `obj_id_to_scenario_id` like `_stable_actor_id`; (b) set `force_reuse_object_name=True` globally. | (a): one normalization rule, no global MetaDrive naming change. | `dynamic`/`interactions`/`static` populated. | Approved by user 2026-09-07 |
| `DEC-AF-002` | implementation detail | How to make the double `observe()` harmless. | (a) guard only the context-row append; (b) cache the whole batch per `(scenario, step)`. | (b): the stale mid-step call also mixes the live vehicle pose of step `k+1` with the context of step `k`; returning the batch built for step `k` is strictly more consistent and saves one full build per step. | `context_history` correct; ~half the observation build cost. | Approved by user 2026-09-07 (fix as recommended) |
| `DEC-AF-003` | specification deviation (runtime source) | Where the live signal state is read. | (a) simplified light object state (current); (b) raw source sequence at the current frame. | (b): the approved mapping table only applies to raw states; still causal (current frame only). | Flashing red priced as RED; flashing yellow `NOT_APPLICABLE`. | Approved by user 2026-09-07 |
| `DEC-AF-004` | implementation detail | Chunk boundary continuation. | (a) forward `last_observations` like the ACL driver; (b) mark truncation before the reset. | (a): no episode is lost; also forward per-slot episode lengths so the abort-closure guard stays correct. | Replay/PPO targets no longer cross chunk boundaries. | Approved by user 2026-09-07 |

## 7. Proposed Design

- `perception.py`: `_stable_object_id(mapping, obj)` resolves `obj.id` through
  the traffic manager mapping fetched once per sweep; ego and candidates use
  it. `FirstHitLidarSweep.actor_ids` therefore carries the same ids as
  `ActorSnapshot.actor_id`.
- `causal_semantic.py` (`PerceptionBoundedSemanticBatchBuilder`): `build()`
  caches the last `(scenario_id, step_index)` batch and returns it on a repeat;
  `reset()` clears the cache. `_dynamic_features` override replaces the
  station difference with `actor_s − mission_s_m`.
- `metadrive_live.py`: `live_signal_states_by_physical_id` reads the raw state
  from `manager._episode_light_data[pid]["object_state"][clamp(episode_step)]`
  when available, else falls back to the object state; mapping unchanged.
- `agent.py`: new optional `initial_episode_lengths`; summary gains
  `last_episode_lengths`. `train_loop.py`: carries both across chunks on the
  vectorized path; the carry is cleared wherever the environment is closed or
  rebuilt.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-AF-01` | `AC-AF-01` sweep ids equal snapshot ids for a replayed actor | `envs/observations/perception.py::_stable_object_id`, `FirstHitLidarAdapter.sweep` | `tests/test_perception_actor_identity.py::test_sweep_resolves_scenario_ids_through_traffic_manager_mapping`, `::test_sweep_keeps_object_ids_without_scenario_mapping`, `tests/test_perception_first_hit_scenario_ids.py::test_scenario_online_env_sweep_ids_match_live_snapshot_ids` | Implemented |
| `REQ-AF-02` | `AC-AF-02` dynamic `Δs` uses the committed station on a self-approaching route | `causal_semantic.py::PerceptionBoundedSemanticBatchBuilder._dynamic_features` | `tests/test_perception_bounded_semantic.py::test_v13_dynamic_station_difference_uses_committed_mission_station` | Implemented |
| `REQ-AF-03` | `AC-AF-03` two builds at the same step, then one at the next step: mask sum 2, continuity flags preserved | `causal_semantic.py::PerceptionBoundedSemanticBatchBuilder.build` | `tests/test_perception_bounded_semantic.py::test_v13_repeated_build_for_the_same_step_does_not_duplicate_the_context_row` | Implemented |
| `REQ-AF-04` | `AC-AF-04` raw `FLASHING_STOP` → `RED`, `FLASHING_CAUTION` → `FLASHING_YELLOW`, index clamped, fallback preserved | `metadrive_live.py::live_signal_states_by_physical_id` | `tests/test_rulebook_v2_metadrive_live.py::test_metadrive_signal_provider_reads_raw_source_state_when_object_state_is_simplified`, `::test_metadrive_signal_provider_clamps_raw_index_past_scenario_length` | Implemented |
| `REQ-AF-05` | `AC-AF-05` second chunk with carried state performs no `reset()` and continues from the carried observations and lengths | `agent/agent.py::Agent.train_vectorized`, `runtime/loops/train_loop.py` | `tests/test_train_vectorized_chunk_continuation.py::test_carried_observations_skip_reset_and_continue_episode_lengths`, `::test_train_loop_forwards_last_observations_between_chunks` | Implemented |

## 9. Test Strategy

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-AF-01` | Unit | Sweep maps object ids through the traffic-manager mapping | Fake engine with `obj_id_to_scenario_id`, patched `DistanceDetector.perceive` | `actor_ids == {"77"}`, ego excluded | `REQ-AF-01` |
| `TEST-AF-02` | Unit | Without a mapping the object id is kept | Same, mapping absent | `actor_ids == {object id}` | `REQ-AF-01` |
| `TEST-AF-03` | Integration | Real `ScenarioOnlineEnv` with one traffic vehicle 15 m ahead | Synthetic descriptor | sweep ids ∩ snapshot ids non-empty | `REQ-AF-01` |
| `TEST-AF-04` | Unit | Repeated `build()` at one step | Fixture context, same step twice then step+1 | `context_history_mask.sum() == 2`, continuity flag of row `k` intact, cached batch identical | `REQ-AF-03` |
| `TEST-AF-05` | Unit | Dynamic `Δs` on a U-shaped route with committed station on the near branch | Route with parallel return branch, ego lateral 1.6 m | `dynamic[slot,-1,17] ≈ 10/50`, not `−1` | `REQ-AF-02` |
| `TEST-AF-06` | Unit | Raw flashing states | Fake manager with `_episode_light_data` and simplified object states | `RED`, `FLASHING_YELLOW` | `REQ-AF-04` |
| `TEST-AF-07` | Unit | Raw index clamp past length | `episode_step` beyond sequence | last state used | `REQ-AF-04` |
| `TEST-AF-08` | Unit | `train_vectorized` with carried state | Fake two-slot vector env counting resets | `reset()` not called; first stored batch obs == carried; summary lengths continue | `REQ-AF-05` |
| `TEST-AF-09` | Unit | `train_loop` forwards the carry | Source inspection helper `chunk_carry_kwargs` | returns `initial_observations`/`initial_episode_lengths` from a summary, `{}` when cleared | `REQ-AF-05` |

Commands (verified available, run inside the provisioned container):
`uv run --no-sync python -m pytest -q <files>`; `make lint`;
`make format-check PYTHON_QUALITY_PATHS="<changed files>"`; `make smoke`.

## 10. Milestones

- [x] M1 A1 + dynamic anchor: `perception.py`, `causal_semantic.py`, tests.
- [x] M2 A2 batch cache: `causal_semantic.py`, test.
- [x] M3 A3 raw signal state: `metadrive_live.py`, tests.
- [x] M4 A5 chunk continuation: `agent.py`, `train_loop.py`, tests.
- [x] M5 Validation: focused tests, full suite, lint, focused format check,
  smoke on the production observation path; documentation and index update.

## 11. Progress And Findings Log

- 2026-09-07: audit completed (three parallel reviews, all claims re-verified
  in code; A1 and A3 reproduced empirically with synthetic descriptors in the
  dev container). User approved A1 (variant a), A2, A3, A5. Plan created.
- 2026-09-07: M1-M4 implemented with regression tests; see §14 for results.
- 2026-09-07: **overlap with an unmerged branch.** `worktree-audit-block-a-fixes`
  (two commits on top of the *local* `main` at `9c4c2b0`, i.e. without
  `origin/main`'s crash-safe-resume commits `a246068`/`422ba99`) fixes the same
  two defects as A2 and A5 under its own identifiers `C18` (row-level guard in
  `_append_timestamped_context_row`) and `C17` (`previous_chunk_last_observations`
  forwarded when `curriculum_manager is None`, also into the PPO overshoot
  collection). This plan's branch is based on `origin/main` (`422ba99`). Both
  branches touch `causal_semantic.py` (different hunks, auto-mergeable) and
  `train_loop.py` (same hunk around `extra_train_kwargs`, textual conflict).
  This plan's versions are supersets: the batch cache subsumes the row guard
  and also removes the second observation build per step; `chunk_carry_kwargs`
  adds the environment-identity check and the per-slot episode lengths. That
  branch also registers `C13`-`C20` in `docs/open_items.md`, colliding with
  `origin/main`'s `C13`; this plan follows the register rule (next free number
  on `origin/main`, `C14`-`C17`). Whichever branch merges second must
  renumber. **Not resolved here**: the other branch's owner must rebase it onto
  `origin/main`; its own conflicts with the resume feature (`driver.py`,
  `train_loop.py`) are unrelated to this plan.

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-AF-001` | Live signal state read from the MetaDrive light object | Read from the source sequence at the current frame, object state as fallback | MetaDrive discards the flashing distinction before the object is readable | User 2026-09-07 (`DEC-AF-003`) | `tests/test_rulebook_v2_metadrive_live.py` |

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/envs/observations/perception.py` | Modified | `REQ-AF-01` |
| `src/thesis_rl/envs/observations/causal_semantic.py` | Modified | `REQ-AF-02`, `REQ-AF-03` |
| `src/thesis_rl/rulebook/v2/context/metadrive_live.py` | Modified | `REQ-AF-04` |
| `src/thesis_rl/agent/agent.py` | Modified | `REQ-AF-05` |
| `src/thesis_rl/runtime/loops/train_loop.py` | Modified | `REQ-AF-05` |
| `tests/test_perception_actor_identity.py` | Added | `TEST-AF-01/02` |
| `tests/test_perception_first_hit_scenario_ids.py` | Added | `TEST-AF-03` |
| `tests/test_perception_bounded_semantic.py` | Modified | `TEST-AF-04/05` |
| `tests/test_rulebook_v2_metadrive_live.py` | Modified | `TEST-AF-06/07` |
| `tests/test_train_vectorized_chunk_continuation.py` | Added | `TEST-AF-08/09` |
| `docs/implementation/observation_signal_audit_fixes_v1_exec_plan.md` | Added | this plan |
| `docs/open_items.md`, `docs/project_index.md` | Modified | register |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| Focused pytest: `tests/test_perception_actor_identity.py tests/test_perception_bounded_semantic.py tests/test_rulebook_v2_metadrive_live.py tests/test_train_vectorized_chunk_continuation.py tests/test_perception_first_hit_scenario_ids.py tests/test_observation_v13_corrections.py tests/test_causal_semantic_batch.py tests/test_train_vectorized_data_abort.py tests/test_train_vectorized_geometry_abort.py` (dev container) | `PASS` | 2026-09-07 | 110 passed in 10.08 s |
| Same new tests with the five pre-fix source files bind-mounted read-only over the worktree | `PASS` (as evidence) | 2026-09-07 | 7 failed / 35 passed: both mapping tests, the `ScenarioOnlineEnv` integration test, the raw-state and raw-unknown signal tests, the repeated-build test and the station-anchor test fail on the pre-fix tree; `tests/test_train_vectorized_chunk_continuation.py` does not import (`chunk_carry_kwargs` absent). Every behavioural regression test therefore detects its defect. |
| Full suite `uv run --no-sync python -m pytest -q -p no:cacheprovider` (dev container) | `PASS` | 2026-09-07 | 1730 passed, 1 warning, 32 min 38 s. Two aborted `git rebase` attempts briefly swapped source files on disk during this run (see §11); no failure resulted, and the focused files were re-run afterwards. |
| `ruff check src tests scripts` | `PASS` | 2026-09-07 | All checks passed |
| `ruff format --check` on the five new/materially modified Python files | `PASS` after `ruff format` | 2026-09-07 | `perception.py`, `metadrive_live.py` and the new integration test were reformatted (three whitespace-only hunks in pre-existing lines of the two source files); `causal_semantic.py`, `agent.py`, `train_loop.py` were not reformatted because the repository formatting baseline for them is not clean and the change is a few lines each. |
| Production-path smoke on CPU: `presets/test/smoke_train obs=semantic_v3 agent/planner/encoder=lq_v3 reward=scalar_reward env.vectorized.enabled=true env.vectorized.num_envs=2 device=cpu` | `PARTIAL` | 2026-09-07 | Run `outputs/obs_audit_fix_smoke/td3_sb3/seed_42/20260907_110938`: both training chunks completed (2000 steps, chunk boundary crossed with the carried observations, 5 episodes per chunk, no abort, no error), validations at 1000 and 2000, final panels `test_waymo_empirical` and `test_pg` completed; the 60-minute wrapper timeout killed the run during the third final panel (`run_interrupted`). Training path fully exercised; the final-panel tail is covered by the GPU run below. |
| Production-path smoke on GPU: same overrides with `compose.gpu.yaml` and `device=cuda` | `PASS` | 2026-09-07 | Run `outputs/obs_audit_fix_smoke_gpu/td3_sb3/seed_42/20260907_121017` (GH200): exit 0; two chunks (2000 steps), two intermediate validations, all three final panels (`evaluation_finished` x7), no abort, no `run_interrupted`. |

## 15. Final Reconciliation

| Requirement | AC | Status | Notes |
|---|---|---|---|
| `REQ-AF-01` | `AC-AF-01` | `VERIFIED` | Unit tests with a fake engine, integration test on a real `ScenarioOnlineEnv`; the pre-fix reproduction (disjoint id sets) is now the integration test. |
| `REQ-AF-02` | `AC-AF-02` | `VERIFIED` | U-shaped route fixture where the un-anchored projection provably picks the far branch. |
| `REQ-AF-03` | `AC-AF-03` | `VERIFIED` | Production call pattern (stale build then committed build) over 24 steps yields the full 21-row mask. |
| `REQ-AF-04` | `AC-AF-04` | `VERIFIED` | Raw flashing states mapped, index clamped, raw `UNKNOWN` still fails closed, fallback preserved for fakes. A live reproduction with a real flashing-red descriptor was done before the fix (audit probe), not re-run after it. |
| `REQ-AF-05` | `AC-AF-05` | `VERIFIED` | Agent-level continuation test plus helper tests; the loop wiring is pinned by source inspection and exercised end-to-end by the smoke run (two chunks). |

**Known limitations.**
- Episode statistics of the episode straddling a chunk boundary are split
  across the two chunk summaries (length and return of the part before the
  boundary belong to the previous chunk). The replay content is correct; only
  the per-chunk reporting is affected, as it already was on the ACL path.
- The dynamic block still projects *other* actors without a station anchor
  (none is available for them); only the ego station was corrected.
- The raw-state read depends on the light manager's private
  `_episode_light_data` attribute, as the existing provider already depended on
  `_scenario_id_to_obj_id`. A MetaDrive upgrade renaming it silently restores
  the simplified-state fallback; the regression tests would not notice because
  they use fakes. A follow-up could assert the raw sequence is present on
  ScenarioNet environments.

**Deferred, user decisions (not part of this plan).** Reward contract around
red lights (A4), ADR-061 route extension for controls past the route end, goal
placement relative to stop lines, normalization scales, and the other
non-blocking audit items.

**Readiness.** The observation now contains other road users; every policy
trained on `semantic_v3` before this change is not comparable with later ones.
