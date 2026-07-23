# Runtime Scenario Data-Abort v1 ExecPlan

## 1. Metadata

- Feature / plan ID: typed runtime scenario data-abort and quarantine / `RSA-V1`.
- Status: `IMPLEMENTED` (focused and fixture-level validation only; not `VERIFIED`).
- Created: 2026-07-23. Last updated: 2026-07-23 (evaluation-path exclusion, ACL and non-ACL run-local quarantine persistence across resume, forensic JSONL wiring, and a prefix-preserving PPO/GAE boundary implementation replacing the earlier full-rollout discard, all added and tested).
- Authority: user-approved amendments recorded in ADR-024; Rulebook v4.7, Transition Replay v1, ACL v1, RL Baselines v1, and ScenarioNet v1.1.
- Related ADR: `docs/decisions/ADR-024-runtime-scenario-data-abort.md`.

## 2. Objective and scope

Preserve only valid trajectory prefixes when an explicitly typed live Rulebook-scenario data defect occurs. Covers worker transport, off-policy replay/N-step, PPO/GAE, ACL, run-local persistence, evaluation, forensic logs, and comparison artifacts. Excludes dataset rebuilding, preflight of all scenarios, reward synthesis, and recovery from untyped, numerical, serialization, assertion, CUDA, or learner errors.

## 3. Authoritative requirements

| ID | Requirement |
|---|---|
| `REQ-RSA-001` | Recover only the enumerated typed runtime scenario errors; all others are fatal. |
| `REQ-RSA-002` | Discard failed action/reward; truncate only the previous valid transition with final observation. |
| `REQ-RSA-003` | Flush N-step and PPO GAE at the slot-local boundary without crossing reset. |
| `REQ-RSA-004` | Keep worker alive and reset only the aborted vector slot. |
| `REQ-RSA-005` | Quarantine run-locally, persist/resume it, and keep dataset artifacts immutable. |
| `REQ-RSA-006` | Suppress every ACL learning/buffer/MAB outcome for the aborted episode. |
| `REQ-RSA-007` | Exclude runtime-invalid evaluation episodes from aggregates and publish coverage/per-UID data. |
| `REQ-RSA-008` | Write forensic JSONL and aggregation artifacts. |

## 4. Current repository analysis

- `VERIFIED`: `controls.evaluate_signal_transition()` translates the invalid pre/post signal-state case to `RuntimeScenarioNotEvaluableError(INVALID_SIGNAL_TRANSITION)`; every other raise in that function remains a plain, fatal `ValueError`. Only this one reason code is wired to a call site today; the other four enumerated codes are reserved for future call sites.
- `VERIFIED`: `deterministic_subproc_vec_env.py` translates only `RuntimeScenarioNotEvaluableError` into a distinct data-abort worker reply and keeps the worker alive; every other worker exception remains a fatal envelope that terminates the worker.
- `VERIFIED`: `PrioritizedNStepReplayBuffer` computes N-step at sampling time; the retroactive close (`close_previous_transition_as_data_abort`) works on the already-stored prior row, and an invalid row's priority is forced to zero so it can never be sampled as a window start. Uniform (non-PER) TD3/SAC replay lacks this machinery and remains deliberately fatal on a typed abort, by design (see Findings).
- `VERIFIED`: PPO now preserves the valid rollout prefix at an abort boundary, matching the literal wording of `REQ-RSA-003`. An owned `MaskedRolloutBuffer` (`sb3_extensions/rollout/masked.py`) keeps SB3's shared, lockstep write cursor but adds a per-cell validity mask: the aborted env's cell for that timestep is never populated with real data (only its `episode_starts` flag is written, which is what makes the *previous* row's GAE computation correctly stop propagating advantage across the boundary), and the minibatch sampler (`get()`) filters invalid cells out before they can ever reach a gradient step. `close_previous_transition_as_data_abort` bootstraps the preceding valid row's reward with `gamma * V(final_observation)`, mirroring the existing `TimeLimit`-truncation convention already used elsewhere in this backend. This replaces the earlier full-rollout-discard implementation (see Findings) and required no change to `compute_returns_and_advantage` itself.
- `VERIFIED`: evaluation (`Agent._evaluate_parallel`) now recognizes a `RuntimeScenarioDataAbort` result from `step_slots`, excludes that episode from aggregates, quarantines its `scenario_uid`, and reports `data_abort_coverage` (attempted/valid/invalid/per-episode reason codes). Previously this path crashed with an unhandled `TypeError`.
- `VERIFIED`: run-local quarantine is persisted alongside the checkpoint and restored on resume for both the standard/vectorized training path (`train_loop.py`, sidecar `latest_quarantine_state.json`, reusing `RuntimeScenarioQuarantine`) and the ACL vectorized path (`AclVectorState.quarantined_scenario_uids`, `VECTOR_STATE_VERSION` bumped to `2`). A run that never touches these paths (e.g. a fresh, non-resumed run) starts with an empty quarantine, as required.
- `VERIFIED`: the forensic JSONL (`append_data_abort_record`) is now called from `Agent.train_vectorized` at the exact point a data-abort is detected, with `run_id`, `scenario_uid`, `reason_code`, step, and traceback; confirmed end-to-end with a fixture that drives the real `train_vectorized` loop through an injected abort and inspects the resulting file on disk.

## 5. Design and invariants

`RuntimeScenarioNotEvaluableError` contains reason, diagnostic payload and original cause. Worker replies use a data-abort marker, not a worker-error marker. Parent collectors receive a slot-indexed abort event with valid post-step observation and failed action held only transiently for logging. The off-policy/PER path avoids sampling the failed slot's row (zero priority, non-addressable) and retroactively patches the last same-slot valid transition into a boundary (`timeout=true`). PPO now uses the identical retroactive-boundary idiom through its own `close_previous_transition_as_data_abort`/`MaskedRolloutBuffer`: the previous row's reward is bootstrapped and the failed step's cell is masked out of GAE/minibatch sampling rather than ever being written with real (or synthetic) data. All three algorithms (PPO/TD3/SAC) therefore share the same `agent.py` handling path with no algorithm-specific branch. Quarantine is owned by a runtime component (`ThesisScenarioEnv.scenario_excluded_uids`, broadcast via `env_method`), injected as provider exclusions, and persisted alongside run/ACL checkpoint state so a resume restores it; a fresh run starts with an empty quarantine and dataset artifacts are never mutated.

## 6. Acceptance and mandatory tests

| Test | Expected behavior | Requirement | Status |
|---|---|---|---|
| `TEST-RSA-001` | First-step typed abort inserts zero rows, quarantines, continues. | 1, 2, 5 | Covered: worker-level (`test_typed_runtime_scenario_abort_keeps_worker_alive`) plus a full `train_vectorized` first-step-of-episode abort (`test_train_vectorized_writes_forensic_jsonl_and_quarantines_on_data_abort`), which also exercises the `episode_len <= 1` guard that skips a non-existent prior-row close. |
| `TEST-RSA-002` | Later abort preserves prefix; failed action absent; prior row truncated. | 2 | Covered: `test_per_data_abort_leaf_is_non_addressable_and_closes_previous_transition`. |
| `TEST-RSA-003` | `n=1` and `n=3` close queues, retain valid bootstrap, never cross reset. | 3 | Covered for off-policy/PER (`n=3` case in `test_per_data_abort_leaf_is_non_addressable_and_closes_previous_transition`; sampling boundary logic is exercised by the existing PER sampling tests). `n=1` is not separately exercised for PER; not blocking since the boundary logic is `n_steps`-independent. |
| `TEST-RSA-004` | PPO GAE ends at boundary and uses final-state value. | 3 | Covered: buffer-level (`test_ppo_masked_rollout_buffer.py`, four tests covering reward bootstrap, the missing-previous-row/invalid-previous-row guards, minibatch exclusion of the invalid cell, and no advantage leakage across the boundary) and backend-level with a real `Sb3PpoPlannerBackend`/2-env `DummyVecEnv` (`test_ppo_sb3_data_abort.py::test_ppo_sb3_data_abort_preserves_peer_env_and_bootstraps_boundary`), which drives a real abort through `close_previous_transition_as_data_abort` + masked `observe_transition_batch`, confirms the peer env's row is untouched, and confirms the rollout completes and trains (finite actor/critic loss) afterward. |
| `TEST-RSA-005` | Only failed vector slot resets; peers progress. | 4 | Covered indirectly: `test_train_vectorized_writes_forensic_jsonl_and_quarantines_on_data_abort` has a peer slot that keeps stepping normally while slot 1 aborts and resets. No dedicated multi-slot ordering assertion (see `TEST-RSA-011`). |
| `TEST-RSA-006` | ACL performs no LP/MAB/insert/update and removes existing record. | 6 | Covered: `driver.py`'s `vector_episode_end_callback` filtering and `buffer.remove_scenario_id` are exercised by existing ACL vectorized fixtures; no regression added in this pass beyond the quarantine-persistence tests below. |
| `TEST-RSA-007` | Resume keeps quarantine; new run does not inherit it/dataset stays immutable. | 5 | Covered for both paths: non-ACL (`test_quarantine_state_round_trips_across_checkpoint_and_resume`, `test_quarantine_load_is_a_no_op_without_a_persisted_file`) and ACL (`test_acl_vector_state_persists_runtime_quarantine_across_resume`, `test_acl_vector_state_rejects_a_mismatched_version`). |
| `TEST-RSA-008` | Evaluation aggregates exclude invalid and report coverage/per-UID. | 7 | Covered: `test_parallel_evaluation_excludes_data_abort_episode_from_aggregates`. |
| `TEST-RSA-009` | Untyped worker error remains fatal. | 1 | Covered (pre-existing): `test_worker_python_exception_reports_remote_traceback`. |
| `TEST-RSA-010` | JSONL has scenario, cause, traceback, step and fingerprints. | 8 | Covered: format/hashing at `test_data_abort_jsonl_keeps_observation_out_of_text_record`; end-to-end wiring from a real training loop at `test_train_vectorized_writes_forensic_jsonl_and_quarantines_on_data_abort`. |
| `TEST-RSA-011` | First and last vector slots preserve order/observations. | 4 | Not covered by a dedicated test; only implied by the two-slot fixture above (slot 0 first, slot 1 aborts). No assertion on ordering across more than two slots or on last-slot behavior specifically. |

Commands executed this pass: focused `docker compose run --rm dev uv run --no-sync python -m pytest -q` over the modules listed in §8, and the full suite (`877 passed`, plus 4 pre-existing failures in `test_causal_semantic_batch.py` unrelated to this feature — route-projection geometry, tracked separately); focused Ruff check/format over the same files; `git diff --check`. `make smoke` (or any real SB3 + MetaDrive end-to-end training run) was **not** executed; all validation above is unit-, buffer-, or fixture-level (including one real `Sb3PpoPlannerBackend` + `DummyVecEnv` backend-level test for PPO).

## 7. Milestones

- [x] M1: authority, current code, amendment, ADR, and test matrix.
- [x] M2: typed error, diagnostics, worker/parent protocol, and quarantine state.
- [x] M3: off-policy/PER N-step boundary and PPO/GAE boundary both preserve the valid prefix and are tested (buffer-level and, for PPO, backend-level with a real SB3 model).
- [x] M4: ACL, checkpoint/resume, evaluation/reporting, and forensic artifacts.
- [~] M5: focused regressions, the full test suite, Ruff, and `git diff --check` pass; the mandatory representative smoke test (`make smoke` or an equivalent real SB3/MetaDrive run) has not been executed, so full reconciliation is still pending.

## 8. Traceability

| Requirement | Implementation | Test(s) |
|---|---|---|
| `REQ-RSA-001` | `rulebook/v2/errors.py` (`RuntimeScenarioNotEvaluableError`/`Reason`); `rulebook/v2/components/controls.py::evaluate_signal_transition`; `runtime/execution/deterministic_subproc_vec_env.py` (`_worker`, `RuntimeScenarioDataAbort`) | `test_rulebook_v2_signal.py::test_signal_unknown_is_typed_runtime_scenario_data_abort`; `test_deterministic_subproc_vec_env.py::test_typed_runtime_scenario_abort_keeps_worker_alive`, `::test_worker_python_exception_reports_remote_traceback` |
| `REQ-RSA-002` | `sb3_extensions/replay/prioritized.py::PrioritizedNStepReplayBuffer.add/close_previous_transition_as_data_abort` | `test_transition_replay_per.py::test_per_data_abort_leaf_is_non_addressable_and_closes_previous_transition` |
| `REQ-RSA-003` | Off-policy: as above. PPO: `sb3_extensions/rollout/masked.py::MaskedRolloutBuffer` (`add`, `close_previous_transition_as_data_abort`, `get`); `agent/planners/algorithms/ppo_sb3.py::observe_transition_batch/close_previous_transition_as_data_abort` | `test_ppo_masked_rollout_buffer.py` (four tests); `test_ppo_sb3_data_abort.py::test_ppo_sb3_data_abort_preserves_peer_env_and_bootstraps_boundary` |
| `REQ-RSA-004` | `runtime/execution/deterministic_subproc_vec_env.py::reset_slots(force=True)`; `agent/agent.py::train_vectorized` per-slot abort handling | `test_train_vectorized_data_abort.py::test_train_vectorized_writes_forensic_jsonl_and_quarantines_on_data_abort` |
| `REQ-RSA-005` | `envs/thesis_scenario_env.py::quarantine_scenario_uid/get_quarantined_scenario_uids`; `runtime/data_abort.py::RuntimeScenarioQuarantine`; `runtime/loops/train_loop.py::_save_quarantine_state/_load_quarantine_state`; `curriculum/scenario_acl/vectorized.py::AclVectorState.quarantined_scenario_uids` (`VECTOR_STATE_VERSION=2`); `curriculum/scenario_acl/driver.py` load/save wiring | `test_runtime_quarantine_checkpoint.py` (both tests); `test_scenario_acl_vectorized_state.py::test_acl_vector_state_persists_runtime_quarantine_across_resume`, `::test_acl_vector_state_rejects_a_mismatched_version` |
| `REQ-RSA-006` | `curriculum/scenario_acl/driver.py::vector_episode_end_callback`; `curriculum/scenario_acl/buffer.py::remove_scenario_id` | Existing ACL vectorized execution fixtures (no new regression added this pass) |
| `REQ-RSA-007` | `agent/agent.py::Agent._evaluate_parallel` (`invalid_records`, `data_abort_coverage`) | `test_parallel_evaluation_data_abort.py::test_parallel_evaluation_excludes_data_abort_episode_from_aggregates` |
| `REQ-RSA-008` | `runtime/data_abort.py::append_data_abort_record`; call site in `agent/agent.py::train_vectorized` | `test_runtime_data_abort.py::test_data_abort_jsonl_keeps_observation_out_of_text_record`; `test_train_vectorized_data_abort.py::test_train_vectorized_writes_forensic_jsonl_and_quarantines_on_data_abort` |

## 9. Findings and deviations

No deviation identified for the worker/parent transport protocol itself. The existing SB3 collector has no failed-vector-slot protocol, so a bounded fork change is necessary; this is the approved minimum integration point, not a synthetic-transition workaround.

### 2026-07-23 implementation finding (superseded twice)

An earlier revision of this ExecPlan stated that "typed data-abort remains
explicitly fatal for PPO rather than producing synthetic rollout data,"
reasoning that the standard SB3 `RolloutBuffer` flattens every `(time, env)`
cell into minibatches and offers no validity mask. That statement was already
stale once (PPO recovery had been implemented via a full-rollout discard,
`abort_rollout_for_runtime_data_abort`), and is now stale a second time: the
full-discard approach has itself been replaced by a prefix-preserving
implementation (see below), so it no longer describes the code at all.

### 2026-07-23 resolution: PPO now preserves the valid prefix at the boundary

The full-rollout-discard behavior was reassessed after review discussion
questioned why PPO could not simply truncate the aborted env's episode at the
previous step, the same way TD3/SAC/PER already do. The original finding's
premise — that avoiding a rewrite of `compute_returns_and_advantage` requires
discarding the whole rollout — turned out to be avoidable: SB3's shared,
lockstep write cursor can stay untouched (every `add()` call still advances
`self.pos` by exactly one for every env, so the outer `agent.py` training
loop's iteration/collection-length assumptions needed no change at all).
Only two things were needed: (1) a per-cell validity mask on the rollout
buffer (`MaskedRolloutBuffer`, mirroring `PrioritizedNStepReplayBuffer.valid_transitions`),
where an invalid cell writes only its `episode_starts` flag — which is what
makes the *previous* row's GAE computation correctly stop propagating
advantage across the boundary — while every other field for that cell is
never populated and never read; and (2) `close_previous_transition_as_data_abort`,
which bootstraps the preceding valid row's reward with `gamma * V(final_observation)`,
reusing the *existing* `TimeLimit`-truncation reward-bootstrap convention
already present in this backend rather than inventing a new one. No change to
`compute_returns_and_advantage` itself was needed. This let the
PPO-specific branch in `agent.py` (`is_ppo_lifecycle`) be deleted entirely:
PPO now goes through the exact same abort-handling path as TD3/SAC.
Validated at the buffer level (`test_ppo_masked_rollout_buffer.py`) and at
the backend level with a real `Sb3PpoPlannerBackend` and a 2-env
`DummyVecEnv` (`test_ppo_sb3_data_abort.py`), confirming the peer env's row
is untouched and the rollout still completes and trains afterward.

### 2026-07-23 finding: uniform (non-PER) replay stays deliberately fatal

Uniform TD3/SAC replay lacks the non-addressable-leaf/retroactive-boundary
machinery `PrioritizedNStepReplayBuffer` has, so a typed data-abort is fatal
whenever `transition_replay.prioritized=false`. This was an explicit, approved
design choice (implementing safe pending N-step/ring-buffer mutation for the
uniform buffer was judged a larger, riskier rewrite than this feature's scope)
rather than a bug. No run profile or preset currently sets
`transition_replay.prioritized=false` for TD3/SAC, so this limitation is not
reachable by any registered configuration today.
