# ExecPlan: Resumable Training After Abrupt Interruption v1

## 1. Metadata

- Feature: crash-safe training snapshots and a validated resume path after an
  abrupt process termination.
- Plan ID: `RESUME-ABRUPT-001`
- Status: `IMPLEMENTED` (M1–M4 complete; kill-and-resume smoke `PASS`, full
  suite `1704 passed` — evidence in §11; `VERIFIED` withheld pending the two
  §12 limitations)
- Created: 2026-09-06. Last update: 2026-09-06.
- Tracking issue:
  [#3](https://github.com/EmanueleRsp/thesis-metadrive/issues/3).
- Authoritative specifications:
  - `docs/specifications/transition_replay_v1_specification.md`
    (`TRANSITION-REPLAY` v1.0) as amended by
    `docs/specifications/transition_replay_v1.1_amendment.md`
    (`TRANSITION-REPLAY-V1.1`, `APPROVED` 2026-09-06): REQ-023, REQ-024,
    REQ-024a, REQ-025, REQ-025a, REQ-033, §8.9, §9.1.
  - `docs/specifications/rl_baselines_v1_specification.md` (checkpoint policy,
    unchanged).
- Related ADRs: ADR-079 (this plan's decisions; supersedes ADR-017), ADR-016
  (ACL resume restart policy, unchanged), ADR-024 (quarantine as resume state,
  unchanged).
- Related open items: `C13` (this defect; numbered `C12` while the branch was
  open, renumbered after the merge because `C12` had meanwhile been taken on
  `main` by the intra-chunk progress and fail-fast defect, whose identifier is
  referenced from source code), `V2` (parked 2026-09-01, now
  resolved by `DEC-RES-003`), `D4`, `F5`.
- Branch: `worktree-resume-after-abrupt-interruption`.

## 2. Objective And Scope

**Observable capability.** After a training process is killed at an arbitrary
instant (SIGKILL, OOM-killer, node crash, SIGTERM from `docker stop` or a
scheduler), relaunching with `checkpoint.resume.enabled=true` resumes from the
most recent *complete and internally consistent* snapshot, losing at most one
snapshot interval of work, and never resumes from a torn or mismatched set of
files. When the replay buffer cannot be continued, the run says so explicitly
(REQ-025 `replay_reset=true`) instead of continuing silently.

**Why.** Every production profile lost the whole run on a crash: the periodic
`latest` checkpoint carried no replay buffer, was written in place, and no
signal other than SIGINT reached the save handler. `open_items` `D4` records
two production runs lost this way; `V2` records the diagnosis and the
2026-09-01 decision to park mid-run replay continuation, reopened here with
the cost quantified (§4.6).

**In scope.** Both training loops (`runtime/loops/train_loop.py` and
`curriculum/scenario_acl/driver.py`), the TD3/SAC/PPO SB3 backends' save path,
Hydra keys under `checkpoint.*` and `transition_replay.persistence.*`, the run
profiles, `compose.yaml`, the spec amendment and ADR-079.

**Out of scope.** Bitwise-identical continuation (REQ-025 disclaims it);
stitching two `run_dir`s into one logical run in the analysis layer;
persisting the in-flight asynchronous-evaluation queue; the geometry-abort
ledger (`geometry_abort_quarantine_v1_exec_plan.md` owns it); the non-ACL
scenario-provider draw position (`DEC-RES-005`, deferred).

**Compatibility.** Checkpoint file names and directory layout are preserved;
the periodic snapshot adds companion files next to `periodic/step_X.zip`. Old
runs remain loadable for evaluation and resumable: a snapshot without
`model_num_timesteps` is accepted with a warning. `persistence.trigger=final_or_manual`
(v1.0) is still accepted.

## 3. Authoritative Requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-RES-001` | Every artifact a resume reads is either the previous complete version or the new complete version; never a partial one. | v1.1 REQ-024 (atomic publication, state file last). |
| `REQ-RES-002` | A model-only resume of an off-policy learner fails closed unless explicitly allowed; when allowed it starts a new empty replay segment, logs `replay_reset=true`, resets `beta_progress_env_steps`. | v1.1 REQ-025. |
| `REQ-RES-003` | Model, replay artifact, pair manifest and training/curriculum state come from one save: `model_num_timesteps` must match the loaded model; `seed` must match. | v1.1 REQ-033 (extended). |
| `REQ-RES-004` | The step counter recorded in a snapshot equals the number of environment steps the saved model was trained on. | v1.1 §8.9 (snapshots only at chunk boundaries, `DEC-RES-007`). |
| `REQ-RES-005` | SIGTERM produces the same graceful end as SIGINT. | `DEC-RES-002`. |
| `REQ-RES-006` | The replay buffer is paired with every periodic model checkpoint (crossing semantics, `keep_last=1`), and `periodic` resolves to the newest complete pair. | v1.1 REQ-024, REQ-025a. |
| `REQ-RES-007` | Best-checkpoint comparison keys survive a resume. | `checkpoint.save_best_*` semantics. |
| `REQ-RES-008` | Intermediate replay artifacts are removed once `final` carries the pair. | v1.1 REQ-024a (`DEC-RES-006`). |

## 4. Current Repository Analysis (as found on 2026-09-06, before the change)

All statements `VERIFIED` by reading the code unless labelled otherwise.

### 4.1 Save path

- `save_intermediate_checkpoints` (`train_loop.py`) ran at every chunk
  boundary **and** from the asynchronous-evaluation completion callback, which
  `AsyncEvaluationManager.poll()` invokes from `drain_event_messages()` inside
  a training chunk: on the non-curriculum path `latest` was therefore written
  **mid-chunk** with the live model and the evaluated job's (stale)
  `global_step`. Under `save_latest_each_chunk` it wrote `latest.zip`,
  `latest_training_state.yaml`, `latest_rng_state.pkl`,
  `latest_quarantine_state.json`; no replay buffer, no pair manifest.
- The ACL loops wrote the same set plus, when persistence was enabled,
  `latest_replay_buffer.pkl` and `latest_checkpoint_pair.json` every chunk;
  they wrote **no periodic checkpoints**.
- `Agent.save` → `planner.save` → plain `self.model.save(path)`: in-place,
  not atomic. `_save_training_state` (`OmegaConf.save`) and `_save_rng_state`
  (`pickle.dump`) were not atomic; ACL `scenario_acl_state.json`,
  `scenario_buffer.json`, `scenario_coverage_state.json` used bare
  `write_text`. `_save_json_atomically`, `_save_replay_buffer_atomically` and
  `save_acl_vector_state` were atomic.
- The periodic checkpoint fired only when `current_global_step % interval == 0`
  exactly; with `eval_interval=50000` and `periodic_interval_steps=125000`
  (`thesis`, `long`, `tune`) that skipped every other periodic checkpoint.
- `sb3_extensions/checkpointing.py` (atomic generations + pointer) was unused
  by training (`DEC-EP-002`); not adopted here (`DEC-RES-001`).

### 4.2 Interrupt handlers

- Only `except KeyboardInterrupt` existed; no `signal` usage in `src/`.
- The standard handler saved `latest` with `current_global_step` up to one
  chunk stale, dropped `beta_progress_env_steps`; the ACL handler saved the
  RNG only under `persistence_enabled` and re-wrote none of the ACL state
  files; the vectorized ACL path had no handler at all (dispatch outside the
  `try`).

### 4.3 Resume path

- Replay validation ran only under `persistence_enabled`; otherwise the replay
  block was skipped and training silently continued on an empty buffer
  (`grep -rn replay_reset src/` was empty).
- `_validate_checkpoint_pair` failed closed on a missing pair; best keys were
  never restored; `seed` was written but never compared.

### 4.4 Configuration

`checkpoint.resume.{enabled,run_dir,checkpoint_name,restore_rng_state}`;
`run_profile.replay_persistence` `true` only in `smoke` (ADR-017);
`replay/config.py` rejected any trigger but `final_or_manual`.

### 4.5 Tests

Unit coverage for the pair validator, sidecars, quarantine and ACL vector
state; nothing for the RNG/state writers, the interrupt handlers, best-key
restoration or a kill-and-resume flow.

### 4.6 Cost of persisting the replay buffer (`VERIFIED` from config and code)

| Quantity | Value |
|---|---|
| Bytes per stored transition (`semantic_v3`, dim 3011, float32 obs + next obs, float64 priority, bool mask) | ≈ 24.1 kB |
| Full-buffer artifact (`buffer_size = 300 000`) | ≈ 7.2 GB |
| Peak disk during atomic replace (`keep_last=1`) | ≈ 14.4 GB |
| Writes per 1.5 M-step `thesis` run at `periodic_interval_steps` (125k) cadence | 12 |
| Smoke measurement (§11): 1 000 transitions of `lidar_state` obs | 1.32 MB, < 1 s |

## 5. Assumptions And Invariants

- Snapshot commit order: model zip and sidecars, replay artifact (if due),
  pair manifest, RNG, quarantine, (ACL: buffer, coverage, vector state), and
  **last** the training/curriculum state file as commit marker. A resume
  whose state `model_num_timesteps` differs from the loaded model's counter
  fails before training.
- `os.replace` on the same filesystem is atomic; temporaries are siblings
  named `.<name>.<pid>.tmp` (suffix kept so SB3 does not append `.zip`).
- The SIGTERM handler is installed in the main thread of the training process
  only; worker subprocesses keep the default disposition.
- Snapshots are written only at chunk boundaries; mid-chunk the model is
  ahead of every chunk-level counter (`DEC-RES-007`).
- Steps are environment steps summed over vectorized workers, as elsewhere.

## 6. Decisions And Approval Gates

| ID | Category | Decision | Status |
|---|---|---|---|
| `DEC-RES-001` | Implementation detail | tmp + `os.replace` on the existing file names instead of adopting `checkpointing.py` generations. | Recorded. |
| `DEC-RES-002` | New convention | SIGTERM raises `KeyboardInterrupt`; `compose.yaml` `stop_grace_period: 120s`. | **Approved 2026-09-06** (ADR-079). |
| `DEC-RES-003` | Specification amendment | `TRANSITION-REPLAY` v1.1 `periodic_and_final`: replay paired with every periodic checkpoint, cadence `checkpoint.periodic_interval_steps` (crossing semantics), `keep_last=1`; `replay_persistence=true` on every profile; supersedes ADR-017. Cost §4.6. | **Approved 2026-09-06** (ADR-079). |
| `DEC-RES-004` | Specification clarification | `checkpoint.resume.allow_replay_reset` (default `false`); REQ-025 event `replay_reset=true` in `events.jsonl` and `run_metadata.yaml`. | **Approved 2026-09-06** (ADR-079). |
| `DEC-RES-005` | Scope | Non-ACL scenario-provider draw position not persisted (deferred). | **Approved 2026-09-06** (ADR-079). |
| `DEC-RES-006` | Data policy | After the `final` pair is committed, intermediate replay artifacts are deleted. | **Approved 2026-09-06** (user addition, ADR-079). |
| `DEC-RES-007` | Implementation decision (observable, recorded for review) | Interrupt handlers no longer write a mid-chunk snapshot; the chunk-boundary snapshot is the resumable one. Replaces the plan's earlier idea of recording `planner.num_timesteps` in the handler, which would have fixed the step counter but not the chunk-level curriculum/ACL state; the async-callback finding (§4.1) showed that any mid-chunk write is unsafe. Ctrl+C/SIGTERM lose at most one chunk, like a kill. | Implemented; reversible if the user prefers mid-chunk Ctrl+C snapshots. |
| `DEC-RES-008` | Implementation detail | The periodic replay snapshot rides the existing `periodic/step_X` checkpoint (`checkpoint_name=periodic` alias) instead of a new artifact family; no `periodic_frequency_steps` knob. Periodic "due" uses crossing semantics, which also fixes the pre-existing skip in §4.1. | Recorded. |

## 7. Design As Implemented

- `src/thesis_rl/runtime/io/atomic.py` (new): `atomic_publish(path, writer)`,
  `atomic_write_text/bytes`, `atomic_pickle_dump`, `atomic_omegaconf_save`,
  `temporary_sibling`.
- `src/thesis_rl/runtime/io/resume_snapshot.py` (new): snapshot layout
  (`resume_artifact_paths`), `periodic` alias resolution, crossing test
  (`periodic_snapshot_due`), `planner_trained_timesteps`, torn-snapshot and
  seed checks, REQ-025 classification (`classify_replay_resume`),
  `write_checkpoint_pair` (adds `model_num_timesteps`, `beta_progress_env_steps`),
  pruning (`prune_replay_snapshots`, `prune_old_periodic_checkpoints`,
  `prune_periodic_companions`) and `remove_replay_snapshots_after_final`.
- `src/thesis_rl/runtime/signals.py` (new): `install_sigterm_as_keyboard_interrupt`.
- Backends `sac_sb3.py`, `td3_sb3.py`, `ppo_sb3.py`: `save()` through
  `atomic_publish`.
- `replay/config.py`: `persistence_trigger`, `periodic_replay_persistence`,
  triggers `{final_or_manual, periodic_and_final}`.
- `train_loop.py`: SIGTERM install; startup validation (periodic trigger needs
  `save_periodic` and a positive interval); resume setup through
  `resume_artifact_paths` with the `periodic` alias; resume block with
  torn-snapshot/seed checks, REQ-025 classification, `replay_reset` event,
  best-key restoration; `save_intermediate_checkpoints(write_resume_snapshot=...)`
  keeps only the best checkpoints when called from the async callback;
  `write_resume_snapshot_files` writes `latest` and, when due, the paired
  periodic snapshot with its own training/RNG/quarantine state, prunes and
  logs `replay_snapshot_written`; the non-curriculum chunk end calls it with
  live counters; `final` pair carries `model_num_timesteps`, then
  `remove_replay_snapshots_after_final`; interrupt handler no longer saves.
- `driver.py` (ACL): atomic state files; `_write_acl_resume_snapshot` shared by
  both loops (latest, RNG independent of persistence, live curriculum state
  with state file last, paired periodic snapshot with frozen curriculum copy
  under `periodic/step_X_acl/`, pruning); resume root chosen by checkpoint
  name (`_acl_state_dir_for_checkpoint`); consistency checks, REQ-025
  classification, RNG restore independent of persistence; finals carry
  `model_num_timesteps` and clean up; interrupt handlers no longer save; the
  vectorized loop gained a `KeyboardInterrupt` handler.
- Configuration: `checkpoint.resume.allow_replay_reset: false`;
  `trigger: periodic_and_final` in `td3_sb3.yaml`/`sac_sb3.yaml`;
  `replay_persistence: true` in `default`, `fast`, `medium`, `long`, `thesis`,
  `tune`; `compose.yaml` `stop_grace_period: 120s`.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-RES-001` | `AC-RES-001`: after a failed/killed write the previous snapshot is intact, no temporary left | `runtime/io/atomic.py`; backends `save`; `train_loop.py` `_save_training_state`/`_save_rng_state`; `driver.py` state writers | `tests/test_resume_snapshot.py::test_atomic_publish_*` (`TEST-RES-001`, `002`), `test_acl_state_files_are_written_with_state_last`; `TEST-RES-010` smoke | Done |
| `REQ-RES-002` | `AC-RES-002`: model-only resume fails closed without the flag; with it emits `replay_reset=true`, β=0 | `classify_replay_resume`; resume blocks in both loops | `test_model_only_resume_fails_closed_without_flag` (`TEST-RES-007`), `test_model_only_resume_with_flag_is_classified_as_reset` (`TEST-RES-008`), `test_partially_committed_pair_is_rejected_even_with_flag` | Done (event emission verified by code reading; no loop-level harness) |
| `REQ-RES-003` | `AC-RES-003`: `model_num_timesteps` mismatch or seed mismatch fails before training | `assert_snapshot_model_consistent`, `assert_snapshot_seed_consistent`; resume blocks | `test_model_consistency_check_*` (`TEST-RES-005`), `test_seed_consistency_check`, `test_checkpoint_pair_carries_identity_fields` | Done |
| `REQ-RES-004` | `AC-RES-004`: snapshot counters equal the model's trained steps | chunk-boundary-only snapshots (`DEC-RES-007`), `write_resume_snapshot_files` with live counters | `TEST-RES-010` smoke: `latest_training_state.yaml` `global_steps_done == model_num_timesteps == 1000` | Done |
| `REQ-RES-005` | `AC-RES-005`: SIGTERM → `KeyboardInterrupt` path | `runtime/signals.py`, `run_training`, `compose.yaml` | `test_sigterm_is_raised_as_keyboard_interrupt` (`TEST-RES-009`, unit form) | Done (process-level SIGTERM not exercised end to end; see §12) |
| `REQ-RES-006` | `AC-RES-006`: pair exactly at crossings, `periodic` alias picks newest complete pair, previous pair pruned | `periodic_snapshot_due`, `write_resume_snapshot_files`, `_write_acl_resume_snapshot`, `resolve_resume_checkpoint_name`, `prune_replay_snapshots` | `test_periodic_snapshot_due_uses_crossing_semantics` (`TEST-RES-011`), `test_periodic_alias_*`, `test_prune_replay_snapshots_keeps_only_the_newest_pair`, `test_acl_resume_snapshot_writes_latest_and_paired_periodic_when_due`, `test_transition_replay_config.py::test_periodic_and_final_trigger_is_accepted` (`TEST-RES-006`); `TEST-RES-010` smoke | Done |
| `REQ-RES-007` | `AC-RES-007`: best keys restored on resume | `_best_keys_payload`, `_restore_best_key`, resume block | `test_best_keys_round_trip_through_training_state` (`TEST-RES-013`) | Done (restoration wiring verified by code reading) |
| `REQ-RES-008` | `AC-RES-008`: intermediate replay copies removed after `final` | `remove_replay_snapshots_after_final` in both loops | `test_remove_replay_snapshots_after_final_drops_intermediate_copies`; `TEST-RES-010` smoke event `replay_snapshots_removed_after_final` | Done |

Regression added for a defect found by `TEST-RES-010`:
`tests/test_transition_replay_persistence.py::test_checkpoint_pair_validation_accepts_periodic_checkpoint_name`
(the validator compared `periodic/step_X.zip` with the recorded basename).

## 9. Test Strategy

Mandatory matrix as in §8; commands (all existing):

```bash
docker compose -p thesis-metadrive run --rm -T dev uv run --no-sync python -m pytest -q \
  tests/test_resume_snapshot.py tests/test_transition_replay_config.py \
  tests/test_transition_replay_persistence.py tests/test_scenario_acl_buffer.py \
  tests/test_hydra_preset_run_configs.py tests/test_run_metadata.py
docker compose -p thesis-metadrive run --rm -T dev uv run --no-sync python -m pytest -q tests
make lint
make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/runtime/io/atomic.py src/thesis_rl/runtime/io/resume_snapshot.py src/thesis_rl/runtime/signals.py src/thesis_rl/sb3_extensions/replay/config.py tests/test_resume_snapshot.py"
make config
```

Tests changed under ADR-079 (contract change, approved):
`test_replay_persistence_is_enabled_only_for_smoke_by_default` →
`test_replay_persistence_is_enabled_on_every_profile_by_default`;
`test_periodic_replay_persistence_is_rejected` →
`test_periodic_frequency_steps_is_not_a_setting`; the `final_or_manual`
assertion in `test_final_scalar_pipeline_defaults_compose` → `periodic_and_final`.

## 10. Milestones

- [x] **M1 — crash-safe snapshot and consistent counters.** Atomic writers,
  chunk-boundary-only snapshots, torn-snapshot/seed checks, best keys,
  `model_num_timesteps` in pair and state.
- [x] **M2 — SIGTERM.** `runtime/signals.py`, `run_training`, `compose.yaml`.
- [x] **M3 — periodic paired replay snapshot.** Spec amendment v1.1, ADR-079,
  `replay/config.py`, both loops, profiles, `allow_replay_reset`, cleanup after
  `final`.
- [x] **M4 — reconciliation.** `open_items` `C13`/`V2`, `project_index.md`,
  `validation_commands.md` §8, ADR-017 marked superseded.

## 11. Progress And Findings Log

- **2026-09-06 (analysis)** — Findings recorded in issue #3 and §4; plan
  written with gates `DEC-RES-002..005`.
- **2026-09-06 (approval)** — User approved all gates and added `DEC-RES-006`;
  instructed full implementation.
- **2026-09-06 (finding, during implementation)** — On the non-curriculum
  path the asynchronous-evaluation completion callback wrote `latest` mid-chunk
  with the evaluated job's step (§4.1). Resolved by `DEC-RES-007`/`DEC-RES-008`:
  the resume snapshot is written only at chunk boundaries with live counters;
  the callback keeps only the best-checkpoint logic.
- **2026-09-06 (finding)** — The ACL loops wrote no periodic checkpoints and
  the vectorized ACL path had no `KeyboardInterrupt` handler; both added.
- **2026-09-06 (finding)** — Periodic checkpoints were skipped when
  `periodic_interval_steps` was not a multiple of `eval_interval` (`thesis`,
  `long`, `tune`: 125k vs 50k). Fixed by crossing semantics.
- **2026-09-06 (tests)** — Focused suite: `74 passed` (`test_resume_snapshot.py`
  27 tests, `test_transition_replay_config.py`, `test_transition_replay_persistence.py`,
  `test_scenario_acl_buffer.py`, `test_hydra_preset_run_configs.py`,
  `test_run_metadata.py`); ruff lint clean on all touched files; ruff format
  clean on the new modules and the new test file (the pre-existing formatting
  debt in `train_loop.py`/`driver.py` was not touched, per `AGENTS.md`).
- **2026-09-06 (`TEST-RES-010`, kill -9 and resume)** — Smoke preset
  (`presets/test/smoke_train`, TD3, MetaDrive `lidar_state`, non-curriculum →
  asynchronous validation) with `total_timesteps=3000 eval_interval=500
  periodic_interval_steps=1000`, fixed `paths.run_dir`
  `outputs/RESUME-ABRUPT-SMOKE/td3_sb3/seed_0/run_a`. At step 1000 the periodic
  snapshot was written (`step_00001000.zip` 5.2 MB, `_replay_buffer.pkl`
  1.32 MB, pair, RNG, training state with `global_steps_done == model_num_timesteps == 1000`,
  `best_keys` populated). The container was killed with `docker kill -s KILL`
  during chunk 3 (`run_metadata.yaml` still `status: running`). First relaunch
  with `checkpoint.resume.checkpoint_name=periodic` **failed** before training:
  `Checkpoint pair model identity mismatch: manifest='step_00001000.zip',
  expected='periodic/step_00001000.zip'` — a defect in `_validate_checkpoint_pair`
  (fixed, regression test added, `3 passed`). Second relaunch: `run_resumed`
  from `periodic/step_00001000.zip`, chunks 3 and 4 completed, the paired
  snapshot at step 2000 was written and pruned the step-1000 pair
  (`removed_previous`), and the run then died on the **unrelated** issue #2 /
  `C7` defect (`Signal transition requires valid, known pre/post states`) during
  its evaluation, which is not part of this plan. Third relaunch resumed from
  `periodic/step_00002000.zip`, ran chunks 5 and 6 to the full
  `total_timesteps=3000`, wrote the paired snapshot at 3000 and then `final`,
  and went on to finish the three final evaluation panels
  (`test_waymo_empirical`, `test_pg`, `test_arm_stratified` at step 3000);
  `run_metadata.yaml` ends at `status: completed`, `finished_at 22:20:17`,
  `duration_seconds 848.02`. So the resumed run terminated normally, evaluation
  included, not merely up to the last training chunk.
- **2026-09-06 (`TEST-RES-010` result, PASS)** — Final state of the run
  directory after one SIGKILL and two resumes:

  | evidence | observed |
  |---|---|
  | `checkpoint_index.csv` `global_step` | 500, 1000, 1000 (periodic), 1500, 2000, 2000 (periodic), 2500, 3000, 3000 (periodic), 3000 (final) — strictly monotone across both resume points |
  | `latest_training_state.yaml` | `global_steps_done = model_num_timesteps = beta_progress_env_steps = 3000`, `chunk_id = 6`, `seed = 42` |
  | `final` artifacts | `final.zip`, `final_replay_buffer.pkl`, `final_checkpoint_pair.json` present |
  | `DEC-RES-006` cleanup | `replay_snapshots_removed_after_final` event; `periodic/` retains only the three model zips with their manifest, RNG and training-state companions — no `_replay_buffer.pkl`, no `_checkpoint_pair.json` |
  | replay snapshot sizes | 1.32 MB (1 000 transitions), 2.64 MB (2 000), 3.95 MB (3 000) — linear in stored transitions, as `__getstate__` truncation predicts |
  | run outcome | `status: completed`; final panels `test_waymo_empirical`, `test_pg`, `test_arm_stratified` recorded in `evals.csv` at step 3000 |

  `AC-RES-001`, `AC-RES-003`, `AC-RES-004`, `AC-RES-006` and `AC-RES-008`
  observed end to end.
- **2026-09-06 (full suite)** — `1704 passed in 1680.40s` (`python -m pytest -q
  tests` in the compose `dev` service), zero failures.

## 12. Known Limitations

- Process-level SIGTERM (`docker stop`) is covered by the unit test of the
  handler and by code reading, not by an end-to-end run.
- The REQ-025 `replay_reset` event emission and the best-key restoration are
  wired in the loops and verified by code reading; no loop-level harness exists
  for them (the loops need a live environment).
- The ACL companion files (buffer, coverage, vector state) are each atomic but
  not jointly atomic with `scenario_acl_state.json`; the write window is
  sub-second and they are now written contiguously.
- `checkpoint_index.csv` is append-only and not deduplicated: a resumed run
  appends rows after the crash point; rows of the lost chunk are absent, not
  duplicated.
- Replay write duration at full buffer (≈ 7.2 GB) is not measured; the smoke
  measured 1.32 MB in under a second.
