# ExecPlan: Resumable Training After Abrupt Interruption v1

## 1. Metadata

- Feature: crash-safe training snapshots and a validated resume path after an
  abrupt process termination.
- Plan ID: `RESUME-ABRUPT-001`
- Status: `AWAITING_DECISIONS` (proposal; no production code changed yet)
- Created: 2026-09-06. Last update: 2026-09-06.
- Tracking issue:
  [#3](https://github.com/EmanueleRsp/thesis-metadrive/issues/3).
- Authoritative specifications touched:
  - `docs/specifications/transition_replay_v1_specification.md`
    (`TRANSITION-REPLAY` v1.0, `AUTHORITATIVE`): REQ-023, REQ-024, REQ-025,
    REQ-033, §8.9, §9.1. **M3 requires an amendment (v1.1)**; M1 and M2 do not
    change it.
  - `docs/specifications/rl_baselines_v1_specification.md` (checkpoint policy,
    unchanged).
- Related ADRs: ADR-016 (ACL resume restart policy), ADR-017 (smoke-only replay
  persistence, **to be revised by M3**), ADR-024 (quarantine as resume state).
- Related open items: `V2` (parked 2026-09-01), `D4`, `F5`; new `C12`.
- Branch: `worktree-resume-after-abrupt-interruption`.

## 2. Objective And Scope

**Observable capability.** After a training process is killed at an arbitrary
instant (SIGKILL, OOM-killer, node crash, SIGTERM from `docker stop` or a
scheduler), relaunching with `checkpoint.resume.enabled=true` resumes from the
most recent *complete and internally consistent* snapshot, losing at most one
snapshot interval of work, and never resumes from a torn or mismatched set of
files. When the replay buffer cannot be continued, the run says so explicitly
(REQ-025 `replay_reset=true`) instead of continuing silently.

**Why.** Every production profile today loses the whole run on a crash: the
periodic `latest` checkpoint carries no replay buffer, is written in place, and
no signal other than SIGINT reaches the save handler. `open_items` `D4`
records two production runs lost this way; `V2` records the diagnosis and the
2026-09-01 decision to park mid-run replay continuation. This plan re-opens
that decision with the cost quantified (§4.6) and splits the work so that the
zero-cost correctness fixes (M1) do not wait for the storage decision (M3).

**Success.** `TEST-RES-010` (kill -9 at a random point of a smoke run, resume,
finish with the same total budget and a consistent `checkpoint_index.csv`)
passes, and the M1 unit matrix passes.

**In scope.** Both training loops (`runtime/loops/train_loop.py` and
`curriculum/scenario_acl/driver.py`), TD3/SAC/PPO SB3 backends' save path,
Hydra keys under `checkpoint.*` and `transition_replay.persistence.*`, the
`smoke` profile's resume coverage.

**Out of scope.** Bitwise-identical continuation (REQ-025 already disclaims
it); stitching two `run_dir`s into one logical run in the analysis layer;
persisting the in-flight asynchronous-evaluation queue; the geometry-abort
ledger (`geometry_abort_quarantine_v1_exec_plan.md` owns it); the non-ACL
scenario-provider draw position (recorded as `DEC-RES-005`, deferred).

**Compatibility.** Checkpoint file names and directory layout are preserved.
Old runs remain loadable for evaluation. A snapshot written by the new code is
resumable by the new code only (it adds a consistency check that old snapshots
cannot satisfy; the check is skipped with a warning when the new fields are
absent).

## 3. Authoritative Requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-RES-001` | Every artifact a resume reads is either the previous complete version or the new complete version; never a partial one. | Derived from `TRANSITION-REPLAY` REQ-024 ("replay replacement shall be atomic") and REQ-033 ("partially committed pairs fail before resume"), extended to the model, training state and RNG files. |
| `REQ-RES-002` | A model-only resume starts a new empty replay segment, logs `replay_reset=true`, resets `beta_progress_env_steps` to 0, and is never reported as replay-equivalent continuation. | `TRANSITION-REPLAY` REQ-025 (already authoritative, currently not implemented). |
| `REQ-RES-003` | Model, replay artifact and pair manifest share `checkpoint_id`, `training_timestep`, `replay_segment_id`; mismatch fails before resume. | `TRANSITION-REPLAY` REQ-033 (implemented; extended to also compare the model's own `num_timesteps`). |
| `REQ-RES-004` | The step counter recorded in a snapshot equals the number of environment steps the saved model was trained on. | Derived from RL Baselines v1 checkpoint policy and REQ-025 (β progress is defined on that counter). |
| `REQ-RES-005` | SIGTERM produces the same graceful snapshot as SIGINT. | New convention, `DEC-RES-002`. |
| `REQ-RES-006` | A periodic replay snapshot paired with `latest` may be written at a configured cadence; `keep_last=1`. | **Amendment** to REQ-023/REQ-024/§8.9 (`TRANSITION-REPLAY` v1.1), `DEC-RES-003`. |
| `REQ-RES-007` | Best-checkpoint comparison keys survive a resume. | Derived from `checkpoint.save_best_*` semantics (a resumed run must not demote the true best). |

## 4. Current Repository Analysis

All statements below are `VERIFIED` by reading the code on 2026-09-06 unless
labelled otherwise.

### 4.1 Save path

- `save_intermediate_checkpoints` (`train_loop.py:1251-1496`) runs at every
  chunk boundary (chunk = `experiment.eval_interval`, 10k–50k steps by profile)
  and, under `checkpoint.save_latest_each_chunk`, writes `latest.zip`
  (`agent.save`, `:1410`), `latest_training_state.yaml` (`:1435`),
  `latest_rng_state.pkl` (`:1437`), `latest_quarantine_state.json` (`:1438`),
  and appends to `checkpoint_index.csv`. **No replay buffer and no pair
  manifest** are written here.
- The ACL loop writes the same set plus, when
  `transition_replay.persistence_enabled`, `latest_replay_buffer.pkl` and
  `latest_checkpoint_pair.json` every chunk (`driver.py:1254-1266`).
- Replay + pair on the standard loop are written only at `final`
  (`train_loop.py:2451-2466`) and in the `KeyboardInterrupt` handler
  (`:3031-3043`).
- `Agent.save` (`agent/agent.py:2929-2941`) calls `planner.save`, which is a
  plain `self.model.save(path)` (`sac_sb3.py:613-616`, `td3_sb3.py:663-666`,
  `ppo_sb3.py:587-590`): **in-place write, not atomic**. The two sidecars
  (`.reward_semantics.json`, `.manifest.json`) are atomic.
- `_save_training_state` (`train_loop.py:232-234`) and `_save_rng_state`
  (`:342-353`) are **not atomic**. `_save_json_atomically` (`:237-249`) and
  `_save_replay_buffer_atomically` (`:307-318`) are.
- ACL state files `scenario_acl_state.json` (`driver.py:1240-1253`),
  `scenario_buffer.json`, `scenario_coverage_state.json` (`:452-476`) use bare
  `write_text`. `scenario_acl_vector_state.json` is atomic
  (`vectorized.py:361-374`).
- `sb3_extensions/checkpointing.py` provides an atomic generation + pointer
  mechanism, unused by training (`DEC-EP-002`,
  `evaluation_protocol_v1.0_exec_plan.md:312`). This plan does **not** adopt it
  (`DEC-RES-001`): the file layout is a public contract used by the evaluation
  protocol, videos and analysis, and tmp+rename on the existing names is
  sufficient.

### 4.2 Interrupt handlers

- Only `except KeyboardInterrupt` exists (`train_loop.py:3008`,
  `driver.py:2613`). No `signal` module usage anywhere in `src/`.
- `current_global_step` is updated only at chunk end (`train_loop.py:1596`),
  so the handler records a counter up to one chunk stale (`:3058`) for a model
  that has already been trained past it. The planner's own counter is exact:
  `self.model.num_timesteps += collected` per collected batch
  (`sac_sb3.py:435-436`, `td3_sb3.py:447-448`, `ppo_sb3.py:406`).
- The handler's training-state payload omits `beta_progress_env_steps`
  (`:3055-3068` vs `:1433`).
- The ACL handler nests the RNG save under `persistence_enabled`
  (`driver.py:2614-2623`) and re-writes none of the ACL state files.

### 4.3 Resume path

- `train_loop.py:820-935`, `1159-1182`: loads training state, planner
  (`load_planner`, reward-semantics and manifest checks), adapter, then **only
  if `persistence_enabled`** validates the pair and loads the replay buffer;
  restores RNG, quarantine and counters. When persistence is disabled the
  replay block is skipped and training continues on an empty buffer with no
  log line (`grep -rn replay_reset src/` is empty).
- `_validate_checkpoint_pair` (`:252-293`) fails closed on a missing pair.
- Best keys are initialised to `None` (`:721-726`) and never restored.
- `seed` is written (`:1431`) but not compared with `cfg.seed`.
- ACL resume (`driver.py:478-547`, `590-730`) restores buffer, bandit,
  coverage, vector state and RNG; missing coverage file is fatal (`:517-525`).

### 4.4 Configuration

- `conf/config.yaml:96-110`: `checkpoint.save_latest_each_chunk`,
  `save_periodic`, `periodic_interval_steps`, `keep_last_periodic`,
  `save_rng_state`, `resume.{enabled,run_dir,checkpoint_name,restore_rng_state}`.
- `run_profile.replay_persistence` is `true` only in `smoke`;
  `periodic_interval_steps` is 125k (`thesis`, `long`, `tune`), 100k
  (`medium`), 250k (`default`), 50k (`fast`), 10k (`smoke`).
- `sb3_extensions/replay/config.py:72-78` rejects any
  `persistence.trigger != final_or_manual` and any
  `periodic_frequency_steps`.

### 4.5 Tests

Unit coverage exists for the pair validator, sidecars, quarantine and ACL
vector state round-trips (`tests/test_transition_replay_persistence.py`,
`test_checkpoint_manifest_sidecar.py`, `test_runtime_quarantine_checkpoint.py`,
`test_scenario_acl_vectorized_state.py`). **Nothing** covers
`_save_rng_state`/`_load_rng_state`, `_save_training_state`,
`save_intermediate_checkpoints`, either interrupt handler, best-key
restoration, or a kill-and-resume flow.

### 4.6 Cost of persisting the replay buffer (`VERIFIED` from config and code)

`buffer_size = 300 000`, `semantic_v3` flat dim 3011, float32 observations and
next observations, `optimize_memory_usage` forbidden, float64 raw priorities,
bool validity mask:

| Quantity | Value |
|---|---|
| Bytes per stored transition | ≈ 24.1 kB |
| Full-buffer artifact | ≈ 7.2 GB |
| Peak disk during atomic replace (`keep_last=1`) | ≈ 14.4 GB |
| Writes per 1.5 M-step `thesis` run at `eval_interval` cadence (50k) | 30 |
| Writes per run at `periodic_interval_steps` cadence (125k) | 12 |

`PrioritizedNStepReplayBuffer.__getstate__` truncates to the active rows and
drops the sum-tree (`replay/prioritized.py:284-303`), so artifacts before step
300k are proportionally smaller. Wall-clock per write on local disk is
`INFERRED` at 10–60 s; to be measured in M3 (`TEST-RES-012`).

## 5. Assumptions And Invariants

- Snapshot commit order: model zip and sidecars, replay artifact (if any),
  RNG, quarantine, ACL state, pair manifest, and **last** the training state
  file, which acts as the commit marker. A resume that finds a training state
  whose `global_steps_done` differs from the model's `num_timesteps` (or from
  the pair's `training_timestep`) fails before training (`REQ-RES-003`).
- `os.replace` on the same filesystem is atomic; the temporary file lives next
  to the target (`.<name>.tmp`), as `_save_json_atomically` already does.
- Signal handlers are installed only in the main process of the training CLI,
  after the subprocess vector environment has been created, so workers keep
  the default disposition and are torn down by the parent's normal cleanup.
- Units: steps are environment steps summed over vectorized workers, as
  everywhere else in the loops.
- The RNG state file covers python, numpy and torch (CPU + all CUDA devices).
  Environment RNG is re-derived from `run_seed` and the restored counters,
  unchanged.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-RES-001` | Implementation detail | How to make model/state/RNG writes atomic. | (a) tmp + `os.replace` on the existing file names; (b) adopt `checkpointing.py` generations + `latest.json` pointer. | **(a)**: no layout change, evaluation/video/analysis consumers untouched, ~40 lines. | None observable except crash safety. | Recorded, no approval needed (`DEC-EP-002` already chose the layout). |
| `DEC-RES-002` | New convention | Handle SIGTERM. | (a) install a SIGTERM handler raising `KeyboardInterrupt` in the training CLI main process; (b) leave SIGTERM fatal. | **(a)**. `docker stop` and schedulers send SIGTERM, then SIGKILL after a grace period (Docker default 10 s; raise `stop_grace_period` in `compose.yaml` to cover a replay write). Exit code stays 130 as for Ctrl+C. | Graceful snapshot on the most common non-crash termination. | **Awaiting approval.** |
| `DEC-RES-003` | Specification amendment | Allow a periodic replay snapshot paired with `latest`. | (a) `TRANSITION-REPLAY` v1.1: `persistence.trigger` gains `periodic_and_final`; `periodic_frequency_steps` accepts a positive multiple of `eval_interval`; ACL and standard loops share one save helper; ADR-017 revised so production profiles enable persistence with `periodic_frequency_steps = ${checkpoint.periodic_interval_steps}`. (b) Keep v1; rely on `DEC-RES-004` only (resume with empty buffer). (c) Persist every chunk. | **(a)** with cadence tied to `periodic_interval_steps` (12 writes per `thesis` run, ≤125k steps lost on crash). (c) triples the writes for a small gain; (b) is not a continuation. | Storage ≈ 7.2 GB steady per off-policy run, 14.4 GB peak; a pause of the measured write time every 125k steps. | **Awaiting approval.** Reverses the 2026-09-01 "park it" decision on `V2`. |
| `DEC-RES-004` | Specification clarification | REQ-025 empty-segment resume is currently silent. | (a) new key `checkpoint.resume.allow_replay_reset` (default `false`): when the pair is missing or persistence is disabled, fail closed unless the key is `true`, in which case log `replay_reset=true`, write it to `run_metadata.yaml` and `events.jsonl`, reset β progress; (b) always allow with a warning. | **(a)**: fail closed by default is consistent with the current behaviour for persistence-on runs and with ADR-017's "must not be claimed". | Persistence-off runs that resume today silently would now require the flag. | **Awaiting approval.** |
| `DEC-RES-005` | Scope | Non-ACL scenario-provider draw position is not persisted. | (a) persist per-worker generator state + `FixedSequenceScenarioProvider._position` in the snapshot; (b) defer. | **(b)**: the default configuration uses the ACL loop, whose selection state is persisted; the non-ACL path is used by the learnability screening arms, whose comparability across a crash is already compromised by the lost buffer. Reopen if a non-ACL production run is planned. | A resumed non-ACL run re-draws scenarios from the start of the sequence. | **Awaiting approval** (deferral). |

Dependent work: M1 depends on none of the gates. M2 depends on
`DEC-RES-002`. M3 depends on `DEC-RES-003` and `DEC-RES-004`.

## 7. Proposed Design

### 7.1 M1: crash-safe snapshot, consistent counters (no contract change)

- `runtime/io/atomic.py` (new, ~40 lines): `atomic_write_bytes`,
  `atomic_write_text`, `atomic_pickle_dump`, `atomic_omegaconf_save`, and
  `atomic_call(path, writer)` that hands the writer a `.tmp` sibling and
  `os.replace`s on success, unlinking the temporary on failure. fsync of file
  and directory as in `checkpointing._write_json` / `_fsync_directory`.
- Planner backends: `save()` writes through `atomic_call` (SB3 `model.save`
  accepts any path; the `.zip` suffix is appended by SB3 only when absent, so
  the temporary is named `.<stem>.zip.tmp`). Adapter `.pt` likewise.
- `train_loop._save_training_state`, `_save_rng_state`, and the three ACL
  `write_text` sites move to the atomic helpers. Commit order as in §5.
- Interrupt handlers: `global_steps_done = int(planner.num_timesteps)` (new
  read-only property on the backend protocol, already an attribute on all
  three SB3 backends); `beta_progress_env_steps` carried; ACL RNG save moved
  out of the persistence branch; the ACL handler re-writes its state files
  through the same helper used at chunk boundaries (extract
  `_write_acl_snapshot`).
- Resume: after `load_planner`, assert
  `planner.num_timesteps == training_state.global_steps_done` and, when a pair
  is loaded, `== pair.training_timestep`; assert
  `training_state.seed == cfg.seed`. Snapshots lacking the new
  `model_num_timesteps` field log a warning and skip the first check
  (backward compatibility with runs produced before this plan).
- Best keys: `best_checkpoints.yaml` already stores the metrics that produced
  each best; on resume rebuild the three keys from it through the existing
  `_lexicographic_key` / `_rulebook_*_key` helpers.

### 7.2 M2: SIGTERM (`DEC-RES-002`)

`cli/train.py`: `signal.signal(SIGTERM, _raise_keyboard_interrupt)` installed
inside `run_training` after environment construction; `compose.yaml`
`stop_grace_period` raised to cover one snapshot. Handler logs
`run_interrupted` with `signal=SIGTERM`.

### 7.3 M3: periodic paired replay snapshot (`DEC-RES-003`, `DEC-RES-004`)

- `replay/config.py`: accept `trigger ∈ {final_or_manual, periodic_and_final}`
  and `periodic_frequency_steps` (positive int, must be a multiple of
  `experiment.eval_interval`; validated at resolution time). `keep_last` stays 1.
- One helper `write_latest_snapshot(...)` used by both loops at chunk
  boundaries: when `global_step % periodic_frequency_steps == 0` it writes
  replay + pair before the training state; otherwise it **removes** the stale
  `latest_checkpoint_pair.json` so that `latest` is honestly model-only (REQ-033
  identity would otherwise mismatch by `training_timestep`, which already fails
  closed; removing it makes the `DEC-RES-004` path reachable instead).
- Resume classification: pair present and consistent → continuation; pair
  absent → `allow_replay_reset` gate → `replay_reset=true` event, β reset
  (REQ-025).
- Profiles: `replay_persistence: true` with
  `periodic_frequency_steps: ${checkpoint.periodic_interval_steps}` on every
  profile; `smoke` keeps 10k. ADR-017 superseded by a new ADR.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation (planned) | Tests | Status |
|---|---|---|---|---|
| `REQ-RES-001` | `AC-RES-001`: after a kill injected during any write of a snapshot, the previous snapshot is intact and resumable. | `runtime/io/atomic.py`; backends `save`; `train_loop.py`; `driver.py` | `TEST-RES-001`, `TEST-RES-002`, `TEST-RES-010` | Planned |
| `REQ-RES-002` | `AC-RES-002`: model-only resume emits `replay_reset=true` in `events.jsonl` and `run_metadata.yaml`, β progress 0; without the flag it fails before training. | `train_loop.py` resume block | `TEST-RES-007`, `TEST-RES-008` | Planned (M3) |
| `REQ-RES-003` | `AC-RES-003`: `num_timesteps` ≠ `global_steps_done` fails before training with a message naming both. | `train_loop.py` resume block | `TEST-RES-005` | Planned |
| `REQ-RES-004` | `AC-RES-004`: after Ctrl+C mid-chunk, `latest_training_state.yaml.global_steps_done == planner.num_timesteps`, and `beta_progress_env_steps` is present. | interrupt handlers | `TEST-RES-003`, `TEST-RES-004` | Planned |
| `REQ-RES-005` | `AC-RES-005`: SIGTERM to the training process yields `run_interrupted` and a complete snapshot, exit 130. | `cli/train.py` | `TEST-RES-009` | Planned (M2) |
| `REQ-RES-006` | `AC-RES-006`: with `periodic_frequency_steps=N`, a pair exists exactly at multiples of N and is absent otherwise; resume from a paired `latest` continues the buffer (same length and priorities). | `replay/config.py`, snapshot helper | `TEST-RES-006`, `TEST-RES-011`, `TEST-RES-012` | Planned (M3) |
| `REQ-RES-007` | `AC-RES-007`: after resume, a worse evaluation does not overwrite `best_*.zip`. | `train_loop.py` | `TEST-RES-013` | Planned |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-RES-001` | Unit | `atomic_call` leaves the old file when the writer raises | writer that writes half then raises | old bytes unchanged, no `.tmp` left | `REQ-RES-001` |
| `TEST-RES-002` | Unit | backend `save` never exposes a partial zip | monkeypatched `model.save` raising after partial write | target absent or previous version | `REQ-RES-001` |
| `TEST-RES-003` | Unit | interrupt payload uses `planner.num_timesteps` | fake planner with `num_timesteps=1234`, `current_global_step=1000` | `global_steps_done == 1234` | `REQ-RES-004` |
| `TEST-RES-004` | Unit | interrupt payload carries `beta_progress_env_steps` | as above | key present and equal to loop value | `REQ-RES-004` |
| `TEST-RES-005` | Unit | resume rejects model/state mismatch | state 1000, model 1500 | `ValueError` naming both values before `begin_training` | `REQ-RES-003` |
| `TEST-RES-006` | Unit | `resolve_transition_replay_config` accepts `periodic_and_final` with a multiple of `eval_interval`, rejects a non-multiple | config mappings | accepted / `ValueError` | `REQ-RES-006` |
| `TEST-RES-007` | Unit | model-only resume without flag fails closed | pair absent, `allow_replay_reset=false` | `ValueError` | `REQ-RES-002` |
| `TEST-RES-008` | Unit | model-only resume with flag logs `replay_reset=true`, β=0 | pair absent, flag true | event row and metadata field present | `REQ-RES-002` |
| `TEST-RES-009` | Integration | SIGTERM handled as graceful | `smoke` run, `os.kill(pid, SIGTERM)` after first chunk | exit 130, `run_interrupted` event, snapshot consistent | `REQ-RES-005` |
| `TEST-RES-010` | Integration | kill -9 mid-run then resume | `smoke` TD3 run killed by SIGKILL at a random time after the first snapshot, relaunched with `checkpoint.resume.enabled=true run_dir=<same>` | resumed run finishes; `checkpoint_index.csv` steps monotone; final `global_steps_done == total_timesteps` | `REQ-RES-001`, `REQ-RES-003` |
| `TEST-RES-011` | Unit | pair present exactly at snapshot cadence | fake loop at steps N, 2N, N+eval | pair file exists / removed | `REQ-RES-006` |
| `TEST-RES-012` | Measurement | replay write duration and size at full buffer | 300k synthetic transitions | recorded in this plan's log (no threshold) | `REQ-RES-006` |
| `TEST-RES-013` | Unit | best keys restored from `best_checkpoints.yaml` | yaml with three bests, worse metrics on first eval | no `best_*.zip` rewrite | `REQ-RES-007` |

Commands (all existing): focused tests
`uv run --no-sync python -m pytest -q tests/test_resume_snapshot.py`
(new file); lint/format on touched files
`make lint`, `make format-check PYTHON_QUALITY_PATHS="<touched files>"`;
smoke `make smoke`; compose `make config`. `TEST-RES-009`/`010` run through
the compose environment as `make smoke` does; no CI target exists for them
yet, so they are recorded as manual evidence in §11 until one is added.

## 10. Milestones

- [ ] **M1 — crash-safe snapshot and consistent counters.** No gate. Files:
  `src/thesis_rl/runtime/io/atomic.py`, `runtime/loops/train_loop.py`,
  `curriculum/scenario_acl/driver.py`,
  `agent/planners/algorithms/{sac,td3,ppo}_sb3.py`,
  `agent/planners/interfaces/backend.py`, `tests/test_resume_snapshot.py`.
  Tests `TEST-RES-001..005`, `013`. Evidence: focused suite, `make lint`,
  focused `make format-check`.
- [ ] **M2 — SIGTERM.** Gate `DEC-RES-002`. Files: `src/thesis_rl/cli/train.py`,
  `compose.yaml`. Test `TEST-RES-009`.
- [ ] **M3 — periodic paired replay snapshot.** Gates `DEC-RES-003`,
  `DEC-RES-004`. Files: spec amendment `transition_replay_v1.1_specification.md`
  (through `incoming/` → review → approval), new ADR superseding ADR-017,
  `sb3_extensions/replay/config.py`, snapshot helper, `conf/run_profile/*.yaml`,
  `conf/config.yaml` (`checkpoint.resume.allow_replay_reset`). Tests
  `TEST-RES-006..008`, `011`, `012`, then `TEST-RES-010` and `make smoke`.
- [ ] **M4 — reconciliation.** `open_items` `C12`/`V2`/`D4` updated,
  `project_index.md` row, `docs/setup/validation_commands.md` resume section
  updated with the in-place resume override set
  (`checkpoint.resume.enabled=true checkpoint.resume.run_dir=<dir> paths.run_dir=<dir>`).

## 11. Progress And Findings Log

- **2026-09-06** — Analysis only. Findings recorded in issue #3 and §4.
  Decision gates `DEC-RES-002..005` await the user. No production code
  changed. Next step: user decision on the gates; M1 can start independently.
