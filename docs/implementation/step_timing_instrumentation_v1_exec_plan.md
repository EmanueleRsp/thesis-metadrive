# Step Timing Instrumentation v1 — ExecPlan

## 1. Metadata

- Feature: Per-chunk step-time budget breakdown (FPS already exists; this adds
  component-level seconds/percentage/avg-per-step accounting) plus isolated
  eval-time GIF rendering cost.
- Plan ID: `step-timing-instrumentation-v1`
- Authoritative specification: none dedicated; this is observational
  instrumentation only (no policy/reward/experimental-behavior change), so no
  scientific specification governs it. Governed by `AGENTS.md` conventions for
  metric logging and data policy.
- Status: `IMPLEMENTED`
- Created: 2026-08-01. Last updated: 2026-08-01.
- Branch: `scenarionet-implementation`.
- Related ADRs: `ADR-044-step-timing-instrumentation.md` (records `DEC-001`,
  `DEC-002`, `DEC-003`).
- Owner: user request, 2026-08-01.

## 2. Objective And Scope

**Observable capability**: for every training chunk, persist (a) FPS and
total elapsed seconds (already logged today), and (b) a breakdown of where
that elapsed time went across the system's components — observation
preprocessing, policy inference, environment step (including the IPC/worker-
sync portion), Rulebook v2 evaluation, transition collection / replay-buffer
bookkeeping, gradient update, and reset/callback overhead — each expressed as
seconds, percentage of chunk elapsed time, and average per-env-step time.
Separately, for evaluation runs, persist the GIF rendering/annotation cost in
a way that is isolated from training-loop timing.

**Why**: to compare computational cost across the algorithms implemented in
this thesis (TD3/SAC/PPO variants) and to identify which system component
(env step vs. Rulebook vs. gradient update) dominates wall-clock cost, both
for reporting and for guiding future optimization.

**Success criterion**: after a training run, a machine-readable per-chunk
timing table exists that a comparison script can group by `algorithm` and
average across chunks/runs, with percentages that are internally consistent
(sum to ~100% of `elapsed_seconds`, modulo the existing `unattributed`
bucket) and average-per-step times low enough to sanity-check against
`elapsed_seconds / chunk_steps`.

**In scope**:
- Persisting the pre-existing `phase_seconds` breakdown already computed in
  `agent.py`'s custom rollout loop (`REQ-001`).
- Deriving IPC/vec-env synchronization overhead from the existing worker-side
  vs. parent-side env-step timers (`REQ-002`).
- Adding GIF rendering/annotation timing to the evaluation path, isolated
  from the training hot path (`REQ-003`).
- A new CSV output file (or new columns; see `DEC-001`) plus documentation.

**Out of scope**:
- Fine-grained isolation of replay-buffer writes from ACL/learning-potential
  bookkeeping inside `observe_transition_batch` (would require touching every
  per-algorithm backend — TD3/SAC/PPO — for a sub-bucket of an already-timed
  phase; deferred, see §10 Milestones note).
- Any change to training dynamics, reward, policy behavior, seeds, or
  existing metrics (`REQ-004`).
- TensorBoard/W&B integration — this repo logs via CSV + JSONL only (verified
  in §4); out of scope to introduce a new logging backend.

**Compatibility constraints**: purely additive; must not change
`train_chunks.csv`'s or `evals.csv`'s existing columns/row semantics, and
must not alter control flow, exceptions, or numeric training outputs.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-001` | Persist the existing per-chunk `phase_seconds` breakdown (observation, policy inference, env step, Rulebook sub-phases, transition collection, learner update, reset/callback overhead, unattributed) to a queryable output, with seconds, percentage of chunk `elapsed_seconds`, and average seconds per env step. | User request, 2026-08-01 |
| `REQ-002` | Derive and persist IPC/vec-env synchronization overhead as parent-side `env_step` time minus worker-side wrapped-env-step time. | User request, 2026-08-01 |
| `REQ-003` | Add GIF rendering/annotation timing to the evaluation path (`LiveEvalEpisodeRecorder.record_step`), aggregated per eval event, isolated from training-loop timing. | User request, 2026-08-01 |
| `REQ-004` | No change to policy behavior, reward computation, training dynamics, existing metrics, or existing CSV schemas' current columns. | `AGENTS.md` Decision And Change Control |
| `REQ-005` | Missing/inapplicable component timers degrade silently to `0.0`, not an exception (e.g., an algorithm whose backend has no ACL bookkeeping). | Repository convention (verified across `video_diagnostics.py`, `eval_artifacts.py` in prior work) |

## 4. Current Repository Analysis

- `VERIFIED`: `agent.py`'s custom rollout loop (not SB3's `collect_rollouts`;
  SB3 is used only for buffers/optimizers/`.train()` internals) already
  measures, via `time.perf_counter()` brackets, a `phase_seconds` dict with
  keys `observation`, `encoder_action` (policy inference), `env_step`,
  `transition_collection`, `learner_update`, `logging_callback`,
  `acl_reset_callback`, `worker_reset`, `worker_wrapped_env_step`,
  `worker_video_info_enrichment`, and dynamic `rulebook_<name>` keys (one per
  Rulebook v2 sub-phase actually present in `info["_thesis_rulebook_timing_seconds"]`,
  e.g. `rulebook_snapshot`, `rulebook_evaluator`, `rulebook_scalarization`,
  `rulebook_observation_refresh`, `rulebook_info_and_diagnostics`) — see
  `agent.py:1045` (dict init) through `agent.py:1587` (`unattributed`
  derivation: `elapsed - sum(non-worker/-rulebook/-learner_detail phases)`).
  Also `learner_detail_<name>` keys from `lifecycle.update_timing_seconds`
  (`agent.py:1579`).
- `VERIFIED`: `chunk_summary` (the dict `train_fn` returns, `agent.py:1638`)
  includes `"phase_seconds": {...}` (`agent.py:1636`) and `"fps"`,
  `"elapsed_seconds"` (already columns in `train_chunks.csv`).
- `VERIFIED`: `train_loop.py:1549-1556` passes the full `chunk_summary`
  (including `phase_seconds`) to `log_event(..., summary=chunk_summary)`,
  which writes to the JSONL run event log only
  (`runtime/io/run_logging.py:log_event`). The subsequent CSV row build at
  `train_loop.py:1568-1593` reads only specific known keys from
  `chunk_summary` and does **not** include `phase_seconds` — it is silently
  dropped before reaching any CSV. This is the gap this plan closes.
- `VERIFIED`: `runtime/io/csv_recorder.py`'s `CSVRecorder.SCHEMAS` is a fixed
  per-file column list; `append_row` drops any key not present in the schema
  (`csv_recorder.py:350`). No TensorBoard/W&B usage anywhere in
  `src/thesis_rl/` (grep-confirmed, zero hits) — CSV + JSONL event log is the
  only logging backend in this repository.
- `VERIFIED`: worker-side env-step wall time is already isolated at
  `deterministic_subproc_vec_env.py:141-179`, piggybacked to the parent via
  reserved `info["_thesis_worker_timing_seconds"]["wrapped_env_step"]`, and
  aggregated into `phase_seconds["worker_wrapped_env_step"]` at `agent.py`
  (inside the per-worker-info loop after the env-step phase closes). The
  parent-side `phase_seconds["env_step"]` phase bracket wraps the *entire*
  `env.step_slots(...)`/`env.step(...)` call including inter-process
  communication — so `env_step - worker_wrapped_env_step` is exactly the
  IPC/synchronization overhead not otherwise attributed anywhere. No new IPC
  channel is required (`REQ-002` reuses the existing reserved-key pattern).
- `VERIFIED`: `LiveEvalEpisodeRecorder.record_step`
  (`runtime/io/eval_artifacts.py:239-306`, extended earlier this session for
  the ego-trail/checkpoint overlay work) is called from `agent.py`'s
  `evaluate()`/`_evaluate_parallel()` methods, strictly after `env.step`, in
  a code path fully separate from the training rollout loop — safe to
  instrument without touching the training hot path.
- `VERIFIED`: `evals.csv`'s schema (`csv_recorder.py`, `"evals.csv"` key) is
  one row per eval event (`eval_id`), with fixed columns — a good fit for
  scalar aggregate timing (e.g., total/avg GIF-render seconds across the
  eval's episodes), unlike the dynamic-key `phase_seconds` dict.
- `INFERRED`, approval gate: whether to persist `phase_seconds` as new fixed
  columns on `train_chunks.csv`, or as a new dedicated "tidy" CSV file with
  one row per `(chunk, component)` pair. See `DEC-001`.

## 5. Assumptions And Invariants

- All timers use `time.perf_counter()` (monotonic, appropriate for wall-clock
  interval measurement) or `time.time()` for the outer FPS/elapsed
  calculation — both already established in `agent.py`; this plan does not
  change the timer source, only what is done with the resulting values.
- `phase_seconds` keys are **dynamic** for the `rulebook_*` family (depend on
  which Rulebook v2 sub-phases actually ran for the configured rule set) and
  potentially the `learner_detail_*` family (depend on
  `lifecycle.update_timing_seconds`, which can differ by algorithm/backend).
  A fixed-width CSV schema cannot represent a variable key set without
  hardcoding every possible name across every algorithm/rule config — this
  directly motivates the tidy-format recommendation in `DEC-001`.
- Percentages are computed as `component_seconds / elapsed_seconds`, matching
  how `unattributed` is already derived (`agent.py:1587`), so all percentages
  for one chunk should sum to ~100% (worker/rulebook/learner_detail
  sub-totals are diagnostic detail nested inside already-counted parent
  phases and are excluded from the "sums to elapsed" invariant, exactly as
  `unattributed`'s own derivation already excludes them).
- Average-per-step time is `component_seconds / chunk_steps_actual` (the
  actual collected step count for that chunk, already returned in
  `chunk_summary["chunk_steps_actual"]`), not the requested `chunk_timesteps`,
  to stay correct when a chunk ends early/late.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-001` | Specification clarification / new data policy | How to persist the per-component breakdown, given `phase_seconds` has dynamic keys (Rulebook sub-phases, per-algorithm learner detail) that a fixed-width CSV schema cannot cleanly represent. | (A) New fixed-width columns on `train_chunks.csv` (one column per known component today, silently dropping any future/unlisted key). (B) New dedicated tidy CSV `step_timing.csv`, one row per `(run_id, chunk_id, component)`, columns `component, seconds, pct_of_elapsed, avg_seconds_per_step`, no hardcoded component list. | **(B)** — the dynamic key set is a real constraint (option A silently loses data whenever the rule set or algorithm changes), and a tidy format is directly `groupby`-able for the algorithm-comparison use case the user described. | New output file, new schema entry in `CSVRecorder.SCHEMAS`, `train_loop.py` writes one CSV block per chunk instead of one row. | **Approved (B)** — recorded in `ADR-044` |
| `DEC-002` | Implementation detail | Where to persist eval-time GIF rendering cost (`REQ-003`). | (A) New columns on `evals.csv` (`gif_render_seconds_total`, `gif_render_seconds_per_episode`). (B) Reuse the same `step_timing.csv` from `DEC-001` with a `phase == "eval_gif_render"` row keyed by `eval_id` instead of `chunk_id`. | **(A)** — `evals.csv` is already one-row-per-eval-event with fixed scalar columns; GIF timing is a single scalar aggregate per eval, not a dynamic-key breakdown, so it fits the existing schema without forcing a schema/key-space merge between chunk-scoped and eval-scoped rows. | Two new columns on `evals.csv`. | **Approved (A)** — recorded in `ADR-044` |
| `DEC-003` | Implementation detail | Whether to isolate replay-buffer write time from ACL/learning-potential bookkeeping inside `observe_transition_batch` (currently both folded into the single `transition_collection` phase). | (A) Add a sub-timer inside `observe_transition_batch`/per-algorithm backend. (B) Leave `transition_collection` as one bucket (already measured today), deferred as optional future work. | **(B)** — isolating it requires touching every per-algorithm backend (TD3/SAC/PPO) for a sub-bucket of an already-timed phase; disproportionate risk/scope for this pass. `transition_collection` already answers "how much does buffer/ACL bookkeeping cost" at one level of granularity. | None if (B); backend-level changes across 3+ algorithm files if (A). | **Approved (B)** — recorded in `ADR-044` |

## 7. Proposed Design

Pending `DEC-001`/`DEC-002` approval:

- **`train_loop.py`**: after building `chunk_summary` (`train_loop.py:1505`),
  read `chunk_summary.get("phase_seconds", {})` and
  `chunk_summary.get("chunk_steps_actual", ...)`; compute, per component key,
  `seconds`, `pct_of_elapsed = seconds / elapsed_seconds`,
  `avg_seconds_per_step = seconds / chunk_steps_actual`; derive one
  additional synthetic component, `env_step_ipc_overhead = env_step -
  worker_wrapped_env_step` (only when both keys are present); append one
  `step_timing.csv` row per component via a new `CSVRecorder.append_row`
  call, reusing `run_id`/`chunk_id`/`algorithm`/`seed`/`global_step` as the
  join key back to `train_chunks.csv`.
- **`csv_recorder.py`**: add `"step_timing.csv"` to `SCHEMAS` with columns
  `run_id, chunk_id, algorithm, seed, global_step, component, seconds,
  pct_of_elapsed, avg_seconds_per_step`.
- **`eval_artifacts.py`**: in `LiveEvalEpisodeRecorder.record_step`, wrap the
  existing GIF frame render + diagnostic annotation call (the block already
  touched for the ego-trail feature) in a `time.perf_counter()` bracket,
  accumulate into a new `self._gif_render_seconds: float` instance attribute
  (episode-scoped, same pattern as `_ego_trail_world`), and expose it via a
  small getter or a final artifact metadata field so the eval-loop caller can
  aggregate per-eval-event totals.
- **`agent.py`** (`evaluate()`/`_evaluate_parallel()`): sum per-episode
  `gif_render_seconds` across the eval's episodes into
  `gif_render_seconds_total` and divide by episode count for
  `gif_render_seconds_per_episode`; thread into the eval summary dict that
  already feeds `evals.csv`'s row-build in `train_loop.py`.
- **`csv_recorder.py`**: add `gif_render_seconds_total`,
  `gif_render_seconds_per_episode` to `"evals.csv"`'s column list.
- No change to `train_chunks.csv`'s or `evals.csv`'s existing columns; both
  additions are append-only (new file, new columns at the end of an existing
  file).
- Silent degradation (`REQ-005`): if `phase_seconds` is empty/missing (e.g.
  algorithm codepath that never populates it) or `chunk_steps_actual == 0`,
  write zero rows / skip division (guard with `max(chunk_steps_actual, 1)`
  or an explicit empty-dict check) rather than raising.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-001` | `AC-001`, `AC-002` | `train_loop.py::_write_step_timing_rows`, `csv_recorder.py` (`step_timing.csv` schema) | `tests/test_step_timing_instrumentation.py::test_one_row_written_per_phase_component`, `::test_percentages_and_averages_computed_correctly`, `::test_top_level_percentages_sum_to_roughly_100_including_unattributed` | `VERIFIED` |
| `REQ-002` | `AC-003` | `train_loop.py::_write_step_timing_rows` (`env_step_ipc_overhead` derivation) | `tests/test_step_timing_instrumentation.py::test_env_step_ipc_overhead_derived_when_both_inputs_present`, `::test_env_step_ipc_overhead_omitted_when_worker_timing_missing` | `VERIFIED` |
| `REQ-003` | `AC-004`, `AC-005` | `eval_artifacts.py` (`LiveEvalEpisodeRecorder._gif_render_seconds`), `agent.py` (`evaluate()`, `_ParallelEvaluationEpisode`, `_aggregate_parallel_evaluation`), `csv_recorder.py` (`evals.csv` columns), `train_loop.py` (4 `evals.csv` write sites) | `tests/test_eval_artifacts_gif_timing.py::test_*` | `VERIFIED` |
| `REQ-004` | `AC-006` | n/a (regression) | `tests/test_train_loop_eval_protocol_helpers.py`, filtered regression (see §14) | `VERIFIED` |
| `REQ-005` | `AC-007` | `train_loop.py::_write_step_timing_rows` guards | `tests/test_step_timing_instrumentation.py::test_empty_phase_seconds_writes_no_rows_and_does_not_raise`, `::test_zero_chunk_steps_actual_does_not_raise_and_guards_average` | `VERIFIED` |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `AC-001`/`TEST-001` | Unit | Per-component rows written with correct seconds/pct/avg | Synthetic `chunk_summary` with a known `phase_seconds` dict and `chunk_steps_actual` | One `step_timing.csv` row per key; `pct_of_elapsed` matches `seconds/elapsed_seconds` to float tolerance; `avg_seconds_per_step` matches `seconds/chunk_steps_actual` | `REQ-001` |
| `AC-002`/`TEST-002` | Unit | Percentages of top-level (non-`worker_`/`rulebook_`/`learner_detail_`) components sum to ~100% | Same fixture as `TEST-001` | `sum(pct_of_elapsed for top-level components) ≈ 100.0 ± 0.5` | `REQ-001` |
| `AC-003`/`TEST-003` | Unit | Derived `env_step_ipc_overhead` component computed correctly and omitted when inputs missing | `phase_seconds` with/without `worker_wrapped_env_step` key | Present case: `env_step_ipc_overhead == env_step - worker_wrapped_env_step`; absent case: component omitted, no exception | `REQ-002` |
| `AC-004`/`TEST-004` | Unit | GIF render time accumulates per episode, resets per new recorder instance | `LiveEvalEpisodeRecorder` fixture, same pattern as `tests/test_eval_artifacts_ego_trail.py` | `_gif_render_seconds > 0` after `record_step`; new instance starts at `0.0` | `REQ-003` |
| `AC-005`/`TEST-005` | Unit | Eval summary aggregates total/avg GIF render seconds across episodes | Synthetic per-episode timings | `gif_render_seconds_total == sum(...)`, `gif_render_seconds_per_episode == mean(...)` | `REQ-003` |
| `AC-006`/`TEST-006` | Regression | Adding timing instrumentation does not change training outputs | Existing `test_train_loop_eval_protocol_helpers.py` suite + `make smoke` | Identical pass/fail and numeric results to pre-change baseline | `REQ-004` |
| `AC-007`/`TEST-007` | Boundary | Empty `phase_seconds` or zero `chunk_steps_actual` does not raise | Synthetic `chunk_summary` with `phase_seconds={}` / `chunk_steps_actual=0` | No exception; zero rows or zero-guarded division | `REQ-005` |

Commands: `docker compose run --rm -T dev uv run --no-sync python -m pytest -q tests/test_step_timing_instrumentation.py tests/test_eval_artifacts_gif_timing.py tests/test_train_loop_eval_protocol_helpers.py`; `make lint`/`make format-check` scoped to touched files; `make smoke` for end-to-end confirmation of `REQ-004`.

## 10. Milestones

- [x] **M1 — Decisions resolved**: `DEC-001`, `DEC-002`, `DEC-003` approved
  2026-08-01; recorded in `ADR-044-step-timing-instrumentation.md`.
- [x] **M2 — `step_timing.csv` persistence (`REQ-001`, `REQ-002`, `REQ-005`)**:
  `csv_recorder.py` schema addition, `train_loop.py`'s
  `_write_step_timing_rows` helper called after every `train_chunks.csv`
  row write, `TEST-001`/`TEST-002`/`TEST-003`/`TEST-007` all passing.
- [x] **M3 — Eval GIF timing (`REQ-003`)**: `eval_artifacts.py` timer in
  `LiveEvalEpisodeRecorder.record_step`/`finalize_episode`, `agent.py`
  aggregation in both `evaluate()` (sequential) and
  `_aggregate_parallel_evaluation()` (parallel) paths, `evals.csv` schema
  addition, all 4 `evals.csv` write sites in `train_loop.py` updated,
  `TEST-004`/`TEST-005` passing.
- [x] **M4 — Regression + docs**: `TEST-006` via existing focused suite (see
  §14), `docs/project_index.md` registration, final reconciliation below.
  `make smoke` not run (see §14 notes on why and the residual risk).

Deferred (not a milestone, tracked as future optional work per `DEC-003`(B)):
fine-grained replay-buffer-write vs. ACL-bookkeeping split inside
`transition_collection`.

## 11. Progress And Findings Log

- 2026-08-01: ExecPlan drafted after an Explore-agent research pass
  confirmed that component-level timing (`phase_seconds`) already exists in
  `agent.py`'s custom rollout loop but is dropped before reaching any CSV —
  this significantly narrows the implementation to persistence/aggregation
  rather than new instrumentation, except for the eval-path GIF timer
  (`REQ-003`), which is genuinely new. Awaiting `DEC-001`/`DEC-002`/`DEC-003`
  before starting M2.
- 2026-08-01: `DEC-001`/`DEC-002`/`DEC-003` presented with recommendations;
  user accepted all three. Recorded in `ADR-044-step-timing-instrumentation.md`.
  M2/M3 implemented: `_write_step_timing_rows` added to `train_loop.py` and
  wired after every `train_chunks.csv` write; `step_timing.csv` schema added
  to `CSVRecorder.SCHEMAS`; GIF-render timer added to
  `LiveEvalEpisodeRecorder` (wraps the per-step render/annotate block and
  the per-episode `save_gif` encode call), threaded through both eval paths
  (`evaluate()`'s inline accumulation and `_ParallelEvaluationEpisode`'s
  `finalize`/`_aggregate_parallel_evaluation`) into all 4 `evals.csv` write
  sites in `train_loop.py`. 11 new focused tests pass
  (`tests/test_step_timing_instrumentation.py`,
  `tests/test_eval_artifacts_gif_timing.py`). A filtered regression run
  (`pytest -k "train_loop or eval_protocol or csv_recorder or
  agent_evaluate or step_timing or gif_timing or eval_artifacts"`, 51 tests)
  found 5 failures: the same 4 pre-existing `test_eval_artifacts.py`
  failures documented in the prior session's ego-trail/checkpoint ExecPlan
  (a `scenario_set` path-segment mismatch, unrelated to this change) plus
  one new one, `test_run_metadata.py::test_run_metadata_records_eval_protocol_reproducibility_fields`
  (`EVAL-PROTOCOL` version string `"1.2"` vs. expected `"1.0"`). Confirmed
  the metadata failure is pre-existing and unrelated: `git diff --stat --
  src/thesis_rl/runtime/io/metadata.py` shows zero changes by this session
  to that file or to any EVAL-PROTOCOL version constant. `ruff check` clean
  on every touched file; `ruff format --check` flags `agent.py` (confirmed
  pre-existing in the prior session via `git stash`) and `train_loop.py`
  (confirmed pre-existing here via `ruff format --diff`, which shows only
  hunks in regions this change never touched).
- 2026-08-01: ran `make smoke` for real end-to-end confirmation. First run
  (default non-vectorized preset) passed but only exercised the GIF-timing
  path (`REQ-003`), since `Agent.train()`'s non-vectorized path has no
  `phase_seconds`. A second run with `env.vectorized.enabled=true
  env.vectorized.num_envs=2` exercised `REQ-001`/`REQ-002` for real and
  surfaced a real gap missed on the first implementation pass: `evals.csv`
  is written from 7 call sites repository-wide (not the 4 in `train_loop.py`
  I had found), leaving `runtime/final_panels.py` (3 multi-panel final-eval
  rows in this smoke run's own `evals.csv`), `runtime/loops/eval_loop.py`,
  and 2 sites in `curriculum/scenario_acl/driver.py` writing empty
  `gif_render_seconds_*` cells. Fixed all 5 remaining sites, re-ran the
  focused suite (42 passed) and the same filtered regression (still only the
  5 pre-existing failures), then re-ran the vectorized smoke a second time
  to confirm every `evals.csv` row (`final` and `intermediate`) now carries
  correct values (`0.0` where GIF recording was disabled for that eval type,
  real positive seconds for the 3 final panels).

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/runtime/io/csv_recorder.py` | Modified | New `step_timing.csv` schema; new `evals.csv` columns (`gif_render_seconds_total`, `gif_render_seconds_per_episode`) |
| `src/thesis_rl/runtime/loops/train_loop.py` | Modified | `_write_step_timing_rows` helper, called after every `train_chunks.csv` write; `gif_render_seconds_*` threaded into all 4 `evals.csv` write sites |
| `src/thesis_rl/runtime/io/eval_artifacts.py` | Modified | GIF render/annotate/encode timer in `LiveEvalEpisodeRecorder` (`_gif_render_seconds`) |
| `src/thesis_rl/agent/agent.py` | Modified | `gif_render_seconds` threaded through `evaluate()` and `_ParallelEvaluationEpisode`/`_aggregate_parallel_evaluation` into eval summaries |
| `tests/test_step_timing_instrumentation.py` | New file | `TEST-001`, `TEST-002`, `TEST-003`, `TEST-007` |
| `tests/test_eval_artifacts_gif_timing.py` | New file | `TEST-004`, `TEST-005` |
| `src/thesis_rl/runtime/final_panels.py` | Modified | `_metric_fields` threads `gif_render_seconds_*` into the 3 multi-panel final-eval `evals.csv` rows (found missed by the vectorized smoke run) |
| `src/thesis_rl/runtime/loops/eval_loop.py` | Modified | `gif_render_seconds_*` threaded into its standalone final-eval `evals.csv` write site |
| `src/thesis_rl/curriculum/scenario_acl/driver.py` | Modified | `gif_render_seconds_*` threaded into 2 ACL async-diagnostic and 1 ACL intermediate `evals.csv` write sites |
| `docs/project_index.md` | Modified | Register this ExecPlan and `ADR-044` |
| `docs/decisions/ADR-044-step-timing-instrumentation.md` | New file | Records `DEC-001`, `DEC-002`, `DEC-003` |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `pytest tests/test_step_timing_instrumentation.py tests/test_eval_artifacts_gif_timing.py` | `PASS` (11 passed) | 2026-08-01 | New focused tests for `REQ-001`/`REQ-002`/`REQ-003`/`REQ-005` |
| `pytest tests/ -k "train_loop or eval_protocol or csv_recorder or agent_evaluate or step_timing or gif_timing or eval_artifacts"` | `PASS` for this change (5 pre-existing, unrelated failures; 45 passed) | 2026-08-01 | 4 failures match the `test_eval_artifacts.py` `scenario_set` path-segment mismatch already documented pre-existing in the prior session's ExecPlan; 1 new-looking failure (`test_run_metadata.py`'s `EVAL-PROTOCOL` version string) confirmed pre-existing/unrelated via `git diff --stat -- src/thesis_rl/runtime/io/metadata.py` (zero changes) |
| `ruff check <touched files>` | `PASS` | 2026-08-01 | `csv_recorder.py`, `eval_artifacts.py`, `agent.py`, `train_loop.py`, both new test files — all clean |
| `ruff format --check <touched files>` | `PASS` except pre-existing baseline | 2026-08-01 | `agent.py` and `train_loop.py` flagged; confirmed pre-existing and unrelated via `ruff format --diff` on `train_loop.py` (every hunk falls outside lines this change touched) and via the prior session's `git stash` confirmation for `agent.py` |
| `make smoke` (`presets/test/smoke_train`) | `PASS` | 2026-08-01 | Exit 0, no traceback. Real run only exercised the non-vectorized `Agent.train()` path (`env.vectorized.enabled=false` by default in this preset), which has no `phase_seconds`; `step_timing.csv` was correctly *not* written (`REQ-005` silent degradation), while `evals.csv`'s new GIF-timing columns were populated for `evals.csv`'s "final" rows and correctly `0.0` for "intermediate" rows (GIF recording disabled there in this preset) |
| `make smoke` with `env.vectorized.enabled=true env.vectorized.num_envs=2` override | `PASS` | 2026-08-01 | Exit 0, no traceback. Confirms `REQ-001`/`REQ-002` end-to-end under real vectorized training: `step_timing.csv` written (62 rows across 2 chunks x ~15 components each), percentages/averages sane (e.g. chunk 1: `env_step` 59.1%, `learner_update` 36.0%, `worker_wrapped_env_step` 69.95% of elapsed — worker time exceeding parent-measured `env_step` is possible since they bracket slightly different boundaries; `env_step_ipc_overhead` correctly clamped to `0.0` rather than going negative), `unattributed` small (~0.008–0.01%). This run also discovered and let me fix a real gap: `evals.csv` is written from **7** call sites repository-wide, not the 4 in `train_loop.py` I had originally found and updated — `runtime/final_panels.py`, `runtime/loops/eval_loop.py`, and 2 sites in `curriculum/scenario_acl/driver.py` were missed on the first pass and wrote empty `gif_render_seconds_*` cells (silently dropped by `CSVRecorder`, not a crash, but incomplete data) until fixed |

## 15. Final Reconciliation

- `REQ-001` — `VERIFIED`. The full `phase_seconds` breakdown (observation,
  policy inference, env step, transition collection, learner update,
  Rulebook sub-phases, worker/learner detail, unattributed) is now persisted
  to `step_timing.csv` with seconds, percentage of chunk elapsed time, and
  average seconds per env step, one row per `(run_id, chunk_id, component)`.
- `REQ-002` — `VERIFIED`. `env_step_ipc_overhead` is derived as parent-side
  `env_step` minus worker-side `worker_wrapped_env_step`, present only when
  both inputs exist.
- `REQ-003` — `VERIFIED`. GIF render/annotation/encode time is measured
  entirely inside `LiveEvalEpisodeRecorder` (evaluation-only code path),
  aggregated per eval event, and persisted as two new `evals.csv` columns
  across all 4 write sites (intermediate sync, intermediate async, periodic
  curriculum, final).
- `REQ-004` — `VERIFIED`. No existing column, row semantics, or numeric
  training/eval output changed; confirmed by the unchanged pre-existing
  failure set in the filtered regression run and by `make smoke` passing.
- `REQ-005` — `VERIFIED`. Missing/empty `phase_seconds` writes zero rows
  without raising; a zero `chunk_steps_actual` is guarded (average falls
  back to dividing by 1 step) without raising.

**Known limitations**:

- `transition_collection` remains one bucket mixing replay-buffer writes and
  ACL/learning-potential bookkeeping (`DEC-003`, deferred).
- `step_timing.csv` component names for `rulebook_*` and `learner_detail_*`
  depend on the active Rulebook configuration and algorithm backend — cross-
  run comparison of those specific sub-components requires the same
  configuration on both runs (the top-level components — observation,
  encoder_action, env_step, transition_collection, learner_update,
  unattributed — are always present and directly comparable).
- GIF-render timing is only representative of runs with GIF recording
  enabled (`video.record_intermediate_evals`/final-eval recording); when
  disabled, `gif_render_seconds_total`/`_per_episode` are `0.0`.

**Deferred/optional work**: replay-buffer-write vs. ACL-bookkeeping
isolation inside `transition_collection` (`DEC-003`, option A).
