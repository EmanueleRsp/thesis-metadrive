# ADR-044: Per-Chunk Step-Timing Instrumentation And Output Format

- Status: `APPROVED`
- Date: 2026-08-01
- Decision owners: thesis repository maintainer
- Approval evidence: user answered the presented `DEC-001`/`DEC-002`/`DEC-003`
  options with all three recommendations on 2026-08-01, in
  `docs/implementation/step_timing_instrumentation_v1_exec_plan.md`
- Related plan: `docs/implementation/step_timing_instrumentation_v1_exec_plan.md`

## Context

The user requested a way to compare computational cost across the algorithms
implemented in this thesis (TD3/SAC/PPO variants): FPS was already logged in
`train_chunks.csv`, but there was no persisted per-component breakdown of
where chunk wall-clock time went (observation preprocessing, policy
inference, environment step including IPC/worker-sync overhead, Rulebook v2
evaluation, transition collection, gradient update), nor any timing for
evaluation-time GIF rendering/annotation cost.

Investigation found the per-component timing (`phase_seconds`) already
existed in `agent.py`'s custom rollout loop but was computed and then dropped
before reaching any CSV — it only reached the JSONL run event log via
`log_event(..., summary=chunk_summary)`. This made the change primarily one
of persistence and derivation, not new instrumentation, except for GIF
rendering, which had no existing timer.

This establishes a new metric/data-logging convention (a new CSV output file
and new columns on an existing one), which `AGENTS.md`'s Decision And Change
Control requires to be approved and recorded via ADR.

## Decision

1. **Output format (`DEC-001`)**: persist the per-chunk component breakdown
   as a new tidy CSV, `step_timing.csv`, with one row per
   `(run_id, chunk_id, component)` — columns `algorithm, seed, run_id,
   chunk_id, global_step, component, seconds, pct_of_elapsed,
   avg_seconds_per_step` — rather than fixed columns on `train_chunks.csv`.
   The component key set is dynamic (Rulebook v2 sub-phases and
   per-algorithm learner detail vary with configuration), so a fixed-width
   schema would silently drop unlisted keys; a tidy row-per-component format
   has no such limitation and is directly `groupby`-able for cross-algorithm
   comparison.
2. **Derived IPC/vec-env sync overhead**: one additional synthetic
   component, `env_step_ipc_overhead`, computed as parent-side `env_step`
   time minus worker-side `worker_wrapped_env_step` time, present only when
   both inputs are available (silent omission otherwise).
3. **Eval GIF timing (`DEC-002`)**: persist evaluation-time GIF
   rendering/annotation/encoding cost as two new scalar columns on the
   existing `evals.csv` (`gif_render_seconds_total`,
   `gif_render_seconds_per_episode`), rather than a new tidy table, since
   `evals.csv` already has one row per eval event with fixed scalar columns
   and GIF timing is a single scalar aggregate per eval, not a dynamic-key
   breakdown.
4. **Replay-buffer write isolation deferred (`DEC-003`)**: the existing
   `transition_collection` phase bucket (replay-buffer writes + ACL/
   learning-potential bookkeeping combined) is left as-is. Isolating replay
   writes further would require touching every per-algorithm backend
   (TD3/SAC/PPO); deferred as optional future work.

Percentages are `component_seconds / elapsed_seconds`, matching how
`unattributed` was already derived in `agent.py`; average-per-step time is
`component_seconds / chunk_steps_actual`.

## Consequences

Positive:

- reuses timing already measured by `agent.py`'s rollout loop; no new
  instrumentation for training-loop components, only persistence;
- the tidy format survives future changes to the Rulebook sub-phase set or
  the addition of new algorithms without a schema migration;
- GIF timing is isolated from training-loop timing by construction (measured
  entirely inside the evaluation-only `LiveEvalEpisodeRecorder`).

Negative:

- `step_timing.csv` is a new output file every run now produces, increasing
  per-run disk usage roughly linearly in `(chunks × distinct components)`;
- joining `step_timing.csv` back to `train_chunks.csv` requires a
  `(run_id, chunk_id)` merge rather than reading a single wide row;
- `transition_collection` remains a single bucket mixing replay-buffer
  writes and ACL bookkeeping, so it cannot yet answer "how much does the
  replay buffer alone cost" without further work.

## Alternatives rejected

- Fixed columns on `train_chunks.csv` for the component breakdown (`DEC-001`
  option A): rejected because the component key set is dynamic and a fixed
  schema would silently lose data for any future Rulebook sub-phase or
  algorithm-specific learner-detail key not anticipated today.
- A shared tidy table for both chunk-level and eval-level timing (`DEC-002`
  option B): rejected to avoid mixing two different row granularities
  (`chunk_id` vs. `eval_id`) in one table.
- Isolating replay-buffer write time from ACL bookkeeping now (`DEC-003`
  option A): rejected for this pass as disproportionate scope/risk (touches
  every per-algorithm backend) relative to the benefit of one additional
  sub-bucket; `transition_collection` already answers the question at one
  level of granularity.
