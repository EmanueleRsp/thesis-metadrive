# ADR-018: Deterministic Parallel Evaluation And Final Test

- Status: `Approved`
- Date: 2026-07-21
- Decision owner: thesis repository maintainer
- Approval evidence: explicit user instruction to implement the agreed
  evaluation/test parallelization; asynchronous train/evaluation overlap was
  explicitly deferred.
- Related specification: `RL-BASELINES` v1.0
- Related protocol: `docs/protocols/live_eval_video_protocol.md`
- Affected ExecPlan: `docs/implementation/parallel_evaluation_test_v1_exec_plan.md`

## Context

The repository's primary evaluation loop executes one scenario at a time and
the final live video recorder runs in that same loop. Training already has a
process-based vector environment, but evaluation must preserve the exact
scenario order, episode metrics, termination boundaries, and official video
identity of the sequential protocol.

## Decision

1. Parallelize episodes within validation/evaluation/final test using spawned
   process workers; do not overlap evaluation with training in this change.
2. Use explicit parent-controlled reset commands rather than worker auto-reset
   so native seeds and ScenarioNet runtime indices are assigned exactly.
3. Keep policy inference in the parent and execute environment stepping and
   rendering in isolated workers. Supported baselines remain feed-forward and
   non-recurrent.
4. Sort completed episode records by canonical `episode_id` before aggregate
   reduction and CSV emission.
5. Reconstruct the ScenarioNet provider-0 sequence in the parent and force each
   worker to the corresponding runtime index; workers must not race provider
   RNG state.
6. Default to 12 validation workers and 8 final-test workers, cap each by the
   episode count, and allow explicit configuration overrides.

## Consequences

- Evaluation can use a different worker count from scientific training without
  changing learner `n_envs`, PPO rollout semantics, replay shape, or checkpoint
  compatibility.
- Official videos remain tied to the same live episodes as quantitative metrics.
- Parent-side frame transfer and GIF encoding may become the next throughput
  bottleneck; the implementation must report failures rather than silently
  converting official artifacts into offline replays.
- Async train/evaluation overlap requires a separate design for immutable
  snapshots, curriculum gates, resource contention, and failure recovery.

## Approval Record

- Approved by: user.
- Approval date: 2026-07-21.
