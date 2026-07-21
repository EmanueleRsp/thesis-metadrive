# ADR-019: Asynchronous Evaluation Queue During Training

- Status: `Approved`
- Date: 2026-07-21
- Decision owner: thesis repository maintainer
- Approval evidence: explicit user confirmation in the implementation thread
- Related specification: `RL-BASELINES` v1.0
- Related ADRs: `ADR-018-parallel-evaluation-and-test.md`, `ADR-016`
- Affected ExecPlan: `docs/implementation/asynchronous_evaluation_v1_exec_plan.md`

## Context

Evaluation episodes are already parallelized within an evaluation run. The
remaining throughput opportunity is to let ordinary validation and Scenario ACL
diagnostic evaluation run while the learner continues training. Staged
curriculum evaluation cannot use this model because its result gates promotion.
The UI must keep training and evaluation visible together, and final test must
remain a distinct post-training operation.

## Decision

1. Ordinary validation and Scenario ACL diagnostic evaluation are asynchronous;
   staged curriculum evaluation remains synchronous and is a mandatory barrier
   before promotion.
2. Each triggered evaluation creates an immutable model snapshot and a FIFO job.
   Every queued job runs to completion with its own snapshot, configuration,
   episode budget, and seed schedule; no job is skipped, replaced, or cancelled.
3. At most one evaluation process is active at a time. Training continues while
   the active job runs and continues to enqueue later jobs.
4. Any evaluation failure is fatal to the whole training run. The parent drains
   and surfaces the failure rather than silently continuing with incomplete
   validation data.
5. The learner keeps using its live model and mutable training state. Snapshots
   are evaluator-only copies and do not replace or pause the learner.
6. The Rich UI is owned by the training parent and shows training progress,
   training monitor, merged train/evaluation event log, current evaluation
   progress/status, and a last-completed-evaluation table. The latter is updated
   atomically only when a complete result is received.
7. At training completion, all queued evaluations are drained before the final
   test. Final test remains a separate section below training and is not queued
   with diagnostics.
8. Evaluation and training use the normal configured CPU/GPU resources. No
   implicit CPU-only fallback or resource partition is introduced.

## Consequences

- Diagnostic validation no longer blocks learner progress, but its result is
  timestamped to the immutable snapshot that produced it.
- A long evaluation queue can delay final test because completeness is required;
  this is intentional and preserves the user's all-jobs contract.
- Staged curriculum timing and promotion semantics remain unchanged.
- Snapshot storage is temporary job state and is removed only after the job has
  completed successfully; failures stop the run and retain diagnostic context.

## Approval Record

- Approved by: user.
- Approval date: 2026-07-21.
