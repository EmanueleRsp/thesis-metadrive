# ADR-016: Deterministic Parent-Controlled Scenario ACL Vectorization

- Status: `Approved`
- Date: `2026-07-20`
- Decision owner: thesis repository maintainer
- Approval evidence: explicit user approval of DEC-VEC-001--DEC-VEC-005 on 2026-07-20
- Related ADRs: ADR-001, ADR-014
- Affected ExecPlan: `docs/implementation/scenario_acl_vectorized_execution_v1_exec_plan.md`

## Context

Scenario ACL must execute multiple frozen ScenarioNet episodes concurrently while
keeping one curriculum decision and one learning-potential attribution per slot
and logical episode. The existing generic vector environment auto-resets a
completed worker before the parent can install the next ACL selection, and the
learner summaries do not retain enough episode provenance for ACL feedback.

## Decision

For the ScenarioNet ACL vector path:

1. The parent owns selection, buffer, MAB, RNG, and transaction state. Completed
   slots are reset selectively only after their terminal outcome is committed.
2. Fresh selections in one parent batch must have distinct frozen catalog
   identities. Replay selections and independently sampled non-ACL selections
   retain the ScenarioNet duplicate contract.
3. TD3/SAC learning potential is computed from collected transitions carrying
   slot/episode provenance and aggregated per episode; replay-batch updates are
   diagnostics only and never ACL feedback. PPO carries slot/episode provenance
   through rollout storage and preserves partial episodes across chunks.
4. Simultaneous completions are committed in ascending
   `(collection_tick, worker_id)` order, independent of process arrival order.
5. Vector state is versioned and persists active selections, partial episode
   accumulators, parent RNG state, collection tick, and `n_envs`; incompatible
   or mismatched state is rejected.
6. On checkpoint resume, newly spawned ACL workers reset each persisted active
   slot with its persisted `reset_seed` before the first resumed step. The
   logical episode id, selection generation, scenario provenance, and parent
   accumulator remain unchanged; the simulator trajectory prefix is not
   reconstructed because MetaDrive/Rulebook state is not serialized by the
   learner checkpoint. The restart is emitted as an explicit event and is a
   documented limitation, not a silent observation-only continuation.
7. The generic non-ACL auto-reset protocol remains unchanged. ACL workers use
   `spawn` and the six frozen ScenarioNet arms A0--A5; mutation and dataset
   writes remain prohibited under ADR-014.

## Consequences

The ACL vector path has an explicit control-plane protocol and cannot silently
fall back to single-environment execution. It adds provenance to collector and
rollout data, while preserving aggregate learning-potential metrics for
diagnostics. A logical trace is reproducible for identical software, data,
configuration, seed, `num_envs`, and checkpoint state; cross-platform bitwise
identity is not promised.

## Approval Record

- Approved by: user.
- Approval evidence: explicit approval of DEC-VEC-001--DEC-VEC-005 on
  2026-07-20 and subsequent instruction to proceed with the recommended
  deterministic active-slot restart policy on 2026-07-20.
