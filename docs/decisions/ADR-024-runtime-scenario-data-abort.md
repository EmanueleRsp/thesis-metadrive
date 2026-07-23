# ADR-024: Typed Runtime Scenario Data-Abort and Run-Local Quarantine

- Status: APPROVED
- Date: 2026-07-23
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-23
- Supersedes: the runtime fail-fast policy for explicitly classified Rulebook scenario non-evaluability only; generic worker and software failures remain governed by ADR-019 and existing fail-fast contracts.
- Affected specifications: Rulebook v4.7, Transition Replay v1, Automatic Curriculum Learning v1, RL Baselines v1, ScenarioNet Integration v1.1, and Algorithm Comparison Protocol.
- Affected ExecPlan: `docs/implementation/runtime_scenario_data_abort_v1_exec_plan.md`

## Context

Offline Rulebook eligibility prevents known invalid scenarios from entering a frozen runtime view, but a live signal state or mapping can still make a scenario non-evaluable after the simulator advances. The former policy stopped the whole run. Converting arbitrary exceptions into an episode outcome would conceal software errors and is prohibited.

## Decision

Only `RuntimeScenarioNotEvaluableError`, with a stable reason code and original exception as its cause, is recoverable. Allowed codes are `UNKNOWN_SIGNAL_STATE`, `INVALID_SIGNAL_TRANSITION`, `MISSING_SIGNAL_MAPPING`, `INCOMPLETE_SIGNAL_TIMELINE`, and `UNRESOLVED_PHYSICAL_SIGNAL`. The current invalid pre/post signal-state error is translated to `INVALID_SIGNAL_TRANSITION`.

The failing step is discarded. If a preceding valid transition exists, it is retrospectively closed as `terminated=false`, `truncated=true`, `data_abort=true`, `truncation_reason=runtime_scenario_not_evaluable`, with the valid post-advance observation as `final_observation` for bootstrap. No synthetic reward may enter a learner, rollout, replay, or metric. A first-step abort has no transition.

The worker sends a distinct structured data-abort response and remains alive. The parent resets only that slot, quarantines the `scenario_uid` for the run, and preserves all other slots. Generic exceptions and worker failures remain fatal. Quarantine is a run artifact/checkpoint-resume state, never a mutation of `ScenarioDescription`, catalog, split, runtime database, fingerprint, or persisted eligibility.

ACL receives no LP, usefulness, MAB feedback, insert, or update for an aborted episode; an existing scenario-buffer record is removed. Evaluation excludes aborted episodes from policy aggregates while reporting attempted, valid, invalid, coverage, per-scenario results, and reason-code diagnostics. Final comparisons use the union exclusion set across compared methods/seeds.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Continue fail-fast | Simple and exposes defects | Loses valid data and stops unrelated slots | Superseded only for the explicitly typed data defect |
| Catch all worker errors | Keeps runs alive | Hides programming/numerical/learner defects | Violates fail-fast scientific controls |
| Insert failed step with zero reward | Simple collector path | Invents training data and biases learning | Prohibited |
| Mutate frozen dataset eligibility | Persistent filtering | Changes dataset/fingerprint and comparability | Prohibited |

## Consequences

Replay and PPO collectors need a slot-local retroactive boundary. N-step frontiers must flush at the boundary, while PPO GAE bootstraps from the valid final observation and never crosses the reset. Run artifacts gain quarantine and forensic JSONL records. Reporting discloses coverage and primary cross-method metrics use a common valid scenario set.

## Validation And Traceability

Mandatory regressions `TEST-RSA-001` through `TEST-RSA-011` in the linked ExecPlan cover first/later step, N-step, PPO/GAE, vector slots, ACL, resume, new run, evaluation, fatal untyped errors, forensic logging, and slot ordering.

## Approval Record

- Approved by: user
- Approval evidence: explicit approval and complete behavioral contract in the Codex conversation on 2026-07-23.
