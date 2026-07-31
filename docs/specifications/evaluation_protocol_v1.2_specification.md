# Evaluation protocol amendment: full-pool multi-panel execution

## Metadata

- Specification ID: `EVAL-PROTOCOL`
- Version: `1.2`
- Status: `APPROVED`
- Date: `2026-07-31`
- Supersedes: version `1.1` only for named-panel size, validation timing, and
  panel execution/reporting rules. Every unamended version `1.0`/`1.1`
  requirement remains authoritative, including `final.zip` as the sole
  official checkpoint.
- Related specification: `docs/specifications/scenarionet_integration_v1.3_specification.md`
- Related ADR: `ADR-042`
- Authoritative: `YES`

### Approval record

The user explicitly approved this amendment by requesting implementation of
the “ScenarioNet multi-panel evaluation, full-pool protocol, and terminal
monitor” ExecPlan on 2026-07-31.

## Amended requirements

### REQ-004 — frozen panel plan

ScenarioNet official cardinality is resolved from the frozen evaluation plan,
never from `eval_episodes` or `final_eval_episodes`. The five panel names,
cardinalities, roles, subset identities, and scopes are exactly those in
`SCENARIONET-INTEGRATION` v1.3. Legacy scalar evaluation remains available to
non-ScenarioNet environments.

### REQ-007 and REQ-013 — reporting and validation

At every profile `eval_interval`, both validation panels are admitted as one
evaluation batch from one immutable checkpoint snapshot and global step. Jobs
are FIFO. When the configured one-batch queue capacity is reached, training
waits; a required boundary may never be dropped, coalesced, or silently
delayed. The two panel results are reported separately. Learning curves,
tables, plots, aggregate metrics, per-episode data, rule/subrule data,
data-aborts, videos, trajectories, and events retain their existing metric
families for every panel.

### REQ-006 and final-test execution

After training, exactly three final evaluations execute serially from
`checkpoints/final.zip`: Waymo empirical, PG, and arm-stratified. Internal
parallelism through `test_workers` remains permitted for each panel. One
complete final result is written per panel.

### REQ-011 — comparison isolation

Analysis requires the expected named panel set for the declared profile scope
and rejects mixed protocol versions, frozen selection hashes, panel hashes, or
evaluation scopes. The primary status is only full-pool
`test_waymo_empirical`; PG remains secondary and arm-level claims use only
`test_arm_stratified`. Diagnostic `smoke` and `fast` outputs remain
inspectable but are excluded from official comparison blocks.

### Terminal monitor

Interactive terminals use one Rich live renderer showing learner progress,
current batch/snapshot, each panel’s queued/running/completed state and
progress, and completed per-panel sample mean ± sample standard deviation for
success, collision, and route completion. Non-interactive output remains
complete and machine-readable.

