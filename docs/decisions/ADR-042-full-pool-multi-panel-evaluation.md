# ADR-042: Full-pool multi-panel evaluation and diagnostic isolation

- Status: Approved
- Date: 2026-07-31
- Approval evidence: explicit user request to implement the approved
  multi-panel ExecPlan in this conversation.
- Affected specifications: `SCENARIONET-INTEGRATION` v1.3 and
  `EVAL-PROTOCOL` v1.2.

## Decision

Use complete frozen pools for `default`, `medium`, `tune`, `long`, and
`thesis`; use only frozen profile-specific diagnostic subsets for `smoke` and
`fast`. Both validation panels run at every existing profile interval from one
shared snapshot. The final three panels execute serially from `final.zip`.

## Consequences

The data-selection hash remains distinct from panel and subset identity.
Diagnostic results cannot enter full-pool comparison aggregates. Queue
backpressure protects the required validation schedule at the cost of waiting
when an evaluation batch is outstanding.

