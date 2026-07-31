# ADR-039: Label-Blind PG Holdout Replenishment

- Status: Approved
- Date: 2026-07-31
- Approval evidence: explicit user instruction “procedi” after review of the
  proposed equal-profile, eligibility-count-only replenishment policy.
- Affected specification: `SCENARIONET-INTEGRATION` v1.2 §3.3.

## Decision

When the declared PG holdout population leaves fewer than 450
Rulebook-eligible candidates, generate additional batches of 90 scenarios per
profile, using new disjoint seed windows. Rebuild and filter after each batch;
stop at feasibility or after five batches total.

The trigger is the total number of Rulebook-eligible PG holdout candidates.
It must not inspect or optimize source-by-arm counts. Each batch and observed
eligible count is recorded in the split manifest.

## Consequences

The empirical PG holdout preserves its frozen equal generation-profile mixture
and label-blind selection. Generation cost is bounded; a remaining shortfall
fails loudly rather than relaxing Rulebook eligibility or panel sizes.
