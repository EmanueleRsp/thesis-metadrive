# ADR-040: V1.2 Balanced Train Selection

- Status: Approved
- Date: 2026-07-31
- Approval evidence: explicit user confirmation that the v1.1 train result was
  satisfactory, after the distinction between a residual-only train and the
  prior balanced allocation was explained in this conversation.
- Affected specification: `SCENARIONET-INTEGRATION` v1.2 §3.1, §3.4, §3.5.

## Decision

The v1.2 train pool contains exactly 1,100 Waymo and 1,100 PG scenarios. It
uses the v1.1 `balanced_arm_source` selection policy after the empirical and
stratified holdouts have been frozen: total arm counts are near-uniform,
source allocation within an arm prefers 50/50, and structural empty cells are
compensated in other cells while the exact train source totals remain hard.

## Consequences

The final v1.2 selected dataset again contains 3,500 scenarios: 2,200 train,
300 validation, and 1,000 test. The empirical holdout reservation remains
label-blind and unchanged; only the residual-to-train selection is revised.
This supersedes the residual-only/minimum-per-arm train rule introduced by
ADR-037 for the v1.2 train pool.
