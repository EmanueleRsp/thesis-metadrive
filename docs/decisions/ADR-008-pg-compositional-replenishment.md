# ADR-008: PG Compositional Replenishment After Split Infeasibility

- Status: `Approved; amended by ADR-009`
- Date: `2026-07-17`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-17`
- Supersedes: `NONE`
- Amends: ADR-005, extending the trigger condition only
- Amended by: ADR-009, replacing equal complete blocks with targeted profile allocation when arm deficits are known
- Affected specification: `docs/specifications/scenarionet_integration_v1.1_specification.md`, version `1.1`
- Affected ExecPlan: `docs/implementation/scenarionet_integration_spec_v1.1_exec_plan.md`

## Context

The PG pool can contain enough total runtime-eligible records while remaining
infeasible for the strict source/arm/group split contract. Previously the
controller replenished PG only for a raw runtime-count shortfall, then stopped
or attempted unrelated Waymo expansion when the failure was specifically
`pg/train` composition.

## Decision

When the split report identifies a PG source-target infeasibility and the PG
runtime-count shortfall is zero, automatically generate up to the configured
`pg.max_composition_replenishment_blocks` complete PG profile blocks
(`count_per_profile=350`) before another feasibility cycle. The approved
default is `2`. The seed base is computed as the next disjoint one-million
profile-stride block after the highest existing PG seed. Rebuild catalog,
Rulebook, and splits after each block. If the composition failure persists
after the configured limit, stop with the diagnostics; do not loop, relax
constraints, or acquire more Waymo solely for a PG failure.

## Consequences

The controller can repair compositional PG scarcity without requiring a manual
restart, while preserving fixed profile semantics, seed disjointness, strict
Rulebook/signal policy, and the source targets. At most the configured number
of extra blocks is generated per pipeline invocation, so unresolved
infeasibility remains explicit and bounded.

## Approval Record

- Approved by: user
- Approval evidence: explicit user message “procedi” after the recommendation
  to diagnose the source/arm/group infeasibility and activate an additional
  complete PG block only when the diagnostic confirms it.
