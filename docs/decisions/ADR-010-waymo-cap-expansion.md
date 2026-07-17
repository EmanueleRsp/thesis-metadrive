# ADR-010: Waymo Acquisition Cap Expansion

- Status: `Approved`
- Date: `2026-07-17`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-17`
- Supersedes: `NONE`
- Amends: ADR-007
- Affected specification: `docs/specifications/scenarionet_integration_spec_v1.1.md`, version `1.1`
- Affected ExecPlan: `docs/implementation/scenarionet_integration_spec_v1.1_exec_plan.md`

## Decision

Retain deterministic Waymo acquisition batches of 64 unseen shards and 16
conversion workers, and increase the cumulative maximum from 128 to 256 new
shards. The controller still acquires at most one batch per feasibility cycle,
rebuilds catalog and Rulebook artifacts, and stops immediately when strict
targets are feasible. The cap remains fail-closed; no filter or quota is
relaxed.

## Rationale

The prior 128-shard cap was reached while the strict split still lacked the
Waymo-only A4 population. Doubling the bounded cap gives the approved pipeline
two additional 64-shard cycles without changing source targets, grouping,
signal policy, Rulebook semantics, or arm formulas.

## Approval Record

- Approved by: user
- Approval evidence: explicit user instruction “raddoppia il cap waymo” on
  2026-07-17.
