# ADR-006: Incremental ScenarioNet Feasibility Cycles

- Status: `Approved`
- Date: `2026-07-17`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-17`
- Supersedes: `NONE`
- Affected specifications: `docs/specifications/scenarionet_integration_v1.1_specification.md`, version `1.1`
- Affected ExecPlans: `docs/implementation/scenarionet_integration_spec_v1.1_exec_plan.md`

## Context

The feasibility loop previously regenerated PG scenarios and re-evaluated the
full static Rulebook catalog after each Waymo batch. This preserved correctness
but made resumable acquisition unnecessarily slow.

## Decision

Use resumable incremental cycles while retaining global validation:

1. Reuse existing deterministic PG seeds unless explicit PG overwrite is set.
2. Skip redundant Waymo database status scans when the parent cycle has already
   established the deficit and requests exactly one batch; the next catalog
   cycle performs the authoritative status check.
3. Cache Rulebook eligibility by scenario UID, scenario file size/mtime
   fingerprint, geometry-config hash, calibration hash, and Rulebook schema.
4. Evaluate new or changed records with the configured worker pool, merge cached
   and new audit records, and rerun the complete source/split/arm contract.
5. Invalidate the cache on any fingerprint, policy, geometry, calibration, or
   schema mismatch; never reuse a stale or unverified result.

## Consequences

The expensive work is proportional to new or changed scenarios after the first
compatible cache artifact. Final population accounting and leakage checks remain
global, so incremental execution cannot weaken scientific acceptance behavior.

## Approval Record

- Approved by: user
- Approval evidence: explicit user instruction to proceed with the incremental
  implementation while maintaining correctness.
