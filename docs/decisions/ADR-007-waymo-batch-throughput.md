# ADR-007: Waymo Acquisition Batch Throughput

- Status: `Approved`
- Date: `2026-07-17`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-17`
- Supersedes: `NONE`
- Amends: ADR-001, decision 5, only for the batch-size parameter
- Affected specification: `docs/specifications/scenarionet_integration_spec_v1.1.md`, version `1.1`
- Affected ExecPlan: `docs/implementation/scenarionet_integration_spec_v1.1_exec_plan.md`

## Context

The post-Rulebook feasibility controller rebuilds the catalog and eligibility
artifacts after each acquisition cycle. The approved policy used 16 unseen
Waymo shards per cycle and a cumulative cap of 128 new shards. The resulting
download/conversion work is correct but incurs avoidable catalog and Rulebook
cycle overhead when a larger bounded batch can be processed safely.

## Decision

Use 64 unseen Waymo shards per acquisition cycle, retain the cumulative cap of
128 newly processed shards, and retain 16 conversion workers. The controller
continues to acquire at most one batch per feasibility cycle, then rebuilds the
catalog, reuses compatible Rulebook results, and rechecks the strict split
contract. Existing converted shards remain write-once and are never downloaded
again.

## Consequences

At most two 64-shard acquisition cycles are available under the existing cap,
reducing orchestration overhead while preserving post-Rulebook feasibility
checks and fail-closed eligibility. A single conversion cycle uses more disk
space and can run longer than the former 16-shard cycle; the persistent
converter and 16-worker setting remain unchanged. No source targets, grouping,
signal policy, Rulebook semantics, or split criteria change.

## Approval Record

- Approved by: user
- Approval evidence: explicit user message “procedi” immediately after the
  proposed `batch_shards: 64` change and its policy implications were stated.
