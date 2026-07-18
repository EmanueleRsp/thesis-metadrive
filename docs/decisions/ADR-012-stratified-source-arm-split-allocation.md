# ADR-012: Stratified Source-Arm Allocation Across Primary Splits

- Status: `Approved`
- Date: `2026-07-18`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-18`
- Amends: ADR-001, dataset split selection only
- Affected specification: `docs/specifications/scenarionet_integration_v1.1_specification.md`, version `1.1`
- Affected ExecPlan: `docs/implementation/scenarionet_integration_spec_v1.1_exec_plan.md`

## Context

The completed dataset satisfied the hard source totals and near-uniform arm
totals, but the greedy arm-to-split transport concentrated source-arm cells in
different primary splits. In particular, train could contain whole arms from
only PG or only Waymo even when both sources were selected for that arm across
the complete dataset. This made train, validation, and test composition
unrepresentative of the selected global source-arm policy.

## Decision

First determine the exact global quota for each `source × arm`, preserving the
approved exact source totals, near-uniform six-arm totals, hard eligibility,
group disjointness, and structural `A4_vru × PG = 0`. Then allocate each
source-arm quota across train, validation, and test using deterministic
minimum-cost transport. The allocation must preserve exact split source totals
and arm totals while minimizing the sum of absolute deviations from each
source-arm quota's proportional split distribution.

No filter, arm label, source total, group boundary, or scenario record may be
relaxed, changed, or duplicated to improve this objective.

## Consequences

The selected composition of every split now approximates the selected global
composition for each source-arm cell, subject only to integral rounding and
hard capacities. The global 50/50 Waymo-PG target remains exact; source balance
within an arm remains best effort and A4 remains Waymo-only. The implementation
uses an in-repository deterministic min-cost-flow construction and adds no
dependency.

## Approval Record

- Approved by: user
- Approval evidence: explicit user message in this Codex conversation accepting
  the option that preserves exact global 50/50 Waymo-PG composition.
