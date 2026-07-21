# ADR-023: Scoped Vehicle-Yield Source Contract

- Status: APPROVED
- Date: 2026-07-21
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-21
- Supersedes: NONE
- Affected specification: `docs/specifications/rulebook_v4.7_specification.md`,
  version 4.7-final-implementation-complete, authoritative
- Affected ExecPlan:
  `docs/implementation/rulebook_synthetic_scenario_descriptors_v1_exec_plan.md`

## Context

The pure vehicle-yield evaluator was implemented, but production transition
wiring supplied a permanently empty domain.  Source geometry can derive
unambiguous lane movements and conflict zones, but cannot legally infer
pairwise priority or roundabout membership.

## Decision

1. Derive lane movements, conflict zones, occupancy, and STOP-versus-NONE
   priority causally from normalized source records and live snapshots.
2. Accept pairwise priority and roundabout entry/circulating relations only as
   explicit validated source metadata.
3. Treat malformed metadata and duplicate pairwise records as ineligible; do
   not add a fallback or infer priority from geometry.
4. Keep scenarios without a scoped predicate in the `NOT_APPLICABLE` domain.

## Consequences

The four predicates in specification §7.9 are representable in live
evaluation.  Future PG/Waymo ingestion that supplies explicit priority or
roundabout facts must populate `metadata.rulebook_vehicle_yield`; unannotated
ambiguous intersections remain outside the component domain.  No policy input,
reward-vector shape, or training-data selection rule changes.

## Approval Record

- Approved by: user
- Approval evidence: explicit instruction "procedi ad implementarla" in the
  Codex conversation on 2026-07-21, after the alternatives and source-contract
  consequences were explained.
