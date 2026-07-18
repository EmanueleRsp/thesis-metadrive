# ADR-009: Targeted PG Replenishment By Observed Arm Deficits

- Status: `Approved`
- Date: `2026-07-17`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-17`
- Supersedes: `NONE`
- Amends: ADR-008
- Affected specification: `docs/specifications/scenarionet_integration_v1.1_specification.md`, version `1.1`
- Affected ExecPlan: `docs/implementation/scenarionet_integration_spec_v1.1_exec_plan.md`

## Context

Complete equal-size PG profile blocks can overproduce A0/A1 while leaving
complex PG arms scarce. The observed profile-to-arm pilot matrix is retained as
an empirical planning prior; it does not replace post-generation arm
classification.

## Decision

After a strict PG split-composition failure with zero raw runtime-count
shortfall, compute PG arm deficits from the replenishment report and allocate a
maximum candidate budget of `pg.replenishment_candidate_budget` (default `1750`)
across the profiles with the highest observed yield for those deficits. The
planner uses disjoint seeds and the existing PG generator, validation, feature
extraction, Rulebook filter, and final A0-A5 classifier. If no PG arm deficit is
reported, no additional PG is generated. The existing profile parameters remain
frozen; calibration is deferred.

The existing bound of
`pg.max_composition_replenishment_blocks` (default `2`) remains the maximum
number of targeted cycles per pipeline invocation. Each cycle rebuilds the
catalog, Rulebook artifact, and split attempt.

## Consequences

The pipeline avoids spending generation time on profiles that cannot address a
reported PG arm deficit, while preserving the scientific arm definitions and
strict runtime contract. Targeted allocation is a deterministic heuristic and
does not guarantee feasibility; the split solver remains authoritative and
must reject unresolved shortages.

## Approval Record

- Approved by: user
- Approval evidence: explicit user approval to split each 1,750-candidate
  replenishment cycle among missing arms and avoid an expensive parameter
  calibration pilot at this stage.
