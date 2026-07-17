# ADR-005: Incremental ScenarioNet Candidate Replenishment

- Status: `Approved`
- Date: `2026-07-17`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-17`
- Supersedes: `NONE`
- Affected specifications: `docs/specifications/scenarionet_integration_spec_v1.1.md`, version `1.1`
- Affected ExecPlans: `docs/implementation/scenarionet_integration_spec_v1.1_exec_plan.md`

## Context

The strict v1.1 source targets require 1,750 runtime-eligible PG records, but
the first generated pool retained only 968 after quality, signal, and Rulebook
filters. Re-running the complete PG generation would waste deterministic seeds
and invalidate catalog/Rulebook caches. Additional candidates must preserve
profile semantics and seed disjointness while remaining auditable.

## Decision

1. Permit valid-scenario PG replenishment in addition to bounded Waymo shard
   expansion when the final eligible population is below the approved source
   targets.
2. Generate replenishment in complete configured profile blocks of 350
   candidates, preserving the existing five-profile distribution.
3. Start the first replenishment block at seed `5920000`, disjoint from the
   existing `920000` seed windows and their one-million profile stride.
4. Persist replenishment reports separately from the canonical PG pilot report;
   existing scenarios and reports are never overwritten by the replenishment
   command unless explicitly requested.
5. Rebuild the unified catalog and Rulebook audit after replenishment, then
   stop or continue based only on the strict runtime-eligible source/split
   contract. Do not relax quality, signal, Rulebook, or split constraints.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Generate exactly the reported deficit as candidates | Minimal nominal work | Does not account for invalid or Rulebook-excluded candidates | The report's deficit is a lower bound on successful eligible records |
| Regenerate the original PG seed windows | Simple command | Wastes work and invalidates reproducibility caches | Existing deterministic scenarios are preserved |
| Waymo-only expansion | Avoids changing PG population | Cannot satisfy the hard PG source target | Source targets are strict and source-specific |
| Generate one complete profile block | Preserves profile semantics and provides an auditable bounded step | May require another block if acceptance is unusually low | Approved first bounded replenishment step |

## Consequences

The pipeline gains a resumable PG replenishment target and separate audit
reports. The first block adds 1,750 candidates; only those passing the existing
quality, signal, and Rulebook policies count toward the final PG target. The
full source/split and arm diagnostics remain mandatory after every iteration.

## Approval Record

- Approved by: user
- Approval evidence: explicit user message in this Codex conversation:
  “vai va bene” in response to the recommendation to use Waymo expansion plus
  bounded PG replenishment in complete profile blocks.
