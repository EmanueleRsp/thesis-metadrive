# ADR-041: Preserve Scarce Waymo A0 for Training

- Status: Approved
- Date: 2026-07-31
- Approval evidence: explicit user agreement with the recommendation to retain
  scarce Waymo A0 capacity for training after reviewing the real split result.
- Affected specification: `SCENARIONET-INTEGRATION` v1.2 §3.4, §3.5.

## Decision

After empirical holds are frozen, reserve at least 20
`A0_simple_low_traffic × Waymo` records for the train pool. The 300-record
stratified test remains near-uniform across arms (50 records per arm), but its
A0 source composition may deviate from the best-effort 50/50 preference to
honour this train reserve.

## Consequences

The empirical Waymo and PG pools remain label-blind and unchanged. The
stratified test remains an arm-balanced competence endpoint, while its A0
source composition is explicitly reported and no longer consumes virtually
the whole scarce Waymo A0 population. The train retains exact 1,100/1,100
source totals and the near-uniform total-arm allocation approved by ADR-040.
