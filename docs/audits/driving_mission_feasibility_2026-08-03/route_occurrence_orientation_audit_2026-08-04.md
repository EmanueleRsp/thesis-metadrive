# Route occurrence orientation audit — 2026-08-04

## Scope and method

This was a read-only prototype audit over the frozen
`data/scenarionet/frozen/scenario_selection_index.json` and the aligned source
root `/scratch/e.respino/thesis-metadrive/data/scenarionet`. It processed all
3,500 records in 70 independent batches.

The frozen `assigned_route_lane_ids` sequence was never changed. For each
distinct occurrence, valid SDC poses were associated using the static lane
polygon and mandatory vertical-compatible projection onto that lane's source
centerline. The ordered source stations were classified as `FORWARD` or
`REVERSED` only when one signed direction had at least 60% of non-null changes.
No heading, planning, shortest path, native navigation, or runtime fallback was
used. The prototype then checked oriented endpoint connectivity and computed
the reset-to-terminal mission-local XY length.

These are audit results, not approval to exclude or replace records.

## Aggregate results

| Measure | PG | Waymo | Total |
|---|---:|---:|---:|
| Records | 1,695 | 1,805 | 3,500 |
| `FORWARD` occurrences | 8,114 | 4,181 | 12,295 |
| `REVERSED` occurrences | 0 | 2 | 2 |
| Positive connected mission-local routes | 1,694 | 1,765 | 3,459 |
| Reported cases | 1 | 40 | 41 |
| Missions with at least one reversed occurrence | 0 | 1 | 1 |

The two reversed occurrences are in Waymo records
`waymo:training_20s:6042e6c648fca15d` and
`waymo:training_20s:5e7bbc00b872c2ba`. The first also fails oriented endpoint
connectivity under this conservative prototype. The second is the target
record and is uniquely resolved as reversed.

## Target record

For `waymo:training_20s:5e7bbc00b872c2ba`:

- valid poses: `199`;
- assigned route: `("89",)`;
- candidate lanes at every valid pose: exactly `("89",)`;
- source station: `59.823227593 m` at reset and `5.376686225 m` at terminal;
- station changes: `187` negative, `0` positive, `11` zero;
- orientation: uniquely `REVERSED`;
- oriented reset station on the full source lane: `101.060441681 m`;
- oriented goal station on the full source lane: `155.428924978 m`;
- mission-local trimmed length: `54.368483298 m`;
- source lane XY length: `160.909795401 m`;
- alternative lane candidates: none.

The positive mission length is therefore the result of reversing the assigned
occurrence and trimming it between the causal reset and terminal goal. It is
not an absolute-difference fallback.

## Reported cases requiring resolution

The conservative audit reported the following 41 records:

```text
waymo:training_20s:6042e6c648fca15d       oriented_route_disconnected
waymo:training_20s:1e215c9f95a467b6       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:9473dddf2dfa0325       orientation_ambiguous_occurrence_4_no_progress
waymo:training_20s:daccf49e0efc62c4       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:d56c64cbee72ec33       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:f16f82e862814845       orientation_ambiguous_occurrence_2_no_progress
waymo:training_20s:b715653c21962e4b       orientation_ambiguous_occurrence_4_no_progress
pg:scenarionet_v1:PGMap-6000153            orientation_ambiguous_occurrence_3_no_progress
waymo:training_20s:32f4d4e13ef20671       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:9bf03f2582691a0a       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:93dc44d4cbd36da8       orientation_ambiguous_occurrence_2_no_progress
waymo:training_20s:d1879bfe0a6acd1f       orientation_ambiguous_occurrence_2_no_progress
waymo:training_20s:e7b275d7914ab8f4       orientation_ambiguous_occurrence_5_no_progress
waymo:training_20s:36be9a1cd684a47d       orientation_ambiguous_occurrence_3_no_progress
waymo:training_20s:b46e0017ffa72646       orientation_ambiguous_occurrence_2_no_progress
waymo:training_20s:23eb3a96bc381cb7       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:75d4410ee4904b7c       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:acd9e9a70bfae46d       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:f5ef710fe5f961ba       orientation_ambiguous_occurrence_1_no_progress
waymo:training_20s:b90fb93bd41291e8       orientation_ambiguous_occurrence_1_no_progress
waymo:training_20s:973b5bc2b7fbea7b       orientation_ambiguous_occurrence_2_no_progress
waymo:training_20s:ac3ff477d842a07d       orientation_ambiguous_occurrence_2_no_progress
waymo:training_20s:df0edcbc8e89bf30       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:809cf2a3b99b2757       orientation_ambiguous_occurrence_1_no_progress
waymo:training_20s:8222f13dc412e044       orientation_ambiguous_occurrence_3_no_progress
waymo:training_20s:93c7d419abe72a46       orientation_ambiguous_occurrence_8_no_progress
waymo:training_20s:e028754632ec8c1c       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:ca24a3a9146e34da       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:d2124381cf164080       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:939c9aee08126d2b       orientation_ambiguous_occurrence_1_no_progress
waymo:training_20s:9483d74b7bc523d6       orientation_ambiguous_occurrence_1_no_progress
waymo:training_20s:230f35ad15b63057       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:64057bf5c347fe65       orientation_ambiguous_occurrence_1_no_progress
waymo:training_20s:4fd490143c28dbf7       orientation_ambiguous_occurrence_1_no_progress
waymo:training_20s:ec498a05e75e057d       orientation_ambiguous_occurrence_4_no_progress
waymo:training_20s:706f68684eb7a3dc       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:46f1f23235dc0f5e       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:33cc6dd5f49c182d       orientation_ambiguous_occurrence_2_no_progress
waymo:training_20s:653f9899e3695df       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:aadb60ca3fc9cea6       orientation_ambiguous_occurrence_0_no_progress
waymo:training_20s:7935931452c917a9       orientation_ambiguous_occurrence_0_no_progress
```

The `no_progress` category means that the current static-polygon association
did not provide enough temporal station changes for that occurrence. It is not
evidence that the record is invalid; the next approved amendment work must
evaluate whether transition samples and sequence-constrained association
provide the missing evidence. No record was excluded.

## Legal-direction conflict status

The audit identified two mission occurrences whose mission-local orientation
is reversed relative to the source centerline point order. The current frozen
schemas do not expose an independent, authoritative legal-direction field for
these occurrences. Therefore the audit reports a source-centerline-direction
disagreement, not a definitive Rulebook legal conflict. A definitive conflict
classification requires inspection of the source-declared lane-direction
semantics and must not mutate Rulebook data.

## Required schema/runtime impact

The amendment requires `RouteOccurrence` serialization, oriented 3D geometry,
orientation provenance, source station bounds, source geometry hash, and a
mission-local canonical polyline with explicit reset and goal endpoints. The
current `route_offset_m`-based runtime path is not sufficient as the authority
representation. Tracker initialization, route samples, final-gate tangent,
mission hash, and dependent artifact generation must consume the trimmed
polyline directly. This schema/runtime impact was subsequently implemented
under the approved amendment.

## Global two-state audit after approval

The approved global resolver was then executed over all 3,500 records. It uses
two orientation states per occurrence, temporal net-station evidence when
available, end-to-start horizontal/vertical continuity, reset/goal anchor
membership, and the source orientation as the deterministic default when local
movement is insufficient.

Results:

- `3,499` records built successfully;
- `1,695` PG records built successfully;
- `1,804` Waymo records built successfully;
- `1` record has no globally valid orientation configuration;
- no record was excluded, replaced, or written back to source data.

The sole globally invalid record is
`waymo:training_20s:6042e6c648fca15d` (frozen-index position `75`). Its route
is `("184", "180", "182")`. The SDC evidence requires lane `184` to be
`REVERSED` (source station approximately `37.846 m -> 0 m`), while the only
end-to-start connected assignment requires all three source-oriented lanes to
be `FORWARD`. The intermediate lane `180` has no sufficient local movement, so
it is correctly resolved by the global continuity/default hierarchy; that
does not remove the contradiction on lane `184`. No orientation combination
satisfies both the temporal evidence and frozen geometric continuity.

This was initially treated as a concrete global-invalid-route decision. A
follow-up inspection found that the audit association was not temporal: it
re-scanned all route polygons for every pose. Because lane 184's polygon
overlaps the later junction geometry, late poses were incorrectly projected
back onto lane 184 and reversed its apparent station progression.

## Corrected sequence-constrained audit

The builder was corrected to associate poses monotonically with the frozen
occurrence sequence. It retains the current occurrence while it provides a
vertically compatible projection and advances only to a later occurrence; once
advanced, it never reassigns a pose to an earlier occurrence. This is an
offline association rule, not a route change or a runtime continuity fallback.

The corrected read-only sweep over all 3,500 records produced:

- `3,500/3,500` successful mission builds;
- `1,695` PG records and `1,805` Waymo records;
- `12,405` `FORWARD` occurrences and `1` `REVERSED` occurrence;
- `3,500/3,500` positive mission-local routes;
- `s_goal` range `9.493406 m` to `593.031749 m`, with zero non-positive values;
- zero orientation errors, zero gate-construction errors, and zero exclusions.

For `waymo:training_20s:6042e6c648fca15d`, the corrected temporal association
is:

| Occurrence | Associated source stations | Orientation |
|---|---:|---|
| lane `184` | `37.846 -> 41.196 m` | `FORWARD` |
| lane `180` | `0.077 -> 26.788 m` | `FORWARD` |
| lane `182` | `0.267 -> 74.179 m` | `FORWARD` |

The resulting mission-local route is connected and has `s_goal = 104.572425
m`. The scenario is therefore valid under the approved amendment and is not
excluded or replaced. The earlier `3,499/3,500` result was an audit bug, not a
source-data or mission-validity defect.
