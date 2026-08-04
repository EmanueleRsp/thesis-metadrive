# Geometric Final-Gate Audit — 2026-08-04

Read-only prototype against the canonical 3,500-record index and frozen
ScenarioNet pickle root.

## Summary — initial polygon-consistency prototype

| Outcome | PG | Waymo | Total |
|---|---:|---:|---:|
| built | 1,695 | 1,801 | 3,496 |
| empty | 0 | 0 | 0 |
| final occurrence missing | 0 | 0 | 0 |
| ambiguous component selection | 0 | 4 | 4 |

Constants: geometry merge tolerance `0.01 m`; vertical compatibility tolerance
`3.0 m`. The builder uses lane polygons and static lane centerline tangents;
neighbor, boundary, and road identity metadata are not prerequisites.

## Ambiguous records

Each has two components containing the final-occurrence interval, so no segment
was selected silently:

| UID | merged components | discarded candidates: no-cross / opposite-perpendicular / vertical |
|---|---:|---:|
| `waymo:training_20s:dffbfc6327b84335` | 4 | 29 / 37 / 6 |
| `waymo:training_20s:b1128b1ac9f4e56a` | 2 | 30 / 30 / 7 |
| `waymo:training_20s:dd1ea48c3239f622` | 3 | 21 / 34 / 0 |
| `waymo:training_20s:fc680771abf4226f` | 8 | 111 / 129 / 0 |

These are geometric builder ambiguities, not invalid scenarios. No exclusion or
replacement is authorized.

## Candidate-level exclusion diagnostics

Across all records: `139,146` no cross-section intersections, `152,379`
opposite/perpendicular direction failures, `4,257` vertical incompatibilities,
and one null intersection. These are candidate-level counts, not record counts.

## Representative successful gates

- `waymo:training_20s:b2c31d815b05d8d2`: three-lane contiguous component,
  approximately `10.01 m`.
- `waymo:training_20s:215d94eecfef01aa`: singleton terminal component,
  approximately `6.16 m`.
- `waymo:training_20s:6c803c457cac865c`: three-lane contiguous component,
  approximately `11.03 m`.

The initial result is retained for comparison only. It is superseded by the
canonical-anchor result below.

## Final canonical-anchor result

The refined builder uses `g=r(s_goal)`, `x(u)=g+u*n`, and selects the unique
merged interval satisfying `a-epsilon <= 0 <= b+epsilon`. The final occurrence
polygon is consistency-only.

| Outcome | PG | Waymo | Total |
|---|---:|---:|---:|
| unique anchor-covering gate | 1,695 | 1,805 | 3,500 |
| zero anchor-covering components | 0 | 0 | 0 |
| multiple anchor-covering components | 0 | 0 | 0 |

The four former cases are now unique: `dffbfc6327b84335` (11.009 m),
`b1128b1ac9f4e56a` (4.281 m), `dd1ea48c3239f622` (7.484 m), and
`fc680771abf4226f` (6.913 m). Final-occurrence consistency passed for all
four. No metadata tie-break or record exclusion was used.

The audit is diagnostic and does not modify source data, production code, or
`project_index.md`.
