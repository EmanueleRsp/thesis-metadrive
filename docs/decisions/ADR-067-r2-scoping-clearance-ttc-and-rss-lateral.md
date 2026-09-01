# ADR-067: `clearance` is scoped to the roadway, `ttc` moves to 0.95 s, `rss_lateral` is left unchanged

- Status: **Approved** — carried by `RULEBOOK-V5.1`, approved 2026-08-14
- Date: 2026-08-10
- Approval evidence: pending; carried by `rulebook_v5.0_UNDER_REVIEW`.
- Affected specification: `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
  §5.2, §5.3 (amends `rulebook_v4.7_specification.md` §6.4 and the v4.8 scoped
  lateral-RSS metric).

## Context

Blame attribution over the expert's residual penalty, in reward units, named two
sub-rules as the dominant remaining cost after `rss` was removed:

| sub-rule | share of residual penalty | dominant in negative episodes |
|---|---:|---:|
| `rss_lateral` | 36.2 % | 44 / 116 |
| `clearance` | 30.5 % | 26 / 116 |

This table is the evidence that motivated the scoping investigated below, and it
is measured on the `proposed` counterfactual — the rulebook without
`dashed_line`, without `speed_limit` and with the original TTC thresholds. **It
is not the attribution of the rulebook this ADR helps specify.** That one is
measured separately in RULEBOOK-V5.0 §4.10, where the same two sub-rules own
49 of 97 below-standstill episodes rather than the majority this table suggests,
and where a comparably sized lane-marking cluster appears that cannot be seen
here because `dashed_line` is absent from `proposed`. Carrying this table over to
the specified rulebook would be an extrapolation across four sub-rules.

Both hypotheses were then tested against the data.

**`clearance`** charges any pedestrian or cyclist within 1.0 m of the ego
footprint, with **no test of where that VRU is standing**. Scoping to VRUs whose
footprint centroid lies on the drivable surface: **0.327 % -> 0.199 %** of steps.

**`rss_lateral`** adds the neighbour's worst-case inward displacement to the
ego's when computing `d_safe^lat`. Removing the neighbour's ability to raise the
ego's requirement: **0.312 % -> 0.267 %**, a 14 % reduction.

**`ttc`** uses 0.8 s for vehicles and 1.0 s for VRU. Swept uniformly:

| threshold | 0.4 s | 0.6 s | 0.8 s | 0.95 s | 1.2 s |
|---|---:|---:|---:|---:|---:|
| % steps | 0.056 | 0.077 | 0.118 | **0.156** | 0.258 |

## Decision

- `clearance` is scoped: only pedestrians and cyclists whose footprint centroid
  lies on the drivable surface are candidates. Threshold and graded cost
  unchanged.
- `ttc` uses **0.95 s uniformly** across actor classes.
- `rss_lateral` is left **unchanged**.

## Rationale

**`clearance`.** A person waiting on the kerb of a narrow street is inside 1 m
of every passing vehicle and is in conflict with none of them. The unscoped rule
priced normal urban driving. The centre-entry criterion is the same one that
took `wrong_carriageway` to zero expert violations, so the rulebook applies one
geometric convention rather than two.

**`ttc`.** 0.95 s is nuPlan's `least_min_ttc` bound. Adopting it costs 0.04
percentage points of expert violation and replaces a repository-chosen pair of
values with a published one. At every swept threshold `ttc` remains an order of
magnitude below the other sub-rules, so it needs no tolerance of its own.

**`rss_lateral` is deliberately not changed.** The scoped variant is
implementable in two lines, needs no new input, and is a strict relaxation
verified by `TEST-RSEC-017`. But the hypothesis that motivated it - that the
cost came from the neighbour's motion - is falsified by its own measurement: at
a 14 % reduction, the cost comes almost entirely from the ego's own lateral
motion, which it is legitimate to charge. One fewer deviation to justify, for a
benefit that does not appear in the data.

A related implementation defect was found while building the variant and is
recorded here so it is not reintroduced: simply zeroing the neighbour's inward
speed is **not** a relaxation, because that displacement term goes negative for
a receding neighbour and therefore acts as a credit. The scoped form must take
the minimum of the full and the zero-neighbour safe distances.

## Consequences

Recorded in `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md`
and in `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`.
This ADR is not an implementation authorisation on its own: the specification
must be promoted to `APPROVED` first.
