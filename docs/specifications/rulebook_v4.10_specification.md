# Specification: Rulebook v2 R2/R3 cost-activation corrections

## Metadata

- Feature: `rulebook_v2_cost_activation_corrections`
- Specification ID: `rulebook-v2-r2-r3-cost-activation`
- Version: `4.10`
- Status: `APPROVED`
- Date: `2026-08-01`
- Supersedes: `docs/specifications/rulebook_v4.7_specification.md`, version
  `4.7-final-implementation-complete` (only for the subset of §6.2.1, §6.2.4,
  and §7.2.1 amended below, plus the new §7.3-bis; the rest of v4.7 remains
  authoritative and unchanged)
- Related specifications:
  - `docs/specifications/rulebook_v4.7_specification.md` (base for every
    section not touched by this version)
  - `docs/specifications/rulebook_v4.8_specification.md` (independent R2
    lateral-clearance amendment; no interaction)
  - `docs/specifications/rulebook_v4.9_specification.md` (independent R1
    injury-risk amendment; no interaction)
- Related ADRs: `docs/decisions/ADR-046-drivable-surface-seam-closing.md`,
  `ADR-047-ego-braking-physical-bound.md`,
  `ADR-048-rss-standstill-applicability.md`,
  `ADR-049-wrong-carriageway-subrule.md`
- Related ExecPlan:
  `docs/implementation/rulebook_v2_cost_activation_corrections_v1_exec_plan.md`
- Authoritative: YES

## 1. Purpose And Context

A user review of `RUN_PROFILE=medium` evaluation GIFs, followed by a full
audit
(`docs/audits/rulebook_v2_cost_activation_audit_2026-08-01/findings.md`),
confirmed four defects in the Rulebook v2 R2/R3 cost activation that this
version corrects:

1. the `offroad` reference surface (v4.7 §7.2.1) is not watertight across
   adjacent lane polygons, producing spurious sub-cm² off-road activations
   (F5a);
2. the ego RSS braking bound (v4.7 §6.2.4 step 10) discards a measured factor
   of 2.7 of ego braking capability and is inconsistent with the value
   already assumed for surrounding traffic (F3a);
3. longitudinal RSS (v4.7 §6.2.1) remains applicable at standstill, charging
   a stopped vehicle in a normal queue indefinitely with no cost-reducing
   action available (F3b/F5b);
4. no sub-rule measures occupancy of a lane whose legal direction opposes the
   assigned route (F1): `wrong_way` is, correctly, a signed-velocity rule
   (v4.7 §7.3.2) and `offroad`'s reference surface unions lanes regardless of
   legal direction (v4.7 §7.2.1) by design, so an ego driving forward in the
   oncoming lane of an empty road carries zero cost.

The scalarizer's binarisation of graded sub-rule costs at
`numerical_tolerance = 1e-8` (`bounded_satisfaction_rank`,
`rulebook_scalarization_v1.0_specification.md`) amplifies every defect above
but is a separate, deferred correction (out of scope of this version by
explicit user decision: "una cosa alla volta"). No tolerance introduced below
is sized or justified by that binarisation; each stands on its own geometric
or physical merits.

## 2. Amendment To §7.2.1 (Drivable Surface)

v4.7 §7.2.1 defines `C_drive` as the union of every vertically compatible
drivable lane polygon for the current step, with no post-processing. This
version adds: **the union is morphologically closed** with
`C_drive := union.buffer(+eps_close).buffer(-eps_close)`,
`eps_close = 0.10 m`. This is a numerical-tolerance addition, not a semantic
change to the reference surface: adjacent lane polygons are built
independently per source (per-point left/right widths on Waymo, block
geometry on PG, a centreline buffer in the fallback path) and their shared
edges do not coincide to floating-point accuracy, leaving hairline interior
holes in the union (measured: 681 holes of `1e-4..5e-4 m²` on one PG map).
The closing operator is monotone non-decreasing in area, so it can only add
surface — it can only lower an `offroad` cost, never raise one. `epsilon_A`
(the absolute area tolerance on the footprint/surface difference) is
unchanged at `1e-4 m²`. See `ADR-046`.

## 3. Amendment To §6.2.4 Step 10 (Ego Braking Calibration Bound)

v4.7 §6.2.4 step 10 fixes `b_e = min(4.0, floor(10 * b_meas) / 10)` m/s².
This version replaces the bound: `b_e = min(8.0, floor(10 * b_meas) / 10)`
m/s². `8.0 m/s²` is the dry-asphalt tyre-road deceleration limit and equal to
`b_i` (the braking already assumed in §6.2.1 for the identical-class front
vehicle). The quantile/rounding formula (`floor` of the lower 5th-percentile
braking sample, rounded to `0.1 m/s²`) is unchanged; only the upper bound
moves. A persisted calibration artifact records the bound (`cap_mps2`) and is
rejected if it does not match this constant, so an artifact produced under
the previous bound cannot be silently reused. See `ADR-047`.

## 4. Amendment To §6.2.1 (Longitudinal RSS Applicability)

v4.7 §6.2.1 defines the longitudinal RSS candidate set with no lower-speed
exclusion. This version adds an applicability clause: a candidate pair is
excluded from RSS evaluation when both `v_e <= 0.1 m/s` and
`v_i <= 0.1 m/s` (standstill). If every candidate in a step is excluded
this way, the `rss` component is `NOT_APPLICABLE` for that step; if some
candidates remain (e.g. a stopped leader alongside a moving actor), only the
standstill ones are excluded and the rest are evaluated per the unchanged
§6.2.1 formula. The response-time terms of `safe_distance_m` price an
acceleration the stopped ego is not performing; a stopped vehicle correctly
queued behind a stopped leader has no action available that reduces the
resulting cost. See `ADR-048`.

## 5. New §7.3-bis — Opposing-Carriageway Occupancy

A new R3 sub-rule `wrong_carriageway`, structurally analogous to `offroad`
(§7.2):

```
q_wrong_carriageway = A(P_e ∩ (C_opp \ C_aligned)) / A(P_e)
```

Over the same vertically compatible drivable lanes §7.2.1 uses:

- `C_aligned` is the union of lane polygons whose centreline tangent at the
  ego's projection satisfies `t_lane . t_route >= cos(60°)` against the
  canonical route tangent at the ego's route projection;
- `C_opp` is the union of those satisfying `t_lane . t_route <= -cos(60°)`;
- a lane in neither cone (a crossing branch inside a junction) contributes to
  neither surface.

`epsilon_A = 1e-4 m²` (the same absolute area tolerance as §7.2.2) applies to
the invaded-area computation before the ratio. The rule is memoryless: no
sub-rule state persists across steps, and no time ramp is applied — an ego
must be charged for occupying the opposing carriageway on the step it does
so, exactly as `offroad` charges off-road occupancy, with no grace period.

v1 charges this cost unconditionally, including when the separating marking
permits passing (e.g. a broken centreline legal for overtaking); the class of
the nearest separating marking is recorded as a diagnostic for a future
revision to consider suppressing the cost during a legal overtake, once that
diagnostic shows whether it is needed.

`wrong_carriageway` joins the `road_traffic_compliance` (R3) aggregation
group, evaluated by `aggregate_max_component` alongside every other R3
sub-rule per the unchanged v4.7 §7.1 max-aggregation rule. See `ADR-049`.

## 6. Compatibility

- The observation vector's dimension is unchanged by this version. Its
  *content* changes only via the separate, non-normative lane-marking
  adapter-coverage fix tracked in the ExecPlan (M3), not by anything in this
  specification.
- `RulebookResult.components` gains the key `wrong_carriageway`; consumers
  that iterate the mapping are unaffected, consumers that assume a fixed key
  set must be updated (the fixed Rulebook v2 component registry already
  enforces this at construction time).
- A calibration artifact produced before §3 of this version is rejected
  fail-closed on load; it must be regenerated.
- Episode returns on scenarios where `wrong_carriageway` activates, or where
  RSS standstill scoping or the off-road seam closing changes an existing
  cost, are not comparable with runs completed before this version.
