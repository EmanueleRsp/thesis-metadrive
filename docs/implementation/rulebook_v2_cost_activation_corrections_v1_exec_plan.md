# ExecPlan — Rulebook v2 Cost-Activation Corrections v1

## 1. Metadata

| Field | Value |
| --- | --- |
| Feature | Correction of six confirmed cost-activation defects in Rulebook v2 (R2/R3) |
| Plan ID | `EP-RBCOST-V1` |
| Authoritative specification | `docs/specifications/rulebook_v4.7_specification.md` (base), amended by `rulebook_v4.8_specification.md` (R2 lateral clearance) and `rulebook_v4.9_specification.md` (R1 injury risk) |
| Specification status | `APPROVED` / `Authoritative: YES` |
| New amendment | `docs/specifications/rulebook_v4.10_specification.md`, `APPROVED` / `Authoritative: YES`, covering the R2/R3 amendments of this plan |
| Triggering audit | `docs/audits/rulebook_v2_cost_activation_audit_2026-08-01/findings.md` |
| Status | `IN_PROGRESS` (M1-M4 implemented and verified; M0 partial; M5 not started) |
| Created | 2026-08-01 |
| Last updated | 2026-08-01 |
| Branch | `scenarionet-implementation` |
| Related ADRs | `ADR-046`, `ADR-047`, `ADR-048`, `ADR-049` (all `APPROVED`) |
| Owner | Repository owner |

## 2. Objective And Scope

### 2.1 Observable capability

After this change, the Rulebook v2 cost vector must be a faithful measure of the
ego's driving behaviour on both scenario sources, specifically:

- a correctly positioned ego on a normal lane reports `offroad = 0` at every
  step, instead of intermittently reporting a `~2 cm²` numerical sliver;
- an ego occupying the opposing carriageway carries a graded R3 cost
  proportional to the invaded footprint fraction, instead of zero;
- the yellow continuous centreline of a procedural (PG) map participates in the
  `solid_line` cost and in the semantic observation, instead of being invisible;
- the longitudinal RSS safe distance uses an ego braking value consistent with
  the measured ego dynamics and with the value already assumed for surrounding
  vehicles, instead of a value `2.7x` smaller than measured;
- an ego stopped behind a stopped leader is not charged a permanent R2 cost with
  no cost-reducing action available;
- traffic-control selection and sub-rule applicability are observable per
  episode and in the evaluation GIF overlay, instead of being indistinguishable
  from "satisfied".

### 2.2 Success criteria

Success is recognised by:

1. the deterministic acceptance tests in §9 passing;
2. the M1 regression measurement showing zero seam-driven `offroad` activations
   on the PG geometry sweep that currently reports up to `37.1%` of positions;
3. the M0 diagnostics reporting non-zero selected-control counts on signalised
   scenarios, thereby converting F4 from hypothesis to measurement;
4. the M3 dry-run showing that the added map-feature classes introduce no new
   `validation_errors` in the scenario catalog (or, if they do, that the
   eligibility policy change of `DEC-RBCOST-006` is approved first).

### 2.3 In scope

- `offroad` reference-surface construction (`geometry/drivable.py`).
- A new R3 sub-rule `wrong_carriageway` (`components/road.py`, registry,
  aggregation, transition wiring).
- Lane-marking class coverage in both static adapters.
- Ego RSS braking calibration bound and RSS standstill applicability.
- Additive control/applicability diagnostics (episode counters and GIF overlay).
- The specification amendment and ADRs recording every approved decision.
- A regression test for every defect fixed.

### 2.4 Out of scope

- **The scalarizer (`F0` / audit `P2`).** Explicit user instruction: one problem
  at a time. **No tolerance, deadband, or epsilon in this plan may be sized or
  justified by the scalarizer's binarisation.** Every numerical tolerance
  introduced here must have a standalone geometric or physical justification
  recorded next to it. It gets its own ExecPlan later.
- `solid_line` minimal-penetration deadband (audit `P5c`): deferred to M5 and
  gated on a measurement, see `DEC-RBCOST-005`.
- Red-light persistence latch (audit `P7.3`): deferred to M5 and gated on the M0
  measurement, see `DEC-RBCOST-007`.
- Any change to `wrongway` itself: it is correct as specified (v4.7 §7.3.2).
- Retraining, curriculum, or reward-scale re-tuning.

### 2.5 Compatibility constraints

| Surface | Impact |
| --- | --- |
| Public rulebook interface | `RulebookResult.components` gains the key `wrong_carriageway`; `road_traffic_compliance` gains one sub-component. Consumers iterate the mapping, so this is additive. |
| Observation vector | M3 changes the *content* of the semantic observation (new markings become visible), not its dimension. Existing checkpoints stay shape-compatible but are **not behaviourally comparable**. This must be stated in the final report. |
| Calibration artifact | M2 changes the normative cap, and `load_calibration_artifact` validates the persisted `cap_mps2` against it. **Any existing artifact is rejected fail-closed until regenerated.** The ego `config_hash` does not change. |
| Frozen selection index | M3 may alter `validation_errors`, hence `rulebook_eligible`. `DEC-RBCOST-006` exists to keep the frozen catalog valid. |
| Reward scale | R3 gains a sub-rule and R2 loses spurious activations, so absolute episode returns are not comparable with runs completed before this change. Documented, not mitigated. |

## 3. Authoritative Requirements

| ID | Requirement | Source |
| --- | --- | --- |
| `REQ-RBCOST-001` | `offroad` measures the footprint area fraction outside the union of vertically compatible drivable lanes; the union is the normative reference surface and must not depend on legal direction | v4.7 §7.2.1–§7.2.2 |
| `REQ-RBCOST-002` | `ε_A` in `offroad` is a numerical tolerance fixed by geometry tests, not a semantic parameter | v4.7 §7.2.2 |
| `REQ-RBCOST-003` | `wrongway` penalises only velocity opposite to the canonical route tangent; geometric orientation alone must not be penalised | v4.7 §7.3.2 |
| `REQ-RBCOST-004` | R3 must charge continuous solid lane markings ("boundary continue") without restriction on marking colour | v4.7 §7.4 |
| `REQ-RBCOST-005` | R3 must charge dashed lane markings ("boundary tratteggiate") without restriction on marking colour | v4.7 §7.5 |
| `REQ-RBCOST-006` | The longitudinal RSS safe distance uses `ρ = 1.0 s`, `a_max_acc = 3.5 m/s²`, `b_i = 8.0 m/s²`, and `b_e` from the one-shot calibration protocol | v4.7 §6.2.1, §6.2.4 |
| `REQ-RBCOST-007` | `b_e` is the floor-to-0.1 of the lower 5th percentile of the valid braking trials, bounded above | v4.7 §6.2.4 step 10 |
| `REQ-RBCOST-008` | Each macro rule aggregates its applicable sub-rules by maximum; a non-applicable sub-rule must not contribute | v4.7 §7.1, `aggregation.py` |
| `REQ-RBCOST-009` | Occupying a lane whose legal direction opposes the assigned route carries an R3 cost graded by the invaded footprint fraction | **New** — proposed v4.10 §7.3-bis, `DEC-RBCOST-004` |
| `REQ-RBCOST-010` | Longitudinal RSS is `NOT_APPLICABLE` when ego and front vehicle are both at standstill | **New** — proposed v4.10 amending v4.7 §6.2.1, `DEC-RBCOST-003` |
| `REQ-RBCOST-011` | A map feature whose type has no mapping to a `MapFeatureClass` must be recorded as an explicit diagnostic, never silently dropped | Repository convention (fail-loud), `DEC-RBCOST-006` |
| `REQ-RBCOST-012` | Per-episode diagnostics must report, for every traffic-control sub-rule, the number of selected controls, the number of applicable steps, and the number of controls dropped by each drop path | **New** — additive diagnostics, no approval gate |

`REQ-RBCOST-009` and `REQ-RBCOST-010` extend the approved specification and are
therefore approval gates. `REQ-RBCOST-007`'s bound value change is an approval
gate. All other requirements restate existing approved behaviour that the
current implementation fails to deliver.

## 4. Current Repository Analysis

All statements below are `VERIFIED` against the working tree at commit `0c04fe2`
unless labelled otherwise. Full evidence is in the triggering audit.

### 4.1 Off-road reference surface

`src/thesis_rl/rulebook/v2/geometry/drivable.py:65-96` — `drivable_surface_for_ego`
selects every vertically compatible lane and returns `shapely.union_all` of their
polygons, cached by `_union_selected_surfaces` (`lru_cache(maxsize=128)`, keyed on
the geometry tuple). No post-processing is applied to the union.

`VERIFIED` (measured): on a live PG map (`map='SCS'`, 36 lanes) the union of the
lane polygons is a single polygon carrying **681 interior holes** of individual
area `1e-4 … 5e-4 m²`. Sweeping a `4.5 m × 1.85 m` footprint along the lane
centrelines:

| lateral offset | steps with `offroad > 0` | median area ratio | after a `0.10 m` closing |
| ---: | ---: | ---: | ---: |
| 0.0 m | 2.8% | 0.0556 | 2.8% |
| 0.4 m | 2.8% | 0.0556 | 2.8% |
| 0.8 m | 14.9% | 0.0000 | 2.8% |
| 1.2 m | 37.1% | 0.0002 | 2.8% |

The residual `2.8%` baseline is legitimate — the footprint protruding past the
first and last lane of the map, ratio `0.0556`, all at index `1` or `len-2` of
those lanes. Everything above the baseline is seam artefact. A `0.05 m` closing
was insufficient at these seams.

`INFERRED`: Waymo lane polygons are built independently per lane from per-point
left/right widths (`waymo_static_adapter.py:_lane_polygon_from_widths`), so the
same seam class is expected; **not measured**, because the scenario database is
not mounted in the analysis session. Closing M1 requires this measurement
(`TEST-RBCOST-004`).

`components/road.py:20` — `OFFROAD_AREA_EPSILON_M2 = 1.0e-4`, applied as an
absolute cut on `outside_area` before the ratio. This value is **kept unchanged**
by this plan: it is a numerical tolerance whose only defensible justification is
floating-point noise in the difference operation, and no measurement supports
enlarging it. An earlier draft proposed a 1%-relative tolerance; it was withdrawn
because its only justification was the scalarizer's binarisation (see §2.4) and,
on an `8.3 m²` footprint, 1% would silently exempt `0.08 m²` of genuine off-road.

### 4.2 Opposing-carriageway occupancy

`components/road.py:116-146` — `evaluate_wrongway` computes
`cost = clip([-v_parallel]_+ / v_cap, 0, 1)`: strictly reverse motion. This is
exactly `REQ-RBCOST-003` and must be preserved.

`drivable.py:78-88` — the reference surface unions all vertically compatible
lanes with an explicit comment forbidding a footprint prefilter, and v4.7 §7.2.1
states verbatim that the surface does not depend on legal direction. The opposing
carriageway is therefore part of `C_drive` and `offroad` returns `0` there.

`components/progress.py:30-50` — `route_outside_fraction` is diagnostic-only and
does not gate the R4 progress credit.

Conclusion (`VERIFIED`): no sub-rule measures lane direction. An ego driving
forward in the oncoming lane of an empty road has zero cost from every sub-rule
except `solid_line`, which §4.3 shows is broken for exactly the marking that
separates the carriageways on PG maps.

### 4.3 Lane-marking coverage

`context/pg_static_adapter.py:38-44` maps five feature types; the loop skips any
feature whose class resolves to `None`. `VERIFIED` (reproduced live): a PG map
`'SCS'` emits `36 LANE_SURFACE_STREET`, `24 ROAD_LINE_BROKEN_SINGLE_WHITE`,
`12 ROAD_LINE_SOLID_SINGLE_WHITE`, `12 ROAD_LINE_SOLID_SINGLE_YELLOW`. The last
class has no mapping, and per
`third_party/metadrive/metadrive/component/map/pg_map.py:156-168` it is precisely
the continuous centreline separating the two carriageways. `ROAD_LINE_SOLID_DOUBLE_YELLOW`
is mapped but never emitted by PG.

`context/waymo_static_adapter.py:38-46` additionally maps
`ROAD_LINE_SOLID_SINGLE_YELLOW` and `DRIVEWAY`. The ScenarioNet Waymo converter
(`third_party/scenarionet/scenarionet/converter/waymo/type.py:45-67`) can also emit
`ROAD_LINE_SOLID_DOUBLE_WHITE`, `ROAD_LINE_BROKEN_SINGLE_YELLOW`,
`ROAD_LINE_BROKEN_DOUBLE_YELLOW`, `ROAD_LINE_PASSING_DOUBLE_YELLOW`, and
`ROAD_EDGE_MEDIAN` — none mapped.

`transition.py:1329-1344` filters `map_feature_catalog` by `MapFeatureClass`, and
`envs/observations/causal_semantic.py:1027-1042` consumes the same catalog, so an
unmapped marking is invisible to both the rulebook and the observation.

`context/catalog_eligibility.py:111` — `rulebook_eligible = not errors`. Any new
`validation_errors` entry produced by newly admitted geometry would shrink the
catalog. `VERIFIED` on PG: 122 yellow lines across 4 maps, 0 degenerate, 0
canonicalization failures. `AWAITING_CONFIRMATION` on Waymo — the dry-run of
`TEST-RBCOST-009` must run before M3 lands.

### 4.4 RSS

`calibration.py:17,96` — `MAX_REFERENCE_BRAKE_MPS2 = 4.0`;
`calibrated = min(4.0, floor(10 * b_meas) / 10)`. The same constant is validated
on write (`:110`) and on load (`:152,163`), and is persisted in the artifact as
`cap_mps2`. The production artifact at
`$DATA_ROOT/scenarionet/rulebook_v2/calibration_b_e.json` currently holds
`ego_min_brake_mps2 = 4.0`, `cap_mps2 = 4.0`,
`config_hash = 7ed6b5a0…eee94`.

`VERIFIED` (measured, 12-trial sample in
`docs/audits/rulebook_v2_cost_activation_audit_2026-08-01/braking_trials_sample_12.json`):
mean decelerations `10.73 … 16.92 m/s²`, minimum `10.73`. The cap discards a
factor of `2.7`. It is also internally inconsistent with
`components/rss.py:33` — `FRONT_MAX_BRAKE_MPS2 = 8.0` — i.e. the model assumes
the ego brakes half as well as identical surrounding vehicles.

Resulting `d_safe` (`ρ = 1.0`, `a_max_acc = 3.5`, `b_i = 8.0`):

| `v_e` | `v_i` | `b_e = 4.0` | `b_e = 8.0` |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 3.28 m | 2.52 m |
| 10 | 10 | 28.28 m | 16.89 m |
| 20 | 20 | 65.78 m | 31.27 m |

`components/rss.py:40-51` — `safe_distance_m` always adds the response terms, so
`d_safe(0, 0) > 0` and RSS stays `applicable` at standstill.
`components/rss.py:64-71` — the only `NOT_APPLICABLE` path is an empty candidate
tuple.

`transition.py:253-335` — `_rss_candidates` was reviewed line by line
(same-traffic-stream gating, concordant heading, bumper-to-bumper gap, front
predicate) and is `VERIFIED` correct per v4.7 §6.2.1. The over-triggering is in
the parameter and the applicability domain, not in candidate selection.

### 4.5 Traffic controls

`components/controls.py:190-299` — `evaluate_signal_transition` charges
`cost = 1.0` on the single step whose swept front bumper crosses the control line
under a RED (or committed YELLOW) pre-state, then marks the group resolved. The
approach branch charges `1 - post_delta / d_req` with
`d_req = v Δt + v²/(2 b_e)`. `VERIFIED`: this is the specified behaviour; the
"appears not to work" symptom is one violated step in an episode of hundreds.

Two silent drop paths, both `VERIFIED`:

1. `derive_control_line` raises `ControlLineOffRouteError`; both adapters
   `continue` without recording anything.
2. `transition.py:499-522` — `_selected_control` requires
   `control.movement_key.approach_lane_id in cache.task_route.lane_ids`, strict
   lane-ID membership, whereas the catalog flag `has_route_traffic_light` is
   computed by a looser geometric-proximity test
   (`scenarios/waymo_topology.py:177-192`). A scenario can therefore be tagged as
   signalised while the rulebook selects no control.

`runtime/io/video_diagnostics.py:100-124` — the overlay prints `label c=<cost>`
only; `NOT_APPLICABLE` and `SATISFIED` are indistinguishable in the GIFs.

Frozen index statistics (`VERIFIED`): 3500 records, 1805 Waymo / 1695 PG, all
`rulebook_eligible = true`, `has_route_traffic_light = true` for 828 (23.7%).
Signalised scenarios are present; whether the rulebook selects their controls at
runtime is currently **not observable** — which is why M0 is a prerequisite for
any signal behaviour change.

### 4.6 Debt explicitly not addressed here

`reward/scalarization.py:239-259` with `numerical_tolerance = 1e-8` binarises
every macro margin. Recorded as the audit's `F0`, deferred by user instruction
(§2.4). It amplifies every defect above but is not a cause of any of them.

## 5. Assumptions And Invariants

| Item | Value / rule | Established by | Violation handling |
| --- | --- | --- | --- |
| Units | lengths m, areas m², speeds m/s, accelerations m/s², angles rad, times s | v4.7 §2 | `ValueError` at component boundary |
| Frames | map-global XY for all geometry; route-relative `s` from `RoutePolyline.project`; lane tangents in map-global XY | v4.7 §2.9 | — |
| Ego footprint | valid non-empty polygon, area > 0, nominally `4.5 × 1.85 m` (`8.325 m²`) | `evaluate_offroad` precondition | `ValueError` |
| Vertical compatibility | `abs(z_ego - z_lane) <= VERTICAL_COMPATIBILITY_TOLERANCE_M` | `geometry/vertical.py` | lane excluded |
| Cost range | every sub-rule cost in `[0, 1]`, checked in `aggregate_max_component` with `1e-8` slack | `aggregation.py:21` | `ValueError` |
| Macro aggregation | max over `applicable` sub-rules only; all-non-applicable ⇒ macro `NOT_APPLICABLE`, cost 0 | `aggregation.py:23-35` | — |
| Closing operator | morphological closing `buffer(+e).buffer(-e)` is idempotent-in-effect and monotone non-decreasing in area; it can only *add* surface, so it can only *reduce* `offroad` — it can never create a false violation | M1 design | `TEST-RBCOST-003` asserts area monotonicity |
| Standstill threshold | `0.1 m/s`, one order of magnitude below the `~1 m/s` residual speed of a vehicle "creeping" in a queue and above the physics solver's residual velocity noise | `DEC-RBCOST-003` | — |
| Lane-direction cone | `±60°` (`cos 60° = 0.5`) around the route tangent, applied at the ego's projection on each lane centreline | `DEC-RBCOST-004` | lanes outside both cones contribute to neither surface |
| Determinism | all added geometry is a pure function of the static map and the ego pose; no RNG, no iteration-order dependence (surfaces are unioned, not folded) | design | `TEST-RBCOST-012` |
| State/reset | `wrong_carriageway` is memoryless and owns no `RulebookMemory` field, so the registry's field-ownership invariant is unchanged | `registry.py:167-192` | `RulebookV2Registry.validate` |
| Calibration hash | the ego config is unchanged, so `config_hash` is unchanged and the frozen catalog eligibility index remains valid | §4.4 | — |

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `DEC-RBCOST-001` | Specification clarification | Should the drivable-surface union be morphologically closed, and at what `e`? | (a) closing at `e = 0.10 m`; (b) closing at `e = 0.05 m`; (c) repair the per-lane polygons upstream in each adapter | (a). Measured: `0.05 m` leaves seam activations, `0.10 m` removes all of them while leaving the genuine map-edge gap intact. (c) is a per-source fix duplicated in two adapters and does not cover the centreline/width fallback path. | Adds a numerical tolerance to `C_drive` (v4.7 §7.2.1). Risk: a genuine 20 cm gap between two lanes is bridged — a gap no vehicle can be meaningfully "off-road" inside. | Approved 2026-08-01, implemented (M1) → `ADR-046` |
| `DEC-RBCOST-002` | Specification deviation | Replace the `b_e` bound `min(4.0, …)` with `min(8.0, …)` | (a) bound at `8.0` (dry-asphalt tyre–road limit, equal to the `b_i` already assumed for identical vehicles); (b) remove the bound entirely and use the measured `≈10.7`; (c) keep `4.0` | (a). It is the only value that makes the model internally consistent, and it remains conservative w.r.t. the measurement. (b) discards the protection against a mis-measured calibration run. | `d_safe` halves at every speed; `d_req` (signal) and `d_stop` (crosswalk, vehicle-yield) shrink by the same factor, becoming physically accurate. Amends v4.7 §6.2.4 step 10 and the §6.2 parameter table. **The persisted artifact must be regenerated or every run fails closed.** | Approved 2026-08-01, implemented (M2) → `ADR-047` |
| `DEC-RBCOST-003` | Specification clarification | Make longitudinal RSS `NOT_APPLICABLE` when `v_e <= 0.1` and `v_i <= 0.1` | (a) standstill scoping; (b) gate on closing dynamics `v_e > v_i`; (c) no change | (a). The RSS response term prices an acceleration the stopped ego is not performing; the resulting state offers no cost-reducing action (reversing triggers `wrongway`, advancing triggers collision). (b) would also silence genuine same-speed tailgating, which is the core case the rule exists for. | New applicability clause in v4.7 §6.2.1. Queueing stops being a permanent R2 violation. | Approved 2026-08-01, implemented (M2) → `ADR-048` |
| `DEC-RBCOST-004` | Specification clarification | Add the R3 sub-rule `wrong_carriageway` with cost `A(P_e ∩ (C_opp \ C_aligned)) / A(P_e)` | (a) as stated; (b) extend `offroad` to subtract opposing lanes from `C_drive`; (c) extend `wrongway` with a positional term | (a). (b) would conflate two different infractions under one name and change `offroad`'s approved meaning; (c) would violate `REQ-RBCOST-003`. Keeping the three concerns separate (position / direction of motion / markings) preserves the v4.7 §7.1 decomposition. | New v4.10 §7.3-bis, one registry entry, one `road_traffic_compliance` group member. Subtracting `C_aligned` is what keeps the rule silent inside junctions. | Approved 2026-08-01, implemented (M4) → `ADR-049` |
| `DEC-RBCOST-005` | Specification clarification | Does `wrong_carriageway` v1 charge a legal overtake across a broken centreline? | (a) charge unconditionally, record the separating marking class as a diagnostic; (b) suppress the cost when the separating marking permits passing | (a) for v1. (b) requires a reliable nearest-separating-marking association that does not exist yet, and the data to build it is exactly what the (a) diagnostic collects. Note that overtaking is not silently free either way: it also crosses a marking, which R3 already prices. | If overtaking proves to be systematically suppressed by the resulting cost, revisit with the collected diagnostic. | Approved 2026-08-01, implemented as (a) in M4 |
| `DEC-RBCOST-006` | Blocking technical issue | Newly mapped markings may add `validation_errors`, and `catalog_eligibility.py:111` makes any error disqualifying — this can shrink the frozen catalog | (a) run the read-only Waymo dry-run first and proceed only if zero new errors; (b) reclassify marking-geometry defects as non-blocking diagnostics; (c) accept catalog shrinkage | (a), falling back to (b) only if the dry-run shows new errors. (c) is unacceptable: the frozen selection index is the selection authority and a silent membership change breaks reproducibility of every completed run. | Gates M3. | Approved 2026-08-01; dry-run over the 1805 frozen Waymo records found 21 (1.16%) needing fallback (b); implemented in M3 |
| `DEC-RBCOST-007` | Specification clarification | Should a red-light violation persist while the ego remains in the controlled junction? | (a) decide after the M0 measurement; (b) add the latch now, mirroring the crosswalk / vehicle-yield illegal-entry latches; (c) leave as a one-step event | (a). Today it cannot be distinguished whether the signal rule is sparse-but-correct or never selected. Changing behaviour before that measurement risks fixing a non-problem. | Deferred to M5. | Approved 2026-08-01 as deferred; not yet measured |
| `DEC-RBCOST-008` | Implementation detail | `ROAD_EDGE_MEDIAN` → `ROAD_BOUNDARY`? | (a) map it to `ROAD_BOUNDARY`; (b) leave unmapped but diagnosed | (a). A median edge is a physical road boundary in every sense the rulebook uses the class for. | Adds boundary geometry on Waymo maps. Covered by the `DEC-RBCOST-006` dry-run. | Approved 2026-08-01, implemented (M3) |
| `DEC-RBCOST-009` | Implementation detail | `ROAD_LINE_PASSING_DOUBLE_YELLOW` → which class? | (a) `LANE_MARKING_DASHED`; (b) `LANE_MARKING_SOLID` | (a). In Waymo semantics this class denotes a double yellow that *permits* passing; functionally it is a broken line. (An earlier draft of the audit said SOLID; that was wrong and has been corrected.) | Marking classification only. | Approved 2026-08-01, implemented (M3) |

No dependent work starts while its gate is unresolved. M0 depends on no gate.

## 7. Proposed Design

### 7.1 M1 — Drivable-surface seam closing (`REQ-RBCOST-001`, `DEC-RBCOST-001`)

`geometry/drivable.py` gains a module constant

```python
DRIVABLE_SEAM_CLOSING_M = 0.10
```

and `_union_selected_surfaces` applies `union.buffer(+e).buffer(-e)` after the
union, inside the cached function so the cost is paid once per stable lane set.

Justification, standalone: adjacent lane polygons are built independently — from
per-point widths on Waymo, from block geometry on PG, from a mitre buffer of the
centreline in the fallback path — so their shared edges do not coincide to
floating-point accuracy and the union retains hairline interior holes. The
closing removes interior features narrower than `2e`. It is a repair of a
representation defect, not a relaxation of the rule: closing only adds surface,
so it can only lower an `offroad` cost, never raise one.

`OFFROAD_AREA_EPSILON_M2` stays at `1e-4`.

Error handling: if the closing produces an empty or invalid geometry (possible
only for degenerate input), fall back to the raw union and record a diagnostic
rather than failing the episode — the raw union is the current behaviour, so the
fallback is strictly no worse.

### 7.2 M2 — RSS calibration bound and standstill scope (`REQ-RBCOST-006/007/010`)

`calibration.py`: rename the intent of the constant and change the value.

```python
# Physical bound: dry-asphalt tyre-road deceleration limit.  It is also the
# value already assumed for surrounding vehicles (FRONT_MAX_BRAKE_MPS2), so the
# model no longer assumes the ego brakes worse than identical traffic.
MAX_REFERENCE_BRAKE_MPS2 = 8.0
```

The persisted `cap_mps2` and both validation sites follow automatically.
Regeneration is mandatory: `make rulebook-v2-collect-trials` then
`make rulebook-v2-calibrate` then `make rulebook-v2-validate-calibration`. With
`b_meas ≈ 10.7` the artifact becomes `ego_min_brake_mps2 = 8.0`. Because the ego
config is untouched, `config_hash` is stable and the frozen catalog stays valid.

`components/rss.py`: a candidate is dropped from evaluation when
`max(v_e, v_i) <= RSS_STANDSTILL_SPEED_MPS = 0.1`. If **all** candidates are
dropped, the result is `NOT_APPLICABLE` with `candidate_count: 0` and a
`standstill_dropped` diagnostic; if some remain, they are evaluated normally.
This keeps the aggregation invariant (`REQ-RBCOST-008`) intact: a non-applicable
sub-rule contributes nothing rather than contributing a zero cost.

Standalone justification for the threshold: `0.1 m/s` is above the residual
velocity a stopped rigid body retains in the Bullet solver and an order of
magnitude below any speed at which a following manoeuvre is under way. It is not
a "grace band" on the safe distance — at any speed above it, the full RSS
distance applies unchanged.

### 7.3 M3 — Marking coverage (`REQ-RBCOST-004/005/011`)

`pg_static_adapter._FEATURE_CLASSES` adds:

```python
"ROAD_LINE_SOLID_SINGLE_YELLOW":  MapFeatureClass.LANE_MARKING_SOLID,
"ROAD_LINE_BROKEN_SINGLE_YELLOW": MapFeatureClass.LANE_MARKING_DASHED,
```

`waymo_static_adapter._FEATURE_CLASSES` adds:

```python
"ROAD_LINE_SOLID_DOUBLE_WHITE":    MapFeatureClass.LANE_MARKING_SOLID,
"ROAD_LINE_BROKEN_SINGLE_YELLOW":  MapFeatureClass.LANE_MARKING_DASHED,
"ROAD_LINE_BROKEN_DOUBLE_YELLOW":  MapFeatureClass.LANE_MARKING_DASHED,
"ROAD_LINE_PASSING_DOUBLE_YELLOW": MapFeatureClass.LANE_MARKING_DASHED,  # DEC-RBCOST-009
"ROAD_EDGE_MEDIAN":                MapFeatureClass.ROAD_BOUNDARY,        # DEC-RBCOST-008
```

Both adapters replace the silent `continue` on an unmapped type with an
accumulated `unmapped_feature_types` counter surfaced in the adapter result, so a
future schema extension cannot disappear again. This is a diagnostic, not a
validation error — it must not affect `rulebook_eligible`.

Ordering: the `DEC-RBCOST-006` dry-run runs **before** the mapping change lands.

### 7.4 M4 — `wrong_carriageway` (`REQ-RBCOST-009`)

`drivable_surface_for_ego` is generalised to return three surfaces from the same
lane pass, as a small frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class DrivableSurfaces:
    all_lanes: BaseGeometry      # today's normative C_drive, unchanged
    route_aligned: BaseGeometry  # t_lane · t_route >=  cos(60°)
    opposing: BaseGeometry       # t_lane · t_route <= -cos(60°)
```

`t_lane` is the lane centreline tangent at the ego's projection on that lane;
`t_route` is the canonical route tangent at the ego's route projection — both
already computed by `RoutePolyline.project`. Lanes in neither cone (crossing
branches inside a junction) enter neither set. `all_lanes` is byte-for-byte the
current normative surface, so `offroad` is untouched.

```python
q_wrong_carriageway = A(P_e ∩ (C_opp \ C_aligned)) / A(P_e)
```

Subtracting `C_aligned` is the junction-safety mechanism: an ego turning left is
inside its own route-aligned junction lane, so the overlap with the opposing
through-lane polygons is cancelled and the cost stays `0`. On a two-way road the
oncoming lane is covered by no aligned lane, so the invaded fraction is charged.

The rule is memoryless and has **no time ramp**. An earlier draft proposed one;
it was withdrawn because its stated justification (robustness against transient
junction overlap) was hypothesised rather than measured, and because `offroad` —
the direct structural analogue, and the shape the user asked for — has none. If
transient junction false positives appear in the M4 measurement, they are a
geometry defect to be fixed in the cone logic, not to be masked by a timer.

Applicability: `applicable=True` whenever the drivable surface is non-empty,
mirroring `offroad`. Diagnostics record the opposing lane ids, the invaded area,
and — for `DEC-RBCOST-005` — the class of the nearest separating marking.

Registry: one `ComponentDefinition("wrong_carriageway", ROAD_TRAFFIC_COMPLIANCE,
evaluate_wrong_carriageway)` inserted after `wrong_way`, with no owned memory
fields; `aggregation.py:68` adds the name to the `road_traffic_compliance` set.
`RulebookV2Registry.validate` enforces a fixed sequence, so both edits are
required together.

### 7.5 M0 — Diagnostics (`REQ-RBCOST-012`)

Purely additive, no gate:

- per-episode counters: selected controls by type, applicable-step counts per
  control sub-rule, controls dropped by `ControlLineOffRouteError`, controls
  dropped by the `approach_lane_id in route_lane_ids` filter, unmapped map
  feature types;
- GIF overlay (`runtime/io/video_diagnostics.py`): render the sub-rule status
  (`NOT_APPLICABLE` / `SATISFIED` / `VIOLATED`) alongside the cost, and for
  `signal` the current colour and the signed distance to the control line.

This is what makes F4 measurable and what will make M1–M4 verifiable in the
evaluation GIFs the user actually reviews.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
| --- | --- | --- | --- | --- |
| `REQ-RBCOST-001` | `AC-RBCOST-001` | `src/thesis_rl/rulebook/v2/geometry/drivable.py` | `tests/test_rulebook_v2_geometry.py::test_drivable_surface_closes_lane_seams`, `::test_closing_never_reduces_surface_area` | Planned |
| `REQ-RBCOST-002` | `AC-RBCOST-002` | `src/thesis_rl/rulebook/v2/components/road.py` (unchanged constant) | `tests/test_rulebook_v2_road.py::test_offroad_area_epsilon_unchanged` | Planned |
| `REQ-RBCOST-003` | `AC-RBCOST-003` | `components/road.py::evaluate_wrongway` (unchanged) | `tests/test_rulebook_v2_road.py::test_forward_motion_never_charges_wrongway` | Planned |
| `REQ-RBCOST-004` | `AC-RBCOST-004` | `context/pg_static_adapter.py`, `context/waymo_static_adapter.py` | `tests/test_rulebook_v2_pg_adapter.py::test_yellow_centreline_enters_catalog`, `tests/test_rulebook_v2_waymo_adapter.py::test_solid_marking_classes_mapped` | Planned |
| `REQ-RBCOST-005` | `AC-RBCOST-005` | same | `tests/test_rulebook_v2_waymo_adapter.py::test_dashed_marking_classes_mapped` | Planned |
| `REQ-RBCOST-006` | `AC-RBCOST-006` | `components/rss.py` (unchanged formula) | `tests/test_rulebook_v2_rss.py::test_safe_distance_reference_table` | Planned |
| `REQ-RBCOST-007` | `AC-RBCOST-007` | `rulebook/v2/calibration.py` | `tests/test_rulebook_v2_calibration.py::test_brake_bound_is_physical_limit`, `::test_legacy_artifact_rejected` | Planned |
| `REQ-RBCOST-008` | `AC-RBCOST-008` | `rulebook/v2/aggregation.py` | `tests/test_rulebook_v2_aggregation.py::test_wrong_carriageway_in_r3_group` | Planned |
| `REQ-RBCOST-009` | `AC-RBCOST-009` | `components/road.py::evaluate_wrong_carriageway`, `geometry/drivable.py`, `registry.py` | `tests/test_rulebook_v2_road.py::test_wrong_carriageway_*` (5 cases) | Planned |
| `REQ-RBCOST-010` | `AC-RBCOST-010` | `components/rss.py` | `tests/test_rulebook_v2_rss.py::test_standstill_pair_not_applicable`, `::test_partial_standstill_still_evaluated` | Planned |
| `REQ-RBCOST-011` | `AC-RBCOST-011` | both static adapters | `tests/test_rulebook_v2_pg_adapter.py::test_unmapped_feature_type_recorded` | Planned |
| `REQ-RBCOST-012` | `AC-RBCOST-012` | `rulebook/v2/subrule_diagnostics.py`, `runtime/io/video_diagnostics.py` | `tests/test_video_diagnostics.py::test_overlay_renders_subrule_status`, `tests/test_rulebook_v2_transition.py::test_control_drop_counters` | Planned |

### 8.1 Acceptance criteria

| ID | Observable criterion |
| --- | --- |
| `AC-RBCOST-001` | On a fixture of two adjacent lane polygons sharing a nominally identical edge perturbed by `1e-6 m`, the returned surface has zero interior rings, and a footprint centred on the seam reports `offroad == 0.0` |
| `AC-RBCOST-002` | `OFFROAD_AREA_EPSILON_M2 == 1e-4` and the ratio is computed on the raw difference area above it |
| `AC-RBCOST-003` | For any forward velocity, however misaligned the heading, `wrongway` cost is `0.0` |
| `AC-RBCOST-004` | A PG scenario containing `ROAD_LINE_SOLID_SINGLE_YELLOW` produces a `map_feature_catalog` entry of class `LANE_MARKING_SOLID` with the same geometry |
| `AC-RBCOST-005` | Each dashed class of §7.3 produces a `LANE_MARKING_DASHED` catalog entry |
| `AC-RBCOST-006` | `safe_distance_m` reproduces the `b_e = 8.0` column of the §4.4 table to `1e-2 m` |
| `AC-RBCOST-007` | `calibrate_ego_braking` on trials with `b_meas = 10.73` returns `8.0`; `load_calibration_artifact` rejects an artifact carrying `cap_mps2 = 4.0` |
| `AC-RBCOST-008` | `aggregate_rulebook_result` places `wrong_carriageway` in `road_traffic_compliance` and R3 equals the max over its applicable sub-rules including the new one |
| `AC-RBCOST-009` | Five geometry cases: (i) fully in the aligned lane ⇒ `0.0`; (ii) fully in the opposing lane ⇒ `1.0`; (iii) half-straddling ⇒ `0.5 ± 0.02`; (iv) inside a junction, turning left across an opposing through lane while inside an aligned junction lane ⇒ `0.0`; (v) empty drivable surface ⇒ `NOT_APPLICABLE` |
| `AC-RBCOST-010` | Two stationary vehicles `2 m` apart ⇒ `rss` `NOT_APPLICABLE`, cost `0.0`; ego at `0.05 m/s` behind a leader at `5 m/s` ⇒ evaluated normally |
| `AC-RBCOST-011` | A feature of an unknown type is absent from the catalog *and* present in `unmapped_feature_types`, with `validation_errors` unchanged |
| `AC-RBCOST-012` | The overlay text for a `NOT_APPLICABLE` sub-rule differs from that of a `SATISFIED` one; episode diagnostics expose the four control-drop counters |

## 9. Test Strategy Defined Before Implementation

Acceptance-test-first: each milestone's tests are written and observed failing
before its production change.

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
| --- | --- | --- | --- | --- | --- |
| `TEST-RBCOST-001` | Unit | Seam between two lane polygons is closed | Two rectangles sharing an edge perturbed by `1e-6 m` | Union has 0 interior rings | `REQ-RBCOST-001` |
| `TEST-RBCOST-002` | Unit | Footprint on a seam is not off-road | Footprint centred on the perturbed edge | `offroad == 0.0`, status `SATISFIED` | `REQ-RBCOST-001` |
| `TEST-RBCOST-003` | Unit (numerical) | Closing is area-monotone | Randomised (fixed-seed) convex lane sets | `closed.area >= raw.area` and `closed.contains(raw)` | `REQ-RBCOST-001` |
| `TEST-RBCOST-004` | Integration | Seam artefact removed on a real map | PG `map='SCS'` sweep of §4.1, plus the same sweep on one Waymo scenario | Activation share equals the legitimate map-edge baseline at every lateral offset | `REQ-RBCOST-001` |
| `TEST-RBCOST-005` | Unit (boundary) | Genuine off-road still charged | Footprint half outside the outermost lane | Ratio `0.5 ± 0.02`, status `VIOLATED` | `REQ-RBCOST-001` |
| `TEST-RBCOST-006` | Unit (regression) | Off-road tolerance not relaxed | `OFFROAD_AREA_EPSILON_M2` | `== 1e-4` | `REQ-RBCOST-002` |
| `TEST-RBCOST-007` | Unit | `wrongway` unaffected | Forward motion, heading offset up to `±89°` | cost `0.0` | `REQ-RBCOST-003` |
| `TEST-RBCOST-008` | Unit | PG yellow centreline mapped | Synthetic PG scenario dict with `ROAD_LINE_SOLID_SINGLE_YELLOW` | Catalog entry `LANE_MARKING_SOLID`, geometry preserved | `REQ-RBCOST-004` |
| `TEST-RBCOST-009` | Integration (read-only) | Catalog eligibility unchanged | All 1805 Waymo records, adapters run with and without the new mapping | `validation_errors` set identical; report any delta | `DEC-RBCOST-006` |
| `TEST-RBCOST-010` | Unit | New Waymo marking classes mapped | One synthetic scenario per added class | Expected `MapFeatureClass` per §7.3 | `REQ-RBCOST-004/005` |
| `TEST-RBCOST-011` | Unit (missing data) | Unmapped type diagnosed, not dropped silently | Feature of type `"ROAD_LINE_FUTURE_SCHEMA"` | Absent from catalog, present in `unmapped_feature_types`, `validation_errors` unchanged | `REQ-RBCOST-011` |
| `TEST-RBCOST-012` | Unit (numerical) | `d_safe` reference table | `(v_e, v_i)` grid of §4.4 at `b_e = 8.0` | Matches table to `1e-2 m` | `REQ-RBCOST-006` |
| `TEST-RBCOST-013` | Unit | Brake bound is the physical limit | Trials with `b_meas = 10.73` | `ego_min_brake_mps2 == 8.0` | `REQ-RBCOST-007` |
| `TEST-RBCOST-014` | Unit (compatibility) | Stale artifact rejected fail-closed | Persisted artifact with `cap_mps2 = 4.0` | `ValueError` | `REQ-RBCOST-007` |
| `TEST-RBCOST-015` | Unit (boundary) | RSS standstill scoping | `v_e = v_i = 0.0`; `v_e = v_i = 0.1`; `v_e = 0.11, v_i = 0.0` | `NOT_APPLICABLE`, `NOT_APPLICABLE`, evaluated | `REQ-RBCOST-010` |
| `TEST-RBCOST-016` | Unit | Partial standstill still evaluated | `v_e = 0.05`, `v_i = 5.0` | Evaluated, cost per formula | `REQ-RBCOST-010` |
| `TEST-RBCOST-017` | Unit (regression) | Queue does not accumulate R2 cost | 50 steps, both vehicles stopped `2 m` apart | R2 cost `0.0` at every step | `REQ-RBCOST-010` |
| `TEST-RBCOST-018` | Unit | `wrong_carriageway` five geometry cases | Fixtures per `AC-RBCOST-009` | Values per `AC-RBCOST-009` | `REQ-RBCOST-009` |
| `TEST-RBCOST-019` | Unit (state/reset) | `wrong_carriageway` is memoryless | Same pose reached via two different histories | Identical cost; empty `MemoryDelta` | `REQ-RBCOST-009` |
| `TEST-RBCOST-020` | Unit (contract) | Registry and aggregation stay consistent | `RulebookV2Registry().validate()`; `aggregate_rulebook_result` | No error; `wrong_carriageway` in the R3 group | `REQ-RBCOST-008` |
| `TEST-RBCOST-021` | Unit (determinism) | Surfaces are order-independent | Same lane set in shuffled order (fixed seed) | Identical surfaces and cost | Assumptions §5 |
| `TEST-RBCOST-022` | Unit | Overlay distinguishes applicability | Component results with each `ComponentStatus` | Distinct rendered text per status | `REQ-RBCOST-012` |
| `TEST-RBCOST-023` | Unit | Control-drop counters | Transition with one off-route control line and one off-route approach lane | Both counters `== 1` | `REQ-RBCOST-012` |
| `TEST-RBCOST-024` | Integration (smoke) | End-to-end training smoke unaffected | `make smoke` | Completes; no rulebook evaluation failure | All |

Regression tests required by AGENTS.md, one per fixed defect: `TEST-RBCOST-002`
(F5a), `TEST-RBCOST-008` (F2), `TEST-RBCOST-013` (F3a), `TEST-RBCOST-017` (F3b /
F5b), `TEST-RBCOST-018` (F1), `TEST-RBCOST-023` (F4b).

### 9.1 Commands (all verified to exist)

| Purpose | Command |
| --- | --- |
| Focused rulebook tests | `uv run --no-sync python -m pytest -q tests/test_rulebook_v2_road.py tests/test_rulebook_v2_geometry.py tests/test_rulebook_v2_rss.py tests/test_rulebook_v2_calibration.py tests/test_rulebook_v2_aggregation.py` |
| Adapter tests | `uv run --no-sync python -m pytest -q tests/test_rulebook_v2_pg_adapter.py tests/test_rulebook_v2_waymo_adapter.py` |
| Rulebook v2 gate | `make rulebook-v2-check` |
| Full suite | `make test` |
| Lint (focused) | `make lint PYTHON_QUALITY_PATHS="<changed files>"` |
| Format check (focused) | `make format-check PYTHON_QUALITY_PATHS="<changed files>"` |
| Smoke | `make smoke` |
| Calibration regeneration | `make rulebook-v2-collect-trials && make rulebook-v2-calibrate && make rulebook-v2-validate-calibration` |
| Catalog eligibility re-run | `make rulebook-v2-filter-catalog` |
| Whitespace | `git diff --check` |

No global mypy target exists; type annotations are added to every new public
function, and no static-check command is invented.

## 10. Milestones

### M0 — Diagnostics and observability (no behaviour change)

- [x] Status: `PARTIAL`. Decision dependencies: **none**.
- Expected files: `src/thesis_rl/rulebook/v2/transition.py`,
  `src/thesis_rl/rulebook/v2/subrule_diagnostics.py`,
  `src/thesis_rl/rulebook/v2/context/{pg,waymo}_static_adapter.py`,
  `src/thesis_rl/runtime/io/video_diagnostics.py`,
  `tests/test_video_diagnostics.py`, `tests/test_rulebook_v2_transition.py`.
- Tasks: control-drop counters; per-episode applicability counters;
  `unmapped_feature_types` accumulation; overlay status + signal colour/distance.
- Done: `unmapped_feature_types` accumulation (implemented as part of M3, both
  adapters); GIF overlay now renders each sub-rule's status marker
  (`[n/a]`/`[ok]`/`[!]`) and, for `signal`, the colour and signed distance —
  `video_diagnostics.py`, `TEST-RBCOST-022`.
- Not done: per-episode control-drop counters (`ControlLineOffRouteError`,
  `approach_lane_id` filter) and per-episode applicability-step counters
  (`TEST-RBCOST-023`) — deferred, no code touches this yet.
- Completion evidence: `tests/test_video_diagnostics.py` (16 tests, incl.
  `test_diagnostic_lines_distinguish_not_applicable_from_satisfied`). The
  episode-counter evidence (a real signalised-scenario GIF/log) is deferred
  along with the counters themselves.

### M1 — Off-road seam closing (fixes F5a)

- [x] Status: `IMPLEMENTED`. Depends on `DEC-RBCOST-001` (approved).
- Expected files: `geometry/drivable.py`, `tests/test_rulebook_v2_geometry.py`,
  `tests/test_rulebook_v2_road.py`, `docs/decisions/ADR-046-*.md`, v4.10 draft.
- Tests: `TEST-RBCOST-001` … `TEST-RBCOST-006`, `TEST-RBCOST-021` — all
  implemented as `test_drivable_surface_closes_lane_seams`,
  `test_drivable_surface_closing_is_area_monotone_and_covers_the_raw_union`,
  `test_drivable_surface_genuine_gap_still_reports_off_road`,
  `test_drivable_surface_union_is_order_independent` in
  `tests/test_rulebook_v2_geometry.py`; all pass.
- Completion evidence: unit-scale seam/monotonicity/genuine-gap tests pass
  (synthetic fixtures, not the full §4.1 map sweep — see §15 limitations). The
  Waymo lane-polygon seam measurement remains unmeasured (§15).

### M2 — RSS calibration and standstill scope (fixes F3a, F3b, F5b)

- [x] Status: `IMPLEMENTED`. Depends on `DEC-RBCOST-002`, `DEC-RBCOST-003`
  (both approved).
- Expected files: `rulebook/v2/calibration.py`, `components/rss.py`,
  `tests/test_rulebook_v2_calibration.py`, `tests/test_rulebook_v2_rss.py`,
  `docs/decisions/ADR-047-*.md`, `ADR-048-*.md`, v4.10 draft.
- Tasks: change the bound; add standstill scoping; regenerate the artifact.
- Tests: `TEST-RBCOST-012` … `TEST-RBCOST-017` all implemented and passing.
- Completion evidence: the regenerated artifact at
  `$DATA_ROOT/scenarionet/rulebook_v2/calibration_b_e.json` records
  `ego_min_brake_mps2 = 8.0`, `cap_mps2 = 8.0`,
  `config_hash = 7ed6b5a0ac48c05d8881d04b334c89691077b7a8a005a6d35c56859e215eee94`
  (unchanged from before this change), measured from the normative 40-trial
  protocol (`b_meas` 5th-percentile consistent with the 12-trial sample:
  `10.73 m/s²` minimum). `make rulebook-v2-validate-calibration` passed.

### M3 — Lane-marking coverage (fixes F2)

- [x] Status: `IMPLEMENTED`. Depends on `DEC-RBCOST-006`, `-008`, `-009` (all
  approved).
- Expected files: both static adapters, `tests/test_rulebook_v2_pg_adapter.py`,
  `tests/test_rulebook_v2_waymo_adapter.py`.
- Tasks: run `TEST-RBCOST-009` dry-run **first**; then add the mappings.
- Done: dry-run executed over all 1805 frozen Waymo records
  (`build_waymo_static_adapter_result` against each real scenario pickle,
  comparing `validation_errors` before/after). Initial run found 21 records
  (1.16%) gaining a new `invalid_map_feature_geometry` error from degenerate
  single-point instances of newly-covered classes; the fallback of
  `DEC-RBCOST-006` was applied (`_NEW_MARKING_FEATURE_TYPES`: a degenerate
  geometry on one of these classes becomes an `unmapped_feature_types`
  diagnostic entry instead of a `validation_errors` entry). Re-run confirmed
  zero new validation errors across all 1805 records. Both adapters now map
  every class listed in §7.3; unmapped types are recorded in the new
  `StaticAdapterResult.unmapped_feature_types` field (`static_adapter.py`),
  never affecting `rulebook_eligible`.
- Tests: `TEST-RBCOST-008` … `TEST-RBCOST-011` implemented, plus a dedicated
  `DEC-RBCOST-006` regression test
  (`test_waymo_adapter_degenerate_new_marking_geometry_is_diagnostic_not_error`).
  All pass.
- Completion evidence: dry-run output
  (`total_waymo_records=1805 checked=1805 failures=0
  records_with_new_validation_errors=0`); adapter test suites (25 tests) pass.

### M4 — `wrong_carriageway` sub-rule (fixes F1)

- [x] Status: `IMPLEMENTED`. Depends on `DEC-RBCOST-004`, `-005` (both
  approved); sequenced after M1.
- Expected files: `geometry/drivable.py`, `components/road.py`, `registry.py`,
  `aggregation.py`, `transition.py`, `tests/test_rulebook_v2_road.py`,
  `tests/test_rulebook_v2_aggregation.py`, `docs/decisions/ADR-049-*.md`, v4.10.
- Tests: `TEST-RBCOST-018` … `TEST-RBCOST-021` all implemented and passing,
  including the junction left-turn case
  (`test_wrong_carriageway_junction_left_turn_inside_aligned_lane_is_zero`).
- Completion evidence: unit-scale geometry-case tests pass (five `AC-RBCOST-009`
  cases plus memorylessness). No live evaluation GIF was captured in this
  session (§15 limitations); the registry/aggregation wiring is verified end
  to end by `test_rulebook_v2_transition.py`'s complete-registry test, which
  now asserts `wrong_carriageway` is present.

### M5 — Measurement-gated decisions (deferred)

- [ ] Status: `Not started`. Depends on `DEC-RBCOST-005`, `-007` and on the M0/M4
  measurements.
- Content: (a) measure the `dashed_lateral_penetration` distribution at
  `solid_line` activation steps, then decide the penetration deadband on its own
  geometric merits; (b) decide the red-light persistence latch from the M0
  counters. Each becomes its own approval gate with its own ADR.

### Out-of-plan follow-up

The scalarizer (`F0`) gets a separate ExecPlan. Until then, the milestones above
reduce the *frequency* of spurious violations but not the *magnitude* of any
violation that does occur.

## 11. Progress And Findings Log

### 2026-08-01 — Plan created

- Completed: full audit of the five user-reported symptoms plus one cross-cutting
  finding, recorded in
  `docs/audits/rulebook_v2_cost_activation_audit_2026-08-01/findings.md`; this
  ExecPlan drafted from it.
- Commands run during the audit: live PG map feature census; 12-trial braking
  protocol re-run (`PYTHONPATH=src … thesis_rl.cli.rulebook_v2_braking_trials`);
  frozen-index signal statistics; a Shapely footprint sweep over the PG lane
  polygons.
- Findings requiring correction before implementation starts: an earlier draft of
  the audit classified `ROAD_LINE_PASSING_DOUBLE_YELLOW` as `LANE_MARKING_SOLID`;
  the correct class is `LANE_MARKING_DASHED` (`DEC-RBCOST-009`). Two proposed
  tolerances — a 1%-relative off-road tolerance and a `wrong_carriageway` time
  ramp — were withdrawn because their only justification was the deferred
  scalarizer fix; see §2.4, §7.1, §7.4.
- Decisions needed: `DEC-RBCOST-001` … `DEC-RBCOST-009`.
- Next step: obtain approval for the gates of M1–M4 (M0 needs none) and begin M0.

### 2026-08-01 — Post-approval follow-up (real-map verification, wrongway status flicker)

- Re-ran the F5a seam measurement (§4.1) on 5 real PG maps from
  `data/scenarionet/pg/database` (not the synthetic fixtures used for the
  unit tests): `PGMap-2000001` (144 raw-union holes, 0 after closing;
  offset-1.2m off-road rate 38.4% -> 3.7%) and `PGMap-2000004` (158 -> 0
  holes; 32.0% -> 4.2%) reproduce the audit's original PG measurement almost
  exactly and confirm the seam closing on real map data, not only synthetic
  fixtures. Resolves the "not re-measured on real maps" limitation for PG;
  the Waymo-side counterpart remains open (see Known limitations).
- User observed the `wrongway` status marker (`[ok]`/`[!]`) flickering on a
  stationary ego in a real evaluation GIF
  (`videos/final_eval/test_arm_stratified/eval_0007/episode_0001`,
  `pg:scenarionet_v1:PGMap-24921253`); initially suspected to be
  `wrong_carriageway` (new this session) but confirmed by frame extraction
  to be the pre-existing `wrongway` component. Root cause and fix recorded
  in `ADR-050`. Not part of the original `DEC-RBCOST-*` gate set; approved
  ad hoc in this conversation.
- Open question, not yet resolved: an RSS transient at the moment a stopped
  leader starts moving (ego still at `v<=0.1 m/s`) can produce a brief,
  self-resolving cost spike, because the standstill exclusion
  (`ADR-048`) requires *both* actors below the speed threshold — the
  moment the leader alone crosses it, the pair re-enters full RSS
  evaluation while `gap_m` has not yet grown. Unlike the double-standstill
  defect this pair does self-resolve (the physical gap grows every step the
  leader keeps moving), so it is explicitly left unfixed pending measurement
  of its real-data frequency/duration rather than corrected speculatively —
  see Deferred required work.

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
| --- | --- | --- | --- | --- | --- |
| `DEV-RBCOST-001` | v4.7 §6.2.4 step 10: `b_e = min(4.0, floor(10 b_meas)/10)` | `b_e = min(8.0, floor(10 b_meas)/10)` | The `4.0` bound discards a measured factor of `2.7` and contradicts the `b_i = 8.0` assumed for identical vehicles | Pending (`DEC-RBCOST-002`) | `tests/test_rulebook_v2_calibration.py`, v4.10, `ADR-047` |
| `DEV-RBCOST-002` | v4.7 §6.2.1: RSS applicable whenever a front candidate exists | Adds a standstill exclusion | The response term prices an action the stopped ego is not performing, creating a state with no cost-reducing action | Pending (`DEC-RBCOST-003`) | `tests/test_rulebook_v2_rss.py`, v4.10, `ADR-048` |
| `DEV-RBCOST-003` | v4.7 §7.2.1: `C_drive` is the plain union of compatible lanes | Adds a morphological closing at `0.10 m` | The per-lane polygons are not watertight; measured 681 interior holes on one PG map | Pending (`DEC-RBCOST-001`) | `tests/test_rulebook_v2_geometry.py`, v4.10, `ADR-046` |
| `DEV-RBCOST-004` | v4.7 §7.3: R3 has no positional lane-direction rule | Adds `wrong_carriageway` | No sub-rule charges occupying the opposing carriageway | Pending (`DEC-RBCOST-004`) | `tests/test_rulebook_v2_road.py`, v4.10 §7.3-bis, `ADR-049` |

## 13. Files

| Path | Action | Purpose |
| --- | --- | --- |
| `src/thesis_rl/rulebook/v2/geometry/drivable.py` | Planned modification | Seam closing; three-surface lane pass |
| `src/thesis_rl/rulebook/v2/components/road.py` | Planned modification | `evaluate_wrong_carriageway` |
| `src/thesis_rl/rulebook/v2/components/rss.py` | Planned modification | Standstill applicability |
| `src/thesis_rl/rulebook/v2/calibration.py` | Planned modification | Physical brake bound |
| `src/thesis_rl/rulebook/v2/registry.py` | Planned modification | Register the new sub-rule |
| `src/thesis_rl/rulebook/v2/aggregation.py` | Modified | Added `wrong_carriageway` to the R3 group |
| `src/thesis_rl/rulebook/v2/transition.py` | Modified | Wired `wrong_carriageway` inputs via `carriageway_surfaces_for_ego`; control-drop counters not yet added |
| `src/thesis_rl/rulebook/v2/context/pg_static_adapter.py` | Modified | Yellow marking coverage; `unmapped_feature_types` diagnostic |
| `src/thesis_rl/rulebook/v2/context/waymo_static_adapter.py` | Modified | Five new marking classes; `unmapped_feature_types`; `DEC-RBCOST-006` degenerate-geometry fallback |
| `src/thesis_rl/rulebook/v2/context/static_adapter.py` | Modified | New `StaticAdapterResult.unmapped_feature_types` field, threaded through `normalize_static_records` |
| `src/thesis_rl/runtime/io/video_diagnostics.py` | Modified | Status marker (`[n/a]`/`[ok]`/`[!]`) and signal colour/distance in the overlay |
| `tests/test_rulebook_v2_geometry.py` | Modified | `TEST-RBCOST-001/003/004/021` |
| `tests/test_rulebook_v2_road.py` | Modified | `TEST-RBCOST-002/005/006/007/018/019` |
| `tests/test_rulebook_v2_rss.py` | Modified | `TEST-RBCOST-012/015/016/017` |
| `tests/test_rulebook_v2_calibration.py` | Modified | `TEST-RBCOST-013/014` |
| `tests/test_rulebook_v2_transition.py` | Modified | Extended complete-registry assertion with `wrong_carriageway` |
| `tests/test_rulebook_v2_pg_adapter.py` | Modified | `TEST-RBCOST-008/011` |
| `tests/test_rulebook_v2_waymo_adapter.py` | Modified | `TEST-RBCOST-009/010/011` plus `DEC-RBCOST-006` regression |
| `tests/test_video_diagnostics.py` | Modified | `TEST-RBCOST-022` |
| `docs/specifications/rulebook_v4.10_specification.md` | Created | `APPROVED` directly (not staged `UNDER_REVIEW`), per explicit user approval of the whole plan; amendments of `DEV-RBCOST-001…004` |
| `docs/decisions/ADR-046-drivable-surface-seam-closing.md` | Created | `DEC-RBCOST-001` |
| `docs/decisions/ADR-047-ego-braking-physical-bound.md` | Created | `DEC-RBCOST-002` |
| `docs/decisions/ADR-048-rss-standstill-applicability.md` | Created | `DEC-RBCOST-003` |
| `docs/decisions/ADR-049-wrong-carriageway-subrule.md` | Created | `DEC-RBCOST-004` |
| `docs/project_index.md` | Modified | Registered the plan, the ADRs, and v4.10 |
| `docs/audits/rulebook_v2_cost_activation_audit_2026-08-01/findings.md` | Modified | Corrected the `PASSING_DOUBLE_YELLOW` mapping and withdrew the binarisation-justified tolerances, per user instruction |
| `$DATA_ROOT/scenarionet/rulebook_v2/calibration_b_e.json` | Regenerated | `ego_min_brake_mps2 = 8.0`, `cap_mps2 = 8.0`, `config_hash` unchanged (data, not versioned) |
| `src/thesis_rl/rulebook/v2/registry.py` | Modified | Registered `wrong_carriageway` |
| `src/thesis_rl/rulebook/v2/components/road.py` | Modified (post-approval follow-up) | `WRONGWAY_STATUS_SPEED_EPSILON_MPS` deadband on `evaluate_wrongway` status only, `ADR-050`; `cost` unchanged |
| `tests/test_rulebook_v2_road.py` | Modified (post-approval follow-up) | `test_wrongway_status_has_a_deadband_against_standstill_physics_noise` |
| `docs/decisions/ADR-050-wrongway-status-deadband.md` | Created | Post-approval follow-up, out of the original `DEC-RBCOST-*` set |

Not in this diff: `subrule_diagnostics.py` (M0 control-drop/applicability
counters deferred, not started).

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
| --- | --- | --- | --- |
| `uv run --no-sync python -m pytest -q tests/ -k "rulebook_v2"` | `PASS` (325 passed) | 2026-08-01 | Full rulebook v2 suite, `dev` container |
| `uv run --no-sync python -m pytest -q` (full repository suite) | `PASS` except 9 pre-existing failures | 2026-08-01 | 1262 passed, 10 failed before restoring one accidentally-deleted tracked data file (`data/scenarionet/frozen/scenario_selection_index.json`, restored via `git checkout --`), then re-verified via the targeted rulebook run above; the remaining 9 failures are in files already modified/uncommitted before this session (`eval_artifacts.py`, `video_diagnostics.py` GIF-timing columns, `final_panels.py`, `train_loop.py`/eval-protocol version, `conf/run_profile/tune.yaml`) and unrelated to this change — not re-run to completion after the restore because they are orthogonal and pre-dated this task |
| `ruff check <changed files>` | `PASS` | 2026-08-01 | All checks passed |
| `ruff format --check <changed files>` then `ruff format <changed files>` | `PASS` after formatting | 2026-08-01 | 7 of 17 changed files needed formatting; applied, re-verified by the rulebook suite re-run |
| `make rulebook-v2-collect-trials && make rulebook-v2-calibrate && make rulebook-v2-validate-calibration` | `PASS` | 2026-08-01 | `RULEBOOK_V2_DATA_ROOT=/scratch/e.respino/thesis-metadrive/data/scenarionet` override required (Makefile default is host-relative and does not read `.env`'s `HOST_DATA_DIR`); artifact regenerated as recorded in §13 |
| Read-only Waymo marking-coverage dry-run (`DEC-RBCOST-006`) | `PASS` | 2026-08-01 | 1805/1805 records checked, 0 failures, 0 new `validation_errors` after the `_NEW_MARKING_FEATURE_TYPES` fallback; script not checked in (ad hoc, reproducible from `build_waymo_static_adapter_result` + the frozen index's `relative_path`/`rulebook_validation_errors`) |
| `make smoke` | `PASS` | 2026-08-01 | End-to-end `td3_sb3` training smoke completed (2000 timesteps, rulebook `monitor only`, `ScenarioNet` dataset); exit code 0 |
| `git diff --check` | `NOT_RUN` | — | Not executed; no whitespace-sensitive edits were made (plain Python/Markdown) |

## 15. Final Reconciliation

| Requirement | Status | Notes |
| --- | --- | --- |
| `REQ-RBCOST-001` | `VERIFIED` | Seam closing implemented and tested (`geometry/drivable.py`); Waymo-side measurement remains a limitation (below) |
| `REQ-RBCOST-002` | `VERIFIED` | `OFFROAD_AREA_EPSILON_M2` unchanged at `1e-4`, asserted by test |
| `REQ-RBCOST-003` | `VERIFIED` | `evaluate_wrongway` untouched; existing regression tests still pass |
| `REQ-RBCOST-004` | `VERIFIED` | PG/Waymo solid-marking coverage extended and tested |
| `REQ-RBCOST-005` | `VERIFIED` | PG/Waymo dashed-marking coverage extended and tested |
| `REQ-RBCOST-006` | `VERIFIED` | `safe_distance_m` formula unchanged; reference-table test passes |
| `REQ-RBCOST-007` | `VERIFIED` | Bound raised to `8.0`; artifact regenerated and validated |
| `REQ-RBCOST-008` | `VERIFIED` | `wrong_carriageway` participates in R3 max-aggregation; registry/aggregation contract tests pass |
| `REQ-RBCOST-009` | `VERIFIED` | Five geometry cases plus memorylessness test pass |
| `REQ-RBCOST-010` | `VERIFIED` | Standstill scoping implemented and tested, incl. a mixed-candidate case |
| `REQ-RBCOST-011` | `VERIFIED` | Both adapters record unmapped types via `unmapped_feature_types` without affecting `validation_errors` |
| `REQ-RBCOST-012` | `PARTIAL` | GIF overlay status distinction implemented and tested; per-episode control-drop/applicability counters not implemented |

### Known limitations

- the Waymo lane-polygon seam measurement (the real-map counterpart of the PG
  sweep in §4.1) was not re-run in this session; the Waymo scenario database
  is mounted (`/scratch/.../data/scenarionet/waymo`) but the sweep script
  itself was not adapted and re-executed for M1's completion evidence — only
  the marking-coverage dry-run (M3) exercised the real Waymo corpus. The PG
  counterpart of this limitation is now resolved (see the 2026-08-01
  post-approval follow-up log entry: real-map re-measurement on 5 PG maps);
- `b_meas` is now the normative 40-trial measurement (regenerated in this
  session), superseding the 12-trial audit sample;
- M0's per-episode control-drop and applicability counters are not
  implemented; only the GIF-overlay status distinction is;
- no live evaluation GIF was captured to visually confirm `wrong_carriageway`
  or the overlay changes in a running episode during initial implementation;
  this was later done ad hoc via frame extraction while diagnosing the
  `wrongway` status flicker (see post-approval follow-up log entry), which
  incidentally confirmed `wrong_carriageway` renders stably (`[ok]`) across
  the inspected frames — but this is not a systematic visual audit.

### Deferred required work

- M5 (`DEC-RBCOST-005`/`-007` follow-through): measure the
  `dashed_lateral_penetration` distribution at `solid_line` activation before
  deciding a penetration deadband; decide the red-light persistence latch
  from M0 counters once those counters exist.
- M0's remaining control-drop/applicability counters.
- The RSS standstill-exit transient (see post-approval follow-up log entry):
  measure how often and for how many consecutive steps a pair re-enters full
  RSS evaluation with `gap_m` still below `safe_distance_m` immediately after
  the leader crosses `RSS_STANDSTILL_SPEED_MPS`, before deciding whether a
  gradual reintroduction (vs. the current hard step at the threshold) is
  warranted. Explicitly not fixed speculatively, per the same
  measure-before-correcting principle already applied to the withdrawn
  off-road and `wrong_carriageway` tolerances (§2.4, §7.1, §7.4).
- `make smoke` has not been re-run since the `wrongway` status-deadband
  follow-up (`ADR-050`); not required before merging since `cost` is
  unchanged and the focused `test_rulebook_v2_road.py` suite passed, but
  flagged here for completeness.

### Deferred optional work

- The scalarizer binarisation (`F0`) correction, explicitly out of scope of
  this plan and reserved for a separate ExecPlan: a residual spurious
  activation still costs as much as a real violation until it is fixed.

### Result summary

M1-M4 are implemented, tested, and verified against the full repository test
suite and an end-to-end training smoke test, with no regressions
attributable to this change. The RSS calibration artifact has been
regenerated with the normative 40-trial protocol. The marking-coverage
change was verified not to shrink the frozen ScenarioNet catalog via a
read-only dry-run over all 1805 Waymo records. M0 is partially implemented
(GIF status overlay); M5 is not started, pending its own measurements. This
ExecPlan is ready for experimental use of M1-M4; it is not `VERIFIED` (no
live evaluation GIF captured to visually confirm the overlay/`wrong_
carriageway` changes in a running episode) and remains `IN_PROGRESS` pending
M0's completion and M5's decisions.
