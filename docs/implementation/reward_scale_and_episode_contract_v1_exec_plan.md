# ExecPlan: Reward scale and episode contract v1

## 1. Metadata

- Feature: correction of the reward scale and of the episode contract that
  together make a standing-still policy optimal.
- Plan ID: `RSEC-V1`.
- Authoritative specifications:
  - `docs/specifications/rulebook_v4.13_specification.md` (`AWAITING_APPROVAL`;
    amends v4.7 §2.9.5/§2.9.6 and §7.3).
  - `docs/specifications/scenarionet_integration_v1.4_specification.md`
    (`AWAITING_APPROVAL`; amends v1.1 §49).
  - `docs/specifications/evaluation_protocol_v1.3_specification.md`
    (`AWAITING_APPROVAL`; amends `EVAL-PROTOCOL` v1.2 reporting).
  - Unchanged and still authoritative for everything not amended:
    `rulebook_v4.7`..`v4.12`, `SCAL-V1.1`, `DRIVING-MISSION-V1.1.1`,
    `OBS-V1.3`, `OBS-LIDAR-V2.0`.
- Status: `AWAITING_DECISIONS` (M0 complete; `DEC-RSEC-001` and the newly raised
  `DEC-RSEC-008` gate every remaining milestone that touches cost).
- Created: 2026-08-09. Last updated: 2026-08-09.
- Branch: `scenarionet-implementation`.
- Related ADRs: `ADR-058`, `ADR-059`, `ADR-060`, `ADR-061`, `ADR-062`;
  supersedes `ADR-050` and `ADR-056`.

## 2. Objective And Scope

### Observed defect

In the `smoke-gpu` run the ego stands still and only starts moving when the
logged scenario ends and every replay actor is removed. Two independent causes
were identified by code inspection and arithmetic:

1. **Reward scale (primary).** With `bounded_priority_weighted_rank`
   (`priority_base=3`) the per-step reward is
   `27·[(sat₁−1)+m₁] + 9·[(sat₂−1)+m₂] + 3·[(sat₃−1)+m₃] + m₄`.
   Longitudinal RSS with the frozen `ρ = 1.0 s` declares a safe distance of
   `1.4375·v + 2.52` m — `16.9 m` at `10 m/s`, i.e. a `1.7 s` headway. The
   implied exchange rates (Knox et al.'s sanity-check method) are: one step of
   R2 violation is worth ~20 steps of maximal progress, and a collision is worth
   0.3 s of sub-RSS-distance following.

   **Correction, 2026-08-09.** The plan's first draft asserted that `c₂ > 0` is
   the *permanent* state of urban car-following and derived a per-episode cost
   of `−1800 ÷ −2700`. The M0 probe does **not** support that: over 20 records
   and 800 steps the expert violates the R2 macro on **7.6%** of steps, not
   continuously — largely because the expert is frequently stopped (median speed
   `1.9 m/s`), where the `ADR-048` standstill exclusion makes RSS inapplicable.
   The magnitude is roughly an order of magnitude smaller than first stated. The
   *conclusion* survives the correction, because standing still scores `0` and
   any net-negative driving return still loses to it, but the mechanism is
   "recurrent violations" rather than "permanent violation", and the fix must be
   sized to the measured distribution rather than to the original estimate. The
   authoritative figures are those of the full M0 run recorded in §11.
2. **Episode contract (secondary, and the visible symptom).** At
   `episode_step >= scenario_length` MetaDrive removes every replay participant
   (`scenario_traffic_manager.py:135`) and freezes every traffic light
   (`scenario_light_manager.py:68`), while
   `extra_steps_after_scenario = 50` grants 5 s of empty world. 31.9% of Waymo
   missions are shorter than 40 m and are therefore completable inside that
   window at ~8 m/s.

### Success

- The expert (logged SDC) trajectory is R2-compliant in nominal driving and
  non-compliant only in genuinely critical states (`AC-RSEC-001`).
- No episode contains a period in which the world is empty of logged actors
  while the episode is still running (`AC-RSEC-004`).
- Reverse motion below a manoeuvre threshold is not charged (`AC-RSEC-006`).
- Reported primary progress metric is continuous and benchmark-comparable on
  the Waymo panel (`AC-RSEC-008`).

### In scope

Episode tail length; evaluation reporting and the making-progress gate; the
unified `wrong_direction` R3 sub-rule; the traffic-control route-reference
polyline; the ego LiDAR detection mask; and the R2 cost scale decision
(`DEC-RSEC-001`), whose option is selected by the M0 measurement.

### Out of scope (deferred by explicit user decision, 2026-08-09)

- **Terminal reward channel.** Previously agreed, then deferred: *"vediamo se
  anche senza reward terminale funziona"*. Consequence recorded in
  `§6 DEC-RSEC-003`: closing the early-off-road-exit shortcut now rests
  entirely on the R2 scale fix.
- **R4 normalizer.** `v_max = 22.2 m/s` (MetaDrive's per-type constant) is
  retained; the user decided not to act now.
- **Stagnation / per-step time penalty.** Rejected on the evidence of Knox et
  al. (the cumulative waiting penalty exceeds the collision penalty and the
  agent crashes instead of waiting) — which is the failure mode already
  observed in this repository's own earlier tests.
- Lexicographic RL algorithms; the scalarizer's mode set.

### Compatibility

Every item changes the scale or the meaning of the episode return. Runs and
checkpoints produced before this plan are **not** comparable to runs after it.
The items are therefore released as one block, followed by a new reference run.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-RSEC-001` | R2's aggregate cost is zero for driving that a competent human expert performs, and strictly positive in critical states | rulebook v4.13 §2 (pending `DEC-RSEC-001`) |
| `REQ-RSEC-002` | `extra_steps_after_scenario = 0` for every source; the only temporal truncation is the exported scenario length | scenarionet_integration v1.4 §1 |
| `REQ-RSEC-003` | R3 exposes exactly one direction sub-rule, `wrong_direction`, aggregating a kinematic and a positional component by `max` | rulebook v4.13 §3 |
| `REQ-RSEC-004` | The kinematic component is memoryless: a function of the current state only, with no additional `RulebookMemory` field | rulebook v4.13 §3.2 |
| `REQ-RSEC-005` | Reverse motion is tolerated up to `2 m/s` and fully charged at `6 m/s` | rulebook v4.13 §3.2 |
| `REQ-RSEC-006` | Traffic-control route coordinates are derived on a route polyline extended along unambiguous successors within a bounded distance `D`, prefix-identical to the canonical route on `[0, L]` | rulebook v4.13 §4 |
| `REQ-RSEC-007` | The ego LiDAR does not detect traffic-light air walls; traffic IDM policies and `vehicle.red_light` are unaffected | `ADR-062`, OBS-LIDAR-V2.0 amendment §1 |
| `REQ-RSEC-008` | `route_completion` is the primary reported progress metric; the geometric final gate remains the binary success event; a making-progress gate at `0.2` scores degenerate episodes zero | evaluation_protocol v1.3 §2-§3 |
| `REQ-RSEC-009` | Benchmark comparability is claimed only for the Waymo panel | evaluation_protocol v1.3 §4 |

## 4. Current Repository Analysis

All statements below are `VERIFIED` by reading the named symbol unless marked
otherwise.

| Area | Location | Current behavior |
|---|---|---|
| Scalarization | `src/thesis_rl/reward/scalarization.py:303` | `base^k · ((sat−1) + m)`; `priority_base=3`; weights 27/9/3/1 |
| RSS safe distance | `src/thesis_rl/rulebook/v2/components/rss.py:59` | `RESPONSE_TIME_S = 1.0`, `MAX_RESPONSE_ACCEL_MPS2 = 3.5`, `FRONT_MAX_BRAKE_MPS2 = 8.0`; `b_e = 8.0` per `ADR-047` |
| RSS applicability | `rss.py:101` | `ADR-048` standstill exclusion at `0.1 m/s` |
| TTC | `components/ttc.py:18` | horizon `3.0 s`; thresholds `0.8 s` vehicles / `1.0 s` VRU — already threshold-shaped, close to nuPlan's `0.95 s` |
| Lateral RSS | `components/rss_lateral.py:19` | `μ = 0.10 m`, `ρ_lat = 0.5 s`; inert for normal abreast driving |
| `wrongway` | `components/road.py:175` | Instantaneous reverse speed normalized by `v_max = 22.2 m/s`, deadband `0.1 m/s` (`ADR-056`). Full cost only at 22 m/s reverse — unreachable in practice |
| `wrong_carriageway` | `components/road.py:121` | Footprint-area fraction on the opposing surface, memoryless |
| R3 group | `rulebook/v2/aggregation.py:71` | `max` over `{offroad, wrongway, wrong_way, wrong_carriageway, solid_line, dashed_line, signal, stop, crosswalk, vehicle_yield}` |
| `solid_line` | `components/road.py:220` | Cost is **binary 1.0** on contact, not graded |
| Control-line derivation | `rulebook/v2/geometry/controls.py:80` | `ControlLineOffRouteError` when the control line does not intersect the canonical route polyline |
| Control selection | `rulebook/v2/transition.py:502,530` | `approach_lane_id ∈ route ∪ {unique successor}` (`ADR-051`) |
| Episode tail | `envs/thesis_scenario_env.py:16,62,920` | `scenario_time_limit_reached`; `extra_steps_after_scenario` default `50`; `conf/env/scenarionet.yaml` sets 50 |
| Replay actor removal | upstream `manager/scenario_traffic_manager.py:110-139` | at `episode_step >= current_scenario_length` every replay participant is cleared |
| Light freeze | upstream `manager/scenario_light_manager.py:68` | `after_step` returns early past the scenario length |
| Traffic-light air wall | upstream `component/traffic_light/base_traffic_light.py:45,102-120` | ghost box `0.25 × lane_width × 1.5 m`; into-mask `InvisibleWall` when red/yellow, `AllOff` when green |
| Ego LiDAR mask | upstream `component/sensors/lidar.py:27` | `self.mask = CollisionGroup.can_be_lidar_detected()` = `Vehicle | InvisibleWall | TrafficObject | TrafficParticipants` |
| IDM light compliance | upstream `policy/idm_policy.py:239,346,479` | uses `lidar.get_surrounding_objects()` (a broad-phase `contactTest`), **not** `perceive()`'s `self.mask` |
| Red-light contact flag | upstream `component/vehicle/base_vehicle.py:771` | `_state_check` sets `vehicle.red_light`; governed by `(Vehicle, InvisibleWall, True)`, not by the LiDAR mask |
| Mission success | `mission/tracker.py:229`, `envs/thesis_scenario_env.py:769` | geometric final-gate crossing; `completion_instant`/`completion_max = clip(s/s_goal, 0, 1)` already computed |
| Evaluation CSV | `runtime/io/csv_recorder.py:92` | `route_completion` already recorded per episode and aggregated |
| ACL gate | `curriculum/config.py:34` | `route_completion_min = 0.85`, a third unrelated threshold |
| Observation history | `envs/observations/causal_semantic.py:572,2425`; `conf/obs/semantic_v3.yaml` | `history_length = 5` (signed route-relative velocity reconstructible); `context_history_length = 21` but carries `hypot(velocity)`, i.e. **unsigned** |

`INFERRED`: that the LiDAR raycast actually returns the traffic-light air wall
is inferred from the upstream comment (*"add to dynamic world so the lidar can
detect it"*) and the `(InvisibleWall, LidarBroadDetector, True)` collision
pair. `TEST-RSEC-011` converts this inference into a verified fact before
`REQ-RSEC-007` is implemented.

### Dataset facts (`VERIFIED`, frozen index, 3500 records)

- Waymo scenario length 198–200 steps (~20 s); PG 501 (~50 s).
- Waymo mission distance: p10 `25.6 m`, median `113.7 m`, p90 `289.3 m`.
- Waymo missions with `D ≤ 40 m`: **31.9%**; `D ≤ 60 m`: 44.9%.
- Split sizes: Waymo train 1100, validation 150, test 555.

## 5. Assumptions And Invariants

- Control period `Δt = 0.1 s` (`physics_world_step_size 0.02 × decision_repeat 5`),
  matching the 10 Hz logged data.
- `v∥` is the ego velocity projected on the canonical route tangent; the route
  follows source-declared **legal** lane direction, preserved deliberately by
  `DRIVING-MISSION-V1.1.1` §6. Where the ego is far from its route the tangent
  is no longer the local legal direction — a declared limitation.
- All sub-rule costs remain in `[0, 1]`; the `SCAL-V1.1` algebraic guarantee
  (`priority_base = 3`) depends on it and must not be weakened.
- Termination semantics are unchanged: collision, physical out-of-road, and
  final-gate success terminate; the time limit truncates.
- The R4 signal `m₄ = clip(Δs/(v_max·Δt), −1, 1)` is unchanged.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-RSEC-001` | Specification deviation | The reward scores the logged human expert at `−179.4` per median episode against `0.0` for standing still (M0, 218,289 steps) | (1) calibrate `ρ`; (2) binary RSS-DS indicator; (3) deadband reparameterization of `c₂`, calibrated so the expert's ordinary driving is compliant and its risky moments are not; (4) remove the categorical jump | **(3)**, alone or with (1). M0 **refutes (1) alone** (`−20.9` even at `ρ=0.3 s`) and shows (4) reaches only `−8.5` while breaking the per-step `R2≻R3` order. The jump accounts for `−171` of `−231`, so the fix must reduce violation *frequency*, which is what (3) does | Reward scale, all runs, all checkpoints | **Open — awaiting user decision** |
| `DEC-RSEC-008` | Blocking technical issue | `offroad` (1.20% of expert steps, p50 cost `0.058`) and `wrong_carriageway` (0.84%, p50 `0.031`) charge the human expert with sub-footprint-fraction violations, the signature of lane-polygon seam artifacts rather than of real behavior | (a) treat as geometry defect and fix the surface; (b) absorb into the R2/R3 scale fix | (a). A human expert does not leave the drivable surface on 1.2% of steps; re-scaling would conceal the defect rather than correct it | R3 cost, off-road termination predicate | **Open — new, raised by M0** |
| `DEC-RSEC-002` | Specification deviation | Episode tail grants 5 s of empty world | 0 / 10 / 50 steps; or freeze actors instead of removing them | `0` (Waymax, V-Max and nuPlan all end at the logged horizon; freezing has no precedent) | Episode length, returns | **Approved 2026-08-09** (`ADR-058`) |
| `DEC-RSEC-003` | Specification clarification | Off-road early exit was cheaper than driving | terminal channel / R2 fix / promote off-road to its own rule | R2 fix only; terminal channel deferred by the user | If M0's fix is insufficient the shortcut may persist | **Approved (deferred), 2026-08-09** |
| `DEC-RSEC-004` | Specification deviation | `wrongway` charges any reverse motion above physics noise, criminalizing manoeuvre reversing | keep / nuPlan 1 s displacement window / memoryless speed thresholds | Memoryless speed thresholds `2` and `6 m/s` | R3 cost; supersedes `ADR-050`/`ADR-056` | **Approved 2026-08-09** (`ADR-060`) |
| `DEC-RSEC-005` | Specification clarification | 25.2% of signalised Waymo scenarios never select their signal | polyline extension / multi-hop lane filter / proximity fallback | Bounded-distance polyline extension along unambiguous successors | Signal coverage | **Approved 2026-08-09** (`ADR-061`) |
| `DEC-RSEC-006` | Specification deviation | The LiDAR arm receives an unintended traffic-light channel | suppress the object / restrict the ego LiDAR mask / keep and declare | Restrict the ego LiDAR raycast mask | Observation content of the LiDAR arm only | **Approved 2026-08-09** (`ADR-062`) |
| `DEC-RSEC-007` | Specification deviation | Binary success is unattainable with a zero tail for a policy slower than the expert | ratio primary / binary primary | Ratio primary, gate unchanged as the binary event | Reporting only | **Approved 2026-08-09** (`ADR-059`) |

`DEC-RSEC-001` is a hard gate: no production change to R2 may begin before M0
reports.

## 7. Proposed Design

### M0 — measurement (no production change)

`scripts/measure_expert_rulebook_costs.py` drives the **logged SDC track**
through the production adapter and evaluates `rss`, `ttc`, `offroad`,
`solid_line` and `wrong_carriageway` per step. It imports the runtime's own
`_rss_candidates` and `_vertical_actor_ids` rather than reimplementing candidate
selection, so measurement and runtime cannot diverge. `--rho-sweep` recomputes
only the longitudinal safe distance for alternative response times from the same
candidate gaps and speeds; that one duplicated formula is asserted equal to
production at `ρ = 1.0` by `TEST-RSEC-001`.

Because the ego pose comes from the recording, the result is a property of the
metric definition, not of any policy.

### M2 — `wrong_direction`

```
q_rev  = clip((max(0, −v∥) − 2) / (6 − 2), 0, 1)      # m/s, memoryless
q_carr = existing evaluate_wrong_carriageway
wrong_direction = max(q_rev, q_carr)
```

Anchors are nuPlan's Driving Direction Compliance thresholds (2 m tolerated,
6 m full violation, over 1 s) **reinterpreted as speeds**. The reinterpretation
is required by observability: the only observation block carrying *signed*
route-relative velocity is 5 steps deep (0.5 s), while the 21-step block carries
unsigned speed, so a 1 s displacement window would make the cost depend on
information the agent does not have. The divergence is a brief high-speed
reverse burst (0.3 s at 5 m/s = 1.5 m), compliant under nuPlan and partially
charged here; declared, not hidden.

`WRONGWAY_SPEED_EPSILON_MPS` and its two ADRs become unnecessary: a `2 m/s`
tolerance is 20× the `0.1 m/s` physics noise floor they were patching.

### M3 — control reference polyline

A second polyline, prefix-identical to the canonical route on `[0, L]` and
extended along unambiguous single successors while the cumulative extension
stays below `D`, is used **only** as the `route` argument of
`derive_control_line`. Prefix identity is the correctness condition that keeps
`control.route_s_m` and the ego's `front_s` in the same frame; it is asserted in
code and by `TEST-RSEC-008`. A control beyond `L` stays selectable (`front_s`
saturates at `L`) but can never register a crossing, so only the graded approach
cost applies — which is the correct signal when the mission goal is a stop line.

### M4 — ego LiDAR mask

Set the engine LiDAR instance's `mask` to `can_be_lidar_detected() & ~InvisibleWall`.
Only `perceive()` consults `mask`; IDM uses the broad-phase `contactTest` and
`vehicle.red_light` uses the chassis contact test, so both survive. `InvisibleWall`
is used elsewhere only by tollgate blocks (ID `$`), absent from every PG profile
in `scenarios/pg/profiles.py`.

### M5 — evaluation reporting

`route_completion` (already recorded) becomes the primary reported metric; the
binary gate rate is retained and relabelled; a making-progress gate at `0.2`
multiplies the composite score to zero; the ACL threshold is aligned; and
benchmark comparability is claimed only for the Waymo panel, because
`route_completion = s/s_goal` is *own-route completion*, whereas nuPlan's metric
is *ego progress ÷ expert progress* — the two coincide only where the goal is
the expert's endpoint, and PG has no expert at all.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-RSEC-001` | `AC-RSEC-001` | pending `DEC-RSEC-001` | `TEST-RSEC-002` | Blocked |
| `REQ-RSEC-002` | `AC-RSEC-004` | `conf/env/scenarionet.yaml`, `envs/thesis_scenario_env.py` | `TEST-RSEC-003`, `TEST-RSEC-004` | Planned |
| `REQ-RSEC-003` | `AC-RSEC-005` | `rulebook/v2/components/road.py`, `registry.py`, `aggregation.py` | `TEST-RSEC-005` | Planned |
| `REQ-RSEC-004` | `AC-RSEC-007` | `rulebook/v2/components/road.py` | `TEST-RSEC-007` | Planned |
| `REQ-RSEC-005` | `AC-RSEC-006` | `rulebook/v2/components/road.py` | `TEST-RSEC-006` | Planned |
| `REQ-RSEC-006` | `AC-RSEC-009` | `rulebook/v2/context/{waymo,pg}_static_adapter.py`, `geometry/controls.py` | `TEST-RSEC-008`, `TEST-RSEC-009` | Planned |
| `REQ-RSEC-007` | `AC-RSEC-010` | `envs/thesis_scenario_env.py` | `TEST-RSEC-010`, `TEST-RSEC-011` | Planned |
| `REQ-RSEC-008` | `AC-RSEC-008` | `runtime/io/csv_recorder.py`, `analysis/run_analysis.py`, `curriculum/config.py` | `TEST-RSEC-012` | Planned |
| `REQ-RSEC-009` | `AC-RSEC-011` | `docs/specifications/evaluation_protocol_v1.3_specification.md`, `analysis/run_analysis.py` | `TEST-RSEC-013` | Planned |

## 9. Test Strategy Defined Before Implementation

### Acceptance criteria

- `AC-RSEC-001`: on the frozen train split, the expert trajectory's aggregate
  R2 cost is `0` for at least the 90th percentile of measured steps, and
  strictly positive on steps independently flagged as conflicts by the catalog.
- `AC-RSEC-004`: no episode step exists at which the logged actors have been
  removed while the episode has not terminated or truncated.
- `AC-RSEC-005`: R3 exposes `wrong_direction` and no `wrongway` component.
- `AC-RSEC-006`: `q_rev(1.9 m/s) = 0`, `q_rev(4.0 m/s) = 0.5`, `q_rev(6.0 m/s) = 1`.
- `AC-RSEC-007`: `RulebookMemory` gains no field; two identical states with
  different histories produce the same `wrong_direction` cost.
- `AC-RSEC-008`: `route_completion` is present and continuous in the aggregate
  report; an episode with `route_completion < 0.2` scores zero.
- `AC-RSEC-009`: on the frozen catalog, the fraction of `has_route_traffic_light`
  Waymo records with zero selectable `SIGNAL` control is strictly lower than the
  25.2% baseline, and `route_s_m` is unchanged for every control already built.
- `AC-RSEC-010`: a red traffic light contributes no LiDAR return to the ego
  point cloud, while `vehicle.red_light` still latches and IDM traffic still
  stops.
- `AC-RSEC-011`: the aggregate report labels benchmark comparability on the
  Waymo panel only.

### Mandatory matrix

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-RSEC-001` | Unit | Sensitivity formula equals production at `ρ=1.0` | Speed grid | `parametric_safe_distance_m == safe_distance_m` | `REQ-RSEC-001` |
| `TEST-RSEC-002` | Unit | Selected R2 option zeroes nominal following and keeps critical states positive | Synthetic follow + cut-in | Per `DEC-RSEC-001` | `REQ-RSEC-001` |
| `TEST-RSEC-003` | Unit | `scenario_time_limit_reached` truncates at exactly the scenario length | `extra_steps=0` | Boundary exact | `REQ-RSEC-002` |
| `TEST-RSEC-004` | Integration | No step runs with the world emptied | Waymo smoke scenario | Truncation at `length` | `REQ-RSEC-002` |
| `TEST-RSEC-005` | Unit | R3 aggregation contains `wrong_direction`, not `wrongway` | Component set | Group membership | `REQ-RSEC-003` |
| `TEST-RSEC-006` | Unit | Reverse-speed anchors | `1.9 / 4.0 / 6.0 / 8.0 m/s` | `0 / 0.5 / 1 / 1` | `REQ-RSEC-005` |
| `TEST-RSEC-007` | Unit | Memorylessness | Same state, two histories | Identical cost; no new memory field | `REQ-RSEC-004` |
| `TEST-RSEC-008` | Unit | Prefix identity of the control reference polyline | Route + successor | `s` identical on `[0, L]` | `REQ-RSEC-006` |
| `TEST-RSEC-009` | Regression | A signal on a lane two hops past the route end becomes selectable; an ambiguous branch does not | Synthetic topology | Selected / `NOT_APPLICABLE` | `REQ-RSEC-006` |
| `TEST-RSEC-010` | Unit | Ego LiDAR mask excludes `InvisibleWall` and nothing else | Constructed env config | Mask equality | `REQ-RSEC-007` |
| `TEST-RSEC-011` | Integration | Raycast returns nothing for a red light; `red_light` still latches | Live env, red then green | No return; flag set | `REQ-RSEC-007` |
| `TEST-RSEC-012` | Unit | Making-progress gate zeroes a degenerate episode | `route_completion = 0.1` | Composite `0` | `REQ-RSEC-008` |
| `TEST-RSEC-013` | Unit | Panel comparability labelling | PG + Waymo panels | Waymo only | `REQ-RSEC-009` |

Regression protection: the `ADR-056` test
`test_wrongway_deadband_zeroes_cost_for_standstill_noise` and its siblings are
**replaced**, not deleted — their intent (a stopped ego is never charged) is
re-asserted against `wrong_direction`, which satisfies it a fortiori.

### Commands (existing, not invented)

- `make rulebook-v2-check`
- `uv run --no-sync python -m pytest -q tests/test_rulebook_v2_road.py`
- `uv run --no-sync python -m pytest -q`
- `make lint`, `make format-check PYTHON_QUALITY_PATHS="..."`
- `make smoke`
- `make config`, `make config-gpu`
- `git diff --check`

## 10. Milestones

- [x] **M0 — Measurement.** `scripts/measure_expert_rulebook_costs.py` written,
      parallelized and executed on the full Waymo train split (218,289 steps).
      Reported in §11. `DEC-RSEC-001` is now decidable and awaits the user;
      `DEC-RSEC-008` was raised by the same measurement.
- [ ] **M1 — R2 scale.** Blocked on `DEC-RSEC-001`.
- [ ] **M2 — `wrong_direction`.** `components/road.py`, `registry.py`,
      `aggregation.py`; removes `WRONGWAY_SPEED_EPSILON_MPS`.
- [ ] **M3 — Control reference polyline.** Both static adapters and
      `geometry/controls.py`; re-run the 828-record coverage dry-run.
- [ ] **M4 — Ego LiDAR mask.** `envs/thesis_scenario_env.py`; `TEST-RSEC-011`
      first, to convert the `INFERRED` fact.
- [ ] **M5 — Evaluation reporting.** CSV/analysis/ACL alignment.
- [ ] **M6 — Reconciliation.** Full suite, lint, smoke, index update, new
      reference run.

## 11. Progress And Findings Log

### 2026-08-09 — Diagnosis and decisions

- Established by inspection that **exactly one** R2 component is active in
  nominal driving: longitudinal RSS. TTC (`0.8/1.0 s` thresholds), lateral RSS
  (`μ = 0.10 m`) and clearance (VRU-only) are all inert in normal traffic. This
  narrows `DEC-RSEC-001` from an architectural question to a parameter question.
- Established that the standstill attractor is **not** an artifact of scalar
  scalarization: a lexicographic agent ranks actions by `Q₂`, the discounted sum
  of future R2 costs; "remain stopped" has `Q₂ ≈ 0` and every driving policy has
  `Q₂ < 0`, so with `η = 0` the strict lexicographic agent never consults `Q₃`
  or `Q₄` either. The R2 fix is a prerequisite for **both** algorithm families.
  This corrects an earlier claim in this conversation.
- Established that R3 shares R2's structure: its sub-rules split into *event*
  (`signal`, `stop`, `crosswalk`, `vehicle_yield` — one step) and *persistent
  state* (`offroad`, `wrong_carriageway`, `solid_line` — every step the
  condition holds), and `solid_line`'s cost is binary `1.0`, so 1 s on a solid
  line costs `−60`. M0 was extended to cover the persistent sub-rules.
- Verified that suppressing the traffic-light air wall at the object would have
  broken IDM red-light compliance (`idm_policy.py:346`); the ego-side LiDAR mask
  is the correct intervention. This corrects an earlier proposal.
- Verified that a 1 s reverse-displacement window is **not** reconstructible
  from the observation (`history_length = 5`; the 21-step context row carries
  unsigned speed), which is why `q_rev` is memoryless.
- User decisions recorded: tail `0`; ratio-primary reporting; unified
  `wrong_direction`; no stagnation penalty; terminal channel deferred; R4
  normalizer untouched; benchmark comparability claimed on the Waymo panel only.

### 2026-08-09 — M0 harness, parallelization, and preliminary numbers

- Wrote `scripts/measure_expert_rulebook_costs.py`. It reuses the runtime's
  `_rss_candidates` and `_vertical_actor_ids`, so candidate selection cannot
  diverge from production; the only duplicated formula is
  `parametric_safe_distance_m` for the response-time sweep, pinned to production
  at `ρ = 1.0` by `TEST-RSEC-001`.
- Parallelized on user request (*"non conviene parallelizzare?"*). Records are
  independent, so each worker returns a partial `Measurement` and the parent
  merges; percentiles are always computed from the merged sample lists, never
  from per-worker percentiles, so the result is order-independent. **Verified
  by construction and by experiment**: the 20-record run with `--workers 8`
  produces a byte-identical summary to the sequential run. Wall time per record
  fell from ~3 s to ~0.4 s, which made the definitive run affordable at full
  10 Hz resolution (`--step-stride 1`) instead of the 2 Hz subsample originally
  planned.
- First test run of `tests/test_measure_expert_rulebook_costs.py` failed all 44
  cases with `AttributeError: 'NoneType' object has no attribute '__dict__'`.
  Cause was the test loader, not the script: `@dataclass` resolves
  `sys.modules[cls.__module__]` while processing the class body, so a module
  executed via `spec.loader.exec_module` without prior registration in
  `sys.modules` cannot be loaded if it declares a dataclass. Fixed by
  registering the module first; the fix is commented in place because the
  existing `tests/test_validate_torch_install.py` pattern does not need it and
  would otherwise be copied again. The same run confirmed the rest of the suite
  is green (1347 passed, 0 unrelated failures).
- **Preliminary probe (20 records, 800 steps, 2 Hz)** — superseded by the full
  run, recorded here because it corrected the plan's own diagnosis:

  | quantity | value |
  |---|---|
  | expert speed, p50 / p90 | `1.9` / `9.49 m/s` |
  | expert front gap, p10 / p50 | `3.88` / `10.85 m` |
  | R2 macro violated | **7.6%** of steps (mean cost when violated `0.40`) |
  | `rss` violated | 18.9% of *applicable* steps (applicable on 307/800) |
  | `ttc` violated | 0.4% of steps |
  | `solid_line` violated | **5.0%** of steps, always at cost `1.0` |
  | `wrong_carriageway` violated | 4.3% of steps |
  | `offroad` violated | 0.8% of steps |
  | `rss` violated vs `ρ` | `1.0 s`: 18.9% · `0.75`: 10.7% · `0.5`: 6.8% · `0.4`: 5.5% · `0.3`: 4.9% |

  Two consequences. First, the R2 defect is real but roughly an order of
  magnitude smaller than the plan's first estimate — see the correction in §2.
  Second, `solid_line` is confirmed as an independent problem of the same
  family: the expert sits on a solid marking 5% of the time and each such step
  costs the full `−6`, which is the R3 half of the structural issue and is why
  M0 was extended beyond R2.
- The `ρ` sweep is close to linear in this range and does **not** by itself
  reach zero expert violation, which is evidence against option (1) being
  sufficient alone. No decision is taken on 20 records; `DEC-RSEC-001` stays
  open until the full run reports.

### 2026-08-09 — M0 definitive result: the reward prefers standing still to expert driving

Full run: **1100 Waymo train records, 218,289 steps at 10 Hz, zero scenarios
skipped**, 48 workers, `b_e = 8.0`.

| sub-rule | violated / all steps | applicable steps | mean cost when violated | p50 / p90 |
|---|---:|---:|---:|---|
| `rss` | 8.43% | 100,188 (46%) | — | `0.216` / `0.835` |
| `ttc` | 0.12% | 218,057 | — | `0.429` / `1.0` |
| `offroad` | 1.20% | 218,289 | `0.097` | `0.058` / `0.281` |
| `solid_line` | 1.12% | 214,168 | `1.000` | `1.0` / `1.0` |
| `wrong_carriageway` | 0.84% | 218,090 | `0.066` | `0.031` / `0.173` |
| **R2 macro** | **8.54%** | 218,057 | **`0.342`** | `0.218` / `0.841` |

Expert speed p50 `2.62 m/s`, p90 `12.21 m/s`; front gap p10 `3.93 m`,
p50 `9.79 m`.

**Decisive quantity.** Scoring the logged human expert with the production
reward over a median episode (198 steps, 113.7 m):

| channel | value |
|---|---:|
| R2 | `−204.0` |
| R3 | `−26.5` (upper bound; sub-rules treated as disjoint) |
| R4 | `+51.2` |
| **total** | **`−179.4`**, against **`0.0`** for standing still |

This is a Knox et al. sanity-check failure in its strongest form: the reward's
optimal policy is not the expert's behavior, by a wide margin. The standstill
attractor is confirmed as a property of the reward definition, independent of
any policy, algorithm or scalarization mode.

**Where the deficit comes from.** Decomposing the `−231` of cost:

| term | value |
|---|---:|
| R2 categorical rank jump (`8.54% × −9`) | `−152.1` |
| R2 continuous severity | `−51.9` |
| R3 categorical rank jump (`3.16% × −3`) | `−18.8` |
| R3 continuous severity | `−7.7` |

**The categorical jump is `−171` of `−231`.** The dominant term is the
*frequency* with which a violation is declared at all, not the severity of the
violations. Any fix that acts only on severity, or only on `d_safe`, addresses
the minority of the deficit.

**`DEC-RSEC-001` option (1) is refuted.** The `ρ` sweep, applied to the same
candidate gaps and speeds:

| `ρ` | violated / applicable | expert episode total |
|---:|---:|---:|
| `1.0 s` (production) | 18.37% | `−176.4` |
| `0.75 s` | 7.95% | `−72.5` |
| `0.5 s` | 5.15% | `−39.7` |
| `0.4 s` | 4.31% | `−29.5` |
| `0.3 s` | 3.63% | `−20.9` |

Even at `ρ = 0.3 s` — an aggressive value that would be hard to defend for the
ego's response time — the expert still scores `−20.9`, i.e. still loses to
standing still. The plan's stated preference for option (1) was wrong and is
withdrawn. Note also that as `ρ` falls the violation rate drops but the median
cost *when* violated rises (`0.22 → 0.54`), which is the desired sparse-severe
shape but does not by itself close the gap.

**Counterfactuals on the same measured distribution:**

| candidate | expert episode total |
|---|---:|
| production | `−179.4` |
| `ρ = 0.5 s` alone | `−39.7` |
| no categorical jump at all (pure severity) | `−8.5` |
| deadband on R2 at the expert's p90 (`~0.85%` of steps) | `+5.0` |
| deadband on R2 **and** R3 | `+29.2` |

**New finding requiring a separate decision.** R3's `−26.5` is not one problem
but two:

| sub-rule | contribution | character |
|---|---:|---|
| `solid_line` | `−13.4` | genuine: binary cost `1.0`, expert on a solid marking 1.12% of steps |
| `offroad` | `−7.8` | suspect: p50 cost `0.058`, i.e. ~6% of the footprint — consistent with lane-polygon seam artifacts (`ADR-046`'s subject), not with a human driving off-road |
| `wrong_carriageway` | `−5.3` | suspect: p50 cost `0.031`, same signature |

The `offroad` and `wrong_carriageway` charges have the profile of a **geometry
defect**, not of a scale defect: a human expert does not leave the drivable
surface on 1.2% of steps. If confirmed, they must be fixed as geometry and not
compensated by re-scaling, which would hide the defect. This is recorded as
`DEC-RSEC-008`.

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-RSEC-001` | scenarionet_integration v1.1 §49 freezes `extra_steps_after_scenario = 50` | `0` | 50 was inherited from visual replay; it creates an empty-world window exploitable in 31.9% of Waymo missions | Approved 2026-08-09 | `ADR-058`, v1.4 |
| `DEV-RSEC-002` | rulebook v4.7 §7.3 `wrongway` normalized by `v_max` with a `0.1 m/s` deadband | `q_rev` with `2`/`6 m/s` anchors, merged into `wrong_direction` | Charged legal manoeuvre reversing; full cost unreachable in practice | Approved 2026-08-09 | `ADR-060`, v4.13 |
| `DEV-RSEC-003` | nuPlan DDC is a 1 s displacement window | Reinterpreted as instantaneous speeds | Preserves Markov observability given `history_length = 5` | Approved 2026-08-09 | `ADR-060` §Consequences |
| `DEV-RSEC-004` | rulebook v4.7 §2.9.5/§2.9.6 derive `route_s_m` on the canonical route only | Prefix-identical bounded extension | 25.2% of signalised scenarios never see their signal | Approved 2026-08-09 | `ADR-061`, v4.13 |
| `DEV-RSEC-005` | `EVAL-PROTOCOL` reports binary success as primary | `route_completion` primary | A zero tail makes binary success unattainable for a policy slower than the expert; benchmarks report a ratio | Approved 2026-08-09 | `ADR-059`, v1.3 |

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `scripts/measure_expert_rulebook_costs.py` | Created | M0 measurement |
| `docs/decisions/ADR-058..ADR-062` | Created | Approved decisions |
| `docs/specifications/rulebook_v4.13_specification.md` | Created | §7.3 and §2.9.5/§2.9.6 amendments |
| `docs/specifications/scenarionet_integration_v1.4_specification.md` | Created | §49 amendment |
| `docs/specifications/evaluation_protocol_v1.3_specification.md` | Created | Reporting amendment |
| `docs/specifications/observation_lidar_v2.0.1_amendment.md` | Created | Traffic-light blindness limitation |
| `conf/env/scenarionet.yaml` | Planned modification | `extra_steps_after_scenario: 0` |
| `src/thesis_rl/envs/thesis_scenario_env.py` | Planned modification | Default tail; ego LiDAR mask |
| `src/thesis_rl/rulebook/v2/components/road.py` | Planned modification | `wrong_direction` |
| `src/thesis_rl/rulebook/v2/{registry,aggregation}.py` | Planned modification | Sub-rule registration and grouping |
| `src/thesis_rl/rulebook/v2/geometry/controls.py` | Planned modification | Control reference polyline |
| `src/thesis_rl/rulebook/v2/context/{waymo,pg}_static_adapter.py` | Planned modification | Build and pass the extended polyline |
| `src/thesis_rl/runtime/io/csv_recorder.py`, `src/thesis_rl/analysis/run_analysis.py` | Planned modification | Primary metric and gate |
| `src/thesis_rl/curriculum/config.py` | Planned modification | Threshold alignment |
| `tests/test_rulebook_v2_road.py`, `tests/test_rulebook_v2_transition.py`, and new files | Planned modification | Mandatory matrix |
| `docs/project_index.md` | Planned modification | Authority rows |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `scripts/measure_expert_rulebook_costs.py --limit 3 --step-stride 10` | `PASS` | 2026-08-09 | Harness smoke: 3 scenarios, 60 steps, no adapter rejection |
| `scripts/measure_expert_rulebook_costs.py --limit 20 --step-stride 5` sequential vs `--workers 8` | `PASS` | 2026-08-09 | Byte-identical summaries; establishes order-independence of the parallel merge |
| `scripts/measure_expert_rulebook_costs.py --split train --source waymo --step-stride 1 --workers 48` | `PASS` | 2026-08-09 | 1100/1100 records measured, 218,289 steps, zero skipped; result in §11 |
| `make lint PYTHON_QUALITY_PATHS="scripts/measure_expert_rulebook_costs.py tests/test_measure_expert_rulebook_costs.py"` | `PASS` | 2026-08-09 | Ruff clean |
| `make format-check PYTHON_QUALITY_PATHS="scripts/... tests/..."` | `PASS` | 2026-08-09 | Both files formatted (`make format` applied once to the script) |
| `make test` (full suite) | `FAIL` | 2026-08-09 | First run: 1347 passed, 44 failed — all in the new test file, caused by the dataclass/`sys.modules` loader defect described in §11. No pre-existing failure observed |
| `make test` (full suite, after the loader fix) | `PASS` | 2026-08-09 | **1391 passed, 0 failed**, 274 s |
| `git diff --check` | `PASS` | 2026-08-09 | No whitespace defects |
| `make rulebook-v2-check` | `NOT_RUN` | — | No Rulebook production change made yet; required at M2/M3 |
| `make smoke` | `NOT_RUN` | — | Deferred to M6; no production change to exercise yet |
| `make config`, `make config-gpu` | `NOT_RUN` | — | Required when `conf/env/scenarionet.yaml` changes at the tail milestone |

## 15. Final Reconciliation

Not reached. M0 is the only executed milestone. `DEC-RSEC-001` is decidable but
unresolved and gates M1; `DEC-RSEC-008` was opened by M0's own result. No
production code has been modified, so no requirement can yet be marked
`IMPLEMENTED`.

M0's own validity rests on two properties that were verified rather than
assumed: the measurement reuses the runtime's candidate selection, and the
parallel merge was shown byte-identical to the sequential run on the same
records. Its main limitation is that the R3 macro is an upper bound — the three
persistent sub-rules were summed as if disjoint, so the true R3 contribution is
at most `−26.5` and possibly less where they co-activate.

Known limitations already accepted:

- `wrong_direction`'s reference tangent is the ego's own route; far off-route it
  is not the local legal direction.
- The `q_rev` speed reinterpretation diverges from nuPlan on brief high-speed
  reverse bursts.
- With the terminal channel deferred, the early-off-road-exit shortcut is
  addressed only indirectly, through the R2 scale fix. If M0's fix proves
  insufficient, the deferral must be revisited rather than patched around.
- Sidewalk and guardrail impacts remain outside R1 (`is_traffic_object` covers
  only cones, barriers and traffic objects); explicitly not addressed here.
