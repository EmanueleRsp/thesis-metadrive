# Semantic Observation Causal-Correctness and Dead-Dimension Cleanup ExecPlan

**Plan ID:** PLAN-OBS-CORR-V1.3
**Status:** IMPLEMENTED — M0-M7 complete, awaiting final user sign-off
**Authoritative specification (current):** `docs/specifications/observation_v1.2_specification.md`
(OBS-V1.2, version `1.2-perception-bounded`, `APPROVED`, `Authoritative: YES`),
which retains the field definitions of
`docs/specifications/observation_v1.1_specification.md` (OBS-V1.1) for every
inherited group
**Authoritative specification (target, not yet drafted):** OBS-V1.3 + ENC-V1.3
**Related specifications:** ENC-V1.1, ENC-V1.2 (LQ variant), RULEBOOK-V4.7/4.8/4.9
**Related decisions:** ADR-022, ADR-026, ADR-033 (`docs/decisions/ADR-033-rulebook-latch-exclusion-from-policy-observation.md`, approved 2026-07-29)
**Created:** 2026-07-29
**Last update:** 2026-07-29 (implementation complete)
**Branch:** `scenarionet-implementation`

---

## 1. Objective

Restore the causal and geometric validity of the `semantic_v3` observation
(`PerceptionBoundedSemanticBatchBuilder`) and remove the dimensions that carry
no information.

The change is driven by a static review of the builder against OBS-V1.2 and the
inherited OBS-V1.1 field contracts. It identified three distinct classes of
defect:

1. **geometric defects** — features whose numeric value is not the physical
   quantity the contract names (lane width, boundary clearance, static
   dimensions);
2. **causality defects** — channels that expose either a Rulebook result or
   non-local/global knowledge to the policy;
3. **dead dimensions** — features that are constant, or exact duplicates of
   another feature already present in the same observation.

Success is recognised when: every policy-visible value is either derivable from
a perception-gated measurement, a declared map/mission prior, or ego-owned
memory; every named quantity matches its physical definition; and the
deterministic regression matrix below passes.

## 2. Scope

### 2.1 In scope

- `src/thesis_rl/envs/observations/causal_semantic.py`, both
  `CausalSemanticBatchBuilder` methods inherited by the V1.2 path and the
  `PerceptionBoundedSemanticBatchBuilder` overrides;
- `src/thesis_rl/contracts/observation_schema.py` group shapes and `flat_dim`
  (Phase B only);
- `src/thesis_rl/agent/planners/encoders/factory.py` per-group input dimensions
  (Phase B only);
- the OBS-V1.3 / ENC-V1.3 specification documents (Phase B only);
- deterministic unit and regression tests for every defect;
- `docs/project_index.md` authority rows.

### 2.2 Out of scope

- Rulebook v4.7/4.8/4.9 formulas, memory, timers, or component behaviour. The
  Rulebook keeps its internal latches; this plan only stops the *observation*
  from reading them.
- Calibrated sensor noise, learned perception, RGB, V2I/SPaT, multi-plane LiDAR.
- The legacy `semantic_v2` path and `CausalSemanticBatchBuilder.build`, which
  remain historical reproducible contracts. Shared private helpers touched by
  this plan (`_lane_width`, `_build_lane_road`, `_build_interactions`,
  `_approach_control`) are used by both paths — see `DEC-008`.
- Extending map ingestion with lateral lane adjacency (see `DEC-002`).
- Carrying a fine static sub-taxonomy through `ActorSnapshot` (see `DEC-003`).

### 2.3 Compatibility constraints

- Phase A changes no shape and no `flat_dim`, but **does** change the numeric
  meaning of values a trained policy consumes. Checkpoints remain structurally
  loadable and become semantically invalid.
- Phase B changes `flat_dim` from `3064` to `3009` and four group shapes.
  Checkpoints become structurally incompatible. OBS-V1.2 §13 already forbids
  automatic migration; OBS-V1.3 inherits that position.
- The latent-query raw token count stays `143` in both phases: only per-group
  feature widths change, not the group cardinalities.

## 3. Authoritative requirements

Requirement IDs are stable. The `C-item` column is the review label used in the
findings log so the two can be cross-referenced.

### 3.1 Geometric correctness (Phase A)

| ID | C-item | Requirement | Contract section |
|---|---|---|---|
| `REQ-001` | C1 | Lane width is the transverse width of the lane polygon at the queried point, invariant under rotation of the map frame | OBS-V1.1 §7.4, §7.6 ("Larghezza della lane associata"); OBS-V1.2 §6.3 |
| `REQ-002` | C2 | Left/right boundary clearance and type describe the **nearest** qualifying boundary on that side | OBS-V1.1 §7.6 |
| `REQ-003` | C3 | Boundary clearance is a signed footprint clearance: positive before contact, zero at contact, negative during overlap | OBS-V1.1 §7.6 (explicit) |
| `REQ-004` | C4 | The side assignment of a boundary uses the same point used for its distance | OBS-V1.1 §7.6; OBS-V1.2 §6.1 (closest point/segment for long geometries) |
| `REQ-005` | C5 | Static map-feature dimensions describe a local footprint window, not the bounding box of the whole geometry | OBS-V1.2 §6.1 |

### 3.2 Causality (Phase A)

| ID | C-item | Requirement | Contract section |
|---|---|---|---|
| `REQ-006` | C6 | No policy-visible value is expressed in the world frame | OBS-V1.2 §4; OBS-V1.1 §9.1 ("Le coordinate globali non vengono esposte") |
| `REQ-007` | C7 | No Rulebook decision, latch, or timer is policy-visible | OBS-V1.2 §1, §8, §12 |
| `REQ-008` | C8 | Pre-existing conflict-zone occupancy is reconstructed from current geometry plus ego-owned memory | OBS-V1.2 §4, §6.3 |
| `REQ-025` | C26 | Traffic-control association is limited to the local control horizon | OBS-V1.2 §6.2 ("not global knowledge of every scene control") |
| `REQ-026` | C27 | Continuity indicators are false when the previous sample is not step `k-1` | OBS-V1.2 §8 |
| `REQ-027` | C28 | Conflict-zone geometry is a function of map topology and the assigned ego route only, never of a specific actor's state | OBS-V1.2 §6.3 |

### 3.3 Consistency with the Rulebook and with the contract text (Phase A)

| ID | C-item | Requirement | Contract section |
|---|---|---|---|
| `REQ-009` | C9 | Route distances to a traffic control are measured from the ego front bumper | OBS-V1.1 §7.7; OBS-V1.2 §8 (twice: `compliance[22]`, yellow onset) |
| `REQ-010` | C10 | The active-control type is derived from the control record, never inferred from its signal-state string | OBS-V1.2 §8 rows `15:17` |
| `REQ-011` | C11 | A stop control emits the `not-signal` state; a non-observable signal emits an all-zero state with `state_valid = 0` | OBS-V1.1 §7.7 |
| `REQ-012` | C12 | Route lateral offset preserves the left/right sign | OBS-V1.1 §7.5, §8.2 — see `DEC-006` |

### 3.4 Selection and instrumentation (Phase A)

| ID | C-item | Requirement | Contract section |
|---|---|---|---|
| `REQ-013` | C13 | Interaction tokens are ranked by criticality, using only causally available criteria | OBS-V1.1 §8.4; OBS-V1.2 §7 |
| `REQ-014` | C14 | Overflow diagnostics exist for dynamic, static, controls and interactions | OBS-V1.1 §8.5 |
| `REQ-015` | C15 | Interaction entry/exit distances are the true route-curvilinear limits of the zone | OBS-V1.1 §7.8 |
| `REQ-016` | C16 | Control ranking prefers controls associated with the ego approach lane | OBS-V1.1 §8.3 criteria 1-2 (criterion 3 is excluded, see `DEC-009`) |

### 3.5 Dead dimensions and hygiene (Phase B, except `REQ-022`/`REQ-023`)

| ID | C-item | Requirement | Contract section |
|---|---|---|---|
| `REQ-017` | C17, C18 | The control token carries no value that is a deterministic function of `yellow_onset_memory` | New in OBS-V1.3 |
| `REQ-018` | C20 | The compliance row carries no value identical by construction to another value in the same row | New in OBS-V1.3 |
| `REQ-019` | C7, C19 | The interaction token carries no Rulebook latch and no duplicate of its own zone-type one-hot | New in OBS-V1.3 |
| `REQ-020` | C21 | The lane/road group carries no permanently constant field | New in OBS-V1.3 |
| `REQ-021` | C22 | The static type one-hot encodes a taxonomy the source can actually discriminate | OBS-V1.2 §7 (restated in OBS-V1.3) |
| `REQ-022` | C23, C25 | The V1.2 builder contains no unreachable or non-functional method and no comment describing absent logic | Repository convention |
| `REQ-023` | C24 | A non-projectable static actor degrades the affected token, never the whole observation | OBS-V1.2 §11 strict-failure policy, read together with the route-projection diagnostics precedent |

## 4. Current repository analysis

All statements below are `VERIFIED` by direct reading unless labelled otherwise.
Line numbers refer to `src/thesis_rl/envs/observations/causal_semantic.py` at
commit `eaa520c` unless another file is named.

### 4.1 Call flow of the V1.2 path

`PerceptionBoundedSemanticBatchBuilder.build` (`:1412`) calls, in order:
`_context_v12` → `_record_ego_frame` (inherited, `:320`) → `_build_ego_history`
(inherited, `:395`) → `_build_route_v12` (`:1472`) → `_build_dynamic_v12`
(`:1540`) → `_build_static_v12` (`:1600`) → `_build_lane_road`
(**inherited**, `:997`) → `_build_controls_v12` (`:1694`) →
`_build_interactions` (**inherited**, `:1142`) → `_append_compliance_row`
(`:1882`).

`_build_lane_road` and `_build_interactions` are **not** overridden. Every
lane/road and interaction defect in this plan is therefore V1.1 code that was
never revised for V1.2, and any fix is shared with `semantic_v2` (see
`DEC-008`).

### 4.2 Defect evidence

| C-item | Location | Verified behaviour |
|---|---|---|
| C1 | `:466`, `:479` | `polygon_xy.bounds[3] - bounds[1]` is the world-frame Y extent. Correct only for lanes roughly parallel to the X axis; for a north-south lane it returns the lane *length*, which `_clip(..., 6.0)` saturates to `1.0`. Feeds `lane_road[0]`, `route[:,5]`, and the `1.5 * lane_width` threshold in `_lane_relation` (`:659`) |
| C2 | `:1023` | `if signed < boundaries[side][0] * 50.0`. The `* 50.0` matches the normalized sentinel `1.0` only on the first iteration; afterwards the slot holds metres and the threshold becomes ~50x, so every later candidate wins. The retained value is the last iterated feature within 10 m, in `map_feature_catalog` iteration order |
| C3 | `:1022` | `-distance` where `shapely.distance` is `0.0` for intersecting geometries, so the value is `-0.0`. The negative branch required by OBS-V1.1 §7.6 is unreachable |
| C4 | `:1009` vs `:1010` | Side from `representative_point()` of the whole geometry, distance from `geometry.distance(ego.footprint)`. On a long boundary the two can fall on opposite sides |
| C5 | `:1654`, `:1660` | `feature.geometry.bounds` of the entire geometry. Inconsistent with actors, which use `_dimensions` / `minimum_rotated_rectangle` (`:133`) |
| C6 | `:1659` + `:1682` | Map features are emitted with `heading = 0.0`; the payload then computes `_heading_sincos(0.0 - ego.heading_rad)`, i.e. `sin(-psi_ego), cos(-psi_ego)` — the ego world heading |
| C7 | `:1303`-`:1307` | Reads `context.memory.preexisting_ego_occupancy_zone_ids`, `vehicle_yield_illegal_entries`, `crosswalk_illegal_entries`. `context.memory` is `RulebookMemory` (`src/thesis_rl/contracts/causal_scene_context.py:21`) |
| C9 | `:1735`, `:1794`, `:1857` | `control.route_s_m - current_s` with `current_s` from the ego **centre** projection (`:1418`). The Rulebook decides control-line crossing with the swept front bumper (`src/thesis_rl/rulebook/v2/transition.py:452`) |
| C10 | `:1736`-`:1750` + `:1855` | A non-observable SIGNAL keeps `state = "not-signal"`; the compliance row types the control with `state != "not-signal"`, yielding index 1 = stop |
| C11 | `:1752` | A stop control emits five zeros; OBS-V1.1 §7.7 places `not-signal` at index 4, which the V1.1 builder still does (`:1086`) |
| C12 | `:698`, `:1687` | `_clip(lateral, 50.0, lower=0.0)` on a signed quantity (`src/thesis_rl/rulebook/v2/geometry/route.py:160`) collapses the entire right half-plane to `0.0` |
| C13 | `:1226` | `sort(key=lambda item: (item.zone_id, item.actor.actor_id))` — only criterion 8 of OBS-V1.1 §8.4 |
| C14 | `:1565`-`:1573` | `SemanticOverflowDiagnostics` is populated with a `"dynamic"` key only |
| C15 | `:1219` | Crosswalk `route_entry_s_m` and `route_exit_s_m` are both the centroid projection |
| C17 | `:1769`, `:1804`-`:1807` | `_yellow_required_stop_distance` is `v*dt + v^2/(2a)` with constant `a`, a bijection of `_yellow_onset_speed_mps`, which is `yellow_onset_memory[2]` (`:1814`) |
| C18 | `:1769` vs `:1813` | The same `self._yellow_onset_distance_m` scalar, clipped identically |
| C19 | `:1302` vs `:1261` | `float(candidate.zone_type_index == 3)` is bit 3 of the zone-type one-hot |
| C20 | `:1853` vs `:1858` | `controls_ego` in `active_trace` is `True` iff `is_active` fired (`:1753`), i.e. iff `control_id is not None`. Identical to `float(active_control)` |
| C21 | `:1032`-`:1033` | Two hardcoded `0.0`. `EpisodeCache` (`src/thesis_rl/rulebook/v2/types.py:339`) and `RouteLaneRecord` (`src/thesis_rl/rulebook/v2/geometry/lanes.py:38`) expose no lateral adjacency, only `successor_lane_ids` |
| C22 | `:1615`, `:1661` | Every static candidate is emitted with type index 4. `_actor_class` (`src/thesis_rl/rulebook/v2/context/metadrive_live.py:80`) does discriminate `cone`/`barrier`/`building` before collapsing them into `ActorClass.STATIC_COLLIDABLE` |
| C23 | `:928`-`:995`, `:874`-`:914`, `:1894` | `_build_compliance_v12` references attributes that do not exist on either class (`_previous_dashed_key`, `_compliance_history`, `_compliance_steps`, `_yellow_group_id`): it is dead **and** non-functional. The base `_build_controls_v12` is shadowed by the subclass override. `_active_dashed_feature_id` is never called, while the inline version actually used (`:1829`) takes an arbitrary first match and applies no elevation filter |
| C24 | `:547`-`:559` + `:84`-`:87` | `_route_s`/`_route_lateral` return `-inf`/`+inf` on projection failure; `_clip` raises on non-finite input. The live-static branch (`:1616`) has no guard, the map-feature branch (`:1637`) does |
| C25 | `:657` | The comment describes an adjacency-record lookup that does not exist in the function |
| C26 | `:1121`-`:1124` | Iterates the whole `episode_cache.traffic_control_catalog` with no range or elevation filter, unlike `_build_controls_v12` (`:1705`) |
| C27 | `:1851`, `:1854` | Continuity compares against the previous `build` call. `self._last_compliance_step` exists (`:1355`) but is never read |
| C29 | `:1255` vs `:1305` | **Tuple-order mismatch.** The observation builds `pair = (zone_id, actor.actor_id)`; both Rulebook latch sets are keyed `(actor_id, zone_id)` (`components/controls.py:435` `illegal_keys.add((actor_id, zone_id))`, `transition.py:495` and `:727` `for actor_id, zone_id in ...`). The membership test can only succeed when `zone_id == actor_id`, so `incompatible entry latched` is **constant `0.0` at runtime**. Discovered 2026-07-29 while resolving `DEC-005`; it downgrades the severity of `C7` but not the decision — see §11 |

### 4.3 Verified-clean findings

Recorded so the reasoning is not repeated later.

- **Conflict-zone provenance is causally clean, but only incidentally.** Zones
  are built by iterating every actor in the snapshot with no visibility filter
  (`src/thesis_rl/rulebook/v2/transition.py:877`). This does not leak, because
  (a) the zone polygon is a function of the ego corridor and the approach lane
  polygon only — the actor is a computation trigger, not an input to the
  geometry (`transition.py:920`-`:934`) — and (b) `_build_interactions` emits a
  token only for a currently visible actor whose `live_lane_id` matches the
  approach lane (`:1150`), for which the pair would be computed at that same
  step regardless. The property is load-bearing for the causality claim and is
  currently protected by no test: hence `REQ-027`.
- **Movement priorities are a declared map prior, not intent.** They are decoded
  from explicit scenario metadata `rulebook_vehicle_yield`
  (`src/thesis_rl/rulebook/v2/context/static_adapter.py:59`), never from future
  tracks. `derive_lane_movement_key`
  (`src/thesis_rl/rulebook/v2/geometry/lanes.py:53`) returns `None` when a lane
  has several successors, so another actor's exit lane is exposed only when it
  is topologically forced.
- **The perception gate itself is structurally sound.** `_dynamic_candidates`
  (`:1463`), `_build_static_v12` (`:1605`) and `commit_context` (`:1386`) all
  filter on `self._visible_actor_ids`; `_zone_actor_ids` and
  `_build_interactions` both go through `_dynamic_candidates`; signal states go
  through `mapped_signal_visibility` (`:1721`). Occlusion produces masked gaps,
  covered by `tests/test_perception_bounded_semantic.py`.
- **`ego speed` in the compliance row is correct.** OBS-V1.2 §8 row 0 says
  "ego speed"; `hypot(*ego.velocity_xy)` conforms. No change required.

### 4.4 Directly relevant debt

- `VERIFIED` 2026-07-29 (second check, after a numbering collision) — a
  concurrent line of work created `ADR-032-acl-generate-catalog-decoupling.md`
  in the same working tree while this plan was being written, so this record was
  renumbered to `ADR-033`. Any further ADR from this plan must re-check the
  directory immediately before creation.
- `VERIFIED` 2026-07-29 — `docs/decisions/` still contains two `ADR-025-*`
  files, and two former `ADR-028-*` duplicates appear renamed to `ADR-030` and
  `ADR-031` as staged, uncommitted index entries not produced by this work. The
  next free number is therefore `ADR-033`, which this plan uses; it must be
  re-verified at creation time because the renames are not yet committed.
- `AWAITING_CONFIRMATION` — the crosswalk interaction branch (`:1177`-`:1191`)
  applies an elevation filter but no range filter. Bounded in practice because
  the paired actor must be a visible dynamic candidate (<= 50 m), so no leak is
  demonstrated. Recorded, not scheduled.
- `docs/decisions/` contains three distinct `ADR-028-*` files. Not caused by
  this work and not addressed here; the new record must therefore be numbered
  `ADR-033` and its number verified at creation time.

## 5. Assumptions and invariants

| Item | Value | How established | Violation handling |
|---|---|---|---|
| Control period | `dt = 0.1 s` | OBS-V1.2 §4 | `control_timestep_s <= 0` already rejected (`:185`) |
| Frame | Ego frame at step `k`: `x` forward, `y` left, angles in `[-pi, pi]` | OBS-V1.2 §4 | `REQ-006` makes this testable for the static group |
| Units | metres, m/s, rad, rad/s | OBS-V1.2 §4 | `_finite`/`_clip` raise on non-finite (`:84`, `:91`) |
| Route lateral sign | Positive to the left of the route tangent | Verified from `route.py:160` (cross-product form) | `REQ-012` |
| Dynamic history | Five positions mapping to exact steps `k-4..k`, absent samples zeroed and masked | OBS-V1.2 §7 | Already implemented at `:1554`-`:1564`; must not regress |
| Compliance history | 21 positions mapping to exact steps `k-20..k` | OBS-V1.2 §8 | Already implemented at `:1870`-`:1879`; `REQ-026` adds continuity contiguity |
| `flat_dim` | `3064` in Phase A; `3009` in Phase B | OBS-V1.2 §5; computed in §7.4 | Schema validation in `observation_schema.py` |
| Raw LQ tokens | `143`, unchanged in both phases | OBS-V1.2 §10 | ENC test suite |
| Reset semantics | Every episode-local cache cleared on scenario change | `reset` (`:1357`) | New builder-owned state from `REQ-008` and `REQ-026` must be added to `reset` |
| Determinism | Ranking keys must be total orders with a stable ID tie-break | OBS-V1.2 §7 | `REQ-013`, `REQ-016` |

## 6. Decisions and approval gates

No production code may be modified for a requirement whose gate is unresolved.

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-001` | Specification deviation | Phase A alone (`D` unchanged) or Phase A + Phase B in one intervention | A: Phase A now, Phase B later — two restarts. B: both, one restart | **B**. Phase A already changes the numeric meaning of features a trained policy consumes, so a restart is forced regardless; the marginal cost of the dimension change is code work with zero extra compute | Determines whether OBS-V1.3/ENC-V1.3 are drafted now | **APPROVED 2026-07-29 — option B**: both phases, one restart |
| `DEC-002` | Specification clarification | `lane_road[11:13]` adjacent-lane availability is permanently `0.0` | A: remove the two dimensions. B: extend map ingestion with lateral adjacency from the ScenarioNet lane schema | **A**. Verified that neither `EpisodeCache` nor `RouteLaneRecord` carries lateral adjacency; option B is an upstream work item of its own, justified only when lane-change behaviour is studied | `REQ-020`; `-2` dimensions | **APPROVED 2026-07-29 — option A** |
| `DEC-003` | Specification clarification | Static type one-hot is constant | A: redefine the five slots as `{live detected obstacle, road boundary, other non-drivable, stationary vehicle, unknown}`. B: additionally carry `cone`/`barrier`/`wall` through a new `ActorSnapshot` field | **A**. It restores the distinction that actually governs how the rest of the token is interpreted, at zero dimensional and contract cost. B modifies a Rulebook contract type for a taxonomy the Rulebook itself does not use (`injury_risk.py:88` maps all statics to one curve) | `REQ-021`; `0` dimensions | **APPROVED 2026-07-29 — option A** |
| `DEC-004` | Blocking technical issue | The three production runs | A: stop all three before implementation. B: let `td3-0` continue | **A**. `sac-0` and `ppo-0` are already stopped since ENC-V1.2; `td3-0` would produce results under an observation contract that no longer exists | Experiment continuity | **APPROVED 2026-07-29 — option A**: all three runs stopped before implementation |
| `DEC-005` | Specification deviation | OBS-V1.1 §7.8 lists `Incompatible entry latched` as contractual; OBS-V1.2 §1/§12 forbids any Rulebook latch | A: OBS-V1.2 §12 wins, feature removed, contradiction recorded in an ADR. B: OBS-V1.2 §12 amended to allow it | **A**. The bit is a function of the violation the Rulebook is about to penalise: with it active, "learned to yield" and "learned to read the flag" are experimentally indistinguishable | `REQ-007`, `REQ-019`; ADR-033 | **APPROVED 2026-07-29 — option A**, recorded in ADR-033. Re-explained before approval. `C29` shows the latch read is inert at runtime, so no past experiment is contaminated; the decision is unchanged because the coupling to `RulebookMemory` is still present in the source and would become live the moment the tuple order were corrected |
| `DEC-006` | Specification clarification | Route lateral distance: signed, absolute, or clamped | A: signed, spec text updated to "signed lateral offset". B: `abs()`, spec text unchanged | **A**. Left/right of the corridor is decision-relevant for overtaking, merging and conflict resolution. The current clamped form is the only one of the three that destroys information | `REQ-012`; spec text | **APPROVED 2026-07-29 — option A**: signed |
| `DEC-007` | Implementation detail | Geometric definition of negative boundary clearance | A: split the ego footprint by the boundary line, take the piece not containing the ego centroid, report the negated maximum distance of its vertices from the line. B: negated intersection area / chord length | **A**. On a convex footprint the maximum distance from a line is attained at a vertex, so A is exact and constant-cost. B is not a length | `REQ-003`; spec text must pin the chosen definition | **DECIDED 2026-07-29 (delegated) — option A.** Classified as an implementation detail; the exact definition is pinned in the OBS-V1.3 text and in `AC-003` |
| `DEC-008` | Specification deviation | `_build_lane_road`, `_build_interactions`, `_lane_width` and `_approach_control` are shared with the legacy `semantic_v2` path | A: fix in place, accepting that `semantic_v2` behaviour changes. B: override the fixed versions in `PerceptionBoundedSemanticBatchBuilder`, freezing `semantic_v2` | **B**. OBS-V1.2 §12 requires that "existing legacy observation modes retain their historical contracts"; `semantic_v2` experiments must stay reproducible. Cost: four additional overrides and a test asserting `semantic_v2` output is byte-identical before and after | All Phase A requirements touching those four methods | **DECIDED 2026-07-29 (delegated) — option B.** Evidence: `conf/config.yaml:7` still selects `obs: semantic_v2` as the repository default, and `SemanticStateObservationV2` consumes the shared base builder through `set_batch_builder` (`thesis_scenario_env.py:427`-`:432`). Fixing in place would silently change the default observation. The four overrides are marked for deletion when `semantic_v2` is retired |
| `DEC-009` | Specification deviation | OBS-V1.1 §8.3 criterion 3 ("non ancora risolto") for control ranking | A: permanently drop it. B: reconstruct it from ego-owned memory | **A**. It was implemented by reading `context.memory.resolved_*_group_ids` (`:1062`-`:1065`), a Rulebook latch. Reconstructing it builder-side would recreate a rule-aligned shortcut for marginal ranking benefit | `REQ-016`; spec text | **DECIDED 2026-07-29 (delegated) — option A**: criterion permanently dropped |

## 7. Proposed design

### 7.1 Geometric corrections

- **`REQ-001`** — replace `_lane_width` with a local measurement: project the
  query point onto `RouteLaneRecord.centerline`, build a segment along the
  normal, intersect it with `polygon_xy`, return the intersection length.
  Deterministic fallback `polygon.area / centerline.length_m` when the
  intersection degenerates; the existing `navigation.get_current_lane_width`
  fallback and the final `CausalSemanticObservationError` are preserved.
- **`REQ-002`/`REQ-004`** — sentinel `float("inf")`, comparison `signed < best`,
  final mapping `inf -> 50.0` with type index 3. Side and distance both taken
  from `nearest_points(feature.geometry, ego.footprint)`.
- **`REQ-003`** — per `DEC-007`.
- **`REQ-005`** — `feature.geometry.intersection(Point(ego_xy).buffer(R))` with
  `R = 10.0` m before `_dimensions`; on an empty intersection the token keeps
  the nearest-point position and reports zeroed dimensions.

### 7.2 Causality corrections

- **`REQ-006`** — replace the constant `0.0` heading of map features with the
  orientation of the nearest boundary segment, still expressed as
  `_heading_sincos(theta_local - ego.heading_rad)`. This removes the world-frame
  leak *and* replaces a constant with a real physical quantity.
- **`REQ-007`** — drop the two latch reads. In Phase A the fields are written as
  constant `0.0`; in Phase B the dimensions are removed.
- **`REQ-008`** — new builder-owned `dict[str, bool]`: the first time a zone id
  enters the candidate set, record `ego.footprint.intersects(zone.polygon)`;
  keep the flag while the zone remains a candidate; clear on `reset`.
- **`REQ-025`** — `_approach_control` gains the `control_radius_m` and
  `vertical_tolerance_m` filters already used by `_build_controls_v12`, and
  returns `ApproachControl.UNKNOWN` outside the horizon.
- **`REQ-026`** — read `self._last_compliance_step`; continuity indicators are
  `0.0` unless the previous row was step `k-1`.
- **`REQ-027`** — regression test only, no production change.

### 7.3 Consistency corrections

- **`REQ-009`** — one helper `_front_bumper_s(ego)` projecting
  `p_ego + (L/2) * u(psi_ego)` onto the route, used by the control token, the
  compliance row and the yellow-onset latch.
- **`REQ-010`** — `active_trace` carries `control.control_type` instead of the
  type being inferred from the state string.
- **`REQ-011`** — restore index 4 for stop controls; an unobservable signal keeps
  the all-zero vector with `state_valid = 0`.
- **`REQ-013`** — ranking key: `(not ego_in, not other_in, not intervals_overlap,
  not preexisting, entry_distance, t_in, actor_distance, zone_id, actor_id)`.
  All terms derive from values the token already computes.
- **`REQ-015`** — intersect the route polyline with the zone polygon, take the
  minimum and maximum `s` of the intersection; fall back to the centroid
  projection with entry `==` exit only when the intersection is empty.

### 7.4 Phase B dimensional change

| Group | Current | New | Delta | Removed |
|---|---|---|---|---|
| `lane_road` | `(14,)` | `(12,)` | `-2` | adjacent-lane availability x2 |
| `control` | `(8, 17)` | `(8, 15)` | `-16` | yellow-onset distance, required stopping distance |
| `interaction` | `(8, 35)` | `(8, 33)` | `-16` | incompatible-entry latch, roundabout-relation duplicate |
| `compliance_history` | `(21, 24)` | `(21, 23)` | `-21` | "control governs ego movement" |

`3064 - 55 = 3009`. Raw LQ tokens stay at `143`.

Rejected alternative: keeping `D = 3064` and writing constant zeros in the freed
slots. It preserves checkpoint shape but not checkpoint validity, so it buys
nothing while leaving 55 dimensions to explain in the thesis.

## 8. Traceability

Populated during implementation; `Status` starts at `Planned` for every row.

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-001` | `AC-001` | `causal_semantic.py::_lane_width` | `TEST-001`, `TEST-002` | Planned |
| `REQ-002` | `AC-002` | `causal_semantic.py::_build_lane_road` | `TEST-003` | Planned |
| `REQ-003` | `AC-003` | `causal_semantic.py::_build_lane_road` | `TEST-004` | Planned |
| `REQ-004` | `AC-004` | `causal_semantic.py::_build_lane_road` | `TEST-005` | Planned |
| `REQ-005` | `AC-005` | `causal_semantic.py::_build_static_v12` | `TEST-006` | Planned |
| `REQ-006` | `AC-006` | `causal_semantic.py::_build_static_v12` | `TEST-007`, `TEST-008` | Planned |
| `REQ-007` | `AC-007` | `causal_semantic.py::_build_interactions` | `TEST-009` | Planned |
| `REQ-008` | `AC-008` | `causal_semantic.py::_build_interactions`, `reset` | `TEST-010`, `TEST-011` | Planned |
| `REQ-009` | `AC-009` | `causal_semantic.py::_front_bumper_s` | `TEST-012`, `TEST-013` | Planned |
| `REQ-010` | `AC-010` | `causal_semantic.py::_build_controls_v12` | `TEST-014` | Planned |
| `REQ-011` | `AC-011` | `causal_semantic.py::_build_controls_v12` | `TEST-015` | Planned |
| `REQ-012` | `AC-012` | `causal_semantic.py::_dynamic_features`, `_build_static_v12` | `TEST-016` | Planned |
| `REQ-013` | `AC-013` | `causal_semantic.py::_build_interactions` | `TEST-017`, `TEST-018` | Planned |
| `REQ-014` | `AC-014` | `causal_semantic.py`, `SemanticOverflowDiagnostics` | `TEST-019` | Planned |
| `REQ-015` | `AC-015` | `causal_semantic.py::_build_interactions` | `TEST-020` | Planned |
| `REQ-016` | `AC-016` | `causal_semantic.py::_build_controls_v12` | `TEST-021` | Planned |
| `REQ-017` | `AC-017` | `observation_schema.py`, `causal_semantic.py`, `encoders/factory.py` | `TEST-022`, `TEST-026` | Planned |
| `REQ-018` | `AC-018` | same | `TEST-023`, `TEST-026` | Planned |
| `REQ-019` | `AC-019` | same | `TEST-024`, `TEST-026` | Planned |
| `REQ-020` | `AC-020` | same | `TEST-025`, `TEST-026` | Planned |
| `REQ-021` | `AC-021` | `causal_semantic.py::_build_static_v12` | `TEST-027` | Planned |
| `REQ-022` | `AC-022` | `causal_semantic.py` | `TEST-028` | Planned |
| `REQ-023` | `AC-023` | `causal_semantic.py::_build_static_v12` | `TEST-029` | Planned |
| `REQ-025` | `AC-025` | `causal_semantic.py::_approach_control` | `TEST-030` | Planned |
| `REQ-026` | `AC-026` | `causal_semantic.py::_append_timestamped_compliance_row` | `TEST-031` | Planned |
| `REQ-027` | `AC-027` | none (property test) | `TEST-032` | Planned |
| all Phase A | `AC-028` | `causal_semantic.py` | `TEST-033` | Planned |

## 9. Test strategy

### 9.1 Acceptance criteria

- `AC-001` Lane width is invariant when the whole fixture map is rotated by 90
  degrees; the value equals the analytic width within `1e-6` m.
- `AC-002` With three boundaries on the same side at 1, 3 and 7 m, the reported
  clearance is 1 m and the type is that of the 1 m boundary, for every catalog
  insertion order.
- `AC-003` A footprint overlapping a boundary by `d` reports `-d` within
  `1e-6` m; a footprint exactly touching reports `0.0`.
- `AC-004` A 200 m boundary whose representative point lies on the opposite side
  of the ego from its nearest point is assigned the side of the nearest point.
- `AC-005` A 200 m guard-rail reports local dimensions bounded by the 10 m
  window, not the full extent.
- `AC-006` Rotating the ego heading with the scene fixed leaves every static
  map-feature heading pair unchanged; the emitted heading equals the local
  boundary tangent relative to the ego.
- `AC-007` A context whose `RulebookMemory` carries non-empty
  `crosswalk_illegal_entries` and `vehicle_yield_illegal_entries` produces a
  tensor bit-identical to the same context with an empty memory.
- `AC-008` A zone the ego already occupies at the first observation reports
  pre-existing occupancy; a zone the ego enters later does not; `reset` clears
  the flag.
- `AC-009` For an ego of length `L` at distance `d` from a control line, the
  control token, the compliance row and the yellow-onset latch all report
  `d - L/2`.
- `AC-010` An occluded signal produces control type `signal`, state one-hot all
  zero, `state_valid = 0`.
- `AC-011` A stop control produces `not-signal` at index 4 with
  `state_valid = 1`.
- `AC-012` An actor `y` metres to the right of the route reports a negative
  lateral offset of the same magnitude as an actor `y` metres to the left.
- `AC-013` With nine candidates of which the ninth alphabetically is the only
  one with the ego inside the zone, that candidate is selected.
- `AC-014` `SemanticOverflowDiagnostics` reports totals, selected, dropped and
  ranking keys for all four groups.
- `AC-015` A crosswalk of route extent `[s0, s1]` reports entry `s0` and exit
  `s1`, with `s1 > s0`.
- `AC-016` A control on the ego approach lane at 60 m ranks before a control not
  on the ego lane at 20 m.
- `AC-017` to `AC-021` The observation matches the OBS-V1.3 shapes, `flat_dim`
  is `3009`, raw tokens are `143`, and the encoder builds and runs a forward
  pass at the new dimensions.
- `AC-022` No unreachable method remains in the V1.2 builder; the dashed-feature
  lookup applies the elevation filter and selects the nearest feature.
- `AC-023` A static actor whose route projection fails is excluded from the
  static tokens; the observation is still produced and finite.
- `AC-025` A control 200 m from the ego does not set another actor's approach
  control; the value is `unknown`.
- `AC-026` After a step gap, both continuity indicators are `0.0`.
- `AC-027` Two scenarios identical in map and route but with different vehicles
  on the same approach lane produce identical conflict-zone geometry.
- `AC-028` `semantic_v2` output is bit-identical before and after the change
  (per `DEC-008`).

### 9.2 Mandatory test matrix

Frozen before any production change. `tests/test_perception_bounded_semantic.py`
is the primary home; schema and encoder cases go to
`tests/test_observation_schema_v12.py` (renamed in Phase B) and
`tests/test_encoders_v11.py`.

| ID | Level | Behaviour | Fixture | Expected | Requirement |
|---|---|---|---|---|---|
| `TEST-001` | Unit | Lane width, axis-aligned lane | Synthetic rectangular lane | Analytic width | `REQ-001` |
| `TEST-002` | Unit | Lane width, rotation invariance | Same fixture rotated 90 deg | Identical value | `REQ-001` |
| `TEST-003` | Unit | Nearest boundary wins in any order | Three boundaries, both insertion orders | 1 m boundary | `REQ-002` |
| `TEST-004` | Unit | Negative clearance on overlap | Footprint crossing a line by 0.4 m | `-0.4` | `REQ-003` |
| `TEST-005` | Unit | Side from nearest point | 200 m diagonal boundary | Correct side | `REQ-004` |
| `TEST-006` | Unit | Local static dimensions | 200 m guard-rail | Window-bounded | `REQ-005` |
| `TEST-007` | Causality | No world heading | Two ego headings, same scene | Equal static headings | `REQ-006` |
| `TEST-008` | Unit | Local tangent heading | Curved boundary | Tangent at nearest point | `REQ-006` |
| `TEST-009` | Causality | Rulebook latch invariance | Populated vs empty `RulebookMemory` | Identical tensors | `REQ-007` |
| `TEST-010` | Unit | Pre-existing occupancy from geometry | Ego spawned inside a zone | Flag set | `REQ-008` |
| `TEST-011` | State/reset | Flag cleared on reset | Scenario change | Flag cleared | `REQ-008` |
| `TEST-012` | Unit | Front-bumper distance | Known `L`, known `d` | `d - L/2` | `REQ-009` |
| `TEST-013` | Integration | Same value in all three consumers | Single step | Three equal values | `REQ-009` |
| `TEST-014` | Regression | Occluded signal is not a stop | Occluded signal fixture | Type `signal`, state zero | `REQ-010` |
| `TEST-015` | Regression | Stop emits `not-signal` | Stop fixture | Index 4 set | `REQ-011` |
| `TEST-016` | Regression | Signed lateral offset | Actors left and right | Opposite signs | `REQ-012` |
| `TEST-017` | Unit | Critical interaction retained | 9 candidates | Critical one selected | `REQ-013` |
| `TEST-018` | Determinism | Stable order across runs | Shuffled inputs | Identical order | `REQ-013` |
| `TEST-019` | Unit | Diagnostics for four groups | Overflow in each | All keys present | `REQ-014` |
| `TEST-020` | Unit | Real crosswalk entry/exit | Crosswalk across the route | `s1 > s0` | `REQ-015` |
| `TEST-021` | Unit | Ego-lane control ranks first | Two controls | Ego-lane one first | `REQ-016` |
| `TEST-022` | Compatibility | Control token width 15 | Any build | Shape `(8, 15)` | `REQ-017` |
| `TEST-023` | Compatibility | Compliance width 23 | Any build | Shape `(21, 23)` | `REQ-018` |
| `TEST-024` | Compatibility | Interaction width 33 | Any build | Shape `(8, 33)` | `REQ-019` |
| `TEST-025` | Compatibility | Lane/road width 12 | Any build | Shape `(12,)` | `REQ-020` |
| `TEST-026` | Compatibility | `flat_dim` and token count | Any build | `3009`, `143` | `REQ-017`-`REQ-020` |
| `TEST-027` | Unit | Static taxonomy discriminates | Live obstacle + road boundary | Different indices | `REQ-021` |
| `TEST-028` | Unit | Nearest dashed feature, elevation filtered | Two dashed features | Nearest, correct level | `REQ-022` |
| `TEST-029` | Missing data | Non-projectable static degrades | Off-route static actor | Token dropped, build succeeds | `REQ-023` |
| `TEST-030` | Causality | Distant control not exposed | Control at 200 m | `unknown` | `REQ-025` |
| `TEST-031` | Boundary | Continuity false after a gap | Steps `k-2`, `k` | Both `0.0` | `REQ-026` |
| `TEST-032` | Causality | Zone geometry is actor-independent | Two vehicle sets, same lane | Identical geometry | `REQ-027` |
| `TEST-033` | Compatibility | `semantic_v2` unchanged | Legacy fixture | Bit-identical | `DEC-008` |

Existing tests that must keep passing unchanged: the occlusion-gap and
fail-closed adjacent-lane cases in `tests/test_perception_bounded_semantic.py`,
the first-hit preflights in `tests/test_perception_first_hit_preflight.py`, and
the checkpoint-manifest schema identity cases.

### 9.3 Commands

Run from the repository root.

- Focused suite, inside a provisioned container:
  `uv run --no-sync python -m pytest -q tests/test_perception_bounded_semantic.py tests/test_causal_semantic_batch.py tests/test_observation_schema_v12.py tests/test_encoders_v11.py tests/test_semantic_state_v3.py`
- Full suite through the primary environment: `make test`
- Rulebook regression for `REQ-027`: `make rulebook-v2-check`
- Lint, focused: `make lint PYTHON_QUALITY_PATHS="src/thesis_rl/envs/observations/causal_semantic.py src/thesis_rl/contracts/observation_schema.py tests/test_perception_bounded_semantic.py"`
- Format check, focused: `make format-check PYTHON_QUALITY_PATHS="<same paths>"`
- End-to-end smoke: `make smoke`
- Whitespace: `git diff --check`

Type checking: no global mypy target exists in this repository; annotations are
added to the new and materially modified functions, without inventing a command.

## 10. Milestones

### M0 — Decision gate — `COMPLETE` (2026-07-29)

- [x] `DEC-001` to `DEC-009` resolved: `DEC-001`, `DEC-002`, `DEC-003`,
      `DEC-004`, `DEC-005`, `DEC-006` approved by the user; `DEC-007`,
      `DEC-008`, `DEC-009` decided under explicit delegation
- [x] ADR-033 written and approved, covering `DEC-005` and `DEC-009`
- [ ] Production runs stopped per `DEC-004` — user-side action, pending
      confirmation before M2 touches production code

No production change before M0 closes. Only the test scaffolding of M1 may be
written in advance.

### M1 — Frozen test matrix — `COMPLETE`

Objective: every `TEST-*` exists and fails for the documented reason.
Files: `tests/test_perception_bounded_semantic.py`,
`tests/test_observation_schema_v12.py`, `tests/test_causal_semantic_batch.py`.
Depends on: `DEC-008` (whether `TEST-033` is required).

### M2 — Geometry — `COMPLETE`

`REQ-001` to `REQ-005`. Depends on `DEC-007`, `DEC-008`.
Evidence: `TEST-001` to `TEST-006` pass; focused suite green.

### M3 — Causality — `COMPLETE`

`REQ-006`, `REQ-007`, `REQ-008`, `REQ-025`, `REQ-026`, `REQ-027`.
Depends on `DEC-005`.
Evidence: `TEST-007` to `TEST-011`, `TEST-030` to `TEST-032` pass;
`make rulebook-v2-check` green.

### M4 — Rulebook consistency — `COMPLETE`

`REQ-009` to `REQ-012`, `REQ-016`. Depends on `DEC-006`.
Evidence: `TEST-012` to `TEST-016`, `TEST-021` pass.

### M5 — Selection and instrumentation — `COMPLETE`

`REQ-013`, `REQ-014`, `REQ-015`, plus hygiene `REQ-022`, `REQ-023`.
Evidence: `TEST-017` to `TEST-020`, `TEST-028`, `TEST-029` pass.

End of Phase A. `flat_dim` still `3064`. Full suite and `make smoke` recorded
here, so Phase A is independently shippable if `DEC-001` resolves to option A.

### M6 — OBS-V1.3 and ENC-V1.3 specifications — `COMPLETE`

Objective: write both documents, obtain explicit approval, set `APPROVED` /
`Authoritative: YES`, move them into `docs/specifications/`, update
`docs/project_index.md`. Depends on `DEC-001`, `DEC-002`, `DEC-003`, `DEC-005`,
`DEC-006`, `DEC-007`, `DEC-009`.

Beyond the dimension table, the specification text must also correct the
statements the review found to be wrong about the *implementation intent*:
`Lane offset` and `Local lane curvature` are assigned-route quantities, not lane
quantities, and the observation is correct as written — it is the OBS-V1.1
wording that must change.

### M7 — Dimensional change — `COMPLETE`

`REQ-017` to `REQ-021`. Depends on M6 approval.
Evidence: `TEST-022` to `TEST-027` pass; full suite, `make smoke`, and one
checkpoint save/load cycle at the new dimensions.

## 11. Progress and findings log

**2026-07-29 — plan created.**

Completed: full static review of `causal_semantic.py` against OBS-V1.2 and the
inherited OBS-V1.1 field contracts; provenance verification of
`CausalSceneContext.memory`, `RouteProjection.lateral_distance_m`,
`RouteLaneRecord`, `EpisodeCache`, conflict-zone construction, movement-priority
decoding, and `_actor_class`.

Commands: none. No test was executed — `pytest` is not importable outside the
container, and no production code has been modified.

Findings: 28 items, `C1` to `C28`, mapped to `REQ-001` to `REQ-027`. Three
findings are causality defects (`C6`, `C7`, `C26`), one is a latent crash
(`C24`), five are exact duplications or constants (`C17` to `C21`), and one is a
non-functional dead method (`C23`). Four provenance checks came back clean and
are recorded in §4.3 so they are not re-investigated.

Severity note: `C7` is the only finding that can invalidate an experimental
result rather than merely degrade it, because the exposed bit is a function of
the violation the Rulebook is about to penalise. **Superseded by the 2026-07-29
entry below.**

Decisions needed: `DEC-001` to `DEC-009`.

Next step: resolve M0.

**2026-07-29 — decision round 1 and severity correction.**

Completed: `DEC-001` (option B), `DEC-004` (option A) approved by the user;
`DEC-008` and `DEC-009` decided under explicit delegation. `DEC-005` re-explained
and still open. `DEC-002`, `DEC-003`, `DEC-006`, `DEC-007` remain open.

Commands: none. No production change.

Findings: new item `C29`. While preparing the `DEC-005` explanation, the tuple
order of the two Rulebook latch sets was checked against the observation's
membership test. The latches are keyed `(actor_id, zone_id)`
(`components/controls.py:435`; consumed as `for actor_id, zone_id in ...` at
`transition.py:495` and `:727`), while the observation constructs
`pair = (zone_id, actor.actor_id)` (`causal_semantic.py:1255`). The test at
`:1305` therefore never succeeds: `incompatible entry latched` is constant
`0.0` at runtime.

Severity correction: the 2026-07-29 plan-creation entry stated that `C7` was the
only finding able to invalidate an experimental result. That is wrong for the
illegal-entry half of `C7`, which has never been functional and therefore cannot
have contaminated any completed run. What remains genuinely live is the
`preexisting_ego_occupancy_zone_ids` read at `:1303`, which uses a plain
`frozenset[str]` of zone ids and does succeed: it is a real `RulebookMemory`
read, but its content is benign (it marks a zone the ego already occupied before
the event became attributable) and `REQ-008` reconstructs it causally rather than
removing it.

Consequence for `DEC-005`: the recommendation is unchanged. Removing a channel
that never worked has no behavioural cost, and leaving the coupling to
`RulebookMemory` in the source means a future correction of the tuple order
would silently activate the leak. `TEST-009` (populated vs empty memory produces
identical tensors) remains the right guard and would today fail only on the
pre-existing-occupancy component.

Evidence for `DEC-008`: `conf/config.yaml:7` still selects `obs: semantic_v2` as
the repository default, and `SemanticStateObservationV2` receives the shared base
builder via `set_batch_builder` (`thesis_scenario_env.py:427`-`:432`). The four
shared helpers must therefore be overridden, not edited in place.

Decisions needed: none.

**2026-07-29 — decision round 2, M0 closed.**

Completed: `DEC-002`, `DEC-003`, `DEC-005` and `DEC-006` approved by the user,
all on the recommended option; `DEC-007` decided under delegation; ADR-033
written and approved. Plan status moved from `AWAITING_DECISIONS` to `APPROVED`.

Commands: none. No production change.

Findings: none new.

Next step: M1 — write the 33-case matrix so that every case fails for its
documented reason before any production edit.

**2026-07-29 — M1-M7 implemented.**

Completed: the 33-case matrix (`tests/test_observation_v13_corrections.py`,
written as a separate module so the OBS-V1.2 contract tests stay legible);
every Phase A correction as an override on
`PerceptionBoundedSemanticBatchBuilder`; the OBS-V1.3 dimensional change; both
specifications; the authority index rows.

Commands and results: see §14. Baseline before any production edit was 26
failed / 7 passed on the new matrix; after implementation 33 passed, full suite
1094 passed, rulebook 240 passed, and the `semantic_v3` path ran a full smoke at
`D=3009`.

Findings during implementation:

1. **`DEC-004` was already satisfied, and not in the way the plan assumed.** No
   production training run was alive. `tmux list-sessions` shows only
   `sac-lite-cmp` and `sac-micro-cmp`, both created 2026-07-27 and both sitting
   at an idle `bash`. The `td3-0` process visible in `ps` (PID 1946106) is the
   tmux *server*, which retains the command line of the first session ever
   created on it; the `td3-0` session itself no longer exists. Nothing was
   killed. Three containers idle since 9 days hold hung `pytest` runs from an
   earlier session, and one container was running a `smoke_train` started by
   concurrent work; none were touched.
2. **Two fixture assumptions in the frozen matrix were wrong and were corrected
   before implementing.** `shapely.representative_point()` on the first
   candidate L-shape returned a vertex on the *near* side, so that fixture would
   not have exercised the `C4` side-assignment defect; it was replaced with a
   shape verified to place the representative point on the opposite side from
   the nearest point. And `AC-006` was reformulated: "rotating the ego leaves
   the static heading unchanged" is wrong, because a relative orientation
   *should* depend on ego heading. The correct property, and the one now tested,
   is that a **rigid rotation of the whole scene** leaves the static tokens
   unchanged.
3. **The axis-aligned bounding box is not rotation invariant.** The first
   implementation of the local static window used `bounds`, which passed the
   dimension test but failed the rotation test. Replaced by the minimum rotated
   rectangle, falling back to the clipped length for line-like geometries.
4. **`REQ-025` resolves to `none`, not `unknown`.** Inside the control horizon
   the local map legitimately establishes that an actor's lane carries no
   control; only knowledge *beyond* the horizon is inadmissible. The acceptance
   criterion was corrected accordingly: the distant stop must not leak in as
   `stop`, and the reported value is `none`.
5. **`LatentQueryEncoderV2` shares the projection module with the V3 encoder.**
   The first dimensional edit changed both and broke two `test_encoders_v10.py`
   cases. The legacy widths (14, 17, 35) were restored; ENC-V1.3 §3 now states
   the exclusion normatively.

Decisions needed: none.

Next step: user sign-off, then M8 closure.

## 12. Deviations

| ID | Original contract | Proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-001` | OBS-V1.1 §7.8 lists `Incompatible entry latched` | Feature removed | Contradicts OBS-V1.2 §1/§12; rule-aligned shortcut. Per `C29` the channel is also inert at runtime, so removal has no behavioural cost | `DEC-005`, ADR-033 | `TEST-009`, `TEST-024`, OBS-V1.3 |
| `DEV-002` | OBS-V1.1 §7.6 `Left/Right adjacent lane available` | Fields removed | Not derivable from any available source | `DEC-002` | `TEST-025`, OBS-V1.3 |
| `DEV-003` | OBS-V1.1 §7.7 yellow-onset distance and required stopping distance in the control token | Fields removed | Exact duplicate and deterministic function of `yellow_onset_memory` | `DEC-001` | `TEST-022`, OBS-V1.3 |
| `DEV-004` | OBS-V1.2 §8 index 23 `control governs the ego movement` | Field removed | Identical by construction to index 13 | `DEC-001` | `TEST-023`, OBS-V1.3 |
| `DEV-005` | OBS-V1.1 §8.3 criterion 3 for control ranking | Criterion permanently dropped | Only implementable by reading a Rulebook latch | `DEC-009` | `TEST-021`, OBS-V1.3 |
| `DEV-006` | OBS-V1.1 §7.5/§8.2 "Route lateral distance" | Redefined as a signed offset | Current clamped form destroys the right half-plane | `DEC-006` | `TEST-016`, OBS-V1.3 |
| `DEV-007` | OBS-V1.1 §7.4 `Lane offset`, §7.6 `Current lane curvature` | Specification text corrected to name the assigned route | The implementation is correct; the contract text is not | `DEC-001` | OBS-V1.3 |
| `DEV-008` | OBS-V1.2 §7 static classes cone/barrier/wall/stationary vehicle/generic | Taxonomy redefined to what the source discriminates | The runtime emits `generic` for every static object | `DEC-003` | `TEST-027`, OBS-V1.3 |

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/envs/observations/causal_semantic.py` | Planned modification | `REQ-001` to `REQ-016`, `REQ-021` to `REQ-026` |
| `src/thesis_rl/contracts/observation_schema.py` | Planned modification | `REQ-017` to `REQ-020` (Phase B) |
| `src/thesis_rl/agent/planners/encoders/factory.py` | Planned modification | New group widths (Phase B) |
| `tests/test_perception_bounded_semantic.py` | Planned modification | Most of the `TEST-*` matrix |
| `tests/test_observation_schema_v12.py` | Planned modification | `TEST-022` to `TEST-026` |
| `tests/test_causal_semantic_batch.py` | Planned modification | `TEST-033` legacy invariance |
| `tests/test_encoders_v11.py` | Planned modification | Encoder at new dimensions |
| `tests/test_checkpoint_manifest_sidecar.py`, `tests/test_checkpointing.py` | Planned modification | `3064` -> `3009` |
| `docs/specifications/observation_v1.3_specification.md` | Planned creation | Phase B contract |
| `docs/specifications/encoder_v1.3_specification.md` | Planned creation | Phase B encoder contract |
| `docs/decisions/ADR-033-rulebook-latch-exclusion-from-policy-observation.md` | Created 2026-07-29 | `DEC-005`, `DEC-009` |
| `docs/project_index.md` | Planned modification | Authority rows |

## 14. Validation results

| Command | Result | Date | Notes |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_observation_v13_corrections.py` | `PASS` | 2026-07-29 | 33 passed. Baseline before implementation was 26 failed / 7 passed |
| `make test` (full suite via the dev container) | `PASS` | 2026-07-29 | 1094 passed, 1 pre-existing warning in `test_pg_profiles.py` |
| `make rulebook-v2-check` | `PASS` | 2026-07-29 | 240 passed, scoped ruff clean, `git diff --check` clean |
| `make smoke` | `PASS` | 2026-07-29 | Default preset (`obs: lidar_state`), 2000 steps |
| `... python -m thesis_rl.cli.train --config-name presets/test/smoke_train obs=semantic_v3` | `PASS` | 2026-07-29 | The OBS-V1.3 path end to end at `D=3009`: 2000 steps, 5 episodes, no failure |
| `make lint PYTHON_QUALITY_PATHS="<changed src and tests>"` | `PASS` | 2026-07-29 | All checks passed |
| `make format-check PYTHON_QUALITY_PATHS="tests/test_observation_v13_corrections.py"` | `PASS` | 2026-07-29 | New module formatted |
| `make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/envs/observations/causal_semantic.py"` | `FAIL` | 2026-07-29 | **Pre-existing.** Verified that the file at `HEAD` (`eaa520c`) already fails the same check, so this is the repository-wide formatting baseline AGENTS.md describes, not a regression. Mass-formatting it inside a semantic change is explicitly discouraged there. Follow-up, once a dedicated formatting change is approved: `make format PYTHON_QUALITY_PATHS="src/thesis_rl/envs/observations/causal_semantic.py"` |
| `git diff --check` | `PASS` | 2026-07-29 | No whitespace defects |

## 15. Final reconciliation

Not applicable: no requirement has been implemented. Every `REQ-*` is
`NOT_IMPLEMENTED`, gated on M0.

Known limitations that this plan will **not** remove, to be carried into the
OBS-V1.3 limitations section:

1. Semantic tracking and classification remain ideal after physical admission
   (OBS-V1.2 §9, unchanged).
2. Other actors' lane association (`live_lane_id`) is exact. This is the
   heaviest remaining idealisation: it drives the same-lane relation, the
   conflict-zone pairing and the approach-control association, and a real stack
   is least certain exactly where it matters most — an actor straddling a lane
   boundary mid-manoeuvre.
3. Ego localisation on the HD map is exact, which the dashed-boundary and
   clearance features depend on.
4. Static map features are admitted without an occlusion test. This is
   deliberate (OBS-V1.2 §6.1): it models map knowledge, not a detector.
5. Right-of-way availability depends on scenario metadata carrying
   `rulebook_vehicle_yield`; its coverage across the dataset is a separate
   question from its causal validity.

Deferred optional work: lateral lane adjacency from the ScenarioNet lane schema
(`DEC-002` option B); the fine static sub-taxonomy through `ActorSnapshot`
(`DEC-003` option B); a range filter on the crosswalk interaction branch (§4.4).
