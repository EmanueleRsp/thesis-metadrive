# ExecPlan: R2 Lateral-RSS Clearance Replacement (v4.8)

## 1. Metadata

- Feature: `r2_lateral_rss_clearance_replacement`
- Plan ID: `R2-LATERAL-RSS-CLEARANCE-V4.8`
- Authoritative specification: `docs/specifications/rulebook_v4.8_specification.md`,
  version `4.8`, status `APPROVED`, `Authoritative: YES` (amends only v4.7
  §6.4; v4.7 remains authoritative for everything else)
- Status: `VERIFIED`
- Created: 2026-07-24
- Last updated: 2026-07-24
- Branch: `scenarionet-implementation`
- Related ADR: `docs/decisions/ADR-025-r2-lateral-rss-clearance-replacement.md` (`APPROVED`)
- Owner: n/a (single session)

## 2. Objective And Scope

Implement the scoped lateral-RSS metric that replaces `q_clear,vehicle`,
demote `q_clear,static` to diagnostic-only, and update `R2`'s aggregation,
per `rulebook_v4.8_specification.md` §6-9. In scope: new component
`rss_lateral`, its candidate construction from `pre_state`, the longitudinal
gate reusing the existing RSS formula, `clearance.py` scoped to VRU only,
`aggregation.py`/`registry.py` wiring, and the `REQ-R2-06` regression test.
Out of scope: RSS-longitudinal, TTC, VRU clearance, vehicle-yield, and the
actor-classification logic itself (already conformant, no change needed).

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| REQ-R2-01 | Vehicle clearance removed from the cost path | v4.8 §6 |
| REQ-R2-02 | Static clearance diagnostic-only | v4.8 §6 |
| REQ-R2-03 | New scoped lateral-RSS metric | v4.8 §6-7 |
| REQ-R2-04 | VRU clearance unchanged | v4.8 §6 |
| REQ-R2-05 | New R2 aggregation | v4.8 §6-7 |
| REQ-R2-06 | Actor classification invariant (parked vehicle stays `VEHICLE`) | v4.8 §6 |

## 4. Current Repository Analysis

- `src/thesis_rl/rulebook/v2/components/clearance.py`: previously
  `CLEARANCE_THRESHOLDS_M = {VEHICLE: 0.8, PEDESTRIAN: 1.0, CYCLIST: 1.0,
  STATIC_COLLIDABLE: 0.5}`, worst-of cost over all four classes.
- `src/thesis_rl/rulebook/v2/aggregation.py:65-67` (pre-change):
  `dynamic_interaction_safety = {"rss", "ttc", "clearance"}`.
- `src/thesis_rl/rulebook/v2/registry.py:47-49` (pre-change): `rss`, `ttc`,
  `clearance` `ComponentDefinition` entries, no lateral component.
- `src/thesis_rl/rulebook/v2/context/metadrive_live.py::_actor_class`
  (VERIFIED): purely type-name based, no velocity-based reclassification
  anywhere in the repository — `REQ-R2-06` was already satisfied by
  existing code; only the regression test was missing.
- RSS-longitudinal/TTC candidate construction reads `pre_state`
  (`transition.py::_rss_candidates`, `component_inputs["ttc"]`); the
  pre-change `clearance` component read `post_state`. The new lateral
  component reads `pre_state` (`DEC-R2-02`).
- Reusable primitives (all VERIFIED and reused, no duplication): `geometry/
  route.py::RoutePolyline.project` (`tangent_xy`, `lateral_distance_m`),
  `geometry/lanes.py::associate_route_lane`/`footprint_route_coordinates`/
  `bumper_to_bumper_gap`, `components/rss.py::safe_distance_m`.

## 5. Assumptions And Invariants

- Shared tangent/normal frame per pair: the tangent at the ego footprint's
  centroid projection onto the task's canonical `RoutePolyline` (same
  `route` object already used by RSS-longitudinal); `normal = (-tangent_y,
  tangent_x)`. Vertex-level `s_m`/`lateral_distance_m` are each projected
  independently (same simplification RSS-longitudinal already accepts for
  gently-curved roads; not a new risk).
- Inward-speed sign convention: `direction = +1` when the actor's center
  lateral coordinate is `>=` ego's on the shared frame, else `-1`;
  `ego_inward = direction * ego_normal_speed`, `actor_inward = -direction *
  actor_normal_speed`. Documented in `transition.py::_rss_lateral_candidates`.
- Applicability proxy for "same road structure or adjacent compatible
  lanes, concordant directions" (v4.8 §8): both ego and the actor must
  resolve a valid `LaneAssociation` via `associate_route_lane`, and each
  must have heading concordant with its own lane's tangent (`heading ·
  tangent > 0`), mirroring RSS-longitudinal's existing concordance check
  but *without* RSS-longitudinal's same-lane-id restriction, since the
  lateral metric must also apply to adjacent lanes (AC-R2-05). No new lane
  adjacency data structure was introduced (none exists in `RouteLaneRecord`
  and inventing one would violate the project's no-unverified-source-policy
  precedent, ADR-023) — this is the resolution of `DEC-R2-01`
  (not formally re-opened, since the specification already constrains
  applicability to compatible geometries and leaves the exact operational
  test as an implementation detail).
- Longitudinal gate role assignment (`I_long,unsafe`, v4.8 §7 step 3-4):
  when the two tangent-axis intervals do not overlap, the existing RSS
  formula (`safe_distance_m`) is applied with the real ego always in the
  "ego" (calibrated-braking) role and the other actor always in the "front
  vehicle" role, regardless of which one is geometrically ahead — the only
  choice consistent with the runtime only calibrating ego's braking
  (`RSSCalibrationArtifact.ego_min_brake_mps2`); recorded as `DEC-R2-05`
  below.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| DEC-R2-02 | Specification clarification | `pre_state` vs `post_state` | `pre_state` | `pre_state` | Matches RSS-long/TTC | **APPROVED** 2026-07-24 by the user |
| DEC-R2-04 | Implementation detail | How to expose `static_polygon_distance_m` | (A) `diagnostics` only; (B) separate component | (A) | Logging only | **APPROVED** 2026-07-24, resolved in the specification concurrently with approval |
| DEC-R2-05 | Implementation detail | Role assignment (ego vs "front vehicle") in the reused longitudinal gate formula when the other actor is geometrically behind | (A) always real ego = "ego" role; (B) invent/assume unmeasured braking for the other vehicle | (A) | No new unverified assumption; consistent with the only calibration the runtime has | Approved as implementation detail (no spec deviation — the specification only says "reuse the existing formula", this resolves *how*) |

No decision here required new user approval beyond `DEC-R2-02` (already
obtained) and the overall specification approval; `DEC-R2-04`/`DEC-R2-05`
are implementation details within Codex's decision authority per `AGENTS.md`.

## 7. Proposed Design

- **Geometry** (`src/thesis_rl/rulebook/v2/geometry/lanes.py`):
  `FootprintRouteCoordinates` gained `center_tangent_xy`, `center_lateral_m`,
  `lateral_min_m`, `lateral_max_m` (computed in the existing
  `footprint_route_coordinates`, backward-compatible since construction is
  keyword-only and it was the sole call site). New pure function
  `lateral_edge_to_edge_gap(ego, other) -> (gap_m, direction)`, mirroring
  `bumper_to_bumper_gap` on the normal axis.
- **Component** (`src/thesis_rl/rulebook/v2/components/rss_lateral.py`,
  new file): `LateralRSSCandidate` (actor_id, lateral_gap_m,
  ego_inward_speed_mps, actor_inward_speed_mps, longitudinal_unsafe);
  module constants `RHO_LAT_S=0.5`, `LATERAL_ACC_MAX_MPS2=0.2`,
  `LATERAL_BRAKE_MIN_MPS2=0.8`, `LATERAL_MARGIN_MU_M=0.10` (frozen, mirrors
  how RSS-longitudinal hardcodes its own frozen constants in `rss.py` rather
  than exposing them via `RulebookTransitionConfig`); `lateral_safe_distance_m`
  implements v4.8 §7's `Δ_e`/`Δ_i`/`d_safe^lat` exactly; `evaluate_rss_lateral`
  is worst-of over candidates, `NOT_APPLICABLE` when empty, `0.0` cost when
  `longitudinal_unsafe` is false or `d_safe^lat=0`.
- **Candidate construction** (`src/thesis_rl/rulebook/v2/transition.py`):
  `_longitudinal_unsafe_gate` (the §7 gate, reusing `safe_distance_m`) and
  `_rss_lateral_candidates` (mirrors `_rss_candidates`'s structure and
  heading-concordance check, but iterates all `VEHICLE` actors with a valid
  association rather than only same-lane frontal ones). Called once per
  transition from `pre_state`, wired into `component_inputs["rss_lateral"]
  = {"candidates": rss_lateral_candidates}`.
- **Clearance** (`src/thesis_rl/rulebook/v2/components/clearance.py`):
  `CLEARANCE_THRESHOLDS_M` now only `{PEDESTRIAN: 1.0, CYCLIST: 1.0}`;
  `STATIC_COLLIDABLE` actors are tracked separately into
  `diagnostics["static_polygon_distance_m"]` (worst-of/minimum), never
  reaching `raw`/`cost`.
- **Aggregation/registry**: `dynamic_interaction_safety` group now
  `{"rss", "rss_lateral", "ttc", "clearance"}`; new `ComponentDefinition("rss_lateral",
  MacroRule.DYNAMIC_INTERACTION_SAFETY, evaluate_rss_lateral)` registered
  between `rss` and `ttc`.
- **Classification** (`tests/test_rulebook_v2_metadrive_live.py`): added
  `test_metadrive_stationary_vehicle_stays_classified_as_vehicle` — no
  production code change needed (`REQ-R2-06` was already satisfied).

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| REQ-R2-01 | AC-R2-01 | `clearance.py` (`CLEARANCE_THRESHOLDS_M` scoped to VRU) | `tests/test_rulebook_v2_clearance.py::test_clearance_excludes_vehicle_class_per_rulebook_v4_8` | Verified |
| REQ-R2-02 | AC-R2-06 | `clearance.py` (`diagnostics["static_polygon_distance_m"]`) | `tests/test_rulebook_v2_clearance.py::test_clearance_static_distance_is_diagnostic_only_per_rulebook_v4_8` | Verified |
| REQ-R2-03 | AC-R2-01, AC-R2-02, AC-R2-03, AC-R2-05 | `components/rss_lateral.py`, `transition.py::_rss_lateral_candidates` | `tests/test_rulebook_v2_rss_lateral.py`, `tests/test_rulebook_v2_geometry.py::test_lateral_edge_to_edge_gap_and_side_on_shared_route_frame`, `tests/test_rulebook_synthetic_scenarios.py[rss_front_vehicle-rss_lateral-True-False]` | Verified |
| REQ-R2-04 | — | `clearance.py` (VRU path unchanged) | Existing VRU clearance tests, unaffected | Verified (no change) |
| REQ-R2-05 | — | `aggregation.py`, `registry.py` | `tests/test_rulebook_v2_transition.py::test_transition_invokes_complete_registry_and_keeps_vehicle_yield_not_applicable` (component set) | Verified |
| REQ-R2-06 | AC-R2-04 | `context/metadrive_live.py::_actor_class` (already conformant) | `tests/test_rulebook_v2_metadrive_live.py::test_metadrive_stationary_vehicle_stays_classified_as_vehicle` | Verified |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| TEST-R2-01 | Unit | `lateral_safe_distance_m` matches the reference numeric value at zero inward speed | `rho_lat=0.5, acc=0.2, brake=0.8, mu=0.10`, both speeds 0 | `~0.1625 m` | REQ-R2-03, AC-R2-03 |
| TEST-R2-02 | Unit | `evaluate_rss_lateral` is `NOT_APPLICABLE` with no candidates | `candidates=()` | `applicable=False`, `cost=0.0` | REQ-R2-03 |
| TEST-R2-03 | Unit | A longitudinally-safe candidate contributes zero cost regardless of lateral gap | `longitudinal_unsafe=False`, tiny gap | `cost=0.0` | REQ-R2-03, AC-R2-01/05 |
| TEST-R2-04 | Unit | A longitudinally-unsafe candidate with insufficient gap produces positive bounded cost | `longitudinal_unsafe=True`, closing speed | `0 < cost <= 1` | REQ-R2-03, AC-R2-02 |
| TEST-R2-05 | Unit | Worst-of over multiple candidates | one safe, one unsafe | worst actor selected | REQ-R2-03 |
| TEST-R2-06 | Unit | Negative gap raises | `lateral_gap_m=-0.1` | `ValueError` | REQ-R2-03 |
| TEST-R2-07 | Unit | `lateral_edge_to_edge_gap` matches `bumper_to_bumper_gap`'s pattern on the normal axis, both directions and overlap | shared-route footprints offset in y | correct gap/side | REQ-R2-03 |
| TEST-R2-08 | Unit | VEHICLE actor excluded from clearance cost entirely | close VEHICLE actor | `NOT_APPLICABLE`, empty `raw["actors"]` | REQ-R2-01 |
| TEST-R2-09 | Unit | Close STATIC_COLLIDABLE actor logged in diagnostics only | close static actor | `diagnostics["static_polygon_distance_m"]` set, `raw["actors"]` empty | REQ-R2-02 |
| TEST-R2-10 | Unit | Stationary VEHICLE-typed object stays classified `VEHICLE` | `velocity=(0,0)` mock | `actor_class is VEHICLE` | REQ-R2-06 |
| TEST-R2-11 | Integration | Live transition includes `rss_lateral` for a same-lane front vehicle | `rss_front_vehicle` synthetic scenario | `applicable=True` | REQ-R2-03, REQ-R2-05 |
| TEST-R2-12 | Regression | Full rulebook v2 suite unaffected elsewhere | `make rulebook-v2-check` | all pass | Backward compatibility |

Commands:

```bash
docker compose run --rm dev uv run --no-sync python -m pytest -q -k rulebook
make rulebook-v2-check
```

## 10. Milestones

- [x] M0: Specification approved (`docs/specifications/rulebook_v4.8_specification.md`,
  `Status: APPROVED`), ADR-025 approved, `docs/project_index.md` updated.
- [x] M1: Lateral-extent geometry primitive (`FootprintRouteCoordinates`
  extension, `lateral_edge_to_edge_gap`). Tests: TEST-R2-07.
- [x] M2: `q_RSS,lat` evaluator component (`components/rss_lateral.py`),
  `DEC-R2-04` resolved (diagnostics-only static exposure). Tests:
  TEST-R2-01..06.
- [x] M3: `aggregation.py`/`registry.py` wiring, `transition.py` candidate
  construction from `pre_state`, `clearance.py` scoped to VRU with
  diagnostic-only static distance. Tests: TEST-R2-08, TEST-R2-09, TEST-R2-11,
  plus updated pre-existing tests (component-name-set assertions, clearance
  worst-actor assertions) to reflect the approved behavioral change.
- [x] M4: `REQ-R2-06` regression test. Tests: TEST-R2-10.
- [x] M5: Full reconciliation — `make rulebook-v2-check` and the complete
  `-k rulebook` suite (306 tests), scoped `ruff format --diff` on every
  modified/new file.

## 11. Progress And Findings Log

- 2026-07-24: User approved `docs/specifications/rulebook_v4.8_specification.md`
  ("Approvo la specifica v4.8, procedi con l'implementazione"). Moved the
  spec from `incoming/` (untracked, gitignored) to `docs/specifications/`,
  set `Status: APPROVED`/`Authoritative: YES`, resolved `DEC-R2-04`, filled
  in and approved ADR-025, updated `docs/project_index.md` (new spec row,
  ADR-025 row, two ExecPlan registry rows).
- 2026-07-24: Implemented M1-M5 directly (no additional Plan-agent dispatch;
  design was completed inline given the mathematical contract was already
  fully specified). Regenerated `tests/fixtures/rulebook_scenarios/manifest.json`
  after changing `FIXTURE_COMPONENTS` for `rss_front_vehicle`
  (`clearance` → `rss_lateral`, since a `VEHICLE`-only fixture no longer
  triggers clearance at all).
- 2026-07-24, finding: initially assumed `rss_rear_vehicle` would also
  produce an `rss_lateral` candidate (since the lateral candidate builder,
  unlike RSS-longitudinal, does not filter by front/rear position). The
  live synthetic-scenario test showed `NOT_APPLICABLE` instead — the rear
  vehicle's position falls outside the synthetic lane polygon's coverage
  (the same reason RSS-longitudinal's own `rss` component is also
  `NOT_APPLICABLE` for that fixture), so `associate_route_lane` returns
  `None` before either candidate builder's front/rear logic is ever
  reached. Adjusted the test expectation to match; not a defect, both
  components fail closed identically for that fixture's specific geometry.
- 2026-07-24: `make rulebook-v2-check` (226 scoped tests + ruff) and the
  full `-k rulebook` suite (306 tests) pass. `git diff --check` clean.
  Formatting verified with scoped `ruff format --diff` on every
  modified/new file; only pre-existing, untouched lines remain outside the
  repository-wide formatting baseline, left unchanged per `AGENTS.md`.

## 12. Deviations

No deviations identified. `DEC-R2-01` (lateral frame convention) and
`DEC-R2-03` (longitudinal gate procedure) were already fully specified by
the approved specification text and required no reinterpretation; only
`DEC-R2-04` (implementation detail, resolved) and the newly-recorded
`DEC-R2-05` (role assignment in the reused gate formula) needed a
resolution beyond the literal specification text, and both are
implementation details within Codex's decision authority, not deviations
from approved acceptance behavior.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `docs/specifications/rulebook_v4.8_specification.md` | Created (moved from `incoming/`) | Approved specification |
| `docs/decisions/ADR-025-r2-lateral-rss-clearance-replacement.md` | Modified | Approved ADR |
| `docs/project_index.md` | Modified | New spec/ADR/ExecPlan registry entries |
| `src/thesis_rl/rulebook/v2/geometry/lanes.py` | Modified | `FootprintRouteCoordinates` lateral fields, `lateral_edge_to_edge_gap` |
| `src/thesis_rl/rulebook/v2/components/rss_lateral.py` | Created | New scoped lateral-RSS evaluator |
| `src/thesis_rl/rulebook/v2/components/clearance.py` | Modified | VRU-only cost path, static diagnostic-only |
| `src/thesis_rl/rulebook/v2/aggregation.py` | Modified | New `dynamic_interaction_safety` group membership |
| `src/thesis_rl/rulebook/v2/registry.py` | Modified | New `ComponentDefinition("rss_lateral", ...)` |
| `src/thesis_rl/rulebook/v2/transition.py` | Modified | `_longitudinal_unsafe_gate`, `_rss_lateral_candidates`, wiring |
| `tests/test_rulebook_v2_rss_lateral.py` | Created | TEST-R2-01..06 |
| `tests/test_rulebook_v2_geometry.py` | Modified | TEST-R2-07 |
| `tests/test_rulebook_v2_clearance.py` | Modified | TEST-R2-08, TEST-R2-09, updated existing threshold test |
| `tests/test_rulebook_v2_metadrive_live.py` | Modified | TEST-R2-10 |
| `tests/test_rulebook_v2_f6_contracts.py` | Modified | Updated pre-existing worst-actor/diagnostic assertions |
| `tests/test_rulebook_v2_contracts.py` | Modified | Registry index assertion made name-based instead of brittle-index |
| `tests/test_rulebook_v2_transition.py` | Modified | Component-name-set assertion includes `rss_lateral` |
| `tests/test_rulebook_synthetic_scenarios.py` | Modified | TEST-R2-11, updated `rss_front_vehicle`/`rss_rear_vehicle` expectations |
| `tests/rulebook_scenario_fixtures.py` | Modified | `FIXTURE_COMPONENTS` updated |
| `tests/fixtures/rulebook_scenarios/manifest.json` | Regenerated | Reflects updated `FIXTURE_COMPONENTS` |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q -k rulebook` | PASS | 2026-07-24 | 306 passed, 598 deselected |
| `make rulebook-v2-check` | PASS | 2026-07-24 | 226 passed (scoped `test_rulebook_v2_*.py`); `ruff check` clean; `git diff --check` clean |
| `docker compose run --rm dev uv run --no-sync ruff format --diff <every modified/new file>` | PASS (scoped) | 2026-07-24 | Every newly written line matches canonical formatting; a small number of pre-existing, untouched lines across several files remain outside the repository-wide formatting baseline per `AGENTS.md` — intentionally not reformatted (unrelated to this change) |

## 15. Final Reconciliation

- REQ-R2-01: IMPLEMENTED, VERIFIED.
- REQ-R2-02: IMPLEMENTED, VERIFIED.
- REQ-R2-03: IMPLEMENTED, VERIFIED.
- REQ-R2-04: VERIFIED (no change; confirmed by unaffected existing tests).
- REQ-R2-05: IMPLEMENTED, VERIFIED.
- REQ-R2-06: VERIFIED (no production change needed; regression test added).

Known limitations: the applicability proxy for "same road structure or
adjacent compatible lanes" (§5, `DEC-R2-01`'s resolution) uses heading
concordance plus valid lane association rather than an explicit lane-
adjacency graph, which does not exist in the current `RouteLaneRecord` data
model; this is consistent with the specification's own scope limitation
(full RSS with generalized route-geometry handling is explicitly out of
scope) and with the project's no-unverified-source-policy precedent
(ADR-023). No deferred required work remains within this plan's scope.
Compatibility: `R2`'s cost distribution changes as documented in the
specification §11; no already-run experiment baseline calibrated on v4.7's
`R2` is directly comparable after this change. Ready for experimental use.
