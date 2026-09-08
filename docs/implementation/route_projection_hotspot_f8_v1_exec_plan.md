# Route-Projection Hotspot (F8) v1 — ExecPlan

## 1. Metadata

- Feature: remove the production hotspot that makes every environment step on
  Waymo scenarios spend most of its Python time in `RoutePolyline.project`
  (`docs/open_items.md`, row `F8`).
- Plan ID: `route-projection-hotspot-f8-v1`
- Authoritative specification: `docs/specifications/rulebook_v4.7_specification.md`
  (`AUTHORITATIVE`, approved 2026-07-17), §2.9.2 (route polyline and
  curvilinear projection) and §2.9.4 (2.5D drivable surface). This plan changes
  **no** specified behaviour: both sections explicitly allow an accelerated
  selection provided the result is equivalent (§2.9.2 step 1 "mediante spatial
  index"; §2.9.4 "Uno spatial index può accelerare la selezione, ma il risultato
  deve essere equivalente all'unione di tutte le lane verticalmente
  compatibili").
- Status: `VERIFIED` (M1, M2, adversarial review and the full `make gate`
  recorded in §11)
- Created: 2026-09-08. Last updated: 2026-09-08.
- Branch: `worktree-f8-route-projection-hotspot` (worktree off `main` at
  `7f4ae9c`).
- Related ADRs: none required; `DEC-F8-001` is a test-tolerance decision
  recorded here (approved verbally by the user, 2026-09-08).
- Owner: user request ("dammi l'handoff per F8"), 2026-09-08.

## 2. Objective And Scope

**Observable capability.** The Rulebook step on Waymo scenarios gets
substantially cheaper with the reward, observation, termination and diagnostics
outputs unchanged (up to a declared floating-point tolerance for the
vectorised projection). The four `integration` cases of
`tests/test_reward_return_ordering_runtime.py` are the yardstick: the two Waymo
cases took 588 s / 584 s against 41 s / 41 s for PG (gate log
`20260908T145526Z`), which sets the gate floor at ~10 min.

**Why.** The learnability screening runs A and B are gated on this: every
vector worker runs the same code, so the hotspot taxes training as well as the
test suite.

**Success.** (a) An equivalence test proves the new `project` selects the same
segment as the old one on every probe point sampled from the frozen panels and
returns values within the declared tolerance; (b) a unit test proves the
drivable-surface pre-classification returns the same lane set as the
projection-based gate; (c) a unit test proves the per-step memo computes the
footprint-exit geometry once per `(adapter, ego pose)`; (d) the Waymo case of
the return-ordering test is measured before and after without the profiler and
the headline number is reported in `docs/open_items.md` `F8`.

**In scope.** `RoutePolyline.project` (vectorisation), `drivable_surface_for_ego`
and `carriageway_surfaces_for_ego` (vertical pre-classification),
`SceneContextAdapter._rulebook_full_footprint_exit` (per-step memo), their
tests, `docs/open_items.md` `F8`, `docs/project_index.md`.

**Out of scope.** `RoutePolyline.point_at` and `projection_diagnostics`
(not in the profile); any change to the tie rule, the continuity preference,
`GEOMETRY_EPSILON_M`, `VERTICAL_COMPATIBILITY_TOLERANCE_M`; the Shapely union
cache; the snapshotter; the `DrivableLaneRecord` re-validation per call unless
the post-M1 profile shows it (recorded in §11 if deferred).

**Compatibility.** No public interface, configuration key, checkpoint, dataset
or log schema changes. `RouteProjection` keeps its five fields and types. One
implicit structural contract tightens: `drivable_surface_for_ego` and
`carriageway_surfaces_for_ego` now read `centerline.z_range_m`, so a
duck-typed centerline exposing only `project` (as two test fakes did) no longer
works; `DrivableLaneRecord.centerline` was already typed `RoutePolyline`.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-F8-01` | Projection selects among vertically compatible segments (when `position_z` is supplied) the minimum planar distance, keeps candidates within `d_min + eps_geom`, prefers the smallest `s` at reset, `min abs(s - s_previous)` afterwards, then the smallest segment index; returns `(s, tangent_xy, z_at_s, lateral_distance, segment_index)`; no compatible segment fails closed. | v4.7 §2.9.2 steps 1–7 |
| `REQ-F8-02` | `max_s_jump_m` bounds the move from `previous_s_m` as a *preference* (unbounded selection kept when no plausible candidate). | `route.py:155-165` docstring and inline comment (code contract, `VERIFIED`; not in §2.9.2, see `DEC-F8-003`) |
| `REQ-F8-03` | The step's normative drivable surface is the union of every and only the vertically compatible drivable lanes; acceleration is allowed if the result is equivalent. | v4.7 §2.9.4 |
| `REQ-F8-04` | `carriageway_surfaces_for_ego` mirrors the same vertical gate and partitions by direction with the ±60° cone. | `drivable.py:119-172` (REQ-RBCOST-009 / DEC-RBCOST-004) |
| `REQ-F8-05` | The physical road-exit predicate is the Rulebook full-footprint geometry on the current snapshot; repeated evaluation within one step must return the same classification. | `scene_context.py:83-118` (code contract, `VERIFIED`) |
| `REQ-F8-06` | Performance: the Waymo case of the return-ordering test must be measured before and after, without the profiler, and reported. | `docs/open_items.md` `F8` (measurement requirement, not a scientific one) |

## 4. Current Repository Analysis

All statements `VERIFIED` on `main` at `7f4ae9c` unless labelled.

- `src/thesis_rl/rulebook/v2/geometry/route.py:147-244` — `project` loops in
  Python over every segment, allocating a `RouteProjection` per segment, then
  filters by `max_s_jump_m`, computes `planar_distance` again per candidate
  (twice: once for the minimum, once for the tie set), and reduces with `min`.
  `__post_init__` precomputes `_segment_starts_m` and `_segment_lengths_m`
  as tuples.
- `src/thesis_rl/rulebook/v2/geometry/drivable.py:85-116` — for every lane,
  `lane.centerline.project(ego_position_xy)` **without** `position_z`, then
  compares `projection.z_m` against `ego_position_z` with
  `VERTICAL_COMPATIBILITY_TOLERANCE_M` (3.0 m). Only `z_m` is read.
  `carriageway_surfaces_for_ego` (`:133-172`) does the same and additionally
  reads `tangent_xy`.
- Callers of `drivable_surface_for_ego`: `transition.py:1462` (once per step,
  post-state) and `scene_context.py:105` via `_rulebook_full_footprint_exit`,
  which is reached from `is_physically_out_of_road` (`thesis_scenario_env.py:929`
  `_is_out_of_road`, invoked by MetaDrive's native `done_function`,
  `reward_function` and `cost_function` — `scenario_env.py:169,233,303` — and by
  the thesis `done_function` at `:962`) and from `get_physical_road_diagnostics`
  (`:1043`). Each call re-captures the snapshot (`snapshotter(env)`) and
  recomputes the surface: 4–5 identical computations per step.
- Profile (F8 row, one Waymo case, cProfile ×3): `project` 890 952 calls,
  67 % of cumulative time; callers 644 903 from `drivable_surface_for_ego`,
  107 333 from `carriageway_surfaces_for_ego`, 76 288 from
  `causal_semantic._build_static_v12`; the remainder from `lanes.py`
  (association, footprint coordinates), `transition.py:141,1304`,
  `controls.py`.
- Tests fixing the contract: `tests/test_rulebook_v2_geometry.py`
  (`test_route_projection_uses_reset_and_previous_s_tie_breaks_at_self_intersection`,
  `test_route_projection_rejects_incompatible_vertical_level`,
  `test_route_projection_continuity_bound_rejects_a_far_branch_jump`,
  `test_drivable_surface_*`), `tests/test_thesis_scenario_env.py:125-175`
  (footprint exit; fakes `centerline` as a `SimpleNamespace(project=...)`),
  `tests/test_reward_return_ordering_runtime.py` (integration yardstick).
- `INFERRED` (to be confirmed by the post-M1 profile): the per-call
  construction of `DrivableLaneRecord` tuples (`transition.py:1466`,
  `scene_context.py:109`) runs Shapely `is_valid` on every lane polygon each
  call.

## 5. Assumptions And Invariants

- Units: metres; XY in the scenario frame; `z` metres; `s` XY arc length.
- `RoutePolyline.points_xyz` is consolidated (no two consecutive points closer
  than `PRECISION_GRID_M`), so every `_segment_lengths_m[i] > 0` (established
  by `__post_init__`, raises otherwise).
- Interpolated `z_m` on a segment lies within `[min(z_i, z_{i+1}), max(...)]`
  up to one floating-point rounding; therefore the lane's z-range over
  `points_xyz` bounds every projection's `z_m`. The pre-classification uses a
  guard band `Z_RANGE_GUARD_M = 1e-6` so a lane is decided without projection
  only when the range test holds with margin; anything closer to the boundary
  is projected exactly as today (mechanism argument; unit-tested at the
  boundary).
- Memo validity: within one `env.step()` the physics state is fixed after the
  simulator advance, so every `is_physically_out_of_road` call sees the same
  snapshot ego pose. The memo is keyed on the adapter object (by identity —
  one adapter per reset) and the snapshot ego `(position_xy, position_z,
  heading_rad)`; a pose change invalidates it, so semantics are identical by
  construction. The memo holds one entry.
- Seeds, termination/truncation, dataset policy: untouched.
- NaN/inf: the input validation of `project` is preserved verbatim. NumPy
  arithmetic on finite inputs with positive lengths produces finite values for
  every coordinate magnitude this repository sees (|xy| < 1e4 m); the one
  divergence from the reference is unreachable in practice: an overflowing dot
  product (|xy| ≳ 1e154) gives `nan` where `min(1.0, max(0.0, nan))` gave `0.0`.
- Input types: `project` widens `point_xy` to Python floats before the NumPy
  pass, so a `float32` input would now be projected in double precision where
  the reference kept single-precision arithmetic (~1e-4 m). No caller passes
  `float32` (every path converts explicitly or comes from `point_at`).
- Planar distance: `np.hypot` and `math.hypot` may differ by one ULP
  (~1e-14 m), so the `d_min + eps_geom` tie set is identical up to one ULP of
  the distance at the boundary, not bit-identical; this is inside
  `DEC-F8-001` and is why the criterion is a tolerance.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-F8-001` | specification clarification (test acceptance) | Equivalence criterion between the Python and NumPy `project` | bit-exact / declared tolerance | declared tolerance: identical `segment_index` on every probe, `s_m`, `z_m`, `lateral_distance_m`, `tangent_xy` within **1e-9 m** (values are O(1e2) m, so this is ~1e-11 relative, three orders above double rounding on the summed arc length and six below `GEOMETRY_EPSILON_M`) | test only; no runtime behaviour | **Approved** by the user 2026-09-08 ("chissene frega dell'equivalenza bit a bit, se cambia di pochissimo va bene") |
| `DEC-F8-002` | implementation detail | Where the per-step memo lives | `SceneContextAdapter` instance / module `lru_cache` on `drivable_surface_for_ego` / snapshotter | `SceneContextAdapter` instance (one per env), keyed as in §5 | none observable | Decided |
| `DEC-F8-003` | implementation detail | Vertical pre-classification without projection | project every lane (today) / z-range test with guard band | z-range test: fully inside → select, fully outside → skip, straddling → project | none observable (equivalent by mechanism, §5) | Decided |
| `DEC-F8-004` | implementation detail | `max_s_jump_m` semantics are documented in code, not in §2.9.2 | — | preserve the code contract verbatim (`REQ-F8-02`); do not touch | none | Noted, no change |

## 7. Proposed Design

**M1 — cheap gates and one computation per step.**

1. `RoutePolyline` gains a read-only `z_range_m: tuple[float, float]`
   (min/max over consolidated points), computed in `__post_init__`.
2. `drivable.py`: a private `_vertical_compatibility(lane_centerline, ego_z)`
   returns `True` / `False` / `None` (decided / undecided) from the z-range with
   the guard band; `drivable_surface_for_ego` and `carriageway_surfaces_for_ego`
   project only when undecided (`carriageway_surfaces_for_ego` still projects
   compatible lanes because it needs the tangent).
3. `scene_context.py`: `_rulebook_full_footprint_exit` keeps a one-entry memo
   `(adapter, pose_key) -> (fully_outside, outside_area, ego_area)` on the
   adapter instance.

**M2 — vectorised `project`.** Precompute once per polyline (lazily, cached on
the frozen instance via `object.__setattr__`) NumPy arrays: segment starts
(`x0, y0, z0`), deltas, lengths, unit tangents, cumulative starts. `project`
computes fraction, projected point, `z_m`, planar distance and signed lateral
distance for all segments in one pass; applies the vertical mask, the
`max_s_jump_m` plausibility mask (preference), the `d_min + eps` tie mask,
then the lexicographic reduction exactly as today. Only the selected segment
materialises a `RouteProjection`. The planar distance uses the same formula as
the current `planar_distance` (hypot from the point re-projected from `s`), not
`abs(lateral)`, so the tie set is identical up to one ULP of the distance (§5).
Error messages preserved verbatim.

Fallback/logging: none added. Errors: identical.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-F8-01` | `AC-F8-01`: on ≥ 2 000 (probe point, option set) pairs from the frozen panels (route polylines and lane centerlines; on-centerline, laterally offset, near joints, roundabout self-approach) the NumPy `project` returns the same `segment_index` and values within 1e-9 m of the reference Python implementation for every combination of `position_z` / `previous_s_m` / `max_s_jump_m`; the same on synthetic routes that exercise each rule | `route.py::RoutePolyline.project`, `::_projection_arrays` | `tests/test_route_projection_equivalence.py::test_vectorised_projection_matches_reference_on_frozen_panel_records[waymo,pg]` (skips without the prepared validation runtime — a gate log must show it ran), `::test_vectorised_projection_matches_reference_on_synthetic_routes[7 routes]`, `::test_vectorised_projection_breaks_bit_identical_ties_on_the_lower_segment_index`, `::test_vectorised_projection_preserves_argument_validation`, `::test_route_polyline_equality_and_hash_ignore_the_projection_cache`; existing `test_rulebook_v2_geometry.py` projection tests unchanged | Implemented, tests pass |
| `REQ-F8-02` | `AC-F8-02`: existing `test_route_projection_continuity_bound_rejects_a_far_branch_jump` passes unchanged; the equivalence test covers `max_s_jump_m` with and without a plausible candidate | as above | as above (option sets `max_s_jump_m` 5.0 and 0.0 from a far `previous_s_m`) | Implemented, tests pass |
| `REQ-F8-03` | `AC-F8-03`: the lane set selected by the pre-classification equals the projection-based gate on constructed lanes fully below, fully above, straddling and exactly at / within 1e-7 m of the tolerance boundary, for six ego elevations; only undecided lanes are projected; existing drivable tests pass | `drivable.py::_vertical_compatibility_from_range`, `::drivable_surface_for_ego` | `tests/test_rulebook_v2_geometry.py::test_drivable_surface_vertical_preclassification_matches_projection_gate[6 ego_z]` | Implemented, tests pass; pre-fix failure executed (extra lanes projected) |
| `REQ-F8-04` | `AC-F8-04`: same lane partition for `carriageway_surfaces_for_ego` | `drivable.py::carriageway_surfaces_for_ego` | same test | Implemented, tests pass |
| `REQ-F8-05` | `AC-F8-05`: two calls with the same adapter and pose compute the surface once; a pose change or a new adapter recomputes; existing footprint-exit tests pass | `scene_context.py::SceneContextAdapter.__init__`, `::_rulebook_full_footprint_exit` | `tests/test_thesis_scenario_env.py::test_scene_context_footprint_exit_is_computed_once_per_pose` | Implemented, tests pass; pre-fix failure executed (3 computations against 1) |
| `REQ-F8-06` | `AC-F8-06`: before/after step-loop time of `a_native` × `validation_waymo_empirical` without profiler, reported in `F8` | — | measurement script `outputs/f8/f8_profile_case.py` on the outputs volume (git-ignored); numbers in §11 and in `docs/open_items.md` `F8` | Measured: 1135 → 929 (M1) → 312 ms/step (M1+M2) |

## 9. Validation Commands

- Focused: `docker compose -p thesis-metadrive run --rm -T dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_geometry.py tests/test_route_projection_equivalence.py tests/test_thesis_scenario_env.py`
- Ruff, focused: `make lint PYTHON_QUALITY_PATHS="<touched files>"`,
  `make format-check PYTHON_QUALITY_PATHS="<touched files>"`
- Working loop: `make check`
- Yardstick: `make gate GATE_ARGS="tests/test_reward_return_ordering_runtime.py"` (PARTIAL)
- Merge gate: `make gate`
- Measurement: `docker compose -p thesis-metadrive run --rm -T dev uv run --no-sync python /workspace/outputs/f8/f8_profile_case.py a_native validation_waymo_empirical [--profile]`

## 10. Milestones

| Milestone | Content | Verification | Status |
|---|---|---|---|
| M0 | Baseline measurement without profiler | `TOTAL` line of `outputs/f8/baseline_a_native_waymo.log`: 593 steps, step loop 673.3 s, **1135 ms/step**, wall 710 s | Done |
| M1 | z-range pre-classification + per-step memo (commit `7c4500e`) | AC-F8-03/04/05 pass; `make check` PASS (PARTIAL by design, 1881 passed, 6 skipped, 2m37s); re-measured with the reference `project` swapped back in: **929 ms/step** (550.7 s, −18 %) | Done |
| M2 | NumPy `project` (commit `f78faf6`) | AC-F8-01/02 pass; re-measured: **312 ms/step** (185.0 s, 3.6× over baseline; wall 217 s) | Done |
| M3 | Gate, adversarial review, docs, PR | adversarial review recorded below; `F8`/`F9` rows and index updated; `make gate` **PASS, FULL** at `3b3ca66`: 1893 passed, 5 skipped, pytest 4m01s (log `outputs/gate/20260908T172858Z-3b3ca66.log`; the two Waymo return-ordering cases 224 s / 222 s against 588 s / 584 s in `20260908T145526Z`) | Done |

## 11. Progress Log, Findings, Limitations

- 2026-09-08: plan created; worktree from `main` `7f4ae9c`; baseline measured
  (tmux `f8-baseline`): 1135 ms/step on `a_native` × `validation_waymo_empirical`
  (three behaviours, 593 steps, no profiler).
- 2026-09-08, M1: the three behaviours' returns (9.521790397819428,
  198.71053857982088, 386.40149605475256) and route completions are identical
  to every printed digit before and after M1 and after M2, so on this case the
  change is not merely within tolerance but invisible.
- 2026-09-08, M1 finding: **the caller count did not tell the story.** 644 903
  of 890 952 `project` calls came from `drivable_surface_for_ego`, yet removing
  them (M1) saved only 18 % of the step: the remaining ~300 calls per step are
  on the *route* polyline (357 segments on the profiled record) and on longer
  lane centerlines, each 2–3× the cost of a typical lane projection. M1 alone
  would not have brought the Waymo case near the PG one; M2 was necessary, and
  the numbers are recorded rather than assumed.
- 2026-09-08, M2 micro-benchmark (`outputs/f8/microbench_project.py`, min of
  5×50 repeats, in-container): route of `sd_waymo_training_20s_141f284b8bb5d418`,
  357 segments: 743 µs → 23 µs plain, 594 µs → 24 µs with `position_z` and
  continuity (×33 / ×25); its longest lane centerline, 213 segments: 444 µs →
  20 µs (×22).
- 2026-09-08, cost of the new test: the equivalence test runs the pure-Python
  reference on ~10 000 probe/option pairs per route; after halving the probe
  density and computing the reset projection once per probe, the slowest case
  (`[waymo]`, three records) takes 33 s, the others < 5 s; the module runs in
  parallel with the rest under `pytest-xdist`, so it adds no wall time to
  `make check`.
- 2026-09-08, **adversarial review** (AGENTS.md step 10) by a session that did
  not write the change (Opus, read-only, given this plan and the pre-change
  code). Verdict: no blocking defect; the NumPy `project` traced operation by
  operation as bit-identical to the reference on float64 inputs except for
  `np.hypot` vs `math.hypot` (one ULP, §5); the z-range gate sound (checked all
  six ego elevations against the eight constructed lanes by hand); the memo
  unable to go stale on the production path (new adapter per reset, frozen
  `EpisodeCache`, footprint a pure function of the pose key, `heading_rad` a
  required snapshot field). Findings acted on in the same session: focused
  `ruff format` on the three test files (five lines over 100 columns); the
  vacuous `isinstance(x, float)` type checks replaced by exact-type checks
  (`np.float64` subclasses `float`); a regression test pinning that the lazy
  array cache stays out of `__eq__`/`__hash__` (`causal_semantic` compares
  routes with `!=` on the observation path); a test for the secondary
  segment-index key on bit-identical continuity keys; the equivalence test's
  cost reduced; this plan's status, §2 compatibility note, §5 invariants (NaN
  overflow, `float32`, ULP) and §8 units corrected. Findings recorded, not
  changed: the memo keeps a strong reference to the previous episode's adapter
  until the next episode's first query (harmless); the memo sits after
  `snapshotter(env)`, so the 4–5 snapshot captures per step remain (the
  snapshotter is out of scope, §2); the pre-classification test's `undecided`
  set restates the implementation's inequalities while its lane-set assertions
  use the independent pre-F8 gate as oracle.
- Limitation: the frozen-panel half of the equivalence test skips when the
  prepared ScenarioNet validation runtime is absent, so a gate log must show
  `test_vectorised_projection_matches_reference_on_frozen_panel_records` as
  passed rather than skipped for AC-F8-01's panel evidence to count.
- 2026-09-08, **remaining cost after M2** (cProfile of the same case, 593
  steps, 283 s of step loop under the profiler, ~1.5× overhead): `project` is
  down to 310 459 calls and 14 s (**4 %**). The step is now the semantic
  observation, 153 s (**54 %**): `causal_semantic._build_static_v12` 123 s, of
  which `_local_tangent_rad` 105 s — 75 904 calls (128 per step) that build one
  Shapely `LineString` per boundary segment of every local map feature (5.7 M
  `LineString` constructions, 6.5 M `distance` calls). Second is the Rulebook
  transition, 100 s (**35 %**): `associate_route_lane` 36 s (61 calls per step,
  one per actor, reading `polygon.bounds` on every route lane — 6.9 M `bounds`
  calls). Environment build is 16 s per env, 9.5 s of it parsing the 10 500-record
  scenario catalog, not a per-step cost. Recorded as `docs/open_items.md` `F9`;
  out of this plan's scope (observation code, not Rulebook geometry).
