# ExecPlan: Driving Mission v1.1 Route Coordinate and v1.1.1 Amendment

## 1. Status and authorization

- Specification: `DRIVING-MISSION-V1.1`, `APPROVED`, `Authoritative: YES`
- ADR: `ADR-054`, `APPROVED`
- Plan status: `APPROVED`
- Amendment: `DRIVING-MISSION-V1.1.1`, `APPROVED`, `Authoritative: YES`
- Amendment ADR: `ADR-055`, `APPROVED`
- Production implementation authorization for the amendment: `YES`, explicit user approval recorded 2026-08-04
- Project index: v1.1.1 registration authorized
- Branch: `codex/route-coordinate-mission`
- Last updated: `2026-08-06`

This plan is the authorized implementation record for v1.1 and the approved
v1.1.1 amendment. It does not authorize scientific fallback, source-data
mutation, record exclusion/replacement, or route-planning alternatives.

## 2. Objective and scope

Prepare one immutable source-independent route mission for all 3,500 frozen scenarios. Preserve UID/source/split identity; orient distinct occurrences offline using the frozen assigned lane-ID sequence and the complete SDC trajectory only as an offline orientation witness; trim one mission-local canonical XY-arc-length route from reset to terminal goal; then apply the approved exact cursor, shared `MissionSnapshot`, pure R4, separate completion values, fixed observations, and one passive final gate. During dataset generation, build and validate this mission immediately after Rulebook filtering and before split selection. Records that fail mission construction are reported and removed from the candidate pool before split feasibility is evaluated, allowing the existing replenishment loop to acquire more source scenarios. Exclude no record from the already audited 3,500 frozen source selection under the approved migration.

Out of scope: source changes, future-SDC runtime access, native navigation authority, R1--R3 changes, new dependencies, online replanning, runtime lane/neighbor/boundary search, proximity fallback, component expansion, and implementation before approval.

## 3. Authoritative requirements and acceptance

| Requirement | Acceptance | Status |
|---|---|---|
| normalized route | immutable ordered oriented 3D occurrences and directly serialized mission-local route, finite and connected, no invented joins | implemented |
| reset/goal | explicit reset/goal endpoints, `s_start=0`, positive trimmed `s_goal` | implemented |
| exact cursor | contiguous search, exact unsaturated `s`, no clamp/freeze/recovery | implemented |
| R4/completion | exact formulas and separate monotone maximum | implemented |
| shared snapshot | semantic/LiDAR/reward/Rulebook/metrics consume one snapshot | implemented (M8); Rulebook wrong-way/wrong-carriageway intentionally excluded per v1.1.1 §6 legal-direction independence |
| final gate | one offline geometric component segment covering the canonical anchor, passive runtime | audit: 3,500 unique |
| migration | preserve all 3,500 UID/path/split/source identities | implemented; pre-split eligibility integrated |

Stable acceptance IDs: `AC-RCM-001` canonical route round-trip and finite
geometry; `AC-RCM-002` first-occurrence reset normalization; `AC-RCM-003`
exact cursor without jump envelope; `AC-RCM-004` signed delta-s R4; `AC-RCM-005`
instantaneous/max completion; `AC-RCM-006` shared snapshot identity;
`AC-RCM-007` ten route samples; `AC-RCM-008` anchor gate and directed crossing;
`AC-RCM-009` termination/truncation; `AC-RCM-010` full migration identity.

Amendment acceptance IDs: `AC-RCM-011` occurrence orientation from temporal
source-station progression; `AC-RCM-012` occurrence connectivity without route
replacement or connectors; `AC-RCM-013` mission-local reset/goal trimming and
positive XY length; `AC-RCM-014` absence of trajectory in serialized/runtime
state; `AC-RCM-015` explicit reporting of mission/legal-direction conflicts.

## 4. Current repository analysis

The canonical index is `data/scenarionet/frozen/scenario_selection_index.json`; source files are under `/scratch/e.respino/thesis-metadrive/data/scenarionet`. Index and records are byte-identical after alignment. Read-only availability/content checks passed 3,500/3,500.

The geometric builder uses the canonical `0.01 m` geometry tolerance and a vertical compatibility check. It audits every static lane polygon at the goal cross-section, applies static tangent concordance, parameterizes the section as `x(u)=g+u*n` with `g=r(s_goal)`, forms intervals, merges overlaps and tolerance-adjacent intervals, and selects the unique component covering `u=0` with a closed tolerance-equivalent `covers` predicate. The final occurrence is only a consistency check. Results: PG 1,695 built; Waymo 1,805 built; 3,500 total; zero anchor-missing gates; zero anchor ambiguities.

- `waymo:training_20s:dffbfc6327b84335`
- `waymo:training_20s:b1128b1ac9f4e56a`
- `waymo:training_20s:dd1ea48c3239f622`
- `waymo:training_20s:fc680771abf4226f`

The four formerly ambiguous records now have one anchor-covering component each and are not excluded. More than one anchor-covering component after tolerance merging is a builder error; no tie-break is allowed. The earlier 210 Waymo metadata-unresolved result is retained only as historical diagnostic evidence.

The initial occurrence-orientation audit used a global polygon scan. Because
lane polygons overlap at junctions, that method could assign late poses back
to an earlier occurrence and invert its apparent station progression. The
builder now uses a sequence-constrained association that advances only through
the frozen occurrence order and never revisits a prior occurrence. A corrected
read-only sweep over all 3,500 records built 3,500 positive mission-local
routes: 1,695 PG and 1,805 Waymo, with 12,405 `FORWARD` and 1 `REVERSED`
occurrence classifications and no unresolved orientation.

For `waymo:training_20s:5e7bbc00b872c2ba`, all 199 valid poses are contained by
exactly one static lane candidate (`89`), the source station sequence has 187
negative, 0 positive, and 11 zero changes, and the occurrence is uniquely
`REVERSED`. Its projected reset is source station `59.823228 m`, its terminal
is `5.376686 m`, and the trimmed mission-local length is `54.368483 m`.

`VERIFIED`: v1 code in `src/thesis_rl/mission/` modeled ordered sections and
graph distance. `VERIFIED`: the v1.1 implementation adds the normalized route
fields, frozen `FinalGateSegment`, source builder, route-coordinate tracker,
passive runtime path, signed-delta progress, and explicit completion metrics.
The approved v1.1.1 contract represents source centerline orientation per frozen
route occurrence and trims the canonical route to reset and goal offline. The
implementation and artifact regeneration are authorized. The Parquet catalog
additionally requires the repository's optional `pyarrow` environment.

### Dataset-pipeline placement

The authoritative generation path is now:

`source generation/acquisition → raw catalog → Rulebook v2 filtering and driving-mission build → split feasibility and replenishment → thresholds → runtime views → freeze`

`filter_rulebook_v2_catalog` constructs a mission for every Rulebook-eligible
candidate and writes `driving_mission` into the filtered catalog. It also emits
`rulebook_v2/driving_mission_eligibility.json`, including deterministic builder
identity, per-record errors, and source counts. A candidate without a valid
v1.1.1 mission is not passed to split selection. The split CLIs accept
`--require-driving-mission` and reject a catalog that is not mission-ready; the
normal v1.1/v1.2 orchestration enables this flag. The freeze CLI always requires
the approved mission schema, so a post-freeze repair cannot silently produce an
official dataset.

The existing `materialize_driving_missions` command remains a migration and
diagnostic tool for already frozen catalogs. It is not the normal generation
path and cannot replace the pre-split eligibility stage.

## 5. Decisions and approval gates

| ID | Question | Status |
|---|---|---|
| `DEC-RCM-001` | approve complete specification and ADR | approved 2026-08-04 |
| `DEC-RCM-002` | enforce unique anchor-covering component and builder errors for zero/multiple components | approved by anchor-based contract; implementation test required |
| `DEC-RCM-003` | regenerate dependent artifacts without changing source records | approved policy; implementation pending |
| `DEC-RCM-004` | internal module/schema decomposition | deferred until contract approval |

The final gate schema is one frozen `final_gate_segment`, not a required complete source-semantic lane set and not a runtime envelope. It contains world geometry, static tangent, elevation, final occurrence identity, provenance/evidence, source geometry hash, and builder identity.

## 6. Planned milestones after approval

- [x] M0 — explicit approval of v1.1 specification and ADR (2026-08-04).
- [x] M1 — freeze v1.1 normalized-record schema and correction-first migration inputs.
- [x] M2 — implement the previously approved v1.1 route, cursor, and snapshot baseline with deterministic tests.
- [x] M3 — integrate the previously approved v1.1 R4, completion, observation, and consumer baseline.
- [x] M4 — complete the previously approved v1.1 anchor-gate audit.
- [x] M5 — approve and implement v1.1.1 occurrence orientation and mission-local trimming (2026-08-04).
- [x] M6 — implement sequence-constrained occurrence association and run the corrected 3,500-record amended audit (2026-08-04).
- [x] M7 — integrate offline mission eligibility before split selection; require the mission schema in split and freeze paths; add catalog/report and regression coverage (2026-08-04).
- [x] M8 — wire semantic observation (OBS-V1.3/`causal_semantic.py`) and the LiDAR 22D navigation block (`assigned_route.py`) to the mission's canonical route and the shared per-step `MissionSnapshot.s_m`, closing `AC-RCM-006` for these two consumers; keep Rulebook wrong-way/wrong-carriageway (`road.py`) on the legacy source-declared-direction route per v1.1.1 §6 (2026-08-06).

## 7. Mandatory validation matrix

Test finite/connected route geometry; first-occurrence reset association; goal containment and positive global station; self-intersection/roundabout/parallel/vertical cursor cases; exact forward/reverse/off-route R4 and speed-cap invariance; completion decrease and maximum monotonicity; shared snapshot identity; cross-section interval creation/merging, singleton and multi-lane contiguous components, non-guidable gaps, opposite/perpendicular/vertical exclusions, missing-final-occurrence detection, and ambiguity reporting; directed bumper crossing; termination/truncation; future-SDC causality; migration non-overwrite; and PG/Waymo smoke tests.

The focused implementation tests map as follows: `tests/test_driving_mission_v11.py`
(`AC-RCM-001`, `AC-RCM-003`, `AC-RCM-008`),
`tests/test_driving_mission_materialize.py` (`AC-RCM-010`), and
`tests/test_rulebook_v2_progress.py` (`AC-RCM-004`).

## 8. Validation commands

Executed commands include the focused mission/route/progress suite, the
ScenarioNet pipeline regression suite, the complete read-only
sequence-constrained builder sweep, focused Ruff checks, and shell syntax
validation. The real catalog regeneration remains a data operation and was not
run in this implementation pass.

## 9. Findings and reconciliation

The 13 reset and 25 terminal preliminary failures were uniquely repairable offline; `PGMap-6000153` has a positive global goal station after route concatenation. These approved migration decisions remain in scope. The geometric final-gate audit supersedes the former source-metadata completeness requirement. The first orientation audit falsely reported one invalid route because it globally reused overlapping lane polygons. The corrected sequence-constrained builder resolves all 3,500 records, including `waymo:training_20s:6042e6c648fca15d`, with no exclusion or replacement. The stale pre-normalization generated JSON was moved to `/tmp` and is not canonical.

## 10. Progress and findings log

- 2026-08-04: user approved specification, ADR, index promotion, and implementation.
- 2026-08-04: promoted documents and corrected the pre-existing bounded/clamp index entry.
- 2026-08-04: implemented v1.1 route record, anchor gate, route tracker, passive runtime, signed-delta progress, and completion fields.
- 2026-08-04: focused v1.1.1 mission/materialization/progress tests passed (`12 passed`); the target false conflict was traced to global polygon reassociation, fixed with sequence-constrained occurrence association, and the corrected 3,500-record sweep passed (`3,500/3,500`).
- 2026-08-04: Parquet catalog promotion not completed because `pyarrow` is unavailable; no dependency was added.
- 2026-08-04: integrated mission construction into the Rulebook pre-split filter; records failing mission construction are reported and excluded before split feasibility/replenishment, while split and freeze reject non-mission-ready catalogs. The focused pipeline and mission suite passed (`105 passed`).
- 2026-08-04: the user ran `make scenarionet-rebuild-existing` against the already-materialized sources and reported that the regenerated frozen index did not match the previously frozen `data/scenarionet/frozen/scenario_selection_index.json` (1,750 Waymo / 1,750 PG vs. the historical 1,805 Waymo / 1,695 PG; only 1,460 UIDs in common; 729 shared UIDs changed split). Root cause: `scenarionet-rebuild-existing` drives `build_splits` (the superseded v1.1 `balanced_arm_source` CLI with literal per-source targets from `conf/scenarios/pipeline_v1.yaml`), while the currently frozen index was built by the v1.2 empirical-holdout methodology (`build_splits_v1_2`, `SCENARIONET-INTEGRATION` v1.3, `AUTHORITATIVE`, approved 2026-07-31, ADR-037/ADR-040/ADR-041/ADR-042): exact 1,100 Waymo + 1,100 PG train, 150+150 validation, `test_waymo_empirical` 400 + `test_pg` 300 + `test_arm_stratified` 300, PG holdout batches instead of literal PG targets. This is not a new deviation, just the wrong existing entry point: `build_splits_v1_2` already carries `--require-driving-mission` and the `driving_mission_eligibility` wiring from the same 2026-08-04 integration (`scenarionet-v1-2-rebuild` in the Makefile), so no additional mission-integration code change was needed. Added `scenarionet-v1-2-rebuild-existing` (`Makefile`), chaining `scenarionet-v1-2-rebuild` -> `scenarionet-v1-2-build-panels` -> `scenarionet-v1-2-freeze`, as the v1.2-methodology counterpart of `scenarionet-rebuild-existing`: it revalidates/rebuilds from already materialized Waymo/PG sources (no PG holdout regeneration, no Waymo acquisition) and is the correct command to pick up the driving-mission integration without silently switching dataset methodology. `make -n scenarionet-v1-2-rebuild-existing` confirmed the expanded command chain includes `--mission-eligibility-output` and `--require-driving-mission`.
- 2026-08-04: ran `make scenarionet-v1-2-rebuild-existing` (64 workers, real host, `HOST_DATA_DIR=/scratch/e.respino/thesis-metadrive/data`) with the mission integration but before the fix below. Aggregate composition matched the historical frozen index exactly (3,500 total; 1,805 Waymo / 1,695 PG; train 1,100/1,100; validation 150/150; test 555/445; all 3,500 records carry a `driving_mission`). UID-level identity did not: 1,192/3,500 records (34%) were replaced by different scenarios relative to the pre-run backup (`/scratch/.../frozen/backup-before-index-alignment-20260804_131833/scenario_selection_index.json`). Cross-checked against `driving_mission_eligibility.json`: 0 of the 1,192 displaced records were mission-excluded (`mission_excluded_records` = 4,574, entirely disjoint from the displaced set) -- they were simply not reselected. Root cause isolated to `_solve_scalable_singleton_group_assignment` in `src/thesis_rl/scenarios/pipeline.py` (reached from `assign_training_pool_from_residual` -> `assign_arm_balanced_splits_to_targets`, the v1.2 train-pool allocator): each `(source, arm)` candidate cell was ordered with `np.random.default_rng(seed).shuffle(group_ids)`, whose output is a function of the shuffled list's *length*, not just the seed -- population-size-dependent, so excluding ~27.5% of eligible Waymo candidates (mission-invalid) reshuffled the ordering for every remaining candidate in a cell, not only the excluded ones. This directly contradicts `SCENARIONET-INTEGRATION` v1.2's population-size-independence requirement (`REQ-005`, "a frozen empirical holdout must be provably unaffected by scenarios acquired afterward"), already upheld elsewhere in the same module via `_stable_permutation_key` (used by `reserve_empirical_holdouts` and the stratified-pool per-cell ordering) but never applied to this one call site. Not a new deviation to approve: fixing it aligns the code with the already-approved REQ-005, using the module's own established pattern.
- 2026-08-04: fixed `_solve_scalable_singleton_group_assignment` (`src/thesis_rl/scenarios/pipeline.py`) to order each `(source, arm)` cell with `sorted(..., key=lambda name: _stable_permutation_key(...))` instead of `np.random.default_rng(seed).shuffle(...)`. Added regression test `test_arm_balanced_selector_is_stable_when_unselected_candidates_are_removed` (`tests/test_scenarionet_pipeline.py`): removing 10 never-selected candidates from a 480-entry synthetic pool must not change any other candidate's split. Confirmed the test fails without the fix (2/24 selected assignments changed) and passes with it. Full `tests/test_scenarionet_pipeline.py` (57 -> 58 tests) plus the focused mission/progress suites: `71 passed`. `ruff check` and `ruff format --check` on both touched files: PASS.
- 2026-08-04: re-ran `make scenarionet-v1-2-rebuild-existing` twice in a row with the fix (64 workers, real host, unchanged sources). Both runs produced the historical aggregate composition (3,500 total; 1,805 Waymo / 1,695 PG; train 1,100/1,100; validation 150/150; test 555/445; all 3,500 records carry `driving_mission`). Compared against the pre-fix run and the original (pre-mission, 2026-07-31) frozen index, substantial turnover remains (~1,500/3,500 records), but this is expected and one-time: it is driven jointly by (a) the mission-eligibility filter itself excluding 4,574 previously-unvetted candidates (22% of the Waymo pool) -- a required scientific change, not an artifact -- and (b) switching the allocator from the unstable shuffle to the stable key is itself a one-time re-derivation. What the fix actually guarantees, and what was verified directly, is **idempotency going forward**: comparing the two consecutive post-fix runs record-by-record (`scenario_uid`, `source`, `split`, and the full `driving_mission` payload) found **0/3,500 differing records**; only the top-level `created_at` freeze timestamp (and the `selection_hash` derived from it) differ. This confirms `_solve_scalable_singleton_group_assignment` no longer reshuffles the selection under a stable candidate pool, closing the churn risk for future re-runs that do not change the eligible pool. `AC-RCM-010`/the migration requirement is satisfied prospectively from this frozen index onward, not retroactively against the pre-mission 2026-07-31 baseline (that one-time transition is unavoidable, per the two causes above). The new frozen index lives at `/scratch/e.respino/thesis-metadrive/data/scenarionet/frozen/scenario_selection_index.json` (host mount); the git-tracked copy at `data/scenarionet/frozen/scenario_selection_index.json` was not touched by this run and remains the 2026-07-31 pre-mission snapshot -- syncing/committing the new ~196 MB index is a separate decision pending user confirmation.
- 2026-08-06: **M8 — shared-consumer unification (`AC-RCM-006`)**. A prior investigation session found that R4/completion/success/termination already consumed the shared `MissionSnapshot` (via `MissionRuntime`), but semantic observation (OBS-V1.3, `causal_semantic.py`), the LiDAR 22D navigation block (`assigned_route.py`), and Rulebook wrong-way/wrong-carriageway (`road.py`) still independently reprojected the ego onto a legacy `RoutePolyline` (`episode_cache.route_polyline`, built from `assigned_route_lane_ids` in declared order, not the mission's trimmed/oriented `canonical_route_points_xyz`) with an unanchored global-nearest search -- contradicting `driving_mission_v1.1_specification.md` §3/§5 (*"R4, completion, semantic/LiDAR observation, Rulebook route relevance, metrics, and video consume that snapshot"*; *"never reproject ego independently"*). This work item was already authorized (ADR-052 consumer list, `AC-RCM-006`, ExecPlan row `shared snapshot | ... | planned`) and did not require new approval, only implementation.

  Re-reading `driving_mission_v1.1.1_amendment.md` §6 during planning surfaced a correction to that investigation: §6 explicitly states the amendment *"does not mutate ... source lane direction used by Rulebook legality, wrong-way semantics, or wrong-carriageway semantics"* and requires an explicit conflict report if mission-local orientation disagrees with source-declared legal direction. Since exactly one of the 12,406 frozen route occurrences is `REVERSED` (the mission traverses that occurrence opposite to its declared direction), swapping `road.py`'s route to the mission-oriented route would have silently flipped wrong-way legality for that record. `road.py` was therefore **excluded** from this milestone and kept on the legacy source-declared-direction route; a regression test now pins this (`test_wrongway_uses_source_declared_direction_independent_of_mission_orientation`, `tests/test_rulebook_v2_road.py`).

  Implementation: `MissionRuntime` (`src/thesis_rl/mission/runtime.py`) gained a public `route` property exposing the tracker's canonical `RoutePolyline` (fail-closed for non-route-coordinate missions). `CausalSceneContext` gained a required `mission_route` field, threaded from `self._mission_runtime.route` at both the per-reset (`thesis_scenario_env.py::_prepare_initial_causal_context`) and per-step (`RulebookV2Adapter.mission_route` -> `RulebookV2MonitorWrapper._publish_causal_context`) publication points, replacing `episode_cache.route_polyline`. `causal_semantic.py`'s and `assigned_route.py`'s ego-position projection call sites (`self.route.project(ego.position_xy, ...)`) now read `context.snapshot.mission_snapshot.s_m` directly instead of independently searching; call sites that also need the route tangent/lateral offset at the ego (heading error, lane offset, front-bumper station) use an anchored query (`previous_s_m=mission_s_m`) and raise if the result disagrees with the committed station beyond `1e-6 m`, so a future regression cannot silently reintroduce independent reprojection. Other-actor and static-feature route projections remain independent per-object searches (they are not "the ego"), but now run against the mission's canonical route object instead of the legacy one, keeping every route-relative value in the same coordinate frame as R4/completion.

  A pre-existing, unrelated bug was discovered and fixed as a byproduct: `MissionRuntime.snapshot` called `self._tracker.snapshot()` (method-call style), but `RouteCoordinateMissionTracker.snapshot` is a `@property` (only the legacy `MissionTracker.snapshot` is a plain method), so `MissionRuntime.snapshot` raised `TypeError: 'MissionSnapshot' object is not callable` for every v1.1.1 route-coordinate mission -- i.e. for the entire current frozen dataset, at the very first read in `_install_mission_runtime`. This was masked in the pre-existing unit-test suite (which drives trackers through `.update()`, not the `.snapshot` property, for its assertions) and was only exposed once the real `test_rulebook_v2_scenarionet_integration.py` integration test could reach `env.reset()`. Confirmed pre-existing via `git stash` against the unmodified branch. Fixed with a type-dispatched read in `MissionRuntime.snapshot`; regression test added in `tests/test_driving_mission_v11.py` asserting `runtime.snapshot` (not just `runtime.update()`) is readable immediately after construction and agrees with a subsequent `update()`.

  Because the route source changes (mission-trimmed canonical route vs. legacy untrimmed route) even for ordinary `FORWARD` scenarios, the observation's numeric route-derived values change; this is the explicitly authorized `DEC-MSN-004` consequence (*"observation schema change policy: new schema IDs, same tensor widths where sufficient"*), not a new deviation. `SemanticObservationSchemaV11.version`/`SemanticObservationSchemaV12.version` were bumped (`1.1-final` -> `1.1-final-mission-route-v1`, `1.2-perception-bounded` -> `1.2-perception-bounded-mission-route-v1`) so `observation_schema_version`/checkpoint-manifest identity correctly flags pre-existing checkpoints as incompatible; `required_observation_schema` in the four `conf/agent/planner/encoder/*.yaml` presets was updated to match. `encoder_architecture_version`/`architecture_version` (a separate, encoder-network-identity axis, config-driven, unaffected by this change) was left untouched.

  A follow-up audit (same day, in response to a user question about consumer coverage) found one more independent-reprojection consumer not covered by the milestone as originally scoped: `diagnostic_geometry` in `src/thesis_rl/runtime/io/video_diagnostics.py` (the qualitative-video/GIF route-ahead overlay) read `causal_scene_context.route_polyline` (the legacy, forwarded from `episode_cache.route_polyline`) and called `route.project(ego_position)` with no anchor, an independent global-nearest search -- the same defect class as the observation/LiDAR gap, but for the video overlay explicitly named in spec §3's consumer list. `transition.py`'s use of `cache.route_polyline` for control-line crossing (`_front_s`) and approach-speed tangent was re-checked and confirmed *not* a gap: it is the same source-declared-direction carve-out as `road.py` (v1.1.1 §6), not an oversight. Fixed `diagnostic_geometry` to read `context.mission_route` and anchor the projection to `context.snapshot.mission_snapshot.s_m` (`previous_s_m=mission_s_m`), matching the established pattern; the now-unreachable legacy `episode_cache.route_polyline` fallback branch was removed since `context` being unavailable already short-circuits the whole overlay block. `CausalSceneContext.route_polyline` remains defined (harmless, no other caller found repo-wide) but is no longer read by any production code. Updated the three affected fixtures in `tests/test_video_diagnostics.py` to populate `mission_route`/`snapshot.mission_snapshot.s_m` instead of `route_polyline`; 15/16 tests in the file pass, the one pre-existing failure (`test_geometry_extraction_projects_mission_gates_with_progress_colours`, an unrelated screen-offset bug in the mission-gate overlay) confirmed via `git stash` to reproduce identically on the unmodified branch.

  Two more follow-ups closed the same day, both user-requested. First, coverage symmetry: `road.py`'s source-declared-direction carve-out had a dedicated regression test, but `transition.py`'s equivalent (`_front_s`, control-line crossing) did not -- only manual code reading. Added `test_front_s_uses_source_declared_direction_independent_of_mission_orientation` (`tests/test_rulebook_v2_transition.py`), mirroring the `road.py` test: it calls `transition_module._front_s` directly with a source-declared route (`(0,0)->(20,0)`) and a mission-reversed route (`(20,0)->(0,0)`) for the same ego footprint, and shows the geometric maximum-`s` vertex flips from the vehicle's real leading edge (`s=6.0`, correct) to its real trailing edge (`s=16.0`, wrong) when fed the mission-oriented route -- concretely demonstrating why `evaluate_transition` must never receive it. Built with a self-contained `MissionSnapshot` via the file's own `_snapshot()` helper rather than routing through `evaluate_transition` end-to-end, because the file's `_snapshot()` fixture leaves `MissionSnapshot.s_m` unset, which is the same pre-existing, unrelated defect behind 14 of this file's other failing tests (confirmed unchanged before/after: `14 failed, 6 passed`, versus `14 failed, 5 passed` before this addition -- net effect is exactly the one new passing test). Second, the video-overlay pre-existing failure noted above was investigated on request: root cause is a wrong hardcoded expectation in the test itself, not the production code -- `to_screen()` in `video_diagnostics.py` is camera-relative (subtracts `camera_pixel - screen_center`) consistently for every overlay (`route_future`, `ego_trail`, `mission_gates`), but the `mission_gates` fixture's expected coordinates were authored as raw `pos2pix(point)` without accounting for the camera offset. Corrected the expected values in `tests/test_video_diagnostics.py` (recomputed directly from the production formula, not guessed); all 16 tests in the file now pass, no production code changed. Both changes verified with focused Ruff check/format PASS and `git diff --check` PASS.

## 11. Deviations

The v1.1.1 occurrence orientation and mission-local trimming amendment is now
approved and implemented in the mission builder/runtime path. The legacy v1
classes remain only for deserialization and compatibility tests; the v1.1.1
route path does not use intermediate gates, graph distance, envelope expansion,
native navigation, or a source-station offset as authority.

## 12. Files

| Path | Action |
|---|---|
| specification | approved authority |
| ADR-054 | approved decision |
| this ExecPlan | approved implementation record |
| audit findings | append geometric audit evidence |
| `data/scenarionet/frozen/scenario_selection_mission_v1_1_1.json` | pending | atomic promotion still requires the provisioned artifact-generation environment (`pyarrow`); builder audit is complete |
| `docs/project_index.md` | modified | register v1.1.1 authority and ADR-055 |
| `src/thesis_rl/scenarios/mission_eligibility.py` | added | deterministic pre-split mission builder/audit adapter |
| `scripts/prepare_scenarionet_dataset.sh` | modified | build mission before split and retry feasibility after mission filtering |
| `Makefile` | modified | added `scenarionet-v1-2-rebuild-existing`, the v1.2-methodology (authoritative) counterpart of `scenarionet-rebuild-existing`, so revalidation against already materialized sources uses the correct, currently frozen split methodology |
| `src/thesis_rl/scenarios/pipeline.py` | modified | bugfix: `_solve_scalable_singleton_group_assignment` now orders each source/arm candidate cell with the population-size-independent `_stable_permutation_key` instead of a length-dependent `numpy` shuffle, so excluding non-selected candidates no longer reshuffles the rest of the train-pool selection |
| `tests/test_scenarionet_pipeline.py` | modified | added `test_arm_balanced_selector_is_stable_when_unselected_candidates_are_removed` regression test |
| `src/thesis_rl/mission/runtime.py` | modified | M8: public `route` property (fail-closed for non-route-coordinate missions); bugfix: `snapshot` property now dispatches on tracker type instead of always calling `.snapshot()` |
| `src/thesis_rl/contracts/causal_scene_context.py` | modified | M8: added required `mission_route` field |
| `src/thesis_rl/envs/thesis_scenario_env.py` | modified | M8: `_build_causal_frame_builder` and `_install_causal_observation_builder` source `route` from `self._mission_runtime.route`; `_prepare_initial_causal_context` populates `CausalSceneContext.mission_route`; `RulebookV2Adapter` construction passes `mission_route` |
| `src/thesis_rl/rulebook/v2/wrapper.py` | modified | M8: `RulebookV2Adapter`/`RulebookV2MonitorWrapper` carry and publish `mission_route` into every per-step `CausalSceneContext` |
| `src/thesis_rl/runtime/wiring/builders.py` | modified | M8: threads `mission_route` from the adapter into `RulebookV2MonitorWrapper` construction |
| `src/thesis_rl/envs/observations/causal_semantic.py` | modified | M8: ego route-station reads come from `context.snapshot.mission_snapshot.s_m`; tangent/lateral queries anchored to it with a divergence guard; route-identity guard checks `context.mission_route` |
| `src/thesis_rl/envs/observations/assigned_route.py` | modified | M8: `AssignedRouteWaypointAdapter`/`MapRouteNavigationObservation22` read `s_m` from a `mission_provider` instead of independently projecting the ego |
| `src/thesis_rl/contracts/observation_schema.py` | modified | M8: bumped `SemanticObservationSchemaV11`/`V12` `version` (`DEC-MSN-004`) |
| `conf/agent/planner/encoder/{lq,lq_v3,lq_v3_lite,lq_v3_micro}.yaml` | modified | M8: `required_observation_schema` updated to match the bumped schema versions |
| `tests/test_rulebook_v2_road.py` | modified | M8: regression test pinning wrong-way legality independence from mission orientation |
| `tests/test_driving_mission_v11.py` | modified | M8: regression test for the `MissionRuntime.snapshot` property bugfix |
| `tests/test_causal_semantic_batch.py`, `tests/test_observation_v13_corrections.py`, `tests/test_assigned_route_observation.py`, `tests/test_causal_lidar.py`, `tests/test_causal_scene_context.py`, `tests/test_rulebook_v2_causal_context.py`, `tests/test_thesis_scenario_env.py` | modified | M8: fixtures updated for the new `mission_route`/`mission_snapshot` wiring; added `test_ego_route_station_is_bit_identical_to_the_committed_mission_snapshot` |
| `tests/test_observation_schema_v11.py`, `tests/test_observation_schema_v12.py`, `tests/test_checkpointing.py`, `tests/test_hydra_preset_test_configs.py` | modified | M8: updated pinned schema-version literals |
| `src/thesis_rl/runtime/io/video_diagnostics.py` | modified | M8 follow-up: `diagnostic_geometry`'s route-ahead overlay reads `context.mission_route` anchored to `context.snapshot.mission_snapshot.s_m` instead of the legacy `route_polyline` with an unanchored global projection |
| `tests/test_video_diagnostics.py` | modified | M8 follow-up: three fixtures updated for `mission_route`/`snapshot.mission_snapshot.s_m`; corrected the `mission_gates` test's camera-relative expected coordinates (pre-existing test-only bug, unrelated to route/mission semantics) |
| `tests/test_rulebook_v2_transition.py` | modified | M8 follow-up: added `test_front_s_uses_source_declared_direction_independent_of_mission_orientation`, coverage-symmetric with the `road.py` regression test |

## 13. Validation results

| Command | Result | Notes |
|---|---|---|
| focused mission/progress tests | PASS | 12 tests including reversed occurrence and mission-local endpoint regression |
| 3,500-record sequence-constrained builder audit | PASS | 3,500/3,500 routes built; 3,500/3,500 positive `s_goal`; no exclusions |
| `git diff --check` | PASS | 2026-08-04 |
| Parquet catalog promotion | NOT_RUN | `pyarrow` missing; install/use provisioned environment before catalog build |
| full `make test` | NOT_REPEATED | Earlier baseline run: 1,266 passed and 51 failures outside this scoped pipeline integration; the current acceptance evidence is the focused 105-test suite below |
| `make lint`, format, config, smoke | NOT_RUN | final reconciliation pending |
| focused ScenarioNet/mission pipeline pytest suite | PASS | 105 tests in Docker, 2026-08-04 |
| focused Ruff check and format check | PASS | all modified Python files, 2026-08-04 |
| `bash -n scripts/prepare_scenarionet_dataset.sh` | PASS | 2026-08-04 |
| M8 focused suite (mission, observation, Rulebook road/wrapper/causal-context, checkpoint identity, hydra presets) | PASS | 484 tests, Docker `dev` service, 2026-08-06 |
| M8 real-data integration (`tests/test_rulebook_v2_scenarionet_integration.py`) | PASS | `env.reset()` + `env.step()` against real frozen PG/Waymo ScenarioNet data, 2 scenarios, 2026-08-06; exercises the full M8 wiring end-to-end (mission runtime install, per-step `CausalSceneContext` publication, semantic/LiDAR observation build) |
| `git stash` differential check confirming pre-existing failures are unrelated | PASS | 15 failures in `test_rulebook_v2_transition.py`/`test_thesis_scenario_env.py::test_thesis_reward_suppresses_native_short_route_bonus` reproduce identically on the unmodified branch, 2026-08-06 |
| M8 focused Ruff check and format check | PASS | all M8-modified Python files, 2026-08-06 |
| `git diff --check` | PASS | 2026-08-06 |
| `make smoke` | NOT_REPEATED | superseded by `make smoke-gpu`, below (this host has a GPU) |
| full `make test` | NOT_REPEATED | M8 acceptance evidence is the focused suite plus the real-data integration test and `make smoke-gpu` |
| `make smoke-gpu` | PASS | 2026-08-06. The `pyarrow` gap no longer applies (pyarrow 25.0.0 is installed in the `dev` image). The specific missing artifact, `scenario_catalog_driving_mission_v1_1_1.parquet`, turned out to already exist in substance: `scenario_catalog.parquet` on the host `HOST_DATA_DIR` mount (`/scratch/.../data/scenarionet/catalog/`, built 2026-08-04 by `scenarionet-v1-2-rebuild-existing`) already carries `driving_mission`/`rulebook_eligible` for all 3,500 records -- it was simply never materialized under the filename `MISSION_CATALOG_FILENAME` (`src/thesis_rl/envs/factory.py`) expects for evaluation. Copied it to that filename (additive, host filesystem, outside git) to unblock evaluation. Ran GPU training end-to-end (2000/2000 env steps, GH200 480GB): `exit 0`, no errors/tracebacks; the periodic evaluation at step 1000 (`eval_interval: 1000`) and the final evaluation both completed against real frozen PG scenarios (`validation_pg`, 10/10 episodes per batch), exercising the full M8 wiring (mission runtime, per-step `CausalSceneContext`, semantic/LiDAR observation, video diagnostics overlay) under GPU training, not just unit/integration tests. Low success/route-completion numbers are expected and uninformative at this budget (2000 steps, an untrained policy) -- the acceptance signal is pipeline correctness (no crash, no shape/consistency error), not policy quality. **Follow-up decision needed**: whether to formalize `scenario_catalog_driving_mission_v1_1_1.parquet` production (e.g. a dedicated build/rename step in `scripts/materialize_frozen_scenarionet.sh` or `scenarionet-v1-2-*`) instead of the ad hoc copy made here, which is not reproducible from a clean host. |
| `tests/test_video_diagnostics.py` (M8 follow-up: video/GIF overlay) | PASS (16/16) | initially 15/16; the one failure was a wrong hardcoded expectation in the test itself (camera-relative `to_screen` offset), corrected; all 16 pass, 2026-08-06 |
| focused Ruff check/format on `video_diagnostics.py`/`test_video_diagnostics.py` | PASS | 2026-08-06 |
| `tests/test_rulebook_v2_transition.py::test_front_s_uses_source_declared_direction_independent_of_mission_orientation` | PASS | new regression test, coverage-symmetric with `road.py`; rest of file unchanged at `14 failed, 5 passed` baseline (now `14 failed, 6 passed`), 2026-08-06 |
| focused Ruff check/format on `test_rulebook_v2_transition.py` | PASS | 2026-08-06 |
| `git diff --check` | PASS | 2026-08-06, after both follow-ups |

## 14. Final reconciliation

`REQ-RCM-001` through `REQ-RCM-007` and amendment requirements
`AC-RCM-011` through `AC-RCM-015` are implemented for every record in the
corrected builder audit. The required 3,500/3,500 route-builder acceptance is
met: all routes are connected, positive, and have unique orientation/gate
construction, including `waymo:training_20s:6042e6c648fca15d`. The Parquet
catalog cannot be emitted in the current host without `pyarrow`; this is an
environmental artifact-generation limitation, not a mission-validity issue.
The normal pipeline now performs mission validation before split selection and
the official freeze path requires the validated mission schema. No source
pickle, planning route, or runtime fallback was introduced. The approved
3,500-record source selection remains unchanged; any future source generation
pass may replenish records only through the existing post-filter feasibility
loop.

`AC-RCM-006` is now implemented for semantic observation and LiDAR navigation:
both consume the mission's canonical route and the shared per-step
`MissionSnapshot.s_m` for the ego, matching R4/completion/success. Rulebook
wrong-way/wrong-carriageway is a deliberate, spec-mandated exception (v1.1.1
§6) and remains on the legacy source-declared-direction route, pinned by a
regression test. No new approval gate was required: the consumer-unification
decision was already authorized by ADR-052/`AC-RCM-006`; this milestone is
implementation, not a new material choice. One pre-existing, unrelated bug
(`MissionRuntime.snapshot` broken for every v1.1.1 route-coordinate mission)
was discovered and fixed with a regression test, since it blocked this
milestone's own real-data validation. `make smoke`/full `make test` remain
blocked on the same pre-existing missing-`pyarrow` artifact-generation gap
recorded in the M7 entry above; the real single-scenario ScenarioNet
integration test exercised the identical `reset`/`step` wiring against real
PG and Waymo data as an equivalent smoke signal.

The ChatGPT project source files changed during this task are
`docs/project_index.md`; replace that exact source in the project. The other
two synchronized sources, `docs/engineering_workflow.md` and
`docs/templates/specification_template.md`, were not changed.
