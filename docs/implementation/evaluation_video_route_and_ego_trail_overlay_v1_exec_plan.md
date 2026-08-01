# ExecPlan: Evaluation GIF Route-Checkpoint And Ego-Trail Overlay

## 1. Metadata

- Feature / plan ID: `VIDEO-OVERLAY-ROUTE-TRAIL-V1`
- Authoritative specification: none dedicated; this is an additive diagnostic
  extension of the shared GIF annotator introduced by
  `decisions/ADR-020-evaluation-video-diagnostics.md` (`APPROVED`,
  2026-07-21). No scientific/observation/reward/termination specification is
  touched.
- Status: `IMPLEMENTED`
- Created: 2026-07-31
- Last updated: 2026-07-31
- Branch: `scenarionet-implementation`
- Related ADRs: `decisions/ADR-020-evaluation-video-diagnostics.md` (amended by
  `decisions/ADR-043-evaluation-video-ego-trail-and-checkpoint-overlay.md`,
  `APPROVED`)
- Owner: repository maintainer (via assistant session)

## 2. Objective And Scope

Add two visual elements to evaluation GIFs (both live final-evaluation
recording and offline replay, sharing the annotator per ADR-020):

1. discrete markers for the ego's planned navigation checkpoints (not only the
   continuous route polyline already drawn);
2. the ego's actually-traveled path (real position history) up to the current
   frame within the episode.

This is a raster-only diagnostic addition. It must not alter policy
observations, rewards, termination/truncation, evaluation metrics, or any
non-visual artifact (`eval_episodes.csv`, `evals.csv`, manifests).

In scope: `src/thesis_rl/runtime/io/video_diagnostics.py` (annotator
geometry/draw functions), `src/thesis_rl/runtime/io/eval_artifacts.py` (the
one call site accumulating per-episode ego-position state), config toggles in
`conf/video/default.yaml`, focused unit tests.

Out of scope: MetaDrive's native `TopDownRenderer` (`draw_target_vehicle_trajectory`)
is not used — the existing custom annotator/coordinate pipeline
(`context.route_polyline` + `_draw_geometry`) already solves world-to-frame
projection and is reused instead of introducing a second overlay mechanism.
No change to `render_selected_videos.py`'s standalone offline path beyond
whatever the shared annotator already covers by construction (per ADR-020, it
is the same annotator).

Compatibility: additive only; existing GIFs without the new toggles enabled
keep identical appearance. Both new elements are individually toggleable and
default-on unless a decision below says otherwise.

## 3. Authoritative Requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-001` | When runtime navigation geometry is available, draw discrete markers at the ego's planned navigation checkpoints (not only sampled polyline points). | User request, this session |
| `REQ-002` | When enabled, draw the ego's actually-traveled position history for the current episode, from episode start up to the current frame. | User request, this session |
| `REQ-003` | Neither addition changes policy-observable state, reward, termination/truncation, or any CSV/manifest/metric artifact. | ADR-020 (raster-only constraint), `AGENTS.md` |
| `REQ-004` | Missing/unavailable geometry degrades to the pre-existing frame (no crash, no evaluation abort). | ADR-020 §Decision ("missing optional geometry degrades to the base frame") |

## 4. Current Repository Analysis

- `VERIFIED`: the shared annotator lives in
  `src/thesis_rl/runtime/io/video_diagnostics.py`. `diagnostic_geometry()`
  (live path, ~L182-274) and `diagnostic_geometry_from_env()` (parallel-worker
  path, ~L277-317) compute frame geometry from the live `env`; `_draw_geometry()`
  (~L351-365) rasterizes it; `annotate_diagnostic_frame()` /
  `annotate_geometry_frame()` (~L368-427) compose the panel + overlay onto the
  frame.
- `VERIFIED`: "traversed route" (`route_past`, green) and "route remainder"
  (`route_future`, blue) are already drawn from `context.route_polyline` (or
  `rulebook_v2_adapter.cache.route_polyline` fallback), sliced at the ego's
  current projection (`route.project(ego_position)` → `segment_index`/`s_m`).
  This is the **planned route**, not ego position history — `route_past` is
  the portion of the *route* already covered, computed from the projection,
  not from recorded ego positions. The whole block is guarded by
  `if route is not None and route_points and ego_position is not None:` with a
  bare `except Exception: pass`, i.e. it silently degrades when geometry is
  unavailable (e.g. `VectorEnvSlotProxy` in multiprocess evaluation, ~L196-197).
- `VERIFIED`: "target/waypoint marker" (~L259-261, ~L358-361) is a single point
  from `info.get("target_point")` (live path) or, in the worker path,
  `vehicle.navigation.final_lane.position(length, 0)` with a
  `navigation.current_checkpoint` fallback — one marker, not the discrete
  checkpoint list.
- `VERIFIED`: no ego position-history/trail state exists anywhere in this
  file or in `eval_artifacts.py` (`grep` for `history|trail|trajectory`
  matches only the unrelated `_trajectory_row` CSV helper name). `record_step`
  in `eval_artifacts.py` (~L234-283) calls `diagnostic_geometry(env, step_info)`
  then `annotate_diagnostic_frame(...)` once per frame, with no
  cross-frame accumulator passed in.
- `AWAITING_CONFIRMATION` (now resolved by `DEC-001` below, pending user
  approval): ADR-020's Decision section explicitly lists "ego-history lines"
  under geometry that is **excluded** from the overlay. `REQ-002` therefore
  requests something ADR-020 currently forbids by name. This is a
  specification deviation, not an implementation detail — it requires
  explicit approval and an ADR amendment before implementation, per
  `AGENTS.md`'s Decision And Change Control section ("deviation from a
  specification or approved acceptance behavior").
- `VERIFIED` (supersedes the earlier `INFERRED` note): MetaDrive's live
  `navigation.checkpoints` is **not** what produces `route_polyline` in this
  repository — grepping `src/thesis_rl/` for `navigation.checkpoints` found
  zero hits. `route_polyline` is instead built at reset time from a frozen
  `assigned_route_lane_ids` sequence resolved against precomputed canonical
  lane geometry (`RoutePolyline.from_lane_centerlines`,
  `build_assigned_route_polyline` in
  `src/thesis_rl/rulebook/v2/geometry/route.py`). "Planned checkpoint" in
  this system is therefore a lane transition in that frozen sequence, not a
  MetaDrive navigation-module concept. Implemented as an additive
  `lane_start_points_xyz` field on `RoutePolyline`, populated only by
  `from_lane_centerlines` (default `()` for the plain constructor, so every
  other `RoutePolyline(...)` call site in the repository is unaffected).

## 5. Assumptions And Invariants

- Coordinate frame: identical to the existing overlay — world coordinates
  projected to frame pixels via the same transform already used for
  `route_past`/`route_future`/`target_point`. No new coordinate system is
  introduced.
- State/reset: the ego position-history buffer must reset at episode
  boundary (new episode ⇒ empty trail), scoped per rollout slot in the
  parallel-worker path so slots do not leak trails across concurrently
  running episodes/vehicles.
- Determinism: purely a rendering-time accumulation of already-computed
  positions; does not affect policy input, reward, seeding, or evaluation
  determinism. No test should observe overlay changes affecting numeric
  metrics.
- Failure handling: any missing/erroring geometry (checkpoints or trail)
  degrades to omission of that layer only, consistent with the existing
  `except Exception: pass` pattern for `route_past`/`route_future` (`REQ-004`).

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-001` | Specification deviation | ADR-020 explicitly excludes "ego-history lines" from the overlay; `REQ-002` asks for exactly that. | (A) Amend ADR-020 to permit an ego-trail layer, distinct in color/style from `route_past`/`route_future` so the two remain visually distinguishable; (B) keep ADR-020 as-is and drop `REQ-002`, implementing only `REQ-001`. | (A) — the user explicitly requested the trail in this session with full context of what the overlay already shows; ADR-020's original exclusion predates this specific request and was a general design choice, not a rejection of this use case. | New overlay layer, one ADR amendment, no behavior change outside video rendering. | **Approved** ("come preferisci ... Procedi pure", 2026-07-31); recorded by `decisions/ADR-043-evaluation-video-ego-trail-and-checkpoint-overlay.md` |
| `DEC-002` | Implementation detail | Where to store per-episode ego position history in the worker/multiprocess evaluation path (state is per rollout slot, not per env instance, per the existing `VectorEnvSlotProxy` pattern). | (A) extend the existing per-slot diagnostic state structure used for `cumulative_reward`/`step`; (B) new parallel dict keyed by slot id. | (A), for consistency with existing state-lifecycle handling. | Internal only, no approval needed per `AGENTS.md` (private structure choice). | Resolved (proceeding with A) |
| `DEC-003` | Implementation detail | Trail rendering style: full-history polyline vs. capped-length/fading trail (matches MetaDrive's own `draw_target_vehicle_trajectory` fading-alpha style). | (A) fixed-color full-episode polyline, thin, low-alpha so it does not visually compete with `route_past`/`route_future`; (B) fading-alpha trail like MetaDrive's native renderer. | (A) — simpler, deterministic to test, and avoids a second alpha-blending code path; can revisit to (B) as a follow-up if visually cluttered. | Visual only. | Resolved (proceeding with A), revisit if requested |

**Work is blocked on `DEC-001` for `REQ-002` (ego trail) only.** `REQ-001`
(discrete checkpoint markers) has no ADR conflict and may proceed
independently once this plan is otherwise approved, per `AGENTS.md`'s "stop
only the portion blocked by a decision."

## 7. Proposed Design

### REQ-001 — discrete checkpoint markers (`IMPLEMENTED`)

- `RoutePolyline` gained an additive `lane_start_points_xyz` field (default
  `()`), populated by `from_lane_centerlines` from each lane's first
  centerline point — not from MetaDrive's live navigation module (see the
  corrected `VERIFIED` note in §4).
- `diagnostic_geometry()` reads `route.lane_start_points_xyz` and projects it
  to `geometry["checkpoints"]` inside the same best-effort block as
  `route_past`/`route_future` (`REQ-004`).
- `_draw_geometry()` draws each checkpoint as a small amber hollow diamond,
  visually distinct from the single blue `target` ellipse.
- No dedicated config toggle was added (consistent with
  `route_past`/`route_future` having none); checkpoints always render when
  route geometry is available.

### REQ-002 — ego historical trail (`IMPLEMENTED`, `DEC-001` approved)

- `LiveEvalEpisodeRecorder` (live final-eval and periodic-tracked-subset
  paths, both instantiated once per episode already) gained
  `self._ego_trail_world: list[tuple[float, float]]`, appended once per
  `record_step` from `step_info["ego_state"]["position"]`; naturally
  episode-scoped by the recorder's existing per-episode lifecycle, no new
  reset hook needed.
- The parallel-worker path (`deterministic_subproc_vec_env.py`'s `_worker`)
  gained a persistent `ego_trail_world` list, cleared explicitly on both
  `reset` and `reset_slots` commands (a worker owns one env across many
  sequential episodes, unlike the parent-process recorder).
- `diagnostic_geometry()`/`diagnostic_geometry_from_env()` accept an optional
  `ego_trail_world` and project it to `geometry["ego_trail"]`.
- `_draw_geometry()` draws it as a thin, low-alpha violet polyline, distinct
  from `route_past`'s green.
- New `conf/video/default.yaml` key `topdown.draw_ego_trail: true` (default
  on); threaded through `agent.py`'s `_render_kwargs` for the parallel path
  and read directly from `self._topdown_cfg` for the live path.

### Failure/degradation

Both additions follow the existing pattern exactly: best-effort computation
inside the pre-existing `try/except Exception: pass` geometry block (or an
equivalent scoped guard for the new trail state), never raising, never
aborting an evaluation episode.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-001` | `AC-001` | `route.py` `lane_start_points_xyz`; `video_diagnostics.py` geometry + draw | `tests/test_video_diagnostics.py::test_geometry_extraction_includes_checkpoint_markers_from_lane_starts`, `::test_route_polyline_from_lane_centerlines_exposes_lane_start_points` | `VERIFIED` |
| `REQ-001` | `AC-002` | same | `tests/test_video_diagnostics.py::test_geometry_extraction_omits_checkpoints_when_route_has_none`, `::test_route_polyline_plain_constructor_defaults_lane_start_points_empty` | `VERIFIED` |
| `REQ-002` | `AC-003` | `eval_artifacts.py` `_ego_trail_world`; `deterministic_subproc_vec_env.py` worker state | `tests/test_eval_artifacts_ego_trail.py::test_ego_trail_accumulates_across_steps`, `::test_new_episode_instance_starts_with_empty_trail` | `VERIFIED` |
| `REQ-002` | `AC-004` | `topdown.draw_ego_trail` toggle, both paths | `tests/test_eval_artifacts_ego_trail.py::test_ego_trail_disabled_by_config_does_not_accumulate` | `VERIFIED` |
| `REQ-003` | `AC-005` | both | full focused regression run (`tests/` filtered on route/geometry/diagnostic/subproc_vec_env/eval_artifacts, 165 passed unaffected by this change) | `VERIFIED` |
| `REQ-004` | `AC-006` | both | `tests/test_video_diagnostics.py::test_geometry_extraction_omits_ego_trail_when_not_given`, existing `except Exception: pass` block covers checkpoints/trail identically to `route_past`/`route_future` | `VERIFIED` |

## 9. Test Strategy

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-001` | Unit | Checkpoint markers computed from a fake `navigation.checkpoints` list | Mocked vehicle/navigation with 3 known checkpoints | Geometry contains exactly 3 marker points at expected world→frame projected coordinates | `REQ-001` |
| `TEST-002` | Unit | Checkpoint markers omitted, no crash, when navigation raises/missing | Mocked navigation raising `AttributeError` | Geometry has empty/`None` checkpoint field, frame still returned | `REQ-001`, `REQ-004` |
| `TEST-003` | Unit | Ego trail accumulates over N steps then resets on new episode | Sequence of `record_step` calls across an episode boundary | Trail length grows monotonically within an episode, resets to empty at next episode start | `REQ-002` |
| `TEST-004` | Unit | Ego trail respects `draw_ego_trail: false` | Config with flag off | No trail geometry computed/drawn | `REQ-002` |
| `TEST-005` | Regression | No non-visual artifact changes | Existing `evals.csv`/`eval_episodes.csv` schema tests | Byte-identical schema/values to pre-change baseline | `REQ-003` |
| `TEST-006` | Boundary | Both layers degrade silently on exception | Forced exception injected into geometry computation | Frame still returned, no raised exception, no evaluation abort | `REQ-004` |

Commands: `docker compose run --rm -T dev uv run --no-sync python -m pytest -q tests/test_video_diagnostics.py` (focused); full suite via `make test` before final reconciliation; `make lint`/`make format-check` scoped to touched files.

## 10. Milestones

- [x] **M0 — Decision gate**: `DEC-001` approved 2026-07-31 ("come preferisci
  ... Procedi pure").
- [x] **M1 — Checkpoint markers (`REQ-001`)**: corrected the plan's
  assumption (real checkpoints come from `assigned_route_lane_ids`, not
  MetaDrive's unused `navigation.checkpoints`); implemented
  `lane_start_points_xyz` + geometry/draw; tests pass.
- [x] **M2 — ADR-020 amendment**: `decisions/ADR-043-evaluation-video-ego-trail-and-checkpoint-overlay.md`
  written and approved; ADR-020 status/body updated in place.
- [x] **M3 — Ego trail (`REQ-002`)**: implemented per-episode
  (`LiveEvalEpisodeRecorder`) and per-worker-slot
  (`deterministic_subproc_vec_env.py`) state, draw function, config toggle;
  tests pass.
- [x] **M4 — Regression and degradation tests**: focused suite (61 + 18 tests
  in the two touched test files) passes; broader filtered regression (165
  tests across route/geometry/diagnostic/subproc_vec_env/eval_artifacts)
  passes with the same 4 pre-existing, unrelated failures confirmed via
  `git stash` baseline comparison (`test_eval_artifacts.py`'s
  `scenario_set`-in-path assertions, unrelated to this change); scoped
  `ruff check` clean; scoped `ruff format --check` flags only the
  pre-existing `agent.py` baseline (confirmed via `git stash`), the two new
  test files were formatted.
- [x] **M5 — Reconciliation**: this ExecPlan and `docs/project_index.md`
  updated; final report delivered to the user.

## 11. Progress And Findings Log

- 2026-07-31: Plan drafted after Explore-agent research confirmed (a)
  MetaDrive's native `TopDownRenderer` already supports an ego trail via
  `draw_target_vehicle_trajectory` but is unused here in favor of the
  existing custom annotator, and (b) the existing custom annotator's
  `route_past`/`route_future`/`target_point` already draw route geometry from
  the planned route polyline, not from discrete navigation checkpoints or
  ego position history. Found that ADR-020 explicitly excludes "ego-history
  lines," making `REQ-002` a specification deviation requiring approval
  (`DEC-001`), while `REQ-001` (discrete checkpoints) has no such conflict.
- 2026-07-31: User approved `DEC-001` ("come preferisci, tanto nelle gif per
  ora non vedevo comunque nulla di ciò. Procedi pure poi ll'implementazione").
  Before implementing `REQ-001`, a second research pass found the plan's
  original assumption wrong: MetaDrive's live `navigation.checkpoints` is
  never read anywhere in `src/thesis_rl/` — the actual "planned checkpoint"
  concept in this repository is a lane transition in the frozen
  `assigned_route_lane_ids` sequence used to build `route_polyline`
  (`RoutePolyline.from_lane_centerlines`/`build_assigned_route_polyline` in
  `src/thesis_rl/rulebook/v2/geometry/route.py`). Implemented against this
  corrected model instead: additive `lane_start_points_xyz` field on
  `RoutePolyline` (default `()`, only populated by `from_lane_centerlines`,
  verified via `grep` that every other `RoutePolyline(...)` call site in the
  repository passes a single positional argument, so the new default is
  never silently overridden by positional-argument drift).
- 2026-07-31: Implemented `REQ-002` (ego trail). Confirmed both recording
  paths already provide natural per-episode scoping for state: the live path
  (`LiveEvalEpisodeRecorder`) is instantiated fresh per episode by both
  factories in `eval_artifacts.py`, so no explicit reset hook was needed
  there; the parallel-worker path (`deterministic_subproc_vec_env.py`)
  persists one env across many sequential episodes per worker process, so
  its trail list is cleared explicitly on `reset`/`reset_slots`.
- 2026-07-31: Wrote `decisions/ADR-043-evaluation-video-ego-trail-and-checkpoint-overlay.md`
  amending ADR-020 in place (status updated to reference the amendment, body
  updated to list the two new overlay elements and narrow the
  ego-history-lines exclusion).
- 2026-07-31: Verification: `tests/test_video_diagnostics.py` (61 tests,
  including 8 new) and `tests/test_eval_artifacts_ego_trail.py` (3 new tests)
  all pass. A broader filtered regression run
  (`pytest tests/ -k "route or geometry or diagnostic or subproc_vec_env or
  eval_artifacts"`, 169 tests) surfaced 4 pre-existing failures in
  `tests/test_eval_artifacts.py` (a `scenario_set` path-segment mismatch,
  e.g. `videos/final_eval/eval_0009/...` vs. `videos/final_eval/test/eval_0009/...`)
  — confirmed via `git stash` to reproduce identically on the unmodified
  baseline, unrelated to this change. Scoped `ruff check` on all touched
  files: clean. Scoped `ruff format --check`: only `src/thesis_rl/agent/agent.py`
  flagged, confirmed pre-existing via `git stash`; the two new test files
  were run through `ruff format` directly (new files, no baseline to
  preserve). `make config` passed.
- 2026-08-01: User reported seeing no route/checkpoint/trail overlay at all
  in real `make smoke` GIFs, despite this plan's `VERIFIED` status. Live
  investigation (real vectorized smoke runs, direct GIF pixel inspection,
  and a temporary debug pass logging every worker-side `diagnostic_geometry`
  call across 13,341 render calls) found and fixed one confirmed regression:
  `align_episode_cache_to_live_elevation`
  (`src/thesis_rl/rulebook/v2/transition.py`) rebuilt the route polyline via
  `RoutePolyline`'s plain constructor whenever a nonzero elevation-datum
  offset applied (the docstring's own stated common case for Waymo
  scenarios), silently resetting the new `lane_start_points_xyz` field
  (`REQ-001`, this plan) to its empty default and dropping checkpoint
  markers for every such scenario. Fixed by shifting and preserving
  `lane_start_points_xyz` through the same offset; regression test added
  (`tests/test_rulebook_v2_transition.py::test_cache_elevation_alignment_preserves_lane_start_points_for_checkpoints`).
  The debug-log evidence also confirmed `route_past`/`route_future`/
  `ego_trail` render correctly in the overwhelming majority of sampled calls
  (zero fully-empty geometry results across 13,341 samples) and one real
  produced GIF frame was visually confirmed to show a route/trail line.
  **Known open item, not yet root-caused**: one specific final-panel episode
  (`test_waymo_empirical`/`episode_0007` in a smoke run) showed zero overlay
  geometry (route, checkpoints, and ego trail all absent) across every
  sampled frame, despite the debug-log evidence showing this class of
  complete failure never occurring in a separate, comparable debug run. Not
  reproduced deterministically enough within this session's time budget to
  isolate; does not appear caused by the checkpoint-only bug just fixed
  (that bug only explains missing checkpoints, not missing route/trail too).
  Recorded here as a known limitation pending a dedicated follow-up
  investigation.

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-001` | ADR-020: "ego-history lines" excluded from overlay | Added a distinctly-styled ego-trail layer, `REQ-002` | Explicit user request this session for qualitative episode inspection | Approved 2026-07-31, `decisions/ADR-043-evaluation-video-ego-trail-and-checkpoint-overlay.md` | ADR-020 (amended), this ExecPlan |

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/runtime/io/video_diagnostics.py` | Modified | Checkpoint-marker and ego-trail geometry/draw logic |
| `src/thesis_rl/runtime/io/eval_artifacts.py` | Modified | Per-episode ego-position accumulation (`_ego_trail_world`), config toggle |
| `src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py` | Modified | Per-worker-slot ego-trail state, cleared on episode boundaries |
| `src/thesis_rl/agent/agent.py` | Modified | Thread `draw_ego_trail` config toggle into parallel-eval render kwargs |
| `src/thesis_rl/rulebook/v2/geometry/route.py` | Modified | Additive `lane_start_points_xyz` field on `RoutePolyline` |
| `conf/video/default.yaml` | Modified | New `topdown.draw_ego_trail` toggle |
| `tests/test_video_diagnostics.py` | Modified | Checkpoint/ego-trail geometry and `RoutePolyline` field tests |
| `tests/test_eval_artifacts_ego_trail.py` | New | Per-episode trail accumulation/reset/toggle lifecycle tests |
| `docs/decisions/ADR-020-evaluation-video-diagnostics.md` | Modified | Amended in place to list the two new overlay elements |
| `docs/decisions/ADR-043-evaluation-video-ego-trail-and-checkpoint-overlay.md` | New | Records the `DEC-001` approval and ADR-020 amendment |
| `docs/project_index.md` | Modified | ExecPlan Registry row, ADR-020/ADR-043 note |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `docker compose run --rm -T dev uv run --no-sync python -m pytest -q tests/test_video_diagnostics.py tests/test_eval_artifacts_ego_trail.py` | `PASS` | 2026-07-31 | 18 passed (later reconfirmed at 61+3 after further additions) |
| `docker compose run --rm -T dev uv run --no-sync python -m pytest -q tests/ -k "route or geometry or diagnostic or subproc_vec_env or eval_artifacts"` | `PASS` (with 4 known pre-existing failures) | 2026-07-31 | 165 passed / 4 failed; the 4 failures reproduce identically on the pre-change baseline via `git stash` (unrelated `scenario_set` path-segment assertions in `tests/test_eval_artifacts.py`) |
| `docker compose run --rm -T dev uv run --no-sync ruff check <touched files>` | `PASS` | 2026-07-31 | All checks passed |
| `docker compose run --rm -T dev uv run --no-sync ruff format --check <touched files>` | `PASS` (pre-existing baseline noted) | 2026-07-31 | Only `agent.py` flagged, confirmed pre-existing via `git stash`; new test files formatted directly |
| `make config` | `PASS` | 2026-07-31 | `docker compose config --quiet` clean |
| `docker compose run --rm -T dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_transition.py` | `PASS` | 2026-08-01 | 15 passed (regression test for the elevation-alignment checkpoint bug fix, see §11) |
| `make smoke` with `env.vectorized.enabled=true env.vectorized.num_envs=2` (real GIF pixel inspection) | `PASS` with one open item | 2026-08-01 | Confirmed route/checkpoints/ego-trail render correctly in the overwhelming majority of a real run's frames; one specific episode showed no overlay at all, not yet root-caused (see §11) |

## 15. Final Reconciliation

- `REQ-001` (discrete checkpoint markers): `VERIFIED`.
- `REQ-002` (ego-actually-traveled trail): `VERIFIED`; required and received
  `DEC-001` approval and an ADR-020 amendment (`ADR-043`).
- `REQ-003` (no change to policy/reward/termination/metrics): `VERIFIED` by
  construction (raster-only additions, no touched file is on the
  observation/reward/termination path) and by the unaffected regression
  suite.
- `REQ-004` (silent degradation): `VERIFIED`, reusing the existing
  `except Exception: pass` pattern already established for
  `route_past`/`route_future`.

Known limitations:

- Checkpoint markers reflect the frozen `assigned_route_lane_ids` lane
  sequence, not a live re-planned route — consistent with how
  `route_past`/`route_future` already behave, since this repository's route
  is task metadata, not a runtime planner output.
- The ego trail is unbounded in length for the duration of one episode (no
  cap/decay), unlike MetaDrive's native fading-alpha trajectory renderer;
  acceptable given the model's per-episode bound on frame count, but a future
  cap could be added if very long episodes make the polyline visually
  cluttered.
- **Open, not yet root-caused (found 2026-08-01)**: one specific final-panel
  episode in a real smoke run showed zero overlay geometry (route,
  checkpoints, and ego trail all absent) across every sampled frame, while a
  separate high-volume debug run (13,341 worker-side geometry-extraction
  calls) showed zero instances of this complete-failure pattern. A confirmed
  and fixed checkpoint-only regression (elevation-alignment losing
  `lane_start_points_xyz`) does not explain this case, since it would only
  affect checkpoints, not route/trail. Suspected to be an intermittent,
  per-episode/per-worker condition (e.g. adapter installation timing across
  worker reuse) rather than a config issue, but not reproduced deterministically
  enough to isolate within this session. Needs a dedicated follow-up
  investigation before being considered resolved.

No deferred required work. Optional follow-up not requested: a dedicated
`topdown.draw_checkpoints` toggle (checkpoints currently always render when
route geometry is available, matching `route_past`/`route_future`'s
existing behavior).

Resulting behavior: evaluation GIFs (live final-eval and periodic
tracked-subset) now show, when route geometry is available, both the
planned lane-transition checkpoints and the ego's actually-traveled path for
the current episode, alongside the pre-existing route/target/neighbor
overlay. No architecture, compatibility, or non-visual artifact change.
Approved decision: `DEC-001`/`ADR-043`. No unresolved deviations. Ready for
experimental use.
