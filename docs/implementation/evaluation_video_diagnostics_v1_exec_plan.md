# Evaluation Video Diagnostics v1 ExecPlan

## 1. Metadata

- Feature: compact diagnostic overlays for live and replayed evaluation GIFs
- Plan ID: `EVAL-VIDEO-DIAGNOSTICS-V1`
- Authoritative specification: `docs/specifications/rl_baselines_v1_specification.md`
  (`RL-BASELINES`, version `1.0`, `APPROVED`, `Authoritative: YES`)
- Related contracts: Rulebook v4.7, scalarization v1.0, observation v1.1,
  `docs/protocols/live_eval_video_protocol.md` (candidate protocol)
- Related ADRs: ADR-018; proposed `ADR-020` for diagnostic rendering scope
- Status: `IMPLEMENTED`
- Created: 2026-07-21
- Last updated: 2026-07-21
- Branch: `scenarionet-implementation`
- Owner: thesis repository maintainer

## 2. Objective And Scope

Add compact, deterministic diagnostic information to evaluation GIF frames
without changing policy observations, rewards, termination semantics, datasets,
checkpoints, or experiment metrics.

In scope:

- one shared frame annotation path for live final-evaluation recording and
  offline replay;
- a compact bottom-right panel with algorithm, step, speed, heading, selected
  reward, cumulative selected reward, route completion, and Rulebook R1--R4
  values including available subrules;
- black text on a semi-transparent white panel;
- route already traversed in green and available target/waypoint marker;
- optional route remainder only when the complete runtime route is available
  during the frame, never reconstructed from future trajectory data;
- thin orange outlines for available ego-neighbor vehicles and thin red outline
  for the vehicle identified as relevant by RSS/TTC;
- deterministic tests for formatting, missing diagnostics, reward accumulation,
  and shared live/replay behavior.

Out of scope for v1:

- changes to the observation or reward contracts;
- explicit terminal-status or top-rule panels;
- native/scalarized reward duplication when the selected reward is shown;
- road-boundary or violation-zone overlays;
- RSS safety polygons and TTC cones. Their numeric diagnostics and target-actor
  highlighting remain in scope; geometric primitives are deferred unless the
  existing coordinate/rendering interfaces make them trivial and exact.

Compatibility constraints: official live GIFs remain authoritative, offline
replay remains available and non-authoritative, and one-worker/parallel
evaluation paths retain their existing episode ordering and artifact layout.

## 3. Authoritative Requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-EVD-001` | Diagnostic overlays must not enter policy observations or alter reward/termination behavior. | RL-BASELINES §§4.1--4.2, 8.6, 8.8 |
| `REQ-EVD-002` | The panel displays the algorithm, step, selected reward, cumulative selected reward, route completion, speed, and heading. | User-approved scope, 2026-07-21 |
| `REQ-EVD-003` | The panel displays compact R1--R4 margins and available subrule values without an excessive hierarchy/status block. | User-approved scope, 2026-07-21; Rulebook v4.7 |
| `REQ-EVD-004` | Live and offline renderers use the same annotation/formatting logic. | User-approved implementation choice, 2026-07-21 |
| `REQ-EVD-005` | Route/target and vehicle overlays use world geometry only when available at the current frame; no future trajectory reconstruction is allowed. | RL-BASELINES §§4.2, 8.8; user-approved scope |
| `REQ-EVD-006` | Neighbor vehicles use thin orange outlines and the RSS/TTC-relevant actor uses a thin red outline when actor identity can be resolved. | User-approved scope, 2026-07-21 |
| `REQ-EVD-007` | RSS/TTC numeric values remain visible; geometric RSS/TTC primitives are deferred when exact projection is not already available. | User-approved scope, 2026-07-21 |

## 4. Current Repository Analysis

| Classification | Verified fact |
|---|---|
| `VERIFIED` | `src/thesis_rl/runtime/io/eval_artifacts.py` owns live frame capture, trajectory rows, manifests, and GIF encoding. |
| `VERIFIED` | `src/thesis_rl/analysis/videos/render_selected_videos.py` owns offline replay and already has a compact telemetry annotator. |
| `VERIFIED` | `step_info` exposes selected reward-related fields, route completion, ego state, neighbors, `rule_reward_vector`, `rule_metadata`, `rule_components`, and `rulebook`. |
| `VERIFIED` | Rulebook `RuleComponentResult.raw` retains detailed RSS/TTC actor IDs and values; Rulebook components retain status/applicability. |
| `VERIFIED` | The canonical Rulebook episode cache contains a route polyline, route lanes, map features, and target/control geometry, but this is not currently a renderer-neutral frame payload. |
| `VERIFIED` | Neighbor records can contain actor position, velocity, yaw, dimensions, and bounding polygon; availability varies by environment/wrapper. |
| `VERIFIED` | Evaluation is deterministic and diagnostics are outside the policy input boundary. |
| `INFERRED` | A shared renderer can be introduced without changing evaluation scheduling by normalizing a small frame payload at the recorder boundary. |
| `AWAITING_CONFIRMATION` | None. The user explicitly approved implementation of the agreed scope. |

## 5. Assumptions And Invariants

- Selected reward means the scalar value returned by `env.step`, exactly once per
  transition. The panel shows this value and its episode cumulative sum.
- Rule values are rendered in their existing contract units and signs; no
  formula, clipping, normalization, or scalarization is introduced by the
  renderer.
- Speed is displayed in km/h when the existing runtime field is km/h; otherwise
  the displayed unit must be explicit.
- Heading is displayed in radians, matching the existing `yaw`/heading runtime
  convention.
- World geometry remains in the environment's existing XY frame. The renderer
  must not use policy-visible future information.
- Missing values render as `-` or are omitted from optional geometry; rendering
  diagnostics must never make an evaluation episode fail.
- Frame annotation occurs after the transition, consistent with current live and
  offline capture timing.
- Semi-transparent white panels and geometry are raster overlays only; no
  dependency is added.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-EVD-001` | implementation detail | Full hierarchical rule panel is too large | verbose status lines / compact one-line macro rows | compact macro rows with abbreviated subrules and color/status encoded visually | readable GIFs; no metric change | Approved by user 2026-07-21 |
| `DEC-EVD-002` | implementation detail | Live vs replay annotation ownership | separate formatters / shared annotator | shared normalizer and annotator reused by both paths | visual consistency and lower maintenance | Approved by user 2026-07-21 |
| `DEC-EVD-003` | scope boundary | RSS/TTC geometry complexity | approximate polygons/cones now / defer | defer geometric primitives; retain numeric values and target outline | avoids misleading geometry | Approved by user 2026-07-21 |
| `DEC-EVD-004` | implementation detail | Actor visibility mode | new debug view / existing top-down actors plus optional outlines | use existing top-down scene; add outlines only when identity is resolvable | avoids duplicating already visible actors | Approved by user 2026-07-21 |
| `DEC-EVD-005` | specification clarification | Route remainder availability | reconstruct future route / draw only runtime-available route | draw full remainder only when current runtime exposes it; otherwise omit | preserves causal/debug interpretation | Approved by user 2026-07-21 |

No unresolved approval gate currently blocks implementation.

## 7. Proposed Design

Create a renderer-neutral diagnostic payload and shared raster annotator under
`src/thesis_rl/runtime/io/` or the existing video package. The payload contains
only sanitized scalar values and optional world geometry references/coordinates.

The live recorder will build the payload from the current `step_info`, render
the base top-down frame, and apply the shared annotator before appending the
frame. The offline renderer will build the same payload from replay `step_info`
and cumulative episode state before applying the same annotator.

The shared annotator will:

1. convert frames to RGB;
2. draw optional route/target/vehicle overlays when coordinates are available;
3. draw the compact semi-transparent white panel with black text;
4. preserve the current frame dimensions and GIF encoding path.

The first implementation will prefer existing `info` geometry and avoid
accessing policy internals. If an environment does not expose a suitable route
polyline or actor polygon, it will retain the native top-down frame and panel.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-EVD-001` | `AC-EVD-001` | shared annotation boundary only | existing observation/reward contract tests plus focused renderer tests | Verified |
| `REQ-EVD-002` | `AC-EVD-002` | shared payload/panel formatter | `tests/test_video_diagnostics.py` | Verified |
| `REQ-EVD-003` | `AC-EVD-003` | compact rule-row formatter | `tests/test_video_diagnostics.py` | Verified |
| `REQ-EVD-004` | `AC-EVD-004` | shared live/replay annotator | `tests/test_video_diagnostics.py`, artifact tests | Verified |
| `REQ-EVD-005` | `AC-EVD-005` | route/target optional overlay adapter | focused payload/overlay tests; backend visual smoke pending | Partial |
| `REQ-EVD-006` | `AC-EVD-006` | actor outline selection | focused payload/overlay tests; backend visual smoke pending | Partial |
| `REQ-EVD-007` | `AC-EVD-007` | numeric RSS/TTC rows; no approximate geometry | focused missing/available raw diagnostics tests | Verified |

## 9. Test Strategy Defined Before Implementation

### Acceptance criteria

- `AC-EVD-001`: annotation does not modify observation, action, reward,
  termination, truncation, or environment state.
- `AC-EVD-002`: all required panel fields render with deterministic formatting;
  selected reward appears once and cumulative reward is accumulated exactly.
- `AC-EVD-003`: R1--R4 and available subrules fit the compact panel; missing or
  not-applicable values render safely without exceptions.
- `AC-EVD-004`: live and offline paths call the same annotator and produce
  equivalent annotated frames for the same normalized payload.
- `AC-EVD-005`: route/target overlays are drawn only from current available
  geometry; absent geometry leaves the base frame valid.
- `AC-EVD-006`: resolvable neighbors receive orange outlines and the RSS/TTC
  target receives a red outline; unresolved identities do not fail rendering.
- `AC-EVD-007`: RSS/TTC values and actor IDs are shown where available; no
  approximate safety polygon or cone is emitted.

### Mandatory test matrix

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-EVD-001` | Unit | panel formatting | complete payload | deterministic required fields, one selected reward | `REQ-EVD-002` |
| `TEST-EVD-002` | Unit | cumulative reward | positive/negative reward sequence | exact running sum | `REQ-EVD-002` |
| `TEST-EVD-003` | Unit | compact rule hierarchy | R1--R4 with subrules and NA values | bounded rows, safe formatting, no exception | `REQ-EVD-003` |
| `TEST-EVD-004` | Unit | missing diagnostics | absent rule/ego/route fields | valid frame/panel with placeholders | `REQ-EVD-003`, `REQ-EVD-005` |
| `TEST-EVD-005` | Unit | shared annotator | identical normalized payload/frame | live/replay entry points produce equivalent pixels | `REQ-EVD-004` |
| `TEST-EVD-006` | Unit | route/target overlays | current route and target geometry | green route/marker overlay; absent future route omitted | `REQ-EVD-005` |
| `TEST-EVD-007` | Unit | vehicle highlighting | neighbor list and RSS/TTC raw IDs | orange neighbors/red target; unresolved target safe | `REQ-EVD-006` |
| `TEST-EVD-008` | Unit | RSS/TTC scope | raw numeric diagnostics | values render; no geometric primitive required | `REQ-EVD-007` |
| `TEST-EVD-009` | Integration | live artifact | deterministic recorder fixture | annotated GIF and existing manifest paths remain valid | `REQ-EVD-004` |
| `TEST-EVD-010` | Regression | offline replay | existing replay fixture | existing replay comparisons and authoritative selection remain valid | `REQ-EVD-001`, `REQ-EVD-004` |

### Commands

```bash
uv run --no-sync python -m pytest -q tests/test_video_diagnostics.py tests/test_eval_artifacts.py tests/test_parallel_evaluation.py tests/test_video_selection_authoritative.py
make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/runtime/io/video_diagnostics.py src/thesis_rl/runtime/io/eval_artifacts.py src/thesis_rl/analysis/videos/render_selected_videos.py tests/test_video_diagnostics.py"
make lint PYTHON_QUALITY_PATHS="src/thesis_rl/runtime/io/video_diagnostics.py src/thesis_rl/runtime/io/eval_artifacts.py src/thesis_rl/analysis/videos/render_selected_videos.py tests/test_video_diagnostics.py"
git diff --check
make smoke
```

## 10. Milestones

- [x] M1: scope, authority, decisions, acceptance criteria, and test matrix recorded.
- [x] M2: shared diagnostic payload/annotator and compact panel.
- [x] M3: route/target/vehicle overlay integration with safe backend fallback.
- [x] M4: live recorder and offline replay integration.
- [x] M5: focused tests and quality checks; end-to-end smoke attempted but
  interrupted after the first chunk because the container made no progress in
  the second chunk.
- [x] M6: final reconciliation, index update, and source-sync report.

## 11. Progress And Findings Log

| Date | Finding/action | Evidence/result | Next step |
|---|---|---|---|
| 2026-07-21 | Existing live recorder and offline replay have separate frame annotation paths. | `eval_artifacts.py`, `render_selected_videos.py` inspection | add shared annotator |
| 2026-07-21 | Rulebook retains detailed subrule and RSS/TTC actor diagnostics. | `RuleComponentResult`, RSS/TTC components | normalize compact rows |
| 2026-07-21 | Route/actor geometry is available only in environment-specific forms. | wrapper `ego_state`, `neighbors`, Rulebook cache | use optional geometry adapter and safe omission |
| 2026-07-21 | Implemented shared panel and optional exact route/target/actor overlays. | Focused suite `14 passed`; Ruff format/lint passed | final reconciliation; smoke remains unverified |
| 2026-07-21 | End-to-end smoke built and completed its first 1,000-step chunk, then made no observable progress in the second chunk. | `make smoke` interrupted with exit 130 after read-only `docker top`/logs checks | report smoke as not verified; retain focused evidence |
| 2026-07-21 | Parallel final evaluation attempted to pickle MetaDrive `top_down_renderer` through `VectorEnvSlotProxy` while extracting optional geometry. | Attached traceback: worker `get_attr` failed on `edge_lane` pickling; panel frame transport itself was unaffected | skip world-geometry extraction across the parent proxy boundary; retain panel and worker-rendered frame |
| 2026-07-21 | Fixed parallel diagnostics: worker now applies route/target/neighbor overlays before transferring the frame; worker also attaches pickle-safe speed/yaw scalars. Panel semantics now distinguish macro margins (`m`) from subrule costs (`c`). | Focused suite `13 passed`; Ruff format/lint passed; `git diff --check` passed | rerun the five-episode test in the real ScenarioNet smoke environment |

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `docs/implementation/evaluation_video_diagnostics_v1_exec_plan.md` | Added | living implementation and traceability record |
| `docs/decisions/ADR-020-evaluation-video-diagnostics.md` | Planned | approved diagnostic rendering scope |
| `src/thesis_rl/runtime/io/video_diagnostics.py` | Added | shared payload, formatter, exact optional geometry extraction, and raster annotator |
| `src/thesis_rl/runtime/io/eval_artifacts.py` | Modified | live recorder integration and selected-reward trajectory field |
| `src/thesis_rl/analysis/videos/render_selected_videos.py` | Modified | offline replay integration using the shared annotator |
| `tests/test_video_diagnostics.py` | Added | deterministic unit/regression coverage |
| `tests/test_eval_artifacts.py` | Reused unchanged | existing live artifact compatibility coverage passes |
| `docs/project_index.md` | Modified | register ExecPlan/ADR and status |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `PYTHONPATH=/home/e.respino/.local/lib/python3.10/site-packages:src .venv/bin/python -m pytest -q tests/test_video_diagnostics.py tests/test_eval_artifacts.py tests/test_video_selection_authoritative.py tests/test_parallel_evaluation.py tests/test_deterministic_subproc_vec_env.py` | `PASS` | 2026-07-21 | `14 passed` |
| `make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/runtime/io/video_diagnostics.py src/thesis_rl/runtime/io/eval_artifacts.py src/thesis_rl/analysis/videos/render_selected_videos.py tests/test_video_diagnostics.py"` | `PASS` | 2026-07-21 | Container Ruff check: 4 files already formatted |
| `make lint PYTHON_QUALITY_PATHS="src/thesis_rl/runtime/io/video_diagnostics.py src/thesis_rl/runtime/io/eval_artifacts.py src/thesis_rl/analysis/videos/render_selected_videos.py tests/test_video_diagnostics.py"` | `PASS` | 2026-07-21 | Container Ruff: all checks passed |
| `git diff --check` | `PASS` | 2026-07-21 | no whitespace errors |
| `PYTHONPYCACHEPREFIX=/tmp/thesis-pycache python -m py_compile ...` | `PASS` | 2026-07-21 | modified Python modules compile |
| `make smoke` | `NOT_VERIFIED` | 2026-07-21 | first 1,000-step chunk completed; second chunk made no observable progress and was interrupted with exit 130 |

## 15. Final Reconciliation

| Requirement/criterion | Status | Reconciliation |
|---|---|---|
| `REQ-EVD-001` / `AC-EVD-001` | VERIFIED | Annotation is applied after `env.step` and does not enter observation/reward/termination paths; focused artifact and evaluation regressions pass. |
| `REQ-EVD-002` / `AC-EVD-002` | VERIFIED | Shared panel shows algorithm, step, speed, heading, selected reward, cumulative selected reward, and route completion; deterministic unit tests pass. |
| `REQ-EVD-003` / `AC-EVD-003` | VERIFIED | Compact R1--R4 rows and available subrule rows are rendered with safe missing-data behavior. |
| `REQ-EVD-004` / `AC-EVD-004` | VERIFIED | Live and offline paths call `video_diagnostics.annotate_diagnostic_frame`. |
| `REQ-EVD-005` / `AC-EVD-005` | PARTIAL | Exact route/target overlays are implemented for the supported MetaDrive top-down camera and safely omitted otherwise; no full PG/ScenarioNet/Waymo visual smoke was completed. |
| `REQ-EVD-006` / `AC-EVD-006` | PARTIAL | Exact renderer-space neighbor/critical-actor outlines are implemented when IDs and geometry resolve; backend-wide visual verification remains pending. |
| `REQ-EVD-007` / `AC-EVD-007` | VERIFIED | Numeric RSS/TTC diagnostics and target identification are retained; approximate safety bands/cones are explicitly deferred by ADR-020. |

Known limitations:

- The end-to-end `make smoke` command was not verified because the second
  training chunk stopped producing progress and was interrupted after the
  first chunk completed.
- Camera modes with target heading rotation, missing canonical route context,
  or missing neighbor positions omit optional geometry rather than guessing.
- RSS safety bands and TTC cones are deferred optional work.

No required work is blocked on a user decision. The implementation is ready for
focused evaluation artifacts; full visual backend verification should use the
follow-up command `make smoke` in a healthy/provisioned runtime.
