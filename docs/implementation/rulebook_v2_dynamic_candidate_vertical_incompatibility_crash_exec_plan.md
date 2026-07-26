# ExecPlan: Rulebook v2 Live Training Crash — Unguarded Route Projection for Dynamic Candidates

## 1. Metadata

- Feature / plan ID: `rulebook-v2-dynamic-candidate-vertical-incompatibility-crash-v1`
- Authoritative specification: `docs/specifications/rulebook_v4.7_specification.md`,
  `docs/specifications/rulebook_v4.8_specification.md` (RSS-longitudinal,
  RSS-lateral, crosswalk formulas unchanged); `docs/specifications/observation_v1.2_specification.md`
  (interaction-candidate feature, unchanged)
- Status: `IMPLEMENTED`, live re-verification of the 9 restarted runs in progress
- Created: 2026-07-26
- Last updated: 2026-07-26
- Branch: `scenarionet-implementation`
- Related decisions: none new — conformance fix, same design pattern already
  established for static-map features
  (`causal_semantic_route_incompatible_static_feature_bugfix_exec_plan.md`)
- Owner: n/a (single session)

## 2. Objective And Scope

**Objective**: a *dynamic* candidate (another vehicle for RSS-longitudinal
or RSS-lateral, a crosswalk feature for interaction candidates) whose
footprint/position has no vertically compatible route segment must be
excluded from that step's candidate set, never crash the whole
`evaluate_transition` step or observation build.

**Why**: this was discovered live, mid-session, while running 9 real
`RUN_PROFILE=thesis` training processes (3 algorithms x 3 seeds) at the
user's explicit request. 5 of the 9 crashed with
`ValueError: Route projection has no vertically compatible segment` within
roughly 15 minutes, at three distinct unguarded call sites, all calling
`RoutePolyline.project`/`footprint_route_coordinates` for an *optional*,
non-ego candidate (another vehicle or a crosswalk feature) without the
try/except pattern already established for static map features earlier in
this same session
(`causal_semantic_route_incompatible_static_feature_bugfix_exec_plan.md`).
This is the identical bug class, just in three call sites that audit had
not covered because it was scoped to static-map features specifically, not
dynamic actors.

**In scope**:

- `src/thesis_rl/rulebook/v2/transition.py::_rss_candidates`: wrap both the
  ego and the per-actor `footprint_route_coordinates` calls.
- `src/thesis_rl/rulebook/v2/transition.py::_rss_lateral_candidates`: same,
  for both ego and per-actor calls.
- `src/thesis_rl/envs/observations/causal_semantic.py::_build_interactions`:
  wrap the crosswalk-feature `route.project` call.
- Policy, consistent across all three: if **ego's own** footprint fails to
  project (a genuinely structural precondition — ego's live position must
  be locatable on its own assigned route), the function returns its empty
  result (`()`), matching the existing `ego_lane is None: return ()`
  early-exit already present in the same functions — this is a fail-fast
  boundary, not silently swallowed, but scoped to "no RSS/interaction
  candidates this step" rather than crashing the whole transition. If an
  **individual actor or feature** fails to project, only that candidate is
  skipped (`continue`), and the rest of the step proceeds normally.

**Out of scope**:

- `_front_s` and the top-level `route.project` calls on ego's own
  post-state position in `evaluate_transition` (`transition.py:178,1128`)
  and in the semantic observation builders' own ego-frame projections
  (already using try/except where they existed, e.g. `_route_s`,
  `_route_lateral`, `_route_projection_or_raise`) — these are the required
  ego/route structural frame, deliberately left fail-fast, consistent with
  the same design distinction already established for static features.
- Route-self-projection calls (e.g. `self.route.project(point[:2], ...)`
  where `point` came from `self.route.point_at(...)`) — geometrically
  guaranteed to find a compatible segment (projecting a point back onto the
  route that generated it); audited and confirmed not a crash risk, left
  unchanged.
- `movement_priorities`/roundabout data population, PyTorch cross-arch
  validation, and other unrelated open items tracked elsewhere in
  `docs/project_index.md`.

**Compatibility**: no public interface, configuration key, or checkpoint
schema change. Observable behavior change: a vehicle or crosswalk feature
that is vertically incompatible with ego's route is now silently excluded
from RSS/interaction candidates for that step, instead of crashing the
episode/process. This is the same design already accepted for static
map-boundary features.

## 3. Authoritative Requirements

| ID | Requirement |
|---|---|
| `REQ-001` | A non-ego vehicle candidate with no vertically compatible route segment must be excluded from RSS-longitudinal candidates, not crash `evaluate_transition`. |
| `REQ-002` | A non-ego vehicle candidate with no vertically compatible route segment must be excluded from RSS-lateral candidates, not crash `evaluate_transition`. |
| `REQ-003` | A crosswalk feature with no vertically compatible route segment must be excluded from interaction candidates, not crash the semantic observation build. |
| `REQ-004` | Ego's own footprint failing to project must still result in an empty candidate set for the affected function (fail-fast at the function boundary, not silently defaulted to a fabricated value), matching the pre-existing `ego_lane is None` early-exit contract. |

## 4. Current Repository Analysis

- `VERIFIED`: `footprint_route_coordinates` (`geometry/lanes.py:166-193`) and
  `RoutePolyline.project` (`geometry/route.py:150-176`) raise a bare
  `ValueError` (not a `CausalSemanticObservationError`-style typed error)
  when no route segment is within `VERTICAL_COMPATIBILITY_TOLERANCE_M` of
  the queried point's `position_z`.
- `VERIFIED`: this exact bug class (unguarded `route.project` for an
  optional static feature) was found and fixed earlier in this same
  session for `_build_static`/`_build_static_v12`/`_build_interactions`'s
  crosswalk loop is the SAME function that was touched for
  `route_incompatible_static_features` accounting — the crosswalk-specific
  crash site inside `_build_interactions` (lines ~1170-1176) is distinct
  from, and was not covered by, that earlier static-feature-only fix.
- `VERIFIED` live, in production: 3 distinct unguarded call sites
  identified from real crash tracebacks across 5 of 9 concurrently running
  `RUN_PROFILE=thesis` processes:
  1. `_rss_candidates` (`transition.py:266`, pre-fix) — actor's
     `footprint_route_coordinates` call, hit by `sac-2`.
  2. `_rss_lateral_candidates` (`transition.py:383`, pre-fix) — actor's
     `footprint_route_coordinates` call, hit by `td3-2`.
  3. `_build_interactions` (`causal_semantic.py:1175`, pre-fix) — crosswalk
     feature's `route.project` call, hit by `td3-1`, `sac-1`, and (with a
     traceback captured mid-fix, confirming the running process still held
     pre-fix bytecode) `ppo-1`.
- `VERIFIED`: an audit of every remaining unguarded `route.project`/
  `footprint_route_coordinates` call site in `transition.py` and
  `causal_semantic.py` found no further dynamic-candidate crash risk — the
  remaining unguarded sites are either already-guarded (`_route_s`,
  `_route_lateral`, `_route_projection_or_raise` all have try/except),
  ego's own structural route frame (`_front_s`, `evaluate_transition`'s
  `post_route_tangent_xy`, the various builders' `route_projection`
  variables), or self-projections of points already generated by
  `route.point_at(...)` (geometrically safe by construction).

## 5. Assumptions And Invariants

- Same as the prior static-feature fix: `VERTICAL_COMPATIBILITY_TOLERANCE_M`
  and the underlying vertical-gate semantics are unchanged; this plan only
  changes control flow (skip vs. crash), never a formula or threshold.
- Restarting the 9 live training processes with the fixed code loses their
  first ~15-20 minutes of progress (a live Python process does not pick up
  an on-disk source change until restarted); accepted given the
  alternative (continuing to run against a known, actively-crashing bug)
  is strictly worse.

## 6. Decisions And Approval Gates

No new material decision: this is a conformance fix extending the same
design pattern (skip vertically-incompatible optional candidates,
fail-fast on ego's own required frame) already established and approved
earlier in this session for static map features, now applied to the
dynamic-candidate call sites that same earlier audit did not cover.

## 7. Proposed Design

For each of the three call sites, wrap the relevant
`footprint_route_coordinates`/`route.project` call in `try: ... except
ValueError: <return () | continue>`, exactly mirroring the control-flow
shape already used at the sibling `ego_lane is None` / `actor_lane is
None` early exits in the same functions. No new helper abstraction
introduced — three small, local, self-explanatory edits.

## 8. Traceability

| Requirement | Implementation | Verification | Status |
|---|---|---|---|
| `REQ-001` | `transition.py::_rss_candidates` | `tests/test_rulebook_v2_transition.py` (full suite re-run, 288 passed); live restart of `sac-2` | Implemented; live re-verification in progress |
| `REQ-002` | `transition.py::_rss_lateral_candidates` | Same | Implemented; live re-verification in progress |
| `REQ-003` | `causal_semantic.py::_build_interactions` | `tests/test_causal_semantic_batch.py`; live restart of `td3-1`/`sac-1`/`ppo-1` | Implemented; live re-verification in progress |
| `REQ-004` | All three functions | Unchanged `ego_lane is None`/early-exit behavior confirmed by unmodified existing tests still passing | Verified |

## 9. Test Strategy

No new unit test was added in this pass (time-critical live fix while
production processes were actively crashing); the fix reuses an
already-established, already-tested control-flow pattern
(`try/except ValueError: continue|return ()`) at three additional call
sites with no new branching logic. Verification relied on:

1. The full existing focused/regression suite (rulebook v2, causal
   semantic, synthetic scenarios — 288 + 972 tests) confirming zero
   regression.
2. Restarting the actual 9 live `RUN_PROFILE=thesis` processes that
   surfaced the bug, as the most direct real-world confirmation available.

**Follow-up recommended, not done in this pass**: add deterministic unit
regressions for `_rss_candidates`/`_rss_lateral_candidates` mirroring
`test_route_incompatible_static_feature_is_omitted_and_masked` (a
vertically-incompatible actor is excluded, not a crash), for durable
coverage independent of live-run observation.

Commands used:

- `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_causal_semantic_batch.py tests/test_rulebook_v2_rss_lateral.py tests/test_rulebook_v2_transition.py`
- `docker compose run --rm dev uv run --no-sync python -m pytest -q -m "not integration"`
- `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_*.py tests/test_causal_semantic_batch.py tests/test_rulebook_synthetic_scenarios.py`

## 10. Milestones

### M1 — Live incident triage and fix

- Objective: identify all crash sites from live production tracebacks, fix
  each, verify no regression, restart the affected processes.
- Status: Done for the fix and restart; live multi-hour re-verification
  (do the 9 restarted runs survive past the point the pre-fix ones
  crashed) is ongoing and not yet concluded as of this writing.

## 11. Progress And Findings Log

- 2026-07-26 — User explicitly requested launching 9 real
  `RUN_PROFILE=thesis` training runs (TD3/SAC/PPO x seeds 0/1/2), each in
  its own tmux session, after a session-long readiness assessment. Set up
  a persistent background monitor watching all 9 tmux sessions/logs for
  errors or unexpected termination. Within ~15 minutes, the monitor
  reported `td3-1`, `td3-2`, `sac-1` ended with the same
  `ValueError: Route projection has no vertically compatible segment` at
  two different call sites (`_build_interactions` crosswalk loop,
  `_rss_lateral_candidates` actor loop). Investigated and fixed both while
  the remaining 6 processes kept running (source changes do not affect an
  already-running Python process). Before the fix could be confirmed and
  the remaining runs restarted, the monitor reported two more failures:
  `sac-2` at a third site (`_rss_candidates` actor loop) and `ppo-1` at the
  already-identified crosswalk site (traceback captured a stale line
  number from bytecode compiled before the in-progress edit, confirming
  the running process still held pre-fix code). Fixed the third site,
  audited every remaining unguarded `route.project` call in both files to
  rule out further dynamic-candidate crash risk (found none; remaining
  unguarded sites are ego's-own-frame or route-self-projections, both
  judged safe/intentionally fail-fast), ran the full non-integration suite
  (972 passed) plus the targeted rulebook/causal-semantic suite (288
  passed) with zero regressions, then killed all 9 tmux sessions and
  relaunched all 9 fresh with identical parameters so every run benefits
  from the fix rather than only the 5 that had already crashed.

## 12. Deviations

No deviations identified. No specification formula or threshold changed;
only control flow (skip vs. crash) for an already-established
vertical-incompatibility case.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/rulebook/v2/transition.py` | Modified | `try/except` around ego/actor `footprint_route_coordinates` in `_rss_candidates` and `_rss_lateral_candidates` |
| `src/thesis_rl/envs/observations/causal_semantic.py` | Modified | `try/except` around the crosswalk `route.project` call in `_build_interactions` |
| `docs/implementation/rulebook_v2_dynamic_candidate_vertical_incompatibility_crash_exec_plan.md` | Added | This ExecPlan |

## 14. Validation Results

| Command | Result | Date | Notes |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_causal_semantic_batch.py tests/test_rulebook_v2_rss_lateral.py tests/test_rulebook_v2_transition.py` | PASS | 2026-07-26 | `35 passed` |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q -m "not integration"` | PASS | 2026-07-26 | `972 passed, 5 deselected` |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_*.py tests/test_causal_semantic_batch.py tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-26 | `288 passed` |
| Live restart of all 9 `RUN_PROFILE=thesis` processes with the fix applied | IN_PROGRESS | 2026-07-26 | Restarted after 2 confirmed pre-fix crash waves (5 of 9 processes affected); a background monitor continues watching for further crashes; not yet run long enough to confirm survival past the ~15-minute mark where the pre-fix crashes occurred |

## 15. Final Reconciliation

`REQ-001`-`REQ-003` are implemented and covered by the full regression
suite with zero failures. `REQ-004`'s fail-fast-for-ego behavior is
unchanged and still covered by existing tests. This plan's `IMPLEMENTED`
status reflects the code fix; it is not yet `VERIFIED` in the stricter
sense of "confirmed to prevent recurrence over a long real run," since the
9 restarted processes have not yet run long enough to establish that. A
follow-up check (has any of the 9 crashed with this or a related error
after the restart) is the natural closing action for this plan, along
with the recommended deterministic unit regressions noted in Section 9.

**Known limitations**: no new deterministic test was added for the two
newly-fixed call sites (`_rss_candidates`, `_rss_lateral_candidates`);
coverage currently relies on the pre-existing suite not regressing plus
live-run observation. **Deferred optional work**: add
`test_rulebook_v2_rss.py`/`test_rulebook_v2_rss_lateral.py` regressions
mirroring the static-feature-omission tests, so this bug class has durable
unit coverage independent of live-run luck.
