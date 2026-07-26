# ExecPlan: Crosswalk Illegal-Entry Latch Never Persisted When Zone Not Selected

## 1. Metadata

- Feature / plan ID: `crosswalk-latch-cleanup-v1`
- Authoritative specification: `docs/specifications/rulebook_v4.7_specification.md`,
  version 4.7-final-implementation-complete, `Authoritative: YES` (crosswalk
  yield semantics are read but not changed by this plan)
- Status: `VERIFIED`
- Created: 2026-07-25
- Last updated: 2026-07-25
- Branch: `scenarionet-implementation`
- Related ADR: none new — this is a conformance fix restoring the same
  intended behavior already decided for `vehicle_yield` in
  `docs/decisions/ADR-025-vehicle-yield-latch-cleanup-decoupled-from-zone-selection.md`,
  applied here to a second, independently affected component
- Owner: n/a (single session)

## 2. Objective And Scope

**Objective**: once the ego footprint has genuinely and fully left a
crosswalk conflict zone, `memory.crosswalk_illegal_entries` must stop
reporting an active illegal entry for that zone.

**Why**: a full audit of every rulebook v2 component prompted by the
ADR-025 vehicle_yield fix found the same underlying bug class in
`evaluate_crosswalk_yield`/`_crosswalk_inputs`, but with a different,
more severe failure mechanism than vehicle_yield's:

- `select_first_ahead_or_occupied_zone`'s epsilon-scoped "ahead" filter
  excludes a crosswalk zone from candidacy before the ego footprint
  (vehicle-length scoped) actually stops intersecting it — identical
  geometric mechanism to the vehicle_yield bug.
- Every fallback branch in `_crosswalk_inputs` (including the "zone no
  longer selected" case) sets `vertical_applicable=False`.
- `evaluate_crosswalk_yield`'s `not vertical_applicable` early-return path
  returned a **completely empty `MemoryDelta()`** — no write at all for
  `crosswalk_illegal_entries` — rather than persisting the (already
  correctly computed) pruned value. An unwritten field is never updated by
  `merge_memory_deltas`, so it silently keeps whatever value it already
  had. This is a stricter failure than vehicle_yield's original bug: even
  the wiring-layer cleanup considered as a fix candidate first (mirroring
  `_cleared_vehicle_yield_illegal_entries`) was not sufficient on its own,
  because the correctly cleared value was computed but then discarded by
  the component's own early-return branch before it could reach memory.

**In scope**:

- A wiring-layer helper `_cleared_crosswalk_illegal_entries` in
  `src/thesis_rl/rulebook/v2/transition.py`, mirroring
  `_cleared_vehicle_yield_illegal_entries`, that prunes stale
  `(actor_id, zone_id)` entries using cached zone geometry
  (`cache.conflict_zones`), independent of this step's zone selection.
- Registering the selected crosswalk zone into the shared
  `cache.conflict_zones` cache (via a new `CacheDelta` returned by
  `_crosswalk_inputs`), since crosswalk zones were not previously cached
  there at all (only `vehicle_yield` zones were).
- A fix inside `evaluate_crosswalk_yield`
  (`src/thesis_rl/rulebook/v2/components/controls.py`): the
  `not vertical_applicable` early-return branch now writes
  `previous_illegal_entries` back to memory instead of an empty
  `MemoryDelta()`.
- An end-to-end regression test driving `evaluate_transition` through and
  past a crosswalk conflict zone across multiple steps.

**Out of scope**:

- Any change to `select_first_ahead_or_occupied_zone` (unchanged, same
  rationale as ADR-025).
- Any change to `evaluate_crosswalk_yield`'s cost formula, its existing
  unit tests, or the meaning of `vertical_applicable` for the genuinely
  non-applicable cases (no crosswalk feature in the scenario, ambiguous
  movement key, no geometric candidates) — those still return `cost=0.0`,
  `NOT_APPLICABLE`, unchanged.
- `movement_priorities`/`roundabout_priority_records` population gaps
  (separately tracked, require new data-sourcing work, not a code bug).

**Compatibility**: no public interface, configuration key, checkpoint
schema, or reward-vector shape change. Only the temporal accuracy of the
`crosswalk_illegal_entries` memory field changes, plus a new (namespaced,
collision-free) entry type in the shared `cache.conflict_zones` mapping.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-001` | Once the ego footprint no longer intersects a crosswalk conflict zone's cached polygon, no `(actor_id, zone_id)` entry for that zone may remain in `memory.crosswalk_illegal_entries` after the transition step that observes the exit. | Derived from the same zone-selection contract as vehicle_yield (§2.8.2 pattern); memory-management correction, not a specification behavior change. |
| `REQ-002` | The currently selected zone's cost/gap computation for the step must be unchanged for all previously passing scenarios. | Unchanged `evaluate_crosswalk_yield` cost formula. |

## 4. Current Repository Analysis

- `VERIFIED`: `_crosswalk_inputs` (`src/thesis_rl/rulebook/v2/transition.py`)
  falls back to sentinel zone ids (`"__no_crosswalk__"`,
  `"__ambiguous_crosswalk__"`, `"__no_relevant_crosswalk__"`) or, for a
  genuinely selected zone whose occupancy prediction fails, keeps the real
  `zone_id` — in every one of these paths `vertical_applicable` was `False`.
- `VERIFIED`: `evaluate_crosswalk_yield`
  (`src/thesis_rl/rulebook/v2/components/controls.py:302-315`, pre-fix)
  returned `MemoryDelta()` unconditionally whenever
  `vertical_applicable=False`, discarding whatever
  `previous_illegal_entries` value it was given.
- `VERIFIED`: unlike `vehicle_yield` (whose `empty` fallback dict in
  `_vehicle_yield_inputs` has no early-return-and-discard shortcut inside
  `evaluate_vehicle_yield`, so it always persists
  `previous_illegal_entries`), crosswalk's architecture funnels every
  "not this step" case through a single shortcut that used to drop the
  memory write entirely. This made the bug's impact worse in practice:
  once a crosswalk zone stopped being selected, `crosswalk_illegal_entries`
  became permanently frozen at whatever value it last had, for the rest of
  the episode, regardless of any wiring-layer cleanup upstream.
- `VERIFIED`: prior to this plan, crosswalk zones were never registered
  into `cache.conflict_zones` (only `vehicle_yield` wrote
  `CacheDelta.new_conflict_zones`); `_crosswalk_zone_id` already
  namespaces its hash payload with `"namespace": "crosswalk"`
  (`src/thesis_rl/rulebook/v2/geometry/conflict_zones.py:146-165`), so
  sharing the same cache mapping introduces no key-collision risk with
  vehicle_yield zone ids.
- `VERIFIED`: `_build_interactions` in
  `src/thesis_rl/envs/observations/causal_semantic.py:1111` already guards
  `if zone.other_movement_key is None: continue` before consuming any
  cached `ConflictZoneRecord`, so registering crosswalk records (which have
  no `other_movement_key`) into the shared cache cannot be misread by the
  observation-side interaction builder.
- `VERIFIED` empirically: reproduced the failure by tracing
  `_cleared_crosswalk_illegal_entries`'s own return value across a 5-step
  transition sequence — it correctly computed an empty pruned set once
  ego's footprint left the zone, yet the persisted
  `memory.crosswalk_illegal_entries` after that same step still contained
  the stale entry. Isolated the discrepancy to the
  `not vertical_applicable` early return in `evaluate_crosswalk_yield`
  (temporary debug prints, removed before finalizing the fix).
- Existing tests: `tests/test_rulebook_v2_crosswalk.py` (pure evaluator
  unit tests, `vertical_applicable=True` only — did not exercise this
  path), `tests/test_rulebook_v2_transition.py` (no crosswalk integration
  test existed before this plan).

## 5. Assumptions And Invariants

- Same route/footprint assumptions as ADR-025 (monotonic curvilinear `s`,
  canonical stable zone polygons for the lifetime of an episode).
- The fix must not alter `evaluate_crosswalk_yield`'s cost/status for any
  case where `vertical_applicable=True` (`REQ-002`).
- For genuinely non-applicable cases (no crosswalk feature, ambiguous
  movement key, no geometric candidates), `previous_illegal_entries` is
  passed through unchanged from `cleared_illegal_entries`, so writing it
  back is a no-op relative to the current step's own logic — it only
  matters when the upstream value has actually changed (the "zone left"
  case).

## 6. Decisions And Approval Gates

No new material decision: this is a conformance fix restoring the intended
memory-management behavior already established as correct for
`vehicle_yield` (ADR-025), applied to a second component with an
independently discovered variant of the same bug. No specification
formula, threshold, or acceptance behavior changes.

## 7. Proposed Design

1. `_cleared_crosswalk_illegal_entries(*, cache, memory, ego_footprint)` —
   pure helper mirroring `_cleared_vehicle_yield_illegal_entries`.
2. `_crosswalk_inputs` computes this once, from `post.ego.footprint`, and
   uses it for `previous_illegal_entries` in every return path; it now
   returns `tuple[dict, CacheDelta]`, registering a `ConflictZoneRecord`
   for the selected zone (`other_movement_key=None`) whenever one is
   selected, so the next step's cleanup helper can find its polygon.
3. `evaluate_transition` merges the new `crosswalk_cache_delta` with the
   existing `vehicle_cache_delta` into one `combined_cache_delta` before
   passing it as `pending_cache_delta`.
4. `evaluate_crosswalk_yield`'s `not vertical_applicable` branch now
   returns `MemoryDelta(writer="crosswalk", writes=(("crosswalk_illegal_entries", previous_illegal_entries),))`
   instead of `MemoryDelta()`.

No new component, no new memory field, no new configuration, no fallback
policy change.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-001` | `AC-001` | `transition.py::_cleared_crosswalk_illegal_entries`, `_crosswalk_inputs`; `components/controls.py::evaluate_crosswalk_yield` | `tests/test_rulebook_v2_transition.py::test_transition_clears_crosswalk_illegal_entry_latch_after_ego_fully_exits_zone` | Verified |
| `REQ-002` | `AC-002` | (no change to cost formula) | `tests/test_rulebook_v2_crosswalk.py` (unmodified, still passing) | Verified |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-001` | Integration (end-to-end `evaluate_transition`, multi-step, `apply_cache_delta` chained) | Illegal-entry latch clears once ego footprint has fully left the crosswalk zone | Ego drives through and past a static crosswalk zone (x∈[8,12]) with a stationary pedestrian inside it, recording an illegal entry on first occupied entry | `next_memory.crosswalk_illegal_entries` is empty once ego's footprint no longer intersects the zone | `REQ-001` |
| `TEST-002` (regression guard) | Unit | Existing pure-evaluator tests remain valid and unmodified | `tests/test_rulebook_v2_crosswalk.py` (unchanged) | All pass unchanged | `REQ-002` |

Commands:

- `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_transition.py tests/test_rulebook_v2_crosswalk.py`
- `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_*.py tests/test_rulebook_synthetic_scenarios.py`
- `make rulebook-v2-check`

## 10. Milestones

### M1 — Fix and regression test

- Objective: implement `_cleared_crosswalk_illegal_entries`, register
  crosswalk zones into `cache.conflict_zones`, fix
  `evaluate_crosswalk_yield`'s early-return memory write, add `TEST-001`.
- Status: Done and verified (Section 14).
- Expected files: `src/thesis_rl/rulebook/v2/transition.py`,
  `src/thesis_rl/rulebook/v2/components/controls.py`,
  `tests/test_rulebook_v2_transition.py`.
- Tests/commands: see Section 9.
- Completion evidence: see Section 14.

## 11. Progress And Findings Log

- 2026-07-25 — Found during a requested full audit of all rulebook v2
  components for the same bug class as ADR-025. Initial hypothesis (a
  wiring-layer-only fix, exactly mirroring
  `_cleared_vehicle_yield_illegal_entries`) was implemented first and
  appeared syntactically correct, but the first regression test run still
  failed with the stale entry persisting. Added temporary debug prints
  inside `_cleared_crosswalk_illegal_entries` and traced a full 5-step
  transition sequence: confirmed the helper itself correctly computed an
  empty pruned set at the exact step the ego footprint left the zone, yet
  the resulting `memory.crosswalk_illegal_entries` still contained the
  stale entry. Isolated the discrepancy to
  `evaluate_crosswalk_yield`'s `not vertical_applicable` early return,
  which discarded any `previous_illegal_entries` value via an empty
  `MemoryDelta()` rather than persisting it. Fixed by writing
  `previous_illegal_entries` back to memory in that branch. Re-ran the
  regression test: passes. Removed all temporary debug prints before
  finalizing. Full `tests/test_rulebook_v2_*.py` suite (273 tests) and
  `make rulebook-v2-check` both pass after the fix.

## 12. Deviations

No deviations identified. `select_first_ahead_or_occupied_zone` is
unchanged; `evaluate_crosswalk_yield`'s cost formula and status logic for
`vertical_applicable=True` are unchanged.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/rulebook/v2/transition.py` | Modified | Add `_cleared_crosswalk_illegal_entries`; `_crosswalk_inputs` returns `(dict, CacheDelta)` and registers the selected crosswalk zone into `cache.conflict_zones`; merge crosswalk and vehicle cache deltas before `pending_cache_delta` |
| `src/thesis_rl/rulebook/v2/components/controls.py` | Modified | `evaluate_crosswalk_yield`'s `not vertical_applicable` branch now persists `previous_illegal_entries` |
| `tests/test_rulebook_v2_transition.py` | Modified | Add `TEST-001` end-to-end regression test, new `MapFeatureClass`/`MapFeatureRecord` imports |
| `docs/implementation/crosswalk_illegal_entry_latch_cleanup_exec_plan.md` | Added | This ExecPlan |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_transition.py -k crosswalk` | PASS | 2026-07-25 | `1 passed` (the new `TEST-001`). |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_*.py tests/test_rulebook_synthetic_scenarios.py` | PASS | 2026-07-25 | `273 passed` (full rulebook v2 suite, up from 272 before this plan). |
| `make rulebook-v2-check` | PASS | 2026-07-25 | `228 passed` (scoped `test_rulebook_v2_*.py`); `ruff check` clean; `git diff --check` clean. |
| Manual trace confirming the bug before the `evaluate_crosswalk_yield` fix | FAIL (expected, pre-fix) | 2026-07-25 | With only the wiring-layer helper in place (no fix to `evaluate_crosswalk_yield`), `memory.crosswalk_illegal_entries` still contained the stale entry at steps after ego fully left the zone, despite `_cleared_crosswalk_illegal_entries` itself correctly returning an empty set — proving the early-return memory-write gap was the actual root cause, not the wiring layer. |

## 15. Final Reconciliation

- `REQ-001`: `IMPLEMENTED` and `VERIFIED` — `TEST-001` passes; the latch now
  clears at the exact step ego's footprint leaves the zone, regardless of
  whether the zone is still selected for the current step.
- `REQ-002`: `IMPLEMENTED` and `VERIFIED` — `tests/test_rulebook_v2_crosswalk.py`
  passes unmodified; no change to the cost formula or to the
  `vertical_applicable=True` code path.

**Resulting behavior**: `memory.crosswalk_illegal_entries` no longer
retains an `(actor_id, zone_id)` entry once the ego footprint has
genuinely left that zone's cached polygon. Crosswalk zones are now also
registered into `cache.conflict_zones`, consistent with vehicle_yield.

**Architecture/compatibility**: no public interface, configuration key,
checkpoint schema, or reward-vector shape changed.

**Executed checks**: see Section 14.

**Approved decisions**: none required (conformance fix, no material
deviation).

**Deviations**: none (Section 12).

**Update 2026-07-25**: `crosswalk`'s `"ego_entered"` detection was changed
to use the swept front bumper (pre→post), mirroring `vehicle_yield`'s
DEC-005/REQ-VY-02 convention, for defensive consistency. However, a
geometric analysis attempting to construct a regression test proving a
behavioral difference found that, for straight-line ego motion, the
scenario this refinement is meant to catch (footprint entry missed by
the plain post-state polygon check but caught by the swept bumper) is
**mathematically unreachable** given how `select_first_ahead_or_occupied_zone`
couples zone selection to the same footprint/position: its "ahead"
criterion (`route_exit_s_m >= front_s - 0.05`) fails as soon as the front
edge has advanced meaningfully past the zone, well before the discrete
footprint (vehicle-length scoped) could plausibly still miss it while the
swept bumper catches it — the two conditions cannot hold simultaneously
for translational motion. This mirrors the same finding for
`vehicle_yield`'s own REQ-VY-02, which likewise has no dedicated
regression test in this repository for that exact reason. The change was
kept (harmless, consistent with the established convention, and the swept
bumper would matter for a rotating/turning ego where the footprint's shape
changes non-trivially between pre and post — a scenario not constructed
here) but no new test was added for it, since one could not be written
without a misleading or artificial setup. Full `tests/test_rulebook_v2_*.py`
suite (273 tests, unchanged) re-run and confirmed no regression.

**Known limitations**:
- `make test` (the full suite) was not run; the focused rulebook v2 suite
  (273 tests) plus `make rulebook-v2-check` were used as the practical
  equivalent for this change's blast radius, which is confined to
  `transition.py` and `components/controls.py`.

**Deferred optional work**: none identified.
