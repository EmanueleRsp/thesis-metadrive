# ExecPlan: PG Roundabout Vehicle-Yield Priority Data Population

## 1. Metadata

- Feature / plan ID: `pg-roundabout-vehicle-yield-priority-v1`
- Authoritative specification: `specifications/rulebook_v4.7_specification.md`
  (vehicle_yield roundabout priority predicate, unchanged); this plan only
  populates a pre-existing, already-approved data schema
  (`RoundaboutPriorityRecord`, `rulebook_vehicle_yield.roundabout_priorities`
  metadata key) that was never populated by the real PG generation pipeline
- Status: `VERIFIED`
- Created: 2026-07-25
- Last updated: 2026-07-25
- Branch: `scenarionet-implementation`
- Related decisions: none new — no scientific formula, threshold, or
  acceptance behavior changes; this is a data-sourcing fix for an existing,
  already-approved consumption mechanism
- Owner: n/a (single session)

## 2. Objective And Scope

**Objective**: for PG-generated scenarios containing a MetaDrive
`Roundabout` block (`block.ID == "O"`), populate
`scenario["metadata"]["rulebook_vehicle_yield"]["roundabout_priorities"]`
so that `vehicle_yield`'s `roundabout_priority` predicate
(`src/thesis_rl/rulebook/v2/transition.py`) can actually grant circulating
traffic priority over entering traffic, instead of the mechanism being
permanently inert (always empty) as it was before this plan.

**Why**: a full audit of the rulebook v2 system found that
`RoundaboutPriorityRecord`/`roundabout_priority_records` — a mechanism
that already exists, is already consumed by `vehicle_yield`, and is
already unit-tested via synthetic fixtures — was never populated by the
real PG scenario generation pipeline. `vehicle_yield_records_from_metadata`
(`src/thesis_rl/rulebook/v2/context/static_adapter.py`) reads
`metadata["rulebook_vehicle_yield"]["roundabout_priorities"]`, but no
script in `src/thesis_rl/scenarios/pg/` ever wrote that key; `cache.roundabout_priority_records`
was consequently always `()` for real scenarios, and the `roundabout_priority`
OR-branch in `vehicle_yield`'s priority predicate never fired. This also
affects the policy observation feature at
`src/thesis_rl/envs/observations/causal_semantic.py:1535-1557`, which
emits the "roundabout" conflict-zone type index only when a matching
`roundabout_priority_records` entry exists — so the same gap silently
starved a designed observation channel, not only the reward.

**In scope**:

- A new pure function `roundabout_priority_records(map_obj)`
  (`src/thesis_rl/scenarios/pg/roundabout_priority.py`) that derives
  entry/circulating lane-id pairs directly from a live MetaDrive
  `Roundabout` block's own authoritative graph structure (never inferred
  from generic topology, consistent with the existing
  `RoundaboutPriorityRecord` design constraint).
- Wiring this into `generate_pg_scenario`
  (`src/thesis_rl/scenarios/pg/generator.py`), writing the derived records
  into the exported scenario's metadata before the MetaDrive environment
  is closed.
- An end-to-end integration test that generates a real roundabout scenario
  (profile `P2_merge_or_roundabout`, seed 14, deterministically samples a
  single-block `("O",)` map) and verifies the exported metadata resolves
  against the real `map_features` lane ids.

**Out of scope**:

- Waymo/real-dataset roundabout detection (rejected: would require a
  topology-only geometric heuristic, violating the existing
  `RoundaboutPriorityRecord` design constraint; not attempted here).
- `movement_priorities` (pairwise right-of-way at uncontrolled
  intersections): investigated and found infeasible without inventing
  data — MetaDrive's `X`/`T` intersection blocks are generated fully
  symmetric, with no major/minor-road concept to derive from (confirmed by
  reading `third_party/metadrive/metadrive/component/pgblock/intersection.py`,
  `t_intersection.py`, and `pg_space.py`). Not implemented; documented as a
  known, permanent limitation of PG-sourced scenarios.
- Any change to `vehicle_yield`'s priority predicate logic itself
  (`_vehicle_yield_inputs`/`_pre_state_priority_and_gap` in
  `transition.py`), which already correctly consumes
  `cache.roundabout_priority_records` — this plan only supplies real data
  to an already-correct consumer.

**Compatibility**: no public interface, configuration key, or checkpoint
schema change. Adds a new (optional, empty-by-default for non-roundabout
scenarios) metadata key to PG-exported scenarios. `movement_priorities`
under the same `rulebook_vehicle_yield` metadata key is left unset (PG
generation still supplies no data for it, matching pre-existing behavior).

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-001` | For every `Roundabout` block instance in a generated PG map, the exported scenario's `rulebook_vehicle_yield.roundabout_priorities` metadata must contain one record per (entry lane, circulating lane) pair, with lane ids matching the exported `map_features` keys exactly. | Data-sourcing fix for the already-approved `RoundaboutPriorityRecord` consumption contract; no specification change. |
| `REQ-002` | The derivation must be based only on the live MetaDrive block's own authoritative graph structure, never on inferred/heuristic topology. | Existing design constraint already documented in `RoundaboutPriorityRecord`'s docstring (`types.py`). |
| `REQ-003` | Non-roundabout PG scenarios must be unaffected (no metadata key added, no behavior change). | Compatibility. |

## 4. Current Repository Analysis

- `VERIFIED`: `vehicle_yield_records_from_metadata`
  (`src/thesis_rl/rulebook/v2/context/static_adapter.py:52-110`) reads
  `metadata["rulebook_vehicle_yield"]["roundabout_priorities"]`, returning
  `()` if the key is absent — confirmed via `grep` that, before this plan,
  no script under `src/thesis_rl/scenarios/` or `scripts/` ever wrote that
  key; it appeared only in `tests/rulebook_scenario_fixtures.py`.
- `VERIFIED`: `vehicle_yield`'s `roundabout_priority` predicate
  (`transition.py:689-692`, `:900-903`) already correctly checks
  `record.entry_lane_id == ego_lane.lane_id and record.circulating_lane_id == other_lane.lane_id`
  against `cache.roundabout_priority_records` — no change needed on the
  consumption side.
- `VERIFIED`: `causal_semantic.py:1535-1557`'s `_conflict_zone_type_index`
  emits the roundabout type index (3) only when a matching
  `roundabout_priority_records` entry exists, so this plan also un-starves
  that observation feature as a side effect (no code change there
  required).
- `VERIFIED` (via `Explore` agent, reading MetaDrive source directly, no
  container execution): `third_party/metadrive/metadrive/component/pgblock/intersection.py`
  and `t_intersection.py` build all arms of `X`/`T` blocks symmetrically;
  `lane.priority` is only ever propagated forward from a preceding lane,
  never assigned asymmetrically during PG intersection construction; the
  only non-trivial `priority` assignment in the codebase comes from
  imported SUMO maps (`metadrive/utils/sumo/map_utils.py:138`), not PG
  generation. This rules out deriving `movement_priorities` from PG data
  without inventing a new convention.
- `VERIFIED` empirically (real `docker compose run --rm dev` execution,
  `MetaDriveEnv({"map": "SORO"})`, two live `Roundabout` block instances
  inspected): `PGBlock.node(block_idx, part_idx, road_idx)` produces node
  names of the exact form `f"{block_idx}{ID}{part_idx}_{road_idx}_"` (e.g.
  `"2O0_2_"`), confirming `DASH = "_"`. For a `Roundabout` block
  (`SOCKET_NUM = 3`, 4 total arms including the pre-connected one), the
  block's own `block_network.graph` contains: 8 "ring" edges (road_idx 0/1
  on both endpoints, forming the closed circulating loop), 4 "entry"
  edges (landing on a road_idx-0 ring node, originating from either the
  external predecessor block or one of the 3 socket negative roads), plus
  4 "exit" edges and 4 ring-to-socket "connector" edges not needed for
  this plan. Cross-checked all derived entry/ring lane ids (`str(lane.index)`)
  against `map_obj.get_map_features()` keys: zero missing keys, for two
  independent roundabout instances at different block indices (2 and 4).
- `VERIFIED` empirically: `env.current_map.road_network` is a
  `NodeRoadNetwork`; its `get_map_features()` keys are
  `str(lane.index)` where `lane.index` is the `(from_node, to_node,
  lane_position)` tuple — exactly matching the lane-id format the rulebook
  PG adapter already reads from `scenario["map_features"]`.
- `VERIFIED` empirically: `generate_pg_scenario("P2_merge_or_roundabout",
  seed=14, ...)` deterministically samples a single-block `("O",)` map
  (checked via `GenerationSpec.sample` across seeds 0-29); used as the
  integration test's fixed seed.
- `VERIFIED` end-to-end (real container execution, separate from the
  automated test suite): loaded the exported scenario `.pkl` through
  `build_pg_static_adapter_result` (the actual rulebook PG adapter) —
  128 `RoundaboutPriorityRecord`s parsed with zero validation errors,
  confirming the full pipeline (generation → export → rulebook adapter
  ingestion) works end-to-end, not only at the unit level.

## 5. Assumptions And Invariants

- `PGBlock.node()`'s naming scheme (`{block_idx}{ID}{part_idx}_{road_idx}_`,
  road_idx 0/1 = ring, 2/3 = socket branch) is MetaDrive's own internal,
  versioned convention (pinned `third_party/metadrive` submodule); a
  future MetaDrive upgrade that changes this convention would silently
  break `roundabout_priority_records`'s classification logic (it would
  likely produce empty or malformed records, not a crash, since the regex
  simply wouldn't match) — flagged as a fragility to watch on submodule
  upgrades, not fixed here.
- The derivation is intentionally conservative/generous: every entry lane
  is paired with every circulating-ring lane of the same roundabout
  instance (not just the geometrically nearest segment). This matches the
  real-world roundabout rule ("yield to all circulating traffic", not only
  the adjacent segment) and is safely scoped by the pre-existing
  conflict-zone/movement-key matching in `vehicle_yield`, which only ever
  evaluates this predicate for actor/lane pairs already linked by a real
  geometric conflict zone.
- No randomness introduced: `roundabout_priority_records` is a pure
  function of the live map's block/lane graph.

## 6. Decisions And Approval Gates

No new material decision: this is a data-sourcing fix for an
already-approved, already-implemented consumption mechanism
(`RoundaboutPriorityRecord`, approved as part of the v4.7 vehicle_yield
design). No specification formula, threshold, or acceptance behavior
changes.

## 7. Proposed Design

`src/thesis_rl/scenarios/pg/roundabout_priority.py` (new module):

```python
def roundabout_priority_records(map_obj) -> list[dict[str, str]]:
    ...
```

For every `block.ID == "O"` in `map_obj.blocks`: classify each edge of
`block.block_network.graph` via a regex over MetaDrive's own node-naming
convention (`^-?{block.name}(\d+)_(\d+)_$`) into "ring" (both endpoints
road_idx 0/1) or "entry" (destination road_idx 0, origin not itself a
ring node); emit one `{component_id, entry_lane_id, circulating_lane_id}`
dict per (entry lane, ring lane) pair, `lane_id = str(lane.index)`.

`generate_pg_scenario` (`generator.py`) calls this against
`env.current_map` before `env.close()`, and — only if non-empty — sets
`scenario["metadata"]["rulebook_vehicle_yield"]["roundabout_priorities"]`.

No change to the rulebook v2 consumption side (`static_adapter.py`,
`transition.py`) — it already correctly reads this schema.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-001` | `AC-001` | `pg/roundabout_priority.py::roundabout_priority_records`, `pg/generator.py::generate_pg_scenario` | `tests/test_pg_generator_integration.py::test_roundabout_generation_populates_vehicle_yield_priority_metadata` | Verified |
| `REQ-002` | `AC-002` | Derivation reads only `block.ID`, `block.name`, `block.block_network.graph`, `lane.index` — no geometric/topology heuristic | Same test; additionally cross-checked manually against real `map_features` (Section 4) | Verified |
| `REQ-003` | `AC-003` | `generate_pg_scenario` only sets the metadata key `if roundabout_priorities:` (non-empty) | `tests/test_pg_generator_integration.py::test_p0_generation_exports_reloadable_scenario` (P0_simple, no roundabout, unmodified, still passing) | Verified |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-001` | Integration (`@pytest.mark.integration`, real MetaDrive PG generation) | Roundabout scenario export populates valid, resolvable priority metadata | `generate_pg_scenario("P2_merge_or_roundabout", seed=14, ...)` | Exported `.pkl`'s `metadata["rulebook_vehicle_yield"]["roundabout_priorities"]` is non-empty; every `entry_lane_id`/`circulating_lane_id` resolves in `map_features`; no entry equals its own circulating lane | `REQ-001`, `REQ-002` |
| `TEST-002` (regression guard) | Integration | Non-roundabout PG generation is unaffected | `tests/test_pg_generator_integration.py::test_p0_generation_exports_reloadable_scenario` (unchanged) | Passes unchanged | `REQ-003` |

Commands:

- `docker compose run --rm dev uv run --no-sync python -m pytest -q -m integration tests/test_pg_generator_integration.py`
- `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_*.py tests/test_rulebook_synthetic_scenarios.py tests/test_pg_*.py`
- `make rulebook-v2-check`

## 10. Milestones

### M1 — Empirical structure verification (no code changes)

- Objective: determine, from real MetaDrive execution, whether a
  `Roundabout` block's entry/circulating lane structure is derivable
  without inventing data.
- Status: Done (Section 4).

### M2 — Implementation and end-to-end test

- Objective: implement `roundabout_priority_records`, wire it into
  `generate_pg_scenario`, add `TEST-001`, verify end-to-end through the
  real rulebook PG adapter.
- Status: Done and verified (Section 14).
- Expected files: `src/thesis_rl/scenarios/pg/roundabout_priority.py`
  (new), `src/thesis_rl/scenarios/pg/generator.py`,
  `tests/test_pg_generator_integration.py`.

## 11. Progress And Findings Log

- 2026-07-25 — Requested as the third of three follow-up fixes after the
  vehicle_yield/crosswalk latch-cleanup fixes. Investigated feasibility
  first (per user's explicit "without inventing data" constraint): used
  the `Explore` agent to confirm PG's `X`/`T` intersection blocks are
  symmetric (ruling out `movement_priorities`), then personally verified,
  via three separate real `docker compose run --rm dev` executions, the
  `Roundabout` block's node-naming convention, the ring/entry
  classification rule, and the exact byte-for-byte lane-id format used by
  both MetaDrive's `NodeRoadNetwork.get_map_features()` and the rulebook's
  PG adapter — cross-checking derived lane ids against real
  `map_features` keys (zero mismatches across two independent roundabout
  instances) before writing any production code. Implemented, added an
  end-to-end integration test using a seed (14) verified to deterministically
  produce a single-roundabout map, and additionally ran a standalone
  end-to-end check loading the exported scenario through the real
  `build_pg_static_adapter_result` (not only the test's own assertions),
  confirming 128 records parse with zero validation errors. All temporary
  debug/verification scripts were removed before finalizing.

## 12. Deviations

No deviations identified. No specification formula, threshold, or
acceptance behavior changed; `vehicle_yield`'s consumption logic is
unmodified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/scenarios/pg/roundabout_priority.py` | Added | `roundabout_priority_records(map_obj)` pure derivation function |
| `src/thesis_rl/scenarios/pg/generator.py` | Modified | `generate_pg_scenario` calls the new function and writes non-empty results into scenario metadata before `env.close()` |
| `tests/test_pg_generator_integration.py` | Modified | Add `TEST-001` end-to-end integration test and a `_load_exported_scenario` helper |
| `docs/implementation/pg_roundabout_vehicle_yield_priority_exec_plan.md` | Added | This ExecPlan |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q -m integration tests/test_pg_generator_integration.py -v` | PASS | 2026-07-25 | `2 passed` (existing `test_p0_generation_exports_reloadable_scenario` unchanged, new `test_roundabout_generation_populates_vehicle_yield_priority_metadata`). |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_*.py tests/test_rulebook_synthetic_scenarios.py tests/test_pg_*.py` | PASS | 2026-07-25 | `284 passed`. |
| `make rulebook-v2-check` | PASS | 2026-07-25 | `228 passed`; `ruff check` clean; `git diff --check` clean. |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/scenarios/pg/roundabout_priority.py src/thesis_rl/scenarios/pg/generator.py tests/test_pg_generator_integration.py` | PASS | 2026-07-25 | `All checks passed!` |
| `docker compose run --rm dev uv run --no-sync ruff format --check` (same files) | PARTIAL | 2026-07-25 | `generator.py` "would be reformatted"; `ruff format --diff` shows the only hunks are two pre-existing lines (`lane_num=...`, `output = ...`) this plan did not touch — pre-existing repository formatting debt per `AGENTS.md`, not introduced here. `roundabout_priority.py` and the test file are already formatted. |
| Standalone end-to-end check through `build_pg_static_adapter_result` (real PG generation + real rulebook adapter, not part of the automated suite) | PASS | 2026-07-25 | `128` `RoundaboutPriorityRecord`s parsed, `validation_errors: ()`. |

## 15. Final Reconciliation

- `REQ-001`: `IMPLEMENTED` and `VERIFIED` — `TEST-001` passes; independently
  confirmed end-to-end through the real rulebook PG adapter.
- `REQ-002`: `IMPLEMENTED` and `VERIFIED` — derivation is a pure function of
  the live block's own graph/lane objects; no geometric/topology
  heuristic used.
- `REQ-003`: `IMPLEMENTED` and `VERIFIED` — the existing P0 (non-roundabout)
  integration test passes unmodified.

**Resulting behavior**: PG-generated scenarios containing a `Roundabout`
block now export real vehicle-yield roundabout priority data; `vehicle_yield`'s
`roundabout_priority` predicate can fire for these scenarios (previously
always inert); the `causal_semantic.py` "roundabout" conflict-zone-type
observation feature is un-starved as a side effect, with no code change on
that side.

**Architecture/compatibility**: no public interface, configuration key, or
checkpoint schema changed. Non-roundabout scenarios are unaffected
(verified by the unmodified P0 integration test).

**Executed checks**: see Section 14.

**Approved decisions**: none required (data-sourcing fix for an
already-approved consumption mechanism).

**Deviations**: none (Section 12).

**Known limitations**:
- Waymo-sourced scenarios remain without roundabout priority data (out of
  scope by design — see Section 2).
- `movement_priorities` (pairwise priority at uncontrolled intersections)
  remains unpopulated for PG scenarios: confirmed infeasible without
  inventing a new generation-time convention, since MetaDrive's `X`/`T`
  blocks carry no major/minor-road concept. Documented as a permanent
  limitation, not a deferred task.
- Fragility to MetaDrive submodule upgrades that might change the
  `PGBlock.node()` naming convention (Section 5) — not mitigated here;
  would need a re-verification pass (rerun the empirical checks in
  Section 4) after any `third_party/metadrive` version bump.
- `make test` (the full suite) was not run; the focused rulebook v2 +
  PG-scoped suite (284 tests) plus the dedicated integration test plus
  `make rulebook-v2-check` were used as the practical equivalent for this
  change's blast radius.

**Deferred optional work**: none identified.
