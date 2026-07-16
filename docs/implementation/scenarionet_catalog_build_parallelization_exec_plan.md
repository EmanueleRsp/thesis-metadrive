# ScenarioNet Catalog Build Parallelization ExecPlan

## 1. Metadata

- Feature and plan ID: ScenarioNet catalog build parallelization, `SCB-PAR-001`
- Historical authoritative specification at implementation: `docs/specifications/scenarionet_integration_v1_specification.md`, version `v1`
- Current authoritative specification: `docs/specifications/scenarionet_integration_spec_v1.1.md`, version `1.1`; reconciliation is tracked in `docs/implementation/scenarionet_integration_spec_v1.1_exec_plan.md`
- Related specification: `docs/specifications/rulebook_v4.6_specification.md`, version `4.6-final-implementation-complete`, `AUTHORITATIVE`
- Status: `IMPLEMENTED`
- Created: 2026-07-16
- Last updated: 2026-07-16
- Branch: current working branch
- Related ADRs: none; implementation-level performance change
- Owner: repository maintainer

## 2. Objective And Scope

Parallelize the CPU-heavy per-file loading and feature extraction performed by
the ScenarioNet catalog builder, and expose Rich progress for the operation.
Keep final Parquet, group mapping, and report writes in the parent process so
artifact ordering, atomicity, and scientific semantics remain unchanged.

In scope:

- bounded spawned process pool shared by catalog loading and existing Rulebook
  eligibility evaluation;
- configurable `workers` for Waymo and PG catalog loading;
- deterministic result ordering based on the existing sorted file order;
- parent-owned Rich progress for Waymo/PG loading;
- regression tests comparing serial and parallel loader outputs.

Out of scope:

- changes to feature formulas, quality policy, grouping, arm assignment, split
  policy, report schema, or Parquet schema;
- parallel artifact writers or distributed execution;
- resuming partial catalog builds.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-SCB-001` | Waymo and PG remain represented through the canonical ScenarioDescription/catalog interfaces. | ScenarioNet v1 §2.3–§2.5, §7 |
| `REQ-SCB-002` | Catalog paths are relative to the configured data root and records remain reproducible/deterministic. | ScenarioNet v1 §4, §7, §24 |
| `REQ-SCB-003` | Scenario validation, feature extraction, quality filtering, and arm assignment semantics remain unchanged. | ScenarioNet v1 §16–§17 |
| `REQ-SCB-004` | Pipeline logging exposes operational progress while machine-readable outputs remain available. | ScenarioNet v1 §24 |
| `REQ-SCB-005` | Parallel work is bounded and configurable without a new dependency. | User request 2026-07-16; repository conventions |

## 4. Current Repository Analysis

- `src/thesis_rl/cli/scenarios/build_catalog.py` loads Waymo and PG serially
  inside one Rich status spinner, then writes the catalog and groups in the
  parent process.
- `src/thesis_rl/scenarios/waymo.py::load_converted_waymo_entries` loops over
  sorted pickle files, loading each file and computing features, grouping,
  quality policy, and records.
- `src/thesis_rl/scenarios/pg/loader.py::load_exported_pg_entries` performs the
  analogous serial work, including per-scenario generation-manifest lookup and
  validation.
- `src/thesis_rl/rulebook/v2/context/catalog_eligibility.py` already contains a
  bounded spawned process implementation introduced by the related catalog
  filter change. It now delegates to the shared ordered process-map helper
  added for catalog loading while preserving its public behavior.
- `src/thesis_rl/cli/scenarios/ui.py::make_progress` is the repository-standard
  Rich progress layout and writes human output to stderr.
- `rich` is already a declared dependency; no dependency addition is required.

Classification:

- `SPECIFIED`: canonical records, relative paths, deterministic grouping, and
  unchanged validation/data policy.
- `VERIFIED`: current serial call graph, sorted file discovery, existing Rich
  UI, and existing process-pool precedent.
- `INFERRED`: per-file loading is independent after file discovery and can be
  safely evaluated in spawned processes; final aggregation remains serial.

## 5. Assumptions And Invariants

- `workers` must be positive; effective workers are capped at the number of
  discovered files.
- At most `2 * effective_workers` tasks are in flight to bound parent memory.
- The parent owns all progress output and all artifact writes.
- Results are reconstructed in the original sorted file order, regardless of
  completion order.
- Worker functions are top-level and picklable under the repository-standard
  `spawn` context.
- A worker exception propagates as a catalog-build failure, matching current
  fail-fast loader behavior; no partial catalog is committed.
- Filtering by PG seed window happens inside the worker before results are
  returned, preserving the existing selected subset.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-SCB-001` | implementation detail | Parallel unit | batch database / per-file task / threads | Per-file spawned process task with bounded queue | Uses available CPU without changing data policy | Accepted by user request |
| `DEC-SCB-002` | implementation detail | Artifact ordering | completion order / sorted reconstruction | Preserve sorted discovery order in parent | Identical Parquet/groups/report semantics | Accepted by user request |
| `DEC-SCB-003` | implementation detail | Shared infrastructure | duplicate pool code / generic helper | Shared ordered bounded process-map helper | Keeps stage 3 and Rulebook filter behavior consistent | Accepted by user request |

No unresolved approval gate remains. No scientific contract or artifact schema
is changed.

## 7. Proposed Design

Add `src/thesis_rl/scenarios/parallel.py` with a generic ordered bounded process
map accepting a top-level worker function, positive worker count, and optional
`(completed, total)` callback. Refactor Rulebook eligibility to use it.

Refactor each loader into:

1. deterministic file discovery and validation in the parent;
2. a top-level per-file worker that performs the existing pickle load,
   extraction, validation, metadata lookup, quality policy, and record creation;
3. parent-side ordered aggregation and existing source-specific assertions.

Extend `load_converted_waymo_entries` and `load_exported_pg_entries` with
keyword-only `workers=1` and progress callback parameters. Update
`build_catalog.py` with `--waymo-workers` and `--pg-workers`, use
`make_progress`, and give each source a Rich task with its own total. The
pipeline stage passes the already resolved `SCENARIONET_WAYMO_WORKERS` and
`SCENARIONET_PG_WORKERS` values to the builder.

The final writer remains serial:

- `write_scenario_catalog`;
- `group_ids_for_entries` and JSON group mapping;
- feature/arm report calculation and report write.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-SCB-001` | `AC-SCB-001` loader outputs remain canonical entries | Waymo/PG loaders | existing loader tests | Implemented and verified |
| `REQ-SCB-002` | `AC-SCB-002` serial/parallel outputs have identical ordered UIDs/paths | parallel helper/loaders | loader determinism tests | Implemented and verified |
| `REQ-SCB-003` | `AC-SCB-003` feature/quality/arm values are unchanged | per-file workers | serial-vs-parallel regression tests | Implemented and verified |
| `REQ-SCB-004` | `AC-SCB-004` CLI shows source progress and preserves stdout JSON | build CLI | CLI fixture smoke | Implemented and verified |
| `REQ-SCB-005` | `AC-SCB-005` worker validation and bounded execution | helper/CLI/config wiring | invalid worker tests | Implemented and verified |

## 9. Test Strategy Defined Before Implementation

Acceptance criteria:

- `AC-SCB-001`: Waymo and PG loader fixtures still return valid canonical
  entries and source-specific grouping/metadata.
- `AC-SCB-002`: `workers=1` and `workers=2` produce identical ordered records,
  group mappings, and selected PG windows.
- `AC-SCB-003`: Existing feature extraction, validation status, quality
  warnings, primary arms, and signal reliability are identical between paths.
- `AC-SCB-004`: Catalog CLI with a fixture emits Rich loading progress to stderr
  and still emits its JSON report on stdout.
- `AC-SCB-005`: Non-positive worker counts fail explicitly; empty/small inputs
  cap effective workers without changing results.

Mandatory test matrix:

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-SCB-001` | Regression | Existing Waymo loader | vendored Waymo fixture | Existing entries/groups unchanged | `REQ-SCB-001` |
| `TEST-SCB-002` | Regression | Existing PG loader | generated/export fixture where available | Existing entries/metadata unchanged | `REQ-SCB-001` |
| `TEST-SCB-003` | Integration | Waymo serial vs parallel | multiple fixture files | Exact ordered equality | `REQ-SCB-002` |
| `TEST-SCB-004` | Integration | PG serial vs parallel/window filter | multiple fixture files | Exact ordered equality and filtering | `REQ-SCB-002` |
| `TEST-SCB-005` | Unit | Invalid workers/progress | small deterministic input | Explicit validation and monotonic callbacks | `REQ-SCB-004`, `REQ-SCB-005` |
| `TEST-SCB-006` | CLI smoke | Build catalog Rich output | small temporary catalog root | Artifacts and stdout report remain valid | `REQ-SCB-004` |

Validation commands:

- `uv run --no-sync python -m pytest -q tests/test_scenario_waymo.py tests/test_scenarionet_pipeline.py tests/test_scenario_catalog_build.py`
- `uv run --no-sync ruff check src/thesis_rl/scenarios/parallel.py src/thesis_rl/scenarios/waymo.py src/thesis_rl/scenarios/pg/loader.py src/thesis_rl/cli/scenarios/build_catalog.py src/thesis_rl/rulebook/v2/context/catalog_eligibility.py tests/test_scenario_catalog_build.py`
- `uv run --no-sync ruff format --check ...` on the same modified Python files
- `bash -n scripts/prepare_scenarionet_dataset.sh`
- `git diff --check`
- Representative catalog fixture smoke; full data-dependent pipeline remains
  separate and must not run concurrently with another dataset pipeline.

## 10. Milestones

- [x] M1 — Plan and mandatory test matrix approved.
- [x] M2 — Add shared bounded process map and refactor Rulebook evaluator;
      existing Rulebook tests pass.
- [x] M3 — Parallelize Waymo/PG loaders and add loader regressions.
- [x] M4 — Wire builder CLI/pipeline workers and Rich source progress.
- [x] M5 — Update docs/index, run validations, and reconcile requirements.

## 11. Progress And Findings Log

### 2026-07-16 — Initial analysis

- Confirmed phase `[3/9]` is serial in both source loaders.
- Confirmed final catalog/group/report writing is already parent-owned and should
  remain serial.
- Confirmed the repository has Rich and a bounded `spawn` process-pool pattern.
- Added the shared ordered process map, refactored Rulebook eligibility to use
  it, and parallelized Waymo/PG per-file catalog loading.
- Added separate CLI worker options and Rich progress tasks for Waymo and PG;
  the pipeline passes the already resolved `waymo.workers` and `pg.workers`.
- Focused loader, CLI, Waymo, and Rulebook tests pass.
- Next step: final validation review.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/scenarios/parallel.py` | Added | Shared bounded ordered process map |
| `src/thesis_rl/rulebook/v2/context/catalog_eligibility.py` | Modified | Reuse shared process map |
| `src/thesis_rl/scenarios/waymo.py` | Modified | Parallel per-file Waymo loading |
| `src/thesis_rl/scenarios/pg/loader.py` | Modified | Parallel per-file PG loading |
| `src/thesis_rl/cli/scenarios/build_catalog.py` | Modified | Workers argument and Rich source progress |
| `scripts/prepare_scenarionet_dataset.sh` | Modified | Pass Waymo/PG workers to stage 3 |
| `tests/test_scenario_catalog_build.py` | Added | Loader and CLI regressions |
| `docs/setup/scenarionet_waymo_conversion.md` | Modified | Stage 3 operational guidance |
| `docs/setup/validation_commands.md` | Modified | Stage 3 worker/progress guidance |
| `docs/project_index.md` | Modified | Register ExecPlan |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| Focused tests | `PASS` | 2026-07-16 | `16 passed` in Docker |
| Related regression tests | `PASS` | 2026-07-16 | `143 passed` for Rulebook v2, ScenarioNet, Waymo, and catalog builder |
| Ruff check | `PASS` | 2026-07-16 | Modified loader, CLI, helper, and test files clean |
| Ruff format check | `PASS` | 2026-07-16 | Passed after formatting modified files |
| Shell/diff checks | `PASS` | 2026-07-16 | `bash -n` and `git diff --check` passed |
| Pipeline YAML resolver | `PASS` | 2026-07-16 | Emits Waymo, PG, Rulebook, and check worker values as 16 |
| Catalog fixture smoke | `PASS` | 2026-07-16 | CLI wrote Parquet/groups/report and Rich output was captured |

## 15. Final Reconciliation

| Requirement / criterion | Result | Evidence and limitation |
|---|---|---|
| `REQ-SCB-001` / `AC-SCB-001` | `VERIFIED` | Waymo and PG loader regressions pass |
| `REQ-SCB-002` / `AC-SCB-002` | `VERIFIED` | Serial/parallel records and PG seed window are identical |
| `REQ-SCB-003` / `AC-SCB-003` | `VERIFIED` | Feature/quality/arm output equality covered by loader comparisons |
| `REQ-SCB-004` / `AC-SCB-004` | `VERIFIED` | CLI fixture confirms Rich stderr and JSON/artifacts |
| `REQ-SCB-005` / `AC-SCB-005` | `VERIFIED` | Positive validation and bounded helper behavior tested |

Resulting behavior: stage `[3/9]` loads Waymo and PG records in bounded spawned
workers using the configured worker counts, reports per-source progress through
Rich, and keeps all final artifact writes deterministic and parent-owned. No
scientific or artifact-schema deviation was introduced.
