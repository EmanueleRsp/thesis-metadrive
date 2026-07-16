# Rulebook v2 Catalog Filter Parallelization ExecPlan

## 1. Metadata

- Feature and plan ID: Rulebook v2 catalog filter parallelization, `RBCF-PAR-001`
- Authoritative specifications:
  - `docs/specifications/rulebook_v2_spec.md`, version `4.6-final-implementation-complete`, `AUTHORITATIVE`
  - `docs/specifications/scenarionet_integration_spec_v1.md`, version `v1`, `AUTHORITATIVE`
- Status: `IMPLEMENTED`
- Created: 2026-07-16
- Last updated: 2026-07-16
- Branch: current working branch
- Related ADRs: none; this is an implementation-level performance and observability change
- Owner: repository maintainer

## 2. Objective And Scope

The Rulebook v2 static catalog filter currently evaluates every catalog entry
sequentially and emits no progress until the complete artifact is ready. Add
bounded process parallelism and a parent-owned Rich progress dashboard so large
catalog runs use available CPU and expose completion, rate, elapsed time,
remaining time, worker count, and final eligibility counts.

In scope:

- configurable process worker count for the catalog eligibility evaluator and
  the ScenarioNet pipeline;
- bounded task submission to avoid an unbounded future queue;
- deterministic result ordering and unchanged eligibility semantics/artifact
  schema;
- Rich progress and summary output owned by the CLI parent process;
- focused tests, configuration wiring, and operational documentation.

Out of scope:

- changes to Rulebook formulas, thresholds, geometry, map matching, source
  policy, split policy, or eligibility decisions;
- changing artifact schemas or adding a new runtime dependency;
- distributed execution across machines or GPU acceleration;
- resuming an interrupted filter from partial eligibility records.

Success is recognized when a multi-entry run produces the same ordered
eligibility results and filtered catalog as the sequential path, while the CLI
reports Rich progress and accepts a validated worker count.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-RBCF-001` | Static eligibility remains offline, fail-closed, and based on the canonical task route and source adapters. | Rulebook v2 §15.11; §7.6.1; ScenarioNet v1 §17 |
| `REQ-RBCF-002` | Catalog processing remains deterministic and does not mutate the catalog policy or split inputs. | ScenarioNet v1 §22.3; §23.1; §24 |
| `REQ-RBCF-003` | Eligibility and exclusion causes remain auditable for every catalog record. | Rulebook v2 §7.6.1; §15.11; ScenarioNet v1 §24 |
| `REQ-RBCF-004` | The pipeline exposes operational progress and preserves machine-readable artifact output. | ScenarioNet v1 §24; existing repository CLI/output convention |
| `REQ-RBCF-005` | Parallel execution is bounded and configurable without requiring a new dependency. | User request 2026-07-16; verified repository convention |

## 4. Current Repository Analysis

Verified findings before implementation:

- `src/thesis_rl/rulebook/v2/context/catalog_eligibility.py::evaluate_catalog_entries`
  sorts entries by `scenario_uid` and evaluates them sequentially. Each entry
  loads one pickle and runs a CPU-heavy static adapter/map-matching path.
- `src/thesis_rl/cli/scenarios/filter_rulebook_v2_catalog.py::main` waits for
  the complete tuple before writing `scenario_catalog_rulebook_v2.parquet` and
  `catalog_eligibility.json`; this explains the absence of intermediate output.
- `scripts/prepare_scenarionet_dataset.sh` invokes the CLI at stage `[4/9]` but
  does not currently pass a Rulebook worker setting.
- `src/thesis_rl/cli/scenarios/pipeline_config.py` resolves versioned YAML
  values into environment variables; `conf/scenarios/pipeline_v1.yaml` is the
  scientific/data-policy source of truth.
- `src/thesis_rl/scenarios/pg/report.py` already uses
  `ProcessPoolExecutor`, `spawn`, and a parent progress callback for bounded
  process-based work. This is the repository precedent.
- `rich` is already declared in `pyproject.toml`; no dependency addition is
  required.
- Existing behavior catches per-scenario adapter/load failures and retains a
  typed audit result. The parallel path must preserve that behavior.

Classification of findings:

- `SPECIFIED`: offline/fail-closed validation, deterministic catalog/split
  behavior, audit artifact, and no future-track runtime fallback.
- `VERIFIED`: current sequential call graph, output timing, existing Rich
  dependency, and process-pool convention.
- `INFERRED`: process-based parallelism is appropriate because the workload is
  CPU-bound and uses Python/Shapely objects; this does not change scientific
  behavior and is recorded as an implementation choice below.

## 5. Assumptions And Invariants

- `workers` is a positive integer. The executor uses at most
  `min(workers, catalog_size)` processes.
- The parent process is the only Rich console writer. Worker stdout/stderr is
  not used for progress reporting, preventing interleaved dashboards.
- Entries are sorted by `scenario_uid` before dispatch, and results are stored
  by original sorted index. Completion order is not artifact order.
- The bounded in-flight queue is limited to a small multiple of the active
  worker count; it is an operational memory bound, not a sampling policy.
- `workers=1` remains a deterministic sequential execution path and is used by
  tests as the reference behavior.
- Eligibility hashes, route geometry, coordinate frames, formulas, thresholds,
  and error semantics remain unchanged.
- Human progress is written to stderr through Rich; JSON and final CLI result
  output remain machine-readable on stdout where the current CLI emits it.
- An interrupted run does not produce a resumable partial artifact. Existing
  atomic catalog writing and end-of-run eligibility JSON writing remain in
  effect.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-RBCF-001` | implementation detail | How to parallelize CPU-bound static adapter work | threads / process pool / external distributed runner | `ProcessPoolExecutor` with `spawn`, matching PG generation | Uses multiple CPU cores while isolating pickle/geometry workloads; no scientific effect | Accepted by user request and repository convention |
| `DEC-RBCF-002` | implementation detail | How to preserve deterministic artifacts while showing completion-order progress | ordered blocking / completion-order collection with indexed results | Completion-order progress plus index-based result reconstruction | Faster feedback without changing catalog order or audit records | Accepted by user request |
| `DEC-RBCF-003` | implementation detail | Default worker count | sequential default / fixed conservative default / host CPU count | Fixed YAML default of 8, overridable in the versioned pipeline config | Reproducible operational behavior and explicit resource control | Accepted by user request; no scientific impact |

No unresolved approval gate remains. The selected choices do not alter the
authoritative scientific contracts.

## 7. Proposed Design

### 7.1 Evaluation API

Extend `evaluate_catalog_entries` with keyword-only `workers: int = 1` and an
optional progress callback carrying `(completed, total)`. Keep the existing
sequential behavior for `workers=1`.

For `workers > 1`, use a top-level picklable worker function that calls the
existing `evaluate_catalog_entry`. Submit only a bounded number of tasks,
collect completed futures as they finish, update the callback, and place each
result at its sorted input index. A worker exception that is not already
converted into an eligibility result propagates as before; ordinary per-entry
load/adapter failures remain fail-closed results.

### 7.2 CLI dashboard

Add `--workers` to `filter_rulebook_v2_catalog`, validate it as positive, and
construct a Rich `Progress` with total entries, completed count, rate, elapsed
time, and ETA. Emit start/end summaries with worker count, eligible and excluded
counts. The CLI passes the callback to the evaluator and keeps artifact writes
after all results are available.

### 7.3 Pipeline configuration

Add `rulebook_v2.workers: 8` to `conf/scenarios/pipeline_v1.yaml`, resolve it as
`SCENARIONET_RULEBOOK_V2_WORKERS`, and pass it from
`scripts/prepare_scenarionet_dataset.sh` to stage 4. The standalone Make target
uses the CLI default unless a future explicit override is added; it remains
compatible with `--workers`.

### 7.4 Compatibility and resource behavior

The filtered Parquet and JSON schemas are unchanged. The only expected
operational change is lower wall-clock time on machines with available CPU and
additional stderr progress output. Users can reduce workers when memory or
thermal limits require it.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-RBCF-001` | `AC-RBCF-001` same eligibility semantics and fail-closed errors | `catalog_eligibility.py` | existing per-entry tests; CLI regression | Implemented and verified by focused tests |
| `REQ-RBCF-002` | `AC-RBCF-002` sequential and parallel results have identical ordered UIDs/results | `catalog_eligibility.py` | parallel determinism test | Implemented and verified by focused tests |
| `REQ-RBCF-003` | `AC-RBCF-003` audit artifact contains all records and counts | filter CLI | existing artifact test; worker-count CLI test | Implemented and verified by focused tests |
| `REQ-RBCF-004` | `AC-RBCF-004` Rich progress callback/dashboard runs with total and completion | filter CLI | callback/dashboard-focused CLI test | Implemented and verified by focused tests |
| `REQ-RBCF-005` | `AC-RBCF-005` positive worker validation and bounded executor configuration | evaluator and CLI | invalid workers tests | Implemented and verified by focused tests |

## 9. Test Strategy Defined Before Implementation

Acceptance criteria:

- `AC-RBCF-001`: Existing eligible and unmappable fixtures retain exactly the
  same `rulebook_eligible` and `validation_errors` values.
- `AC-RBCF-002`: For a deterministic multi-entry fixture, `workers=1` and
  `workers=2` return identical results in sorted UID order, including mixed
  eligible/excluded entries.
- `AC-RBCF-003`: CLI output artifacts retain all eligibility records and exact
  eligible/excluded counts when workers are enabled.
- `AC-RBCF-004`: The evaluator invokes the progress callback once per completed
  entry with monotonic completion values ending at the catalog total; CLI Rich
  mode renders without worker output interleaving.
- `AC-RBCF-005`: `workers <= 0` is rejected by both evaluator and CLI argument
  validation; executor worker count is capped by catalog size.

Mandatory test matrix:

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-RBCF-001` | Unit | Existing eligible entry | current PG fixture | Eligible, no errors | `REQ-RBCF-001` |
| `TEST-RBCF-002` | Unit/regression | Existing unmappable entry | current off-lane fixture | Excluded with exact typed error | `REQ-RBCF-001` |
| `TEST-RBCF-003` | Integration | Sequential vs process pool | two or more deterministic PG entries | Identical ordered results | `REQ-RBCF-002` |
| `TEST-RBCF-004` | Unit | Progress callback | deterministic small catalog | `1..N` monotonic callbacks, final `N` | `REQ-RBCF-004` |
| `TEST-RBCF-005` | Unit | Invalid worker count | `0`, negative values | `ValueError`/CLI parser error | `REQ-RBCF-005` |
| `TEST-RBCF-006` | Integration | CLI artifacts with workers | small catalog, `--workers 2` | Same artifact counts and complete records | `REQ-RBCF-003` |

Validation commands defined before implementation:

- `uv run --no-sync python -m pytest -q tests/test_rulebook_v2_catalog_eligibility.py`
- `uv run --no-sync ruff check src/thesis_rl/rulebook/v2/context/catalog_eligibility.py src/thesis_rl/cli/scenarios/filter_rulebook_v2_catalog.py tests/test_rulebook_v2_catalog_eligibility.py`
- `uv run --no-sync ruff format --check src/thesis_rl/rulebook/v2/context/catalog_eligibility.py src/thesis_rl/cli/scenarios/filter_rulebook_v2_catalog.py tests/test_rulebook_v2_catalog_eligibility.py`
- `git diff --check`
- Representative data-dependent smoke: `make rulebook-v2-filter-catalog` only
  if the active dataset run is stopped or finished and the user authorizes a
  full artifact regeneration; otherwise it remains not run to avoid competing
  with the live pipeline.

## 10. Milestones

- [x] M1 — Plan and test matrix approved; no production changes yet.
- [x] M2 — Add bounded evaluator parallelism and progress callback; focused unit
      tests pass.
- [x] M3 — Add Rich CLI dashboard and YAML/pipeline worker wiring; CLI tests and
      lint pass.
- [x] M4 — Update operational documentation and project index; diff reviewed.
- [x] M5 — Run focused checks and reconcile requirements/results. The full
      data-dependent smoke remains deferred because the prior live pipeline run
      is still active.

## 11. Progress And Findings Log

### 2026-07-16 — Initial analysis

- Verified the stage-4 command evaluates a 13,077-entry catalog sequentially.
- Verified the active run used one Python process at 100% CPU and emitted no
  output artifacts until completion.
- Verified `rich` is already a project dependency and PG generation provides a
  process-pool/progress precedent.
- Implemented a `spawn` process pool with at most `2 * workers` in-flight tasks,
  indexed result reconstruction, a `--workers` CLI option, and a parent Rich
  dashboard.
- Focused test suite passes after adding eligible/excluded parallel fixtures.
- Next step: complete documentation reconciliation and broader validation.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/rulebook/v2/context/catalog_eligibility.py` | Modified | Bounded process pool and deterministic result collection |
| `src/thesis_rl/cli/scenarios/filter_rulebook_v2_catalog.py` | Modified | Rich dashboard, worker argument, and summaries |
| `src/thesis_rl/cli/scenarios/pipeline_config.py` | Modified | Resolve Rulebook worker configuration |
| `conf/scenarios/pipeline_v1.yaml` | Modified | Versioned default worker count |
| `scripts/prepare_scenarionet_dataset.sh` | Modified | Pass workers to stage 4 |
| `tests/test_rulebook_v2_catalog_eligibility.py` | Modified | Parallelism, determinism, progress, and validation tests |
| `docs/setup/scenarionet_waymo_conversion.md` | Modified | Operational worker/progress guidance |
| `docs/setup/validation_commands.md` | Modified | CLI worker/progress guidance |
| `docs/project_index.md` | Modified | Register the new ExecPlan |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| Focused pytest | `PASS` | 2026-07-16 | `6 passed` in Docker |
| Related regression pytest | `PASS` | 2026-07-16 | `133 passed` for Rulebook v2 and ScenarioNet tests in Docker |
| Focused Ruff check | `PASS` | 2026-07-16 | Modified source/test files clean |
| Focused Ruff format check | `PASS` | 2026-07-16 | Passed after formatting two modified files |
| `git diff --check` | `PASS` | 2026-07-16 | No whitespace errors |
| `bash -n scripts/prepare_scenarionet_dataset.sh` | `PASS` | 2026-07-16 | Pipeline script syntax valid |
| Pipeline YAML resolver | `PASS` | 2026-07-16 | Emits `SCENARIONET_RULEBOOK_V2_WORKERS 8` in Docker |
| Full data-dependent smoke | `NOT_RUN` | 2026-07-16 | Avoid competing with the active dataset pipeline |

## 15. Final Reconciliation

| Requirement / criterion | Result | Evidence and limitation |
|---|---|---|
| `REQ-RBCF-001` / `AC-RBCF-001` | `VERIFIED` | Existing eligible/excluded semantics pass unchanged |
| `REQ-RBCF-002` / `AC-RBCF-002` | `VERIFIED` | Sequential and spawned results are equal and UID-ordered |
| `REQ-RBCF-003` / `AC-RBCF-003` | `VERIFIED` | CLI artifact test passes with two workers |
| `REQ-RBCF-004` / `AC-RBCF-004` | `VERIFIED` | Rich start/completion logs and progress callback are tested |
| `REQ-RBCF-005` / `AC-RBCF-005` | `VERIFIED` | Positive validation and bounded in-flight process pool implemented |
| Representative full-catalog smoke | `PARTIAL` | Not run to avoid competing with the existing live dataset run |

Resulting behavior: the full ScenarioNet pipeline now defaults to eight spawned
workers for Rulebook v2 static eligibility, keeps at most twice that number of
tasks in flight, reports Rich progress from the parent process, and reconstructs
artifacts deterministically. No scientific or artifact-schema deviation was
introduced.

Known limitation to preserve: an interrupted run cannot resume from partial
eligibility results; output artifacts are still committed only after complete
evaluation.
