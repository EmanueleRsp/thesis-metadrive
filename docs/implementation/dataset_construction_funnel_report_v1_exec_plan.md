# ExecPlan: dataset-construction funnel report

## 1. Metadata

- **Feature:** reproducible reconstruction of the ScenarioNet dataset-construction
  funnel as thesis-ready tables and figures
- **Plan ID:** `DFR`
- **Authoritative specifications** (read-only inputs; this plan changes none of
  them):
  - `docs/specifications/scenarionet_integration_v1.1_specification.md`
    (`SCENARIONET-INTEGRATION` v1.1, `APPROVED`) — §5, §6, §7, §12, §15, §16, §17
  - `docs/specifications/scenarionet_integration_v1.2_specification.md`
    (v1.2, `APPROVED`) — holdout-first policy, grouping, pool sizes
  - `docs/specifications/scenarionet_integration_v1.3_specification.md`
    (v1.3, `APPROVED`) — panel cardinalities and frozen subsets
  - `docs/specifications/driving_mission_v1.1_specification.md`
    (`DRIVING-MISSION-V1.1`, `APPROVED`) — the offline route stage
- **Status:** `IMPLEMENTED`
- **Created:** 2026-09-05
- **Last updated:** 2026-09-05
- **Branch:** `dataset-funnel-report` (git worktree
  `.claude/worktrees/dataset-funnel-report`)
- **Related ADRs:** ADR-001, ADR-012, ADR-031, ADR-037, ADR-038, ADR-040
- **Owner:** thesis repository maintainer

## 2. Objective And Scope

The thesis chapter on dataset design needs one auditable account of how 66,854
converted scenarios became the 3,500 frozen records. The pipeline never wrote
that account in a single place: every stage left its own artifact, with
different conventions, and one of them — `catalog/split_report.json` — is a
**stale `SCENARIONET-INTEGRATION` v1.1 report** that still claims 1,000 training
records per source against the frozen v1.2 index's 1,100. Publishing from it
would put wrong numbers in the thesis.

**In scope:** a read-only reporter that reconstructs the funnel from the
immutable pipeline artifacts, emits the tables and vector figures the chapter
needs, and records the sha256 of every input it consumed.

**Out of scope:** rebuilding the dataset, changing any selection or eligibility
rule, editing the frozen index, and writing the chapter prose itself. The
reporter is descriptive; it introduces no dataset semantics of its own.

Success is recognized when the five stage counts are reproduced from artifacts
alone, cross-check against `arm_report.json` (3,500), and every published number
is traceable to a hashed input.

## 3. Authoritative Requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-DFR-001` | Reconstruct the funnel S0–S4, each stage attributed to the exact artifact that establishes it | v1.1 §17, v1.2 §3.5 |
| `REQ-DFR-002` | Never read `catalog/split_report.json`; it is a stale v1.1 artifact | Repository evidence, §4 |
| `REQ-DFR-003` | Count exclusion causes **per record**, not per message | v1.1 §17.4 |
| `REQ-DFR-004` | Publish the frozen selection by split, holdout pool, source and arm | v1.2 §3.3, ADR-037 |
| `REQ-DFR-005` | Publish the arm thresholds actually in force, unmodified | v1.1 §15, ADR-031 |
| `REQ-DFR-006` | Publish the declared-versus-observed PG profile mixture | v1.2 `DEC-003` |
| `REQ-DFR-007` | Publish the fifteen panel manifests with size, seed, scope, parent and hash | v1.3 |
| `REQ-DFR-008` | Record the sha256 and path of every consumed artifact, plus the frozen `selection_hash` | v1.1 §3, ADR-001 |
| `REQ-DFR-009` | Report a stage as `NOT_COMPUTED` rather than estimate it when its input is unreadable | AGENTS.md, testing policy |
| `REQ-DFR-010` | Add no project dependency | AGENTS.md |

## 4. Current Repository Analysis

- `VERIFIED` — `scripts/audit_frozen_scenarionet.py` is the precedent for a
  standalone, stdlib-only, read-only reporting script writing CSV/JSON to an
  explicit output directory. This plan follows it.
- `VERIFIED` — the stage artifacts and their semantics:
  - `catalog/catalog_report.json` — converted pool aggregates (`by_source`).
  - `rulebook_v2/catalog_eligibility.json` — per-record `rulebook_eligible`.
    Its header keys are **misleading**: `eligible_records` (16,159) is the count
    after the *driving-mission* stage, while `counts_by_source[*].eligible`
    (20,733) is the Rulebook stage. Established by reading
    `src/thesis_rl/cli/scenarios/filter_rulebook_v2_catalog.py:274-330`, where
    `selected` is the mission-filtered tuple and `source_counts` is computed
    from `eligibility`.
  - `rulebook_v2/driving_mission_eligibility.json` — the mission stage in
    isolation (20,733 in, 16,159 out).
  - `catalog/scenario_catalog_rulebook_v2.parquet` — the split builder's input.
  - `frozen/scenario_selection_index.json` — the 3,500 selected records.
- `VERIFIED` — `src/thesis_rl/cli/scenarios/build_splits_v1_2.py:239-247`
  restricts the candidate pool to `validation_status in {valid, warning}` and
  `rulebook_eligible is True`, and calls `assign_holdout_first_splits` **without**
  `allowed_signal_reliabilities`. So v1.2 applies no signal-reliability filter,
  unlike v1.1; 352 `partial` records are in the frozen selection.
- `VERIFIED` — `src/thesis_rl/scenarios/quality.py:53-62`
  (`mark_hard_quality_failures`) sets `validation_status = "invalid"` exactly
  when an offline quality filter fires, so an invalid record's
  `validation_warnings` are its rejection reasons. This is what makes the S3
  cause table meaningful.
- `VERIFIED` — `catalog/split_report.json` (2026-08-04 18:46 local) predates the
  v1.2 rebuild (23:21) and reports v1.1 split counts. Its `population_counts`
  happen to agree with the recomputation (16,159 / 10,763), but its `counts`
  do not, so the whole file is excluded as a source.
- `VERIFIED` — `pyarrow` is present in the `dataset-pipeline` image and absent
  from the host interpreter, so the S3 stage must degrade explicitly.

## 5. Assumptions And Invariants

- The reporter opens **no** `ScenarioDescription` file and writes **only** into
  its `--output-dir`. It never mutates the dataset.
- Units are preserved verbatim from the artifacts: `length` in control steps at
  `Δt = 0.1 s`, `route_length_m` in metres, agent counts as the q90 temporal
  quantile within the 50 m relevance radius (v1.1 §12–§15).
- A record may trigger several exclusion categories, so the per-category record
  counts sum to at least the excluded-record count. The report states this.
- Percentiles are linear-interpolated on the sorted sample; `n`, `min` and `max`
  are exact.
- The frozen index is the single authority for the selection; `arm_report.json`
  is used only as an independent cross-check of the 3,500 total.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-DFR-001` | Tooling | How to render figures without `matplotlib`, which is not a project dependency | (A) request approval to add `matplotlib`; (B) emit SVG from the standard library | **B** | No dependency added; vector output usable directly in the thesis | Decided, internal |
| `DEC-DFR-002` | Data policy | `catalog/split_report.json` disagrees with the frozen index | (A) publish it; (B) exclude it and recompute | **B** | Prevents publishing v1.1 numbers as v1.2 | Decided, internal |
| `DEC-DFR-003` | Reporting | The pipeline's `excluded_by_cause` counts messages | (A) republish as-is; (B) recount per record | **B** | Causes become interpretable as record counts | Decided, internal |
| `DEC-DFR-004` | Scope | The chapter also needs the S3 rejection reasons, which no artifact aggregates | (A) leave the 5,396-record drop unexplained; (B) aggregate from `validation_warnings` | **B** | Explains the largest unexplained drop in the funnel | Decided, internal |

No decision here changes observable dataset behavior, so none is an approval
gate. Nothing in this plan is a specification deviation.

## 7. Proposed Design

One standalone script, `scripts/build_dataset_funnel_report.py`:

1. resolve artifact paths from `--data-root` (default `SCENARIONET_DATA_ROOT`);
2. load the JSON artifacts and assemble `FunnelStage` records (S0, S1, S2, S4);
3. read the filtered catalog through `pyarrow` for S3, or mark it
   `NOT_COMPUTED`;
4. aggregate exclusion causes per record for S1, S2 and S3;
5. write seven CSV tables, five SVG figures, `funnel.md`, and `provenance.json`.

Pure helpers (`percentile`, `describe`, `error_category`,
`records_by_error_category`, `build_funnel`, `FunnelReport.rows`, the four
`figure_*` builders) are module-level and importable, which is what the test
module exercises.

## 8. Traceability

| Requirement | Implementation | Test |
|---|---|---|
| `REQ-DFR-001` | `build_funnel`, `FunnelReport.rows` | `TEST-DFR-005`, `TEST-DFR-007` |
| `REQ-DFR-002` | artifact list in `main`; `funnel.md` states the exclusion | Manual review of `provenance.json` inputs |
| `REQ-DFR-003` | `records_by_error_category` | `TEST-DFR-003`, `TEST-DFR-004` |
| `REQ-DFR-004` | `distribution` aggregation; `tables/split_source_arm.csv` | Cross-check against `arm_report.json` |
| `REQ-DFR-005` | `tables/arm_thresholds.csv` (verbatim copy) | Manual diff against `splits/arm_thresholds.json` |
| `REQ-DFR-006` | `tables/pg_profile_mixture.csv`, `figure_pg_mixture` | `TEST-DFR-008` |
| `REQ-DFR-007` | `tables/panels.csv` | Manual review (17 rows) |
| `REQ-DFR-008` | `provenance.json` | Manual review |
| `REQ-DFR-009` | `parquet_note` path | `TEST-DFR-006` |
| `REQ-DFR-010` | stdlib-only SVG emitters | `TEST-DFR-008`, `TEST-DFR-009` |

## 9. Test Strategy Defined Before Implementation

`tests/test_dataset_funnel_report.py`, nine tests, all deterministic and
offline (the module is loaded by path with `importlib`; no dataset is read):

- `TEST-DFR-001` percentile interpolation and endpoints, empty input raises;
- `TEST-DFR-002` `describe` skips `None`, returns `None` on an empty feature;
- `TEST-DFR-003` category is the prefix before the first colon;
- `TEST-DFR-004` causes counted per record, not per message;
- `TEST-DFR-005` five ordered stages with per-source counts;
- `TEST-DFR-006` an uncomputed stage stays empty rather than being interpolated;
- `TEST-DFR-007` retention percentages, stage-over-previous and over-converted;
- `TEST-DFR-008` all four figures parse as XML with an `svg` root;
- `TEST-DFR-009` figure text is XML-escaped.

## 10. Milestones

| ID | Milestone | Status |
|---|---|---|
| `M0` | Establish stage semantics from the pipeline source and reconcile the artifact header discrepancy | `DONE` |
| `M1` | Implement the reporter, tables and figures | `DONE` |
| `M2` | Add the S3 rejection-cause aggregation | `DONE` |
| `M3` | Tests, lint, format, and a real run against the frozen dataset | `DONE` |

## 11. Progress And Findings Log

- **2026-09-05, `M0`.** The 16,159-versus-20,733 discrepancy in
  `catalog_eligibility.json` is a **naming** problem in the artifact, not a data
  error: its header's `eligible_records` is post-mission and its
  `counts_by_source` is post-Rulebook. Resolved by reading the writer.
- **2026-09-05, `M0`.** `catalog/split_report.json` is stale and excluded
  (`DEC-DFR-002`).
- **2026-09-05, `M2`.** The largest single unexplained drop in the funnel is
  S2→S3: 5,396 Waymo records (12,030 → 6,634). Cause: offline hard quality
  filters — 3,056 "too many dynamic objects", 2,837 "degenerate SDC route",
  145 "possible overpass/elevation artifact" (categories overlap, so they sum
  above 5,396). PG loses one record in the whole stage.
- **2026-09-05, `M3`.** Measured funnel: 66,854 → 20,733 → 16,159 → 10,763 →
  3,500 (31.0 %, 77.9 %, 66.6 %, 32.5 % stage retention; 5.2 % overall).
  Independent cross-check: `arm_report.json` total = 3,500.

## 12. Deviations

None. No specification, dataset artifact, or production module was modified.

## 13. Files

| Path | Change |
|---|---|
| `scripts/build_dataset_funnel_report.py` | New. Read-only reporter. |
| `tests/test_dataset_funnel_report.py` | New. Nine tests. |
| `docs/audits/dataset_construction_2026-09-05/funnel.md` | New. Generated report. |
| `docs/audits/dataset_construction_2026-09-05/provenance.json` | New. Input hashes. |
| `docs/audits/dataset_construction_2026-09-05/tables/*.csv` | New. Eight generated tables. |
| `docs/audits/dataset_construction_2026-09-05/figures/*.svg` | New. Five generated figures. |
| `docs/implementation/dataset_construction_funnel_report_v1_exec_plan.md` | New. This plan. |

## 14. Validation Results

All commands run from the worktree root with
`COMPOSE_PROJECT_NAME=thesis-metadrive` so the existing image is reused (the
worktree has no populated `third_party/` submodules and must not trigger an
image rebuild).

| Check | Command | Result | Date |
|---|---|---|---|
| Unit tests | `docker compose run --rm -T dataset-pipeline uv run --no-sync python -m pytest -q tests/test_dataset_funnel_report.py` | **PASS**, 9 passed | 2026-09-05 |
| Lint | `... ruff check scripts/build_dataset_funnel_report.py tests/test_dataset_funnel_report.py` | **PASS**, all checks passed | 2026-09-05 |
| Format | `... ruff format` on the same two files | **PASS**, 2 files reformatted then clean | 2026-09-05 |
| Real run | `... python scripts/build_dataset_funnel_report.py --data-root /workspace/data/scenarionet --output-dir .../docs/audits/dataset_construction_2026-09-05 --overwrite` | **PASS**, five stages reported, cross-check 3,500 | 2026-09-05 |

Not run, with reason:

- Full `python -m pytest -q`: out of scope for a new, isolated reporting module
  that imports nothing from `src/thesis_rl`; the repository suite was not
  touched. Follow-up command if wanted:
  `COMPOSE_PROJECT_NAME=thesis-metadrive docker compose run --rm -T dataset-pipeline uv run --no-sync python -m pytest -q`.
- `make smoke`: no runtime, environment, reward or observation code was changed,
  so the training path cannot be affected.

## 15. Final Reconciliation

Every requirement `REQ-DFR-001`..`REQ-DFR-010` is implemented and covered by a
test or an explicit manual check in §8. The generated report reproduces the
frozen dataset's own totals and adds no number that is not derived from a hashed
artifact. Known limitations, all declared in the generated `funnel.md` or here:

- exclusion categories overlap within a stage, so per-category record counts sum
  to at least the excluded-record count;
- S3 requires `pyarrow`, so a host-only run reports that stage as
  `NOT_COMPUTED`;
- the reporter describes the dataset that exists; it does not re-derive the arm
  thresholds, whose heuristic provenance remains recorded in ADR-031.
