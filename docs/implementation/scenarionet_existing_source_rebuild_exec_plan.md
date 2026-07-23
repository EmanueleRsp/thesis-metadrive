# ScenarioNet Existing-Source Rebuild ExecPlan

## 1. Metadata

- Feature and plan ID: `SCENARIONET-REBUILD-EXISTING`
- Authoritative specification: `docs/specifications/scenarionet_integration_v1.1_specification.md`, version `1.1`, `APPROVED`
- Status: `IMPLEMENTED`
- Created / last updated: 2026-07-20
- Owner: thesis repository maintainer

## 2. Objective And Scope

Provide one Make target that revalidates and rebuilds a ScenarioNet dataset from
already materialized PG/Waymo source files, without discovery, acquisition, or
source generation. Preserve visible Rich progress output by retaining Docker
TTY allocation and allow the caller to choose the Rulebook worker count.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| REQ-REBUILD-001 | Re-evaluate Rulebook eligibility before split selection. | ScenarioNet v1.1 §§6, 17 |
| REQ-REBUILD-002 | Rebuild balanced source/split selections and runtime views from existing sources. | ScenarioNet v1.1 §§6, 24 |
| REQ-REBUILD-003 | Recreate the frozen selection index after successful rebuild. | ScenarioNet v1.1 §27 |
| REQ-REBUILD-004 | Do not perform discovery, acquisition, or source generation. | ScenarioNet v1.1 §27 |
| REQ-REBUILD-005 | Keep progress visible and expose an optional worker override. | User request, 2026-07-20 |

## 4. Current Repository Analysis

- `VERIFIED`: `scenarionet-from-frozen` trusts recorded frozen eligibility and
  is insufficient for stale-eligibility recovery.
- `VERIFIED`: `filter_rulebook_v2_catalog` supports `--no-incremental` and
  performs full static revalidation.
- `VERIFIED`: the canonical derived stages are `build_splits`,
  `compute_arm_thresholds`, `build_runtime_databases`, and `freeze_dataset`.
- `VERIFIED`: omitting Docker `-T` preserves interactive Rich progress output.

## 5. Decisions

| ID | Category | Decision | Status |
|---|---|---|---|
| DEC-REBUILD-001 | implementation detail | Use `SCENARIONET_REBUILD_WORKERS` as an optional Make override; omit `--workers` when unset so the CLI default applies. | Approved by user request |
| DEC-REBUILD-002 | implementation detail | Preserve the verified dataset targets: Waymo/PG `1000/250/500`, split seed `0`, balanced-arm selection. | Approved by user request |

## 6. Traceability

| Requirement | Acceptance criteria | Implementation | Validation |
|---|---|---|---|
| REQ-REBUILD-001 | Full no-cache filtering is invoked. | `Makefile::scenarionet-rebuild-existing` | Make dry-run |
| REQ-REBUILD-002 | Split, threshold, and runtime stages are chained. | Same target | Existing pipeline/runtime checks |
| REQ-REBUILD-003 | Frozen index is recreated last. | Same target | Frozen replay verification |
| REQ-REBUILD-004 | No acquisition/discovery/generation command appears. | Same target | Make dry-run inspection |
| REQ-REBUILD-005 | No `-T`; worker override is supported. | Same target | Dry-run with and without workers |

## 7. Validation Results

| Command | Result | Date | Notes |
|---|---|---|---|
| `make -n scenarionet-rebuild-existing` | PASS | 2026-07-20 | Emits no `-T`, uses `--no-incremental`, and chains all five rebuild stages; host/container existence checks were subsequently corrected to run inside the dataset-pipeline mount. |
| `make -n scenarionet-rebuild-existing SCENARIONET_REBUILD_WORKERS=64` | PASS | 2026-07-20 | Emits `--workers "64"` only for the full Rulebook revalidation stage |
| `git diff --check` | PASS | 2026-07-20 | No whitespace errors |

## 8. Deviations

No deviations identified.

## 9. Files

| Path | Action | Purpose |
|---|---|---|
| `Makefile` | Modified | Add existing-source revalidation/rebuild target |
| `docs/implementation/scenarionet_existing_source_rebuild_exec_plan.md` | Added | Record behavior, decisions, and validation |

## 10. Final Reconciliation

The target is implemented pending dry-run and whitespace validation. It does
not replace the frozen index as the selection authority; it regenerates the
derived selection and then creates a new frozen index.
