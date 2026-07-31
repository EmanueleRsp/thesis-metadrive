# ExecPlan: ScenarioNet multi-panel evaluation, full-pool protocol, and terminal monitor

## 1. Metadata

- Plan ID: `SN-MULTIPANEL-EVAL-001`
- Authoritative specifications: `docs/specifications/scenarionet_integration_v1.3_specification.md` (`SCENARIONET-INTEGRATION` v1.3, APPROVED) and `docs/specifications/evaluation_protocol_v1.2_specification.md` (`EVAL-PROTOCOL` v1.2, APPROVED).
- Status: `IN_PROGRESS`
- Created / last updated: 2026-07-31
- Branch: `scenarionet-implementation`
- Related ADR: `ADR-042`

## 2. Objective and scope

Implement five named ScenarioNet evaluation endpoints across frozen data,
Hydra resolution, all training/evaluation paths, artifacts, analysis, and a
single Rich terminal monitor. Full profiles consume complete frozen pools;
`smoke` and `fast` consume frozen deterministic subsets. Checkpoint semantics,
legacy non-ScenarioNet scalar evaluation, deterministic inference, and all
existing metric families are preserved.

## 3. Authoritative requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-MP-001` | Freeze five complete pools and ten deterministic diagnostic child manifests. | SN v1.3 |
| `REQ-MP-002` | Resolve official ScenarioNet size from frozen plan, not scalar overrides. | EVAL v1.2 REQ-004 |
| `REQ-MP-003` | Admit both validation panels at every profile boundary from one snapshot with FIFO backpressure. | EVAL v1.2 REQ-007/013 |
| `REQ-MP-004` | Execute all final panels serially from `final.zip`. | EVAL v1.2 REQ-006 |
| `REQ-MP-005` | Preserve panel-labelled artifacts and show panel-aware Rich status. | EVAL v1.2 terminal monitor |
| `REQ-MP-006` | Analyse each panel/scope separately and reject incompatible comparison inputs. | EVAL v1.2 REQ-011 |

## 4. Current repository analysis

- `src/thesis_rl/scenarios/frozen.py` already embeds five canonical panel
  manifests, but the existing payload has old partial cardinalities and no
  profile subsets. **VERIFIED**.
- `src/thesis_rl/runtime/async_evaluation.py` has a single FIFO job queue and
  independently snapshots each enqueue. **VERIFIED**.
- `src/thesis_rl/runtime/loops/train_loop.py`, staged paths, and
  `curriculum/scenario_acl/driver.py` invoke single-panel evaluation and use
  scalar episode counts. **VERIFIED**.
- `runtime/io/csv_recorder.py` and analysis carry `scenario_set`, but final
  aggregation assumes one final row. **VERIFIED**.
- The repository already depends on Rich; no dependency is added. **VERIFIED**.

## 5. Invariants

- The frozen data-selection hash and panel/subset identity are distinct.
- `smoke`/`fast` sampling seeds are 20260731/20260732 and never use learner
  seed. Child UID order is frozen and replayed exactly.
- Both validation jobs reference one immutable snapshot and global step.
- Failure is fatal; queue capacity is one batch; no evaluation boundary drops.
- `final.zip` remains the only final policy source.

## 6. Decisions and approval gates

| ID | Category | Decision | Status |
|---|---|---|---|
| `DEC-MP-001` | specification clarification | Full profiles use all 150/150/400/300/300 endpoints; smoke/fast use frozen diagnostic subsets. | Approved by ADR-042 |
| `DEC-MP-002` | specification clarification | Both validation panels use every existing profile interval and one snapshot. | Approved by ADR-042 |
| `DEC-MP-003` | implementation detail | One Rich Live renderer and additive CSV columns; no new dependency. | Approved by user plan |

## 7. Proposed design

`scenarios/panel_manifest.py` and `frozen.py` build, embed, restore, and
validate canonical and child manifests. A panel-plan resolver loads the frozen
index and validates the profile-specific expected set. Runtime passes a shared
snapshot and batch identity into panel jobs. CSV/event/artifact paths carry
panel, scope, hashes, and batch identity. Analysis treats final CSV as a set
of panel rows and validates identity before grouping.

## 8. Traceability

| Requirement | Acceptance criterion | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-MP-001` | Complete/child manifests are deterministic and tamper-evident. | `frozen.py`, `panel_manifest.py` | frozen/panel tests | Planned |
| `REQ-MP-002` | Profile resolution ignores ScenarioNet scalar count overrides. | panel plan resolver/config | config tests | Planned |
| `REQ-MP-003` | Shared snapshot; FIFO backpressure; no lost batch. | async manager/training adapters | async runtime tests | Planned |
| `REQ-MP-004` | Three ordered `final.zip` results. | final evaluator | final runtime tests | Planned |
| `REQ-MP-005` | All artifacts are panel scoped; UI renders µ±σ. | IO/console | UI/artifact tests | Planned |
| `REQ-MP-006` | Multi-row finals and compatible scope/hash aggregation only. | analysis | aggregate tests | Planned |

## 9. Mandatory test strategy

| ID | Level | Expected behavior | Requirement |
|---|---|---|---|
| `TEST-MP-001` | Unit | Full pool/diagnostic sizes, fixed seeds, replay, and tamper rejection. | `REQ-MP-001` |
| `TEST-MP-002` | Config | Every profile resolves the exact five panels and ignores obsolete scalars. | `REQ-MP-002` |
| `TEST-MP-003` | Runtime | Two panels share one snapshot; FIFO backpressure is lossless and failures are fatal. | `REQ-MP-003` |
| `TEST-MP-004` | Runtime | Exactly three serial final panels use `final.zip` and have isolated paths. | `REQ-MP-004` |
| `TEST-MP-005` | UI/IO | Rich states and three µ±σ metrics render without corrupting progress. | `REQ-MP-005` |
| `TEST-MP-006` | Analysis | Multiple final rows remain panel-separated; incompatible scope/hash is rejected. | `REQ-MP-006` |

Commands: focused `pytest`, focused Ruff lint/format, `git diff --check`,
`make config`, `make config-gpu`, `make test`, and `make smoke`.

## 10. Milestones

- [x] M0: approve amendments, ADR, and this ExecPlan.
- [x] M1: implement frozen complete/diagnostic manifest schema and resolver.
- [x] M2: implement shared runtime panel execution, artifacts, and Rich UI.
- [x] M3: implement analysis validation and panel-aware reports.
- [ ] M4: execute mandatory validation and reconcile requirements.

## 11. Progress and findings log

- 2026-07-31: user approved the policy and explicitly requested implementation.
  Existing index embeds five manifests but represents the superseded partial
  pools. The materialized dataset volume is not in this checkout; it must be
  regenerated with the new builder before an official run is admitted.
- 2026-07-31: implemented v1.3 manifest child generation/replay and a
  fail-closed resolver; changed the materialization target to
  150/150/400/300/300. Implemented shared-snapshot asynchronous batches,
  panel-labelled CSV fields, Rich panel state rendering, Scenario-ACL and
  baseline validation wiring, video panel namespacing, and aggregation guards.
- 2026-07-31: replaced the ScenarioNet branches of the duplicated synchronous
  final-evaluation paths with `runtime/final_panels.py`; standard training,
  Scenario-ACL (single and vectorized), and standalone evaluation now execute
  the three frozen final panels serially from `final.zip`.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/scenarios/{panel_manifest,frozen}.py` | Modify | frozen panel definitions |
| `conf/evaluation/` | Add | frozen plan configuration |
| `src/thesis_rl/runtime/` | Modify | panel jobs, artifacts, UI |
| `src/thesis_rl/runtime/final_panels.py` | Add | serial final-panel executor shared by every ScenarioNet entry point |
| `src/thesis_rl/analysis/` | Modify | multi-panel reporting |
| `tests/` | Modify/add | mandatory matrix |

## 14. Validation results

| Command | Result | Date | Notes |
|---|---|---|---|
| AST syntax checks and `git diff --check` | PASS | 2026-07-31 | 13 modified/new Python modules/tests parsed; whitespace clean. |
| Direct async-batch harness | PASS | 2026-07-31 | One saved snapshot was shared by two FIFO panel jobs; Rich headers include the three requested `μ ± σ` metrics. |
| `make config` and `make config-gpu` | PASS | 2026-07-31 | Compose configurations validate. |
| `make smoke` | NOT_RUN | 2026-07-31 | Docker daemon socket is unavailable to this sandbox (`permission denied`); rerun in the normal Docker-enabled development environment after regenerating the v1.3 frozen manifests. |
| Focused tests and Ruff | NOT_RUN | 2026-07-31 | The supplied `.venv` has neither pytest nor ruff; `uv run` cannot access its externally mounted cache in this sandbox. |
| `make config`, `make config-gpu`, `make test`, `make smoke` | NOT_RUN | 2026-07-31 | Pending implementation. |

## 15. Final reconciliation

Pending implementation and validation.
