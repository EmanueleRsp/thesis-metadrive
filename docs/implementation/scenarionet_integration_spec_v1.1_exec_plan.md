# ScenarioNet Integration v1.1 ExecPlan

## 1. Metadata

- Feature and plan ID: ScenarioNet integration, `SCENARIONET-INTEGRATION-V1.1`
- Authoritative specification: `docs/specifications/scenarionet_integration_spec_v1.1.md`, ID `SCENARIONET-INTEGRATION`, version `1.1`, `APPROVED`
- Status: `IN_PROGRESS`
- Created: `2026-07-16`
- Last updated: `2026-07-18`
- Branch: current working branch
- Related ADRs: `docs/decisions/ADR-001-scenarionet-v1-1-dataset-policy.md`, `docs/decisions/ADR-007-waymo-batch-throughput.md`, `docs/decisions/ADR-008-pg-compositional-replenishment.md`, `docs/decisions/ADR-009-pg-targeted-replenishment.md`, `docs/decisions/ADR-010-waymo-cap-expansion.md`
- Owner: thesis repository maintainer

## 2. Objective And Scope

Deliver a reproducible ScenarioNet v1.1 pipeline that builds, validates,
balances, freezes, and samples a unified Waymo `training_20s` plus offline
MetaDrive PG dataset through the existing thesis environment and reward
interfaces. Success requires exact source totals, near-uniform A0–A5 semantic
arm totals, Rulebook-eligible runtime pools, deterministic evaluation, and the
approved `scenario.length + 50` truncation contract.

In scope: catalog and manifest schema, final eligibility accounting, grouping,
acquisition and PG preparation, split selection, features and existing-arm
classification, runtime views, providers, environment reconciliation, logging,
tests, documentation, and artifact reconciliation.

Out of scope: a new semantic-observation contract, a new Rulebook formula,
custom PG VRU/signals/right-of-way generation, scenario mutation, and changes
to the historical seven-arm generator ACL mechanism.

Compatibility constraints: preserve the existing six semantic arm names and
formulas; keep historical v1 documents; do not add dependencies without an
explicit approval; use only relative catalog paths; preserve termination versus
truncation distinction.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-SN-001` | Use Waymo `training_20s`, offline PG, `ScenarioDescription`, configurable root, and a frozen software/data manifest. | §§2–4 |
| `REQ-SN-002` | Construct primary splits with source totals W/PG: 1000/1000 train, 250/250 validation, 500/500 test. | §5 |
| `REQ-SN-003` | Maintain candidate, eligible, and selected populations; select deterministic, no-leakage grouped splits. | §6.1–§6.6 |
| `REQ-SN-004` | Group Waymo by verified log/segment; otherwise use original scenario ID and retain TFRecord source only as provenance. | §6.1, ADR-001 |
| `REQ-SN-005` | Require validity, hard quality, allowed signal reliability, and `rulebook_eligible=true` in runtime pools. | §§6.5, 17.3–17.5 |
| `REQ-SN-006` | Acquire only unseen Waymo shards in deterministic batches of 64, stop at feasibility or 256 new shards, and report deficits. | §6.7, ADR-001, ADR-007, ADR-010 |
| `REQ-SN-007` | Generate and validate PG offline with native blocks, disjoint seeds, and pilot diagnostics. | §§9–11 |
| `REQ-SN-008` | Extract specified features, calculate train-only Q40/Q75, and retain the existing A0–A5 taxonomy unchanged. | §§12–16 |
| `REQ-SN-009` | Build runtime views and validate ScenarioNet plus thesis reset/rollout behavior. | §§4, 17 |
| `REQ-SN-010` | Apply the `ThesisScenarioEnv` contract: custom reward reuse, physical-out-of-road separation, and `length + 50` truncation. | §§18–21 |
| `REQ-SN-011` | Provide strict uniform, arm-uniform, ACL, and fixed evaluation providers; ACL semantic MAB uses the same six arms (`K=6`). | §22 |
| `REQ-SN-012` | Use process-based vectorization with deterministic per-worker seeds and reliable cleanup. | §23 |
| `REQ-SN-013` | Record episode/run source×arm statistics and all dataset population, target, selected, and deficit artifacts. | §24 |
| `REQ-SN-014` | Prevent leakage at the ScenarioNet pipeline boundary; defer the complete observation contract to its own specification. | §§2.1, 27.3, ADR-001 |
| `REQ-SN-015` | Supply reproducible CLIs, mandatory tests, smoke training, and final artifact reconciliation. | §§25–28 |
| `REQ-SN-016` | Make the orchestration command restartable after an interrupted stage while preserving explicit overwrite protection for source PG/Waymo data, bootstrap and validate required Rulebook calibration artifacts without synthetic values, and report failures with actionable stage context. | §25 |

## 4. Current Repository Analysis

| Statement | Status | Evidence and consequence |
|---|---|---|
| ScenarioNet and MetaDrive are pinned locally at `d4acdb5…` and `85e5dad…`. | VERIFIED | No arbitrary submodule update is permitted. |
| `src/thesis_rl/scenarios/arms.py` exposes the six preserved semantic arms and conflict-based formulas. | VERIFIED | v1.1 must reconcile, not rename, this taxonomy. |
| `src/thesis_rl/curriculum/scenario_acl/arms.py` uses the shared ScenarioNet arm list; `conf/curriculum/scenario_acl_scenarionet.yaml` sets `num_arms: 6`. | VERIFIED | `REQ-SN-011` is a reconciliation target. |
| `src/thesis_rl/scenarios/pipeline.py` has a greedy arm-balanced selector and separate validity/signal handling. | VERIFIED | It lacks the approved final population contract and must be replaced or bounded by strict validation. |
| `src/thesis_rl/scenarios/waymo.py` receives `metadata.source_file` from the converter. | VERIFIED | Treat it as provenance unless a true group field is available. |
| `src/thesis_rl/envs/thesis_scenario_env.py` already implements `length + extra_steps_after_scenario` and a line/physical-boundary adapter. | VERIFIED | Verify it against v1.1, including the fixed default of 50. |
| `conf/scenarios/pipeline_v1.yaml` contains obsolete arm minima and v1 policy assumptions. | VERIFIED | Replace with the approved v1.1 policy, without mass unrelated cleanup. |
| Existing ScenarioNet v1 plans and artifacts are historical. | VERIFIED | The v1.1 plan owns reconciliation; `scenarionet_pipeline_restructure_v2_exec_plan.md` is superseded. |
| Full semantic-observation specification is absent. | VERIFIED | Do not expand `REQ-SN-014` beyond the pipeline boundary. |

## 5. Assumptions And Invariants

| Invariant | Basis | Violation handling |
|---|---|---|
| Catalog paths are normalized and relative to `SCENARIONET_DATA_ROOT`. | SPECIFIED §4 | Reject absolute/traversal paths. |
| Waymo source is only `training_20s`; PG seeds and Waymo groups are split-disjoint. | SPECIFIED §§6.1–6.3 | Fail before runtime views are written. |
| Primary source totals are exact; arm totals differ by at most one; within-arm source balance is best effort. | SPECIFIED §§5–6 | Emit deficit and fail when hard constraints cannot be met. |
| Runtime records are Rulebook eligible and retain exclusion causes in audit artifacts. | SPECIFIED §17.5, ADR-001 | Never relax filters to fill quotas. |
| Existing A0–A5 formula, names, and ACL semantic arm count remain unchanged. | SPECIFIED §8, ADR-001 | Test exact boundaries and reject foreign names. |
| Physics is 0.02 s with repeat 5, hence 10 Hz; horizon is actual scenario length plus 50. | SPECIFIED §§18–19 | Mark time limit as truncation, not termination. |
| Equal seeds, worker count, config, catalog, and manifests reproduce provider sequences. | SPECIFIED §23 | Test deterministic sequence equality. |

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-SN-001` | specification clarification | Waymo grouping with converter `source_file` only | shard group / scenario ID absent verified log or segment | Scenario ID; `source_file` remains provenance | Split feasibility and leakage | Approved; ADR-001 |
| `DEC-SN-002` | specification clarification | Runtime Rulebook eligibility | retain partial records / require eligibility | Require `rulebook_eligible=true` | Dataset population and safety | Approved; ADR-001 |
| `DEC-SN-003` | specification clarification | Semantic arm contract | new arm taxonomy / preserve existing six | Preserve existing A0–A5; ACL `K=6` | Catalog, ACL, tests | Approved; ADR-001 |
| `DEC-SN-004` | specification clarification | Episode tail | 0 / 50 / later pilot choice | Fixed `+50` | Truncation and bootstrap | Approved; ADR-001 |
| `DEC-SN-005` | specification clarification | Waymo cap and throughput | unbounded / configured cap | batch 64, 256 new shards | Reproducibility and cost | Approved; ADR-001, ADR-007, ADR-010 |

No unresolved approval gate exists. Selector decomposition and implementation
algorithm are private details only if every hard invariant is validated.

## 7. Proposed Design

```text
candidate inventories
→ official + thesis validation and Rulebook eligibility
→ auditable final eligible population
→ features, frozen thresholds, preserved A0–A5 classification
→ deterministic grouped balanced selection with strict feasibility validation
→ split/catalog/threshold/runtime manifests and views
→ strict providers and ThesisScenarioEnv
→ vectorized training, evaluation sequence, and source×arm logging
```

Add `rulebook_eligible` and diagnostic errors to catalog records or an
equivalently joined, immutable eligibility artifact. The runtime builder must
consume only selected records and verify `runtime_index → scenario_uid`.
The selected-split solver may use deterministic constructive search, but it
must not silently choose a reduced output or weaken any hard constraint.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-SN-001` | `AC-SN-001` | `scenarios/paths.py`, manifests, configs | `TEST-SN-001` | Planned |
| `REQ-SN-002`–`REQ-SN-006` | `AC-SN-002`–`AC-SN-006` | pipeline, splits, waymo pool, reports | `TEST-SN-002`–`TEST-SN-006` | Planned |
| `REQ-SN-007`–`REQ-SN-009` | `AC-SN-007`–`AC-SN-009` | pg, features, catalog, runtime, validation | `TEST-SN-007`–`TEST-SN-009` | Partial/reconcile |
| `REQ-SN-010`–`REQ-SN-012` | `AC-SN-010`–`AC-SN-012` | envs, provider, ACL runtime | `TEST-SN-010`–`TEST-SN-012` | Partial/reconcile |
| `REQ-SN-013`–`REQ-SN-015` | `AC-SN-013`–`AC-SN-015` | runtime wiring, CLIs, docs | `TEST-SN-013`–`TEST-SN-015` | Planned |
| `REQ-SN-016` | `AC-SN-016`–`AC-SN-018` | `scripts/prepare_scenarionet_dataset.sh`, `src/thesis_rl/cli/scenarios/pipeline_config.py`, `Makefile` | `TEST-SN-016`–`TEST-SN-018` | In progress |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `AC-SN-001` / `TEST-SN-001` | Unit | Manifest/path contract | valid and invalid manifests/paths | frozen IDs and relative paths only | `REQ-SN-001` |
| `AC-SN-002` / `TEST-SN-002` | Unit | Exact source and arm totals | sufficient singleton records | requested totals and arm difference ≤1 | `REQ-SN-002` |
| `AC-SN-003` / `TEST-SN-003` | Unit | Group and seed isolation | Waymo and PG groups | no cross-split overlap | `REQ-SN-003`–`004` |
| `AC-SN-004` / `TEST-SN-004` | Unit | TFRecord provenance | converter-like metadata | scenario ID grouping absent real log/segment | `REQ-SN-004` |
| `AC-SN-005` / `TEST-SN-005` | Unit | Final eligibility | invalid/partial/Rulebook-ineligible fixtures | runtime excludes; audit retains causes | `REQ-SN-005` |
| `AC-SN-006` / `TEST-SN-006` | Unit | Bounded acquisition | unseen shard inventory | deterministic 64-shard batches; failure at 128 | `REQ-SN-006` |
| `AC-SN-007` / `TEST-SN-007` | Unit | PG seed/profile handling | native-profile fixture | reproducible, diagnostic classification | `REQ-SN-007` |
| `AC-SN-008` / `TEST-SN-008` | Unit | Existing arm formula | exact boundary fixtures | unchanged A0–A5 labels and priority | `REQ-SN-008` |
| `AC-SN-009` / `TEST-SN-009` | Integration | Runtime database mapping | Waymo/PG fixture views | loaded ID matches selected UID | `REQ-SN-009` |
| `AC-SN-010` / `TEST-SN-010` | Unit/integration | Termination/truncation | collision, line, boundary, timeout | native collisions, physical exit, `length+50` | `REQ-SN-010` |
| `AC-SN-011` / `TEST-SN-011` | Unit | Provider/ACL arm contract | balanced and A4-only pools | strict no fallback; semantic `K=6` | `REQ-SN-011` |
| `AC-SN-012` / `TEST-SN-012` | Integration | Vector seed and cleanup | repeated spawned workers | deterministic sequence and clean close | `REQ-SN-012` |
| `AC-SN-013` / `TEST-SN-013` | Unit | Logging/report schema | episode and split fixtures | all source×arm and deficit keys present | `REQ-SN-013` |
| `AC-SN-014` / `TEST-SN-014` | Unit | Pipeline leakage boundary | forbidden future/config metadata | rejected/not forwarded to policy | `REQ-SN-014` |
| `AC-SN-015` / `TEST-SN-015` | Smoke | Full small-count CLI pipeline | local fixtures | artifacts and random-policy run succeed | `REQ-SN-015` |
| `AC-SN-016` / `TEST-SN-016` | Shell integration | Restart after derived artifacts were partially written | mocked pipeline commands that reject existing outputs without `--overwrite` | every catalog, Rulebook, split, threshold, and runtime stage receives derived-artifact overwrite; PG source generation remains separately protected | `REQ-SN-016` |
| `AC-SN-017` / `TEST-SN-017` | Shell integration | Actionable orchestration failure | deterministic mocked stage failure | nonzero exit plus stage, operation, failed command, remediation, and retry command in the terminal summary | `REQ-SN-016` |
| `AC-SN-018` / `TEST-SN-018` | Shell/real integration | Missing Rulebook prerequisite artifacts | fresh data root without ego config, braking trials, or calibration | install the frozen canonical ego config non-destructively, collect real trials, create and validate the hash-bound calibration in the CPU pipeline service, then continue filtering | `REQ-SN-016` |

Mandatory commands:

```text
uv run --no-sync python -m pytest -q tests/test_scenario_*.py tests/test_thesis_scenario_env.py tests/test_scenarionet_*.py
make rulebook-v2-check
make lint
make format-check PYTHON_QUALITY_PATHS="src tests scripts"
make smoke
git diff --check
```

Use focused pytest paths and focused Ruff scope for each milestone. Full
real-data acquisition is not a mandatory automated check because it requires
external credentials and must not overwrite frozen data.

## 10. Milestones

- [x] M1 — Completed. Reconciled record/manifest/report eligibility schema and audit population accounting, including record-level Rulebook provenance, strict pre-freeze split validation, and canonical YAML manifests. Depends on `DEC-SN-001`–`005`; validates the available portions of `TEST-SN-001`–`006`.
- [x] M2 — Completed. Implemented strict grouped `balanced_arm_source` selection and bounded acquisition/PG replenishment interfaces. The selector enforces per-split source and arm capacity, uses exhaustive exact search for small grouped fixtures (≤18 groups), and uses a deterministic transportation-based constructive solver for the normal large singleton-group population. The shell pipeline repeats catalog → Rulebook → split feasibility and acquires at most one forced configured batch (now 64 shards) per cycle under the cumulative 256-shard cap. `build_splits` now writes a PG replenishment report on both success and selection failure, separating a true filtered-PG count shortage from joint Waymo/group infeasibility. Validates the implemented portions of `TEST-SN-002`–`008`.
- [x] M3 — Completed implementation reconciliation for runtime views, providers, ACL six-arm interface, environment behavior, and logging. The environment factory rejects an audit or legacy catalog containing any valid/warning record without `rulebook_eligible=true`; provider construction and runtime mapping use only this checked population. The provider layer supports strict uniform-by-arm sampling with explicit one-sided source-cell fallback and carries sampling metadata into reset and terminal-step info. Runtime aggregation records reset/step/episode matrices by `source × arm`, and evaluation CSVs persist ScenarioNet identity, sampling, and completion fields. The real runtime-database/vector smoke remains M4 because it requires a prepared external fixture. Validates the implemented portions of `TEST-SN-009`–`014`.
- [ ] M4 — In progress. The latest reconciled catalog contains 28,483 records and
  8,398 Rulebook-eligible records. Strict split construction now reaches the
  PG composition boundary (`pg/train=550` versus 1,000), while the PG report
  shows no raw runtime-count shortfall; the former Waymo acquisition cap of 128
  new shards has also been reached. The controller now plans at most 1,750 targeted
  PG candidates per cycle from the reported arm deficits; the current report
  has no remaining PG arm deficit, so no further PG is generated. The vectorized smoke still reaches its
  intended stale-runtime-fixture skip after the import-cycle fix.
  Documentation reconciliation and non-mutating final audit remain pending.
  Validates `TEST-SN-015` and mandatory checks.
- [ ] M5 — In progress. Complete correctness-preserving incremental feasibility
  validation: reuse deterministic PG seeds unless explicit PG overwrite is
  requested; skip duplicate full Waymo status scans when the caller already
  validated the catalog deficit; cache Rulebook eligibility by scenario UID,
  payload fingerprint, geometry hash, and calibration hash; and exercise the
  configured bound of two targeted profile replenishment cycles with disjoint
  seed bases. Preserve the full merged audit and split contract. Validate
  full-vs-incremental equivalence and cache invalidation before marking complete.
- [x] M6 — Completed. Make the top-level dataset command safely restartable
  after partial derived outputs and add actionable terminal failure summaries.
  Source PG/Waymo overwrite controls remain unchanged. Validation includes
  mocked shell integration regressions, syntax, focused pytest, and a real
  catalog-stage resume; unavailable ShellCheck is recorded below.
- [ ] M7 — In progress. Close the Rulebook prerequisite gap exposed by the
  first real stage-4 run: provide the documented canonical ego calibration
  config, use the CPU pipeline image for real braking trials/calibration, and
  bootstrap only missing artifacts before static eligibility filtering.

## 11. Progress And Findings Log

### 2026-07-16 — Approval and planning

- Read the approved v1.1 specification, Rulebook v4.6 eligibility requirements,
  ADR template, existing v1 and v2 ScenarioNet plans, code, configuration, and
  focused test sources.
- Promoted the approved specification, created ADR-001, and updated the
  authority index.
- Verified current six-arm ScenarioNet/ACL semantic contract and v1 pipeline
  gaps described in §4.
- Added serializable Rulebook eligibility and exclusion causes to
  `ScenarioRecord`, with a round-trip regression test; the remaining M1
  eligibility selection and manifest work is pending.
- The Rulebook filter CLI now persists eligibility on selected catalog records;
  split eligibility rejects every explicitly Rulebook-ineligible record.
- `balanced_arm_source` now derives remainder arms deterministically from the
  split seed. Before writing a catalog or manifest, the CLI rejects any
  selected runtime population that does not have exact source totals, the
  seed-derived near-uniform arm totals, accepted signal reliability, valid or
  warning validation status, and `rulebook_eligible=true`.
- The split manifest is now canonical YAML and records requested source
  targets plus the v1.1 balancing contract. The preparation script uses the
  same path and no longer applies the obsolete post-split arm-trimming stage.
- The pipeline configuration no longer contains historical per-source arm
  minima. Its Waymo-only A4 acquisition lower bound remains explicit under
  `waymo.required_arms`, separate from final split selection.
- The split CLI now derives candidate, valid/warning, Rulebook-eligible,
  runtime-eligible, and selected population counts from the audit catalog and
  writes them to both the manifest and report. It feeds only the strict
  runtime-eligible population to selectors.
- The arm selector now treats both the per-split source capacity and
  seed-derived per-arm capacity as non-relaxable during candidate selection.
  The accompanying regression covers unequal per-source split targets on a
  feasible singleton-group pool.
- The canonical split manifest now also records the approved Waymo acquisition
  ordering seed, batch size, and cap. The shell pipeline passes these values
  from the sole YAML policy source into the split CLI.
- Runtime defense is now duplicated at the environment boundary: an
  unverified, false, or legacy-null Rulebook eligibility field produces an
  actionable failure before the runtime view is mapped or a provider is
  created. This prevents accidental use of an audit catalog in training.
- Small fixture and smoke pools are now solved exactly under the indivisible
  group, source-total, and seed-derived arm-total constraints. Larger pools
  retain deterministic greedy selection plus strict pre-freeze rejection; this
  prevents a false success while the scalable solver remains M2 work.
- Split reports and manifests now contain the requested, selected, and
  deficit `split × arm × source` matrices plus an explicit compensation from
  equal within-arm source share. Structural A4 Waymo-only and empirically rare
  Waymo A0 cases are therefore auditable rather than treated as failures.
- `waymo_pool_status` can now consume the Rulebook-annotated catalog and
  require `rulebook_eligible=true` in its accounting. This closes the
  testable status interface needed by acquisition.
- The orchestration script now uses that interface: auto-expansion is deferred
  until a Rulebook-filtered split attempt fails because Waymo coverage is
  insufficient, then exactly one unseen batch is acquired and the full
  catalog/Rulebook/split cycle repeats. It supports an initially absent Waymo
  directory without treating that bootstrap state as a catalog-loader error.
- The scalable selector handles the repository's verified normal grouping
  condition—singleton Waymo scenarios when no true log/segment is exposed and
  singleton PG generation seeds—by allocating exact source and arm quotas as a
  deterministic transportation problem. Multi-record groups remain indivisible
  and use the prior exact-small/strict-greedy path; the pre-freeze contract
  continues to reject any unresolved infeasibility rather than changing a
  target.
- `build_splits` now emits `pg/replenishment_report.json` from the
  Rulebook-filtered catalog. It records candidate through runtime-eligible PG
  populations, profile/arm matrices, selected counts, hard count shortfall,
  and any joint selection error. It recommends more PG seeds only for a true
  filtered-PG count shortage, so a Waymo or grouped-split limitation cannot
  cause blind procedural regeneration.
- The M3 provider gap identified on resumption is closed: the catalog provider
  now supports uniform selection over all six semantic arms, followed by the
  specified conditional source selection. A source-cell fallback is permitted
  and recorded only when the selected arm has exactly one available source;
  an absent semantic arm is a strict error. The environment propagates this
  metadata through reset and step info, including terminal episode info.
- Runtime counters now retain source and arm jointly for resets, steps, and
  completed episodes, in addition to the existing source-only/arm-only totals.
  The parent aggregation preserves the JSON-safe `source → arm → count`
  matrices across vector workers.
- The evaluation path now retains ScenarioNet reset/terminal metadata per
  episode and writes it to evaluation CSV rows: scenario UID/ID, source,
  split, semantic arm, worker, sampling mode, requested arm, source-cell
  fallback, termination reason, and separate terminated/truncated flags.
  This replaces a synthetic seed-only scenario ID whenever the environment
  provides the frozen-catalog identity.
- `docker compose run --rm dev uv run --no-sync ruff check
  src/thesis_rl/scenarios/provider.py src/thesis_rl/scenarios/__init__.py
  src/thesis_rl/envs/factory.py src/thesis_rl/envs/thesis_scenario_env.py
  src/thesis_rl/runtime/wiring/builders.py tests/test_scenario_provider.py
  tests/test_thesis_scenario_env.py` and `python -m pytest -q
  tests/test_scenario_provider.py tests/test_thesis_scenario_env.py` passed:
  28 tests.
- `docker compose run --rm dev uv run --no-sync ruff check
  src/thesis_rl/agent/agent.py src/thesis_rl/runtime/loops/eval_loop.py
  src/thesis_rl/runtime/loops/train_loop.py tests/test_agent_pipeline.py` and
  `python -m pytest -q tests/test_agent_pipeline.py tests/test_eval_artifacts.py
  tests/test_scenario_provider.py tests/test_thesis_scenario_env.py` passed:
  40 tests.
- M4 fixture inspection found `data/scenarionet/runtime/train/dataset_summary.pkl`
  and a catalog, but the catalog contains valid PG records with
  `rulebook_eligible != true`. The vectorized integration tests therefore
  skip after the intended fail-closed runtime validation rather than exercising
  the fixture. This is a stale artifact discrepancy, not an implementation
  failure. The user subsequently approved the frozen-artifact rebuild.
- A full-corpus Rulebook filter rebuild was started with `--workers 1` after
  duplicate attempts were detected, but that serial execution was explicitly
  stopped by the user after about 71 minutes. No output artifacts were
  published. The user will restart the exact filter command manually with
  `--workers 16`; do not run another concurrent filter process.
- The user completed the 16-worker rebuild successfully in 25 minutes 13
  seconds: `eligible=5179`, `excluded=7898`. The published catalog contains
  3,532 Waymo and 1,647 PG records, with `rulebook_eligible=true` on every
  published record. This confirms the stale-flag discrepancy is repaired but
  proves a hard PG deficit of 103 records against the approved final target of
  1,750 PG scenarios.
- Running `tests/test_scenarionet_vectorized_integration.py` after the rebuild
  initially failed during collection with an import cycle between
  `thesis_rl.contracts.causal_scene_context` and `thesis_rl.rulebook.v2.wrapper`.
  The package wrapper exports are now lazy, preserving the public API; the
  focused regression and wrapper/contract tests pass (23 tests). The vector
  test now collects and reaches its intended stale-runtime-fixture skip.
- The split diagnostic against the rebuilt catalog fails at the strict runtime
  contract with `waymo/train` selected `483` versus requested `1000`. The
  static Waymo population is 3,532, but only 1,240 records are both
  valid/warning and signal-allowed (`752` not_applicable, `488` complete);
  305 partial and one missing-signal record are excluded, and 2,152 Waymo
  records are invalid. The PG report records 1,646 runtime-eligible records
  versus 1,750 requested (shortfall 104). The existing pool therefore needs
  both approved PG replenishment and the pipeline's bounded Waymo expansion;
  rerunning `build_splits` alone cannot succeed.
- The first incremental optimization pass is implemented. Existing PG output is
  reused unless `SCENARIONET_PG_OVERWRITE=true`; the auto-expansion path can skip
  its redundant pre-download and post-conversion full Waymo status scans when
  the parent pipeline already proved the deficit and requests one batch; and
  Rulebook eligibility artifacts now retain scenario file fingerprints and
  reuse only records matching the current geometry/calibration hashes. New or
  changed records remain evaluated with the configured worker pool, and the
  merged audit is still written and checked globally. The shell cycle now stops
  on a reported PG hard shortfall before acquiring more Waymo, avoiding an
  unrelated external download when the PG pool is the blocking source.
- Focused incremental validation passed: shell syntax, Ruff format/lint, and
  14 Rulebook eligibility/Waymo pool tests.
- The user approved ADR-005: replenish PG with one complete 350-candidates-per-
  profile block at seed start `5920000`, persist a separate replenishment report,
  and retain strict source/split eligibility. `make scenarionet-pg-replenish`
  now exposes this bounded operation with 16 workers; focused report-output and
  incremental tests pass (15 tests in the combined focused run). The catalog
  builder now has an explicit all-PG mode so replenishment seed windows are
  included without changing the legacy filtered-window default; the broader
  focused validation passes 21 tests.
- The first pipeline retry after enabling reuse exposed an unbound state
  variable in the forced one-batch Waymo path when the redundant initial status
  scan was skipped. The path now initializes incomplete/unknown state explicitly
  and passes shell syntax, ShellCheck, and whitespace validation. The failed
  retry stopped before downloading a shard.
- The next retry reused all 13,098 Rulebook results, then acquired and converted
  the first 16 unseen Waymo shards (`00163`–`00178`) successfully. Its log
  revealed two follow-up defects: `.env` overwrote the pipeline's configured
  16 converter workers with 8, and the deferred A4 status was printed as a
  fabricated `0/584`. The expansion script now preserves an explicit worker
  override and reports deferred A4 status without inventing a measurement.
- The same retry exposed a host/container path bug in the PG-shortfall guard:
  the host shell checked `/workspace/data/...` directly, so it missed the
  current report (`hard_count_shortfall=770`) and expanded Waymo unnecessarily.
  The guard now checks the host-mounted report and passes the container path
  only to the JSON reader; a resumed cycle will stop before further Waymo
  acquisition while the PG shortfall remains.
  The approved acquisition policy is now 64 shards per cycle under ADR-007;
  the cumulative cap is now 256 new shards under ADR-010, and post-Rulebook
  feasibility checks remain unchanged.
- Validation after this correction: `bash -n scripts/expand_waymo_pool.sh`,
  `shellcheck scripts/expand_waymo_pool.sh`, and `git diff --check` pass.
- The user approved the throughput amendment on 2026-07-17. The single YAML
  source now resolves `batch_shards=64`, `max_new_shards=256`, and
  `waymo.workers=16`; ADR-007 and the authoritative v1.1 specification record
  the amendment. The read-only `pipeline_config` container check passed.
- The first 64-shard retry exposed a state-reconciliation defect: forced
  one-batch conversion deferred the status scan without recording the selected
  shard names, so the next cycle reprocessed `00163`–`00178` inside
  `batch_00163_00226`. The expansion script now reconstructs shard state from
  finalized batch directory ranges and records every successful batch. The
  Waymo loader deterministically retains the first record for an already
  reprocessed UID and emits a runtime warning, allowing the existing pool to
  be reconciled without deleting user data. Focused catalog/loader validation
  passes (7 tests), plus shell syntax, ShellCheck, and whitespace checks.
- The post-reconciliation catalog reports 1,315 Rulebook/runtime-eligible
  Waymo records of 1,750 required, with a total deficit of 435. The explicit
  A4_vru lower-bound check is the dominant constraint at 202/584 (deficit
  382); PG replenishment is now sufficient (`hard_count_shortfall=0`). The
  remaining bounded Waymo cycles must therefore be allowed to complete before
  judging feasibility; no filter or quota relaxation is permitted.
- A preliminary exclusion breakdown from the failed pre-deduplication catalog
  indicates that the rigid route/Rulebook boundary is a material bottleneck:
  9,557 records fail task-route association, 2,646 fail assigned-route
  continuity, 1,450 fail signal-state checks, and 1,199 fail map geometry.
  These counts include the known reprocessed-batch duplicates and must be
  recomputed on the reconciled catalog. The observation vector shape itself is
  not a catalog eligibility predicate; the route metadata and Rulebook checks
  are the direct causes. No relaxation is approved; a post-reconciliation
  sensitivity audit is required before proposing any behavioral change.
- The completed 64-shard Waymo cycle moved the failure boundary: the subsequent
  split attempt no longer failed on `waymo/train=845`, but on
  `pg/train=550`. The run reached the configured 128-shard Waymo cap, while
  the PG report still showed no raw runtime-count shortfall; this is therefore
  a strict PG source/arm/group selection infeasibility, not a reversal of the
  earlier Waymo count. The existing duplicate-batch warning (1,139 UIDs) was
  handled by the deterministic loader and did not abort catalog construction.
- The PG report now includes source/arm/split diagnostics on selection failure.
  The controller triggers up to the configured number of targeted PG cycles
  (currently `2`) when the failure explicitly targets a PG split and the raw PG
  runtime-count shortfall is zero. Each cycle has a maximum candidate budget of
  `1,750`, allocated by the observed profile-to-arm yield matrix and the arm
  deficits in the replenishment report; if no PG arm deficit remains, it does
  not generate more PG and proceeds to the normal bounded stop/acquisition
  decision. Each next seed base is disjoint and the full feasibility cycle is
  rebuilt.
- The first resumed run reached this intended branch but exposed an inline
  Python f-string escaping error while reading pg/replenishment_report.json.
  The parser now uses a format string valid under the container's Python
  version, and a regression test executes the exact extracted command against
  a PG composition-failure report.
- The user approved doubling the Waymo cap from 128 to 256 new shards. The
  single YAML source, expansion default, ADR-010, and v1.1 specification now
  resolve the amended cap while retaining 64-shard batches and 16 workers.
- The first targeted PG run exposed a progress-accounting defect: unspecified
  profiles retained the default 350 count, while the progress total counted
  only the explicit profile map. Profile overrides now zero unspecified
  profiles, and the regression suite verifies that a P5-only plan generates
  exactly 1,750 tasks.
- `docker compose run --rm dev uv run --no-sync ruff check
  src/thesis_rl/scenarios/pipeline.py src/thesis_rl/scenarios/reports.py
  src/thesis_rl/cli/scenarios/build_splits.py tests/test_scenarionet_pipeline.py`
  and `python -m pytest -q tests/test_scenarionet_pipeline.py` passed: 19
  tests. The new coverage proves exact split/source/arm contract preservation
  above the former 18-group exact-search threshold and PG shortage reporting.
- `docker compose run --rm dev uv run --no-sync python -m pytest -q
  tests/test_scenario_records.py tests/test_scenario_catalog.py` passed: 12
  tests.
- Direct Ruff format and lint checks passed after the image-level `make` target
  proved unavailable.

Next step: resume the already-started M3 runtime/provider/environment/logging
reconciliation, then perform the M4 fixture pipeline and smoke validation.

### 2026-07-18 — Restartability and error-reporting audit

- Reproduced the first-cycle catalog collision against the user-owned partial
  dataset. The root cause was that only feasibility cycles after cycle zero
  received `--overwrite`, although catalog, eligibility, split, threshold, and
  runtime files are deterministic derived artifacts.
- The top-level pipeline now refreshes all derived outputs on every cycle while
  retaining separate explicit overwrite protection for PG and Waymo source
  scenarios. A real stage-3 run rebuilt 1,750 PG catalog entries successfully.
- Added a terminal failure summary with stage, operation, failed command, line,
  exit code, remediation hint, and retry command while preserving the detailed
  underlying error above it.
- Corrected the Waymo cap counter so PG-only replenishment cycles no longer
  consume a Waymo batch slot. Added pre-arithmetic bounds checks and typed YAML
  validation with argparse errors instead of tracebacks.
- Moved PG replenishment from the unavailable `dev` image to the existing CPU
  `dataset-pipeline` service and replaced the hard-coded 16 workers with the
  YAML-resolved worker count.
- Corrected `scenarionet-recatalog` to skip both PG and Waymo source mutation;
  it now matches its documented derived-artifact-only behavior.
- The real rebuilt catalog contains one expected invalid PG record,
  `pg:scenarionet_v1:PGMap-1920069`, due to a 0.007 m degenerate SDC route. It
  remains in the audit catalog and is excluded by the existing hard validity
  policy; no quality threshold was relaxed.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `docs/specifications/scenarionet_integration_spec_v1.1.md` | Added | Approved authoritative specification |
| `docs/decisions/ADR-001-scenarionet-v1-1-dataset-policy.md` | Added | Approved material dataset/runtime policy |
| `docs/implementation/scenarionet_integration_spec_v1.1_exec_plan.md` | Added | Living v1.1 implementation record |
| `docs/project_index.md` | Modified | Authority and ExecPlan registry |
| `src/thesis_rl/scenarios/{records,catalog,pipeline,splits,manifests,reports,waymo_pool}.py` | Modified/planned modification | Eligibility, split, audit, manifest, scalable singleton-group selection, and PG replenishment-report contract |
| `src/thesis_rl/cli/scenarios/build_splits.py` | Modified | Strict split CLI and PG replenishment artifact on success/failure |
| `src/thesis_rl/cli/scenarios/plan_pg_replenishment.py`, `src/thesis_rl/scenarios/pg/replenishment.py` | Added | Deterministic targeted profile allocation from reported PG arm deficits |
| `src/thesis_rl/cli/scenarios/generate_pg_dataset.py`, `src/thesis_rl/scenarios/pg/report.py` | Modified | Profile-specific PG candidate counts for targeted replenishment |
| `scripts/{prepare_scenarionet_dataset,expand_waymo_pool}.sh` | Modified | Post-Rulebook bounded Waymo acquisition loop and PG report path |
| `Makefile` | Modified | Run PG replenishment in the CPU pipeline service with YAML-resolved workers |
| `docs/setup/scenarionet_waymo_conversion.md` | Modified | Document restart and failure-summary behavior |
| `src/thesis_rl/cli/scenarios/pipeline_config.py` | Modified | Validate YAML types and ranges before shell orchestration |
| `src/thesis_rl/scenarios/{features,arms,provider,runtime_database,validation}.py` | Modified/planned reconciliation | Preserved arm taxonomy, strict arm-uniform provider, and runtime behavior |
| `src/thesis_rl/runtime/wiring/builders.py` | Modified | Aggregates ScenarioNet source×arm runtime matrices across workers |
| `src/thesis_rl/envs/{factory,thesis_scenario_env}.py` | Modified/planned reconciliation | Strict provider construction, sampling metadata, episode contract, and source×arm counters |
| `src/thesis_rl/agent/agent.py`, `src/thesis_rl/runtime/loops/{train_loop,eval_loop}.py` | Modified | Preserve and persist per-episode ScenarioNet identity, sampling, and completion metadata |
| `src/thesis_rl/envs/{thesis_scenario_env,scene_context,scenario_env_factory}.py` | Planned verification/modification | Environment contract |
| `conf/scenarios/pipeline_v1.yaml`, `conf/env/scenarionet.yaml`, `conf/curriculum/scenario_acl_scenarionet.yaml` | Modified/planned modification | Frozen v1.1 policy and documented strict provider modes |
| `tests/test_scenarionet_pipeline.py` | Modified | Restart, failure UX, cap accounting, config validation, and CPU replenishment regressions |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| Complete v1.1 specification review | PASS | 2026-07-16 | Candidate reviewed before approval; promoted after explicit approval |
| `git diff --check` | PASS | 2026-07-16 | No whitespace errors before v1.1 implementation work |
| `docker compose run --rm dev make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/scenarios/records.py tests/test_scenario_records.py"` | FAIL | 2026-07-16 | Container image has no `make`; equivalent direct Ruff commands used below |
| `docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/scenarios/records.py tests/test_scenario_records.py` | PASS | 2026-07-16 | Both files formatted |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/scenarios/records.py tests/test_scenario_records.py` | PASS | 2026-07-16 | All checks passed |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scenario_records.py tests/test_scenario_catalog.py` | PASS | 2026-07-16 | 12 passed; validates Rulebook eligibility serialization regression and catalog compatibility |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_catalog_eligibility.py tests/test_scenarionet_pipeline.py tests/test_scenario_records.py tests/test_scenario_catalog.py` | PASS | 2026-07-16 | 29 passed after filter and pipeline eligibility integration |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scenario_*.py tests/test_scenarionet_*.py tests/test_thesis_scenario_env.py` | PASS | 2026-07-16 | 149 passed in 24.75 s; applicable ScenarioNet regression suite |
| `docker compose run --rm dev uv run --no-sync ruff format ... && ruff check ... && python -m pytest -q tests/test_scenarionet_pipeline.py tests/test_scenario_manifests.py tests/test_scenario_records.py tests/test_rulebook_v2_catalog_eligibility.py tests/test_waymo_pool.py` | PASS | 2026-07-16 | 41 passed; verifies strict Rulebook/signal/source/arm freeze contract, v1.1 manifest schema, and Waymo pool policy |
| `docker compose run --rm dev uv run --no-sync python -m thesis_rl.cli.scenarios.pipeline_config --config conf/scenarios/pipeline_v1.yaml` | PASS | 2026-07-16 | Resolved approved batch 16, cap 128, exact source targets, and Waymo A4 feasibility lower bound |
| `bash -n scripts/prepare_scenarionet_dataset.sh` | PASS | 2026-07-16 | Shell syntax valid after canonical YAML manifest and removal of post-split trimming |
| `git diff --check` | PASS | 2026-07-16 | No whitespace errors after the M1 changes |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scenario_*.py tests/test_scenarionet_*.py tests/test_thesis_scenario_env.py` | PASS | 2026-07-16 | 154 passed in 25.27 s after M1 completion and source-capacity selector changes |
| `docker compose run --rm dev uv run --no-sync ruff format ... && ruff check ... && python -m pytest -q tests/test_scenario_manifests.py tests/test_scenarionet_pipeline.py` | PASS | 2026-07-16 | 20 passed; validates the canonical manifest acquisition fields and selector regression |
| `docker compose run --rm dev uv run --no-sync ruff format src/thesis_rl/envs/factory.py tests/test_scenarionet_pipeline.py && ruff check ... && python -m pytest -q tests/test_scenarionet_pipeline.py tests/test_scenario_provider.py` | PASS | 2026-07-16 | 25 passed; verifies the Rulebook-eligibility runtime boundary and provider regressions |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scenario_*.py tests/test_scenarionet_*.py tests/test_thesis_scenario_env.py` | PASS | 2026-07-16 | 153 passed, 2 skipped in 11.73 s; skips require an externally prepared ScenarioNet runtime dataset |
| `docker compose run --rm dev uv run --no-sync ruff format src/thesis_rl/scenarios/pipeline.py tests/test_scenarionet_pipeline.py && ruff check ... && python -m pytest -q tests/test_scenarionet_pipeline.py` | PASS | 2026-07-16 | 17 passed; covers exact small-pool grouped selection and strict split-contract regressions |
| `docker compose run --rm dev uv run --no-sync ruff format src/thesis_rl/scenarios/pipeline.py src/thesis_rl/cli/scenarios/build_splits.py tests/test_scenarionet_pipeline.py && ruff check ... && python -m pytest -q tests/test_scenarionet_pipeline.py tests/test_scenario_manifests.py` | PASS | 2026-07-16 | 22 passed; verifies source×arm compensation diagnostics and split-manifest compatibility |
| `docker compose run --rm dev uv run --no-sync ruff format src/thesis_rl/scenarios/waymo_pool.py src/thesis_rl/cli/scenarios/waymo_pool_status.py tests/test_waymo_pool.py && ruff check ... && python -m pytest -q tests/test_waymo_pool.py` | PASS | 2026-07-16 | 8 passed; verifies optional Rulebook-required Waymo feasibility from an annotated catalog |
| `bash -n setup.sh scripts/*.sh && docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/cli/scenarios/build_catalog.py tests/test_scenario_catalog_build.py && python -m pytest -q tests/test_scenario_catalog_build.py tests/test_waymo_pool.py` | PASS | 2026-07-16 | 14 passed; verifies empty-Waymo bootstrap support and shell syntax without external downloads |
| `docker compose run --rm dev uv run --no-sync ruff format src/thesis_rl/scenarios/pipeline.py src/thesis_rl/scenarios/reports.py src/thesis_rl/cli/scenarios/build_splits.py && ruff check src/thesis_rl/scenarios/pipeline.py src/thesis_rl/scenarios/reports.py src/thesis_rl/cli/scenarios/build_splits.py tests/test_scenarionet_pipeline.py && python -m pytest -q tests/test_scenarionet_pipeline.py` | PASS | 2026-07-16 | 19 passed; verifies the scalable singleton-group solver and the PG replenishment report |
| `bash -n setup.sh scripts/*.sh && docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/scenarios/pipeline.py src/thesis_rl/scenarios/reports.py src/thesis_rl/cli/scenarios/build_splits.py src/thesis_rl/cli/scenarios/build_catalog.py tests/test_scenarionet_pipeline.py tests/test_scenario_catalog_build.py && ruff check ... && python -m pytest -q tests/test_scenario_*.py tests/test_scenarionet_*.py tests/test_thesis_scenario_env.py && git diff --check` | PASS | 2026-07-16 | 157 passed, 2 expected skips requiring an external prepared ScenarioNet runtime dataset; six focused files formatted, lint and shell/whitespace checks passed |
| `bash -n setup.sh scripts/*.sh && shellcheck setup.sh scripts/*.sh && git diff --check` | PASS | 2026-07-16 | Shell syntax, ShellCheck, and whitespace validation pass after preserving dynamic configuration export semantics |
| `bash -n scripts/prepare_scenarionet_dataset.sh scripts/expand_waymo_pool.sh && shellcheck scripts/prepare_scenarionet_dataset.sh scripts/expand_waymo_pool.sh && git diff --check` | PASS | 2026-07-17 | Bounded PG compositional replenishment trigger and batch-state reconciliation pass shell/whitespace validation |
| `docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/scenarios/reports.py tests/test_scenarionet_pipeline.py && ruff check ... && python -m pytest -q tests/test_scenarionet_pipeline.py` | PASS | 2026-07-16 | 19 passed after making joint selection-failure reporting explicitly distinct from a completed PG selection |
| `docker compose run --rm -T dev uv run --no-sync ruff format --check src/thesis_rl/cli/scenarios/build_splits.py tests/test_scenarionet_pipeline.py && ruff check ... && python -m pytest -q tests/test_scenarionet_pipeline.py` | PASS | 2026-07-17 | 19 passed after adding source/arm/split diagnostics for PG composition failures |
| `docker compose run --rm dev uv run --no-sync ruff format src/thesis_rl/scenarios/provider.py src/thesis_rl/scenarios/__init__.py src/thesis_rl/envs/factory.py src/thesis_rl/envs/thesis_scenario_env.py src/thesis_rl/runtime/wiring/builders.py tests/test_scenario_provider.py tests/test_thesis_scenario_env.py && ruff check ... && python -m pytest -q tests/test_scenario_provider.py tests/test_thesis_scenario_env.py` | PASS | 2026-07-16 | 28 passed; verifies arm-uniform source-cell behavior and source×arm aggregation |
| `docker compose run --rm dev uv run --no-sync ruff format src/thesis_rl/agent/agent.py src/thesis_rl/runtime/loops/eval_loop.py src/thesis_rl/runtime/loops/train_loop.py tests/test_agent_pipeline.py && ruff check ... && python -m pytest -q tests/test_agent_pipeline.py tests/test_eval_artifacts.py tests/test_scenario_provider.py tests/test_thesis_scenario_env.py` | PASS | 2026-07-16 | 40 passed; verifies per-episode ScenarioNet metadata propagation and CSV-compatible evaluation metrics |
| `bash -n setup.sh scripts/*.sh && shellcheck setup.sh scripts/*.sh && docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/scenarios/provider.py src/thesis_rl/scenarios/__init__.py src/thesis_rl/envs/factory.py src/thesis_rl/envs/thesis_scenario_env.py src/thesis_rl/runtime/wiring/builders.py src/thesis_rl/agent/agent.py src/thesis_rl/runtime/loops/eval_loop.py src/thesis_rl/runtime/loops/train_loop.py tests/test_scenario_provider.py tests/test_thesis_scenario_env.py tests/test_agent_pipeline.py && ruff check ... && python -m pytest -q tests/test_scenario_*.py tests/test_scenarionet_*.py tests/test_thesis_scenario_env.py tests/test_agent_pipeline.py tests/test_eval_artifacts.py && git diff --check` | PASS | 2026-07-16 | 171 passed, 2 expected skips requiring an external prepared ScenarioNet runtime dataset; formatting, lint, shell, and whitespace checks passed |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q -rs tests/test_scenarionet_vectorized_integration.py` | SKIP | 2026-07-16 | 2 skips: local runtime fixture catalog has valid records without `rulebook_eligible=true`; M4 smoke needs an approved fixture rebuild/replacement |
| `docker compose run --rm -T dev uv run --no-sync python -m thesis_rl.cli.scenarios.pipeline_config --config conf/scenarios/pipeline_v1.yaml` | PASS | 2026-07-17 | Resolved Waymo batch 64, cap 256, 16 workers, and PG composition replenishment limit 2 |
| `docker compose run --rm -T dev uv run --no-sync ruff format --check tests/test_scenarionet_pipeline.py && ruff check tests/test_scenarionet_pipeline.py && python -m pytest -q tests/test_scenarionet_pipeline.py` | PASS | 2026-07-17 | 20 passed; regression covers PG report parsing for compositional split failure |
| `bash -n scripts/prepare_scenarionet_dataset.sh scripts/expand_waymo_pool.sh && shellcheck scripts/prepare_scenarionet_dataset.sh scripts/expand_waymo_pool.sh && git diff --check` | PASS | 2026-07-17 | Validates the corrected PG report parser and orchestration shell syntax |
| `docker compose run --rm -T dev uv run --no-sync ruff format src/thesis_rl/scenarios/pg/report.py src/thesis_rl/scenarios/pg/replenishment.py src/thesis_rl/cli/scenarios/generate_pg_dataset.py src/thesis_rl/cli/scenarios/plan_pg_replenishment.py tests/test_pg_replenishment.py && ruff check ... && python -m pytest -q tests/test_pg_replenishment.py tests/test_scenarionet_pipeline.py` | PASS | 2026-07-17 | 24 passed; validates profile-count overrides, exact targeted task totals, and global source×arm deficits |
| Read-only targeted-plan simulation on the current catalog | PASS | 2026-07-17 | Waymo A4 and PG A5 are the remaining global composition gaps; the planner assigns the 1,750-candidate budget to `P5_complex_mixed`, while A4 remains Waymo-only |
| `docker compose run --rm -T dev uv run --no-sync python -m thesis_rl.cli.scenarios.pipeline_config --config conf/scenarios/pipeline_v1.yaml` | PASS | 2026-07-17 | Resolves targeted PG budget 1,750 and two-cycle bound |
| `bash -n scripts/prepare_scenarionet_dataset.sh scripts/expand_waymo_pool.sh && shellcheck scripts/prepare_scenarionet_dataset.sh scripts/expand_waymo_pool.sh && git diff --check` | PASS | 2026-07-17 | Batch-throughput amendment and host/container guard fixes pass shell and whitespace validation |
| Focused pytest command | NOT_RUN | 2026-07-16 | Host environment lacks `uv` and `python`; run in provisioned container |
| Full real-data acquisition | NOT_RUN | 2026-07-16 | Requires credentials and can mutate dataset artifacts |
| `docker compose run --rm -T dataset-pipeline uv run --no-sync python -m pytest -q tests/test_scenarionet_pipeline.py tests/test_scenario_catalog.py tests/test_scenario_catalog_build.py tests/test_scenario_runtime_database.py tests/test_rulebook_v2_catalog_eligibility.py` | PASS | 2026-07-18 | 48 passed; covers restart overwrite, actionable failures, independent PG/Waymo counters, source-safe recatalog, config validation, catalog, Rulebook, and runtime regressions |
| `docker compose run --rm -T dataset-pipeline uv run --no-sync ruff format --check src/thesis_rl/cli/scenarios/pipeline_config.py tests/test_scenarionet_pipeline.py` | PASS | 2026-07-18 | Both modified Python files are formatted; focused Ruff check also passed |
| `bash -n scripts/prepare_scenarionet_dataset.sh` and `git diff --check` | PASS | 2026-07-18 | Shell syntax and patch whitespace are valid |
| `make config` | PASS | 2026-07-18 | Docker Compose configuration is valid |
| Real stage-3 `build_catalog --overwrite` on the mounted dataset | PASS | 2026-07-18 | Rebuilt the former collision target with 1,750 PG entries; one degenerate-route record remains correctly marked invalid for downstream exclusion |
| `shellcheck scripts/prepare_scenarionet_dataset.sh` | NOT_RUN | 2026-07-18 | ShellCheck is unavailable on the host and in the CPU container; the `dev` image cannot build on this ARM host because the configured CUDA PyTorch wheel has no matching platform build |
| Full `make scenarionet-pipeline` with Waymo acquisition | NOT_RUN | 2026-07-18 | Would start long-running external downloads and mutate the user-owned candidate pool; restart behavior was verified with shell integration tests and the real formerly failing catalog stage |

## 15. Final Reconciliation

All v1.1 requirements are `NOT_IMPLEMENTED` or `PARTIAL` pending milestones.
Existing v1 behavior is not evidence of v1.1 compliance until each requirement,
acceptance criterion, artifact, and mandatory test is reconciled.

Known limitations: no authoritative semantic-observation specification; no
current final v1.1 dataset artifact; external Waymo acquisition is unavailable
in this planning environment.

Deferred required work: M1–M4. Optional future work: custom PG VRU generation,
Waymo-natural evaluation, and solver optimization beyond a correct deterministic
selector.
