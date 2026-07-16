# ScenarioNet Pipeline Integrity And Restructure v2

## 1. Metadata

- Plan ID: `SCENARIO-PIPELINE-RESTRUCTURE-V2`
- Feature: end-to-end ScenarioNet dataset acquisition, eligibility, splitting,
  reporting, and runtime freeze
- Historical specification at creation: `docs/specifications/scenarionet_integration_v1_specification.md`, version v1
- Current authoritative specification: `docs/specifications/scenarionet_integration_spec_v1.1.md`, version 1.1
- Status: `SUPERSEDED`
- Created: 2026-07-16
- Last updated: 2026-07-16
- Branch: current working branch
- Related ADRs: none; `docs/decisions/` is currently absent
- Owner: thesis repository maintainers

> Superseded by [`scenarionet_integration_spec_v1.1_exec_plan.md`](scenarionet_integration_spec_v1.1_exec_plan.md).
> Its open v1 policy gates are resolved by the approved v1.1 specification and ADR-001.

## 2. Objective And Scope

The objective is to replace the current patchwork of pre-filter acquisition,
best-effort split selection, optional arm trimming, and incomplete reports with
one explicit, reproducible pipeline contract.

The restructured pipeline must determine final eligibility before freezing
source counts, acquire or generate additional candidates only when the final
eligible pool is insufficient, fail with an actionable deficit report when the
target cannot be reached, and preserve the scientific distinction between
dataset composition and semantic-arm diagnostics.

In scope:

- Waymo and PG acquisition/generation readiness accounting;
- common final eligibility accounting, including quality, signal policy, and
  Rulebook v2 when enabled;
- exact source/split target construction and leakage checks;
- arm classification and diagnostic reporting;
- deterministic manifests, cache keys, iteration state, and failure reports;
- incremental recomputation and bounded parallelism where it does not change
  scientific behavior;
- regression, integration, and end-to-end fixture coverage.

Out of scope unless separately approved:

- changing the approved Rulebook v2 semantics;
- changing the A0-A5 scientific definitions;
- inventing PG VRU generation or a custom Waymo arm classifier;
- changing ScenarioEnv, provider, reward, observation, or training semantics;
- forcing equal arm counts when the authoritative specification treats rare arms
  as diagnostic and sampler/curriculum inputs.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-PIPE-001` | Use Waymo `training_20s`, PG offline generation, common `ScenarioDescription`, and frozen train/validation/test runtime views. | §§2.3-2.5, 4, 6, 28 |
| `REQ-PIPE-002` | The baseline source targets are Waymo/PG `1000/1000` train, `250/250` validation, and `500/500` test. A reduction must be explicit and documented. | §§5.1, 6.4 |
| `REQ-PIPE-003` | Exclude invalid scenarios from runtime data; preserve valid/warning semantics and signal reliability policy without imputing missing states. | §§17.3-17.4 |
| `REQ-PIPE-004` | Preserve deterministic, leakage-free grouping for Waymo and seed-disjoint PG splits. | §§6.1-6.4, 27.5 |
| `REQ-PIPE-005` | Compute traffic thresholds only on a source-balanced train candidate and never on evaluation data. | §15 |
| `REQ-PIPE-006` | Assign A0-A5 from realized scenario features, report rare arms, and do not regenerate the dataset solely to make arm counts equal. Rare arms are handled by the later sampler/curriculum. | §§8, 11, 16.4 |
| `REQ-PIPE-007` | Preserve strict provider behavior: no silent source/arm fallback when a requested combination is empty. | §22.1 |
| `REQ-PIPE-008` | Persist manifest, catalog, split, threshold, validation, and diagnostic artifacts with enough identifiers to reproduce a run. | §§3, 4, 6.4, 10, 24, 28 |

## 4. Current Repository Analysis

### 4.1 Verified call flow

The current shell orchestrator is
`scripts/prepare_scenarionet_dataset.sh`:

```text
pre-Rulebook Waymo pool expansion
→ fixed-count PG generation
→ raw catalog build
→ Rulebook v2 static filtering
→ auto/balanced split selection
→ train-only thresholds and arm classification
→ optional arm trimming
→ runtime database build
→ mapping and official checks
```

The following facts are `VERIFIED` from the code and the 2026-07-16 run:

1. Waymo expansion uses `src/thesis_rl/scenarios/waymo_pool.py` and
   `scripts/expand_waymo_pool.sh`. Its eligibility predicate checks quality and
   allowed signal reliability, but it runs before catalog construction and
   Rulebook filtering. The shell loop targets total Waymo availability and only
   one extra arm requirement, `A4_vru`.
2. PG generation uses a fixed number of attempts per profile. It does not
   replenish the pool based on post-catalog, post-Rulebook, or post-split valid
   counts.
3. Rulebook filtering writes a catalog containing static-eligible records, but
   the split selector applies `eligible_entries()` again and removes invalid
   records. The current report therefore presents the Rulebook-eligible count as
   the input while the actual usable population is smaller.
4. `assign_arm_balanced_splits_to_targets()` is a best-effort greedy selector.
   It processes smaller split totals first and does not receive or enforce the
   configured `arm_minimums`. Those minimums are diagnostic in this path.
5. `balance.enabled=false` skips stage 7. If enabled, `balance_arm_distribution`
   can only remove records to approach equal arm totals; it cannot acquire or
   generate missing records and it does not enforce source/split quotas.
6. Auto split mode writes reduced effective counts and a deficit report instead
   of failing the pipeline. The runtime stages do not reject a manifest whose
   effective counts differ from the configured baseline targets.
7. The authoritative specification explicitly says not to regenerate the
   dataset merely to equalize arms. The current `balanced_arm_source`, manual
   arm minimums, `balance` section, and Waymo A4 acquisition target are an
   implementation extension whose scientific authority is not established in
   the project index.

### 4.2 Current run evidence

The attached 2026-07-16 run and local artifacts show:

```text
raw catalog                              13077
Rulebook static eligible                   5179
quality valid/warning after Rulebook       3026
quality + allowed signal                  2886
selected final catalog                    2886
train                                     1386 (Waymo 430, PG 956)
validation                                 500
test                                      1000
```

The final usable arm/source pool before split selection contained:

```text
Waymo: A0=7, A1=164, A2=297, A3=428, A4=184, A5=160
PG:    A0=694, A1=612, A2=231, A3=42, A4=0, A5=67
```

Thus A4 and A5 were not eliminated completely by the filters. The selected
records placed all A4 and A5 records in validation/test, leaving train with
`A4=0` and `A5=0`. This is a selection-policy failure, compounded by the fact
that the final pool was too small for the configured source totals and any
equal-arm target.

### 4.3 Directly relevant debt and conflicts

- The implementation plan claims a 3500-record, balanced final dataset, while
  the current artifacts and current code path produce 2886 records. The plan is
  stale relative to the current run and cannot be used as evidence of readiness.
- The specification's “rare arms are documented and handled by the sampler”
  policy conflicts with the later implementation plan's arm quotas and
  `balanced_arm_source` policy. This is an approval gate, not an implementation
  detail.
- Reports mix static Rulebook eligibility, quality eligibility, signal
  eligibility, split availability, and selected counts without a single named
  population contract.

## 5. Assumptions And Invariants

| Invariant | Basis | Handling |
|---|---|---|
| Waymo source is `training_20s` | SPECIFIED, §6.1 | Reject other official variants. |
| PG seeds are disjoint across splits | SPECIFIED, §§6.2-6.3 | Validate before writing split/runtime artifacts. |
| Invalid records never enter runtime | SPECIFIED, §17.4 | Exclude from final eligible population and report by reason. |
| `partial`/`missing` signal records are not silently imputed | SPECIFIED, §17.3 | Exclude when policy disallows them and report separately. |
| Arms are descriptive, not equal-count quotas, unless explicitly approved otherwise | SPECIFIED, §§11 and 16.4 | Report distribution and provider availability; do not trim/regenerate for equality by default. |
| Thresholds use train-only, source-balanced data | SPECIFIED, §15 | Fail if either source is empty; persist sample count and seed. |
| Final artifacts are reproducible for a fixed input pool, seed, policy, and software manifest | SPECIFIED, §§3, 6.4, 28 | Include hashes and policy versions in manifests/cache keys. |

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-PIPE-001` | specification clarification | Should A0-A5 be equalized/quotas enforced, or remain descriptive rare-arm diagnostics? | A: retain `balanced_arm_source`; B: align with §11 and remove equalization as a dataset requirement | B | Changes split selection, config, tests, and curriculum assumptions | Awaiting user approval |
| `DEC-PIPE-002` | specification clarification | What is the final acquisition eligibility predicate? | A: pre-Rulebook quality/signal; B: quality + signal + enabled Rulebook static eligibility | B when Rulebook is enabled | Changes when downloads/generation stop and dataset size | Awaiting user approval |
| `DEC-PIPE-003` | observable behavior | What happens when final eligible candidates are below source/split targets? | A: best-effort reduced dataset; B: strict failure with deficit report; C: explicit development-only reduced mode | B by default, C only for development | Prevents accidental training on an undersized dataset | Awaiting user approval |
| `DEC-PIPE-004` | implementation detail with scientific impact | How should additional candidates be obtained? | A: Waymo-only expansion; B: Waymo expansion plus PG valid-scenario replenishment; C: no automatic replenishment | B, bounded by configured caps and remote/seed inventory | Affects source composition and reproducibility | Awaiting user approval |
| `DEC-PIPE-005` | implementation detail | Should acquisition and eligibility be one monolithic loop or a resumable state machine with cached artifacts? | A: shell loop around full rebuild; B: resumable iteration manifest with incremental catalog/filter work | B | Improves restartability and runtime without changing semantics | Awaiting user approval |

No production implementation should depend on these decisions until they are
approved and, where scientific behavior changes, recorded in an ADR.

## 7. Proposed Design

Subject to the approval gates above, the recommended architecture is:

```text
source inventories and frozen manifest
→ candidate acquisition/generation iteration
→ raw catalog and quality report
→ Rulebook eligibility (when enabled)
→ one final eligible population
→ strict source/split target feasibility check
→ deterministic grouped split
→ train-only thresholds and arm classification
→ descriptive arm/source/split report
→ runtime views and validation
→ frozen dataset manifest
```

The acquisition controller should compare targets only against the final
eligible population. It should add unseen Waymo shards or fresh PG seeds,
rebuild only affected artifacts when their cache keys change, and repeat until
the source targets are feasible or an explicit cap/inventory exhaustion occurs.

The split builder should use exact per-source counts by default, preserving
group and seed constraints. If exact assignment is impossible, it should fail
before writing runtime views and persist the reason, available counts, group
sizes, and requested counts. A separately named development mode may allow a
reduced dataset, but it must be impossible to confuse it with the baseline
dataset.

Arm classification should remain after split/threshold preparation as required
by the specification. Arm reports should include counts by source and split,
but should not silently transform those counts into equalization quotas. The
provider/curriculum remains responsible for sampling or reporting rare arms.

Reports should distinguish these populations explicitly:

```text
raw catalog
quality-valid catalog
Rulebook-static-eligible catalog
final eligible catalog
requested source/split counts
assigned source/split counts
classified arm distribution
runtime-validated counts
```

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-PIPE-001` | `AC-PIPE-001`: all final runtime records use the common format and source policy | Existing catalog/runtime modules; restructure to preserve | Existing catalog/runtime integration tests plus end-to-end fixture | Partial |
| `REQ-PIPE-002` | `AC-PIPE-002`: exact configured source/split counts or explicit strict failure | `build_splits.py`, orchestration, manifest validation | New strict target and reduced-mode tests | Partial |
| `REQ-PIPE-003` | `AC-PIPE-003`: one final eligibility report reconciles quality, signal, and Rulebook counts | `waymo_pool.py`, Rulebook filter, new eligibility coordinator | New population reconciliation tests | Partial |
| `REQ-PIPE-004` | `AC-PIPE-004`: exact split assignment preserves no leakage | `pipeline.py`, `splits.py` | Existing split tests plus end-to-end mixed-source test | Implemented/needs re-verification |
| `REQ-PIPE-005` | `AC-PIPE-005`: thresholds use a deterministic balanced train subset only | `thresholds.py`, `compute_arm_thresholds.py` | Existing threshold tests plus target-count gate | Implemented/needs re-verification |
| `REQ-PIPE-006` | `AC-PIPE-006`: arms are reported without artificial equalization by default | `arms.py`, reports, config | New rare-arm regression test | Deviates currently |
| `REQ-PIPE-007` | `AC-PIPE-007`: empty requested provider combinations fail explicitly | `provider.py` and curriculum integration | Existing provider tests plus final-catalog availability test | Implemented/needs re-verification |
| `REQ-PIPE-008` | `AC-PIPE-008`: manifest and reports identify every population and policy version | `manifests.py`, reports, orchestration | New artifact schema/reconciliation test | Partial |

## 9. Test Strategy Defined Before Implementation

Mandatory new tests, pending decision approval:

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-PIPE-001` | Unit | Final eligibility composition | records split across invalid, partial, Rulebook-ineligible, and valid | counts and exclusion reasons reconcile exactly | `REQ-PIPE-003` |
| `TEST-PIPE-002` | Unit | Exact source/split target assignment | sufficient singleton and grouped records | exact counts; deterministic mapping | `REQ-PIPE-002`, `REQ-PIPE-004` |
| `TEST-PIPE-003` | Unit | Strict shortage behavior | insufficient final eligible Waymo or PG records | no runtime output; actionable deficit | `REQ-PIPE-002` |
| `TEST-PIPE-004` | Unit | Rare arms remain diagnostics | valid A4/A5 records plus common arms | records are not trimmed or reassigned to equalize arms | `REQ-PIPE-006` |
| `TEST-PIPE-005` | Unit | PG replenishment accounting | generation failures and post-filter invalid records | controller requests additional seeds until valid target or cap | `REQ-PIPE-002`, `REQ-PIPE-003` |
| `TEST-PIPE-006` | Unit/integration | Waymo replenishment accounting | new shard changes final eligibility after Rulebook | controller does not stop at pre-Rulebook target | `REQ-PIPE-002`, `REQ-PIPE-003` |
| `TEST-PIPE-007` | Integration | Full fixture pipeline | mixed Waymo/PG records with quality and signal failures | exact runtime counts, reports, thresholds, and no overlap | `REQ-PIPE-001`-`REQ-PIPE-008` |
| `TEST-PIPE-008` | Regression | Current failure shape | A4/A5 available only in a small pool | train allocation follows approved policy; no silent disappearance | `REQ-PIPE-006` |
| `TEST-PIPE-009` | Regression | Restart/cache behavior | interrupted acquisition iteration | resume is deterministic and does not duplicate shards/seeds | `REQ-PIPE-008` |

Commands available for validation:

```text
docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_scenarionet_pipeline.py tests/test_scenario_splits.py tests/test_waymo_pool.py tests/test_scenario_thresholds.py tests/test_scenario_manifests.py
make rulebook-v2-check
make lint
make format-check PYTHON_QUALITY_PATHS="src tests scripts"
git diff --check
bash -n setup.sh scripts/*.sh
```

The full real-data acquisition pipeline is not a mandatory design-validation
command until the approved architecture is implemented; it requires external
Waymo credentials and can overwrite active artifacts.

## 10. Milestones

- [x] M0: read the authoritative specification and inspect the current
      orchestration, selection, acquisition, and reports.
- [ ] M1: approve the scientific policy gates `DEC-PIPE-001` through
      `DEC-PIPE-004`.
- [ ] M2: implement one final eligibility/population accounting contract.
- [ ] M3: implement resumable deficit-driven acquisition/generation.
- [ ] M4: implement strict deterministic source/split construction and remove
      or isolate the obsolete best-effort arm balancing path.
- [ ] M5: reconcile manifests, reports, documentation, and tests.
- [ ] M6: run focused regressions, representative smoke, and final diff audit.

## 11. Progress And Findings Log

### 2026-07-16 — Initial audit

- Read the complete authoritative ScenarioNet v1 specification and repository
  authority index; no applicable ADRs exist.
- Verified that the current run has 5179 Rulebook-static-eligible records but
  only 2886 final usable records after quality and signal filtering.
- Verified that the current Waymo expansion stops before Rulebook filtering and
  only tracks total eligibility plus A4.
- Verified that PG generation has no post-filter replenishment loop.
- Verified that the current arm-balanced split is best-effort and does not
  enforce configured arm minimums.
- Verified that current artifacts contradict the existing implementation plan's
  claim of a 3500-record final dataset.
- Focused regression suite passed: 27 tests; shell syntax and whitespace checks
  also passed.
- Decision gates `DEC-PIPE-001` through `DEC-PIPE-005` are open.

Next step: obtain the policy decisions, then implement the smallest coherent
architecture and its regression matrix.

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-PIPE-001` | The specification treats rare arms as diagnostic and does not require equal arm counts. | Current code adds `balanced_arm_source`, arm minimums, optional global arm trimming, and a Waymo A4 acquisition target. | Historical attempts to solve source/arm imbalance incrementally. | Not approved in the authority index | Current pipeline tests and implementation plan |
| `DEV-PIPE-002` | Baseline source/split targets must be realized or explicitly reduced. | Current auto mode emits a reduced catalog and continues into runtime without a strict failure. | Best-effort group-aware selection. | Not approved | Split reports, runtime acceptance, pipeline tests |

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `docs/implementation/scenarionet_pipeline_restructure_v2_exec_plan.md` | Created | Audit findings, decision gates, and proposed redesign |
| `scripts/prepare_scenarionet_dataset.sh` | Planned review | Replace stage-local acquisition with final-eligibility orchestration |
| `scripts/expand_waymo_pool.sh` | Planned review | Make acquisition state/report compatible with final eligibility |
| `src/thesis_rl/scenarios/waymo_pool.py` | Planned review | Separate candidate inventory from final eligibility accounting |
| `src/thesis_rl/scenarios/pipeline.py` | Planned modification | Remove or isolate best-effort arm quotas and enforce approved split policy |
| `src/thesis_rl/cli/scenarios/build_splits.py` | Planned modification | Strict target gate and reconciled population report |
| `src/thesis_rl/cli/scenarios/generate_pg_dataset.py` | Planned review | Valid-scenario replenishment contract |
| `src/thesis_rl/cli/scenarios/filter_rulebook_v2_catalog.py` | Planned review | Stable final eligibility artifact contract |
| `conf/scenarios/pipeline_v1.yaml` | Planned modification | Single unambiguous acquisition/split/arm policy |
| `tests/` | Planned additions | Regression and end-to-end contract coverage |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| Read authoritative specification and authority index | PASS | 2026-07-16 | Complete ScenarioNet v1 specification read; no applicable ADRs found |
| Read-only artifact population counts | PASS | 2026-07-16 | 5179 Rulebook records; 3026 valid; 2886 final usable; train 1386 |
| Current focused ScenarioNet tests | PASS | 2026-07-16 | 27 passed in 0.53s in the Docker development environment |
| `bash -n setup.sh scripts/*.sh` and `git diff --check` | PASS | 2026-07-16 | Shell syntax and patch whitespace clean |
| Full real-data pipeline | NOT_RUN | 2026-07-16 | Requires external credentials and is not safe during design audit |

## 15. Final Reconciliation

This plan is not implemented. Requirements are currently `Partial` or
`Deviates currently` as listed in the traceability table. No readiness claim is
made for the current dataset until the approval gates are resolved and the
restructured pipeline is validated against the final eligible population.

Known limitations at this stage:

- the exact scientific policy for equal arm quotas versus descriptive rare-arm
  reporting is unresolved between current implementation history and the
  authoritative specification;
- current acquisition cannot guarantee post-Rulebook source targets;
- current documentation claims a historical successful run that does not match
  the current artifacts.

Deferred optional work: solver-based multi-constraint selection, custom PG VRU
generation, and global arm equalization remain out of scope unless separately
approved.
