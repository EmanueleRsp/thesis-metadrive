# ExecPlan: Empirical Holdouts, Dual Test Panels, and Waymo Co-location Grouping

## 1. Metadata

- **Feature:** replace the arm-balanced train/validation/test allocation with a
  holdout-first policy that reserves *empirical* (label-blind) validation and
  test pools before any arm balancing, adds an explicitly stratified challenge
  test panel alongside the empirical one, and strengthens the Waymo no-leakage
  grouping key.
- **Plan ID:** `SN-SPLIT-V1.2`
- **Authoritative specification:** `docs/specifications/scenarionet_integration_v1.2_specification.md`
  (ID `SCENARIONET-INTEGRATION`, version `1.2`, `APPROVED`, `Authoritative: YES`,
  approved 2026-07-31, amending `docs/specifications/scenarionet_integration_v1.1_specification.md`
  version `1.1`, approved 2026-07-16, for §5, §6.1, §6.2, §6.4, §6.5/§6.6,
  §6.8 only).
- **Secondary authoritative specification:** `docs/specifications/evaluation_protocol_v1.1_specification.md`
  (ID `EVAL-PROTOCOL`, version `1.1`, `APPROVED`, `Authoritative: YES`,
  approved 2026-07-31, amending `docs/specifications/evaluation_protocol_v1.0_specification.md`
  version `1.0` for `REQ-004`/`DEC-005`, `REQ-007`, `REQ-013`, §8 items 3/7
  only). `REQ-006`/`DEC-002` (`final.zip` as the sole official checkpoint) is
  **explicitly retained unchanged** — see `DEC-004` below.
- **Status:** `APPROVED` (M0 complete; `M2`-`M7` implementation not started)
- **Created:** 2026-07-31
- **Last updated:** 2026-07-31
- **Branch:** `scenarionet-implementation`
- **Related ADRs (existing):** ADR-001 (dataset policy), ADR-009 (PG targeted
  replenishment), ADR-012 (stratified source-arm split allocation) — ADR-012 is
  materially superseded by this plan for the validation/test splits.
- **Related ADRs:** ADR-037 (empirical holdout policy and dual test panels,
  `APPROVED` 2026-07-31), ADR-038 (Waymo map-based grouping key, adopted
  unconditionally per `DEC-002`, `APPROVED` 2026-07-31).
- **Owner:** repository maintainer

## 2. Objective And Scope

### 2.1 Problem

Three verified defects in the current frozen dataset and evaluation path:

1. **The holdouts are not empirical.** `_split_arm_targets`
   ([`src/thesis_rl/scenarios/pipeline.py:804`](../../src/thesis_rl/scenarios/pipeline.py))
   imposes near-uniform `A0..A5` quotas on *every* split, and
   [`pipeline.py:780`](../../src/thesis_rl/scenarios/pipeline.py) re-asserts
   this as a fatal invariant. Validation and test therefore measure a uniform
   macro-average over competences, not performance on the empirical eligible
   distribution of either source. This is defensible as a benchmark but is
   currently *described* as a split, with no empirical endpoint anywhere in the
   protocol.
2. **The evaluation panels re-balance a second time.**
   [`src/thesis_rl/scenarios/panel_manifest.py:1`](../../src/thesis_rl/scenarios/panel_manifest.py)
   builds panels by a "deterministic seed-driven balanced draw across the six
   scenario arms", and `EVAL-PROTOCOL` §269 fixes 100 validation and 300 test
   episodes. Only 300 of the 1000 frozen test scenarios are ever evaluated.
   Changing the split policy without changing the panel policy would change
   nothing observable.
3. **Waymo group-disjointness is effectively absent.**
   [`pipeline.py:53-57`](../../src/thesis_rl/scenarios/pipeline.py) discards
   `source_log_id` whenever it starts with `training_20s.tfrecord-` and falls
   back to `f"scenario:{uid}"`. In the frozen index *every* Waymo record has a
   TFRecord shard name in that field, so the Waymo grouping is per-scenario,
   i.e. no grouping at all. This conforms to `SCENARIONET-INTEGRATION` §6.1
   (which anticipated the absence of a superior identifier) but leaves
   co-located 20 s windows free to straddle train and test. Any empirical test
   endpoint built on top of it would be optimistic for leakage, not only for
   balancing.

### 2.2 Observable capability after this change

- A frozen dataset whose validation and test pools are drawn **before** and
  **independently of** the `A0..A5` labels, from a deterministically permuted
  candidate stream, per source.
- Three declared, frozen, hashed test panels: `test_waymo_empirical` (primary
  generalization endpoint), `test_arm_stratified` (competence endpoint, the
  only one with sufficient per-arm power), `test_pg` (procedural-generalization
  endpoint, secondary).
- Validation panels declared per source, with the Waymo empirical panel as the
  primary learning-curve panel.
- A training pool built from the residual population with **per-arm minimums**
  instead of exact equal quotas, so that `UniformScenarioProvider` and
  `ArmUniformScenarioProvider`
  ([`src/thesis_rl/scenarios/provider.py:47`](../../src/thesis_rl/scenarios/provider.py),
  [`:159`](../../src/thesis_rl/scenarios/provider.py)) stop collapsing onto the
  same sampling distribution and remain distinguishable baselines.
- A Waymo grouping key that reflects map co-location, *if and only if* the
  `M1` audit shows material co-location.
- Post-hoc `A0..A5` distribution reporting for every split and panel.

### 2.3 In scope

- `src/thesis_rl/scenarios/`: `pipeline.py`, `splits.py`, `panel_manifest.py`,
  `waymo.py`, `records.py`, `pg/profiles.py`, `pg/replenishment.py`.
- `src/thesis_rl/cli/scenarios/`: `build_splits.py`, `build_panel_manifest`
  entry point (`scripts/build_panel_manifest.py`), `freeze_dataset.py`.
- `scripts/prepare_scenarionet_dataset.sh` argument surface.
- `conf/` dataset target keys.
- Split manifest schema (new version), frozen selection index schema.
- Specification amendment v1.2, EVAL-PROTOCOL panel amendment, ADRs,
  `docs/project_index.md`.

### 2.4 Out of scope

- Changing `EVAL-PROTOCOL` `REQ-006`/`DEC-002`: `checkpoints/final.zip` remains
  the sole official checkpoint (`DEC-004` below).
- Changing arm definitions, `A0..A5` fixed thresholds
  ([`arms.py:19-26`](../../src/thesis_rl/scenarios/arms.py)), the Q40/Q75
  diagnostic tags ([`thresholds.py:74`](../../src/thesis_rl/scenarios/thresholds.py)),
  quality filters, or Rulebook eligibility.
- Using the official Waymo `validation`/`testing` splits. `training_20s`
  remains the sole Waymo source (`splits.py:46`).
- ACL v2.0 arm-selection semantics (separate `UNDER_REVIEW` specification).

### 2.5 Compatibility

**Breaking by construction.** The new dataset gets a new `selection_hash` and a
new split manifest schema version. Runs produced against the current frozen
dataset (`selection_hash`
`7c0d6f3b15ca69ff79dbb3a0daefefdae326254065a9792c0b73af1ffa730fb5`,
created 2026-07-31) **cannot be resumed against, or aggregated with, runs on
the new dataset**. This is acceptable only because official experiments have
not started; that precondition must be reconfirmed at approval time.

## 3. Authoritative Requirements

Requirements marked `AMEND` require the v1.2 specification amendment to be
approved before implementation.

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-001` | Validation and test pools are drawn from a deterministic pseudo-random permutation of the eligible candidate stream, per source, with `A0..A5` labels ignored by the allocator | `AMEND` §6.6 (new "holdout-first" policy), generalizes existing §6.8 |
| `REQ-002` | The test pool is reserved first, the validation pool second, and both are frozen before any training-pool construction | `AMEND` §6.5 |
| `REQ-003` | An arm-stratified challenge test pool is reserved after the empirical holdouts, disjoint from them and from train | `AMEND` §6.5/§6.8 |
| `REQ-004` | The training pool is built from the residual eligible population under **per-arm minimums** and exact per-source totals, not exact equal arm quotas | `AMEND` §6.6 (replaces constraint 5 of `balanced_arm_source`) |
| `REQ-005` | Any post-freeze scenario acquisition may enter the training pool only; validation and test pools are immutable once frozen | `AMEND` §6.5, §6.7 |
| `REQ-006` | The Waymo grouping key uses map co-location evidence when the audit demonstrates shared-map dependence between scenarios; the chosen field and its evidence are recorded in the split manifest | §6.1 (existing "strongest equivalent metadata available" clause) + `AMEND` |
| `REQ-007` | PG holdout scenarios are generated from a declared, frozen generation-profile mixture with a seed range disjoint from every other split | `AMEND` §4/§6.2 |
| `REQ-008` | Every split and every panel records its observed `A0..A5` distribution and its source composition as post-hoc reporting | `AMEND` §6.4 |
| `REQ-009` | Three frozen, hashed test panels exist: `test_waymo_empirical`, `test_arm_stratified`, `test_pg`; empirical panels are drawn without arm balancing | `EVAL-PROTOCOL` `REQ-004`/`DEC-005` `AMEND` |
| `REQ-010` | Validation panels are declared per source; `validation_waymo_empirical` is the primary learning-curve panel | `EVAL-PROTOCOL` `REQ-013` `AMEND` |
| `REQ-011` | No primary metric aggregates Waymo and PG; any macro-source average is explicitly labelled as an artificial 50/50 summary | `EVAL-PROTOCOL` §7 `AMEND` |
| `REQ-012` | `checkpoints/final.zip` remains the sole official checkpoint; no panel introduced here participates in checkpoint selection | `EVAL-PROTOCOL` `REQ-006`/`DEC-002`, **unchanged** |
| `REQ-013` | Group disjointness holds pairwise across train, validation-empirical, test-empirical, and test-stratified pools | §6.3, extended to the new pools |

## 4. Current Repository Analysis

| Statement | Label | Evidence |
|---|---|---|
| Near-uniform arm quotas are imposed per split and re-asserted as a fatal invariant | `VERIFIED` | `pipeline.py:804` (`_split_arm_targets`), `pipeline.py:780` |
| The frozen dataset is train 2000 (1000 W + 1000 PG), validation 500 (250+250), test 1000 (500+500) | `VERIFIED` | `data/scenarionet/frozen/scenario_selection_index.json` `split_manifest.counts` |
| Evaluation panels re-balance across arms and use 100 / 300 episodes | `VERIFIED` | `panel_manifest.py:1` module docstring; `EVAL-PROTOCOL` §269 |
| `test_waymo_natural` is specified (§6.8) but has zero implementation | `VERIFIED` | no occurrence of `waymo_natural` in `src/`, `tests/`, `conf/`, `scripts/` |
| Waymo grouping degenerates to per-scenario | `VERIFIED` | `pipeline.py:53-57`; every Waymo record in the frozen index has `source_log_id` matching `training_20s.tfrecord-*` |
| The converted Waymo eligible pool contains exactly 26 `A0`, 143 `A1`, 144 `A2` scenarios | `VERIFIED` (derived) | For these arms the balanced selector fell short of the ideal 50/50 waymo share, so it consumed the entire available capacity: selected Waymo totals across the three splits are `A0`=26, `A1`=143, `A2`=144 |
| Waymo `A3`, `A4`, `A5` eligible capacity is at least 463 / 583 / 391 | `VERIFIED` (lower bound) | same index; these arms were quota-limited, not capacity-limited |
| A 300-scenario empirical Waymo panel would contain on the order of 4 `A0` and 20 `A1` scenarios | `INFERRED` from the capacities above | motivates `REQ-003` |
| The PG profile mixture in the frozen inventory (P5 48%, P0 18%, P1 14%, P2 10%, P3 10%) is an artifact of arm-deficit-driven replenishment, not a neutral generative distribution | `VERIFIED` | `pg/replenishment.py:74` (`plan_profile_counts`), ADR-009; inventory counts in the frozen index |
| Waymo acquisition is shard-batched and arm-agnostic *within* a shard; only the stopping rule is deficit-driven | `VERIFIED` | §6.7 procedure; `waymo_pool.py:56` (`summarize_waymo_pool`) |
| `UniformScenarioProvider` and `ArmUniformScenarioProvider` both exist as baselines | `VERIFIED` | `provider.py:47`, `provider.py:159` |
| Converted scenarios expose `map_features` as a mapping, usable for a co-location fingerprint | `VERIFIED` | `features.py:132`, `features.py:220`, `features.py:379` |
| The converted candidate pool (parquet catalog) is required to re-split without re-converting; it is not present on the host filesystem, only inside the pipeline container volume | `AWAITING_CONFIRMATION` | `artifacts.catalog.relative_path` = `catalog/scenario_catalog.parquet`; `conf/env/scenarionet.yaml:7` resolves the data root to `/workspace/data` |
| Official experiments have not yet started, so a dataset re-freeze does not invalidate recorded results | `AWAITING_CONFIRMATION` | user confirmation required at approval time |

## 5. Assumptions And Invariants

- **Determinism.** Every draw introduced here is driven by an explicit integer
  seed recorded in the split manifest. Re-running the pipeline with the same
  seeds, the same candidate pool, and the same filters must reproduce identical
  UID sets and identical panel hashes.
- **Ordering independence.** The candidate stream must be permuted *before* any
  content is inspected, so that shard order, file order, and acquisition order
  cannot correlate with the holdout membership.
- **Label blindness.** The empirical holdout allocator must not read
  `primary_arm`, `tags`, `low_traffic`, or `dense_traffic`. This is enforced by
  construction (the allocator receives a projection of the record) and asserted
  by test.
- **Freeze order.** test-empirical → validation-empirical → test-stratified →
  train. No later stage may move a record into an earlier one.
- **Threshold provenance.** Q40/Q75 remain computed on the training pool only
  (`thresholds.py:44` already rejects non-train entries) and are applied to the
  holdouts afterwards to produce comparable diagnostic tags. They never
  influence selection, and they never move a scenario between `A0..A5`
  (`arms.py:19-26` uses fixed thresholds).
- **Structural empty cell.** `A4_vru x PG` remains structurally empty; the
  `A0/A1/A2 x Waymo` scarcity documented in §4 is an empirical property of the
  pool and is reported, never corrected by relabelling.
- **Units and frames** are unchanged by this plan.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-001` | Specification deviation | Validation/test are currently arm-balanced by construction | (A) holdout-first empirical pools + separate stratified pool; (B) empirical only; (C) status quo with corrected wording | **A** | Replaces ADR-012 for validation/test; new dataset fingerprint | **Approved 2026-07-31** (user) |
| `DEC-002` | Specification deviation | Waymo grouping degenerates to per-scenario | (A) add map-co-location grouping unconditionally; (B) defer to a separate plan; (C) accept and document | **A, unconditionally** — the fix costs the same regardless of the audit's outcome (the fingerprint used to *measure* co-location is the same one used to *fix* it), so there is no scenario where waiting for a magnitude estimate changes the implementation decision. The `M1` census is retained only as descriptive evidence for the thesis limitations section, not as a gate | May reduce usable Waymo capacity; changes group-disjointness semantics | **Approved 2026-07-31** (user); no longer conditional on `M1` completing |
| `DEC-003` | Specification clarification | There is no empirical PG distribution to sample: the pool mixture is a replenishment artifact | (A) declare and freeze an explicit generation-profile mixture for holdouts; (B) sample the existing biased stream; (C) drop PG from the holdouts | **A**, with an **equiprobable mixture**: 20% each over `P0_simple`, `P1_vehicle_interaction`, `P2_merge_or_roundabout`, `P3_intersection`, `P5_complex_mixed` | Requires PG regeneration with a disjoint seed range; the resulting arm distribution is an *observed consequence* of the generator, never a target | **Approved 2026-07-31** (user) |
| `DEC-004` | Specification clarification | Whether to move the official checkpoint from `final.zip` to a validation-selected checkpoint | (A) keep `final.zip`; (B) `final.zip` + mean of last K evaluations; (C) argmax on validation | **A** | None; `EVAL-PROTOCOL` `DEC-002` retained | **Approved 2026-07-31** (user) |
| `DEC-005` | Specification clarification | Split and panel sizes | see §7.3 | §7.3 table as written | Governs training capacity and evaluation cost | **Approved 2026-07-31** (user) |
| `DEC-006` | Specification clarification | Periodic-validation cost: adding a PG validation panel doubles the cost of every evaluation point (60 points x 3 seeds x N conditions) | (A) Waymo panel every 25k steps, PG panel at a coarser cadence; (B) both every 25k steps; (C) PG only at final test | **A**, with `validation_waymo_empirical` every 25,000 steps (primary curve) and `validation_pg` every 100,000 steps (diagnostic, ~15 points) | Compute budget +25% instead of +100%; PG curve resolution reduced by 4x | **Approved 2026-07-31** (user) |
| `DEC-007` | Blocking technical issue | Whether the converted candidate pool and parquet catalog are still available so that re-splitting does not require re-downloading and re-converting Waymo | (A) reuse existing pool; (B) re-acquire | **A** | Determines whether `M1`/`M3` are hours or days of work | **Resolved 2026-07-31**: confirmed present at `/scratch/e.respino/thesis-metadrive/data/scenarionet/` (`catalog/scenario_catalog.parquet`, `waymo/` 65G, `pg/` 7.0G) |

No open gates remain outside the `M0` specification-amendment approval itself.
`M1` is unblocked and read-only; `M2` onward must not start before the v1.2
amendment is approved.

`DEC-006` requires the periodic-validation cadence to become panel-specific:
`EVAL-PROTOCOL` §679 currently fixes a single 25,000-step cadence for one
100-scenario panel. The amendment must express the cadence per panel rather
than globally.

## 7. Proposed Design

### 7.1 Allocation pipeline

```text
frozen generation profiles, filters, arm rules, fixed arm thresholds
                              |
                    build candidate pool (per source)
                              |
        conversion + technical validation + quality filters
        + rulebook eligibility + feature extraction
                              |
                    eligible pool (per source)
                              |
        deterministic pseudo-random permutation of GROUPS  <-- holdout_seed
                              |
   +--------------------------+--------------------------+
   |  reserve N_test_empirical groups  (labels ignored)   |
   |  reserve N_val_empirical  groups  (labels ignored)   |
   +--------------------------+--------------------------+
                              |
                    FREEZE empirical holdouts
                              |
        reserve arm-stratified challenge test pool from the residual
                              |
                    FREEZE stratified holdout
                              |
        build training pool from the residual, per-arm MINIMUMS
        + exact per-source totals
                              |
        compute Q40/Q75 on train only, freeze, apply to holdouts as tags
                              |
        report observed A0..A5 distributions for every split and panel
```

New public helpers in `pipeline.py`:

- `reserve_empirical_holdouts(entries, *, per_source_counts, seed) -> HoldoutReservation`
  — operates on a label-free projection of each record; returns reserved UID
  sets plus the permutation seed and the group ordering digest.
- `reserve_stratified_holdout(entries, *, per_arm_counts, seed)` — the existing
  balanced machinery, restricted to the residual population.
- `assign_training_pool_with_arm_minimums(entries, *, source_totals, arm_minimums, seed)`
  — replaces `assign_arm_balanced_splits_to_targets` for the training split
  only. The existing function is retained for the stratified pool.

`assign_arm_balanced_splits_to_targets` is **not** deleted: it becomes the
stratified-pool allocator. The invariant at `pipeline.py:780` is narrowed to
apply to the stratified pool only.

### 7.2 Grouping

`waymo_group_id` ([`waymo.py:148`](../../src/thesis_rl/scenarios/waymo.py))
gains a map-co-location fallback: when no true log/segment id is available, the
group key becomes a stable digest over the scenario's map identity — the sorted
map-feature identifiers together with a quantized spatial anchor, so that two
20 s windows recorded over the same road geometry hash to the same group. The
exact digest definition is fixed in `M1` from the audit evidence and recorded
in the split manifest under `grouping.waymo.evidence`.

If `M1` finds no material co-location, `DEC-002` resolves to "document as a
limitation" and `M2` is dropped, with the finding recorded in §11.

### 7.3 Sizes (`DEC-005`, awaiting approval)

Recommended allocation, assuming the existing 3500-scenario budget is retained
and noting that today 700 of the 1000 frozen test scenarios are never
evaluated:

| Pool | Waymo | PG | Derived panels |
|---|---:|---:|---|
| `test_empirical` | 400 | 300 | `test_waymo_empirical` 300 (primary), `test_pg` 200 (secondary) |
| `test_stratified` | ~180 | ~120 | `test_arm_stratified` 300 (50 per arm) |
| `validation` | 150 | 150 | `validation_waymo_empirical` 100 (primary curve), `validation_pg` 100 |
| `train` | residual | residual | per-arm minimums, sampler-controlled exposure |

Rationale: statistical precision is governed by the *panel* size and, far more,
by the three training seeds — not by the pool size. Enlarging the pools beyond
what the panels consume buys nothing, while the residual capacity is worth more
in training. The stratified pool is sized so that every arm reaches 50 test
episodes, which is the minimum at which a per-arm comparison between ACL and
the baselines is worth reporting.

Feasibility constraint from §4: the Waymo pool holds only 26 `A0`, 143 `A1`,
144 `A2`. The stratified pool's `A0` cell will therefore be PG-dominated, and
`A0 x Waymo` may be structurally unfillable at 50. This is reported, not
corrected. `M3` must fail loudly with a capacity report rather than silently
relaxing a quota.

### 7.4 Panels

`panel_manifest.py` gains a `draw_policy` field with two values:

- `arm_balanced` (existing behaviour, used by `test_arm_stratified`);
- `empirical` (new: uniform draw over the pool's groups, labels ignored,
  source-conditioned).

Every manifest records `draw_policy`, `source`, `split`, `size`, the observed
per-arm counts, and the ordered UID SHA-256. The existing hash and ordering
semantics are preserved.

### 7.5 Errors and fallbacks

No silent fallback anywhere. Insufficient capacity in any pool is a fatal
error carrying a per-source-per-arm capacity report. A missing or mismatched
panel hash invalidates the evaluation, as today.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-001` | `AC-001` | `pipeline.py::reserve_empirical_holdouts` | `tests/test_scenario_splits.py::test_empirical_holdout_ignores_arm_labels` | Planned |
| `REQ-002` | `AC-002` | `pipeline.py` freeze order | `::test_holdouts_are_reserved_before_training_pool` | Planned |
| `REQ-003` | `AC-003` | `pipeline.py::reserve_stratified_holdout` | `::test_stratified_pool_disjoint_from_empirical` | Planned |
| `REQ-004` | `AC-004` | `pipeline.py::assign_training_pool_with_arm_minimums` | `::test_training_pool_respects_arm_minimums_not_exact_quotas` | Planned |
| `REQ-005` | `AC-005` | `freeze_dataset.py` | `::test_post_freeze_acquisition_enters_train_only` | Planned |
| `REQ-006` | `AC-006` | `waymo.py::waymo_group_id` | `tests/test_waymo_grouping.py::test_colocated_windows_share_group` | Planned |
| `REQ-007` | `AC-007` | `pg/profiles.py`, `generate_pg_dataset.py` | `tests/test_pg_holdout_mixture.py::test_holdout_seed_range_disjoint` | Planned |
| `REQ-008` | `AC-008` | split manifest writer | `::test_manifest_reports_observed_arm_distribution` | Planned |
| `REQ-009` | `AC-009` | `panel_manifest.py` | `tests/test_panel_manifest.py::test_empirical_panel_is_not_arm_balanced` | Planned |
| `REQ-010` | `AC-010` | `conf/`, evaluator wiring | `tests/test_hydra_preset_run_configs.py` | Planned |
| `REQ-011` | `AC-011` | analysis entry point | `::test_no_primary_metric_aggregates_sources` | Planned |
| `REQ-012` | `AC-012` | unchanged | existing `EVAL-PROTOCOL` tests | Planned |
| `REQ-013` | `AC-013` | `splits.py::assert_no_group_overlap` extension | `::test_pairwise_group_disjointness_across_four_pools` | Planned |

## 9. Test Strategy Defined Before Implementation

### Acceptance criteria

- `AC-001` Given a synthetic eligible pool whose arm labels are permuted
  arbitrarily, the reserved empirical holdout UID set is **identical** for every
  permutation, at fixed seed.
- `AC-002` Reserved holdout UIDs are byte-identical whether or not the
  training-pool construction runs afterwards.
- `AC-003` The stratified pool shares no group with either empirical holdout or
  with train.
- `AC-004` The training pool satisfies every configured per-arm minimum and its
  exact per-source totals; arm counts are *not* required to be equal, and a
  configuration with unequal arm counts is accepted.
- `AC-005` Adding new eligible scenarios after the freeze changes the training
  pool only; the holdout UID sets and their hashes are unchanged.
- `AC-006` Two synthetic scenarios sharing the same map identity receive the
  same group id; two with different map identities do not.
- `AC-007` PG holdout seeds intersect no other split's seed range, and the
  holdout profile mixture matches the declared frozen mixture.
- `AC-008` Every split and panel record carries observed per-arm and per-source
  counts.
- `AC-009` An `empirical` panel's per-arm distribution matches the pool's
  distribution within sampling error and is *not* uniform; an `arm_balanced`
  panel remains uniform.
- `AC-010` The Hydra preset matrix resolves for every new panel key.
- `AC-011` No primary metric in the analysis output aggregates Waymo and PG;
  any macro-source average is labelled.
- `AC-012` The official checkpoint remains `final.zip`; no new panel is
  reachable from any checkpoint-selection code path.
- `AC-013` Pairwise group intersection is empty across all four pools.

### Mandatory test matrix

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-001` | Unit | Label blindness of the empirical allocator | Synthetic pool, arm labels permuted 10 ways, fixed seed | Identical reserved UID set in all 10 | `REQ-001` |
| `TEST-002` | Unit | Stream permutation precedes inspection | Pool whose file order correlates perfectly with arm | Reserved arm distribution matches the pool distribution, not the file order | `REQ-001` |
| `TEST-003` | Unit | Freeze ordering | Pipeline run with and without the training stage | Identical holdout hashes | `REQ-002` |
| `TEST-004` | Unit | Disjointness of the stratified pool | Synthetic pool | Empty pairwise group intersection | `REQ-003`, `REQ-013` |
| `TEST-005` | Unit | Per-arm minimums accepted with unequal arm counts | Pool with a scarce arm | Success, with the observed distribution reported | `REQ-004` |
| `TEST-006` | Unit | Capacity failure is loud | Pool that cannot satisfy a minimum | `ValueError` carrying the per-source-per-arm capacity report | `REQ-004` |
| `TEST-007` | Unit | Post-freeze acquisition | Frozen holdouts + new eligible entries | Holdout hashes unchanged; only train grows | `REQ-005` |
| `TEST-008` | Unit | Map co-location grouping | Two synthetic scenarios, same/different map identity | Same/different group id | `REQ-006` |
| `TEST-009` | Unit | PG holdout seed disjointness and mixture | Declared mixture, holdout seed range | Disjoint ranges; mixture matches within rounding | `REQ-007` |
| `TEST-010` | Unit | Empirical panel is not arm-balanced | Pool with a skewed arm distribution | Panel distribution tracks the pool; uniformity test fails | `REQ-009` |
| `TEST-011` | Unit | Arm-balanced panel unchanged | Existing fixture | Existing expectations hold | `REQ-009` |
| `TEST-012` | Unit | Panel hash and ordering semantics preserved | Existing fixture | Existing expectations hold | `REQ-009` |
| `TEST-013` | Integration | Full allocation on a synthetic multi-source pool | Deterministic fixture | All four pools materialize with the declared sizes and the recorded diagnostics | `REQ-001`..`REQ-005`, `REQ-008` |
| `TEST-014` | Determinism | Re-run reproducibility | Same seeds and pool | Identical UID sets and identical panel hashes | `REQ-001`, `REQ-009` |
| `TEST-015` | Compatibility | Hydra preset matrix | New panel keys | All presets resolve | `REQ-010` |
| `TEST-016` | Regression | Checkpoint policy untouched | Existing `EVAL-PROTOCOL` fixtures | `final.zip` remains the sole official checkpoint | `REQ-012` |
| `TEST-017` | Smoke | End-to-end training smoke on the regenerated dataset | `make smoke` | Passes | all |

### Available commands

- Focused tests: `uv run --no-sync python -m pytest -q tests/test_scenario_splits.py tests/test_panel_manifest.py`
- Full suite through the primary environment: `make test`
- Lint (focused): `make lint PYTHON_QUALITY_PATHS="src/thesis_rl/scenarios tests/test_scenario_splits.py"`
- Format check (focused): `make format-check PYTHON_QUALITY_PATHS="..."`
- Compose validation: `make config`, `make config-gpu`
- End-to-end smoke: `make smoke`
- Whitespace: `git diff --check`

Type checking: no global mypy target is configured; new public interfaces in
`pipeline.py` and `panel_manifest.py` will carry full annotations, checked ad
hoc only.

## 10. Milestones

### M0 — Specification amendment and ADRs (approval gate) — COMPLETE

- [x] Draft `docs/specifications/scenarionet_integration_v1.2_specification.md`
      amending §5 (targets), §6.1 (grouping), §6.2 (PG seeds and mixture),
      §6.4 (manifest schema), §6.5/§6.6 (holdout-first policy, arm minimums),
      §6.8 (promotion of `test_waymo_natural` to a primary endpoint).
- [x] Draft `docs/specifications/evaluation_protocol_v1.1_specification.md`
      (`REQ-004`/`DEC-005`, `REQ-007`, `REQ-013`, §8 cadence), explicitly
      reaffirming `REQ-006`/`DEC-002`.
- [x] Resolve `DEC-007` (data confirmed present, no re-acquisition needed).
- [x] Resolve `DEC-002` unconditionally (map-based Waymo grouping adopted
      regardless of audit magnitude; see the Decisions table).
- [x] Draft and approve ADR-037 (empirical holdout policy and dual test
      panels) and ADR-038 (Waymo map-based grouping key).
- [x] Obtain explicit user approval of both specification documents
      ("mi sembra vada tutto bene", 2026-07-31); promoted each to
      `APPROVED`/`Authoritative: YES`, removed `_UNDER_REVIEW` from the
      filenames, updated `docs/project_index.md`.

### M1 — Waymo co-location audit (read-only, optional/descriptive; no longer a gate)

`DEC-002`/ADR-038 already resolved unconditionally; this milestone produces
descriptive evidence for the thesis limitations section only and does not
block `M2`. A partial run (script at
`/tmp/.../scratchpad/m1_colocation_audit.py`, this-session working copy) was
interrupted at 35,000/54,104 scenarios once it was established that no
result would change the `M2` decision.

- [ ] Resume or re-run the script under `docs/audits/waymo_colocation_2026-07-31/`,
      computing a map identity digest for every converted Waymo scenario.
- [ ] Report: number of distinct map groups, group size distribution, and how
      many currently-frozen v1.1 train/test pairs share a map group.
- [ ] Record the count in `scenarionet_integration_v1.2` §3.6 as descriptive
      evidence (already written to accept this as non-blocking).
- **No production code changes in this milestone.**

### M2 — Grouping key

- [ ] `waymo.py::waymo_group_id` map fallback; `records.py` field for the
      digest; manifest evidence field.
- [ ] `TEST-008`; focused lint/format.

### M3 — Holdout-first allocator

- [ ] `reserve_empirical_holdouts`, `reserve_stratified_holdout`,
      `assign_training_pool_with_arm_minimums` in `pipeline.py`.
- [ ] Narrow the `pipeline.py:780` invariant to the stratified pool.
- [ ] Extend `splits.py::assert_no_group_overlap` to four pools.
- [ ] `TEST-001`..`TEST-007`, `TEST-013`, `TEST-014`.

### M4 — PG holdout generation

- [ ] Declared frozen profile mixture; disjoint holdout seed range in
      `pg/profiles.py` and `generate_pg_dataset.py`.
- [ ] `TEST-009`.

### M5 — Panels

- [ ] `draw_policy` in `panel_manifest.py`; empirical draw; extended manifest
      fields.
- [ ] `scripts/build_panel_manifest.py` argument surface.
- [ ] `TEST-010`..`TEST-012`.

### M6 — Configuration, CLI, and manifest schema

- [ ] `build_splits.py`, `freeze_dataset.py`, `prepare_scenarionet_dataset.sh`,
      `conf/` target keys, split manifest schema version bump.
- [ ] `TEST-015`; `make config`, `make config-gpu`.

### M7 — Regeneration, freeze, reconciliation

- [ ] Regenerate the dataset; record the new `selection_hash` and every panel
      hash.
- [ ] Audit report with the observed `A0..A5` distribution of every split and
      panel, and the source composition.
- [ ] `make test`, `make smoke`, focused lint/format, `git diff --check`.
- [ ] Reconcile every `REQ`/`AC` against code and tests; update
      `docs/project_index.md`; record the incompatibility of prior runs.

## 11. Progress And Findings Log

### 2026-07-31 — Plan created

Completed: repository analysis of the current split, panel, grouping, and
threshold behaviour; derivation of the Waymo per-arm eligible capacity from the
frozen selection index; identification of the three defects in §2.1.

Findings:

1. The evaluation panels, not the splits, determine what is measured. Any
   change confined to the split policy would have been observationally inert.
   This reframes the whole task around `panel_manifest.py`.
2. `SCENARIONET-INTEGRATION` §6.8 already specifies `test_waymo_natural` with
   exactly the required semantics (group-disjoint, no arm rebalancing, reported
   separately, never used for calibration or checkpoint selection), but marks it
   optional and disabled by default, and it has never been implemented. The
   amendment is a promotion of an existing clause rather than a new concept.
3. The Waymo grouping key degenerates to per-scenario. This *conforms* to §6.1,
   which anticipated the absence of a superior identifier, so it is a
   specification gap rather than an implementation bug — but it undermines any
   empirical endpoint and must be revisited before, not after, the re-split.
4. The Waymo eligible pool contains only 26 `A0`, 143 `A1`, and 144 `A2`
   scenarios. A purely empirical 300-scenario Waymo test panel would therefore
   carry roughly 4 `A0` and 20 `A1` episodes, which is why the stratified
   challenge panel is retained as a co-endpoint rather than dropped.
5. There is no empirical PG distribution available: the frozen PG mixture is an
   artifact of `plan_profile_counts` deficit targeting.

Decisions taken (user, 2026-07-31): `DEC-001` = A (empirical + stratified test
panels); `DEC-002` = A, conditional on the `M1` evidence; `DEC-003` = A with an
equiprobable five-profile PG mixture; `DEC-004` = A (`final.zip` retained);
`DEC-005` = the §7.3 table as written; `DEC-006` = A (Waymo validation every
25k steps, PG validation every 100k steps).

Decisions needed: `DEC-007` only — whether the converted candidate pool and
`catalog/scenario_catalog.parquet` are still present in the pipeline container
volume. The host filesystem does not expose them (`conf/env/scenarionet.yaml:7`
resolves the data root to `/workspace/data`), so this must be confirmed before
`M1` can run.

Next step: confirm `DEC-007`, then run the `M1` audit (read-only); `M0`
drafting proceeds in parallel. No production code changes until the v1.2
amendment is approved.

### 2026-07-31 — DEC-007 resolved, M1 reframed, M0 drafted

`DEC-007` resolved by direct inspection: the converted candidate pool is
present at `/scratch/e.respino/thesis-metadrive/data/scenarionet/`
(`catalog/scenario_catalog.parquet`, `waymo/` 65G across 54,104 converted
scenarios in 768 shards, `pg/` 7.0G), reachable through the existing
`dataset-pipeline` Docker Compose service via `HOST_DATA_DIR`. No
re-acquisition is required.

An `M1` co-location audit was started (label-free map-identity fingerprint
over all 54,104 converted Waymo scenarios, cross-checked against the frozen
v1.1 split for straddling groups) and reached 35,000/54,104 files before
being stopped deliberately. Reframing: `DEC-002` does not actually depend on
the audit's outcome, because the fingerprint used to *measure* co-location
is the same fingerprint needed to *fix* it (§3.2 of the v1.2 amendment) — so
no possible measured magnitude would change the implementation decision.
`DEC-002` is resolved to **adopt map-based Waymo grouping unconditionally**;
the audit is retained only as optional descriptive evidence for the thesis
limitations section (`scenarionet_integration_v1.2` §3.6), not as a
blocking prerequisite. This removed one full milestone's worth of blocking
wait time from the critical path.

Drafted both `M0` amendment documents:

- `docs/specifications/scenarionet_integration_v1.2_specification_UNDER_REVIEW.md`
  — amends v1.1 §5, §6.1, §6.2, §6.4, §6.5/§6.6, §6.8.
- `docs/specifications/evaluation_protocol_v1.1_specification_UNDER_REVIEW.md`
  — amends `EVAL-PROTOCOL` v1.0 `REQ-004`/`DEC-005`, `REQ-007`, `REQ-013`,
  §8 items 3 and 7; explicitly reaffirms `REQ-006`/`DEC-002` (`final.zip`)
  unchanged.

`docs/project_index.md` updated with pointers to both `UNDER_REVIEW`
documents under the existing `ScenarioNet integration` and `Evaluation and
algorithm comparison protocol` rows.

Decisions needed: none remain outside the M0 approval gate itself — the user
must read and approve (or request changes to) the two `UNDER_REVIEW`
documents before `M1`(descriptive)/`M2`-`M7` production work begins.

Next step: user review and approval of both `UNDER_REVIEW` specification
documents. On approval: promote both to `APPROVED`/`Authoritative: YES`,
drop `_UNDER_REVIEW` from filenames, move under `docs/specifications/`
(already there), draft ADR-037/ADR-038, then begin `M2` (grouping key).

### 2026-07-31 — M0 approved and closed

User approval received ("mi sembra vada tutto bene") for both
`UNDER_REVIEW` documents. Completed the promotion sequence:

- `scenarionet_integration_v1.2_specification_UNDER_REVIEW.md` renamed to
  `scenarionet_integration_v1.2_specification.md`; `Status: APPROVED`,
  `Authoritative: YES`, approval record added.
- `evaluation_protocol_v1.1_specification_UNDER_REVIEW.md` renamed to
  `evaluation_protocol_v1.1_specification.md`; `Status: APPROVED`,
  `Authoritative: YES`, approval record added.
- `docs/decisions/ADR-037-empirical-holdout-policy-and-dual-test-panels.md`
  and `docs/decisions/ADR-038-waymo-map-based-grouping-key.md` created,
  `Status: Approved`.
- `docs/project_index.md` updated: both document-authority rows now point to
  the approved v1.2/v1.1 amendments with ADR references; the ADR table gained
  the ADR-037/ADR-038 rows.
- This ExecPlan's own metadata updated: `Status: APPROVED`, specification
  references repointed to the approved (non-`_UNDER_REVIEW`) paths.

M0 is complete. No open decision gates remain. `M1` is now explicitly
optional/descriptive (not a prerequisite for `M2`, per the unconditional
resolution of `DEC-002`/ADR-038).

Next step: begin `M2` (Waymo map-identity grouping key implementation) or,
at the user's preference, `M1` first purely for the descriptive audit count
— both are unblocked and neither depends on the other.
