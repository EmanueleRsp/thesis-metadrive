# Specification amendment: Dual test panels, per-source primary metrics, and panel-specific validation cadence

## Metadata

- Feature: dual test panels (empirical + arm-stratified), per-source primary
  reporting, panel-specific periodic-validation cadence
- Specification ID: `EVAL-PROTOCOL`
- Version: `1.1`
- Status: `APPROVED`
- Date: `2026-07-31`
- Supersedes: `docs/specifications/evaluation_protocol_v1.0_specification.md`,
  version `1.0` (only for the subset of `REQ-004`/`DEC-005`, `REQ-007`,
  `REQ-013`, and §8 items 3/7 amended below; the rest of v1.0 remains
  authoritative and unchanged, including `REQ-001`-`REQ-003`, `REQ-005`,
  `REQ-006`/`DEC-002`, `REQ-008`-`REQ-012`, `REQ-014`-`REQ-019`, and all of
  §7 not listed as amended)
- Related specifications: `docs/specifications/scenarionet_integration_v1.2_specification.md`
  (introduces the `test_waymo_empirical`, `test_arm_stratified`, `test_pg`,
  `validation_waymo_empirical`, and `validation_pg` pools this amendment
  consumes)
- Related ADRs: ADR-037
- Related ExecPlan: `docs/implementation/empirical_holdout_split_and_dual_test_panels_v1.2_exec_plan.md`
- Authoritative: `YES`

### Approval Record

- Date: `2026-07-31`
- Evidence: explicit user approval in this conversation ("mi sembra vada
  tutto bene") after review of the drafted document alongside its companion
  `SCENARIONET-INTEGRATION` v1.2 amendment.
- Approved scope: the complete v1.1 amendment as drafted. `REQ-006`/`DEC-002`
  (`checkpoints/final.zip` as the sole official checkpoint) is **explicitly
  retained unchanged** by this amendment; no requirement in this document
  touches checkpoint selection.

## 1. Motivation

`scenarionet_integration_v1.2_UNDER_REVIEW` replaces the arm-balanced
train/validation/test allocation with an empirical-holdout-first policy and
introduces a separate arm-stratified challenge pool. Under v1.0, `REQ-004`
freezes a single validation panel (100 episodes) and a single test panel
(300 episodes), both drawn by a balanced cross-arm sampler regardless of the
underlying split's composition — so a change to the split policy alone would
have no observable effect on what is evaluated. This amendment changes the
panel policy to match: it declares multiple named panels, assigns each a
role (primary/secondary/diagnostic), and states which panels feed the
primary comparison versus which are descriptive only.

## 2. Amendments

### 2.1 Amendment to REQ-004 — Freeze Validation And Test Scenario Panels

The following invariants of v1.0 `REQ-004` are replaced:

> - intended panel sizes under `thesis` are 100 validation episodes and 300
>   final-test episodes;

is replaced by:

- five named panels are declared, each with its own size, source split, and
  draw policy:

| Panel | Split | Source | Size | Draw policy | Role |
|---|---|---:|---:|---|---|
| `validation_waymo_empirical` | `validation` | Waymo | 100 | empirical (label-blind, uniform over the empirical pool) | primary learning-curve panel |
| `validation_pg` | `validation` | PG | 100 | empirical | diagnostic |
| `test_waymo_empirical` | `test_empirical` | Waymo | 300 | empirical | **primary final-test endpoint** |
| `test_pg` | `test_empirical` | PG | 200 | empirical | secondary (procedural generalization) |
| `test_arm_stratified` | `test_stratified` | Waymo + PG | 300 | arm-balanced (50 per arm), the v1.0 draw policy unchanged | competence endpoint, only panel with sufficient per-arm power |

The remaining `REQ-004` invariants (identical frozen panel across every
condition and seed within a comparison block, hashing, deduplication,
stored attempted UID sequence) apply to every panel in the table above,
individually. An `empirical` draw is defined as: uniform sampling over the
panel's source pool without reading `primary_arm`, `tags`, `low_traffic`, or
`dense_traffic`; an `arm_balanced` draw is the v1.0 `panel_manifest.py`
policy, unchanged.

**Proposed decision** (`DEC-005`, amends v1.0 panel sizing): panel sizes are
as declared in the table above. `test_waymo_empirical` is the sole primary
final-test endpoint (`REQ-017` unchanged: still test-split-only, still no
model-selection influence). `test_arm_stratified` and `test_pg` are
secondary; neither is aggregated into a single primary scalar with
`test_waymo_empirical` (see `REQ-007` amendment, §2.2).

### 2.2 Amendment to REQ-007 — Report Method-Neutral Primary Outcomes

The following invariant is added to v1.0 `REQ-007`:

> - `test_waymo_empirical` results are the primary quantitative outcome. Any
>   summary combining `test_waymo_empirical` and `test_pg`, or combining
>   Waymo and PG counts within `test_arm_stratified`, into one number shall
>   be labelled explicitly as an artificial macro-source average and shall
>   never replace the per-source primary report;
> - the per-arm breakdown (`A0`-`A5`) is read from `test_arm_stratified`
>   only; `test_waymo_empirical`'s observed arm distribution is reported
>   descriptively (post hoc), never used for a per-arm comparison claim,
>   because its small per-arm counts (see `scenarionet_integration_v1.2`
>   §4) do not support one.

### 2.3 Amendment to REQ-013 — Report Learning Curves On The Environment-Step Axis

The following invariant is added to v1.0 `REQ-013`:

> - the primary learning curve uses `validation_waymo_empirical` only;
>   `validation_pg` produces a secondary, lower-resolution curve (§2.4) and
>   is never interpolated onto the primary curve's step axis.

### 2.4 Amendment to §8 — Applicability, State, And Timing

Item 3 of v1.0 §8 ("Periodic validation occurs every 25,000 environment
timesteps under `run_profile=thesis` and uses the fixed 100-scenario
validation panel") is replaced by:

> 3. Periodic validation occurs under `run_profile=thesis` on a
>    panel-specific cadence: `validation_waymo_empirical` (100 episodes)
>    every 25,000 environment timesteps; `validation_pg` (100 episodes)
>    every 100,000 environment timesteps. Both cadences are frozen before
>    official training starts, identical across every condition and seed
>    within a comparison block, and produce independently timestamped
>    evaluation points — `validation_pg` points are not interpolated onto
>    the `validation_waymo_empirical` step axis.

Item 7 of v1.0 §8 ("Final test uses the frozen 300-scenario test panel and
deterministic inference") is replaced by:

> 7. Final test evaluates `test_waymo_empirical`, `test_pg`, and
>    `test_arm_stratified` once each, using deterministic inference, after
>    training and after any validation-only decision are complete (`REQ-005`,
>    `REQ-017`, unchanged). None of the three participates in checkpoint
>    selection (`REQ-006`/`DEC-002`, unchanged).

**Proposed decision** (`DEC-006`, new): panel-specific periodic-validation
cadence as stated above. Rationale: a uniform 25,000-step cadence across two
validation panels doubles the evaluation cost of every comparison block
(≈60 evaluation points × 3 seeds × N conditions); the asymmetric cadence
keeps the primary Waymo curve at full resolution while retaining a
diagnostic PG curve at ≈15 points, at ≈25% additional cost instead of 100%.

## 3. Not modified by this amendment

`REQ-001`-`REQ-003` (comparison blocks, sample budget, three training
seeds), `REQ-005` (deterministic read-only evaluation), `REQ-006`/`DEC-002`
(sole official checkpoint `final.zip`), `REQ-008`-`REQ-012` (seed-level
metrics, cross-seed aggregation, paired differences, failure handling,
data-abort coverage), `REQ-014`-`REQ-019` (qualitative case selection,
reproducibility metadata, core comparison report, test-split-only final
evaluation, ablation separation, future-algorithm admission), and all of §7
except §7.7's implicit dependency on the panel table in §2.1 (unchanged in
substance: aggregation still occurs at exactly the recorded evaluation-step
index, per panel).

## 4. Compatibility and impact

This amendment has no effect until `scenarionet_integration_v1.2` is
approved and the new panels exist. It does not change `checkpoints/final.zip`
as the sole official checkpoint, and it does not change the three-seed
protocol or the 1,500,000-step budget. Any comparison block that mixes
results evaluated under v1.0 panels with results evaluated under v1.1 panels
is invalid per v1.0 `REQ-019`/`AC-009` (unchanged): a new panel definition
requires a new approved protocol version before combination.
