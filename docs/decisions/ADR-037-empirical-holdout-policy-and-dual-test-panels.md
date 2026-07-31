# ADR-037: Empirical Holdout Policy And Dual Test Panels

- Status: Approved
- Date: 2026-07-31
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-31
- Supersedes: NONE (materially supersedes ADR-012 for the validation/test
  splits specifically; ADR-012 remains historical for its own scope)
- Affected specifications:
  - `docs/specifications/scenarionet_integration_v1.2_specification.md`, ID
    `SCENARIONET-INTEGRATION`, version `1.2` (amends v1.1 §5, §6.1, §6.2,
    §6.4, §6.5/§6.6, §6.8)
  - `docs/specifications/evaluation_protocol_v1.1_specification.md`, ID
    `EVAL-PROTOCOL`, version `1.1` (amends v1.0 `REQ-004`/`DEC-005`,
    `REQ-007`, `REQ-013`, §8 items 3/7)
- Affected ExecPlans:
  `docs/implementation/empirical_holdout_split_and_dual_test_panels_v1.2_exec_plan.md`
  (`DEC-001`, `DEC-003` through `DEC-006`)

## Context

The frozen v1.1 dataset (`selection_hash`
`7c0d6f3b15ca69ff79dbb3a0daefefdae326254065a9792c0b73af1ffa730fb5`) imposes
near-uniform `A0`-`A5` arm quotas on every split, including validation and
test (`src/thesis_rl/scenarios/pipeline.py:804`, re-asserted as a fatal
invariant at `:780`). Validation and test therefore measure a uniform
macro-average over competences, not performance on the empirical eligible
distribution of either source — while being described, informally, as if
they were ordinary holdouts. Separately, the evaluation panels
(`src/thesis_rl/scenarios/panel_manifest.py`) apply their own balanced draw
on top of the split, using only 300 of the 1000 frozen test scenarios;
changing the split policy alone would therefore have had no observable
effect on what a comparison actually measures.

The converted Waymo eligible pool is also capacity-limited for the
low-traffic arms: exactly 26 `A0`, 143 `A1`, and 144 `A2` scenarios are
available (derived from the frozen v1.1 index, where the balanced selector
consumed the entire available capacity for these three arms rather than
being quota-limited). A purely empirical Waymo test panel of typical size
would therefore carry on the order of 4 `A0` and 20 `A1` episodes — too few
to support a per-arm comparison between curriculum and baseline conditions,
which is a central object of this thesis.

## Decision

1. **Holdout-first allocation.** Validation and test pools are reserved from
   a deterministic pseudo-random permutation of the eligible candidate
   stream, per source, with `A0`-`A5` labels ignored by the allocator
   (label-blind projection). Test is reserved first, validation second, both
   frozen before any training-pool construction. Any post-freeze acquisition
   enters the training pool only.
2. **Dual test/validation endpoints**, declared and frozen in advance:
   `test_waymo_empirical` (primary generalization endpoint, 300 episodes),
   `test_pg` (secondary, 200 episodes), `test_arm_stratified` (competence
   endpoint, 50 episodes/arm, the only panel with sufficient per-arm power),
   `validation_waymo_empirical` (primary learning curve, 100 episodes, every
   25,000 steps), `validation_pg` (diagnostic, 100 episodes, every 100,000
   steps).
3. **Training pool with per-arm minimums**, not exact equal arm quotas,
   built from the residual population after both holdouts are frozen. This
   keeps `UniformScenarioProvider` and `ArmUniformScenarioProvider`
   distinguishable baselines, which an exactly-balanced pool would collapse
   onto the same sampling distribution.
4. **No primary metric aggregates Waymo and PG.** Any macro-source average
   is explicitly labelled as artificial. The per-arm breakdown is read from
   `test_arm_stratified` only.
5. **`checkpoints/final.zip` remains the sole official checkpoint**
   (`EVAL-PROTOCOL` `REQ-006`/`DEC-002`, explicitly unchanged by this
   decision).
6. **PG holdout mixture**: equiprobable 20% across the five generation
   profiles (`P0_simple`, `P1_vehicle_interaction`, `P2_merge_or_roundabout`,
   `P3_intersection`, `P5_complex_mixed`), disjoint seed range from every
   other split. The training PG pool may continue deficit-driven
   replenishment (ADR-008/ADR-009) since training does not require an
   empirical distribution.
7. **Known structural capacity limit, reported not corrected**: the
   `A0 x Waymo` cell of `test_arm_stratified` cannot reach 50 scenarios given
   the pool's 26-scenario Waymo `A0` capacity; the shortfall is filled from
   PG per the unchanged §6.6 policy and recorded in the split manifest.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Empirical test only, no stratified panel | Simplest; matches a strict reading of "empirical generalization" | Per-arm ACL-vs-baseline comparison becomes statistically uninterpretable for `A0`/`A1` given Waymo pool capacity | Rejected: the per-arm breakdown is a primary object of this thesis |
| Keep the status quo balanced test, correct only its description | No re-split work; cheapest | No empirical generalization endpoint exists at all; 700/1000 frozen test scenarios remain permanently unused by the panel draw | Rejected: does not address the underlying measurement gap |
| Move official checkpoint to best-validation selection | Could reduce end-of-training instability | Argmax over ~60 noisy evaluation points with 3 seeds risks a selection bias that differs by algorithm (winner's curse), confounding the comparison | Rejected; `final.zip` retained (item 5) |

## Consequences

- **Scientific validity**: introduces a genuine empirical generalization
  endpoint for the first time, alongside a properly justified competence
  benchmark, with per-source reporting that avoids conflating Waymo and PG
  performance.
- **Compatibility**: breaking by construction. The regenerated dataset
  receives a new `selection_hash` and a new split-manifest schema version.
  Runs against the v1.1 frozen dataset cannot be resumed against, or
  aggregated with, runs on the v1.2 dataset. Acceptable only because
  official experiments have not started.
- **Reproducibility**: every new draw is seeded and recorded in the split
  manifest; label-blindness of the holdout allocator is enforced by
  construction and by test.
- **Operational cost**: periodic validation cost increases by roughly 25%
  (asymmetric cadence) rather than 100% (uniform cadence across both
  validation panels).
- **Known limitation carried forward**: `A0 x Waymo` structural scarcity in
  `test_arm_stratified` (item 7).

## Validation And Traceability

Affected requirements: `SCENARIONET-INTEGRATION` v1.2 all amended
requirements (§3 of the specification); `EVAL-PROTOCOL` v1.1 all amended
requirements (§2 of the specification). Mandatory tests: `TEST-001` through
`TEST-017` of the linked ExecPlan (label blindness, freeze ordering, pairwise
group disjointness across four pools, per-arm minimum feasibility, PG seed
disjointness, panel draw-policy correctness, determinism, Hydra preset
resolution, checkpoint-policy non-regression). New regression risk: any
future change to the allocator must preserve label-blindness of the
empirical holdout stage, verified by `TEST-001`.

## Approval Record

- Approved by: repository maintainer (explicit user approval)
- Approval evidence: user message "mi sembra vada tutto bene" in this
  conversation, following review of the drafted `UNDER_REVIEW` specification
  amendments and the preceding decision-by-decision approvals (`DEC-001`
  through `DEC-007`, recorded 2026-07-31 in the ExecPlan)
- Notes: NONE
