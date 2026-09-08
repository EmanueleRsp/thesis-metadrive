# ExecPlan: Rulebook v5.1 six-level hierarchy and `SCAL-V1.4`

## 1. Metadata

- Feature: bring `src/thesis_rl/` from the three-macro-rule / four-margin
  rulebook of v4.7 + `SCAL-V1.1` to the approved six-level hierarchy of
  `RULEBOOK-V5.1` with `SCAL-V1.4`.
- Plan ID: `RB51`.
- Authoritative specification: `docs/specifications/rulebook_v5.1_specification.md`
  (`RULEBOOK-V5.1`, version `5.1`, **`APPROVED`** 2026-08-14, amended 2026-08-20).
  Evidence of record for the per-sub-rule Test A / Test B tables it inherits:
  `rulebook_v5.0_UNDER_REVIEW_specification.md` (`SUPERSEDED`, never approved).
- Status: `IN_PROGRESS` — every gate resolved. `DEC-RB51-001`, `-002` and
  `-005` were **approved on 2026-08-20**, each on the recommendation given, so
  no milestone is blocked.
- Created: 2026-08-20. Last updated: 2026-08-20.
- Branch: `scenarionet-implementation`.
- Related ADRs: `ADR-063` .. `ADR-076`. `ADR-058`, `ADR-059`, `ADR-061` and
  `ADR-062` are already implemented and are preserved, not revisited.
- Relationship to `RSEC-V1`
  (`reward_scale_and_episode_contract_v1_exec_plan.md`, `AWAITING_DECISIONS`):
  that plan's `DEC-RSEC-001` asked how to fix the reward scale. `RULEBOOK-V5.1`
  **answers it**, so this plan supersedes `RSEC-V1` for everything downstream of
  that gate. `RSEC-V1`'s completed episode-contract work (`ADR-058`) stands.

## 2. Objective And Scope

### Observable capability

After this plan, a training run evaluates the rulebook the approved
specification describes: six ordered levels, the atomic cost vector of §3.4
exposed alongside them, and a scalar reward computed by `SCAL-V1.4` at the
measured weights. The measured expert panel is the acceptance reference: the
same instrument run against production must reproduce the offline figures.

### Success is recognized when

The production evaluator, driven over the frozen Waymo `train` panel, produces
per-step channel values that **match the measurement instrument to numerical
tolerance** on the same records. This is the only end-to-end check available
offline, and it is strong: the instrument is what the specification's numbers
were derived from.

### In scope

Six-level hierarchy and its aggregations; the L4 channel and the new L6 channel;
the sub-rule redefinitions v5.1 inherits from v5.0 §4; the at-fault gate; the
`speed_limit` sub-rule; `SCAL-V1.4`; the diagnostics of §7.

### Out of scope

- **The lexicographic and distributional arms.** They consume §3.4's vector,
  which this plan produces, but choosing and implementing a thresholded-lex
  algorithm — including `τ₄` and `d₂` — is separate work, sequenced after the
  rulebook is frozen (open item `D1`).
- Retraining, and any decision about the three stopped production runs
  (open item `D4`).
- PG coverage measurement (`F6`).

### Compatibility

**This change breaks checkpoint compatibility**, intentionally and by two
independent routes: the observation gains a `speed_limit` input (v5.1 §6), and
the margin vector changes shape from 4 to 6. Both were approved as part of the
specification. Coordinate with `D4`.

## 3. Authoritative Requirements

| ID | Requirement | Spec section |
|---|---|---|
| `REQ-RB51-01` | Six levels, strictly ordered, with the sub-rule membership of the §3 table | §3 |
| `REQ-RB51-02` | `max` across objects within a sub-rule, and within L2 and within L3 | §3.3 |
| `REQ-RB51-03` | L5 aggregates by normalized sum with a **fixed** denominator of 3 | §3.3 |
| `REQ-RB51-04` | L6 has one sub-rule; denominator stays explicit at 1 | §3.3 |
| `REQ-RB51-05` | The atomic cost vector of §3.4 is exposed alongside the aggregated channels | §3.4 |
| `REQ-RB51-06` | `Δq_t = clip((s_{t+1} − s_t)/D_REF, −1, +1)`, `D_REF = v_ref·Δt = 2.2222 m`; L4 carries the bare advance | §4.1 |
| `REQ-RB51-07` | `Σ_t Δq_t = (s_T − s_0)/D_REF` to numerical tolerance below the clip | §4.2 |
| `REQ-RB51-08` | `c_L6 = 1 − clip(Δq, 0, 1)`; standstill and reverse both cost 1 | §4.6 |
| `REQ-RB51-09` | `SCAL-V1.4` as written in §5.1, at `a=2.2, σ=0, φ=0.25, λ₄=2.0, η=1.0, λ₆=0.2` at approval; `a=2.5, σ=0.30` since ADR-081 (2026-09-07), the other four unchanged | §5.1, §5.5, ADR-081 |
| `REQ-RB51-10` | The §5.4 rank-preservation predicate rejects inadmissible weights before use | §5.4 |
| `REQ-RB51-11` | `rss` longitudinal is a reported diagnostic, never in the reward | §3, ADR-063 |
| `REQ-RB51-12` | `wrongway` is deleted; `dashed_line` retained at L5 | ADR-066 |
| `REQ-RB51-13` | Traffic-control persistence latches removed | ADR-064 |
| `REQ-RB51-14` | `offroad`, `solid_line`, `wrong_carriageway` use the v5.0 §4 geometric criteria | ADR-065 |
| `REQ-RB51-15` | `clearance` scoped to VRU on the roadway; `ttc` at nuPlan's 0.95 s; `rss_lateral` scoped | ADR-067 |
| `REQ-RB51-16` | `speed_limit` sub-rule at L3, posted limit admitted **by provenance** only | ADR-068 |
| `REQ-RB51-17` | At-fault gate: interaction sub-rules inapplicable at or below the gate speed | ADR-070 |
| `REQ-RB51-18` | At-fault collision terminates and charges; not-at-fault truncates and charges nothing | ADR-071 |
| `REQ-RB51-19` | Diagnostics of §7, including `l4_clip_binding_steps`, `l5_reached_steps`, `mean_ego_speed_by_source` | §7 |
| `REQ-RB51-20` | One shared discount in every algorithm configuration, shaping discount tracking it; `γ = 1` at approval (ADR-075), `γ = 0.996` since 2026-09-07 (ADR-081, together with `a = 2.5`, `σ = 0.30` in `conf/scalarization/default.yaml`) | §4.4, ADR-075, ADR-081 |

`REQ-RB51-20` is **already met** (six configs updated 2026-08-20, guarded by
`tests/test_hydra_agent_presets.py::test_every_algorithm_shares_one_hierarchy_preserving_discount`,
renamed from `test_every_algorithm_is_undiscounted` when ADR-081 replaced `γ = 1`
with `γ = 0.996`); it is
listed for traceability, not as work.

## 4. Current Repository Analysis

All `VERIFIED` by inspection on 2026-08-20 unless labelled otherwise.

| Concern | Current state | Path |
|---|---|---|
| Macro levels | **Three**, plus a progress margin | `rulebook/v2/types.py:99` `MacroRule` |
| Margin vector | **Four** floats, three costs | `rulebook/v2/types.py:427` `RulebookResult` |
| Aggregation | `max` for all three macro rules; no sum, no L5, no L6 | `rulebook/v2/aggregation.py` |
| Components | 14 registered: `collision`, `rss`, `rss_lateral`, `ttc`, `clearance`, `offroad`, `wrong_way`, `wrong_carriageway`, `solid_line`, `dashed_line`, `signal`, `stop`, `crosswalk`, `vehicle_yield` | `rulebook/v2/registry.py:42` |
| Missing component | **`speed_limit` does not exist** | — |
| To delete | `wrong_way` | `registry.py:54` |
| To demote | `rss` is normative today | `registry.py:49` |
| Progress | `raw_delta = post_mission.s_m − pre_mission.s_m` already computed | `rulebook/v2/components/progress.py:87` |
| Scalarization | `SCAL-V1.1`, four modes, `priority_base` 3.0 | `reward/scalarization.py:10` |
| Reward wiring | `scalar_rule_reward` from the wrapper into the hybrid manager | `rulebook/v2/wrapper.py:247`, `reward/managers/hybrid_rulebook_manager.py` |
| Route geometry | `build_assigned_route_polyline` builds the polyline `s` projects onto | `rulebook/v2/geometry/route.py:300` |
| PG adapter | `build_pg_static_adapter_result` exists and reads `polygon` | `rulebook/v2/context/pg_static_adapter.py:144` |
| Offline reference | The instrument implements every v5.1 channel already | `scripts/measure_expert_rulebook_transition.py` |

**Behaviour to preserve**: the component/memory/cache protocol
(`ComponentDefinition`, `MemoryDelta`, `CacheDelta`), the fail-closed
`NOT_EVALUABLE` handling in `aggregate_max_component`, the applicability masks,
and every `ADR-058`/`ADR-061`/`ADR-062` episode-contract behaviour.

**Directly relevant debt**: `C1` (RSS standstill-exit transient), `C2` (`SIGNAL`
never selected on 25.2 % of scenarios that have traffic lights) and `C3` in
`docs/open_items.md`. `C2` in particular means `REQ-RB51-13`'s latch removal
lands on a sub-rule that is already under-firing; the two must not be conflated
when reading results.

## 5. Assumptions And Invariants

- Units: metres, seconds, m/s. `Δt = 0.1 s`, `T_REF = 1 s`. `VERIFIED`.
- `s_m` is arc length along the assigned route polyline, not a chord.
  `VERIFIED` — the arc length of `canonical_route_points_xyz` equals
  `s_goal − s_start` to the metre (mean 90.17 m, chord 87.29 m).
- Every sub-rule cost is in `[0, 1]`; every channel is in `[0, 1]` except L4,
  which is in `[−1, 1]`. `SPECIFIED`.
- `max_speed_km_h = 80` is not overridden, so the §4.1 clip never binds on an
  agent trajectory. `VERIFIED`; `AC-RB5.1-05` depends on it and open item `D3`
  records the decision not to change it.
- Episodes terminate at the logged horizon or on mission success; the tail is
  zero (`ADR-058`). `VERIFIED`.
- Determinism: the evaluator is a pure function of the transition snapshot plus
  the declared memory fields. Violations fail closed.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-RB51-001` | Specification clarification | `speed_limit` needs an observation input, which amends `OBS-V1.3` and `OBS-LIDAR-V2.0` — both separately approved documents | (a) amend both now, under this plan; (b) amend them in their own change and defer `M4` | (a), because a rulebook without `speed_limit` is not the approved rulebook and `M6`'s weights were calibrated with it | Observation schema, encoder input width, checkpoint compatibility | **Approved 2026-08-20**, option (a) |
| `DEC-RB51-002` | Scope | Is `ADR-071` (at-fault classification and truncation) in this plan? | (a) yes, as `M7`; (b) separate plan | (a) — it is an approved part of the same contract (`B1`), and leaving it out means the rulebook is still not frozen | Episode contract: termination vs truncation, bootstrap value | **Approved 2026-08-20**, option (a) |
| `DEC-RB51-003` | Implementation detail | Keep `SCAL-V1.1`'s four legacy modes alongside `SCAL-V1.4`? | (a) keep, add the new mode; (b) remove | (a) — they are historical baselines and removing them would make earlier runs unreproducible | Config surface only | Proposed, no approval needed |
| `DEC-RB51-004` | Implementation detail | Where does L6 live in the component protocol? | (a) a `progress_rate` component in the registry like any other; (b) computed in aggregation from L4 | (a) — uniform with §3.4's atomic vector, and it keeps aggregation free of channel-specific arithmetic | Internal | Proposed, no approval needed |
| `DEC-RB51-005` | Specification clarification | The `MacroRule` **string values** reach recorded CSVs, eval artifacts and analysis tables — 75 occurrences across ~20 files — so renaming them is an output-contract change, not an internal one. §3 names the six channels `collision_safety`, `interaction_risk`, `non_relaxable_compliance`, `mission_progress`, `relaxable_lane_compliance`, `progress_rate`, while production carries `collision_impact`, `dynamic_interaction_safety`, `road_traffic_compliance` | (a) rename to the specification's names and declare the artifact-schema break alongside the already-approved checkpoint break; (b) keep the legacy values and accept that L3 is recorded as `road_traffic_compliance` after the relaxable road rules have left it | (a). Leaving a level called `road_traffic_compliance` when the road rules a driver may relax are no longer in it is exactly the stale naming that causes later misreadings, and the change already breaks the artifact schema through the margin vector's shape | Recorded CSV columns, eval artifacts, analysis tables (`EVAL-PROTOCOL` reporting) | **Approved 2026-08-20**, option (a) |

All three were approved on 2026-08-20 on the recommendation given, so every
milestone is open. `DEC-RB51-005` in particular authorizes the artifact-schema
break: the recorded channel names become §3's, alongside the margin vector's
change of shape and the already-approved checkpoint break.

## 7. Proposed Design

`MacroRule` gains `RELAXABLE_LANE_COMPLIANCE` and `PROGRESS_RATE`, and
`ROAD_TRAFFIC_COMPLIANCE` is renamed to `NON_RELAXABLE_COMPLIANCE` with the
relaxable sub-rules moved out of it. `MACRO_RULE_ORDER` becomes the six-level
order and is the single source of truth for the ordering; nothing else may
hard-code it.

`RulebookResult.margins` becomes a six-tuple and `costs` a five-tuple (L1, L2,
L3, L5, L6), with L4 kept separately as the signed advance, so that a channel
that is a *cost* is never confused with the one channel that is a *utility*.

`aggregation.py` gains `aggregate_sum_component`, taking an explicit denominator
so the L5 fixed-3 rule and the L6 explicit-1 rule are the same code path with
different declared constants — the fixed denominator is what `TEST-RB5.1-11`
guards and it must not be inferred from how many sub-rules happened to apply.

`reward/scalarization.py` gains a `six_level_priority_weighted_rank` mode
implementing §5.1, with the §5.4 predicate as a constructor-time check that
refuses inadmissible weights rather than pricing them.

The measurement instrument stays as it is and becomes the **oracle**: `M8`
asserts production against it record by record.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-RB51-01`..`05` | `AC-RB5.1-01`, `-08` | `rulebook/v2/{types,registry,aggregation}.py` | `tests/test_rulebook_v51_levels.py` | Planned |
| `REQ-RB51-06`..`08` | `AC-RB5.1-05`, `-07` | `rulebook/v2/components/progress.py`, new `progress_rate.py` | `tests/test_rulebook_v51_l4_l6.py` | Planned |
| `REQ-RB51-09`, `-10` | `AC-RB5.1-06`, `-13`, `-14` | `reward/scalarization.py` | `tests/test_scal_v14.py` | Planned |
| `REQ-RB51-11`..`15` | v5.0 §4 Test A tables | `rulebook/v2/components/*`, `registry.py` | existing rulebook suite, extended | Planned |
| `REQ-RB51-16` | `AC-RB5.1` speed-limit rows | new `components/speed_limit.py` | `tests/test_speed_limit_subrule.py` | Planned |
| `REQ-RB51-17` | ADR-070 gate | `components/{ttc,clearance,rss_lateral}.py` | `tests/test_at_fault_gate.py` | Planned |
| `REQ-RB51-18` | `AC-RB5.1-14` | `rulebook/v2/lifecycle.py`, env done/truncation | `tests/test_rulebook_v51_orderings.py::…-14` | Planned |
| `REQ-RB51-19` | §7 | `rulebook/v2/wrapper.py`, monitor | `tests/test_rulebook_v51_diagnostics.py` | Planned |
| `REQ-RB51-20` | `AC-RB5.1-16` | `conf/agent/planner/algorithm/*.yaml` | `test_every_algorithm_shares_one_hierarchy_preserving_discount` | **Done** |

## 9. Test Strategy Defined Before Implementation

Acceptance criteria are `RULEBOOK-V5.1` §9's `AC-RB5.1-01` .. `-17`, which this
plan does not restate or weaken. The mandatory matrix below is frozen before any
production change.

| ID | Level | Behaviour | Fixture | Expected | Requirement |
|---|---|---|---|---|---|
| `T-RB51-01` | Unit | Six levels in the declared order, nothing hard-codes it twice | `MACRO_RULE_ORDER` | Order matches §3 | `REQ-RB51-01` |
| `T-RB51-02` | Unit | `max` within L2 and L3; `c_L2 = 0` iff all three L2 sub-rules are 0 | synthetic component results | exact | `REQ-RB51-02` |
| `T-RB51-03` | Unit | L5 denominator stays 3 when a sub-rule is inapplicable | one applicable, two not | `c/3`, not `c/1` | `REQ-RB51-03` |
| `T-RB51-04` | Unit | L4 signed, clipped; `Σ Δq` telescopes below the clip | station sequence | `(s_T−s_0)/D_REF` | `REQ-RB51-06`, `-07` |
| `T-RB51-05` | Unit | `c_L6 = 1 − clip(Δq,0,1)`; standstill and reverse cost 1 | `Δq ∈ {−0.4, 0, 0.5, 1}` | `{1, 1, 0.5, 0}` | `REQ-RB51-08` |
| `T-RB51-06` | Unit | `SCAL-V1.4` equals §5.1 term by term | channel vector | exact | `REQ-RB51-09` |
| `T-RB51-07` | Unit | §5.4 predicate refuses inadmissible weights at construction | `λ₆ = 2.0` | raises | `REQ-RB51-10` |
| `T-RB51-08` | Unit | `rss` longitudinal never contributes to any channel | violating `rss` | all channels unchanged | `REQ-RB51-11` |
| `T-RB51-09` | Unit | `wrongway` is absent from the registry | registry | `KeyError` | `REQ-RB51-12` |
| `T-RB51-10` | Unit | At-fault gate makes interaction sub-rules inapplicable at standstill | stopped ego, hazard | `NOT_APPLICABLE` | `REQ-RB51-17` |
| `T-RB51-11` | Property | Every channel in `[0,1]`, L4 in `[−1,1]`, on random component vectors | hypothesis-free random grid | invariant holds | `REQ-RB51-01` |
| `T-RB51-12` | Integration | Production evaluator vs the measurement instrument, per step, on 20 frozen records | frozen panel subset | equal to `1e-9` | all |
| `T-RB51-13` | Integration | Full frozen Waymo `train` panel reproduces the §5.5 table | 1100 records | mean +70.70, below-standstill 3.36 % | all |
| `T-RB51-14` | Smoke | End-to-end training step runs under the six-level reward | `make smoke` | completes | all |
| `T-RB51-15` | Regression | The O1–O6 fixtures still hold against the **production** channels, not only the instrument's | `tests/test_rulebook_v51_orderings.py`, re-pointed | pass | `REQ-RB51-01`..`09` |

`T-RB51-12` and `T-RB51-13` are the load-bearing ones: they are what makes the
specification's published numbers a claim about production rather than about a
script.

### Commands

| Purpose | Command |
|---|---|
| Full suite | `make test` |
| Focused | `uv run --no-sync python -m pytest tests/test_rulebook_v51_*.py -q` (inside the container) |
| Rulebook v2 suite, scoped lint, whitespace | `make rulebook-v2-check` |
| Lint | `make lint` |
| Format check, focused | `make format-check PYTHON_QUALITY_PATHS="<paths>"` |
| Smoke | `make smoke` |
| Whitespace | `git diff --check` |

Type checking: **unavailable** — no global mypy target or configuration exists,
and this plan does not invent one. New public interfaces are annotated.

## 10. Milestones

- [x] **M0 — Frozen test matrix.** `DONE 2026-08-21`. 30 tests in
  `tests/test_rulebook_v51_levels.py`, verified failing against the
  three-macro-rule rulebook for the right reason (`aggregate_sum_component`
  absent). No production change in this milestone.
- [x] **M1 — Six levels.** `DONE 2026-08-21`, pending the last of its own
  fallout. `MacroRule` renamed to §3's names and extended to six;
  `MACRO_RULE_ORDER` is the sole source of the ordering and `COST_MACRO_RULES`
  separates the five cost levels from the one utility level; `RulebookResult`
  validates its own shape against the order instead of fixing four margins, and
  gained `margin_for` / `cost_for`; `aggregate_sum_component` added with a
  **declared** denominator that fails closed; `aggregate_rulebook_result` takes
  level membership from the registry rather than from a second name list;
  `wrong_way` deregistered, `rss` demoted, the three lane rules moved to L5.
  Consumers updated: `subrule_diagnostics`, `usefulness` (ACL weights),
  `video_diagnostics` (`R1..R4` -> `L1..L6`).
- [x] **M2 — L4 and L6.** `DONE 2026-08-21`. **L4 needed no change**:
  `evaluate_progress` already computed `clip(delta_s / (v_ref * dt), -1, +1)`
  with `v_ref * dt = 2.2222 m`, i.e. `REQ-RB51-06` was already met. New
  `components/progress_rate.py` implements `c_L6 = 1 - clip(dq, 0, 1)` as the
  `advance_shortfall` sub-rule, wired into `transition.py` from the same station
  delta L4 reads. `T-RB51-04` and `-05` still to be written as their own file.
- [x] **M3 — Inherited sub-rule redefinitions.** `DONE 2026-09-01`. `ADR-067`:
  `ttc` moves to nuPlan's uniform **0.95 s** (the per-class 0.8/1.0 pair had no
  source), with class *eligibility* split out of the threshold table so
  "eligible" and "how urgent" stopped being one object; `clearance` is scoped to
  VRU whose centroid lies on the drivable surface, which required threading the
  surface into the component. `ADR-065`: `offroad` takes the **0.3 m** band,
  `wrong_carriageway` the **centroid** gate, `solid_line` **graded penetration**
  at tolerance 0.3 with the marking buffered by its real half width 0.075 m.
  `ADR-064`: the persistence latches leave the cost of `crosswalk` and
  `vehicle_yield`; the illegal-entry set is still tracked and now reported as
  `latched_illegal_entry`, so "not charged" never becomes "not observed", and
  `REQ-EF-08`'s applicability coupling goes with the latch because there is no
  longer a latched cost for `aggregate_max_component` to discard.

  **One judgment recorded rather than assumed.** `ADR-065` replaces "the binary
  1.0 **on any contact**", and the swept-front-bumper crossing was one of the two
  ways that 1.0 was reached, so it leaves the cost and becomes a diagnostic.
  Three reasons, in order of weight: the measured 0.349 % was produced by a
  variant with no swept term, so pricing it here would put production and the
  oracle out of agreement before `T-RB51-12` is even run; an event term on a
  binary scale beside a state term on a graded one is exactly the confusion the
  redefinition removes; and under `ADR-072` a completed crossing that leaves the
  ego correctly placed is the *relaxation* L5 exists to permit, so charging it
  1.0 against sustained straddling's 0.328 would invert the intended ordering.
- [x] **M4 — `speed_limit`.** `DONE 2026-09-01`. New `components/speed_limit.py`
  at L3; `RouteLaneRecord` gains `posted_speed_limit_mps`, populated by the Waymo
  adapter through a **provenance** gate (`speed_limit_mph` must be present) and
  left `None` unconditionally by the PG adapter. The v1 extractor's
  `max_speed_km_h` fallback is not carried over, as `ADR-068` requires.

  Both observation amendments written and applied: `OBS-V1.3.1` (`lane_road`
  12 -> 14, `D` 3009 -> **3011**) and `OBS-LIDAR-V2.0.2` (frame 308 -> **310**,
  stacked 6489 -> **6531**). Checkpoint compatibility is broken by both, as
  approved.

  **The observation calls the reward's own lookup.** §7 requires the
  "unavailable" encoding to fire under *exactly* the condition that makes the
  sub-rule inapplicable, so `associated_speed_limit_mps` was made public and both
  paths call it after the same `associate_route_lane`. Two implementations of
  "is there a limit here" would eventually disagree and the failure would be
  silent. Two values rather than a sentinel, because a sentinel inside the
  normalized channel is indistinguishable from a real limit at that value once
  the encoder has projected it.
- [x] **M5 — At-fault gate (`ADR-070`).** `DONE 2026-09-01`. New
  `components/at_fault_gate.py` holds the threshold, the covered sub-rules and
  the inapplicable-result shape **once**: three components applying one rule from
  three private constants is how one rule becomes three slightly different rules.
  The gate reads the **post**-transition ego velocity for all three, including
  the two whose costs come from the pre state, because it asks whether the ego is
  stopped at the state the cost is charged against. `tests/test_at_fault_gate.py`
  additionally pins the magnitude argument that licenses transplanting an
  evaluation threshold into a reward: `5e-02 / v_ref < 0.005`, so crawling under
  the gate earns nothing.
- [x] **M6 — `SCAL-V1.4`.** `DONE 2026-09-01`. The formula, the §5.4
  constructor gate and `tests/test_scal_v14.py` were already in the tree; what
  was missing was that **nothing selected the mode**:
  `conf/scalarization/default.yaml` still named `bounded_priority_weighted_rank`
  at `priority_base 3.0`, a four-margin mode, while the rulebook has emitted six
  since `M1`. Switched to `six_level_priority_weighted_rank` at
  `a = 2.2, sigma = 0, phi = 0.25, lambda4 = 2.0, eta = 1.0, lambda6 = 0.2`,
  with `vector_schema_id = rulebook_v5_1_six_level_v1` (the distinct id of §3.4,
  which reaches the checkpoint identity as part of the already-approved break).
  `_REQUIRED_MARGIN_COUNT_BY_MODE` was declared but never read, with 4 and 6
  restated at the two `_canonicalize_bounded` call sites; the table is now the
  single source and `expected` lost its default so no caller can omit it.
- [x] **M7 — At-fault classification (`ADR-071`).** `DONE 2026-09-01`. New
  `components/collision_fault.py` implements nuPlan's taxonomy in the order that
  matters; `evaluate_collision_impact` charges **only** at-fault contacts and
  reports the rest. Undeterminable lane containment resolves to **at fault**: an
  input that cannot be resolved must never buy an exculpation, because the
  failure would be silent and would reward exactly the states where the geometry
  is hardest to resolve.

  The truncation half is in `done_function`, which already snapshots the
  post-state and so can classify without waiting for the outer Rulebook wrapper.
  It calls the **same** pure classifier the reward calls, so "charged" and
  "terminated" cannot drift. A not-at-fault-only step clears the crash flags and
  sets `MAX_STEP`, which is the repository's existing truncation channel: the
  episode ends and the value target bootstraps from `V(s)`, so provoking the
  impact buys the agent exactly what continuing to drive would have.
- [x] **M-DIAG — §7 diagnostics (`REQ-RB51-19`).** `DONE 2026-09-01`.
  `l4_clip_binding_steps`, `l5_reached_steps` and the per-step ego speed feeding
  `mean_ego_speed_by_source`. Two of the three exist to make a claim falsifiable
  rather than to describe a run, and the tests assert that reading: a non-zero
  L4 clip count says the 80 km/h cap was overridden and the Test A figures no
  longer bound what the agent can earn, and an `l5_reached_steps` that stays zero
  says the restructure bought nothing and must be reported as such.

- [ ] **M8 — Reconciliation.** `T-RB51-12` .. `-15`, `make test`, `make lint`,
  `make smoke`; reconcile every `AC-RB5.1-*`; update `docs/project_index.md` and
  close `B1`/`B2` in `docs/open_items.md`.

  **Three coverage measurements ride along with `M8`'s panel run** (approved
  2026-09-01). They are measurements *against* the frozen definition, not
  changes to it, and both expand what is measured rather than what is excused:
  each can only add violations, never remove them, so if the channels stay clean
  afterwards the frozen claim is strictly stronger. They are scheduled here and
  not later because `M8` must re-derive `T-RB51-13`'s targets against production
  anyway — folding them in costs one run, deferring them costs two.

  - **`M8a` — PG dispatch (closes `F6`).**
    `scripts/measure_expert_rulebook_transition.py:1634` calls
    `build_waymo_static_adapter_result` unconditionally; dispatch on the record's
    source so `build_pg_static_adapter_result` runs on PG. Then report
    **applicability rates per sub-rule and geometric sanity** on the PG `train`
    panel. This is **not** Test A and must never be reported as such: a PG
    record's logged ego is `IDMPolicy`, which makes the headway rules circular
    (limitation 5) and the positional rules vacuous (limitation 6), so replay
    can establish nothing about satisfiability there. What it does establish is
    whether the rulebook is *applicable* on PG at all — a sub-rule never
    applicable on half the training distribution is a sub-rule silently absent
    from it, which is `C2`'s failure mode transplanted to a whole data source.
    Instrument-only: no production code changes, so no rulebook regression is
    possible.
  - **`M8b` — decompose the 209 (answers `C2`, closes `C3`).** ADR-051 already
    reports the residual as **52** controls lost at adapter construction plus
    **157** that are *"genuinely unrelated approaches **or** route ends more than
    one lane short"*. That `or` is the whole point: the 157 mix correct
    exclusions — a light governing an approach the ego never enters — with
    recoverable ones, and the single-hop version cannot separate them. So `C2`'s
    headline 25.2 % is an upper bound on a defect whose true size is unknown.
    `control_line_diagnostics(cache)`
    (`src/thesis_rl/rulebook/v2/transition.py:556`) already computes exactly this
    and is already static per scenario; `C3` is precisely the missing
    aggregation. Aggregate it over the 828 signalised records and report the
    split. Read-only.
  - **`M8c` — per-scenario applicability.** The instrument already reports
    `applicable_steps` and `violated_fraction_of_applicable` per sub-rule; what
    it does not report is the **per-scenario** view — on how many scenarios a
    sub-rule is never applicable at all. That is the granularity `C2` needs and
    the one Test A is structurally blind to (limitation 6: a rule that never
    fires passes by never being tested).

- [ ] **M9 — conditional: multi-hop route reachability.** *Gated on `M8b`.* If
  the decomposition shows the recoverable share is material,
  `route_reachable_control_lane_ids` (`transition.py:502`) extends from one hop
  to a chain, walking successors while each remains unique. This is the
  follow-up ADR-051 named and deliberately deferred for want of measurement, not
  a new idea, and it preserves the load-bearing property unchanged — an
  ambiguous branch is never guessed — which already carries its own regression
  test. It amends `rulebook_v4.11` §2.9.5's operational predicate and therefore
  needs its own ADR, but it does **not** reopen `RULEBOOK-V5.1` §3/§4/§5:
  hierarchy, `L4`/`L6` and `SCAL-V1.4` are untouched.

## 11. Progress And Findings Log

**2026-09-01 (later) — M8's measurements. Three findings, one of them large.**

`T-RB51-12` and `T-RB51-13` **pass**, and passing them is what converts §5.5 from
a claim about a script into a claim about production.

- **`oracle_max_divergence` is 0.0** for `clearance`, `solid_line`, `ttc` and
  `wrong_carriageway` across all 1100 Waymo records, and `5.97e-06` for
  `offroad`. That residual is production's `OFFROAD_AREA_EPSILON_M2 = 1e-4`
  clamp, which the variant does not apply: `1e-4` over a ~8 m² footprint is
  `1.25e-5`, so the observed value sits inside the epsilon by construction. A
  deliberate numerical net, not a drift.
- **The panel reproduces §5.5 exactly** at the selected weights: mean **70.70**,
  p1 **-61.81**, p5 **+5.29**, p50 **+51.85**, below standstill **3.36 %**. 1100
  measured, 0 skipped, 0 errors.

**The oracle was itself broken, and had been since `M1`.** Before it could be
used it had to be found: the script still built the four-margin vector while
production has emitted six since 2026-08-21, so `scalarize_rulebook_margins`
raised on **every** record. The failure was invisible because
`ScalarizationEvaluationError` subclasses `ValueError` and the replay's own
handler counts that as a *skipped scenario* -- the report said "0 measured", and
nobody read it because the script had not been run. Three further v4.7-era
assumptions surfaced behind it: the self-check compared the old `r2`/`r3`
aggregates against channels production no longer computes, and the `clearance`
reproduction compared the unscoped rule against a production that now scopes and
gates it. Each would have reported an *approved redefinition* as an instrument
defect.

**`M8b` -- `C2` decomposed, and it is worse for `stop` than for `signal`.**
On the same 1100 records:

| | total | dropped by route reachability |
|---|---:|---:|
| `SIGNAL` controls | 744 | **313 (42.1 %)** |
| `STOP` controls | 536 | **400 (74.6 %)** |

546 records carry a signal that survived adapter construction, and **122 of them
have every signal dropped** by the filter. A further **13 569** controls are lost
at adapter construction (`control_line_off_route_drop_count`). So `C2`'s 25.2 %
headline is, on this panel, 122 records attributable to route reachability --
the part `M9`'s multi-hop extension could recover -- and the rest to adapter
construction, which is a different defect with a different fix. **`STOP` was
never measured before and is the worse case**; `C2` should be restated to cover
both.

**`M8c` -- three L3 sub-rules are nearly inert on Waymo.** Records on which the
sub-rule is applicable at least once, out of 1100: `crosswalk` **15 (1.4 %)`,
`stop` **115 (10.5 %)**, `signal` **409 (37.2 %)**. Test A can only reject, so a
rule that almost never applies passes it almost for free. Their clean §4 figures
must be read against these denominators.

**`M8a` -- the largest finding. PG and Waymo are not under the same rulebook.**
The dispatch was one call site, as limitation 5 said. With it, 1089 of 1100 PG
records replay (11 skipped on a missing `length` field), `oracle_max_divergence`
is 0.0 throughout, and the expert scores mean **+142.06** with **5.6 %** below
standstill against Waymo's +70.70 and 3.36 %.

Records on which each sub-rule is ever applicable:

| sub-rule | Waymo /1100 | PG /1089 |
|---|---:|---:|
| `offroad`, `solid_line`, `dashed_line`, `wrong_carriageway` | 1005-1100 | **1089** |
| `ttc` | 1100 | 909 |
| `rss_lateral` | 806 | 381 |
| `vehicle_yield` | 725 | 196 |
| `clearance` | 674 | **0** |
| `signal` | 409 | **0** |
| `stop` | 115 | **0** |
| `crosswalk` | 15 | **0** |
| `speed_limit` | 1100 | **0** |

**Five of the six L3 sub-rules never apply on PG**, so L3 there is `offroad`
alone, and `clearance` never applies either because PG has no VRU. This is a much
stronger statement than limitation 13, which records only the speed regime: the
two sources are graded by *substantially different rulebooks*, and PG carries no
traffic controls at all (`control_line_coverage` is zero throughout). It bears
directly on open item `D5` (arm/source confounding), and it is the reason `F6`
was worth promoting out of *Deferred* rather than measuring after training.

**None of this is a reason to reopen the definition.** Every figure is a
measurement *against* the frozen hierarchy, and the hierarchy reproduced its own
published numbers to the decimal on the way. What they change is what may be
*claimed*: the admissibility evidence covers Waymo, and on PG the rulebook is
largely inapplicable rather than largely satisfied.


**2026-09-01 — M6 complete. The defect was the wiring, not the formula.**

The full suite was run first, as a measurement rather than as a gate: **2 failed,
1484 passed**. Both failures were informative and neither was in `SCAL-V1.4`'s
arithmetic.

1. *Nothing selected the new mode.* `conf/scalarization/default.yaml` still named
   a four-margin mode while the rulebook has emitted six since `M1`, so
   `tests/test_rulebook_v2_wrapper.py` failed with
   `Scalarization requires 4 macro margins, got 6`. Worth stating precisely,
   because the first reading was wrong: this is **not** an unguarded hole. The
   arity contract is enforced, has its own two tests
   (`test_six_level_mode_rejects_a_four_margin_vector`,
   `test_legacy_modes_still_reject_a_six_margin_vector`), and it is what produced
   the error. What was wrong is that the arity was *written in three places* —
   the table, and a literal at each call site — with the table never read. The
   table is now the single source and `expected` lost its default.

   The consequence was larger than one red test: with that config, **a live
   training run would have raised on its first step**, which is why `make smoke`
   could not have passed either.

2. *A fixture below its own tolerance.*
   `test_progress_weight_cannot_overturn_a_non_relaxable_violation` used `-1e-9`
   as "the smallest possible L3 violation", but `numerical_tolerance` is `1e-8`
   and `_canonicalize_bounded` clamps at or under it to exactly zero **by
   design**. The margin was therefore not a small violation but no violation,
   the satisfaction indicator correctly did not fire, and the test asserted
   `-0.12 > 2.0`. Corrected to `-1e-6`, with the clamped case now asserted
   explicitly alongside it so the distinction cannot be silently undone. The
   test is strengthened, not relaxed: before this it could not fail for the
   reason it was written to check.

Both test changes are migrations of the same class as `M1`'s 50-test ripple —
they encoded the four-level contract — and neither weakens an acceptance
criterion. `tests/test_hydra_preset_run_configs.py`'s pinned default mode was
updated for the same reason.

**2026-08-21 — M0, M1, M2 complete; two self-inflicted defects found and
fixed with regressions.**

1. *Overloaded flag.* `rss` was demoted with `normative_output=False`, but that
   flag means "not evaluated at all", while ADR-063 demoted it to **reported but
   never in the reward**. It would have silently stopped being published. Split
   into `normative_output` (is it evaluated) and `contributes_to_channel` (does
   its cost reach a level).
2. *Level/sub-rule name collision.* L6's sub-rule was first named
   `progress_rate`, the same as its level, so the aggregated result overwrote
   the atomic one in the component map and §3.4's "every sub-rule cost is
   exposed" was quietly broken. Nothing failed, because for a single-sub-rule
   level the two values coincide. Sub-rule renamed `advance_shortfall`; the
   collision is now rejected, with a regression test.

Also found: `evaluate_progress` already satisfied `REQ-RB51-06`, so M2 reduced
to L6 alone. `evaluate_wrongway` is kept as a pure geometric function with its
existing unit tests and only *deregistered*, since ADR-066 deletes the rule, not
the arithmetic.

Ripple from the 4->6 vector and the rename: 50 tests initially failed, all
encoding the old contract. Fixed by widening the constructed literals, updating
the enum references, and retargeting the `wrong_way` synthetic descriptor onto
`wrong_carriageway`. Next step: full-suite green, then `M3`.

**2026-08-20 — gates resolved.** `DEC-RB51-001`, `-002` and `-005` approved,
each on the recommendation. Status `AWAITING_DECISIONS` -> `IN_PROGRESS`.
Starting `M0`.

**2026-08-20 — plan created.** Repository analysis complete; the gap is larger
than "add two levels": production is at `SCAL-V1.1` with four margins and
fourteen components, one of which (`wrong_way`) the specification deletes, one
of which (`rss`) it demotes, and one of which (`speed_limit`) does not exist.
Two approval gates raised. Next step: `DEC-RB51-001` and `DEC-RB51-002`, then
`M0`.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/rulebook/v2/types.py` | Planned modification | Six levels, six-margin result |
| `src/thesis_rl/rulebook/v2/aggregation.py` | Planned modification | Summed L5/L6 with declared denominators |
| `src/thesis_rl/rulebook/v2/registry.py` | Planned modification | Level re-mapping, `wrong_way` removed, `rss` demoted |
| `src/thesis_rl/rulebook/v2/components/progress.py` | Planned modification | `Δq` over fixed `D_REF` |
| `src/thesis_rl/rulebook/v2/components/progress_rate.py` | Planned addition | L6 |
| `src/thesis_rl/rulebook/v2/components/speed_limit.py` | Planned addition | `REQ-RB51-16`, gated |
| `src/thesis_rl/reward/scalarization.py` | Planned modification | `SCAL-V1.4` |
| `src/thesis_rl/rulebook/v2/wrapper.py` | Planned modification | §7 diagnostics |
| `tests/test_rulebook_v51_*.py` | Planned addition | Mandatory matrix |

## 14. Validation Results

| Command | Result | Date | Notes |
|---|---|---|---|
| `pytest tests/test_rulebook_v51_levels.py` | **fail (expected)** | 2026-08-21 | `M0`: collection error, `aggregate_sum_component` absent — the intended reason |
| `pytest tests/test_rulebook_v51_levels.py` | **30 passed** | 2026-08-21 | `M1` meets the frozen matrix |
| `pytest tests/test_rulebook_v2_transition.py` | **20 passed** | 2026-08-21 | after wiring L6 and deregistering `wrong_way` |
| `make test` | 50 failed, 1418 passed | 2026-08-21 | first ripple measurement |
| `make test` | 51 failed, 1417 passed | 2026-08-21 | after the first fixes; remaining causes isolated |
| `make test` | *in progress* | 2026-08-21 | after the two defect fixes |
| `make test` | **2 failed, 1484 passed** | 2026-09-01 | `M6` entry measurement: the stale four-margin default config, and the sub-tolerance fixture. Both diagnosed in §11 |
| `uv run --no-sync python -m pytest -q` | **1486 passed** | 2026-09-01 | after `M6`; no test skipped, weakened or xfailed |
| `uv run --no-sync ruff check src tests scripts` | **All checks passed** | 2026-09-01 | fixes a pre-existing `F821` in `aggregation.py:140` introduced by `M1`: `RulebookV2Registry` was annotated but never imported. Guarded under `TYPE_CHECKING`, since `registry` pulls in every evaluator |
| `ruff format --check` (5 changed files) | **PASS** | 2026-09-01 | focused scope per `AGENTS.md`; `aggregation.py`'s single reformatted line is `M1`'s, in a file this change materially modifies |
| `git diff --check` | **clean** | 2026-09-01 | |
| `pytest -q` (full suite) | **1528 passed** | 2026-09-01 | after `M3`, `M4`, `M5`, `M7` and the §7 diagnostics |
| `ruff check src tests scripts` | **All checks passed** | 2026-09-01 | |
| `ruff format --check` (50 changed files) | **PASS** | 2026-09-01 | focused scope; 18 materially modified files formatted |
| `T-RB51-12` — oracle divergence, 20 records | **PASS** | 2026-09-01 | 0.0 on every redefined sub-rule |
| `T-RB51-13` — full Waymo `train` panel | **PASS** | 2026-09-01 | 1100 measured, 0 skipped, 0 errors; mean **70.70**, p1 −61.81, p5 +5.29, p50 +51.85, below standstill **3.36 %** — §5.5 reproduced to the decimal. `offroad` divergence 5.97e-06, explained by production's area epsilon |
| `M8a` — PG coverage panel | **RUN** | 2026-09-01 | 1089/1100 measured, 11 skipped (missing `length`); mean +142.06, below standstill 5.6 %; five of six L3 sub-rules never applicable |
| `M8b` — control-line decomposition | **RUN** | 2026-09-01 | `SIGNAL` 313/744 dropped, `STOP` **400/536** dropped, 122 records with no selectable signal |
| `M8c` — per-scenario applicability | **RUN** | 2026-09-01 | `crosswalk` 15/1100, `stop` 115/1100, `signal` 409/1100 |
| `make smoke` | **PASS** (exit 0) | 2026-09-01 | The direct check of what `M6` claimed was broken: an end-to-end training run under the six-level reward, which under the previous config would have raised on its first step. First smoke since `ADR-058` and the only one ever run under `gamma = 1`. **Not a substitute for `M8`'s smoke**: `M3`, `M4`, `M5` and `M7` are still missing, so this exercises the wiring, not the finished rulebook |
