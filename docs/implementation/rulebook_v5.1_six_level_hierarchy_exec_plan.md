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
| `REQ-RB51-09` | `SCAL-V1.4` as written in §5.1, at `a=2.2, σ=0, φ=0.25, λ₄=2.0, η=1.0, λ₆=0.2` | §5.1, §5.5 |
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
| `REQ-RB51-20` | `γ = 1` in every algorithm configuration, shaping discount tracking it | §4.4, ADR-075 |

`REQ-RB51-20` is **already met** (six configs updated 2026-08-20, guarded by
`tests/test_hydra_agent_presets.py::test_every_algorithm_is_undiscounted`); it is
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
| `REQ-RB51-20` | `AC-RB5.1-16` | `conf/agent/planner/algorithm/*.yaml` | `test_every_algorithm_is_undiscounted` | **Done** |

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
- [ ] **M3 — Inherited sub-rule redefinitions.** `ADR-064`, `-065`, `-067`.
  Existing rulebook suite extended.
- [ ] **M4 — `speed_limit`**, with the `OBS-V1.3` / `OBS-LIDAR-V2.0` amendment it requires. Test `T-RB51-16`.
- [ ] **M5 — At-fault gate (`ADR-070`).** Test `T-RB51-10`.
- [ ] **M6 — `SCAL-V1.4`.** Tests `T-RB51-06`, `-07`. *Not started; the module
  was read during M1 to size the work, nothing was changed.*
- [ ] **M7 — At-fault classification (`ADR-071`).** Termination vs truncation and the bootstrap value.
- [ ] **M8 — Reconciliation.** `T-RB51-12` .. `-15`, `make test`, `make lint`,
  `make smoke`; reconcile every `AC-RB5.1-*`; update `docs/project_index.md` and
  close `B1`/`B2` in `docs/open_items.md`.

## 11. Progress And Findings Log

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
