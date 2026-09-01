# ExecPlan: ACL v2.0, outcome-based windowed learning progress

## 1. Metadata

- Feature: ScenarioNet automatic curriculum learning, arm-level teacher
- Plan ID: `ACL-PROG-004`
- Authoritative specification:
  `docs/specifications/automatic_curriculum_learning_v2.0_specification.md`
  (`ACL-SN-EMA-001`, `v2.0`, `APPROVED` 2026-09-01, `Authoritative: YES`)
- Status: `AWAITING_DECISIONS` (`DEC-EP-002` gates `M4`; `M1`, `M2`, `M3` are
  unblocked)
- Created: 2026-09-01
- Last update: 2026-09-01
- Branch: `scenarionet-implementation`
- Related ADRs: ADR-077 (this contract), ADR-029/ADR-030/ADR-032/ADR-028
  (carried forward), ADR-016 (commit ordering), ADR-024 (data abort),
  ADR-063/ADR-066/ADR-068/ADR-070/ADR-071/ADR-072/ADR-076 (`RULEBOOK-V5.1`
  decisions this contract consumes)
- Supersedes as the implementation record for `ACL-SN-EMA-001`:
  `automatic_curriculum_learning_v1.3_exec_plan.md` (`ACL-SN-CAT-003`)

## 2. Objective And Scope

### Observable capability

The curriculum teacher stops deriving its per-arm feedback from a
prediction-error learning potential and derives it instead from a windowed
ordinal comparison of environment and Rulebook **outcomes**. Observably:

- a structurally noisy but stationary arm receives neutral feedback and no
  selection advantage, where `v1.3` gave it the highest score (`FIND-006`);
- the neutral state of the curriculum is exactly uniform sampling, and the
  teacher departs from it only on gated, measured evidence;
- the curriculum is **identical across PPO, TD3 and SAC** given identical
  outcomes, so the reward-setting x curriculum comparison compares one
  curriculum rather than three;
- the scenario buffer becomes an arm-balanced recency memory instead of a
  usefulness-ranked one.

### Success

Every `AC-201`…`AC-217` of the specification passes as an executed test, the
full suite and `make smoke` pass, and the measurement instrumentation of `M1`
produces per-episode level statistics usable to compute `SNR` offline.

### In scope

Everything in §2 "In Scope" of the specification, plus one addition this plan
records explicitly: the **per-episode measurement sink must work with
`curriculum=disabled`**, because `LIM-201` requires the `SNR` to be measured on
a uniform-sampling run and the A/C cells of the experimental matrix (§2.1) are
exactly such runs.

### Out of scope

Everything in §2 "Out Of Scope" of the specification. In particular: no change
to reward scalarization, the observation schema, transition replay, dataset
splits, or the frozen catalog; no per-record progress estimation; no ablation of
`H`, `buffer_capacity`, `band_level` or `tau`.

### 2.1 Experimental matrix this plan serves

Recorded from the user's instruction of 2026-09-01. A 2x2 learnability screen,
fixed at semantic observation, MLP encoder/decoder, SAC, `n_steps = 3`, PER
enabled:

| cell | reward | curriculum | needs this plan? |
|---|---|---|---|
| A | MetaDrive native (`monitor_only`) | `disabled` | only `M1` |
| B | MetaDrive native (`monitor_only`) | `scenario_acl` v2.0 | full |
| C | rulebook scalarized (`scalar_reward`) | `disabled` | only `M1` |
| D | rulebook scalarized (`scalar_reward`) | `scenario_acl` v2.0 | full |

The letter-to-cell assignment above is the assistant's reading of "reward
nativo, rulebook scalarizzato, con e senza ACL" and is `AWAITING_CONFIRMATION`;
nothing in the plan depends on the lettering, only on the four cells.

Two consequences drive the milestone order:

1. **A and C are runnable after `M1` alone**, and they are the two cells that
   answer "is the rulebook+scalarization reward learnable at all, compared with
   the native reward" — the question `FIND-013` left open and `RULEBOOK-V5.1`
   was written to fix.
2. **The `SNR` of `LIM-201` can only be measured on those same runs**, and only
   if `M1`'s instrumentation is already recording per-episode level statistics.
   Running A/C without `M1` answers the learnability question but not the `SNR`
   question, and the runs would have to be repeated.

The teacher consumes Rulebook outcomes, not reward: under `monitor_only` the
Rulebook is still evaluated, so cells B and D are both well defined. Only SAC
is exercised, so `AC-215` (algorithm independence) is established by test and
not by this matrix.

### Compatibility

Breaking, as declared in specification §11: checkpoint schema
`acl_ema_v3` -> `acl_progress_v1` with no migration; `ScenarioRecord` gains
`last_generate_step` and loses the ranking meaning of `usefulness`; the RNG call
order changes. Runs in progress must restart — already required independently by
`gamma = 1`, `speed_limit` and the observation amendments (open item `D4`).

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-001` | Five teacher dimensions in the fixed order `L1 -> L2 -> L3 -> T -> L5` | §6 `REQ-001` |
| `REQ-002` | Episodic outcome statistics per level and for the task, with the L1 / step-fraction asymmetry | §6 `REQ-002` |
| `REQ-003` | Comparable keys and per-arm per-dimension sliding windows of `2H`, Generate-only | §6 `REQ-003` |
| `REQ-004` | Vargha-Delaney ordinal progress measure `G` | §6 `REQ-004` |
| `REQ-005` | Exact conditional permutation neutrality band, uncorrected, priority-ordered selection | §6 `REQ-005` |
| `REQ-006` | Learnability gate `4g(1-g)` on the per-dimension statistic | §6 `REQ-006` |
| `REQ-007` | Signed arm feedback `A = 0.5 + D(G - 0.5)` | §6 `REQ-007` |
| `REQ-008` | Balanced calibration of `K * 2H` valid Generate episodes and Replay activation | §6 `REQ-008` |
| `REQ-009` | EMA score, temperature softmax, exploration floor, uniform neutral state | §6 `REQ-009` |
| `REQ-010` | Arm-balanced recency buffer, admission and eviction on `last_generate_step` | §6 `REQ-010` |
| `REQ-011` | Replay sampling `0.70 * p_i/n_i + 0.30 * staleness` | §6 `REQ-011` |
| `REQ-012` | Generate eligibility and per-arm coverage cycles, unchanged from `v1.3` | §6 `REQ-012` |
| `REQ-013` | Prediction-error LP retained, strictly inert, diagnostic-only | §6 `REQ-013` |
| `REQ-014` | Persistence under `acl_progress_v1`, schema rejection, resume | §6 `REQ-014` |

No requirement is excluded.

## 4. Current Repository Analysis

All statements below are `VERIFIED` by inspection on 2026-09-01 unless labelled
otherwise.

### Rulebook interface (the input side)

- `src/thesis_rl/rulebook/v2/wrapper.py:285` emits `rule_reward_vector` (the six
  level margins) and `:295` emits `rule_components`, which contains the macro
  results keyed by channel value alongside the atomic ones
  (`aggregation.py:219-231`). Both assumptions of specification §3.2 hold under
  the renamed levels.
- `src/thesis_rl/agent/agent.py:2617` `_extract_rule_margins` and `:2642`
  `_extract_rule_applicability` already read exactly these two structures, on
  the sequential path, for `EVAL-PROTOCOL` `REQ-008`. They are reusable as-is.
- `components/road.py:138` makes `offroad` unconditionally applicable, so L3 is
  applicable at every step; `components/collision.py:112`/`:224` make L1
  applicable only on a new **at-fault** contact; `aggregation.py:63`
  `aggregate_sum_component` implements L5's fixed-denominator normalized sum.

### ACL package (the change surface)

- `curriculum/scenario_acl/mab.py:19` `ScenarioArmBandit` carries the EMA score,
  `probabilities(eligible_mask=...)` with the `ADR-028` renormalization, and the
  per-arm reward-scale machinery (`:126`, `:133`, `:147`) that `REQ-013`
  demotes to diagnostic.
- `buffer.py:20` `ScenarioBuffer` maintains a usefulness rank (`_refresh_ranks`,
  `:37`) and `insert` (`:43`) replaces the worst record by usefulness —
  precisely what `REQ-010` removes.
- `ranking.py:11` `assign_ranks` and `:18` `compute_replay_probabilities`
  implement the `rank^-beta` mixture that `REQ-011` replaces.
- `record.py:8` `ScenarioRecord` carries `usefulness`/`usefulness_norm`/`rank`;
  `catalog_state.py:45` already holds a per-record last-Generate episode id, so
  `REQ-010`'s eviction key has an existing analogue to align with.
- `usefulness.py` holds the whole prediction-error chain that `REQ-013` keeps
  running and makes inert.
- `selection.py:86` filters candidates by `record.primary_arm`; `:154`
  implements the `ADR-032` decoupling that `REQ-012` carries forward unchanged.
- `driver.py` owns the commit path, quarantine handling (`:742`) and persistence
  (`:1256` writes the periodic checkpoint pair).

### Per-episode data available today

- `agent.py:1362-1385` builds the per-episode ACL payload on the vectorized
  path. It carries `reward`, `success`, `collision`, `out_of_road`,
  `route_completion`, `env_reward` and `scalar_rule_reward` — **no per-level
  cost, applicability or violated-step counts**. This is the gap `REQ-002`
  closes and the reason `FIND-013` could not reconstruct R2.
- `envs/thesis_scenario_env.py:1034` puts the record's `primary_arm` into the
  episode payload as `scenario_arm`/`arm`, **independently of the curriculum**,
  so arm-grouping is available on a `curriculum=disabled` run.
- `runtime/io/csv_recorder.py` schemas: `train_chunks.csv` is per chunk,
  `eval_episodes.csv` is per evaluation episode and already carries
  `primary_arm`. **There is no per-training-episode sink outside the ACL
  driver.** `M1` must add one, otherwise the A/C cells produce no `SNR` data.

### Behavior to preserve

`ADR-016` commit ordering `(collection_tick, worker_id, episode_id)`;
`ADR-024` data-abort and quarantine semantics including the run-local
persistence and vectorized propagation; `ADR-028` eligibility renormalization as
a residual safety net; `ADR-032` Generate/catalog decoupling and per-arm
coverage cycles; the vectorized selective-reset protocol.

## 5. Assumptions And Invariants

| Item | Value | Established by | Violation handling |
|---|---|---|---|
| Level margins | `m_k(t) in [-1, 0]`, cost `= -m` | specification §3.2, `aggregation.py` | fatal |
| Level applicability | bool per level per step, read not recomputed | `wrapper.py:295` | fatal if absent while ACL enabled |
| `H` | `20`, so windows are `2H = 40` and calibration is `240` valid Generate episodes | `DEC-205` | configuration validation |
| Permutation arithmetic | exact `int64`; `C(40,20) = 137846528820 < 2^63` | specification §7 | reject `H` whose `C(2H,H)` overflows |
| Commit order | `(collection_tick, worker_id, episode_id)` | `ADR-016` | fatal on ordering violation |
| Data abort | produces no teacher observation of any kind | `ADR-024` | episode discarded from all ACL state |
| Not-at-fault truncation | ordinary episode, no special handling | `DEC-207` | none; rate logged (`LIM-210`) |
| Windows | Generate episodes only | `RAT-205` | Replay must not append |
| Neutral state | every `A_i = 0.5` implies `p` uniform `1/6` | `REQ-009` | asserted by `AC-209` |

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-EP-001` | Implementation detail | Where do the per-episode level statistics land for a `curriculum=disabled` run? | A: new `train_episodes.csv` with a fixed schema, written on every training path / B: extend the JSONL run-event log only / C: extend `train_chunks.csv` with aggregates | **A**, mirroring `eval_episodes.csv`, because the `SNR` procedure needs per-episode rows grouped by arm and ordered by global step, which an aggregate cannot provide and a JSONL-only sink makes awkward to analyse with the existing `make analyze` tooling | New artifact, additive; no schema of an existing CSV changes | Assistant decision, recorded |
| `DEC-EP-002` | Mandatory test change | Three mandatory tests pin semantics `REQ-010`/`REQ-011` remove | A: rewrite the three to pin the new contract / B: keep and mark expected-to-fail / C: delete | **A**. `test_scenario_buffer_replaces_worst_when_capacity_is_full` and `test_compute_replay_probabilities_mix_usefulness_and_staleness` (`tests/test_scenario_acl_buffer.py:60`, `:72`) and `test_learning_potential_alone_determines_buffer_usefulness` (`tests/test_scenario_acl_usefulness.py:47`) assert usefulness-ranked admission and replay, which the approved specification replaces. No assertion is weakened: each is rewritten to pin the arm-balanced contract with the same strictness | Blocks `M4` | **Awaiting approval** |
| `DEC-EP-003` | Implementation detail | Does the diagnostic LP chain keep the reward-scale EMA and rank window, or only `U_raw`? | A: keep the whole chain / B: keep `U_raw` plus reward magnitude and reconstruct offline | **A**, as specification `REQ-013` requires it verbatim; `AC-212` makes inertness mechanical, so the cost is persistence size only | Persistence payload larger | Assistant decision, recorded |

`DEC-EP-002` is the only gate. `M1`, `M2` and `M3` touch none of the three
tests and may proceed.

## 7. Proposed Design

### New module

`src/thesis_rl/curriculum/scenario_acl/progress.py` — pure, no I/O, no RNG:

- `EpisodeOutcome`: the per-episode statistics of `REQ-002` (`C_k`, `N_app_k`,
  `N_viol_k`, `F_k`, `v_1`, `success`, `route_completion`).
- `DimensionWindow`: the `2H` FIFO of one dimension of one arm, with
  `is_available`, `older`, `recent`.
- `vargha_delaney(recent, older, key)` -> `G` (`REQ-004`), computed through
  pooled midranks.
- `exact_permutation_p(recent, older, key)` -> `p^perm` (`REQ-005`), dynamic
  programming over tie groups in doubled integer rank units, `int64` only, with
  an explicit rejection of any `H` whose `C(2H,H)` would overflow. Reference
  construction and validation are recorded in
  `docs/audits/acl_v2_teacher_power_analysis_2026-07-31/`.
- `learnability_gate(g)` -> `D` (`REQ-006`), `arm_feedback(G, D)` -> `A`
  (`REQ-007`), and `select_dimension(...)` -> `d*` implementing the
  `(L1, L2, L3, T, L5)` priority order.

### Modified modules

- `agent.py` — per-slot accumulation of the four consumed level costs, their
  applicability and violated-step counts over each episode, on the vectorized
  path, reusing `_extract_rule_margins`/`_extract_rule_applicability`; the
  per-episode ACL payload gains the `REQ-002` statistics; the same accumulation
  feeds `DEC-EP-001`'s sink so it is produced with the curriculum disabled.
- `mab.py` — `feedback: windowed_outcome_progress`; the reward-scale EMA and
  the rank normalizer stay but become diagnostic outputs only.
- `buffer.py`, `ranking.py`, `record.py` — arm-balanced admission and eviction
  on `last_generate_step`; `P_progress = p_i^full / n_i`; the usefulness rank
  and `beta` are removed from the ranking path.
- `driver.py` — calibration phase and Replay gate (`REQ-008`), window updates in
  `ADR-016` order at commit time, teacher-update logging (§10), persistence
  under `acl_progress_v1` with rejection of the four legacy schemas.
- `config.py` and `conf/curriculum/scenario_acl.yaml` — `progress.*` fields,
  rejection of `replay_sampling.beta`, of the legacy `mab.feedback` value and of
  `progress.multiplicity_correction`.

### Errors and fallbacks

No new fallback is introduced. Every failure listed in specification §10 as
fatal fails closed with a message naming `ACL-SN-EMA-001 v2.0`.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-001` | `AC-206` | `progress.py` (`select_dimension`) | `TEST-PROG-006` | Planned |
| `REQ-002` | `AC-203`, `AC-203b`, `AC-204` | `agent.py`, `progress.py` (`EpisodeOutcome`) | `TEST-PROG-003`, `-003b`, `-004` | Planned |
| `REQ-003` | `AC-211` | `progress.py` (`DimensionWindow`), `driver.py` | `TEST-PROG-011` | Planned |
| `REQ-004` | `AC-201`, `AC-202`, `AC-202b`, `AC-217` | `progress.py` (`vargha_delaney`) | `TEST-PROG-001`, `-002`, `-002b`, `-017` | Planned |
| `REQ-005` | `AC-205`, `AC-206` | `progress.py` (`exact_permutation_p`) | `TEST-PROG-005`, `-006` | Planned |
| `REQ-006` | `AC-207`, `AC-203b` | `progress.py` (`learnability_gate`) | `TEST-PROG-007` | Planned |
| `REQ-007` | `AC-207`, `AC-209`, `AC-202b` | `progress.py` (`arm_feedback`) | `TEST-PROG-007`, `-009` | Planned |
| `REQ-008` | `AC-208` | `driver.py` | `TEST-PROG-008` | Planned |
| `REQ-009` | `AC-209` | `mab.py` | `TEST-PROG-009` | Planned |
| `REQ-010` | `AC-210`, `AC-213` | `buffer.py`, `record.py` | `TEST-PROG-010`, `-013` | Planned |
| `REQ-011` | `AC-210` | `ranking.py` | `TEST-PROG-010` | Planned |
| `REQ-012` | `AC-213` | `selection.py` (unchanged) | existing coverage suite | Planned |
| `REQ-013` | `AC-212`, `AC-215` | `usefulness.py`, `driver.py` | `TEST-PROG-012`, `-015` | Planned |
| `REQ-014` | `AC-214` | `driver.py` | `TEST-PROG-014` | Planned |

## 9. Test Strategy

Acceptance criteria are those of the specification (`AC-201`…`AC-217`); they are
not restated here. Minimum mandatory matrix, frozen before production changes:

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-PROG-001` | Unit | Scale invariance of `G` | two exchangeable windows, noise scale over `250x` | mean `G` flat at `0.5`, fire rate flat, no monotone trend | `REQ-004` |
| `TEST-PROG-002` | Unit | Monotone response to improvement | three separation levels, and windows exchanged | `G > 0.5` increasing; symmetric below `0.5` when swapped | `REQ-004` |
| `TEST-PROG-002b` | Integration | Noisy stationary arm not preferred, closed loop | seeded 6-arm loop, one stationary arm at several times the noise | its mean `A` is `0.5`, fewest Generate draws, terminal `p_i` at the floor | `REQ-004`, `REQ-007`, `REQ-009` |
| `TEST-PROG-003` | Unit | L1 statistic on collision-free and not-at-fault episodes | episodes with no contact, with two at-fault contacts, with one not-at-fault contact | `C_1 = 0/0.7/0`; `v_1 = 0/1/0`; window always appended; `T` appended in every case | `REQ-002` |
| `TEST-PROG-003b` | Unit | Step-fraction gate does not saturate | 100-step episodes, L2 applicable at 80, violated at 12 and at 1; and 40 standstill steps | `F = 0.15` and `0.0125`; `D > 0` on a window of violating episodes; standstill steps absent from `N_app` | `REQ-002`, `REQ-006` |
| `TEST-PROG-004` | Unit | Unobserved level is not zero cost | episode with L2 applicable at zero steps; episode with L5 applicable at zero steps | no observation appended, availability unchanged, `C` undefined | `REQ-002` |
| `TEST-PROG-005` | Unit | Exact permutation band | constant windows; fully separated windows; six inputs at `H = 10` incl. two tie-heavy | `p = 1.0`; `p = 2/C(2H,H)`; agreement to `1e-12` with exhaustive enumeration; zero RNG draws; `H` overflowing `2^63` rejected | `REQ-005` |
| `TEST-PROG-006` | Unit | Priority-ordered selection, no correction | L1 neutral, L2 fires, `T` fires with smaller `p`; then only `T` and `L5` fire | `d* = L2`; then `d* = T`; threshold independent of how many dimensions are available; `multiplicity_correction` rejected by config validation | `REQ-001`, `REQ-005` |
| `TEST-PROG-007` | Unit | Gate closes at both extremes, on the right statistic | `g in {0, 1, 0.5}` with `G = 0.8`, on each dimension | `A = 0.5, 0.5, 0.8`; consuming `v` on L2/L3/L5 or `F` on L1 fails | `REQ-006`, `REQ-007` |
| `TEST-PROG-008` | Integration | Calibration completeness and Replay gate | seeded run from fresh state | all-Generate until `2H` per arm; `q_i` unchanged; counts differ by at most one; data-aborts do not count; Replay only after both conditions | `REQ-008` |
| `TEST-PROG-009` | Unit | Neutral state is uniform | repeated `A_i = 0.5`; then one arm at `0.8` | `p` uniform within `1e-9`; then that arm above `1/6` with the floor respected elsewhere | `REQ-009` |
| `TEST-PROG-010` | Unit | Buffer balance, eviction key, replay mass | full buffer at 250, unbalanced Generate commits | `abs(n_i - n_i*) <= 1`; eviction by smallest `last_generate_step` even when a Replay refreshed `last_seen_step`; `sum P_progress` over an arm equals `p_i^full` within `1e-9` | `REQ-010`, `REQ-011` |
| `TEST-PROG-011` | Unit | Replay does not move the teacher | arbitrary Replay commits on a full-window arm | `q_i`, windows and `N_gen` unchanged; only `last_seen_step`, visits and diagnostics change | `REQ-003`, `REQ-009`, `REQ-010` |
| `TEST-PROG-012` | Integration | Diagnostic LP is inert | two runs, identical seed, LP forced to different finite values | mode, arm, scenario, `q_i`, buffer and evictions bit-identical; only `U`/`U_scaled`/`U_norm` differ | `REQ-013` |
| `TEST-PROG-013` | Integration | Data-abort isolation | Generate episode ending in a typed data abort | no window, no counter, no score change; record quarantined and removed from the buffer; still marked visited | `REQ-002`, `REQ-008`, `REQ-010`, `REQ-012` |
| `TEST-PROG-014` | Integration | Checkpoint round-trip and rejection | mid-run state, partial windows, mixed availability, incomplete calibration, full buffer | exact round-trip and identical subsequent sequence; `acl_ema_v1/v2/v3` and cumulative payloads rejected explicitly | `REQ-014` |
| `TEST-PROG-015` | Unit | Algorithm independence | identical outcomes replayed under PPO, TD3, SAC | identical `G`, `D`, `A_i`, `q_i` trajectories | `REQ-002`, `REQ-013` |
| `TEST-PROG-016` | Config | Default composition and rejected fields | root config with no override; `curriculum=disabled`; configs carrying `replay_sampling.beta` or the legacy feedback value | v2.0 profile; disabled profile; explicit validation failure | `REQ-009`, `REQ-010` |
| `TEST-PROG-017` | Integration | Emergence, plateau, handover | seeded closed loop, three arms improving at staggered onsets, stated `SNR >= 1.25` | each `p_i` rises above `1/K`, peaks in onset order, returns to `1/K`; the stated `SNR` is asserted in the test name or docstring | `REQ-004`…`REQ-009` |
| `TEST-PROG-018` | Instrumentation | Per-episode statistics exist with the curriculum disabled | short `curriculum=disabled` run | the `DEC-EP-001` sink contains one row per training episode with arm, global step and the four levels' statistics | `REQ-002` (measurement half) |

### Validation commands

All exist in the repository; none is invented.

- focused: `uv run --no-sync python -m pytest -q tests/test_scenario_acl_*.py`
- new module: `uv run --no-sync python -m pytest -q tests/test_scenario_acl_progress.py`
- full suite: `make test` (or `uv run --no-sync python -m pytest -q` inside a
  provisioned container)
- lint: `make lint`
- formatting, focused: `make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/curriculum/scenario_acl tests/test_scenario_acl_progress.py"`
- smoke: `make smoke`, and a second run with `env.vectorized.enabled=true`
- config: `make config` and `make config-gpu`
- whitespace: `git diff --check`

No global mypy target exists; static checking stays scoped to the new module's
annotations. Coverage has no configured threshold and none is invented.

## 10. Milestones

### `M0` — acceptance tests first (pure functions)

- [ ] Status: not started. Depends on nothing.
- Files: `tests/test_scenario_acl_progress.py` (new).
- Tasks: write `TEST-PROG-001`…`-007` against the not-yet-existing
  `progress.py` API, including the exhaustive-enumeration cross-check at
  `H = 10`.
- Command: the focused pytest invocation above; expected to fail until `M2`.

### `M1` — measurement instrumentation only

- [ ] Status: not started. **Unblocks cells A and C of the experimental
  matrix**, and is the milestone that makes the `LIM-201` `SNR` measurable.
- Files: `src/thesis_rl/agent/agent.py`,
  `src/thesis_rl/runtime/io/csv_recorder.py`,
  `src/thesis_rl/runtime/loops/train_loop.py`, plus the sequential path.
- Tasks: accumulate per slot, per episode, the cost, applicability and
  violated-step counts of L1, L2, L3 and L5; extend the ACL episode payload;
  add the `DEC-EP-001` per-training-episode sink carrying arm, global step,
  source and the four levels' statistics; make it work with the curriculum
  disabled.
- Tests: `TEST-PROG-018`, plus `TEST-PROG-003`/`-003b`/`-004` at the statistics
  level.
- Inert by construction: nothing here feeds a teacher, a reward or an
  observation.
- Completion evidence: a short `curriculum=disabled` smoke whose sink contains
  the expected rows, and the measured per-step overhead (`LIM-208` item 1).

### `M2` — `progress.py`

- [ ] Status: not started. Depends on `M0`.
- Files: `src/thesis_rl/curriculum/scenario_acl/progress.py` (new).
- Tasks: keys, windows, `G`, exact permutation band, gate, feedback, dimension
  selection.
- Tests: `M0`'s suite must now pass, plus `TEST-PROG-002b`.
- Completion evidence: the `AC-201` flatness and the `AC-205` exhaustive
  agreement reproduced by the implementation, not only by the audit script; the
  per-call cost measured (`LIM-208` item 2).

### `M3` — bandit and configuration

- [ ] Status: not started. Depends on `M2`.
- Files: `mab.py`, `config.py`, `conf/curriculum/scenario_acl.yaml`.
- Tasks: `windowed_outcome_progress` feedback; reward-scale machinery demoted to
  diagnostic; `progress.*` fields; rejection of the three legacy fields.
- Tests: `TEST-PROG-009`, `TEST-PROG-016`.

### `M4` — buffer and replay sampling

- [ ] Status: **blocked by `DEC-EP-002`**.
- Files: `buffer.py`, `ranking.py`, `record.py`.
- Tasks: arm-balanced admission and eviction on `last_generate_step`;
  `P_progress = p_i^full / n_i`; usefulness removed from the ranking path.
- Tests: `TEST-PROG-010`, plus the three rewritten mandatory tests.

### `M5` — calibration, commit path, teacher wiring

- [ ] Status: not started. Depends on `M2`, `M3`, `M4`.
- Files: `driver.py`, `vectorized.py`, `runtime.py`.
- Tests: `TEST-PROG-008`, `-011`, `-013`, `-017`.

### `M6` — persistence and resume

- [ ] Status: not started. Depends on `M5`.
- Files: `driver.py`, `record.py`.
- Tests: `TEST-PROG-014`.

### `M7` — inertness and diagnostics

- [ ] Status: not started. Depends on `M5`.
- Files: `usefulness.py`, `driver.py`, logging sites.
- Tasks: `AC-212` inertness; the §10 per-update log fields; the `Delta s`
  diagnostic of `LIM-203`.
- Tests: `TEST-PROG-012`, `TEST-PROG-015`.

### `M8` — integration and cost

- [ ] Status: not started. Depends on `M6`, `M7`.
- Tasks: full suite, lint, focused format-check, `make config`/`make config-gpu`,
  and two `make smoke` runs (sequential and vectorized), recording the three
  costs of `LIM-208` rather than assuming them.

### `M9` — `SNR` measurement and the `H` decision

- [ ] Status: not started. Depends on `M1` **and on the A/C runs existing**, not
  on `M2`…`M8`.
- Tasks: apply the `LIM-201` procedure to a `curriculum=disabled` run under the
  `RULEBOOK-V5.1` reward; report `SNR` per arm and per dimension; confirm
  `H = 20` or trigger `DEC-205`'s declared revision to `H = 30`.
- Note: this is a post-hoc validation, explicitly **not** a gate on any earlier
  milestone.

## 11. Progress And Findings Log

- **2026-09-01** — Plan created after `ACL-SN-EMA-001 v2.0` was approved
  (ADR-077). Repository analysis of §4 performed and recorded. Key finding
  driving the milestone order: there is **no per-training-episode sink outside
  the ACL driver**, so a `curriculum=disabled` learnability run produces no data
  from which `LIM-201`'s `SNR` could be computed. `M1` is therefore separated
  from the rest of the implementation and placed first, so that cells A and C of
  the experimental matrix answer the learnability question and the `SNR`
  question in one pass instead of two. Next step: approval of `DEC-EP-002`, then
  `M0`.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/curriculum/scenario_acl/progress.py` | Planned addition | `REQ-003`…`REQ-007` |
| `src/thesis_rl/curriculum/scenario_acl/mab.py` | Planned modification | `REQ-009`, `REQ-013` |
| `src/thesis_rl/curriculum/scenario_acl/buffer.py` | Planned modification | `REQ-010` |
| `src/thesis_rl/curriculum/scenario_acl/ranking.py` | Planned modification | `REQ-011` |
| `src/thesis_rl/curriculum/scenario_acl/record.py` | Planned modification | `REQ-010`, `REQ-014` |
| `src/thesis_rl/curriculum/scenario_acl/driver.py` | Planned modification | `REQ-008`, `REQ-014`, §10 |
| `src/thesis_rl/curriculum/scenario_acl/usefulness.py` | Planned modification | `REQ-013` |
| `src/thesis_rl/agent/agent.py` | Planned modification | `REQ-002` accumulation |
| `src/thesis_rl/runtime/io/csv_recorder.py` | Planned modification | `DEC-EP-001` sink |
| `src/thesis_rl/runtime/loops/train_loop.py` | Planned modification | `DEC-EP-001` sink |
| `conf/curriculum/scenario_acl.yaml` | Planned modification | §9 configuration |
| `tests/test_scenario_acl_progress.py` | Planned addition | `TEST-PROG-001`…`-007`, `-017` |
| `tests/test_scenario_acl_buffer.py` | Planned modification | `DEC-EP-002` |
| `tests/test_scenario_acl_usefulness.py` | Planned modification | `DEC-EP-002` |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| — | `NOT_RUN` | — | No production change has been made yet; the plan is awaiting `DEC-EP-002`. |

## 15. Final Reconciliation

Not applicable yet: no requirement is implemented. This section is completed at
`M8`, and `M9` is recorded separately because it depends on a training run
rather than on code.
