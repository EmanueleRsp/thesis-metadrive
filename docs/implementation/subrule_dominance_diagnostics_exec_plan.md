# ExecPlan: Sub-Rule Dominance And Cost Diagnostics

## 1. Metadata

- Feature: `subrule_dominance_diagnostics`
- Plan ID: `EP-SUBRULE-DIAG`
- Authoritative specification: none amended. This is additive diagnostic
  reporting under `docs/specifications/evaluation_protocol_v1.0_specification.md`
  (`EVAL-PROTOCOL`, version `1.0`, `AUTHORITATIVE`); see `DEC-SUB-001`, which
  is an approval gate before implementation starts.
- Related specifications (read, not changed):
  - `docs/specifications/rulebook_v4.7_specification.md` §6 (R2/R3 sub-metrics)
  - `docs/specifications/rulebook_v4.8_specification.md` (R2 composition)
  - `docs/specifications/rulebook_v4.9_specification.md` (R1 cost)
- Related ADRs: none yet; `DEC-SUB-001` and `DEC-SUB-002` require one if approved
  as material.
- Status: `IMPLEMENTED`
- Created: 2026-07-26
- Last updated: 2026-07-26
- Branch: `scenarionet-implementation`

## 2. Objective And Scope

### Problem being solved

The rulebook aggregates several heterogeneous sub-metrics into one macro cost
by maximum (`aggregate_max_component`,
`src/thesis_rl/rulebook/v2/aggregation.py`):

```
c_2(t) = max{ q_RSS,long , q_RSS,lat , q_TTC , q_clear,VRU }
c_3(t) = max{ offroad, wrongway, solid_line, dashed_line, signal, stop,
              crosswalk, vehicle_yield }
```

Censi et al. Definition 16 licenses this: aggregating *equi-ranked* rules
through any `alpha` that embeds the product pre-order into `R+`, which `max`
satisfies. What the framework does **not** supply is any guarantee that a cost
of 0.5 from TTC and a cost of 0.5 from RSS represent comparable real risk:
each is normalized against its own threshold, not against a common risk scale.
That assumption of intra-priority commensurability is currently undocumented
and, more importantly, **unmeasured**.

Its practical weight differs sharply by consumer:

- under the default scalarizer `bounded_satisfaction_rank` (`SCAL-V1.0` §7.5)
  the categorical term uses `I_k = 1[m_k = 0]`, a boolean. The cardinal value
  of the `max` reaches the reward only through the continuous tie-breaker,
  bounded to `±0.25`. Low impact;
- a **lexicographic learner optimizes `J_2(pi) = E[sum_t gamma^t m_2(t)]`
  directly**, with no boolean gate in between. There the cardinal value of the
  `max` *is* the signal. If one sub-metric saturates far more readily than its
  siblings it will dominate the maximum almost always, and `R2` silently
  degenerates into a single-sub-rule objective while still being reported as
  "interaction safety". High impact.

Nothing in the current analysis output can detect that: verified, the whole
reporting chain carries **macro rules only** (`wrapper.py:211` sets
`rule_names = [rule.value for rule in MACRO_RULE_ORDER]` and
`rule_reward_vector = result.margins`, a 4-vector; `Agent._extract_rule_margins`
reads exactly those). Sub-rule results exist per step in
`step_info["rule_components"]` and reach disk only inside
`*.trajectory.jsonl`, which is written for a **bounded tracked subset** of
episodes (`eval_artifacts.py:289` tracked-subset filter, plus the
`max_final_videos` cap at `eval_artifacts.py:441`) — a sample selected for
video rendering, not a representative one.

### Objective

Make sub-rule behaviour visible in the standard end-of-run analysis report,
representatively over all evaluation episodes, so that three questions can be
answered from artifacts rather than assumed:

1. **Dominance** — within each macro rule, how often is each sub-rule the
   `worst_component`, counted only over steps where the macro is actually
   violated?
2. **Calibration** — how are the sub-rules' cost distributions positioned
   relative to each other? A sub-rule whose median cost is an order of
   magnitude above its siblings is the mechanism behind any dominance found.
3. **Liveness** — is any sub-rule effectively never applicable, or never the
   worst? (Plausible for `clearance`, restricted to VRUs by v4.8.)

Success is recognised when `make analyze RUN_PROFILE=<profile>` emits the two
tables and two plots of §7 for a multi-seed run, disaggregated by scenario
source, without changing any pre-existing metric value.

### In Scope

- Online per-sub-rule aggregation during evaluation, alongside the existing
  macro-rule aggregation in `Agent.evaluate()`.
- New `subrule_metrics.csv` run artifact and its aggregation into
  `subrule_metrics_all_runs.csv`.
- Two tables and two plots in the analysis output (§7).
- Tests for aggregation correctness, the violated-only conditioning, and
  schema stability.

### Out Of Scope

- Any change to the rulebook, to `aggregate_max_component`, to sub-rule
  thresholds, or to macro costs. This plan **measures**, it does not modify.
- Any change to existing metrics, tables, plots, or their values.
- Any change to scalarization or to the reward.
- Acting on the findings. If the diagnostics reveal a dominance or calibration
  problem, the response is a separate approved decision, not this plan.
- Trajectory-log-based analysis (rejected: non-representative sample, §4).

### Compatibility constraints

Purely additive. New CSV, new columns, new outputs. `aggregate_runs.py` must
tolerate runs recorded before this feature (missing `subrule_metrics.csv`)
without failing, so historical runs stay analysable — see `DEC-SUB-003`.

## 3. Requirements

| ID | Requirement | Rationale |
|---|---|---|
| `REQ-SUB-01` | Per-sub-rule applicability, violation rate, and cost statistics are aggregated over **all** evaluation episodes | §2; the tracked subset is not representative |
| `REQ-SUB-02` | Dominance share is counted only over steps where the parent macro rule is violated (`cost > 0`) | A `max` over all-zero costs picks an arbitrary winner and would pollute the frequencies |
| `REQ-SUB-03` | Sub-rule statistics use cost terminology (`[0,1]`, positive), never margin terminology | Sub-rules are cost-based; macro margins are `[-1,0]`. Mirrors the R1--R3/R4 separation `EVAL-PROTOCOL` REQ-007 already enforces |
| `REQ-SUB-04` | Outputs are disaggregated by scenario source | Waymo-urban and PG-highway distributions differ; a pooled mean hides both |
| `REQ-SUB-05` | Simultaneous-violation share is reported per macro rule | If two sub-rules are almost never violated together, the `max` is nearly inert and the commensurability concern largely dissolves |
| `REQ-SUB-06` | No pre-existing metric, table, or plot changes value | Additive-only constraint |
| `REQ-SUB-07` | Outputs are explicitly labelled diagnostic, never primary comparison metrics | `DEC-SUB-001` |

## 4. Current Repository Analysis

All rows `VERIFIED` by reading the code unless marked otherwise.

| Item | Path | Note |
|---|---|---|
| Sub-rule results per step | `src/thesis_rl/rulebook/v2/wrapper.py:220` | `info["rule_components"] = {name: component.to_dict()}` — **all** components, sub-rules and macros |
| Component payload | `types.py:385` `RuleComponentResult.to_dict` | `name`, `cost`, `raw`, `applicable`, `evaluable`, `status`, `diagnostics` |
| Dominance source | `aggregation.py:41,47` | `worst_component` written into both `raw` and `diagnostics` of each macro component |
| Macro-only reporting | `wrapper.py:211`; `agent.py:2544` | `rule_names` = `MACRO_RULE_ORDER` only; this is why sub-rules are absent downstream |
| Applicability reader (precedent to mirror) | `agent.py:2568` `_extract_rule_applicability` | Already reads `rule_components`; the new extractor sits beside it |
| Macro aggregation site | `agent.py` ~1849--1935 | `ep_rule_*` accumulators per episode, then seed-level `per_rule` rows |
| CSV emission | `eval_loop.py:44`, `train_loop.py:501` | `_append_rule_metrics_rows`, **duplicated in both loops**; 1 call site in `eval_loop`, 5 in `train_loop` |
| CSV schema | `csv_recorder.py:180` | Fixed column list per file |
| Run aggregation | `analysis/aggregate/aggregate_runs.py:18,59` | File allow-list plus per-file extra group keys (`rule_metrics.csv` → `("eval_type", "scenario_set")`) |
| Table builder | `analysis/tables/make_rulebook_tables.py` | Produces `rulebook_compliance.*` and `rule_violation_by_rule.*` |
| Plot builder | `analysis/plots/make_plots.py:416` | `_plot_rule_violation_by_rule` |
| Trajectory persistence | `eval_artifacts.py:289,441` | Tracked-subset filter + `max_final_videos` cap — the reason online aggregation is required |
| Scenario source availability | `csv_recorder.py:123,128` | `rule_metrics.csv`/`eval_episodes.csv` carry `scenario_set`, `scenario_seed`, `scenario_id`, none of which is a Waymo/PG source key. **Corrected during implementation**: `eval_episodes.csv` (a sibling file) already carries a `source` column populated per episode from `Agent._scenario_metadata()`/`ScenarioRecord.source`, in turn set to the literal `"pg"` (`scenarios/pg/loader.py:125`, `scenarios/pg/exporter.py:73`) or `"waymo"` (`scenarios/waymo.py:230`) at scenario-load time. `DEC-SUB-004` resolved by (a): reuse that existing per-episode `source` value, propagated into the new `subrule_metrics.csv` at the same point `scenario_metadata` is already read (`agent.py` `episode_scenario_metadata`/`record["scenario_metadata"]`). No schema change to `eval_episodes.csv` needed. |

Behaviour to preserve: every existing column of `rule_metrics.csv`, the
`EVAL-PROTOCOL` REQ-008 applicability-aware aggregation, and the R1--R3 / R4
reporting separation.

## 5. Assumptions And Invariants

- Sub-rule `cost` is in `[0,1]`, non-negative, with `0` meaning satisfied;
  enforced upstream by `aggregate_max_component`'s range validation.
- `applicable=False` means the sub-rule had no evaluable candidate this step;
  such steps must be excluded from that sub-rule's violation rate, mirroring
  `EVAL-PROTOCOL` REQ-008's treatment of macro rules.
- `worst_component` is present in a macro component's `diagnostics` whenever
  that macro is applicable; absent when `NOT_APPLICABLE`.
- Ties in `max` are broken by `(cost, name)`, so dominance counting is
  deterministic and reproducible across runs and seeds.
- The diagnostics are read-only observers: no rulebook state, no reward, no
  policy input is touched.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-SUB-001` | Specification clarification | `EVAL-PROTOCOL` v1.0 is authoritative for evaluation reporting. Does adding sub-rule diagnostic outputs require an amendment? | (a) additive diagnostics, no amendment, provided they never enter a primary metric or comparison table and are labelled diagnostic; (b) formal `EVAL-PROTOCOL` amendment | (a) — nothing existing changes value or meaning, and the protocol's primary-metric set is untouched | If (b), an approved amendment must precede implementation | **Resolved: (a).** User confirmed "procedi come hai suggerito" (2026-07-26); no `EVAL-PROTOCOL` amendment. |
| `DEC-SUB-002` | Clarification | Should dominance also be reported for `R1`? | (a) `R2`/`R3` only — `R1`'s "sub-components" are contact onsets per actor, not heterogeneous metrics, so the commensurability question does not arise; (b) all macro rules | (a) | Scope and output size | **Resolved: (a)**, adopted as part of `DEC-SUB-001`'s approval. |
| `DEC-SUB-003` | Implementation detail | Behaviour when a run predates this feature and has no `subrule_metrics.csv` | (a) skip with a recorded note; (b) fail | (a) — historical runs must stay analysable | Analysis robustness | **Resolved: (a)**. `aggregate_runs.py`'s existing per-file `if not path.exists(): continue` already gives this for free; no special-casing added. |
| `DEC-SUB-004` | Blocking technical issue | `REQ-SUB-04` needs a Waymo/PG source key per episode; no such field was found in the `rule_metrics.csv`/`subrule_metrics.csv` eval CSV schema | (a) derive it from `scenario_id`/`scenario_set` if they encode source; (b) add an explicit source column; (c) drop `REQ-SUB-04` | Determine (a)'s feasibility first by inspecting real run artifacts; fall back to (b) | Without it, distributions from two very different regimes are pooled | **Resolved: variant of (a).** The original survey only checked `rule_metrics.csv`'s own schema; the sibling `eval_episodes.csv` already carries a `source` column (`csv_recorder.py:...`, populated by `Agent._scenario_metadata()` from `ScenarioRecord.source`, itself `"pg"`/`"waymo"` literals set at load time). Reused directly, no schema change. |
| `DEC-SUB-005` | Implementation detail | `_append_rule_metrics_rows` is duplicated in `eval_loop` and `train_loop` | (a) duplicate the new emitter the same way, matching existing structure; (b) refactor both into a shared helper | (a) — a refactor of 6 call sites is unrelated cleanup and out of scope per AGENTS.md | Consistency vs. scope | Decided internally; implemented as (a). |

All gates resolved 2026-07-26; implementation complete (§11).

## 7. Proposed Design

### 7.1 Collection (`Agent.evaluate()`)

A new static extractor beside `_extract_rule_applicability`:

```python
_extract_subrule_costs(step_info) -> dict[str, SubRuleStep]
    # name -> (parent_macro, cost, applicable, is_worst_within_parent)
```

reading `step_info["rule_components"]`, using each macro component's
`diagnostics["worst_component"]` to mark the winner, and mapping sub-rule to
parent through the same grouping `aggregation.py` uses.

Per-episode accumulators mirroring the existing `ep_rule_*` ones:
applicable-step count, violated-step count, cost sum and max, worst-component
count, and a per-macro count of steps with `>= 2` violated sub-rules
(`REQ-SUB-05`). Aggregated to seed level into `metrics["per_subrule"]`,
exactly as `per_rule` is today.

### 7.2 Persistence

New `subrule_metrics.csv` (schema in `csv_recorder.py`, emitters mirroring
`_append_rule_metrics_rows` per `DEC-SUB-005`):

```
<base identity fields>, eval_id, eval_type, scenario_set, chunk_id, stage,
stage_index, global_step, macro_rule, subrule_name,
applicability_rate, violation_rate, mean_cost, max_cost,
dominance_share, worst_component_count, macro_violated_step_count,
multi_violation_share,
applicable_episode_count, excluded_episode_count
```

`aggregate_runs.py`: add the file to the allow-list with group keys
`("eval_type", "scenario_set")`, producing `subrule_metrics_all_runs.csv`.

### 7.3 Reporting

**`subrule_metrics_by_subrule.csv/md`** — per `(condition, macro_rule,
subrule)`: applicability rate, violation rate, mean/max cost, episode counts.
Answers *liveness*. Cost terminology throughout (`REQ-SUB-03`).

**`subrule_dominance_within_macro.csv/md`** — per `(condition, macro_rule,
subrule)`: `dominance_share` over macro-violated steps only, with
`macro_violated_step_count` as the visible denominator, plus
`multi_violation_share` per macro. Answers *dominance*.

**`subrule_dominance_stacked_bar.png`** — stacked bars, one per
`(condition × macro_rule)`, segments = dominance share. One segment filling a
bar means that macro is de facto a single rule; this is the figure intended
for the thesis.

**`subrule_cost_distribution_boxplot.png`** — cost distribution over
applicable-and-violated steps, one box per sub-rule, grouped by macro.
Answers *calibration*: it shows **why** any dominance occurs.

All four carry an explicit diagnostic label (`REQ-SUB-07`).

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-SUB-01` | `AC-SUB-01` all eval episodes contribute, not only tracked-subset ones | `agent.py` (`Agent.evaluate()`, `_ParallelEvaluationEpisode`, `_aggregate_parallel_evaluation`), `subrule_diagnostics.py` | `tests/test_subrule_diagnostics.py::test_all_episodes_contribute_and_are_disaggregated_by_source` | Done |
| `REQ-SUB-02` | `AC-SUB-02` a step with all-zero sub-rule costs contributes no dominance count | `subrule_diagnostics.py::SubruleEpisodeAccumulator` | `::test_dominance_counts_only_violated_macro_steps` | Done |
| `REQ-SUB-03` | `AC-SUB-03` no margin-named column in the sub-rule outputs | `csv_recorder.py` (`subrule_metrics.csv` schema), `make_subrule_tables.py` | schema inspection: all columns are `*_cost`/`*_rate`/`*_share`/`*_count`, none named `margin` | Done |
| `REQ-SUB-04` | `AC-SUB-04` rows are keyed by scenario source | `agent.py` (reuses `scenario_metadata["source"]`), `subrule_diagnostics.py::aggregate_subrule_episodes` | `::test_all_episodes_contribute_and_are_disaggregated_by_source`, `::test_missing_source_is_grouped_as_unknown_not_dropped` | Done |
| `REQ-SUB-05` | `AC-SUB-05` simultaneous-violation share matches a hand-computed fixture | `subrule_diagnostics.py::SubruleEpisodeAccumulator` | `::test_multi_violation_share_fixture` | Done |
| `REQ-SUB-06` | `AC-SUB-06` no pre-existing metric, table, or plot changes value | `agent.py`, `csv_recorder.py`, `aggregate_runs.py`, `run_analysis.py`, `make_plots.py` (additive-only edits, verified by diff review) | full suite (`1018 passed`, was `1008` pre-feature: `+10` new, `0` changed) | Done |
| `REQ-SUB-07` | `AC-SUB-07` outputs carry the diagnostic label | `make_subrule_tables.py::DIAGNOSTIC_LABEL` (written into both `.md` files), `make_plots.py` titles ("diagnostic, EP-SUBRULE-DIAG"), gated under the existing `include_diagnostics` plot flag | manual inspection of generated `.md`/plot titles (§11) | Done |

## 9. Validation

| Category | Requirement | Covered by |
|---|---|---|
| nominal and boundary behavior | `REQUIRED` | fixture with known sub-rule cost sequences |
| invalid and incomplete inputs | `REQUIRED` | missing `rule_components`, `NOT_APPLICABLE` macro, absent `worst_component` |
| aggregation correctness | `REQUIRED` | hand-computed dominance and multi-violation shares |
| compatibility and migration | `REQUIRED` | run without `subrule_metrics.csv` still analysable (`DEC-SUB-003`) |
| regression: no change to existing outputs | `REQUIRED` | `AC-SUB-06` |
| upstream/downstream integration | `REQUIRED` | `make analyze` on a fixture run directory |

Commands: `docker compose run --rm dev uv run --no-sync python -m pytest -q
tests/test_subrule_diagnostics.py`, the full suite, scoped `ruff check`, and
`make analyze RUN_PROFILE=<profile>` on a real multi-condition run.

## 10. Milestones

- `M0` — resolve `DEC-SUB-001` and `DEC-SUB-004`. **Blocking.**
- `M1` — one-off exploratory pass over existing tracked-subset trajectory logs
  to get an early signal on dominance. Not representative and not a
  deliverable; it exists to de-risk the design and may reveal that a sub-rule
  is never applicable at all. Unblocked, needs no approval.
- `M2` — collection in `Agent.evaluate()` plus `subrule_metrics.csv` and its
  tests.
- `M3` — `aggregate_runs.py` wiring and `subrule_metrics_all_runs.csv`.
- `M4` — the two tables and two plots, with the additive-only regression test.
- `M5` — full suite, lint, `make analyze` on a real run, plan reconciliation,
  `project_index.md` update.

## 11. Execution Record

`DEC-SUB-001` and `DEC-SUB-004` resolved 2026-07-26 (user: "Procedi come hai
suggerito ed implementa il piano."); `M0`-`M5` completed in one pass.

### Changes

- `src/thesis_rl/rulebook/v2/subrule_diagnostics.py` (new): `extract_subrule_step`
  (pure per-step reader, scoped to `DIAGNOSTIC_MACRO_RULES`), `SubruleEpisodeAccumulator`
  (per-episode accumulation), `aggregate_subrule_episodes` (seed-level rows,
  keyed by `(scenario_source, macro_rule, subrule_name)`).
- `src/thesis_rl/agent/agent.py`: `_ParallelEvaluationEpisode` gained a
  `SubruleEpisodeAccumulator`, observed every step, finalized into
  `subrule_summary`; the sequential `evaluate()` loop mirrors this with a
  local `subrule_acc`; both paths add `metrics["per_subrule"]` via
  `aggregate_subrule_episodes`, sourced from each episode's existing
  `scenario_metadata["source"]` (`DEC-SUB-004`).
- `src/thesis_rl/runtime/io/csv_recorder.py`: new `subrule_metrics.csv` schema.
- `src/thesis_rl/runtime/loops/eval_loop.py` / `train_loop.py`: new
  `_append_subrule_metrics_rows`, called beside every existing
  `_append_rule_metrics_rows` call (1 in `eval_loop`, 4 in `train_loop`,
  `DEC-SUB-005`).
- `src/thesis_rl/analysis/aggregate/aggregate_runs.py`: `subrule_metrics.csv`
  added to `CSV_FILENAMES`/`INFERRED_FIELDS_BY_FILE`; the existing
  per-file `if not path.exists(): continue` gives `DEC-SUB-003` for free.
- `src/thesis_rl/analysis/tables/make_subrule_tables.py` (new):
  `subrule_metrics_by_subrule.csv/md` (liveness) and
  `subrule_dominance_within_macro.csv/md` (dominance), both prefixed with
  `DIAGNOSTIC_LABEL` in the `.md` file (`REQ-SUB-07`); wired into
  `run_analysis.py::_build_tables_for_root` unconditionally, since the label
  itself (not a build flag) satisfies `DEC-SUB-001`.
- `src/thesis_rl/analysis/plots/make_plots.py`: `_plot_subrule_dominance_stacked_bar`
  and `_plot_subrule_cost_distribution_boxplot`, titled "(diagnostic,
  EP-SUBRULE-DIAG)"; gated under the pre-existing `include_diagnostics` flag
  (the same mechanism already used for the other non-core diagnostic plots),
  which satisfies `REQ-SUB-07` without a new mechanism.
- `tests/test_subrule_diagnostics.py` (new, 10 tests): the extractor, the
  accumulator (`REQ-SUB-02`, `REQ-SUB-05`), and seed-level aggregation
  (`REQ-SUB-01`, `REQ-SUB-04`), all against hand-computed fixtures.

### Commands executed and results

- `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_subrule_diagnostics.py`
  → 10 passed.
- `docker compose run --rm dev uv run --no-sync python -m pytest -q
  tests/test_subrule_diagnostics.py tests/test_train_loop_eval_protocol_helpers.py
  tests/test_rulebook_v2_lifecycle.py tests/test_eval_artifacts.py
  tests/test_parallel_evaluation.py tests/test_rulebook_v2_contracts.py
  tests/test_parallel_evaluation_data_abort.py tests/test_agent_pipeline.py
  tests/test_sac_sb3_porting.py tests/test_eval_protocol_req008_applicability_aggregation.py
  tests/test_rulebook_evaluator.py` → 105 passed (every test touching
  `Agent.evaluate()`/parallel evaluation).
- `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_analysis_optional_ci_and_r4_split.py`
  → 6 passed.
- `docker compose run --rm dev uv run --no-sync python -m pytest -q` (full
  suite) → **1018 passed** (was 1008 before this feature: +10 new, 0 changed
  — satisfies `AC-SUB-06`).
- `docker compose run --rm dev uv run --no-sync ruff check <all touched
  files>` → All checks passed.
- `docker compose run --rm dev uv run --no-sync ruff format --check <all
  touched files>` → the 3 wholly-new files formatted clean; the remaining
  6 pre-existing files each showed only pre-existing, unrelated formatting
  deviations on `ruff format --diff` inspection (confirmed line-by-line), so
  per AGENTS.md's formatting-baseline note they were **not** mass-reformatted;
  the 3 new files (`subrule_diagnostics.py`, `make_subrule_tables.py`,
  `test_subrule_diagnostics.py`) were formatted and reverified clean.
- Manual smoke test (no fixture run directory exists in this environment):
  `build_subrule_tables` and both new plot functions were exercised directly
  against a hand-built `subrule_metrics_all_runs.csv` fixture in a temp
  directory inside the dev container; both tables and both `.png` files were
  produced correctly, including the diagnostic label text.
- `git diff --check` → clean.

### Known limitation / residual risk

`make analyze RUN_PROFILE=<profile>` was **not** run end-to-end against a
real multi-seed run directory (none exists in this environment), so the full
pipeline integration (`aggregate_runs` → `subrule_metrics_all_runs.csv` →
`build_subrule_tables`/`make_plots`) is verified by construction and by the
isolated smoke test above, not by an actual run. Follow-up command, to run
once a real run is available:
`docker compose run --rm dev uv run --no-sync python -m thesis_rl.analysis.run_analysis --run-profile <profile> --seed-list 0,1,2 --include-diagnostic-plots` and inspect
`analysis/<profile>/tables/subrule_*` and `analysis/<profile>/plots/subrule_*.png`.
