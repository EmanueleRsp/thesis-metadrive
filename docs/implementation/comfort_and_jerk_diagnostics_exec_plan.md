# ExecPlan: Comfort And Jerk Evaluation Diagnostics

## 1. Metadata

- Feature: `comfort_and_jerk_diagnostics`
- Plan ID: `EP-COMFORT-DIAG`
- Authoritative specification: none amended. This is additive diagnostic
  reporting explicitly licensed by
  `docs/specifications/rulebook_v5.1_specification.md` (`RULEBOOK-V5.1`,
  version `5.1`, `AUTHORITATIVE`) §13, which excludes comfort and jerk from the
  rulebook and the reward while permitting them to be *logged as diagnostics
  only*. Reporting sits under `EVAL-PROTOCOL`
  (`docs/specifications/evaluation_protocol_v1.0_specification.md`, amended by
  v1.2 and v1.3), whose primary/secondary metric contract this plan does not
  change.
- Related specifications (read, not changed):
  - `docs/specifications/rulebook_v5.0_UNDER_REVIEW_specification.md` §13, which
    records comfort as a deliberately excluded future extension anchored on
    nuPlan's expert-derived acceleration and jerk percentiles
  - `docs/specifications/evaluation_protocol_v1.3_specification.md` §2.1
    (primary metric `route_completion`; unchanged here)
- Related ADRs: none. `DEC-CMF-001` and `DEC-CMF-002` were approved by the user
  directly on 2026-09-01; `DEC-CMF-003` is a recorded limitation, not a
  specification deviation, because no specification defines comfort values.
- Status: `IMPLEMENTED`
- Created: 2026-09-01
- Last updated: 2026-09-01
- Branch: `scenarionet-implementation`

## 2. Objective And Scope

### Problem being solved

Nothing in the repository measures ride comfort. Verified on 2026-09-01: the
strings `jerk` and `comfort` appear in `docs/` and in one test comment, and in
**no** production module under `src/thesis_rl/`. Concretely, none of the six
CSV schemas in `src/thesis_rl/runtime/io/csv_recorder.py` carries an
acceleration, jerk, or comfort column; `Agent.evaluate` and
`Agent._aggregate_parallel_evaluation` accumulate no such quantity; and no
analysis table or plot reports one.

That is a real reporting gap rather than a modelling one. `RULEBOOK-V5.1` §13
deliberately keeps comfort out of the reward, so a trained policy is under no
pressure to produce smooth control and may well satisfy every rulebook level
with a bang-bang throttle. Without a diagnostic channel the thesis cannot say
whether that happened, and cannot compare arms on a dimension every published
planning benchmark reports.

### Observable capability

Every evaluation episode gains the seven kinematic comfort statistics nuPlan's
`ego_is_comfortable` is built from, plus the boolean those bounds imply; every
evaluation gains their seed-level aggregate; and the analysis pipeline gains a
diagnostic table per condition, kept separate from the primary comparison
tables.

### Success

A completed evaluation writes populated comfort columns to
`eval_episodes.csv`, `evals.csv`, and `final_eval.csv`; `make analyze` emits
`comfort_diagnostics.{csv,md}` per analysis root and per comparison view; the
serial and parallel evaluation paths produce identical values from identical
episodes; and no reward, rulebook, checkpoint, or primary metric changes.

### In scope

- Per-step extraction of ego longitudinal/lateral acceleration and yaw from the
  already-populated `info["ego_state"]`, at evaluation time only.
- Per-episode reduction to the nuPlan statistic set and the comfort boolean.
- Seed-level aggregation over evaluation episodes.
- CSV schema extension for the three evaluation-scoped files.
- A separate diagnostic analysis table, wired into the analysis pipeline.

### Out of scope

- Any reward, rulebook, scalarization, or termination change (`RULEBOOK-V5.1`
  §13 forbids the first two; the rest are untouched).
- Training-time logging (`DEC-CMF-002`): `train_chunks.csv` is unchanged.
- Action smoothing or any downstream filter. §13 says smoothing, if ever
  wanted, belongs downstream of the policy and frozen across arms; this plan
  only measures.
- Backfilling comfort columns onto runs recorded before this change.

### Compatibility

Additive only. `CSVRecorder.append_row` projects each row onto its schema, so
new keys are inert for files that do not declare them, and runs recorded before
this change simply lack the columns; `aggregate_runs` unions source headers
(`_extend_unique`, `aggregate_runs.py:493-496`), so mixed run sets aggregate
without error and the analysis table leaves missing values blank rather than
failing. No checkpoint, observation, or action space is touched.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-CMF-01` | Comfort and jerk are logged as diagnostics only, never entering the rulebook, the reward, or any termination decision | `RULEBOOK-V5.1` §13 |
| `REQ-CMF-02` | Per-episode statistics are the seven nuPlan comfort channels, with the published bound values, plus the conjunction boolean | `RULEBOOK-V5.0` §13 (anchor of record); `RULEBOOK-V5.1` §13 (scope) |
| `REQ-CMF-03` | Per-episode values are written to `eval_episodes.csv` | `EVAL-PROTOCOL` v1.0 §7 (per-episode artifact) |
| `REQ-CMF-04` | Seed-level aggregates are written to `evals.csv` and `final_eval.csv` | `EVAL-PROTOCOL` v1.0 §7 |
| `REQ-CMF-05` | The serial and the parallel evaluation paths produce identical values for identical episodes | `EVAL-PROTOCOL` v1.2 (multi-panel execution parity) |
| `REQ-CMF-06` | Analysis reports the diagnostic per condition, separately from the primary comparison tables | `EVAL-PROTOCOL` v1.3 §2.1 (primary metric unchanged) |
| `REQ-CMF-07` | Episodes whose kinematics are undefined are excluded from the aggregate and their count reported, never silently counted as zero or as comfortable | `EVAL-PROTOCOL` v1.0 REQ-008 applicability convention, followed by analogy |

## 4. Current Repository Analysis

| Statement | Label | Evidence |
|---|---|---|
| No comfort/jerk metric exists in production code | `VERIFIED` | `grep -rniE "jerk\|comfort" src/` returns nothing (2026-09-01) |
| `info["ego_state"]["acceleration"]` already carries `x`, `y`, `longitudinal`, `lateral` | `VERIFIED` | `src/thesis_rl/envs/wrappers.py:619-646` `_extract_physical_acceleration`, projecting `(v_t - v_{t-1})/dt` onto the heading and its normal |
| `info["ego_state"]["yaw"]` carries `heading_theta` in radians | `VERIFIED` | `src/thesis_rl/envs/wrappers.py:186-189` |
| The differentiation timestep is not exported to `info` | `VERIFIED` | `_extract_dt` (`wrappers.py:648`) is consumed internally and discarded |
| `dt = physics_world_step_size * decision_repeat = 0.02 * 5 = 0.1 s` for every shipped env config | `VERIFIED` | `conf/env/{metadrive,scenarionet,metadrive_native_strict,metadrive_native_relaxed}.yaml` |
| Evaluation has two paths that must stay in agreement | `VERIFIED` | serial `Agent.evaluate` (`agent.py:1667`) and parallel `_EpisodeTracker` + `Agent._aggregate_parallel_evaluation` (`agent.py:105`, `agent.py:2423`) |
| `eval_episodes.csv` is written from five sites, all indexing the same `per_episode` vectors | `VERIFIED` | `eval_loop.py:551`, `train_loop.py:1019/1930/2798`, `final_panels.py:105` |
| `evals.csv` / `final_eval.csv` are written from thirteen sites that enumerate metric keys explicitly | `VERIFIED` | `grep -rn '"evals.csv"\|"final_eval.csv"' src/thesis_rl` |
| `CSVRecorder.append_row` drops keys absent from the target schema | `VERIFIED` | `csv_recorder.py:370-378` |
| `aggregate_runs` unions source headers, so new columns propagate with no change | `VERIFIED` | `aggregate_runs.py:493-496` |
| Comparison views copy every column of the aggregated CSVs through to the view | `VERIFIED` | `make_comparison_views.py:_write_view_aggregated`, which writes `source_fieldnames` unchanged |
| `make_final_tables` tolerates a metric absent from the source, emitting blank cells | `VERIFIED` | `make_final_tables.py:135-152` |
| `EP-SUBRULE-DIAG` is the approved precedent for an additive evaluation diagnostic: accumulator module, per-episode summary, seed-level aggregate, separate diagnostic table | `VERIFIED` | `docs/implementation/subrule_dominance_diagnostics_exec_plan.md`; `rulebook/v2/subrule_diagnostics.py`; `analysis/tables/make_subrule_tables.py` |
| `scipy` is neither a declared nor an installed dependency | `VERIFIED` | `pyproject.toml` `dependencies`; `python -c "import scipy"` fails |

Behaviour to preserve: the reward path, the rulebook margins, the
applicability-aware REQ-008 aggregation, the per-episode CSV column order of
existing columns, and the primary/secondary metric contract of `EVAL-PROTOCOL`
v1.3.

## 5. Assumptions And Invariants

- **Units.** Accelerations in `m/s^2`, jerks in `m/s^3`, yaw in `rad`, yaw rate
  in `rad/s`, yaw acceleration in `rad/s^2`, `dt` in `s`. Established from
  `_extract_physical_acceleration` (SI velocities differentiated by an SI
  timestep) and from `heading_theta` being radians in MetaDrive.
- **Frame.** `longitudinal` is the projection on the ego heading, `lateral` on
  its left normal, both from `wrappers.py:637-641`. The comfort statistics
  inherit that frame; no re-projection is performed.
- **Timing.** Derivatives are backward finite differences over consecutive
  *policy* steps, on the same `dt` the wrapper used to produce the
  acceleration. A step whose kinematics are unusable breaks the chain: the
  accumulator resets its previous-step state so no derivative is ever taken
  across a gap.
- **Angle wrapping.** Yaw differences are wrapped to `(-pi, pi]` before being
  divided by `dt`, so a heading crossing `+-pi` cannot fabricate a spurious
  yaw rate of order `2*pi/dt`.
- **Definedness.** `max_abs_lon_accel`-class statistics need one valid step;
  the jerk and yaw-rate statistics need two consecutive; yaw acceleration needs
  three. `is_comfortable` is defined only when all seven are, and is `None`
  otherwise. `None` propagates as an empty CSV cell and as an exclusion from
  the aggregate, with the exclusion counted (`REQ-CMF-07`).
- **Determinism.** The accumulator is a pure function of the ordered step-info
  sequence; it holds no RNG and no global state, so it cannot perturb seeding
  or replay.
- **Reset.** One accumulator instance per episode, constructed at episode start
  in both evaluation paths, mirroring `SubruleEpisodeAccumulator`.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-CMF-001` | Specification clarification | Which metric set? | A: full nuPlan `ego_is_comfortable` suite (seven statistics + boolean) / B: minimal continuous set (RMS jerk, mean lateral acceleration) / C: both | A | Benchmark comparability; column count | **Approved by user 2026-09-01** |
| `DEC-CMF-002` | Specification clarification | Which run scope? | A: evaluation only / B: evaluation plus per-chunk training aggregates | A | Surface modified; no env or training-loop change | **Approved by user 2026-09-01** |
| `DEC-CMF-003` | Implementation detail | nuPlan derives its comfort channels from Savitzky-Golay-filtered trajectories; this implementation uses raw backward differences | A: raw differences, deviation documented / B: reimplement the filter (needs `scipy`, an unapproved dependency) / C: hand-rolled local polynomial derivative | A | Max-statistics are noisier and upper-biased relative to nuPlan, so `is_comfortable` is **conservative**: it fails more readily than nuPlan's would on the same trajectory | Decided; recorded as a known limitation |
| `DEC-CMF-004` | Implementation detail | Where do the metrics surface in analysis? | A: separate diagnostic table / B: added to `FINAL_METRICS` in the primary `final_evaluation` table | A | Keeps the `EVAL-PROTOCOL` v1.3 primary/secondary contract intact; follows the `EP-SUBRULE-DIAG` precedent | Decided |
| `DEC-CMF-005` | Implementation detail | `dt` is needed at evaluation time but not exported | A: export `ego_state["dt"]` beside the acceleration it belongs to / B: re-read the engine config in the accumulator / C: hard-code `0.1` | A | One additive info key, guaranteed present exactly when `acceleration` is; no invented fallback | Decided |

`DEC-CMF-003` is not a specification deviation: no approved specification
defines a comfort value, so there is no contract to deviate from. It is
recorded because it makes the reported numbers non-identical to nuPlan's on the
same trajectory, which a reader comparing against the benchmark must know.

## 7. Proposed Design

### Data flow

```
RuleRewardWrapper._enrich_runtime_info        (envs/wrappers.py)
  -> info["ego_state"] = {..., acceleration:{x,y,longitudinal,lateral}, yaw, dt}
       |
       v
ComfortEpisodeAccumulator.observe(step_info)  (runtime/comfort_diagnostics.py)
  serial:   Agent.evaluate                    (agent.py)
  parallel: _EpisodeTracker.observe_step      (agent.py)
       |
       v  .finalize() -> per-episode summary dict
aggregate_comfort_episodes(summaries)         -> flat metrics keys
       |
       +--> metrics["per_episode"]["comfort"] -> eval_episodes.csv  (5 sites)
       +--> metrics[comfort aggregate keys]   -> evals.csv, final_eval.csv
       |
       v
aggregate_runs (header union, unchanged) -> build_comfort_tables
       -> tables/comfort_diagnostics.{csv,md}
```

### New module `src/thesis_rl/runtime/comfort_diagnostics.py`

Placed under `runtime/` rather than `rulebook/` on purpose: `RULEBOOK-V5.1` §13
puts comfort outside the rulebook, and filing it under `rulebook/` would
suggest a channel that does not exist.

- `ComfortBounds` frozen dataclass and `NUPLAN_COMFORT_BOUNDS`, the published
  `ego_is_comfortable` defaults: `max_lon_accel 2.40`, `min_lon_accel -4.05`,
  `max_abs_lat_accel 4.89`, `max_abs_mag_jerk 8.37`, `max_abs_lon_jerk 4.13`,
  `max_abs_yaw_rate 0.95`, `max_abs_yaw_accel 1.93`.
- `extract_comfort_step(step_info)` -> `ComfortStep | None`, tolerant of a
  missing or malformed `step_info` exactly as `extract_subrule_step` is.
- `ComfortEpisodeAccumulator.observe(step_info)` / `.finalize()`.
- `aggregate_comfort_episodes(summaries)` -> the flat aggregate keys.
- `COMFORT_EPISODE_COLUMNS`, `COMFORT_AGGREGATE_COLUMNS`, and the two row
  helpers `comfort_episode_fields(per_episode, index)` and
  `comfort_aggregate_fields(metrics)`, so the thirteen writer sites each gain
  one `**helper(...)` term instead of a copied block of keys.

### Column names

Per episode (`eval_episodes.csv`): `comfort_valid_step_count`,
`comfort_is_comfortable`, `comfort_max_lon_accel`, `comfort_min_lon_accel`,
`comfort_max_abs_lat_accel`, `comfort_max_abs_mag_jerk`,
`comfort_max_abs_lon_jerk`, `comfort_max_abs_yaw_rate`,
`comfort_max_abs_yaw_accel`.

Aggregate (`evals.csv`, `final_eval.csv`): `comfort_rate`,
`comfort_episode_count`, `comfort_excluded_episode_count`, and the seven
`mean_comfort_*` means over the episodes where each statistic is defined.

The per-bound violation rates are deliberately not stored: each is a threshold
comparison on a per-episode column that is already present, so nothing is lost
and the aggregate schema stays small.

### Analysis

`src/thesis_rl/analysis/tables/make_comfort_tables.py` mirrors
`make_subrule_tables.py`: it reads `final_eval_all_runs.csv`, groups by
`condition_id`, and writes `comfort_diagnostics.csv` and `.md` carrying an
explicit diagnostic label. It is wired into `_build_tables_for_root` beside
`build_subrule_tables`, so it runs for the analysis root and for every
comparison view without further wiring.

### Errors and fallbacks

No fallback is introduced. A step without usable kinematics is not counted and
breaks the derivative chain; an episode without a defined statistic reports an
empty cell and is excluded from that statistic's mean, with the exclusion
counted. A run set with no comfort columns yields a header-only diagnostic
table, not an error.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-CMF-01` | `AC-CMF-01` | `runtime/comfort_diagnostics.py` (reads `step_info`, returns values; no writes) | `tests/test_comfort_diagnostics.py::test_module_never_touches_reward_or_rulebook_paths` | Implemented |
| `REQ-CMF-02` | `AC-CMF-02` | `NUPLAN_COMFORT_BOUNDS`, `ComfortEpisodeAccumulator.finalize` | `::test_constant_jerk_trajectory_exact_statistics`, `::test_each_bound_flips_is_comfortable` | Implemented |
| `REQ-CMF-03` | `AC-CMF-03` | `csv_recorder.py` `eval_episodes.csv` schema; `comfort_episode_fields` at five writer sites | `::test_eval_episodes_schema_declares_comfort_columns`, `::test_comfort_episode_fields_indexes_per_episode_vector` | Implemented |
| `REQ-CMF-04` | `AC-CMF-04` | `csv_recorder.py` `evals.csv`/`final_eval.csv` schemas; `comfort_aggregate_fields` | `::test_eval_and_final_schemas_declare_comfort_columns` | Implemented |
| `REQ-CMF-05` | `AC-CMF-05` | `Agent.evaluate` and `_EpisodeTracker`/`_aggregate_parallel_evaluation` both call the same accumulator and aggregator | `::test_serial_and_parallel_paths_agree_on_identical_episodes` | Implemented |
| `REQ-CMF-06` | `AC-CMF-06` | `analysis/tables/make_comfort_tables.py`; `run_analysis._build_tables_for_root` | `::test_build_comfort_tables_groups_by_condition`, `::test_build_comfort_tables_tolerates_legacy_runs` | Implemented |
| `REQ-CMF-07` | `AC-CMF-07` | `aggregate_comfort_episodes` exclusion counting | `::test_undefined_episodes_excluded_and_counted` | Implemented |

Acceptance criteria:

- `AC-CMF-01`: importing and running the accumulator over a step sequence
  changes no reward, margin, or rulebook value; the module imports nothing from
  `thesis_rl.reward` or `thesis_rl.rulebook`.
- `AC-CMF-02`: on an analytically known trajectory the seven statistics equal
  the closed-form values to `1e-9`, and each bound in isolation flips
  `is_comfortable` from `True` to `False`.
- `AC-CMF-03`: `eval_episodes.csv` declares the nine per-episode columns and a
  row carries the values of its own episode index.
- `AC-CMF-04`: `evals.csv` and `final_eval.csv` declare the ten aggregate
  columns.
- `AC-CMF-05`: the same episode sequence reduced through the serial and the
  parallel aggregation yields bit-identical comfort values.
- `AC-CMF-06`: `build_comfort_tables` emits per-condition mean/SD/seed-value
  rows with the diagnostic label, and emits a header-only table for a run set
  without comfort columns.
- `AC-CMF-07`: an episode with fewer than three consecutive valid steps has an
  empty `comfort_is_comfortable`, is absent from `comfort_episode_count`, and
  is counted in `comfort_excluded_episode_count`.

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-CMF-01` | Unit | Closed-form kinematics | Constant-jerk series, `dt = 0.1` | Seven statistics match analytic values to `1e-9` | `REQ-CMF-02` |
| `TEST-CMF-02` | Unit | Bound sensitivity | Seven series each breaching one bound | `is_comfortable` `False` in each, `True` on the compliant series | `REQ-CMF-02` |
| `TEST-CMF-03` | Unit, boundary | Yaw wrap | Heading crossing `+pi` | Yaw rate is the small wrapped value, not `~2*pi/dt` | `REQ-CMF-02` |
| `TEST-CMF-04` | Unit, missing data | Gap handling | Series with a step lacking `ego_state` | No derivative taken across the gap; valid-step count excludes it | `REQ-CMF-07` |
| `TEST-CMF-05` | Unit, invalid | Malformed input | `None`, `{}`, non-numeric acceleration, absent `dt` | Treated as absent, no exception | `REQ-CMF-07` |
| `TEST-CMF-06` | Unit, boundary | Short episode | One- and two-step episodes | `is_comfortable is None`; defined statistics still reported | `REQ-CMF-07` |
| `TEST-CMF-07` | Unit | Aggregate exclusion | Mixed defined/undefined episodes | Means over defined only; `comfort_episode_count` and `comfort_excluded_episode_count` exact | `REQ-CMF-07` |
| `TEST-CMF-08` | Contract | CSV schemas | `CSVRecorder.SCHEMAS` | The nine per-episode and ten aggregate columns are declared | `REQ-CMF-03`, `REQ-CMF-04` |
| `TEST-CMF-09` | Contract | Row helpers | `per_episode` vector, index | Correct episode's values; out-of-range index yields empties | `REQ-CMF-03` |
| `TEST-CMF-10` | Integration | Path parity | Same episode summaries through both reducers | Identical comfort aggregates | `REQ-CMF-05` |
| `TEST-CMF-11` | Integration | Diagnostic table | Synthetic `final_eval_all_runs.csv`, two conditions, three seeds | Per-condition mean/SD/seed values and the diagnostic label | `REQ-CMF-06` |
| `TEST-CMF-12` | Compatibility | Legacy runs | Aggregated CSV without comfort columns | Header-only table, no exception | `REQ-CMF-06` |
| `TEST-CMF-13` | Regression | Isolation | Accumulator over a full series | No import of reward/rulebook modules; no mutation of the input `step_info` | `REQ-CMF-01` |

Commands (all already supported by the repository):

- focused: `uv run --no-sync python -m pytest -q tests/test_comfort_diagnostics.py`
- regression: `uv run --no-sync python -m pytest -q`
- lint: `make lint`
- formatting, focused scope: `make format-check PYTHON_QUALITY_PATHS="..."`
- smoke: `make smoke`
- whitespace: `git diff --check`

Type checking: no global mypy target is configured; not claimed.

## 10. Milestones

- [x] **M1 — Kinematic source.** Export `ego_state["dt"]` beside the
  acceleration it belongs to (`DEC-CMF-005`). Files:
  `src/thesis_rl/envs/wrappers.py`.
- [x] **M2 — Diagnostic module.** `runtime/comfort_diagnostics.py` with the
  bounds, extractor, accumulator, aggregator, and the two row helpers. Tests:
  `TEST-CMF-01`..`07`, `13`.
- [x] **M3 — Evaluation wiring.** Both evaluation paths in `agent/agent.py`.
  Tests: `TEST-CMF-10`.
- [x] **M4 — CSV schemas and writers.** `csv_recorder.py` plus the thirteen
  writer sites. Tests: `TEST-CMF-08`, `TEST-CMF-09`.
- [x] **M5 — Analysis.** `make_comfort_tables.py` wired into `run_analysis`.
  Tests: `TEST-CMF-11`, `TEST-CMF-12`.
- [x] **M6 — Validation and reconciliation.** Focused tests, full suite, lint,
  focused format check, smoke, index update.

## 11. Progress And Findings Log

**2026-09-01 — investigation.** Established the gap and its licence. `grep`
over `src/` returns no comfort or jerk symbol; `RULEBOOK-V5.1` §13 permits the
diagnostic explicitly. Found that the hard part was already done: the wrapper
computes heading-projected acceleration on the correct `dt`, so only
differentiation and reduction remained. Confirmed `dt = 0.1 s` for every
shipped env config, which happens to be the rate nuPlan reports at, making the
finite differences directly comparable in rate if not in filtering.

**2026-09-01 — decisions.** User approved the full nuPlan suite
(`DEC-CMF-001`) and evaluation-only scope (`DEC-CMF-002`). Found `scipy`
absent from `pyproject.toml`, which rules out reproducing nuPlan's
Savitzky-Golay derivative without a dependency approval; recorded
`DEC-CMF-003` as a conservative-bias limitation instead of adding a dependency
or hand-rolling an unapproved filter.

**2026-09-01 — implementation.** Milestones M1-M6 implemented. The thirteen
aggregate writer sites and five episode writer sites were each reduced to one
helper call rather than a copied key block, which is why the diff is wide but
shallow. Measured results are in §14.

**2026-09-01 — `BUG-CMF-001`, found by the mandatory matrix.** `TEST-CMF-04`
failed on first run. `ComfortBounds.satisfied_by` iterated the channels once,
returning `False` at the first breached bound; a channel that was *undefined*
later in the order was therefore never reached. An episode with two valid steps
either side of a gap — no jerk measurable at all, but a large recorded
acceleration — was reported as **uncomfortable** rather than as unmeasured,
which is exactly the silent-zero failure `REQ-CMF-07` exists to prevent, and it
would have biased `comfort_rate` downward on any panel with dropped frames.
Severity: would have corrupted the headline metric. Fixed by settling
definedness across every channel before testing any bound. Regression test:
`::test_undefined_channel_wins_over_a_breached_one`, plus the two gap tests that
caught it. No approval needed: the fix restores the behaviour the plan
specified.

**2026-09-01 — one test expectation corrected, not weakened.**
`::test_yaw_rate_wraps_across_pi` originally asserted `is_comfortable is True`
on a heading that crossed `+-pi` and then stopped rotating. The wrap assertion
was right and passed; the comfort assertion was wrong, because abruptly ending
a rotation genuinely breaches the yaw-acceleration bound. The fixture now
carries a *steady* rotation across the branch cut, which is the case the test
was meant to exercise, and still asserts the wrapped rate rather than the
`2*pi/dt` artefact.

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-CMF-001` | nuPlan derives the comfort channels from Savitzky-Golay-filtered trajectories | Raw backward finite differences at `dt = 0.1 s` | `scipy` is not a declared dependency and AGENTS.md forbids adding one without approval; a hand-rolled filter would be an unapproved numerical convention | Not a specification deviation (no specification defines comfort values); recorded as `DEC-CMF-003` and as a known limitation | Module docstring; `make_comfort_tables` diagnostic label; §15 |

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/runtime/comfort_diagnostics.py` | Added | Bounds, extractor, accumulator, aggregator, CSV row helpers |
| `src/thesis_rl/envs/wrappers.py` | Modified | Export `ego_state["dt"]` (`DEC-CMF-005`) |
| `src/thesis_rl/agent/agent.py` | Modified | Accumulate and aggregate in both evaluation paths |
| `src/thesis_rl/runtime/io/csv_recorder.py` | Modified | Declare the per-episode and aggregate columns |
| `src/thesis_rl/runtime/loops/eval_loop.py` | Modified | Emit comfort fields |
| `src/thesis_rl/runtime/loops/train_loop.py` | Modified | Emit comfort fields at the evaluation writer sites |
| `src/thesis_rl/runtime/final_panels.py` | Modified | Emit comfort fields |
| `src/thesis_rl/curriculum/scenario_acl/driver.py` | Modified | Emit comfort fields at its evaluation writer sites |
| `src/thesis_rl/analysis/tables/make_comfort_tables.py` | Added | Diagnostic table per condition |
| `src/thesis_rl/analysis/run_analysis.py` | Modified | Wire the diagnostic table into the pipeline |
| `tests/test_comfort_diagnostics.py` | Added | `TEST-CMF-01`..`13` |
| `docs/implementation/comfort_and_jerk_diagnostics_exec_plan.md` | Added | This plan |
| `docs/project_index.md` | Modified | Register the diagnostic |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_comfort_diagnostics.py` | `PASS` | 2026-09-01 | 49 passed. An earlier run of this same file failed 3 tests, which is how `BUG-CMF-001` below was found |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q` | `PASS` | 2026-09-01 | 1577 passed, 1 warning, 0 failed, in 255 s. Baseline before this change was 1528 |
| `make lint` | `PASS` | 2026-09-01 | `ruff check src tests scripts`: all checks passed |
| `make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/runtime/comfort_diagnostics.py src/thesis_rl/analysis/tables/make_comfort_tables.py tests/test_comfort_diagnostics.py"` | `PASS` | 2026-09-01 | Focused scope per the repository formatting policy; two files were reformatted with `make format` first, then re-checked clean |
| `git diff --check` | `PASS` | 2026-09-01 | No whitespace errors |
| `make smoke` | `SMOKE_RESULT` | 2026-09-01 | End-to-end training smoke |

Not run: `make analyze` end-to-end against a real multi-seed run, because no
such run set exists in this environment. `build_comfort_tables` is covered
instead by four focused tests against hand-built aggregated CSVs, including the
legacy-run and missing-file cases. Follow-up command once a run set exists:
`make analyze RUN_PROFILE=<profile>`, then inspect
`<analysis-root>/<profile>/tables/comfort_diagnostics.md`.

## 15. Final Reconciliation

| Requirement | Status |
|---|---|
| `REQ-CMF-01` | `VERIFIED` |
| `REQ-CMF-02` | `VERIFIED` |
| `REQ-CMF-03` | `VERIFIED` |
| `REQ-CMF-04` | `VERIFIED` |
| `REQ-CMF-05` | `VERIFIED` |
| `REQ-CMF-06` | `VERIFIED` |
| `REQ-CMF-07` | `VERIFIED` |

### Known limitations

1. **Unfiltered derivatives (`DEV-CMF-001`).** The statistics are raw backward
   differences, where nuPlan smooths first. Simulator contact impulses and
   controller chatter therefore inflate the max-statistics, so `comfort_rate`
   is a **lower bound** on the rate nuPlan's instrument would report on the same
   trajectories. Comparisons *between arms* in this repository are unaffected,
   since every arm is measured with the identical unfiltered instrument;
   comparisons *against published nuPlan numbers* are not like-for-like.
2. **No expert calibration.** Unlike every rulebook sub-rule, these thresholds
   were not falsified against the logged Waymo expert. The published nuPlan
   values are taken as-is. Running the expert panel through the accumulator
   would establish what the expert actually scores and is the natural follow-up.
3. **Evaluation only (`DEC-CMF-002`).** Comfort during training is not
   observable; only evaluation checkpoints are measured.
4. **No backfill.** Runs recorded before this change have empty comfort columns
   and are excluded from the diagnostic table rather than being re-derived from
   stored trajectories.

### Deferred optional work

- Calibrating the bounds against the Waymo expert panel (limitation 2), which
  would turn `comfort_rate` from an absolute claim into a human-relative one.
- A comfort learning curve, which requires lifting `DEC-CMF-002`.

### Resulting behaviour

Evaluation now measures ride comfort on nuPlan's seven kinematic channels and
reports it per episode, per evaluation, and per condition, entirely outside the
reward and the rulebook as `RULEBOOK-V5.1` §13 requires. No reward, rulebook,
scalarization, checkpoint, observation, or primary metric changed, and the
addition is inert for runs recorded before it.
