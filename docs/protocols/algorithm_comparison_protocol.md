# Algorithm Comparison Protocol

This document defines how to compare algorithms in a scientific and
reproducible way. It complements `csv_evaluation_objectives.md`, which defines
the CSV format and logging requirements.

## 1) Units Of Analysis

> Amendment RSA-1 (approved 2026-07-23; ADR-024): record per-scenario UID
> evaluation results. Primary comparisons exclude the union of UIDs that are
> runtime-invalid in at least one compared algorithm/seed, report each
> method's exclusions/reason codes and coverage, and never compare implicit
> different scenario sets.

Hierarchy:

- `run` = algorithm + seed + configuration + timestamp
- `experiment` = set of runs for the same algorithm/config across multiple
  seeds
- `comparison` = comparison between different experiments

Rule:

- A single run is useful for debugging and traceability.
- Experimental results should be reported by aggregating across multiple seeds.

## 2) Comparison Objectives

The comparison is not just "higher reward wins."

Thesis priorities:

1. Safety and rulebook compliance.
2. Driving and task performance.
3. Sample efficiency and curriculum progression.
4. Robustness and stability across seeds and scenarios.

## 3) Primary Outcomes

Main metrics for final ranking:

- `collision_rate` (lower is better)
- `out_of_road_rate` (lower is better)
- `top_rule_violation_rate` or aggregated rulebook metrics (lower is better)
- `success_rate` (higher is better)
- `route_completion` (higher is better)

Secondary metrics:

- `mean_reward`, `std_reward`
- `mean_env_reward`, `std_env_reward` (native MetaDrive reward)
- `mean_scalar_rule_reward`, `std_scalar_rule_reward` (scalarized rulebook
  reward)
- `episode_length_mean/std`

Note:

- When trade-offs appear, safety and compliance take precedence over global
  reward.
- In analysis and reporting, always show in parallel:
  - final policy reward (`mean_reward`)
  - native environment reward (`mean_env_reward`)
  - scalarized rulebook reward (`mean_scalar_rule_reward`, when available)

## 4) Sample Efficiency And Curriculum Progression

Metrics to report:

- `steps_to_stage_2`, `steps_to_stage_3`, ..., `steps_to_final_stage`
- `final_stage_reached` (percentage of seeds)
- number of failed evaluations before promotion
- stage reached at `total_timesteps`

Purpose:

- Measure not only "how well it performs," but also "how quickly and stably it
  reaches harder levels."

## 5) Robustness / Generalization

Evaluate, when available:

- scenarios and seeds seen during curriculum
- unseen scenarios and seeds
- final stage and/or stress set

Metrics:

- mean and variability of primary metrics
- tail quantiles (for example 5th percentile)
- worst-case or CVaR-like metrics, when formally defined

## 6) Required Figures

### Figure 1 - Main Learning Curves

Recommended panels (vs `global_step`):

- `success_rate`
- `collision_rate`
- `top_rule_violation_rate` (or equivalent rulebook metric)
- `route_completion`

Lines = algorithms, bands = across-seed variability (std, stderr, or CI).

### Figure 2 - Curriculum Progression

- `stage_index` vs `global_step` (aggregated across seeds), or a "steps to
  stage" table.

### Figure 3 - Safety / Performance Trade-Off

Recommended scatter:

- X axis: safety metric (for example `collision_rate` or `Avg EV`)
- Y axis: task metric (for example `success_rate` or `route_completion`)

## 7) Required Tables

### Table A - Final Evaluation (Multi-Seed)

Minimum fields:

- `success_rate`
- `collision_rate`
- `out_of_road_rate`
- `top_rule_violation_rate` (or equivalent)
- `route_completion`
- `mean_reward`
- `mean_env_reward`
- `mean_scalar_rule_reward` (when `reward.mode=rulebook`)

Format:

- Official convention: `mean ± 95% CI`.

### Table B - Rulebook Compliance

When V2 metrics are available:

- `Avg EV`
- `Max EV`
- `Counterexample ratio`
- `% violated rules`
- `# unique violation patterns`

Semantic note:

- `EV` is derived from rule margins (violation if `margin < 0`,
  non-violation if `margin >= 0`).

### Table C - Curriculum / Sample Efficiency

- `final_stage_reached`
- `steps_to_final_stage`
- `failed_evals_before_promotion`

### Table D - Ablation (If Present)

Comparison of variants (for example TD3, scalar-rulebook, lexicographic,
distributional, curriculum on/off).

## 8) Statistical Aggregation Rules

- Compare algorithms under identical budget (`total_timesteps`) and evaluation
  protocol.
- Always aggregate by seed, not by isolated single runs.
- Define in advance:
  - number of seeds
  - variability metric (adopted: `95% CI`)
  - optional curve smoothing (default: none)
- Do not change metrics or aggregation after the fact between algorithms.

### Seed Protocol (Fixed)

- Number of runs per algorithm in the main comparison: `10`.
- Official seed list, shared by all algorithms:
  - `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Balancing rule:
  - each algorithm must have the same number of valid runs on the same seed
    list
- Failure handling:
  - if a run fails, relaunch the same configuration with the same seed
  - do not replace the seed with a different value

Operational rules:

- Report main curves without smoothing.
- Any smoothed version is secondary and must declare the fixed window
  explicitly.

## 9) Run Validity And Selection

A run is valid for comparison if:

- it contains `artifacts/run_metadata.yaml` with `status: completed`
- it contains `csv/final_eval.csv`
- it uses the same `total_timesteps` budget planned for the comparison
- it uses a consistent evaluation protocol (`eval_episodes` and
  `final_eval_episodes`) with the compared group
- it is not an explicitly excluded debug or dev run

Recommended fields for automatic filtering (in config or metadata):

```yaml
analysis:
  include_in_comparison: true
  experiment_group: td3_curriculum_v1
```

## 10) Question-To-CSV Mapping

- How it evolves during training: `evals.csv`
- When and how it gets promoted: `promotions.csv` (+ gate columns in
  `evals.csv`)
- Evaluation episode distribution: `eval_episodes.csv`
- Final run state: `final_eval.csv` (V2; in V1.5 derivable from final eval)
- Per-rule analysis: `rule_metrics.csv` (V2)

Current V2 availability:

- `evals.csv`: also includes `avg_error_value`, `max_error_value`,
  `counterexample_rate`, `violated_rules_ratio`, `unique_violation_patterns`
- `evals.csv`: also includes the aggregated reward triplet
  (`mean_reward`, `mean_env_reward`, `mean_scalar_rule_reward` + std)
- `eval_episodes.csv`: includes `scenario_seed`, `scenario_id`, `error_value`,
  `violated_rules`, `violation_pattern`, `video_path` (optional, may be
  `null`)
- `eval_episodes.csv`: also includes `env_reward`, `scalar_rule_reward`,
  `rule_rewards_by_rule` for per-episode and per-rule analysis

## 11) Final Evaluation Protocol

- Intermediate evaluations use `experiment.eval_episodes`.
- Final evaluation uses `experiment.final_eval_episodes` (recommended default:
  `100`), separate from intermediate evaluations.
- `final_eval.csv` contains one row per run/seed with the official final
  metrics.

## 12) Required Analysis Outputs

```text
<outputs-root>/analysis/aggregated/
  evals_all_runs.csv
  final_eval_all_runs.csv
  promotions_all_runs.csv
  eval_episodes_all_runs.csv
  rule_metrics_all_runs.csv

<outputs-root>/analysis/tables/
  final_evaluation.(csv|tex)
  rulebook_compliance.(csv|tex)
  curriculum_efficiency.(csv|tex)

<outputs-root>/analysis/plots/
  learning_curves_success_collision_rule_route.(png|pdf)
  curriculum_progression.(png|pdf)
  safety_performance_tradeoff.(png|pdf)
```

## 13) Recommended Reporting Statement

Expected final message from the comparison:

> The lexicographic algorithm, possibly with a distributional variant, reduces
> the frequency and severity of violations of the highest-priority rules,
> preserves competitive driving performance, and progresses through the
> curriculum more stably and efficiently than scalarized baselines.
