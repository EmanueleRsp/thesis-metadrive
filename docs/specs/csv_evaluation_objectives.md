# CSV Evaluation Objectives

This document collects the CSV evaluation and logging objectives for runs, with
an incremental plan designed to avoid incompatibilities across experiments.

## Objective

Standardize the output in `outputs/.../csv/` to support:

- comparisons across algorithms, reward modes, and curriculum settings
- comparisons across seeds
- curriculum progression analysis
- distributional analysis of evaluation episodes
- extensibility toward advanced rulebook metrics

## Principles

- Always include identifying columns: `algorithm`, `reward_mode`,
  `curriculum_name`, `seed`, `run_id`, `stage`, `stage_index`, `global_step`
- Stable and ordered schema (fixed field names)
- No "invented" metrics: each column must have a clear definition and data
  source in the code
- Backward compatibility: new columns should be additive only

## CSV Granularity

The main CSV files are not step-by-step logs.

Plots such as `metric vs timesteps` should be interpreted as:

- `x = global_step` (accumulated environment interactions)
- `y = aggregated metric` of the policy evaluated at that `global_step`

In our loop (`train chunk -> evaluation -> gate check -> optional promotion`),
each CSV has a specific granularity:

- `train_chunks.csv`: 1 row = 1 training chunk
- `evals.csv`: 1 row = 1 aggregated evaluation (over N episodes)
- `eval_episodes.csv`: 1 row = 1 evaluation episode
- `promotions.csv`: 1 row = 1 relevant curriculum event (for example
  `promoted`)
- `rule_metrics.csv` (V2): 1 row = 1 rule evaluated in 1 evaluation
- `final_eval.csv` (V2): 1 row = 1 final evaluation of the run

Do not use the main CSV files to store `1 row = 1 env step`.
Step-by-step data remains diagnostic and debug material and should be stored
separately (JSONL logs, trajectory dumps, and so on).

## Phase V1.5 (Priority Implementation)

### Expected CSV Files

- `train_chunks.csv`
- `evals.csv`
- `promotions.csv`
- `eval_episodes.csv`

### 1) `train_chunks.csv` (One Row Per Chunk)

Purpose: chunk-level training monitoring (`chunk -> eval`).

V1.5 columns:

```csv
algorithm,reward_mode,curriculum_name,seed,run_id,chunk_id,stage,stage_index,steps_start,steps_end,global_step,chunk_steps,episodes,ep_rew_mean,ep_rew_std,ep_rew_ci_95,ep_env_rew_mean,ep_scalar_rule_rew_mean,ep_hybrid_rew_mean,ep_len_mean,ep_len_std,ep_len_ci_95,ep_success_rate,ep_collision_rate,ep_out_of_road_rate,ep_route_completion_mean,actor_loss,critic_loss,actor_loss_ema,critic_loss_ema,learning_rate,n_updates,fps,elapsed_seconds,train_reset_seed_first,train_reset_seed_last,train_reset_seed_unique_count
```

Notes:

- `ep_success_rate`, `ep_collision_rate`, `ep_out_of_road_rate`, and
  `ep_route_completion_mean` are episode-level aggregates within the chunk
- `train_reset_seed_*` fields track the effective scenario-seed window sampled
  during the chunk
- If no episode terminates within the chunk, these fields may be null

### 2) `evals.csv` (One Row Per Aggregated Evaluation)

Purpose: main performance curve and curriculum gate tracking.

V1.5 columns:

```csv
algorithm,reward_mode,curriculum_name,seed,run_id,eval_id,chunk_id,stage,stage_index,global_step,eval_episodes,deterministic,mean_reward,std_reward,mean_env_reward,std_env_reward,mean_scalar_rule_reward,std_scalar_rule_reward,mean_rule_saturation_max,collision_rate,collision_rate_std,out_of_road_rate,success_rate,success_rate_std,route_completion,top_rule_violation_rate,success_rate_min,collision_rate_max,out_of_road_rate_max,top_rule_violation_rate_max,route_completion_min,gate_success_pass,gate_collision_pass,gate_out_of_road_pass,gate_top_rule_pass,gate_route_completion_pass,passed_eval_gates,consecutive_passes,warmup_evals_required,consecutive_evals_required,promoted,next_stage
```

Notes:

- Separate gates avoid the ambiguity of a single `passed_eval_gates`
- The thresholds used should be saved to preserve historical interpretability
  of the runs

### 3) `promotions.csv` (Promotion Events Only)

Purpose: track stage progression and time to promotion.

V1.5 columns:

```csv
algorithm,reward_mode,curriculum_name,seed,run_id,event_type,from_stage,to_stage,from_stage_index,to_stage_index,eval_id,chunk_id,global_step,stage_steps_done,stage_steps_min_required,passed_eval_gates,consecutive_passes,success_rate,collision_rate,out_of_road_rate,top_rule_violation_rate,route_completion,reason
```

Guidelines:

- Write only actual promotion events (`event_type=promoted`)
- Failed gates stay in `evals.csv` with no duplication

### 4) `eval_episodes.csv` (One Row Per Evaluation Episode)

Purpose: distributional analysis, worst-case, and best/median/worst episodes.

V1.5 columns:

```csv
algorithm,reward_mode,curriculum_name,seed,run_id,eval_id,episode_id,stage,stage_index,global_step,deterministic,reward,env_reward,scalar_rule_reward,rule_rewards_by_rule,episode_length,success,collision,out_of_road,timeout,route_completion,top_rule_violation_rate
```

Notes:

- In V1.5 we do not yet require `scenario_id/scenario_seed/video_path`
  because they are not stabilized in the current episode payload
- `top_rule_violation_rate` per episode is derived from the per-episode data
  already available in `Agent.evaluate(..., return_episode_metrics=True)`
- `reward` = final reward used by the agent during evaluation
  (after possible rulebook wrapping)
- `env_reward` = native MetaDrive reward (always tracked)
- `scalar_rule_reward` = scalarized rulebook contribution
  (present when available in the rulebook wrapper)
- `rule_rewards_by_rule` = per-episode JSON with cumulative contribution or
  margin for each rule

Added V2 columns:

```csv
scenario_seed,scenario_id,error_value,violated_rules,violation_pattern,video_path
```

V2 notes:

- `scenario_seed` is the actual seed used in `env.reset(...)` for the episode
- `scenario_id` is a deterministic identifier in the form
  `seed_<scenario_seed>`
- `video_path` is optional: it may currently be `null` if per-episode video
  recording is not active

## Phase V2 (Implemented / Requested Extensions For Final Analysis)

### Additional Files

- `rule_metrics.csv`
- `final_eval.csv`

### Candidate Advanced Metrics

- `avg_error_value`
- `max_error_value`
- `counterexample_rate`
- `violated_rules_ratio`
- `unique_violation_patterns`

Agreed definitions:

- Rule margin is the base signal. Convention: violation if `margin < 0`,
  satisfaction or non-violation if `margin >= 0`
- `error_value` is derived from negative margins
  (no metric disconnected from margins)
- At episode level:
  - `EV_episode = sum_i (w_i * max(0, -margin_i_min_episode))`
  - `margin_i_min_episode` = worst margin observed in the episode for rule `i`
  - `w_i` = priority weight, monotonic with respect to priority and
    implemented consistently with the active rulebook
- At evaluation level:
  - `avg_error_value = mean(EV_episode)`
  - `max_error_value = max(EV_episode)`

Per-rule violation representation:

- In `rule_metrics.csv` (long format): one row per (`eval_id`, `rule_name`)
  with at least:
  - `violated` (aggregated bool)
  - `violation_rate`
  - margin statistics (`mean`, `min`, `max`)
- In `eval_episodes.csv` (V2):
  - `violated_rules` as a canonical string ordered by priority
    (alphabetical tie-break)
  - `violation_pattern` using the same serialization
  - value `none` when no rule is violated

Scenario and video identifiers (V2):

- `scenario_seed`: actual seed used in `env.reset(...)`
- `scenario_id`: deterministic identifier (`seed_<scenario_seed>`)
- `video_path`: optional (`null` if no video is saved)

V2 implementation status:

- `rule_metrics.csv`: implemented
- `final_eval.csv`: implemented
- `evals.csv` / `final_eval.csv`: include the aggregated reward triplet
  (`mean_reward`, `mean_env_reward`, `mean_scalar_rule_reward` and respective
  std; rulebook fields may be null outside `reward.mode=rulebook`)
- Aggregated V2 columns in `evals.csv`
  (`avg_error_value`, `max_error_value`, `counterexample_rate`,
  `violated_rules_ratio`, `unique_violation_patterns`): implemented
- Episode-level V2 columns in `eval_episodes.csv`
  (`scenario_seed`, `scenario_id`, `error_value`, `violated_rules`,
  `violation_pattern`, `video_path`): implemented

## Analysis Pipeline (Post-Run)

Recommended pipeline for turning run CSVs into comparable results:

```text
analysis/
  aggregate_runs.py
  make_final_tables.py
  make_learning_curves.py
  make_curriculum_plots.py
  make_rulebook_tables.py
  make_tradeoff_plots.py
```

Expected outputs:

```text
<outputs-root>/analysis/aggregated/
  evals_all_runs.csv
  final_eval_all_runs.csv
  promotions_all_runs.csv
  eval_episodes_all_runs.csv
  rule_metrics_all_runs.csv
```

Notes:

- Multi-run aggregation is a prerequisite for robust plots and tables
- Protocol tables and figures should be generated from `*_all_runs.csv`, not
  from single runs

## Protocol Decisions Linked To V2

- Tabular reporting: `mean ± 95% CI` as the official convention
- Main curves: no smoothing in the primary analysis
- Optional smoothing only in follow-up analysis and always declared explicitly
  with a fixed window
- Separate and robust final evaluation:
  - intermediate evaluations: `experiment.eval_episodes`
  - final evaluation: `experiment.final_eval_episodes`
    (recommended default: `100`)

## V1.5 Acceptance Criteria

- Each training run produces the 4 CSV files in `paths.csv_dir`
- Headers are stable across different runs
- Rows in `evals.csv` are aligned 1:1 with executed evaluations
- Rows in `promotions.csv` are aligned 1:1 with actual promotions
- `eval_episodes.csv` contains `eval_episodes` rows for each evaluation
