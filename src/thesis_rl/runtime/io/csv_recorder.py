from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from thesis_rl.runtime.comfort_diagnostics import (
    COMFORT_AGGREGATE_COLUMNS,
    COMFORT_EPISODE_COLUMNS,
)
from thesis_rl.runtime.route_adherence_diagnostics import (
    ROUTE_ADHERENCE_EPISODE_COLUMNS,
)


class CSVRecorder:
    """Append structured rows to run-scoped CSV files with fixed schemas."""

    SCHEMAS: dict[str, list[str]] = {
        "train_chunks.csv": [
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum_name",
            "rulebook_config",
            "seed",
            "run_id",
            "chunk_id",
            "stage",
            "stage_index",
            "steps_start",
            "steps_end",
            "global_step",
            "chunk_steps",
            "episodes",
            "ep_rew_mean",
            "ep_rew_std",
            "ep_rew_ci_95",
            "ep_env_rew_mean",
            "ep_scalar_rule_rew_mean",
            "ep_hybrid_rew_mean",
            "ep_len_mean",
            "ep_len_std",
            "ep_len_ci_95",
            "ep_success_rate",
            "ep_collision_rate",
            "ep_out_of_road_rate",
            "ep_route_completion_mean",
            # For TD3 and DDPG, `actor_loss` is exactly `-mean Q1(s, pi(s))`
            # (`third_party/stable-baselines3/.../td3/td3.py:215`), so **negating
            # this column gives the mean critic value of the on-policy action**.
            # That is the quantity that drifts when the discount cannot damp a
            # bootstrapped horizon, and it needs no separate column: a rising
            # `-actor_loss` alongside a rising `critic_loss` is value drift, a
            # rising `critic_loss` with a flat `-actor_loss` is not.
            # The identity does not hold for SAC, whose actor loss carries the
            # entropy term `ent_coef * log_prob` as well.
            "actor_loss",
            "critic_loss",
            "actor_loss_ema",
            "critic_loss_ema",
            "learning_rate",
            "update_calls",
            "n_updates",
            "fps",
            "elapsed_seconds",
            "train_reset_seed_first",
            "train_reset_seed_last",
            "train_reset_seed_unique_count",
        ],
        "evals.csv": [
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum_name",
            "rulebook_config",
            "seed",
            "run_id",
            "eval_id",
            "eval_type",
            "scenario_set",
            "evaluation_batch_id",
            "panel_name",
            "evaluation_scope",
            "panel_sha256",
            "parent_panel_sha256",
            "frozen_selection_hash",
            "checkpoint_identity",
            "chunk_id",
            "stage",
            "stage_index",
            "global_step",
            "eval_episodes",
            "deterministic",
            "mean_reward",
            "std_reward",
            "mean_env_reward",
            "std_env_reward",
            "mean_scalar_rule_reward",
            "std_scalar_rule_reward",
            "mean_hybrid_reward",
            "std_hybrid_reward",
            "mean_rule_saturation_max",
            "collision_rate",
            "collision_rate_std",
            "out_of_road_rate",
            "success_rate",
            "success_rate_std",
            "route_completion",
            "top_rule_violation_rate",
            "avg_error_value",
            "max_error_value",
            "counterexample_rate",
            "violated_rules_ratio",
            "unique_violation_patterns",
            "success_rate_min",
            "collision_rate_max",
            "out_of_road_rate_max",
            "top_rule_violation_rate_max",
            "route_completion_min",
            "gate_success_pass",
            "gate_collision_pass",
            "gate_out_of_road_pass",
            "gate_top_rule_pass",
            "gate_route_completion_pass",
            "passed_eval_gates",
            "consecutive_passes",
            "warmup_evals_required",
            "consecutive_evals_required",
            "promoted",
            "next_stage",
            "data_abort_attempted",
            "data_abort_valid",
            "data_abort_invalid",
            "data_abort_coverage",
            # step_timing_instrumentation_v1 REQ-003: GIF render/annotation
            # cost, isolated from training-loop timing (docs/implementation/
            # step_timing_instrumentation_v1_exec_plan.md).
            "gif_render_seconds_total",
            "gif_render_seconds_per_episode",
            # EP-COMFORT-DIAG: seed-level ride-comfort diagnostics
            # (docs/implementation/comfort_and_jerk_diagnostics_exec_plan.md).
            # Diagnostic only -- `RULEBOOK-V5.1` §13 excludes comfort and jerk
            # from the rulebook and the reward.
            *COMFORT_AGGREGATE_COLUMNS,
        ],
        "eval_episodes.csv": [
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum_name",
            "rulebook_config",
            "seed",
            "run_id",
            "eval_id",
            "eval_type",
            "scenario_set",
            "evaluation_batch_id",
            "panel_name",
            "evaluation_scope",
            "panel_sha256",
            "parent_panel_sha256",
            "frozen_selection_hash",
            "checkpoint_identity",
            "episode_id",
            "stage",
            "stage_index",
            "global_step",
            "scenario_seed",
            "scenario_id",
            "scenario_uid",
            "source",
            "split",
            "primary_arm",
            "worker_id",
            "termination_reason",
            "terminated",
            "truncated",
            "sampling_mode",
            "requested_arm",
            "source_cell_fallback",
            "deterministic",
            "reward",
            "env_reward",
            "scalar_rule_reward",
            "hybrid_reward",
            "rule_rewards_by_rule",
            "episode_length",
            "success",
            "collision",
            "out_of_road",
            "timeout",
            "route_completion",
            "top_rule_violation_rate",
            "error_value",
            "violated_rules",
            "violation_pattern",
            "video_path",
            "video_authoritative_path",
            "video_manifest_path",
            "trajectory_log_path",
            "video_recorded_live",
            "replay_warning",
            # EP-COMFORT-DIAG: per-episode ride-comfort diagnostics; an empty
            # cell means the channel was undefined for that episode, which is
            # not the same as comfortable (`REQ-CMF-07`).
            *COMFORT_EPISODE_COLUMNS,
            # `REQ-EF-15` / `D14` and RULEBOOK-V5.1 §7: reported, never priced.
            # `route_fully_outside_max_run` is the one to read -- a mean cannot
            # separate a clipped corner from driving the wrong carriageway.
            *ROUTE_ADHERENCE_EPISODE_COLUMNS,
        ],
        "promotions.csv": [
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum_name",
            "rulebook_config",
            "seed",
            "run_id",
            "event_type",
            "from_stage",
            "to_stage",
            "from_stage_index",
            "to_stage_index",
            "eval_id",
            "chunk_id",
            "global_step",
            "stage_steps_done",
            "stage_steps_min_required",
            "passed_eval_gates",
            "consecutive_passes",
            "success_rate",
            "collision_rate",
            "out_of_road_rate",
            "top_rule_violation_rate",
            "route_completion",
            "reason",
        ],
        "rule_metrics.csv": [
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum_name",
            "rulebook_config",
            "seed",
            "run_id",
            "eval_id",
            "eval_type",
            "scenario_set",
            "evaluation_batch_id",
            "panel_name",
            "evaluation_scope",
            "panel_sha256",
            "parent_panel_sha256",
            "frozen_selection_hash",
            "checkpoint_identity",
            "chunk_id",
            "stage",
            "stage_index",
            "global_step",
            "rule_name",
            "rule_priority",
            "violated",
            "violation_rate",
            "violation_count",
            "mean_margin",
            "min_margin",
            "max_margin",
            # EVAL-PROTOCOL v1.0 REQ-008 (spec §7.2): the count of episodes
            # with >=1 applicable step for this rule feeding `violation_rate`
            # /`mean_margin`, and the count excluded because they had zero
            # applicable steps for this rule.
            "applicable_episode_count",
            "excluded_episode_count",
        ],
        # EP-SUBRULE-DIAG: additive R2/R3 sub-rule dominance/cost diagnostics
        # (docs/implementation/subrule_dominance_diagnostics_exec_plan.md).
        # Diagnostic only -- never a primary comparison metric (`DEC-SUB-001`).
        "subrule_metrics.csv": [
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum_name",
            "rulebook_config",
            "seed",
            "run_id",
            "eval_id",
            "eval_type",
            "scenario_set",
            "evaluation_batch_id",
            "panel_name",
            "evaluation_scope",
            "panel_sha256",
            "parent_panel_sha256",
            "frozen_selection_hash",
            "checkpoint_identity",
            "chunk_id",
            "stage",
            "stage_index",
            "global_step",
            "scenario_source",
            "macro_rule",
            "subrule_name",
            "applicability_rate",
            "violation_rate",
            "mean_cost",
            "max_cost",
            "dominance_share",
            "worst_component_count",
            "macro_violated_step_count",
            "multi_violation_share",
            "applicable_episode_count",
            "excluded_episode_count",
        ],
        "final_eval.csv": [
            "algorithm",
            "reward_type",
            "reward_behavior",
            "curriculum_name",
            "rulebook_config",
            "curriculum_enabled",
            "seed",
            "run_id",
            "eval_type",
            "scenario_set",
            "evaluation_batch_id",
            "panel_name",
            "evaluation_scope",
            "panel_sha256",
            "parent_panel_sha256",
            "frozen_selection_hash",
            "checkpoint_identity",
            "total_timesteps",
            "final_stage",
            "final_stage_index",
            "final_stage_reached",
            "steps_to_final_stage",
            "final_eval_episodes",
            "deterministic",
            "mean_reward",
            "std_reward",
            "mean_env_reward",
            "std_env_reward",
            "mean_scalar_rule_reward",
            "std_scalar_rule_reward",
            "mean_hybrid_reward",
            "std_hybrid_reward",
            "mean_rule_saturation_max",
            "collision_rate",
            "collision_rate_std",
            "out_of_road_rate",
            "success_rate",
            "success_rate_std",
            "route_completion",
            "top_rule_violation_rate",
            "avg_error_value",
            "max_error_value",
            "counterexample_rate",
            "violated_rules_ratio",
            "unique_violation_patterns",
            "checkpoint_path",
            "checkpoint_type",
            "checkpoint_global_step",
            "checkpoint_hash",
            "checkpoint_role",
            "data_abort_attempted",
            "data_abort_valid",
            "data_abort_invalid",
            "data_abort_coverage",
            # EP-COMFORT-DIAG: seed-level ride-comfort diagnostics
            # (docs/implementation/comfort_and_jerk_diagnostics_exec_plan.md).
            # Diagnostic only -- `RULEBOOK-V5.1` §13 excludes comfort and jerk
            # from the rulebook and the reward.
            *COMFORT_AGGREGATE_COLUMNS,
        ],
        # step_timing_instrumentation_v1 REQ-001/REQ-002: tidy per-chunk,
        # per-component wall-clock breakdown (docs/implementation/
        # step_timing_instrumentation_v1_exec_plan.md). One row per
        # (chunk, component) rather than fixed columns, because the
        # component key set is dynamic (Rulebook v2 sub-phases and
        # per-algorithm learner detail vary with configuration).
        "step_timing.csv": [
            "algorithm",
            "seed",
            "run_id",
            "chunk_id",
            "global_step",
            "component",
            "seconds",
            "pct_of_elapsed",
            "avg_seconds_per_step",
        ],
    }

    def __init__(self, csv_dir: str | Path) -> None:
        self.csv_dir = Path(csv_dir)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

    def append_row(self, filename: str, row: dict[str, Any]) -> None:
        if filename not in self.SCHEMAS:
            raise KeyError(f"Unsupported CSV schema: {filename}")

        path = self.csv_dir / filename
        fieldnames = self.SCHEMAS[filename]
        file_exists = path.exists()
        clean_row = {key: row.get(key) for key in fieldnames}

        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(clean_row)
