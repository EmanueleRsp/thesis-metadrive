from __future__ import annotations

from collections.abc import Mapping

from thesis_rl.curriculum.config import CurriculumConfig, StageConfig


class CurriculumManager:
    """Track staged curriculum state and decide automatic promotion."""

    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config
        self._stage_idx = self._resolve_initial_stage_index()
        self._stage_steps_done = 0
        self._eval_count_at_stage = 0
        self._consecutive_passes = 0
        self._last_eval_passed = False

    def get_current_stage(self) -> StageConfig:
        if not self.config.stages:
            raise ValueError("Curriculum requires at least one configured stage")
        return self.config.stages[self._stage_idx]

    def get_env_config(self, evaluation: bool = False) -> dict[str, object]:
        stage = self.get_current_stage()
        if evaluation and stage.eval_env:
            merged = dict(stage.env)
            merged.update(stage.eval_env)
            return merged
        return dict(stage.env)

    def record_train_steps(self, num_steps: int) -> None:
        self._stage_steps_done += int(num_steps)

    def record_eval_metrics(self, metrics: Mapping[str, float]) -> bool:
        self._eval_count_at_stage += 1
        self._last_eval_passed = self._passes_all_gates(metrics)
        if self._last_eval_passed:
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0
        return self._last_eval_passed

    def should_promote(self) -> bool:
        if not self.config.enabled:
            return False
        if self.config.mode.lower() != "auto":
            return False
        if self._is_last_stage():
            return False
        if self._stage_steps_done < self.config.promotion.min_stage_steps:
            return False
        if self._eval_count_at_stage <= self.config.promotion.warmup_evals:
            return False
        return self._consecutive_passes >= self.config.promotion.consecutive_evals

    def promote(self) -> bool:
        if not self.should_promote():
            return False

        self._stage_idx += 1
        self._stage_steps_done = 0
        self._eval_count_at_stage = 0
        self._consecutive_passes = 0
        self._last_eval_passed = False
        return True

    def is_finished(self) -> bool:
        return self._is_last_stage()

    @property
    def stage_index(self) -> int:
        return self._stage_idx

    @property
    def stage_steps_done(self) -> int:
        return self._stage_steps_done

    @property
    def eval_count_at_stage(self) -> int:
        return self._eval_count_at_stage

    @property
    def consecutive_passes(self) -> int:
        return self._consecutive_passes

    def _resolve_initial_stage_index(self) -> int:
        if not self.config.stages:
            return 0

        if self.config.mode.lower() != "fixed":
            return 0

        target = self.config.fixed_stage
        for idx, stage in enumerate(self.config.stages):
            if stage.name == target:
                return idx
        return 0

    def _is_last_stage(self) -> bool:
        if not self.config.stages:
            return True
        return self._stage_idx >= len(self.config.stages) - 1

    def _passes_all_gates(self, metrics: Mapping[str, float]) -> bool:
        gates = self.config.promotion.gates

        collision_rate = self._read_metric(metrics, "collision_rate")
        top_violation_rate = self._read_metric(metrics, "top_rule_violation_rate")
        out_of_road_rate = self._read_metric(metrics, "out_of_road_rate")

        if collision_rate is None or collision_rate > gates.safety.collision_rate_max:
            return False
        if top_violation_rate is None or top_violation_rate > gates.safety.top_rule_violation_rate_max:
            return False
        if out_of_road_rate is None or out_of_road_rate > gates.safety.out_of_road_rate_max:
            return False

        success_rate = self._read_metric(metrics, "success_rate")
        route_completion = self._read_metric(metrics, "route_completion")
        if success_rate is None or success_rate < gates.task.success_rate_min:
            return False
        if route_completion is None or route_completion < gates.task.route_completion_min:
            return False

        mean_reward = self._read_metric(metrics, "mean_reward")
        if mean_reward is None or mean_reward < gates.quality.mean_reward_min:
            return False
        return True

    @staticmethod
    def _read_metric(metrics: Mapping[str, float], key: str) -> float | None:
        value = metrics.get(key)
        if value is None:
            return None
        return float(value)
