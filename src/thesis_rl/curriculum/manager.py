from __future__ import annotations

from collections.abc import Mapping

from thesis_rl.curriculum.config import CurriculumConfig, StageConfig


class CurriculumManager:
    """Track staged curriculum state and decide automatic promotion."""

    def __init__(self, config: CurriculumConfig) -> None:
        '''Initializes the curriculum manager with the given configuration.'''
        self.config = config
        self._validate_auto_mode_eval_split()
        self._stage_idx = self._resolve_initial_stage_index()
        self._stage_steps_done = 0
        self._eval_count_at_stage = 0
        self._consecutive_passes = 0
        self._last_eval_passed = False

    def _validate_auto_mode_eval_split(self) -> None:
        """Fail-fast validation for curriculum auto mode.

        In auto mode, train-time evaluation is used for promotion gates, so we enforce
        explicit eval pools disjoint from train pools on every stage.
        """
        if not self.config.enabled:
            return
        if str(self.config.mode).lower() != "auto":
            return

        for stage in self.config.stages:
            if not stage.eval_env:
                raise ValueError(
                    "Curriculum auto mode requires `eval_env` for every stage to avoid "
                    f"train/eval scenario overlap. Missing eval_env for stage '{stage.name}'."
                )

            train_start = self._read_positive_int(stage.env, "start_seed", stage.name, "env")
            train_count = self._read_positive_int(stage.env, "num_scenarios", stage.name, "env")
            eval_start = self._read_positive_int(stage.eval_env, "start_seed", stage.name, "eval_env")
            eval_count = self._read_positive_int(stage.eval_env, "num_scenarios", stage.name, "eval_env")

            train_end = train_start + train_count - 1
            eval_end = eval_start + eval_count - 1
            overlap = max(train_start, eval_start) <= min(train_end, eval_end)
            if overlap:
                raise ValueError(
                    "Curriculum auto mode requires disjoint train/eval scenario pools per stage. "
                    f"Stage '{stage.name}' overlaps: "
                    f"train=[{train_start}, {train_end}] eval=[{eval_start}, {eval_end}]."
                )

    @staticmethod
    def _read_positive_int(
        payload: Mapping[str, object],
        key: str,
        stage_name: str,
        section: str,
    ) -> int:
        if key not in payload:
            raise ValueError(
                f"Missing `{section}.{key}` for stage '{stage_name}' in curriculum configuration."
            )
        value = int(payload[key])  # type: ignore[arg-type]
        if value <= 0 and key == "num_scenarios":
            raise ValueError(
                f"Invalid `{section}.{key}` for stage '{stage_name}': expected > 0, got {value}."
            )
        return value

    def get_current_stage(self) -> StageConfig:
        '''Returns the current curriculum stage configuration.'''
        if not self.config.stages:
            raise ValueError("Curriculum requires at least one configured stage")
        return self.config.stages[self._stage_idx]

    def get_env_config(self, evaluation: bool = False) -> dict[str, object]:
        '''Returns the environment configuration for the current stage, optionally merging evaluation overrides.'''
        stage = self.get_current_stage()
        if evaluation and stage.eval_env:
            merged = dict(stage.env)
            merged.update(stage.eval_env)
            return merged
        return dict(stage.env)

    def record_train_steps(self, num_steps: int) -> None:
        '''Records the given number of training steps done at the current stage.'''
        self._stage_steps_done += int(num_steps)

    def record_eval_metrics(self, metrics: Mapping[str, float]) -> bool:
        '''Records the given evaluation metrics and updates promotion state. Returns True if all gates were passed.'''
        self._eval_count_at_stage += 1
        self._last_eval_passed = self._passes_all_gates(metrics)
        if self._last_eval_passed:
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0
        return self._last_eval_passed

    def should_promote(self) -> bool:
        '''Determines whether promotion criteria are met based on the current state and configuration.'''

        # Check basic promotion criteria that do not depend on metrics:
        if not self.config.enabled:
            return False    # Promotion is disabled in config
        if self.config.mode.lower() != "auto":
            return False    # Promotion is only automatic in "auto" mode, otherwise it's manual
        if self._is_last_stage():
            return False    # Cannot promote if we're already at the last stage
        if self._stage_steps_done < self._min_stage_steps_for_current_stage():
            return False    # Need to have done enough training steps at the current stage before we can promote
        if self._eval_count_at_stage <= self.config.promotion.warmup_evals:
            return False    # Need to have done enough evaluations at the current stage to warm up before we can promote
        
        # We require consecutive passes to promote
        return self._consecutive_passes >= self.config.promotion.consecutive_evals

    def _min_stage_steps_for_current_stage(self) -> int:
        stage_name = self.get_current_stage().name
        per_stage = self.config.promotion.per_stage_min_steps
        if stage_name in per_stage:
            return int(per_stage[stage_name])
        if self.config.promotion.default_min_stage_steps > 0:
            return int(self.config.promotion.default_min_stage_steps)
        raise ValueError(
            "Missing valid min-stage-steps threshold for curriculum promotion: "
            f"stage='{stage_name}' not found in `promotion.per_stage`, and "
            "`promotion.default_min_stage_steps` is not > 0."
        )

    def promote(self) -> bool:
        '''
            Promotes to the next curriculum stage if promotion criteria are met. 
            Returns True if promotion occurred.
        '''
        # Check if we can promote based on the current state and configuration
        if not self.should_promote():
            return False

        # Perform promotion by advancing to the next stage and resetting relevant counters
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
        '''Determines the initial curriculum stage index based on the configuration.'''
        # If no stages are configured, start at index 0
        if not self.config.stages:
            return 0

        # For fixed curriculum mode, find the index of the specified fixed stage
        if self.config.mode.lower() != "fixed":
            return 0

        # Find the index of the fixed stage in the configured stages
        target = self.config.fixed_stage
        for idx, stage in enumerate(self.config.stages):
            if stage.name == target:
                return idx
        raise ValueError(f"Fixed curriculum stage '{target}' not found among configured stages")

    def _is_last_stage(self) -> bool:
        if not self.config.stages:
            return True
        return self._stage_idx >= len(self.config.stages) - 1

    def _passes_all_gates(self, metrics: Mapping[str, float]) -> bool:
        '''Evaluates the given metrics against all promotion gates and returns True if all gates are passed.'''
        # Recover gate thresholds
        gates = self.config.promotion.gates

        # Check safety gates first:
        # if any of them is not passed, we fail immediately without checking task or stability gates
        collision_rate = self._read_metric(metrics, "collision_rate")
        top_violation_rate = self._read_metric(metrics, "top_rule_violation_rate")
        out_of_road_rate = self._read_metric(metrics, "out_of_road_rate")
        if collision_rate is None or collision_rate > gates.safety.collision_rate_max:
            return False
        if top_violation_rate is None or top_violation_rate > gates.safety.top_rule_violation_rate_max:
            return False
        if out_of_road_rate is None or out_of_road_rate > gates.safety.out_of_road_rate_max:
            return False

        # Check task gates
        success_rate = self._read_metric(metrics, "success_rate")
        route_completion = self._read_metric(metrics, "route_completion")
        if success_rate is None or success_rate < gates.task.success_rate_min:
            return False
        if route_completion is None or route_completion < gates.task.route_completion_min:
            return False

        # Check stability gates
        success_rate_std = self._read_metric(metrics, "success_rate_std")
        collision_rate_std = self._read_metric(metrics, "collision_rate_std")
        if success_rate_std is None or success_rate_std > gates.stability.success_rate_std_max:
            return False
        if collision_rate_std is None or collision_rate_std > gates.stability.collision_rate_std_max:
            return False
        
        # If we passed all gates, return True
        return True

    @staticmethod
    def _read_metric(metrics: Mapping[str, float], key: str) -> float | None:
        '''Safely reads the specified metric from the given mapping, returning None if the metric is not present.'''
        value = metrics.get(key)
        if value is None:
            return None
        return float(value)
