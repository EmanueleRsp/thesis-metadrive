from __future__ import annotations

from collections.abc import Mapping

from thesis_rl.curriculum.config import StageConfig, StagedCurriculumConfig
from thesis_rl.curriculum.state import CurriculumState


class StagedCurriculum:
    """Track staged curriculum state and decide automatic promotion."""

    def __init__(self, config: StagedCurriculumConfig) -> None:
        self.config = config
        self._validate_auto_mode_eval_split()
        self._state = CurriculumState(stage_index=self._resolve_initial_stage_index())

    def _validate_auto_mode_eval_split(self) -> None:
        """Fail-fast validation for curriculum auto mode.

        In auto mode, train-time evaluation is used for promotion gates, so we enforce
        explicit eval pools disjoint from train pools on every stage.
        """
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

    @property
    def state(self) -> CurriculumState:
        return self._state

    def load_state(self, state: CurriculumState) -> None:
        self._state = CurriculumState(
            stage_index=int(state.stage_index),
            stage_steps_done=int(state.stage_steps_done),
            eval_count_at_stage=int(state.eval_count_at_stage),
            consecutive_passes=int(state.consecutive_passes),
            last_eval_passed=bool(state.last_eval_passed),
        )

    def get_current_stage(self) -> StageConfig:
        if not self.config.stages:
            raise ValueError("Curriculum requires at least one configured stage")
        return self.config.stages[self._state.stage_index]

    def get_env_config(self, evaluation: bool = False) -> dict[str, object]:
        stage = self.get_current_stage()
        if evaluation and stage.eval_env:
            merged = dict(stage.env)
            merged.update(stage.eval_env)
            return merged
        return dict(stage.env)

    def record_train_steps(self, num_steps: int) -> None:
        self._state.stage_steps_done += int(num_steps)

    def record_eval_metrics(self, metrics: Mapping[str, float]) -> bool:
        self._state.eval_count_at_stage += 1
        self._state.last_eval_passed = self._passes_all_gates(metrics)
        if self._state.last_eval_passed:
            self._state.consecutive_passes += 1
        else:
            self._state.consecutive_passes = 0
        return self._state.last_eval_passed

    def should_promote(self) -> bool:
        if str(self.config.mode).lower() != "auto":
            return False
        if self._is_last_stage():
            return False
        if self._state.stage_steps_done < self._min_stage_steps_for_current_stage():
            return False
        if self._state.eval_count_at_stage <= self.config.promotion.warmup_evals:
            return False
        return self._state.consecutive_passes >= self.config.promotion.consecutive_evals

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
        if not self.should_promote():
            return False
        self._state.stage_index += 1
        self._state.stage_steps_done = 0
        self._state.eval_count_at_stage = 0
        self._state.consecutive_passes = 0
        self._state.last_eval_passed = False
        return True

    def is_finished(self) -> bool:
        return self._is_last_stage()

    def _resolve_initial_stage_index(self) -> int:
        if not self.config.stages:
            return 0
        if str(self.config.mode).lower() != "fixed":
            return 0

        target = self.config.fixed_stage
        for idx, stage in enumerate(self.config.stages):
            if stage.name == target:
                return idx
        raise ValueError(f"Fixed curriculum stage '{target}' not found among configured stages")

    def _is_last_stage(self) -> bool:
        if not self.config.stages:
            return True
        return self._state.stage_index >= len(self.config.stages) - 1

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

        success_rate_std = self._read_metric(metrics, "success_rate_std")
        collision_rate_std = self._read_metric(metrics, "collision_rate_std")
        if success_rate_std is None or success_rate_std > gates.stability.success_rate_std_max:
            return False
        if collision_rate_std is None or collision_rate_std > gates.stability.collision_rate_std_max:
            return False

        return True

    @staticmethod
    def _read_metric(metrics: Mapping[str, float], key: str) -> float | None:
        value = metrics.get(key)
        if value is None:
            return None
        return float(value)
