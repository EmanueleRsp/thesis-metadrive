from __future__ import annotations

from collections.abc import Mapping

from thesis_rl.curriculum.config import ScenarioAclConfig, StageConfig
from thesis_rl.curriculum.state import CurriculumState


class ScenarioAclCurriculum:
    """Phase-1 placeholder strategy for scenario-level curriculum wiring."""

    def __init__(self, config: ScenarioAclConfig) -> None:
        self.config = config
        self._state = CurriculumState()
        self._stage = StageConfig(name="scenario_acl", env={}, eval_env={})

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
        return self._stage

    def get_env_config(self, evaluation: bool = False) -> dict[str, object]:
        del evaluation
        return {}

    def record_train_steps(self, num_steps: int) -> None:
        self._state.stage_steps_done += int(num_steps)

    def record_eval_metrics(self, metrics: Mapping[str, float]) -> bool:
        del metrics
        self._state.eval_count_at_stage += 1
        self._state.last_eval_passed = False
        self._state.consecutive_passes = 0
        return False

    def should_promote(self) -> bool:
        return False

    def promote(self) -> bool:
        return False

    def is_finished(self) -> bool:
        return False
