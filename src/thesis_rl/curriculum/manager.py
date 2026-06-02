from __future__ import annotations

from collections.abc import Mapping

from thesis_rl.curriculum.config import CurriculumConfig, StageConfig
from thesis_rl.curriculum.interfaces import CurriculumStrategy
from thesis_rl.curriculum.registry import build_curriculum_strategy
from thesis_rl.curriculum.state import CurriculumState


class CurriculumManager:
    """Wrap a curriculum strategy and expose state helpers."""

    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config
        if not config.enabled:
            raise ValueError("CurriculumManager requires curriculum.enabled=true.")
        if str(config.kind).lower() == "disabled":
            raise ValueError("CurriculumManager cannot be created with kind=disabled.")
        self._strategy: CurriculumStrategy = build_curriculum_strategy(config)

    def get_current_stage(self) -> StageConfig:
        return self._strategy.get_current_stage()

    def get_env_config(self, evaluation: bool = False) -> dict[str, object]:
        return self._strategy.get_env_config(evaluation=evaluation)

    def record_train_steps(self, num_steps: int) -> None:
        self._strategy.record_train_steps(num_steps)

    def record_eval_metrics(self, metrics: Mapping[str, float]) -> bool:
        return self._strategy.record_eval_metrics(metrics)

    def should_promote(self) -> bool:
        return self._strategy.should_promote()

    def promote(self) -> bool:
        return self._strategy.promote()

    def is_finished(self) -> bool:
        return self._strategy.is_finished()

    def load_state(self, payload: Mapping[str, object] | CurriculumState | None) -> None:
        if payload is None:
            return
        if isinstance(payload, CurriculumState):
            state = payload
        elif isinstance(payload, Mapping):
            state = CurriculumState.from_mapping(payload)
        else:
            raise TypeError("Curriculum state must be a mapping or CurriculumState.")
        self._strategy.load_state(state)

    @property
    def state(self) -> CurriculumState:
        return self._strategy.state

    def state_dict(self) -> dict[str, object]:
        return self._strategy.state.to_dict()

    @property
    def stage_index(self) -> int:
        return int(self._strategy.state.stage_index)

    @property
    def stage_steps_done(self) -> int:
        return int(self._strategy.state.stage_steps_done)

    @property
    def eval_count_at_stage(self) -> int:
        return int(self._strategy.state.eval_count_at_stage)

    @property
    def consecutive_passes(self) -> int:
        return int(self._strategy.state.consecutive_passes)

    @property
    def last_eval_passed(self) -> bool:
        return bool(self._strategy.state.last_eval_passed)
