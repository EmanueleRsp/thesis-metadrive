from __future__ import annotations

from typing import cast

import pytest

from thesis_rl.curriculum import (
    CurriculumConfig,
    CurriculumManager,
    CurriculumState,
    CurriculumStrategy,
    build_curriculum_strategy,
    register_curriculum_strategy,
)
from thesis_rl.curriculum.config import StageConfig


def _staged_config() -> CurriculumConfig:
    return CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "staged",
            "staged": {
                "mode": "fixed",
                "fixed_stage": "stage1",
                "stages": [
                    {"name": "stage1", "env": {"map": 1}, "eval_env": {"map": 2}},
                ],
            },
        }
    )


def test_curriculum_state_roundtrip_mapping() -> None:
    state = CurriculumState(
        stage_index=2,
        stage_steps_done=123,
        eval_count_at_stage=4,
        consecutive_passes=2,
        last_eval_passed=True,
    )

    payload = state.to_dict()
    restored = CurriculumState.from_mapping(payload)

    assert restored == state


def test_curriculum_manager_load_state_from_mapping() -> None:
    manager = CurriculumManager(_staged_config())

    manager.load_state(
        {
            "stage_index": 0,
            "stage_steps_done": 42,
            "eval_count_at_stage": 3,
            "consecutive_passes": 1,
            "last_eval_passed": True,
        }
    )

    assert manager.state == CurriculumState(
        stage_index=0,
        stage_steps_done=42,
        eval_count_at_stage=3,
        consecutive_passes=1,
        last_eval_passed=True,
    )
    assert manager.state_dict() == {
        "stage_index": 0,
        "stage_steps_done": 42,
        "eval_count_at_stage": 3,
        "consecutive_passes": 1,
        "last_eval_passed": True,
    }


def test_curriculum_config_rejects_missing_kind() -> None:
    with pytest.raises(ValueError, match="requires `kind`"):
        CurriculumConfig.from_mapping({"enabled": True})


def test_curriculum_config_rejects_staged_without_block() -> None:
    with pytest.raises(ValueError, match="requires a 'staged' block"):
        CurriculumConfig.from_mapping({"enabled": True, "kind": "staged"})


def test_build_curriculum_strategy_returns_protocol_compatible_strategy() -> None:
    strategy = build_curriculum_strategy(_staged_config())

    protocol_view = cast(CurriculumStrategy, strategy)

    assert protocol_view.get_current_stage().name == "stage1"
    assert protocol_view.get_env_config(evaluation=True)["map"] == 2


def test_register_curriculum_strategy_supports_custom_kind() -> None:
    class DummyCurriculum:
        def __init__(self) -> None:
            self._state = CurriculumState()
            self._stage = StageConfig(name="dummy", env={"foo": 1}, eval_env={"foo": 2})

        def get_current_stage(self) -> StageConfig:
            return self._stage

        def get_env_config(self, evaluation: bool = False) -> dict[str, object]:
            payload = dict(self._stage.env)
            if evaluation:
                payload.update(self._stage.eval_env)
            return payload

        def record_train_steps(self, num_steps: int) -> None:
            self._state.stage_steps_done += int(num_steps)

        def record_eval_metrics(self, metrics: dict[str, float]) -> bool:
            self._state.eval_count_at_stage += 1
            self._state.last_eval_passed = True
            return True

        def should_promote(self) -> bool:
            return False

        def promote(self) -> bool:
            return False

        def is_finished(self) -> bool:
            return True

        @property
        def state(self) -> CurriculumState:
            return self._state

        def load_state(self, state: CurriculumState) -> None:
            self._state = state

    register_curriculum_strategy("dummy", lambda _config: DummyCurriculum())

    config = CurriculumConfig(enabled=True, kind="dummy")
    strategy = build_curriculum_strategy(config)

    assert strategy.get_current_stage().name == "dummy"
    assert strategy.get_env_config(evaluation=True)["foo"] == 2
