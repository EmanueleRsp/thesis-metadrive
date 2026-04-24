from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from thesis_rl.reward.reward_manager import HybridRulebookRewardManager
from thesis_rl.rulebook.types import RuleEvalInput, RuleVector


class _FakeEvaluator:
    def __init__(self) -> None:
        self.rules = [SimpleNamespace(name="speed_limit"), SimpleNamespace(name="goal_progress")]

    def evaluate(self, rule_eval_input: RuleEvalInput) -> RuleVector:
        _ = rule_eval_input
        return RuleVector(
            names=["speed_limit", "goal_progress"],
            values=np.asarray([-2.0, 1.0], dtype=np.float32),
            priorities=[0, 1],
            metadata={"evaluator": "fake"},
        )


def test_hybrid_manager_scalarization_and_outputs() -> None:
    manager = HybridRulebookRewardManager(
        evaluator=_FakeEvaluator(),
        a=2.01,
        c=30.0,
        lambda_env=1.0,
        lambda_rule=0.2,
        scales={"speed_limit": 5.0, "goal_progress": 25.0},
        include_violation_vector=True,
    )

    result = manager.compute(env_reward=10.0, info={"episode_length": 3})

    assert isinstance(result.final_reward, float)
    assert isinstance(result.scalar_rule_reward, float)
    assert len(result.rule_reward_vector) == 2
    assert len(result.rule_bounded_vector) == 2
    assert all(-1.0 <= x <= 1.0 for x in result.rule_bounded_vector)
    assert result.rule_components["speed_limit"] == -2.0
    assert result.rule_components["goal_progress"] == 1.0
    assert result.rule_violation_vector == [2.0, 0.0]
    assert "saturation_ratio_by_rule" in result.rule_metadata


def test_manager_from_configs_builds_and_runs() -> None:
    cfg_reward = OmegaConf.create(
        {
            "a": 2.01,
            "c": 30.0,
            "lambda_env": 1.0,
            "lambda_rule": 0.2,
            "include_violation_vector": False,
            "scales": {"speed_limit": 5.0},
        }
    )
    cfg_rulebook = OmegaConf.create(
        {
            "rules": [
                {"name": "speed_limit", "priority": 0},
            ]
        }
    )

    manager = HybridRulebookRewardManager.from_configs(cfg_reward, cfg_rulebook)
    result = manager.compute(env_reward=0.5, info={"speed_limit": 10.0, "velocity": 12.0})

    assert len(result.rule_reward_vector) == 1
    assert float(result.rule_reward_vector[0]) < 0.0
    assert "rule_names" in result.rule_metadata


def test_extract_ego_state_supports_multiple_metadrive_keys() -> None:
    info = {
        "vehicle_position": [1.0, 2.0],
        "velocity": [3.0, 4.0],
        "acceleration": {"x": 0.5, "y": -0.1},
        "raw_action": [0.2, 0.7],
        "carsize": [1.8, 4.5],
        "yaw": 0.3,
    }

    ego_state = HybridRulebookRewardManager._extract_ego_state(info)

    assert ego_state["position"] == [1.0, 2.0]
    assert ego_state["velocity"] == [3.0, 4.0]
    assert ego_state["acceleration"] == {"x": 0.5, "y": -0.1}
    assert ego_state["steer"] == 0.2
    assert ego_state["accel"] == 0.7
    assert ego_state["width"] == 1.8
    assert ego_state["length"] == 4.5
    assert ego_state["yaw"] == 0.3


def test_extract_neighbors_uses_fallback_keys() -> None:
    info = {
        "surrounding_vehicles": [
            {"position": [0.0, 0.0]},
            {"position": [1.0, 1.0]},
        ]
    }

    neighbors = HybridRulebookRewardManager._extract_neighbors(info)
    assert len(neighbors) == 2


def test_saturation_metadata_tracks_ratio() -> None:
    manager = HybridRulebookRewardManager(
        evaluator=_FakeEvaluator(),
        a=2.01,
        c=30.0,
        lambda_env=1.0,
        lambda_rule=0.2,
        scales={"speed_limit": 0.1, "goal_progress": 0.1},
        include_violation_vector=False,
    )

    result = manager.compute(env_reward=0.0, info={})
    ratios = result.rule_metadata["saturation_ratio_by_rule"]

    assert "speed_limit" in ratios
    assert "goal_progress" in ratios
    assert 0.0 <= ratios["speed_limit"] <= 1.0
    assert 0.0 <= ratios["goal_progress"] <= 1.0


def test_scalarization_matches_closed_form_reference() -> None:
    manager = HybridRulebookRewardManager(
        evaluator=_FakeEvaluator(),
        a=2.01,
        c=30.0,
        lambda_env=1.0,
        lambda_rule=0.2,
        scales={"speed_limit": 5.0, "goal_progress": 25.0},
        include_violation_vector=False,
    )

    env_reward = 3.5
    result = manager.compute(env_reward=env_reward, info={})

    margins = np.asarray([-2.0, 1.0], dtype=np.float64)
    scales = np.asarray([5.0, 25.0], dtype=np.float64)
    rho = np.tanh(margins / scales)
    n = 2
    exponents = np.asarray([2, 1], dtype=np.float64)
    sigmoid = 1.0 / (1.0 + np.exp(-30.0 * rho))
    expected_rule = float(np.sum((2.01**exponents) * sigmoid + (rho / n)))
    expected_final = float((1.0 * env_reward + 0.2 * expected_rule) / (1.0 + 0.2))

    assert np.isclose(result.scalar_rule_reward, expected_rule, rtol=1e-10, atol=1e-10)
    assert np.isclose(result.final_reward, expected_final, rtol=1e-10, atol=1e-10)
