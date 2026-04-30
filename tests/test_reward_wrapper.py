from __future__ import annotations

import numpy as np

from thesis_rl.envs.wrappers import RuleRewardWrapper
from thesis_rl.reward.base import RewardComputationResult


class _DummyLane:
    def local_coordinates(self, point):
        _ = point
        return 0.0, 0.0


class _DummyFinalLane:
    length = 5.0

    def position(self, longitudinal, lateral):
        return [float(longitudinal), float(lateral)]


class _DummyNavigation:
    def __init__(self):
        self.current_ref_lanes = [_DummyLane()]
        self.final_lane = _DummyFinalLane()


class _DummyVehicle:
    def __init__(self, position, speed):
        self.position = np.asarray(position, dtype=np.float32)
        self.velocity = np.asarray([speed, 0.0], dtype=np.float32)
        self.speed = float(speed)
        self.speed_km_h = float(speed) * 3.6
        self.heading_theta = 0.2
        self.steering = 0.1
        self.throttle_brake = 0.0
        self.LENGTH = 4.5
        self.WIDTH = 1.8
        self.max_speed_km_h = 50.0
        self.navigation = _DummyNavigation()
        self.lane = None
        self.bounding_box = None


class _DummyEnv:
    def __init__(self) -> None:
        self._ego = _DummyVehicle([0.0, 0.0], 3.0)
        self._neighbor = _DummyVehicle([2.0, 1.0], 1.0)
        self.agents = {"ego": self._ego, "other": self._neighbor}

    def reset(self, **kwargs):
        _ = kwargs
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        return np.array([1.0], dtype=np.float32), 1.0, False, False, {"velocity": 0.0}


class _DummyManager:
    def __init__(self) -> None:
        self.last_info = None

    def reset(self) -> None:
        return None

    def compute(self, env_reward: float, info: dict):
        _ = env_reward
        self.last_info = dict(info)
        return RewardComputationResult(
            final_reward=2.5,
            scalar_rule_reward=1.5,
            rule_reward_vector=[0.1, -0.2],
            rule_bounded_vector=[0.01, -0.02],
            rule_components={"a": 0.1, "b": -0.2},
            rule_metadata={"evaluator": "dummy"},
            rule_violation_vector=[0.0, 0.2],
        )


def test_rule_reward_wrapper_enriches_info_and_overrides_reward() -> None:
    manager = _DummyManager()
    env = RuleRewardWrapper(_DummyEnv(), manager, reward_mode="hybrid", attach_info=True)
    _obs, _info = env.reset()

    _next_obs, reward, _done, _truncated, info = env.step(np.array([0.0], dtype=np.float32))

    assert reward == 2.5
    assert info["env_reward"] == 1.0
    assert info["scalar_rule_reward"] == 1.5
    assert info["hybrid_reward"] == 2.5
    assert info["selected_reward"] == 2.5
    assert info["rule_reward_vector"] == [0.1, -0.2]
    assert info["rule_bounded_vector"] == [0.01, -0.02]
    assert info["rule_components"] == {"a": 0.1, "b": -0.2}
    assert info["rule_violation_vector"] == [0.0, 0.2]

    assert manager.last_info is not None
    assert isinstance(manager.last_info["ego_state"], dict)
    assert "position" in manager.last_info["ego_state"]
    assert "speed" in manager.last_info["ego_state"]
    assert manager.last_info["lane_centerline"] is not None
    assert manager.last_info["target_point"] == [5.0, 0.0]
    assert manager.last_info["speed_limit"] == 50.0
    assert isinstance(manager.last_info["neighbors"], list)
    assert len(manager.last_info["neighbors"]) == 1


def test_rule_reward_wrapper_can_return_env_reward_with_rulebook_diagnostics() -> None:
    manager = _DummyManager()
    env = RuleRewardWrapper(_DummyEnv(), manager, reward_mode="scalar_default", attach_info=True)
    _obs, _info = env.reset()

    _next_obs, reward, _done, _truncated, info = env.step(np.array([0.0], dtype=np.float32))

    assert reward == 1.0
    assert info["env_reward"] == 1.0
    assert info["scalar_rule_reward"] == 1.5
    assert info["hybrid_reward"] == 2.5
    assert info["selected_reward"] == 1.0


def test_rule_reward_wrapper_can_return_scalar_rulebook_reward() -> None:
    manager = _DummyManager()
    env = RuleRewardWrapper(_DummyEnv(), manager, reward_mode="scalar_rulebook", attach_info=True)
    _obs, _info = env.reset()

    _next_obs, reward, _done, _truncated, info = env.step(np.array([0.0], dtype=np.float32))

    assert reward == 1.5
    assert info["selected_reward"] == 1.5
