from __future__ import annotations

import gymnasium as gym
from metadrive.engine.engine_utils import get_global_config
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.utils.math import clip


class ThesisPolicyBridge(EnvInputPolicy):
    """MetaDrive policy bridge used as optional env-side agent_policy.

    It stays compatible with MetaDrive's Policy contract (`act`, `get_input_space`)
    and adds configurable clipping bounds from global config.
    """

    DEBUG_MARK_COLOR = (100, 200, 255, 255)
    POLICY_LOW = -1.0
    POLICY_HIGH = 1.0

    def __init__(self, obj, seed):
        super().__init__(obj, seed)
        self.policy_low = float(self.__class__.POLICY_LOW)
        self.policy_high = float(self.__class__.POLICY_HIGH)

    def act(self, agent_id):
        action = self.engine.external_actions[agent_id]
        if self.engine.global_config.get("action_check", True):
            assert self.get_input_space().contains(action), (
                f"Input {action} is not compatible with action space {self.get_input_space()}"
            )

        to_process = self.convert_to_continuous_action(action) if self.discrete_action else action
        action = [clip(to_process[i], self.policy_low, self.policy_high) for i in range(len(to_process))]
        self.action_info["action"] = action
        self.action_info["policy"] = self.__class__.__name__
        return action

    @classmethod
    def get_input_space(cls):
        # Keep behavior aligned with EnvInputPolicy for compatibility.
        engine_global_config = get_global_config()
        discrete_action = engine_global_config["discrete_action"]
        discrete_steering_dim = engine_global_config["discrete_steering_dim"]
        discrete_throttle_dim = engine_global_config["discrete_throttle_dim"]
        use_multi_discrete = engine_global_config["use_multi_discrete"]

        if not discrete_action:
            return gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=float)

        if use_multi_discrete:
            return gym.spaces.MultiDiscrete([discrete_steering_dim, discrete_throttle_dim])
        return gym.spaces.Discrete(discrete_steering_dim * discrete_throttle_dim)
