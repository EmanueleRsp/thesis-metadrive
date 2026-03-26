from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class EnvInputPolicyBridge:
    """MetaDrive-compatible policy bridge.

    The interface mirrors MetaDrive policy contracts with:
    - act(...): produce raw 2D control
    - get_input_space(...): define accepted external input space
    """

    def __init__(self, low: float, high: float, action_check: bool = True) -> None:
        self.low = float(low)
        self.high = float(high)
        self.action_check = action_check
        self.action_info: dict[str, Any] = {}

    @classmethod
    def get_input_space(cls) -> gym.spaces.Box:
        return gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def act(self, external_input: Any) -> np.ndarray:
        action = np.asarray(external_input, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Policy expects shape (2,), got {action.shape}")

        if self.action_check and not self.get_input_space().contains(action):
            raise ValueError(
                f"Input {action} is not compatible with policy input space {self.get_input_space()}"
            )

        action = np.clip(action, self.low, self.high)
        self.action_info = {"action": action.tolist()}
        return action


class PolicyAdapter:
    """Skeleton adapter aligned with MetaDrive Policy-style action contracts.

    This adapter is the integration point for future policy-driven mappings where
    planner outputs are transformed into raw 2D controls expected by vehicles.
    """

    is_neural = False
    requires_training = False

    def __init__(
        self,
        low: float = -1.0,
        high: float = 1.0,
        clip: bool = True,
        expected_shape: tuple[int, ...] = (2,),
        policy_name: str = "EnvInputPolicy",
        action_check: bool = True,
    ) -> None:
        self.low = low
        self.high = high
        self.clip = clip
        self.expected_shape = expected_shape
        self.policy_name = policy_name

        if self.policy_name != "EnvInputPolicy":
            raise ValueError(
                f"Unsupported policy_name '{self.policy_name}'. "
                "Currently supported: EnvInputPolicy"
            )
        self.policy = EnvInputPolicyBridge(low=low, high=high, action_check=action_check)
        self.last_action_info: dict[str, Any] = {}

    def __call__(self, planner_output: Any) -> np.ndarray:
        action = self.policy.act(planner_output)
        if action.shape != self.expected_shape:
            raise ValueError(f"Expected action shape {self.expected_shape}, got {action.shape}")
        if self.clip:
            action = np.clip(action, self.low, self.high)
        self.last_action_info = dict(self.policy.action_info)
        return action

    def begin_training(self) -> None:
        return None

    def maybe_update(self) -> None:
        return None

    def end_training(self) -> None:
        return None

    def save(self, checkpoint_path: str) -> None:
        _ = checkpoint_path
        return None

    def load(self, checkpoint_path: str) -> None:
        _ = checkpoint_path
        return None
