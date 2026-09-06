"""Logged-ego replay with an optional mid-episode stop.

Diagnostic only. The policy drives the ego along the logged SDC trajectory the
way MetaDrive's ``ReplayEgoCarPolicy`` does and, when
``replay_ego_stop_fraction`` is below one, freezes the ego at the last replayed
pose for the remainder of the episode. This yields reference behaviours for
reward-ordering checks without any learned component:

- ``replay_ego_stop_fraction = 1.0``: full legal completion (the logged expert);
- ``0 < replay_ego_stop_fraction < 1``: partial legal progress followed by a
  standstill.

A standstill from the first step is obtained separately by holding the brake
under ``env_input_policy``. The ``env.step`` action is ignored here, as with
every MetaDrive replay policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

REPLAY_EGO_STOP_FRACTION_KEY = "replay_ego_stop_fraction"


class ReplayEgoStopPolicy(ReplayEgoCarPolicy):
    """Replay the logged ego and optionally hold its pose after a step fraction."""

    def __init__(self, control_object: Any, track: Any = None, random_seed: Any = None) -> None:
        super().__init__(control_object, track, random_seed)
        fraction = float(self.engine.global_config.get(REPLAY_EGO_STOP_FRACTION_KEY, 1.0))
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"{REPLAY_EGO_STOP_FRACTION_KEY} must lie in (0, 1], got {fraction}.")
        self.stop_fraction = fraction
        valid_steps = sum(
            1 for state in self.traj_info if state is not None and bool(state["valid"])
        )
        self.stop_index: int | None = (
            int(np.ceil(fraction * valid_steps)) if fraction < 1.0 else None
        )
        self._frozen_state: dict[str, Any] | None = None

    def _last_valid_state_before(self, index: int) -> dict[str, Any] | None:
        for candidate in range(min(index, len(self.traj_info) - 1), -1, -1):
            state = self.traj_info[candidate]
            if state is not None and bool(state["valid"]):
                return state
        return None

    def act(self, *args: Any, **kwargs: Any):
        index = max(int(self.episode_step), 0)
        if self.stop_index is None or index < self.stop_index:
            return super().act(*args, **kwargs)
        if self._frozen_state is None:
            self._frozen_state = self._last_valid_state_before(self.stop_index)
        state = self._frozen_state
        if state is None:
            return None
        if hasattr(self.control_object, "set_throttle_brake"):
            self.control_object.set_throttle_brake(-1.0)
        if hasattr(self.control_object, "set_steering"):
            self.control_object.set_steering(0.0)
        self.control_object.set_position(state["position"])
        self.control_object.set_velocity(np.zeros(2, dtype=np.float32), in_local_frame=False)
        self.control_object.set_heading_theta(state["heading"])
        self.control_object.set_angular_velocity(0.0)
        if self.engine.global_config.get("set_static", False):
            self.control_object.set_static(True)
        return None
