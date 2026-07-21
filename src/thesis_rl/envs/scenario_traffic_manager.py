"""Scenario traffic lifecycle constrained by source-track support."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from metadrive.manager.scenario_traffic_manager import ScenarioTrafficManager
from metadrive.policy.idm_policy import TrajectoryIDMPolicy


def source_track_is_valid_at_step(track: Mapping[str, Any] | None, step: int) -> bool:
    """Return whether a source track has a valid state at one simulation step."""

    if track is None or step < 0:
        return False
    state = track.get("state")
    if not isinstance(state, Mapping):
        return False
    valid = state.get("valid")
    if valid is None:
        return False
    try:
        if step >= len(valid):
            return False
    except TypeError:
        return False
    return bool(valid[step])


class SourceBoundScenarioTrafficManager(ScenarioTrafficManager):
    """Keep reactive vehicles only while their ScenarioNet source state exists."""

    def before_step(self, *args: Any, **kwargs: Any) -> None:
        """Act valid IDM vehicles and queue source-expired ones for cleanup."""

        self._obj_to_clean_this_frame = []
        for vehicle in self.spawned_objects.values():
            if not self.engine.has_policy(vehicle.id, TrajectoryIDMPolicy):
                continue

            scenario_id = self._obj_id_to_scenario_id[vehicle.id]
            track = self.current_traffic_data.get(scenario_id)
            if not source_track_is_valid_at_step(track, self.episode_step):
                self._obj_to_clean_this_frame.append(scenario_id)
                continue

            policy = self.engine.get_policy(vehicle.name)
            if policy.arrive_destination:
                self._obj_to_clean_this_frame.append(scenario_id)
                continue

            do_speed_control = self.episode_step % self.IDM_ACT_BATCH_SIZE == policy.policy_index
            vehicle.before_step(policy.act(do_speed_control))
