from __future__ import annotations

import pytest

from thesis_rl.envs.factory import _configure_agent_observation


def test_semantic_observation_rejects_future_trajectory_flag() -> None:
    with pytest.raises(ValueError, match="future time-indexed"):
        _configure_agent_observation(
            {},
            {
                "type": "semantic_state",
                "expose_time_indexed_future_trajectory": True,
            },
        )


def test_semantic_observation_rejects_future_signal_phase_flag() -> None:
    with pytest.raises(ValueError, match="future signal"):
        _configure_agent_observation(
            {},
            {"type": "semantic_state", "expose_future_signal_phase": True},
        )


def test_stacked_lidar_observation_freezes_sensor_noise_and_dimensions() -> None:
    config: dict = {}

    _configure_agent_observation(config, {"type": "stacked_lidar_state"})

    assert config["agent_observation"].__name__ == "StackedLidarStateObservation"
    assert config["vehicle_config"]["lidar"] == {
        "num_lasers": 240,
        "distance": 50.0,
        "num_others": 4,
        "add_others_navi": False,
        "gaussian_noise": 0.0,
        "dropout_prob": 0.0,
    }
    assert config["vehicle_config"]["side_detector"]["num_lasers"] == 12
    assert config["vehicle_config"]["lane_line_detector"]["num_lasers"] == 12
