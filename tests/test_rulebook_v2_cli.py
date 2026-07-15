from __future__ import annotations

import json

import pytest

from thesis_rl.cli.rulebook_v2_calibrate import _read_trials
from thesis_rl.cli.rulebook_v2_braking_trials import validate_calibration_config
from thesis_rl.cli.rulebook_v2_pilot import _percentile


def test_calibration_cli_reads_wrapped_trial_json(tmp_path):
    path = tmp_path / "trials.json"
    path.write_text(
        json.dumps(
            {
                "trials": [
                    {
                        "target_speed_mps": 5,
                        "reached_speed_mps": 5.1,
                        "collided": False,
                        "left_lane": False,
                        "mean_deceleration_mps2": 3.2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    trials = _read_trials(path)
    assert len(trials) == 1
    assert trials[0].reached_speed_mps == 5.1


def test_pilot_percentile_is_deterministic_lower_order_statistic():
    assert _percentile([3.0, 1.0, 2.0, 4.0], 0.95) == 4.0
    assert _percentile([], 0.95) == 0.0


def test_braking_runner_requires_straight_empty_track_config():
    config = {
        "map_config": {"type": "block_sequence", "config": "SSSS"},
        "traffic_density": 0.0,
        "random_agent_model": False,
        "num_agents": 1,
        "vehicle_config": {"vehicle_model": "default"},
    }
    validate_calibration_config(config)
    config["traffic_density"] = 0.1
    with pytest.raises(ValueError, match="traffic_density=0"):
        validate_calibration_config(config)
