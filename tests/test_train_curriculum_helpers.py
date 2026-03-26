from __future__ import annotations

from omegaconf import OmegaConf

from thesis_rl.train import _merge_env_config_with_overrides, _missing_curriculum_metrics


def test_merge_env_config_with_overrides_updates_only_config_block() -> None:
    cfg_env = OmegaConf.create(
        {
            "name": "metadrive_base",
            "env_id": "MetaDriveEnv",
            "config": {
                "traffic_density": 0.0,
                "start_seed": 5,
                "num_scenarios": 1,
            },
            "observation": {"type": "lidar_state"},
        }
    )

    merged = _merge_env_config_with_overrides(
        cfg_env,
        {
            "traffic_density": 0.1,
            "start_seed": 100,
        },
    )

    assert float(merged.config.traffic_density) == 0.1
    assert int(merged.config.start_seed) == 100
    assert int(merged.config.num_scenarios) == 1
    assert str(merged.name) == "metadrive_base"
    assert str(merged.observation.type) == "lidar_state"


def test_missing_curriculum_metrics_lists_only_absent_keys() -> None:
    metrics = {
        "mean_reward": 1.0,
        "success_rate": 0.8,
        "collision_rate": 0.01,
        "out_of_road_rate": 0.01,
    }

    missing = _missing_curriculum_metrics(metrics)

    assert "top_rule_violation_rate" in missing
    assert "route_completion" in missing
    assert "mean_reward" not in missing
    assert "success_rate" not in missing
