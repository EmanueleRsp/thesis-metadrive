from __future__ import annotations

from omegaconf import OmegaConf

from thesis_rl.curriculum.config import CurriculumConfig
from thesis_rl.runtime.builders import merge_env_config_with_overrides
from thesis_rl.train import _missing_curriculum_metrics


def test_merge_env_config_with_overrides_updates_only_config_block() -> None:
    cfg_env = OmegaConf.create(
        {
            "name": "metadrive",
            "env_id": "MetaDriveEnv",
            "config": {
                "traffic_density": 0.0,
                "start_seed": 5,
                "num_scenarios": 1,
            },
            "observation": {"type": "lidar_state"},
        }
    )

    merged = merge_env_config_with_overrides(
        cfg_env,
        {
            "traffic_density": 0.1,
            "start_seed": 100,
        },
    )

    assert float(merged.config.traffic_density) == 0.1
    assert int(merged.config.start_seed) == 100
    assert int(merged.config.num_scenarios) == 1
    assert str(merged.name) == "metadrive"
    assert str(merged.observation.type) == "lidar_state"


def test_missing_curriculum_metrics_lists_only_absent_keys() -> None:
    metrics = {
        "success_rate": 0.8,
        "success_rate_std": 0.02,
        "collision_rate": 0.01,
        "collision_rate_std": 0.01,
        "out_of_road_rate": 0.01,
        "top_rule_violation_rate": 0.01,
        "route_completion": 0.9,
    }
    curriculum_cfg = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "mode": "auto",
            "promotion": {
                "gates": {
                    "safety": {
                        "collision_rate_max": 0.05,
                        "top_rule_violation_rate_max": 0.02,
                        "out_of_road_rate_max": 0.03,
                    },
                    "task": {
                        "success_rate_min": 0.8,
                        "route_completion_min": 0.85,
                    },
                    "stability": {
                        "success_rate_std_max": 0.1,
                        "collision_rate_std_max": 0.03,
                    },
                }
            },
        }
    )

    missing = _missing_curriculum_metrics(metrics, curriculum_cfg)
    assert missing == []
