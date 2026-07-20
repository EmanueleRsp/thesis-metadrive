from __future__ import annotations

import yaml
from omegaconf import OmegaConf

from thesis_rl.runtime.io.metadata import save_run_metadata


def test_run_metadata_records_effective_algorithm_and_transition_replay(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "name": "metadata-test",
            "agent": {
                "planner": {
                    "algorithm": {
                        "name": "td3_sb3",
                        "transition_replay": {
                            "enabled": True,
                            "n_steps": 3,
                            "prioritized": True,
                            "persistence": {"enabled": True, "trigger": "final_or_manual"},
                        },
                    }
                }
            },
            "env": {"name": "metadrive"},
            "reward": {"type": "rulebook", "behavior": "scalar_reward"},
            "scalarization": {"specification_id": "SCAL-V1.0", "mode": "bounded_satisfaction_rank"},
            "curriculum": {"name": "scenario_acl_scenarionet"},
            "analysis": {"experiment_group": "metadata-test", "include_in_comparison": False},
            "seed": 42,
            "experiment": {"total_timesteps": 2000, "eval_interval": 1000, "eval_episodes": 2},
        }
    )

    path = save_run_metadata(cfg, tmp_path)
    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert metadata["algorithm"] == "td3_sb3"
    assert metadata["transition_replay"]["n_steps"] == 3
    assert metadata["transition_replay"]["prioritized"] is True
    assert metadata["transition_replay"]["persistence"]["enabled"] is True
