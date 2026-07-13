from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from thesis_rl.curriculum import ScenarioAclScenarioEnvConfig, scenario_env_runtime_config
from thesis_rl.curriculum.scenario_acl.record import ScenarioRecord
from thesis_rl.curriculum.scenario_acl.scenario_env import (
    _build_scenarionet_replay_config,
    build_scenario_replay_env,
)


def _record(dataset_directory: Path, *, scenario_index: int = 7) -> ScenarioRecord:
    return ScenarioRecord(
        scenario_id="waymo-replay-7",
        source="waymo",
        parent_id=None,
        scenario_description_path=str(dataset_directory / "scenario.pkl"),
        scenario_description_hash="hash",
        dataset_directory=str(dataset_directory),
        scenario_index=scenario_index,
        env_config={"split": "validation"},
        reset_seed=scenario_index,
        generator_arm=None,
        mutation_type=None,
        mutation_params=None,
        validation_status="valid",
        rule_criticality=0.0,
        learning_potential=0.0,
        usefulness=0.0,
        usefulness_norm=0.0,
        rank=0,
        num_seen=1,
        last_seen_step=1,
        num_children=0,
        metrics_summary={},
        scenario_arm="A1_traffic",
    )


def test_scenario_env_runtime_config_filters_unsupported_keys() -> None:
    runtime_cfg = scenario_env_runtime_config(ScenarioAclScenarioEnvConfig())

    assert "horizon" in runtime_cfg
    assert "out_of_route_done" in runtime_cfg
    assert "crash_vehicle_done" in runtime_cfg
    assert "reactive_traffic" in runtime_cfg
    assert "on_continuous_line_done" not in runtime_cfg
    assert "on_broken_line_done" not in runtime_cfg
    assert "out_of_road_done" not in runtime_cfg


def test_scenarionet_replay_uses_canonical_episode_contract(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "name": "scenarionet",
                "env_id": "ThesisScenarioEnv",
                "split": "validation",
                "config": {
                    "horizon": None,
                    "allowed_more_steps": None,
                    "truncate_as_terminate": False,
                    "relax_out_of_road_done": False,
                    "out_of_route_done": False,
                    "crash_vehicle_done": True,
                    "crash_object_done": True,
                    "crash_human_done": True,
                    "reactive_traffic": True,
                    "agent_policy": "env_input_policy",
                },
                "episode_control": {"extra_steps_after_scenario": 0},
            }
        }
    )
    record = _record(tmp_path)

    replay_config = _build_scenarionet_replay_config(
        cfg,
        record=record,
        scenario_env_cfg=ScenarioAclScenarioEnvConfig(horizon=1000),
    )

    assert replay_config["horizon"] is None
    assert replay_config["allowed_more_steps"] is None
    assert replay_config["truncate_as_terminate"] is False
    assert replay_config["extra_steps_after_scenario"] == 0
    assert replay_config["start_scenario_index"] == 7
    assert replay_config["num_scenarios"] == 1
    assert replay_config["data_directory"] == str(tmp_path)


def test_scenarionet_acl_replay_constructs_thesis_env(monkeypatch, tmp_path: Path) -> None:
    import thesis_rl.curriculum.scenario_acl.scenario_env as scenario_env_module

    cfg = OmegaConf.create(
        {
            "env": {
                "name": "scenarionet",
                "env_id": "ThesisScenarioEnv",
                "split": "validation",
                "config": {
                    "horizon": None,
                    "allowed_more_steps": None,
                    "truncate_as_terminate": False,
                    "reactive_traffic": True,
                    "agent_policy": "env_input_policy",
                },
                "episode_control": {"extra_steps_after_scenario": 50},
            }
        }
    )
    record = _record(tmp_path)
    captured: dict[str, object] = {}

    class FakeThesisScenarioEnv:
        def __init__(self, config, *, split):
            captured["config"] = config
            captured["split"] = split
            self.current_scenario_record = None

    monkeypatch.setattr(
        "thesis_rl.envs.thesis_scenario_env.ThesisScenarioEnv",
        FakeThesisScenarioEnv,
    )
    monkeypatch.setattr(
        scenario_env_module,
        "maybe_wrap_env_with_reward_manager",
        lambda env, _cfg: env,
    )

    env = build_scenario_replay_env(
        cfg,
        record=record,
        scenario_env_cfg=ScenarioAclScenarioEnvConfig(),
    )

    assert isinstance(env, FakeThesisScenarioEnv)
    assert captured["split"] == "validation"
    assert captured["config"]["horizon"] is None  # type: ignore[index]
    assert captured["config"]["extra_steps_after_scenario"] == 50  # type: ignore[index]
    assert env.current_scenario_record is record
