from __future__ import annotations

from thesis_rl.curriculum import ScenarioAclScenarioEnvConfig, scenario_env_runtime_config


def test_scenario_env_runtime_config_filters_unsupported_keys() -> None:
    runtime_cfg = scenario_env_runtime_config(ScenarioAclScenarioEnvConfig())

    assert "horizon" in runtime_cfg
    assert "out_of_route_done" in runtime_cfg
    assert "crash_vehicle_done" in runtime_cfg
    assert "reactive_traffic" in runtime_cfg
    assert "on_continuous_line_done" not in runtime_cfg
    assert "on_broken_line_done" not in runtime_cfg
    assert "out_of_road_done" not in runtime_cfg
