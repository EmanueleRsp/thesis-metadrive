from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def test_scenarionet_env_config_composes_with_required_semantics() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(
            config_name="config",
            overrides=["env=scenarionet", "reward=monitor_only", "curriculum=disabled"],
        )

    assert str(cfg.env.env_id) == "ThesisScenarioEnv"
    assert cfg.env.config.horizon is None
    assert cfg.env.config.allowed_more_steps is None
    assert bool(cfg.env.config.reactive_traffic) is True
    assert bool(cfg.env.config.relax_out_of_road_done) is False
    assert bool(cfg.env.config.out_of_route_done) is False
    assert bool(cfg.env.config.truncate_as_terminate) is False
    assert int(cfg.env.episode_control.extra_steps_after_scenario) == 50
    assert float(cfg.env.provider.source_probability.waymo) == 0.5
    assert float(cfg.env.provider.source_probability.pg) == 0.5
