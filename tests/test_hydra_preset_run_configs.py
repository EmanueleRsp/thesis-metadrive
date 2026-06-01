from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def test_base_config_composes_with_monitor_only() -> None:
    cfg = _compose("reward=monitor_only", "curriculum=disabled")

    assert cfg.env.name == "metadrive"
    assert cfg.agent.preprocessor.name == "identity"
    assert cfg.agent.adapter.name == "identity"
    assert cfg.reward.name == "monitor_only"
    assert str(cfg.reward.type) == "native"
    assert str(cfg.reward.behavior) == "monitor_only"
    assert str(cfg.reward.rulebook_config) == "selection"
    assert float(cfg.reward.lambda_env) == 1.0
    assert float(cfg.reward.lambda_rule) == 0.0
    assert bool(cfg.curriculum.enabled) is False


def test_native_config_composes_with_off_behavior() -> None:
    cfg = _compose("reward=native", "curriculum=disabled")

    assert cfg.reward.name == "native"
    assert str(cfg.reward.type) == "native"
    assert str(cfg.reward.behavior) == "off"
    assert str(cfg.reward.rulebook_config) == "none"
    assert bool(cfg.curriculum.enabled) is False


def test_reward_variants_compose() -> None:
    cfg_native = _compose("reward=native", "curriculum=stages")
    cfg_monitor_only = _compose("reward=monitor_only", "curriculum=stages")
    cfg_scalar_reward = _compose("reward=scalar_reward", "curriculum=stages")

    assert cfg_native.reward.name == "native"
    assert str(cfg_native.reward.type) == "native"
    assert str(cfg_native.reward.behavior) == "off"
    assert str(cfg_native.reward.rulebook_config) == "none"
    assert bool(cfg_native.curriculum.enabled) is True
    assert str(cfg_native.curriculum.mode) == "auto"
    assert len(cfg_native.curriculum.stages) >= 1

    assert cfg_monitor_only.reward.name == "monitor_only"
    assert str(cfg_monitor_only.reward.type) == "native"
    assert str(cfg_monitor_only.reward.behavior) == "monitor_only"
    assert str(cfg_monitor_only.reward.rulebook_config) == "selection"
    assert float(cfg_monitor_only.reward.lambda_env) == 1.0
    assert float(cfg_monitor_only.reward.lambda_rule) == 0.0
    assert cfg_scalar_reward.reward.name == "scalar_reward"
    assert str(cfg_scalar_reward.reward.type) == "rulebook"
    assert str(cfg_scalar_reward.reward.behavior) == "scalar_reward"
    assert str(cfg_scalar_reward.reward.rulebook_config) == "selection"
    assert float(cfg_scalar_reward.reward.lambda_env) == 0.0
    assert float(cfg_scalar_reward.reward.lambda_rule) == 1.0


def test_run_profile_medium_overrides_experiment_budget() -> None:
    cfg = _compose("run_profile=medium")

    assert cfg.run_profile.name == "medium"
    assert cfg.experiment.name == "medium"
    assert int(cfg.experiment.total_timesteps) == 350000
    assert int(cfg.experiment.eval_episodes) == 50
