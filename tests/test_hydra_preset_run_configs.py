from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def test_base_config_composes_with_scalar_default() -> None:
    cfg = _compose("reward=scalar_default", "curriculum=disabled")

    assert cfg.env.name == "metadrive"
    assert cfg.preprocessor.name == "identity"
    assert cfg.adapter.name == "identity"
    assert cfg.reward.name == "scalar_default"
    assert str(cfg.reward.mode) == "scalar_default"
    assert str(cfg.reward.rulebook) == "selection"
    assert float(cfg.reward.lambda_env) == 1.0
    assert float(cfg.reward.lambda_rule) == 0.0
    assert bool(cfg.curriculum.enabled) is False


def test_scalar_native_config_composes_without_rulebook_fields() -> None:
    cfg = _compose("reward=scalar_native", "curriculum=disabled")

    assert cfg.reward.name == "scalar_native"
    assert str(cfg.reward.mode) == "scalar_native"
    assert "rulebook" not in cfg.reward
    assert bool(cfg.curriculum.enabled) is False


def test_reward_variants_compose() -> None:
    cfg_scalar_native = _compose("reward=scalar_native", "curriculum=stages")
    cfg_scalar_default = _compose("reward=scalar_default", "curriculum=stages")
    cfg_scalar_rulebook = _compose("reward=scalar_rulebook", "curriculum=stages")

    assert cfg_scalar_native.reward.name == "scalar_native"
    assert str(cfg_scalar_native.reward.mode) == "scalar_native"
    assert bool(cfg_scalar_native.curriculum.enabled) is True
    assert str(cfg_scalar_native.curriculum.mode) == "auto"
    assert len(cfg_scalar_native.curriculum.stages) >= 1

    assert cfg_scalar_default.reward.name == "scalar_default"
    assert str(cfg_scalar_default.reward.mode) == "scalar_default"
    assert str(cfg_scalar_default.reward.rulebook) == "selection"
    assert float(cfg_scalar_default.reward.lambda_env) == 1.0
    assert float(cfg_scalar_default.reward.lambda_rule) == 0.0
    assert str(cfg_scalar_rulebook.reward.mode) == "scalar_rulebook"
    assert float(cfg_scalar_rulebook.reward.lambda_env) == 0.0
    assert float(cfg_scalar_rulebook.reward.lambda_rule) == 1.0


def test_run_profile_medium_overrides_experiment_budget() -> None:
    cfg = _compose("run_profile=medium")

    assert cfg.run_profile.name == "medium"
    assert cfg.experiment.name == "medium"
    assert int(cfg.experiment.total_timesteps) == 350000
    assert int(cfg.experiment.eval_episodes) == 50
