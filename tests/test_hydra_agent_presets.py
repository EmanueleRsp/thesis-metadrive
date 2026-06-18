from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose_preset(config_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name=config_name)


def test_td3_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/td3_sb3")
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "td3_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "td3_sb3"
    assert str(cfg.agent.planner.algorithm.policy) == "MlpPolicy"
    assert list(cfg.agent.planner.algorithm.policy_kwargs.net_arch) == [256, 256]


def test_td3_lq_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/td3_lq_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "lq"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.algorithm.name) == "td3_sb3"


def test_td3_mlp_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/td3_mlp_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "mlp"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.algorithm.name) == "td3_sb3"


def test_sac_sb3_algorithm_config_composes() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["agent/planner/algorithm=sac_sb3"])
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"
    assert str(cfg.agent.planner.algorithm.policy) == "MlpPolicy"
    assert list(cfg.agent.planner.algorithm.policy_kwargs.net_arch) == [256, 256]


def test_sac_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/sac_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "sac_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"


def test_sac_lq_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/sac_lq_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "lq"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"


def test_sac_mlp_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/sac_mlp_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "mlp"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"


def test_ppo_sb3_algorithm_config_composes() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["agent/planner/algorithm=ppo_sb3"])
    assert str(cfg.agent.planner.algorithm.name) == "ppo_sb3"
    assert str(cfg.agent.planner.algorithm.policy) == "MlpPolicy"
    assert int(cfg.agent.planner.algorithm.n_steps) == 2048
    assert int(cfg.agent.planner.algorithm.batch_size) == 64
    assert int(cfg.agent.planner.algorithm.n_epochs) == 10
    assert float(cfg.agent.planner.algorithm.ent_coef) == 0.0
    assert list(cfg.agent.planner.algorithm.policy_kwargs.net_arch.pi) == [64, 64]
    assert list(cfg.agent.planner.algorithm.policy_kwargs.net_arch.vf) == [64, 64]


def test_ppo_algorithm_config_composes_with_sb3_faithful_defaults() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["agent/planner/algorithm=ppo"])
    assert str(cfg.agent.planner.algorithm.name) == "ppo"
    assert int(cfg.agent.planner.algorithm.n_steps) == 2048
    assert int(cfg.agent.planner.algorithm.batch_size) == 64
    assert int(cfg.agent.planner.algorithm.n_epochs) == 10
    assert float(cfg.agent.planner.algorithm.ent_coef) == 0.0
    assert float(cfg.agent.planner.algorithm.optimizer_eps) == 1e-5


def test_ppo_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/ppo_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "ppo_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "ppo_sb3"


def test_ppo_lq_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/ppo_lq_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "lq"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.algorithm.name) == "ppo_sb3"


def test_ppo_mlp_sb3_preset_composes() -> None:
    cfg = _compose_preset("presets/agent/ppo_mlp_sb3")
    assert str(cfg.obs.type) == "semantic_state"
    assert str(cfg.agent.planner.encoder.type) == "mlp"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.algorithm.name) == "ppo_sb3"
