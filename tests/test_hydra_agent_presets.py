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


def test_selection_td3_sb3_qual_lidar_thesis_preset_composes() -> None:
    cfg = _compose_preset("presets/selection/td3_sb3_qual_lidar_thesis")
    assert str(cfg.run_profile.name) == "thesis"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.env.name) == "metadrive_native_strict"
    assert bool(cfg.env.config.out_of_road_done) is True
    assert bool(cfg.env.config.on_continuous_line_done) is True
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "td3_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "td3_sb3"
    assert int(cfg.env.vectorized.num_envs) == 5
    assert int(cfg.experiment.eval_interval) == 50000
    assert int(cfg.experiment.eval_episodes) == 20
    assert int(cfg.experiment.final_eval_episodes) == 100


def test_selection_sac_sb3_qual_lidar_thesis_preset_composes() -> None:
    cfg = _compose_preset("presets/selection/sac_sb3_qual_lidar_thesis")
    assert str(cfg.run_profile.name) == "thesis"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.env.name) == "metadrive_native_strict"
    assert bool(cfg.env.config.out_of_road_done) is True
    assert bool(cfg.env.config.on_continuous_line_done) is True
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "sac_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"
    assert int(cfg.env.vectorized.num_envs) == 5
    assert int(cfg.experiment.eval_interval) == 50000
    assert int(cfg.experiment.eval_episodes) == 20
    assert int(cfg.experiment.final_eval_episodes) == 100


def test_selection_ppo_sb3_qual_lidar_thesis_preset_composes() -> None:
    cfg = _compose_preset("presets/selection/ppo_sb3_qual_lidar_thesis")
    assert str(cfg.run_profile.name) == "thesis"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.env.name) == "metadrive_native_strict"
    assert bool(cfg.env.config.out_of_road_done) is True
    assert bool(cfg.env.config.on_continuous_line_done) is True
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "ppo_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "ppo_sb3"
    assert int(cfg.env.vectorized.num_envs) == 5
    assert int(cfg.experiment.eval_interval) == 50000
    assert int(cfg.experiment.eval_episodes) == 20
    assert int(cfg.experiment.final_eval_episodes) == 100


def test_selection_sac_sb3_native_contract_strict_fast_preset_composes() -> None:
    cfg = _compose_preset("presets/selection/sac_sb3_native_contract_strict_fast")
    assert str(cfg.run_profile.name) == "fast"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "sac_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"
    assert str(cfg.env.name) == "metadrive_native_strict"
    assert bool(cfg.env.config.out_of_road_done) is True
    assert bool(cfg.env.config.on_continuous_line_done) is True
    assert bool(cfg.env.config.on_broken_line_done) is False
    assert int(cfg.env.vectorized.num_envs) == 5


def test_selection_sac_sb3_native_contract_relaxed_fast_preset_composes() -> None:
    cfg = _compose_preset("presets/selection/sac_sb3_native_contract_relaxed_fast")
    assert str(cfg.run_profile.name) == "fast"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.encoder.type) == "none"
    assert str(cfg.agent.planner.decoder.name) == "sac_sb3"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"
    assert str(cfg.env.name) == "metadrive_native_relaxed"
    assert bool(cfg.env.config.out_of_road_done) is True
    assert bool(cfg.env.config.on_continuous_line_done) is False
    assert bool(cfg.env.config.on_broken_line_done) is False
    assert int(cfg.env.vectorized.num_envs) == 5


def test_selection_sac_sb3_native_contract_strict_fast_env4_preset_composes() -> None:
    cfg = _compose_preset("presets/selection/sac_sb3_native_contract_strict_fast_env4")
    assert str(cfg.run_profile.name) == "fast"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.obs.type) == "lidar_state"
    assert str(cfg.agent.planner.algorithm.name) == "sac_sb3"
    assert str(cfg.env.name) == "metadrive_native_strict"
    assert bool(cfg.env.config.out_of_road_done) is True
    assert bool(cfg.env.config.on_continuous_line_done) is True
    assert int(cfg.env.vectorized.num_envs) == 4


def test_smoke_train_preset_composes() -> None:
    cfg = _compose_preset("presets/test/smoke_train")
    assert str(cfg.run_profile.name) == "smoke"
    assert str(cfg.experiment.name) == "smoke"
    assert str(cfg.reward.name) == "monitor_only"
    assert str(cfg.curriculum.name) == "disabled"
    assert str(cfg.agent.planner.algorithm.name) == "td3_sb3"
