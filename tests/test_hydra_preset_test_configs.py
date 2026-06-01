from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def test_adapter_groups_compose() -> None:
    cfg_identity = _compose("adapter=identity", "reward=monitor_only")
    cfg_neural = _compose("adapter=neural_adapter", "reward=monitor_only")
    cfg_policy = _compose("adapter=policy_adapter", "reward=monitor_only")

    assert cfg_identity.adapter.name == "identity"
    assert cfg_neural.adapter.name == "neural_adapter"
    assert cfg_policy.adapter.name == "policy_adapter"


def test_curriculum_groups_compose() -> None:
    cfg_disabled = _compose("curriculum=disabled", "reward=monitor_only")
    cfg_stages = _compose("curriculum=stages", "reward=monitor_only")

    assert bool(cfg_disabled.curriculum.enabled) is False
    assert bool(cfg_stages.curriculum.enabled) is True


def test_observation_groups_compose() -> None:
    cfg_lidar = _compose("obs=lidar_state", "reward=monitor_only")
    cfg_semantic = _compose("obs=semantic_state", "reward=monitor_only")

    assert str(cfg_lidar.obs.type) == "lidar_state"
    assert str(cfg_semantic.obs.type) == "semantic_state"


def test_encoder_and_planner_groups_compose() -> None:
    cfg_none_td3 = _compose("encoder=none", "planner=td3", "reward=monitor_only")
    cfg_mlp_sac = _compose("encoder=mlp", "planner=sac", "reward=monitor_only")
    cfg_lq_ppo = _compose("encoder=lq", "planner=ppo", "obs=semantic_state", "reward=monitor_only")

    assert str(cfg_none_td3.encoder.type) == "none"
    assert str(cfg_none_td3.planner.name) == "td3"
    assert str(cfg_mlp_sac.encoder.type) == "mlp"
    assert str(cfg_mlp_sac.planner.name) == "sac"
    assert str(cfg_lq_ppo.encoder.type) == "lq"
    assert str(cfg_lq_ppo.planner.name) == "ppo"
