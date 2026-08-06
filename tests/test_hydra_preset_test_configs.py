from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def test_adapter_groups_compose() -> None:
    cfg_identity = _compose("agent/adapter=identity", "reward=monitor_only")

    assert cfg_identity.agent.adapter.name == "identity"


def test_curriculum_groups_compose() -> None:
    cfg_disabled = _compose("curriculum=disabled", "reward=monitor_only")
    cfg_stages = _compose("curriculum=stages", "reward=monitor_only")
    cfg_scenario_acl = _compose("curriculum=scenario_acl", "reward=monitor_only")

    assert bool(cfg_disabled.curriculum.enabled) is False
    assert bool(cfg_stages.curriculum.enabled) is True
    assert bool(cfg_scenario_acl.curriculum.enabled) is True
    assert str(cfg_scenario_acl.curriculum.kind) == "scenario_acl"
    assert bool(cfg_scenario_acl.curriculum.scenario_acl.use_replay) is True


def test_observation_groups_compose() -> None:
    cfg_lidar = _compose("obs=lidar_state", "reward=monitor_only")
    cfg_semantic = _compose("obs=semantic_state", "reward=monitor_only")
    cfg_semantic_v3 = _compose("obs=semantic_v3", "reward=monitor_only")

    assert str(cfg_lidar.obs.type) == "lidar_state"
    assert str(cfg_semantic.obs.type) == "semantic_state"
    assert str(cfg_semantic_v3.obs.type) == "semantic_v3"


def test_encoder_and_planner_groups_compose() -> None:
    cfg_none_td3 = _compose(
        "agent/planner/encoder=none", "agent/planner/algorithm=td3", "reward=monitor_only"
    )
    cfg_mlp_sac = _compose(
        "agent/planner/encoder=mlp", "agent/planner/algorithm=sac", "reward=monitor_only"
    )
    cfg_lq_ppo = _compose(
        "agent/planner/encoder=lq",
        "agent/planner/algorithm=ppo",
        "obs=semantic_state",
        "reward=monitor_only",
    )
    cfg_lq_v3_ppo = _compose(
        "agent/planner/encoder=lq_v3",
        "agent/planner/algorithm=ppo",
        "obs=semantic_v3",
        "reward=monitor_only",
    )

    assert str(cfg_none_td3.agent.planner.encoder.type) == "none"
    assert str(cfg_none_td3.agent.planner.algorithm.name) == "td3"
    assert str(cfg_mlp_sac.agent.planner.encoder.type) == "mlp"
    assert str(cfg_mlp_sac.agent.planner.algorithm.name) == "sac"
    assert str(cfg_lq_ppo.agent.planner.encoder.type) == "latent_query_v2"
    assert str(cfg_lq_ppo.agent.planner.algorithm.name) == "ppo"
    assert str(cfg_lq_v3_ppo.agent.planner.encoder.type) == "latent_query_v3"
    assert (
        str(cfg_lq_v3_ppo.agent.planner.encoder.required_observation_schema)
        == "1.2-perception-bounded-mission-route-v1"
    )
