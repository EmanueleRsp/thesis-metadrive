from __future__ import annotations

import logging

import pytest

from thesis_rl.agent.planners.backend_names import (
    _reset_legacy_backend_warning_cache,
    planner_backend_family,
    preferred_fork_backed_backend,
    warn_if_legacy_backend,
)
from thesis_rl.sb3_extensions import (
    build_algorithm_spec_from_planner_cfg,
    build_policy_spec_from_planner_cfg,
    build_sb3_specs_from_configs,
    uses_explicit_custom_sb3_policy,
)


def test_build_policy_spec_from_planner_cfg_normalizes_net_arch_shapes() -> None:
    policy_spec = build_policy_spec_from_planner_cfg(
        {
            "policy": "MlpPolicy",
            "policy_kwargs": {
                "net_arch": {
                    "pi": (64, 64),
                    "vf": [64, 64],
                }
            },
        }
    )

    assert policy_spec.policy == "MlpPolicy"
    assert policy_spec.policy_kwargs == {
        "net_arch": {
            "pi": [64, 64],
            "vf": [64, 64],
        }
    }


def test_build_algorithm_spec_from_planner_cfg_carries_custom_replay_buffer_hooks() -> None:
    class _DummyReplayBuffer:
        pass

    algorithm_spec = build_algorithm_spec_from_planner_cfg(
        {
            "replay_buffer_class": _DummyReplayBuffer,
            "replay_buffer_kwargs": {"alpha": 0.6},
            "algorithm_kwargs": {"optimize_memory_usage": True},
        }
    )

    assert algorithm_spec.replay_buffer_class is _DummyReplayBuffer
    assert algorithm_spec.replay_buffer_kwargs == {"alpha": 0.6}
    assert algorithm_spec.has_custom_replay_buffer is True
    assert algorithm_spec.merged_algorithm_kwargs(verbose=0) == {
        "optimize_memory_usage": True,
        "verbose": 0,
    }


def test_uses_explicit_custom_sb3_policy_detects_non_default_bridge_inputs() -> None:
    class _CustomPolicy:
        pass

    assert uses_explicit_custom_sb3_policy({"policy": _CustomPolicy}) is True
    assert uses_explicit_custom_sb3_policy(
        {"policy": "MlpPolicy", "policy_kwargs": {"features_extractor_class": object}}
    ) is True
    assert uses_explicit_custom_sb3_policy({"policy": "MlpPolicy"}) is False


def test_build_sb3_specs_from_configs_rejects_silently_ignored_thesis_encoder_path() -> None:
    with pytest.raises(ValueError, match="decoder dropout"):
        build_sb3_specs_from_configs(
            "td3_sb3",
            {"policy": "MlpPolicy", "policy_kwargs": {"net_arch": [256, 256]}},
            encoder_cfg={"type": "lq"},
            decoder_cfg={"name": "mlp_encoded", "type": "mlp", "hidden_layers": [256, 256], "dropout": 0.1},
        )


def test_build_sb3_specs_from_configs_allows_custom_policy_bridge_override() -> None:
    class _CustomPolicy:
        pass

    policy_spec, algorithm_spec = build_sb3_specs_from_configs(
        "td3_sb3",
        {"policy": _CustomPolicy, "policy_kwargs": {"net_arch": [256, 256]}},
        encoder_cfg={"type": "lq"},
        decoder_cfg={"name": "td3_sb3"},
    )

    assert policy_spec.policy is _CustomPolicy
    assert algorithm_spec.has_custom_replay_buffer is False


def test_build_sb3_specs_from_configs_builds_encoder_feature_extractor_bridge() -> None:
    pytest.importorskip("stable_baselines3")

    policy_spec, _algorithm_spec = build_sb3_specs_from_configs(
        "sac_sb3",
        {"policy": "MlpPolicy", "policy_kwargs": {}},
        encoder_cfg={"type": "lq", "output_dim": 256},
        decoder_cfg={"name": "mlp_encoded", "type": "mlp", "hidden_layers": [256, 256], "activation": "relu"},
        obs_cfg={"type": "semantic_state"},
    )

    assert policy_spec.policy == "MlpPolicy"
    assert policy_spec.policy_kwargs["net_arch"] == [256, 256]
    assert policy_spec.policy_kwargs["features_extractor_class"].__name__ == (
        "ThesisEncoderFeatureExtractor"
    )
    assert policy_spec.policy_kwargs["features_extractor_kwargs"]["cfg_encoder"]["type"] == "lq"


def test_build_sb3_specs_from_configs_builds_ppo_decoder_arch_for_both_heads() -> None:
    policy_spec, _algorithm_spec = build_sb3_specs_from_configs(
        "ppo_sb3",
        {"policy": "MlpPolicy", "policy_kwargs": {}},
        encoder_cfg={"type": "none"},
        decoder_cfg={"name": "mlp_encoded", "type": "mlp", "hidden_layers": [256, 256], "activation": "relu"},
        obs_cfg={"type": "semantic_state"},
    )

    assert policy_spec.policy_kwargs["net_arch"] == {
        "pi": [256, 256],
        "vf": [256, 256],
    }


def test_build_sb3_specs_from_configs_keeps_algorithm_policy_kwargs_for_same_name_baseline_decoder() -> None:
    policy_spec, _algorithm_spec = build_sb3_specs_from_configs(
        "td3_sb3",
        {"policy": "MlpPolicy", "policy_kwargs": {"net_arch": [256, 256]}},
        encoder_cfg={"type": "none"},
        decoder_cfg={"name": "td3_sb3", "type": "mlp", "hidden_layers": [400, 300], "activation": "relu"},
        obs_cfg={"type": "lidar_state"},
    )

    assert policy_spec.policy_kwargs == {"net_arch": [256, 256]}


def test_planner_backend_family_separates_legacy_and_sb3_fork_backends() -> None:
    assert planner_backend_family("td3") == "legacy"
    assert planner_backend_family("sac") == "legacy"
    assert planner_backend_family("ppo") == "legacy"
    assert planner_backend_family("td3_sb3") == "sb3_fork"
    assert planner_backend_family("sac_sb3") == "sb3_fork"
    assert planner_backend_family("ppo_sb3") == "sb3_fork"


def test_preferred_fork_backed_backend_maps_legacy_to_sb3_fork() -> None:
    assert preferred_fork_backed_backend("td3") == "td3_sb3"
    assert preferred_fork_backed_backend("sac") == "sac_sb3"
    assert preferred_fork_backed_backend("ppo") == "ppo_sb3"
    assert preferred_fork_backed_backend("td3_sb3") == "td3_sb3"


def test_warn_if_legacy_backend_logs_once_per_backend(caplog) -> None:
    _reset_legacy_backend_warning_cache()
    caplog.set_level(logging.WARNING, logger="thesis_rl.agent.planners.backend_names")

    warn_if_legacy_backend("ppo")
    warn_if_legacy_backend("ppo")

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "temporary legacy implementation" in messages[0]
    assert "ppo_sb3" in messages[0]
