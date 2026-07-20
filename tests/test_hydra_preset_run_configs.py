from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from thesis_rl.runtime.wiring.builders import _resolve_planner_cfg


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def test_base_config_composes_with_monitor_only() -> None:
    cfg = _compose("env=metadrive", "reward=monitor_only", "curriculum=disabled")

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
    cfg_scenario_acl = _compose("reward=monitor_only", "curriculum=scenario_acl")

    assert cfg_native.reward.name == "native"
    assert str(cfg_native.reward.type) == "native"
    assert str(cfg_native.reward.behavior) == "off"
    assert str(cfg_native.reward.rulebook_config) == "none"
    assert bool(cfg_native.curriculum.enabled) is True
    assert str(cfg_native.curriculum.staged.mode) == "auto"
    assert len(cfg_native.curriculum.staged.stages) >= 1

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
    assert cfg_scenario_acl.reward.name == "monitor_only"
    assert str(cfg_scenario_acl.curriculum.kind) == "scenario_acl"
    assert bool(cfg_scenario_acl.curriculum.scenario_acl.use_replay) is True


def test_scenarionet_staged_curriculum_composes_with_semantic_arms() -> None:
    cfg = _compose("env=scenarionet", "curriculum=stages_scenarionet")

    names = [str(stage.name) for stage in cfg.curriculum.staged.stages]
    assert names == [
        "A0_simple_low_traffic",
        "A1_traffic",
        "A2_junction",
        "A3_complex_junction",
        "A4_vru",
        "A5_critical_mixed",
    ]
    assert str(cfg.curriculum.staged.stages[3].env.provider.arm) == "A3_complex_junction"
    assert float(cfg.curriculum.staged.stages[4].env.provider.source_probability.pg) == 0.0
    assert all("start_seed" not in stage.env for stage in cfg.curriculum.staged.stages)
    assert all("num_scenarios" not in stage.env for stage in cfg.curriculum.staged.stages)


def test_scenarionet_acl_composes_with_six_semantic_arms() -> None:
    cfg = _compose("env=scenarionet", "curriculum=scenario_acl_scenarionet")

    assert int(cfg.curriculum.scenario_acl.mab.num_arms) == 6
    assert bool(cfg.curriculum.scenario_acl.use_scenario_buffer) is True
    assert bool(cfg.curriculum.scenario_acl.use_replay) is True
    assert str(cfg.agent.planner.algorithm.name) == "td3_sb3"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.encoder.name) == "latent_query_v2"


def test_run_profile_medium_overrides_experiment_budget() -> None:
    cfg = _compose("run_profile=medium")

    assert cfg.run_profile.name == "medium"
    assert cfg.experiment.name == "medium"
    assert int(cfg.experiment.total_timesteps) == 350000
    assert int(cfg.experiment.eval_episodes) == 50


def test_td3_run_profiles_use_budget_appropriate_warmup_and_batch() -> None:
    expected = {
        "default": (600000, 10000, 256),
        "smoke": (2000, 100, 64),
        "fast": (120000, 5000, 256),
        "medium": (350000, 10000, 256),
        "long": (700000, 10000, 256),
        "tune": (500000, 10000, 256),
        "thesis": (1500000, 10000, 256),
    }

    for profile, (timesteps, learning_starts, batch_size) in expected.items():
        cfg = _compose(
            f"run_profile={profile}",
            "agent/planner/algorithm=td3_sb3",
        )
        resolved_planner_cfg = _resolve_planner_cfg(cfg)
        assert int(cfg.experiment.total_timesteps) == timesteps
        assert int(resolved_planner_cfg.learning_starts) == learning_starts
        assert int(resolved_planner_cfg.batch_size) == batch_size


def test_sb3_sac_run_profiles_use_budget_appropriate_warmup_and_batch() -> None:
    expected = {
        "default": (600000, 10000, 256),
        "smoke": (2000, 100, 64),
        "fast": (120000, 1000, 256),
        "medium": (350000, 5000, 256),
        "long": (700000, 10000, 256),
        "tune": (500000, 5000, 256),
        "thesis": (1500000, 10000, 256),
    }

    for profile, (timesteps, learning_starts, batch_size) in expected.items():
        cfg = _compose(
            f"run_profile={profile}",
            "agent/planner/algorithm=sac_sb3",
        )
        resolved_planner_cfg = _resolve_planner_cfg(cfg)
        assert int(cfg.experiment.total_timesteps) == timesteps
        assert int(resolved_planner_cfg.learning_starts) == learning_starts
        assert int(resolved_planner_cfg.batch_size) == batch_size


def test_sb3_ppo_smoke_uses_diagnostic_rollout_override() -> None:
    cfg = _compose("run_profile=smoke", "agent/planner/algorithm=ppo_sb3")
    resolved_planner_cfg = _resolve_planner_cfg(cfg)

    assert int(resolved_planner_cfg.n_steps) == 16
    assert int(resolved_planner_cfg.batch_size) == 8


def test_rule_margin_logging_is_enabled_only_for_smoke_by_default() -> None:
    profiles = ("default", "smoke", "fast", "medium", "long", "tune", "thesis")

    for profile in profiles:
        cfg = _compose(f"run_profile={profile}")
        path = cfg.reward.rule_margin_log_path
        if profile == "smoke":
            assert str(path).endswith("/logs/rule_margins.jsonl")
        else:
            assert path is None


def test_trajectory_logging_is_enabled_only_for_smoke_by_default() -> None:
    profiles = ("default", "smoke", "fast", "medium", "long", "tune", "thesis")

    for profile in profiles:
        cfg = _compose(f"run_profile={profile}")
        assert bool(cfg.video.save_trajectory_log) is (profile == "smoke")


def test_replay_persistence_is_enabled_only_for_smoke_by_default() -> None:
    profiles = ("default", "smoke", "fast", "medium", "long", "tune", "thesis")

    for algorithm in ("td3_sb3", "sac_sb3"):
        for profile in profiles:
            cfg = _compose(
                f"run_profile={profile}",
                f"agent/planner/algorithm={algorithm}",
            )
            assert bool(cfg.agent.planner.algorithm.transition_replay.persistence.enabled) is (
                profile == "smoke"
            )


def test_replay_persistence_can_be_enabled_explicitly_outside_smoke() -> None:
    cfg = _compose(
        "run_profile=medium",
        "agent/planner/algorithm=td3_sb3",
        "agent.planner.algorithm.transition_replay.persistence.enabled=true",
    )

    assert bool(cfg.agent.planner.algorithm.transition_replay.persistence.enabled) is True


def test_explicit_td3_diagnostic_preset_keeps_margin_logging_enabled() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="presets/td3/td3_scalar_reward_scale_tuning_no_curr")

    assert str(cfg.reward.rule_margin_log_path).endswith("/logs/rule_margins.jsonl")


def test_run_profile_tune_overrides_experiment_budget() -> None:
    cfg = _compose("run_profile=tune")

    assert cfg.run_profile.name == "tune"
    assert cfg.experiment.name == "tune"
    assert int(cfg.experiment.total_timesteps) == 500000
    assert int(cfg.experiment.eval_interval) == 25000
    assert int(cfg.experiment.eval_episodes) == 20
    assert int(cfg.experiment.final_eval_episodes) == 50


def test_run_profile_planner_overrides_algorithm_defaults() -> None:
    cfg = _compose("run_profile=thesis", "agent/planner/algorithm=sac")

    assert int(cfg.agent.planner.algorithm.batch_size) == 128

    resolved_planner_cfg = _resolve_planner_cfg(cfg)
    assert int(resolved_planner_cfg.batch_size) == 128
    assert int(resolved_planner_cfg.learning_starts) == 20000
    assert str(resolved_planner_cfg.name) == "sac"


def test_final_scalar_pipeline_defaults_compose() -> None:
    cfg = _compose()

    assert str(cfg.env.name) == "scenarionet"
    assert str(cfg.obs.name) == "semantic_v2"
    assert str(cfg.agent.planner.encoder.name) == "latent_query_v2"
    assert str(cfg.agent.planner.encoder.architecture_version) == "1.0-final"
    assert str(cfg.agent.planner.decoder.name) == "mlp_encoded"
    assert str(cfg.agent.planner.algorithm.name) == "td3_sb3"
    assert str(cfg.reward.name) == "scalar_reward"
    assert str(cfg.reward.behavior) == "scalar_reward"
    assert str(cfg.scalarization.mode) == "bounded_satisfaction_rank"
    assert str(cfg.curriculum.name) == "scenario_acl_scenarionet"
    assert str(cfg.rulebook.version) == "4.7-final-implementation-complete"

    assert bool(cfg.env.provider.strict) is True
    assert bool(cfg.env.provider.allow_fallback) is False
    assert int(cfg.env.config.num_scenarios) == -1
    assert bool(cfg.env.vectorized.enabled) is False
    assert int(cfg.env.vectorized.num_envs) == 1

    assert int(cfg.agent.planner.algorithm.transition_replay.n_steps) == 3
    assert bool(cfg.agent.planner.algorithm.transition_replay.prioritized) is True
    assert bool(cfg.agent.planner.algorithm.transition_replay.persistence.enabled) is True
    assert (
        str(cfg.agent.planner.algorithm.transition_replay.persistence.trigger) == "final_or_manual"
    )
    assert bool(cfg.checkpoint.save_latest_each_chunk) is True
    assert bool(cfg.checkpoint.save_final) is True
    assert bool(cfg.checkpoint.save_rng_state) is True
    assert bool(cfg.checkpoint.resume.enabled) is False

    assert bool(cfg.video.enabled) is True
    assert int(cfg.video.max_final_videos) == 0
    assert str(cfg.reward.rule_margin_log_path).endswith("/logs/rule_margins.jsonl")
    assert bool(cfg.reward.include_violation_vector) is False
    assert bool(cfg.reward.runtime_info_debug_enabled) is False

    assert str(cfg.run_profile.name) == "smoke"
    assert str(cfg.experiment.name) == "run"
    resolved_planner_cfg = _resolve_planner_cfg(cfg)
    assert int(resolved_planner_cfg.learning_starts) == 100
    assert int(resolved_planner_cfg.batch_size) == 64
