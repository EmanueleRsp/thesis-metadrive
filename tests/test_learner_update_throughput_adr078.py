"""ADR-078 regression tests: SAC update-to-data ratio, batch 512 profiles, and
the switchable post-update learning-potential diagnostic (SAC and TD3)."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from thesis_rl.agent.planners.algorithms import Sb3SacPlannerBackend, Sb3Td3PlannerBackend
from thesis_rl.agent.types import Transition
from thesis_rl.runtime.wiring.builders import _resolve_planner_cfg

pytest.importorskip("stable_baselines3")
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


@pytest.fixture
def env():
    env_instance = gym.make("Pendulum-v1")
    try:
        yield env_instance
    finally:
        env_instance.close()


def _compose(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def _sac_planner(env: gym.Env, **overrides) -> Sb3SacPlannerBackend:
    cfg = {
        "policy": "MlpPolicy",
        "learning_starts": 0,
        "batch_size": 8,
        "buffer_size": 128,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "tau": 0.005,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "target_entropy": "auto",
        **overrides,
    }
    return Sb3SacPlannerBackend.build(
        env=env, cfg_planner=OmegaConf.create(cfg), device="cpu", seed=123
    )


def _td3_planner(env: gym.Env, **overrides) -> Sb3Td3PlannerBackend:
    cfg = {
        "policy": "MlpPolicy",
        "learning_starts": 0,
        "batch_size": 8,
        "buffer_size": 128,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "tau": 0.005,
        "action_noise_type": "normal",
        "action_noise_sigma": 0.1,
        "action_noise_mean": 0.0,
        **overrides,
    }
    return Sb3Td3PlannerBackend.build(
        env=env, cfg_planner=OmegaConf.create(cfg), device="cpu", seed=123
    )


class _SampleSpy:
    """Count replay-buffer samples drawn outside ``model.train``."""

    def __init__(self, planner) -> None:
        self.samples_outside_train = 0
        self._inside_train = False
        original_train = planner.model.train
        original_sample = planner.replay_buffer.sample

        def train(*args, **kwargs):
            self._inside_train = True
            try:
                return original_train(*args, **kwargs)
            finally:
                self._inside_train = False

        def sample(*args, **kwargs):
            if not self._inside_train:
                self.samples_outside_train += 1
            return original_sample(*args, **kwargs)

        planner.model.train = train
        planner.replay_buffer.sample = sample


def _collect_one_and_update(planner, env: gym.Env) -> dict:
    """Collect one transition through the lifecycle and return the update metrics."""

    lifecycle = planner.get_lifecycle()
    lifecycle.begin_training(chunk_timesteps=1, global_total_timesteps=1, global_steps_done=0)
    obs, _ = env.reset(seed=123)
    action = lifecycle.act(np.asarray(obs, dtype=np.float32), deterministic=False)
    next_obs, reward, terminated, truncated, info = env.step(action)
    lifecycle.observe_transition(
        Transition(
            observation=np.asarray(obs, dtype=np.float32),
            env_action=np.asarray(action, dtype=np.float32),
            buffer_action=lifecycle.to_buffer_action(np.asarray(action, dtype=np.float32)),
            scalar_reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            next_observation=np.asarray(next_obs, dtype=np.float32),
            terminal_observation=(
                np.asarray(next_obs, dtype=np.float32) if (terminated or truncated) else None
            ),
            info=dict(info),
        )
    )
    return planner.maybe_update(
        collected_steps=1,
        step_count=lifecycle.step_count + 1,
        global_total_timesteps=None,
        global_steps_done=0,
    )


# --- REQ-078-01: `gradient_steps: auto` honours `update_to_data_ratio` -------


@pytest.mark.parametrize("backend", [Sb3SacPlannerBackend, Sb3Td3PlannerBackend])
@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(None, 20), (1.0, 20), (0.5, 10), (0.25, 5), (0.01, 1)],
)
def test_auto_gradient_steps_scale_with_update_to_data_ratio(backend, ratio, expected) -> None:
    vec_env = DummyVecEnv([lambda: gym.make("Pendulum-v1") for _ in range(20)])
    try:
        cfg = {"gradient_steps": "auto", "train_freq": 1}
        if ratio is not None:
            cfg["update_to_data_ratio"] = ratio
        assert backend._resolve_gradient_steps(vec_env, cfg) == expected
    finally:
        vec_env.close()


def test_sac_explicit_gradient_steps_ignore_update_to_data_ratio() -> None:
    vec_env = DummyVecEnv([lambda: gym.make("Pendulum-v1") for _ in range(4)])
    try:
        cfg = {"gradient_steps": 3, "train_freq": 1, "update_to_data_ratio": 0.5}
        assert Sb3SacPlannerBackend._resolve_gradient_steps(vec_env, cfg) == 3
    finally:
        vec_env.close()


@pytest.mark.parametrize("backend", [Sb3SacPlannerBackend, Sb3Td3PlannerBackend])
@pytest.mark.parametrize("ratio", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_non_positive_update_to_data_ratio(backend, ratio) -> None:
    vec_env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    try:
        with pytest.raises(ValueError, match="update_to_data_ratio"):
            backend._resolve_gradient_steps(
                vec_env, {"gradient_steps": "auto", "update_to_data_ratio": ratio}
            )
    finally:
        vec_env.close()


# --- REQ-078-02: run profiles resolve the approved SAC learner setting -------


@pytest.mark.parametrize("algorithm", ["sac_sb3", "td3_sb3"])
@pytest.mark.parametrize("profile", ["fast", "default", "medium", "long", "tune", "thesis"])
def test_off_policy_run_profiles_resolve_batch_512_and_half_utd(
    algorithm: str, profile: str
) -> None:
    cfg = _compose(
        f"agent/planner/algorithm={algorithm}",
        "curriculum=disabled",
        f"run_profile={profile}",
    )
    planner_cfg = _resolve_planner_cfg(cfg)
    assert int(planner_cfg.batch_size) == 512
    assert str(planner_cfg.gradient_steps) == "auto"
    assert float(planner_cfg.update_to_data_ratio) == pytest.approx(0.5)
    assert bool(planner_cfg.update_learning_potential_diagnostic) is False


@pytest.mark.parametrize("algorithm", ["sac_sb3", "td3_sb3"])
def test_off_policy_smoke_profile_keeps_its_diagnostic_batch(algorithm: str) -> None:
    cfg = _compose(
        f"agent/planner/algorithm={algorithm}", "curriculum=disabled", "run_profile=smoke"
    )
    planner_cfg = _resolve_planner_cfg(cfg)
    assert int(planner_cfg.batch_size) == 64
    assert float(planner_cfg.update_to_data_ratio) == pytest.approx(0.5)


def test_ppo_run_profile_is_untouched_by_adr078() -> None:
    cfg = _compose("agent/planner/algorithm=ppo_sb3", "curriculum=disabled", "run_profile=thesis")
    planner_cfg = _resolve_planner_cfg(cfg)
    assert "update_to_data_ratio" not in planner_cfg
    assert "update_learning_potential_diagnostic" not in planner_cfg


# --- REQ-078-03: the post-update learning potential is switchable ------------


def test_sac_update_skips_learning_potential_batch_when_disabled(env) -> None:
    planner = _sac_planner(env, update_learning_potential_diagnostic=False)
    spy = _SampleSpy(planner)

    metrics = _collect_one_and_update(planner, env)

    assert int(metrics["update_calls"]) == 1
    assert metrics["learning_potential"] is None
    assert spy.samples_outside_train == 0


def test_sac_update_keeps_learning_potential_batch_by_default(env) -> None:
    planner = _sac_planner(env)
    spy = _SampleSpy(planner)

    metrics = _collect_one_and_update(planner, env)

    assert spy.samples_outside_train == 1
    assert metrics["learning_potential"] is not None
    assert np.isfinite(float(metrics["learning_potential"]))
    assert float(metrics["learning_potential"]) >= 0.0


def test_td3_update_skips_learning_potential_batch_when_disabled(env) -> None:
    planner = _td3_planner(env, update_learning_potential_diagnostic=False)
    spy = _SampleSpy(planner)

    metrics = _collect_one_and_update(planner, env)

    assert int(metrics["update_calls"]) == 1
    assert metrics["learning_potential"] is None
    assert metrics["timing_acl_replay_learning_potential_seconds"] == 0.0
    assert spy.samples_outside_train == 0


def test_td3_update_keeps_learning_potential_batch_by_default(env) -> None:
    planner = _td3_planner(env)
    spy = _SampleSpy(planner)

    metrics = _collect_one_and_update(planner, env)

    assert spy.samples_outside_train == 1
    assert metrics["learning_potential"] is not None
    assert np.isfinite(float(metrics["learning_potential"]))
