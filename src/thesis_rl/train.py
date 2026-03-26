from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from thesis_rl.adapters.base import BaseAdapter
from thesis_rl.adapters.direct_action import DirectActionAdapter
from thesis_rl.adapters.neural_adapter import NeuralAdapter
from thesis_rl.adapters.policy_adapter import PolicyAdapter
from thesis_rl.agents.agent import Agent
from thesis_rl.agents.planner_agent import Td3PlannerBackend
from thesis_rl.curriculum.config import CurriculumConfig
from thesis_rl.curriculum.manager import CurriculumManager
from thesis_rl.envs.factory import make_env
from thesis_rl.envs.wrappers import RuleRewardWrapper
from thesis_rl.preprocessors.identity import IdentityPreprocessor
from thesis_rl.reward.reward_manager import HybridRulebookRewardManager


def _build_preprocessor(cfg: DictConfig) -> IdentityPreprocessor:
    if cfg.preprocessor.name != "identity":
        raise ValueError(f"Unsupported preprocessor: {cfg.preprocessor.name}")
    return IdentityPreprocessor(cast_to_float32=bool(cfg.preprocessor.cast_to_float32))


def _build_adapter(cfg: DictConfig) -> BaseAdapter:
    name = str(cfg.adapter.name)
    common_kwargs = {
        "low": float(cfg.adapter.low),
        "high": float(cfg.adapter.high),
        "clip": bool(cfg.adapter.clip),
        "expected_shape": tuple(cfg.adapter.expected_shape),
    }

    if name == "direct_action":
        return DirectActionAdapter(**common_kwargs)

    if name == "neural_adapter":
        return NeuralAdapter(
            **common_kwargs,
            hidden_dim=int(cfg.adapter.get("hidden_dim", 64)),
            learning_rate=float(cfg.adapter.get("learning_rate", 1e-3)),
            batch_size=int(cfg.adapter.get("batch_size", 64)),
            update_interval=int(cfg.adapter.get("update_interval", 1)),
            buffer_capacity=int(cfg.adapter.get("buffer_capacity", 10000)),
            device=str(cfg.device),
        )

    if name == "policy_adapter":
        return PolicyAdapter(
            **common_kwargs,
            policy_name=str(cfg.adapter.get("policy_name", "EnvInputPolicy")),
            action_check=bool(cfg.adapter.get("action_check", True)),
        )

    raise ValueError(f"Unsupported adapter: {name}")


def _maybe_wrap_env_with_reward_manager(env, cfg: DictConfig):
    if str(cfg.reward.mode) != "hybrid":
        return env

    if cfg.get("rulebook") is None:
        raise ValueError("reward.mode=hybrid requires a rulebook config")

    manager = HybridRulebookRewardManager.from_configs(
        cfg_reward=cfg.reward,
        cfg_rulebook=cfg.rulebook,
    )
    return RuleRewardWrapper(
        env=env,
        reward_manager=manager,
        attach_info=bool(cfg.reward.get("attach_info", True)),
    )


def _merge_env_config_with_overrides(cfg_env: DictConfig, env_overrides: dict[str, Any]) -> DictConfig:
    merged_cfg_env = OmegaConf.create(OmegaConf.to_container(cfg_env, resolve=True))
    merged_cfg_env.config = OmegaConf.merge(merged_cfg_env.config, dict(env_overrides))
    return merged_cfg_env


def _build_env(cfg: DictConfig, env_overrides: dict[str, Any] | None = None):
    cfg_env = cfg.env
    if env_overrides:
        cfg_env = _merge_env_config_with_overrides(cfg.env, env_overrides)

    env = make_env(cfg_env)
    return _maybe_wrap_env_with_reward_manager(env, cfg)


def _missing_curriculum_metrics(metrics: dict[str, float]) -> list[str]:
    required = (
        "collision_rate",
        "top_rule_violation_rate",
        "out_of_road_rate",
        "success_rate",
        "route_completion",
        "mean_reward",
    )
    return [name for name in required if name not in metrics]


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== TRAIN CONFIG ===")
    print(OmegaConf.to_yaml(cfg))

    preprocessor = _build_preprocessor(cfg)
    adapter = _build_adapter(cfg)
    curriculum_cfg = CurriculumConfig.from_experiment_cfg(cfg.experiment)

    curriculum_manager: CurriculumManager | None = None
    current_train_overrides: dict[str, Any] | None = None
    if curriculum_cfg.enabled and curriculum_cfg.stages:
        curriculum_manager = CurriculumManager(curriculum_cfg)
        current_train_overrides = curriculum_manager.get_env_config(evaluation=False)

    env = _build_env(cfg, current_train_overrides)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    planner = Td3PlannerBackend.build(
        env=env,
        cfg_agent=cfg.agent,
        cfg_planner=cfg.planner,
        device=str(cfg.device),
    )
    agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter)

    total_timesteps = int(cfg.experiment.total_timesteps)
    log_interval = int(cfg.experiment.get("log_interval", 1000))
    eval_interval = int(cfg.experiment.get("eval_interval", total_timesteps))
    if eval_interval <= 0:
        eval_interval = total_timesteps

    remaining = total_timesteps
    while remaining > 0:
        chunk_steps = min(eval_interval, remaining)
        current_stage_name = "baseline"
        if curriculum_manager is not None:
            current_stage_name = curriculum_manager.get_current_stage().name

        print(
            f"Training chunk on stage '{current_stage_name}': "
            f"steps={chunk_steps}, remaining_after={remaining - chunk_steps}"
        )
        agent.train(
            env=env,
            total_timesteps=chunk_steps,
            deterministic=False,
            log_interval=log_interval,
        )
        remaining -= chunk_steps

        if curriculum_manager is not None:
            curriculum_manager.record_train_steps(chunk_steps)

        # MetaDrive uses a global engine singleton: close training env before creating eval env.
        env.close()
        eval_env_overrides = None
        if curriculum_manager is not None:
            eval_env_overrides = curriculum_manager.get_env_config(evaluation=True)
        eval_env = _build_env(cfg, eval_env_overrides)
        planner.set_env(eval_env)
        metrics = agent.evaluate(
            env=eval_env,
            n_eval_episodes=int(cfg.experiment.eval_episodes),
            deterministic=bool(cfg.experiment.eval_deterministic),
        )
        eval_env.close()

        print(f"Evaluation metrics after chunk: {metrics}")

        if curriculum_manager is None:
            env = _build_env(cfg, current_train_overrides)
            planner.set_env(env)
            continue

        if curriculum_cfg.mode.lower() == "auto":
            missing_metrics = _missing_curriculum_metrics(metrics)
            if missing_metrics:
                raise ValueError(
                    "Curriculum auto mode requires metrics: "
                    f"{missing_metrics}. "
                    "Implement metric extraction in Agent.evaluate before enabling auto mode."
                )

        curriculum_manager.record_eval_metrics(metrics)
        if curriculum_manager.promote():
            previous_stage = current_stage_name
            next_stage = curriculum_manager.get_current_stage().name
            print(f"Curriculum promoted: {previous_stage} -> {next_stage}")
            current_train_overrides = curriculum_manager.get_env_config(evaluation=False)

        env = _build_env(cfg, current_train_overrides)
        planner.set_env(env)

    checkpoints_dir = Path(str(cfg.paths.checkpoints_dir))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoints_dir / f"{cfg.experiment.name}_td3"
    agent.save(ckpt_path)
    adapter_ckpt_path = agent.adapter_checkpoint_path(ckpt_path)

    env.close()
    final_eval_env_overrides = None
    if curriculum_manager is not None:
        final_eval_env_overrides = curriculum_manager.get_env_config(evaluation=True)
    eval_env = _build_env(cfg, final_eval_env_overrides)

    metrics = agent.evaluate(
        env=eval_env,
        n_eval_episodes=int(cfg.experiment.eval_episodes),
        deterministic=bool(cfg.experiment.eval_deterministic),
    )
    print(f"Evaluation metrics after training: {metrics}")
    print(f"Checkpoint saved at: {ckpt_path}.zip")
    if bool(getattr(adapter, "requires_training", False)):
        print(f"Adapter checkpoint saved at: {adapter_ckpt_path}")

    eval_env.close()


if __name__ == "__main__":
    main()
