from __future__ import annotations

from dataclasses import asdict
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from thesis_rl.agents.agent import Agent
from thesis_rl.curriculum.config import CurriculumConfig
from thesis_rl.curriculum.manager import CurriculumManager
from thesis_rl.runtime.builders import (
    adapter_space_kwargs,
    build_adapter,
    build_env,
    build_planner,
    build_preprocessor,
)


def _required_curriculum_metrics(curriculum_cfg: CurriculumConfig) -> list[str]:
    """Derive required metric names from configured curriculum gates.

    Gate keys are expected to follow `<metric>_min` or `<metric>_max`.
    """
    gates_payload = asdict(curriculum_cfg.promotion.gates)
    required: list[str] = []
    seen: set[str] = set()

    for gate_group in gates_payload.values():
        if not isinstance(gate_group, dict):
            continue
        for gate_key in gate_group.keys():
            metric_name = str(gate_key)
            if metric_name.endswith("_min"):
                metric_name = metric_name[: -len("_min")]
            elif metric_name.endswith("_max"):
                metric_name = metric_name[: -len("_max")]
            if metric_name and metric_name not in seen:
                seen.add(metric_name)
                required.append(metric_name)
    return required


def _missing_curriculum_metrics(
    metrics: dict[str, float], curriculum_cfg: CurriculumConfig
) -> list[str]:
    required = _required_curriculum_metrics(curriculum_cfg)
    return [name for name in required if name not in metrics]


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_env_spaces(env: Any, seed: int) -> None:
    action_space = getattr(env, "action_space", None)
    if action_space is not None and hasattr(action_space, "seed"):
        action_space.seed(seed)
    observation_space = getattr(env, "observation_space", None)
    if observation_space is not None and hasattr(observation_space, "seed"):
        observation_space.seed(seed)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== TRAIN CONFIG ===")
    print(OmegaConf.to_yaml(cfg))

    run_seed = int(cfg.seed)
    _set_global_seed(run_seed)

    ###################
    ###### SETUP ######
    ###################
    
    # Curriculum manager
    curriculum_cfg = CurriculumConfig.from_curriculum_cfg(cfg.curriculum)
    curriculum_manager: CurriculumManager | None = None
    current_train_overrides: dict[str, Any] | None = None
    if curriculum_cfg.enabled and curriculum_cfg.stages:
        curriculum_manager = CurriculumManager(curriculum_cfg)
        current_train_overrides = curriculum_manager.get_env_config(evaluation=False)

    # Environment
    env = build_env(cfg, current_train_overrides)
    _seed_env_spaces(env, run_seed)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Agent
    preprocessor = build_preprocessor(cfg)
    adapter = build_adapter(
        cfg,
        adapter_space_kwargs(env.action_space),
    )
    planner = build_planner(cfg, env, seed=run_seed)
    agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter)

    # Training and evaluation params
    total_timesteps = int(cfg.experiment.total_timesteps)
    log_interval = int(cfg.experiment.get("log_interval", 1000))
    eval_interval = int(cfg.experiment.get("eval_interval", total_timesteps))
    if eval_interval <= 0:
        eval_interval = total_timesteps

    ##################
    ###### LOOP ######
    ##################

    # Training loop with periodic evaluation and optional curriculum progression
    remaining = total_timesteps
    while remaining > 0:

        # Stage info and chunk size 
        chunk_steps = min(eval_interval, remaining)
        current_stage_name = "baseline"
        if curriculum_manager is not None:
            current_stage_name = curriculum_manager.get_current_stage().name
        print(
            f"Training chunk on stage '{current_stage_name}': "
            f"steps={chunk_steps}, remaining_after={remaining - chunk_steps}"
        )

        ###### TRAINING ######

        # Agent training
        agent.train(
            env=env,
            chunk_timesteps=chunk_steps,
            global_total_timesteps=total_timesteps,
            global_steps_done=total_timesteps - remaining,
            deterministic=False,
            log_interval=log_interval,
            reset_seed=run_seed + (total_timesteps - remaining),
        )
        remaining -= chunk_steps

        # Record training steps in curriculum manager for potential stage progression
        if curriculum_manager is not None:
            curriculum_manager.record_train_steps(chunk_steps)

        # MetaDrive uses a global engine singleton: close training env before creating eval env.
        env.close()
        
        ###### EVALUATION ######

        # Config setup
        eval_env_overrides = None
        if curriculum_manager is not None:
            eval_env_overrides = curriculum_manager.get_env_config(evaluation=True)
        
        # Environment
        eval_env = build_env(cfg, eval_env_overrides)
        _seed_env_spaces(eval_env, run_seed + 100_000 + (total_timesteps - remaining))
        planner.set_env(eval_env)   # Update planner's env reference

        # Evaluation
        metrics = agent.evaluate(
            env=eval_env,
            n_eval_episodes=int(cfg.experiment.eval_episodes),
            deterministic=bool(cfg.experiment.eval_deterministic),
            base_seed=run_seed + 200_000 + (total_timesteps - remaining),
        )
        eval_env.close()
        print(f"Evaluation metrics after chunk: {metrics}")

        ###### CURRICULUM PROGRESSION ######

        # If no curriculum, just rebuild the env
        if curriculum_manager is None:
            env = build_env(cfg, current_train_overrides)
            _seed_env_spaces(env, run_seed + 300_000 + (total_timesteps - remaining))
            planner.set_env(env)
            continue

        # Metrics check
        if curriculum_cfg.mode.lower() == "auto":
            missing_metrics = _missing_curriculum_metrics(metrics, curriculum_cfg)
            if missing_metrics:
                raise ValueError(
                    "Curriculum auto mode requires metrics: "
                    f"{missing_metrics}. "
                    "Implement metric extraction in Agent.evaluate before enabling auto mode."
                )

        # Record eval metrics
        curriculum_manager.record_eval_metrics(metrics)

        # Check for promotion and update env config if promoted
        if curriculum_manager.promote():
            previous_stage = current_stage_name
            next_stage = curriculum_manager.get_current_stage().name
            print(f"Curriculum promoted: {previous_stage} -> {next_stage}")
            current_train_overrides = curriculum_manager.get_env_config(evaluation=False)

        # Rebuild env with new config (if changed) and update planner's env reference
        env = build_env(cfg, current_train_overrides)
        _seed_env_spaces(env, run_seed + 400_000 + (total_timesteps - remaining))
        planner.set_env(env)    # Update planner's env reference

    ##########################
    ###### FINALIZATION ######
    ##########################

    # Save agent checkpoints
    checkpoints_dir = Path(str(cfg.paths.checkpoints_dir))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoints_dir / f"{cfg.experiment.name}"
    agent.save(ckpt_path)
    adapter_ckpt_path = agent.adapter_checkpoint_path(ckpt_path)

    ###### FINAL EVAL ######

    # Environment
    env.close()
    final_eval_env_overrides = None
    if curriculum_manager is not None:
        final_eval_env_overrides = curriculum_manager.get_env_config(evaluation=True)
    eval_env = build_env(cfg, final_eval_env_overrides)
    _seed_env_spaces(eval_env, run_seed + 500_000)

    # Evaluation
    metrics = agent.evaluate(
        env=eval_env,
        n_eval_episodes=int(cfg.experiment.eval_episodes),
        deterministic=bool(cfg.experiment.eval_deterministic),
        base_seed=run_seed + 600_000,
    )
    print(f"Evaluation metrics after training: {metrics}")
    print(f"Checkpoint saved at: {ckpt_path}.zip")
    if bool(getattr(adapter, "requires_training", False)):
        print(f"Adapter checkpoint saved at: {adapter_ckpt_path}")

    eval_env.close()


if __name__ == "__main__":
    main()
