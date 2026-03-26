from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from thesis_rl.adapters.base import BaseAdapter
from thesis_rl.adapters.direct_action import DirectActionAdapter
from thesis_rl.adapters.neural_adapter import NeuralAdapter
from thesis_rl.adapters.policy_adapter import PolicyAdapter
from thesis_rl.agents.agent import Agent
from thesis_rl.agents.planner_agent import Td3PlannerBackend
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


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== EVAL CONFIG ===")
    print(OmegaConf.to_yaml(cfg))

    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path is None:
        raise ValueError(
            "checkpoint_path is required. Example: "
            "uv run python -m thesis_rl.evaluate checkpoint_path=checkpoints/baseline_td3.zip"
        )

    ckpt = Path(str(checkpoint_path))
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    env = make_env(cfg.env)
    env = _maybe_wrap_env_with_reward_manager(env, cfg)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    preprocessor = _build_preprocessor(cfg)
    adapter = _build_adapter(cfg)
    planner = Td3PlannerBackend.load(checkpoint_path=ckpt, env=env, device=str(cfg.device))
    agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter)
    agent.load_adapter(checkpoint_path=ckpt, strict=True)

    metrics = agent.evaluate(
        env=env,
        n_eval_episodes=int(cfg.experiment.eval_episodes),
        deterministic=bool(cfg.experiment.eval_deterministic),
    )

    print(f"Evaluation metrics: {metrics}")
    env.close()


if __name__ == "__main__":
    main()
