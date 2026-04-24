from __future__ import annotations

from omegaconf import OmegaConf

from thesis_rl.envs.wrappers import RuleRewardWrapper
from thesis_rl.runtime.builders import maybe_wrap_env_with_reward_manager


class _DummyEnv:
    pass


def test_wiring_wraps_env_in_rulebook_mode() -> None:
    cfg = OmegaConf.create(
        {
            "reward": {
                "mode": "rulebook",
                "rulebook": "selection",
                "attach_info": True,
                "a": 2.01,
                "c": 30.0,
                "lambda_env": 1.0,
                "lambda_rule": 0.2,
                "include_violation_vector": False,
                "scales": {"speed_limit": 5.0},
            },
        }
    )
    wrapped = maybe_wrap_env_with_reward_manager(_DummyEnv(), cfg)
    assert isinstance(wrapped, RuleRewardWrapper)


def test_wiring_is_noop_in_scalar_default_mode() -> None:
    cfg = OmegaConf.create({"reward": {"mode": "scalar_default"}})
    env = _DummyEnv()
    assert maybe_wrap_env_with_reward_manager(env, cfg) is env
