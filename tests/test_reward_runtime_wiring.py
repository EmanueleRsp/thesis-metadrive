from __future__ import annotations

from omegaconf import OmegaConf

from thesis_rl.envs.wrappers import RuleRewardWrapper
from thesis_rl.evaluate import _maybe_wrap_env_with_reward_manager as eval_wrap
from thesis_rl.train import _maybe_wrap_env_with_reward_manager as train_wrap


class _DummyEnv:
    pass


def _build_hybrid_cfg():
    return OmegaConf.create(
        {
            "reward": {
                "mode": "hybrid",
                "attach_info": True,
                "a": 2.01,
                "c": 30.0,
                "lambda_env": 1.0,
                "lambda_rule": 0.2,
                "include_violation_vector": False,
                "scales": {"speed_limit": 5.0},
            },
            "rulebook": {
                "rules": [
                    {"name": "speed_limit", "priority": 0},
                ]
            },
        }
    )


def test_train_wiring_wraps_env_in_hybrid_mode() -> None:
    cfg = _build_hybrid_cfg()
    wrapped = train_wrap(_DummyEnv(), cfg)
    assert isinstance(wrapped, RuleRewardWrapper)


def test_evaluate_wiring_wraps_env_in_hybrid_mode() -> None:
    cfg = _build_hybrid_cfg()
    wrapped = eval_wrap(_DummyEnv(), cfg)
    assert isinstance(wrapped, RuleRewardWrapper)


def test_wiring_is_noop_when_not_hybrid() -> None:
    cfg = OmegaConf.create({"reward": {"mode": "scalar"}})
    env = _DummyEnv()
    assert train_wrap(env, cfg) is env
    assert eval_wrap(env, cfg) is env
