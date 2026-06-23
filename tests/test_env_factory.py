from __future__ import annotations

from thesis_rl.envs.factory import _resolve_agent_policy


def test_resolve_agent_policy_accepts_canonical_class_names() -> None:
    env_input_cls = _resolve_agent_policy("EnvInputPolicy")
    expert_cls = _resolve_agent_policy("ExpertPolicy")
    idm_cls = _resolve_agent_policy("IDMPolicy")

    assert env_input_cls.__name__ == "EnvInputPolicy"
    assert expert_cls.__name__ == "ExpertPolicy"
    assert idm_cls.__name__ == "IDMPolicy"
