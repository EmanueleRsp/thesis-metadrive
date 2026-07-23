from __future__ import annotations

from thesis_rl.agent.planners.core.utils import call_env_method


class _VecEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def env_method(self, method_name: str, *args):
        self.calls.append((method_name, args))
        return ["ok"]


class _WrapperWithoutOwnEnvMethod:
    """Mimics gym.Wrapper: no `env_method` of its own, only `get_wrapper_attr`."""

    def __init__(self, inner: _VecEnv) -> None:
        self._inner = inner

    def get_wrapper_attr(self, name: str):
        return getattr(self._inner, name)


def test_call_env_method_prefers_get_wrapper_attr_when_available() -> None:
    inner = _VecEnv()
    wrapper = _WrapperWithoutOwnEnvMethod(inner)

    result = call_env_method(wrapper, "quarantine_scenario_uid", "waymo:fixture")

    assert result == ["ok"]
    assert inner.calls == [("quarantine_scenario_uid", ("waymo:fixture",))]


def test_call_env_method_falls_back_to_plain_env_method() -> None:
    env = _VecEnv()

    result = call_env_method(env, "get_quarantined_scenario_uids")

    assert result == ["ok"]
    assert env.calls == [("get_quarantined_scenario_uids", ())]
