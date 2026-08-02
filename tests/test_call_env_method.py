from __future__ import annotations

import warnings

import gymnasium as gym

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


class _RawVecEnvWithMethod:
    """A raw SB3-style `VecEnv` stand-in: has `env_method` and `unwrapped`
    (returning itself, matching SB3's `VecEnv.unwrapped` for a non-wrapper
    vec env) but no `get_wrapper_attr` of its own, unlike a `gym.Env`.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    @property
    def unwrapped(self) -> "_RawVecEnvWithMethod":
        return self

    def env_method(self, method_name: str, *args):
        self.calls.append((method_name, args))
        return ["ok"]


def test_call_env_method_falls_back_through_unwrapped_without_deprecation_warning() -> None:
    """Regression: the fallback used to be `env.env_method(...)`, a plain attribute
    access on a real `gym.Wrapper` chain. That trips `gym.Wrapper.__getattr__`,
    which unconditionally logs gymnasium's "deprecated attribute forwarding"
    warning before resolving the attribute, at every wrapper level along the
    way to the base env. This is exactly the shape of the real bug: `get_wrapper_attr`
    recursion reaches the innermost SB3 `VecEnv`, which has no `get_wrapper_attr`
    of its own, so it raises `AttributeError` and `call_env_method` falls back.
    Going through `env.unwrapped.env_method(...)` instead reaches the same base
    env via a real property, with no warning.
    """

    inner = _RawVecEnvWithMethod()
    wrapped = gym.Wrapper(inner)  # type: ignore[arg-type]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = call_env_method(wrapped, "quarantine_scenario_uid", "waymo:fixture")

    assert result == ["ok"]
    assert inner.calls == [("quarantine_scenario_uid", ("waymo:fixture",))]
