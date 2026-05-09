from __future__ import annotations

import multiprocessing as mp
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def _extract_seed_bounds(env: gym.Env) -> tuple[Optional[int], Optional[int]]:
    try:
        if hasattr(env, "get_worker_seed_bounds"):
            start, count = env.get_worker_seed_bounds()  # type: ignore[misc]
            return int(start), int(count)
        if hasattr(env, "start_index") and hasattr(env, "num_scenarios"):
            return int(getattr(env, "start_index")), int(getattr(env, "num_scenarios"))
    except Exception:
        return None, None
    return None, None


def _normalize_seed_to_window(seed: int, start: int, count: int) -> int:
    if count <= 0:
        raise ValueError(f"Invalid scenario window size: {count}")
    return int(start + ((int(seed) - int(start)) % int(count)))


def _next_seed_in_window(current_seed: int, start: int, count: int) -> int:
    if count <= 0:
        raise ValueError(f"Invalid scenario window size: {count}")
    return int(start + ((int(current_seed) - int(start) + 1) % int(count)))


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[Dict[str, Any]] = {}
    last_reset_seed: Optional[int] = None
    auto_reset_seed: Optional[int] = None

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = observation
                    start, count = _extract_seed_bounds(env)
                    if start is not None and count is not None and count > 0:
                        if auto_reset_seed is None:
                            if last_reset_seed is None:
                                auto_reset_seed = int(start)
                            else:
                                auto_reset_seed = _normalize_seed_to_window(last_reset_seed, start, count)
                        auto_reset_seed = _next_seed_in_window(auto_reset_seed, start, count)
                        observation, reset_info = env.reset(seed=auto_reset_seed)
                    else:
                        observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                seed_arg, options_arg = data
                maybe_options = {"options": options_arg} if options_arg else {}
                normalized_seed = seed_arg
                if seed_arg is not None:
                    start, count = _extract_seed_bounds(env)
                    if start is not None and count is not None and count > 0:
                        normalized_seed = _normalize_seed_to_window(int(seed_arg), start, count)
                    last_reset_seed = int(normalized_seed)
                    auto_reset_seed = int(normalized_seed)
                observation, reset_info = env.reset(seed=normalized_seed, **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class DeterministicSubprocVecEnv(VecEnv):
    """
    SubprocVecEnv variant that keeps worker resets deterministic after done transitions.

    SB3 default worker auto-resets with `env.reset()` (seed=None), which may introduce
    non-determinism for environments that sample a new scenario when seed is omitted.
    This class instead advances deterministic per-worker seeds inside each worker
    scenario window when available.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)  # type: ignore[assignment]
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos  # type: ignore[return-value]

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        for pipe in self.remotes:
            pipe.send(("render", None))
        outputs = [pipe.recv() for pipe in self.remotes]
        return outputs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space) -> VecEnvObs:
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    if isinstance(space, spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))  # type: ignore[index]
    return np.stack(obs)  # type: ignore[arg-type]
