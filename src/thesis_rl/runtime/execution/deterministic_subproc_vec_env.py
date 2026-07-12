from __future__ import annotations

import multiprocessing as mp
import warnings
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import cloudpickle
import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv as Sb3VecEnv
except ModuleNotFoundError:  # pragma: no cover
    class Sb3VecEnv:  # type: ignore[no-redef]
        def __init__(self, num_envs: int, observation_space: spaces.Space, action_space: spaces.Space):
            self.num_envs = int(num_envs)
            self.observation_space = observation_space
            self.action_space = action_space
            self.reset_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
            self._seeds: list[Optional[int]] = [None for _ in range(self.num_envs)]
            self._options: list[Optional[dict[str, Any]]] = [None for _ in range(self.num_envs)]

        def _reset_seeds(self) -> None:
            self._seeds = [None for _ in range(self.num_envs)]

        def _reset_options(self) -> None:
            self._options = [None for _ in range(self.num_envs)]

VecEnvIndices = Union[None, int, Sequence[int], np.ndarray]
VecEnvObs = Union[np.ndarray, dict[str, np.ndarray], tuple[np.ndarray, ...]]
VecEnvStepReturn = tuple[VecEnvObs, np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]


class CloudpickleWrapper:
    def __init__(self, var: Any) -> None:
        self.var = var

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, payload: bytes) -> None:
        self.var = cloudpickle.loads(payload)


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


def _is_wrapped(env: gym.Env, wrapper_class: Type[gym.Wrapper]) -> bool:
    current = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, wrapper_class):
            return True
        current = current.env
    return False


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    parent_remote.close()
    env = env_fn_wrapper.var()
    reset_info: dict[str, Any] = {}
    last_reset_seed: Optional[int] = None
    auto_reset_seed: Optional[int] = None

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                done = bool(terminated or truncated)
                info = dict(info)
                info["TimeLimit.truncated"] = bool(truncated and not terminated)
                if done:
                    info["terminal_observation"] = observation
                    start, count = _extract_seed_bounds(env)
                    if start is not None and count is not None and count > 0:
                        provider_env = getattr(env, "unwrapped", env)
                        if getattr(provider_env, "scenario_provider", None) is not None:
                            # Provider-driven ScenarioNet resets must sample the
                            # next record; forcing a seed here would bypass the
                            # worker/source partition and break strict sampling.
                            observation, reset_info = env.reset()
                        else:
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
                base_env = getattr(env, "unwrapped", env)
                method = getattr(base_env, data[0], None)
                if not callable(method):
                    method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(_is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class DeterministicSubprocVecEnv(Sb3VecEnv):
    """
    Subprocess vector env with deterministic per-worker auto-resets.

    When an episode ends in a worker, this env auto-resets that worker with the
    next deterministic seed inside the worker scenario window (if available).
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes: list[mp.Process] = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(self.num_envs, observation_space, action_space)
        self.render_mode = getattr(getattr(env_fns[0], "__self__", None), "render_mode", None)
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

    def seed(self, seed: int | None = None) -> list[Optional[int]]:
        if seed is None:
            self._seeds = [None for _ in range(self.num_envs)]
        else:
            base = int(seed)
            self._seeds = [base + idx for idx in range(self.num_envs)]
        return list(self._seeds)

    def _reset_seeds(self) -> None:
        self._seeds = [None for _ in range(self.num_envs)]

    def _reset_options(self) -> None:
        self._options = [None for _ in range(self.num_envs)]

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, reset_infos = zip(*results)
        self.reset_infos = list(reset_infos)
        return (
            _flatten_obs(obs, self.observation_space),
            np.asarray(rews, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            tuple(dict(info) for info in infos),
        )

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        self.step_async(actions)
        return self.step_wait()

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, reset_infos = zip(*results)
        self.reset_infos = list(reset_infos)
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
            self.waiting = False
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
        return [bool(remote.recv()) for remote in target_remotes]

    def _get_indices(self, indices: VecEnvIndices) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, (int, np.integer)):
            idx = int(indices)
            if idx < 0:
                idx = self.num_envs + idx
            if idx < 0 or idx >= self.num_envs:
                raise IndexError(f"Env index out of range: {idx}")
            return [idx]
        return [int(i) for i in list(indices)]

    def _get_target_remotes(self, indices: VecEnvIndices) -> list[Any]:
        return [self.remotes[i] for i in self._get_indices(indices)]


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs, ...]], space: spaces.Space) -> VecEnvObs:
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    if isinstance(space, spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))
    return np.stack(obs)
