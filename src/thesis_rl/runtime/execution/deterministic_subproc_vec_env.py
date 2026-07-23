from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
import warnings
from collections import OrderedDict
from multiprocessing.connection import wait
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Type, Union

import cloudpickle
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from thesis_rl.runtime.execution.numeric_threads import start_process_with_numeric_thread_limit

try:
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv as Sb3VecEnv
except ModuleNotFoundError:  # pragma: no cover

    class Sb3VecEnv:  # type: ignore[no-redef]
        def __init__(
            self, num_envs: int, observation_space: spaces.Space, action_space: spaces.Space
        ):
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
_WORKER_ERROR_MARKER = "__thesis_rl_worker_error__"
_RUNTIME_DATA_ABORT_MARKER = "__thesis_rl_runtime_data_abort__"
_CHILD_JOIN_TIMEOUT_S = 1.0


class SubprocessWorkerError(RuntimeError):
    """A vector-environment worker failed before completing its command."""


class RuntimeScenarioDataAbort:
    """Non-fatal worker response for an explicitly typed scenario data defect."""

    def __init__(self, *, slot: int, payload: Mapping[str, Any]) -> None:
        self.slot = int(slot)
        self.payload = dict(payload)


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
    acl_mode: bool = False,
    auto_reset: bool = True,
    torch_num_threads: int | None = None,
) -> None:
    parent_remote.close()
    if torch_num_threads is not None:
        if torch_num_threads <= 0:
            raise ValueError("Worker PyTorch thread count must be positive")
        import torch

        torch.set_num_threads(torch_num_threads)
        torch.set_num_interop_threads(torch_num_threads)
    try:
        env = env_fn_wrapper.var()
    except Exception as exc:
        _send_worker_error(remote, command="initialize", error=exc)
        remote.close()
        return
    reset_info: dict[str, Any] = {}
    last_reset_seed: Optional[int] = None
    auto_reset_seed: Optional[int] = None
    worker_step_index = 0

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                step_started = time.perf_counter()
                try:
                    observation, reward, terminated, truncated, info = env.step(data)
                except Exception as exc:
                    from thesis_rl.rulebook.v2.errors import RuntimeScenarioNotEvaluableError

                    if not isinstance(exc, RuntimeScenarioNotEvaluableError):
                        raise
                    remote.send(
                        (
                            _RUNTIME_DATA_ABORT_MARKER,
                            {
                                "pid": os.getpid(),
                                "reason_code": exc.reason.value,
                                "original_message": str(exc.__cause__ or exc),
                                "exception_message": str(exc),
                                "traceback": traceback.format_exc(),
                                "diagnostics": dict(exc.diagnostics),
                                "final_observation": getattr(exc, "final_observation", None),
                                "failed_action": data,
                                "worker_step_index": worker_step_index,
                            },
                        )
                    )
                    worker_step_index += 1
                    continue
                environment_step_seconds = time.perf_counter() - step_started
                enrich_started = time.perf_counter()
                from thesis_rl.runtime.io.video_diagnostics import enrich_step_info_from_env

                info = enrich_step_info_from_env(env, info)
                enrich_seconds = time.perf_counter() - enrich_started
                done = bool(terminated or truncated)
                info = dict(info)
                info["_thesis_worker_timing_seconds"] = {
                    "wrapped_env_step": environment_step_seconds,
                    "video_info_enrichment": enrich_seconds,
                    "worker_step_index": worker_step_index,
                }
                worker_step_index += 1
                info["TimeLimit.truncated"] = bool(truncated and not terminated)
                if done:
                    info["terminal_observation"] = observation
                    info["terminated"] = bool(terminated)
                    info["truncated"] = bool(truncated)
                if done and not acl_mode and auto_reset:
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
                                    auto_reset_seed = _normalize_seed_to_window(
                                        last_reset_seed, start, count
                                    )
                            auto_reset_seed = _next_seed_in_window(auto_reset_seed, start, count)
                            observation, reset_info = env.reset(seed=auto_reset_seed)
                    else:
                        observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "configure_acl_selection":
                base_env = getattr(env, "unwrapped", env)
                method = getattr(base_env, "configure_acl_selection", None)
                if not callable(method):
                    method = getattr(env, "configure_acl_selection", None)
                if not callable(method):
                    raise RuntimeError(
                        "ACL worker environment does not support selection configuration."
                    )
                remote.send(method(data))
            elif cmd == "reset_slots":
                seed_arg, options_arg = data
                maybe_options = {"options": options_arg} if options_arg else {}
                reset_started = time.perf_counter()
                observation, reset_info = env.reset(seed=seed_arg, **maybe_options)
                reset_info = dict(reset_info)
                worker_timing = dict(reset_info.get("_thesis_worker_timing_seconds", {}))
                worker_timing["reset"] = time.perf_counter() - reset_started
                reset_info["_thesis_worker_timing_seconds"] = worker_timing
                worker_step_index = 0
                remote.send((observation, reset_info))
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
                worker_step_index = 0
                remote.send((observation, reset_info))
            elif cmd == "render":
                render_kwargs = {} if data is None else dict(data)
                diagnostic_overlay = bool(render_kwargs.pop("diagnostic_geometry", False))
                render_env = getattr(env, "unwrapped", env)
                try:
                    frame = render_env.render(**render_kwargs)
                except TypeError as exc:
                    if "unexpected keyword argument 'mode'" not in str(exc):
                        raise
                    render_kwargs.pop("mode", None)
                    frame = render_env.render(**render_kwargs)
                if diagnostic_overlay and frame is not None:
                    from thesis_rl.runtime.io.video_diagnostics import (
                        annotate_geometry_frame,
                        diagnostic_geometry_from_env,
                    )

                    frame = annotate_geometry_frame(
                        frame,
                        diagnostic_geometry_from_env(env),
                    )
                remote.send(frame)
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
                    method = getattr(env, data[0], None)
                remote.send(method(*data[1], **data[2]) if callable(method) else None)
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
        except Exception as exc:
            _send_worker_error(remote, command=cmd, error=exc)
            remote.close()
            break


def _send_worker_error(remote: mp.connection.Connection, *, command: str, error: Exception) -> None:
    """Best-effort delivery of a Python worker failure to its parent."""

    payload = {
        "command": str(command),
        "pid": os.getpid(),
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(),
    }
    try:
        remote.send((_WORKER_ERROR_MARKER, payload))
    except (BrokenPipeError, EOFError, OSError):
        pass


class DeterministicSubprocVecEnv(Sb3VecEnv):
    """
    Subprocess vector env with deterministic per-worker auto-resets.

    When an episode ends in a worker, this env auto-resets that worker with the
    next deterministic seed inside the worker scenario window (if available).
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        start_method: Optional[str] = None,
        *,
        acl_mode: bool = False,
        auto_reset: bool = True,
        torch_num_threads: int | None = None,
        numeric_library_num_threads: int | None = None,
    ):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        self.acl_mode = bool(acl_mode)
        self.auto_reset = bool(auto_reset)
        self.torch_num_threads = torch_num_threads
        self.numeric_library_num_threads = numeric_library_num_threads

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes: list[mp.Process] = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (
                work_remote,
                remote,
                CloudpickleWrapper(env_fn),
                self.acl_mode,
                self.auto_reset,
                self.torch_num_threads,
            )
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            start_process_with_numeric_thread_limit(process, self.numeric_library_num_threads)
            self.processes.append(process)
            work_remote.close()

        self._send(0, "get_spaces", None)
        observation_space, action_space = self._receive(0, command="get_spaces")
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
        for index, action in enumerate(actions):
            self._send(index, "step", action)
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        try:
            responses = self._receive_many(range(self.num_envs), command="step")
            results = [responses[index] for index in range(self.num_envs)]
        finally:
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
        for env_idx in range(self.num_envs):
            self._send(env_idx, "reset", (self._seeds[env_idx], self._options[env_idx]))
        responses = self._receive_many(range(self.num_envs), command="reset")
        results = [responses[index] for index in range(self.num_envs)]
        obs, reset_infos = zip(*results)
        self.reset_infos = list(reset_infos)
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space)

    def configure_acl_selection(self, slot: int, selection: Mapping[str, Any]) -> Any:
        """Install one parent-owned selection before resetting its worker."""
        if not self.acl_mode:
            raise RuntimeError("ACL selection protocol is available only in acl_mode.")
        index = self._get_indices(slot)[0]
        self._send(index, "configure_acl_selection", dict(selection))
        return self._receive(index, command="configure_acl_selection")

    def reset_slots(
        self,
        slots: Sequence[int],
        *,
        seeds: Mapping[int, int | None] | None = None,
        options: Mapping[int, dict[str, Any] | None] | None = None,
        force: bool = False,
    ) -> dict[int, tuple[Any, dict[str, Any]]]:
        """Reset only completed ACL slots after their next selection is installed."""
        if self.auto_reset and not self.acl_mode and not force:
            raise RuntimeError(
                "Selective reset protocol requires auto_reset=False or acl_mode=True."
            )
        selected = [int(slot) for slot in slots]
        for slot in selected:
            self._send(
                slot,
                "reset_slots",
                ((seeds or {}).get(slot), (options or {}).get(slot)),
            )
        responses = self._receive_many(selected, command="reset_slots")
        for slot, (_observation, reset_info) in responses.items():
            self.reset_infos[slot] = dict(reset_info)
        return responses

    def step_slots(self, actions: Mapping[int, Any]) -> dict[int, Any]:
        """Step only active workers, preserving manual-reset episode boundaries."""

        selected = sorted(int(slot) for slot in actions)
        for slot in selected:
            self._send(slot, "step", actions[slot])
        return self._receive_many(selected, command="step")

    def render_slots(
        self,
        render_kwargs: Mapping[str, Any] | None = None,
        *,
        slots: Sequence[int] | None = None,
    ) -> dict[int, Any]:
        """Render selected workers concurrently and return slot-indexed frames."""

        kwargs = dict(render_kwargs or {})
        selected = (
            list(range(self.num_envs)) if slots is None else sorted({int(slot) for slot in slots})
        )
        for slot in selected:
            self._send(slot, "render", kwargs)
        return self._receive_many(selected, command="render")

    def get_slot_proxy(self, slot: int) -> "VectorEnvSlotProxy":
        """Return a single-worker proxy suitable for callbacks and recorders."""

        return VectorEnvSlotProxy(self, int(slot))

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            self._abort_workers()
            return
        for index in range(self.num_envs):
            try:
                self.remotes[index].send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        self._close_remotes()
        self._reap_children(terminate_live=True)
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        for index in range(self.num_envs):
            self._send(index, "render", None)
        responses = self._receive_many(range(self.num_envs), command="render")
        return [responses[index] for index in range(self.num_envs)]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        target_remotes = self._get_target_remotes(indices)
        target_indices = self._get_indices(indices)
        for index, _remote in zip(target_indices, target_remotes):
            self._send(index, "get_attr", attr_name)
        responses = self._receive_many(target_indices, command="get_attr")
        return [responses[index] for index in target_indices]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_remotes = self._get_target_remotes(indices)
        target_indices = self._get_indices(indices)
        for index, _remote in zip(target_indices, target_remotes):
            self._send(index, "set_attr", (attr_name, value))
        self._receive_many(target_indices, command="set_attr")

    def env_method(
        self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs
    ) -> List[Any]:
        target_remotes = self._get_target_remotes(indices)
        target_indices = self._get_indices(indices)
        command = f"env_method:{method_name}"
        for index, _remote in zip(target_indices, target_remotes):
            self._send(index, "env_method", (method_name, method_args, method_kwargs))
        responses = self._receive_many(target_indices, command=command)
        return [responses[index] for index in target_indices]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        target_remotes = self._get_target_remotes(indices)
        target_indices = self._get_indices(indices)
        for index, _remote in zip(target_indices, target_remotes):
            self._send(index, "is_wrapped", wrapper_class)
        responses = self._receive_many(target_indices, command="is_wrapped")
        return [bool(responses[index]) for index in target_indices]

    def _receive(self, index: int, *, command: str) -> Any:
        """Receive one response, applying the same fatal cleanup as batches."""

        return self._receive_many([index], command=command)[index]

    def _send(self, index: int, command: str, data: Any) -> None:
        """Send one command or fail the complete vector environment."""

        try:
            self.remotes[index].send((command, data))
        except (BrokenPipeError, EOFError, OSError) as exc:
            error = self._transport_error(index, command, exc)
            self._abort_workers()
            raise error from exc

    def _receive_many(self, indices: Sequence[int], *, command: str) -> dict[int, Any]:
        """Receive a batch in readiness order while returning slot-indexed replies."""

        pending = {int(index) for index in indices}
        results: dict[int, Any] = {}
        try:
            while pending:
                ready = wait([self.remotes[index] for index in pending])
                remote_to_index = {id(self.remotes[index]): index for index in pending}
                for remote in ready:
                    index = remote_to_index[id(remote)]
                    results[index] = self._receive_response(index, command=command)
                    pending.remove(index)
        except SubprocessWorkerError:
            self._abort_workers()
            raise
        return results

    def _receive_response(self, index: int, *, command: str) -> Any:
        """Decode one ready response without changing sibling worker state."""

        try:
            result = self.remotes[index].recv()
        except EOFError as exc:
            raise self._transport_error(index, command, exc) from exc
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], str)
            and result[0] == _WORKER_ERROR_MARKER
            and isinstance(result[1], Mapping)
        ):
            payload = result[1]
            raise SubprocessWorkerError(
                "Vector environment worker raised an exception: "
                f"slot={index}, command={payload.get('command', command)!r}, "
                f"pid={payload.get('pid')}, type={payload.get('exception_type')}, "
                f"message={payload.get('exception_message')}\n"
                f"Remote traceback:\n{payload.get('traceback', '<unavailable>')}"
            )
        if (
            command == "step"
            and isinstance(result, tuple)
            and len(result) == 2
            and result[0] == _RUNTIME_DATA_ABORT_MARKER
            and isinstance(result[1], Mapping)
        ):
            return RuntimeScenarioDataAbort(slot=index, payload=result[1])
        return result

    def _transport_error(
        self, index: int, command: str, error: BaseException | None = None
    ) -> SubprocessWorkerError:
        process = self.processes[index]
        detail = "" if error is None else f" transport_error={type(error).__name__}: {error}."
        return SubprocessWorkerError(
            "Vector environment worker terminated without a response: "
            f"slot={index}, command={command!r}, pid={process.pid}, "
            f"exitcode={process.exitcode}. This usually indicates a native crash, "
            f"signal, OOM kill, or an exception before the worker could report it.{detail}"
        )

    def _abort_workers(self) -> None:
        """Close pipes and stop only this vector environment's child processes."""

        if self.closed:
            return
        self.waiting = False
        self._close_remotes()
        self._reap_children(terminate_live=True)
        self.closed = True

    def _close_remotes(self) -> None:
        for remote in self.remotes:
            try:
                remote.close()
            except OSError:
                pass

    def _reap_children(self, *, terminate_live: bool) -> None:
        for process in self.processes:
            process.join(timeout=_CHILD_JOIN_TIMEOUT_S)
        if terminate_live:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            for process in self.processes:
                process.join(timeout=_CHILD_JOIN_TIMEOUT_S)

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


class VectorEnvSlotProxy:
    """Remote single-slot view used by evaluation callbacks and recorders."""

    def __init__(self, vector_env: DeterministicSubprocVecEnv, slot: int) -> None:
        object.__setattr__(self, "_vector_env", vector_env)
        object.__setattr__(self, "_slot", int(slot))
        object.__setattr__(self, "env", None)
        object.__setattr__(self, "_rendered_frame", None)

    @property
    def unwrapped(self) -> "VectorEnvSlotProxy":
        return self

    def __getattr__(self, name: str) -> Any:
        return self._vector_env.get_attr(name, indices=self._slot)[0]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_vector_env", "_slot", "env", "_rendered_frame"}:
            object.__setattr__(self, name, value)
            return
        self._vector_env.set_attr(name, value, indices=self._slot)

    def set_rendered_frame(self, frame: Any) -> None:
        object.__setattr__(self, "_rendered_frame", frame)

    def consume_rendered_frame(self) -> Any:
        frame = self._rendered_frame
        object.__setattr__(self, "_rendered_frame", None)
        return frame

    def render(self, **kwargs: Any) -> Any:
        cached = self.consume_rendered_frame()
        if cached is not None:
            return cached
        return self._vector_env.render_slots(kwargs)[self._slot]

    def close(self) -> None:
        return None


def _flatten_obs(
    obs: Union[List[VecEnvObs], Tuple[VecEnvObs, ...]], space: spaces.Space
) -> VecEnvObs:
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), (
            "non-dict observation for environment with Dict observation space"
        )
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    if isinstance(space, spaces.Tuple):
        assert isinstance(obs[0], tuple), (
            "non-tuple observation for environment with Tuple observation space"
        )
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))
    return np.stack(obs)
