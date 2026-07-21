"""Parent-owned FIFO orchestration for asynchronous training evaluation."""

from __future__ import annotations

import json
import multiprocessing as mp
import queue
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table


class AsyncEvaluationError(RuntimeError):
    """Raised when an asynchronous evaluation cannot produce a complete result."""


@dataclass(frozen=True)
class EvaluationJob:
    """Immutable evaluation payload captured at queue admission time."""

    eval_id: int
    global_step: int
    stage: str
    stage_index: int
    episode_count: int
    base_seed: int | None
    env_seed: int
    workers: int
    cfg: dict[str, Any]
    env_overrides: dict[str, Any]
    checkpoint_stem: str
    scenario_arm_schedule: tuple[str, ...] | None = None
    scenario_source_schedule: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _JobState:
    job: EvaluationJob
    status: str = "queued"
    completed: int = 0
    error: str | None = None
    metrics: dict[str, Any] | None = None
    process: mp.Process | None = None


def _evaluation_worker_main(job: EvaluationJob, output_queue: Any) -> None:
    """Rebuild an isolated evaluator and execute one immutable job."""

    try:
        from thesis_rl.agent.agent import Agent
        from thesis_rl.runtime.execution.seeding import set_global_seed, seed_env_spaces
        from thesis_rl.runtime.wiring.builders import (
            adapter_space_kwargs,
            build_adapter,
            build_eval_env,
            build_preprocessor,
            load_planner,
        )

        cfg = OmegaConf.create(job.cfg)
        set_global_seed(int(cfg.seed))
        env = build_eval_env(
            cfg,
            job.env_overrides,
            n_eval_episodes=job.episode_count,
            workers=job.workers,
            scenario_arm_schedule=job.scenario_arm_schedule,
            scenario_source_schedule=job.scenario_source_schedule,
        )
        seed_env_spaces(env, job.env_seed)
        preprocessor = build_preprocessor(cfg)
        adapter = build_adapter(cfg, adapter_space_kwargs(env.action_space))
        planner = load_planner(cfg, checkpoint_path=f"{job.checkpoint_stem}.zip", env=env)
        agent = Agent(
            preprocessor=preprocessor,
            planner=planner,
            adapter=adapter,
            ema_alpha=float(cfg.agent.planner.algorithm.get("monitor_ema_alpha", 0.1)),
        )
        agent.load_adapter(checkpoint_path=f"{job.checkpoint_stem}.zip", strict=True)
        output_queue.put(("started", job.eval_id, job.episode_count))

        def progress_callback(completed: int, total: int) -> None:
            output_queue.put(("progress", job.eval_id, int(completed), int(total)))

        def before_episode_reset(target_env: Any, episode_index: int) -> None:
            if job.scenario_arm_schedule is not None:
                target_env.scenario_arm = job.scenario_arm_schedule[episode_index]
            if job.scenario_source_schedule is not None:
                target_env.scenario_source = job.scenario_source_schedule[episode_index]

        metrics = agent.evaluate(
            env=env,
            n_eval_episodes=job.episode_count,
            deterministic=bool(cfg.experiment.eval_deterministic),
            base_seed=job.base_seed,
            return_episode_metrics=True,
            error_priority_base=float(cfg.reward.get("a", 2.01)),
            show_progress=False,
            progress_callback=progress_callback,
            before_episode_reset_callback=before_episode_reset,
        )
        if not isinstance(metrics, dict) or len(metrics.get("per_episode", {})) == 0:
            raise RuntimeError("Asynchronous evaluation returned no complete per-episode metrics.")
        output_queue.put(("finished", job.eval_id, metrics))
    except BaseException as exc:  # propagate every evaluator failure to the parent
        output_queue.put(
            (
                "error",
                job.eval_id,
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        )
    finally:
        try:
            env.close()  # type: ignore[name-defined]
        except Exception:
            pass


class AsyncEvaluationManager:
    """FIFO coordinator whose Rich state is safe to render from the parent."""

    def __init__(
        self,
        *,
        checkpoints_dir: Path,
        on_complete: Callable[[EvaluationJob, dict[str, Any]], None] | None = None,
        process_factory: Callable[..., mp.Process] | None = None,
        start_method: str = "spawn",
    ) -> None:
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"Unsupported asynchronous evaluation start method: {start_method}")
        self._ctx = mp.get_context(start_method)
        self._queue = self._ctx.Queue()
        self._process_factory = process_factory or self._ctx.Process
        self._checkpoints_dir = checkpoints_dir / "async_eval"
        self._on_complete = on_complete
        self._pending: deque[_JobState] = deque()
        self._active: _JobState | None = None
        self._jobs: dict[int, _JobState] = {}
        self._last_completed: _JobState | None = None
        self._event_messages: deque[str] = deque(maxlen=32)
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            expand=True,
        )
        self._progress_task = self._progress.add_task("Evaluation idle", total=1, completed=1)
        self._fatal_error: AsyncEvaluationError | None = None

    @property
    def active(self) -> EvaluationJob | None:
        return self._active.job if self._active is not None else None

    @property
    def queued_count(self) -> int:
        return len(self._pending)

    @property
    def last_completed_metrics(self) -> dict[str, Any] | None:
        return None if self._last_completed is None else self._last_completed.metrics

    def enqueue(
        self,
        *,
        agent: Any,
        cfg: DictConfig,
        eval_id: int,
        global_step: int,
        stage: str,
        stage_index: int,
        episode_count: int,
        base_seed: int | None,
        env_seed: int,
        env_overrides: Mapping[str, Any] | None = None,
        workers: int | None = None,
        scenario_arm_schedule: tuple[str, ...] | None = None,
        scenario_source_schedule: tuple[str, ...] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EvaluationJob:
        """Snapshot the live learner and admit a complete FIFO job."""

        self.poll()
        if self._fatal_error is not None:
            raise self._fatal_error
        if int(episode_count) <= 0:
            raise ValueError("Asynchronous evaluation requires episode_count > 0.")
        resolved_workers = int(
            workers if workers is not None else evaluation_num_workers_from_cfg(cfg, final=False)
        )
        if resolved_workers <= 0:
            raise ValueError("Asynchronous evaluation workers must be > 0.")
        resolved_workers = min(resolved_workers, int(episode_count))
        job_index = len(self._jobs) + 1
        stem = self._checkpoints_dir / f"eval_{job_index:06d}_step_{int(global_step):012d}"
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        agent.save(stem)
        snapshot_payload = {
            "eval_id": int(eval_id),
            "global_step": int(global_step),
            "stage": stage,
            "stage_index": int(stage_index),
            "episode_count": int(episode_count),
            "base_seed": base_seed,
            "env_seed": int(env_seed),
            "env_overrides": dict(env_overrides or {}),
        }
        Path(f"{stem}.json").write_text(
            json.dumps(snapshot_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(resolved_cfg, dict):
            raise TypeError("Resolved asynchronous evaluation config must be a mapping.")
        job = EvaluationJob(
            eval_id=int(eval_id),
            global_step=int(global_step),
            stage=str(stage),
            stage_index=int(stage_index),
            episode_count=int(episode_count),
            base_seed=None if base_seed is None else int(base_seed),
            env_seed=int(env_seed),
            workers=resolved_workers,
            cfg=resolved_cfg,
            env_overrides=dict(env_overrides or {}),
            checkpoint_stem=str(stem),
            scenario_arm_schedule=scenario_arm_schedule,
            scenario_source_schedule=scenario_source_schedule,
            metadata=dict(metadata or {}),
        )
        state = _JobState(job=job)
        self._jobs[job.eval_id] = state
        self._pending.append(state)
        self._event_messages.append(
            f"[EVAL] Queued evaluation {job.eval_id} | step={job.global_step} | episodes={job.episode_count}"
        )
        self._launch_next()
        return job

    def poll(self) -> None:
        """Process all currently available IPC messages and launch the next job."""

        if self._fatal_error is not None:
            raise self._fatal_error
        while True:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                break
            self._handle_message(message)
        if self._active is not None and self._active.process is not None:
            while (
                self._active is not None
                and not self._active.process.is_alive()
                and self._active.status not in {"finished", "failed"}
            ):
                try:
                    message = self._queue.get(timeout=0.05)
                except queue.Empty:
                    self._fail(
                        self._active,
                        "Evaluator exited without a completion or error message.",
                    )
                    break
                self._handle_message(message)
        if self._fatal_error is not None:
            raise self._fatal_error

    def drain(self) -> None:
        """Wait until every admitted evaluation has completed, or fail fatally."""

        while self._pending or self._active is not None:
            self.poll()
            if self._fatal_error is not None:
                raise self._fatal_error
            if self._pending or self._active is not None:
                try:
                    message = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._handle_message(message)
        self.poll()
        if self._fatal_error is not None:
            raise self._fatal_error

    def drain_event_messages(self) -> list[str]:
        self.poll()
        messages = list(self._event_messages)
        self._event_messages.clear()
        return messages

    def renderables(self) -> list[Any]:
        """Return the evaluation progress bar and last-complete result table."""

        self.poll()
        active = self._active
        current = f"{active.completed}/{active.job.episode_count}" if active else "0/0"
        progress_description = (
            "Evaluation episodes" if active is None else f"Evaluation episodes ({current})"
        )
        self._progress.update(
            self._progress_task,
            description=progress_description,
            total=max(active.job.episode_count, 1) if active is not None else 1,
            completed=active.completed if active is not None else 1,
        )
        last = Table(title="Last Completed Evaluation", expand=True)
        last.add_column("Metric", style="cyan", no_wrap=True)
        last.add_column("Value", style="white")
        if self._last_completed is None or self._last_completed.metrics is None:
            values = {
                "Global step": "-",
                "Reward": "-",
                "Success rate": "-",
                "Collision rate": "-",
                "Out-of-road rate": "-",
                "Route completion": "-",
                "Top rule violation": "-",
                "Avg error value": "-",
            }
        else:
            job = self._last_completed.job
            metrics = self._last_completed.metrics
            values = {
                "Global step": str(job.global_step),
                "Reward": (
                    f"{float(metrics.get('mean_reward', 0.0)):.4g} ± "
                    f"{float(metrics.get('std_reward', 0.0)):.4g}"
                ),
                "Success rate": f"{float(metrics.get('success_rate', 0.0)):.4g}",
                "Collision rate": f"{float(metrics.get('collision_rate', 0.0)):.4g}",
                "Out-of-road rate": f"{float(metrics.get('out_of_road_rate', 0.0)):.4g}",
                "Route completion": f"{float(metrics.get('route_completion', 0.0)):.4g}",
                "Top rule violation": f"{float(metrics.get('top_rule_violation_rate', 0.0)):.4g}",
                "Avg error value": f"{float(metrics.get('avg_error_value', 0.0)):.4g}",
            }
        for metric, value in values.items():
            last.add_row(metric, value)
        return [self._progress, Panel(last, title="Evaluation Monitor")]

    def close(self) -> None:
        """Close IPC resources after a successful drain."""

        if self._active is not None and self._active.process is not None:
            self._active.process.join(timeout=1.0)
        self._queue.close()
        self._queue.join_thread()

    def _launch_next(self) -> None:
        if self._active is not None or not self._pending:
            return
        state = self._pending.popleft()
        state.status = "running"
        process = self._process_factory(
            target=_evaluation_worker_main,
            args=(state.job, self._queue),
            daemon=False,
        )
        state.process = process
        self._active = state
        process.start()
        self._event_messages.append(
            f"[EVAL] Started evaluation {state.job.eval_id} | step={state.job.global_step}"
        )

    def _handle_message(self, message: Any) -> None:
        if not isinstance(message, tuple) or len(message) < 2:
            raise AsyncEvaluationError(f"Malformed asynchronous evaluation message: {message!r}")
        kind, eval_id = message[0], int(message[1])
        state = self._jobs.get(eval_id)
        if state is None:
            raise AsyncEvaluationError(f"Message references unknown evaluation {eval_id}.")
        if kind == "started":
            state.status = "running"
            return
        if kind == "progress":
            state.completed = int(message[2])
            return
        if kind == "finished":
            metrics = message[2]
            if not isinstance(metrics, dict) or not metrics.get("per_episode"):
                self._fail(state, "Evaluation completion did not contain complete metrics.")
                return
            state.status = "finished"
            state.completed = state.job.episode_count
            state.metrics = metrics
            self._last_completed = state
            self._event_messages.append(
                f"[EVAL] Evaluation {eval_id} completed | step={state.job.global_step}"
            )
            if self._on_complete is not None:
                self._on_complete(state.job, metrics)
            self._finish_active(state)
            return
        if kind == "error":
            self._fail(state, str(message[2]))
            return
        self._fail(state, f"Unknown asynchronous evaluation message kind: {kind!r}")

    def _finish_active(self, state: _JobState) -> None:
        if self._active is state:
            if state.process is not None:
                # The worker sends completion before its ``finally`` closes the
                # environment; wait for that cleanup before launching the next
                # FIFO job so there is never more than one evaluator process.
                state.process.join()
            self._remove_snapshot(state.job)
            self._active = None
            self._launch_next()

    def _fail(self, state: _JobState, reason: str) -> None:
        state.status = "failed"
        state.error = reason
        self._event_messages.append(f"[EVAL] Evaluation {state.job.eval_id} failed: {reason}")
        if state.process is not None and state.process.is_alive():
            state.process.terminate()
            state.process.join(timeout=5.0)
        self._fatal_error = AsyncEvaluationError(
            f"Asynchronous evaluation {state.job.eval_id} failed; training terminated.\n{reason}"
        )
        self._active = None

    def _remove_snapshot(self, job: EvaluationJob) -> None:
        stem = Path(job.checkpoint_stem)
        for path in (
            stem.with_suffix(".zip"),
            stem.with_name(f"{stem.name}.adapter.pt"),
            stem.with_name(f"{stem.name}.reward_semantics.json"),
            stem.with_suffix(".json"),
        ):
            path.unlink(missing_ok=True)


def evaluation_num_workers_from_cfg(cfg: DictConfig, *, final: bool) -> int:
    key = "test_workers" if final else "eval_workers"
    return int(cfg.experiment.get(key, 1))
