"""Parent-owned FIFO orchestration for asynchronous training evaluation."""

from __future__ import annotations

import json
import multiprocessing as mp
import queue
import statistics
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from thesis_rl.runtime.execution.numeric_threads import start_process_with_numeric_thread_limit


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
    batch_id: str | None = None
    panel_name: str | None = None
    evaluation_scope: str | None = None
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
        planner = load_planner(
            cfg,
            checkpoint_path=f"{job.checkpoint_stem}.zip",
            env=env,
            validate_rollout_geometry=False,
        )
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
        numeric_library_num_threads: int | None = None,
    ) -> None:
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"Unsupported asynchronous evaluation start method: {start_method}")
        self._ctx = mp.get_context(start_method)
        self._queue = self._ctx.Queue()
        self._process_factory = process_factory or self._ctx.Process
        self._numeric_library_num_threads = numeric_library_num_threads
        self._checkpoints_dir = checkpoints_dir / "async_eval"
        self._on_complete = on_complete
        self._pending: deque[_JobState] = deque()
        self._active: _JobState | None = None
        self._jobs: dict[int, _JobState] = {}
        self._snapshot_refcounts: dict[str, int] = {}
        self._batch_admission_in_progress = False
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
        batch_id: str | None = None,
        panel_name: str | None = None,
        evaluation_scope: str | None = None,
        checkpoint_stem: str | Path | None = None,
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
        stem = (
            Path(checkpoint_stem)
            if checkpoint_stem is not None
            else self._checkpoints_dir / f"eval_{job_index:06d}_step_{int(global_step):012d}"
        )
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        own_snapshot = checkpoint_stem is None
        if own_snapshot:
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
        if own_snapshot:
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
            batch_id=batch_id or f"eval-{int(eval_id)}",
            panel_name=panel_name,
            evaluation_scope=evaluation_scope,
            scenario_arm_schedule=scenario_arm_schedule,
            scenario_source_schedule=scenario_source_schedule,
            metadata=dict(metadata or {}),
        )
        state = _JobState(job=job)
        self._jobs[job.eval_id] = state
        self._snapshot_refcounts[str(stem)] = self._snapshot_refcounts.get(str(stem), 0) + 1
        self._pending.append(state)
        self._event_messages.append(
            f"[EVAL] Queued evaluation {job.eval_id} | step={job.global_step} | episodes={job.episode_count}"
        )
        if not self._batch_admission_in_progress:
            self._launch_next()
        return job

    def enqueue_batch(
        self,
        *,
        agent: Any,
        cfg: DictConfig,
        batch_id: str,
        global_step: int,
        stage: str,
        stage_index: int,
        jobs: tuple[Mapping[str, Any], ...],
    ) -> tuple[EvaluationJob, ...]:
        """Admit a complete panel batch from exactly one immutable snapshot.

        The manager intentionally permits only one admitted batch.  It drains
        a prior batch before saving the next snapshot, so training cannot lose
        a required validation boundary by coalescing or unbounded queueing.
        """
        if not batch_id:
            raise ValueError("evaluation batch_id must be non-empty")
        if not jobs:
            raise ValueError("an evaluation batch must contain at least one panel")
        self.poll()
        if self._active is not None or self._pending:
            self.drain()
        if len({str(item.get("panel_name", "")) for item in jobs}) != len(jobs):
            raise ValueError("an evaluation batch cannot contain duplicate panel names")
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        stem = self._checkpoints_dir / f"batch_{batch_id}_step_{int(global_step):012d}"
        agent.save(stem)
        Path(f"{stem}.json").write_text(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "global_step": int(global_step),
                    "stage": stage,
                    "stage_index": int(stage_index),
                    "panels": [dict(item.get("metadata", {})) for item in jobs],
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        admitted: list[EvaluationJob] = []
        self._batch_admission_in_progress = True
        try:
            for item in jobs:
                admitted.append(
                    self.enqueue(
                        agent=agent,
                        cfg=cfg,
                        eval_id=int(item["eval_id"]),
                        global_step=global_step,
                        stage=stage,
                        stage_index=stage_index,
                        episode_count=int(item["episode_count"]),
                        base_seed=item.get("base_seed"),
                        env_seed=int(item["env_seed"]),
                        env_overrides=item.get("env_overrides"),
                        workers=item.get("workers"),
                        scenario_arm_schedule=item.get("scenario_arm_schedule"),
                        scenario_source_schedule=item.get("scenario_source_schedule"),
                        metadata=item.get("metadata"),
                        batch_id=batch_id,
                        panel_name=item.get("panel_name"),
                        evaluation_scope=item.get("evaluation_scope"),
                        checkpoint_stem=stem,
                    )
                )
        finally:
            self._batch_admission_in_progress = False
        self._launch_next()
        return tuple(admitted)

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
        """Return one panel-aware Rich progress renderer and status table."""

        self.poll()
        active = self._active
        progress_state = active or self._last_completed
        current = (
            f"{progress_state.completed}/{progress_state.job.episode_count}"
            if progress_state
            else "0/0"
        )
        panel = progress_state.job.panel_name if progress_state is not None else None
        progress_description = "Evaluation idle" if panel is None else f"Evaluation {panel} ({current})"
        self._progress.update(
            self._progress_task,
            description=progress_description,
            total=max(progress_state.job.episode_count, 1) if progress_state else 1,
            completed=progress_state.completed if progress_state else 0,
        )
        table = Table(title="Evaluation panels", expand=True)
        table.add_column("Batch / panel", style="cyan")
        table.add_column("State")
        table.add_column("Progress", justify="right")
        table.add_column("Success μ ± σ", justify="right")
        table.add_column("Collision μ ± σ", justify="right")
        table.add_column("Route completion μ ± σ", justify="right")
        for state in self._jobs.values():
            job = state.job
            metrics = state.metrics or {}
            per_episode = metrics.get("per_episode", {}) if isinstance(metrics, dict) else {}
            table.add_row(
                f"{job.batch_id or '-'} / {job.panel_name or job.eval_id}",
                state.status,
                f"{state.completed}/{job.episode_count}",
                _sample_mean_sd(per_episode.get("success", ())),
                _sample_mean_sd(per_episode.get("collision", ())),
                _sample_mean_sd(per_episode.get("route_completion", ())),
            )
        if not self._jobs:
            table.add_row("-", "idle", "0/0", "-", "-", "-")
        return [self._progress, Panel(table, title="Evaluation Monitor")]

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
        start_process_with_numeric_thread_limit(process, self._numeric_library_num_threads)
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
            self._release_snapshot(state.job)
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

    def _release_snapshot(self, job: EvaluationJob) -> None:
        stem = Path(job.checkpoint_stem)
        key = str(stem)
        remaining = self._snapshot_refcounts.get(key, 0) - 1
        if remaining > 0:
            self._snapshot_refcounts[key] = remaining
            return
        self._snapshot_refcounts.pop(key, None)
        for path in (
            stem.with_suffix(".zip"),
            stem.with_name(f"{stem.name}.adapter.pt"),
            stem.with_name(f"{stem.name}.reward_semantics.json"),
            stem.with_suffix(".json"),
        ):
            path.unlink(missing_ok=True)


def _sample_mean_sd(values: Any) -> str:
    """Render the requested sample mean ± sample standard deviation."""
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return "-"
    deviation = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
    return f"{statistics.fmean(numeric):.3f} ± {deviation:.3f}"


def evaluation_num_workers_from_cfg(cfg: DictConfig, *, final: bool) -> int:
    key = "test_workers" if final else "eval_workers"
    return int(cfg.experiment.get(key, 1))
