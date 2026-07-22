from __future__ import annotations

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import thesis_rl.runtime.async_evaluation as async_module
from thesis_rl.runtime.async_evaluation import (
    AsyncEvaluationError,
    AsyncEvaluationManager,
)


class _FakeProcess:
    def __init__(self, *, target, args, daemon):
        self._target = target
        self._args = args
        self._alive = False
        self.daemon = daemon

    def start(self):
        self._alive = True
        self._target(*self._args)
        self._alive = False

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        del timeout

    def terminate(self):
        self._alive = False


class _FakeAgent:
    def __init__(self, root: Path):
        self.root = root
        self.saved: list[Path] = []

    def save(self, stem: Path) -> None:
        self.saved.append(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".zip").write_text("snapshot", encoding="utf-8")


class _EnvironmentCapturingProcess(_FakeProcess):
    observed_openblas_threads: str | None = None

    def start(self):
        type(self).observed_openblas_threads = os.environ.get("OPENBLAS_NUM_THREADS")
        super().start()


def _cfg():
    return OmegaConf.create(
        {
            "seed": 7,
            "device": "cpu",
            "experiment": {
                "eval_workers": 2,
                "eval_deterministic": True,
            },
        }
    )


def test_fifo_jobs_keep_distinct_snapshots_and_complete_in_order(tmp_path, monkeypatch):
    completed: list[int] = []

    def fake_worker(job, output_queue):
        output_queue.put(("started", job.eval_id, job.episode_count))
        output_queue.put(("progress", job.eval_id, job.episode_count, job.episode_count))
        output_queue.put(
            (
                "finished",
                job.eval_id,
                {
                    "per_episode": {"returns": [float(job.eval_id)]},
                    "mean_reward": float(job.eval_id),
                },
            )
        )

    monkeypatch.setattr(async_module, "_evaluation_worker_main", fake_worker)
    manager = AsyncEvaluationManager(
        checkpoints_dir=tmp_path,
        on_complete=lambda job, metrics: completed.append(job.eval_id),
        process_factory=_FakeProcess,
    )
    agent = _FakeAgent(tmp_path)

    manager.enqueue(
        agent=agent,
        cfg=_cfg(),
        eval_id=1,
        global_step=10,
        stage="baseline",
        stage_index=0,
        episode_count=2,
        base_seed=100,
        env_seed=200,
    )
    manager.enqueue(
        agent=agent,
        cfg=_cfg(),
        eval_id=2,
        global_step=20,
        stage="baseline",
        stage_index=0,
        episode_count=3,
        base_seed=300,
        env_seed=400,
    )
    manager.drain()

    assert completed == [1, 2]
    assert [path.name for path in agent.saved] == [
        "eval_000001_step_000000000010",
        "eval_000002_step_000000000020",
    ]
    assert manager.last_completed_metrics == {
        "per_episode": {"returns": [2.0]},
        "mean_reward": 2.0,
    }
    manager.close()


def test_evaluation_error_is_fatal(tmp_path, monkeypatch):
    def fake_worker(job, output_queue):
        output_queue.put(("error", job.eval_id, "deterministic evaluator failure"))

    monkeypatch.setattr(async_module, "_evaluation_worker_main", fake_worker)
    manager = AsyncEvaluationManager(checkpoints_dir=tmp_path, process_factory=_FakeProcess)
    manager.enqueue(
        agent=_FakeAgent(tmp_path),
        cfg=_cfg(),
        eval_id=1,
        global_step=10,
        stage="baseline",
        stage_index=0,
        episode_count=1,
        base_seed=100,
        env_seed=200,
    )
    with pytest.raises(AsyncEvaluationError, match="training terminated"):
        manager.poll()


def test_renderables_expose_progress_and_last_complete_monitor(tmp_path, monkeypatch):
    def fake_worker(job, output_queue):
        output_queue.put(
            (
                "finished",
                job.eval_id,
                {"per_episode": {"returns": [1.0]}, "mean_reward": 1.0, "success_rate": 0.5},
            )
        )

    monkeypatch.setattr(async_module, "_evaluation_worker_main", fake_worker)
    manager = AsyncEvaluationManager(checkpoints_dir=tmp_path, process_factory=_FakeProcess)
    manager.enqueue(
        agent=_FakeAgent(tmp_path),
        cfg=_cfg(),
        eval_id=1,
        global_step=10,
        stage="baseline",
        stage_index=0,
        episode_count=1,
        base_seed=100,
        env_seed=200,
    )
    renderables = manager.renderables()
    assert len(renderables) == 2
    assert renderables[1].title == "Evaluation Monitor"
    assert [column.header for column in renderables[1].renderable.columns] == [
        "Metric",
        "Value",
    ]
    assert "Evaluation Queue" not in str(renderables)
    manager.close()


def test_renderables_keep_completed_episode_count_after_job_finishes(tmp_path, monkeypatch):
    def fake_worker(job, output_queue):
        output_queue.put(("started", job.eval_id, job.episode_count))
        output_queue.put(("progress", job.eval_id, job.episode_count, job.episode_count))
        output_queue.put(
            (
                "finished",
                job.eval_id,
                {
                    "per_episode": {"returns": [1.0, 2.0]},
                    "mean_reward": 1.5,
                },
            )
        )

    monkeypatch.setattr(async_module, "_evaluation_worker_main", fake_worker)
    manager = AsyncEvaluationManager(
        checkpoints_dir=tmp_path,
        process_factory=_FakeProcess,
    )
    manager.enqueue(
        agent=_FakeAgent(tmp_path),
        cfg=_cfg(),
        eval_id=1,
        global_step=10,
        stage="baseline",
        stage_index=0,
        episode_count=2,
        base_seed=100,
        env_seed=200,
    )

    manager.renderables()

    task = manager._progress.tasks[0]
    assert task.completed == 2
    assert task.total == 2
    assert task.description == "Evaluation episodes (2/2)"
    manager.close()


def test_async_evaluator_inherits_numeric_thread_limit_before_spawn(tmp_path, monkeypatch) -> None:
    def fake_worker(job, output_queue):
        output_queue.put(("finished", job.eval_id, {"per_episode": {"returns": [1.0]}}))

    monkeypatch.setattr(async_module, "_evaluation_worker_main", fake_worker)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "17")
    _EnvironmentCapturingProcess.observed_openblas_threads = None
    manager = AsyncEvaluationManager(
        checkpoints_dir=tmp_path,
        process_factory=_EnvironmentCapturingProcess,
        numeric_library_num_threads=2,
    )
    manager.enqueue(
        agent=_FakeAgent(tmp_path),
        cfg=_cfg(),
        eval_id=1,
        global_step=10,
        stage="baseline",
        stage_index=0,
        episode_count=1,
        base_seed=100,
        env_seed=200,
    )
    manager.drain()
    assert _EnvironmentCapturingProcess.observed_openblas_threads == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "17"
    manager.close()
